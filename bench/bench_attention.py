#!/usr/bin/env python3
"""Time sageattn (default), sageattn with the hand-written kernels off, and torch SDPA.

Presets:
  default  B=1, H=48, D=128, contiguous HND, N in {4096, 8771, 16384}, causal off and on.
  h3       B=1, H=56, D=128, bf16, non-causal, q/k/v as views into one fused QKV buffer (the way
           MiniMax-H3 passes them), N in {27280, 31248, 61488}.

The three variants are timed in interleaved repetitions in one process: one repetition times each
variant once, and the order rotates from one repetition to the next, so slow drift in clocks or
temperature hits all of them alike. Each timing is the mean of `--iters` calls between two GPU
events, after `--warmup` untimed calls. The table shows the median over repetitions and the
min-max range. The "off" variant is `sageattn(..., sk1_backend=False)`, i.e. PR #368's path.

Before timing, every cell checks that the hand-written kernel really serves it and prints the
reason when it does not. A cell that falls back is timed anyway, but its default and off columns
then measure the same path.

    python bench/bench_attention.py [--preset default|h3] [--dtype fp16|bf16]
                                    [--reps 15] [--iters 5] [--warmup 3] [--json out.json]
"""
import argparse
import json
import platform
import statistics
import sys

import torch
import torch.nn.functional as F

PRESETS = {
    "default": dict(heads=48, shapes=[4096, 8771, 16384], causal=(False, True), fused=False),
    "h3": dict(heads=56, shapes=[27280, 31248, 61488], causal=(False,), fused=True),
}
D = 128


def make_qkv(n, heads, dtype, fused):
    g = torch.Generator(device="cuda").manual_seed(0)
    if not fused:
        return [torch.randn(1, heads, n, D, device="cuda", dtype=dtype, generator=g)
                for _ in range(3)]
    # One (N, 3*H*D) projection output split into q, k and v, then viewed as (1, H, N, D).
    qkv = torch.randn(n, 3 * heads * D, device="cuda", dtype=dtype, generator=g)
    return [t.view(n, heads, D).transpose(0, 1).unsqueeze(0)
            for t in qkv.split(heads * D, dim=-1)]


def time_calls(fn, iters):
    """Mean milliseconds per call over `iters` calls, measured with GPU events."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def served(q, k, v, causal):
    """Whether the hand-written kernel serves this call, and the refusal reason if it does not."""
    from sageattention.sk1_backend import try_sk1_t1
    out, reason = try_sk1_t1(q, k, v, tensor_layout="HND", is_causal=causal)
    return out is not None, reason


def fmt(samples):
    if not samples:
        return "n/a"
    return "%7.2f (%.2f-%.2f)" % (statistics.median(samples), min(samples), max(samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
    ap.add_argument("--dtype", choices=("fp16", "bf16"), default=None,
                    help="default: fp16 for the default preset, bf16 for h3")
    ap.add_argument("--reps", type=int, default=15, help="interleaved repetitions per cell")
    ap.add_argument("--iters", type=int, default=5, help="calls averaged inside one timing")
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls before timing a cell")
    ap.add_argument("--json", help="also write the raw samples to this file")
    args = ap.parse_args()

    import sageattention
    from sageattention import sageattn

    preset = PRESETS[args.preset]
    dtype_name = args.dtype or ("bf16" if args.preset == "h3" else "fp16")
    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float16
    heads = preset["heads"]

    props = torch.cuda.get_device_properties(0)
    print("device: %s (%s) | %s | torch %s | sageattention %s"
          % (props.name, props.gcnArchName, platform.platform(), torch.__version__,
             sageattention.__version__))
    print("preset %s: B=1 H=%d D=%d %s%s; %d reps x %d calls, %d warm-up calls; ms per call, "
          "median (min-max)" % (args.preset, heads, D, dtype_name,
                                ", fused QKV views" if preset["fused"] else "",
                                args.reps, args.iters, args.warmup))

    rows = []
    for n in preset["shapes"]:
        q, k, v = make_qkv(n, heads, dtype, preset["fused"])
        for causal in preset["causal"]:
            ok, reason = served(q, k, v, causal)
            variants = {
                "default": lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=causal),
                "off": lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=causal,
                                        sk1_backend=False),
                "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
            }
            samples = {name: [] for name in variants}
            failed = set()
            for name, fn in variants.items():
                try:
                    for _ in range(args.warmup):
                        fn()
                    torch.cuda.synchronize()
                except RuntimeError as exc:
                    failed.add(name)
                    print("%s failed for N=%d: %s" % (name, n, str(exc).splitlines()[0]),
                          file=sys.stderr)
            names = list(variants)
            for rep in range(args.reps):
                for name in names[rep % len(names):] + names[:rep % len(names)]:
                    if name not in failed:
                        samples[name].append(time_calls(variants[name], args.iters))
            rows.append(dict(n=n, causal=causal, served=ok, reason=reason, samples=samples))
        del q, k, v
        torch.cuda.empty_cache()

    print()
    print("%6s %6s %-8s | %-25s | %-25s | %-25s | %8s %8s %9s"
          % ("N", "causal", "kernel", "sageattn default", "sageattn kernel off", "torch SDPA",
             "off/def", "sdpa/def", "def TFLOPS"))
    for r in rows:
        s = r["samples"]
        med = {name: statistics.median(v) if v else None for name, v in s.items()}
        flops = 4.0 * heads * r["n"] * r["n"] * D * (0.5 if r["causal"] else 1.0)
        ratio = lambda a, b: "%.2fx" % (med[a] / med[b]) if med[a] and med[b] else "n/a"
        tflops = "%.1f" % (flops / (med["default"] * 1e-3) / 1e12) if med["default"] else "n/a"
        print("%6d %6s %-8s | %-25s | %-25s | %-25s | %8s %8s %9s"
              % (r["n"], "on" if r["causal"] else "off", "yes" if r["served"] else "NO",
                 fmt(s["default"]), fmt(s["off"]), fmt(s["sdpa"]),
                 ratio("off", "default"), ratio("sdpa", "default"), tflops))
        if not r["served"]:
            print("%22s not served by the hand-written kernel: %s" % ("", r["reason"]))
    print("\noff/def and sdpa/def are median ratios; above 1 means the default is faster.")
    print("TFLOPS uses 4*B*H*N^2*D flops (halved when causal) and the default variant's median.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(device=props.name, arch=props.gcnArchName, platform=platform.platform(),
                           torch=torch.__version__, sageattention=sageattention.__version__,
                           preset=args.preset, dtype=dtype_name, heads=heads, rows=rows),
                      f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
