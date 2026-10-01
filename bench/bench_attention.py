#!/usr/bin/env python3
"""Time sageattn (default), sageattn with the hand-written kernel disabled, and torch SDPA.

Shapes: B=1, H=48, D=128, fp16, N in {4096, 8771, 16384}, causal off and on.

The three variants are timed in interleaved repetitions: one repetition times each variant once
(the order rotates from one repetition to the next), so slow drift in clocks or temperature hits
all of them alike. Each timing is the mean of `--iters` calls between two CUDA events, after
`--warmup` untimed calls. The table shows the median over repetitions and the min-max range.

`SAGEATTN_SK1_BACKEND` is read when `sageattention` is imported, so the "hand-written kernel
disabled" variant runs in a worker subprocess started with `SAGEATTN_SK1_BACKEND=0`. The parent
and the worker take turns on the GPU and never run at the same time.

    python bench/bench_attention.py [--reps 15] [--iters 5] [--warmup 3]
"""
import argparse
import os
import statistics
import subprocess
import sys

import torch
import torch.nn.functional as F

B, H, D = 1, 48, 128
SHAPES = [4096, 8771, 16384]


def make_qkv(n):
    g = torch.Generator(device="cuda").manual_seed(0)
    return [torch.randn(B, H, n, D, device="cuda", dtype=torch.float16, generator=g)
            for _ in range(3)]


def time_calls(fn, iters):
    """Mean milliseconds per call over `iters` calls, measured with CUDA events."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def worker():
    """Line protocol on stdin/stdout: `setup N causal warmup`, then repeated `time iters`."""
    from sageattention import sageattn
    state = {}
    for line in sys.stdin:
        cmd, *args = line.split()
        try:
            if cmd == "setup":
                n, causal, warmup = int(args[0]), bool(int(args[1])), int(args[2])
                q, k, v = make_qkv(n)
                state["fn"] = lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=causal)
                for _ in range(warmup):
                    state["fn"]()
                torch.cuda.synchronize()
                print("OK", flush=True)
            elif cmd == "time":
                print("OK %.6f" % time_calls(state["fn"], int(args[0])), flush=True)
            elif cmd == "quit":
                return
        except Exception as exc:  # report to the parent instead of dying silently
            print("ERR %s: %s" % (type(exc).__name__, str(exc).splitlines()[0] if str(exc) else ""),
                  flush=True)


class Worker:
    def __init__(self):
        env = dict(os.environ, SAGEATTN_SK1_BACKEND="0")
        self.proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     env=env)

    def ask(self, line):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()
        reply = self.proc.stdout.readline().strip()
        if not reply.startswith("OK"):
            raise RuntimeError(reply or "worker exited")
        return reply[2:].strip()

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        self.proc.wait(timeout=60)


def fmt(samples):
    if not samples:
        return "n/a"
    return "%7.2f (%.2f-%.2f)" % (statistics.median(samples), min(samples), max(samples))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=15, help="interleaved repetitions per cell")
    ap.add_argument("--iters", type=int, default=5, help="calls averaged inside one timing")
    ap.add_argument("--warmup", type=int, default=3, help="untimed calls before timing a cell")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        worker()
        return 0

    from sageattention import sageattn

    props = torch.cuda.get_device_properties(0)
    print("device: %s (%s), torch %s" % (props.name, props.gcnArchName, torch.__version__))
    print("B=%d H=%d D=%d fp16; %d reps x %d calls, %d warm-up calls; ms per call, median (min-max)"
          % (B, H, D, args.reps, args.iters, args.warmup))

    worker_proc = Worker()
    rows = []
    try:
        for n in SHAPES:
            q, k, v = make_qkv(n)
            for causal in (False, True):
                variants = {
                    "default": lambda: sageattn(q, k, v, tensor_layout="HND", is_causal=causal),
                    "off": None,
                    "sdpa": lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal),
                }
                samples = {name: [] for name in variants}
                failed = set()
                try:
                    worker_proc.ask("setup %d %d %d" % (n, int(causal), args.warmup))
                except RuntimeError as exc:
                    failed.add("off")
                    print("worker setup failed for N=%d: %s" % (n, exc), file=sys.stderr)
                for name, fn in variants.items():
                    if fn is None:
                        continue
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
                    order = names[rep % len(names):] + names[:rep % len(names)]
                    for name in order:
                        if name in failed:
                            continue
                        if name == "off":
                            samples[name].append(float(worker_proc.ask("time %d" % args.iters)))
                        else:
                            samples[name].append(time_calls(variants[name], args.iters))
                rows.append((n, causal, samples))
            del q, k, v
            torch.cuda.empty_cache()
    finally:
        worker_proc.close()

    print()
    print("%6s %6s | %-22s | %-22s | %-22s | %8s %8s %9s"
          % ("N", "causal", "sageattn default", "sageattn SK1 off", "torch SDPA",
             "off/def", "sdpa/def", "def TFLOPS"))
    for n, causal, s in rows:
        med = {k: statistics.median(v) if v else None for k, v in s.items()}
        flops = 4.0 * B * H * n * n * D * (0.5 if causal else 1.0)
        ratio = lambda a, b: "%.2fx" % (med[a] / med[b]) if med[a] and med[b] else "n/a"
        tflops = "%.1f" % (flops / (med["default"] * 1e-3) / 1e12) if med["default"] else "n/a"
        print("%6d %6s | %-22s | %-22s | %-22s | %8s %8s %9s"
              % (n, "on" if causal else "off", fmt(s["default"]), fmt(s["off"]), fmt(s["sdpa"]),
                 ratio("off", "default"), ratio("sdpa", "default"), tflops))
    print("\noff/def and sdpa/def are median ratios; above 1 means the default is faster.")
    print("TFLOPS uses 4*B*H*N^2*D flops (halved when causal) and the default variant's median.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
