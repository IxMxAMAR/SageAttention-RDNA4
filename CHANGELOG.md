# Changelog

The heading gives the release and the wheel's version string. The `+amd.gfx12.N` part is a local
version identifier, so every wheel can be traced to the release it came from. The numbers below are
summaries; [docs/FINDINGS.md](docs/FINDINGS.md) has the measurements behind them and
[docs/METHOD.md](docs/METHOD.md) how they were taken.

## v0.4.0 — 2.2.0+amd.gfx12.4

**The quantization step before the kernel is now hand-written HIP as well, so short calls are about
even with PR #368's instead of up to 1.6× slower, and nothing is compiled on first use.**

- **A HIP prologue replaces the Triton one.** Every call quantizes K and V before the attention
  kernel runs. That used to be 3–4 Triton launches with a JIT compiler behind them. It is now 1–2
  launches of a prebuilt code object, on every default path (fp16 with int8 Q·K, fp16 with fp8
  Q·K, bf16; contiguous, strided and fused-QKV layouts). What that changes:
  - **Short calls are faster.** Timed one at a time, a call of 288 to 1 024 tokens takes 0.85–1.13
    of PR #368's time, where v0.3.0 took 1.12–1.59. Production sizes are unchanged within noise.
  - **No compile on first use.** In ComfyUI's environment with an empty Triton cache, the first
    call takes 0.48 s instead of 1.9 s, and nothing is compiled.
  - **No 2^31-element limit.** v0.3.0 stopped serving at about 100 000 tokens at H3's layout,
    because the Triton prologue computed offsets in 32 bits. The HIP prologue uses 64-bit offsets.
  - **Same output.** With `smooth_k` off, the quantized bytes and the attention output are
    bit-identical to v0.3.0, including all-zero rows, values past fp8's range, and NaN/Inf. With
    `smooth_k` on, the key mean is now summed in fp64, which is deterministic and more accurate
    than the old fp32 sum. A small share of outputs therefore differ from v0.3.0 by rounding, with
    the same error against exact attention.
  - `SAGEATTN_SK1_PROLOGUE=triton` brings back v0.3.0's Triton prologue.
- **RX 9060 XT (gfx1200), opt in.** The wheel carries gfx1200 builds of every kernel the default
  paths use. Their machine code is byte-identical to the gfx1201 builds, but nobody has run them on
  a 9060 XT yet, so they stay off unless `SAGEATTN_SK1_GFX1200=1` is set. The device always picks
  its own builds. If you have the card, [docs/GFX1200-TESTING.md](docs/GFX1200-TESTING.md) says
  what to run and what to send back.
- **The build records the torch it was built against.** The runtime check compares against that
  stamp instead of a hard-coded `2.13.0+rocm10.0.0`, which is what wheels for other torch versions
  and for Linux need.

Not in this release:
- **head_dim 64** still uses PR #368's kernel. A head_dim 64 kernel on the new prologue was built
  and measured, and it is still 1.03–5.3× slower than PR #368's. That needs its own design.
- **ROCm 10.1 / PyTorch 2.14** came later, on 2026-10-29 (#3): a second Windows wheel,
  `sageattention-2.2.0+amd.gfx12.4-214-cp312-cp312-win_amd64.whl`, built against
  `2.14.0+rocm10.1.0`. Same code; it passes 97 of 97 tests on two machines and its output is
  bit-identical to the 2.13 wheel. PyTorch 2.14 itself needs a recent driver (it failed on
  32.0.31041.1004 and runs on Adrenalin 26.8.1 and 26.10.1).
- **A Linux wheel.** Linux runs from source; Linux testing of this release happens in a pull
  request.

## v0.3.0 — 2.2.0+amd.gfx12.3

- **Fused q/k/v views are served.** MiniMax-H3, and any model that fuses q, k and v into one
  projection, passes them as views of one tensor, so a row's stride is three heads wide. Those
  calls used to fall back to PR #368's path without saying so. The acceptance rule is now the raw
  launch's own requirement, and a refused call carries its reason.
- **Against PR #368 at production sizes** (N = 27 280 to 99 792): 0.87–0.94 of its time. In a real
  MiniMax-H3 w4a8 render at 864x480 and 124 frames, the sampling phase took 0.950 (4 steps) and
  0.847 (8 steps) of PR #368's time.
- **Linux runs from source**, thanks to [boxwrench](https://github.com/boxwrench)
  ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)). The wheel is Windows-only.
- The version strings in `setup.py`, `sageattention/__init__.py` and `sageattention/_version.py`
  agree again.

Known issue outside this package: with the ROCm-int8 ComfyUI pack's `OTUNetLoaderW8A8`, the first
forward pass of a w4a8 checkpoint hung the driver on the test machine (6 of 6 attempts). ComfyUI's
own `UNETLoader` handled the same checkpoint 5 times out of 5.

## v0.2.0 — 2.2.0+amd.gfx12.2

- **int8 Q·K is the default for fp16**, head_dim 128. Calls it cannot serve fall back to the fp8
  kernel, then to PR #368's path.
- **bf16 runs on the hand-written kernels**, also by default (fp8 Q·K; there is no int8 bf16
  kernel).
- `SAGEATTN_SK1_INT8=0` and `SAGEATTN_SK1_BF16=0` restore v0.1.0's behaviour exactly. `=1` makes
  the switch strict: it raises when its kernel is missing or fails to load.

## v0.1.0 — 2.2.0+amd.gfx12.1

- First release: a hand-written HIP fp8 attention kernel for fp16, head_dim 128, used by default on
  gfx1201, with a silent fallback to PR #368's path everywhere else and on any load failure.
