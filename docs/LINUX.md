# Linux

There is no prebuilt Linux wheel. The package builds from source on Linux, and the prebuilt kernel
objects load there through `libamdhip64.so.7`
([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1), by
[boxwrench](https://github.com/boxwrench)).

| Version | GPU | OS | PyTorch | Checked |
|---|---|---|---|---|
| v0.2.0 | Radeon AI PRO R9700 (`gfx1201`) | Ubuntu 24.04 | `2.9.1+rocm7.2.1` | kernels load, 88/88 tests pass |

Every speed and accuracy number in the README was measured on Windows. The Linux H3 timings in #1
ran on PR #368's path, because H3's fused-QKV views were only served from v0.3.0 on, so there are no
Linux timings of these kernels yet.

## Requirements

- A `gfx1201` GPU: RX 9070, RX 9070 XT or Radeon AI PRO R9700. `gfx1200` (RX 9060 XT) is opt-in,
  as on Windows ([GFX1200-TESTING.md](GFX1200-TESTING.md)).
- Python 3.12.
- A ROCm build of PyTorch and a ROCm toolchain to compile the extensions against. setup.py takes the
  first of: the ROCm SDK Python packages (`rocm-sdk path --root`), the ROCm that PyTorch finds
  (normally `/opt/rocm`), `$ROCM_HOME`.

The `.hsaco` kernel objects are not compiled; they ship prebuilt in `sageattention/sk1_backend/` and
are the same files on Windows and Linux.

## Build

```
git clone https://github.com/IxMxAMAR/SageAttention-RDNA4
cd SageAttention-RDNA4
export PYTORCH_ROCM_ARCH=gfx1201
pip install . --no-build-isolation --no-deps
```

Build isolation has to be off because setup.py imports your PyTorch. The compile takes tens of
minutes. `pip wheel . --no-build-isolation --no-deps -w dist` builds a wheel instead of installing.

The build records the PyTorch version it compiled against in `sageattention/_build_stamp.py`, and
the import-time check compares with that, so no override is needed for your PyTorch version. The
result only works with that PyTorch version; rebuild after upgrading it.

## Check

Run the tests from outside the checkout, so the installed package is imported and not the source
tree:

```
cd ~
python -m pytest path/to/SageAttention-RDNA4/tests -q
```

On a `gfx1201` GPU every test should pass. If they are all skipped, PyTorch did not report a
`gfx1201` device.

Benchmarks, from the checkout:

```
python bench/bench_attention.py
python bench/bench_attention.py --dtype bf16
python bench/bench_attention.py --preset h3
```

The `kernel` column says whether the hand-written kernel served that row; it should say `yes`
everywhere. The environment switches are the same as on Windows (README, "Use").
