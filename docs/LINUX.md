# Linux

The package builds from source on Linux, and the prebuilt kernel
objects load there through `libamdhip64.so.7`
([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1), by
[boxwrench](https://github.com/boxwrench)).

| Version | GPU | OS | PyTorch | Checked |
|---|---|---|---|---|
| v0.2.0 | Radeon AI PRO R9700 (`gfx1201`) | Ubuntu 24.04 | `2.9.1+rocm7.2.1` | kernels load, 88/88 tests pass |
| v0.4.0 | Radeon AI PRO R9700, ~300 W | Linux | `2.9.1+rocm7.2.1` | source build with no override, 97/97 tests pass, every benchmark row served; H3 bit-identical to a direct kernel call ([boxwrench](https://github.com/boxwrench), #2) |
| v0.4.0 | Radeon AI PRO R9700, capped at 220 W | Linux | `2.14.0+rocm10.2.0a20261012` (nightly) | 97/97 tests pass, every benchmark row served (VRTest20, #2) |

### Timings

Full `sageattn` call, B=1, D=128, non-causal unless noted, `bench/bench_attention.py`; our time
compared with PR #368's path (`kernel off`) and with PyTorch SDPA, as speed-ups:

| Setup | Shape | vs PR #368 | vs SDPA |
|---|---|---|---|
| boxwrench, torch 2.9.1 | H3 bf16 fused QKV, H=56, N = 27 280 / 31 248 / 61 488 | 1.50 / 1.49 / 1.46× | 3.05 / 3.11 / 3.26× |
| boxwrench, torch 2.9.1 | default presets, fp16 and bf16 | 1.28–1.85× | |
| VRTest20, torch 2.14 nightly | H3 bf16 fused QKV, same N | 1.23 / 1.23 / 1.19× | 1.98 / 2.04 / 2.07× |
| VRTest20, torch 2.14 nightly | fp16, H=48, N = 4096 / 8771 / 16384 | 1.17 / 1.27 / 1.14× | 1.41 / 1.75 / 1.89× |
| VRTest20, torch 2.14 nightly | bf16, same N | 1.21 / 1.30 / 1.16× | 1.49 / 1.80 / 1.92× |

The hand-written kernel runs at the same speed as on Windows (about 142 ms at N = 27 280 against
140 ms on an RX 9070 XT). The larger margins with torch 2.9.1 come from PR #368's path and SDPA
being slower on that older ROCm stack. These were measured on other people's machines with their
own protocol, so they are not part of the README's tables.

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
