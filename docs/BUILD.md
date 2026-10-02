# Building from source

There are two independent build products:

1. **The Python package and its two native extensions** (`_qattn_gfx12_native`, `_fused`). These
   come from `csrc/`, the gfx12 port from PR #368, and are built by `setup.py`.
2. **The hand-written kernels' code objects** (`sageattention/sk1_backend/*.hsaco`), built from
   `kernels/hip/` by `tools/build_hsaco.py`. They ship prebuilt in the repo, so you only need this
   step if you change a kernel.

## Requirements

- Windows 10/11 with the MSVC build tools (C++ workload).
- Python 3.12 with a ROCm build of PyTorch. The prebuilt wheel was built against
  `2.13.0+rocm10.0.0`.
- The ROCm SDK for gfx120X in the same environment. The Python SDK packages work: their
  `_rocm_sdk_devel` package provides clang and the HIP headers.

## 1. The package

From a shell with the MSVC environment loaded (for example the "x64 Native Tools" prompt):

```
set PYTORCH_ROCM_ARCH=gfx1201
pip wheel . --no-build-isolation --no-deps -w dist
```

- `--no-build-isolation` is required because `setup.py` imports torch to configure the HIP
  extensions.
- A missing `gfx1201` target is a hard error on purpose. Without it the wheel would build, but
  without the native extension, and would quietly fall back to PyTorch attention.
- The compile takes about 50 minutes on a desktop CPU; most of it is the 440 kB
  `qk_int_sv_gfx12_native.cu`.

The extensions link against the PyTorch C++ ABI, so a wheel only works with the PyTorch version it
was built against.

To package extensions you have already built (for example after changing only Python files):

```
python setup.py --no-build-ext bdist_wheel
```

## 2. The kernel code objects

```
python tools/build_hsaco.py            # rebuild all shipped objects in place
python tools/build_hsaco.py --check    # build to a temp dir and compare with the shipped files
```

The script compiles each kernel device-only for `gfx1201` and unbundles the raw ELF that
`hipModuleLoadData` expects. It needs no GPU. It finds clang from, in order: `SK1_CLANG`,
`SK1_ROCM_DEVEL`, the `_rocm_sdk_devel` package, `ROCM_PATH`/`HIP_PATH`, then `PATH`.

`--check` compares the machine code, kernel descriptors and resource metadata, and ignores symbol
tables. A comment change renames an internal symbol, so the files are not byte-identical after any
source edit.

Any change to a kernel has to pass the gates in [METHOD.md](METHOD.md) §3 before it is timed. Those
are: at most 240 VGPRs, no spills, LDS within budget, and no barrier reachable by only some waves.

## 3. Tests and benchmark

```
python -m pytest tests -q              # needs a gfx1201 GPU; skipped otherwise
python bench/bench_attention.py        # default vs hand-written kernel off vs PyTorch SDPA
```

Run the tests from outside the checkout, or with the installed package first on `PYTHONPATH`. The
checkout's own `sageattention/` has no compiled extensions and would shadow the installed one.

The tests cover, in fp16 and bf16:
- agreement with PyTorch SDPA;
- 36 edge-length cells: N from 1 to 1000, causal and non-causal, `smooth_k` on and off;
- a key with |V| = 20 000;
- NHD input bit-identical to HND;
- int8 Q·K as the fp16 default, and `SAGEATTN_SK1_INT8=0` giving the fp8 kernel bit for bit;
- the kernels switched off via `SAGEATTN_SK1_BACKEND=0` or `SAGEATTN_SK1_BF16=0`;
- head_dim 64 falling back to PR #368's kernel.
