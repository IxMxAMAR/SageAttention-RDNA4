# SageAttention-RDNA4

Quantized attention for AMD Radeon RX 9070 / RX 9070 XT (RDNA4, `gfx1201`) on Windows, and on Linux
when built from source.

This is a drop-in build of the `sageattention` package. Underneath it is SageAttention 2.2 and the
gfx12 port from [SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368). On top of
that sit hand-written HIP attention kernels that take over the common case: head_dim 128, in fp16
or bf16. Against PR #368's own kernels they are **1.5–2.0× faster on causal attention and
1.06–1.3× faster on non-causal attention** in fp16, and **2.2× and 1.3× faster** in bf16, measured
over the full call.

fp16 calls compute Q·K in int8 with one scale per token, which is more precise than fp8 at about the
same speed. bf16 calls compute Q·K in fp8.

ComfyUI users get it through the normal `--use-sage-attention` switch. No node and no code change
is needed.

---

## Results

### fp16, against PR #368, full call (quantization included)

Time of this package divided by time of PR #368 on the same tensors. Below 1.0 means faster.
B=1, H=48, D=128. Each cell is the median of interleaved runs in one process. Measured with the
fp8 Q·K kernel that was the default in v0.1.0 (`SAGEATTN_SK1_INT8=0` today).

| N | causal | causal + smooth_k | non-causal | non-causal + smooth_k |
|---|---|---|---|---|
| 8768 | 0.56 | 0.51 | 0.81 | 0.89 |
| 8771 | 0.56 | 0.51 | 0.77 | 0.82 |
| 16384 | 0.65 | 0.58 | 0.85 | 0.94 |
| 47520 | 0.64 | 0.56 | 0.85 | 0.92 |

The int8 Q·K default against that fp8 kernel, `smooth_k` on:

| N | non-causal | causal |
|---|---|---|
| 8771 (Krea2) | 1.000 | 0.983 |
| 47520 (720p video) | 1.017 | 1.010 |

At image sizes nothing changes. On long, video-sized sequences int8 gives back up to 1.7 % of the
lead. `SAGEATTN_SK1_INT8=0` takes it back if you prefer speed there.

### bf16, against PR #368's bf16 path, `smooth_k` on

| N | non-causal | causal |
|---|---|---|
| 8771 | 0.76 | 0.44 |
| 47520 | 0.77 | 0.45 |

When the GPU's memory is nearly full, PR #368's bf16 path slows down by up to 1.9× and this one
does not, so on a full card the gap is wider. The table is measured without that pressure.

All of these are ratios. On this card a software clock check can confirm that the clock did not
move during a block, but it cannot pin the absolute clock. So the absolute milliseconds are not
quoted here and the ratios are. [docs/METHOD.md](docs/METHOD.md) explains why that matters and
how each cell was measured.

### In a real render (ComfyUI's Krea2 model, 8 steps)

Seconds per sampling step: median of 9 renders (3 prompts × 3 seeds). Each attention backend was
called exactly as stock ComfyUI calls it, so every package used its own defaults.

fp16, a Krea2 int8 fine-tune with no LoRAs:

| | 1 MP | 2 MP (1448²) |
|---|---|---|
| PyTorch SDPA | 1.029 s | 2.387 s |
| SageAttention 1.x (community RDNA4 build) | 0.997 s | 2.167 s |
| **This package** | **0.973 s** | **2.059 s** |
| attention time per call (this / SDPA) | 5.0 / 9.9 ms | 14.1 / 29.6 ms |

These were taken with the fp8 kernel. Switching to the int8 default changed the step time by
−0.3 % at 1 MP and −0.05 % at 2 MP.

bf16, Krea2 Turbo (fp8-scaled weights, computing in bf16):

| | 1 MP | 2 MP |
|---|---|---|
| PyTorch SDPA | 1.156 s | 2.567 s |
| SageAttention 1.x (community RDNA4 build) | 1.113 s | 2.330 s |
| **This package** | **1.076 s** | **2.202 s** |

The hand-written kernels served every self-attention call in these renders. Most of a step is
spent outside attention, in the model's matmuls, which is why a 2× faster attention call becomes
a 5–14 % faster step.

### Accuracy

For fp16, Q and K are quantized to int8 and V to fp8 (e4m3), each with **one scale per token**,
and the attention weights are folded into fp8 before the P·V product. For bf16, Q and K are fp8
per token instead. Measured on real activations captured from Krea2 and Flux2-Klein:

- **No bad regime.** PR #368 quantizes Q·K to int8 with one K scale per block of 64 tokens and one
  Q scale per 32 rows. On layers with a few extreme keys, such as Krea2's first block, one outlier
  sets the scale for the whole block, and its error was up to 200× this package's. Per-token
  scaling does not have that failure mode. On ten real captures, per-token int8 with `smooth_k`
  was more precise than PR #368 on all ten and than per-token fp8 on nine (F154). Inside real
  Krea2 renders its per-call error is 0.82× the fp8 kernel's (F157).
- **`smooth_k` is on by default** and should stay on. Subtracting the per-head key mean before
  quantizing is exact in real arithmetic, and it reduced the fp8 kernel's error on 10 of 10 real
  captures, by 3.2× on average (F063). The cost is 0.4–4 % per call.
- **bf16** error is within 2.4 % of the fp16 fp8 kernel's on every test cell. With values large
  enough to overflow, it produces non-finite output only where exact fp32 attention overflows too
  (F158, F159).

In the Krea2 render tests the images from this package were closer to full-precision attention
than SageAttention 1.x's: median PSNR 18.5 vs 13.5 dB at 1 MP and 21.7 vs 15.6 dB at 2 MP in fp16,
and 23.0 vs 15.0 dB and 18.7 vs 15.0 dB in bf16. Take per-image numbers from an 8-step sampler
with care, though. A change of 3·10⁻⁶ in the text conditioning alone moves the final latent by
4–58 %, so single-image differences are mostly trajectory divergence, not precision.

---

## Install

The prebuilt wheel targets exactly this stack:

- Windows 10/11, Python 3.12
- AMD Radeon RX 9070 or RX 9070 XT (`gfx1201`)
- PyTorch `2.13.0+rocm10.0.0` (the ROCm Python packages AMD publishes for gfx120X)

Download the wheel from [Releases](../../releases) and install it into the Python environment
that runs your models. For ComfyUI that means ComfyUI's own venv. Close ComfyUI first.

```
python -m pip install --no-deps sageattention-2.2.0+amd.gfx12.2-cp312-cp312-win_amd64.whl
```

`--no-deps` keeps pip from touching your PyTorch install. The compiled extensions are built
against that exact PyTorch version. For any other version, build from source (see below).

### Linux

There is no prebuilt Linux wheel, but the kernels load on Linux too, thanks to
[boxwrench](https://github.com/boxwrench) ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)). It was tested on a Radeon AI PRO R9700, which is
also `gfx1201`, under Ubuntu 24.04 with PyTorch `2.9.1+rocm7.2.1`. Build from source with
`PYTORCH_ROCM_ARCH=gfx1201`. [docs/BUILD.md](docs/BUILD.md) describes the Windows build. The
package's runtime check expects PyTorch `2.13.0+rocm10.0.0`, the version the Windows wheel is
built against, so set `SAGEATTENTION_ALLOW_UNVERIFIED_TORCH=1` when you build against any other.

## Use

```python
from sageattention import sageattn

out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)   # q, k, v: (B, H, N, D)
```

ComfyUI: start it with `--use-sage-attention`.

The hand-written kernels are used automatically when **all** of these hold. Anything else goes to
the gfx12 kernel from PR #368, with the same results contract:

- the GPU is `gfx1201`, the HIP runtime resolves, and the code object loads;
- fp16 or bf16 inputs and head_dim 128;
- HND or NHD layout (strided views are fine);
- query length equal to key length, either fully causal or non-causal;
- no attention mask, `return_lse=False`, `smooth_v=False`.

An fp16 call tries the int8 Q·K kernel first, then the fp8 kernel, then PR #368's. A bf16 call
tries the bf16 kernel, then PR #368's.

| environment variable | effect |
|---|---|
| `SAGEATTN_SK1_BACKEND` unset | default: use the kernels where they apply, fall back silently elsewhere |
| `SAGEATTN_SK1_BACKEND=0` | never use them |
| `SAGEATTN_SK1_BACKEND=1` | require them: raise instead of falling back when they cannot load |
| `SAGEATTN_SK1_INT8` unset | default: int8 Q·K for fp16 calls |
| `SAGEATTN_SK1_INT8=0` | fp8 Q·K for fp16 calls, bit-identical to v0.1.0 |
| `SAGEATTN_SK1_INT8=1` | require int8: raise when an fp16 call cannot use it |
| `SAGEATTN_SK1_BF16` unset | default: bf16 calls use the hand-written kernel |
| `SAGEATTN_SK1_BF16=0` | bf16 calls go to PR #368's path, as in v0.1.0 |
| `SAGEATTN_SK1_BF16=1` | require it: raise when a bf16 call cannot use it |

The variables are read once, when the package is imported. `SAGEATTN_SK1_BACKEND=0` turns all of
them off. The same switches exist per call: `sageattn(..., sk1_backend=, sk1_int8=, sk1_bf16=)`.

If the kernels cannot be used at all, for example on a different GPU, the package logs one line
per process and carries on with the fallback.

## Build from source

[docs/BUILD.md](docs/BUILD.md) covers the native extensions (`setup.py`, ROCm SDK,
`PYTORCH_ROCM_ARCH=gfx1201`) and rebuilding the kernels' code objects from `kernels/hip/` with
`tools/build_hsaco.py`.

## How this was built

- [docs/JOURNEY.md](docs/JOURNEY.md): the whole path, from a Triton kernel that lost to PR #368
  to hand-written kernels that beat it, including the dead ends.
- [docs/FINDINGS.md](docs/FINDINGS.md): every finding, confirmed or refuted, with its numbers.
- [docs/METHOD.md](docs/METHOD.md): how measurements were taken, and the mistakes that shaped
  those rules.

## Limitations

- The prebuilt kernels target `gfx1201` only. Other GPUs use the fallback path.
- Every speed and accuracy number here was measured on Windows. On Linux the kernels load and the
  test suite passes, but nothing has been timed there yet.
- head_dim 128 only. A head_dim 64 kernel was built and tested, but it was slower than PR #368's,
  so head_dim 64 still uses the fallback (F156).
- On long sequences (around 47k tokens, non-causal) the int8 default is up to 1.7 % slower per
  attention call than the fp8 kernel.
- End-to-end validation so far is one model family (Krea2), in fp16 and bf16. The kernels
  themselves are checked bit for bit across shapes and layouts, and against an fp64 reference.

## Credits and license

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

- [SageAttention](https://github.com/thu-ml/SageAttention) by the THU-ML group: the
  quantized-attention method and the package this builds on.
- The gfx12 port in [PR #368](https://github.com/thu-ml/SageAttention/pull/368) by DELUXA, which
  provides the native fallback kernels and the baseline this work is measured against.
- [boxwrench](https://github.com/boxwrench), for Linux support in the kernel loader ([#1](https://github.com/IxMxAMAR/SageAttention-RDNA4/pull/1)).
