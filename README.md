# SageAttention-RDNA4

Quantized attention for AMD Radeon RX 9070 / RX 9070 XT (RDNA4, `gfx1201`) on Windows.

This is a drop-in build of the `sageattention` package. Underneath it is SageAttention 2.2 and the
gfx12 port from [SageAttention PR #368](https://github.com/thu-ml/SageAttention/pull/368). On top of
that sits a hand-written HIP fp8 attention kernel that takes over the common case: fp16,
head_dim 128. Against PR #368's own kernel it is **1.5–2.0× faster on causal attention and
1.06–1.3× faster on non-causal attention**, measured over the full call.

ComfyUI users get it through the normal `--use-sage-attention` switch. No node and no code change
is needed.

---

## Results

### Against PR #368, full call (quantization included)

Time of this package divided by time of PR #368 on the same tensors. Below 1.0 means faster.
B=1, H=48, D=128, fp16. Each cell is the median of interleaved runs in one process.

| N | causal | causal + smooth_k | non-causal | non-causal + smooth_k |
|---|---|---|---|---|
| 8768 | 0.56 | 0.51 | 0.81 | 0.89 |
| 8771 | 0.56 | 0.51 | 0.77 | 0.82 |
| 16384 | 0.65 | 0.58 | 0.85 | 0.94 |
| 47520 | 0.64 | 0.56 | 0.85 | 0.92 |

All of these are ratios. On this card a software clock check can confirm that the clock did not
move during a block, but it cannot pin the absolute clock. So the absolute milliseconds are not
quoted here and the ratios are. [docs/METHOD.md](docs/METHOD.md) explains why that matters and
how each cell was measured.

### In a real render (ComfyUI's Krea2 model, 8 steps)

Seconds per sampling step: median of 9 renders (3 prompts × 3 seeds). Each attention backend was
called exactly as stock ComfyUI calls it, so every package used its own defaults. The model was a
Krea2 int8 fine-tune running in fp16, with no LoRAs.

| | 1 MP | 2 MP (1448²) |
|---|---|---|
| PyTorch SDPA | 1.029 s | 2.387 s |
| SageAttention 1.x (community RDNA4 build) | 0.997 s | 2.167 s |
| **This package** | **0.973 s** | **2.059 s** |
| attention time per call (this / SDPA) | 5.0 / 9.9 ms | 14.1 / 29.6 ms |

The hand-written kernel served every self-attention call in these renders. Most of a step is
spent outside attention, in the model's matmuls, which is why a 2× faster attention call becomes
a 5–14 % faster step.

### Accuracy

Q, K and V are quantized to fp8 (e4m3) with **one scale per token**, and the attention weights
are folded into fp8 before the P·V product. Two consequences, both measured on real activations
captured from Krea2 and Flux2-Klein:

- **No bad regime.** PR #368 computes Q·K in int8, with one K scale per block of 64 tokens. On
  well-behaved layers that is more precise than this package. On layers with a few extreme keys,
  such as Krea2's first block, one outlier sets the scale for all 64 tokens, and its error was up
  to 55× this package's. Per-token scaling does not have that failure mode: the
  error stays flat across blocks, models, timesteps and resolutions. Details are in
  [docs/FINDINGS.md](docs/FINDINGS.md) (F058).
- **`smooth_k` is on by default** and should stay on. Subtracting the per-head key mean before
  quantizing is exact in real arithmetic, and it reduced this kernel's error on 10 of 10 real
  captures, by 3.2× on average (F063). The cost is 0.4–4 % per call.

In the Krea2 render test the images from this package were closer to full-precision attention
than SageAttention 1.x's: median PSNR 18.5 vs 13.5 dB at 1 MP and 21.7 vs 15.6 dB at 2 MP. Take
per-image numbers from an 8-step sampler with care, though. A change of 3·10⁻⁶ in the text
conditioning alone moves the final latent by 4–58 %, so single-image differences are mostly
trajectory divergence, not precision.

---

## Install

The prebuilt wheel targets exactly this stack:

- Windows 10/11, Python 3.12
- AMD Radeon RX 9070 or RX 9070 XT (`gfx1201`)
- PyTorch `2.13.0+rocm10.0.0` (the ROCm Python packages AMD publishes for gfx120X)

Download the wheel from [Releases](../../releases) and install it into the Python environment
that runs your models. For ComfyUI that means ComfyUI's own venv. Close ComfyUI first.

```
python -m pip install --no-deps sageattention-2.2.0+amd.gfx12.1-cp312-cp312-win_amd64.whl
```

`--no-deps` keeps pip from touching your PyTorch install. The compiled extensions are built
against that exact PyTorch version. For any other version, build from source (see below).

## Use

```python
from sageattention import sageattn

out = sageattn(q, k, v, tensor_layout="HND", is_causal=False)   # q, k, v: (B, H, N, D)
```

ComfyUI: start it with `--use-sage-attention`.

The hand-written kernel is used automatically when **all** of these hold. Anything else goes to
the gfx12 kernel from PR #368, with the same results contract:

- the GPU is `gfx1201`, the HIP runtime resolves, and the code object loads;
- fp16 inputs and head_dim 128;
- HND or NHD layout (strided views are fine);
- query length equal to key length, either fully causal or non-causal;
- no attention mask, `return_lse=False`, `smooth_v=False`.

| environment variable | effect |
|---|---|
| `SAGEATTN_SK1_BACKEND` unset | default: use the kernel where it applies, fall back silently elsewhere |
| `SAGEATTN_SK1_BACKEND=0` | never use it |
| `SAGEATTN_SK1_BACKEND=1` | require it: raise instead of falling back when it cannot load |

If the kernel cannot be used at all, for example on a different GPU, the package logs one line
per process and carries on with the fallback.

## Build from source

[docs/BUILD.md](docs/BUILD.md) covers the native extensions (`setup.py`, ROCm SDK,
`PYTORCH_ROCM_ARCH=gfx1201`) and rebuilding the kernel's code object from `kernels/hip/` with
`tools/build_hsaco.py`.

## How this was built

- [docs/JOURNEY.md](docs/JOURNEY.md): the whole path, from a Triton kernel that lost to PR #368
  to a hand-written kernel that beats it, including the dead ends.
- [docs/FINDINGS.md](docs/FINDINGS.md): every finding, confirmed or refuted, with its numbers.
- [docs/METHOD.md](docs/METHOD.md): how measurements were taken, and the mistakes that shaped
  those rules.

## Limitations

- The prebuilt kernel targets `gfx1201` only. Other GPUs use the fallback path.
- Tested on Windows only.
- The hand-written kernel handles fp16 and head_dim 128. bf16 models and other head dims use the
  fallback.
- End-to-end validation so far is one model family (Krea2). The kernel itself is checked
  bit-for-bit across shapes and layouts, and against an fp64 reference.

## Credits and license

Apache-2.0, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

- [SageAttention](https://github.com/thu-ml/SageAttention) by the THU-ML group: the
  quantized-attention method and the package this builds on.
- The gfx12 port in [PR #368](https://github.com/thu-ml/SageAttention/pull/368) by DELUXA, which
  provides the native fallback kernels and the baseline this work is measured against.
