# Triton fp8 attention kernel (research)

`fa_fp8.py` is the Triton fp8 flash-attention kernel for gfx1201 that came before the hand-written
HIP kernel in `kernels/hip/`. `quant_triton.py` is the quantiser module it imports (it is the same
file the package ships as `sageattention/sk1_backend/quant_triton.py`).

This code is not part of the installed package and nothing in `sageattention/` imports `fa_fp8.py`.
It is kept for reference: the flags and comments record what was tried (fp8 P conversion variants,
emulated exp2, lazy softmax rescale, per-channel V scaling, int8 QK, tail handling) and the
measured result of each, including several that did not pay off.

The kernel needs a ROCm build of torch and Triton on a gfx1201 GPU. `fa_fp8.py` imports
`quant_triton` as a top-level module, so run it with this directory on `PYTHONPATH`:

```
cd research/triton
python -c "import fa_fp8"
```
