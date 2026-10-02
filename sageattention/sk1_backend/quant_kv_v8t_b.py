#!/usr/bin/env python3
"""The bf16 K/V prologue: the same Triton kernel as the fp16 path, called with bf16 tensors.

There is no separate bf16 quantiser. Every load in `quant_kv_v8t._quant_kv_pad_v8t_fp8` and in
`quant_triton.kseq_mean_fp8` is followed by `.to(tl.float32)`, and a bf16 -> fp32 widening is exact
for every bf16 bit pattern. So the scales `sk = max(max|k|,1)/448` and `sv = max(max|v|,1)/448` and
the fp8 bytes `K8`/`V8T` come out bit-identical to what the same kernel produces for the same values
stored as fp16, whenever the values are inside fp16's range. Above 65504 only the bf16 form can
represent them at all.

Why this file exists, given the kernel is unchanged: `quant_kv_v8t.prologue_fp8_fused` also computes
`q8, sq = quant_rows_fp8(q)`, which the SK1 caller discards, so it costs one wasted launch per call.
This module is K/V only, like the strided `quant_kv_v8t_n.prologue_fp8_fused_strided` (which the
strided bf16 path reuses as is).

`quant_triton.py` and `quant_kv_v8t.py` are not edited.
"""
from __future__ import annotations

import torch

__all__ = ["prologue_bf16_fused", "n_pad_for"]


def prologue_bf16_fused(k, v, n_pad=None, smooth_k=False, block_n=64):
    """bf16 `(B,H,N,D)` contiguous k/v -> `(k8, sk, v8t, sv)`, via the fp16 fused kernel.

    `smooth_k=True` uses the same `kseq_mean_fp8(k)` mean the shipped `prologue_fp8_fused` uses, so
    the K bytes are comparable on that axis. The pad rows are written by the kernel itself
    (`K8`/`V8T` = 0, `SK` = 1.0, `SV` = 0.0).
    """
    from quant_triton import kseq_mean_fp8
    from quant_kv_v8t import quant_kv_pad_v8t_fp8, n_pad_for as _npf

    if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise ValueError("prologue_bf16_fused expects bf16 k/v; got %s / %s" % (k.dtype, v.dtype))
    if k.shape != v.shape:
        raise ValueError("k and v shapes differ: %s vs %s" % (tuple(k.shape), tuple(v.shape)))
    k, v = k.contiguous(), v.contiguous()
    if smooth_k:
        mu = kseq_mean_fp8(k)
        return quant_kv_pad_v8t_fp8(k, v, n_pad=n_pad, block_n=block_n, mean=mu,
                                    n_seq=k.shape[2])
    return quant_kv_pad_v8t_fp8(k, v, n_pad=n_pad, block_n=block_n)


def n_pad_for(N: int) -> int:
    """The `do_pad_kv` contract: SK1's N_pad is a multiple of 64."""
    return ((int(N) + 63) // 64) * 64
