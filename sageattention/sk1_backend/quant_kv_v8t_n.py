#!/usr/bin/env python3
"""The strided variant of the fused SK1 prologue: K/V read from a general (b,h)-strided view
instead of a contiguous `(B*H, N, D)` tensor.

Why a separate module
---------------------
`quant_kv_v8t.py` and `quant_triton.py` stay as they are. This file imports the shared primitives
(`_cvt2d`, `SCALE_MAX`, `EPS`, `_next_pow2`, `_kseq_finalize_mean`) and adds nothing to them.

What changes, and why it is free
--------------------------------
The contiguous prologue addresses K/V as `(bh*N + n)*stride_r + cols*stride_d` and therefore
requires `k.reshape(BH, N, D)` to be a view, i.e. `k` must be contiguous `(B,H,N,D)`. ComfyUI sends
either a non-contiguous HND `rearrange` view (strides `(N*H*D, D, H*D, 1)`) or NHD `(B, N, H, D)`.
Here the plane base is `b*stride_b + h*stride_h` and the row stride is explicit, so the same kernel
reads either layout with no copy at all. K and V are re-written into K8/SK/V8T/SV by this kernel
anyway, so the strided read moves exactly the same number of bytes as the contiguous one.

The arithmetic is copied from `_quant_kv_pad_v8t_fp8` verbatim (same masked load with `other=0.0`,
same `max(|x|,1)/448`, same `x/s[:,None]`, same hardware pack `_cvt2d`, same pad-region writes), so
the four output tensors are bit-identical to the contiguous path on the same data.

The sequence-axis mean (`smooth_k`) is the same two-stage, atomics-free reduction as
`quant_triton.kseq_mean_fp8` (F065), with the same CHUNK/BLOCK_D grid and the same summation order,
so the mean is bit-identical too; stage 2 is the same `_kseq_finalize_mean` kernel.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from quant_triton import EPS, SCALE_MAX, _cvt2d, _kseq_finalize_mean, _next_pow2

__all__ = ["quant_kv_pad_v8t_strided", "prologue_fp8_fused_strided", "kseq_mean_fp8_strided",
           "n_pad_for"]


@triton.jit
def _quant_kv_pad_v8t_fp8_strided(K, V, K8, SK, V8T, SV, N, N_pad, D, H,
                                  k_sb, k_sh, k_sr, v_sb, v_sh, v_sr,
                                  BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                                  MEAN=None, SUB_MEAN: tl.constexpr = 0):
    """K and V -> (K8, SK) row-major padded, (V8T, SV) transposed padded, in ONE pass over V.

    `K`/`V` are the logical `(B, H, N, D)` tensors with `stride(-1) == 1`; `k_sb`/`k_sh`/`k_sr` are
    their (batch, head, row) element strides.  grid = (cdiv(N_pad, BLOCK_N), B*H).  Every element of
    all four outputs is written, including the pad region, so the caller may pass `torch.empty`.

    Identical arithmetic to `quant_kv_v8t._quant_kv_pad_v8t_fp8`; only the K/V read addressing
    differs.
    """
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh - b * H
    kbase = b * k_sb + h * k_sh
    vbase = b * v_sb + h * v_sh
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)      # key index in [0, N_pad)
    cols = tl.arange(0, BLOCK_D)                   # head dim
    live = n < N                                   # a real key (not pad)
    dm = cols < D
    m = live[:, None] & dm[None, :]

    # ---- K: per-ROW fp8, exactly `_quant_rows_fp8`'s arithmetic -------------------------------
    k = tl.load(K + kbase + n[:, None] * k_sr + cols[None, :], mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        mu = tl.load(MEAN + bh * D + cols, mask=dm, other=0.0)
        k = k - mu[None, :]
    # The pad rows must be re-zeroed (see `quant_kv_v8t.py`).
    k = tl.where(live[:, None], k, 0.0)
    sk = tl.maximum(tl.max(tl.abs(k), 1) / SCALE_MAX, EPS)
    tl.store(SK + bh * N_pad + n, tl.where(live, sk, 1.0), mask=n < N_pad)
    tl.store(K8 + ((bh * N_pad + n)[:, None] * D + cols[None, :]),
             _cvt2d(k / sk[:, None], BLOCK_N, BLOCK_D),
             mask=(n < N_pad)[:, None] & dm[None, :])

    # ---- V: per-ROW fp8, stored TRANSPOSED into V8T (B*H, D, N_pad) ---------------------------
    v = tl.load(V + vbase + n[:, None] * v_sr + cols[None, :], mask=m, other=0.0).to(tl.float32)
    v = tl.where(live[:, None], v, 0.0)
    sv = tl.maximum(tl.max(tl.abs(v), 1) / SCALE_MAX, EPS)
    tl.store(SV + bh * N_pad + n, tl.where(live, sv, 0.0), mask=n < N_pad)
    tl.store(V8T + ((bh * D + cols[None, :]) * N_pad + n[:, None]),
             _cvt2d(v / sv[:, None], BLOCK_N, BLOCK_D),
             mask=(n < N_pad)[:, None] & dm[None, :])


@triton.jit
def _kseq_partial_sum_strided(X, P, N_SEQ, D, H, stride_b, stride_h, stride_r,
                              CHUNK: tl.constexpr, BLOCK_D: tl.constexpr):
    """Stage 1 of the sequence-axis mean for a general (b,h)-strided `(B,H,N,D)` view.

    Grid `(B*H, D_blocks, n_chunks)`.  Writes `P[(bh, d, chunk)]`.  DETERMINISTIC BY CONSTRUCTION
    (no atomics), exactly as `quant_triton._kseq_partial_sum`.
    """
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    ck = tl.program_id(2)
    b = bh // H
    h = bh - b * H
    base = b * stride_b + h * stride_h
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r in range(0, CHUNK, 16):
        rows = ck * CHUNK + r + tl.arange(0, 16)
        rm = rows < N_SEQ
        m = rm[:, None] & dm[None, :]
        x = tl.load(X + base + rows[:, None] * stride_r + cols[None, :],
                    mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    n_ch = tl.cdiv(N_SEQ, CHUNK)
    tl.store(P + (bh * n_ch + ck) * D + cols, acc, mask=dm)


def kseq_mean_fp8_strided(k, H, block_d=None, chunk=1024):
    """`quant_triton.kseq_mean_fp8` for a general (b,h)-strided `(B,H,N,D)` tensor.

    Same CHUNK/BLOCK_D grid, same partial-then-reduce order, and the SAME
    `_kseq_finalize_mean` stage 2, so the result is bit-identical to `kseq_mean_fp8` on the same
    data made contiguous.  No copy: unlike `kseq_mean_fp8` (which calls `.contiguous()` on a
    non-contiguous `reshape`) this reads the strided view directly.
    """
    B, Hh, N, D = k.shape
    if Hh != H:
        raise ValueError("kseq_mean_fp8_strided: H=%r but k.shape[1]=%r" % (H, Hh))
    if k.stride(3) != 1:
        raise ValueError("kseq_mean_fp8_strided requires stride(-1)==1; got %r" % (k.stride(3),))
    BH = B * Hh
    bd = block_d or max(16, _next_pow2(D))
    n_ch = triton.cdiv(N, chunk)
    part = torch.empty((BH, n_ch, D), device=k.device, dtype=torch.float32)
    mean = torch.empty((BH, D), device=k.device, dtype=torch.float32)
    _kseq_partial_sum_strided[(BH, triton.cdiv(D, bd), n_ch)](
        k, part, N, D, H, k.stride(0), k.stride(1), k.stride(2),
        CHUNK=chunk, BLOCK_D=bd, num_warps=4)
    _kseq_finalize_mean[(BH, triton.cdiv(D, bd))](
        part, mean, N, D, n_ch, BLOCK_D=bd, num_warps=4)
    return mean.view(B, Hh, D)


def n_pad_for(N: int) -> int:
    """The `do_pad_kv` contract: SK1's N_pad is a multiple of 64."""
    return ((int(N) + 63) // 64) * 64


def quant_kv_pad_v8t_strided(k, v, n_pad=None, block_n=64, block_d=None, mean=None):
    """k, v: logical `(B,H,N,D)` fp16 with `stride(-1) == 1` and any (b,h,row) strides -> the four
    SK1 tensors.  Shapes: K8 (B*H, N_pad, D) e4m3; SK (B*H, N_pad) fp32; V8T (B*H, D, N_pad) e4m3;
    SV (B*H, N_pad) fp32.  Pad rows: K8/V8T = 0, SK = 1.0, SV = 0.0.
    """
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("quant_kv_pad_v8t_strided expects (B,H,N,D); got %s / %s"
                         % (tuple(k.shape), tuple(v.shape)))
    B, H, N, D = k.shape
    if tuple(v.shape) != (B, H, N, D):
        raise ValueError("k and v shapes differ: %s vs %s" % (tuple(k.shape), tuple(v.shape)))
    if k.stride(3) != 1 or v.stride(3) != 1:
        raise ValueError("quant_kv_pad_v8t_strided requires stride(-1)==1; got k=%r v=%r"
                         % (k.stride(3), v.stride(3)))
    BH = B * H
    n_pad = int(n_pad) if n_pad is not None else n_pad_for(N)
    if n_pad < N or n_pad % 64:
        raise ValueError("n_pad=%d must be >= N=%d and a multiple of 64" % (n_pad, N))
    bd = block_d or max(16, _next_pow2(D))
    k8 = torch.empty((BH, n_pad, D), dtype=torch.float8_e4m3fn, device=k.device)
    sk = torch.empty((BH, n_pad), dtype=torch.float32, device=k.device)
    v8t = torch.empty((BH, D, n_pad), dtype=torch.float8_e4m3fn, device=k.device)
    sv = torch.empty((BH, n_pad), dtype=torch.float32, device=k.device)
    grid = (triton.cdiv(n_pad, block_n), BH)
    if mean is not None:
        _quant_kv_pad_v8t_fp8_strided[grid](
            k, v, k8, sk, v8t, sv, N, n_pad, D, H,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            MEAN=mean.reshape(-1, D), SUB_MEAN=1, BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    else:
        _quant_kv_pad_v8t_fp8_strided[grid](
            k, v, k8, sk, v8t, sv, N, n_pad, D, H,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    return k8, sk, v8t, sv


def prologue_fp8_fused_strided(k, v, n_pad=None, smooth_k=False, block_n=64):
    """K/V only -- Q is quantised by the SK1 kernel itself and `prologue_fp8_fused`'s `q8`/`sq`
    outputs are discarded by `try_sk1_t1` (`__init__.py`), so this path does not compute them.

    Returns `(k8, sk, v8t, sv)`.  `smooth_k=True` uses the strided sequence mean above, which is
    bit-identical to the contiguous `kseq_mean_fp8` on the same data.
    """
    if smooth_k:
        mu = kseq_mean_fp8_strided(k, k.shape[1])
        return quant_kv_pad_v8t_strided(k, v, n_pad=n_pad, block_n=block_n, mean=mu)
    return quant_kv_pad_v8t_strided(k, v, n_pad=n_pad, block_n=block_n)
