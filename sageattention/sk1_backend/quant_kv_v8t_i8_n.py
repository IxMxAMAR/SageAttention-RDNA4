#!/usr/bin/env python3
"""The strided variant of the INT8 fused SK1 prologue: K per token in int8, V unchanged (fp8 e4m3),
read from a general (b,h)-strided view instead of a contiguous `(B*H, N, D)` tensor.

Why a separate module
---------------------
`quant_kv_v8t_i8.py`, `quant_kv_v8t.py`, `quant_kv_v8t_n.py` and `quant_triton.py` stay as they are.
This file imports the shared primitives and edits nothing:

  * K's int8 arithmetic is the same as `quant_kv_v8t_i8._quant_kv_pad_v8t_i8` (round-to-nearest-even
    by `floor` plus the exact half-even test, clamp [-128,127], scale `max(max|k|)/127`, floor
    `1e-12`, pad rows re-zeroed before the amax);
  * V's fp8 branch is the same as `quant_kv_v8t_n._quant_kv_pad_v8t_fp8_strided`;
  * the sequence mean is the same `kseq_mean_fp8_strided` first pass plus the same
    `quant_triton._kseq_finalize_mean` second pass, so it is bit-identical to the contiguous
    `kseq_mean_fp8` on the same data made contiguous.

The only difference from `quant_kv_v8t_i8.py` is the K/V read addressing: the plane base is
`b*stride_b + h*stride_h` and the row stride is explicit, so K and V are read straight out of
ComfyUI's non-contiguous HND `rearrange` view or NHD `(B,N,H,D)` with no copy. K and V are
re-written into K8/SK/V8T/SV by this kernel anyway, so the strided read moves the same number of
bytes as the contiguous one, and `K8/SK/V8T/SV` are bit-identical to the contiguous int8 prologue on
the same data.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from quant_triton import EPS, SCALE_MAX, _cvt2d, _next_pow2

__all__ = ["quant_kv_pad_v8t_i8_strided", "prologue_i8_fused_strided",
           "prologue_i8_fused_contig", "kseq_mean_fp8_strided", "n_pad_for"]

I8_MAX = 127.0


def n_pad_for(N: int) -> int:
    """The `do_pad_kv` contract: SK1's N_pad is a multiple of 64."""
    return ((int(N) + 63) // 64) * 64


# `kseq_mean_fp8_strided` is layout-generic (it reads `(b,h,row)` strides and never touches Q), so it
# is reused from the fp8 strided module rather than written again.
from quant_kv_v8t_n import kseq_mean_fp8_strided  # noqa: E402


@triton.jit
def _quant_kv_pad_v8t_i8_strided(K, V, K8, SK, V8T, SV, N, N_pad, D, H,
                                 k_sb, k_sh, k_sr, v_sb, v_sh, v_sr, FLOOR,
                                 BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                                 MEAN=None, SUB_MEAN: tl.constexpr = 0):
    """K -> (K8 int8, SK fp32) row-major padded; V -> (V8T e4m3, SV fp32) transposed padded.

    `K`/`V` are the logical `(B, H, N, D)` tensors with `stride(-1) == 1`; `k_sb`/`k_sh`/`k_sr` are
    their (batch, head, row) element strides.  grid = (cdiv(N_pad, BLOCK_N), B*H).  Every element of
    all four outputs is written, including the pad region (K8/V8T = 0, SK = 1.0, SV = 0.0), so the
    caller may pass `torch.empty`.

    Same arithmetic as `quant_kv_v8t_i8._quant_kv_pad_v8t_i8` (K) and
    `quant_kv_v8t_n._quant_kv_pad_v8t_fp8_strided` (V); only the K/V read addressing differs.
    """
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh - b * H
    kbase = b * k_sb + h * k_sh
    vbase = b * v_sb + h * v_sh
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)      # key index in [0, N_pad)
    cols = tl.arange(0, BLOCK_D)                   # head dim
    live = n < N
    dm = cols < D
    m = live[:, None] & dm[None, :]

    # ---- K: per-row int8, the same arithmetic as `quant_kv_v8t_i8` -----------------------------
    k = tl.load(K + kbase + n[:, None] * k_sr + cols[None, :], mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        mu = tl.load(MEAN + bh * D + cols, mask=dm, other=0.0)
        k = k - mu[None, :]
    # the pad rows must be re-zeroed before the amax, as in the contiguous int8 module
    k = tl.where(live[:, None], k, 0.0)
    sk = tl.maximum(tl.max(tl.abs(k), 1) / 127.0, FLOOR)
    tl.store(SK + bh * N_pad + n, tl.where(live, sk, 1.0), mask=n < N_pad)
    # round-to-nearest-even, exactly: up = d > 0.5 or (d == 0.5 and floor is odd)
    x = k / sk[:, None]
    fl = tl.floor(x)
    d = x - fl
    odd = (fl.to(tl.int32) & 1) == 1
    q = tl.where((d > 0.5) | ((d == 0.5) & odd), fl + 1.0, fl)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(K8 + ((bh * N_pad + n)[:, None] * D + cols[None, :]),
             q.to(tl.int8), mask=(n < N_pad)[:, None] & dm[None, :])

    # ---- V: per-ROW fp8, stored TRANSPOSED into V8T (B*H, D, N_pad) as in the fp8 file
    v = tl.load(V + vbase + n[:, None] * v_sr + cols[None, :], mask=m, other=0.0).to(tl.float32)
    v = tl.where(live[:, None], v, 0.0)
    sv = tl.maximum(tl.max(tl.abs(v), 1) / SCALE_MAX, EPS)
    tl.store(SV + bh * N_pad + n, tl.where(live, sv, 0.0), mask=n < N_pad)
    tl.store(V8T + ((bh * D + cols[None, :]) * N_pad + n[:, None]),
             _cvt2d(v / sv[:, None], BLOCK_N, BLOCK_D),
             mask=(n < N_pad)[:, None] & dm[None, :])


def quant_kv_pad_v8t_i8_strided(k, v, n_pad=None, block_n=64, block_d=None, mean=None,
                                floor=1e-12):
    """k, v: logical `(B,H,N,D)` fp16 with `stride(-1) == 1` and any (b,h,row) strides -> the four
    SK1 tensors.  Shapes: K8 (B*H, N_pad, D) int8; SK (B*H, N_pad) fp32; V8T (B*H, D, N_pad) e4m3;
    SV (B*H, N_pad) fp32.  Pad rows: K8/V8T = 0, SK = 1.0, SV = 0.0.
    """
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("quant_kv_pad_v8t_i8_strided expects (B,H,N,D); got %s / %s"
                         % (tuple(k.shape), tuple(v.shape)))
    B, H, N, D = k.shape
    if tuple(v.shape) != (B, H, N, D):
        raise ValueError("k and v shapes differ: %s vs %s" % (tuple(k.shape), tuple(v.shape)))
    if k.stride(3) != 1 or v.stride(3) != 1:
        raise ValueError("quant_kv_pad_v8t_i8_strided requires stride(-1)==1; got k=%r v=%r"
                         % (k.stride(3), v.stride(3)))
    BH = B * H
    n_pad = int(n_pad) if n_pad is not None else n_pad_for(N)
    if n_pad < N or n_pad % 64:
        raise ValueError("n_pad=%d must be >= N=%d and a multiple of 64" % (n_pad, N))
    bd = block_d or max(16, _next_pow2(D))
    k8 = torch.empty((BH, n_pad, D), dtype=torch.int8, device=k.device)
    sk = torch.empty((BH, n_pad), dtype=torch.float32, device=k.device)
    v8t = torch.empty((BH, D, n_pad), dtype=torch.float8_e4m3fn, device=k.device)
    sv = torch.empty((BH, n_pad), dtype=torch.float32, device=k.device)
    grid = (triton.cdiv(n_pad, block_n), BH)
    if mean is not None:
        _quant_kv_pad_v8t_i8_strided[grid](
            k, v, k8, sk, v8t, sv, N, n_pad, D, H,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), float(floor),
            MEAN=mean.reshape(-1, D), SUB_MEAN=1, BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    else:
        _quant_kv_pad_v8t_i8_strided[grid](
            k, v, k8, sk, v8t, sv, N, n_pad, D, H,
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2), float(floor),
            BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    return k8, sk, v8t, sv


def prologue_i8_fused_strided(k, v, n_pad=None, smooth_k=False, block_n=64, floor=1e-12):
    """K/V only -- Q is quantised by the SK1 kernel itself, so `q8`/`sq` are not computed.

    Returns `(k8, sk, v8t, sv)`.  `smooth_k=True` uses the strided sequence mean, which is
    bit-identical to the contiguous `kseq_mean_fp8` on the same data.
    """
    if smooth_k:
        mu = kseq_mean_fp8_strided(k, k.shape[1])
        return quant_kv_pad_v8t_i8_strided(k, v, n_pad=n_pad, block_n=block_n, mean=mu, floor=floor)
    return quant_kv_pad_v8t_i8_strided(k, v, n_pad=n_pad, block_n=block_n, floor=floor)


def prologue_i8_fused_contig(k, v, n_pad=None, smooth_k=False, block_n=64, floor=1e-12):
    """The contiguous-HND int8 prologue, K/V only.

    Same arithmetic as `quant_kv_v8t_i8.prologue_i8_fused` minus the discarded `quant_rows_fp8(q)`,
    and with the mean taken from the contiguous `kseq_mean_fp8`. Returns `(k8, sk, v8t, sv)`.
    """
    from quant_triton import kseq_mean_fp8
    from quant_kv_v8t_i8 import quant_kv_pad_v8t_i8
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("prologue_i8_fused_contig expects contiguous k/v")
    if smooth_k:
        mu = kseq_mean_fp8(k)
        return quant_kv_pad_v8t_i8(k, v, n_pad=n_pad, block_n=block_n, mean=mu,
                                   n_seq=k.shape[2], floor=floor)
    return quant_kv_pad_v8t_i8(k, v, n_pad=n_pad, block_n=block_n, floor=floor)
