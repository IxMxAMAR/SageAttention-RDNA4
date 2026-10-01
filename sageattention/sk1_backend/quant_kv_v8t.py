#!/usr/bin/env python3
"""The fused SK1 prologue: K/V quantised, K zero-padded, V written already transposed.

Why this file is separate from `quant_triton.py`
------------------------------------------------
Adding the kernel to `quant_triton.py` would change that file, which earlier bit-identity checks
compare against. So this module imports the shared primitives (`_cvt2d`, `SCALE_MAX`, `EPS`,
`_next_pow2`) and adds nothing to them.

What the unfused path costs, and what this removes
--------------------------------------------------
`kernels/hip/sk1.hip` needs, per (b,h):
  K8   (N_pad, 128)  e4m3, rows >= N zero
  SK   (N_pad,)      fp32, rows >= N set to 1.0   (the `do_pad_kv` contract; the pad value is
                                                   irrelevant because K8 is 0 there)
  V8T  (128, N_pad)  e4m3, V TRANSPOSED, rows >= N zero
  SV   (N_pad,)      fp32, rows >= N set to 0.0
The unfused path produced these with `prologue_fp8` (3 launches of `_quant_rows_fp8`), four torch
pad ops (`_pad_tokens`) and one `transpose(-1,-2).contiguous()`. At (1,48,8771,128) that is about
486 MB of traffic; the fused kernel moves about 162 MB and issues one launch.

Bit-identity argument
---------------------
Every arithmetic step is copied from `_quant_rows_fp8` verbatim: the same masked load with
`other=0.0`, the same `s = max(max(|x|,1)/448, 1e-12)` (a max reduction, which is exact and
order-independent), the same `x / s[:, None]`, the same hardware pack `_cvt2d`. The pad regions are
written by the kernel itself: rows >= N load as 0.0, so `K8`/`V8T` are 0 and `SV` is 0.0, and `SK`
is written explicitly as 1.0. `V8T[bh, d, n] = v8[bh, n, d]`, exactly the
`transpose(-1,-2).contiguous()` of the unfused path.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from quant_triton import EPS, SCALE_MAX, _cvt2d, _next_pow2

__all__ = ["quant_kv_pad_v8t_fp8", "prologue_fp8_fused", "n_pad_for"]


@triton.jit
def _quant_kv_pad_v8t_fp8(K, V, K8, SK, V8T, SV, N, N_pad, D,
                          stride_kr, stride_kd, stride_vr, stride_vd,
                          BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                          MEAN=None, SUB_MEAN: tl.constexpr = 0):
    """K and V -> (K8, SK) row-major padded, (V8T, SV) transposed padded, in ONE pass over V.

    grid = (cdiv(N_pad, BLOCK_N), B*H).  Every element of all four outputs is written by this
    kernel, including the pad region, so the caller may pass `torch.empty` buffers.
    """
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)      # key index in [0, N_pad)
    cols = tl.arange(0, BLOCK_D)                   # head dim
    live = n < N                                   # a real key (not pad)
    dm = cols < D
    m = live[:, None] & dm[None, :]

    # ---- K: per-ROW fp8, exactly `_quant_rows_fp8`'s arithmetic -------------------------------
    k = tl.load(K + ((bh * N + n)[:, None] * stride_kr + cols[None, :] * stride_kd),
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        mu = tl.load(MEAN + bh * D + cols, mask=dm, other=0.0)
        k = k - mu[None, :]
    # The pad rows must be re-zeroed: with SUB_MEAN they arrive as `-mu`, not 0, and the unfused
    # path's `_pad_tokens(k8, fill=0)` writes zeros there. SK1's output is the same either way (the
    # padded keys are fully masked), so only a tensor-level comparison of K8 can see it.
    k = tl.where(live[:, None], k, 0.0)
    sk = tl.maximum(tl.max(tl.abs(k), 1) / SCALE_MAX, EPS)
    tl.store(SK + bh * N_pad + n, tl.where(live, sk, 1.0), mask=n < N_pad)
    tl.store(K8 + ((bh * N_pad + n)[:, None] * D + cols[None, :]),
             _cvt2d(k / sk[:, None], BLOCK_N, BLOCK_D),
             mask=(n < N_pad)[:, None] & dm[None, :])

    # ---- V: per-ROW fp8, stored TRANSPOSED into V8T (B*H, D, N_pad) ---------------------------
    v = tl.load(V + ((bh * N + n)[:, None] * stride_vr + cols[None, :] * stride_vd),
                mask=m, other=0.0).to(tl.float32)
    v = tl.where(live[:, None], v, 0.0)
    sv = tl.maximum(tl.max(tl.abs(v), 1) / SCALE_MAX, EPS)
    tl.store(SV + bh * N_pad + n, tl.where(live, sv, 0.0), mask=n < N_pad)
    tl.store(V8T + ((bh * D + cols[None, :]) * N_pad + n[:, None]),
             _cvt2d(v / sv[:, None], BLOCK_N, BLOCK_D),
             mask=(n < N_pad)[:, None] & dm[None, :])


def n_pad_for(N: int) -> int:
    """The `do_pad_kv` contract: SK1's N_pad is a multiple of 64."""
    return ((int(N) + 63) // 64) * 64


def quant_kv_pad_v8t_fp8(k, v, n_pad=None, block_n=64, block_d=None,
                         mean=None, n_seq=None, out=None):
    """k, v: (B,H,N,D) contiguous fp16 -> (K8, SK, V8T, SV).

    Shapes: K8 (B*H, N_pad, D) e4m3; SK (B*H, N_pad) fp32; V8T (B*H, D, N_pad) e4m3;
    SV (B*H, N_pad) fp32.  Pad rows: K8/V8T = 0, SK = 1.0, SV = 0.0.

    `mean`/`n_seq` apply `smooth_k` to K only, exactly as `quant_rows_fp8(mean=...)` does; the mean
    is subtracted before the absmax (F065).  `v` is never touched by `mean`.

    `out`, when given, is the 4-tuple of caller-owned buffers. A test can pre-fill them with a
    sentinel to check that every element is written.
    """
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("quant_kv_pad_v8t_fp8 expects (B,H,N,D); got %s / %s" % (tuple(k.shape), tuple(v.shape)))
    B, H, N, D = k.shape
    if tuple(v.shape) != (B, H, N, D):
        raise ValueError("k and v shapes differ: %s vs %s" % (tuple(k.shape), tuple(v.shape)))
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("quant_kv_pad_v8t_fp8 expects contiguous k/v (the fused path must not copy)")
    BH = B * H
    n_pad = int(n_pad) if n_pad is not None else n_pad_for(N)
    if n_pad < N or n_pad % 64:
        raise ValueError("n_pad=%d must be >= N=%d and a multiple of 64" % (n_pad, N))
    bd = block_d or max(16, _next_pow2(D))
    if out is None:
        k8 = torch.empty((BH, n_pad, D), dtype=torch.float8_e4m3fn, device=k.device)
        sk = torch.empty((BH, n_pad), dtype=torch.float32, device=k.device)
        v8t = torch.empty((BH, D, n_pad), dtype=torch.float8_e4m3fn, device=k.device)
        sv = torch.empty((BH, n_pad), dtype=torch.float32, device=k.device)
    else:
        k8, sk, v8t, sv = out
        want = [(BH, n_pad, D), (BH, n_pad), (BH, D, n_pad), (BH, n_pad)]
        got = [tuple(t.shape) for t in out]
        if got != want:
            raise ValueError("out shapes %s != %s" % (got, want))
    kv = k.reshape(BH, N, D)
    vv = v.reshape(BH, N, D)
    grid = (triton.cdiv(n_pad, block_n), BH)
    if mean is not None:
        if n_seq is None:
            raise ValueError("quant_kv_pad_v8t_fp8: `mean` requires `n_seq`")
        _quant_kv_pad_v8t_fp8[grid](kv, vv, k8, sk, v8t, sv, N, n_pad, D,
                                    kv.stride(1), kv.stride(2), vv.stride(1), vv.stride(2),
                                    MEAN=mean.reshape(-1, D), SUB_MEAN=1,
                                    BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    else:
        _quant_kv_pad_v8t_fp8[grid](kv, vv, k8, sk, v8t, sv, N, n_pad, D,
                                    kv.stride(1), kv.stride(2), vv.stride(1), vv.stride(2),
                                    BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    return k8, sk, v8t, sv


def prologue_fp8_fused(q, k, v, n_pad=None, smooth_k=False, block_n=64, out=None):
    """The SK1 prologue: Q via the shipped quantiser, K/V via the fused pad+transpose kernel.

    Returns `(q8, sq, k8, sk, v8t, sv)`.  `smooth_k=True` uses the SAME `kseq_mean_fp8(k)` tensor
    the shipped `prologue_fp8` uses (F065), so the K bytes are comparable on that axis.
    """
    from quant_triton import quant_rows_fp8, kseq_mean_fp8
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    q8, sq = quant_rows_fp8(q)
    if smooth_k:
        mu = kseq_mean_fp8(k)
        k8, sk, v8t, sv = quant_kv_pad_v8t_fp8(k, v, n_pad=n_pad, block_n=block_n,
                                               mean=mu, n_seq=k.shape[2], out=out)
    else:
        k8, sk, v8t, sv = quant_kv_pad_v8t_fp8(k, v, n_pad=n_pad, block_n=block_n, out=out)
    return q8, sq, k8, sk, v8t, sv
