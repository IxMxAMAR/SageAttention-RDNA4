#!/usr/bin/env python3
"""The INT8 variant of the fused SK1 prologue: K per token in int8, V unchanged (fp8 e4m3).

Sibling of `quant_kv_v8t.py`, which is not edited. The V branch, the padding contract, the `smooth_k`
plumbing and the `S` staging are copied from `quant_kv_v8t.py`; only the K branch changes, from

    sk  = max(max(|k|)/448, 1e-12);   K8 = e4m3(k/sk)

to

    sk  = max(max(|k|)/127, 1e-12);   K8 = int8(clamp(rne(k/sk), -128, 127))

which is what `kernels/hip/sk1_t6i.hip` expects: `SK` is `amax_k/127`, and the kernel's
`c_q = sm_scale*LOG2E*(amax_q/127)` makes `acc*c_q*SK = sm_scale*LOG2E*sum(q*k)`.

Rounding is round-to-nearest-even, written in pure Triton (`floor` plus the exact half-even test)
rather than through libdevice, so it is `torch.round`'s rule and does not depend on which OCML
bindings the installed Triton exposes. This matters because the hardware `v_cvt_pk_i16_f32`
truncates, and a truncating quantiser is biased.

Why int8 and not e4m3 for K: int8 is a fixed-point grid, so the per-token error is uniform in
absolute terms, and `smooth_k` already removes the per-channel mean that dominates K's dynamic range.
K's scale is not shared with Q's.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from quant_triton import EPS, SCALE_MAX, _cvt2d, _next_pow2

__all__ = ["quant_kv_pad_v8t_i8", "prologue_i8_fused"]

I8_MAX = 127.0


@triton.jit
def _quant_kv_pad_v8t_i8(K, V, K8, SK, V8T, SV, N, N_pad, D,
                         stride_kr, stride_kd, stride_vr, stride_vd, FLOOR,
                         BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                         MEAN=None, SUB_MEAN: tl.constexpr = 0):
    """K -> (K8 int8, SK fp32) row-major padded; V -> (V8T e4m3, SV fp32) transposed padded.

    grid = (cdiv(N_pad, BLOCK_N), B*H).  Every element of all four outputs is written here,
    including the pad region (K8/V8T = 0, SK = 1.0, SV = 0.0), so `torch.empty` is safe.
    """
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    n = pid * BLOCK_N + tl.arange(0, BLOCK_N)      # key index in [0, N_pad)
    cols = tl.arange(0, BLOCK_D)                   # head dim
    live = n < N
    dm = cols < D
    m = live[:, None] & dm[None, :]

    # ---- K: per-row int8 ----------------------------------------------------------------------
    k = tl.load(K + ((bh * N + n)[:, None] * stride_kr + cols[None, :] * stride_kd),
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        mu = tl.load(MEAN + bh * D + cols, mask=dm, other=0.0)
        k = k - mu[None, :]
    k = tl.where(live[:, None], k, 0.0)            # the pad rows must be re-zeroed before the amax
    # (127.0 spelled out: a module-level global is not visible inside a @jit function.)
    # FLOOR is the scale floor applied after the division, matching the fp8 prologue's
    # `tl.maximum(tl.max(tl.abs(x), 1) / SCALE_MAX, EPS)` with `EPS = 1e-12` (the `1` there is
    # `tl.max`'s axis, not a floor). 1e-12 is the default; a floor of 1.0 is accepted only so its
    # cost can be measured, and it is much less accurate.
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
    v = tl.load(V + ((bh * N + n)[:, None] * stride_vr + cols[None, :] * stride_vd),
                mask=m, other=0.0).to(tl.float32)
    v = tl.where(live[:, None], v, 0.0)
    sv = tl.maximum(tl.max(tl.abs(v), 1) / SCALE_MAX, EPS)
    tl.store(SV + bh * N_pad + n, tl.where(live, sv, 0.0), mask=n < N_pad)
    tl.store(V8T + ((bh * D + cols[None, :]) * N_pad + n[:, None]),
             _cvt2d(v / sv[:, None], BLOCK_N, BLOCK_D),
             mask=(n < N_pad)[:, None] & dm[None, :])


def quant_kv_pad_v8t_i8(k, v, n_pad=None, block_n=64, block_d=None, mean=None, n_seq=None, out=None,
                        floor=1e-12):
    """k, v: (B,H,N,D) contiguous fp16 -> (K8 int8, SK fp32, V8T e4m3, SV fp32).

    Same contract as `quant_kv_v8t.quant_kv_pad_v8t_fp8` except `K8` is `torch.int8`.
    """
    if k.dim() != 4 or v.dim() != 4:
        raise ValueError("quant_kv_pad_v8t_i8 expects (B,H,N,D); got %s / %s"
                         % (tuple(k.shape), tuple(v.shape)))
    B, H, N, D = k.shape
    if tuple(v.shape) != (B, H, N, D):
        raise ValueError("k and v shapes differ: %s vs %s" % (tuple(k.shape), tuple(v.shape)))
    if not (k.is_contiguous() and v.is_contiguous()):
        raise ValueError("quant_kv_pad_v8t_i8 expects contiguous k/v")
    BH = B * H
    n_pad = int(n_pad) if n_pad is not None else ((int(N) + 63) // 64) * 64
    if n_pad < N or n_pad % 64:
        raise ValueError("n_pad=%d must be >= N=%d and a multiple of 64" % (n_pad, N))
    bd = block_d or max(16, _next_pow2(D))
    if out is None:
        k8 = torch.empty((BH, n_pad, D), dtype=torch.int8, device=k.device)
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
    args = [kv, vv, k8, sk, v8t, sv, N, n_pad, D,
            kv.stride(1), kv.stride(2), vv.stride(1), vv.stride(2), float(floor)]
    if mean is not None:
        if n_seq is None:
            raise ValueError("quant_kv_pad_v8t_i8: `mean` requires `n_seq`")
        _quant_kv_pad_v8t_i8[grid](*args, MEAN=mean.reshape(-1, D), SUB_MEAN=1,
                                   BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    else:
        _quant_kv_pad_v8t_i8[grid](*args, BLOCK_N=block_n, BLOCK_D=bd, num_warps=4)
    return k8, sk, v8t, sv


def prologue_i8_fused(q, k, v, n_pad=None, smooth_k=False, block_n=64, floor=1e-12):
    """The int8 prologue.  Returns `(q8, sq, k8, sk, v8t, sv)`; `q8`/`sq` are unused by SK1 (the
    kernel quantises Q itself) and are produced only so the return shape matches the shipped
    prologue."""
    from quant_triton import quant_rows_fp8, kseq_mean_fp8
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    q8, sq = quant_rows_fp8(q)
    if smooth_k:
        mu = kseq_mean_fp8(k)
        k8, sk, v8t, sv = quant_kv_pad_v8t_i8(k, v, n_pad=n_pad, block_n=block_n,
                                              mean=mu, n_seq=k.shape[2], floor=floor)
    else:
        k8, sk, v8t, sv = quant_kv_pad_v8t_i8(k, v, n_pad=n_pad, block_n=block_n, floor=floor)
    return q8, sq, k8, sk, v8t, sv
