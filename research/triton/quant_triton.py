"""Quantization prologue kernels for the fp8 attention path on gfx1201.

DESIGN PRINCIPLE (the main departure from the incumbent port):
    The prologue must be **layout-preserving**. No transpose, no pad, no permute.
    The NVIDIA SageAttention path permutes V's head_dim axis with
    [0,1,8,9,2,3,10,11,4,5,12,13,6,7,14,15] purely to satisfy mma.sync fragment
    layouts. RDNA4's v_wmma fragment layout differs, so carrying that permute
    over is pure cost with no benefit. We quantize in the natural NHD layout and
    let the attention kernel's LDS staging handle operand layout.

Two kernel shapes cover everything:

  * per-ROW quant (reduce over head_dim)      -> Q
  * per-CHANNEL quant (reduce over sequence)  -> K and V, with optional centering

Centering is mathematically free for K:
      s_cent[m,t] = q[m].(k[t]-km) = s_true[m,t] - (q[m].km)
The subtracted term depends only on the query row m, and softmax over t is
invariant to a per-row constant shift. So K-centering changes nothing
mathematically; it only shrinks the dynamic range fp8 has to represent.
(No analogous trick exists for Q: (q-qm).k[t] subtracts a term varying with t.
Hence SageAttention has smooth_k/smooth_v and no smooth_q.)

For V the mean DOES have to be returned, because smooth_v changes the output:
      out = attn(v - vm) + vm     (exact, since the weights sum to 1)

Every division is floored (upstream issue #164: an all-equal row yields a
zero-variance scale and NaNs the entire output).
"""
import torch
import triton
import triton.language as tl

FP8 = tl.float8e4nv
SCALE_MAX = tl.constexpr(448.0)   # OCP e4m3 max normal
INT8_MAX = tl.constexpr(127.0)    # F083: symmetric int8, no zero point
EPS = tl.constexpr(1e-12)


# ---------------------------------------------------------------------------
# Hardware fp32 -> e4m3 pack (see F006).
#
# Triton's default `.to(tl.float8e4nv)` does NOT use gfx1201's
# `v_cvt_pk_fp8_f32`; it lowers to ~27 bit-manipulation instructions per
# element. The prologue was paying that cost too, which is why it ran at
# ~128 GB/s instead of near memory speed.
#
# Reaching the instruction needs constraints="=v,v,v" (VGPR, not "=r") and an
# int16 output with pack=2; the fp8 bytes land in the LOW half, so the even
# int16 element carries the value and callers de-interleave by 2.
# ---------------------------------------------------------------------------
@triton.jit
def _cvt_pk_raw(x):
    return tl.inline_asm_elementwise(
        asm="v_cvt_pk_fp8_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[x],
        dtype=tl.int16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _cvt2d(x, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    """Hardware fp32->e4m3 for a (BLOCK_R, BLOCK_D) tile. BLOCK_D must be even."""
    h = _cvt_pk_raw(tl.minimum(x, 448.0))
    lo, _hi = tl.split(tl.reshape(h, (BLOCK_R, BLOCK_D // 2, 2)))
    b = lo.to(tl.uint16, bitcast=True)
    return tl.interleave((b & 0xFF).to(tl.uint8),
                         (b >> 8).to(tl.uint8)).to(tl.float8e4nv, bitcast=True)


# ---------------------------------------------------------------------------
# F083 -- hardware fp32 -> int8 pack (the "8+8 split"'s QK half).
#
# `v_cvt_pk_fp8_f32` above CANNOT make int8; the matching packed instruction is
# `v_cvt_pk_i16_f32`, which converts two fp32 to two SIGNED 16-bit ints and
# writes a full 32-bit VGPR -- so, unlike the fp8 pack, BOTH halves of the
# register are meaningful: element 2i is i16(x[2i]) and element 2i+1 is
# i16(x[2i+1]). The int8 is the low byte of each.
#
# This is the whole reason `_rndne` is here: `v_cvt_pk_i16_f32` truncates toward zero, it does not
# round. That was measured, not assumed: comparing the emitted codes against `torch.round`, `trunc`,
# `floor`, `ceil` and round-half-away shows that `trunc` matches exactly while `torch.round` differs
# on 48.85 % of elements. The F083 convention is `torch.round(x/s).clamp_(-127,127)`, so a bare
# `v_cvt_pk_i16_f32` would have silently shipped a biased quantiser (mean residual +0.005, always
# toward zero). `v_cvt_i32_f32` truncates as well.
#
# So the round happens in float first: `v_rndne_f32` is round-to-nearest-even, which is exactly
# `torch.round`'s tie rule, and the subsequent truncating convert is then exact. Checked bit-exact
# against `torch.round` over 1024 random values and over the explicit tie set
# {+-0.5, +-1.5, +-2.5, +-126.5}. Cost: one extra VGPR instruction per element.
#
# The caller must pre-scale so that |x| <= 127 (the quantiser divides by
# `amax/127`); the explicit clamp below makes that a guarantee rather than an
# expectation, since a rounded 127.0000001 would otherwise land on +128 and wrap
# to -128 in the low byte.
#
# Same two non-obvious requirements as the fp8 pack: `=v` (VGPR, not `=r`) and
# an int16 output dtype with pack=2. Symmetric, no zero point.
# ---------------------------------------------------------------------------
@triton.jit
def _rndne(x):
    """fp32 -> fp32, round to nearest even (integral). = `torch.round`."""
    return tl.inline_asm_elementwise(
        asm="v_rndne_f32 $0, $1",
        constraints="=v,v",
        args=[x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def _cvt_pk_i8_raw(x):
    """fp32 -> two int16 in one VGPR. TRUNCATES -- feed it `_rndne`'d input."""
    return tl.inline_asm_elementwise(
        asm="v_cvt_pk_i16_f32 $0, $1, $2",
        constraints="=v,v,v",
        args=[x],
        dtype=tl.int16,
        is_pure=True,
        pack=2,
    )


@triton.jit
def _cvt2d_i8(x, BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr):
    """Hardware fp32->int8 for a (BLOCK_R, BLOCK_D) tile. BLOCK_D must be even.

    `x` must already be divided by the row scale, i.e. lie in [-127, 127].
    Round-to-nearest-even, then a truncating convert -- bit-exact `torch.round`.
    """
    xc = tl.minimum(tl.maximum(x, -127.0), 127.0)
    h = _cvt_pk_i8_raw(_rndne(xc))
    lo, hi = tl.split(tl.reshape(h, (BLOCK_R, BLOCK_D // 2, 2)))
    a = (lo.to(tl.uint16, bitcast=True) & 0xFF).to(tl.uint8)
    b = (hi.to(tl.uint16, bitcast=True) & 0xFF).to(tl.uint8)
    return tl.interleave(a, b).to(tl.int8, bitcast=True)


@triton.jit
def _quant_row_fp8(X, X8, S, D,
                   stride_xr, stride_xd,
                   BLOCK_D: tl.constexpr):
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    m = cols < D
    x = tl.load(X + r * stride_xr + cols * stride_xd, mask=m, other=0.0).to(tl.float32)
    s = tl.maximum(tl.max(tl.abs(x)) / SCALE_MAX, EPS)
    tl.store(S + r, s)
    tl.store(X8 + r * stride_xr + cols * stride_xd, (x / s).to(FP8), mask=m)


@triton.jit
def _quant_channel_fp8(X, X8, S, MEAN, N, D,
                       stride_xb, stride_xn, stride_xd,
                       BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
                       CENTER: tl.constexpr, WRITE_MEAN: tl.constexpr):
    """Per-(batch*head, d) scale reduced over the sequence axis.

    Three passes over X at most (mean, absmax, quantize). For our shapes X fits
    in L2 — an 8192x128 fp16 tensor is 16.8 MB and this part has 64 MB of
    Infinity Cache — so repeat passes are L2 reads, not HBM.
    """
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    dm = offs_d < D
    base = X + pid * stride_xb

    mean = tl.zeros([BLOCK_D], dtype=tl.float32)
    if CENTER:
        acc_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
        acc_cnt = tl.zeros([BLOCK_D], dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            n = n0 + offs_n
            nm = n < N
            x = tl.load(base + n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                        mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
            acc_sum += tl.sum(x, 0)
            acc_cnt += tl.sum(tl.where(nm, 1.0, 0.0), 0)
        mean = acc_sum / tl.maximum(acc_cnt, 1.0)
        if WRITE_MEAN:
            tl.store(MEAN + pid * D + offs_d, mean, mask=dm)

    amax = tl.zeros([BLOCK_D], dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        n = n0 + offs_n
        nm = n < N
        x = tl.load(base + n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        if CENTER:
            x = x - mean[None, :]
        amax = tl.maximum(amax, tl.max(tl.abs(x), 0))

    s = tl.maximum(amax / SCALE_MAX, EPS)
    tl.store(S + pid * D + offs_d, s, mask=dm)

    for n0 in range(0, N, BLOCK_N):
        n = n0 + offs_n
        nm = n < N
        x = tl.load(base + n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                    mask=nm[:, None] & dm[None, :], other=0.0).to(tl.float32)
        if CENTER:
            x = x - mean[None, :]
        tl.store(X8 + pid * stride_xb + n[:, None] * stride_xn + offs_d[None, :] * stride_xd,
                 (x / s[None, :]).to(FP8), mask=nm[:, None] & dm[None, :])


def _next_pow2(n):
    return 1 << max(0, (n - 1).bit_length())


# ---------------------------------------------------------------------------
# FAST PATH: one per-ROW quantizer handles Q, K and V alike.
#
# This exists because of a measured result: the per-CHANNEL quantizer below
# needs a reduction over the whole sequence axis, which forces a grid of only
# B*H programs (8 for a single sequence) and measured 3.15 ms per tensor on
# B1 H8 N8192 D128 — 19x slower than torch's own amax (0.165 ms), and 6.5 ms
# for the full prologue.
#
# We avoid the sequence reduction entirely by quantizing V per ROW (over
# head_dim) instead of per CHANNEL, and folding V's dequantization scale into P
# inside the attention kernel:
#
#     v8[n,d] = v[n,d] / sv[n]                     (sv depends only on n)
#     p8[m,n] = p[m,n] * sv[n] * P_SCALE           (quantized)
#     sum_n p8[m,n]*v8[n,d] = P_SCALE * sum_n p[m,n]*v[n,d]
#
# The fold is exact, costs one broadcast multiply on a tile we already scale for
# P_SCALE, and removes the only sequence-axis reduction in the prologue. All
# three tensors are now quantized by the same embarrassingly-parallel kernel.
# ---------------------------------------------------------------------------
@triton.jit
def _quant_rows_fp8(X, X8, S, R, D, stride_xr, stride_xd,
                    BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
                    MEAN=None, SUB_MEAN: tl.constexpr = 0, N_SEQ: tl.constexpr = 0):
    """Row-wise fp8 quantiser.

    F065 (3): `SUB_MEAN=1` subtracts a per-(b,h,channel) sequence-axis mean from X
    BEFORE taking the row absmax and quantising. `MEAN` points at a `(B*H, D)` fp32
    tensor and `N_SEQ` is the sequence length, so the flattened row index maps to
    `bh = row // N_SEQ`.

    The mean is subtracted BEFORE the absmax, which is the whole point: it is what
    shrinks K's row amax (F061/F063: 294.4 -> 6.42) and therefore the quantisation
    error. Subtracting after the scale would do nothing.

    `N_SEQ` is a `tl.constexpr` and the divide is an integer divide, so the kernel
    is specialised per sequence length. `SUB_MEAN=0` (the default) compiles to exactly
    the pre-F065 kernel: the `if` is a constexpr branch, so no code is emitted for it.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    rm = rows < R
    m = rm[:, None] & (cols < D)[None, :]
    x = tl.load(X + rows[:, None] * stride_xr + cols[None, :] * stride_xd,
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        bh = rows // N_SEQ
        mu = tl.load(MEAN + bh[:, None] * D + cols[None, :],
                     mask=(cols < D)[None, :], other=0.0)
        x = x - mu
    s = tl.maximum(tl.max(tl.abs(x), 1) / SCALE_MAX, EPS)
    tl.store(S + rows, s, mask=rm)
    tl.store(X8 + rows[:, None] * stride_xr + cols[None, :] * stride_xd,
             _cvt2d(x / s[:, None], BLOCK_R, BLOCK_D), mask=m)


@triton.jit
def _quant_rows_i8(X, X8, S, R, D, stride_xr, stride_xd,
                   BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
                   MEAN=None, SUB_MEAN: tl.constexpr = 0, N_SEQ: tl.constexpr = 0):
    """F083: row-wise int8 quantiser -- `_quant_rows_fp8` with int8/127 and a
    round-to-nearest-even clamp in place of the e4m3 cast.

    Same `mean`/`n_seq` (smooth_k) contract as the fp8 kernel: the mean is
    subtracted BEFORE the absmax, which is the whole point.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    rm = rows < R
    m = rm[:, None] & (cols < D)[None, :]
    x = tl.load(X + rows[:, None] * stride_xr + cols[None, :] * stride_xd,
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        bh = rows // N_SEQ
        mu = tl.load(MEAN + bh[:, None] * D + cols[None, :],
                     mask=(cols < D)[None, :], other=0.0)
        x = x - mu
    s = tl.maximum(tl.max(tl.abs(x), 1) / INT8_MAX, EPS)
    tl.store(S + rows, s, mask=rm)
    tl.store(X8 + rows[:, None] * stride_xr + cols[None, :] * stride_xd,
             _cvt2d_i8(x / s[:, None], BLOCK_R, BLOCK_D), mask=m)


@triton.jit
def _kseq_partial_sum(X, P, N_SEQ, D, stride_xr, stride_xd, CHUNK: tl.constexpr,
                      BLOCK_D: tl.constexpr):
    """Stage 1 of the sequence-axis mean: partial sums over CHUNK tokens.

    Grid `(B*H, D_blocks, n_chunks)`. Writes `P[(bh, d, chunk)]`.

    DETERMINISTIC BY CONSTRUCTION -- no atomics. `tl.atomic_add` on fp32 would make
    the mean depend on block execution order, and every result is expected to be
    reproducible. A fixed partial-then-reduce order gives a
    bit-stable result for a given shape.
    """
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    ck = tl.program_id(2)
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    base = bh * N_SEQ
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r in range(0, CHUNK, 16):
        rows = ck * CHUNK + r + tl.arange(0, 16)
        rm = rows < N_SEQ
        m = rm[:, None] & dm[None, :]
        x = tl.load(X + (base + rows)[:, None] * stride_xr
                    + cols[None, :] * stride_xd, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    n_ch = tl.cdiv(N_SEQ, CHUNK)
    tl.store(P + (bh * n_ch + ck) * D + cols, acc, mask=dm)


@triton.jit
def _kseq_finalize_mean(P, MEAN, N_SEQ, D, n_ch, BLOCK_D: tl.constexpr):
    """Stage 2: sum the chunk partials and divide by N_SEQ. Grid `(B*H, D_blocks)`."""
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for c in range(0, n_ch):
        acc += tl.load(P + (bh * n_ch + c) * D + cols, mask=dm, other=0.0)
    tl.store(MEAN + bh * D + cols, acc / N_SEQ, mask=dm)


def kseq_mean_fp8(k, block_d=None, chunk=1024):
    """F065 (3): K's mean over the SEQUENCE axis, shape (B, H, D), in ONE read of K.

    This replaces `torch`'s `k.float().mean(dim=2)`, which materialised a full
    `(B,H,N,D)` fp32 temporary (2x K's fp16 bytes) and was measured at **+2.69 ms,
    +286% on the prologue, 15.0% of a full non-causal call** -- far past the 5% ship
    gate. This reads K once and writes only `(B*H, D)` and `(B*H, D, n_chunks)` fp32,
    so the temporary is ~N/CHUNK times smaller and there is no separate torch pass.

    `chunk` trades the partial-buffer size against the reduction length: the partial
    buffer is `B*H*D*cdiv(N, chunk)` fp32. At B*H=48, D=128, N=8768, chunk=1024 that
    is 48*128*9*4 = 221 KB.
    """
    B, H, N, D = k.shape
    BH = B * H
    xv = k.reshape(BH, N, D)
    # F088: `_kseq_partial_sum` reconstructs the row address as `(bh*N_SEQ + row) * stride_xr` and
    # is never given `stride(0)`, i.e. it assumes `stride(0) == N*stride(1)`. On a non-contiguous
    # `(B,H,N,D)` view that assumption is false: with B == 1 a transposed `(N,D)` block reshapes to
    # a view whose stride(0) is the head stride, so every `bh >= 1` reads out of bounds. Seen on
    # gfx1201 as a hard driver abort (exit 0xC0000409) on a captured Klein activation whose V had
    # strides (4325376,128,4096,1); the same defect silently returns a NaN mean when the garbage
    # lands in mapped memory. `flash_attn_fp8` masks this with `.contiguous()`; a direct call does
    # not. `is_contiguous()` is True on the normal path, so this is a zero-copy no-op there.
    if not xv.is_contiguous():
        xv = xv.contiguous()
    bd = block_d or max(16, _next_pow2(D))
    n_ch = triton.cdiv(N, chunk)
    part = torch.empty((BH, n_ch, D), device=k.device, dtype=torch.float32)
    mean = torch.empty((BH, D), device=k.device, dtype=torch.float32)
    _kseq_partial_sum[(BH, triton.cdiv(D, bd), n_ch)](
        xv, part, N, D, xv.stride(1), xv.stride(2),
        CHUNK=chunk, BLOCK_D=bd, num_warps=4)
    _kseq_finalize_mean[(BH, triton.cdiv(D, bd))](
        part, mean, N, D, n_ch, BLOCK_D=bd, num_warps=4)
    return mean.view(B, H, D)


# ---------------------------------------------------------------------------
# F100: the per-CHANNEL absmax over the sequence axis.
#
# Structurally `kseq_mean_fp8` with `max` in place of `sum`: two stages, one read
# of `x`, no atomics, so it is deterministic by construction.
#
# Why it is not the old slow path: `_quant_channel_fp8` (above) needs a grid of
# only `B*H` programs and measured 3.15 ms/tensor.  This one is gridded
# `(B*H, D/BLOCK_D, cdiv(N, CHUNK))` -- 48*1*9 = 432 programs at B1 H48 N8771 D128
# chunk=1024 -- exactly the shape `kseq_mean_fp8` already uses at ~2 % of a full
# call.  The cost of this kernel was not timed on its own; the claim is structural, from
# `kseq_mean_fp8`'s measured cost.
# ---------------------------------------------------------------------------
@triton.jit
def _kseq_partial_amax(X, P, N_SEQ, D, stride_xr, stride_xd, CHUNK: tl.constexpr,
                       BLOCK_D: tl.constexpr, MEAN=None, SUB_MEAN: tl.constexpr = 0):
    """Stage 1 of the sequence-axis absmax: per-chunk maxima. Grid
    `(B*H, D_blocks, n_chunks)`. `SUB_MEAN=1` centres by `MEAN` first, so the
    result is `amax|X - mu|` -- the quantity the `sv`-fold actually needs."""
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    ck = tl.program_id(2)
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    base = bh * N_SEQ
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for r in range(0, CHUNK, 16):
        rows = ck * CHUNK + r + tl.arange(0, 16)
        rm = rows < N_SEQ
        m = rm[:, None] & dm[None, :]
        x = tl.load(X + (base + rows)[:, None] * stride_xr
                    + cols[None, :] * stride_xd, mask=m, other=0.0).to(tl.float32)
        if SUB_MEAN:
            mu = tl.load(MEAN + bh * D + cols, mask=dm, other=0.0)
            x = x - mu[None, :]
        acc = tl.maximum(acc, tl.max(tl.abs(x), 0))
    n_ch = tl.cdiv(N_SEQ, CHUNK)
    tl.store(P + (bh * n_ch + ck) * D + cols, acc, mask=dm)


@triton.jit
def _kseq_finalize_amax(P, AMAX, n_ch, D, BLOCK_D: tl.constexpr):
    """Stage 2: max the chunk partials. Grid `(B*H, D_blocks)`."""
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for c in range(0, n_ch):
        acc = tl.maximum(acc, tl.load(P + (bh * n_ch + c) * D + cols,
                                      mask=dm, other=0.0))
    tl.store(AMAX + bh * D + cols, acc, mask=dm)


def kseq_amax_fp8(x, mean=None, block_d=None, chunk=1024):
    """`(B,H,D)` per-channel absmax over the SEQUENCE axis, in ONE read of `x`.

    `mean` (shape `(B*H,D)`, from `kseq_mean_fp8`) subtracts the sequence-axis mean
    before the absmax, i.e. returns `amax_n |x[n,d] - mu[d]|` -- the scale that
    makes V's quantisation benefit from `smooth_v`.
    """
    B, H, N, D = x.shape
    BH = B * H
    xv = x.reshape(BH, N, D)
    if not xv.is_contiguous():
        xv = xv.contiguous()
    bd = block_d or max(16, _next_pow2(D))
    n_ch = triton.cdiv(N, chunk)
    part = torch.empty((BH, n_ch, D), device=x.device, dtype=torch.float32)
    amax = torch.empty((BH, D), device=x.device, dtype=torch.float32)
    _kseq_partial_amax[(BH, triton.cdiv(D, bd), n_ch)](
        xv, part, N, D, xv.stride(1), xv.stride(2), CHUNK=chunk, BLOCK_D=bd,
        MEAN=(mean.reshape(-1, D) if mean is not None else xv),
        SUB_MEAN=(1 if mean is not None else 0), num_warps=4)
    _kseq_finalize_amax[(BH, triton.cdiv(D, bd))](
        part, amax, n_ch, D, BLOCK_D=bd, num_warps=4)
    return amax.view(B, H, D)


# ---------------------------------------------------------------------------
# F104: ONE read of V -> BOTH the sequence-axis mean and the CENTRED
# per-channel absmax.
#
# WHY IT IS EXACT.  `_kseq_partial_amax` computes, per channel,
#     amax[d] = max over the rows it LOADS of |fl(v_row[d] - mu[d])|
# and it loads `n_ch * CHUNK` rows (rows >= N come from `other=0.0` and then have
# `mu` subtracted, so each padded row contributes |0 - mu| = |mu|).  x -> fl(x - mu)
# is MONOTONE (round-to-nearest is monotone) and |.| over a monotone range attains
# its maximum at an endpoint, so that max is exactly
#     max( |fl(max(amax_v[d], 0) - mu[d])| , |fl(min(amin_v[d], 0) - mu[d])| )
# with the 0 folded in because a padded row IS the value 0.  The `max(.,0)` /
# `min(.,0)` therefore appear AUTOMATICALLY if the stage-1 kernel accumulates
# max/min over the loaded tile (`other=0.0` supplies the 0 for every masked lane).
#
# Checked bit-exact on CPU over 323 fp16-valued cases including N = 1, 15, 17,
# 579, 1023, 1025, 8770, 8771 and D = 16..128.
#
# This replaces `kseq_mean_fp8(v)` + `kseq_amax_fp8(v, mean=mu)` -- TWO reads of V
# and FOUR launches -- with ONE read and TWO launches.  It is a new kernel:
# `_kseq_partial_sum` and `_kseq_partial_amax` are untouched, so the shipped paths
# cannot move.
# ---------------------------------------------------------------------------
@triton.jit
def _kseq_partial_sumamax(X, PSUM, PMAX, PMIN, N_SEQ, D, stride_xr, stride_xd,
                          CHUNK: tl.constexpr, BLOCK_D: tl.constexpr):
    """Stage 1: per-chunk sum, max and min in ONE read. Grid
    `(B*H, D_blocks, n_chunks)`. The sum accumulation is `_kseq_partial_sum`'s
    verbatim (`acc += tl.sum(x, axis=0)` over the same 16-row steps) so the mean
    is bit-identical; max/min ride along on the same loaded tile."""
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    ck = tl.program_id(2)
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    base = bh * N_SEQ
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    amx = tl.full([BLOCK_D], float("-inf"), tl.float32)
    amn = tl.full([BLOCK_D], float("inf"), tl.float32)
    for r in range(0, CHUNK, 16):
        rows = ck * CHUNK + r + tl.arange(0, 16)
        rm = rows < N_SEQ
        m = rm[:, None] & dm[None, :]
        x = tl.load(X + (base + rows)[:, None] * stride_xr
                    + cols[None, :] * stride_xd, mask=m, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
        amx = tl.maximum(amx, tl.max(x, axis=0))
        amn = tl.minimum(amn, tl.min(x, axis=0))
    n_ch = tl.cdiv(N_SEQ, CHUNK)
    tl.store(PSUM + (bh * n_ch + ck) * D + cols, acc, mask=dm)
    tl.store(PMAX + (bh * n_ch + ck) * D + cols, amx, mask=dm)
    tl.store(PMIN + (bh * n_ch + ck) * D + cols, amn, mask=dm)


@triton.jit
def _kseq_finalize_meanamax(PSUM, PMAX, PMIN, MEAN, AMAX, N_SEQ, D, n_ch,
                            BLOCK_D: tl.constexpr):
    """Stage 2: sum the partials (`_kseq_finalize_mean`'s verbatim reduction) and
    form `amax = max(|max - mu|, |min - mu|)` -- see the block comment above."""
    bh = tl.program_id(0)
    dc = tl.program_id(1)
    cols = dc * BLOCK_D + tl.arange(0, BLOCK_D)
    dm = cols < D
    s = tl.zeros([BLOCK_D], dtype=tl.float32)
    mx = tl.full([BLOCK_D], float("-inf"), tl.float32)
    mn = tl.full([BLOCK_D], float("inf"), tl.float32)
    for c in range(0, n_ch):
        s += tl.load(PSUM + (bh * n_ch + c) * D + cols, mask=dm, other=0.0)
        mx = tl.maximum(mx, tl.load(PMAX + (bh * n_ch + c) * D + cols,
                                    mask=dm, other=float("-inf")))
        mn = tl.minimum(mn, tl.load(PMIN + (bh * n_ch + c) * D + cols,
                                    mask=dm, other=float("inf")))
    mean = s / N_SEQ
    amax = tl.maximum(tl.abs(mx - mean), tl.abs(mn - mean))
    tl.store(MEAN + bh * D + cols, mean, mask=dm)
    tl.store(AMAX + bh * D + cols, amax, mask=dm)


def kseq_mean_amax_fp8(x, block_d=None, chunk=1024):
    """F104: `(B,H,D)` sequence-axis mean AND centred absmax in ONE read of `x`.

    Returns `(mean, amax)` with `amax[b,h,d] = max_n |x[b,h,n,d] - mean[b,h,d]|`,
    both BIT-IDENTICAL to `kseq_mean_fp8(x)` and `kseq_amax_fp8(x, mean=mean)`.
    """
    B, H, N, D = x.shape
    BH = B * H
    xv = x.reshape(BH, N, D)
    if not xv.is_contiguous():
        xv = xv.contiguous()
    bd = block_d or max(16, _next_pow2(D))
    n_ch = triton.cdiv(N, chunk)
    psum = torch.empty((BH, n_ch, D), device=x.device, dtype=torch.float32)
    pmax = torch.empty((BH, n_ch, D), device=x.device, dtype=torch.float32)
    pmin = torch.empty((BH, n_ch, D), device=x.device, dtype=torch.float32)
    mean = torch.empty((BH, D), device=x.device, dtype=torch.float32)
    amax = torch.empty((BH, D), device=x.device, dtype=torch.float32)
    _kseq_partial_sumamax[(BH, triton.cdiv(D, bd), n_ch)](
        xv, psum, pmax, pmin, N, D, xv.stride(1), xv.stride(2),
        CHUNK=chunk, BLOCK_D=bd, num_warps=4)
    _kseq_finalize_meanamax[(BH, triton.cdiv(D, bd))](
        psum, pmax, pmin, mean, amax, N, D, n_ch, BLOCK_D=bd, num_warps=4)
    return mean.view(B, H, D), amax.view(B, H, D)


def quant_rows_fp8(x, block_r=16, block_d=None, mean=None, n_seq=None, out=None):
    """x: (..., D) contiguous fp16/bf16 -> (x8 fp8, scale (...,) fp32).

    One kernel for Q, K and V. Grid = ceil(rows / block_r).

    F065 (3): pass `mean` (shape `(B*H, D)`, from `kseq_mean_fp8`) and `n_seq` to
    subtract the sequence-axis mean before quantising. Both must be given together:
    `n_seq` is needed to map a flattened row back to its `(b,h)` for the mean lookup,
    and defaulting it would silently index the wrong channel group.
    """
    D = x.shape[-1]
    xv = x.reshape(-1, D)
    R = xv.shape[0]
    if out is None:  # F114 flag: default off, so the path without it is unchanged
        x8 = torch.empty_like(xv, dtype=torch.float8_e4m3fn)
        s = torch.empty(R, device=x.device, dtype=torch.float32)
    else:
        x8, s = out
        if x8.shape != (R, D) or s.shape != (R,):
            raise ValueError(f"quant_rows_fp8: out buffers {tuple(x8.shape)}/{tuple(s.shape)}"
                             f" do not match ({R}, {D})/({R},)")
    bd = block_d or max(16, _next_pow2(D))
    grid = (triton.cdiv(R, block_r),)
    if mean is not None:
        if n_seq is None:
            raise ValueError("quant_rows_fp8: `mean` requires `n_seq`")
        _quant_rows_fp8[grid](xv, x8, s, R, D, xv.stride(0), xv.stride(1),
                              MEAN=mean.reshape(-1, D), SUB_MEAN=1, N_SEQ=int(n_seq),
                              BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    else:
        _quant_rows_fp8[grid](xv, x8, s, R, D, xv.stride(0), xv.stride(1),
                              BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    return x8.view(x.shape), s.view(x.shape[:-1])


def quant_rows_i8(x, block_r=16, block_d=None, mean=None, n_seq=None):
    """F083: `x` (..., D) contiguous fp16/bf16 -> (x8 int8, scale (...,) fp32).

    The int8 twin of `quant_rows_fp8`: same per-row amax, same `max(amax, 1)`
    floor, same epsilon, but `/127` and round-to-nearest-even instead of `/448`
    and an e4m3 cast. `mean`/`n_seq` are the `smooth_k` (sequence-axis mean)
    path and must be given together.
    """
    D = x.shape[-1]
    xv = x.reshape(-1, D)
    R = xv.shape[0]
    x8 = torch.empty_like(xv, dtype=torch.int8)
    s = torch.empty(R, device=x.device, dtype=torch.float32)
    bd = block_d or max(16, _next_pow2(D))
    grid = (triton.cdiv(R, block_r),)
    if mean is not None:
        if n_seq is None:
            raise ValueError("quant_rows_i8: `mean` requires `n_seq`")
        _quant_rows_i8[grid](xv, x8, s, R, D, xv.stride(0), xv.stride(1),
                             MEAN=mean.reshape(-1, D), SUB_MEAN=1, N_SEQ=int(n_seq),
                             BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    else:
        _quant_rows_i8[grid](xv, x8, s, R, D, xv.stride(0), xv.stride(1),
                             BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    return x8.view(x.shape), s.view(x.shape[:-1])


@triton.jit
def _quant_kv_rows_fp8(K, V, K8, V8, SK, SV, R, D,
                       stride_kr, stride_kd, stride_vr, stride_vd,
                       BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
                       MEAN=None, SUB_MEAN: tl.constexpr = 0, N_SEQ: tl.constexpr = 0,
                       VMEAN=None, SUB_VMEAN: tl.constexpr = 0,
                       VCH=None, CHAN_SV: tl.constexpr = 0):
    """Per-row quantization of K and V in ONE launch.

    Both use the same per-token scheme over the same (B,H,N,D) shape, so a single
    kernel halves launch/scheduling overhead and lets the K and V streams
    overlap in the memory system. Input and output element strides are identical
    (same shape), so one set of strides serves both.

    Note: this does NOT reduce the number of passes over memory -- each element
    is still read once and written once. It only removes a launch.

    F082: `SUB_MEAN=1` subtracts K's per-(b,h,channel) SEQUENCE-axis
    mean from K BEFORE K's absmax and quantisation -- the same `smooth_k` the
    single-tensor kernel has had since F065 (`_quant_rows_fp8`), lifted into the
    FUSED path so `flash_attn_fp8` can reach it. `MEAN` points at a `(B*H, D)` fp32
    tensor and `N_SEQ` is the sequence length, so `bh = row // N_SEQ`.

    `smooth_k` IS K ONLY, NEVER V. `smooth_k` is justified by K's row amax
    collapsing (294.4 -> 6.42, F061/F063) and by the softmax-neutrality of the
    removed term (`q . mu` is a per-row constant). V has no such argument: its
    quantisation is per-token over the SAME axis as the mean, so subtracting K's
    mean from V would be a pure error term on the value. F081 puts PV at fp8
    precisely because e4m3 PV is accurate.

    **V's own mean is a different question and it IS supported** — see the
    F086 `SUB_VMEAN` block below. V's mean is subtracted from V and added back to
    the output; K's mean is subtracted from K and never added back. The two must
    never be crossed.

    The mean is subtracted BEFORE the absmax, which is the whole point.
    `SUB_MEAN=0` (the default) compiles to exactly the pre-F082 kernel: the branch
    is constexpr, so no code is emitted for it and the shipped path is unchanged.

    F086: `SUB_VMEAN=1` subtracts **V's** per-(b,h,channel)
    SEQUENCE-axis mean from V BEFORE V's absmax and quantisation. `VMEAN` is a
    second `(B*H, D)` fp32 tensor and shares `N_SEQ` with `MEAN`.

    `smooth_v` IS NOT `smooth_k`. `smooth_k` is softmax-NEUTRAL (the removed
    `q . mu` is a per-query-row constant softmax cancels), so nothing is added
    back. `smooth_v` is NOT neutral, but it is EXACTLY recoverable because
    softmax rows sum to 1:
        sum_j P_ij (V_jd - mu_d) = out_id - mu_d
    so `mu` is added back ONCE to the output, per `(b, h, channel)`, in
    `fa_fp8._attn_fwd_fp8` (`ADD_VMEAN`). No approximation, no free parameter.

    The two means are SEPARATE pointers and separate constexprs: `smooth_v`'s
    mean must never reach K's absmax, and `smooth_k`'s must never reach V's.

    `SUB_VMEAN=0` (the default) emits no code, exactly like `SUB_MEAN=0`.

    F100: `CHAN_SV=1` DECOUPLES V's quantisation grid from
    P's.  The shipped kernel folds V's per-ROW scale `sv` into the P quantisation
    (`p8 = e4m3(p * sv * P_SCALE)`), so P's e4m3 grid is `max(amax_d V, 1)` instead
    of `P_SCALE`, and `smooth_v` shrinks it toward e4m3's subnormal floor (`F097`).
    With `CHAN_SV=1`:
      * V's divisor is `VCH` (`(B*H, D)` fp32), i.e. per-CHANNEL, taken on the
        sequence axis (`kseq_amax_fp8`);
      * the per-KEY `sv` tensor is written as **1.0**, so the fold contributes
        nothing and P keeps e4m3's FULL `P_SCALE` grid;
      * the channel scale is re-applied in the ATTENTION kernel's epilogue
        (`MUL_VCH`), where it is a free `(HEAD_DIM,)` multiply because the channel
        axis is not the reduction axis.
    `CHAN_SV=0` (the default) emits no code: the branch is constexpr, so the
    shipped path is bit-identical.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    rm = rows < R
    m = rm[:, None] & (cols < D)[None, :]
    k = tl.load(K + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
                mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        bh = rows // N_SEQ
        mu = tl.load(MEAN + bh[:, None] * D + cols[None, :],
                     mask=(cols < D)[None, :], other=0.0)
        k = k - mu
    if SUB_VMEAN:
        bhv = rows // N_SEQ
        vmu = tl.load(VMEAN + bhv[:, None] * D + cols[None, :],
                      mask=(cols < D)[None, :], other=0.0)
        v = v - vmu
    sk = tl.maximum(tl.max(tl.abs(k), 1) / SCALE_MAX, EPS)
    sv = tl.maximum(tl.max(tl.abs(v), 1) / SCALE_MAX, EPS)
    tl.store(SK + rows, sk, mask=rm)
    tl.store(SV + rows, sv, mask=rm)
    tl.store(K8 + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
             _cvt2d(k / sk[:, None], BLOCK_R, BLOCK_D), mask=m)
    tl.store(V8 + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
             _cvt2d(v / sv[:, None], BLOCK_R, BLOCK_D), mask=m)
    if CHAN_SV:
        # F100: OVERRIDE, deliberately placed AFTER the shipped block so
        # that the `CHAN_SV=0` path above is the pre-F100 code verbatim and the
        # pruned tail cannot move its codegen. This costs the chan path one
        # redundant row absmax and one redundant V store; that is the price of a
        # provable flag-off identity.
        vch = tl.load(VCH + (rows // N_SEQ)[:, None] * D + cols[None, :],
                      mask=(cols < D)[None, :], other=1.0)
        tl.store(SV + rows, tl.full([BLOCK_R], 1.0, tl.float32), mask=rm)
        tl.store(V8 + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
                 _cvt2d(v / vch, BLOCK_R, BLOCK_D), mask=m)


@triton.jit
def _quant_kv_rows_i8(K, V, K8, V8, SK, SV, R, D,
                      stride_kr, stride_kd, stride_vr, stride_vd,
                      BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
                      MEAN=None, SUB_MEAN: tl.constexpr = 0, N_SEQ: tl.constexpr = 0,
                      VMEAN=None, SUB_VMEAN: tl.constexpr = 0,
                      VCH=None, CHAN_SV: tl.constexpr = 0):
    """F083: per-row quantization of K (int8) and V (e4m3) in ONE launch.

    The K half is `_quant_rows_i8`; the V half is `_cvt2d` (e4m3), UNCHANGED --
    the split is QK int8 / PV fp8, so V must keep the exact fp8 bytes the fp8
    path produces.

    K ONLY for `smooth_k`, NEVER V -- same argument as `_quant_kv_rows_fp8`.

    This is a SEPARATE kernel from `_quant_kv_rows_fp8`, not a constexpr
    branch inside it, so the fp8 kernel's generated instructions cannot move.

    F094: `SUB_VMEAN=1` subtracts **V's** per-(b,h,channel)
    SEQUENCE-axis mean (`VMEAN`, a second `(B*H, D)` fp32 tensor sharing `N_SEQ`)
    from V BEFORE V's absmax and quantisation -- the exact structural twin of the
    F086 block in `_quant_kv_rows_fp8`, so `smooth_v` now exists on the int8 QK
    arm too. The K mean and the V mean are two INDEPENDENT pointers and
    independent constexprs and never cross; `SUB_VMEAN=0` (the default) emits no
    code, so the int8 off path is bit-identical to pre-F094.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    rm = rows < R
    m = rm[:, None] & (cols < D)[None, :]
    k = tl.load(K + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
                mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        bh = rows // N_SEQ
        mu = tl.load(MEAN + bh[:, None] * D + cols[None, :],
                     mask=(cols < D)[None, :], other=0.0)
        k = k - mu
    if SUB_VMEAN:
        bhv = rows // N_SEQ
        vmu = tl.load(VMEAN + bhv[:, None] * D + cols[None, :],
                      mask=(cols < D)[None, :], other=0.0)
        v = v - vmu
    sk = tl.maximum(tl.max(tl.abs(k), 1) / INT8_MAX, EPS)
    sv = tl.maximum(tl.max(tl.abs(v), 1) / SCALE_MAX, EPS)
    tl.store(SK + rows, sk, mask=rm)
    tl.store(SV + rows, sv, mask=rm)
    tl.store(K8 + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
             _cvt2d_i8(k / sk[:, None], BLOCK_R, BLOCK_D), mask=m)
    tl.store(V8 + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
             _cvt2d(v / sv[:, None], BLOCK_R, BLOCK_D), mask=m)
    if CHAN_SV:
        # F100: same override contract as `_quant_kv_rows_fp8` -- the
        # shipped block above is the pre-F100 code verbatim, so `CHAN_SV=0` cannot
        # move its codegen.
        vch = tl.load(VCH + (rows // N_SEQ)[:, None] * D + cols[None, :],
                      mask=(cols < D)[None, :], other=1.0)
        tl.store(SV + rows, tl.full([BLOCK_R], 1.0, tl.float32), mask=rm)
        tl.store(V8 + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
                 _cvt2d(v / vch, BLOCK_R, BLOCK_D), mask=m)


# ---------------------------------------------------------------------------
# Output-buffer reuse.
#
# F009: the prologue KERNEL is not the prologue COST. At N=2048
# `_quant_kv_rows_fp8` runs in 0.0365 ms (348 GB/s) and is already at the
# optimum of a full BLOCK_R x num_warps sweep -- yet the prologue contributes
# 0.094 ms end-to-end. The missing ~0.058 ms is host-side: four `torch.empty`
# allocations plus a launch, on a kernel that only takes 36 us.
#
# These buffers are pure scratch (written by the kernel, read immediately by the
# attention kernel, never aliased by the caller), so they can be reused across
# calls. Keyed by (name, shape, device) so a different shape allocates afresh.
# ---------------------------------------------------------------------------
_BUFCACHE = {}


def _scratch(name, shape, dtype, device):
    key = (name, tuple(shape), device)
    t = _BUFCACHE.get(key)
    if t is None:
        t = torch.empty(shape, dtype=dtype, device=device)
        _BUFCACHE[key] = t
    return t


def quant_kv_rows_fp8(k, v, block_r=16, block_d=None, reuse=False,
                      mean=None, n_seq=None, vmean=None, vch=None):
    """k, v: (B,H,N,D) contiguous fp16 -> (k8, sk, v8, sv). One launch for both.

    `reuse=True` serves the outputs from a module-level scratch cache. It was
    implemented to attack the host-side prologue overhead described above, and
    **measured no faster** (F009): the prologue gap over the standalone kernel
    time was 0.094 ms without it and 0.104 ms with it, i.e. within the measurement
    noise and if anything worse. It is therefore OFF by default -- the hidden
    aliasing (a cached buffer is silently overwritten by the next call) is not
    worth paying for an unproven gain. Kept only so the experiment is
    reproducible.

    F082: pass `mean` (shape `(B,H,D)`, from `kseq_mean_fp8`) and `n_seq`
    to apply `smooth_k` to **K only** inside the fused launch. Both must be given
    together, exactly as in `quant_rows_fp8`: `n_seq` maps a flattened row back to its
    `(b,h)` for the mean lookup, and defaulting it would silently index the wrong
    channel group. `mean=None` (the default) is byte-identical to the pre-F082 call.

    F086: pass `vmean` (shape `(B,H,D)`, also from `kseq_mean_fp8`, but
    computed on **V**) to apply `smooth_v` to **V only** inside the same launch.
    `vmean` shares `n_seq` with `mean`. `vmean` is NOT added back here -- it is
    returned implicitly to the caller, who must add it to the output once (see
    `fa_fp8.flash_attn_fp8`'s `smooth_v`). The two means are independent: giving
    `mean` must not move V's bytes and giving `vmean` must not move K's.

    F100: `vch` is a `(B*H, D)` fp32 per-CHANNEL V divisor; see
    `_quant_kv_rows_fp8`. It requires `n_seq` and is byte-identical to the pre-F100
    call when omitted.
    """
    D = k.shape[-1]
    kv = k.reshape(-1, D)
    vv = v.reshape(-1, D)
    R = kv.shape[0]
    if reuse:
        k8 = _scratch("k8", (R, D), torch.float8_e4m3fn, k.device)
        v8 = _scratch("v8", (R, D), torch.float8_e4m3fn, v.device)
        sk = _scratch("sk", (R,), torch.float32, k.device)
        sv = _scratch("sv", (R,), torch.float32, v.device)
    else:
        k8 = torch.empty_like(kv, dtype=torch.float8_e4m3fn)
        v8 = torch.empty_like(vv, dtype=torch.float8_e4m3fn)
        sk = torch.empty(R, device=k.device, dtype=torch.float32)
        sv = torch.empty(R, device=k.device, dtype=torch.float32)
    bd = block_d or max(16, _next_pow2(D))
    grid = (triton.cdiv(R, block_r),)
    if mean is not None or vmean is not None or vch is not None:
        if n_seq is None:
            raise ValueError("quant_kv_rows_fp8: `mean`/`vmean`/`vch` require `n_seq`")
        # The OFF path is the `else` below and is unchanged. When only one of
        # the two means is given the other pointer is still passed (as the raw
        # input tensor, never dereferenced) because a `tl.constexpr` dead branch
        # emits no code -- the alternative, a third kernel entry point, would
        # multiply the compiled variants for no benefit.
        _quant_kv_rows_fp8[grid](kv, vv, k8, v8, sk, sv, R, D,
                                 kv.stride(0), kv.stride(1), vv.stride(0), vv.stride(1),
                                 MEAN=(mean.reshape(-1, D) if mean is not None else kv),
                                 SUB_MEAN=(1 if mean is not None else 0),
                                 N_SEQ=int(n_seq),
                                 VMEAN=(vmean.reshape(-1, D) if vmean is not None else vv),
                                 SUB_VMEAN=(1 if vmean is not None else 0),
                                 VCH=(vch.reshape(-1, D) if vch is not None else vv),
                                 CHAN_SV=(1 if vch is not None else 0),
                                 BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    else:
        _quant_kv_rows_fp8[grid](kv, vv, k8, v8, sk, sv, R, D,
                                 kv.stride(0), kv.stride(1), vv.stride(0), vv.stride(1),
                                 BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    return (k8.view(k.shape), sk.view(k.shape[:-1]),
            v8.view(v.shape), sv.view(v.shape[:-1]))


def quant_kv_rows_i8(k, v, block_r=16, block_d=None, mean=None, n_seq=None,
                     vmean=None, vch=None):
    """F083: k, v (B,H,N,D) contiguous fp16 -> (k8 int8, sk, v8 e4m3, sv).

    The "8+8 split"'s prologue: **K is int8, V stays e4m3** -- V's bytes are
    produced by the same `_cvt2d` the fp8 path uses, so the PV half is
    untouched. `mean`/`n_seq` apply `smooth_k` to K ONLY (see
    `_quant_kv_rows_i8`), exactly as `quant_kv_rows_fp8` does.

    F094: `vmean` (shape `(B,H,D)`, also from `kseq_mean_fp8`, but
    computed on **V**) applies `smooth_v` to **V only** inside the same launch,
    mirroring `quant_kv_rows_fp8`. It shares `n_seq` with `mean` and is NOT added
    back here -- the caller (`fa_fp8.flash_attn_fp8`) adds it to the output once.
    The two means are independent: `mean` must not move V's bytes and `vmean`
    must not move K's. Both omitted (the default) is byte-identical to pre-F094.
    """
    D = k.shape[-1]
    kv = k.reshape(-1, D)
    vv = v.reshape(-1, D)
    R = kv.shape[0]
    k8 = torch.empty_like(kv, dtype=torch.int8)
    v8 = torch.empty_like(vv, dtype=torch.float8_e4m3fn)
    sk = torch.empty(R, device=k.device, dtype=torch.float32)
    sv = torch.empty(R, device=k.device, dtype=torch.float32)
    bd = block_d or max(16, _next_pow2(D))
    grid = (triton.cdiv(R, block_r),)
    if mean is not None or vmean is not None or vch is not None:
        if n_seq is None:
            raise ValueError("quant_kv_rows_i8: `mean`/`vmean`/`vch` require `n_seq`")
        # The OFF path is the `else` below and is unchanged. When only one of
        # the two means is given the other pointer is still passed (as the raw
        # input tensor, never dereferenced) because a `tl.constexpr` dead branch
        # emits no code -- exactly the F086 contract in `quant_kv_rows_fp8`.
        _quant_kv_rows_i8[grid](kv, vv, k8, v8, sk, sv, R, D,
                                kv.stride(0), kv.stride(1), vv.stride(0), vv.stride(1),
                                MEAN=(mean.reshape(-1, D) if mean is not None else kv),
                                SUB_MEAN=(1 if mean is not None else 0),
                                N_SEQ=int(n_seq),
                                VMEAN=(vmean.reshape(-1, D) if vmean is not None else vv),
                                SUB_VMEAN=(1 if vmean is not None else 0),
                                VCH=(vch.reshape(-1, D) if vch is not None else vv),
                                CHAN_SV=(1 if vch is not None else 0),
                                BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    else:
        _quant_kv_rows_i8[grid](kv, vv, k8, v8, sk, sv, R, D,
                                kv.stride(0), kv.stride(1), vv.stride(0), vv.stride(1),
                                BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    return (k8.view(k.shape), sk.view(k.shape[:-1]),
            v8.view(v.shape), sv.view(v.shape[:-1]))


# ---------------------------------------------------------------------------
# F104: the chan prologue's quantiser WITHOUT the redundant V work.
#
# The F100 `CHAN_SV=1` contract (see `_quant_kv_rows_fp8`) deliberately keeps the
# shipped per-ROW V block and then OVERRIDES it, so the flag-off codegen cannot
# move.  Measured consequence (`F102` attribution): the chan arm pays
#   * one redundant per-row V absmax (register work), and
#   * one redundant `V8` STORE of `(B*H*N*D)` fp8 bytes.
# At `(1,48,8771,128)` that store is 53.88 MB -- half the `sk_chan - sk` byte delta.
#
# These kernels are the FUSED form: K exactly as before, V written ONCE against
# `VCH` with `sv = 1.0`, and no row-`sv` computation at all.  They are NEW kernels
# -- `_quant_kv_rows_fp8`/`_i8` are untouched -- so the flag-off binary cannot move
# (it is not even the same function).
# ---------------------------------------------------------------------------
@triton.jit
def _quant_kv_chan_fused_fp8(K, V, K8, V8, SK, SV, R, D,
                             stride_kr, stride_kd, stride_vr, stride_vd,
                             BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
                             MEAN=None, SUB_MEAN: tl.constexpr = 0,
                             N_SEQ: tl.constexpr = 0,
                             VMEAN=None, SUB_VMEAN: tl.constexpr = 0, VCH=None):
    """F104: K per-row fp8; V per-CHANNEL fp8 in ONE store; `sv` written as 1.0.

    Byte-for-byte the same `K8`/`SK`/`V8`/`SV` as `_quant_kv_rows_fp8` with
    `CHAN_SV=1`, minus the redundant row-`sv` pass and the first `V8` store.
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    rm = rows < R
    m = rm[:, None] & (cols < D)[None, :]
    k = tl.load(K + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
                mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        bh = rows // N_SEQ
        mu = tl.load(MEAN + bh[:, None] * D + cols[None, :],
                     mask=(cols < D)[None, :], other=0.0)
        k = k - mu
    if SUB_VMEAN:
        bhv = rows // N_SEQ
        vmu = tl.load(VMEAN + bhv[:, None] * D + cols[None, :],
                      mask=(cols < D)[None, :], other=0.0)
        v = v - vmu
    sk = tl.maximum(tl.max(tl.abs(k), 1) / SCALE_MAX, EPS)
    vch = tl.load(VCH + (rows // N_SEQ)[:, None] * D + cols[None, :],
                  mask=(cols < D)[None, :], other=1.0)
    tl.store(SK + rows, sk, mask=rm)
    tl.store(SV + rows, tl.full([BLOCK_R], 1.0, tl.float32), mask=rm)
    tl.store(K8 + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
             _cvt2d(k / sk[:, None], BLOCK_R, BLOCK_D), mask=m)
    tl.store(V8 + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
             _cvt2d(v / vch, BLOCK_R, BLOCK_D), mask=m)


@triton.jit
def _quant_kv_chan_fused_i8(K, V, K8, V8, SK, SV, R, D,
                            stride_kr, stride_kd, stride_vr, stride_vd,
                            BLOCK_R: tl.constexpr, BLOCK_D: tl.constexpr,
                            MEAN=None, SUB_MEAN: tl.constexpr = 0,
                            N_SEQ: tl.constexpr = 0,
                            VMEAN=None, SUB_VMEAN: tl.constexpr = 0, VCH=None):
    """F104 int8 twin: K per-row int8, V per-CHANNEL e4m3 in ONE store."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_D)
    rm = rows < R
    m = rm[:, None] & (cols < D)[None, :]
    k = tl.load(K + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
                mask=m, other=0.0).to(tl.float32)
    v = tl.load(V + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
                mask=m, other=0.0).to(tl.float32)
    if SUB_MEAN:
        bh = rows // N_SEQ
        mu = tl.load(MEAN + bh[:, None] * D + cols[None, :],
                     mask=(cols < D)[None, :], other=0.0)
        k = k - mu
    if SUB_VMEAN:
        bhv = rows // N_SEQ
        vmu = tl.load(VMEAN + bhv[:, None] * D + cols[None, :],
                      mask=(cols < D)[None, :], other=0.0)
        v = v - vmu
    sk = tl.maximum(tl.max(tl.abs(k), 1) / INT8_MAX, EPS)
    vch = tl.load(VCH + (rows // N_SEQ)[:, None] * D + cols[None, :],
                  mask=(cols < D)[None, :], other=1.0)
    tl.store(SK + rows, sk, mask=rm)
    tl.store(SV + rows, tl.full([BLOCK_R], 1.0, tl.float32), mask=rm)
    tl.store(K8 + rows[:, None] * stride_kr + cols[None, :] * stride_kd,
             _cvt2d_i8(k / sk[:, None], BLOCK_R, BLOCK_D), mask=m)
    tl.store(V8 + rows[:, None] * stride_vr + cols[None, :] * stride_vd,
             _cvt2d(v / vch, BLOCK_R, BLOCK_D), mask=m)


def _quant_chan_fused_common(k, v, kern, k_dtype, vch, mean, vmean, n_seq,
                             block_r=16, block_d=None):
    if n_seq is None:
        raise ValueError("quant_kv_chan_fused: `vch` requires `n_seq`")
    if vch is None:
        raise ValueError("quant_kv_chan_fused: `vch` is required")
    D = k.shape[-1]
    kv = k.reshape(-1, D)
    vv = v.reshape(-1, D)
    R = kv.shape[0]
    k8 = torch.empty_like(kv, dtype=k_dtype)
    v8 = torch.empty_like(vv, dtype=torch.float8_e4m3fn)
    sk = torch.empty(R, device=k.device, dtype=torch.float32)
    sv = torch.empty(R, device=k.device, dtype=torch.float32)
    bd = block_d or max(16, _next_pow2(D))
    grid = (triton.cdiv(R, block_r),)
    kern[grid](kv, vv, k8, v8, sk, sv, R, D,
               kv.stride(0), kv.stride(1), vv.stride(0), vv.stride(1),
               MEAN=(mean.reshape(-1, D) if mean is not None else kv),
               SUB_MEAN=(1 if mean is not None else 0),
               N_SEQ=int(n_seq),
               VMEAN=(vmean.reshape(-1, D) if vmean is not None else vv),
               SUB_VMEAN=(1 if vmean is not None else 0),
               VCH=vch.reshape(-1, D),
               BLOCK_R=block_r, BLOCK_D=bd, num_warps=4)
    return (k8.view(k.shape), sk.view(k.shape[:-1]),
            v8.view(v.shape), sv.view(v.shape[:-1]))


def quant_kv_chan_fused_fp8(k, v, vch, mean=None, vmean=None, n_seq=None,
                            block_r=16, block_d=None):
    """F104: the fused chan prologue, fp8 QK arm.  See `_quant_kv_chan_fused_fp8`."""
    return _quant_chan_fused_common(k, v, _quant_kv_chan_fused_fp8,
                                    torch.float8_e4m3fn, vch, mean, vmean, n_seq,
                                    block_r=block_r, block_d=block_d)


def quant_kv_chan_fused_i8(k, v, vch, mean=None, vmean=None, n_seq=None,
                           block_r=16, block_d=None):
    """F104: the fused chan prologue, int8 QK arm.  See `_quant_kv_chan_fused_i8`."""
    return _quant_chan_fused_common(k, v, _quant_kv_chan_fused_i8,
                                    torch.int8, vch, mean, vmean, n_seq,
                                    block_r=block_r, block_d=block_d)


def quant_q_fp8(q, block_d=None):
    """q: (..., D) contiguous, fp16/bf16. Returns (q8 fp8, scale (...,) fp32)."""
    D = q.shape[-1]
    x = q.reshape(-1, D)
    R = x.shape[0]
    q8 = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = torch.empty(R, device=q.device, dtype=torch.float32)
    bd = block_d or max(16, _next_pow2(D))
    _quant_row_fp8[(R,)](x, q8, s, D, x.stride(0), x.stride(1), BLOCK_D=bd, num_warps=4)
    return q8.view(q.shape), s.view(q.shape[:-1])


def quant_kv_fp8(x, center=True, write_mean=False, block_n=128, block_d=None):
    """x: (B, H, N, D) contiguous. Per-channel scale reduced over N.

    Returns (x8 fp8 (B,H,N,D), scale (B,H,D) fp32, mean (B,H,D) fp32 or None).
    """
    B, H, N, D = x.shape
    xv = x.reshape(B * H, N, D)
    x8 = torch.empty_like(xv, dtype=torch.float8_e4m3fn)
    s = torch.empty((B * H, D), device=x.device, dtype=torch.float32)
    mean = (torch.empty((B * H, D), device=x.device, dtype=torch.float32)
            if write_mean else s)  # dummy pointer when unused
    bd = block_d or max(16, _next_pow2(D))
    _quant_channel_fp8[(B * H,)](
        xv, x8, s, mean, N, D,
        xv.stride(0), xv.stride(1), xv.stride(2),
        BLOCK_N=block_n, BLOCK_D=bd, CENTER=center, WRITE_MEAN=write_mean,
        num_warps=4)
    m_out = mean.view(B, H, D) if write_mean else None
    return x8.view(B, H, N, D), s.view(B, H, D), m_out


# --------------------------------------------------------------------------
# PyTorch reference implementations — used to VERIFY the Triton kernels above.
# --------------------------------------------------------------------------
def ref_quant_row_fp8(q, scale_max=448.0):
    D = q.shape[-1]
    x = q.reshape(-1, D).float()
    s = x.abs().amax(dim=1).clamp_min(1e-12) / scale_max
    x8 = (x / s[:, None]).to(torch.float8_e4m3fn)
    return x8.view(q.shape), s.view(q.shape[:-1])


def ref_quant_row_i8(x):
    """F083: the torch reference for the int8 row quantiser, in the convention
    F083 fixes -- symmetric, no zero point,
    round-half-to-even, clamp to +-127.

    Matches `_quant_rows_i8` exactly: `s = max(amax, 1e-12) / 127`. There is NO
    `max(amax, 1)` floor anywhere in this project -- in the kernels the `1` in
    `tl.max(tl.abs(x), 1)` is the REDUCTION AXIS, not a clamp. (An earlier version
    of this reference added a spurious `clamp_min(1.0)` and reported a 49 % byte
    mismatch that was entirely its own bug.)

    A 1-ULP scale difference vs the Triton kernels remains and is PRE-EXISTING:
    measured `max|ds| = 2.98e-8` on 3.9 % of rows for BOTH the int8 kernel and the
    untouched fp8 kernel. Byte comparisons against this reference must therefore
    use an LSB tolerance, not `torch.equal`.
    """
    D = x.shape[-1]
    xv = x.reshape(-1, D).float()
    s = (xv.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-12)
    q = torch.round(xv / s).clamp_(-127, 127).to(torch.int8)
    return q.view(x.shape), s.view(x.shape[:-1])


def ref_quant_channel_fp8(x, center=True, write_mean=False, scale_max=448.0):
    B, H, N, D = x.shape
    xf = x.float()
    mean = xf.mean(dim=2) if center else None
    xc = xf - mean[:, :, None, :] if center else xf
    s = xc.abs().amax(dim=2).clamp_min(1e-12) / scale_max
    x8 = (xc / s[:, :, None, :]).to(torch.float8_e4m3fn)
    return x8, s, (mean if write_mean else None)
