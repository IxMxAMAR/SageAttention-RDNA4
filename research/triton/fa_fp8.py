"""FP8 flash-attention for gfx1201 (RDNA4).

Why fp8: AMD's documented RDNA4 matrix rate is **2048 FLOPS/clk/CU for FP8 and INT8 vs 1024 for
FP16/BF16** — a **2.0x** ratio (64 CU x 2.970 GHz => 389.28 fp8 / 194.64 fp16 TFLOP/s). Our own
measurements reach ~196 TFLOP/s for `v_wmma_f32_16x16x16_fp8_fp8` vs ~108 for `..._f16`; that is a
**utilisation** difference (~50% vs ~55% of the respective peaks), NOT an instruction-rate ratio.
An earlier version of this comment called it "~1.81x the rate" — that was wrong, see `F079`.
Torch SDPA (fp16) reaches only 30-68 TFLOP/s on our shapes.

Scale placement
---------------
A per-element scale factors out of a matmul only if it is constant along that
matmul's REDUCTION axis. QK^T reduces over head_dim; PV reduces over the
sequence. So naively V needs a per-CHANNEL scale (reduced over N), which forces
a sequence-axis reduction in the prologue — measured at 3.15 ms/tensor, 19x
slower than torch's own amax, because it can only use B*H = 8 programs.

We remove that reduction entirely by quantizing V per ROW and folding V's
dequantization scale into P instead:

    v8[n,d] = v[n,d] / sv[n]                  (sv depends only on n)
    p8[m,n] = p[m,n] * sv[n] * P_SCALE        (quantized to e4m3)
    sum_n p8[m,n]*v8[n,d] = P_SCALE * sum_n p[m,n]*v[n,d]      -- exact

The fold costs one broadcast multiply on a tile we already scale for P_SCALE,
and it makes all three tensors quantizable by one embarrassingly-parallel
per-row kernel. No transpose, no pad, no permute, no sequence reduction.

P_SCALE
-------
P lies in [0,1] after the max subtraction; e4m3's smallest normal is 2**-6, so a
plain cast leaves small probabilities subnormal. Pre-scaling by 448 shifts them
into the normal range. The factor is constant and folds into the final rescale
for free — strictly better than a plain cast, at zero cost.

Verified: the emitted AMDGCN contains v_wmma_f32_16x16x16_fp8_fp8 and no FMA
fallback (the AMD backend's min_dot_size is (1,1,1) and fails SILENTLY, so this
must be checked for every configuration).
"""
import torch
import triton
import triton.language as tl

from quant_triton import (quant_rows_fp8, quant_kv_rows_fp8, kseq_mean_fp8,
                          kseq_amax_fp8,
                          quant_rows_i8, quant_kv_rows_i8, _cvt2d_i8,
                          kseq_mean_amax_fp8, quant_kv_chan_fused_fp8,
                          quant_kv_chan_fused_i8)

FP8 = tl.float8e4nv
LOG2E = tl.constexpr(1.4426950408889634)
LN2 = tl.constexpr(0.6931471805599453)

# F083 -- the int8-QK ("8+8 split") quantiser constants.
#
# The QK half of the split moves to signed int8 with a per-token scale of
# `amax/127` and a symmetric round-to-nearest-even clamp; the PV half stays
# e4m3. `smooth_k` is a MANDATORY co-requisite for int8 QK (F079, F081):
# without it int8 QK scores 30.60 %/62.24 % cos-sim, with it 99.31 %/
# 99.47 %. It is reachable from here since F082 (the fused K/V prologue).
#
# The `max(amax, 1)` floor and the 1e-12 epsilon are the fp8 path's, kept
# verbatim with 127 in place of 448 so the two quantisers differ in nothing but
# the mantissa budget. `SIGNED` int32 -> fp32 (`sitofp`) is required: the
# products are signed and an unsigned convert would turn every negative score
# into ~4.3e9.
INT8_MAX = 127.0

# F017 experiment switch: which fp32->e4m3 P-conversion path the kernel uses.
# Default 1 == the incumbent `_cvt_pk_fp8_2d`, so importing this module and
# calling flash_attn_fp8() reproduces the previously-shipped behaviour exactly.
# See F017.
PCAST = 1

# F021 experiment switch: division lowering in the two broadcast divides.
#
# The kernel contains 1361 AMDGCN instructions from IEEE fdiv (272 v_rcp, 544
# v_div_scale, 272 v_div_fmas, 272 v_div_fixup = 27% of kernel TEXT). Verified
# from the disassembly that ALL of them are in the PROLOGUE/EPILOGUE -- the KV
# loop body (557 instructions) contains ZERO. So this is a static-text cost, not
# a per-iteration cost, and the expected wall-clock effect is small (largest at
# N=2048, where the prologue is the biggest fraction of total work).
#
# FDIV=0  the incumbent: `qf / qs[:, None]` and `acc / (l_i * P_SCALE)[:, None]`
# FDIV=1  hoist the per-ROW reciprocal and multiply:
#           `qf * (1.0 / qs)[:, None]`, `acc * (1.0 / (l_i * P_SCALE))[:, None]`
#         which is 16+16 divisions per program instead of 128+128, and turns the
#         128 broadcast divides into plain multiplies.
#
# Numerics: the two forms differ by at most 1 ulp of the reciprocal, and both
# division sites feed a 3-mantissa-bit fp8 quantisation (line 228) or the final
# fp16 output (line 308), so the error is far below the representable step. The
# bit-identity of the fp8 codes is the load-bearing claim and is measured, not
# assumed -- see F021.
#
# FDIV=1 is the default because it is measured correct AND non-slower at every
# shape (see F021). FDIV=0 is kept as the exact incumbent for A/B.
FDIV = 1

# F098 experiment switch: the DYNAMIC per-query-row P scale (`PSC_DYN`).
#
# WHY. `F097` established, and then validated out of sample, that the `sv`->P fold
#     ps[m,n] = e4m3( p[m,n] * sv[n] * P_SCALE )       
# pushes the folded tensor toward e4m3's SUBNORMAL floor: the median folded value is
# ~1e-5 while e4m3's smallest subnormal is 2^-9 = 1.95e-3, so the attention weights are
# quantised with an ABSOLUTE error rather than a relative one. `smooth_v` shrinks `sv`
# and therefore makes that penalty worse -- which is why the `smooth_v` A/B reverses
# with `latent_scale` (`F095`, `F096`).
#
# THE FIX. A factor that depends only on the QUERY row `m` commutes with the sum over
# keys, so it can be applied and undone EXACTLY:
#     C[m] = 1 / max_n ( p[m,n] * sv[n] * P_SCALE )
#     ps   = p * sv * P_SCALE * C[m]
#     out  = (ps @ v8) / (P_SCALE * C[m])
#
# A single pass cannot see all keys, so the kernel uses a RUNNING per-query-row max
# `pmax` and scales the accumulator by `r_old/r_new` each time it grows -- folded into
# the EXISTING online-softmax `alpha` multiply, so the steady-state cost is one extra
# (BLOCK_M,) vector multiply plus one extra `tl.max` reduction per KV iteration.
# PROVABLY SAFE: `t * C = t / r <= 1`, so the e4m3 cast can never overflow (the
# hardware convert does NOT saturate -- it yields NaN), and because scaling up by
# s >= 1 never increases e4m3's error, this is elementwise no worse than F097's
# global-max probe.  See F098.
#
# DEFAULT 0. `PSC_DYN=0` emits no code and must be provably bit-identical to the
# pre-F098 path (instruction counts + binary hashes + bit-identical outputs).
PSC_DYN = 0

# F099 experiment switch: fold the per-row P scale into the SOFTMAX ROW MAX (`PSC_FOLD`).
#
# WHY. `F098` implemented `F097`'s exact per-row scale and it removed the `smooth_v` reversal, but
# it cost +15.6 % of a full call (`--isa-dyn`: 1088 -> 1296 instructions, +208) because the scale
# is applied with a `(BLOCK_M, BLOCK_N)` multiply (98 of the 208). A per-row factor can instead be
# applied in the EXPONENT -- `p = exp2(qk - shift)` with a per-row `shift` -- for free.
#
# THE DESIGN. State is `l2u = log2(u)` (the scale to use at the NEXT tile) and `l2r` (the log-ratio
# applied at the previous tile, so `alpha` can carry it):
#     m_new   = max(m_i, rowmax(qk))
#     l2u_used= min(l2u, 448 / (P_SCALE * pumax * svmax))     -- PROVABLE clamp, see below
#     shift   = m_new - l2u_used  ( - log2(P_SCALE) when PSC )
#     alpha   = exp2(m_i - m_new + (l2u_used - l2u) + l2r)    -- carries u_j/u_{j-1} for free
#     p       = exp2(qk - shift)                              -- ALREADY carries u
#     l_i     = l_i*alpha + rowsum(p)
#     ps      = p * vs ( * P_SCALE )                          -- the shipped `sv` fold, unchanged
#     l2r     = -log2(max(rowmax(ps), 1));  l2u = l2u_used + l2r
#     acc     = acc*alpha + dot(e4m3(ps), v8)
#   epilogue: the SHIPPED one -- NO `acc * pmax`, unlike PSC_DYN.
#
# EXACT, for ANY positive per-row `u`: it cancels in `acc / l_i` because `alpha` carries the ratio.
#
# THE SCALE IS NECESSARILY LAGGED, AND THAT IS WHY THIS ARM IS NOT SHIPPABLE.
# `p` must be exponentiated before the tile's own max is known, so `u_j = 1/max(t over tiles < j)`.
# `F098` section 8.4 claimed "t*C = t/pmax <= 1 still holds" -- THAT CLAIM IS FALSE: the current
# tile is quantised at the PREVIOUS tile's scale, so `ps_j = t_j/pmax_{j-1}` can exceed 448 and
# `_cvt_pk_fp8_f32` does NOT saturate (it yields NaN). The clamp above restores the bound
# (`t <= P_SCALE * pumax * svmax` because `p_un <= 1`), so this arm is safe -- but it is not
# ACCURATE: the F099 check measures the lagged scale as WORSE than both `F098`'s
# running max AND the unfixed arm on the `smooth_v` axis. See
# F099.
#
# DEFAULT 0. `PSC_FOLD=0` emits no code and must be bit-identical to the pre-F099 path.
PSC_FOLD = 0

# F100 -- `VSCALE_CHAN`: the `sv`-fold attack.
#
# `F097` identified the root cause of the `smooth_v` reversal: the kernel folds V's
# per-ROW dequantisation scale `sv` into the P quantisation, so the tensor that is
# e4m3-quantised for P is `p * sv * P_SCALE == p * max(amax_d V, 1)`.  P's e4m3 grid
# is therefore V's row amax -- NOT `P_SCALE` -- and `smooth_v` shrinks it toward
# e4m3's subnormal floor.  `F098` repaired the symptom with a per-query-row scale
# (+15.6 % of a full call); `F099` folded that scale into the softmax row max and
# showed the one-tile lag is the defect, closing the cheap-repair route.
#
# `VSCALE_CHAN=1` removes the COUPLING instead of the symptom.  V is quantised
# against a per-CHANNEL divisor `b[d]` (the sequence-axis absmax of the CENTRED V,
# from `kseq_amax_fp8`), the per-KEY `sv` tensor is written as 1.0 so the fold
# contributes nothing, and `b[d]` is re-applied once in the epilogue -- where it is
# a `(HEAD_DIM,)` multiply, because the channel axis is not the PV reduction axis.
# P therefore keeps e4m3's FULL `P_SCALE` grid at every `latent_scale`.
#
# THE CHOICE OF `b[d]` IS FORCED, NOT TUNED.  The fold is exact for ANY per-key
# scale: with `p8 = e4m3(p_un * a)` and `v8 = e4m3(V * P_SCALE / a)` the product is
# `p_un * V * P_SCALE` for every `a`.  e4m3's range is [-448, 448] and `p_un <= 1`,
# so the two constraints are `a <= P_SCALE` (P cannot overflow) and
# `a >= amax|V|` (V cannot overflow).  To keep `a` as LARGE as possible for P we take
# `a = P_SCALE`; that is what makes the per-key `sv` 1.0 and V's divisor
# `b = a/P_SCALE = 1` per channel, rescaled by the channel absmax.  See
# F100.
#
# DEFAULT 0.  `VSCALE_CHAN=0` emits no code in EITHER kernel: the branch is a
# `tl.constexpr`, so the shipped path must be bit-identical.
VSCALE_CHAN = 0

# F116 -- `Q_IN_KERNEL`: the module-level default for whether the
# PROLOGUE skips Q and the attention kernel quantises it in-register.
#
# **The in-kernel quantiser itself is NOT new.** `_attn_fwd_fp8`'s `FUSE_Q` constexpr
# already loads Q fp16, takes its per-row absmax, floors at 1, divides by 448 (127 for
# int8) and converts in-register; its docstring calls it "the incumbent's design (PR #368
# has no Q prologue at all)". `flash_attn_fp8`'s `fuse_q=True` is its DEFAULT and
# activates whenever the caller does not supply `q8` -- so the graph runner
# already quantises Q in-kernel.
#
# What is new here is only the switch that lets the `prologue_fp8` call pattern skip its Q
# pass so the kernel's existing fused path can be used, and the `(None, None)` normalisation
# that makes the two compose.  DEFAULT 0 = the earlier path, byte-identical: the
# Triton kernel source is untouched, so no compiled binary changes.
Q_IN_KERNEL = 0

# F104 -- `VSCALE_CHAN_FUSED`: the FUSED chan prologue.
#
# `F102` priced `VSCALE_CHAN` at +7.28 % (`sk_chan/sk`) / +11.12 % (`sv_chan/sv`)
# composed and FALSIFIED both frozen bars; `F103` then showed the attention kernel's
# own ISA delta is only +102 instructions (≈4 %, 0 spills), so `[INFERENCE]` the cost
# is in the PROLOGUE.  `F104`'s source + launch census attributes it exactly:
#   * `sk_chan` vs `sk`  = +2 launches, +1 read of V (the `kseq_amax_fp8` pass),
#                          +1 redundant `V8` store (53.88 MB at the production shape);
#   * `sv_chan` vs `sv`  = the same +1 read, because `sv` already pays one V read
#                          for the mean, and `sv_chan` pays a SECOND for the amax.
#
# `VSCALE_CHAN_FUSED=1` removes both:
#   * `kseq_mean_amax_fp8` produces the sequence-axis mean AND the centred per-channel
#     absmax in ONE read of V (bit-identical -- see `quant_triton`'s F104 block);
#   * `quant_kv_chan_fused_{fp8,i8}` write V ONCE against `VCH` with `sv = 1.0`,
#     with no row-`sv` pass and no redundant `V8` store.
# It is a NEW path through NEW kernels, so `VSCALE_CHAN_FUSED=0` cannot move the
# shipped code: the flag-off binary is the pre-F104 binary by construction (and is
# proved so by the F104 check).
# DEFAULT 0, and it requires `vscale_chan`.
VSCALE_CHAN_FUSED = 0

# ---------------------------------------------------------------------------
# LAZY softmax rescale.  DEFAULT 0 == no code emitted.
#
# The online-softmax accumulator rescale `acc = acc * alpha[:, None]` is emitted
# unconditionally today, once per KV iteration, over the whole (BLOCK_M, HEAD_DIM)
# tile.  `LAZY_RESCALE=1` makes it CONDITIONAL: the accumulator stays in the scale of
# the last rescale (`m_i` IS that scale, not the running max) and the rescale is taken
# only when the CTA-wide largest per-ROW growth of the tile max since that rescale
# exceeds `LAZY_TAU` log2 units.
#
# Provenance: this is the FA4 `rescale_threshold` (SoftmaxSm100, softmax.py:306, value
# 8.0 for fp16/bf16 and 0.0 for fp8, which instead carries `max_offset=8`) and
# comfy-kitchen PR #194's `kSoftmaxHeadroom = 8.0f` (`int8_attn.hip:65`).  BOTH use a
# WARP-uniform vote over per-row predicates (`cute.arch.vote_ballot_sync`,
# `__any(update_max)`); **Triton cannot express a warp-scope vote**, so this is the
# CTA-uniform analogue and its take-probability is strictly HIGHER (any of the 4 warps
# forces the rescale).  That difference is the central risk of this stage.
#
# EXACT in real arithmetic: `m_i` is advanced only together with the same `alpha`
# applied to `acc` AND `l_i`, so every P is quantised in the scale the accumulator is
# actually in.  The fp8 P consequence is the whole accuracy risk: with a stale max, P
# can reach `2**LAZY_TAU`, so the P scale must satisfy
# `max(amax(V),1) * 2**LAZY_TAU <= 448` (e4m3 max finite 448, min normal 2**-6,
# subnormal step 2**-9; the pack `_cvt_pk_fp8_2d` clamps at 448.0).
#
# DEFAULT 0.  `LAZY_RESCALE=1` requires KQT=0, PSC_FOLD=0, PSC_DYN=0, TAIL_PEEL=0,
# EEXP2=0 and LAZY_TAU>0; the wrapper REFUSES every other combination.
LAZY_RESCALE = 0
LAZY_TAU = 0.0


# ---------------------------------------------------------------------------
# Hardware fp32 -> e4m3 pack.
#
# gfx1201 HAS this instruction: `v_cvt_pk_fp8_f32` packs two fp32 into two e4m3
# in ONE instruction. Triton's default `.to(tl.float8e4nv)` does NOT use it --
# the backend lowers that to ~27 instructions of bit manipulation, which was
# measured at 48% of the KV loop body.
#
# Reaching it from Triton requires two non-obvious things (both discovered the
# hard way; an earlier attempt concluded "gfx12 has no hardware fp8 convert"
# because of them -- that conclusion was WRONG):
#   1. constraints must be `=v` (VGPR), not `=r`. With `=r` LLVM reports
#      "couldn't allocate output register for constraint 'r'" and ABORTS.
#   2. the output dtype must be declared int16 with pack=2. With a sub-32-bit
#      output dtype LLVM hits UNREACHABLE and aborts the process.
# With pack=2, one asm block (2 fp32 inputs) maps to one 32-bit register exposed
# as two int16 elements. The instruction writes the two fp8 bytes into the LOW
# half of that register, so the *element count is unchanged* (N fp32 in, N int16
# out) and the two bytes of the pair (2i, 2i+1) both live in output element 2i:
# byte0 = fp8(x[2i]) in the low byte, byte1 = fp8(x[2i+1]) in bits 15:8. Element
# 2i+1 is register garbage. `reshape(..., (M, N//2, 2))` + `tl.split` recovers
# the even elements; F017 measured that reshape+split to be free (the loop body
# is byte-identical without them).
#
# Numerics: bit-exact vs torch's round-to-nearest-even over 2**20 samples
# including subnormals. Unlike `.to(fp8)` it does NOT saturate (it yields NaN
# above 448), so the explicit clamp is required to make this a drop-in.
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
def _cvt_pk_fp8_2d(x, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    h = _cvt_pk_raw(tl.minimum(x, 448.0))
    lo, _hi = tl.split(tl.reshape(h, (BLOCK_M, BLOCK_N // 2, 2)))
    b = lo.to(tl.uint16, bitcast=True)
    return tl.interleave((b & 0xFF).to(tl.uint8),
                         (b >> 8).to(tl.uint8)).to(tl.float8e4nv, bitcast=True)


# ---------------------------------------------------------------------------
# F017 -- P-conversion variants (PCAST). See F017.
#
# The hypothesis under test: the `tl.reshape` + `tl.split` + `tl.interleave`
# round trip in `_cvt_pk_fp8_2d` costs more than the conversion itself, because
# `interleave` doubles the element count and therefore forces a layout
# conversion.
#
# MEASURED VERDICT (F017): the hypothesis is half right and the half that is
# right does not matter.
#   * `reshape` + `split` are FREE -- PCAST=2, which removes the reshape,
#     emits a loop body that is BYTE-IDENTICAL to PCAST=1 (575 instructions,
#     all 575 lines equal). `tl.split` on the asm result is a pure view.
#   * `interleave` is NOT free but is at its floor: it lowers to 66 `v_perm_b32`
#     per KV iteration (4 warps) = 16.5 per warp, against a floor of 16 (2 fp8
#     bytes per permute, 32 fp8 produced per warp per iteration).
#   * Removing the whole byte tail (mask + shift + interleave) buys NOTHING
#     measurable: 575 -> 497 instructions is 13.6% of the loop body, and the
#     timing difference is inside the spread at every shape. The kernel is not
#     issue-bound; it is latency/LDS-bound.
#
# So PCAST=2 is kept as an exact, bit-identical simplification (one fewer
# compiler-visible op, same code), NOT as a speedup. PCAST=1 remains the
# default so the previously-shipped behaviour is preserved.
#
# PCAST=0  plain `.to(fp8)` -- the software conversion used before the hardware
#          pack. Control for "the conversion itself is expensive": it costs
#          1.41-1.49x the hardware pack in the loop body.
# PCAST=1  the incumbent `_cvt_pk_fp8_2d`:
#          pack -> reshape(M,N/2,2) -> split -> and/shr -> interleave -> bitcast
# PCAST=2  the same code with the `reshape` dropped -- the asm block already
#          produces the (M, N/2) shape that `tl.split` wants after one more
#          reshape, so this is the form without the redundant view. Emits a loop
#          body BYTE-IDENTICAL to PCAST=1 (575 lines, all equal) and a
#          bit-identical output. This is the direct proof that `reshape` + `split`
#          cost nothing. Kept as a simplification, NOT a speedup.
# PCAST=3  reserved; routed to the incumbent. It was meant to price the
#          interleave by skipping it, but skipping it halves P's element count
#          and so requires a pair-summed V that costs more than it measures.
# PCAST=4  as PCAST=1 with the interleave operands swapped. Produces WRONG
#          output (max|diff| 2.6e-2 vs PCAST=1) at the same instruction count,
#          which is how we know the interleave is doing real lane work rather
#          than being a no-op the compiler could elide.
# ---------------------------------------------------------------------------
@triton.jit
def _cvt_pk_fp8_2d_v2(x, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """PCAST=2: byte extraction before the split, on the (M, N//2) int16 pack.

    Bit-identical to `_cvt_pk_fp8_2d` and emits a byte-identical loop body
    (verified: 575 instructions, every line equal), which is the evidence that
    the reshape/split half of the round trip is free.
    """
    h = _cvt_pk_raw(tl.minimum(x, 448.0))
    lo, _hi = tl.split(tl.reshape(h, (BLOCK_M, BLOCK_N // 2, 2)))
    b = lo.to(tl.uint16, bitcast=True)
    return tl.interleave((b & 0xFF).to(tl.uint8),
                         (b >> 8).to(tl.uint8)).to(tl.float8e4nv, bitcast=True)


@triton.jit
def _cvt_pk_fp8_2d_swap(x, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """PCAST=4: the incumbent with the interleave arguments swapped.

    WRONG OUTPUT by construction (measured max|diff| 2.6e-2 vs PCAST=1). Kept
    only because it is the control that proves `tl.interleave` is doing real
    lane work: swapping its two operands changes nothing about the instruction
    count and everything about the result.
    """
    h = _cvt_pk_raw(tl.minimum(x, 448.0))
    lo, _hi = tl.split(tl.reshape(h, (BLOCK_M, BLOCK_N // 2, 2)))
    b = lo.to(tl.uint16, bitcast=True)
    return tl.interleave((b >> 8).to(tl.uint8),
                         (b & 0xFF).to(tl.uint8)).to(tl.float8e4nv, bitcast=True)


# =====================================================================
# F037: EEXP2 -- the EMULATED exp2 (IEEE-754 exponent split + short polynomial)
# =====================================================================
#
# WHY. FlashAttention-4's second named technique is
# "software-emulated exponential ... that reduces non-matmul operations"
# (arXiv 2603.05451 section 3.1.3), on the thesis of ASYMMETRIC HARDWARE SCALING
# -- the tensor cores scale, the SFU does not. Nobody in the surveyed literature
# has measured the exp2 in an fp8 attention kernel, and this project never has
# either: F017 priced the *P-conversion* (the fp32->fp8 cast) and found it at its
# floor, but the exponential itself was never looked at.
#
# WHAT THE CENSUS SAYS WE EMIT TODAY (measured, F037 section 4):
#   `tl.math.exp2` lowers to exactly ONE `v_exp_f32_e32` per element on gfx1201
#   -- 33 in the N=2048 causal loop (32 for the 64x64 score tile over 128 lanes,
#   plus 1 for `alpha`), 17-18 at the other five shapes. It is a VOP1
#   transcendental with NO VOPD (dual-issue) form -- AMD's RDNA4 ISA lists the
#   dual-issue set as exactly 17 `V_DUAL_*` instructions and `V_EXP_F32` is not
#   among them -- so it can never pair into a `v_dual_*` VLIW slot. That is a
#   fact read from the ISA document, not an inference. (The ISA we hold
#   documents no THROUGHPUT figure for `V_EXP_F32`, so no rate claim is made.)
#
# THE CONSTRUCTION. 2^x = 2^floor(x) * 2^(x-floor(x)), with the integer part
# built by writing the IEEE-754 exponent field directly (integer ALU, no
# transcendental) and the fractional part on [0,1) by a short Horner polynomial:
#
#     n  = floor(x)                          -- v_floor_f32
#     f  = x - n                in [0, 1)    -- v_sub_f32
#     e  = bitcast_f32((n + 127) << 23)      -- v_cvt_i32_f32, v_add_u32, v_lshlrev
#     p  = poly(f)                           -- DEG x v_fma_f32
#     2^x = e * p                            -- v_mul_f32
#
# THE CLAMP IS LOAD-BEARING, NOT DEFENSIVE. `x` is `qk - m_new` in this
# kernel and is `-inf` in two common places: every causally masked score (the
# `tl.where(..., float("-inf"))` at the top of the loop) and `alpha` on the
# first iteration (`m_i` starts at `-inf`). Without the clamp,
# `(n + 127) << 23` shifts a NEGATIVE integer for `n < -127` -- poison in LLVM,
# and `-inf - (-inf)` makes the fractional part NaN. Clamping at exactly -127
# makes the exponent field 0, so the result is EXACTLY +0.0, which is what the
# hardware `v_exp_f32` returns for `-inf` and what the masked path needs.
#
# ACCURACY TARGET, STATED EXPLICITLY. The reference is the chunked fp32 SDPA
# of F029/F030 and the shipped kernel's own max-abs error is 1.1994e-03 at
# N=8192 (2.8723e-03 at N=1024) per F019 section 2. The fractional polynomial is
# a Chebyshev fit with its constant term PINNED to exactly 1.0 (so exp2 of an
# exact integer is exact, and `alpha` is exactly 1.0 on the many iterations
# where the running max does not move). Measured on CPU (a separate fitting
# script, not included), fp32 Horner, pessimistic no-FMA rounding, 5M points:
#
#     DEG=2   max rel err 4.74e-03   (79581 fp32 ULP)  -- NOT VIABLE, see below
#     DEG=3   max rel err 2.07e-04   ( 3470 fp32 ULP)  -- the CHEAP variant
#     DEG=4   max rel err 7.24e-06   (  121 fp32 ULP)  -- the DEFAULT
#     DEG=5   max rel err 2.91e-07   (    5 fp32 ULP)
#
# fp16 unit roundoff is 4.88e-04 and the fp8 e4m3 weight P is quantised to 3
# mantissa bits (2^-4 = 6.25% relative), so DEG=4 sits 67x below fp16 roundoff
# and ~8600x below the P quantisation it feeds; DEG=3 sits 2.4x below fp16
# roundoff. **DEG=2 sits 10x ABOVE fp16 roundoff and is excluded on accuracy,
# not on cost** -- which matters, because DEG=2 is by far the cheapest in the
# census (+1.1% loop at N=8192 non-causal, and it fixes dual-issue just as
# well). The accuracy floor, not the instruction count, is what picks the
# degree: DEG=4 by default, DEG=3 for a timing run that wants the cheapest
# accuracy-viable point (+1.8% loop at N=8192 non-causal, 2-4x cheaper than
# DEG=4).
#
# F037 VERDICT: CHANGES-CODEGEN-BUT-UNTESTED-PERFORMANCE, and WORSE BY
# CONSTRUCTION at N=2048/N=4096 NON-CAUSAL. The 64x16_w2 tile has no register
# headroom: the emulation takes it to 256 VGPRs with 32 spills and a 132-byte
# private segment, adding scratch_load_b32/scratch_store_b32 to the steady-state
# loop. That is DEG-independent (42 spills at DEG=2) and split-independent. Do
# not enable this flag at those two shapes. At N=8192 it is a genuine candidate
# (loop +1.8%, v_exp_f32 18->0, v_dual_mul_f32 26->100, v_mul_f32_e32 151->22,
# zero spills, shared memory and all 11 staging/layout opcodes unchanged) --
# but NO PERFORMANCE CLAIM IS MADE: nothing was launched.
#
# AN ISA-DERIVED PREDICTION WAS FALSIFIED HERE, and the lesson is recorded
# rather than hidden. `V_FLOOR_F32` and `V_CVT_I32_F32` are ALSO VOP1 and also
# absent from the VOPD set, so SPLIT=0 trades one non-pairable op for THREE. A
# second split (SPLIT=1, the 1.5*2^23 magic-number trick) was built to use ONLY
# pairable instructions -- and it is NOT better: at N=8192 non-causal it emits
# FEWER dual-issue multiplies (93 vs 100) and a longer loop (576 vs 561). Per-op
# ISA pairability is NECESSARY BUT NOT SUFFICIENT; LLVM's pairing pass is
# constrained by the dependency graph. `EEXP2_SPLIT_DEFAULT` was reverted from
# 1 (the ISA-derived guess) to 0 (measured).
#
# DEFAULT OFF. `EEXP2 = 0` keeps `tl.math.exp2` and the shipped binary. This is
# an ADDITIVE flag: it is proved additive in F037 by comparing the full
# instruction stream of EEXP2=0 against the pre-edit stream (a hash is not
# sufficient evidence, and a raw AMDGCN hash is not comparable across a source
# edit at all).
EEXP2_DEG_DEFAULT = 4
EEXP2_SPLIT_DEFAULT = 0


@triton.jit
def _exp2_emul(x, DEG: tl.constexpr, SPLIT: tl.constexpr):
    """2^x by IEEE-754 exponent split + a DEG-term fractional polynomial.

    SPLIT = 0  floor + float->int convert:
                   v_floor_f32, v_cvt_i32_f32, v_lshl_add_u32
               Fractional range [0, 1).

    SPLIT = 1  magic-number round-to-nearest (the default):
                   t = x + 1.5*2^23        (v_add_f32)
                   e = bitcast((bitcast_i32(t) - 0x4B3FFF81) << 23)
               `0x4B3FFF81 == 0x4B400000 - 127`, so the integer subtract yields
               `round(x) + 127`, the biased exponent field, with NO float->int
               convert and NO floor. Fractional range [-0.5, 0.5).

    WHY SPLIT=1 IS THE DEFAULT, AND IT IS AN ISA RESULT, NOT A TASTE.
    AMD's own machine-readable RDNA4 ISA lists the dual-issue (VOPD) encoding
    set as exactly 17 instructions. `V_EXP_F32`,
    `V_FLOOR_F32` and `V_CVT_I32_F32` are ALL ENC_VOP1 and NONE of them is in
    that set; `V_FMA_F32`/`V_MUL_F32`/`V_ADD_F32`/`V_SUB_F32`/`V_MAX_F32`/
    `V_LSHLREV_B32`/`V_ADD_NC_U32` are (as `V_DUAL_*`). So SPLIT=0 trades ONE
    non-pairable op for THREE, while SPLIT=1 is pairable end to end. Both are
    measured in F037; the instruction-count difference between them is small at
    the large tiles and large at the small ones.

    Coefficients are Chebyshev fits written as `1 + f*Horner(c, f)` so that
    p(0) == 1 exactly (exp2 of an exact integer is then exact, and `alpha` is
    exactly 1.0 on every iteration where the running max does not move).
    """
    xc = tl.maximum(x, -127.0)
    if SPLIT:
        t = xc + 12582912.0                      # 1.5 * 2^23
        e = ((t.to(tl.int32, bitcast=True) - 1262485377)
             << 23).to(tl.float32, bitcast=True)
        f = xc - (t - 12582912.0)                # [-0.5, 0.5), exact
        if DEG == 2:
            p = 1.0 + f * (0.7001061 + f * 0.2414312)
        elif DEG == 3:
            p = 1.0 + f * (0.6931472 + f * (0.2420353 + f * 0.05575465))
        elif DEG == 4:
            p = 1.0 + f * (0.6931368 + f * (0.2402253 + f * (0.05583828
                                                              + f * 0.009656711)))
        else:
            p = 1.0 + f * (0.6931472 + f * (0.2402235 + f * (0.05550381
                                                             + f * (0.009666368
                                                                    + f * 0.00133813))))
    else:
        n = tl.floor(xc)
        f = xc - n                               # [0, 1)
        ni = n.to(tl.int32)                      # n is integral: exact
        e = ((ni + 127) << 23).to(tl.float32, bitcast=True)
        if DEG == 2:
            p = 1.0 + f * (0.6844596 + f * 0.3060535)
        elif DEG == 3:
            p = 1.0 + f * (0.6935328 + f * (0.2334677 + f * 0.07258578))
        elif DEG == 4:
            p = 1.0 + f * (0.6931336 + f * (0.2406543 + f * (0.05342158
                                                              + f * 0.01277613)))
        else:
            p = 1.0 + f * (0.6931476 + f * (0.2402069 + f * (0.05565866
                                                             + f * (0.009196802
                                                                    + f * 0.001789665))))
    return e * p


@triton.jit
def _exp2f(x, EEXP2: tl.constexpr, EEXP2_DEG: tl.constexpr,
           EEXP2_SPLIT: tl.constexpr):
    """The single dispatch point for the exponential.

    With EEXP2 == 0 this is `tl.math.exp2` verbatim, so the shipped path is
    unchanged (and that is proved, not asserted).
    """
    if EEXP2:
        return _exp2_emul(x, EEXP2_DEG, EEXP2_SPLIT)
    else:
        return tl.math.exp2(x)


@triton.jit
def _kv_body(start_n, n_ok, MASKED: tl.constexpr, m_i, l_i, acc, pmax, l2u, l2r,
             q, qs, qk_scale, qt, offs_m, offs_n, offs_d, N_CTX,
             K8, V8, SK, SV, k_base, v_base, sk_base, sv_base,
             stride_kd, stride_kn, stride_vd, stride_vn,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             HEAD_DIM: tl.constexpr, KQT: tl.constexpr,
             PCAST: tl.constexpr, HWCVT: tl.constexpr, PSC: tl.constexpr,
             IS_CAUSAL: tl.constexpr, SPLIT_LOOP: tl.constexpr, EVEN_N: tl.constexpr,
             PAD_KV: tl.constexpr,
             LOAD_UNMASKED: tl.constexpr,
             GATE_MASK: tl.constexpr,
             full, kqt_full,
             P_SCALE: tl.constexpr, EEXP2: tl.constexpr,
             EEXP2_DEG: tl.constexpr, EEXP2_SPLIT: tl.constexpr,
             QK_INT8: tl.constexpr = 0, PSC_DYN: tl.constexpr = 0,
             PSC_FOLD: tl.constexpr = 0, SV_ONE: tl.constexpr = 0,
             LAZY: tl.constexpr = 0, LAZY_TAU: tl.constexpr = 0.0):
    """F060 TAIL_PEEL: the KV-loop body, extracted verbatim from the shipped
    loop so a peeled tail can reuse it.

    `MASKED` replaces the shipped shape-level `EVEN_N` test. The shipped code
    keyed the masked loads off `EVEN_N`, which is a SHAPE property -- so at
    N=8771 (`8771 % 16 = 3`) all 549 iterations carried the masked loads and the
    `tl.where` for a tail only the last iteration has. Keying off `MASKED`
    instead lets the full-block range compile to the unmasked body while the
    single tail iteration stays correct.

    The shipped `EVEN_N=True` path calls this with MASKED=False for every
    iteration, which is byte-equivalent to the original.

    `k`/`v` are deliberately NOT parameters: the body rebinds them to the
    loaded tiles, so a parameter of the same name would be shadowed. Only the
    scale tensors and pointers are passed.
    """
    start_n = tl.multiple_of(start_n, BLOCK_N)
    n_offs = start_n + offs_n
    n_ok = n_offs < N_CTX

    # K loaded already transposed: (HEAD_DIM, BLOCK_N) fp8
    k_ptrs = K8 + k_base + offs_d[:, None] * stride_kd + n_offs[None, :] * stride_kn
    # F031 KQT: the same K tile in its NATURAL orientation, (BLOCK_N,
    # HEAD_DIM), so it can be the A-operand of `tl.dot(kt, qt)` whose
    # contraction dim is HEAD_DIM. Same memory, same tile, one load -- only
    # the operand order of the dot changes. Loaded inside the `if KQT`
    # branch so the shipped path's two loads are untouched.
    if KQT:
        kt_ptrs = (K8 + k_base + n_offs[:, None] * stride_kn
                   + offs_d[None, :] * stride_kd)
    if not MASKED or LOAD_UNMASKED:
        k = tl.load(k_ptrs)
        if KQT:
            kt = tl.load(kt_ptrs)
        ks = tl.load(SK + sk_base + n_offs)
        if not SV_ONE:
            vs = tl.load(SV + sv_base + n_offs)
    else:
        k = tl.load(k_ptrs, mask=n_ok[None, :], other=0.0)
        if KQT:
            kt = tl.load(kt_ptrs, mask=n_ok[:, None], other=0.0)
        ks = tl.load(SK + sk_base + n_offs, mask=n_ok, other=1.0)
        if not SV_ONE:
            vs = tl.load(SV + sv_base + n_offs, mask=n_ok, other=0.0)

    # F031 KQT: `S = K @ Q^T` (shape (BLOCK_N, BLOCK_M)) instead of
    # `S = Q @ K^T` (shape (BLOCK_M, BLOCK_N)). Same matrix, transposed.
    #
    # F083: with QK_INT8 the operands are signed int8 and the product is exact in
    # int32 (`|q|,|k| <= 127`, HEAD_DIM = 128, so `127*127*128 = 2.06e6` is far
    # inside int32). The `.to(tl.float32)` is a SIGNED convert -- Triton emits
    # `arith.sitofp` for a signed int32 source, i.e. `v_cvt_f32_i32`; an unsigned
    # convert would map every negative score to ~4.3e9. Verified in the emitted
    # AMDGCN by the F083 check.
    #
    # The dequant scales `qs`/`ks` are fp32 and stay at exactly this site, and
    # for int8 they are `amax/127` (the prologue's job), so `qs*ks` is the same
    # kind of rank-1 outer product the fp8 path uses.
    if KQT:
        if QK_INT8:
            qk = tl.dot(kt, qt, out_dtype=tl.int32).to(tl.float32)
        else:
            qk = tl.dot(kt, qt, out_dtype=tl.float32)
    else:
        if QK_INT8:
            qk = tl.dot(q, k, out_dtype=tl.int32).to(tl.float32)
        else:
            qk = tl.dot(q, k, out_dtype=tl.float32)
    # The dequant scales are rank-1, so the transpose moves which axis the
    # broadcast runs along and nothing else.
    if KQT:
        qk = qk * (qs[None, :] * ks[:, None]) * qk_scale
    else:
        qk = qk * (qs[:, None] * ks[None, :]) * qk_scale

    if IS_CAUSAL:
        # F030 SPLIT_LOOP: the steady-state range carries no mask
        # construction at all. `start_n < full * BLOCK_N` is a uniform
        # branch on the loop induction variable, so the two bodies become
        # separate code regions and the full-block body is the non-causal
        # body. When SPLIT_LOOP is 0 the constexpr short-circuits and this
        # is byte-for-byte the original single `tl.where`.
        #
        # F031: in the transposed orientation the row index is the key and
        # the column index is the query, so the SAME predicate is written
        # with the two axes swapped. `kqt_full == full` numerically (see the
        # derivation above the loop).
        if KQT:
            if SPLIT_LOOP and start_n < kqt_full * BLOCK_N:
                pass
            else:
                qk = tl.where(n_offs[:, None] <= offs_m[None, :], qk,
                              float("-inf"))
        else:
            if SPLIT_LOOP and start_n < full * BLOCK_N:
                pass
            else:
                qk = tl.where(offs_m[:, None] >= n_offs[None, :], qk, float("-inf"))
    elif not EVEN_N:
        # -------------------------------------------------------------------
        # F060 TAIL FIX part 2: the SCORE mask, gated by a UNIFORM BRANCH on the
        # last block -- SPLIT_LOOP's pattern (F030), NOT a second body.
        #
        # `n_offs = start_n + offs_n`, so `n_ok` is ALL-TRUE exactly when
        # `start_n + BLOCK_N <= N_CTX`. On that side the `tl.where` is a no-op and
        # the branch deletes it. `start_n + BLOCK_N <= N_CTX` is a SCALAR uniform
        # across the block, so the two sides compile to separate code regions and
        # the steady state carries no mask construction -- no spills, one body.
        #
        # THE CLAMP (F062) IS WHAT MAKES THIS SOUND, and it is why this was not
        # possible before: `hi <= cdiv(N_CTX, BLOCK_N)` guarantees `start_n < N_CTX`
        # in every iteration, so the `tl.where` in the OTHER branch is reached at
        # most once, on the final block. Before the clamp the loop could run whole
        # blocks entirely past `N_CTX`, where `n_ok` is all-false and the mask WAS
        # load-bearing.
        #
        # MEASURED VALUE, and it is NOT about padding: this branch is worth
        # 16%. The F060 check sweeps N and finds a CLIFF at exactly N=8769 --
        #   8768: shipped 17.923 / gated 17.922  (identical; mask never needed)
        #   8769: shipped 19.514 / gated 15.070  (shipped pays the mask 549x)
        #   8784: shipped 18.044 / gated 18.049  (identical again)
        # Shipped skips the mask ZERO times, so at every ragged shape it builds and
        # applies `n_ok`/`tl.where` in all ~549 iterations. The 16% is the cost of
        # those 548 wasted masks, not of any load.
        #
        # F066 TRIED AND REVERTED: a compile-time `NEEDS_MASK` constexpr here
        # (False for the main loop, True only for the peeled tail, on the theory
        # that a host-constant would delete the branch and restore bit-identity).
        # IT MADE THINGS WORSE: at N=4096 non-causal the kernel went 237 regs /
        # 0 spills / 1169 insns -> **256 regs / 32 SPILLS / 1555 insns**, and it
        # still was NOT bit-identical. The constexpr did not delete the branch;
        # it changed the allocator's live ranges instead. Reverted -- see
        # F066.
        # -------------------------------------------------------------------
        if GATE_MASK and start_n + BLOCK_N <= N_CTX:
            pass
        else:
            if KQT:
                qk = tl.where(n_ok[:, None], qk, float("-inf"))
            else:
                qk = tl.where(n_ok[None, :], qk, float("-inf"))

    # The online-softmax state (m_i, l_i, alpha) is per QUERY ROW, i.e. per
    # column of the transposed score. Only the reduction axis changes.
    if PSC_FOLD:
        # F099: `qkmax` is kept as a named value because the clamp below needs it.
        if KQT:
            qkmax = tl.max(qk, 0)
        else:
            qkmax = tl.max(qk, 1)
        m_new = tl.maximum(m_i, qkmax)
        # PROVABLE NO-OVERFLOW CLAMP. `p_un <= 1` (m_new >= qkmax), so
        #     t = P_SCALE * p_un * sv <= P_SCALE * pumax * svmax,  pumax = exp2(qkmax - m_new),
        # and requiring `u <= 448 / (P_SCALE * pumax * svmax)` gives `ps = t*u <= 448` for every
        # element of THIS tile. `log2(pumax) = qkmax - m_new` exactly -- the scores are already in
        # log2 units -- so no extra `log2` is needed for it.
        # With `SV_ONE` there is no `vs` to reduce -- the whole tensor is 1.0
        # by construction, so `svmax` is the compile-time constant 1.0 (and the
        # `log2(max(svmax,1e-30))` term below is exactly 0).
        if SV_ONE:
            svmax = tl.max(tl.full([BLOCK_N], 1.0, tl.float32))
        else:
            svmax = tl.max(vs)
        l2u_bound = (tl.math.log2(448.0 / P_SCALE) - (qkmax - m_new)
                     - tl.math.log2(tl.maximum(svmax, 1e-30)))
        l2u_used = tl.minimum(l2u, l2u_bound)
        # `alpha` carries BOTH the softmax max ratio and the scale ratio `u_j/u_{j-1}`, which is
        # what makes the running (lagged) scale equivalent to a fixed one.
        alpha = _exp2f(m_i - m_new + (l2u_used - l2u) + l2r,
                       EEXP2, EEXP2_DEG, EEXP2_SPLIT)
    else:
        if KQT:
            m_tile = tl.maximum(m_i, tl.max(qk, 0))
        else:
            m_tile = tl.maximum(m_i, tl.max(qk, 1))
        if LAZY:
            # Lazy rescale.  `m_i` is the scale the accumulator is in (the max at the last
            # rescale), NOT the running max, so `p = exp2(qk - m_i)` can reach
            # `2**grow` with `grow = m_tile - m_i`.  The rescale is taken only when the
            # CTA-wide largest `grow` exceeds `LAZY_TAU` log2 units.  `alpha` is exactly
            # 1.0 on a skipped iteration, so the shipped `l_i = l_i * alpha + ...`
            # below stays correct without a second code path.
            # `tl.maximum(..., 0.0)` only sanitises the `-inf - -inf` NaN of a wholly
            # masked tile; `grow >= 0` holds by construction because `m_tile >= m_i`.
            alpha = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
            grow = tl.maximum(m_tile - m_i, 0.0)
            if tl.max(grow, 0) > LAZY_TAU:
                alpha = _exp2f(m_i - m_tile, EEXP2, EEXP2_DEG, EEXP2_SPLIT)
                if KQT == 2:
                    acc = acc * alpha[None, :]
                else:
                    acc = acc * alpha[:, None]
                m_i = m_tile
            m_new = m_i
        else:
            m_new = m_tile
            alpha = _exp2f(m_i - m_new, EEXP2, EEXP2_DEG, EEXP2_SPLIT)
    if PSC_FOLD:
        # F099: ONE broadcast subtract, exactly as many as the shipped path -- the per-row scale
        # (and, when PSC, `log2(P_SCALE)`) is folded into the per-row `shift` FIRST, so it costs
        # nothing on the (BLOCK_M, BLOCK_N) tile.
        if KQT == 3:
            qk = tl.trans(qk)
        shift = m_new - l2u_used
        if PSC:
            shift = shift - tl.math.log2(P_SCALE)
        if KQT == 2:
            p = _exp2f(qk - shift[None, :], EEXP2, EEXP2_DEG, EEXP2_SPLIT)
            l_i = l_i * alpha + tl.sum(p, 0)
        else:
            p = _exp2f(qk - shift[:, None], EEXP2, EEXP2_DEG, EEXP2_SPLIT)
            l_i = l_i * alpha + tl.sum(p, 1)
    elif KQT == 3:
        # Transpose the fp32 score back BEFORE the fold and the pack, so the
        # pack and the PV dot are the shipped ones and only the QK dot is
        # reordered. `alpha`/`m_new` were computed from the transposed score,
        # which is elementwise-identical to the shipped one, so the rescale
        # below is the shipped rescale.
        qk = tl.trans(qk)
        if PSC:
            p = _exp2f(qk - m_new[:, None] + tl.math.log2(P_SCALE),
                       EEXP2, EEXP2_DEG, EEXP2_SPLIT)
        else:
            p = _exp2f(qk - m_new[:, None], EEXP2, EEXP2_DEG, EEXP2_SPLIT)
        l_i = l_i * alpha + tl.sum(p, 1)
    elif KQT:
        if PSC:
            p = _exp2f(qk - m_new[None, :] + tl.math.log2(P_SCALE),
                       EEXP2, EEXP2_DEG, EEXP2_SPLIT)
        else:
            p = _exp2f(qk - m_new[None, :], EEXP2, EEXP2_DEG, EEXP2_SPLIT)
        l_i = l_i * alpha + tl.sum(p, 0)
    else:
        if PSC:
            p = _exp2f(qk - m_new[:, None] + tl.math.log2(P_SCALE),
                       EEXP2, EEXP2_DEG, EEXP2_SPLIT)
        else:
            p = _exp2f(qk - m_new[:, None], EEXP2, EEXP2_DEG, EEXP2_SPLIT)
        l_i = l_i * alpha + tl.sum(p, 1)

    v_ptrs = V8 + v_base + n_offs[:, None] * stride_vn + offs_d[None, :] * stride_vd
    if not MASKED or LOAD_UNMASKED:
        v = tl.load(v_ptrs)
    else:
        v = tl.load(v_ptrs, mask=n_ok[:, None], other=0.0)
    # F031 KQT=2: the SAME V tile in its (HEAD_DIM, BLOCK_N) orientation, so
    # it can be the B-operand of `tl.dot(p8, vt)` whose contraction dim is
    # BLOCK_N. This is the second half of the FlyDSL formulation, and it is
    # not optional: with P (BLOCK_N, BLOCK_M) the only legal second GEMM is
    # `(BN,BM) x (BN,HD) -> (BM,HD)` for the SHIPPED (BN, HD) V, which
    # contracts over the QUERY -- the wrong axis. Contracting over the KEY
    # requires V^T. `tl.trans` on a freshly loaded value is a layout
    # annotation, not a data movement, so this is one load, one tile.
    if KQT == 2:
        vt_ptrs = (V8 + v_base + offs_d[:, None] * stride_vd
                   + n_offs[None, :] * stride_vn)
        if not MASKED or LOAD_UNMASKED:
            vt = tl.load(vt_ptrs)
        else:
            vt = tl.load(vt_ptrs, mask=n_ok[None, :], other=0.0)

    # Fold V's per-token dequant scale into P: exact, and free.
    #
    # F031: `vs` is indexed by the KEY, so which axis of `p` it broadcasts
    # along follows `p`'s ORIENTATION, not the KQT flag:
    #   KQT=0  p is (BM, BN) -- key trailing -- so `vs[None, :]`
    #   KQT=2  p is (BN, BM) -- key leading  -- so `vs[:, None]`
    #   KQT=3  p is (BM, BN) again, because KQT=3 transposed the score BACK
    #          before the pack -- so `vs[None, :]`, i.e. the SHIPPED line.
    # Keying this off `KQT == 2` rather than `KQT` is therefore load-bearing:
    # using the transposed broadcast for KQT=3 is a hard compile error.
    #
    # `SV_ONE`: under `VSCALE_CHAN` the prologue writes `sv = 1.0` for
    # EVERY token, so the load above and this fold are dead work. `SV_ONE=1` deletes
    # both, and the fold collapses to the `P_SCALE` multiply the shipped path folds
    # into it. BIT-IDENTICAL, not an approximation: in IEEE fp32 `x * (1.0 * s)`
    # is `x * s` (and `1.0 * 448.0` is exact), and `x * 1.0` is `x` for every finite
    # `x` -- and for the padded columns of the masked tail `p` is exactly 0 (their
    # score is `-inf`), so `0 * P_SCALE` and `0 * (0.0 * P_SCALE)` agree too.
    if SV_ONE:
        if PSC:
            ps = p
        else:
            ps = p * P_SCALE
    elif KQT == 2:
        if PSC:
            ps = p * vs[:, None]
        else:
            ps = p * (vs[:, None] * P_SCALE)
    else:
        if PSC:
            ps = p * vs[None, :]
        else:
            ps = p * (vs[None, :] * P_SCALE)
    if PSC_FOLD:
        # F099: `ps` already carries the per-row scale `u` (it rode in the exponent), so the only
        # work left is to advance the state for the NEXT tile. `u` may only SHRINK -- exactly the
        # running reciprocal max `F098` tracks -- and `l2r` records the ratio just applied so the
        # next tile's `alpha` can carry it.
        if KQT == 2:
            mb = tl.max(ps, 0)
        else:
            mb = tl.max(ps, 1)
        l2r = -tl.math.log2(tl.maximum(mb, 1.0))
        l2u = l2u_used + l2r
    # F098: the DYNAMIC per-query-row P scale. `ps` above is the folded tensor
    # `t = p * sv * P_SCALE`; lift each QUERY ROW of it to e4m3's full range with
    # a RUNNING row max, and remember the ratio by which the accumulator must be
    # rescaled (`cscale`) so the fold stays EXACT. `pmax` is per query row, so the
    # `tl.max` runs over the KEY axis -- which follows `ps`'s ORIENTATION, exactly
    # as the `vs` broadcast above does.
    # `t * (1/pmax) <= 1` by construction, so the e4m3 cast cannot overflow.
    if PSC_DYN:
        p_old = pmax
        if KQT == 2:
            pmax = tl.maximum(p_old, tl.max(ps, 0))
            cscale = p_old / tl.maximum(pmax, 1e-30)
            ps = ps * (1.0 / tl.maximum(pmax, 1e-30))[None, :]
        else:
            pmax = tl.maximum(p_old, tl.max(ps, 1))
            cscale = p_old / tl.maximum(pmax, 1e-30)
            ps = ps * (1.0 / tl.maximum(pmax, 1e-30))[:, None]
    if HWCVT:
        # One hardware instruction per 2 elements, instead of ~27 bit ops.
        # The pack is elementwise in (row, col) of whatever orientation `ps`
        # is in, so it is reused unchanged; only its constexpr shape args
        # follow `ps`.
        #
        # The shape args must name the tensor's LEADING dimension first
        # and its TRAILING dimension second. The reshape inside the pack is
        # `(BLOCK_M, BLOCK_N // 2, 2)` for the shipped (BM, BN) P; for the
        # (BN, BM) P of KQT it must be `(BLOCK_N, BLOCK_M // 2, 2)`. Writing
        # `BLOCK_N // 2` there instead of `BLOCK_M // 2` makes the reshape
        # ambiguous whenever BLOCK_M != BLOCK_N and Triton folds the
        # transpose away, producing a (BM, BN) result. `KQT1`/`KQT2` below
        # are therefore NOT interchangeable.
        if PCAST == 0:
            p8 = ps.to(FP8)
        elif PCAST == 1:
            p8 = (_cvt_pk_fp8_2d(ps, BLOCK_N, BLOCK_M) if KQT == 2
                  else _cvt_pk_fp8_2d(ps, BLOCK_M, BLOCK_N))
        elif PCAST == 2:
            p8 = (_cvt_pk_fp8_2d_v2(ps, BLOCK_N, BLOCK_M) if KQT == 2
                  else _cvt_pk_fp8_2d_v2(ps, BLOCK_M, BLOCK_N))
        elif PCAST == 3:
            # Measurement-only: wrong result, halved P, and the pair-summed
            # V it needs costs more than it measures. Disabled -- routed to
            # the incumbent so the flag cannot produce a bogus number.
            p8 = (_cvt_pk_fp8_2d(ps, BLOCK_N, BLOCK_M) if KQT == 2
                  else _cvt_pk_fp8_2d(ps, BLOCK_M, BLOCK_N))
        else:
            p8 = (_cvt_pk_fp8_2d_swap(ps, BLOCK_N, BLOCK_M) if KQT == 2
                  else _cvt_pk_fp8_2d_swap(ps, BLOCK_M, BLOCK_N))
    else:
        p8 = ps.to(FP8)

    # F031: KQT=3 transposed the fp32 SCORE back to (BM, BN) above, so P and
    # the PV dot are the shipped ones and only the QK dot is reordered.
    # KQT=2 does not: P stays (BN, BM).
    #
    # WHY THE SECOND DOT IS `tl.dot(vt, p8)` AND NOT `tl.dot(p8, vt)`.
    # `tl.dot(a, b)` contracts over `a`'s LAST axis, and P's last axis is the
    # QUERY under KQT. The PV GEMM must contract over the KEY. So the only
    # legal pairing with a (BN, BM) P is
    #       tl.dot(vt, p8) : (HD, BN) x (BN, BM) -> (HD, BM)
    # i.e. V supplied TRANSPOSED and the output accumulated TRANSPOSED.
    # `tl.dot(p8, vt)` with vt (HD, BN) contracts over BN but produces
    # (BM, HD) from operands whose reduction dims are BM and HD -- a hard
    # "input and other must have equal reduction dimensions" error, not a
    # silent wrong answer. This was verified in the F031 check, a
    # standalone 12-case sweep, before it was written here.
    #
    # THE BROADCAST AXIS DIFFERS BETWEEN THE MODES. In the shipped
    # orientation `acc` is (BM, HD) and the per-query vector broadcasts down
    # the ROWS (`alpha[:, None]`). For KQT=2 `acc` is (HD, BM), so it
    # broadcasts along the COLUMNS (`alpha[None, :]`).
    if KQT == 2:
        if LAZY == 0:
            if PSC_DYN:
                acc = acc * (alpha * cscale)[None, :]
            else:
                acc = acc * alpha[None, :]
        acc = tl.dot(vt, p8, acc)
    else:
        if LAZY == 0:
            if PSC_DYN:
                acc = acc * (alpha * cscale)[:, None]
            else:
                acc = acc * alpha[:, None]
        acc = tl.dot(p8, v, acc)
    m_i = m_new
    return m_i, l_i, acc, pmax, l2u, l2r


@triton.jit
def _attn_fwd_fp8(
    Q, Q8, K8, V8, SQ, SK, SV, Out, Lse,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    sq_z, sq_h, sk_z, sk_h, sv_z, sv_h,
    Z, H, N_CTX, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    IS_CAUSAL: tl.constexpr, EVEN_M: tl.constexpr, EVEN_N: tl.constexpr,
    P_SCALE: tl.constexpr, STORE_LSE: tl.constexpr, HWCVT: tl.constexpr,
    FUSE_Q: tl.constexpr, PCAST: tl.constexpr = 1, FDIV: tl.constexpr = 1,
    PSC: tl.constexpr = 0, SPLIT_LOOP: tl.constexpr = 0, KQT: tl.constexpr = 0,
    EEXP2: tl.constexpr = 0, EEXP2_DEG: tl.constexpr = EEXP2_DEG_DEFAULT,
    EEXP2_SPLIT: tl.constexpr = EEXP2_SPLIT_DEFAULT,
    TAIL_PEEL: tl.constexpr = 0, PAD_KV: tl.constexpr = 0,
    LOAD_UNMASKED: tl.constexpr = 0,
    GATE_MASK: tl.constexpr = 0,
    QK_INT8: tl.constexpr = 0,
    # F098: the dynamic per-query-row P scale. Default 0 == no code emitted.
    PSC_DYN: tl.constexpr = 0,
    # F099: the same per-row scale, folded into the softmax row max. Default 0 == no code emitted.
    PSC_FOLD: tl.constexpr = 0,
    # F086: `VM` is V's per-(b,h,channel) sequence-axis mean, shape `(B*H, D)`
    # fp32, and `ADD_VMEAN=1` adds it back to the output once.
    #
    # `VM` is declared LAST, after every constexpr, because every parameter
    # before it that has no default must come first in Python -- and it carries a
    # `None` default so that the many existing direct callers of this kernel
    # (the F017, F018 and F026 checks, ...) keep working unchanged.
    # It is a NON-constexpr parameter, so `None` is only ever legal together
    # with `ADD_VMEAN=0`; the launch in `flash_attn_fp8` always passes a real
    # pointer (the V scale tensor when the add-back is off) precisely so that no
    # arm can dereference a `None`.
    VM=None,
    ADD_VMEAN: tl.constexpr = 0,
    # F100: `VCH` is V's per-(b,h,channel) sequence-axis absmax divisor
    # (`(B*H, D)` fp32, from `kseq_amax_fp8`). `MUL_VCH=1` re-applies it to the
    # output once, undoing the per-channel quantisation of V done by the prologue.
    # Same contract as `VM`: declared last, `None` only legal with `MUL_VCH=0`,
    # and the launch always passes a real pointer so no arm can dereference `None`.
    VCH=None,
    MUL_VCH: tl.constexpr = 0,
    # `SV_ONE=1` asserts that the `sv` tensor this call reads is identically
    # 1.0 -- which the `VSCALE_CHAN` prologue guarantees (`quant_kv_rows_*` /
    # `quant_kv_chan_fused_*` store `sv = 1.0` for every token) -- and deletes the
    # per-K-tile `sv` load and the `* sv` fold from `_kv_body`. Default 0 == no code
    # emitted, so every arm that predates `SV_ONE` is untouched.
    SV_ONE: tl.constexpr = 0,
    # The LAZY softmax rescale (see the module-level flag). `LAZY=0` emits
    # no code at all; `LAZY_TAU` is the log2 headroom.
    LAZY: tl.constexpr = 0,
    LAZY_TAU: tl.constexpr = 0.0,
):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = (off_hz // H).to(tl.int64)
    off_h = (off_hz % H).to(tl.int64)

    q_base = off_z * stride_qz + off_h * stride_qh
    k_base = off_z * stride_kz + off_h * stride_kh
    v_base = off_z * stride_vz + off_h * stride_vh
    o_base = off_z * stride_oz + off_h * stride_oh
    sq_base = off_z * sq_z + off_h * sq_h
    sk_base = off_z * sk_z + off_h * sk_h
    sv_base = off_z * sv_z + off_h * sv_h

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    m_ok = offs_m < N_CTX
    if FUSE_Q:
        # Quantize Q inside the attention kernel. Q is read exactly ONCE per
        # program, so a separate prologue pass over it is pure overhead -- this
        # is the incumbent's design (PR #368 has no Q prologue at all) and
        # it removes one of our three prologue passes.
        #
        # The fp16 and fp8 tensors are both contiguous (B,H,N,D), so the stride
        # arguments are interchangeable; flash_attn_fp8 asserts contiguity.
        q16p = Q + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        if EVEN_M:
            qf = tl.load(q16p).to(tl.float32)
        else:
            qf = tl.load(q16p, mask=m_ok[:, None], other=0.0).to(tl.float32)
        # Same flooring as the prologue (upstream issue #164: an all-equal row
        # gives a zero scale and NaNs the whole output).
        #
        # F083: `/127` for int8 QK, `/448` for fp8. This is the ONLY place the
        # Q scale convention is chosen, and it is a `tl.constexpr` branch, so the
        # fp8 arm emits exactly the instructions it emitted before F083 (proved
        # by the F083 check).
        if QK_INT8:
            qs = tl.maximum(tl.max(tl.abs(qf), 1) / 127.0, 1e-12)
        else:
            qs = tl.maximum(tl.max(tl.abs(qf), 1) / 448.0, 1e-12)
        # Dividing by amax/448 puts the max at exactly 448, so the hardware
        # convert's internal clamp is a no-op here. Same for amax/127 -> 127.
        #
        # F021: the divide here is a BROADCAST divide over (BLOCK_M, HEAD_DIM) --
        # the divisor is constant along head_dim, so the IEEE sequence runs 128
        # times per program when 16 would do. Hoisting the per-row reciprocal
        # and multiplying replaces 128 five-instruction divides with 16 divides
        # plus 128 multiplies.
        #
        # `_cvt_pk_fp8_2d` is an inline-asm `v_cvt_pk_fp8_f32` and CANNOT make
        # int8; the int8 arm uses `_cvt2d_i8` (`v_cvt_pk_i16_f32` + low byte),
        # which is the same helper the int8 K prologue uses.
        if QK_INT8:
            if FDIV:
                q = _cvt2d_i8(qf * (1.0 / qs)[:, None], BLOCK_M, HEAD_DIM)
            else:
                q = _cvt2d_i8(qf / qs[:, None], BLOCK_M, HEAD_DIM)
        else:
            if FDIV:
                q = _cvt_pk_fp8_2d(qf * (1.0 / qs)[:, None], BLOCK_M, HEAD_DIM)
            else:
                q = _cvt_pk_fp8_2d(qf / qs[:, None], BLOCK_M, HEAD_DIM)
    else:
        q_ptrs = Q8 + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
        if EVEN_M:
            q = tl.load(q_ptrs)
            qs = tl.load(SQ + sq_base + offs_m)
        else:
            q = tl.load(q_ptrs, mask=m_ok[:, None], other=0.0)
            qs = tl.load(SQ + sq_base + offs_m, mask=m_ok, other=1.0)

    # F031: the online-softmax running state is per QUERY ROW, so it is always
    # length BLOCK_M -- the query is the accumulator's COLUMN axis for KQT=2
    # (the PV dot is `(BN,BM) x (BN,HD) -> (BM,HD)`, so its ROW axis is the key)
    # and its ROW axis in the shipped orientation. The reduction that produces
    # it therefore runs over the OTHER axis of the score: `tl.max(qk, 1)` on the
    # (BM, BN) score and `tl.max(qk, 0)` on the (BN, BM) one. Both give (BM,).
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    # F098: the running per-query-row max of the FOLDED P tensor `p * sv * P_SCALE`.
    # Dead when PSC_DYN=0 (never read and never written).
    pmax = tl.zeros([BLOCK_M], dtype=tl.float32)
    # F099: `l2u = log2(u)` is the per-row P scale to use at the NEXT tile and `l2r` is the
    # log-ratio applied at the previous tile (so `alpha` can carry `u_j/u_{j-1}`). Both are dead
    # when PSC_FOLD=0. `l2u` starts at 0 (u=1): the FIRST tile has no earlier tile to take a max
    # from, so it is quantised unscaled -- the lag that makes this arm inaccurate (see `PSC_FOLD`).
    l2u = tl.zeros([BLOCK_M], dtype=tl.float32)
    l2r = tl.zeros([BLOCK_M], dtype=tl.float32)

    # -------------------------------------------------------------------
    # F031: KQT -- the `S = K @ Q^T` operand-order swap (from AOTriton
    # 0.14's FlyDSL gfx1201 kernel).
    #
    # The idea: computing `K @ Q^T` instead of `Q @ K^T` makes the score matrix
    # land directly in the orientation the SECOND GEMM wants, avoiding a
    # transpose of P. `S = Q @ K^T` and `S = (K @ Q^T)^T` are the same matrix.
    #
    # IT IS NOT TRUE THAT NO TRANSPOSE IS NEEDED, and the reason is worth
    # recording because it is the whole result of this experiment. `tl.dot(a, b)`
    # contracts over `a`'s LAST axis. P's last axis under the swap is the QUERY;
    # the PV GEMM must contract over the KEY. So the swapped score can never be
    # the A-operand of `P @ V`. The only legal second GEMM that contracts over
    # the key AND uses the swapped P is
    #         tl.dot(vt, p8) : (HD, BN) x (BN, BM) -> (HD, BM)
    # -- V supplied TRANSPOSED, output accumulated TRANSPOSED. The transpose is
    # not avoided; it MOVES from P (per KV iteration) to O (once, in the
    # epilogue). That move is the entire content of the lever, and it is exactly
    # what AOTriton's FlyDSL kernel does.
    #
    #   KQT = 0  shipped. S is (BM, BN); tl.dot(q, k); acc (BM, HD).
    #   KQT = 1  RETIRED. Transposes the fp8 P back per iteration. `tl.trans` on
    #            a value produced by the inline-asm fp8 pack is FOLDED AWAY by
    #            Triton 3.7.1 (see the F031 check), so the dot gets a
    #            (BM, BN) operand where it wants (BN, ...) -- a hard
    #            reduction-dimension error at every non-square tile.
    #   KQT = 2  S is (BN, BM); P is NEVER transposed; V is loaded (HD, BN) as
    #            `vt`; the PV dot is `tl.dot(vt, p8)` giving `acc` (HD, BM); ONE
    #            `tl.trans(acc)` lands in the epilogue. This is the FlyDSL
    #            formulation and the real subject of F031.
    #   KQT = 3  S is (BN, BM); the fp32 score is transposed back per iteration
    #            BEFORE the pack, so the pack and the PV dot are the shipped ones
    #            and only the QK dot is reordered. Isolates "does the QK operand
    #            order itself help" from "does moving the transpose help".
    #
    # `qt` is Q transposed: the SAME tile, in the (HEAD_DIM, BLOCK_M)
    # orientation, so it is the B-operand of `tl.dot(kt, qt)` whose contraction
    # dim is HEAD_DIM. `tl.trans` on a value that was just LOADED is a layout
    # annotation, not a data movement -- which is why it is free here and why it
    # is NOT free on a computed value (that asymmetry is the finding).
    #
    # Nothing in this block executes when KQT == 0: every use is behind a
    # `KQT == n` constexpr test, so the shipped path is byte-identical (proven in
    # the F031 check, not asserted).
    # -------------------------------------------------------------------
    # F060 TAIL_PEEL: qt is bound UNCONDITIONALLY so the peeled loop can pass
    # it to _kv_body regardless of KQT. It was previously defined only under
    # if KQT, which is fine for the shipped in-line loop (every use sits behind
    # a KQT == n constexpr test) but not for a call argument. When KQT == 0 the
    # helper never reads it, so binding it to q costs nothing and is dead.
    qt = tl.trans(q) if KQT else q
    # F031: the accumulator's SHAPE is (BLOCK_M, HEAD_DIM) in every mode -- that
    # is fixed by `tl.dot`'s `M x N` result, and the mma layout behind it is
    # uninfluenceable. What KQT=2 changes is that the
    # SECOND dot is `tl.dot(vt, p8)` = `(HD,BN) x (BN,BM) -> (HD,BM)`, so the
    # accumulator holds O TRANSPOSED during the loop and is transposed once, in
    # the epilogue. See the block comment below.
    if KQT == 2:
        acc = tl.zeros([HEAD_DIM, BLOCK_M], dtype=tl.float32)
    else:
        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = sm_scale * LOG2E

    # F023: PSC -- fold P_SCALE into the exp2 argument.
    #   exp2(x - log2(P_SCALE)) == P_SCALE * exp2(x)
    # so the per-element `* P_SCALE` in `ps = p * (vs * P_SCALE)` becomes a
    # constant the exponent's FMA already carries. The shift is applied to the
    # exponent itself, so the matching normaliser becomes `l_i` instead of
    # `l_i * P_SCALE`.
    #
    # NOTHING is computed here when PSC is off, and that is load-bearing. Both
    # `pscale_shift = ... else 0.0` and `tl.math.log2(P_SCALE) if PSC else 0.0`
    # make the DEFAULT path emit a real `arith.addf %p, %cst` (Triton does not
    # fold an add of a splat zero), which is 21 extra AMDGCN instructions and a
    # different default binary. Measured with the F023 check: the default
    # path went 2003 -> 2034 GCN lines and hash 32412b4f4de75e56 ->
    # 0b09eb2ade4555ad. Selecting the whole EXPRESSION keeps PSC=0 a true no-op.
    # `n_pad` is defined below (F060 PAD_KV) and is `N_CTX` when PAD_KV=0, so
    # this is the shipped bound by default. With PAD_KV the loop must cover the
    # PADDED block count -- the padded columns are then removed by the SCORE mask
    # (`n_ok` keys off the true N_CTX), not by the loop bound.
    # -------------------------------------------------------------------
    # F060 TAIL FIX (PAD_KV): with the F062 clamp below, `hi` never exceeds
    # `cdiv(N_CTX, BLOCK_N)`, so the loop can never address a key block beyond
    # the LAST block that contains a valid token. The only keys it can touch that
    # are outside `[0, N_CTX)` are the tail columns of that final block -- and
    # `n_ok` already marks those.
    #
    # So if K, V, `ks` and `vs` are PADDED to a multiple of BLOCK_N (the wrapper
    # does it), every load is in-bounds in EVERY iteration and the masked-load
    # path becomes unnecessary. That removes the shape-level `EVEN_N` mask
    # that F060 measured at +8.45% -- WITHOUT a
    # second loop body, so no extra spills (F060: the two-body peel cost
    # 0->24/0->56 spills and lost 13-23%).
    #
    # The SCORE mask stays (`elif not EVEN_N:` below). It is required for
    # correctness: `ks` is padded with 1.0 and K with 0, so `qk = 0` for a
    # padded column, which is NOT `-inf` and would be included in the softmax.
    # -------------------------------------------------------------------
    n_pad = (tl.cdiv(N_CTX, BLOCK_N) * BLOCK_N) if PAD_KV else N_CTX

    if IS_CAUSAL:
        # F062 FIX. `hi` was NOT clamped to the number of key blocks that
        # actually exist. When `N % BLOCK_M != 0` the last query block spans
        # `[start_m*BM, (start_m+1)*BM)` with `(start_m+1)*BM > N`, so
        # `cdiv((start_m+1)*BM, BN)` exceeds `cdiv(N, BN)` and the loop iterates
        # KEY BLOCKS PAST THE END of K/V/ks/vs. At `N % BLOCK_N == 0` (`EVEN_N`
        # True) those loads are UNMASKED, so they read past the buffers.
        #
        # fp8 E4M3FN has NaN ENCODINGS (0x7F / 0xFF). If the bytes past the
        # buffer are such an encoding, `qk` is NaN for those columns, the row max
        # is NaN, and NaN propagates to every output element of a VALID row --
        # including rows that never touch the padding. That matches the measured
        # symptom: a NaN count that VARIES call-to-call (F062: [176, 2352, 2352,
        # 2352, 2352]) because it depends on the memory that happens to follow.
        # And it is NOT always NaN: at N=64/BN=32 the frozen kernel returned
        # NO NaN and was 1500x wrong (rms 5.54 vs 3.70e-03) -- silent corruption.
        #
        # The clamp is exact for the rows that matter: a key block at
        # `start_n >= N` can only affect query rows `m >= start_n >= N`, and those
        # rows are outside `[0, N)` -- their `l_i` is 0 and their output is
        # written only under `m_ok`. Dropping those iterations therefore does not
        # change any in-range output. VERIFIED: bit-identical to the frozen kernel
        # at all 35 cells where the frozen kernel is valid, and 0 NaN in all 40
        # (the F062 check).
        #
        # Falsifier (user-supplied, and it HOLDS): every NaN cell must be
        # causal with `N % BLOCK_N == 0` AND `N % BLOCK_M != 0`. Verified on all
        # 4. It is NECESSARY BUT NOT SUFFICIENT -- N=64 with BN=16 and BN=32
        # satisfy it and do NOT NaN, so whether the bug fires also depends on what
        # the out-of-bounds read happens to land on.
        # The bisect (the F062 check) shows the NaN
        # SURVIVES `split_loop=False`, so the defect is this BOUND, not the F030
        # `full`/`hi` split.
        hi = tl.minimum(tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N),
                        tl.cdiv(n_pad, BLOCK_N))
    else:
        hi = tl.cdiv(n_pad, BLOCK_N)

    # -------------------------------------------------------------------
    # F060 TAIL FIX (PAD_KV): with the F062 clamp above, `hi` never exceeds
    # `cdiv(N_CTX, BLOCK_N)`, so the loop can never address a key block beyond
    # the LAST block that contains a valid token. The only keys it can touch that
    # are outside `[0, N_CTX)` are the tail columns of that final block -- and
    # `n_ok` already marks those.
    #
    # So if K, V, `ks` and `vs` are PADDED to a multiple of BLOCK_N (the wrapper
    # does it), every load is in-bounds in EVERY iteration and the masked-load
    # path becomes unnecessary. That removes the shape-level `EVEN_N` mask
    # that F060 measured at +8.45% -- WITHOUT a
    # second loop body, so no extra spills (F060: the two-body peel cost
    # 0->24/0->56 spills and lost 13-23%).
    #
    # The SCORE mask stays (`elif not EVEN_N:` below). It is required for
    # correctness: `ks` is padded with 1.0 and K with 0, so `qk = 0` for a
    # padded column, which is NOT `-inf` and would be included in the softmax.
    # -------------------------------------------------------------------

    # -------------------------------------------------------------------
    # F030: SPLIT_LOOP -- the two-range causal loop, ported from
    # `kernels/fa_fp8_pv.py` (F025/F029) into the SHIPPED kernel.
    #
    # The causal KV loop is split into a full-block range [0, full) that
    # carries NO mask construction at all, and a boundary range [full, hi)
    # that carries the causal `tl.where`. The full range is the steady state
    # and is identical to the non-causal loop body.
    #
    # `full` = the number of BLOCK_N blocks that are unmasked for EVERY row of
    # this BLOCK_M tile. Block b covers columns [b*BN, (b+1)*BN); it is
    # unmasked for all rows iff its largest column is below the tile's first
    # row, i.e. (b+1)*BN - 1 < start_m*BM, i.e. b < (start_m*BM - 1)/BN + 1.
    # Since start_m*BM is a multiple of BN whenever BM is (BM in {64,128},
    # BN in {16,32,64} all satisfy this), that count is exactly
    # start_m*BM // BN.
    #
    # THIS FORMULA IS EXACT AND FROZEN. Do NOT "simplify" it to
    # tl.cdiv((start_m+1)*BM, BN) -- that expression is `hi` above, the number
    # of blocks TOUCHING the diagonal. Using it drops the mask from blocks
    # that need it and silently corrupts the output.
    #
    # `full` is 0 when the lever is off or the call is non-causal, so the
    # `start_n < full * BLOCK_N` test is False everywhere and the masking
    # branch below is the only one reachable. Non-causally the whole block is
    # dead code and the binary is unchanged (F029).
    # -------------------------------------------------------------------
    if SPLIT_LOOP and IS_CAUSAL:
        full = start_m * BLOCK_M // BLOCK_N
    else:
        full = 0

    # F031: the SPLIT_LOOP boundary in the TRANSPOSED orientation.
    #
    # With `S_t = K @ Q^T`, S_t's ROW index is the key (`n_offs`) and its COLUMN
    # index is the query (`offs_m`). The causal mask `offs_m >= n_offs` therefore
    # masks COLUMNS, not rows, and the unmasked part of a key block is its
    # TRAILING columns -- the opposite of the shipped orientation.
    #
    # Key block b covers keys [b*BN, (b+1)*BN). A column j (query start_m*BM + j)
    # is unmasked for every key in the block iff the largest key in the block is
    # <= the smallest query in the block:
    #       (b+1)*BN - 1 <= start_m*BM
    #   <=> b*BN <= start_m*BM + BN - 1 - BN = start_m*BM - 1
    #   <=> b*BN < start_m*BM
    #   <=> b < start_m*BM / BN
    # i.e. exactly `full = start_m * BLOCK_M // BLOCK_N` again -- the same block
    # count as the shipped orientation. What CHANGES is the per-column cut inside
    # a partially-masked block: with key block start_n, a column j is fully
    # unmasked iff its query row `start_m*BM + j >= start_n + BN - 1`, i.e.
    #       j >= (start_n - start_m*BM + BN - 1) // BN
    # The mask is still applied to the whole tile (a uniform branch is not worth
    # the extra complexity here); the cut only matters for a per-column split,
    # which is NOT implemented. What IS used below is the uniform block test,
    # which is `start_n < full * BLOCK_N` -- identical to the shipped one.
    #
    # So `kqt_full` is numerically the same as `full`; it exists as a separate
    # name only so the transposed branch can be read on its own terms and so a
    # future per-column cut has a place to live.
    if KQT and SPLIT_LOOP and IS_CAUSAL:
        kqt_full = start_m * BLOCK_M // BLOCK_N
    else:
        kqt_full = 0

    # -------------------------------------------------------------------
    # F060 TAIL_PEEL: peel the ragged tail so the full-block range runs the
    # UNMASKED body.
    #
    # Measured cost of NOT doing this (F060): +8.45% at N=8771, against a 0.01%
    # bracket. `EVEN_N` is a SHAPE property, so at N=8771 every one of the 549
    # iterations built the masked loads and the
    # `tl.where` for a tail only the LAST iteration has.
    #
    # `hi_full` = the number of blocks that are entirely inside [0, N_CTX). The
    # main loop covers exactly those with MASKED=False; the single remainder
    # block (if any) is handled after the loop with MASKED=True. When N is a
    # multiple of BLOCK_N there is no remainder and the code is the shipped one.
    #
    # The causal `tl.where` is NOT affected: it is driven by `full`/
    # `kqt_full`, which are unchanged, so the causal diagonal stays exact.
    # -------------------------------------------------------------------
    # F066: `NEEDS_MASK` was tried here and REVERTED -- see the note at the
    # `elif not EVEN_N:` branch. It cost 32 spills and did not restore
    # bit-identity.
    hi_full = min(N_CTX // BLOCK_N, hi) if TAIL_PEEL else hi
    n_ok_full = offs_n < BLOCK_N          # all-true: the full range is in bounds
    # `MASKED` here means "the LOADS must be masked". On the DEFAULT path
    # (TAIL_PEEL=0, PAD_KV=0) it must stay `not EVEN_N` -- the shipped behaviour.
    # The first version passed a bare `False`, which silenced the masked loads AND
    # the non-causal `tl.where` (`elif not EVEN_N:`) for every iteration, so
    # `tail_peel=0` at N=8771 produced WRONG OUTPUT (loop 670 -> 552, regs
    # 232 -> 235). Caught by the f060e census, not by the gate -- the gate only
    # compared the two arms against each other.
    # With PAD_KV the buffers are padded to BLOCK_N, so no load needs a mask.
    _masked_full = (False if (TAIL_PEEL or PAD_KV) else (not EVEN_N))
    for start_n in range(0, hi_full * BLOCK_N, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        m_i, l_i, acc, pmax, l2u, l2r = _kv_body(
            start_n, n_ok_full, MASKED=_masked_full,
            m_i=m_i, l_i=l_i, acc=acc, pmax=pmax, l2u=l2u, l2r=l2r,
            q=q, qs=qs, qk_scale=qk_scale, qt=qt, offs_m=offs_m, offs_n=offs_n,
            offs_d=offs_d, N_CTX=N_CTX, K8=K8, V8=V8, SK=SK, SV=SV, k_base=k_base,
            v_base=v_base, sk_base=sk_base, sv_base=sv_base,
            stride_kd=stride_kd, stride_kn=stride_kn, stride_vd=stride_vd,
            stride_vn=stride_vn, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM, KQT=KQT, PCAST=PCAST, HWCVT=HWCVT, PSC=PSC,
            IS_CAUSAL=IS_CAUSAL, SPLIT_LOOP=SPLIT_LOOP, EVEN_N=EVEN_N, PAD_KV=PAD_KV, LOAD_UNMASKED=LOAD_UNMASKED, GATE_MASK=GATE_MASK, full=full,
            kqt_full=kqt_full, P_SCALE=P_SCALE, EEXP2=EEXP2,
            EEXP2_DEG=EEXP2_DEG, EEXP2_SPLIT=EEXP2_SPLIT, QK_INT8=QK_INT8,
            PSC_DYN=PSC_DYN, PSC_FOLD=PSC_FOLD, SV_ONE=SV_ONE,
            LAZY=LAZY, LAZY_TAU=LAZY_TAU)

    if TAIL_PEEL and (N_CTX % BLOCK_N) != 0:
        # Exactly one remainder block, at start_n = hi_full * BLOCK_N.
        #
        # `_sn`/`_n_ok` are built INLINE in the call, not as named locals. The
        # first version named them above the call and the kernel went from 0
        # spills to 24 (non-causal) / 56 (causal) at 232 -> 256 VGPR, with LDS
        # 16384 -> 18432. The tail is a whole extra copy of the body; every value
        # it keeps alive extends across the main loop's live range and pushes the
        # allocator over the ceiling. Shortening those live ranges is the whole
        # game here -- see F060 section 5.
        m_i, l_i, acc, pmax, l2u, l2r = _kv_body(
            hi_full * BLOCK_N, (hi_full * BLOCK_N + offs_n) < N_CTX,
            MASKED=True, m_i=m_i, l_i=l_i, acc=acc, pmax=pmax, l2u=l2u, l2r=l2r,
            q=q, qs=qs, qk_scale=qk_scale, qt=qt, offs_m=offs_m, offs_n=offs_n,
            offs_d=offs_d, N_CTX=N_CTX, K8=K8, V8=V8, SK=SK, SV=SV, k_base=k_base,
            v_base=v_base, sk_base=sk_base, sv_base=sv_base,
            stride_kd=stride_kd, stride_kn=stride_kn, stride_vd=stride_vd,
            stride_vn=stride_vn, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
            HEAD_DIM=HEAD_DIM, KQT=KQT, PCAST=PCAST, HWCVT=HWCVT, PSC=PSC,
            IS_CAUSAL=IS_CAUSAL, SPLIT_LOOP=SPLIT_LOOP, EVEN_N=EVEN_N, PAD_KV=PAD_KV, LOAD_UNMASKED=LOAD_UNMASKED, GATE_MASK=GATE_MASK, full=full,
            kqt_full=kqt_full, P_SCALE=P_SCALE, EEXP2=EEXP2,
            EEXP2_DEG=EEXP2_DEG, EEXP2_SPLIT=EEXP2_SPLIT, QK_INT8=QK_INT8,
            PSC_DYN=PSC_DYN, PSC_FOLD=PSC_FOLD, SV_ONE=SV_ONE,
            LAZY=LAZY, LAZY_TAU=LAZY_TAU)

    # F031 KQT=2: the accumulator has been O^T = (HEAD_DIM, BLOCK_M) for the
    # whole loop. One transpose lands here, in the epilogue, instead of one
    # `tl.trans` per KV iteration on P. Both the transpose and the per-query
    # normaliser then broadcast along `acc`'s COLUMN axis, which is the query.
    if KQT == 2:
        acc = tl.trans(acc)
        if PSC:
            if FDIV:
                acc = acc * (1.0 / l_i)[:, None]
            else:
                acc = acc / l_i[:, None]
        elif FDIV:
            acc = acc * (1.0 / (l_i * P_SCALE))[:, None]
        else:
            acc = acc / (l_i * P_SCALE)[:, None]
    elif PSC:
        # The P_SCALE shift is already in the exponent, so the normaliser is l_i.
        if FDIV:
            acc = acc * (1.0 / l_i)[:, None]
        else:
            acc = acc / l_i[:, None]
    elif FDIV:
        acc = acc * (1.0 / (l_i * P_SCALE))[:, None]
    else:
        acc = acc / (l_i * P_SCALE)[:, None]

    # F098: UNDO the dynamic per-query-row P scale. Inside the loop the accumulator
    # is held as `A / pmax_final` (the kernel rescales it by `r_old/r_new` whenever
    # the running max grows), so the exact inverse is one multiply by the FINAL
    # running max. `pmax` is (BLOCK_M,) and `acc` is (BLOCK_M, HEAD_DIM) in every
    # arm by this point (KQT=2's `tl.trans` has already landed above), so the
    # broadcast axis is unambiguous.
    # It MUST come after the `1/(l_i * P_SCALE)` rescale and before the F086
    # add-back, because `pmax` is part of the P normalisation, not part of V.
    if PSC_DYN:
        acc = acc * pmax[:, None]

    # -------------------------------------------------------------------
    # F100: UNDO V's per-CHANNEL quantisation scale.
    #
    # With `VSCALE_CHAN=1` the prologue divides V by `b[d]` (the sequence-axis
    # absmax of the centred V) and writes `sv = 1.0`, so the fold into P contributes
    # nothing and P keeps e4m3's full `P_SCALE` grid. `b[d]` is constant along the PV
    # REDUCTION axis (the key), so it comes out of the sum exactly:
    #     sum_n p8[m,n] * e4m3(V[n,d]/b[d])  ~  (sum_n p_un[m,n] V[n,d]) / b[d]
    # and one `(HEAD_DIM,)` multiply restores it.
    # It MUST come after the `1/(l_i * P_SCALE)` rescale (it is part of V, not of
    # P) and BEFORE the F086 add-back (which is in V's centred units, so `b` must not
    # touch `mu`). Same placement rule as `PSC_DYN`'s inverse.
    # -------------------------------------------------------------------
    if MUL_VCH:
        vch = tl.load(VCH + off_hz * HEAD_DIM + offs_d)
        acc = acc * vch[None, :]

    # -------------------------------------------------------------------
    # F086: `smooth_v` -- add V's per-channel sequence-axis mean back.
    #
    # EXACT, and NOT the same thing as `smooth_k`. `smooth_k` removes a term
    # softmax cancels (nothing comes back). `smooth_v` removes a term softmax
    # does NOT cancel, but because every softmax row sums to 1,
    #     sum_j P_ij (V_jd - mu_d) = out_id - mu_d,
    # so the correction is a KNOWN additive constant per (b, h, channel) applied
    # once here. No approximation and no free parameter.
    #
    # It goes AFTER the normalisation: `acc` at this point is the true
    # `P @ V_centred`, and `mu` is in the same units. Adding it before the
    # `1/(l_i * P_SCALE)` rescale would be wrong by that factor.
    # It goes BEFORE the `.to(Out.dtype.element_ty)` cast, so the sum is
    # formed in fp32 and rounded to fp16 ONCE. A separate post-pass over the
    # fp16 output would round twice.
    # KQT=2's `tl.trans` has already landed above, so `acc` is (BLOCK_M,
    # HEAD_DIM) in every arm and the broadcast axis is the same one.
    # V ONLY. `VM` is V's mean; K's mean is never added back anywhere.
    # -------------------------------------------------------------------
    if ADD_VMEAN:
        vmu = tl.load(VM + off_hz * HEAD_DIM + offs_d)
        acc = acc + vmu[None, :]

    o_ptrs = Out + o_base + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    if EVEN_M:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty))
    else:
        tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=m_ok[:, None])

    if STORE_LSE:
        lse = m_i * LN2 + tl.math.log(l_i)
        # The per-query vectors are always indexed by the QUERY offsets, so this
        # store is identical in both orientations.
        if EVEN_M:
            tl.store(Lse + off_hz * N_CTX + offs_m, lse)
        else:
            tl.store(Lse + off_hz * N_CTX + offs_m, lse, mask=m_ok)


def _pad_tokens(t, n_pad, fill):
    """F060 TAIL FIX: extend the TOKEN axis of `t` to `n_pad` with `fill`.

    Handles both the 4-D quantised tensors `(B, H, N, D)` and the 3-D per-token
    scale vectors `(B, H, N)`. The token axis is always axis 2.

    Returns `t` unchanged when it is already long enough, so a caller that pads
    conditionally cannot silently change an aligned shape.
    """
    import torch
    n = t.shape[2]
    if n >= n_pad:
        return t
    out = torch.full(t.shape[:2] + (n_pad,) + t.shape[3:], fill,
                     dtype=t.dtype, device=t.device)
    out[:, :, :n] = t
    return out.contiguous()


def plan_kv_path(N, BLOCK_N, pad_kv, load_unmasked, even_n_override=None):
    """F065/F066: decide the K/V padding + `EVEN_N` plan, and REFUSE the unsafe combinations.

    Extracted from `flash_attn_fp8` so it can be unit-tested on CPU, with no GPU and
    no compile. Returns `(even_n, do_pad_kv, n_pad_or_None, load_unmasked_effective)`.

    `even_n_override` tri-state. THE DEFAULT WAS FLIPPED TO THE FAST PATH BY F068
    (F068): F066's bit-identity precondition had already
    failed, and F068's pre-registered tail check found no systematic accuracy bias at
    the F067 failing regime (16/16 fresh inputs had a bit-identical max error, and the
    fast arm was better on mean error in 15/16).
        None         DEFAULT -- the fast path, `EVEN_N = False` at every N. +15.6% to
                     +22.7% at every aligned shape measured (F066, 120 cells, 20 real
                     shapes). It is NOT bit-identical to the pre-F068 default at
                     an aligned NON-CAUSAL N; F068 measured that difference as a random
                     element-level fp16-rounding reshuffle, not a regime-wide shift.
        0 / False    identical to the DEFAULT. Kept as an explicit spelling.
        "shipped"    the OPT-OUT -- `EVEN_N = (N % BLOCK_N == 0)`, the pre-F068
                     computation verbatim, for callers who need the old default.
        True         force the aligned path. At a ragged N this is REFUSED (hazard 2).

    WHY THE FAST PATH IS A TRADE, AND HOW F068 ADJUDICATED IT. F066 item 1 failed:
    the premise was that at an ALIGNED N the gated score mask is all-true so the two
    paths must be bit-identical. **That is FALSE for the NON-CAUSAL path.** Measured:
      * CAUSAL: bit-identical, `max|d| = 0.0e+00` at all 11 shapes, 0 NaN. ✔
      * NON-CAUSAL: `max|d|` = 3e-5 … 1.2e-4, i.e. 0.5–0.9% of the output scale. ✘
    The ragged path is a DIFFERENT COMPILED KERNEL, not a constexpr swap (loop body
    551 vs 418). It emits `v_cndmask_b32_e32` 16 vs 0 -- the `tl.where(n_ok, qk, -inf)`
    select IS materialised because `start_n` is a runtime induction variable -- and its
    presence shifts `v_mul_f32_e32` 153->26 into `v_dual_mul_f32` 27->88, so the
    arithmetic is REASSOCIATED.
    The change IS accuracy-neutral: both arms agree with fp64 truth to 6 significant
    figures (N=4096: 3.091731e-04 vs 3.091733e-04). It is a bit-reproducibility change,
    not an accuracy regression. NO configuration restores bit-identity (7 tried), and
    a compile-time `NEEDS_MASK` constexpr was tried and REVERTED (32 spills, still not
    identical). A second loop body would be F060's `TAIL_PEEL`: 0->24/0->56 spills and
    a 13-23% LOSS. **So this is a genuine trade, and the caller must make it.**
    **F068 (F068) made the call the F066 gate left open:
    the non-causal difference is an element-level fp16-rounding reshuffle (the two arms'
    outputs differ on ~15% of elements; whether the single worst-error element is one of
    them is input-specific), so it is not a systematic accuracy regression.**
    Full evidence: F066, F068.

    `load_unmasked` tri-state:
        None  AUTO (the default) -- unmask the loads iff the buffers were padded.
        True  EXPLICIT request -- refuse if padding did not happen.
        False force masked loads.

    AUTO is not a convenience, it is required for correctness of the DEFAULT.
    `do_pad_kv` is False only when the loads are already unmasked, and an EXPLICIT
    `load_unmasked=1` at such a shape is a caller error and is refused -- which is
    exactly what caught the first version of this default.

    Two hazards, both of which silently read past the end of K/V — the F062 defect,
    where fp8 E4M3FN's NaN encodings (0x7F/0xFF) make `0 * NaN = NaN` in P@V and a
    VALID row goes NaN with a count that depends on the memory that happens to follow:

      1.  `load_unmasked=True` with UNPADDED buffers. The final block's load is no
          longer masked, so it reads past N.
      2.  `even_n_override=True` at a RAGGED N. This claims `N % BLOCK_N == 0` when it
          is not, so the kernel takes the unmasked-load path on a ragged N.

    Both are refused loudly. Neither is a performance question.
    """
    ragged = (N % BLOCK_N) != 0
    # F068: `None` is now the FAST PATH (`EVEN_N=False`). The pre-F068 default
    # (`not ragged`) is preserved under the explicit `"shipped"` spelling. F066's
    # bit-identity precondition failed non-causally; F068's pre-registered tail check
    # found no systematic accuracy bias, so the fast path is the default.
    if even_n_override is None:
        even_n = False                       # F068: the promoted fast path
    elif isinstance(even_n_override, str):
        if even_n_override != "shipped":
            raise ValueError(
                f"even_n_override={even_n_override!r} is not understood. Use None "
                "(the DEFAULT: the fast path, EVEN_N=False at every N), 0/False "
                "(identical to None), \"shipped\" (the OPT-OUT: EVEN_N = "
                "N % BLOCK_N == 0), or True."
            )
        even_n = not ragged                  # the pre-F066 computation, verbatim
    else:
        even_n = bool(even_n_override)

    if even_n and ragged:
        raise ValueError(
            f"even_n_override=1 claims EVEN_N (N % BLOCK_N == 0) but N={N} and "
            f"BLOCK_N={BLOCK_N} give N % BLOCK_N = {N % BLOCK_N}. Forcing the "
            "aligned path on a ragged N takes the UNMASKED load path and reads past "
            "the end of K/V (the F062 defect). Only even_n_override=0 (or None) is "
            "valid at a ragged N."
        )

    # The padding decision follows the EFFECTIVE raggedness, not `N % BLOCK_N`.
    # Forcing the ragged path at an ALIGNED N makes the kernel take the masked path,
    # so it must be padded too. Padding an already-aligned N is harmless: it pads to
    # the same length.
    effective_ragged = ragged or not even_n
    do_pad_kv = bool(pad_kv) and effective_ragged

    if load_unmasked is None:
        load_unmasked_eff = do_pad_kv          # AUTO
    elif load_unmasked:
        if not do_pad_kv:
            raise ValueError(
                "load_unmasked=True was requested explicitly, but the K/V buffers "
                f"were NOT padded (N={N}, BLOCK_N={BLOCK_N}, pad_kv={pad_kv!r}, "
                f"effective_ragged={effective_ragged}). Unmasked loads without "
                "padding read past the end of K/V (the F062 defect) and can silently "
                "produce NaN in valid rows. Pass pad_kv=1, or load_unmasked=False. "
                "(load_unmasked=None means AUTO: unmask only when padding happened.)"
            )
        load_unmasked_eff = True
    else:
        load_unmasked_eff = False

    n_pad = (triton.cdiv(N, BLOCK_N) * BLOCK_N) if do_pad_kv else None

    # -------------------------------------------------------------------
    # THE ALIGNED-N UNMASKED-LOAD INVARIANT -- DELIBERATE, NOT INCIDENTAL.
    #
    # At an ALIGNED N with `even_n=False` this resolves to EVEN_N=0, PAD_KV=1,
    # n_pad=N, LOAD_UNMASKED=1. `_pad_tokens` sees n == n_pad and returns the SAME
    # tensors, so there is NO copy: the loads are UNMASKED over buffers that are
    # exactly N long. That is safe ONLY because of the F062 clamp in the kernel,
    #
    #     hi = tl.minimum(tl.cdiv((start_m + 1) * BLOCK_M, BLOCK_N),
    #                     tl.cdiv(n_pad, BLOCK_N))
    #
    # which bounds the loop at `cdiv(n_pad, BLOCK_N)`, so the highest column any
    # unmasked load can address is `cdiv(N, BLOCK_N) * BLOCK_N - 1` = `N - 1`.
    # IN BOUNDS BY CONSTRUCTION. Remove the clamp and the last query block iterates
    # past the end of K/V/ks/vs, reads fp8 E4M3FN NaN encodings (0x7F/0xFF) past
    # the allocation, and `0 * NaN = NaN` in P@V puts NaN into a VALID row. The
    # regression test is `tools/f067_clamp_regression.py` (static + GPU halves).
    #
    # This holds for the DEFAULT aligned path too, which is NOT the fast path:
    # `_masked_full` is `not EVEN_N` = False at EVEN_N=1, so its loads are unmasked
    # as well and it compiles no score mask at all. THE CLAMP IS LOAD-BEARING FOR
    # BOTH DEFAULTS.
    # -------------------------------------------------------------------
    if load_unmasked_eff:
        _loop_max = triton.cdiv(N, BLOCK_N) * BLOCK_N
        assert n_pad is not None and n_pad >= _loop_max, (
            f"unmasked loads requested with n_pad={n_pad}, but the F062 clamp lets "
            f"the loop address up to {_loop_max} columns (N={N}, BLOCK_N={BLOCK_N}). "
            "Unmasked loads shorter than that read past the end of K/V/ks/vs."
        )
    return even_n, do_pad_kv, n_pad, load_unmasked_eff


def smooth_k_axis(k):
    """F065 (3): K's mean over the SEQUENCE axis, shape (B, H, D).

    SEQUENCE axis, i.e. `dim=2` for a (B, H, N, D) tensor. F061/F063 measured
    that the per-token mean over HEAD_DIM is useless (K's row amax barely moves:
    294.4 -> 292.1), while this one shrinks it 294.4 -> 6.42 (0.022x) and improves
    per-token fp8 KL on 10/10 real captures by 1.44-20.88x.

    Computed in float32 and cast back to `k`'s dtype, so the subsequent quantiser
    sees the same dtype it always did.

    The mean is deliberately NOT returned and NOT added back. F063 check G
    proved `smooth_k` is SOFTMAX-NEUTRAL in exact arithmetic: the removed term
    contributes `q . mu` to every logit in a row, a per-row constant the softmax
    cancels exactly (KL <= 1.86e-6 against the UNCENTRED fp32 reference, where the
    unsmoothed arm gives 2.5e-3..5.2e-3). Not adding it back is also what keeps
    this change INSIDE the prologue: the attention kernel is untouched, so there is
    no new kernel input and no change to the steady-state loop.
    """
    mu = k.float().mean(dim=2, keepdim=True)
    return (k.float() - mu).to(k.dtype)


def prologue_fp8(q, k, v, block_r=16, smooth_k=False, fused_smooth_k=True,
                 smooth_v=False, return_vmean=False, out=None, skip_q=None):
    """Quantize all three inputs with one embarrassingly-parallel kernel type.

    F065 (3): `smooth_k=True` subtracts K's sequence-axis mean before quantising K.

    DEFAULT OFF. `smooth_k=False` is byte-identical to the pre-F065 prologue.

    `fused_smooth_k=True` (the default when smooth_k is on) uses `kseq_mean_fp8`, which
    reads K once. `fused_smooth_k=False` uses the torch reference
    (`smooth_k_axis`), which materialises a full `(B,H,N,D)` fp32 temporary and was
    measured at **+2.69 ms / +286% on the prologue / 15.0% of a full non-causal call**.
    The torch path is kept ONLY as the correctness reference the fused one is checked
    against (the F065 check).

    F086: `smooth_v=True` subtracts **V's** sequence-axis mean before
    quantising V, using the SAME `quant_rows_fp8(mean=...)` machinery (`_quant_rows_fp8`
    is generic over the tensor; nothing about it is K-specific). `smooth_k` and
    `smooth_v` use two independent means and never cross.

    `smooth_v` also needs the mean ADDED BACK TO THE OUTPUT, which this function
    does not do (it has no output). Pass `return_vmean=True` to get it as a 4th
    return value and hand it to `flash_attn_fp8`'s add-back -- or, more simply, call
    `flash_attn_fp8(..., smooth_v=True)` and let it do both. `return_vmean=True` with
    `smooth_v=False` returns `None`.

    F116: `skip_q=True` **skips the Q pass entirely** and returns
    `(None, None)` in Q's slot, so the caller can hand that straight to
    `flash_attn_fp8(q8=..., ...)` and let the ATTENTION KERNEL's pre-existing `FUSE_Q`
    path quantise Q in-register. It is the only pass that can be dropped:
    the kernel's fused path covers K and V too, but `smooth_k` is applied in the
    prologue, so K/V stay here. `skip_q=None` (the default) resolves to the
    module flag `Q_IN_KERNEL`, which is **0** -- so the shipped path is byte-identical.
    The extra `(None, None)` return is a *request*, not a guarantee: with
    `q_in_kernel=False` the kernel falls back to its own separate Q pass, which is the
    flag-off control arm and is bit-identical to the earlier path.
    """
    # F088: mirror `flash_attn_fp8`. `kseq_mean_fp8`
    # is now guarded internally, but a non-contiguous q/k/v reaching a DIRECT
    # `prologue_fp8` call should not depend on that. No-op when already
    # contiguous (zero copy), so it cannot change the shipped path's results.
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    out = out or (None, None, None)  # F114 flag: default off (None) leaves the path unchanged
    _skip_q = bool(Q_IN_KERNEL) if skip_q is None else bool(skip_q)
    if _skip_q:
        q8, sq = None, None
    else:
        q8, sq = quant_rows_fp8(q, block_r=block_r, out=out[0])
    if smooth_k:
        if fused_smooth_k:
            mu = kseq_mean_fp8(k)
            k8, sk = quant_rows_fp8(k, block_r=block_r, mean=mu, n_seq=k.shape[2])
        else:
            k = smooth_k_axis(k)
            k8, sk = quant_rows_fp8(k, block_r=block_r, out=out[1])
    else:
        k8, sk = quant_rows_fp8(k, block_r=block_r, out=out[1])
    if smooth_v:
        vmu = kseq_mean_fp8(v)
        v8, sv = quant_rows_fp8(v, block_r=block_r, mean=vmu, n_seq=v.shape[2])
    else:
        vmu = None
        v8, sv = quant_rows_fp8(v, block_r=block_r, out=out[2])
    if return_vmean:
        return (q8, sq), (k8, sk), (v8, sv), vmu
    return (q8, sq), (k8, sk), (v8, sv)


def prologue_buffers(q, k, v):
    """F114: caller-owned output buffers for `prologue_fp8(..., out=...)`.

    Returns a 3-tuple of `(x8, s)` pairs in `(q, k, v)` order, exactly the shapes
    `quant_rows_fp8` would have allocated.  Nothing here is on the shipped path.
    """
    import torch
    bufs = []
    for t in (q, k, v):
        D = t.shape[-1]
        R = t.numel() // D
        bufs.append((torch.empty((R, D), device=t.device, dtype=torch.float8_e4m3fn),
                     torch.empty((R,), device=t.device, dtype=torch.float32)))
    return tuple(bufs)


def flash_attn_fp8(q, k, v, causal=False, sm_scale=None, return_lse=False,
                   BLOCK_M=None, BLOCK_N=None, num_warps=None, num_stages=1,
                   p_scale=448.0, q8=None, k8=None, v8=None, hwcvt=True,
                   fuse_q=True, pcast=None, waves_per_eu=None, maxnreg=None,
                   fdiv=None, schedule_hint=None, psc=False, split_loop=False,
                   kqt=0, eexp2=False, eexp2_deg=EEXP2_DEG_DEFAULT,
                   eexp2_split=EEXP2_SPLIT_DEFAULT, tail_peel=0, pad_kv=1, load_unmasked=None, gate_mask=1, even_n_override=None,
                   smooth_k=False, fused_smooth_k=True, qk_dtype="fp8",
                   smooth_v=False, psc_dyn=None, psc_fold=None, vscale_chan=None,
                   fused_vch=None, sv_one=None, mul_vch=None, q_in_kernel=None,
                   lazy_rescale=None, lazy_tau=None):
    """q,k,v: (B,H,N,D) fp16, contiguous.

    Pass q8/k8/v8 as ((tensor, scale), ...) to skip the prologue and measure
    kernel-only time.

    `pcast` selects the fp32->e4m3 P-conversion path (see module docstring /
    F017); None means the module-level PCAST, which defaults to 1 = the
    incumbent behaviour.

    `fdiv` selects the division lowering (F021): None means the module-level
    FDIV, which defaults to 1 = hoisted per-row reciprocal + multiply. Pass 0
    for the exact incumbent broadcast IEEE divide.

    `schedule_hint` is passed straight through as an AMD-backend compile option
    (F021 lever 2); e.g. "attention,memory-bound-attention".

    `waves_per_eu` and `maxnreg` (F018) are AMD-backend COMPILE options, not
    kernel arguments. The AMD backend already declares
    `waves_per_eu: int = 0` (triton/backends/amd/compiler.py:48) and emits
    `amdgpu-waves-per-eu = "<min>, <max>"` from it (compiler.py:415), so it is
    reachable from a plain launch -- it just was never plumbed through here.
    `maxnreg` caps VGPRs/thread, the other occupancy lever.

    `split_loop` (F030) enables the two-range causal loop ported from
    `kernels/fa_fp8_pv.py` (F025/F029). It is a NO-OP when `causal=False`:
    there is no mask to remove, so `SPLIT_LOOP=0` and `SPLIT_LOOP=1` compile to
    the same binary. Off by default so the shipped path is unchanged.

    `kqt` (F031) selects the `S = K @ Q^T` operand-order swap from AOTriton
    0.14's FlyDSL gfx1201 kernel. `S = Q @ K^T` and
    `S = (K @ Q^T)^T` are the same matrix, so the swap does NOT avoid a
    transpose -- it MOVES one, from P (every KV iteration) to O (once, in the
    epilogue). `tl.dot(a, b)` contracts over `a`'s last axis, and under the swap
    P's last axis is the QUERY, while the PV GEMM must contract over the KEY.
    The only legal second GEMM is therefore `tl.dot(vt, p8)` with V loaded
    `(HEAD_DIM, BLOCK_N)` and the accumulator held transposed.
    0 = off (shipped, byte-identical); 2 = the FlyDSL form above; 3 = transpose
    the fp32 score back per iteration before the pack, isolating the QK operand
    order from the P-layout change. 1 is retired (see the kernel).
    See F031. Off by default.

    F082: `smooth_k=True` subtracts K's SEQUENCE-axis mean before K is
    quantised, i.e. it finally exposes here what `prologue_fp8` has had since F065 and
    what F063 measured as a **free 10/10 accuracy win (geomean 3.19x, worst-case KL
    5.18e-3 -> 3.02e-3, exact-arithmetic-neutral to 1.86e-6)**. It is **arithmetically
    free** (the removed term `q . mu` is a per-row constant the softmax cancels) and it
    touches **only the prologue** -- the attention kernel is untouched, so there is no
    new kernel input and no change to the steady-state loop.

    **DEFAULT OFF, and `smooth_k=False` must stay byte-identical** to the pre-F082
    path: the fused quantiser branches on a `tl.constexpr`, so no code is emitted for
    the centring when it is off. Verified by the F082 check.

    **K ONLY, NEVER V** -- see `_quant_kv_rows_fp8`'s docstring for why.

    `fused_smooth_k=True` (default) uses `kseq_mean_fp8`, which reads K once.
    `fused_smooth_k=False` uses the torch reference `smooth_k_axis`, which
    materialises a full `(B,H,N,D)` fp32 temporary (+2.69 ms / +286 % on the prologue,
    F065) and exists only as the correctness reference.

    F083: `qk_dtype` selects the QK^T datatype -- the "8+8 split".

        "fp8"   (DEFAULT) Q and K are e4m3 with `amax/448` per-token scales and
                `tl.dot(..., out_dtype=tl.float32)`. This is the SHIPPED path,
                and it is provably unchanged: `qk_dtype` becomes a `tl.constexpr`
                (`QK_INT8`) and every int8 line sits behind it, so the emitted
                AMDGCN opcode sequence and instruction count are identical to the
                pre-F083 kernel (checked by the F083 benchmark).
        "int8"  Q and K are SIGNED int8 with `amax/127` per-token scales, a
                symmetric round-to-nearest-even clamp (+-127, no zero point) and
                `tl.dot(..., out_dtype=tl.int32)` followed by a SIGNED
                int32->fp32 convert. **PV stays e4m3** -- V's bytes are bit-for-bit
                the fp8 path's.

    **`smooth_k` is a MANDATORY co-requisite for `qk_dtype="int8"`**, not a
    tuning knob: F079 / F081 measure int8 QK at 30.60 %/62.24 % cos-sim
    without it and 99.31 %/99.47 % with it. It is NOT enforced here, because P4
    of the F083 pre-registration has to be able to measure the bad arm -- but a
    caller that passes `qk_dtype="int8"` with `smooth_k=False` is asking for the
    arm F083 measured as catastrophic. The scales stay fp32 and stay at the same
    site; overflow is not a risk (`127*127*128 = 2.06e6` vs 2**31).

    F086: `smooth_v=True` subtracts **V's** per-(b,h,channel)
    SEQUENCE-axis mean before V is quantised (in the same fused prologue launch,
    via `_quant_kv_rows_fp8`'s `SUB_VMEAN`) **and adds that mean back to the
    output** inside the attention kernel (`ADD_VMEAN`). It is NOT `smooth_k`:
    `smooth_k` is softmax-neutral and nothing comes back; `smooth_v` is not
    neutral but is EXACTLY recoverable because softmax rows sum to 1, so the
    correction is a known additive constant per `(b,h,channel)`, applied once.

    **DEFAULT OFF, and `smooth_v=False` must stay byte-identical** to the
    pre-F086 path: `SUB_VMEAN` and `ADD_VMEAN` are both `tl.constexpr`, so no code
    is emitted for either when it is off. Verified by the F086 check
    (both the prologue and the attention kernel are fingerprinted).

    **V ONLY.** `smooth_v`'s mean never reaches K's absmax and never reaches the
    QK scale; `smooth_k`'s mean is never added back. The two are independent
    pointers.

    F094: `smooth_v` is now implemented for `qk_dtype="int8"` as well.
    The int8 arm's fused prologue (`quant_kv_rows_i8` / `_quant_kv_rows_i8`) grew
    the same `VMEAN`/`SUB_VMEAN` structure the fp8 kernel has had since F086, and
    this function threads `vmu` into it exactly as it does on the fp8 arm. The
    add-back (`ADD_VMEAN`) is datatype-independent and was already shared, so no
    change was needed there. `smooth_v=False` still takes the pre-F094 `else`
    arm verbatim, and `SUB_VMEAN=0` emits no code, so the int8 off path is
    bit-identical (F094 gate 1).

    When `v8` is supplied pre-quantised, `v` must still be the **uncentred** V
    (the add-back mean is computed from it) and the supplied `v8` must be V's
    **centred** quantisation -- `prologue_fp8(..., smooth_v=True,
    return_vmean=True)` produces exactly that pair.
    """
    B, H, N, D = q.shape
    assert q.dtype == torch.float16
    if qk_dtype not in ("fp8", "int8"):
        raise ValueError(
            f"qk_dtype={qk_dtype!r} is not understood. Use \"fp8\" (the DEFAULT, "
            "the shipped e4m3 QK path) or \"int8\" (signed int8 QK with amax/127 "
            "per-token scales; PV stays e4m3)."
        )
    qk_int8 = (qk_dtype == "int8")
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

    # F086: V's per-(b,h,channel) sequence-axis mean, shape (B, H, D). Computed
    # ONCE here and used twice: subtracted before V's absmax in the prologue, and
    # added back to the output by the attention kernel. `kseq_mean_fp8` is
    # tensor-generic -- it reads V once, exactly as it reads K for `smooth_k`.
    # F104: with `VSCALE_CHAN_FUSED=1` and `smooth_v` the mean is produced by the
    # FUSED stats pass below (`kseq_mean_amax_fp8`), so it is NOT computed here.

    # F100: V's per-(b,h,channel) sequence-axis absmax, used as V's
    # quantisation divisor instead of V's per-ROW amax. `kseq_amax_fp8` centres by
    # `vmu` when `smooth_v` is on, so `b` is `max(amax_n |V[n,d] - mu[d]|, 1)/P_SCALE`
    # -- exactly the quantity that makes V's quantisation benefit from centring.
    # When `smooth_v` is off, `b` is `max(amax_n |V[n,d]|, 1)/P_SCALE`.
    #
    # UNCONDITIONALLY SAFE, and the reason is worth stating: the prologue writes
    # `sv = 1.0`, so P's grid is exactly `P_SCALE` and `p8 = e4m3(p_un * P_SCALE)`
    # with `p_un <= 1` -- P cannot overflow. And V is divided by its OWN channel
    # absmax over `P_SCALE`, so `|V/b| <= P_SCALE` by construction -- V cannot
    # overflow either. Neither bound depends on the data. The `max(., 1)` floor is
    # the shipped path's (`quant_triton._quant_kv_rows_fp8`), kept so an all-zero
    # channel cannot divide by zero.
    _vsc = VSCALE_CHAN if vscale_chan is None else (1 if vscale_chan else 0)
    _fused = VSCALE_CHAN_FUSED if fused_vch is None else (1 if fused_vch else 0)
    if _fused and not _vsc:
        raise ValueError("fused_vch requires vscale_chan=True (the fused prologue is "
                         "the chan path's; there is nothing to fuse otherwise)")
    if _vsc and _fused and smooth_v:
        # F104: ONE read of V -> BOTH the sequence-axis mean and the centred
        # per-channel absmax. Replaces `kseq_mean_fp8(v)` + `kseq_amax_fp8(v, mean=...)`
        # (two reads, four launches) with one read and two launches, bit-identically.
        vmu, _amax = kseq_mean_amax_fp8(v)
    else:
        vmu = kseq_mean_fp8(v) if smooth_v else None
        _amax = None
    if _vsc:
        if _amax is None:
            _amax = kseq_amax_fp8(v, mean=vmu)                  # (B, H, D)
        if not bool(torch.isfinite(_amax).all()):
            raise ValueError("vscale_chan: non-finite V channel absmax")
        vch = (torch.clamp(_amax, min=1.0) / p_scale).reshape(B * H, D)
    else:
        vch = None

    # -------------------------------------------------------------------
    # `SV_ONE`: under `VSCALE_CHAN` the prologue above writes `sv = 1.0`
    # for EVERY token, so `_kv_body`'s per-K-tile `sv` load and its `* sv` fold are
    # dead work. `SV_ONE=1` is the constexpr that deletes both.
    #
    # SOUNDNESS, not an optimisation flag: it asserts a property of the `sv`
    # tensor THIS call will read. That holds exactly when (a) `vch is not None`
    # (`VSCALE_CHAN` is on) and (b) the prologue below actually ran -- a caller that
    # supplies pre-quantised `k8`/`v8` supplies its OWN `sv`, which need not be ones.
    # `sv_one=True` is therefore rejected unless both hold; `sv_one=False` forces the
    # earlier path (it is what reproduces `F105`'s recorded numbers).
    # -------------------------------------------------------------------
    _sv_auto = 1 if (vch is not None and k8 is None and v8 is None) else 0
    if sv_one is None:
        _sv_one = _sv_auto
    else:
        _sv_one = 1 if sv_one else 0
        if _sv_one and not _sv_auto:
            raise ValueError(
                "sv_one=True requires a prologue-written `sv` of ones: it needs "
                "vscale_chan=True with the prologue running (a pre-quantised caller "
                "supplies its own `sv`, which need not be 1.0)")

    # -------------------------------------------------------------------
    # `MUL_VCH` (`F111`): the epilogue's per-channel V divisor
    # re-apply (`_attn_fwd_fp8`'s `acc = acc * vch[None, :]`) was until now
    # switched ONLY by `vch is not None`, i.e. it is welded to `VSCALE_CHAN`.
    # `mul_vch=None` keeps that behaviour EXACTLY (the default path is
    # bit-for-bit unchanged -- see the F111 check).
    #
    # `mul_vch=False` with `vch is not None` yields a NUMERICALLY WRONG
    # output: the V per-channel divisor is not re-applied. It exists ONLY as a
    # TIMING PROBE for `F111` (it prices the epilogue multiply on the kernel),
    # and it is never a shipping configuration. `mul_vch=True` with `vch is
    # None` is rejected: the add-back `VM` pointer is not the divisor, so a
    # truthy `mul_vch` there would multiply by the wrong tensor.
    # -------------------------------------------------------------------
    if mul_vch is None:
        _mul_vch = 1 if vch is not None else 0
    else:
        _mul_vch = 1 if mul_vch else 0
        if _mul_vch and vch is None:
            raise ValueError(
                "mul_vch=True requires vch (VSCALE_CHAN's per-channel divisor); "
                "without it the add-back `VM` pointer would be multiplied in")

    # Tile selection when the caller does not pin one (F011, measured).
    #
    # BN=16 halves the PV operand staging per KV iteration, which was measured at
    # ~72% of loop time; it is worth ~20% at N=8192 non-causal and ~11% at N=8192
    # causal. BN=8 does NOT compile (fp8 WMMA needs K>=16), so 16 is the floor.
    # BM=256/BN=16 is catastrophic (0.11-0.18x) and must not be selected.
    #
    # Measured best per shape, kernel-only, interleaved vs PR #368 (F011-F013,
    # rounds 14-25). Ratio shown is ours/PR #368, so <1 means we win.
    #   N=2048 c=0   64x16 w2  0.2461  1.02x
    #   N=2048 c=1   64x64 w4  0.6553  0.80x  (e2e, round 15)
    #   N=4096 c=0   64x16 w2  0.8021  1.26x  (round 25; 128x16 was 1.28x)
    #   N=4096 c=1   64x32 w4  0.5767  1.10x  (round 25; 64x64 was 1.12x)
    #   N=8192 c=0  128x16 w4  2.7199  1.22x
    #   N=8192 c=1  128x16 w4  1.8275  1.11x  <- we win
    #
    # Tile space is now swept at N=2048, 4096 and 8192 for both causal flags. The
    # wins are small (1-2%) away from N=8192; the large effects were BN=16 itself
    # (F011, ~20%) and the causal/non-causal split. `num_warps` is resolved here
    # too, because the short non-causal optimum differs in warp count as well.
    if BLOCK_M is None or BLOCK_N is None:
        if not causal and N < 8192:
            BLOCK_M, BLOCK_N = 64, 16
            if num_warps is None:
                num_warps = 2
        elif causal and N < 8192:
            BLOCK_M, BLOCK_N = (64, 32) if N >= 4096 else (64, 64)
        else:
            BLOCK_M, BLOCK_N = 128, 16
    if num_warps is None:
        num_warps = 4
    # Quantizing Q inside the attention kernel removes one of the three prologue
    # passes (measured at ~50% overhead at N=2048). Skipped when pre-quantized
    # inputs are supplied, i.e. for kernel-only measurement.
    #
    # F116: `prologue_fp8(skip_q=True)` returns `(None, None)` in
    # Q's slot; that is the "no Q prologue" signal, so normalise it to `q8 = None` BEFORE
    # the `do_fuse_q` test -- otherwise the tuple is truthy and the fused path is skipped,
    # silently leaving `q8 = (None, None)`. `q_in_kernel` overrides the `fuse_q` default
    # per call (None = keep `fuse_q`).
    if isinstance(q8, tuple) and len(q8) == 2 and q8[0] is None:
        q8 = None
    do_fuse_q = (bool(fuse_q) if q_in_kernel is None else bool(q_in_kernel)) and (q8 is None)
    if q8 is None:
        if do_fuse_q:
            q8 = q                    # raw fp16; the kernel reads it via `Q`
            sq = torch.empty((1, 1), device=q.device, dtype=torch.float32)
        elif qk_int8:
            q8, sq = quant_rows_i8(q)
        else:
            q8, sq = quant_rows_fp8(q)
    else:
        q8, sq = q8
    if k8 is None and v8 is None:
        # One launch for both (F008): same shape, same per-row scheme.
        #
        # F082: `smooth_k` centres K (never V) inside that same fused launch.
        # The three arms mirror `prologue_fp8` exactly, so the two entry points
        # agree bit-for-bit on K8/SK (checked by the F082 check).
        #
        # F083: with `qk_dtype="int8"` the FUSED prologue is the separate
        # `quant_kv_rows_i8` kernel -- K to signed int8 (amax/127), V to e4m3 by
        # the identical `_cvt2d` -- so the fp8 prologue's instructions cannot move.
        # F086: V's mean is threaded into the fused launch. With `smooth_v=False`
        # `_vkw` is EMPTY, so the `else` arm below is literally the pre-F086 call
        # `quant_kv_rows_fp8(k, v)` and the off path cannot move.
        # F094: the same `_vkw` is now also threaded into the int8 arms, so
        # `smooth_v` works on `qk_dtype="int8"` too. `_vkw` stays EMPTY when
        # `smooth_v=False`, so both off paths are untouched.
        # F100: `_vkw` additionally carries `vch` when `VSCALE_CHAN=1`. With the
        # flag off it is the pre-F094 dict (or empty), so both off paths are
        # untouched.
        _vkw = {}
        if vmu is not None:
            _vkw.update(vmean=vmu, n_seq=v.shape[2])
        if vch is not None:
            _vkw.update(vch=vch, n_seq=v.shape[2])
        if _fused:
            # F104: the FUSED chan quantiser. K exactly as the arm requires;
            # V written ONCE against `VCH` with `sv = 1.0` -- no row-`sv` pass and no
            # redundant `V8` store. `vch` is always non-None here (enforced above).
            # A NEW kernel, so the flag-off paths below are the pre-F104 calls verbatim.
            _QF = quant_kv_chan_fused_i8 if qk_int8 else quant_kv_chan_fused_fp8
            _fkw = dict(vch=vch, n_seq=v.shape[2])
            if vmu is not None:
                _fkw["vmean"] = vmu
            if smooth_k and not fused_smooth_k:
                k = smooth_k_axis(k)
                k8, sk, v8, sv = _QF(k, v, **_fkw)
            elif smooth_k:
                _mu = kseq_mean_fp8(k)
                k8, sk, v8, sv = _QF(k, v, mean=_mu, **_fkw)
            else:
                k8, sk, v8, sv = _QF(k, v, **_fkw)
        elif qk_int8:
            if smooth_k and not fused_smooth_k:
                k = smooth_k_axis(k)
                k8, sk, v8, sv = quant_kv_rows_i8(k, v, **_vkw)
            elif smooth_k:
                _mu = kseq_mean_fp8(k)
                _kw = dict(mean=_mu, n_seq=k.shape[2])
                _kw.update(_vkw)
                k8, sk, v8, sv = quant_kv_rows_i8(k, v, **_kw)
            elif _vkw:
                k8, sk, v8, sv = quant_kv_rows_i8(k, v, **_vkw)
            else:
                k8, sk, v8, sv = quant_kv_rows_i8(k, v)
        elif smooth_k and not fused_smooth_k:
            # Torch reference: centre K up front, then quantise it normally.
            k = smooth_k_axis(k)
            k8, sk, v8, sv = quant_kv_rows_fp8(k, v, **_vkw)
        elif smooth_k:
            # Fused path (default): one extra read of K for the mean, no temporary.
            _mu = kseq_mean_fp8(k)
            _kw = dict(mean=_mu, n_seq=k.shape[2])
            _kw.update(_vkw)
            k8, sk, v8, sv = quant_kv_rows_fp8(k, v, **_kw)
        elif _vkw:
            k8, sk, v8, sv = quant_kv_rows_fp8(k, v, **_vkw)
        else:
            k8, sk, v8, sv = quant_kv_rows_fp8(k, v)
    else:
        # Pre-quantized inputs are (tensor, scale) PAIRS, matching `q8` above.
        # A bare tensor is a caller error: the dequant scale cannot be recovered
        # from it, so there is no correct thing to do with one.
        #
        # This used to be `k8, sk = k8 if k8 is not None else ...`, which unpacked
        # a 4-D tensor into two names and surfaced as an opaque ValueError --
        # reported as F012 problem 2. Diagnosed and fixed in round 17.
        def _pair(x, name):
            if x is None:
                return None
            if isinstance(x, tuple) and len(x) == 2:
                return x
            raise ValueError(
                f"{name} must be a (tensor, scale) tuple when pre-quantized, got "
                f"{type(x).__name__}. The dequant scale cannot be inferred from a "
                f"bare tensor; call quant_rows_fp8()/quant_kv_rows_fp8() and pass "
                f"the pair it returns."
            )
        kk, vv = _pair(k8, "k8"), _pair(v8, "v8")
        # F083: a missing K in the int8 arm must be quantised to int8, not fp8 --
        # the kernel loads it as int8 and the scale convention is amax/127.
        k8, sk = kk if kk is not None else (quant_rows_i8(k) if qk_int8
                                            else quant_rows_fp8(k))
        v8, sv = vv if vv is not None else quant_rows_fp8(v)

    o = torch.empty_like(q)
    lse = torch.empty((B * H, N), device=q.device, dtype=torch.float32)
    if sm_scale is None:
        sm_scale = D ** -0.5
    if pcast is None:
        pcast = PCAST
    if fdiv is None:
        fdiv = FDIV
    # F098: None means the module-level switch, which defaults to 0 = OFF.
    if psc_dyn is None:
        psc_dyn = PSC_DYN
    # F099: `psc_fold` is the same per-row scale folded into the softmax row max; it is the
    # cheaper but LAGGED alternative to `psc_dyn`, so the two are mutually exclusive.
    if psc_fold is None:
        psc_fold = PSC_FOLD
    if psc_fold and psc_dyn:
        raise ValueError("psc_fold and psc_dyn are mutually exclusive")
    # The LAZY softmax rescale. None means the module-level switch.
    if lazy_rescale is None:
        lazy_rescale = LAZY_RESCALE
    _lazy = 1 if lazy_rescale else 0
    if _lazy:
        if lazy_tau is None:
            lazy_tau = LAZY_TAU
        _lazy_tau = float(lazy_tau)
        # Refuse every combination the kernel does not implement, rather than
        # silently producing a wrong number (as for `sv_one`).
        _unsupported = []
        if kqt:
            _unsupported.append("kqt")
        if psc_fold:
            _unsupported.append("psc_fold")
        if psc_dyn:
            _unsupported.append("psc_dyn")
        if tail_peel:
            _unsupported.append("tail_peel")
        if eexp2:
            _unsupported.append("eexp2")
        if _lazy_tau <= 0.0:
            _unsupported.append("lazy_tau<=0")
        if _unsupported:
            raise ValueError("LAZY_RESCALE is not implemented with: "
                             + ", ".join(_unsupported))
    else:
        _lazy_tau = 0.0

    # -------------------------------------------------------------------
    # F060 TAIL FIX: pad K, V, ks and vs to a multiple of BLOCK_N.
    #
    # With the F062 clamp, `hi <= cdiv(N, BLOCK_N)`, so the loop's last iteration
    # is the block containing token `N-1`; only that block's TAIL columns are
    # outside `[0, N)`. Padding the four K-side buffers makes those columns real,
    # in-bounds memory, so the kernel can use UNMASKED loads in every iteration
    # and drop the `EVEN_N` masked-load path that F060 measured at +8.45%.
    #
    # Pad values are chosen so the SCORE mask can still do the work:
    #     K/V pad = 0  -> `qk = 0` for a padded column (NOT -inf, hence the mask)
    #     vs  pad = 0  -> contributes 0 to P@V
    #     ks  pad = 1  -> keeps `ks` a sane multiplier for the masked column
    # The padded columns are then removed by the existing `tl.where(n_ok, -inf)`,
    # which is the ONE mask that must stay.
    #
    # `sq`/Q are NOT padded: only the key axis needs it. `m_ok` and the store
    # stay keyed off the true `N`, so no padded query row is ever written.
    #
    # Cost: two extra copies of K/V (0.5x each of the pair) plus the scale
    # vectors. Skipped entirely when `N % BLOCK_N == 0`, where nothing is needed.
    # -------------------------------------------------------------------
    # -------------------------------------------------------------------
    # F065/F066: `even_n_override` -- choose the `EVEN_N` constexpr.
    #
    # F068: `None` (the DEFAULT) is the FAST PATH -- `EVEN_N = False` at every N.
    # `even_n_override=0` is an explicit spelling of the same arm. The pre-F066
    # computation (`EVEN_N = N % BLOCK_N == 0`) is the `"shipped"` OPT-OUT, for callers
    # who need the old default. Promotion: F068;
    # per-tile coverage: F069.
    #
    # WHY. F064 found the new kernel at N=8771 (ragged, 15.01 ms) is 16% FASTER than
    # at N=8768 (aligned, 17.98 ms) while doing strictly MORE work (549 key blocks vs
    # 548, plus a padding copy). F065 separated the two things the 8768->8769 cliff
    # flips at once and found the `EVEN_N` choice is worth +19.3% on its own, while
    # Triton's divisibility-by-16 specialisation of `N_CTX` is worth -0.04% (it DOES
    # change the binary -- the F065 check -- it just does not matter).
    #
    # F066 GATED MAKING `0` THE DEFAULT AND IT FAILED. The precondition was that at
    # an ALIGNED N the ragged path is bit-identical. **CAUSAL: yes (0.0e+00). NON-CAUSAL:
    # NO** -- `max|d|` 3e-5…1.2e-4, 0.5–0.9% of the output scale, because the ragged
    # path is a different compiled kernel that reassociates the multiplies. It IS
    # accuracy-neutral (6 significant figures vs fp64 on both arms) and it IS worth
    # +15.6…+22.7% at every aligned shape (120 cells, 20 shapes), but it is a bit
    # change, so the caller opts in. No configuration and no source-level change
    # tried so far restores bit-identity. See F066.
    #
    # The padding plan and both safety refusals live in `plan_kv_path`, so they are
    # unit-testable on CPU with no GPU and no compile (the F065 and F066
    # checks). Do not inline these
    # conditions back here.
    # -------------------------------------------------------------------
    even_n, do_pad_kv, n_pad, load_unmasked_eff = plan_kv_path(
        N, BLOCK_N, pad_kv, load_unmasked, even_n_override)
    if do_pad_kv:
        k8 = _pad_tokens(k8, n_pad, 0)
        v8 = _pad_tokens(v8, n_pad, 0)
        sk = _pad_tokens(sk, n_pad, 1.0)
        sv = _pad_tokens(sv, n_pad, 0.0)

    grid = (triton.cdiv(N, BLOCK_M), B * H)
    # AMD-backend compile options (F018). Only forward what the caller set, so
    # the default code path compiles byte-identically to before this change.
    kopts = {}
    if waves_per_eu is not None:
        kopts["waves_per_eu"] = waves_per_eu
    if maxnreg is not None:
        kopts["maxnreg"] = maxnreg
    if schedule_hint is not None:
        # F021 lever 2: reachable with no patch and no upgrade. The AMD backend
        # declares `schedule_hint: str = ""` and its docstring names our exact
        # case ("memory bound and has a lot of elementwise operations from fused
        # operand dequantizations"). It sets `amdgpu-sched-strategy=iterative-ilp`
        # and, for the "attention" hint, `sink-insts-to-avoid-spills`.
        kopts["schedule_hint"] = schedule_hint
    _attn_fwd_fp8[grid](
        q, q8, k8, v8, sq, sk, sv, o, lse,
        q8.stride(0), q8.stride(1), q8.stride(2), q8.stride(3),
        k8.stride(0), k8.stride(1), k8.stride(2), k8.stride(3),
        v8.stride(0), v8.stride(1), v8.stride(2), v8.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        sq.stride(0), sq.stride(1), sk.stride(0), sk.stride(1),
        sv.stride(0), sv.stride(1),
        B, H, N, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
        IS_CAUSAL=causal, EVEN_M=(N % BLOCK_M == 0), EVEN_N=even_n,
        P_SCALE=p_scale, STORE_LSE=return_lse, HWCVT=hwcvt,
        FUSE_Q=do_fuse_q, PCAST=pcast, FDIV=fdiv, PSC=psc,
        SPLIT_LOOP=1 if split_loop else 0, KQT=kqt,
        EEXP2=1 if eexp2 else 0, EEXP2_DEG=eexp2_deg,
        EEXP2_SPLIT=eexp2_split, TAIL_PEEL=1 if tail_peel else 0,
        PAD_KV=1 if do_pad_kv else 0,
        LOAD_UNMASKED=1 if load_unmasked_eff else 0,
        GATE_MASK=1 if gate_mask else 0,
        QK_INT8=1 if qk_int8 else 0,
        PSC_DYN=1 if psc_dyn else 0,
        PSC_FOLD=1 if psc_fold else 0,
        # F086: the add-back mean, shape (B*H, D) fp32. A REAL pointer is
        # always passed: `sv` stands in when the add-back is off, so no arm can
        # ever dereference `None`. It is never loaded with `ADD_VMEAN=0`.
        VM=(vmu.reshape(B * H, D) if vmu is not None else sv),
        ADD_VMEAN=1 if vmu is not None else 0,
        # F100: V's per-channel divisor, shape (B*H, D) fp32. Same
        # contract as `VM`: a REAL pointer is always passed (`sv` stands in when
        # the flag is off) and it is never loaded with `MUL_VCH=0`.
        VCH=(vch if vch is not None else sv),
        MUL_VCH=_mul_vch,
        # Delete the dead `sv` load + `* sv` fold when `sv` is all ones.
        SV_ONE=_sv_one,
        # The LAZY softmax rescale.
        LAZY=_lazy,
        LAZY_TAU=_lazy_tau,
        num_warps=num_warps, num_stages=num_stages, **kopts,
    )
    return (o, lse) if return_lse else o
