// SK1 D64: the SK1 hand kernel for HEAD_DIM == 64.
//
// Derived by hand from sk1_t4a1n.hip, the variant that carries the `p*amax_v` overflow fix
// (`S`/`SV'`) and the generalised (row, head, batch) Q/O strides that make strided-HND and NHD
// calls work.  The changes are exactly:
//
//   1. `#define HD 128` -> `HD 64`, and `#define KSTRIDE 152` -> `72`.  K's LDS row is now
//      64 data bytes; 72 = 64 + 8 keeps the 16-lane 8-byte A-operand read conflict-free
//      (72*lr mod 128 takes 16 distinct values for lr = 0..15: gcd(72,128) = 8, order 16).
//      `VSTRIDE` stays 72 -- the V tile's row is the KEY axis (BN = 64 keys), which does not
//      change with HD, so V's staging, stride and bank pattern are identical.
//   2. `NDT = HD/16 = 4`: the Q fragments (`qh`/`qf`), the output tiles (`acc`) and the PV
//      unroll are 4 per wave instead of 8.  The QK score fragments (`sc`/`pa`) are 2x2 per
//      iteration in both kernels -- they are sized by BN, not by HD -- so the softmax, the
//      mask, the P fold and the rescale are untouched.
//   3. The staging is re-indexed for a 64-byte row: 4 threads per row at 16 bytes each
//      (256 threads x 16 B = 4096 B = the whole K tile, and the same for V) instead of
//      4 threads per row at 32 bytes on a 128-byte row.
//   4. The body is a template over `BM_T` (query rows per CTA) and `WARPS`, so the same
//      numerics core is instantiated for a 128-row CTA (8 warps) and a 256-row CTA
//      (16 warps).  `BM_T/BN` replaces the literal `2` in the causal `full`/`j_hi` bounds.
//   5. Three entry-point pairs:
//        sk1d64_*   BM=128, 8 warps,  waves_per_eu(6,6)   -- the direct D=64 analogue
//        sk1d64w_*  BM=128, 8 warps,  waves_per_eu(8,8)   -- the 8-waves/SIMD variant
//        sk1d64m_*  BM=256, 16 warps, waves_per_eu(8,8)   -- the larger-query-tile variant
//      `BM_T == WARPS*16` in every case (each warp owns 16 query rows, unchanged).
//
// Everything else is unchanged: the per-token Q quantisation and its `shfl_xor 16` amax, the
// exp2-domain scores, the one-rescale-per-64-keys schedule, the row-sum-before-the-`sv*448`-fold
// order, `cvt_pk_fp8_f32` packing, the CTA-uniform `need_mask` branch, the `S`/`SV'` fold and its
// epilogue `x S`, and the `acc/(l*448) + vmean` epilogue.
//
// Contract (identical to sk1_t4a1n): Q (B*H, N, HD) fp16 raw; K8 (B*H, N_pad, HD) e4m3 padded;
// SK (B*H, N_pad) fp32; V8T (B*H, HD, N_pad) e4m3 (V TRANSPOSED); SV (B*H, N_pad) fp32;
// VMEAN (B*H, HD) fp32 or NULL; S (B*H,) fp32; O (B*H, N, HD) fp16; then H, N, N_pad, sm_scale
// and the six Q/O strides.  N_pad is a multiple of 64.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int      i32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));

#define BN 64
#define HD 64
#define NDT 4          // HD/16: Q fragments, output tiles and the PV unroll per wave
#define KSTRIDE 72     // 64 data bytes + 8 pad; 72*lr mod 128 is 16 distinct slots for lr<16
#define VSTRIDE 72     // unchanged from sk1_t4a1n: the V row is the KEY axis (BN = 64 keys)
#define P_SCALE 448.0f
#define LOG2E 1.4426950408889634f

__device__ __forceinline__ int pk4(float a, float b, float c, float d) {
  int r = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, r, true);
}

__device__ __forceinline__ float clamp448(float x) { return fminf(fmaxf(x, -P_SCALE), P_SCALE); }

__device__ __forceinline__ f32x8 wmma(i32x2 a, i32x2 b, f32x8 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

template <int BM_T, int WARPS, bool CAUSAL>
__device__ __forceinline__ void sk1d64_body(const _Float16* __restrict__ Q, const uint8_t* __restrict__ K8,
                                            const float* __restrict__ SK, const uint8_t* __restrict__ V8T,
                                            const float* __restrict__ SV, const float* __restrict__ VMEAN,
                                            const float* __restrict__ S,
                                            _Float16* __restrict__ O, int H, int N, int N_pad, float sm_scale,
                                            int q_rstride, int q_hstride, int q_bstride,
                                            int o_rstride, int o_hstride, int o_bstride) {
  static_assert(BM_T == WARPS * 16, "one warp owns 16 query rows");
  static_assert(BM_T % BN == 0, "the causal tile bounds need BM_T/BN to be exact");
  __shared__ __attribute__((aligned(16))) uint8_t sK[2][BN * KSTRIDE];
  __shared__ __attribute__((aligned(16))) uint8_t sV[2][HD * VSTRIDE];
  __shared__ __attribute__((aligned(16))) float sS[2][BN];
  __shared__ __attribute__((aligned(16))) float sSV[2][BN];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, lr = lane & 15, lh = lane >> 4;
  const int start_m = blockIdx.x;
  const size_t bh = (size_t)blockIdx.z * H + blockIdx.y;
  // Generalised (b,h) addressing for Q and O.  `q_rstride` is the element stride between
  // consecutive query rows, `q_hstride` between consecutive heads and `q_bstride` between
  // consecutive batches, so the plane base is not `bh*N*HD` in general.  For a contiguous HND
  // (B,H,N,D) tensor (rs,hs,bs) = (HD, N*HD, H*N*HD) and both addresses reduce exactly to the
  // sk1_t4a1s form.
  const size_t qbase = (size_t)blockIdx.z * (size_t)q_bstride
                     + (size_t)blockIdx.y * (size_t)q_hstride;
  const size_t obase = (size_t)blockIdx.z * (size_t)o_bstride
                     + (size_t)blockIdx.y * (size_t)o_hstride;
  // The per-(b,h) rescale the prologue applied to SV.  CTA-uniform, one load.
  const float Sb = S[bh];
  const int q0 = start_m * BM_T + w * 16;
  const int qrow = q0 + lr;
  const int n_tiles = N_pad / BN;
  const int j_hi = CAUSAL ? min(n_tiles, (BM_T / BN) * (start_m + 1)) : n_tiles;
  const int full = CAUSAL ? (BM_T / BN) * start_m : n_tiles;

  // ---- Q: raw fp16 -> e4m3, per-token scale, 8 B fragments (NDT of them at HD=64) ----
  f16x8 qh[NDT];
  float amax = 0.f;
  const _Float16* qp = Q + qbase + (size_t)qrow * (size_t)q_rstride + 8 * lh;
#pragma unroll
  for (int dt = 0; dt < NDT; ++dt) {
    if (qrow < N) {
      qh[dt] = *(const f16x8*)(qp + 16 * dt);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) qh[dt][e] = (_Float16)0.f;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf((float)qh[dt][e]));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
  amax = fmaxf(amax, 1e-12f);
  const float qinv = P_SCALE / amax;
  const float c_q = sm_scale * LOG2E * (amax / P_SCALE);
  i32x2 qf[NDT];
#pragma unroll
  for (int dt = 0; dt < NDT; ++dt) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = clamp448((float)qh[dt][e] * qinv);
    qf[dt] = (i32x2){pk4(v[0], v[1], v[2], v[3]), pk4(v[4], v[5], v[6], v[7])};
  }

  // ---- staging ----
  // 256 threads x 16 B = 4096 B = the whole K tile (64 rows x 64 B) and the whole V tile
  // (64 head-dim rows x 64 keys).  Four threads per row at 16-byte granularity, so every
  // (row, 16 B) start is 16-byte aligned and the row's 64 data bytes are covered exactly.
  // A 512-thread CTA stages with its first 256 threads (the tile is the same size).
  u32x4 kr0, vr0;
  f32x4 sr = {0.f, 0.f, 0.f, 0.f};
  auto gload = [&](int j) {
    if (tid < 256) {
      const int row = tid >> 2, col = (tid & 3) * 16;
      kr0 = *(const u32x4*)(K8 + (bh * N_pad + (size_t)j * BN + row) * HD + col);
      vr0 = *(const u32x4*)(V8T + (bh * HD + row) * N_pad + (size_t)j * BN + col);
    }
    if (tid < 16)
      sr = *(const f32x4*)(SK + bh * N_pad + (size_t)j * BN + 4 * tid);
    else if (tid < 32)
      sr = *(const f32x4*)(SV + bh * N_pad + (size_t)j * BN + 4 * (tid - 16)) * P_SCALE;
  };
  auto lstore = [&](int s) {
    if (tid < 256) {
      const int row = tid >> 2, col = (tid & 3) * 16;
      uint8_t* kd = &sK[s][row * KSTRIDE + col];
      *(u32x2*)(kd + 0) = kr0.xy;
      *(u32x2*)(kd + 8) = kr0.zw;
      uint8_t* vd = &sV[s][row * VSTRIDE + col];
      *(u32x2*)(vd + 0) = vr0.xy;
      *(u32x2*)(vd + 8) = vr0.zw;
    }
    if (tid < 16)
      *(f32x4*)&sS[s][4 * tid] = sr;
    else if (tid < 32)
      *(f32x4*)&sSV[s][4 * (tid - 16)] = sr;
  };

  f32x8 acc[NDT];
#pragma unroll
  for (int nt = 0; nt < NDT; ++nt)
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[nt][e] = 0.f;
  float m_i = -INFINITY, l_i = 0.f;

  gload(0);
  lstore(0);
  __syncthreads();

  for (int j = 0; j < j_hi; ++j) {
    const int s = j & 1;
    const bool more = (j + 1 < j_hi);
    if (more) gload(j + 1);
    const uint8_t* const kb = &sK[s][lr * KSTRIDE + 8 * lh];
    const uint8_t* const vb = &sV[s][lr * VSTRIDE + 8 * lh];
    const float* const skb = &sS[s][8 * lh];
    const float* const svb = &sSV[s][8 * lh];

    // ---------------- phase 1: QK + scale + mask + ONE row max over 64 keys ----------------
    f32x8 sc[2][2];
    bool skp[2];
    float lmax = -INFINITY;
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int base = j * BN + st * 32;
      bool skip, need_mask;
      if (CAUSAL) {
        skip = base > q0 + 15;
        need_mask = (j >= full);
      } else {
        skip = base >= N;
        need_mask = (base + 32 > N);
      }
      skp[st] = skip;
      if (skip) continue;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int e = 0; e < 8; ++e) sc[st][kk][e] = 0.f;
#pragma unroll
      for (int dt = 0; dt < NDT; ++dt)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const i32x2 a = *(const i32x2*)(kb + (st * 32 + kk * 16) * KSTRIDE + dt * 16);
          sc[st][kk] = wmma(a, qf[dt], sc[st][kk]);
        }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const f32x4 k0 = *(const f32x4*)(skb + st * 32 + kk * 16);
        const f32x4 k1 = *(const f32x4*)(skb + st * 32 + kk * 16 + 4);
        const float skv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) sc[st][kk][e] = sc[st][kk][e] * (c_q * skv[e]);
      }
      // The mask is ONE CTA-uniform branch around the whole 32-element select, so the
      // per-element predicate and the exec scaffolding it drives are skipped entirely whenever
      // `need_mask` is false.  Bit-identical: the same products are formed and the same elements
      // are set to -INFINITY, and `fmaxf` is exact.
      if (need_mask) {
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int key = base + kk * 16 + 8 * lh + e;
            if (CAUSAL ? (key > qrow) : (key >= N)) sc[st][kk][e] = -INFINITY;
          }
      }
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int e = 0; e < 8; ++e) lmax = fmaxf(lmax, sc[st][kk][e]);
    }
    lmax = fmaxf(lmax, __shfl_xor(lmax, 16, 32));
    const float m_new = fmaxf(m_i, lmax);
    const float m_use = (m_new == -INFINITY) ? 0.f : m_new;
    const float alpha = __builtin_amdgcn_exp2f(m_i - m_use);
    m_i = m_new;

    // ---------------- phase 2: P = exp2(x - m_use), row sum, pack, ONE rescale ----------------
    float psum = 0.f;
    i32x2 pa[2][2];
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      if (skp[st]) continue;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const f32x4 v0 = *(const f32x4*)(svb + st * 32 + kk * 16);
        const f32x4 v1 = *(const f32x4*)(svb + st * 32 + kk * 16 + 4);
        const float svv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          p[e] = __builtin_amdgcn_exp2f(sc[st][kk][e] - m_use);
          psum += p[e];
          p[e] *= svv[e];  // svv is already x448 (folded in at staging)
        }
        pa[st][kk] = (i32x2){pk4(p[0], p[1], p[2], p[3]), pk4(p[4], p[5], p[6], p[7])};
      }
    }
    psum += __shfl_xor(psum, 16, 32);
    l_i = l_i * alpha + psum;

    float ar[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) ar[e] = __shfl(alpha, 8 * lh + e, 32);
#pragma unroll
    for (int nt = 0; nt < NDT; ++nt)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[nt][e] *= ar[e];

    // ---------------- phase 3: PV ----------------
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      if (skp[st]) continue;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int nt = 0; nt < NDT; ++nt) {
          const i32x2 b = *(const i32x2*)(vb + nt * 16 * VSTRIDE + st * 32 + kk * 16);
          acc[nt] = wmma(pa[st][kk], b, acc[nt]);
        }
    }

    if (more) lstore(s ^ 1);
    __syncthreads();
  }

  // ---- epilogue ----
  float inv[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float l = __shfl(l_i, 8 * lh + e, 32);
    // acc is (448/S) * sum(p*v) because the prologue staged SV' = SV/S, so the
    // epilogue must carry S.  With Sb == 1.0f this is fl(1.0f/(l*448)) exactly.
    inv[e] = l > 0.f ? Sb / (l * P_SCALE) : 0.f;
  }
#pragma unroll
  for (int nt = 0; nt < NDT; ++nt) {
    const int d = nt * 16 + lr;
    const float vm = VMEAN ? VMEAN[bh * HD + d] : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = q0 + 8 * lh + e;
      if (row < N) O[obase + (size_t)row * (size_t)o_rstride + d]
          = (_Float16)(acc[nt][e] * inv[e] + vm);
    }
  }
}

#define SK1D64_ARGS                                                                               \
  const _Float16 *__restrict__ Q, const uint8_t *__restrict__ K8, const float *__restrict__ SK,   \
      const uint8_t *__restrict__ V8T, const float *__restrict__ SV, const float *__restrict__ VMEAN, \
      const float *__restrict__ S, \
      _Float16 *__restrict__ O, int H, int N, int N_pad, float sm_scale, \
      int q_rstride, int q_hstride, int q_bstride, \
      int o_rstride, int o_hstride, int o_bstride

// ---- BM=128, 8 warps, waves_per_eu(6,6) -- the direct D=64 analogue ----
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6, 6)))
sk1d64_attn_fwd_c0(SK1D64_ARGS) {
  sk1d64_body<128, 8, false>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale,
                             q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6, 6)))
sk1d64_attn_fwd_c1(SK1D64_ARGS) {
  sk1d64_body<128, 8, true>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale,
                            q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride);
}

// ---- BM=128, 8 warps, waves_per_eu(8,8) -- asks the allocator for <=192 VGPR ----
extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8)))
sk1d64w_attn_fwd_c0(SK1D64_ARGS) {
  sk1d64_body<128, 8, false>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale,
                             q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(8, 8)))
sk1d64w_attn_fwd_c1(SK1D64_ARGS) {
  sk1d64_body<128, 8, true>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale,
                            q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride);
}

// ---- BM=256, 16 warps, waves_per_eu(8,8) -- the larger-query-tile arm ----
extern "C" __global__ void __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8, 8)))
sk1d64m_attn_fwd_c0(SK1D64_ARGS) {
  sk1d64_body<256, 16, false>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale,
                              q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride);
}

extern "C" __global__ void __launch_bounds__(512) __attribute__((amdgpu_waves_per_eu(8, 8)))
sk1d64m_attn_fwd_c1(SK1D64_ARGS) {
  sk1d64_body<256, 16, true>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale,
                             q_rstride, q_hstride, q_bstride, o_rstride, o_hstride, o_bstride);
}
