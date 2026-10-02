// SK1 T4a1sb: T4a1s with the Q/O element type as a template parameter, so the same kernel source
// serves fp16 and bf16.  Derived from sk1_t4a1s.hip by literal string substitution.
//
// bf16 and fp16 are both 16 bits wide, so the Q vector load, every address and the whole KV loop
// are unchanged.  The only two dtype-dependent operations are the Q widen `(float)qh[..]` and the O
// narrow `(T)(acc*inv + vm)`; K, V and P are e4m3 and every scale is fp32, so the quantisation
// arithmetic is the same text for both.  clang emits the same `global_load_b128` for a bf16 vector
// load as for fp16, widens bf16 to fp32 by bit placement (no `v_cvt`), and narrows fp32 to bf16
// with a 5-VALU round-to-nearest-even sequence plus `global_store_d16_hi_b16`.
//
// Four entry points: `sk1t4a1sp_attn_fwd_c0/_c1` are the `_Float16` instantiation, kept so its code
// can be compared against a build of the unmodified sk1_t4a1s.hip, and
// `sk1t4a1sb_attn_fwd_c0/_c1` are `__bf16`.  The per-(b,h) `S` rescale and the three-phase KV loop
// are described in sk1_t4a1s.hip.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int      i32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef __bf16   bf16x8 __attribute__((ext_vector_type(8)));

// The Q/O element type.  bf16 and fp16 are both 16 bits wide, so the vector load and every
// address are unchanged; only the widen (T -> f32) and the narrow (f32 -> T) differ.
template <typename T> struct QOVec;
template <> struct QOVec<_Float16> { typedef f16x8 type; };
template <> struct QOVec<__bf16>   { typedef bf16x8 type; };

#define BM 128
#define BN 64
#define HD 128
#define KSTRIDE 152
#define VSTRIDE 72
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

template <bool CAUSAL, typename T>
__device__ __forceinline__ void sk1t1_body(const T* __restrict__ Q, const uint8_t* __restrict__ K8,
                                           const float* __restrict__ SK, const uint8_t* __restrict__ V8T,
                                           const float* __restrict__ SV, const float* __restrict__ VMEAN,
                                           const float* __restrict__ S,
                                           T* __restrict__ O, int H, int N, int N_pad, float sm_scale) {
  __shared__ __attribute__((aligned(16))) uint8_t sK[2][BN * KSTRIDE];
  __shared__ __attribute__((aligned(16))) uint8_t sV[2][HD * VSTRIDE];
  __shared__ __attribute__((aligned(16))) float sS[2][BN];
  __shared__ __attribute__((aligned(16))) float sSV[2][BN];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, lr = lane & 15, lh = lane >> 4;
  const int start_m = blockIdx.x;
  const size_t bh = (size_t)blockIdx.z * H + blockIdx.y;
  // The per-(b,h) rescale the prologue applied to SV.  CTA-uniform, one load.
  const float Sb = S[bh];
  const int q0 = start_m * BM + w * 16;
  const int qrow = q0 + lr;
  const int n_tiles = N_pad / BN;
  const int j_hi = CAUSAL ? min(n_tiles, 2 * start_m + 2) : n_tiles;
  const int full = CAUSAL ? 2 * start_m : n_tiles;

  // ---- Q: raw T -> e4m3, per-token scale, 8 B fragments ----
  typedef typename QOVec<T>::type qovec;
  qovec qh[8];
  float amax = 0.f;
  const T* qp = Q + (bh * N + qrow) * HD + 8 * lh;
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    if (qrow < N) {
      qh[dt] = *(const qovec*)(qp + 16 * dt);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) qh[dt][e] = (T)0.f;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf((float)qh[dt][e]));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
  amax = fmaxf(amax, 1e-12f);
  const float qinv = P_SCALE / amax;
  const float c_q = sm_scale * LOG2E * (amax / P_SCALE);
  i32x2 qf[8];
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = clamp448((float)qh[dt][e] * qinv);
    qf[dt] = (i32x2){pk4(v[0], v[1], v[2], v[3]), pk4(v[4], v[5], v[6], v[7])};
  }

  // ---- staging ----
  u32x4 kr0, kr1, vr0, vr1;
  f32x4 sr = {0.f, 0.f, 0.f, 0.f};
  auto gload = [&](int j) {
    const uint8_t* kp = K8 + (bh * N_pad + (size_t)j * BN + (tid >> 2)) * HD + (tid & 3) * 32;
    kr0 = *(const u32x4*)kp;
    kr1 = *(const u32x4*)(kp + 16);
    const uint8_t* vp = V8T + (bh * HD + (tid >> 1)) * N_pad + (size_t)j * BN + (tid & 1) * 32;
    vr0 = *(const u32x4*)vp;
    vr1 = *(const u32x4*)(vp + 16);
    if (tid < 16)
      sr = *(const f32x4*)(SK + bh * N_pad + (size_t)j * BN + 4 * tid);
    else if (tid < 32)
      sr = *(const f32x4*)(SV + bh * N_pad + (size_t)j * BN + 4 * (tid - 16)) * P_SCALE;
  };
  auto lstore = [&](int s) {
    uint8_t* kd = &sK[s][(tid >> 2) * KSTRIDE + (tid & 3) * 32];
    *(u32x2*)(kd + 0) = kr0.xy;
    *(u32x2*)(kd + 8) = kr0.zw;
    *(u32x2*)(kd + 16) = kr1.xy;
    *(u32x2*)(kd + 24) = kr1.zw;
    uint8_t* vd = &sV[s][(tid >> 1) * VSTRIDE + (tid & 1) * 32];
    *(u32x2*)(vd + 0) = vr0.xy;
    *(u32x2*)(vd + 8) = vr0.zw;
    *(u32x2*)(vd + 16) = vr1.xy;
    *(u32x2*)(vd + 24) = vr1.zw;
    if (tid < 16)
      *(f32x4*)&sS[s][4 * tid] = sr;
    else if (tid < 32)
      *(f32x4*)&sSV[s][4 * (tid - 16)] = sr;
  };

  f32x8 acc[8];
#pragma unroll
  for (int nt = 0; nt < 8; ++nt)
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
      for (int dt = 0; dt < 8; ++dt)
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
      // `need_mask` is false (137 of 138 iterations at N=8771 c0).  Bit-identical: the same
      // products are formed and the same elements are set to -INFINITY, and `fmaxf` is exact.
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
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[nt][e] *= ar[e];

    // ---------------- phase 3: PV ----------------
#pragma unroll
    for (int st = 0; st < 2; ++st) {
      if (skp[st]) continue;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
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
  for (int nt = 0; nt < 8; ++nt) {
    const int d = nt * 16 + lr;
    const float vm = VMEAN ? VMEAN[bh * HD + d] : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = q0 + 8 * lh + e;
      if (row < N) O[(bh * N + row) * HD + d] = (T)(acc[nt][e] * inv[e] + vm);
    }
  }
}

#define SK1T1_ARGS(T)                                                                             \
  const T *__restrict__ Q, const uint8_t *__restrict__ K8, const float *__restrict__ SK,          \
      const uint8_t *__restrict__ V8T, const float *__restrict__ SV, const float *__restrict__ VMEAN, \
      const float *__restrict__ S, \
      T *__restrict__ O, int H, int N, int N_pad, float sm_scale

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6, 6)))
sk1t4a1sp_attn_fwd_c0(SK1T1_ARGS(_Float16)) {
  sk1t1_body<false, _Float16>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6, 6)))
sk1t4a1sp_attn_fwd_c1(SK1T1_ARGS(_Float16)) {
  sk1t1_body<true, _Float16>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6, 6)))
sk1t4a1sb_attn_fwd_c0(SK1T1_ARGS(__bf16)) {
  sk1t1_body<false, __bf16>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale);
}

extern "C" __global__ void __launch_bounds__(256) __attribute__((amdgpu_waves_per_eu(6, 6)))
sk1t4a1sb_attn_fwd_c1(SK1T1_ARGS(__bf16)) {
  sk1t1_body<true, __bf16>(Q, K8, SK, V8T, SV, VMEAN, S, O, H, N, N_pad, sm_scale);
}
