// SK1: hand-written fp8 attention forward for gfx1201 (RDNA4), wave32.
// Two points of the design that differ from what one would first assume:
//   * the WMMA B operand is K-major (lane t holds B[8*lh+e][lr]), so Q^T's B fragment is 8
//     CONTIGUOUS head-dims of one query (no gather, per-token Q scale via one shuffle), and
//   * V must be KEY-contiguous for the PV B fragment, so V arrives transposed: V8T (B*H, 128, N_pad).
// These layouts are exactly what sk1_layout_probe.hip checks. Run it first.
//
// Structure (per CTA: 128 queries, 8 waves x 16 queries; per iteration: 64 keys = 2 sub-tiles of 32):
//   S^T = K * Q^T  -> the score D fragment is lane t: S[query lr][key 8*lh+e], which is exactly the
//   PV A fragment, so P is packed to fp8 in registers and never touches LDS.
//   K and V^T are staged through double-buffered LDS with ONE barrier per iteration: the next tile's
//   global loads are issued before compute and written to the other stage after it.
//
// Inputs (HND, B*H = gridDim.z * gridDim.y):
//   Q    (B*H, N, 128)      fp16, raw (quantised to e4m3 in-kernel, per-token scale)
//   K8   (B*H, N_pad, 128)  e4m3, zero-padded past N
//   SK   (B*H, N_pad)       fp32 per-token K scale (padding value irrelevant: K8 is 0 there)
//   V8T  (B*H, 128, N_pad)  e4m3, V transposed, zero-padded past N
//   SV   (B*H, N_pad)       fp32 per-token V scale
//   VMEAN(B*H, 128)         fp32 or nullptr (smooth_v: added back in the epilogue)
//   O    (B*H, N, 128)      fp16
// N_pad must be a multiple of 64. grid = (ceil(N/128), H, B), block = 256.
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef int      i32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x4 __attribute__((ext_vector_type(4)));
typedef float    f32x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));

#define BM 128
#define BN 64
#define HD 128
#define KSTRIDE 152  // staged K row: 128 B + 24 pad. Row r of a phase hits banks 6r mod 32: conflict-free
#define VSTRIDE 72   // staged V^T row: 64 B + 8 pad. Row r hits banks 18r mod 32: conflict-free
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

template <bool CAUSAL>
__device__ __forceinline__ void sk1_body(const _Float16* __restrict__ Q, const uint8_t* __restrict__ K8,
                                         const float* __restrict__ SK, const uint8_t* __restrict__ V8T,
                                         const float* __restrict__ SV, const float* __restrict__ VMEAN,
                                         _Float16* __restrict__ O, int H, int N, int N_pad, float sm_scale) {
  __shared__ __attribute__((aligned(16))) uint8_t sK[2][BN * KSTRIDE];
  __shared__ __attribute__((aligned(16))) uint8_t sV[2][HD * VSTRIDE];
  __shared__ __attribute__((aligned(16))) float sS[2][BN];
  __shared__ __attribute__((aligned(16))) float sSV[2][BN];

  const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5, lr = lane & 15, lh = lane >> 4;
  const int start_m = blockIdx.x;
  const size_t bh = (size_t)blockIdx.z * H + blockIdx.y;
  const int q0 = start_m * BM + w * 16;  // this wave's first query
  const int qrow = q0 + lr;              // the query this lane's score fragment belongs to
  const int n_tiles = N_pad / BN;
  // F062: the causal bound is clamped to the padded key range.
  const int j_hi = CAUSAL ? min(n_tiles, 2 * start_m + 2) : n_tiles;
  const int full = CAUSAL ? 2 * start_m : n_tiles;  // tiles entirely below the diagonal need no mask

  // ---- Q: raw fp16 -> e4m3 with a per-token scale; resident as 8 B fragments (one per 16 dims) ----
  // B = Q^T is K-major: lane t holds Q[qrow][16*dt + 8*lh + e], 8 contiguous halves (16 B).
  f16x8 qh[8];
  float amax = 0.f;
  const _Float16* qp = Q + (bh * N + qrow) * HD + 8 * lh;
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    if (qrow < N) {
      qh[dt] = *(const f16x8*)(qp + 16 * dt);
    } else {
#pragma unroll
      for (int e = 0; e < 8; ++e) qh[dt][e] = (_Float16)0.f;
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) amax = fmaxf(amax, fabsf((float)qh[dt][e]));
  }
  amax = fmaxf(amax, __shfl_xor(amax, 16, 32));  // lanes t and t^16 hold the two halves of the row
  amax = fmaxf(amax, 1e-12f);                     // shipped convention: floor 1e-12, not 1.0
  const float qinv = P_SCALE / amax;              // 1/qs with qs = amax/448
  const float c_q = sm_scale * LOG2E * (amax / P_SCALE);  // qk_scale * qs, exp2 domain
  i32x2 qf[8];
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    float v[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) v[e] = clamp448((float)qh[dt][e] * qinv);
    qf[dt] = (i32x2){pk4(v[0], v[1], v[2], v[3]), pk4(v[4], v[5], v[6], v[7])};
  }

  // ---- staging: 256 threads move one 64-key tile of K (64x128 B) and V^T (128x64 B) ----
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
      sr = *(const f32x4*)(SV + bh * N_pad + (size_t)j * BN + 4 * (tid - 16));
  };
  auto lstore = [&](int s) {
    uint8_t* kd = &sK[s][(tid >> 2) * KSTRIDE + (tid & 3) * 32];  // 8 B aligned (152 = 8*19)
    *(u32x2*)(kd + 0) = kr0.xy;
    *(u32x2*)(kd + 8) = kr0.zw;
    *(u32x2*)(kd + 16) = kr1.xy;
    *(u32x2*)(kd + 24) = kr1.zw;
    uint8_t* vd = &sV[s][(tid >> 1) * VSTRIDE + (tid & 1) * 32];  // 8 B aligned (72 = 8*9)
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
    if (more) gload(j + 1);  // in flight during compute

#pragma unroll
    for (int st = 0; st < 2; ++st) {
      const int base = j * BN + st * 32;  // first key of this 32-key sub-tile
      bool skip, need_mask;
      if (CAUSAL) {
        skip = base > q0 + 15;  // every key is above every query of this wave
        need_mask = (j >= full);
      } else {
        skip = base >= N;       // fully padded sub-tile: exact no-op, skip it
        need_mask = (base + 32 > N);
      }
      if (skip) continue;  // wave-uniform, and there is no barrier inside the sub-tile

      // QK: S^T = K * Q^T for two 16-key tiles. A = K frag: lane t holds K[key lr][16*dt+8*lh+e].
      f32x8 sc[2];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int e = 0; e < 8; ++e) sc[kk][e] = 0.f;
#pragma unroll
      for (int dt = 0; dt < 8; ++dt)
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
          const i32x2 a = *(const i32x2*)&sK[s][(st * 32 + kk * 16 + lr) * KSTRIDE + dt * 16 + 8 * lh];
          sc[kk] = wmma(a, qf[dt], sc[kk]);
        }

      // scale (qk_scale * qs * sk[key]), mask, row max. sc[kk][e] = S[qrow][base + 16*kk + 8*lh + e].
      float lmax = -INFINITY;
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const f32x4 k0 = *(const f32x4*)&sS[s][st * 32 + kk * 16 + 8 * lh];
        const f32x4 k1 = *(const f32x4*)&sS[s][st * 32 + kk * 16 + 8 * lh + 4];
        const float skv[8] = {k0.x, k0.y, k0.z, k0.w, k1.x, k1.y, k1.z, k1.w};
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          float x = sc[kk][e] * (c_q * skv[e]);
          if (need_mask) {
            const int key = base + kk * 16 + 8 * lh + e;
            if (CAUSAL ? (key > qrow) : (key >= N)) x = -INFINITY;
          }
          sc[kk][e] = x;
          lmax = fmaxf(lmax, x);
        }
      }
      lmax = fmaxf(lmax, __shfl_xor(lmax, 16, 32));  // t and t^16: same query, other 8 keys
      const float m_new = fmaxf(m_i, lmax);
      const float m_use = (m_new == -INFINITY) ? 0.f : m_new;  // a fully masked row stays exact
      const float alpha = __builtin_amdgcn_exp2f(m_i - m_use);
      m_i = m_new;

      // P = exp2(x - m): row sum BEFORE the sv fold, then pack e4m3(p * sv * 448) in place.
      float psum = 0.f;
      i32x2 pa[2];
#pragma unroll
      for (int kk = 0; kk < 2; ++kk) {
        const f32x4 v0 = *(const f32x4*)&sSV[s][st * 32 + kk * 16 + 8 * lh];
        const f32x4 v1 = *(const f32x4*)&sSV[s][st * 32 + kk * 16 + 8 * lh + 4];
        const float svv[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        float p[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          p[e] = __builtin_amdgcn_exp2f(sc[kk][e] - m_use);
          psum += p[e];
          p[e] *= svv[e] * P_SCALE;
        }
        pa[kk] = (i32x2){pk4(p[0], p[1], p[2], p[3]), pk4(p[4], p[5], p[6], p[7])};
      }
      psum += __shfl_xor(psum, 16, 32);
      l_i = l_i * alpha + psum;

      // Rescale acc. acc's D layout puts rows 8*lh+e in this lane; fetch their alphas.
      float ar[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) ar[e] = __shfl(alpha, 8 * lh + e, 32);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt)
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[nt][e] *= ar[e];

      // PV: B = V frag, K-major: lane t holds V[key 16*kk+8*lh+e][dim 16*nt+lr], contiguous in V^T.
#pragma unroll
      for (int kk = 0; kk < 2; ++kk)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
          const i32x2 b = *(const i32x2*)&sV[s][(nt * 16 + lr) * VSTRIDE + st * 32 + kk * 16 + 8 * lh];
          acc[nt] = wmma(pa[kk], b, acc[nt]);
        }
    }

    if (more) lstore(s ^ 1);  // stage s^1 was last read before the previous barrier
    __syncthreads();
  }

  // ---- epilogue: out = acc / (l * 448) (+ vmean). acc row 8*lh+e, head-dim 16*nt+lr ----
  float inv[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const float l = __shfl(l_i, 8 * lh + e, 32);
    inv[e] = l > 0.f ? 1.f / (l * P_SCALE) : 0.f;
  }
#pragma unroll
  for (int nt = 0; nt < 8; ++nt) {
    const int d = nt * 16 + lr;
    const float vm = VMEAN ? VMEAN[bh * HD + d] : 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int row = q0 + 8 * lh + e;
      if (row < N) O[(bh * N + row) * HD + d] = (_Float16)(acc[nt][e] * inv[e] + vm);
    }
  }
}

#define SK1_ARGS                                                                                  \
  const _Float16 *__restrict__ Q, const uint8_t *__restrict__ K8, const float *__restrict__ SK,   \
      const uint8_t *__restrict__ V8T, const float *__restrict__ SV, const float *__restrict__ VMEAN, \
      _Float16 *__restrict__ O, int H, int N, int N_pad, float sm_scale

extern "C" __global__ void __launch_bounds__(256) sk1_attn_fwd_c0(SK1_ARGS) {
  sk1_body<false>(Q, K8, SK, V8T, SV, VMEAN, O, H, N, N_pad, sm_scale);
}

extern "C" __global__ void __launch_bounds__(256) sk1_attn_fwd_c1(SK1_ARGS) {
  sk1_body<true>(Q, K8, SK, V8T, SV, VMEAN, O, H, N, N_pad, sm_scale);
}
