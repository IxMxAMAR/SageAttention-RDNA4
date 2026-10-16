// HIP KV prologue: K/V -> (K8, SK, V8T, SV), replacing the Triton prologue on the default SK1
// paths.
//
// The elementwise math is copied from the Triton prologue it replaces (quant_kv_v8t*.py,
// quant_triton.py):
//   K fp8  : sk = max(max_d|k|/448.0, 1e-12); K8 = e4m3(k/sk)
//   K int8 : sk = max(max_d|k|/127.0, FLOOR); x=k/sk; fl=floor(x); d=x-fl;
//            odd=(int(fl)&1)==1; q=where((d>0.5)|((d==0.5)&odd), fl+1, fl); q=clamp(q,-128,127)
//            -- pure round-half-to-even, NOT the truncating hardware v_cvt_pk_i16_f32
//   V      : sv = max(max_d|v|/448.0, 1e-12); V8T[d,n] = e4m3(v[n,d]/sv[n])
//   pad    : k8/v8t = 0, sk = 1.0, sv = 0.0
//   smooth_k: load -> k -= mu -> k = where(live, k, 0) -> absmax -> sk
//
// Two differences from the Triton form:
//   * all addressing is 64-bit (int64 element offsets), so the prologue has no 2^31-element limit;
//   * the smooth_k mean is accumulated in fp64 and is therefore deterministic and
//     order-independent, where Triton accumulates in fp32. "fp64-accumulated" is the whole claim:
//     widening fp16/bf16 to fp64 is exact, but the sum is not (the error bound is relative to the
//     sum of |x|, so it says nothing when terms cancel). It is not "exact" and not "correctly
//     rounded".
// No Q argument is taken or read: the fp8-contiguous Triton prologue's `quant_rows_fp8(q)` output
// is discarded by the caller.
//
// Symbols (all `extern "C"`):
//   h34_quant_kv_f16_i8 / h34_quant_kv_f16_f8 / h34_quant_kv_bf16_f8
//   h34_kseq_mean_f16 / h34_kseq_mean_bf16
#include <hip/hip_runtime.h>
#include <stdint.h>

typedef unsigned int   u32;
typedef unsigned short u16;
typedef unsigned char  u8;
typedef unsigned long long u64;
typedef _Float16       f16;

#define BN   64        // rows per block, matches the Triton BLOCK_N=64
#define NT   256       // threads per block (64 rows x 4 column lanes)
#define HD   128       // head dim (the default paths that run a prologue are D=128 only)
#define VROW 132       // LDS row stride for the fp8 byte tile (64*132 = 8448 B, conflict-free)

// ---------------------------------------------------------------------------------------------
// fp32 -> e4m3, the same instruction Triton's `_cvt_pk_raw` uses. `_cvt2d` clamps only the upper
// side (`tl.minimum(x,448.0)`); this reproduces exactly that: fminf(x, 448.0f), no lower clamp.
// Round-to-nearest-even is the instruction's own rule.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ u32 pk2(float a, float b) {
  return ((u32)__builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false)) & 0xFFFFu;
}
__device__ __forceinline__ u32 fp8_4(float a, float b, float c, float d) {
  // bytes in memory order a, b, c, d -- built explicitly so the word-select semantics of the
  // intrinsic cannot permute them
  return pk2(a, b) | (pk2(c, d) << 16);
}
__device__ __forceinline__ u8 fp8_1(float x) { return (u8)pk2(x, 0.0f); }

// round-half-to-EVEN (floor + the explicit half-even test), then the [-128,127] clamp
__device__ __forceinline__ u8 i8_rne(float x) {
  float fl = floorf(x);
  float d  = x - fl;
  int   odd = ((int)fl) & 1;
  float q = ((d > 0.5f) || (d == 0.5f && odd != 0)) ? (fl + 1.0f) : fl;
  q = fminf(fmaxf(q, -128.0f), 127.0f);
  return (u8)(signed char)q;               // q is integral; the cast truncates exactly
}

template <typename T> __device__ __forceinline__ float tof(T x);
template <> __device__ __forceinline__ float tof<f16>(f16 x)        { return (float)x; }
template <> __device__ __forceinline__ float tof<__bf16>(__bf16 x)  { return (float)x; }

template <typename T> __device__ __forceinline__ T frombits(u16 b) {
  union { u16 u; T f; } x; x.u = b; return x.f;
}

// 8 elements = one 16-byte chunk.  chunk index = j*4+q, column = chunk*8 + t.
template <typename T>
__device__ __forceinline__ void load8(const T* p, float* out) {
  uint4 u = *(const uint4*)p;
  u16 hh[8];
  *(uint4*)hh = u;
  #pragma unroll
  for (int t = 0; t < 8; t++) out[t] = tof<T>(frombits<T>(hh[t]));
}

// =============================================================================================
// The KV quantiser.  grid = (cdiv(N_pad, BN), B*H), block = NT.
//
// Column mapping: lane q in {0,1,2,3} owns the 8-element chunks j*4+q for j=0..3, i.e. columns
// {j*32 + q*8 + t : t=0..7}.  At a fixed j the four q-lanes read 16 consecutive bytes each, so a
// warp's load is 64 contiguous bytes per row (coalesced).
// =============================================================================================
template <typename T, bool KINT8>
__device__ __forceinline__ void quant_kv_body(
    const T* __restrict__ K, const T* __restrict__ V,
    u8* __restrict__ K8, float* __restrict__ SK,
    u8* __restrict__ V8T, float* __restrict__ SV,
    const float* __restrict__ MEAN,
    long long N, long long N_pad, long long H,
    long long k_sb, long long k_sh, long long k_sr,
    long long v_sb, long long v_sh, long long v_sr,
    float FLOOR, int SUB_MEAN)
{
  __shared__ u8 vb[BN * VROW];

  const int tid = threadIdx.x;
  const int rl  = tid >> 2;                 // row within the block, 0..63
  const int q   = tid & 3;                  // column lane, 0..3
  const long long n = (long long)blockIdx.x * BN + rl;
  const int bh = blockIdx.y;
  const int b = (int)(bh / H);
  const int h = (int)(bh - (long long)b * H);
  const long long kbase = (long long)b * k_sb + (long long)h * k_sh;
  const long long vbase = (long long)b * v_sb + (long long)h * v_sh;
  const bool live = (n < N);
  const u8* kdst_row = K8 + ((long long)bh * N_pad + n) * HD;

  // ---------------- K: load -> (mean) -> re-zero pad -> absmax -> sk -> convert ---------------
  float kv[32];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    float tmp[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) tmp[t] = 0.0f;
    if (live) load8<T>(K + kbase + n * k_sr + (long long)j * 32 + (long long)q * 8, tmp);
    #pragma unroll
    for (int t = 0; t < 8; t++) kv[j * 8 + t] = tmp[t];
  }
  if (SUB_MEAN) {
    // subtract before the absmax, and re-zero the pad rows
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      #pragma unroll
      for (int t = 0; t < 8; t++) {
        const long long col = (long long)j * 32 + q * 8 + t;
        kv[j * 8 + t] = live ? (kv[j * 8 + t] - MEAN[(long long)bh * HD + col]) : 0.0f;
      }
    }
  }
  float kmax = 0.0f;
  #pragma unroll
  for (int t = 0; t < 32; t++) kmax = fmaxf(kmax, fabsf(kv[t]));
  kmax = fmaxf(kmax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, kmax, 1));
  kmax = fmaxf(kmax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, kmax, 2));
  // `tl.maximum(amax/SCALE, FLOOR)` -- correctly-rounded division, same as Triton's `arith.divf`.
  // FLOOR is the caller's floor and applies to the int8 K scale only. Triton hard-codes
  // `EPS = 1e-12` for the fp8-K scale and for V's scale; a caller-supplied `floor` there would
  // silently diverge from Triton on every live row whose scale is below it.
  float sk = fmaxf(__fdiv_rn(kmax, KINT8 ? 127.0f : 448.0f), KINT8 ? FLOOR : 1e-12f);
  if (q == 0) SK[(long long)bh * N_pad + n] = live ? sk : 1.0f;

  #pragma unroll
  for (int j = 0; j < 4; j++) {
    u8* dst = (u8*)kdst_row + (long long)j * 32 + q * 8;
    if (KINT8) {
      u64 w = 0;
      #pragma unroll
      for (int t = 0; t < 8; t++) w |= ((u64)i8_rne(__fdiv_rn(kv[j * 8 + t], sk))) << (8 * t);
      *(u64*)dst = w;
    } else {
      // `_cvt2d(k/sk, ...)`: fminf(x,448) then the hardware pack, 4 bytes at a time.
      float c[8];
      #pragma unroll
      for (int t = 0; t < 8; t++) c[t] = fminf(__fdiv_rn(kv[j * 8 + t], sk), 448.0f);
      *(u32*)(dst)     = fp8_4(c[0], c[1], c[2], c[3]);
      *(u32*)(dst + 4) = fp8_4(c[4], c[5], c[6], c[7]);
    }
  }

  // ---------------- V: load -> re-zero pad -> absmax -> sv -> convert -> LDS -> transpose -------
  float vv[32];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    float tmp[8];
    #pragma unroll
    for (int t = 0; t < 8; t++) tmp[t] = 0.0f;
    if (live) load8<T>(V + vbase + n * v_sr + (long long)j * 32 + (long long)q * 8, tmp);
    #pragma unroll
    for (int t = 0; t < 8; t++) vv[j * 8 + t] = tmp[t];
  }
  float vmax = 0.0f;
  #pragma unroll
  for (int t = 0; t < 32; t++) vmax = fmaxf(vmax, fabsf(vv[t]));
  vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, vmax, 1));
  vmax = fmaxf(vmax, __shfl_xor_sync(0xFFFFFFFFFFFFFFFFull, vmax, 2));
  // Triton hard-codes EPS = 1e-12 for V's scale; FLOOR is int8-K only.
  float sv = fmaxf(__fdiv_rn(vmax, 448.0f), 1e-12f);
  if (q == 0) SV[(long long)bh * N_pad + n] = live ? sv : 0.0f;

  #pragma unroll
  for (int j = 0; j < 4; j++) {
    #pragma unroll
    for (int t = 0; t < 8; t++) {
      float x = fminf(__fdiv_rn(vv[j * 8 + t], sv), 448.0f);
      vb[rl * VROW + j * 32 + q * 8 + t] = fp8_1(x);
    }
  }
  __syncthreads();

  // Transposed write: lane -> (d, half), 32 consecutive n per lane -> 2 x 16 B stores.
  // V8T[bh, d, n] = e4m3(v[bh, n, d] / sv[bh, n])
  {
    const int d    = tid >> 1;
    const int half = tid & 1;
    const int n0   = half * 32;
    u8 tmp[32];
    #pragma unroll
    for (int t = 0; t < 32; t++) tmp[t] = vb[(n0 + t) * VROW + d];
    u8* dst = V8T + (long long)bh * HD * N_pad + (long long)d * N_pad
                  + (long long)blockIdx.x * BN + n0;
    *(uint4*)(dst)      = *(uint4*)(tmp);
    *(uint4*)(dst + 16) = *(uint4*)(tmp + 16);
  }
}

extern "C" __global__ void __launch_bounds__(NT)
h34_quant_kv_f16_i8(const f16* K, const f16* V, u8* K8, float* SK, u8* V8T, float* SV,
                    const float* MEAN, long long N, long long N_pad, long long H,
                    long long k_sb, long long k_sh, long long k_sr,
                    long long v_sb, long long v_sh, long long v_sr,
                    float FLOOR, int SUB_MEAN) {
  quant_kv_body<f16, true>(K, V, K8, SK, V8T, SV, MEAN, N, N_pad, H,
                           k_sb, k_sh, k_sr, v_sb, v_sh, v_sr, FLOOR, SUB_MEAN);
}

extern "C" __global__ void __launch_bounds__(NT)
h34_quant_kv_f16_f8(const f16* K, const f16* V, u8* K8, float* SK, u8* V8T, float* SV,
                    const float* MEAN, long long N, long long N_pad, long long H,
                    long long k_sb, long long k_sh, long long k_sr,
                    long long v_sb, long long v_sh, long long v_sr,
                    float FLOOR, int SUB_MEAN) {
  quant_kv_body<f16, false>(K, V, K8, SK, V8T, SV, MEAN, N, N_pad, H,
                            k_sb, k_sh, k_sr, v_sb, v_sh, v_sr, FLOOR, SUB_MEAN);
}

extern "C" __global__ void __launch_bounds__(NT)
h34_quant_kv_bf16_f8(const __bf16* K, const __bf16* V, u8* K8, float* SK, u8* V8T, float* SV,
                     const float* MEAN, long long N, long long N_pad, long long H,
                     long long k_sb, long long k_sh, long long k_sr,
                     long long v_sb, long long v_sh, long long v_sr,
                     float FLOOR, int SUB_MEAN) {
  quant_kv_body<__bf16, false>(K, V, K8, SK, V8T, SV, MEAN, N, N_pad, H,
                               k_sb, k_sh, k_sr, v_sb, v_sh, v_sr, FLOOR, SUB_MEAN);
}

// =============================================================================================
// smooth_k's sequence-axis mean, (B*H, D) fp32.  grid = (B*H, D/32), block = (32, 8).
//
// The sum is order-independent by construction: every K element is widened to fp64 (exact for fp16
// and bf16), accumulated in an fp64 register, reduced over the 8 y-lanes in fp64, and divided by
// (double)N once before the single fp32 rounding.
//
// This does not make the stored fp32 "the correctly-rounded exact mean", and a relative error bound
// of N*2^-53 does not hold in general. Widening is exact; summing is not. The bound is relative to
// the sum of |x|, so it says nothing when terms cancel -- bf16 [2^100, 1, -2^100] sums to fp64 0 and
// yields mu = 0 where the exact mean is 1/3. The claim is: fp64-accumulated, deterministic, and
// more accurate than Triton's fp32 sum. A finite mu shifts every query's scores by a constant,
// which softmax ignores exactly, so this cannot change the attention output. A reference computed
// as `kk.double().mean` is also an fp64 sum and shares this blind spot.
// =============================================================================================
template <typename T>
__device__ __forceinline__ void kseq_mean_body(const T* __restrict__ K, float* __restrict__ MEAN,
                                               long long N, long long H,
                                               long long k_sb, long long k_sh, long long k_sr) {
  __shared__ double red[8][32];
  const int c  = threadIdx.x;                 // 0..31
  const int y  = threadIdx.y;                 // 0..7
  const int bh = blockIdx.x;
  const long long d = (long long)blockIdx.y * 32 + c;
  const int b = (int)(bh / H);
  const int h = (int)(bh - (long long)b * H);
  const long long base = (long long)b * k_sb + (long long)h * k_sh + d;

  double acc = 0.0;
  for (long long n = y; n < N; n += 8) acc += (double)tof<T>(K[base + n * k_sr]);
  red[y][c] = acc;
  __syncthreads();
  if (y == 0) {
    double s = 0.0;
    #pragma unroll
    for (int i = 0; i < 8; i++) s += red[i][c];
    MEAN[(long long)bh * HD + d] = (float)(s / (double)N);
  }
}

extern "C" __global__ void __launch_bounds__(256)
h34_kseq_mean_f16(const f16* K, float* MEAN, long long N, long long H,
                  long long k_sb, long long k_sh, long long k_sr) {
  kseq_mean_body<f16>(K, MEAN, N, H, k_sb, k_sh, k_sr);
}

extern "C" __global__ void __launch_bounds__(256)
h34_kseq_mean_bf16(const __bf16* K, float* MEAN, long long N, long long H,
                   long long k_sb, long long k_sh, long long k_sr) {
  kseq_mean_body<__bf16>(K, MEAN, N, H, k_sb, k_sh, k_sr);
}
