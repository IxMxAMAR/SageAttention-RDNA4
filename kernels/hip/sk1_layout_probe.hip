// Layout probe for SK1: pins the gfx1201 fp8 WMMA register layouts on the device.
// Launch ONE wave (grid 1, block 32). A, B, D are 16x16 row-major fp32 in global memory.
// Fill A and B with small integers in [-4, 4] (exact in e4m3); the host checks D == A @ B.
//   mode 0 (SK1's assumption): lane t, lr = t%16, lh = t/16, element e:
//            A frag = A[lr][8*lh+e],  B frag = B[8*lh+e][lr],  D frag = D[8*lh+e][lr]
//   mode 1 (the alternative layout): B frag = B[lr][8*lh+e]
// If mode 0 matches exactly, A, B, D and the fp8 byte order inside a fragment are all confirmed.
#include <hip/hip_runtime.h>

typedef int   i32x2 __attribute__((ext_vector_type(2)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

__device__ __forceinline__ int pk4(float a, float b, float c, float d) {
  int r = __builtin_amdgcn_cvt_pk_fp8_f32(a, b, 0, false);
  return __builtin_amdgcn_cvt_pk_fp8_f32(c, d, r, true);
}

extern "C" __global__ void __launch_bounds__(32)
sk1_layout_probe(const float* A, const float* B, float* D, int mode) {
  const int t = threadIdx.x, lr = t & 15, lh = t >> 4;
  float a[8], b[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    a[e] = A[lr * 16 + 8 * lh + e];
    b[e] = mode == 0 ? B[(8 * lh + e) * 16 + lr] : B[lr * 16 + 8 * lh + e];
  }
  i32x2 af = {pk4(a[0], a[1], a[2], a[3]), pk4(a[4], a[5], a[6], a[7])};
  i32x2 bf = {pk4(b[0], b[1], b[2], b[3]), pk4(b[4], b[5], b[6], b[7])};
  f32x8 d = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  d = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(af, bf, d);
#pragma unroll
  for (int e = 0; e < 8; ++e) D[(8 * lh + e) * 16 + lr] = d[e];
}
