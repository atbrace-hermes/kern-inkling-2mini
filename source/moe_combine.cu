// SPDX-License-Identifier: Apache-2.0
// SM120 MoE top-8 combine kernels for Inkling-2-Mini (H=2304, TOPK=8):
//   - moe_combine_nvfp4: scales FP32 w2 output by x_amax * w2_amax / (1536 * w2_denom),
//                        rounds each routed row to BF16, sums in FP32 over k=0..7, writes BF16
//   - moe_combine_bf16: unweighted FP32 sum of BF16 w2 output over k=0..7, writes BF16

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {
constexpr int kHidden = 2304;
constexpr int kTopK = 8;
}  // namespace

// Grid: (T, 1, 1), Block: (288, 1, 1) — 288 * 8 = 2304 columns.
extern "C" __global__ void __launch_bounds__(288) moe_combine_nvfp4(
    const float* __restrict__ down_f32,
    const int* __restrict__ src2dst,
    const float* __restrict__ x_amax,
    const float* __restrict__ w2_amax,
    const float* __restrict__ w2_denom,
    const int* __restrict__ sorted_ids,
    __nv_bfloat16* __restrict__ out,
    int T) {
  int const t = blockIdx.x;
  if (t >= T) return;

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  __shared__ int s_dst[kTopK];
  __shared__ float s_scale[kTopK];

  float const rcp = __frcp_rn(__fmul_rn(1536.0f, *w2_denom));
  if (threadIdx.x < kTopK) {
    int d = src2dst[int64_t(t) * kTopK + threadIdx.x];
    s_dst[threadIdx.x] = d;
    s_scale[threadIdx.x] =
        (d >= 0) ? __fmul_rn(__fmul_rn(x_amax[d], w2_amax[sorted_ids[d]]), rcp) : 0.0f;
  }
  __syncthreads();

  int const col = threadIdx.x * 8;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    int d = s_dst[k];
    if (d >= 0) {
      float s = s_scale[k];
      const float4* p = reinterpret_cast<const float4*>(down_f32 + int64_t(d) * kHidden + col);
      float4 v0 = p[0];
      float4 v1 = p[1];
      acc[0] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v0.x, s)));
      acc[1] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v0.y, s)));
      acc[2] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v0.z, s)));
      acc[3] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v0.w, s)));
      acc[4] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v1.x, s)));
      acc[5] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v1.y, s)));
      acc[6] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v1.z, s)));
      acc[7] += __bfloat162float(__float2bfloat16_rn(__fmul_rn(v1.w, s)));
    }
  }

  __nv_bfloat16 out8[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) out8[i] = __float2bfloat16_rn(acc[i]);
  *reinterpret_cast<uint4*>(out + int64_t(t) * kHidden + col) =
      *reinterpret_cast<const uint4*>(out8);

#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Grid: (T, 1, 1), Block: (288, 1, 1)
extern "C" __global__ void __launch_bounds__(288) moe_combine_bf16(
    const __nv_bfloat16* __restrict__ down_bf16,
    const int* __restrict__ src2dst,
    __nv_bfloat16* __restrict__ out,
    int T) {
  int const t = blockIdx.x;
  if (t >= T) return;

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  __shared__ int s_dst[kTopK];
  if (threadIdx.x < kTopK) {
    s_dst[threadIdx.x] = src2dst[int64_t(t) * kTopK + threadIdx.x];
  }
  __syncthreads();

  int const col = threadIdx.x * 8;
  float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    int d = s_dst[k];
    if (d >= 0) {
      uint4 raw = *reinterpret_cast<const uint4*>(down_bf16 + int64_t(d) * kHidden + col);
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += __bfloat162float(h[i]);
    }
  }

  __nv_bfloat16 out8[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) out8[i] = __float2bfloat16_rn(acc[i]);
  *reinterpret_cast<uint4*>(out + int64_t(t) * kHidden + col) =
      *reinterpret_cast<const uint4*>(out8);

#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
