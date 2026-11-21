// SPDX-License-Identifier: Apache-2.0
// SM120 Rank-64 Gated Loop Controller helpers for Inkling-2-Mini.
// Compiled with --fmad=false AND explicit __fsub_rn / __fmul_rn / __fadd_rn
// to strictly prevent FP32 FMA contraction during gated merge.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kMaxWarps = 32;

__device__ __forceinline__ float rsqrt_approx_f32(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffffu, val, offset);
  }
  return val;
}

template <int K>
__device__ __forceinline__ void block_reduce_sum_vec(float vals[K], float out[K]) {
  __shared__ float smem[kMaxWarps][K];
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int num_warps = (blockDim.x + 31) >> 5;

#pragma unroll
  for (int k = 0; k < K; ++k) {
    vals[k] = warp_reduce_sum(vals[k]);
    if (lane == 0) {
      smem[warp][k] = vals[k];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int k = 0; k < K; ++k) {
      float v = (lane < num_warps) ? smem[lane][k] : 0.0f;
      v = warp_reduce_sum(v);
      if (lane == 0) {
        smem[0][k] = v;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int k = 0; k < K; ++k) {
    out[k] = smem[0][k];
  }
}

}  // namespace

// Weightless RMSNorm on FP32 dest (`in_f32`) and FP32 msg (`msg_f32`), rounded
// to BF16 and concatenated into `cat_bf16 [T, 2 * D]`.
// Grid: [T, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_ctrl_rms_concat_f32_bf16(
    const float* __restrict__ in_f32,
    const float* __restrict__ msg_f32,
    __nv_bfloat16* __restrict__ cat_bf16,
    int T,
    int D,
    float eps) {
  const int t = blockIdx.x;
  if (t >= T) return;

  const float* in_row = in_f32 + static_cast<long long>(t) * D;
  const float* msg_row = msg_f32 + static_cast<long long>(t) * D;
  __nv_bfloat16* cat_row = cat_bf16 + static_cast<long long>(t) * (2 * D);

  float sum_sq[2] = {0.0f, 0.0f};
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float a = in_row[d];
    float b = msg_row[d];
    sum_sq[0] = __fadd_rn(sum_sq[0], __fmul_rn(a, a));
    sum_sq[1] = __fadd_rn(sum_sq[1], __fmul_rn(b, b));
  }

  float red[2];
  block_reduce_sum_vec<2>(sum_sq, red);
  const float inv_d = 1.0f / static_cast<float>(D);
  const float rstd_in = rsqrt_approx_f32(__fadd_rn(__fmul_rn(red[0], inv_d), eps));
  const float rstd_msg = rsqrt_approx_f32(__fadd_rn(__fmul_rn(red[1], inv_d), eps));

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    cat_row[d] = __float2bfloat16_rn(__fmul_rn(in_row[d], rstd_in));
    cat_row[D + d] = __float2bfloat16_rn(__fmul_rn(msg_row[d], rstd_msg));
  }
}

// Strict no-FMA FP32 gated merge + fused weighted sublayer RMSNorm -> BF16.
// fg_f32 is [T, 2 * D] with forget_pre in [0..D) and gain_pre in [D..2*D).
// Grid: [T, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_ctrl_merge_rmsnorm_f32_bf16(
    const float* __restrict__ in_f32,
    const float* __restrict__ msg_f32,
    const float* __restrict__ fg_f32,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out_bf16,
    int T,
    int D,
    float eps) {
  const int t = blockIdx.x;
  if (t >= T) return;

  const float* in_row = in_f32 + static_cast<long long>(t) * D;
  const float* msg_row = msg_f32 + static_cast<long long>(t) * D;
  const float* fg_row = fg_f32 + static_cast<long long>(t) * (2 * D);
  __nv_bfloat16* out_row = out_bf16 + static_cast<long long>(t) * D;

  constexpr int kRegCap = 16;
  float reg_merged[kRegCap];
  float sum_sq[1] = {0.0f};

  int idx = 0;
  for (int d = threadIdx.x; d < D; d += blockDim.x, ++idx) {
    float wf = tanhf(fg_row[d]);
    float wg = tanhf(fg_row[D + d]);
    float keep = __fmul_rn(__fsub_rn(1.0f, wf), in_row[d]);
    float gain = __fmul_rn(wg, msg_row[d]);
    float merged = __fadd_rn(keep, gain);
    if (idx < kRegCap) {
      reg_merged[idx] = merged;
    }
    sum_sq[0] = __fadd_rn(sum_sq[0], __fmul_rn(merged, merged));
  }

  float red[1];
  block_reduce_sum_vec<1>(sum_sq, red);
  const float rstd = rsqrt_approx_f32(__fadd_rn(__fmul_rn(red[0], 1.0f / static_cast<float>(D)), eps));

  idx = 0;
  for (int d = threadIdx.x; d < D; d += blockDim.x, ++idx) {
    float merged;
    if (idx < kRegCap) {
      merged = reg_merged[idx];
    } else {
      float wf = tanhf(fg_row[d]);
      float wg = tanhf(fg_row[D + d]);
      float keep = __fmul_rn(__fsub_rn(1.0f, wf), in_row[d]);
      float gain = __fmul_rn(wg, msg_row[d]);
      merged = __fadd_rn(keep, gain);
    }
    float w = __bfloat162float(norm_w[d]);
    out_row[d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(merged, rstd), w));
  }
}
