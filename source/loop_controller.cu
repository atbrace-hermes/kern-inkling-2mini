// SPDX-License-Identifier: Apache-2.0
// SM120 Rank-64 Gated Loop Controller kernels for Inkling-2-Mini.
// Compiled with --fmad=false AND explicit __fsub_rn / __fmul_rn / __fadd_rn
// to strictly prevent FP32 FMA contraction during gated merge.
//
// 100% bit-identical single-pass register-cached implementation (D=2304, block=256, 0 spills):
// - kern_ctrl_rms_concat_2stream_f32_bf16 (T <= 64): grid [T, 2, 1], 24 regs, 2x CTAs saturate SMs.
// - kern_ctrl_rms_concat_f32_bf16         (T >= 65 & standalone ABI): single-pass 18-reg cache (1.67x less DRAM).
// - kern_ctrl_merge_rmsnorm_f32_bf16      (all T): single-pass 9-reg merged cache + streaming fg loads (1.89x less DRAM).

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kD = 2304;
constexpr int kMaxWarps = 32;

__device__ __forceinline__ float rsqrt_approx_f32(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = __fadd_rn(val, __shfl_xor_sync(0xffffffffu, val, offset));
  }
  return val;
}

}  // namespace

// 2-stream single-pass weightless RMSNorm + concat -> BF16 for small T (T <= 64).
// Grid: [T, 2, 1], Block: [256, 1, 1] (exact d = tid + i*256 reduction order, 0 spills).
extern "C" __global__ __launch_bounds__(256, 4) void kern_ctrl_rms_concat_2stream_f32_bf16(
    const float* __restrict__ in_f32,
    const float* __restrict__ msg_f32,
    __nv_bfloat16* __restrict__ cat_bf16,
    int T,
    float eps) {
  const int t = blockIdx.x;
  const int stream = blockIdx.y;
  if (t >= T) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const float* src = (stream == 0 ? in_f32 : msg_f32) + static_cast<long long>(t) * kD;
  float x[9];
  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    float v = src[tid + i * 256];
    x[i] = v;
    sq = __fadd_rn(sq, __fmul_rn(v, v));
  }
  sq = warp_reduce_sum(sq);

  __shared__ float s_red[8];
  __shared__ float s_rstd;
  if (lane == 0) s_red[warp] = sq;
  __syncthreads();

  if (warp == 0) {
    float v = warp_reduce_sum((lane < 8) ? s_red[lane] : 0.0f);
    if (lane == 0) {
      s_rstd = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v, 1.0f / static_cast<float>(kD)), eps));
    }
  }
  __syncthreads();

  const float rstd = s_rstd;
  __nv_bfloat16* dst = cat_bf16 + static_cast<long long>(t) * (2 * kD) + stream * kD;
#pragma unroll
  for (int i = 0; i < 9; ++i) {
    dst[tid + i * 256] = __float2bfloat16_rn(__fmul_rn(x[i], rstd));
  }
}

// Weightless RMSNorm on FP32 dest (`in_f32`) and FP32 msg (`msg_f32`), rounded
// to BF16 and concatenated into `cat_bf16 [T, 2 * D]`.
// Grid: [T, 1, 1], Block: [256, 1, 1]
extern "C" __global__ __launch_bounds__(256, 4) void kern_ctrl_rms_concat_f32_bf16(
    const float* __restrict__ in_f32,
    const float* __restrict__ msg_f32,
    __nv_bfloat16* __restrict__ cat_bf16,
    int T,
    int D,
    float eps) {
  const int t = blockIdx.x;
  if (t >= T) return;

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const float* in_row = in_f32 + static_cast<long long>(t) * D;
  const float* msg_row = msg_f32 + static_cast<long long>(t) * D;
  __nv_bfloat16* cat_row = cat_bf16 + static_cast<long long>(t) * (2 * D);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (D == kD && blockDim.x == 256) {
    // Single-pass register-cached: exact d = tid + i*256 order, 0 spills, 1.67x less DRAM.
    float rx[9], rm[9];
    float sq0 = 0.0f, sq1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      int d = tid + i * 256;
      float a = in_row[d];
      float b = msg_row[d];
      rx[i] = a;
      rm[i] = b;
      sq0 = __fadd_rn(sq0, __fmul_rn(a, a));
      sq1 = __fadd_rn(sq1, __fmul_rn(b, b));
    }
    __shared__ float s_red8[8][2];
    sq0 = warp_reduce_sum(sq0);
    sq1 = warp_reduce_sum(sq1);
    if (lane == 0) {
      s_red8[warp][0] = sq0;
      s_red8[warp][1] = sq1;
    }
    __syncthreads();
    if (warp == 0) {
      float v0 = warp_reduce_sum((lane < 8) ? s_red8[lane][0] : 0.0f);
      float v1 = warp_reduce_sum((lane < 8) ? s_red8[lane][1] : 0.0f);
      if (lane == 0) {
        constexpr float kInvD = 1.0f / static_cast<float>(kD);
        s_red8[0][0] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v0, kInvD), eps));
        s_red8[0][1] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v1, kInvD), eps));
      }
    }
    __syncthreads();
    const float rstd_in = s_red8[0][0];
    const float rstd_msg = s_red8[0][1];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      int d = tid + i * 256;
      cat_row[d] = __float2bfloat16_rn(__fmul_rn(rx[i], rstd_in));
      cat_row[kD + d] = __float2bfloat16_rn(__fmul_rn(rm[i], rstd_msg));
    }
    return;
  }

  __shared__ float s_red[kMaxWarps][2];
  const int num_warps = (blockDim.x + 31) >> 5;

  float sq0 = 0.0f, sq1 = 0.0f;
  for (int d = tid; d < D; d += blockDim.x) {
    float a = in_row[d];
    float b = msg_row[d];
    sq0 = __fadd_rn(sq0, __fmul_rn(a, a));
    sq1 = __fadd_rn(sq1, __fmul_rn(b, b));
  }
  sq0 = warp_reduce_sum(sq0);
  sq1 = warp_reduce_sum(sq1);
  if (lane == 0) {
    s_red[warp][0] = sq0;
    s_red[warp][1] = sq1;
  }
  __syncthreads();
  if (warp == 0) {
    float v0 = warp_reduce_sum((lane < num_warps) ? s_red[lane][0] : 0.0f);
    float v1 = warp_reduce_sum((lane < num_warps) ? s_red[lane][1] : 0.0f);
    if (lane == 0) {
      s_red[0][0] = v0;
      s_red[0][1] = v1;
    }
  }
  __syncthreads();

  const float inv_d = 1.0f / static_cast<float>(D);
  const float rstd_in = rsqrt_approx_f32(__fadd_rn(__fmul_rn(s_red[0][0], inv_d), eps));
  const float rstd_msg = rsqrt_approx_f32(__fadd_rn(__fmul_rn(s_red[0][1], inv_d), eps));
  for (int d = tid; d < D; d += blockDim.x) {
    cat_row[d] = __float2bfloat16_rn(__fmul_rn(in_row[d], rstd_in));
    cat_row[D + d] = __float2bfloat16_rn(__fmul_rn(msg_row[d], rstd_msg));
  }
}

// Strict no-FMA FP32 gated merge + fused weighted sublayer RMSNorm -> BF16.
// fg_f32 is [T, 2 * D] with forget_pre in [0..D) and gain_pre in [D..2*D).
// Grid: [T, 1, 1], Block: [256, 1, 1]
extern "C" __global__ __launch_bounds__(256, 4) void kern_ctrl_merge_rmsnorm_f32_bf16(
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

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const float* in_row = in_f32 + static_cast<long long>(t) * D;
  const float* msg_row = msg_f32 + static_cast<long long>(t) * D;
  const float* fg_row = fg_f32 + static_cast<long long>(t) * (2 * D);
  __nv_bfloat16* out_row = out_bf16 + static_cast<long long>(t) * D;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (D == kD && blockDim.x == 256) {
    // Single-pass register-cached: computes tanh & strict no-FMA merge ONCE, 0 spills, 1.89x less DRAM.
    float merged[9];
    float sq = 0.0f;
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      int d = tid + i * 256;
      float wf = tanhf(__ldcs(&fg_row[d]));
      float wg = tanhf(__ldcs(&fg_row[kD + d]));
      float m = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, wf), in_row[d]), __fmul_rn(wg, msg_row[d]));
      merged[i] = m;
      sq = __fadd_rn(sq, __fmul_rn(m, m));
    }
    __shared__ float s_red8[8];
    sq = warp_reduce_sum(sq);
    if (lane == 0) s_red8[warp] = sq;
    __syncthreads();
    if (warp == 0) {
      float v = warp_reduce_sum((lane < 8) ? s_red8[lane] : 0.0f);
      if (lane == 0) {
        s_red8[0] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v, 1.0f / static_cast<float>(kD)), eps));
      }
    }
    __syncthreads();
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    const float rstd = s_red8[0];
#pragma unroll
    for (int i = 0; i < 9; ++i) {
      int d = tid + i * 256;
      float w = __bfloat162float(__ldg(&norm_w[d]));
      out_row[d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(merged[i], rstd), w));
    }
    return;
  }

  __shared__ float s_red[kMaxWarps];
  const int num_warps = (blockDim.x + 31) >> 5;

  float sq = 0.0f;
  for (int d = tid; d < D; d += blockDim.x) {
    float wf = tanhf(fg_row[d]);
    float wg = tanhf(fg_row[D + d]);
    float merged = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, wf), in_row[d]), __fmul_rn(wg, msg_row[d]));
    sq = __fadd_rn(sq, __fmul_rn(merged, merged));
  }
  sq = warp_reduce_sum(sq);
  if (lane == 0) s_red[warp] = sq;
  __syncthreads();
  if (warp == 0) {
    float v = warp_reduce_sum((lane < num_warps) ? s_red[lane] : 0.0f);
    if (lane == 0) s_red[0] = v;
  }
  __syncthreads();

  const float rstd = rsqrt_approx_f32(__fadd_rn(__fmul_rn(s_red[0], 1.0f / static_cast<float>(D)), eps));
  for (int d = tid; d < D; d += blockDim.x) {
    float wf = tanhf(fg_row[d]);
    float wg = tanhf(fg_row[D + d]);
    float merged = __fadd_rn(__fmul_rn(__fsub_rn(1.0f, wf), in_row[d]), __fmul_rn(wg, msg_row[d]));
    float w = __bfloat162float(norm_w[d]);
    out_row[d] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(merged, rstd), w));
  }
}
