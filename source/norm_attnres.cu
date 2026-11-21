// SPDX-License-Identifier: Apache-2.0
// SM120 normalization, Block Attention Residuals (attn_res), Cross-Pass AttnRes,
// and residual/query-scaling helpers for Inkling-2-Mini.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kMaxSources = 16;
constexpr int kMaxQueries = 8;
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

__device__ __forceinline__ float block_reduce_sum_scalar(float val) {
  float in[1] = {val};
  float out[1];
  block_reduce_sum_vec<1>(in, out);
  return out[0];
}

template <bool HAS_ADDEND, bool WRITE_NORMED>
__device__ void attn_res_core(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ queries,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ out_mix,
    __nv_bfloat16* __restrict__ out_normed,
    long long stack_stride,
    long long out_q_stride,
    int T,
    int D,
    int n_stack,
    int num_q,
    float eps,
    float norm_eps,
    bool has_partial) {
  const int t = blockIdx.x;
  if (t >= T) return;

  const int n_sources = n_stack + (has_partial ? 1 : 0);
  if (n_sources <= 0 || n_sources > kMaxSources || num_q <= 0 || num_q > kMaxQueries) {
    __trap();
  }

  __shared__ float s_inv_rms[kMaxSources];
  __shared__ float s_logits[kMaxQueries][kMaxSources];
  __shared__ float s_weights[kMaxQueries][kMaxSources];
  __shared__ float s_out_rstd[kMaxQueries];

  const float inv_d = 1.0f / static_cast<float>(D);
  const long long row_off = static_cast<long long>(t) * D;

  // Pass 1: per-source inverse RMS and dot products with all pre-scaled queries.
  for (int n = 0; n < n_sources; ++n) {
    const __nv_bfloat16* src_n = (n < n_stack)
                                     ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                     : (partial + row_off);
    const bool use_addend = HAS_ADDEND && (n == n_sources - 1);

    float accum[1 + kMaxQueries];
#pragma unroll
    for (int k = 0; k < 1 + kMaxQueries; ++k) accum[k] = 0.0f;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float x = __bfloat162float(src_n[d]);
      if (use_addend) {
        x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
      }
      accum[0] += x * x;
      for (int q = 0; q < num_q; ++q) {
        accum[1 + q] += queries[static_cast<long long>(q) * D + d] * x;
      }
    }

    float reduced[1 + kMaxQueries];
    block_reduce_sum_vec<1 + kMaxQueries>(accum, reduced);

    if (threadIdx.x == 0) {
      float rstd = rsqrt_approx_f32(reduced[0] * inv_d + eps);
      s_inv_rms[n] = rstd;
      for (int q = 0; q < num_q; ++q) {
        s_logits[q][n] = reduced[1 + q] * rstd;
      }
    }
    __syncthreads();
  }

  // Softmax over sources (normalize_values = true).
  if (threadIdx.x == 0) {
    for (int q = 0; q < num_q; ++q) {
      float max_l = s_logits[q][0];
      for (int n = 1; n < n_sources; ++n) {
        max_l = fmaxf(max_l, s_logits[q][n]);
      }
      float sum_e = 0.0f;
      float exps[kMaxSources];
      for (int n = 0; n < n_sources; ++n) {
        float e = expf(s_logits[q][n] - max_l);
        exps[n] = e;
        sum_e += e;
      }
      float inv_sum = 1.0f / sum_e;
      for (int n = 0; n < n_sources; ++n) {
        s_weights[q][n] = exps[n] * inv_sum * s_inv_rms[n];
      }
    }
  }
  __syncthreads();

  if constexpr (!WRITE_NORMED) {
    // Write unnormalized FP32 mix directly to [num_q, out_q_stride].
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float mix[kMaxQueries];
#pragma unroll
      for (int q = 0; q < kMaxQueries; ++q) mix[q] = 0.0f;

      for (int n = 0; n < n_sources; ++n) {
        const __nv_bfloat16* src_n = (n < n_stack)
                                         ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                         : (partial + row_off);
        float x = __bfloat162float(src_n[d]);
        if (HAS_ADDEND && (n == n_sources - 1)) {
          x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
        }
        for (int q = 0; q < num_q; ++q) {
          mix[q] += s_weights[q][n] * x;
        }
      }
      for (int q = 0; q < num_q; ++q) {
        out_mix[static_cast<long long>(q) * out_q_stride + row_off + d] = mix[q];
      }
    }
  } else {
    // WRITE_NORMED: compute RMS of each query's mix, then normalize and round once to BF16.
    float sum_sq[kMaxQueries];
#pragma unroll
    for (int q = 0; q < kMaxQueries; ++q) sum_sq[q] = 0.0f;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float mix[kMaxQueries];
#pragma unroll
      for (int q = 0; q < kMaxQueries; ++q) mix[q] = 0.0f;

      for (int n = 0; n < n_sources; ++n) {
        const __nv_bfloat16* src_n = (n < n_stack)
                                         ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                         : (partial + row_off);
        float x = __bfloat162float(src_n[d]);
        if (HAS_ADDEND && (n == n_sources - 1)) {
          x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
        }
        for (int q = 0; q < num_q; ++q) {
          mix[q] += s_weights[q][n] * x;
        }
      }
      for (int q = 0; q < num_q; ++q) {
        sum_sq[q] += mix[q] * mix[q];
      }
    }

    float red_sq[kMaxQueries];
    block_reduce_sum_vec<kMaxQueries>(sum_sq, red_sq);
    if (threadIdx.x == 0) {
      for (int q = 0; q < num_q; ++q) {
        s_out_rstd[q] = rsqrt_approx_f32(red_sq[q] * inv_d + norm_eps);
      }
    }
    __syncthreads();

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float mix[kMaxQueries];
#pragma unroll
      for (int q = 0; q < kMaxQueries; ++q) mix[q] = 0.0f;

      for (int n = 0; n < n_sources; ++n) {
        const __nv_bfloat16* src_n = (n < n_stack)
                                         ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                         : (partial + row_off);
        float x = __bfloat162float(src_n[d]);
        if (HAS_ADDEND && (n == n_sources - 1)) {
          x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
        }
        for (int q = 0; q < num_q; ++q) {
          mix[q] += s_weights[q][n] * x;
        }
      }
      for (int q = 0; q < num_q; ++q) {
        float w = __bfloat162float(norm_w[static_cast<long long>(q) * D + d]);
        float y = mix[q] * s_out_rstd[q] * w;
        out_normed[static_cast<long long>(q) * out_q_stride + row_off + d] = __float2bfloat16_rn(y);
      }
    }
  }
}

}  // namespace

// Standard Inkling2 RMSNorm (BF16 -> BF16, NO +1 offset).
// Grid: [T, num_inputs, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_inkling2_rms_norm_bf16(
    const __nv_bfloat16* __restrict__ in,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int T,
    int D,
    int num_inputs,
    float eps) {
  const int t = blockIdx.x;
  const int q = blockIdx.y;
  if (t >= T || q >= num_inputs) return;

  const long long row_off = (static_cast<long long>(t) * num_inputs + q) * D;
  const __nv_bfloat16* x_ptr = in + row_off;
  const __nv_bfloat16* w_ptr = w + static_cast<long long>(q) * D;
  __nv_bfloat16* y_ptr = out + row_off;

  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x = __bfloat162float(x_ptr[d]);
    sum_sq += x * x;
  }
  sum_sq = block_reduce_sum_scalar(sum_sq);
  const float rstd = rsqrt_approx_f32(sum_sq * (1.0f / static_cast<float>(D)) + eps);

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x = __bfloat162float(x_ptr[d]);
    float wt = __bfloat162float(w_ptr[d]);
    y_ptr[d] = __float2bfloat16_rn(x * rstd * wt);
  }
}

// Standard Inkling2 RMSNorm (FP32 -> BF16, NO +1 offset).
// Grid: [T, num_inputs, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_inkling2_rms_norm_f32_bf16(
    const float* __restrict__ in,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    int T,
    int D,
    int num_inputs,
    float eps) {
  const int t = blockIdx.x;
  const int q = blockIdx.y;
  if (t >= T || q >= num_inputs) return;

  const long long row_off = (static_cast<long long>(t) * num_inputs + q) * D;
  const float* x_ptr = in + row_off;
  const __nv_bfloat16* w_ptr = w + static_cast<long long>(q) * D;
  __nv_bfloat16* y_ptr = out + row_off;

  float sum_sq = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x = x_ptr[d];
    sum_sq += x * x;
  }
  sum_sq = block_reduce_sum_scalar(sum_sq);
  const float rstd = rsqrt_approx_f32(sum_sq * (1.0f / static_cast<float>(D)) + eps);

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x = x_ptr[d];
    float wt = __bfloat162float(w_ptr[d]);
    y_ptr[d] = __float2bfloat16_rn(x * rstd * wt);
  }
}

// 4 non-null-pointer AttnRes entry points: Grid [T, 1, 1], Block [256, 1, 1].
extern "C" __global__ void kern_attn_res_norm_bf16(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ queries,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out_normed,
    long long stack_stride,
    long long out_q_stride,
    int T,
    int D,
    int n_stack,
    int num_q,
    float eps,
    float norm_eps) {
  attn_res_core<false, true>(
      stack, partial, nullptr, queries, norm_w, nullptr, out_normed,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, norm_eps, true);
}

extern "C" __global__ void kern_attn_res_addend_norm_bf16(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ queries,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out_normed,
    long long stack_stride,
    long long out_q_stride,
    int T,
    int D,
    int n_stack,
    int num_q,
    float eps,
    float norm_eps) {
  attn_res_core<true, true>(
      stack, partial, addend, queries, norm_w, nullptr, out_normed,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, norm_eps, true);
}

extern "C" __global__ void kern_attn_res_mix_f32(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ queries,
    float* __restrict__ out_mix,
    long long stack_stride,
    long long out_q_stride,
    int T,
    int D,
    int n_stack,
    int num_q,
    float eps) {
  attn_res_core<false, false>(
      stack, partial, nullptr, queries, nullptr, out_mix, nullptr,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, eps, true);
}

extern "C" __global__ void kern_attn_res_addend_mix_f32(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ queries,
    float* __restrict__ out_mix,
    long long stack_stride,
    long long out_q_stride,
    int T,
    int D,
    int n_stack,
    int num_q,
    float eps) {
  attn_res_core<true, false>(
      stack, partial, addend, queries, nullptr, out_mix, nullptr,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, eps, true);
}

// Cross-pass AttnRes read over 2 completed blocks from previous pass -> FP32 [num_q, out_q_stride].
// Grid: [T, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_cross_pass_attn_res_f32(
    const __nv_bfloat16* __restrict__ prev_blocks,
    const float* __restrict__ queries,
    float* __restrict__ out_msgs,
    long long block_stride,
    long long out_q_stride,
    int T,
    int D,
    int num_q,
    float eps) {
  attn_res_core<false, false>(
      prev_blocks, nullptr, nullptr, queries, nullptr, out_msgs, nullptr,
      block_stride, out_q_stride, T, D, 2, num_q, eps, eps, false);
}

// Elementwise FP32 scaling (for derive pre-scaling queries by 1/48).
// Grid: [ceil_div(n, 256), 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_scale_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    long long n,
    float scale) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx] * scale;
  }
}

// Contiguous BF16 copy.
// Grid: [ceil_div(n, 256), 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_copy_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    long long n) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

// Strided tap copy into aux_hidden [T, num_taps, D] BF16.
// Grid: [T, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_tap_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst_taps,
    int T,
    int D,
    int num_taps,
    int tap_idx) {
  const int t = blockIdx.x;
  if (t >= T) return;
  const __nv_bfloat16* s = src + static_cast<long long>(t) * D;
  __nv_bfloat16* d = dst_taps + (static_cast<long long>(t) * num_taps + tap_idx) * D;
  for (int j = threadIdx.x; j < D; j += blockDim.x) {
    d[j] = s[j];
  }
}

// Residual additions in FP32 rounded once to BF16.
// Grid: [ceil_div(n, 256), 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_residual_add2_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    long long n) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float s = __bfloat162float(a[idx]) + __bfloat162float(b[idx]);
    out[idx] = __float2bfloat16_rn(s);
  }
}

extern "C" __global__ void kern_residual_add3_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ c,
    __nv_bfloat16* __restrict__ out,
    long long n) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    float s = __bfloat162float(a[idx]) + __bfloat162float(b[idx]) + __bfloat162float(c[idx]);
    out[idx] = __float2bfloat16_rn(s);
  }
}
