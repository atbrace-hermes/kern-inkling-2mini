// SPDX-License-Identifier: Apache-2.0
// SM120 normalization, Block Attention Residuals (attn_res), Cross-Pass AttnRes,
// and residual/query-scaling helpers for Inkling-2-Mini.
//
// Fast path (D=2304, blockDim.x=256): single-pass register-cached (0 global re-reads,
// 0 stack spills, 2/4 syncthreads) preserving the EXACT thread-to-element mapping
// (d = tid + i*256, i=0..8), 8-warp reduction tree, and n=0..N_STACK accumulation order
// of the reference kernel for bit-for-bit identical outputs.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kMaxSources = 16;
constexpr int kMaxQueries = 8;
constexpr int kMaxWarps = 32;
constexpr int kFastD = 2304;
constexpr int kFastBlock = 256;  // 8 warps x 9 elems/thread = 2304
constexpr int kFastWarps = 8;
constexpr int kElemsPerThread = 9;

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

// Fast single-pass, register-cached, bit-identical AttnRes for D=2304, blockDim.x=256.
// Zero global memory re-reads across Pass 1/2/3, zero stack spills, 2 syncthreads (mix) or 4 (norm).
template <bool HAS_ADDEND, bool WRITE_NORMED, int N_STACK, int NUM_Q>
__device__ __forceinline__ void attn_res_fast_2304(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ queries,
    const __nv_bfloat16* __restrict__ norm_w,
    float* __restrict__ out_mix,
    __nv_bfloat16* __restrict__ out_normed,
    long long stack_stride,
    long long out_q_stride,
    int t,
    float eps,
    float norm_eps,
    float s_red[kFastWarps][20],
    float s_weights[4][5]) {
  constexpr int N_SRC = N_STACK + 1;
  constexpr float kInvD = 1.0f / static_cast<float>(kFastD);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;

  // Prefetch immutable query[0] BEFORE pdl_wait (overlaps with predecessor epilogue).
  float qv0[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    qv0[i] = __ldg(&queries[tid + i * kFastBlock]);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Load all N_STACK sources once as packed BF16 pairs in registers (0 spills) + last source as FP32.
  uint32_t s_pair[N_STACK][4];
  __nv_bfloat16 s_tail[N_STACK];
  float last_x[kElemsPerThread];

#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    const __nv_bfloat16* src_n = stack + static_cast<long long>(n) * stack_stride + row_off;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      __nv_bfloat162 b2;
      b2.x = src_n[tid + (2 * p) * kFastBlock];
      b2.y = src_n[tid + (2 * p + 1) * kFastBlock];
      s_pair[n][p] = *reinterpret_cast<const uint32_t*>(&b2);
    }
    s_tail[n] = src_n[tid + 8 * kFastBlock];
  }

  {
    const __nv_bfloat16* p_ptr = partial + row_off;
    const __nv_bfloat16* a_ptr = HAS_ADDEND ? (addend + row_off) : nullptr;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = tid + i * kFastBlock;
      float x = __bfloat162float(p_ptr[d]);
      if constexpr (HAS_ADDEND) {
        x = __fadd_rn(x, __bfloat162float(a_ptr[d]));
      }
      last_x[i] = x;
    }
  }

  // Pass 1: compute per-source sum_sq and all query dot products in exact d = tid + i*256 order.
  float sum_sq[N_SRC];
  float dots[NUM_Q][N_SRC];

  {
#pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
      float sq = 0.0f, dp = 0.0f;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
        sq += f2.x * f2.x;
        dp += qv0[2 * p] * f2.x;
        sq += f2.y * f2.y;
        dp += qv0[2 * p + 1] * f2.y;
      }
      {
        float x8 = __bfloat162float(s_tail[n]);
        sq += x8 * x8;
        dp += qv0[8] * x8;
      }
      sum_sq[n] = sq;
      dots[0][n] = dp;
    }
    {
      float sq = 0.0f, dp = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        sq += last_x[i] * last_x[i];
        dp += qv0[i] * last_x[i];
      }
      sum_sq[N_STACK] = sq;
      dots[0][N_STACK] = dp;
    }
  }
  if constexpr (NUM_Q > 1) {
#pragma unroll
    for (int q = 1; q < NUM_Q; ++q) {
      const float* q_ptr = queries + static_cast<long long>(q) * kFastD;
      float qv[kElemsPerThread];
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        qv[i] = q_ptr[tid + i * kFastBlock];
      }
#pragma unroll
      for (int n = 0; n < N_STACK; ++n) {
        float dp = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
          dp += qv[2 * p] * f2.x;
          dp += qv[2 * p + 1] * f2.y;
        }
        dp += qv[8] * __bfloat162float(s_tail[n]);
        dots[q][n] = dp;
      }
      {
        float dp = 0.0f;
#pragma unroll
        for (int i = 0; i < kElemsPerThread; ++i) {
          dp += qv[i] * last_x[i];
        }
        dots[q][N_STACK] = dp;
      }
    }
  }

  // Single fused 8-warp block reduction + softmax.
#pragma unroll
  for (int n = 0; n < N_SRC; ++n) {
    float v = warp_reduce_sum(sum_sq[n]);
    if (lane == 0) s_red[warp][n] = v;
  }
#pragma unroll
  for (int q = 0; q < NUM_Q; ++q) {
#pragma unroll
    for (int n = 0; n < N_SRC; ++n) {
      float v = warp_reduce_sum(dots[q][n]);
      if (lane == 0) s_red[warp][N_SRC + q * N_SRC + n] = v;
    }
  }
  __syncthreads();

  if (warp == 0) {
    float inv_rms[N_SRC];
#pragma unroll
    for (int n = 0; n < N_SRC; ++n) {
      float v = (lane < kFastWarps) ? s_red[lane][n] : 0.0f;
      v = warp_reduce_sum(v);
      inv_rms[n] = rsqrt_approx_f32(v * kInvD + eps);
    }
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float logits[N_SRC];
#pragma unroll
      for (int n = 0; n < N_SRC; ++n) {
        float v = (lane < kFastWarps) ? s_red[lane][N_SRC + q * N_SRC + n] : 0.0f;
        v = warp_reduce_sum(v);
        logits[n] = v * inv_rms[n];
      }
      if (lane == 0) {
        float max_l = logits[0];
#pragma unroll
        for (int n = 1; n < N_SRC; ++n) max_l = fmaxf(max_l, logits[n]);
        float exps[N_SRC];
        float sum_e = 0.0f;
#pragma unroll
        for (int n = 0; n < N_SRC; ++n) {
          exps[n] = expf(logits[n] - max_l);
          sum_e += exps[n];
        }
        float inv_sum = 1.0f / sum_e;
#pragma unroll
        for (int n = 0; n < N_SRC; ++n) {
          s_weights[q][n] = exps[n] * inv_sum * inv_rms[n];
        }
      }
    }
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // Pass 2: mix in registers in EXACT n = 0..N_STACK order (zero global re-reads!).
  if constexpr (!WRITE_NORMED) {
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float mix[kElemsPerThread];
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) mix[i] = 0.0f;
#pragma unroll
      for (int n = 0; n < N_STACK; ++n) {
        float wn = s_weights[q][n];
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
          mix[2 * p] += wn * f2.x;
          mix[2 * p + 1] += wn * f2.y;
        }
        mix[8] += wn * __bfloat162float(s_tail[n]);
      }
      {
        float wl = s_weights[q][N_STACK];
#pragma unroll
        for (int i = 0; i < kElemsPerThread; ++i) {
          mix[i] += wl * last_x[i];
        }
      }
      float* dst = out_mix + static_cast<long long>(q) * out_q_stride + row_off;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        dst[tid + i * kFastBlock] = mix[i];
      }
    }
  } else {
    float mix_all[NUM_Q][kElemsPerThread];
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) mix_all[q][i] = 0.0f;
    }
#pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
#pragma unroll
        for (int q = 0; q < NUM_Q; ++q) {
          float wn = s_weights[q][n];
          mix_all[q][2 * p] += wn * f2.x;
          mix_all[q][2 * p + 1] += wn * f2.y;
        }
      }
      {
        float x8 = __bfloat162float(s_tail[n]);
#pragma unroll
        for (int q = 0; q < NUM_Q; ++q) {
          mix_all[q][8] += s_weights[q][n] * x8;
        }
      }
    }
#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float wl = s_weights[q][N_STACK];
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        mix_all[q][i] += wl * last_x[i];
      }
    }

#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float sq = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        sq += mix_all[q][i] * mix_all[q][i];
      }
      float v = warp_reduce_sum(sq);
      if (lane == 0) s_red[warp][q] = v;
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
      for (int q = 0; q < NUM_Q; ++q) {
        float v = (lane < kFastWarps) ? s_red[lane][q] : 0.0f;
        v = warp_reduce_sum(v);
        if (lane == 0) s_weights[q][0] = rsqrt_approx_f32(v * kInvD + norm_eps);
      }
    }
    __syncthreads();

#pragma unroll
    for (int q = 0; q < NUM_Q; ++q) {
      float rstd = s_weights[q][0];
      const __nv_bfloat16* nw_ptr = norm_w + static_cast<long long>(q) * kFastD;
      __nv_bfloat16* dst = out_normed + static_cast<long long>(q) * out_q_stride + row_off;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        int d = tid + i * kFastBlock;
        float w = __bfloat162float(nw_ptr[d]);
        dst[d] = __float2bfloat16_rn(mix_all[q][i] * rstd * w);
      }
    }
  }
}

// Dedicated bit-identical fast path for cross_pass_attn_res_f32 (2 sources, 4 queries, D=2304, block=256).
__device__ __forceinline__ void cross_pass_fast_2304(
    const __nv_bfloat16* __restrict__ prev_blocks,
    const float* __restrict__ queries,
    float* __restrict__ out_msgs,
    long long block_stride,
    long long out_q_stride,
    int t,
    float eps) {
  constexpr float kInvD = 1.0f / static_cast<float>(kFastD);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const __nv_bfloat16* p0 = prev_blocks + row_off;
  const __nv_bfloat16* p1 = prev_blocks + block_stride + row_off;

  float s0[kElemsPerThread], s1[kElemsPerThread];
  float sq0 = 0.0f, sq1 = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    int d = tid + i * kFastBlock;
    s0[i] = __bfloat162float(p0[d]);
    s1[i] = __bfloat162float(p1[d]);
    sq0 += s0[i] * s0[i];
    sq1 += s1[i] * s1[i];
  }

  float d0_q[4], d1_q[4];
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    const float* q_ptr = queries + static_cast<long long>(q) * kFastD;
    float a = 0.0f, b = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      float qv = q_ptr[tid + i * kFastBlock];
      a += qv * s0[i];
      b += qv * s1[i];
    }
    d0_q[q] = a;
    d1_q[q] = b;
  }

  __shared__ float s_red[kFastWarps][10];
  __shared__ float s_w[4][2];
  {
    float r0 = warp_reduce_sum(sq0);
    float r1 = warp_reduce_sum(sq1);
    if (lane == 0) {
      s_red[warp][0] = r0;
      s_red[warp][1] = r1;
    }
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float ra = warp_reduce_sum(d0_q[q]);
      float rb = warp_reduce_sum(d1_q[q]);
      if (lane == 0) {
        s_red[warp][2 + 2 * q] = ra;
        s_red[warp][3 + 2 * q] = rb;
      }
    }
  }
  __syncthreads();

  if (warp == 0) {
    float v0 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][0] : 0.0f);
    float v1 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][1] : 0.0f);
    float inv0 = rsqrt_approx_f32(v0 * kInvD + eps);
    float inv1 = rsqrt_approx_f32(v1 * kInvD + eps);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float l0 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][2 + 2 * q] : 0.0f) * inv0;
      float l1 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][3 + 2 * q] : 0.0f) * inv1;
      if (lane == 0) {
        float ml = fmaxf(l0, l1);
        float e0 = expf(l0 - ml);
        float e1 = expf(l1 - ml);
        float inv_s = 1.0f / (e0 + e1);
        s_w[q][0] = e0 * inv_s * inv0;
        s_w[q][1] = e1 * inv_s * inv1;
      }
    }
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

#pragma unroll
  for (int q = 0; q < 4; ++q) {
    float w0 = s_w[q][0];
    float w1 = s_w[q][1];
    float* dst = out_msgs + static_cast<long long>(q) * out_q_stride + row_off;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      float mix = 0.0f;
      mix += w0 * s0[i];
      mix += w1 * s1[i];
      dst[tid + i * kFastBlock] = mix;
    }
  }
}

template <bool HAS_ADDEND, bool WRITE_NORMED>
__device__ __noinline__ void attn_res_generic(
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
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int n_sources = n_stack + (has_partial ? 1 : 0);
  if (n_sources <= 0 || n_sources > kMaxSources || num_q <= 0 || num_q > kMaxQueries) {
    __trap();
  }

  __shared__ float s_inv_rms[kMaxSources];
  __shared__ float s_logits[kMaxQueries][kMaxSources];
  __shared__ float s_weights[kMaxQueries][kMaxSources];

  const float inv_d = 1.0f / static_cast<float>(D);
  const long long row_off = static_cast<long long>(t) * D;

  for (int n = 0; n < n_sources; ++n) {
    const __nv_bfloat16* src_n = (n < n_stack)
                                     ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                     : (partial + row_off);
    const bool use_addend = HAS_ADDEND && (n == n_sources - 1);

    float sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
      float x = __bfloat162float(src_n[d]);
      if (use_addend) x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
      sq += x * x;
    }
    float red_sq = block_reduce_sum_scalar(sq);
    float rstd = rsqrt_approx_f32(red_sq * inv_d + eps);
    if (threadIdx.x == 0) s_inv_rms[n] = rstd;

    for (int q = 0; q < num_q; ++q) {
      float dp = 0.0f;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float x = __bfloat162float(src_n[d]);
        if (use_addend) x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
        dp += queries[static_cast<long long>(q) * D + d] * x;
      }
      float red_dp = block_reduce_sum_scalar(dp);
      if (threadIdx.x == 0) s_logits[q][n] = red_dp * rstd;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    for (int q = 0; q < num_q; ++q) {
      float max_l = s_logits[q][0];
      for (int n = 1; n < n_sources; ++n) max_l = fmaxf(max_l, s_logits[q][n]);
      float sum_e = 0.0f;
      for (int n = 0; n < n_sources; ++n) sum_e += expf(s_logits[q][n] - max_l);
      float inv_sum = 1.0f / sum_e;
      for (int n = 0; n < n_sources; ++n) {
        s_weights[q][n] = expf(s_logits[q][n] - max_l) * inv_sum * s_inv_rms[n];
      }
    }
  }
  __syncthreads();

  for (int q = 0; q < num_q; ++q) {
    if constexpr (!WRITE_NORMED) {
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float mix = 0.0f;
        for (int n = 0; n < n_sources; ++n) {
          const __nv_bfloat16* src_n = (n < n_stack)
                                           ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                           : (partial + row_off);
          float x = __bfloat162float(src_n[d]);
          if (HAS_ADDEND && (n == n_sources - 1)) x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
          mix += s_weights[q][n] * x;
        }
        out_mix[static_cast<long long>(q) * out_q_stride + row_off + d] = mix;
      }
    } else {
      float sq = 0.0f;
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float mix = 0.0f;
        for (int n = 0; n < n_sources; ++n) {
          const __nv_bfloat16* src_n = (n < n_stack)
                                           ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                           : (partial + row_off);
          float x = __bfloat162float(src_n[d]);
          if (HAS_ADDEND && (n == n_sources - 1)) x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
          mix += s_weights[q][n] * x;
        }
        sq += mix * mix;
      }
      float red_sq = block_reduce_sum_scalar(sq);
      float rstd = rsqrt_approx_f32(red_sq * inv_d + norm_eps);
      for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float mix = 0.0f;
        for (int n = 0; n < n_sources; ++n) {
          const __nv_bfloat16* src_n = (n < n_stack)
                                           ? (stack + static_cast<long long>(n) * stack_stride + row_off)
                                           : (partial + row_off);
          float x = __bfloat162float(src_n[d]);
          if (HAS_ADDEND && (n == n_sources - 1)) x = __fadd_rn(x, __bfloat162float(addend[row_off + d]));
          mix += s_weights[q][n] * x;
        }
        float w = __bfloat162float(norm_w[static_cast<long long>(q) * D + d]);
        out_normed[static_cast<long long>(q) * out_q_stride + row_off + d] = __float2bfloat16_rn(mix * rstd * w);
      }
    }
  }
}

template <bool HAS_ADDEND, bool WRITE_NORMED>
__device__ __forceinline__ void attn_res_dispatch(
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
    float norm_eps) {
  const int t = blockIdx.x;
  if (t >= T) return;

  __shared__ float s_red[kFastWarps][20];
  __shared__ float s_weights[4][5];

  if (D == kFastD && blockDim.x == kFastBlock && n_stack >= 1 && n_stack <= 4 &&
      (num_q == 1 || num_q == 3)) {
    if (num_q == 1) {
      switch (n_stack) {
        case 1:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 1, 1>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
        case 2:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 2, 1>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
        case 3:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 3, 1>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
        case 4:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 4, 1>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
      }
    } else {
      switch (n_stack) {
        case 1:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 1, 3>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
        case 2:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 2, 3>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
        case 3:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 3, 3>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
        case 4:
          attn_res_fast_2304<HAS_ADDEND, WRITE_NORMED, 4, 3>(
              stack, partial, addend, queries, norm_w, out_mix, out_normed,
              stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
          return;
      }
    }
  }
  attn_res_generic<HAS_ADDEND, WRITE_NORMED>(
      stack, partial, addend, queries, norm_w, out_mix, out_normed,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, norm_eps, true);
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

  if (D == kFastD && blockDim.x == kFastBlock) {
    float rx[kElemsPerThread];
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = threadIdx.x + i * kFastBlock;
      rx[i] = __bfloat162float(x_ptr[d]);
      sum_sq += rx[i] * rx[i];
    }
    sum_sq = block_reduce_sum_scalar(sum_sq);
    const float rstd = rsqrt_approx_f32(sum_sq * (1.0f / static_cast<float>(kFastD)) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = threadIdx.x + i * kFastBlock;
      float wt = __bfloat162float(w_ptr[d]);
      y_ptr[d] = __float2bfloat16_rn(rx[i] * rstd * wt);
    }
    return;
  }

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

  if (D == kFastD && blockDim.x == kFastBlock) {
    float rx[kElemsPerThread];
    float sum_sq = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = threadIdx.x + i * kFastBlock;
      rx[i] = x_ptr[d];
      sum_sq += rx[i] * rx[i];
    }
    sum_sq = block_reduce_sum_scalar(sum_sq);
    const float rstd = rsqrt_approx_f32(sum_sq * (1.0f / static_cast<float>(kFastD)) + eps);
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = threadIdx.x + i * kFastBlock;
      float wt = __bfloat162float(w_ptr[d]);
      y_ptr[d] = __float2bfloat16_rn(rx[i] * rstd * wt);
    }
    return;
  }

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
  attn_res_dispatch<false, true>(
      stack, partial, nullptr, queries, norm_w, nullptr, out_normed,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, norm_eps);
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
  attn_res_dispatch<true, true>(
      stack, partial, addend, queries, norm_w, nullptr, out_normed,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, norm_eps);
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
  attn_res_dispatch<false, false>(
      stack, partial, nullptr, queries, nullptr, out_mix, nullptr,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, eps);
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
  attn_res_dispatch<true, false>(
      stack, partial, addend, queries, nullptr, out_mix, nullptr,
      stack_stride, out_q_stride, T, D, n_stack, num_q, eps, eps);
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
  const int t = blockIdx.x;
  if (t >= T) return;
  if (D == kFastD && blockDim.x == kFastBlock && num_q == 4) {
    cross_pass_fast_2304(prev_blocks, queries, out_msgs, block_stride, out_q_stride, t, eps);
    return;
  }
  attn_res_generic<false, false>(
      prev_blocks, nullptr, nullptr, queries, nullptr, out_msgs, nullptr,
      block_stride, out_q_stride, T, D, 2, num_q, eps, eps, false);
}

// 4-stream bit-identical Cross-Pass AttnRes for small T (T <= 64): 4x more CTAs saturate SMs.
// Grid: [T, 4, 1], Block: [256, 1, 1]
extern "C" __global__ __launch_bounds__(256, 4) void kern_cross_pass_attn_res_4q_f32(
    const __nv_bfloat16* __restrict__ prev_blocks,
    const float* __restrict__ queries,
    float* __restrict__ out_msgs,
    long long block_stride,
    long long out_q_stride,
    int T,
    float eps) {
  const int t = blockIdx.x;
  const int q = blockIdx.y;
  if (t >= T) return;

  constexpr float kInvD = 1.0f / static_cast<float>(kFastD);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;

  // Prefetch immutable query vector BEFORE pdl_wait!
  const float* q_ptr = queries + static_cast<long long>(q) * kFastD;
  float qv[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    qv[i] = __ldg(&q_ptr[tid + i * kFastBlock]);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");

  const __nv_bfloat16* p0 = prev_blocks + row_off;
  const __nv_bfloat16* p1 = prev_blocks + block_stride + row_off;

  float s0[kElemsPerThread], s1[kElemsPerThread];
  float sq0 = 0.0f, sq1 = 0.0f, a = 0.0f, b = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    int d = tid + i * kFastBlock;
    s0[i] = __bfloat162float(p0[d]);
    s1[i] = __bfloat162float(p1[d]);
    sq0 += s0[i] * s0[i];
    sq1 += s1[i] * s1[i];
    a += qv[i] * s0[i];
    b += qv[i] * s1[i];
  }

  __shared__ float s_red[kFastWarps][4];
  __shared__ float s_w[2];
  sq0 = warp_reduce_sum(sq0);
  sq1 = warp_reduce_sum(sq1);
  a = warp_reduce_sum(a);
  b = warp_reduce_sum(b);
  if (lane == 0) {
    s_red[warp][0] = sq0;
    s_red[warp][1] = sq1;
    s_red[warp][2] = a;
    s_red[warp][3] = b;
  }
  __syncthreads();

  if (warp == 0) {
    float v0 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][0] : 0.0f);
    float v1 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][1] : 0.0f);
    float inv0 = rsqrt_approx_f32(v0 * kInvD + eps);
    float inv1 = rsqrt_approx_f32(v1 * kInvD + eps);
    float l0 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][2] : 0.0f) * inv0;
    float l1 = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][3] : 0.0f) * inv1;
    if (lane == 0) {
      float ml = fmaxf(l0, l1);
      float e0 = expf(l0 - ml);
      float e1 = expf(l1 - ml);
      float inv_s = 1.0f / (e0 + e1);
      s_w[0] = e0 * inv_s * inv0;
      s_w[1] = e1 * inv_s * inv1;
    }
  }
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const float w0 = s_w[0];
  const float w1 = s_w[1];
  float* dst = out_msgs + static_cast<long long>(q) * out_q_stride + row_off;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    float mix = 0.0f;
    mix += w0 * s0[i];
    mix += w1 * s1[i];
    dst[tid + i * kFastBlock] = mix;
  }
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

// Contiguous BF16 copy (scalar high-CTA for small T, vectorized 8x BF16 for large T).
extern "C" __global__ void kern_copy_bf16(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    long long n) {
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n) {
    dst[idx] = src[idx];
  }
}

extern "C" __global__ void kern_copy_bf16_vec8(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    long long n) {
  const long long base = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 8LL;
  if (base + 8LL <= n && ((reinterpret_cast<uintptr_t>(src) | reinterpret_cast<uintptr_t>(dst)) & 15u) == 0u) {
    *reinterpret_cast<uint4*>(dst + base) = *reinterpret_cast<const uint4*>(src + base);
  } else {
    for (long long i = 0; i < 8LL && base + i < n; ++i) {
      dst[base + i] = src[base + i];
    }
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

// Residual additions in FP32 rounded once to BF16 (scalar for small T, vec8 for large T).
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

extern "C" __global__ void kern_residual_add2_bf16_vec8(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    long long n) {
  const long long base = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 8LL;
  if (base + 8LL <= n &&
      ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(out)) & 15u) == 0u) {
    uint4 va = *reinterpret_cast<const uint4*>(a + base);
    uint4 vb = *reinterpret_cast<const uint4*>(b + base);
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&va);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&vb);
    uint4 vo;
    __nv_bfloat162* ho = reinterpret_cast<__nv_bfloat162*>(&vo);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 fa = __bfloat1622float2(ha[i]);
      float2 fb = __bfloat1622float2(hb[i]);
      ho[i] = __floats2bfloat162_rn(fa.x + fb.x, fa.y + fb.y);
    }
    *reinterpret_cast<uint4*>(out + base) = vo;
  } else {
    for (long long i = 0; i < 8LL && base + i < n; ++i) {
      float s = __bfloat162float(a[base + i]) + __bfloat162float(b[base + i]);
      out[base + i] = __float2bfloat16_rn(s);
    }
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

extern "C" __global__ void kern_residual_add3_bf16_vec8(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ c,
    __nv_bfloat16* __restrict__ out,
    long long n) {
  const long long base = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * 8LL;
  if (base + 8LL <= n &&
      ((reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) | reinterpret_cast<uintptr_t>(c) | reinterpret_cast<uintptr_t>(out)) & 15u) == 0u) {
    uint4 va = *reinterpret_cast<const uint4*>(a + base);
    uint4 vb = *reinterpret_cast<const uint4*>(b + base);
    uint4 vc = *reinterpret_cast<const uint4*>(c + base);
    const __nv_bfloat162* ha = reinterpret_cast<const __nv_bfloat162*>(&va);
    const __nv_bfloat162* hb = reinterpret_cast<const __nv_bfloat162*>(&vb);
    const __nv_bfloat162* hc = reinterpret_cast<const __nv_bfloat162*>(&vc);
    uint4 vo;
    __nv_bfloat162* ho = reinterpret_cast<__nv_bfloat162*>(&vo);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float2 fa = __bfloat1622float2(ha[i]);
      float2 fb = __bfloat1622float2(hb[i]);
      float2 fc = __bfloat1622float2(hc[i]);
      ho[i] = __floats2bfloat162_rn(fa.x + fb.x + fc.x, fa.y + fb.y + fc.y);
    }
    *reinterpret_cast<uint4*>(out + base) = vo;
  } else {
    for (long long i = 0; i < 8LL && base + i < n; ++i) {
      float s = __bfloat162float(a[base + i]) + __bfloat162float(b[base + i]) + __bfloat162float(c[base + i]);
      out[base + i] = __float2bfloat16_rn(s);
    }
  }
}

namespace {

__device__ __forceinline__ float warp_reduce_sum_nofma(float val) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    val = __fadd_rn(val, __shfl_xor_sync(0xffffffffu, val, offset));
  }
  return val;
}

// Strict no-FMA FP32 weightless RMSNorm -> BF16 (100% bit-identical to loop_controller s1).
__device__ __forceinline__ float compute_rstd_nofma_2304(
    const float (&x)[kElemsPerThread], float eps, float s_red8[kFastWarps]) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float sq = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    sq = __fadd_rn(sq, __fmul_rn(x[i], x[i]));
  }
  sq = warp_reduce_sum_nofma(sq);
  if (lane == 0) s_red8[warp] = sq;
  __syncthreads();
  if (warp == 0) {
    float v = warp_reduce_sum_nofma((lane < kFastWarps) ? s_red8[lane] : 0.0f);
    if (lane == 0) {
      s_red8[0] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v, 1.0f / static_cast<float>(kFastD)), eps));
    }
  }
  __syncthreads();
  const float rstd = s_red8[0];
  __syncthreads();
  return rstd;
}

// Single-query AttnRes mix + register-fused no-FMA s1 RMSNorm (N_STACK in 1..4, D=2304, block=256).
template <bool HAS_ADDEND, int N_STACK>
__device__ __forceinline__ void attn_res_mix_1q_s1_2304(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ q_ptr,
    float* __restrict__ out_mix_row,
    __nv_bfloat16* __restrict__ cat0_row,
    __nv_bfloat16* __restrict__ cat1_row,
    __nv_bfloat16* __restrict__ cat2_row,
    bool broadcast_3x,
    long long stack_stride,
    int t,
    float eps,
    float s_red[kFastWarps][20],
    float s_weights[5]) {
  constexpr int N_SRC = N_STACK + 1;
  constexpr float kInvD = 1.0f / static_cast<float>(kFastD);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;

  float qv[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    qv[i] = __ldg(&q_ptr[tid + i * kFastBlock]);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");

  uint32_t s_pair[N_STACK][4];
  __nv_bfloat16 s_tail[N_STACK];
  float last_x[kElemsPerThread];

#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    const __nv_bfloat16* src_n = stack + static_cast<long long>(n) * stack_stride + row_off;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      __nv_bfloat162 b2;
      b2.x = src_n[tid + (2 * p) * kFastBlock];
      b2.y = src_n[tid + (2 * p + 1) * kFastBlock];
      s_pair[n][p] = *reinterpret_cast<const uint32_t*>(&b2);
    }
    s_tail[n] = src_n[tid + 8 * kFastBlock];
  }
  {
    const __nv_bfloat16* p_ptr = partial + row_off;
    const __nv_bfloat16* a_ptr = HAS_ADDEND ? (addend + row_off) : nullptr;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = tid + i * kFastBlock;
      float x = __bfloat162float(p_ptr[d]);
      if constexpr (HAS_ADDEND) x = __fadd_rn(x, __bfloat162float(a_ptr[d]));
      last_x[i] = x;
    }
  }

  float sum_sq[N_SRC];
  float dots[N_SRC];
#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
      sq += f2.x * f2.x;
      dp += qv[2 * p] * f2.x;
      sq += f2.y * f2.y;
      dp += qv[2 * p + 1] * f2.y;
    }
    float x8 = __bfloat162float(s_tail[n]);
    sq += x8 * x8;
    dp += qv[8] * x8;
    sum_sq[n] = sq;
    dots[n] = dp;
  }
  {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      sq += last_x[i] * last_x[i];
      dp += qv[i] * last_x[i];
    }
    sum_sq[N_STACK] = sq;
    dots[N_STACK] = dp;
  }

#pragma unroll
  for (int n = 0; n < N_SRC; ++n) {
    float vs = warp_reduce_sum(sum_sq[n]);
    float vd = warp_reduce_sum(dots[n]);
    if (lane == 0) {
      s_red[warp][n] = vs;
      s_red[warp][N_SRC + n] = vd;
    }
  }
  __syncthreads();

  if (warp == 0) {
    float inv_rms[N_SRC];
    float logits[N_SRC];
#pragma unroll
    for (int n = 0; n < N_SRC; ++n) {
      float vs = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][n] : 0.0f);
      inv_rms[n] = rsqrt_approx_f32(vs * kInvD + eps);
      float vd = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][N_SRC + n] : 0.0f);
      logits[n] = vd * inv_rms[n];
    }
    if (lane == 0) {
      float max_l = logits[0];
#pragma unroll
      for (int n = 1; n < N_SRC; ++n) max_l = fmaxf(max_l, logits[n]);
      float exps[N_SRC];
      float sum_e = 0.0f;
#pragma unroll
      for (int n = 0; n < N_SRC; ++n) {
        exps[n] = expf(logits[n] - max_l);
        sum_e += exps[n];
      }
      float inv_sum = 1.0f / sum_e;
#pragma unroll
      for (int n = 0; n < N_SRC; ++n) {
        s_weights[n] = exps[n] * inv_sum * inv_rms[n];
      }
    }
  }
  __syncthreads();

  float mix[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) mix[i] = 0.0f;
#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    float wn = s_weights[n];
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
      mix[2 * p] += wn * f2.x;
      mix[2 * p + 1] += wn * f2.y;
    }
    mix[8] += wn * __bfloat162float(s_tail[n]);
  }
  {
    float wl = s_weights[N_STACK];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      mix[i] += wl * last_x[i];
      out_mix_row[tid + i * kFastBlock] = mix[i];
    }
  }

  // Register-fused s1 RMSNorm (strict no-FMA, zero re-read of out_mix_row)
  float s_r8_tmp;
  float sq_m = 0.0f;
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    sq_m = __fadd_rn(sq_m, __fmul_rn(mix[i], mix[i]));
  }
  sq_m = warp_reduce_sum_nofma(sq_m);
  if (lane == 0) s_red[warp][0] = sq_m;
  __syncthreads();
  if (warp == 0) {
    float v = warp_reduce_sum_nofma((lane < kFastWarps) ? s_red[lane][0] : 0.0f);
    if (lane == 0) {
      s_red[0][0] = rsqrt_approx_f32(__fadd_rn(__fmul_rn(v, kInvD), eps));
    }
  }
  __syncthreads();
  s_r8_tmp = s_red[0][0];
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    int d = tid + i * kFastBlock;
    __nv_bfloat16 y = __float2bfloat16_rn(__fmul_rn(mix[i], s_r8_tmp));
    cat0_row[d] = y;
    if (broadcast_3x) {
      cat1_row[d] = y;
      cat2_row[d] = y;
    }
  }
}

// 3-query AttnRes mix + register-fused s1 RMSNorm (prefill Full layers, reads stack/partial ONCE).
template <int N_STACK>
__device__ __forceinline__ void attn_res_mix_3q_s1_2304(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ queries,
    float* __restrict__ out_mix,
    __nv_bfloat16* __restrict__ cat0_row,
    __nv_bfloat16* __restrict__ cat1_row,
    __nv_bfloat16* __restrict__ cat2_row,
    long long stack_stride,
    long long out_q_stride,
    int t,
    float eps,
    float s_red[kFastWarps][20],
    float s_weights[3][5]) {
  constexpr int N_SRC = N_STACK + 1;
  constexpr float kInvD = 1.0f / static_cast<float>(kFastD);
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;

  float qv0[kElemsPerThread];
#pragma unroll
  for (int i = 0; i < kElemsPerThread; ++i) {
    qv0[i] = __ldg(&queries[tid + i * kFastBlock]);
  }
  asm volatile("griddepcontrol.wait;" ::: "memory");

  uint32_t s_pair[N_STACK][4];
  __nv_bfloat16 s_tail[N_STACK];
  float last_x[kElemsPerThread];

#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    const __nv_bfloat16* src_n = stack + static_cast<long long>(n) * stack_stride + row_off;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      __nv_bfloat162 b2;
      b2.x = src_n[tid + (2 * p) * kFastBlock];
      b2.y = src_n[tid + (2 * p + 1) * kFastBlock];
      s_pair[n][p] = *reinterpret_cast<const uint32_t*>(&b2);
    }
    s_tail[n] = src_n[tid + 8 * kFastBlock];
  }
  {
    const __nv_bfloat16* p_ptr = partial + row_off;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      last_x[i] = __bfloat162float(p_ptr[tid + i * kFastBlock]);
    }
  }

  float sum_sq[N_SRC];
  float dots[3][N_SRC];
#pragma unroll
  for (int n = 0; n < N_STACK; ++n) {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
      sq += f2.x * f2.x;
      dp += qv0[2 * p] * f2.x;
      sq += f2.y * f2.y;
      dp += qv0[2 * p + 1] * f2.y;
    }
    float x8 = __bfloat162float(s_tail[n]);
    sq += x8 * x8;
    dp += qv0[8] * x8;
    sum_sq[n] = sq;
    dots[0][n] = dp;
  }
  {
    float sq = 0.0f, dp = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      sq += last_x[i] * last_x[i];
      dp += qv0[i] * last_x[i];
    }
    sum_sq[N_STACK] = sq;
    dots[0][N_STACK] = dp;
  }

#pragma unroll
  for (int q = 1; q < 3; ++q) {
    const float* q_ptr = queries + static_cast<long long>(q) * kFastD;
    float qv[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) qv[i] = __ldg(&q_ptr[tid + i * kFastBlock]);
#pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
      float dp = 0.0f;
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
        dp += qv[2 * p] * f2.x;
        dp += qv[2 * p + 1] * f2.y;
      }
      dp += qv[8] * __bfloat162float(s_tail[n]);
      dots[q][n] = dp;
    }
    float dp = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) dp += qv[i] * last_x[i];
    dots[q][N_STACK] = dp;
  }

#pragma unroll
  for (int n = 0; n < N_SRC; ++n) {
    float v = warp_reduce_sum(sum_sq[n]);
    if (lane == 0) s_red[warp][n] = v;
  }
#pragma unroll
  for (int q = 0; q < 3; ++q) {
#pragma unroll
    for (int n = 0; n < N_SRC; ++n) {
      float v = warp_reduce_sum(dots[q][n]);
      if (lane == 0) s_red[warp][N_SRC + q * N_SRC + n] = v;
    }
  }
  __syncthreads();

  if (warp == 0) {
    float inv_rms[N_SRC];
#pragma unroll
    for (int n = 0; n < N_SRC; ++n) {
      float v = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][n] : 0.0f);
      inv_rms[n] = rsqrt_approx_f32(v * kInvD + eps);
    }
#pragma unroll
    for (int q = 0; q < 3; ++q) {
      float logits[N_SRC];
#pragma unroll
      for (int n = 0; n < N_SRC; ++n) {
        float v = warp_reduce_sum((lane < kFastWarps) ? s_red[lane][N_SRC + q * N_SRC + n] : 0.0f);
        logits[n] = v * inv_rms[n];
      }
      if (lane == 0) {
        float max_l = logits[0];
#pragma unroll
        for (int n = 1; n < N_SRC; ++n) max_l = fmaxf(max_l, logits[n]);
        float exps[N_SRC];
        float sum_e = 0.0f;
#pragma unroll
        for (int n = 0; n < N_SRC; ++n) {
          exps[n] = expf(logits[n] - max_l);
          sum_e += exps[n];
        }
        float inv_sum = 1.0f / sum_e;
#pragma unroll
        for (int n = 0; n < N_SRC; ++n) s_weights[q][n] = exps[n] * inv_sum * inv_rms[n];
      }
    }
  }
  __syncthreads();

  __shared__ float s_r8[kFastWarps];
#pragma unroll
  for (int q = 0; q < 3; ++q) {
    float mix[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) mix[i] = 0.0f;
#pragma unroll
    for (int n = 0; n < N_STACK; ++n) {
      float wn = s_weights[q][n];
#pragma unroll
      for (int p = 0; p < 4; ++p) {
        float2 f2 = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&s_pair[n][p]));
        mix[2 * p] += wn * f2.x;
        mix[2 * p + 1] += wn * f2.y;
      }
      mix[8] += wn * __bfloat162float(s_tail[n]);
    }
    float wl = s_weights[q][N_STACK];
    float* dst_mix = out_mix + static_cast<long long>(q) * out_q_stride + row_off;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      mix[i] += wl * last_x[i];
      dst_mix[tid + i * kFastBlock] = mix[i];
    }
    float rstd = compute_rstd_nofma_2304(mix, eps, s_r8);
    if (q == 2) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __nv_bfloat16* cat_q = (q == 0) ? cat0_row : ((q == 1) ? cat1_row : cat2_row);
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      cat_q[tid + i * kFastBlock] = __float2bfloat16_rn(__fmul_rn(mix[i], rstd));
    }
  }
}

}  // namespace

// Dedicated single-query AttnRes + RMSNorm (REG<=56 -> 2x occupancy vs 80-reg generic).
// Grid: [T, 1, 1] (num_q=1) or [T, 3, 1] (num_q=3 for T<=64), Block: [256, 1, 1].
extern "C" __global__ __launch_bounds__(256, 3) void kern_attn_res_norm_1q_bf16(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ queries,
    const __nv_bfloat16* __restrict__ norm_w,
    __nv_bfloat16* __restrict__ out_normed,
    long long stack_stride,
    long long out_q_stride,
    int T,
    int n_stack,
    float eps,
    float norm_eps) {
  const int t = blockIdx.x;
  const int q = blockIdx.y;
  if (t >= T) return;

  __shared__ float s_red[kFastWarps][20];
  __shared__ float s_weights[4][5];
  const float* q_ptr = queries + static_cast<long long>(q) * kFastD;
  const __nv_bfloat16* nw_ptr = norm_w + static_cast<long long>(q) * kFastD;
  __nv_bfloat16* out_ptr = out_normed + static_cast<long long>(q) * out_q_stride;

  switch (n_stack) {
    case 1:
      attn_res_fast_2304<false, true, 1, 1>(
          stack, partial, nullptr, q_ptr, nw_ptr, nullptr, out_ptr,
          stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
      return;
    case 2:
      attn_res_fast_2304<false, true, 2, 1>(
          stack, partial, nullptr, q_ptr, nw_ptr, nullptr, out_ptr,
          stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
      return;
    case 3:
      attn_res_fast_2304<false, true, 3, 1>(
          stack, partial, nullptr, q_ptr, nw_ptr, nullptr, out_ptr,
          stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
      return;
    case 4:
      attn_res_fast_2304<false, true, 4, 1>(
          stack, partial, nullptr, q_ptr, nw_ptr, nullptr, out_ptr,
          stack_stride, out_q_stride, t, eps, norm_eps, s_red, s_weights);
      return;
    default:
      __trap();
  }
}

// Fused Pre-Attention (attn_res_mix + cross_pass_attn_res + loop_controller_3x.s1) for T <= 64.
// Grid: [T, 5, 1] (SWA, in_stride==0) or [T, 7, 1] (Full, in_stride>0), Block: [256, 1, 1].
//   by in 0..2: cross_pass q=by -> writes msg_buf[by] + register-fused s1 -> cat_{by}[D..2D)
//   by == 3:    cross_pass q=3  -> writes msg_buf[3] (for pre_mlp)
//   by >= 4:    attn_res_mix q=by-4 -> writes mix_qkv[q] + register-fused s1 -> cat_{q or 0..2}[0..D)
extern "C" __global__ __launch_bounds__(256, 4) void kern_pre_attn_mix_cross_s1_lowm(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ attn_q,
    const __nv_bfloat16* __restrict__ prev_blocks,
    const float* __restrict__ cross_q,
    float* __restrict__ mix_qkv,
    float* __restrict__ msg_buf,
    __nv_bfloat16* __restrict__ cat0,
    __nv_bfloat16* __restrict__ cat1,
    __nv_bfloat16* __restrict__ cat2,
    long long stack_stride,
    long long out_q_stride,
    long long in_stride,
    int T,
    int n_stack,
    float eps) {
  const int t = blockIdx.x;
  const int by = blockIdx.y;
  if (t >= T) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;

  if (by < 4) {
    constexpr float kInvD = 1.0f / static_cast<float>(kFastD);
    const float* q_ptr = cross_q + static_cast<long long>(by) * kFastD;
    float qv[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) qv[i] = __ldg(&q_ptr[tid + i * kFastBlock]);
    asm volatile("griddepcontrol.wait;" ::: "memory");

    const __nv_bfloat16* p0 = prev_blocks + row_off;
    const __nv_bfloat16* p1 = prev_blocks + stack_stride + row_off;
    float s0[kElemsPerThread], s1[kElemsPerThread];
    float sq0 = 0.0f, sq1 = 0.0f, a = 0.0f, b = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = tid + i * kFastBlock;
      s0[i] = __bfloat162float(p0[d]);
      s1[i] = __bfloat162float(p1[d]);
      sq0 += s0[i] * s0[i];
      sq1 += s1[i] * s1[i];
      a += qv[i] * s0[i];
      b += qv[i] * s1[i];
    }

    __shared__ float s_red4[kFastWarps][4];
    __shared__ float s_w[2];
    sq0 = warp_reduce_sum(sq0);
    sq1 = warp_reduce_sum(sq1);
    a = warp_reduce_sum(a);
    b = warp_reduce_sum(b);
    if (lane == 0) {
      s_red4[warp][0] = sq0;
      s_red4[warp][1] = sq1;
      s_red4[warp][2] = a;
      s_red4[warp][3] = b;
    }
    __syncthreads();
    if (warp == 0) {
      float v0 = warp_reduce_sum((lane < kFastWarps) ? s_red4[lane][0] : 0.0f);
      float v1 = warp_reduce_sum((lane < kFastWarps) ? s_red4[lane][1] : 0.0f);
      float inv0 = rsqrt_approx_f32(v0 * kInvD + eps);
      float inv1 = rsqrt_approx_f32(v1 * kInvD + eps);
      float l0 = warp_reduce_sum((lane < kFastWarps) ? s_red4[lane][2] : 0.0f) * inv0;
      float l1 = warp_reduce_sum((lane < kFastWarps) ? s_red4[lane][3] : 0.0f) * inv1;
      if (lane == 0) {
        float ml = fmaxf(l0, l1);
        float e0 = expf(l0 - ml);
        float e1 = expf(l1 - ml);
        float inv_s = 1.0f / (e0 + e1);
        s_w[0] = e0 * inv_s * inv0;
        s_w[1] = e1 * inv_s * inv1;
      }
    }
    __syncthreads();

    const float w0 = s_w[0];
    const float w1 = s_w[1];
    float* dst_msg = msg_buf + static_cast<long long>(by) * out_q_stride + row_off;
    float mix[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      float m = 0.0f;
      m += w0 * s0[i];
      m += w1 * s1[i];
      mix[i] = m;
      dst_msg[tid + i * kFastBlock] = m;
    }

    if (by < 3) {
      __shared__ float s_r8[kFastWarps];
      float rstd = compute_rstd_nofma_2304(mix, eps, s_r8);
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
      __nv_bfloat16* cat_q = ((by == 0) ? cat0 : ((by == 1) ? cat1 : cat2))
                             + static_cast<long long>(t) * (2 * kFastD) + kFastD;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        cat_q[tid + i * kFastBlock] = __float2bfloat16_rn(__fmul_rn(mix[i], rstd));
      }
    } else {
      asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
    return;
  }

  // by >= 4: attn_res_mix q = by - 4 + register-fused s1 -> cat[0..D)
  const int q = by - 4;
  const bool is_swa = (in_stride == 0);
  __shared__ float s_red[kFastWarps][20];
  __shared__ float s_weights[5];
  const float* q_ptr = attn_q + static_cast<long long>(q) * kFastD;
  float* mix_row = mix_qkv + static_cast<long long>(q) * out_q_stride + row_off;
  const long long cat_off = static_cast<long long>(t) * (2 * kFastD);
  __nv_bfloat16* c0 = is_swa ? (cat0 + cat_off) : (((q == 0) ? cat0 : ((q == 1) ? cat1 : cat2)) + cat_off);
  __nv_bfloat16* c1 = cat1 + cat_off;
  __nv_bfloat16* c2 = cat2 + cat_off;

  switch (n_stack) {
    case 1:
      attn_res_mix_1q_s1_2304<false, 1>(
          stack, partial, nullptr, q_ptr, mix_row, c0, c1, c2, is_swa, stack_stride, t, eps, s_red, s_weights);
      return;
    case 2:
      attn_res_mix_1q_s1_2304<false, 2>(
          stack, partial, nullptr, q_ptr, mix_row, c0, c1, c2, is_swa, stack_stride, t, eps, s_red, s_weights);
      return;
    case 3:
      attn_res_mix_1q_s1_2304<false, 3>(
          stack, partial, nullptr, q_ptr, mix_row, c0, c1, c2, is_swa, stack_stride, t, eps, s_red, s_weights);
      return;
    default:
      __trap();
  }
}

// Fused Pre-Attention (attn_res_mix + cross_pass_attn_res + 3x s1) for Prefill (T >= 65).
// Grid: [T, 2, 1], Block: [256, 1, 1] (0 redundant reads of prev_blocks, mix_qkv, or msg_buf!).
//   by == 0: attn_res_mix (1q SWA or 3q Full) -> writes mix_qkv + register-fused s1 -> cat0/1/2[0..D)
//   by == 1: cross_pass_fast_2304 (4q) -> writes msg_buf[0..3] + register-fused s1 -> cat0/1/2[D..2D)
extern "C" __global__ __launch_bounds__(256, 3) void kern_pre_attn_mix_cross_s1_prefill(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ attn_q,
    const __nv_bfloat16* __restrict__ prev_blocks,
    const float* __restrict__ cross_q,
    float* __restrict__ mix_qkv,
    float* __restrict__ msg_buf,
    __nv_bfloat16* __restrict__ cat0,
    __nv_bfloat16* __restrict__ cat1,
    __nv_bfloat16* __restrict__ cat2,
    long long stack_stride,
    long long out_q_stride,
    long long in_stride,
    int T,
    int n_stack,
    float eps) {
  const int t = blockIdx.x;
  const int by = blockIdx.y;
  if (t >= T) return;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const long long row_off = static_cast<long long>(t) * kFastD;
  const long long cat_off = static_cast<long long>(t) * (2 * kFastD);

  if (by == 1) {
    constexpr float kInvD = 1.0f / static_cast<float>(kFastD);
    asm volatile("griddepcontrol.wait;" ::: "memory");

    const __nv_bfloat16* p0 = prev_blocks + row_off;
    const __nv_bfloat16* p1 = prev_blocks + stack_stride + row_off;
    float s0[kElemsPerThread], s1[kElemsPerThread];
    float sq0 = 0.0f, sq1 = 0.0f;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      int d = tid + i * kFastBlock;
      s0[i] = __bfloat162float(p0[d]);
      s1[i] = __bfloat162float(p1[d]);
      sq0 += s0[i] * s0[i];
      sq1 += s1[i] * s1[i];
    }

    float d0_q[4], d1_q[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float* q_ptr = cross_q + static_cast<long long>(q) * kFastD;
      float a = 0.0f, b = 0.0f;
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        float qv = __ldg(&q_ptr[tid + i * kFastBlock]);
        a += qv * s0[i];
        b += qv * s1[i];
      }
      d0_q[q] = a;
      d1_q[q] = b;
    }

    __shared__ float s_red10[kFastWarps][10];
    __shared__ float s_w[4][2];
    {
      float r0 = warp_reduce_sum(sq0);
      float r1 = warp_reduce_sum(sq1);
      if (lane == 0) {
        s_red10[warp][0] = r0;
        s_red10[warp][1] = r1;
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float ra = warp_reduce_sum(d0_q[q]);
        float rb = warp_reduce_sum(d1_q[q]);
        if (lane == 0) {
          s_red10[warp][2 + 2 * q] = ra;
          s_red10[warp][3 + 2 * q] = rb;
        }
      }
    }
    __syncthreads();

    if (warp == 0) {
      float v0 = warp_reduce_sum((lane < kFastWarps) ? s_red10[lane][0] : 0.0f);
      float v1 = warp_reduce_sum((lane < kFastWarps) ? s_red10[lane][1] : 0.0f);
      float inv0 = rsqrt_approx_f32(v0 * kInvD + eps);
      float inv1 = rsqrt_approx_f32(v1 * kInvD + eps);
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        float l0 = warp_reduce_sum((lane < kFastWarps) ? s_red10[lane][2 + 2 * q] : 0.0f) * inv0;
        float l1 = warp_reduce_sum((lane < kFastWarps) ? s_red10[lane][3 + 2 * q] : 0.0f) * inv1;
        if (lane == 0) {
          float ml = fmaxf(l0, l1);
          float e0 = expf(l0 - ml);
          float e1 = expf(l1 - ml);
          float inv_s = 1.0f / (e0 + e1);
          s_w[q][0] = e0 * inv_s * inv0;
          s_w[q][1] = e1 * inv_s * inv1;
        }
      }
    }
    __syncthreads();

    __shared__ float s_r8[kFastWarps];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      float w0 = s_w[q][0];
      float w1 = s_w[q][1];
      float* dst_msg = msg_buf + static_cast<long long>(q) * out_q_stride + row_off;
      float mix[kElemsPerThread];
#pragma unroll
      for (int i = 0; i < kElemsPerThread; ++i) {
        float m = 0.0f;
        m += w0 * s0[i];
        m += w1 * s1[i];
        mix[i] = m;
        dst_msg[tid + i * kFastBlock] = m;
      }
      if (q < 3) {
        float rstd = compute_rstd_nofma_2304(mix, eps, s_r8);
        __nv_bfloat16* cat_q = ((q == 0) ? cat0 : ((q == 1) ? cat1 : cat2)) + cat_off + kFastD;
#pragma unroll
        for (int i = 0; i < kElemsPerThread; ++i) {
          cat_q[tid + i * kFastBlock] = __float2bfloat16_rn(__fmul_rn(mix[i], rstd));
        }
      }
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }

  // by == 0: attn_res_mix (1q SWA or 3q Full) + register-fused s1
  __shared__ float s_red[kFastWarps][20];
  if (in_stride == 0) {
    __shared__ float s_weights[5];
    switch (n_stack) {
      case 1:
        attn_res_mix_1q_s1_2304<false, 1>(
            stack, partial, nullptr, attn_q, mix_qkv + row_off,
            cat0 + cat_off, cat1 + cat_off, cat2 + cat_off, true, stack_stride, t, eps, s_red, s_weights);
        return;
      case 2:
        attn_res_mix_1q_s1_2304<false, 2>(
            stack, partial, nullptr, attn_q, mix_qkv + row_off,
            cat0 + cat_off, cat1 + cat_off, cat2 + cat_off, true, stack_stride, t, eps, s_red, s_weights);
        return;
      case 3:
        attn_res_mix_1q_s1_2304<false, 3>(
            stack, partial, nullptr, attn_q, mix_qkv + row_off,
            cat0 + cat_off, cat1 + cat_off, cat2 + cat_off, true, stack_stride, t, eps, s_red, s_weights);
        return;
      default:
        __trap();
    }
  } else {
    __shared__ float s_weights3[3][5];
    switch (n_stack) {
      case 1:
        attn_res_mix_3q_s1_2304<1>(
            stack, partial, attn_q, mix_qkv, cat0 + cat_off, cat1 + cat_off, cat2 + cat_off,
            stack_stride, out_q_stride, t, eps, s_red, s_weights3);
        return;
      case 2:
        attn_res_mix_3q_s1_2304<2>(
            stack, partial, attn_q, mix_qkv, cat0 + cat_off, cat1 + cat_off, cat2 + cat_off,
            stack_stride, out_q_stride, t, eps, s_red, s_weights3);
        return;
      case 3:
        attn_res_mix_3q_s1_2304<3>(
            stack, partial, attn_q, mix_qkv, cat0 + cat_off, cat1 + cat_off, cat2 + cat_off,
            stack_stride, out_q_stride, t, eps, s_red, s_weights3);
        return;
      default:
        __trap();
    }
  }
}

// Fused Pre-MLP (attn_res_mix / attn_res_addend_mix + loop_controller.s1) for ALL T.
// Grid: [T, 2, 1], Block: [256, 1, 1]
//   by == 0: attn_res_{mix,addend_mix} -> writes mix_mlp + register-fused s1 -> cat[0..D)
//   by == 1: normalizes msg_buf[3] (strict no-FMA) -> writes cat[D..2D)
template <bool HAS_ADDEND>
__device__ __forceinline__ void pre_mlp_mix_s1_impl(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ mlp_q,
    const float* __restrict__ msg_f32,
    float* __restrict__ mix_mlp,
    __nv_bfloat16* __restrict__ cat_bf16,
    long long stack_stride,
    int T,
    int n_stack,
    float eps) {
  const int t = blockIdx.x;
  const int by = blockIdx.y;
  if (t >= T) return;

  const int tid = threadIdx.x;
  const long long row_off = static_cast<long long>(t) * kFastD;
  const long long cat_off = static_cast<long long>(t) * (2 * kFastD);

  if (by == 1) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    const float* src = msg_f32 + row_off;
    float x[kElemsPerThread];
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) x[i] = src[tid + i * kFastBlock];
    __shared__ float s_r8[kFastWarps];
    float rstd = compute_rstd_nofma_2304(x, eps, s_r8);
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    __nv_bfloat16* dst = cat_bf16 + cat_off + kFastD;
#pragma unroll
    for (int i = 0; i < kElemsPerThread; ++i) {
      dst[tid + i * kFastBlock] = __float2bfloat16_rn(__fmul_rn(x[i], rstd));
    }
    return;
  }

  __shared__ float s_red[kFastWarps][20];
  __shared__ float s_weights[5];
  switch (n_stack) {
    case 1:
      attn_res_mix_1q_s1_2304<HAS_ADDEND, 1>(
          stack, partial, addend, mlp_q, mix_mlp + row_off,
          cat_bf16 + cat_off, nullptr, nullptr, false, stack_stride, t, eps, s_red, s_weights);
      return;
    case 2:
      attn_res_mix_1q_s1_2304<HAS_ADDEND, 2>(
          stack, partial, addend, mlp_q, mix_mlp + row_off,
          cat_bf16 + cat_off, nullptr, nullptr, false, stack_stride, t, eps, s_red, s_weights);
      return;
    case 3:
      attn_res_mix_1q_s1_2304<HAS_ADDEND, 3>(
          stack, partial, addend, mlp_q, mix_mlp + row_off,
          cat_bf16 + cat_off, nullptr, nullptr, false, stack_stride, t, eps, s_red, s_weights);
      return;
    default:
      __trap();
  }
}

extern "C" __global__ __launch_bounds__(256, 4) void kern_pre_mlp_mix_s1_f32_bf16(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const float* __restrict__ mlp_q,
    const float* __restrict__ msg_f32,
    float* __restrict__ mix_mlp,
    __nv_bfloat16* __restrict__ cat_bf16,
    long long stack_stride,
    int T,
    int n_stack,
    float eps) {
  pre_mlp_mix_s1_impl<false>(
      stack, partial, nullptr, mlp_q, msg_f32, mix_mlp, cat_bf16, stack_stride, T, n_stack, eps);
}

extern "C" __global__ __launch_bounds__(256, 4) void kern_pre_mlp_addend_mix_s1_f32_bf16(
    const __nv_bfloat16* __restrict__ stack,
    const __nv_bfloat16* __restrict__ partial,
    const __nv_bfloat16* __restrict__ addend,
    const float* __restrict__ mlp_q,
    const float* __restrict__ msg_f32,
    float* __restrict__ mix_mlp,
    __nv_bfloat16* __restrict__ cat_bf16,
    long long stack_stride,
    int T,
    int n_stack,
    float eps) {
  pre_mlp_mix_s1_impl<true>(
      stack, partial, addend, mlp_q, msg_f32, mix_mlp, cat_bf16, stack_stride, T, n_stack, eps);
}
