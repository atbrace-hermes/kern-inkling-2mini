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
