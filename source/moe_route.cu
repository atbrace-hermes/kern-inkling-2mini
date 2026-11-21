// SPDX-License-Identifier: Apache-2.0
// SM120 MoE routing kernels for Inkling-2-Mini (E=128, TOPK=8, route_scale=8.0):
//   - moe_router_topk: sigmoid(logits) + bias -> deterministic top-8 -> log-sigmoid softmax * 8.0
//   - moe_route_metadata_small: single-CTA counting sort for T <= 1024 (SMEM <= 66.5 KiB)
//   - moe_route_count / moe_route_scan / moe_route_scatter: multi-CTA counting sort for arbitrary T

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr int kNumExperts = 128;
constexpr int kTopK = 8;
constexpr float kRouteScale = 8.0f;

__device__ __forceinline__ uint32_t ord_f32(float f) {
  uint32_t b = __float_as_uint(f);
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ unsigned same_lanes(int expert) {
  unsigned same = 0xFFFFFFFFu;
#pragma unroll
  for (int bit = 0; bit < 7; ++bit) {
    bool set = (expert >> bit) & 1;
    unsigned lanes = __ballot_sync(0xFFFFFFFFu, set);
    same &= set ? lanes : ~lanes;
  }
  // Distinguish out-of-range sentinel (>= 128) from valid experts (0..127)
  bool oob = (expert >= kNumExperts);
  unsigned oob_lanes = __ballot_sync(0xFFFFFFFFu, oob);
  same &= oob ? oob_lanes : ~oob_lanes;
  return same;
}

__device__ __forceinline__ int block_exclusive_scan_128(int* values, int* warp_sums) {
  constexpr int kWarps = 32;
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;
  int v = (threadIdx.x < kNumExperts) ? values[threadIdx.x] : 0;
  int x = v;
#pragma unroll
  for (int s = 1; s < 32; s *= 2) {
    int y = __shfl_up_sync(0xFFFFFFFFu, x, s);
    if (lane >= s) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int w = warp_sums[lane];
#pragma unroll
    for (int s = 1; s < 32; s *= 2) {
      int y = __shfl_up_sync(0xFFFFFFFFu, w, s);
      if (lane >= s) w += y;
    }
    warp_sums[lane] = w;
  }
  __syncthreads();
  int total = warp_sums[kWarps - 1];
  if (threadIdx.x < kNumExperts) {
    values[threadIdx.x] = x - v + (warp ? warp_sums[warp - 1] : 0);
  }
  __syncthreads();
  return total;
}

}  // namespace

// Grid: (ceil_div(T, 8), 1, 1), Block: (256, 1, 1) — 1 warp (32 lanes x 4 experts = 128) per token.
extern "C" __global__ void __launch_bounds__(256) moe_router_topk(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int* __restrict__ ids,
    float* __restrict__ weights,
    int T) {
#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int t = blockIdx.x * 8 + warp;

  if (t < T) {
    const float4 lv = *reinterpret_cast<const float4*>(logits + int64_t(t) * kNumExperts + lane * 4);
    const float4 bv = *reinterpret_cast<const float4*>(bias + lane * 4);
    const float l4[4] = {lv.x, lv.y, lv.z, lv.w};
    const float b4[4] = {bv.x, bv.y, bv.z, bv.w};

    uint64_t keys[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      int e = lane * 4 + j;
      float sig = __fdiv_rn(1.0f, 1.0f + expf(-l4[j]));
      float sel = sig + b4[j];
      keys[j] = (uint64_t(ord_f32(sel)) << 16) | uint64_t(kNumExperts - e);
    }

    int pick_e = 0;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      uint64_t m = max(max(keys[0], keys[1]), max(keys[2], keys[3]));
#pragma unroll
      for (int s = 16; s >= 1; s >>= 1) {
        uint64_t peer = __shfl_xor_sync(0xFFFFFFFFu, m, s);
        if (peer > m) m = peer;
      }
      int my_e = -1;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (keys[j] == m) {
          my_e = lane * 4 + j;
          keys[j] = 0ULL;
        }
      }
      int winner_lane = __ffs(__ballot_sync(0xFFFFFFFFu, my_e >= 0)) - 1;
      int chosen_e = __shfl_sync(0xFFFFFFFFu, my_e, winner_lane);
      if (lane == k) pick_e = chosen_e;
    }

    float log_p = -INFINITY;
    if (lane < kTopK) {
      float z = logits[int64_t(t) * kNumExperts + pick_e];
      log_p = fminf(z, 0.0f) - log1pf(expf(-fabsf(z)));
    }
    float max_lp = log_p;
    max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 1));
    max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 2));
    max_lp = fmaxf(max_lp, __shfl_xor_sync(0xFFFFFFFFu, max_lp, 4));

    float s = (lane < kTopK) ? expf(log_p - max_lp) : 0.0f;
    // Exact ((s0 + s2) + (s1 + s3)) + ((s4 + s6) + (s5 + s7)) summation order
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 2);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 1);
    s += __shfl_xor_sync(0xFFFFFFFFu, s, 4);
    float total = __shfl_sync(0xFFFFFFFFu, s, 0);

    if (lane < kTopK) {
      float logsumexp = max_lp + logf(total);
      float w = expf(log_p - logsumexp) * kRouteScale;
      ids[int64_t(t) * kTopK + lane] = pick_e;
      weights[int64_t(t) * kTopK + lane] = w;
    }
  }
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Single-CTA stable counting sort for n = T * 8 <= 8192 (T <= 1024).
// Dynamic SMEM required: ((n + 31) / 32) * 256 + 1024 bytes (<= 66560 B <= 101376 B on SM120).
// Grid: (1, 1, 1), Block: (1024, 1, 1)
extern "C" __global__ void __launch_bounds__(1024) moe_route_metadata_small(
    const int* __restrict__ ids,
    const float* __restrict__ weights,
    int* __restrict__ sorted_ids,
    int* __restrict__ src2dst,
    int* __restrict__ dst2src,
    float* __restrict__ gammas,
    int* __restrict__ counts,
    int* __restrict__ m_indptr,
    int n) {
  constexpr int kThreads = 1024;
  constexpr int kWarps = 32;
  constexpr int kMaxRounds = 8;

  extern __shared__ __align__(16) unsigned char smem[];
  int const chunks = (n + 31) / 32;
  uint16_t* chunk_counts = reinterpret_cast<uint16_t*>(smem);
  int* expert_start = reinterpret_cast<int*>(smem + size_t(chunks) * kNumExperts * sizeof(uint16_t));
  __shared__ int warp_sums[kWarps];

  int const lane = threadIdx.x & 31;
  int const warp = threadIdx.x >> 5;

  for (int i = threadIdx.x; i < chunks * (kNumExperts / 8); i += kThreads) {
    reinterpret_cast<uint4*>(chunk_counts)[i] = make_uint4(0, 0, 0, 0);
  }
#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int expert[kMaxRounds], rank[kMaxRounds];
  float weight[kMaxRounds];
#pragma unroll
  for (int r = 0; r < kMaxRounds; ++r) {
    int p = r * kThreads + threadIdx.x;
    expert[r] = (p < n) ? ids[p] : kNumExperts;
    weight[r] = (p < n) ? weights[p] : 0.0f;
  }
  __syncthreads();

#pragma unroll
  for (int r = 0; r < kMaxRounds; ++r) {
    if (r * kThreads + warp * 32 >= n) break;
    unsigned same = same_lanes(expert[r]);
    rank[r] = __popc(same & ((1u << lane) - 1));
    if (expert[r] < kNumExperts && lane == __ffs(same) - 1) {
      chunk_counts[(r * kWarps + warp) * kNumExperts + expert[r]] = uint16_t(__popc(same));
    }
  }
  __syncthreads();

  if (threadIdx.x < kNumExperts) {
    int e = threadIdx.x;
    int sum = 0;
    for (int c = 0; c < chunks; ++c) {
      int cnt = chunk_counts[c * kNumExperts + e];
      chunk_counts[c * kNumExperts + e] = uint16_t(sum);
      sum += cnt;
    }
    expert_start[e] = sum;
  }
  __syncthreads();

  int total = block_exclusive_scan_128(expert_start, warp_sums);
  if (threadIdx.x == 0) expert_start[kNumExperts] = total;
  __syncthreads();

  if (threadIdx.x <= kNumExperts) {
    m_indptr[threadIdx.x] = expert_start[threadIdx.x];
  }
  if (threadIdx.x < kNumExperts) {
    counts[threadIdx.x] = expert_start[threadIdx.x + 1] - expert_start[threadIdx.x];
  }

#pragma unroll
  for (int r = 0; r < kMaxRounds; ++r) {
    int p = r * kThreads + threadIdx.x;
    if (p < n) {
      int e = expert[r];
      int dst = expert_start[e] + chunk_counts[(p >> 5) * kNumExperts + e] + rank[r];
      sorted_ids[dst] = e;
      dst2src[dst] = p;
      gammas[dst] = weight[r];
      src2dst[p] = dst;
    }
  }
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Multi-CTA fallback Stage 1: per-block histogram over 256-id chunks.
// Grid: (ceil_div(n, 256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) moe_route_count(
    const int* __restrict__ ids,
    int* __restrict__ block_counts,
    int n) {
  __shared__ int cnt[kNumExperts];
  if (threadIdx.x < kNumExperts) cnt[threadIdx.x] = 0;
  __syncthreads();
#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int p = blockIdx.x * 256 + threadIdx.x;
  if (p < n) {
    int e = ids[p];
    if (e >= 0 && e < kNumExperts) atomicAdd(&cnt[e], 1);
  }
  __syncthreads();
  if (threadIdx.x < kNumExperts) {
    block_counts[blockIdx.x * kNumExperts + threadIdx.x] = cnt[threadIdx.x];
  }
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Multi-CTA fallback Stage 2: scan per-block counts into m_indptr, counts, and block_offsets.
// Grid: (1, 1, 1), Block: (128, 1, 1)
extern "C" __global__ void __launch_bounds__(128) moe_route_scan(
    const int* __restrict__ block_counts,
    int* __restrict__ block_offsets,
    int* __restrict__ counts,
    int* __restrict__ m_indptr,
    int n) {
  int const num_blocks = (n + 255) / 256;
  __shared__ int expert_start[kNumExperts + 1];
  __shared__ int warp_sums[4];
  int e = threadIdx.x;
  int lane = e & 31;
  int warp = e >> 5;

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  int run = 0;
  for (int b = 0; b < num_blocks; ++b) {
    int c = block_counts[b * kNumExperts + e];
    block_offsets[b * kNumExperts + e] = run;
    run += c;
  }
  counts[e] = run;

  int x = run;
#pragma unroll
  for (int s = 1; s < 32; s *= 2) {
    int y = __shfl_up_sync(0xFFFFFFFFu, x, s);
    if (lane >= s) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0 && lane < 4) {
    int w = warp_sums[lane];
#pragma unroll
    for (int s = 1; s < 4; s *= 2) {
      int y = __shfl_up_sync(0xFu, w, s);
      if (lane >= s) w += y;
    }
    warp_sums[lane] = w;
  }
  __syncthreads();
  int base = x - run + (warp ? warp_sums[warp - 1] : 0);
  expert_start[e] = base;
  if (e == 0) expert_start[kNumExperts] = warp_sums[3];
  m_indptr[e] = base;
  if (e == 0) m_indptr[kNumExperts] = expert_start[kNumExperts];
  __syncthreads();

  for (int b = 0; b < num_blocks; ++b) {
    block_offsets[b * kNumExperts + e] += base;
  }
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Multi-CTA fallback Stage 3: stable scatter using per-block offsets.
// Grid: (ceil_div(n, 256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) moe_route_scatter(
    const int* __restrict__ ids,
    const float* __restrict__ weights,
    const int* __restrict__ block_offsets,
    int* __restrict__ sorted_ids,
    int* __restrict__ src2dst,
    int* __restrict__ dst2src,
    float* __restrict__ gammas,
    int n) {
  __shared__ uint16_t warp_counts[8][kNumExperts];
  int warp = threadIdx.x >> 5;
  int lane = threadIdx.x & 31;
  int p = blockIdx.x * 256 + threadIdx.x;

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  for (int i = threadIdx.x; i < 8 * (kNumExperts / 8); i += 256) {
    reinterpret_cast<uint4*>(&warp_counts[0][0])[i] = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();

  int e = (p < n) ? ids[p] : kNumExperts;
  float w = (p < n) ? weights[p] : 0.0f;
  unsigned same = same_lanes(e);
  int rank = __popc(same & ((1u << lane) - 1));
  if (e < kNumExperts && lane == __ffs(same) - 1) {
    warp_counts[warp][e] = uint16_t(__popc(same));
  }
  __syncthreads();

  if (threadIdx.x < kNumExperts) {
    int sum = 0;
#pragma unroll
    for (int g = 0; g < 8; ++g) {
      int c = warp_counts[g][threadIdx.x];
      warp_counts[g][threadIdx.x] = uint16_t(sum);
      sum += c;
    }
  }
  __syncthreads();

  if (p < n) {
    int dst = block_offsets[blockIdx.x * kNumExperts + e] + warp_counts[warp][e] + rank;
    sorted_ids[dst] = e;
    dst2src[dst] = p;
    gammas[dst] = w;
    src2dst[p] = dst;
  }
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
