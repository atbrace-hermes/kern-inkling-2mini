// SPDX-License-Identifier: Apache-2.0
// SM120 embedding, last-row extract, FP32 softcap+vocab mask, FP32 two-stage argmax,
// and FP32 sampling copy helper for Inkling-2-Mini.
// Provenance recorded in upstreams.json (adapted from kern-qwen38 + refs/sglang).

#include <cuda_bf16.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

// Adapted from kern-qwen38/source/embedding.cu with out-of-bounds __trap().
// Grid: [tokens, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_embedding_bf16(
    const int64_t* __restrict__ ids,
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ out,
    int hidden,
    int vocab) {
  int row = blockIdx.x;
  int64_t id = ids[row];
  if (id < 0 || id >= vocab) __trap();
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    out[static_cast<int64_t>(row) * hidden + j] = table[id * hidden + j];
  }
}

// Adapted from kern-qwen38/source/copy_rows.cu:kern_last_row_bf16.
// Grid: [1, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void kern_last_row_bf16(
    __nv_bfloat16* __restrict__ dst,
    const __nv_bfloat16* __restrict__ src,
    int src_stride,
    int width,
    int rows) {
  if (rows <= 0) return;
  const __nv_bfloat16* s = src + static_cast<long long>(rows - 1) * src_stride;
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    dst[j] = s[j];
  }
}

// In-place FP32 logit softcap (softcap * tanh(x / softcap)) and padded vocab masking
// (col >= unpadded_vocab -> -inf) on [rows, padded_vocab].
// Grid: [rows, ceil_div(padded_vocab, 1024), 1], Block: [256, 1, 1]
extern "C" __global__ void kern_softcap_tanh_mask_f32(
    float* __restrict__ logits,
    int rows,
    int padded_vocab,
    int unpadded_vocab,
    float softcap) {
  const int row = blockIdx.x;
  if (row >= rows) return;

  const int base = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
  if (base >= padded_vocab) return;

  float* row_ptr = logits + static_cast<long long>(row) * padded_vocab;
  const float inv_cap = 1.0f / softcap;
  const float neg_inf = -__builtin_inff();

  if (base + 3 < padded_vocab && ((reinterpret_cast<uintptr_t>(row_ptr + base) & 15u) == 0u)) {
    float4 v = *reinterpret_cast<const float4*>(row_ptr + base);
    if (base + 3 < unpadded_vocab) {
      v.x = softcap * tanhf(v.x * inv_cap);
      v.y = softcap * tanhf(v.y * inv_cap);
      v.z = softcap * tanhf(v.z * inv_cap);
      v.w = softcap * tanhf(v.w * inv_cap);
    } else if (base >= unpadded_vocab) {
      v.x = neg_inf;
      v.y = neg_inf;
      v.z = neg_inf;
      v.w = neg_inf;
    } else {
      v.x = (base + 0 < unpadded_vocab) ? (softcap * tanhf(v.x * inv_cap)) : neg_inf;
      v.y = (base + 1 < unpadded_vocab) ? (softcap * tanhf(v.y * inv_cap)) : neg_inf;
      v.z = (base + 2 < unpadded_vocab) ? (softcap * tanhf(v.z * inv_cap)) : neg_inf;
      v.w = (base + 3 < unpadded_vocab) ? (softcap * tanhf(v.w * inv_cap)) : neg_inf;
    }
    *reinterpret_cast<float4*>(row_ptr + base) = v;
  } else {
    for (int k = 0; k < 4; ++k) {
      int col = base + k;
      if (col < padded_vocab) {
        row_ptr[col] = (col < unpadded_vocab) ? (softcap * tanhf(row_ptr[col] * inv_cap)) : neg_inf;
      }
    }
  }
}

namespace {

struct ArgmaxPair {
  float v;
  int i;
};

__device__ __forceinline__ ArgmaxPair pick(ArgmaxPair a, ArgmaxPair b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

template <int BLOCK>
__device__ __forceinline__ ArgmaxPair block_reduce_argmax(ArgmaxPair best) {
  __shared__ float sv[BLOCK];
  __shared__ int si[BLOCK];
  sv[threadIdx.x] = best.v;
  si[threadIdx.x] = best.i;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      ArgmaxPair o = {sv[threadIdx.x + stride], si[threadIdx.x + stride]};
      ArgmaxPair m = pick({sv[threadIdx.x], si[threadIdx.x]}, o);
      sv[threadIdx.x] = m.v;
      si[threadIdx.x] = m.i;
    }
    __syncthreads();
  }
  return {sv[0], si[0]};
}

}  // namespace

// Two-stage FP32 greedy argmax; ties choose the smallest vocabulary index.
// Adapted from kern-qwen38/source/argmax.cu.
// Grid: [tokens, NB, 1], Block: [1024, 1, 1]
extern "C" __global__ void kern_argmax_partial_f32(
    const float* __restrict__ logits,
    float* __restrict__ pmax,
    int* __restrict__ pidx,
    int n) {
  int nb = gridDim.y;
  int chunk = (n + nb - 1) / nb;
  int lo = blockIdx.y * chunk;
  int hi = min(lo + chunk, n);
  const float* row = logits + static_cast<long long>(blockIdx.x) * n;
  ArgmaxPair best = {-__builtin_inff(), INT_MAX};
  for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
    best = pick(best, {row[j], j});
  }
  best = block_reduce_argmax<1024>(best);
  if (threadIdx.x == 0) {
    pmax[blockIdx.x * nb + blockIdx.y] = best.v;
    pidx[blockIdx.x * nb + blockIdx.y] = best.i;
  }
}

// Grid: [tokens, 1, 1], Block: [64, 1, 1]
extern "C" __global__ void kern_argmax_final_i64(
    const float* __restrict__ pmax,
    const int* __restrict__ pidx,
    long long* __restrict__ out,
    int nb) {
  ArgmaxPair best = {-__builtin_inff(), INT_MAX};
  for (int k = threadIdx.x; k < nb; k += blockDim.x) {
    best = pick(best, {pmax[blockIdx.x * nb + k], pidx[blockIdx.x * nb + k]});
  }
  best = block_reduce_argmax<64>(best);
  if (threadIdx.x == 0) {
    out[blockIdx.x] = best.i;
  }
}

// Copy FP32 logits into FlashInfer sampling scratch probs (preserving logits).
// Grid: [seqs, ceil_div(vocab, 1024), 1], Block: [1024, 1, 1]
extern "C" __global__ void kern_sample_copy_f32(
    const float* __restrict__ logits,
    float* __restrict__ probs,
    int vocab) {
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < vocab) {
    size_t at = static_cast<size_t>(blockIdx.x) * vocab + i;
    probs[at] = logits[at];
  }
}
