// SPDX-License-Identifier: Apache-2.0
// SM120 derive kernels for DSpark MTP & Inkling2MarkovHead.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

static __device__ __forceinline__ __nv_bfloat16 d2bf16(double v) {
#if __CUDA_ARCH__ >= 800
  return __double2bfloat16(v);
#else
  return __float2bfloat16_rn((float)v);
#endif
}

// Tiled FP64 matrix multiply: fold_w [V, R] = (lm_head [V, H] @ hidden_w [H, R]).bfloat16()
// Grid: [ceil_div(V, 32), ceil_div(R, 32), 1], Block: [32, 8, 1]
extern "C" __global__ void dspark_fold_fp64(
    const __nv_bfloat16* __restrict__ lm_head,
    const __nv_bfloat16* __restrict__ hidden_w,
    __nv_bfloat16* __restrict__ fold_w,
    int V, int H, int R) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int BK = 16;

  __shared__ double As[BM][BK];
  __shared__ double Bs[BK][BN];

  const int bm = blockIdx.x;
  const int bn = blockIdx.y;
  const int tx = threadIdx.x; // 0..31
  const int ty = threadIdx.y; // 0..7
  const int tid = ty * 32 + tx;

  double acc[4] = {0.0, 0.0, 0.0, 0.0};
  const int col = bn * BN + tx;

  for (int k0 = 0; k0 < H; k0 += BK) {
    // Load As[32][16] (512 elements / 256 threads = 2 per thread)
    for (int e = tid; e < BM * BK; e += 256) {
      int r = e / BK;
      int k = e % BK;
      int gr = bm * BM + r;
      int gk = k0 + k;
      As[r][k] = (gr < V && gk < H) ? (double)__bfloat162float(lm_head[(long long)gr * H + gk]) : 0.0;
    }
    // Load Bs[16][32] (512 elements / 256 threads = 2 per thread)
    for (int e = tid; e < BK * BN; e += 256) {
      int k = e / BN;
      int c = e % BN;
      int gk = k0 + k;
      int gc = bn * BN + c;
      Bs[k][c] = (gk < H && gc < R) ? (double)__bfloat162float(hidden_w[(long long)gk * R + gc]) : 0.0;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      double bval = Bs[k][tx];
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        acc[i] += As[ty * 4 + i][k] * bval;
      }
    }
    __syncthreads();
  }

  if (col < R) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int row = bm * BM + ty * 4 + i;
      if (row < V) {
        fold_w[(long long)row * R + col] = d2bf16(acc[i]);
      }
    }
  }
}

// Warp-per-row FP64 matvec: fold_b [V] = (lm_head [V, H] @ hidden_b [H]).bfloat16()
// Grid: [ceil_div(V, 8), 1, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_fold_bias_fp64(
    const __nv_bfloat16* __restrict__ lm_head,
    const __nv_bfloat16* __restrict__ hidden_b,
    __nv_bfloat16* __restrict__ fold_b,
    int V, int H) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * 8 + warp;
  if (row >= V) return;

  const __nv_bfloat16* rptr = lm_head + (long long)row * H;
  double sum = 0.0;
  for (int k = lane; k < H; k += 32) {
    sum += (double)__bfloat162float(rptr[k]) * (double)__bfloat162float(hidden_b[k]);
  }
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, off);
  }
  if (lane == 0) {
    fold_b[row] = d2bf16(sum);
  }
}

// Prepare derived weights:
// - fuse_alpha [n_alpha_rows, n_taps] BF16 = softmax(fuse_weight.float(), dim=-1).bfloat16()
// - w_state [rows_rnn, state_cols] BF16 = rnn_proj_w[:, :state_cols]
// - w_ctx [rows_rnn, ctx_cols] BF16 = rnn_proj_w[:, state_cols:]
// - rnn_bias [rows_rnn] FP32 = rnn_proj_b.float()
// - inv_freq [head_dim/2] FP32 = 1.0f / powf(rope_theta, 2*k / head_dim)
// Grid: [ceil_div(rows_rnn * (state_cols + ctx_cols), 256), 1, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_prep_weights(
    const __nv_bfloat16* __restrict__ fuse_weight,
    const __nv_bfloat16* __restrict__ rnn_proj_w,
    const __nv_bfloat16* __restrict__ rnn_proj_b,
    __nv_bfloat16* __restrict__ fuse_alpha,
    __nv_bfloat16* __restrict__ w_state,
    __nv_bfloat16* __restrict__ w_ctx,
    float* __restrict__ rnn_bias,
    float* __restrict__ inv_freq,
    int n_alpha_rows, int n_taps,
    int rows_rnn, int state_cols, int ctx_cols,
    int head_dim, float rope_theta) {
  const long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  const int total_cols = state_cols + ctx_cols;
  const long long total_w = (long long)rows_rnn * total_cols;

  if (idx < total_w) {
    int r = (int)(idx / total_cols);
    int c = (int)(idx % total_cols);
    if (c < state_cols) {
      w_state[(long long)r * state_cols + c] = rnn_proj_w[idx];
    } else {
      w_ctx[(long long)r * ctx_cols + (c - state_cols)] = rnn_proj_w[idx];
    }
  }

  if (idx < rows_rnn) {
    rnn_bias[idx] = __bfloat162float(rnn_proj_b[idx]);
  }

  if (idx < n_alpha_rows) {
    int r = (int)idx;
    float vals[16];
    float mx = -__builtin_inff();
    for (int t = 0; t < n_taps; ++t) {
      vals[t] = __bfloat162float(fuse_weight[r * n_taps + t]);
      if (vals[t] > mx) mx = vals[t];
    }
    float denom = 0.0f;
    for (int t = 0; t < n_taps; ++t) {
      vals[t] = expf(vals[t] - mx);
      denom += vals[t];
    }
    for (int t = 0; t < n_taps; ++t) {
      fuse_alpha[r * n_taps + t] = __float2bfloat16_rn(vals[t] / denom);
    }
  }

  const int half_d = head_dim >> 1;
  if (idx < half_d) {
    inv_freq[idx] = 1.0f / powf(rope_theta, (float)(2 * idx) / (float)head_dim);
  }
}
