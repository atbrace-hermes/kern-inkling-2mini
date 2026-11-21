// SPDX-License-Identifier: Apache-2.0
// SM120 state capture, embedding, verify vocab masking, sampling expansion,
// and rollback-safe sconv / SWA / f_rnn commit kernels for DSpark MTP.
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

using B = __nv_bfloat16;

// Capture [rows, hidden] into taps[rows, num_taps, hidden] at tap_idx.
// Grid: [rows, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_tap_capture(
    const B* __restrict__ src, B* __restrict__ taps,
    int tap_idx, int hidden, int num_taps) {
  const long long row = blockIdx.x;
  const B* s = src + row * hidden;
  B* d = taps + (row * num_taps + tap_idx) * hidden;
  if ((hidden & 7) == 0 && (((uintptr_t)s | (uintptr_t)d) & 15) == 0) {
    const int n4 = hidden >> 3;
    const uint4* s4 = reinterpret_cast<const uint4*>(s);
    uint4* d4 = reinterpret_cast<uint4*>(d);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      d4[i] = s4[i];
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      d[i] = s[i];
    }
  }
}

// Build DSpark block embeddings:
// - slot 0: embed_norm(embed_table[anchor[seq]])
// - slots 1..block_size-1: mask_embed
// Grid: [seqs, block_size, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_embed_block(
    const long long* __restrict__ anchor,
    const B* __restrict__ embed_table,
    const B* __restrict__ embed_norm_w,
    const B* __restrict__ mask_embed,
    B* __restrict__ out,
    int block_size, int hidden, float eps) {
  const int seq = blockIdx.x;
  const int slot = blockIdx.y;
  B* dst = out + ((long long)seq * block_size + slot) * hidden;

  if (slot > 0) {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      dst[i] = mask_embed[i];
    }
    return;
  }

  const long long tok = anchor[seq];
  const B* emb = embed_table + tok * hidden;
  float sq = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = __bfloat162float(emb[i]);
    sq += v * v;
  }

  __shared__ float warp_sums[8];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    sq += __shfl_down_sync(0xFFFFFFFFu, sq, off);
  }
  if (lane == 0) warp_sums[warp] = sq;
  __syncthreads();

  if (warp == 0) {
    float s = (lane < 8) ? warp_sums[lane] : 0.0f;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1) {
      s += __shfl_down_sync(0xFFFFFFFFu, s, off);
    }
    if (lane == 0) warp_sums[0] = rsqrtf(s / (float)hidden + eps);
  }
  __syncthreads();

  const float inv_rms = warp_sums[0];
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    float v = __bfloat162float(emb[i]) * inv_rms * __bfloat162float(embed_norm_w[i]);
    dst[i] = __float2bfloat16_rn(v);
  }
}

// Gather last committed token's taps per sequence:
// - prefill (stride_rows=0, is_round=0): row = tokens_total - 1
// - decode (stride_rows=1, is_round=0): row = seq
// - round (stride_rows=8, is_round=1): row = seq * stride_rows + (count[seq] - 1)
// Grid: [seqs, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_gather_last_tap(
    const B* __restrict__ taps,
    const int* __restrict__ count,
    B* __restrict__ out_taps,
    int stride_rows, int is_round, int tokens_total, int tap_elems) {
  const int seq = blockIdx.x;
  const long long row = is_round
      ? (long long)seq * stride_rows + (count[seq] - 1)
      : (stride_rows == 0 ? (long long)(tokens_total - 1) : (long long)seq);
  const B* s = taps + row * tap_elems;
  B* d = out_taps + (long long)seq * tap_elems;
  if ((tap_elems & 7) == 0 && (((uintptr_t)s | (uintptr_t)d) & 15) == 0) {
    const int n4 = tap_elems >> 3;
    const uint4* s4 = reinterpret_cast<const uint4*>(s);
    uint4* d4 = reinterpret_cast<uint4*>(d);
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
      d4[i] = s4[i];
    }
  } else {
    for (int i = threadIdx.x; i < tap_elems; i += blockDim.x) {
      d[i] = s[i];
    }
  }
}

// Load persistent per-sequence f_rnn [hidden] BF16 from dspark_state[lines[seq]].
// Null line (<= 0) writes zeros.
// Grid: [seqs, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_load_frnn(
    const unsigned char* __restrict__ dspark_state,
    const int* __restrict__ lines,
    B* __restrict__ f_rnn,
    long long state_stride, long long frnn_off, int hidden) {
  const int seq = blockIdx.x;
  const long long line = lines[seq];
  B* dst = f_rnn + (long long)seq * hidden;
  if (line <= 0) {
    const B z = __float2bfloat16_rn(0.0f);
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = z;
    return;
  }
  const B* src = reinterpret_cast<const B*>(dspark_state + line * state_stride + frnn_off);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// Store persistent per-sequence f_rnn [hidden] BF16 into dspark_state[lines[seq]].
// Null line (<= 0) is strictly a no-op.
// Grid: [seqs, 1, 1], Block: [256, 1, 1]
extern "C" __global__ void dspark_store_frnn(
    const B* __restrict__ f_rnn,
    unsigned char* __restrict__ dspark_state,
    const int* __restrict__ lines,
    long long state_stride, long long frnn_off, int hidden) {
  const int seq = blockIdx.x;
  const long long line = lines[seq];
  if (line <= 0) return;
  const B* src = f_rnn + (long long)seq * hidden;
  B* dst = reinterpret_cast<B*>(dspark_state + line * state_stride + frnn_off);
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// Rollback-safe sconv commit across all target visits:
// Preloads W-1=3 rows into registers (RAW-safe), writing history ending at accepted[seq]-1.
// Grid: [seqs, ceil_div(sconv_dim, 256), n_visits], Block: [256, 1, 1]
extern "C" __global__ void spec_conv_commit(
    const B* __restrict__ raw,
    unsigned char* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int* __restrict__ count,
    long long state_stride,
    long long sconv_off,
    int max_rows_per_visit,
    int sconv_dim,
    int block_size) {
  const int seq = blockIdx.x;
  const int c = blockIdx.y * blockDim.x + threadIdx.x;
  const int visit = blockIdx.z;
  if (c >= sconv_dim) return;
  const long long line = lines[seq];
  if (line <= 0) return;

  const int n = count[seq];
  B* history = reinterpret_cast<B*>(seq_state + line * state_stride + sconv_off)
      + (long long)visit * (3 * sconv_dim);
  const B* vraw = raw + ((long long)visit * max_rows_per_visit + (long long)seq * block_size) * sconv_dim;

  B v[3];
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    int row = n - 3 + j;
    v[j] = (row < 0) ? history[(row + 3) * sconv_dim + c] : vraw[(long long)row * sconv_dim + c];
  }
  #pragma unroll
  for (int j = 0; j < 3; ++j) {
    history[j * sconv_dim + c] = v[j];
  }
}

// Rollback-safe target SWA ring commit:
// Copies ONLY r = 0 .. count[seq]-1 from staged spec_swa_kv into (anchor_pos + r) % window.
// Grid: [seqs, n_swa_visits, block_size], Block: [128, 1, 1]
extern "C" __global__ void spec_swa_commit(
    const unsigned char* __restrict__ spec_swa_kv,
    unsigned char* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int* __restrict__ lengths,
    const int* __restrict__ count,
    long long state_stride,
    long long swa_off,
    int max_rows_per_visit,
    int window,
    int block_size,
    int token_bytes) {
  const int seq = blockIdx.x;
  const int v_swa = blockIdx.y;
  const int r = blockIdx.z;
  const long long line = lines[seq];
  if (line <= 0 || r >= count[seq]) return;

  const long long anchor_pos = (long long)lengths[seq] - block_size;
  if (anchor_pos + r < 0) return;
  const long long ring_slot = (anchor_pos + r) % window;

  const uint4* src = reinterpret_cast<const uint4*>(
      spec_swa_kv + ((long long)v_swa * max_rows_per_visit + (long long)seq * block_size + r) * token_bytes);
  uint4* dst = reinterpret_cast<uint4*>(
      seq_state + line * state_stride + swa_off
      + ((long long)v_swa * window + ring_slot) * token_bytes);
  const int n_vec = token_bytes >> 4;
  for (int i = threadIdx.x; i < n_vec; i += blockDim.x) {
    dst[i] = src[i];
  }
}

// In-place FP32 softcap + mask col >= vocab_unpadded to -inf.
// Grid: [tokens, ceil_div(vocab_padded, 1024), 1], Block: [1024, 1, 1]
extern "C" __global__ void dspark_verify_mask_vocab_f32(
    float* __restrict__ logits, float softcap, int vocab_padded, int vocab_unpadded) {
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= vocab_padded) return;
  const long long idx = (long long)blockIdx.x * vocab_padded + col;
  if (col >= vocab_unpadded) {
    logits[idx] = -__builtin_inff();
  } else if (softcap > 0.0f) {
    logits[idx] = softcap * tanhf(logits[idx] / softcap);
  }
}

// In-place BF16 softcap + mask col >= vocab_unpadded to -inf.
// Grid: [tokens, ceil_div(vocab_padded, 1024), 1], Block: [1024, 1, 1]
extern "C" __global__ void dspark_verify_mask_vocab_bf16(
    B* __restrict__ logits, float softcap, int vocab_padded, int vocab_unpadded) {
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= vocab_padded) return;
  const long long idx = (long long)blockIdx.x * vocab_padded + col;
  if (col >= vocab_unpadded) {
    logits[idx] = __float2bfloat16_rn(-__builtin_inff());
  } else if (softcap > 0.0f) {
    float x = __bfloat162float(logits[idx]);
    logits[idx] = __float2bfloat16_rn(softcap * tanhf(x / softcap));
  }
}

// Out-of-place FP32 -> BF16 softcap + mask col >= vocab_unpadded to -inf.
// Grid: [tokens, ceil_div(vocab_padded, 1024), 1], Block: [1024, 1, 1]
extern "C" __global__ void dspark_verify_softcap_mask_f32_to_bf16(
    const float* __restrict__ logits_in, B* __restrict__ logits_out,
    float softcap, int vocab_padded, int vocab_unpadded) {
  const int col = blockIdx.y * blockDim.x + threadIdx.x;
  if (col >= vocab_padded) return;
  const long long idx = (long long)blockIdx.x * vocab_padded + col;
  if (col >= vocab_unpadded) {
    logits_out[idx] = __float2bfloat16_rn(-__builtin_inff());
  } else {
    float x = logits_in[idx];
    if (softcap > 0.0f) x = softcap * tanhf(x / softcap);
    logits_out[idx] = __float2bfloat16_rn(x);
  }
}

// Expand per-sequence sampling params to per-row params with Philox offsets
// matching single-token decode: row_lengths[seq * block_size + j] = lengths[seq] - (block_size - 1) + j.
// Grid: [seqs, 1, 1], Block: [32, 1, 1]
extern "C" __global__ void spec_sampling(
    const float* __restrict__ temperature,
    const int* __restrict__ top_k,
    const float* __restrict__ top_p,
    const float* __restrict__ min_p,
    const long long* __restrict__ seed,
    const int* __restrict__ lengths,
    float* __restrict__ row_temperature,
    int* __restrict__ row_top_k,
    float* __restrict__ row_top_p,
    float* __restrict__ row_min_p,
    long long* __restrict__ row_seed,
    int* __restrict__ row_lengths,
    int block_size) {
  const int seq = blockIdx.x;
  const int j = threadIdx.x;
  if (j >= block_size) return;
  const int row = seq * block_size + j;
  row_temperature[row] = temperature[seq];
  row_top_k[row] = top_k[seq];
  row_top_p[row] = top_p[seq];
  row_min_p[row] = min_p[seq];
  row_seed[row] = seed[seq];
  row_lengths[row] = lengths[seq] - (block_size - 1) + j;
}

extern "C" __global__ void spec_zeros_bf16(B* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = __float2bfloat16_rn(0.0f);
}

extern "C" __global__ void spec_copy_bf16(const B* __restrict__ src, B* __restrict__ dst, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dst[i] = src[i];
}
