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

// ===========================================================================
// SM120 (sm_120a) 256-bit direct-register BF16 GEMV/GEMM & fused GateUp+SiLU
// for DSpark 3-layer drafter + Markov projections (M <= 64) with PDL.
// ===========================================================================
namespace ds_gemm {

struct alignas(32) u32x8 {
  uint32_t v[8];
};

__device__ __forceinline__ u32x8 ld_w_v8(const void* ptr) {
  u32x8 r;
  asm volatile(
      "ld.global.nc.L1::no_allocate.L2::256B.v8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(ptr));
  return r;
}

__device__ __forceinline__ u32x8 ld_x_v8(const void* ptr) {
  u32x8 r;
  asm volatile(
      "ld.global.L1::evict_last.L2::evict_last.v8.b32 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
      : "=r"(r.v[0]), "=r"(r.v[1]), "=r"(r.v[2]), "=r"(r.v[3]),
        "=r"(r.v[4]), "=r"(r.v[5]), "=r"(r.v[6]), "=r"(r.v[7])
      : "l"(ptr));
  return r;
}

__device__ __forceinline__ void pdl_wait() {
  asm volatile("griddepcontrol.wait;" ::: "memory");
}

__device__ __forceinline__ void pdl_launch_dependents() {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
}

__device__ __forceinline__ void mma_m16n8k16_acc(
    float (&d)[4],
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_v8_step(
    float (&acc)[4], const u32x8& x0, const u32x8& x1, const u32x8& w) {
  mma_m16n8k16_acc(acc, x0.v[0], x1.v[0], x0.v[1], x1.v[1], w.v[0], w.v[1]);
  mma_m16n8k16_acc(acc, x0.v[2], x1.v[2], x0.v[3], x1.v[3], w.v[2], w.v[3]);
  mma_m16n8k16_acc(acc, x0.v[4], x1.v[4], x0.v[5], x1.v[5], w.v[4], w.v[5]);
  mma_m16n8k16_acc(acc, x0.v[6], x1.v[6], x0.v[7], x1.v[7], w.v[6], w.v[7]);
}

__device__ __forceinline__ bool is_v8_aligned(const B* A, const B* W, int K) {
  return ((K & 63) == 0)
      && ((reinterpret_cast<uintptr_t>(A) & 31u) == 0u)
      && ((reinterpret_cast<uintptr_t>(W) & 31u) == 0u);
}

__device__ __forceinline__ B silu_mul_bf16_exact(float g_f, float u_f) {
  // Match unfused bf16(gate_up) -> FP32 IEEE div silu(g)*u -> bf16(act).
  const float g = __bfloat162float(__float2bfloat16_rn(g_f));
  const float u = __bfloat162float(__float2bfloat16_rn(u_f));
  const float silu = __fdiv_rn(g, 1.0f + expf(-g));
  return __float2bfloat16_rn(silu * u);
}

template <typename OutT>
__device__ __noinline__ void fallback_gemm(
    const B* __restrict__ A, const B* __restrict__ W, OutT* __restrict__ C,
    int M, int N, int K) {
  pdl_wait();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int n = tid; n < N; n += stride) {
    const B* w_row = W + static_cast<int64_t>(n) * K;
    for (int m = 0; m < M; ++m) {
      const B* a_row = A + static_cast<int64_t>(m) * K;
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc = fmaf(__bfloat162float(a_row[k]), __bfloat162float(w_row[k]), acc);
      }
      if constexpr (sizeof(OutT) == 2) {
        C[static_cast<int64_t>(m) * N + n] = __float2bfloat16_rn(acc);
      } else {
        C[static_cast<int64_t>(m) * N + n] = acc;
      }
    }
  }
  pdl_launch_dependents();
}

__device__ __noinline__ void fallback_gate_up_silu(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ Y,
    int M, int N_half, int K) {
  pdl_wait();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int n = tid; n < N_half; n += stride) {
    const B* wg = W + static_cast<int64_t>(n) * K;
    const B* wu = W + static_cast<int64_t>(N_half + n) * K;
    for (int m = 0; m < M; ++m) {
      const B* a_row = A + static_cast<int64_t>(m) * K;
      float ag = 0.0f, au = 0.0f;
      for (int k = 0; k < K; ++k) {
        const float av = __bfloat162float(a_row[k]);
        ag = fmaf(av, __bfloat162float(wg[k]), ag);
        au = fmaf(av, __bfloat162float(wu[k]), au);
      }
      Y[static_cast<int64_t>(m) * N_half + n] = silu_mul_bf16_exact(ag, au);
    }
  }
  pdl_launch_dependents();
}

// ---------------------------------------------------------------------------
// Direct-register 256-bit v8.b32 MMA kernel (OutT = B or float) with PDL.
// Handles M <= M_CAP in 1 weight pass (zero SMEM K-loop, STACK:0).
// ---------------------------------------------------------------------------
template <int M_TILES, bool M_LE_8, typename OutT>
__device__ __forceinline__ void run_v8_mma_pdl(
    const B* __restrict__ A, const B* __restrict__ W, OutT* __restrict__ C,
    int M, int N, int K) {
  constexpr int M_CAP = M_LE_8 ? 8 : (M_TILES * 16);
  if (M <= 0 || N <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_v8_aligned(A, W, K) || M > M_CAP) {
    fallback_gemm<OutT>(A, W, C, M, N, K);
    return;
  }

  constexpr int WARPS_K = 8;
  constexpr int K_STEP = 64 * WARPS_K;  // 512 BF16 = 1024 B/row/iter
  const int bn = (N >= 1024 || N < 128) ? 8 : 4;
  const int num_tiles = (N + bn - 1) / bn;
  if (blockIdx.x >= num_tiles) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  __shared__ float4 s_red[WARPS_K - 1][M_TILES][32];
  const int k0 = warp * 64;
  const B* a_base = A + t * 16;

  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const int n_base = tile * bn;
    const int n_row = n_base + g;
    const bool w_ok = (g < bn) && (n_row < N);
    const B* w_base = W + static_cast<int64_t>(n_row) * K + t * 16;

    u32x8 w_cur = {};
    if (w_ok && k0 < K) {
      w_cur = ld_w_v8(w_base + k0);
    }
    if (tile == blockIdx.x) {
      pdl_wait();
    }

    float acc[M_TILES][4];
#pragma unroll
    for (int mt = 0; mt < M_TILES; ++mt) {
      acc[mt][0] = 0.0f;
      acc[mt][1] = 0.0f;
      acc[mt][2] = 0.0f;
      acc[mt][3] = 0.0f;
    }

    for (int k = k0; k < K; k += K_STEP) {
      const int k_next = k + K_STEP;
      u32x8 w_nxt = {};
      if (w_ok && k_next < K) {
        w_nxt = ld_w_v8(w_base + k_next);
      }

#pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
        const int m0 = mt * 16 + g;
        u32x8 x0 = (m0 < M) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
        u32x8 x1 = {};
        if constexpr (!M_LE_8) {
          const int m1 = m0 + 8;
          if (m1 < M) {
            x1 = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
          }
        }
        mma_v8_step(acc[mt], x0, x1, w_cur);
      }
      w_cur = w_nxt;
    }

    if (warp > 0) {
#pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
        s_red[warp - 1][mt][lane] = make_float4(acc[mt][0], acc[mt][1], acc[mt][2], acc[mt][3]);
      }
    }
    __syncthreads();

    if (tile + gridDim.x >= num_tiles) {
      pdl_launch_dependents();
    }

    if (warp == 0) {
      const int col0 = n_base + 2 * t;
      const int col1 = col0 + 1;
      const bool c0_ok = (2 * t < bn) && (col0 < N);
      const bool c1_ok = (2 * t + 1 < bn) && (col1 < N);
      const bool c_align = ((N & 1) == 0) && ((reinterpret_cast<uintptr_t>(C) & (2 * sizeof(OutT) - 1)) == 0u);
      if (c0_ok) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
          for (int w = 0; w < WARPS_K - 1; ++w) {
            const float4 v = s_red[w][mt][lane];
            acc[mt][0] += v.x;
            acc[mt][1] += v.y;
            acc[mt][2] += v.z;
            acc[mt][3] += v.w;
          }
          const int m0 = mt * 16 + g;
          if (m0 < M) {
            OutT* dst = C + static_cast<int64_t>(m0) * N + col0;
            if constexpr (sizeof(OutT) == 2) {
              if (c1_ok && c_align) {
                *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc[mt][0], acc[mt][1]);
              } else {
                dst[0] = __float2bfloat16_rn(acc[mt][0]);
                if (c1_ok) dst[1] = __float2bfloat16_rn(acc[mt][1]);
              }
            } else {
              if (c1_ok && c_align) {
                *reinterpret_cast<float2*>(dst) = make_float2(acc[mt][0], acc[mt][1]);
              } else {
                dst[0] = acc[mt][0];
                if (c1_ok) dst[1] = acc[mt][1];
              }
            }
          }
          if constexpr (!M_LE_8) {
            const int m1 = m0 + 8;
            if (m1 < M) {
              OutT* dst = C + static_cast<int64_t>(m1) * N + col0;
              if constexpr (sizeof(OutT) == 2) {
                if (c1_ok && c_align) {
                  *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(acc[mt][2], acc[mt][3]);
                } else {
                  dst[0] = __float2bfloat16_rn(acc[mt][2]);
                  if (c1_ok) dst[1] = __float2bfloat16_rn(acc[mt][3]);
                }
              } else {
                if (c1_ok && c_align) {
                  *reinterpret_cast<float2*>(dst) = make_float2(acc[mt][2], acc[mt][3]);
                } else {
                  dst[0] = acc[mt][2];
                  if (c1_ok) dst[1] = acc[mt][3];
                }
              }
            }
          }
        }
      }
    }
    if (tile + gridDim.x < num_tiles) {
      __syncthreads();
    }
  }
}

// ---------------------------------------------------------------------------
// M = 1 specialized 256-bit GEMV (OutT = B or float) with PDL.
// ---------------------------------------------------------------------------
template <typename OutT>
__device__ __forceinline__ void run_v8_gemv_m1_pdl(
    const B* __restrict__ A, const B* __restrict__ W, OutT* __restrict__ C,
    int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_v8_aligned(A, W, K) || M != 1) {
    fallback_gemm<OutT>(A, W, C, M, N, K);
    return;
  }

  constexpr int WARPS = 8;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (N < 1024) {
    // 2 warps per output row -> 4 rows/CTA (N=768 -> 192 CTAs across 110/188 SMs!).
    __shared__ float s_pair[4];
    const int sub_row = warp >> 1;
    const int sub_k = warp & 1;
    const int base_row = blockIdx.x * 4;
    if (base_row >= N) {
      pdl_wait();
      pdl_launch_dependents();
      return;
    }
    for (int tile_row = base_row; tile_row < N; tile_row += gridDim.x * 4) {
      const int row = tile_row + sub_row;
      const bool ok = (row < N);
      const int k0 = (sub_k * 32 + lane) * 16;
      const B* w_row = W + static_cast<int64_t>(row) * K;
      u32x8 w_cur = (ok && k0 < K) ? ld_w_v8(w_row + k0) : u32x8{};
      if (tile_row == base_row) pdl_wait();

      float acc = 0.0f;
      if (ok) {
        for (int k = k0; k < K; k += 1024) {
          u32x8 w_nxt = (k + 1024 < K) ? ld_w_v8(w_row + k + 1024) : u32x8{};
          u32x8 xv = ld_x_v8(A + k);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w_cur.v[i]));
            const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.v[i]));
            acc = fmaf(wf.x, xf.x, acc);
            acc = fmaf(wf.y, xf.y, acc);
          }
          w_cur = w_nxt;
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
      if (sub_k == 1 && lane == 0) s_pair[sub_row] = acc;
      __syncthreads();
      if (tile_row + gridDim.x * 4 >= N) pdl_launch_dependents();
      if (ok && sub_k == 0 && lane == 0) {
        const float sum = acc + s_pair[sub_row];
        if constexpr (sizeof(OutT) == 2) {
          C[row] = __float2bfloat16_rn(sum);
        } else {
          C[row] = sum;
        }
      }
      if (tile_row + gridDim.x * 4 < N) __syncthreads();
    }
    return;
  }

  // N >= 1024: 1 warp per output row (8 rows/CTA).
  const int global_warp = blockIdx.x * WARPS + warp;
  const int total_warps = gridDim.x * WARPS;
  if (global_warp >= N) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  for (int row = global_warp; row < N; row += total_warps) {
    const B* w_row = W + static_cast<int64_t>(row) * K;
    const int k0 = lane * 16;
    u32x8 w_cur = (k0 < K) ? ld_w_v8(w_row + k0) : u32x8{};
    if (row == global_warp) pdl_wait();

    float acc = 0.0f;
    for (int k = k0; k < K; k += 512) {
      u32x8 w_nxt = (k + 512 < K) ? ld_w_v8(w_row + k + 512) : u32x8{};
      u32x8 xv = ld_x_v8(A + k);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w_cur.v[i]));
        const float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.v[i]));
        acc = fmaf(wf.x, xf.x, acc);
        acc = fmaf(wf.y, xf.y, acc);
      }
      w_cur = w_nxt;
    }
    if (row + total_warps >= N) pdl_launch_dependents();
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) {
      if constexpr (sizeof(OutT) == 2) {
        C[row] = __float2bfloat16_rn(acc);
      } else {
        C[row] = acc;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Fused GateUp + SiLU*Mul 256-bit direct-register MMA with PDL.
// W = [W_gate (N_half, K); W_up (N_half, K)], Y = [M, N_half].
// Reads activation A once for both W_gate and W_up; eliminates d.gate_up DRAM
// traffic and dspark_silu_mul launch. Loops over m_base for general M.
// ---------------------------------------------------------------------------
template <int M_TILES, bool M_LE_8, bool PREFETCH = false>
__device__ __forceinline__ void run_v8_gate_up_silu_pdl(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ Y,
    int M, int N_half, int K) {
  constexpr int M_CAP = M_LE_8 ? 8 : (M_TILES * 16);
  if (M <= 0 || N_half <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_v8_aligned(A, W, K)) {
    fallback_gate_up_silu(A, W, Y, M, N_half, K);
    return;
  }

  constexpr int WARPS_K = 8;
  constexpr int K_STEP = 64 * WARPS_K;  // 512 BF16
  constexpr int BN = 8;
  const int num_tiles = (N_half + BN - 1) / BN;
  if (blockIdx.x >= num_tiles) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  __shared__ float4 s_red[WARPS_K - 1][2][M_TILES][32];
  const int k0 = warp * 64;
  const bool y_align4 = ((N_half & 1) == 0) && ((reinterpret_cast<uintptr_t>(Y) & 3u) == 0u);
  bool did_wait = false;

  for (int m_base = 0; m_base < M; m_base += M_CAP) {
    const int m_rem = M - m_base;
    const B* a_base = A + static_cast<int64_t>(m_base) * K + t * 16;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
      const int n_base = tile * BN;
      const int n_row = n_base + g;
      const bool w_ok = (n_row < N_half);
      const B* wg_base = W + static_cast<int64_t>(n_row) * K + t * 16;
      const B* wu_base = W + static_cast<int64_t>(N_half + n_row) * K + t * 16;

      u32x8 wg = {};
      u32x8 wu = {};
      if (w_ok && k0 < K) {
        wg = ld_w_v8(wg_base + k0);
        wu = ld_w_v8(wu_base + k0);
      }
      if (!did_wait) {
        pdl_wait();
        did_wait = true;
      }

      float acc_g[M_TILES][4];
      float acc_u[M_TILES][4];
#pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
        acc_g[mt][0] = 0.0f; acc_g[mt][1] = 0.0f; acc_g[mt][2] = 0.0f; acc_g[mt][3] = 0.0f;
        acc_u[mt][0] = 0.0f; acc_u[mt][1] = 0.0f; acc_u[mt][2] = 0.0f; acc_u[mt][3] = 0.0f;
      }

      for (int k = k0; k < K; k += K_STEP) {
        u32x8 wg_nxt = {};
        u32x8 wu_nxt = {};
        if constexpr (PREFETCH) {
          const int k_next = k + K_STEP;
          if (w_ok && k_next < K) {
            wg_nxt = ld_w_v8(wg_base + k_next);
            wu_nxt = ld_w_v8(wu_base + k_next);
          }
        } else if (k > k0) {
          if (w_ok) {
            wg = ld_w_v8(wg_base + k);
            wu = ld_w_v8(wu_base + k);
          } else {
            wg = {};
            wu = {};
          }
        }
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
          const int m0 = mt * 16 + g;
          u32x8 x0 = (m0 < m_rem) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
          u32x8 x1 = {};
          if constexpr (!M_LE_8) {
            const int m1 = m0 + 8;
            if (m1 < m_rem) {
              x1 = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
            }
          }
          mma_v8_step(acc_g[mt], x0, x1, wg);
          mma_v8_step(acc_u[mt], x0, x1, wu);
        }
        if constexpr (PREFETCH) {
          wg = wg_nxt;
          wu = wu_nxt;
        }
      }

      if (warp > 0) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
          s_red[warp - 1][0][mt][lane] = make_float4(acc_g[mt][0], acc_g[mt][1], acc_g[mt][2], acc_g[mt][3]);
          s_red[warp - 1][1][mt][lane] = make_float4(acc_u[mt][0], acc_u[mt][1], acc_u[mt][2], acc_u[mt][3]);
        }
      }
      __syncthreads();

      const bool is_last = (m_base + M_CAP >= M) && (tile + gridDim.x >= num_tiles);
      if (is_last) {
        pdl_launch_dependents();
      }

      if (warp == 0) {
        const int col0 = n_base + 2 * t;
        const int col1 = col0 + 1;
        const bool c0_ok = (col0 < N_half);
        const bool c1_ok = (col1 < N_half);
        if (c0_ok) {
#pragma unroll
          for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
            for (int w = 0; w < WARPS_K - 1; ++w) {
              const float4 vg = s_red[w][0][mt][lane];
              const float4 vu = s_red[w][1][mt][lane];
              acc_g[mt][0] += vg.x; acc_g[mt][1] += vg.y; acc_g[mt][2] += vg.z; acc_g[mt][3] += vg.w;
              acc_u[mt][0] += vu.x; acc_u[mt][1] += vu.y; acc_u[mt][2] += vu.z; acc_u[mt][3] += vu.w;
            }
            const int m0 = mt * 16 + g;
            if (m0 < m_rem) {
              const B y0 = silu_mul_bf16_exact(acc_g[mt][0], acc_u[mt][0]);
              B* dst = Y + static_cast<int64_t>(m_base + m0) * N_half + col0;
              if (c1_ok) {
                const B y1 = silu_mul_bf16_exact(acc_g[mt][1], acc_u[mt][1]);
                if (y_align4) {
                  *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(y0, y1);
                } else {
                  dst[0] = y0;
                  dst[1] = y1;
                }
              } else {
                dst[0] = y0;
              }
            }
            if constexpr (!M_LE_8) {
              const int m1 = m0 + 8;
              if (m1 < m_rem) {
                const B y2 = silu_mul_bf16_exact(acc_g[mt][2], acc_u[mt][2]);
                B* dst = Y + static_cast<int64_t>(m_base + m1) * N_half + col0;
                if (c1_ok) {
                  const B y3 = silu_mul_bf16_exact(acc_g[mt][3], acc_u[mt][3]);
                  if (y_align4) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(y2, y3);
                  } else {
                    dst[0] = y2;
                    dst[1] = y3;
                  }
                } else {
                  dst[0] = y2;
                }
              }
            }
          }
        }
      }
      if (!is_last) {
        __syncthreads();
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SM120 W8A16 FP8 (E4M3 per-row) -> BF16 direct-register MMA kernel with PDL
// for DSpark `markov.base_logits` (M = 7*seqs, N = 201024, K = 2304).
// Loads 16 FP8 weights via 128-bit ld.global.nc.L1::no_allocate.L2::256B.v4.u32,
// converts to 16 BF16 in registers via 8x 1-cycle `cvt.rn.bf16x2.e4m3x2`, and
// applies per-row FP32 scale at epilogue.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 ld_w_fp8_v4(const void* ptr) {
  uint4 r;
  asm volatile(
      "ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(ptr));
  return r;
}

__device__ __forceinline__ uint32_t cvt_e4m3x2_to_bf16x2(uint16_t raw) {
  uint32_t res;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(res) : "h"(raw));
  return res;
}

__device__ __forceinline__ u32x8 unpack_fp8_v4_to_bf16_v8(const uint4& w) {
  u32x8 out;
  out.v[0] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.x & 0xFFFFu));
  out.v[1] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.x >> 16));
  out.v[2] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.y & 0xFFFFu));
  out.v[3] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.y >> 16));
  out.v[4] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.z & 0xFFFFu));
  out.v[5] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.z >> 16));
  out.v[6] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.w & 0xFFFFu));
  out.v[7] = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w.w >> 16));
  return out;
}

__device__ __forceinline__ bool is_fp8_v4_aligned(const B* A, const uint8_t* W, int K) {
  return ((K & 63) == 0)
      && ((reinterpret_cast<uintptr_t>(A) & 31u) == 0u)
      && ((reinterpret_cast<uintptr_t>(W) & 15u) == 0u);
}

__device__ __noinline__ void fallback_gemm_w8a16_fp8(
    const B* __restrict__ A,
    const uint8_t* __restrict__ W,
    const float* __restrict__ S,
    B* __restrict__ C,
    int M, int N, int K) {
  pdl_wait();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int n = tid; n < N; n += stride) {
    const uint8_t* w_row = W + static_cast<int64_t>(n) * K;
    const float s = S[n];
    for (int m = 0; m < M; ++m) {
      const B* a_row = A + static_cast<int64_t>(m) * K;
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        uint32_t pair = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(w_row[k]));
        float wf = __bfloat162float(*reinterpret_cast<const B*>(&pair));
        acc = fmaf(__bfloat162float(a_row[k]), wf, acc);
      }
      C[static_cast<int64_t>(m) * N + n] = __float2bfloat16_rn(acc * s);
    }
  }
  pdl_launch_dependents();
}

template <int M_TILES, bool M_LE_8, int N_TILES, bool PREFETCH = true>
__device__ __forceinline__ void run_w8a16_fp8_mma_pdl(
    const B* __restrict__ A,
    const uint8_t* __restrict__ W,
    const float* __restrict__ S,
    B* __restrict__ C,
    int M, int N, int K) {
  constexpr int M_CAP = M_LE_8 ? 8 : (M_TILES * 16);
  if (M <= 0 || N <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_fp8_v4_aligned(A, W, K)) {
    fallback_gemm_w8a16_fp8(A, W, S, C, M, N, K);
    return;
  }

  constexpr int WARPS_K = 8;
  constexpr int K_STEP = 64 * WARPS_K;  // 512 elements/iter
  constexpr int BN = 8 * N_TILES;
  const int num_tiles = (N + BN - 1) / BN;
  if (blockIdx.x >= num_tiles) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;  // 0..7
  const int t = lane & 3;   // 0..3

  // Shared reduction buffer holds one N-tile at a time (<= 14 KiB even at M_TILES=4).
  __shared__ float4 s_red[WARPS_K - 1][M_TILES][32];
  const int k0 = warp * 64;
  const bool c_align4 = ((N & 1) == 0) && ((reinterpret_cast<uintptr_t>(C) & 3u) == 0u);
  bool did_wait = false;

  for (int m_base = 0; m_base < M; m_base += M_CAP) {
    const int m_rem = M - m_base;
    const B* a_base = A + static_cast<int64_t>(m_base) * K + t * 16;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
      const int n_base = tile * BN;
      const uint8_t* w_base[N_TILES];
      bool w_ok[N_TILES];
      uint4 w_cur[N_TILES];
#pragma unroll
      for (int nt = 0; nt < N_TILES; ++nt) {
        const int n_row = n_base + nt * 8 + g;
        w_ok[nt] = (n_row < N);
        w_base[nt] = W + static_cast<int64_t>(n_row) * K + t * 16;
        w_cur[nt] = (w_ok[nt] && k0 < K) ? ld_w_fp8_v4(w_base[nt] + k0) : make_uint4(0u, 0u, 0u, 0u);
      }
      if (!did_wait) {
        pdl_wait();
        did_wait = true;
      }

      float acc[M_TILES][N_TILES][4];
#pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
          acc[mt][nt][0] = 0.0f;
          acc[mt][nt][1] = 0.0f;
          acc[mt][nt][2] = 0.0f;
          acc[mt][nt][3] = 0.0f;
        }
      }

      for (int k = k0; k < K; k += K_STEP) {
        uint4 w_nxt[N_TILES];
        if constexpr (PREFETCH) {
          const int k_next = k + K_STEP;
#pragma unroll
          for (int nt = 0; nt < N_TILES; ++nt) {
            w_nxt[nt] = (w_ok[nt] && k_next < K) ? ld_w_fp8_v4(w_base[nt] + k_next) : make_uint4(0u, 0u, 0u, 0u);
          }
        } else if (k > k0) {
#pragma unroll
          for (int nt = 0; nt < N_TILES; ++nt) {
            w_cur[nt] = w_ok[nt] ? ld_w_fp8_v4(w_base[nt] + k) : make_uint4(0u, 0u, 0u, 0u);
          }
        }

        if constexpr (N_TILES == 1) {
          const u32x8 w_bf16 = unpack_fp8_v4_to_bf16_v8(w_cur[0]);
#pragma unroll
          for (int mt = 0; mt < M_TILES; ++mt) {
            const int m0 = mt * 16 + g;
            u32x8 x0 = (m0 < m_rem) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
            u32x8 x1 = {};
            if constexpr (!M_LE_8) {
              const int m1 = m0 + 8;
              if (m1 < m_rem) x1 = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
            }
            mma_v8_step(acc[mt][0], x0, x1, w_bf16);
          }
        } else {
          // Reuse loaded A (x0, x1) across all N_TILES -> cuts A L2 bandwidth by N_TILES!
#pragma unroll
          for (int mt = 0; mt < M_TILES; ++mt) {
            const int m0 = mt * 16 + g;
            u32x8 x0 = (m0 < m_rem) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
            u32x8 x1 = {};
            if constexpr (!M_LE_8) {
              const int m1 = m0 + 8;
              if (m1 < m_rem) x1 = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
            }
#pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
              const u32x8 w_bf16 = unpack_fp8_v4_to_bf16_v8(w_cur[nt]);
              mma_v8_step(acc[mt][nt], x0, x1, w_bf16);
            }
          }
        }
        if constexpr (PREFETCH) {
#pragma unroll
          for (int nt = 0; nt < N_TILES; ++nt) {
            w_cur[nt] = w_nxt[nt];
          }
        }
      }

      const bool is_last = (m_base + M_CAP >= M) && (tile + gridDim.x >= num_tiles);

#pragma unroll
      for (int nt = 0; nt < N_TILES; ++nt) {
        if (warp > 0) {
#pragma unroll
          for (int mt = 0; mt < M_TILES; ++mt) {
            s_red[warp - 1][mt][lane] =
                make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
          }
        }
        __syncthreads();

        if (nt == N_TILES - 1 && is_last) {
          pdl_launch_dependents();
        }

        if (warp == 0) {
          const int col0 = n_base + nt * 8 + 2 * t;
          const int col1 = col0 + 1;
          const bool c0_ok = (col0 < N);
          const bool c1_ok = (col1 < N);
          if (c0_ok) {
            const float s0 = S[col0];
            const float s1 = c1_ok ? S[col1] : 0.0f;
#pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
              for (int w = 0; w < WARPS_K - 1; ++w) {
                const float4 v = s_red[w][mt][lane];
                acc[mt][nt][0] += v.x;
                acc[mt][nt][1] += v.y;
                acc[mt][nt][2] += v.z;
                acc[mt][nt][3] += v.w;
              }
              const int m0 = mt * 16 + g;
              if (m0 < m_rem) {
                const float v0 = acc[mt][nt][0] * s0;
                const float v1 = acc[mt][nt][1] * s1;
                B* dst = C + static_cast<int64_t>(m_base + m0) * N + col0;
                if (c1_ok && c_align4) {
                  *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else {
                  dst[0] = __float2bfloat16_rn(v0);
                  if (c1_ok) dst[1] = __float2bfloat16_rn(v1);
                }
              }
              if constexpr (!M_LE_8) {
                const int m1 = m0 + 8;
                if (m1 < m_rem) {
                  const float v2 = acc[mt][nt][2] * s0;
                  const float v3 = acc[mt][nt][3] * s1;
                  B* dst = C + static_cast<int64_t>(m_base + m1) * N + col0;
                  if (c1_ok && c_align4) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v2, v3);
                  } else {
                    dst[0] = __float2bfloat16_rn(v2);
                    if (c1_ok) dst[1] = __float2bfloat16_rn(v3);
                  }
                }
              }
            }
          }
        }
        if (nt + 1 < N_TILES || !is_last) {
          __syncthreads();
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Fused W8A16 FP8 GateUp + SiLU*Mul with PDL for DSpark 3-layer MLP.
// W = [W_gate (N_half, K); W_up (N_half, K)] in FP8 E4M3, S = [2 * N_half] FP32.
// Uses two-phase shared reduction (<= 14 KiB SMEM even at M_TILES=4) so W is
// read strictly ONCE for M <= 64.
// ---------------------------------------------------------------------------
__device__ __noinline__ void fallback_gate_up_silu_fp8(
    const B* __restrict__ A,
    const uint8_t* __restrict__ W,
    const float* __restrict__ S,
    B* __restrict__ Y,
    int M, int N_half, int K) {
  pdl_wait();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int n = tid; n < N_half; n += stride) {
    const uint8_t* wg = W + static_cast<int64_t>(n) * K;
    const uint8_t* wu = W + static_cast<int64_t>(N_half + n) * K;
    const float sg = S[n];
    const float su = S[N_half + n];
    for (int m = 0; m < M; ++m) {
      const B* a_row = A + static_cast<int64_t>(m) * K;
      float ag = 0.0f, au = 0.0f;
      for (int k = 0; k < K; ++k) {
        const float xf = __bfloat162float(a_row[k]);
        uint32_t pg = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(wg[k]));
        uint32_t pu = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(wu[k]));
        ag = fmaf(xf, __bfloat162float(*reinterpret_cast<const B*>(&pg)), ag);
        au = fmaf(xf, __bfloat162float(*reinterpret_cast<const B*>(&pu)), au);
      }
      Y[static_cast<int64_t>(m) * N_half + n] = silu_mul_bf16_exact(ag * sg, au * su);
    }
  }
  pdl_launch_dependents();
}

template <int M_TILES, bool M_LE_8, bool PREFETCH = false>
__device__ __forceinline__ void run_w8a16_fp8_gate_up_silu_pdl(
    const B* __restrict__ A,
    const uint8_t* __restrict__ W,
    const float* __restrict__ S,
    B* __restrict__ Y,
    int M, int N_half, int K) {
  constexpr int M_CAP = M_LE_8 ? 8 : (M_TILES * 16);
  if (M <= 0 || N_half <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_fp8_v4_aligned(A, W, K)) {
    fallback_gate_up_silu_fp8(A, W, S, Y, M, N_half, K);
    return;
  }

  constexpr int WARPS_K = 8;
  constexpr int K_STEP = 64 * WARPS_K;  // 512 elements/iter
  constexpr int BN = 8;
  const int num_tiles = (N_half + BN - 1) / BN;
  if (blockIdx.x >= num_tiles) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int g = lane >> 2;
  const int t = lane & 3;

  __shared__ float4 s_red[WARPS_K - 1][M_TILES][32];
  const int k0 = warp * 64;
  const bool y_align4 = ((N_half & 1) == 0) && ((reinterpret_cast<uintptr_t>(Y) & 3u) == 0u);
  bool did_wait = false;

  for (int m_base = 0; m_base < M; m_base += M_CAP) {
    const int m_rem = M - m_base;
    const B* a_base = A + static_cast<int64_t>(m_base) * K + t * 16;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
      const int n_base = tile * BN;
      const int n_row = n_base + g;
      const bool w_ok = (n_row < N_half);
      const uint8_t* wg_base = W + static_cast<int64_t>(n_row) * K + t * 16;
      const uint8_t* wu_base = W + static_cast<int64_t>(N_half + n_row) * K + t * 16;

      uint4 wg_cur = (w_ok && k0 < K) ? ld_w_fp8_v4(wg_base + k0) : make_uint4(0u, 0u, 0u, 0u);
      uint4 wu_cur = (w_ok && k0 < K) ? ld_w_fp8_v4(wu_base + k0) : make_uint4(0u, 0u, 0u, 0u);
      if (!did_wait) {
        pdl_wait();
        did_wait = true;
      }

      float acc_g[M_TILES][4];
      float acc_u[M_TILES][4];
#pragma unroll
      for (int mt = 0; mt < M_TILES; ++mt) {
        acc_g[mt][0] = 0.0f; acc_g[mt][1] = 0.0f; acc_g[mt][2] = 0.0f; acc_g[mt][3] = 0.0f;
        acc_u[mt][0] = 0.0f; acc_u[mt][1] = 0.0f; acc_u[mt][2] = 0.0f; acc_u[mt][3] = 0.0f;
      }

      for (int k = k0; k < K; k += K_STEP) {
        uint4 wg_nxt = make_uint4(0u, 0u, 0u, 0u);
        uint4 wu_nxt = make_uint4(0u, 0u, 0u, 0u);
        if constexpr (PREFETCH) {
          const int k_next = k + K_STEP;
          if (w_ok && k_next < K) {
            wg_nxt = ld_w_fp8_v4(wg_base + k_next);
            wu_nxt = ld_w_fp8_v4(wu_base + k_next);
          }
        } else if (k > k0) {
          wg_cur = w_ok ? ld_w_fp8_v4(wg_base + k) : make_uint4(0u, 0u, 0u, 0u);
          wu_cur = w_ok ? ld_w_fp8_v4(wu_base + k) : make_uint4(0u, 0u, 0u, 0u);
        }

        const u32x8 wg_bf16 = unpack_fp8_v4_to_bf16_v8(wg_cur);
        const u32x8 wu_bf16 = unpack_fp8_v4_to_bf16_v8(wu_cur);

#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
          const int m0 = mt * 16 + g;
          u32x8 x0 = (m0 < m_rem) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
          u32x8 x1 = {};
          if constexpr (!M_LE_8) {
            const int m1 = m0 + 8;
            if (m1 < m_rem) x1 = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
          }
          mma_v8_step(acc_g[mt], x0, x1, wg_bf16);
          mma_v8_step(acc_u[mt], x0, x1, wu_bf16);
        }

        if constexpr (PREFETCH) {
          wg_cur = wg_nxt;
          wu_cur = wu_nxt;
        }
      }

      // Phase 1: reduce gate
      if (warp > 0) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
          s_red[warp - 1][mt][lane] = make_float4(acc_g[mt][0], acc_g[mt][1], acc_g[mt][2], acc_g[mt][3]);
        }
      }
      __syncthreads();
      if (warp == 0) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
          for (int w = 0; w < WARPS_K - 1; ++w) {
            const float4 vg = s_red[w][mt][lane];
            acc_g[mt][0] += vg.x; acc_g[mt][1] += vg.y; acc_g[mt][2] += vg.z; acc_g[mt][3] += vg.w;
          }
        }
      }
      __syncthreads();

      // Phase 2: reduce up
      if (warp > 0) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
          s_red[warp - 1][mt][lane] = make_float4(acc_u[mt][0], acc_u[mt][1], acc_u[mt][2], acc_u[mt][3]);
        }
      }
      __syncthreads();

      const bool is_last = (m_base + M_CAP >= M) && (tile + gridDim.x >= num_tiles);
      if (is_last) {
        pdl_launch_dependents();
      }

      if (warp == 0) {
        const int col0 = n_base + 2 * t;
        const int col1 = col0 + 1;
        const bool c0_ok = (col0 < N_half);
        const bool c1_ok = (col1 < N_half);
        if (c0_ok) {
          const float sg0 = S[col0];
          const float su0 = S[N_half + col0];
          const float sg1 = c1_ok ? S[col1] : 0.0f;
          const float su1 = c1_ok ? S[N_half + col1] : 0.0f;
#pragma unroll
          for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
            for (int w = 0; w < WARPS_K - 1; ++w) {
              const float4 vu = s_red[w][mt][lane];
              acc_u[mt][0] += vu.x; acc_u[mt][1] += vu.y; acc_u[mt][2] += vu.z; acc_u[mt][3] += vu.w;
            }
            const int m0 = mt * 16 + g;
            if (m0 < m_rem) {
              const B y0 = silu_mul_bf16_exact(acc_g[mt][0] * sg0, acc_u[mt][0] * su0);
              B* dst = Y + static_cast<int64_t>(m_base + m0) * N_half + col0;
              if (c1_ok) {
                const B y1 = silu_mul_bf16_exact(acc_g[mt][1] * sg1, acc_u[mt][1] * su1);
                if (y_align4) {
                  *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(y0, y1);
                } else {
                  dst[0] = y0; dst[1] = y1;
                }
              } else {
                dst[0] = y0;
              }
            }
            if constexpr (!M_LE_8) {
              const int m1 = m0 + 8;
              if (m1 < m_rem) {
                const B y2 = silu_mul_bf16_exact(acc_g[mt][2] * sg0, acc_u[mt][2] * su0);
                B* dst = Y + static_cast<int64_t>(m_base + m1) * N_half + col0;
                if (c1_ok) {
                  const B y3 = silu_mul_bf16_exact(acc_g[mt][3] * sg1, acc_u[mt][3] * su1);
                  if (y_align4) {
                    *reinterpret_cast<__nv_bfloat162*>(dst) = __halves2bfloat162(y2, y3);
                  } else {
                    dst[0] = y2; dst[1] = y3;
                  }
                } else {
                  dst[0] = y2;
                }
              }
            }
          }
        }
      }
      if (!is_last) {
        __syncthreads();
      }
    }
  }
}

// 2-warp-group M<=64 W8A16 FP8 MMA: warps 0..3 own m=[0..31], warps 4..7 own m=[32..63],
// each warp-group splits K across 4 warps (K_STEP=256 -> exact 9 iters for K=2304),
// sharing W tiles in L1/L2 (single GDDR7 read of W!) with REG<=112, STACK:0, 2 CTAs/SM.
template <int N_TILES>
__device__ __forceinline__ void run_w8a16_fp8_mma_m64_2wg_pdl(
    const B* __restrict__ A,
    const uint8_t* __restrict__ W,
    const float* __restrict__ S,
    B* __restrict__ C,
    int M, int N, int K) {
  constexpr int M_CAP = 64;
  constexpr int M_TILES_WG = 2;  // 32 rows per warp-group
  if (M <= 0 || N <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_fp8_v4_aligned(A, W, K)) {
    fallback_gemm_w8a16_fp8(A, W, S, C, M, N, K);
    return;
  }

  constexpr int WARPS_K = 4;
  constexpr int K_STEP = 64 * WARPS_K;  // 256 elements/iter (2304 = 9 * 256 exact!)
  constexpr int BN = 8 * N_TILES;
  const int num_tiles = (N + BN - 1) / BN;
  if (blockIdx.x >= num_tiles) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }

  const int tid = threadIdx.x;
  const int warp = tid >> 5;       // 0..7
  const int wg = warp >> 2;        // 0 or 1
  const int kw = warp & 3;         // 0..3
  const int lane = tid & 31;
  const int g = lane >> 2;         // 0..7
  const int t = lane & 3;          // 0..3

  __shared__ float4 s_red[2][WARPS_K - 1][M_TILES_WG][32];  // 6 KiB
  const int k0 = kw * 64;
  const bool c_align4 = ((N & 1) == 0) && ((reinterpret_cast<uintptr_t>(C) & 3u) == 0u);
  bool did_wait = false;

  for (int m_base = 0; m_base < M; m_base += M_CAP) {
    const int wg_m_base = m_base + wg * 32;
    const int m_rem = M - wg_m_base;
    const B* a_base = A + static_cast<int64_t>(wg_m_base) * K + t * 16;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
      const int n_base = tile * BN;
      const uint8_t* w_base[N_TILES];
      bool w_ok[N_TILES];
      uint4 w_cur[N_TILES];
#pragma unroll
      for (int nt = 0; nt < N_TILES; ++nt) {
        const int n_row = n_base + nt * 8 + g;
        w_ok[nt] = (n_row < N);
        w_base[nt] = W + static_cast<int64_t>(n_row) * K + t * 16;
        w_cur[nt] = (w_ok[nt] && k0 < K) ? ld_w_fp8_v4(w_base[nt] + k0) : make_uint4(0u, 0u, 0u, 0u);
      }
      if (!did_wait) {
        pdl_wait();
        did_wait = true;
      }

      float acc[M_TILES_WG][N_TILES][4];
#pragma unroll
      for (int mt = 0; mt < M_TILES_WG; ++mt) {
#pragma unroll
        for (int nt = 0; nt < N_TILES; ++nt) {
          acc[mt][nt][0] = 0.0f; acc[mt][nt][1] = 0.0f;
          acc[mt][nt][2] = 0.0f; acc[mt][nt][3] = 0.0f;
        }
      }

      if (m_rem > 0) {
        for (int k = k0; k < K; k += K_STEP) {
          const int k_next = k + K_STEP;
          uint4 w_nxt[N_TILES];
#pragma unroll
          for (int nt = 0; nt < N_TILES; ++nt) {
            w_nxt[nt] = (w_ok[nt] && k_next < K) ? ld_w_fp8_v4(w_base[nt] + k_next) : make_uint4(0u, 0u, 0u, 0u);
          }

#pragma unroll
          for (int mt = 0; mt < M_TILES_WG; ++mt) {
            const int m0 = mt * 16 + g;
            const int m1 = m0 + 8;
            u32x8 x0 = (m0 < m_rem) ? ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k) : u32x8{};
            u32x8 x1 = (m1 < m_rem) ? ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k) : u32x8{};
#pragma unroll
            for (int nt = 0; nt < N_TILES; ++nt) {
              mma_v8_step(acc[mt][nt], x0, x1, unpack_fp8_v4_to_bf16_v8(w_cur[nt]));
            }
          }
#pragma unroll
          for (int nt = 0; nt < N_TILES; ++nt) {
            w_cur[nt] = w_nxt[nt];
          }
        }
      }

      const bool is_last = (m_base + M_CAP >= M) && (tile + gridDim.x >= num_tiles);

#pragma unroll
      for (int nt = 0; nt < N_TILES; ++nt) {
        if (kw > 0) {
#pragma unroll
          for (int mt = 0; mt < M_TILES_WG; ++mt) {
            s_red[wg][kw - 1][mt][lane] =
                make_float4(acc[mt][nt][0], acc[mt][nt][1], acc[mt][nt][2], acc[mt][nt][3]);
          }
        }
        __syncthreads();

        if (nt == N_TILES - 1 && is_last) {
          pdl_launch_dependents();
        }

        if (kw == 0 && m_rem > 0) {
          const int col0 = n_base + nt * 8 + 2 * t;
          const int col1 = col0 + 1;
          const bool c0_ok = (col0 < N);
          const bool c1_ok = (col1 < N);
          if (c0_ok) {
            const float s0 = S[col0];
            const float s1 = c1_ok ? S[col1] : 0.0f;
#pragma unroll
            for (int mt = 0; mt < M_TILES_WG; ++mt) {
#pragma unroll
              for (int w = 0; w < WARPS_K - 1; ++w) {
                const float4 v = s_red[wg][w][mt][lane];
                acc[mt][nt][0] += v.x; acc[mt][nt][1] += v.y;
                acc[mt][nt][2] += v.z; acc[mt][nt][3] += v.w;
              }
              const int m0 = mt * 16 + g;
              if (m0 < m_rem) {
                const float v0 = acc[mt][nt][0] * s0;
                const float v1 = acc[mt][nt][1] * s1;
                B* dst = C + static_cast<int64_t>(wg_m_base + m0) * N + col0;
                if (c1_ok && c_align4) {
                  *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v0, v1);
                } else {
                  dst[0] = __float2bfloat16_rn(v0);
                  if (c1_ok) dst[1] = __float2bfloat16_rn(v1);
                }
              }
              const int m1 = m0 + 8;
              if (m1 < m_rem) {
                const float v2 = acc[mt][nt][2] * s0;
                const float v3 = acc[mt][nt][3] * s1;
                B* dst = C + static_cast<int64_t>(wg_m_base + m1) * N + col0;
                if (c1_ok && c_align4) {
                  *reinterpret_cast<__nv_bfloat162*>(dst) = __floats2bfloat162_rn(v2, v3);
                } else {
                  dst[0] = __float2bfloat16_rn(v2);
                  if (c1_ok) dst[1] = __float2bfloat16_rn(v3);
                }
              }
            }
          }
        }
        if (nt + 1 < N_TILES || !is_last) {
          __syncthreads();
        }
      }
    }
  }
}

}  // namespace ds_gemm

// W8A16 FP8 DSpark lm_head & linear projection entry points (N_TILES=2 for M>16)
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_lm_head_fp8_m8_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_w8a16_fp8_mma_pdl<1, true, 1, true>(A, W, S, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_lm_head_fp8_m16_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_w8a16_fp8_mma_pdl<1, false, 1, true>(A, W, S, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 2) void kern_dspark_lm_head_fp8_m32_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_w8a16_fp8_mma_pdl<2, false, 2, true>(A, W, S, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 2) void kern_dspark_lm_head_fp8_m64_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_w8a16_fp8_mma_m64_2wg_pdl<2>(A, W, S, C, M, N, K);
}

// W8A16 FP8 Fused GateUp + SiLU*Mul entry points (M <= 64, single W pass)
extern "C" __global__ __launch_bounds__(256, 3) void kern_dspark_gate_up_silu_fp8_m8_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_w8a16_fp8_gate_up_silu_pdl<1, true, true>(A, W, S, Y, M, N_half, K);
}
extern "C" __global__ __launch_bounds__(256, 3) void kern_dspark_gate_up_silu_fp8_m16_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_w8a16_fp8_gate_up_silu_pdl<1, false, true>(A, W, S, Y, M, N_half, K);
}
extern "C" __global__ __launch_bounds__(256, 2) void kern_dspark_gate_up_silu_fp8_m32_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_w8a16_fp8_gate_up_silu_pdl<2, false, false>(A, W, S, Y, M, N_half, K);
}
extern "C" __global__ __launch_bounds__(256, 1) void kern_dspark_gate_up_silu_fp8_m64_sm120(
    const B* __restrict__ A, const uint8_t* __restrict__ W, const float* __restrict__ S,
    B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_w8a16_fp8_gate_up_silu_pdl<4, false, false>(A, W, S, Y, M, N_half, K);
}

// BF16-output DSpark GEMV/GEMM entry points (M <= 32, STACK:0)
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_gemv_m1_bf16_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_v8_gemv_m1_pdl<B>(A, W, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_gemm_m8_bf16_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_v8_mma_pdl<1, true, B>(A, W, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_gemm_m16_bf16_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_v8_mma_pdl<1, false, B>(A, W, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 3) void kern_dspark_gemm_m32_bf16_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_v8_mma_pdl<2, false, B>(A, W, C, M, N, K);
}

// F32-output Markov GEMV/GEMM entry points (M <= 8, STACK:0)
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_gemv_m1_f32_sm120(
    const B* __restrict__ A, const B* __restrict__ W, float* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_v8_gemv_m1_pdl<float>(A, W, C, M, N, K);
}
extern "C" __global__ __launch_bounds__(256, 4) void kern_dspark_gemm_m8_f32_sm120(
    const B* __restrict__ A, const B* __restrict__ W, float* __restrict__ C, int M, int N, int K) {
  ds_gemm::run_v8_mma_pdl<1, true, float>(A, W, C, M, N, K);
}

// Fused GateUp + SiLU*Mul entry points (M <= 32, STACK:0)
extern "C" __global__ __launch_bounds__(256, 3) void kern_dspark_gate_up_silu_m8_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_v8_gate_up_silu_pdl<1, true, true>(A, W, Y, M, N_half, K);
}
extern "C" __global__ __launch_bounds__(256, 2) void kern_dspark_gate_up_silu_m16_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_v8_gate_up_silu_pdl<1, false, true>(A, W, Y, M, N_half, K);
}
extern "C" __global__ __launch_bounds__(256, 2) void kern_dspark_gate_up_silu_m32_sm120(
    const B* __restrict__ A, const B* __restrict__ W, B* __restrict__ Y, int M, int N_half, int K) {
  ds_gemm::run_v8_gate_up_silu_pdl<2, false, false>(A, W, Y, M, N_half, K);
}

// ===========================================================================
// Fused verify softcap + vocab mask + FP32 argmax + conditional probs write
// (zero DRAM writes to probs for greedy temperature <= 0.0f rows).
// ===========================================================================
namespace ds_verify_sample {

struct ArgmaxPair {
  float v;
  int i;
};

__device__ __forceinline__ ArgmaxPair pick(ArgmaxPair a, ArgmaxPair b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

__device__ __forceinline__ ArgmaxPair warp_reduce_argmax(ArgmaxPair best) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    float ov = __shfl_xor_sync(0xffffffffu, best.v, off);
    int oi = __shfl_xor_sync(0xffffffffu, best.i, off);
    best = pick(best, {ov, oi});
  }
  return best;
}

__device__ __forceinline__ float softcap_val(float x, float softcap) {
  return (softcap > 0.0f) ? (softcap * tanhf(x / softcap)) : x;
}

}  // namespace ds_verify_sample

// Grid: [tokens, 64, 1], Block: [256, 1, 1]
extern "C" __global__ __launch_bounds__(256) void kern_verify_softcap_argmax_prep_partial_f32(
    const float* __restrict__ logits,
    const float* __restrict__ temperature,
    float* __restrict__ probs,
    float* __restrict__ pmax,
    int* __restrict__ pidx,
    float softcap,
    int vocab_padded,
    int vocab_unpadded) {
  const int row = blockIdx.x;
  const int nb = gridDim.y;
  const int tid = threadIdx.x;
  const bool stochastic = (temperature[row] > 0.0f);
  const float neg_inf = -__builtin_inff();

  const float* row_in = logits + static_cast<int64_t>(row) * vocab_padded;
  float* row_out = probs + static_cast<int64_t>(row) * vocab_padded;

  ds_verify_sample::ArgmaxPair best = {neg_inf, 0x7fffffff};

  if ((vocab_padded & 3) == 0
      && ((reinterpret_cast<uintptr_t>(row_in) & 15u) == 0u)
      && (!stochastic || ((reinterpret_cast<uintptr_t>(row_out) & 15u) == 0u))) {
    const int n4 = vocab_padded >> 2;
    const int chunk4 = (n4 + nb - 1) / nb;
    const int lo4 = blockIdx.y * chunk4;
    const int hi4 = min(lo4 + chunk4, n4);
    const float4* in4 = reinterpret_cast<const float4*>(row_in);
    float4* out4 = reinterpret_cast<float4*>(row_out);
    for (int j4 = lo4 + tid; j4 < hi4; j4 += 256) {
      const int col0 = j4 << 2;
      float4 v;
      if (col0 + 3 < vocab_unpadded) {
        const float4 x = in4[j4];
        v.x = ds_verify_sample::softcap_val(x.x, softcap);
        v.y = ds_verify_sample::softcap_val(x.y, softcap);
        v.z = ds_verify_sample::softcap_val(x.z, softcap);
        v.w = ds_verify_sample::softcap_val(x.w, softcap);
      } else if (col0 >= vocab_unpadded) {
        v = make_float4(neg_inf, neg_inf, neg_inf, neg_inf);
      } else {
        const float4 x = in4[j4];
        v.x = (col0 + 0 < vocab_unpadded) ? ds_verify_sample::softcap_val(x.x, softcap) : neg_inf;
        v.y = (col0 + 1 < vocab_unpadded) ? ds_verify_sample::softcap_val(x.y, softcap) : neg_inf;
        v.z = (col0 + 2 < vocab_unpadded) ? ds_verify_sample::softcap_val(x.z, softcap) : neg_inf;
        v.w = (col0 + 3 < vocab_unpadded) ? ds_verify_sample::softcap_val(x.w, softcap) : neg_inf;
      }
      if (stochastic) {
        out4[j4] = v;
      }
      best = ds_verify_sample::pick(best, {isnan(v.x) ? neg_inf : v.x, col0 + 0});
      best = ds_verify_sample::pick(best, {isnan(v.y) ? neg_inf : v.y, col0 + 1});
      best = ds_verify_sample::pick(best, {isnan(v.z) ? neg_inf : v.z, col0 + 2});
      best = ds_verify_sample::pick(best, {isnan(v.w) ? neg_inf : v.w, col0 + 3});
    }
  } else {
    const int chunk = (vocab_padded + nb - 1) / nb;
    const int lo = blockIdx.y * chunk;
    const int hi = min(lo + chunk, vocab_padded);
    for (int col = lo + tid; col < hi; col += 256) {
      const float v = (col < vocab_unpadded)
                          ? ds_verify_sample::softcap_val(row_in[col], softcap)
                          : neg_inf;
      if (stochastic) row_out[col] = v;
      best = ds_verify_sample::pick(best, {isnan(v) ? neg_inf : v, col});
    }
  }

  best = ds_verify_sample::warp_reduce_argmax(best);
  __shared__ float s_val[8];
  __shared__ int s_idx[8];
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    s_val[warp] = best.v;
    s_idx[warp] = best.i;
  }
  __syncthreads();
  if (warp == 0) {
    ds_verify_sample::ArgmaxPair wb = (lane < 8)
                                          ? ds_verify_sample::ArgmaxPair{s_val[lane], s_idx[lane]}
                                          : ds_verify_sample::ArgmaxPair{neg_inf, 0x7fffffff};
    wb = ds_verify_sample::warp_reduce_argmax(wb);
    if (lane == 0) {
      pmax[row * nb + blockIdx.y] = wb.v;
      pidx[row * nb + blockIdx.y] = wb.i;
    }
  }
  ds_gemm::pdl_launch_dependents();
}

// Grid: [tokens, 1, 1], Block: [64, 1, 1], pdl: True
extern "C" __global__ __launch_bounds__(64) void kern_verify_argmax_final_and_prep(
    const float* __restrict__ pmax,
    const int* __restrict__ pidx,
    const float* __restrict__ temperature,
    const int* __restrict__ top_k,
    const int* __restrict__ seq_len,
    long long* __restrict__ greedy,
    float* __restrict__ chain_temperature,
    int* __restrict__ chain_top_k,
    unsigned long long* __restrict__ offset,
    unsigned* __restrict__ states,
    int tokens,
    int vocab,
    int state_words,
    int nb) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const float neg_inf = -__builtin_inff();

  // Prepare sampling params while waiting for partial argmax via PDL.
  if (tid == 0) {
    const float t = temperature[row];
    chain_temperature[row] = (t > 0.0f) ? t : 1.0f;
    const int k = top_k[row];
    chain_top_k[row] = (k >= 1 && k < vocab) ? k : vocab;
    offset[row] = static_cast<unsigned long long>(seq_len[row]);
  }
  if (row == 0) {
    bool any_stoch = false;
    for (int i = tid; i < tokens; i += 64) {
      if (temperature[i] > 0.0f) any_stoch = true;
    }
    if (__syncthreads_or(any_stoch)) {
      for (int i = tid; i < state_words; i += 64) {
        states[i] = 0u;
      }
    }
  }

  ds_gemm::pdl_wait();
  ds_verify_sample::ArgmaxPair best = {neg_inf, 0x7fffffff};
  if (tid < nb) {
    best = {pmax[row * nb + tid], pidx[row * nb + tid]};
  }
  best = ds_verify_sample::warp_reduce_argmax(best);
  __shared__ float s_val[2];
  __shared__ int s_idx[2];
  const int warp = tid >> 5;
  const int lane = tid & 31;
  if (lane == 0) {
    s_val[warp] = best.v;
    s_idx[warp] = best.i;
  }
  __syncthreads();
  if (tid == 0) {
    const ds_verify_sample::ArgmaxPair fin = ds_verify_sample::pick(
        {s_val[0], s_idx[0]}, {s_val[1], s_idx[1]});
    greedy[row] = static_cast<long long>(fin.i);
  }
}
