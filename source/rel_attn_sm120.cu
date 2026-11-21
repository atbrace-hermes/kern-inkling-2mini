// SPDX-License-Identifier: Apache-2.0
// SM120 (sm_120a) Relative-Bias Attention CUDA Kernels for Inkling-2-Mini.
//
// Uses 16-byte cp.async.cg.shared.global double-buffered FP8 KV staging,
// cooperative FP8 E4M3 -> swizzled BF16 conversion,
// bank-conflict-free ldmatrix.m8n8.x4 / .x4.trans, and native mma.sync.m16n8k16
// (BF16 inputs, FP32 accumulation) with fused relative bias + Gate/Sink epilogue.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>

using bf16 = __nv_bfloat16;

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = 3;
constexpr int kHeadDim = 128;
constexpr int kRelExtent = 1024;
constexpr int kSwaRing = 1024;
constexpr int kSwaWindow = 1024;
constexpr int kBlockN = 64;

constexpr int64_t kSeqStateBytes = 90005376LL;
constexpr int64_t kSwaHeadRingBytes = 131072LL;   // 1024 * 128
constexpr int64_t kSwaHalfVisitBytes = 1048576LL; // 8 * 1024 * 128

constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigmoid_exact(float x) {
  return __fdividef(1.0f, 1.0f + ex2_approx(-x * kLog2E));
}

__device__ __forceinline__ uint32_t cvt_e4m3x2_to_bf16x2(uint16_t raw) {
  uint32_t f16_2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(f16_2) : "h"(raw));
  __half2 h2 = *reinterpret_cast<const __half2*>(&f16_2);
  float2 f2 = __half22float2(h2);
  uint32_t out;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(f2.x), "f"(f2.y));
  return out;
}

__device__ __forceinline__ uint32_t pack_f32x2_to_bf16x2(float a, float b) {
  uint32_t out;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(out) : "f"(a), "f"(b));
  return out;
}

__device__ __forceinline__ void cp_async_cg_16(void* smem_dst, const void* gmem_src, int src_bytes) {
  uint32_t dst_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile(
      "cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
      :: "r"(dst_addr), "l"(gmem_src), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void ldmatrix_x4(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_trans(
    uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(addr));
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float& c0, float& c1, float& c2, float& c3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Swizzle 16-byte chunk index (0..15) by row & 7 for bank-conflict-free [64][128] f16 smem.
__device__ __forceinline__ int swizzle_c16(int row, int c16) {
  return c16 ^ (row & 7);
}

// Cooperatively issue 16B cp.async.cg for one [64, 128] FP8 tile (K or V) from SWA ring or KCur/VCur.
template <int THREADS>
__device__ __forceinline__ void load_swa_kv_stage(
    uint8_t* __restrict__ smem_k_fp8,
    uint8_t* __restrict__ smem_v_fp8,
    const uint8_t* __restrict__ ring_k_base,
    const uint8_t* __restrict__ ring_v_base,
    const uint8_t* __restrict__ kcur_head_base,
    const uint8_t* __restrict__ vcur_head_base,
    int k0,
    int past,
    int seqlen_k,
    int tid) {
  // [64, 128] FP8 = 512 chunks of 16 bytes (8 chunks/row).
  constexpr int kChunksPerRow = kHeadDim / 16;  // 8
  constexpr int kTotalChunks = kBlockN * kChunksPerRow;  // 512
#pragma unroll
  for (int idx = tid; idx < kTotalChunks; idx += THREADS) {
    const int row = idx >> 3;         // 0..63
    const int col = (idx & 7) << 4;   // 0, 16, ..., 112
    const int key = k0 + row;
    const bool valid = (key >= 0) && (key < seqlen_k);
    const int src_bytes = valid ? 16 : 0;
    const uint8_t* k_src;
    const uint8_t* v_src;
    if (key < past) {
      const int64_t off = (static_cast<int64_t>(key & (kSwaRing - 1)) << 7) + col;
      k_src = ring_k_base + off;
      v_src = ring_v_base + off;
    } else {
      const int cur_tok = valid ? (key - past) : 0;
      const int64_t off = static_cast<int64_t>(cur_tok) * (kKvHeads * kHeadDim) + col;
      k_src = kcur_head_base + off;
      v_src = vcur_head_base + off;
    }
    cp_async_cg_16(smem_k_fp8 + row * kHeadDim + col, k_src, src_bytes);
    cp_async_cg_16(smem_v_fp8 + row * kHeadDim + col, v_src, src_bytes);
  }
}

// Cooperatively convert [32, 128] FP8 smem -> swizzled [32, 128] BF16 smem (8 elements per uint4 store).
template <int THREADS>
__device__ __forceinline__ void dequant_fp8_32_to_swizzled_bf16(
    const uint8_t* __restrict__ src_fp8,
    uint4* __restrict__ dst_bf16_swizzled,
    int tid) {
  // 32 rows x 16 chunks of 8 elements (16B in bf16, 8B in fp8) = 512 chunks.
  constexpr int kTotalChunks = 32 * (kHeadDim / 8);  // 512
#pragma unroll
  for (int idx = tid; idx < kTotalChunks; idx += THREADS) {
    const int row = idx >> 4;        // 0..31
    const int c16 = idx & 15;        // 0..15 (8 bf16s = 16 bytes)
    const uint64_t raw8 = *reinterpret_cast<const uint64_t*>(src_fp8 + row * kHeadDim + (c16 << 3));
    uint4 out;
    out.x = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(raw8));
    out.y = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(raw8 >> 16));
    out.z = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(raw8 >> 32));
    out.w = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(raw8 >> 48));
    dst_bf16_swizzled[row * 16 + swizzle_c16(row, c16)] = out;
  }
}

template <int NUM_WARPS>
__device__ __forceinline__ void run_swa_prefill_impl(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ SeqState,
    const uint8_t* __restrict__ KCur,
    const uint8_t* __restrict__ VCur,
    const bf16* __restrict__ Rel,
    const bf16* __restrict__ Gate,
    const bf16* __restrict__ Sink,
    const int* __restrict__ Lines,
    const int* __restrict__ SeqLens,
    bf16* __restrict__ Out,
    int num_tokens,
    int stride_gate,
    int stride_sink,
    float sm_scale,
    float v_scale) {
  constexpr int THREADS = NUM_WARPS * 32;
  constexpr int BLOCK_M = NUM_WARPS * 16;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7
  const int tig = lane & 3;    // 0..3

  const int head = blockIdx.y;
  const int kv_head = head / kGroup;
  const int block_m0 = blockIdx.x * BLOCK_M;
  if (block_m0 >= num_tokens) return;

  const int m0 = block_m0 + warp * 16;
  const int r0 = m0 + gid;
  const int r1 = m0 + gid + 8;
  const bool valid0 = (r0 < num_tokens);
  const bool valid1 = (r1 < num_tokens);

  const int line = Lines[0];
  if (line <= 0) {
    // Null seq_state line -> zero output.
    const int d_col = tig * 2;
    const uint32_t zero2 = 0u;
    if (valid0) {
      bf16* dst0 = Out + static_cast<int64_t>(r0) * (kHeads * kHeadDim) + head * kHeadDim;
#pragma unroll
      for (int di = 0; di < 16; ++di) {
        *reinterpret_cast<uint32_t*>(dst0 + di * 8 + d_col) = zero2;
      }
    }
    if (valid1) {
      bf16* dst1 = Out + static_cast<int64_t>(r1) * (kHeads * kHeadDim) + head * kHeadDim;
#pragma unroll
      for (int di = 0; di < 16; ++di) {
        *reinterpret_cast<uint32_t*>(dst1 + di * 8 + d_col) = zero2;
      }
    }
    return;
  }

  const int seqlen_k = SeqLens[0];
  const int past = seqlen_k - num_tokens;

  const int block_pos_lo = past + block_m0;
  const int block_pos_hi = past + min(block_m0 + BLOCK_M, num_tokens) - 1;
  const int k_lo = max(block_pos_lo - (kSwaWindow - 1), 0);
  const int k_hi = min(block_pos_hi + 1, seqlen_k);
  const int n_lo = k_lo / kBlockN;
  const int n_hi = (k_hi + kBlockN - 1) / kBlockN;

  // Dynamic shared memory layout (48 KB total for both m64 and m128):
  //   k_fp8[stage]    = smem + (stage << 13)         (2 x 8 KB = 16 KB)
  //   v_fp8[stage]    = smem + 16384 + (stage << 13) (2 x 8 KB = 16 KB)
  //   k_bf16[32*128]  = smem + 32768                 (8 KB, swizzled)
  //   v_bf16[32*128]  = smem + 40960                 (8 KB, swizzled)
  extern __shared__ __align__(16) uint8_t smem[];
  uint4* k_bf16_swiz = reinterpret_cast<uint4*>(smem + 32768);
  uint4* v_bf16_swiz = reinterpret_cast<uint4*>(smem + 40960);

  const uint8_t* ring_k_base = SeqState + static_cast<int64_t>(line) * kSeqStateBytes + static_cast<int64_t>(kv_head) * kSwaHeadRingBytes;
  const uint8_t* ring_v_base = ring_k_base + kSwaHalfVisitBytes;
  const uint8_t* kcur_head_base = KCur + static_cast<int64_t>(kv_head) * kHeadDim;
  const uint8_t* vcur_head_base = VCur + static_cast<int64_t>(kv_head) * kHeadDim;

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Load Q [16, 128] for this warp directly into BF16 MMA registers q_reg[8][4] (zero conversion!).
  uint32_t q_reg[8][4];
  const bf16* q_row0 = valid0 ? (Q + static_cast<int64_t>(r0) * (kHeads * kHeadDim) + head * kHeadDim) : nullptr;
  const bf16* q_row1 = valid1 ? (Q + static_cast<int64_t>(r1) * (kHeads * kHeadDim) + head * kHeadDim) : nullptr;
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const int d0 = ks * 16 + tig * 2;
    const int d1 = d0 + 8;
    q_reg[ks][0] = valid0 ? *reinterpret_cast<const uint32_t*>(q_row0 + d0) : 0u;
    q_reg[ks][1] = valid1 ? *reinterpret_cast<const uint32_t*>(q_row1 + d0) : 0u;
    q_reg[ks][2] = valid0 ? *reinterpret_cast<const uint32_t*>(q_row0 + d1) : 0u;
    q_reg[ks][3] = valid1 ? *reinterpret_cast<const uint32_t*>(q_row1 + d1) : 0u;
  }

  float acc[16][4];
#pragma unroll
  for (int di = 0; di < 16; ++di) {
    acc[di][0] = 0.0f;
    acc[di][1] = 0.0f;
    acc[di][2] = 0.0f;
    acc[di][3] = 0.0f;
  }
  float m_i0 = -INFINITY;
  float m_i1 = -INFINITY;
  float l_i0 = 0.0f;
  float l_i1 = 0.0f;

  const float qk_scale = sm_scale * kLog2E;
  const int pos0 = past + r0;
  const int pos1 = past + r1;
  const int warp_pos_lo = past + m0;
  const int warp_pos_hi = past + min(m0 + 16, num_tokens) - 1;
  const bf16* rel_row0 = valid0 ? (Rel + static_cast<int64_t>(r0) * (kHeads * kRelExtent) + head * kRelExtent) : nullptr;
  const bf16* rel_row1 = valid1 ? (Rel + static_cast<int64_t>(r1) * (kHeads * kRelExtent) + head * kRelExtent) : nullptr;

  // Prime pipeline stage 0.
  if (n_lo < n_hi) {
    load_swa_kv_stage<THREADS>(
        smem, smem + 16384,
        ring_k_base, ring_v_base, kcur_head_base, vcur_head_base,
        n_lo * kBlockN, past, seqlen_k, tid);
    cp_async_commit();
  }

  const int k_ld_row_off = ((lane >> 4) << 3) + (lane & 7);
  const int k_ld_c16_off = (lane >> 3) & 1;
  const int v_ld_row_off = lane & 15;
  const int v_ld_c16_off = lane >> 4;

  for (int n = n_lo; n < n_hi; ++n) {
    const int stage = (n - n_lo) & 1;
    const int next_stage = stage ^ 1;
    const int k0 = n * kBlockN;

    // Issue async copy for next tile (overlapped with current tile's compute).
    if (n + 1 < n_hi) {
      load_swa_kv_stage<THREADS>(
          smem + (next_stage << 13), smem + 16384 + (next_stage << 13),
          ring_k_base, ring_v_base, kcur_head_base, vcur_head_base,
          (n + 1) * kBlockN, past, seqlen_k, tid);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Process two 32-col sub-tiles (matches baseline BLOCK_N=32 softmax order & halves live s regs).
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int k_sub = k0 + half * 32;
      if (half > 0) {
        __syncthreads();
      }
      // Cooperatively dequantize 32 rows of K and V from FP8 -> swizzled BF16 SMEM.
      dequant_fp8_32_to_swizzled_bf16<THREADS>(smem + (stage << 13) + (half << 12), k_bf16_swiz, tid);
      dequant_fp8_32_to_swizzled_bf16<THREADS>(smem + 16384 + (stage << 13) + (half << 12), v_bf16_swiz, tid);
      __syncthreads();

      // 1. Q @ K^T -> s[4][4] (16 x 32)
      float s[4][4];
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        s[ni][0] = 0.0f;
        s[ni][1] = 0.0f;
        s[ni][2] = 0.0f;
        s[ni][3] = 0.0f;
      }

#pragma unroll
      for (int ks = 0; ks < 8; ++ks) {
        const int c16_base = ks * 2 + k_ld_c16_off;
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int r = np * 16 + k_ld_row_off;
          uint32_t b0, b1, b2, b3;
          ldmatrix_x4(b0, b1, b2, b3, &k_bf16_swiz[r * 16 + swizzle_c16(r, c16_base)]);
          mma_m16n8k16_bf16(
              s[np * 2][0], s[np * 2][1], s[np * 2][2], s[np * 2][3],
              q_reg[ks][0], q_reg[ks][1], q_reg[ks][2], q_reg[ks][3],
              b0, b1);
          mma_m16n8k16_bf16(
              s[np * 2 + 1][0], s[np * 2 + 1][1], s[np * 2 + 1][2], s[np * 2 + 1][3],
              q_reg[ks][0], q_reg[ks][1], q_reg[ks][2], q_reg[ks][3],
              b2, b3);
        }
      }

      // 2. Scale + Rel bias + causal/window mask
      const bool all_visible =
          (k_sub + 31 <= warp_pos_lo) &&
          (warp_pos_hi - k_sub < kSwaWindow) &&
          (k_sub + 32 <= seqlen_k) &&
          (m0 + 16 <= num_tokens);

      if (all_visible) {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int col0 = k_sub + ni * 8 + tig * 2;
          const int d0 = pos0 - col0;
          const int d1 = pos1 - col0;
          const float b00 = __bfloat162float(__ldg(rel_row0 + d0));
          const float b01 = __bfloat162float(__ldg(rel_row0 + d0 - 1));
          const float b10 = __bfloat162float(__ldg(rel_row1 + d1));
          const float b11 = __bfloat162float(__ldg(rel_row1 + d1 - 1));
          s[ni][0] = __fmaf_rn(s[ni][0], qk_scale, __fmul_rn(b00, kLog2E));
          s[ni][1] = __fmaf_rn(s[ni][1], qk_scale, __fmul_rn(b01, kLog2E));
          s[ni][2] = __fmaf_rn(s[ni][2], qk_scale, __fmul_rn(b10, kLog2E));
          s[ni][3] = __fmaf_rn(s[ni][3], qk_scale, __fmul_rn(b11, kLog2E));
        }
      } else {
#pragma unroll
        for (int ni = 0; ni < 4; ++ni) {
          const int col0 = k_sub + ni * 8 + tig * 2;
          const int col1 = col0 + 1;
          const bool kv0 = (col0 < seqlen_k);
          const bool kv1 = (col1 < seqlen_k);

          const int d00 = pos0 - col0;
          const int d01 = pos0 - col1;
          const bool vis00 = kv0 && (static_cast<unsigned>(d00) < static_cast<unsigned>(kSwaWindow));
          const bool vis01 = kv1 && (static_cast<unsigned>(d01) < static_cast<unsigned>(kSwaWindow));
          if (vis00) {
            float b = valid0 ? __bfloat162float(__ldg(rel_row0 + d00)) : 0.0f;
            s[ni][0] = __fmaf_rn(s[ni][0], qk_scale, __fmul_rn(b, kLog2E));
          } else {
            s[ni][0] = -INFINITY;
          }
          if (vis01) {
            float b = valid0 ? __bfloat162float(__ldg(rel_row0 + d01)) : 0.0f;
            s[ni][1] = __fmaf_rn(s[ni][1], qk_scale, __fmul_rn(b, kLog2E));
          } else {
            s[ni][1] = -INFINITY;
          }

          const int d10 = pos1 - col0;
          const int d11 = pos1 - col1;
          const bool vis10 = kv0 && (static_cast<unsigned>(d10) < static_cast<unsigned>(kSwaWindow));
          const bool vis11 = kv1 && (static_cast<unsigned>(d11) < static_cast<unsigned>(kSwaWindow));
          if (vis10) {
            float b = valid1 ? __bfloat162float(__ldg(rel_row1 + d10)) : 0.0f;
            s[ni][2] = __fmaf_rn(s[ni][2], qk_scale, __fmul_rn(b, kLog2E));
          } else {
            s[ni][2] = -INFINITY;
          }
          if (vis11) {
            float b = valid1 ? __bfloat162float(__ldg(rel_row1 + d11)) : 0.0f;
            s[ni][3] = __fmaf_rn(s[ni][3], qk_scale, __fmul_rn(b, kLog2E));
          } else {
            s[ni][3] = -INFINITY;
          }
        }
      }

      // 3. Online softmax + P @ V for this 32-col sub-tile
      float row_max0 = m_i0;
      float row_max1 = m_i1;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        row_max0 = fmaxf(row_max0, fmaxf(s[j][0], s[j][1]));
        row_max1 = fmaxf(row_max1, fmaxf(s[j][2], s[j][3]));
      }
      row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffffu, row_max0, 1));
      row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffffu, row_max0, 2));
      row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffffu, row_max1, 1));
      row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffffu, row_max1, 2));

      const float m_safe0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
      const float m_safe1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
      const float alpha0 = ex2_approx(m_i0 - m_safe0);
      const float alpha1 = ex2_approx(m_i1 - m_safe1);
      m_i0 = row_max0;
      m_i1 = row_max1;

      uint32_t pa[2][4];
      float psum0 = 0.0f;
      float psum1 = 0.0f;
#pragma unroll
      for (int kp_sub = 0; kp_sub < 2; ++kp_sub) {
        const float p00 = ex2_approx(s[kp_sub * 2][0] - m_safe0);
        const float p01 = ex2_approx(s[kp_sub * 2][1] - m_safe0);
        const float p02 = ex2_approx(s[kp_sub * 2][2] - m_safe1);
        const float p03 = ex2_approx(s[kp_sub * 2][3] - m_safe1);
        const float p10 = ex2_approx(s[kp_sub * 2 + 1][0] - m_safe0);
        const float p11 = ex2_approx(s[kp_sub * 2 + 1][1] - m_safe0);
        const float p12 = ex2_approx(s[kp_sub * 2 + 1][2] - m_safe1);
        const float p13 = ex2_approx(s[kp_sub * 2 + 1][3] - m_safe1);
        psum0 += (p00 + p01) + (p10 + p11);
        psum1 += (p02 + p03) + (p12 + p13);
        pa[kp_sub][0] = pack_f32x2_to_bf16x2(p00, p01);
        pa[kp_sub][1] = pack_f32x2_to_bf16x2(p02, p03);
        pa[kp_sub][2] = pack_f32x2_to_bf16x2(p10, p11);
        pa[kp_sub][3] = pack_f32x2_to_bf16x2(p12, p13);
      }
      psum0 += __shfl_xor_sync(0xffffffffu, psum0, 1);
      psum0 += __shfl_xor_sync(0xffffffffu, psum0, 2);
      psum1 += __shfl_xor_sync(0xffffffffu, psum1, 1);
      psum1 += __shfl_xor_sync(0xffffffffu, psum1, 2);
      l_i0 = __fmaf_rn(l_i0, alpha0, psum0);
      l_i1 = __fmaf_rn(l_i1, alpha1, psum1);

      const int r_k0 = v_ld_row_off;
      const int r_k1 = 16 + v_ld_row_off;
#pragma unroll
      for (int dp = 0; dp < 8; ++dp) {
        const int c16 = dp * 2 + v_ld_c16_off;
        acc[dp * 2][0] = __fmul_rn(acc[dp * 2][0], alpha0);
        acc[dp * 2][1] = __fmul_rn(acc[dp * 2][1], alpha0);
        acc[dp * 2][2] = __fmul_rn(acc[dp * 2][2], alpha1);
        acc[dp * 2][3] = __fmul_rn(acc[dp * 2][3], alpha1);
        acc[dp * 2 + 1][0] = __fmul_rn(acc[dp * 2 + 1][0], alpha0);
        acc[dp * 2 + 1][1] = __fmul_rn(acc[dp * 2 + 1][1], alpha0);
        acc[dp * 2 + 1][2] = __fmul_rn(acc[dp * 2 + 1][2], alpha1);
        acc[dp * 2 + 1][3] = __fmul_rn(acc[dp * 2 + 1][3], alpha1);
        uint32_t vb0, vb1, vb2, vb3;
        ldmatrix_x4_trans(vb0, vb1, vb2, vb3, &v_bf16_swiz[r_k0 * 16 + swizzle_c16(r_k0, c16)]);
        mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3], pa[0][0], pa[0][1], pa[0][2], pa[0][3], vb0, vb1);
        mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3], pa[0][0], pa[0][1], pa[0][2], pa[0][3], vb2, vb3);
        ldmatrix_x4_trans(vb0, vb1, vb2, vb3, &v_bf16_swiz[r_k1 * 16 + swizzle_c16(r_k1, c16)]);
        mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3], pa[1][0], pa[1][1], pa[1][2], pa[1][3], vb0, vb1);
        mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3], pa[1][0], pa[1][1], pa[1][2], pa[1][3], vb2, vb3);
      }
    }
  }

  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const bool empty0 = (l_i0 == 0.0f);
  const bool empty1 = (l_i1 == 0.0f);
  const float l0_safe = empty0 ? 1.0f : l_i0;
  const float l1_safe = empty1 ? 1.0f : l_i1;
  const float lse0_nat = empty0 ? -INFINITY : __fmul_rn(m_i0 + log2f(l0_safe), kLn2);
  const float lse1_nat = empty1 ? -INFINITY : __fmul_rn(m_i1 + log2f(l1_safe), kLn2);

  const float sink0 = valid0 ? __bfloat162float(Sink[static_cast<int64_t>(r0) * stride_sink + head]) : 0.0f;
  const float sink1 = valid1 ? __bfloat162float(Sink[static_cast<int64_t>(r1) * stride_sink + head]) : 0.0f;
  const float sink_g0 = sigmoid_exact(__fsub_rn(lse0_nat, sink0));
  const float sink_g1 = sigmoid_exact(__fsub_rn(lse1_nat, sink1));

  if (valid0) {
    const bf16* gate0 = Gate + static_cast<int64_t>(r0) * stride_gate + head * kHeadDim;
    bf16* dst0 = Out + static_cast<int64_t>(r0) * (kHeads * kHeadDim) + head * kHeadDim;
#pragma unroll
    for (int di = 0; di < 16; ++di) {
      const int d = di * 8 + tig * 2;
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(gate0 + d);
      const float2 gf = __bfloat1622float2(g2);
      const float y0 = (__fdividef(acc[di][0], l0_safe) * v_scale) * sigmoid_exact(gf.x) * sink_g0;
      const float y1 = (__fdividef(acc[di][1], l0_safe) * v_scale) * sigmoid_exact(gf.y) * sink_g0;
      *reinterpret_cast<__nv_bfloat162*>(dst0 + d) = __floats2bfloat162_rn(y0, y1);
    }
  }
  if (valid1) {
    const bf16* gate1 = Gate + static_cast<int64_t>(r1) * stride_gate + head * kHeadDim;
    bf16* dst1 = Out + static_cast<int64_t>(r1) * (kHeads * kHeadDim) + head * kHeadDim;
#pragma unroll
    for (int di = 0; di < 16; ++di) {
      const int d = di * 8 + tig * 2;
      const __nv_bfloat162 g2 = *reinterpret_cast<const __nv_bfloat162*>(gate1 + d);
      const float2 gf = __bfloat1622float2(g2);
      const float y0 = (__fdividef(acc[di][2], l1_safe) * v_scale) * sigmoid_exact(gf.x) * sink_g1;
      const float y1 = (__fdividef(acc[di][3], l1_safe) * v_scale) * sigmoid_exact(gf.y) * sink_g1;
      *reinterpret_cast<__nv_bfloat162*>(dst1 + d) = __floats2bfloat162_rn(y0, y1);
    }
  }
}

}  // namespace

// BLOCK_M=128, BLOCK_N=64, 8 warps (256 threads), 48 KB dynamic SMEM.
extern "C" __global__ void __launch_bounds__(256, 1) kern_rel_swa_prefill_m128_sm120(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ SeqState,
    const uint8_t* __restrict__ KCur,
    const uint8_t* __restrict__ VCur,
    const bf16* __restrict__ Rel,
    const bf16* __restrict__ Gate,
    const bf16* __restrict__ Sink,
    const int* __restrict__ Lines,
    const int* __restrict__ SeqLens,
    bf16* __restrict__ Out,
    int num_tokens,
    int stride_gate,
    int stride_sink,
    float sm_scale,
    float v_scale) {
  run_swa_prefill_impl<8>(
      Q, SeqState, KCur, VCur, Rel, Gate, Sink, Lines, SeqLens, Out,
      num_tokens, stride_gate, stride_sink, sm_scale, v_scale);
}

// BLOCK_M=64, BLOCK_N=64, 4 warps (128 threads), 48 KB dynamic SMEM (2 CTAs/SM).
extern "C" __global__ void __launch_bounds__(128, 2) kern_rel_swa_prefill_m64_sm120(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ SeqState,
    const uint8_t* __restrict__ KCur,
    const uint8_t* __restrict__ VCur,
    const bf16* __restrict__ Rel,
    const bf16* __restrict__ Gate,
    const bf16* __restrict__ Sink,
    const int* __restrict__ Lines,
    const int* __restrict__ SeqLens,
    bf16* __restrict__ Out,
    int num_tokens,
    int stride_gate,
    int stride_sink,
    float sm_scale,
    float v_scale) {
  run_swa_prefill_impl<4>(
      Q, SeqState, KCur, VCur, Rel, Gate, Sink, Lines, SeqLens, Out,
      num_tokens, stride_gate, stride_sink, sm_scale, v_scale);
}
