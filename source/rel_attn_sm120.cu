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
constexpr int kRelRow = 72;  // 72 bf16 = 144 B = 9 x 16B chunks (36 banks -> minimal bank conflicts)
constexpr int kSwaRing = 1024;
constexpr int kSwaWindow = 1024;
constexpr int kBlockN = 64;
constexpr int kPage = 64;
constexpr int kBulkAlign = 128;

constexpr int64_t kSeqStateBytes = 90005376LL;
constexpr int64_t kSwaHeadRingBytes = 131072LL;   // 1024 * 128
constexpr int64_t kSwaHalfVisitBytes = 1048576LL; // 8 * 1024 * 128
constexpr int64_t kStrideKvBlk = 9LL * 2 * kKvHeads * kPage * kHeadDim;  // 1,179,648 B
constexpr int64_t kStrideKvHalf = kKvHeads * kPage * kHeadDim;           // 65,536 B
constexpr int64_t kStrideKvHead = kPage * kHeadDim;                      // 8,192 B

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
  uint32_t out;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(out) : "h"(raw));
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

__device__ __forceinline__ void ldmatrix_x2_trans_b8(
    uint32_t& d0, uint32_t& d1, uint32_t& d2, uint32_t& d3, const void* smem_ptr) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
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

// Swizzle 16B chunk index (0..7) by row & 7 for bank-conflict-free [64, 128] FP8 smem.
__device__ __forceinline__ int swizzle_fp8_c16(int row, int c16) {
  return c16 ^ (row & 7);
}

// Permute key index within each 16-row V tile so ldmatrix.m16n16.x2.trans.b8 + cvt.rn.bf16x2.e4m3x2
// directly matches the Q@K^T C-fragment register layout of P with zero warp shuffles.
__device__ __forceinline__ int permute_v_row_16(int row) {
  return (row & ~15) | (row & 1) | ((row & 8) >> 2) | ((row & 6) << 1);
}

// Cooperatively issue 16B cp.async.cg for one [64, 128] FP8 tile (K swizzled, V permuted+swizzled)
// from SWA ring or KCur/VCur directly into FP8 SMEM (zero SMEM dequantization!).
template <int THREADS>
__device__ __forceinline__ void load_swa_kv_stage_swizzled(
    uint4* __restrict__ smem_k_swiz,
    uint4* __restrict__ smem_v_swiz,
    const uint8_t* __restrict__ ring_k_base,
    const uint8_t* __restrict__ ring_v_base,
    const uint8_t* __restrict__ kcur_head_base,
    const uint8_t* __restrict__ vcur_head_base,
    int k0,
    int past,
    int seqlen_k,
    int tid) {
  constexpr int kTotalChunks = kBlockN * (kHeadDim / 16);  // 512
  const bool all_ring = (k0 + kBlockN <= past);
  const bool all_cur = (k0 >= past) && (k0 + kBlockN <= seqlen_k);
#pragma unroll
  for (int idx = tid; idx < kTotalChunks; idx += THREADS) {
    const int row = idx >> 3;        // 0..63
    const int c16 = idx & 7;         // 0..7
    const int col = c16 << 4;        // 0, 16, ..., 112
    const int k_swiz = swizzle_fp8_c16(row, c16);
    const int v_row = permute_v_row_16(row);
    const int v_swiz = swizzle_fp8_c16(v_row, c16);
    if (all_ring) {
      const int key = k0 + row;
      const int64_t off = (static_cast<int64_t>(key & (kSwaRing - 1)) << 7) + col;
      cp_async_cg_16(&smem_k_swiz[row * 8 + k_swiz], ring_k_base + off, 16);
      cp_async_cg_16(&smem_v_swiz[v_row * 8 + v_swiz], ring_v_base + off, 16);
    } else if (all_cur) {
      const int cur_tok = (k0 - past) + row;
      const int64_t off = static_cast<int64_t>(cur_tok) * (kKvHeads * kHeadDim) + col;
      cp_async_cg_16(&smem_k_swiz[row * 8 + k_swiz], kcur_head_base + off, 16);
      cp_async_cg_16(&smem_v_swiz[v_row * 8 + v_swiz], vcur_head_base + off, 16);
    } else {
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
      cp_async_cg_16(&smem_k_swiz[row * 8 + k_swiz], k_src, src_bytes);
      cp_async_cg_16(&smem_v_swiz[v_row * 8 + v_swiz], v_src, src_bytes);
    }
  }
}

// Cooperatively issue coalesced 16B cp.async.cg for Rel[block_m0..+BLOCK_M-1, head, d_align..+71].
template <int THREADS, int BLOCK_M>
__device__ __forceinline__ void load_swa_rel_stage(
    bf16* __restrict__ smem_rel,
    const bf16* __restrict__ Rel,
    int block_m0,
    int head,
    int past,
    int k0,
    int num_tokens,
    int tid) {
  constexpr int kChunksPerRow = 9;  // 9 * 8 bf16 = 72 bf16 = 144 B
  constexpr int kTotalChunks = BLOCK_M * kChunksPerRow;
  const int rel_head_off = head * kRelExtent;
#pragma unroll
  for (int idx = tid; idx < kTotalChunks; idx += THREADS) {
    const int rm = idx / kChunksPerRow;
    const int ch = idx - rm * kChunksPerRow;
    const int tok = block_m0 + rm;
    const int d_lo = (past + tok) - (k0 + kBlockN - 1);
    const int d_align = d_lo & ~7;
    const int d_c = d_align + (ch << 3);
    const bool ok = (tok < num_tokens) && (static_cast<unsigned>(d_c) < static_cast<unsigned>(kRelExtent));
    const int safe_tok = (tok < num_tokens) ? tok : 0;
    const int safe_dc = ok ? d_c : 0;
    const bf16* src = Rel + static_cast<int64_t>(safe_tok) * (kHeads * kRelExtent) + rel_head_off + safe_dc;
    cp_async_cg_16(smem_rel + rm * kRelRow + (ch << 3), src, ok ? 16 : 0);
  }
}

template <int NUM_WARPS, int REL_STAGES>
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
  constexpr int kRelStageBytes = BLOCK_M * kRelRow * static_cast<int>(sizeof(bf16));

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
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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

  // Dynamic shared memory layout (zero SMEM FP8->BF16 dequantization!):
  //   kv[stage=0..1]: 2 * (8 KB K + 8 KB V) = 32 KB (smem + stage * 16384)
  //   rel[0..REL_STAGES-1]: REL_STAGES * BLOCK_M * 72 * 2 B (smem + 32768 + s * kRelStageBytes)
  //   m128 (REL_STAGES=2): 32768 + 36864 = 69,632 B (1 CTA/SM, 8 warps/SM)
  //   m64  (REL_STAGES=1): 32768 +  9216 = 41,984 B (2 CTAs/SM, 8 warps/SM)
  extern __shared__ __align__(16) uint8_t smem[];

  const uint8_t* ring_k_base = SeqState + static_cast<int64_t>(line) * kSeqStateBytes + static_cast<int64_t>(kv_head) * kSwaHeadRingBytes;
  const uint8_t* ring_v_base = ring_k_base + kSwaHalfVisitBytes;
  const uint8_t* kcur_head_base = KCur + static_cast<int64_t>(kv_head) * kHeadDim;
  const uint8_t* vcur_head_base = VCur + static_cast<int64_t>(kv_head) * kHeadDim;

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Prime stage 0 (KV + Rel) BEFORE loading Q into registers to hide latency.
  if (n_lo < n_hi) {
    const int k0_first = n_lo * kBlockN;
    load_swa_kv_stage_swizzled<THREADS>(
        reinterpret_cast<uint4*>(smem),
        reinterpret_cast<uint4*>(smem + 8192),
        ring_k_base, ring_v_base, kcur_head_base, vcur_head_base,
        k0_first, past, seqlen_k, tid);
    load_swa_rel_stage<THREADS, BLOCK_M>(
        reinterpret_cast<bf16*>(smem + 32768),
        Rel, block_m0, head, past, k0_first, num_tokens, tid);
    cp_async_commit();
  }

  // Load Q [16, 128] BF16 directly into registers matching in-register FP8->BF16 K layout.
  uint32_t q_reg[8][4];
  const bf16* q_row0 = valid0 ? (Q + static_cast<int64_t>(r0) * (kHeads * kHeadDim) + head * kHeadDim) : nullptr;
  const bf16* q_row1 = valid1 ? (Q + static_cast<int64_t>(r1) * (kHeads * kHeadDim) + head * kHeadDim) : nullptr;
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const int d_base = ks * 16 + tig * 4;
    uint2 qv0 = valid0 ? *reinterpret_cast<const uint2*>(q_row0 + d_base) : make_uint2(0u, 0u);
    uint2 qv1 = valid1 ? *reinterpret_cast<const uint2*>(q_row1 + d_base) : make_uint2(0u, 0u);
    q_reg[ks][0] = qv0.x;
    q_reg[ks][1] = qv1.x;
    q_reg[ks][2] = qv0.y;
    q_reg[ks][3] = qv1.y;
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
  const int rm0 = warp * 16 + gid;
  const int rm1 = rm0 + 8;

  const int k_ld_row_off = ((lane >> 4) << 3) + (lane & 7);
  const int k_ld_c16_off = (lane >> 3) & 1;

  for (int n = n_lo; n < n_hi; ++n) {
    const int stage = (n - n_lo) & 1;
    const int next_stage = stage ^ 1;
    const int k0 = n * kBlockN;

    // Issue next tile's KV (+ Rel if double-buffered) overlapped with current tile's compute!
    if (n + 1 < n_hi) {
      const int k0_next = (n + 1) * kBlockN;
      load_swa_kv_stage_swizzled<THREADS>(
          reinterpret_cast<uint4*>(smem + next_stage * 16384),
          reinterpret_cast<uint4*>(smem + next_stage * 16384 + 8192),
          ring_k_base, ring_v_base, kcur_head_base, vcur_head_base,
          k0_next, past, seqlen_k, tid);
      if constexpr (REL_STAGES == 2) {
        load_swa_rel_stage<THREADS, BLOCK_M>(
            reinterpret_cast<bf16*>(smem + 32768 + next_stage * kRelStageBytes),
            Rel, block_m0, head, past, k0_next, num_tokens, tid);
      }
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    const uint4* k_swiz = reinterpret_cast<const uint4*>(smem + stage * 16384);
    const uint4* v_swiz = reinterpret_cast<const uint4*>(smem + stage * 16384 + 8192);
    const bf16* rel_cur = reinterpret_cast<const bf16*>(
        smem + 32768 + ((REL_STAGES == 2) ? (stage * kRelStageBytes) : 0));

    // Warp-level skip if none of this warp's 16 query rows can attend to [k0 .. k0+63].
    const bool warp_active =
        (m0 < num_tokens) &&
        (k0 <= warp_pos_hi) &&
        (k0 + kBlockN - 1 >= warp_pos_lo - (kSwaWindow - 1));

    const int sh0 = (pos0 - (k0 + kBlockN - 1)) & 7;
    const int sh1 = (pos1 - (k0 + kBlockN - 1)) & 7;
    const bf16* srel0 = rel_cur + rm0 * kRelRow + sh0;
    const bf16* srel1 = rel_cur + rm1 * kRelRow + sh1;

    // Process two 32-key sub-tiles (matches baseline BLOCK_N=32 softmax order & halves live s/pa regs!).
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int k_sub = k0 + half * 32;
      const bool sub_active =
          warp_active &&
          (k_sub <= warp_pos_hi) &&
          (k_sub + 31 >= warp_pos_lo - (kSwaWindow - 1));

      uint32_t pa[2][4];
      float alpha0 = 1.0f;
      float alpha1 = 1.0f;

      if (sub_active) {
        // 1. Q @ K^T -> s[4][4] (16 x 32) via ldmatrix_x4 on FP8 K + 1-instr cvt.rn.bf16x2.e4m3x2
        float s[4][4];
#pragma unroll
        for (int np_l = 0; np_l < 2; ++np_l) {
          const int np = half * 2 + np_l;
          float s0_0 = 0.0f, s0_1 = 0.0f, s0_2 = 0.0f, s0_3 = 0.0f;
          float s1_0 = 0.0f, s1_1 = 0.0f, s1_2 = 0.0f, s1_3 = 0.0f;
          const int r = np * 16 + k_ld_row_off;
#pragma unroll
          for (int ks2 = 0; ks2 < 4; ++ks2) {
            const int c16_base = ks2 * 2 + k_ld_c16_off;
            uint32_t b0, b1, b2, b3;
            ldmatrix_x4(b0, b1, b2, b3, &k_swiz[r * 8 + swizzle_fp8_c16(r, c16_base)]);

            const int ks_lo = ks2 * 2;
            const int ks_hi = ks_lo + 1;

            mma_m16n8k16_bf16(s0_0, s0_1, s0_2, s0_3,
                              q_reg[ks_lo][0], q_reg[ks_lo][1], q_reg[ks_lo][2], q_reg[ks_lo][3],
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0)),
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0 >> 16)));
            mma_m16n8k16_bf16(s1_0, s1_1, s1_2, s1_3,
                              q_reg[ks_lo][0], q_reg[ks_lo][1], q_reg[ks_lo][2], q_reg[ks_lo][3],
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2)),
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2 >> 16)));
            mma_m16n8k16_bf16(s0_0, s0_1, s0_2, s0_3,
                              q_reg[ks_hi][0], q_reg[ks_hi][1], q_reg[ks_hi][2], q_reg[ks_hi][3],
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1)),
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1 >> 16)));
            mma_m16n8k16_bf16(s1_0, s1_1, s1_2, s1_3,
                              q_reg[ks_hi][0], q_reg[ks_hi][1], q_reg[ks_hi][2], q_reg[ks_hi][3],
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3)),
                              cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3 >> 16)));
          }
          s[np_l * 2][0] = s0_0;
          s[np_l * 2][1] = s0_1;
          s[np_l * 2][2] = s0_2;
          s[np_l * 2][3] = s0_3;
          s[np_l * 2 + 1][0] = s1_0;
          s[np_l * 2 + 1][1] = s1_1;
          s[np_l * 2 + 1][2] = s1_2;
          s[np_l * 2 + 1][3] = s1_3;
        }

        // 2. Scale + SMEM Rel bias + causal/window mask + row max
        const bool all_visible =
            (k_sub + 31 <= warp_pos_lo) &&
            (warp_pos_hi - k_sub < kSwaWindow) &&
            (k_sub + 32 <= seqlen_k) &&
            (m0 + 16 <= num_tokens);

        float row_max0 = m_i0;
        float row_max1 = m_i1;

        if (all_visible) {
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            const int rev0 = (kBlockN - 1) - (half * 32 + ni * 8 + tig * 2);
            const int rev1 = rev0 - 1;
            const float b00 = __bfloat162float(srel0[rev0]);
            const float b01 = __bfloat162float(srel0[rev1]);
            const float b10 = __bfloat162float(srel1[rev0]);
            const float b11 = __bfloat162float(srel1[rev1]);
            s[ni][0] = __fmaf_rn(s[ni][0], qk_scale, __fmul_rn(b00, kLog2E));
            s[ni][1] = __fmaf_rn(s[ni][1], qk_scale, __fmul_rn(b01, kLog2E));
            s[ni][2] = __fmaf_rn(s[ni][2], qk_scale, __fmul_rn(b10, kLog2E));
            s[ni][3] = __fmaf_rn(s[ni][3], qk_scale, __fmul_rn(b11, kLog2E));
            row_max0 = fmaxf(row_max0, fmaxf(s[ni][0], s[ni][1]));
            row_max1 = fmaxf(row_max1, fmaxf(s[ni][2], s[ni][3]));
          }
        } else {
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) {
            const int col_local0 = half * 32 + ni * 8 + tig * 2;
            const int col0 = k0 + col_local0;
            const int col1 = col0 + 1;
            const int rev0 = (kBlockN - 1) - col_local0;
            const int rev1 = rev0 - 1;
            const bool kv0 = (col0 < seqlen_k);
            const bool kv1 = (col1 < seqlen_k);

            const int d00 = pos0 - col0;
            const int d01 = pos0 - col1;
            const bool vis00 = kv0 && (static_cast<unsigned>(d00) < static_cast<unsigned>(kSwaWindow));
            const bool vis01 = kv1 && (static_cast<unsigned>(d01) < static_cast<unsigned>(kSwaWindow));
            s[ni][0] = vis00 ? __fmaf_rn(s[ni][0], qk_scale, __fmul_rn(__bfloat162float(srel0[rev0]), kLog2E)) : -INFINITY;
            s[ni][1] = vis01 ? __fmaf_rn(s[ni][1], qk_scale, __fmul_rn(__bfloat162float(srel0[rev1]), kLog2E)) : -INFINITY;

            const int d10 = pos1 - col0;
            const int d11 = pos1 - col1;
            const bool vis10 = kv0 && (static_cast<unsigned>(d10) < static_cast<unsigned>(kSwaWindow));
            const bool vis11 = kv1 && (static_cast<unsigned>(d11) < static_cast<unsigned>(kSwaWindow));
            s[ni][2] = vis10 ? __fmaf_rn(s[ni][2], qk_scale, __fmul_rn(__bfloat162float(srel1[rev0]), kLog2E)) : -INFINITY;
            s[ni][3] = vis11 ? __fmaf_rn(s[ni][3], qk_scale, __fmul_rn(__bfloat162float(srel1[rev1]), kLog2E)) : -INFINITY;

            row_max0 = fmaxf(row_max0, fmaxf(s[ni][0], s[ni][1]));
            row_max1 = fmaxf(row_max1, fmaxf(s[ni][2], s[ni][3]));
          }
        }

        // 3. Exact online softmax -> BF16 P registers pa[2][4]
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffffu, row_max0, 1));
        row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffffu, row_max0, 2));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffffu, row_max1, 1));
        row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffffu, row_max1, 2));

        const float m_safe0 = (row_max0 == -INFINITY) ? 0.0f : row_max0;
        const float m_safe1 = (row_max1 == -INFINITY) ? 0.0f : row_max1;
        alpha0 = ex2_approx(m_i0 - m_safe0);
        alpha1 = ex2_approx(m_i1 - m_safe1);
        m_i0 = row_max0;
        m_i1 = row_max1;

        float psum0 = 0.0f;
        float psum1 = 0.0f;
#pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
          const float p00 = ex2_approx(s[kp * 2][0] - m_safe0);
          const float p01 = ex2_approx(s[kp * 2][1] - m_safe0);
          const float p02 = ex2_approx(s[kp * 2][2] - m_safe1);
          const float p03 = ex2_approx(s[kp * 2][3] - m_safe1);
          const float p10 = ex2_approx(s[kp * 2 + 1][0] - m_safe0);
          const float p11 = ex2_approx(s[kp * 2 + 1][1] - m_safe0);
          const float p12 = ex2_approx(s[kp * 2 + 1][2] - m_safe1);
          const float p13 = ex2_approx(s[kp * 2 + 1][3] - m_safe1);
          psum0 += (p00 + p01) + (p10 + p11);
          psum1 += (p02 + p03) + (p12 + p13);
          pa[kp][0] = pack_f32x2_to_bf16x2(p00, p01);
          pa[kp][1] = pack_f32x2_to_bf16x2(p02, p03);
          pa[kp][2] = pack_f32x2_to_bf16x2(p10, p11);
          pa[kp][3] = pack_f32x2_to_bf16x2(p12, p13);
        }
        psum0 += __shfl_xor_sync(0xffffffffu, psum0, 1);
        psum0 += __shfl_xor_sync(0xffffffffu, psum0, 2);
        psum1 += __shfl_xor_sync(0xffffffffu, psum1, 1);
        psum1 += __shfl_xor_sync(0xffffffffu, psum1, 2);
        l_i0 = __fmaf_rn(l_i0, alpha0, psum0);
        l_i1 = __fmaf_rn(l_i1, alpha1, psum1);
      }

      // For single-buffered Rel (m64, 2 CTAs/SM), prefetch Rel_{n+1} during half==1 P @ V!
      if constexpr (REL_STAGES == 1) {
        if (half == 1 && (n + 1 < n_hi)) {
          __syncthreads();
          load_swa_rel_stage<THREADS, BLOCK_M>(
              reinterpret_cast<bf16*>(smem + 32768),
              Rel, block_m0, head, past, (n + 1) * kBlockN, num_tokens, tid);
          cp_async_commit();
        }
      }

      if (sub_active) {
        const bool need_rescale = __any_sync(0xffffffffu, (alpha0 != 1.0f) || (alpha1 != 1.0f));
        if (need_rescale) {
#pragma unroll
          for (int di = 0; di < 16; ++di) {
            acc[di][0] = __fmul_rn(acc[di][0], alpha0);
            acc[di][1] = __fmul_rn(acc[di][1], alpha0);
            acc[di][2] = __fmul_rn(acc[di][2], alpha1);
            acc[di][3] = __fmul_rn(acc[di][3], alpha1);
          }
        }

        // 4. P @ V via hardware ldmatrix.m16n16.x2.trans.b8 + 1-instr cvt.rn.bf16x2.e4m3x2
        const int v_base_row = half * 32 + lane;
#pragma unroll
        for (int dp = 0; dp < 8; ++dp) {
          const int swiz_d = swizzle_fp8_c16(lane, dp);
          uint32_t vb0, vb1, vb2, vb3;
          ldmatrix_x2_trans_b8(vb0, vb1, vb2, vb3, &v_swiz[v_base_row * 8 + swiz_d]);
          mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3],
                            pa[0][0], pa[0][1], pa[0][2], pa[0][3],
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0)),
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0 >> 16)));
          mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3],
                            pa[0][0], pa[0][1], pa[0][2], pa[0][3],
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1)),
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1 >> 16)));
          mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3],
                            pa[1][0], pa[1][1], pa[1][2], pa[1][3],
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2)),
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2 >> 16)));
          mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3],
                            pa[1][0], pa[1][1], pa[1][2], pa[1][3],
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3)),
                            cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3 >> 16)));
        }
      }
    }

    if (n + 1 < n_hi) {
      __syncthreads();
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

// BLOCK_M=128, BLOCK_N=64, 8 warps (256 threads), 69,632 B dynamic SMEM (1 CTA/SM, 2-stage KV+Rel).
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
  run_swa_prefill_impl<8, 2>(
      Q, SeqState, KCur, VCur, Rel, Gate, Sink, Lines, SeqLens, Out,
      num_tokens, stride_gate, stride_sink, sm_scale, v_scale);
}

// BLOCK_M=64, BLOCK_N=64, 4 warps (128 threads), 41,984 B dynamic SMEM (2 CTAs/SM, 2-stage KV + 1-stage Rel).
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
  run_swa_prefill_impl<4, 1>(
      Q, SeqState, KCur, VCur, Rel, Gate, Sink, Lines, SeqLens, Out,
      num_tokens, stride_gate, stride_sink, sm_scale, v_scale);
}

namespace {

// Cooperatively issue 16B cp.async.cg for one paged [64, 128] FP8 K+V page into swizzled SMEM.
template <int THREADS>
__device__ __forceinline__ void load_paged_kv_swizzled_bf16(
    uint4* __restrict__ smem_k_swiz,
    uint4* __restrict__ smem_v_swiz,
    const uint8_t* __restrict__ kv_k_base,
    const uint8_t* __restrict__ kv_v_base,
    int tid) {
  constexpr int kTotalChunks = kPage * (kHeadDim / 16);  // 512
#pragma unroll
  for (int idx = tid; idx < kTotalChunks; idx += THREADS) {
    const int row = idx >> 3;        // 0..63
    const int c16 = idx & 7;         // 0..7
    const int g_off = (row << 7) + (c16 << 4);
    const int k_swiz = swizzle_fp8_c16(row, c16);
    const int v_row = permute_v_row_16(row);
    const int v_swiz = swizzle_fp8_c16(v_row, c16);
    cp_async_cg_16(&smem_k_swiz[row * 8 + k_swiz], kv_k_base + g_off, 16);
    cp_async_cg_16(&smem_v_swiz[v_row * 8 + v_swiz], kv_v_base + g_off, 16);
  }
}

template <int NUM_WARPS>
__device__ __forceinline__ void run_full_prefill_bulk_bf16_impl(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KVState,
    const int* __restrict__ BlockTable,
    const int* __restrict__ SeqLens,
    float* __restrict__ OBulk,
    float* __restrict__ LseBulk,
    int num_tokens,
    int table_width,
    float sm_scale) {
  (void)table_width;
  constexpr int THREADS = NUM_WARPS * 32;
  constexpr int BLOCK_M = NUM_WARPS * 16;
  constexpr int STAGES = 3;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7
  const int tig = lane & 3;    // 0..3

  const int block_m0 = blockIdx.x * BLOCK_M;
  if (block_m0 >= num_tokens) return;
  const int head = blockIdx.y;
  const int kv_head = head / kGroup;

  const int seqlen_k = SeqLens[0];
  const int past = seqlen_k - num_tokens;
  const int aligned_tok = (block_m0 / kBulkAlign) * kBulkAlign;
  const int aligned_pos_lo = past + aligned_tok;
  const int n_bulk_pages = min(max(aligned_pos_lo - kRelExtent + 1, 0) / kPage, seqlen_k / kPage);
  if (n_bulk_pages <= 0) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    return;
  }

  const int m0 = block_m0 + warp * 16;
  const int r0 = m0 + gid;
  const int r1 = m0 + gid + 8;
  const bool valid0 = (r0 < num_tokens);
  const bool valid1 = (r1 < num_tokens);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Load Q [16, 128] BF16 into registers matching the in-register FP8->BF16 K layout:
  // within each 16-elem K-step, thread (gid, tig) holds {d+0, d+1} in a0/a1 and {d+2, d+3} in a2/a3
  // at d = ks*16 + tig*4 (contiguous 8B uint2 load, zero shuffles!).
  uint32_t q_reg[8][4];
  const bf16* q_row0 = valid0 ? (Q + static_cast<int64_t>(r0) * (kHeads * kHeadDim) + head * kHeadDim) : nullptr;
  const bf16* q_row1 = valid1 ? (Q + static_cast<int64_t>(r1) * (kHeads * kHeadDim) + head * kHeadDim) : nullptr;
#pragma unroll
  for (int ks = 0; ks < 8; ++ks) {
    const int d_base = ks * 16 + tig * 4;
    uint2 qv0 = valid0 ? *reinterpret_cast<const uint2*>(q_row0 + d_base) : make_uint2(0u, 0u);
    uint2 qv1 = valid1 ? *reinterpret_cast<const uint2*>(q_row1 + d_base) : make_uint2(0u, 0u);
    q_reg[ks][0] = qv0.x;
    q_reg[ks][1] = qv1.x;
    q_reg[ks][2] = qv0.y;
    q_reg[ks][3] = qv1.y;
  }

  // Dynamic shared memory: 3-stage swizzled FP8 K & permuted-swizzled FP8 V (3 * 16 KB = 48 KB).
  // Zero SMEM dequantization!
  extern __shared__ __align__(16) uint8_t smem[];
  const uint8_t* kv_head_ptr = KVState + static_cast<int64_t>(kv_head) * kStrideKvHead;

  // Prime STAGES - 1 = 2 stages.
  const int prime = min(STAGES - 1, n_bulk_pages);
  for (int p = 0; p < prime; ++p) {
    const int page = max(BlockTable[p], 0);
    const uint8_t* k_base = kv_head_ptr + static_cast<int64_t>(page) * kStrideKvBlk;
    const uint8_t* v_base = k_base + kStrideKvHalf;
    load_paged_kv_swizzled_bf16<THREADS>(
        reinterpret_cast<uint4*>(smem + p * 16384),
        reinterpret_cast<uint4*>(smem + p * 16384 + 8192),
        k_base, v_base, tid);
    cp_async_commit();
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
  const int k_ld_row_off = ((lane >> 4) << 3) + (lane & 7);
  const int k_ld_c16_off = (lane >> 3) & 1;

  int stage = 0;
  int next_issue = prime;
  for (int n = 0; n < n_bulk_pages; ++n) {
    if (next_issue < n_bulk_pages) {
      const int issue_stage = (stage + STAGES - 1) % STAGES;
      const int page = max(BlockTable[next_issue], 0);
      const uint8_t* k_base = kv_head_ptr + static_cast<int64_t>(page) * kStrideKvBlk;
      const uint8_t* v_base = k_base + kStrideKvHalf;
      load_paged_kv_swizzled_bf16<THREADS>(
          reinterpret_cast<uint4*>(smem + issue_stage * 16384),
          reinterpret_cast<uint4*>(smem + issue_stage * 16384 + 8192),
          k_base, v_base, tid);
      cp_async_commit();
      ++next_issue;
      cp_async_wait<STAGES - 1>();
    } else {
      const int rem = n_bulk_pages - 1 - n;
      if (rem == 0) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
    }
    __syncthreads();

    const uint4* k_swiz = reinterpret_cast<const uint4*>(smem + stage * 16384);
    const uint4* v_swiz = reinterpret_cast<const uint4*>(smem + stage * 16384 + 8192);

    // 1. Q @ K^T -> s[8][4] (16 x 64) using ldmatrix_x4 on FP8 K + 1-instr cvt.rn.bf16x2.e4m3x2
    float s[8][4];
    float row_max0 = m_i0;
    float row_max1 = m_i1;
#pragma unroll
    for (int np = 0; np < 4; ++np) {
      float s0_0 = 0.0f, s0_1 = 0.0f, s0_2 = 0.0f, s0_3 = 0.0f;
      float s1_0 = 0.0f, s1_1 = 0.0f, s1_2 = 0.0f, s1_3 = 0.0f;
      const int r = np * 16 + k_ld_row_off;
#pragma unroll
      for (int ks2 = 0; ks2 < 4; ++ks2) {
        const int c16_base = ks2 * 2 + k_ld_c16_off;
        uint32_t b0, b1, b2, b3;
        ldmatrix_x4(b0, b1, b2, b3, &k_swiz[r * 8 + swizzle_fp8_c16(r, c16_base)]);

        const int ks_lo = ks2 * 2;
        const int ks_hi = ks_lo + 1;

        uint32_t k0_lo = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0));
        uint32_t k0_hi = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b0 >> 16));
        mma_m16n8k16_bf16(s0_0, s0_1, s0_2, s0_3,
                          q_reg[ks_lo][0], q_reg[ks_lo][1], q_reg[ks_lo][2], q_reg[ks_lo][3],
                          k0_lo, k0_hi);

        uint32_t k2_lo = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2));
        uint32_t k2_hi = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b2 >> 16));
        mma_m16n8k16_bf16(s1_0, s1_1, s1_2, s1_3,
                          q_reg[ks_lo][0], q_reg[ks_lo][1], q_reg[ks_lo][2], q_reg[ks_lo][3],
                          k2_lo, k2_hi);

        uint32_t k1_lo = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1));
        uint32_t k1_hi = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b1 >> 16));
        mma_m16n8k16_bf16(s0_0, s0_1, s0_2, s0_3,
                          q_reg[ks_hi][0], q_reg[ks_hi][1], q_reg[ks_hi][2], q_reg[ks_hi][3],
                          k1_lo, k1_hi);

        uint32_t k3_lo = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3));
        uint32_t k3_hi = cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(b3 >> 16));
        mma_m16n8k16_bf16(s1_0, s1_1, s1_2, s1_3,
                          q_reg[ks_hi][0], q_reg[ks_hi][1], q_reg[ks_hi][2], q_reg[ks_hi][3],
                          k3_lo, k3_hi);
      }
      s[np * 2][0] = s0_0 * qk_scale;
      s[np * 2][1] = s0_1 * qk_scale;
      s[np * 2][2] = s0_2 * qk_scale;
      s[np * 2][3] = s0_3 * qk_scale;
      s[np * 2 + 1][0] = s1_0 * qk_scale;
      s[np * 2 + 1][1] = s1_1 * qk_scale;
      s[np * 2 + 1][2] = s1_2 * qk_scale;
      s[np * 2 + 1][3] = s1_3 * qk_scale;
      row_max0 = fmaxf(row_max0, fmaxf(fmaxf(s[np * 2][0], s[np * 2][1]), fmaxf(s[np * 2 + 1][0], s[np * 2 + 1][1])));
      row_max1 = fmaxf(row_max1, fmaxf(fmaxf(s[np * 2][2], s[np * 2][3]), fmaxf(s[np * 2 + 1][2], s[np * 2 + 1][3])));
    }

    // 2. Exact online softmax -> BF16 P registers pa[4][4] (ZERO shuffles!)
    row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffffu, row_max0, 1));
    row_max0 = fmaxf(row_max0, __shfl_xor_sync(0xffffffffu, row_max0, 2));
    row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffffu, row_max1, 1));
    row_max1 = fmaxf(row_max1, __shfl_xor_sync(0xffffffffu, row_max1, 2));

    const float alpha0 = ex2_approx(m_i0 - row_max0);
    const float alpha1 = ex2_approx(m_i1 - row_max1);
    m_i0 = row_max0;
    m_i1 = row_max1;

    uint32_t pa[4][4];
    float psum0 = 0.0f;
    float psum1 = 0.0f;
#pragma unroll
    for (int kp = 0; kp < 4; ++kp) {
      const float p00 = ex2_approx(s[kp * 2][0] - row_max0);
      const float p01 = ex2_approx(s[kp * 2][1] - row_max0);
      const float p02 = ex2_approx(s[kp * 2][2] - row_max1);
      const float p03 = ex2_approx(s[kp * 2][3] - row_max1);
      const float p10 = ex2_approx(s[kp * 2 + 1][0] - row_max0);
      const float p11 = ex2_approx(s[kp * 2 + 1][1] - row_max0);
      const float p12 = ex2_approx(s[kp * 2 + 1][2] - row_max1);
      const float p13 = ex2_approx(s[kp * 2 + 1][3] - row_max1);
      psum0 += (p00 + p01) + (p10 + p11);
      psum1 += (p02 + p03) + (p12 + p13);
      pa[kp][0] = pack_f32x2_to_bf16x2(p00, p01);
      pa[kp][1] = pack_f32x2_to_bf16x2(p02, p03);
      pa[kp][2] = pack_f32x2_to_bf16x2(p10, p11);
      pa[kp][3] = pack_f32x2_to_bf16x2(p12, p13);
    }
    psum0 += __shfl_xor_sync(0xffffffffu, psum0, 1);
    psum0 += __shfl_xor_sync(0xffffffffu, psum0, 2);
    psum1 += __shfl_xor_sync(0xffffffffu, psum1, 1);
    psum1 += __shfl_xor_sync(0xffffffffu, psum1, 2);
    l_i0 = __fmaf_rn(l_i0, alpha0, psum0);
    l_i1 = __fmaf_rn(l_i1, alpha1, psum1);

    // 3. P @ V using hardware ldmatrix.m16n16.x2.trans.b8 + 1-instr cvt.rn.bf16x2.e4m3x2 + exact BF16 MMA
#pragma unroll
    for (int dp = 0; dp < 8; ++dp) {
      acc[dp * 2][0] = __fmul_rn(acc[dp * 2][0], alpha0);
      acc[dp * 2][1] = __fmul_rn(acc[dp * 2][1], alpha0);
      acc[dp * 2][2] = __fmul_rn(acc[dp * 2][2], alpha1);
      acc[dp * 2][3] = __fmul_rn(acc[dp * 2][3], alpha1);
      acc[dp * 2 + 1][0] = __fmul_rn(acc[dp * 2 + 1][0], alpha0);
      acc[dp * 2 + 1][1] = __fmul_rn(acc[dp * 2 + 1][1], alpha0);
      acc[dp * 2 + 1][2] = __fmul_rn(acc[dp * 2 + 1][2], alpha1);
      acc[dp * 2 + 1][3] = __fmul_rn(acc[dp * 2 + 1][3], alpha1);

      const int swiz_d = swizzle_fp8_c16(lane, dp);
      uint32_t vb0, vb1, vb2, vb3;

      // Keys 0..31 (pa[0] = keys 0..15, pa[1] = keys 16..31)
      ldmatrix_x2_trans_b8(vb0, vb1, vb2, vb3, &v_swiz[lane * 8 + swiz_d]);
      mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3],
                        pa[0][0], pa[0][1], pa[0][2], pa[0][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0 >> 16)));
      mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3],
                        pa[0][0], pa[0][1], pa[0][2], pa[0][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1 >> 16)));
      mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3],
                        pa[1][0], pa[1][1], pa[1][2], pa[1][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2 >> 16)));
      mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3],
                        pa[1][0], pa[1][1], pa[1][2], pa[1][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3 >> 16)));

      // Keys 32..63 (pa[2] = keys 32..47, pa[3] = keys 48..63)
      ldmatrix_x2_trans_b8(vb0, vb1, vb2, vb3, &v_swiz[(32 + lane) * 8 + swiz_d]);
      mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3],
                        pa[2][0], pa[2][1], pa[2][2], pa[2][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb0 >> 16)));
      mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3],
                        pa[2][0], pa[2][1], pa[2][2], pa[2][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb1 >> 16)));
      mma_m16n8k16_bf16(acc[dp * 2][0], acc[dp * 2][1], acc[dp * 2][2], acc[dp * 2][3],
                        pa[3][0], pa[3][1], pa[3][2], pa[3][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb2 >> 16)));
      mma_m16n8k16_bf16(acc[dp * 2 + 1][0], acc[dp * 2 + 1][1], acc[dp * 2 + 1][2], acc[dp * 2 + 1][3],
                        pa[3][0], pa[3][1], pa[3][2], pa[3][3],
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3)),
                        cvt_e4m3x2_to_bf16x2(static_cast<uint16_t>(vb3 >> 16)));
    }

    __syncthreads();
    stage = (stage + 1 == STAGES) ? 0 : (stage + 1);
  }

  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const bool empty0 = (l_i0 == 0.0f);
  const bool empty1 = (l_i1 == 0.0f);
  const float inv_l0 = empty0 ? 0.0f : __fdividef(1.0f, l_i0);
  const float inv_l1 = empty1 ? 0.0f : __fdividef(1.0f, l_i1);
  const float lse0 = empty0 ? -INFINITY : (m_i0 + log2f(l_i0));
  const float lse1 = empty1 ? -INFINITY : (m_i1 + log2f(l_i1));

  if (valid0) {
    if (tig == 0) {
      LseBulk[static_cast<int64_t>(r0) * kHeads + head] = lse0;
    }
    float* dst0 = OBulk + (static_cast<int64_t>(r0) * kHeads + head) * kHeadDim;
#pragma unroll
    for (int di = 0; di < 16; ++di) {
      const int d = di * 8 + tig * 2;
      *reinterpret_cast<float2*>(dst0 + d) = make_float2(acc[di][0] * inv_l0, acc[di][1] * inv_l0);
    }
  }
  if (valid1) {
    if (tig == 0) {
      LseBulk[static_cast<int64_t>(r1) * kHeads + head] = lse1;
    }
    float* dst1 = OBulk + (static_cast<int64_t>(r1) * kHeads + head) * kHeadDim;
#pragma unroll
    for (int di = 0; di < 16; ++di) {
      const int d = di * 8 + tig * 2;
      *reinterpret_cast<float2*>(dst1 + d) = make_float2(acc[di][2] * inv_l1, acc[di][3] * inv_l1);
    }
  }
}

}  // namespace

// BLOCK_M=128, BLOCK_N=64=PAGE, 8 warps (256 threads), 48 KB SMEM (3-stage FP8 pipeline, exact BF16 MMA).
extern "C" __global__ void __launch_bounds__(256, 1) kern_rel_full_prefill_bulk_bf16_m128_sm120(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KVState,
    const int* __restrict__ BlockTable,
    const int* __restrict__ SeqLens,
    float* __restrict__ OBulk,
    float* __restrict__ LseBulk,
    int num_tokens,
    int table_width,
    float sm_scale) {
  run_full_prefill_bulk_bf16_impl<8>(
      Q, KVState, BlockTable, SeqLens, OBulk, LseBulk, num_tokens, table_width, sm_scale);
}

// BLOCK_M=64, BLOCK_N=64=PAGE, 4 warps (128 threads), 48 KB SMEM (2 CTAs/SM, 3-stage FP8 pipeline, exact BF16 MMA).
extern "C" __global__ void __launch_bounds__(128, 2) kern_rel_full_prefill_bulk_bf16_m64_sm120(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ KVState,
    const int* __restrict__ BlockTable,
    const int* __restrict__ SeqLens,
    float* __restrict__ OBulk,
    float* __restrict__ LseBulk,
    int num_tokens,
    int table_width,
    float sm_scale) {
  run_full_prefill_bulk_bf16_impl<4>(
      Q, KVState, BlockTable, SeqLens, OBulk, LseBulk, num_tokens, table_width, sm_scale);
}
