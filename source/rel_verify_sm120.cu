// SPDX-License-Identifier: Apache-2.0
// SM120 (Blackwell) Target Verify Attention Kernels:
//   1. kern_attn_swa_verify_split: Split-KV sliding-window (W=1024) FP8 E4M3 attention
//      over states.seq_state ring + KCur/VCur for speculative verify (QLEN=8, H=24, HKV=8, D=128).
//      - Transposed MMA geometry: K @ Q^T [64x24] and V^T @ P^T [128x24] (100% Tensor Core util)
//      - Q resident in registers (zero SMEM Q buffer, zero Q LDS in hot loop)
//      - 3-buffer KV SMEM (k[2]+v[1]) + separate rel & p buffers (~32.7 KB -> 3 CTAs/SM on SM120!)
//      - Prefetches K_{it+1} + Rel_{it+1} together during V^T @ P^T (eliminates Rel wait & sync)
//      - Blackwell native 1-instr cvt.rn.bf16x2.e4m3x2 (F2FP.BF16.E4M3)
//      - Optional dedicated cur-only split (ring_splits < num_splits) to eliminate last-split straggler
//      - PDL (griddepcontrol) across prologue & combine.
//   2. kern_attn_verify_combine_s{4,5,8,9,16,17,32}: 1-warp float4 split-KV combine + Gate + Sink.

#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>

using bf16 = __nv_bfloat16;

namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kGroup = 3;
constexpr int kHeadDim = 128;
constexpr int kRelExtent = 1024;
constexpr int kRing = 1024;
constexpr int kWindow = 1024;
constexpr int kQlen = 8;
constexpr int kM = kQlen * kGroup;  // 24 query rows per (seq, kv_head)
constexpr int kBN = 64;             // keys per tile
constexpr int kThreads = 128;       // 4 warps
constexpr int kPRow = 72;           // 72 bf16 = 144 B = 36 banks (gid*4 + tig -> 0 bank conflicts!)

constexpr int64_t kSeqStateBytes = 90005376LL;
constexpr int64_t kSwaHeadRingBytes = 131072LL;   // 1024 * 128
constexpr int64_t kSwaHalfVisitBytes = 1048576LL; // 8 * 1024 * 128

constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;
constexpr float kNegInf = -INFINITY;

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_cg_16(void* dst_smem, const void* src_gmem) {
  uint32_t dst = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src_gmem) : "memory");
}

__device__ __forceinline__ void cp_async_cg_16_pred(void* dst_smem, const void* src_gmem, int src_size) {
  uint32_t dst = smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src_gmem), "r"(src_size) : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t cvt_e4m3x2_to_bf16x2(uint16_t x) {
  uint32_t res;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(res) : "h"(x));
  return res;
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&c)[4],
    const uint32_t (&a)[4],
    uint32_t b0,
    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ float ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float lg2_approx(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + ex2_approx(-x * kLog2E));
}

__device__ __forceinline__ int swiz_q_col(int row, int col_bf16) {
  return col_bf16 ^ ((row & 7) << 3);
}

// Load 64x128 FP8 ring tile (K or V) from SeqState into dst_smem[64][128] with 16B XOR swizzle.
__device__ __forceinline__ void cp_async_ring_tile(
    uint8_t (&dst)[kBN][kHeadDim],
    const uint8_t* __restrict__ ring_base,
    int k0,
    int past,
    int tid,
    bool full_valid) {
  const int c16 = tid & 7;
  const int r0 = tid >> 3;
  const int col_byte = c16 << 4;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int r = r0 + step * 16;
    const int k = k0 + r;
    const int swiz_col = col_byte ^ ((r & 7) << 4);
    const int slot = k & (kRing - 1);
    const uint8_t* src = ring_base + static_cast<int64_t>(slot) * kHeadDim + col_byte;
    if (full_valid) {
      cp_async_cg_16(&dst[r][swiz_col], src);
    } else {
      const int sz = (k >= 0 && k < past) ? 16 : 0;
      cp_async_cg_16_pred(&dst[r][swiz_col], src, sz);
    }
  }
}

// Load Rel[q_start..q_start+7, kv_head*3..+2, d_align..d_align+71] into dst[24][72].
__device__ __forceinline__ void cp_async_rel_tile(
    bf16 (&dst)[kM][kPRow],
    const bf16* __restrict__ Rel,
    int q_start,
    int kv_head,
    int past,
    int k0,
    int tid,
    bool full_interior) {
#pragma unroll
  for (int idx = tid; idx < kM * 9; idx += kThreads) {
    const int m = idx / 9;
    const int chunk = idx - m * 9;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const int d_lo = (past + tok) - (k0 + kBN - 1);
    const int d_align = d_lo & ~7;
    const int d_c = d_align + (chunk << 3);
    const int64_t row_off = static_cast<int64_t>(q_start + tok) * (kHeads * kRelExtent)
                          + static_cast<int64_t>(head) * kRelExtent;
    if (full_interior) {
      if (chunk < 8 || (d_lo & 7) != 0) {
        cp_async_cg_16(&dst[m][chunk << 3], Rel + row_off + d_c);
      }
    } else {
      const bool ok = (static_cast<unsigned>(d_c) < static_cast<unsigned>(kRelExtent));
      const int safe_dc = ok ? d_c : 0;
      cp_async_cg_16_pred(&dst[m][chunk << 3], Rel + row_off + safe_dc, ok ? 16 : 0);
    }
  }
}

// Compute S^T = K[64x128] @ Q^T[128x24] for warp w (keys w*16 .. w*16+15) with Q in registers.
__device__ __forceinline__ void compute_kq_tile_qreg(
    float (&s)[3][4],
    const uint8_t (&smem_k)[kBN][kHeadDim],
    const uint32_t (&q_b0)[8][3],
    const uint32_t (&q_b1)[8][3],
    int w,
    int gid,
    int tig) {
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    s[mt][0] = 0.0f; s[mt][1] = 0.0f; s[mt][2] = 0.0f; s[mt][3] = 0.0f;
  }
  const int k_row0 = w * 16 + gid;
  const int k_row1 = k_row0 + 8;
  const int sw_k0 = (k_row0 & 7) << 4;
  const int sw_k1 = (k_row1 & 7) << 4;

#pragma unroll
  for (int ds = 0; ds < 8; ++ds) {
    const int d_step = ds << 4;
    const int k_col0 = (d_step ^ sw_k0) + tig * 2;
    const int k_col1 = (d_step ^ sw_k1) + tig * 2;
    uint16_t raw0 = *reinterpret_cast<const uint16_t*>(&smem_k[k_row0][k_col0]);
    uint16_t raw1 = *reinterpret_cast<const uint16_t*>(&smem_k[k_row1][k_col1]);
    uint16_t raw2 = *reinterpret_cast<const uint16_t*>(&smem_k[k_row0][k_col0 + 8]);
    uint16_t raw3 = *reinterpret_cast<const uint16_t*>(&smem_k[k_row1][k_col1 + 8]);
    uint32_t a[4];
    a[0] = cvt_e4m3x2_to_bf16x2(raw0);
    a[1] = cvt_e4m3x2_to_bf16x2(raw1);
    a[2] = cvt_e4m3x2_to_bf16x2(raw2);
    a[3] = cvt_e4m3x2_to_bf16x2(raw3);

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      mma_m16n8k16_bf16(s[mt], a, q_b0[ds][mt], q_b1[ds][mt]);
    }
  }
}

// Accumulate O^T += V^T[128xK] @ P^T[Kx24] for warp w (dims w*32 .. w*32+31).
template <int K_STEPS>
__device__ __forceinline__ void compute_vp_tile(
    float (&acc)[2][3][4],
    const uint8_t (&smem_v)[kBN][kHeadDim],
    const bf16 (&smem_p)[kM][kPRow],
    int w,
    int gid,
    int tig) {
#pragma unroll
  for (int ks = 0; ks < K_STEPS; ++ks) {
    const int k_base = ks * 16;
    const int k0 = k_base + tig * 2;
    const int k1 = k0 + 1;
    const int k2 = k0 + 8;
    const int k3 = k2 + 1;
    const int sw0 = (k0 & 7) << 4;
    const int sw1 = (k1 & 7) << 4;
    const int sw2 = (k2 & 7) << 4;
    const int sw3 = (k3 & 7) << 4;

    uint32_t b0[3], b1[3];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m = mt * 8 + gid;
      b0[mt] = *reinterpret_cast<const uint32_t*>(&smem_p[m][k0]);
      b1[mt] = *reinterpret_cast<const uint32_t*>(&smem_p[m][k2]);
    }

#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      const int d_base = w * 32 + dt * 16;
      const int d_off0 = (d_base ^ sw0) + gid;
      const int d_off1 = (d_base ^ sw1) + gid;
      const int d_off2 = (d_base ^ sw2) + gid;
      const int d_off3 = (d_base ^ sw3) + gid;

      uint16_t v_lo0 = static_cast<uint16_t>(smem_v[k0][d_off0])
                     | (static_cast<uint16_t>(smem_v[k1][d_off1]) << 8);
      uint16_t v_hi0 = static_cast<uint16_t>(smem_v[k0][d_off0 + 8])
                     | (static_cast<uint16_t>(smem_v[k1][d_off1 + 8]) << 8);
      uint16_t v_lo1 = static_cast<uint16_t>(smem_v[k2][d_off2])
                     | (static_cast<uint16_t>(smem_v[k3][d_off3]) << 8);
      uint16_t v_hi1 = static_cast<uint16_t>(smem_v[k2][d_off2 + 8])
                     | (static_cast<uint16_t>(smem_v[k3][d_off3 + 8]) << 8);

      uint32_t a[4];
      a[0] = cvt_e4m3x2_to_bf16x2(v_lo0);
      a[1] = cvt_e4m3x2_to_bf16x2(v_hi0);
      a[2] = cvt_e4m3x2_to_bf16x2(v_lo1);
      a[3] = cvt_e4m3x2_to_bf16x2(v_hi1);

#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        mma_m16n8k16_bf16(acc[dt][mt], a, b0[mt], b1[mt]);
      }
    }
  }
}

__device__ __forceinline__ float col_reduce_max8(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  return v;
}

__device__ __forceinline__ float col_reduce_sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  return v;
}

// Total SMEM = 32,256 B (+1 KB runtime = 33,280 B -> 3 CTAs/SM = 99,840 B <= 101,376 B on SM120!)
struct SwaVerifySmem {
  alignas(16) uint8_t k[2][kBN][kHeadDim];  // 16,384 B (k[1] stages Q at prologue; k[0..1] stages O at epilogue)
  alignas(16) uint8_t v[kBN][kHeadDim];     //  8,192 B
  alignas(16) bf16 rel[kM][kPRow];          //  3,456 B (separate from p so Rel_{it+1} prefetches with K_{it+1}!)
  alignas(16) bf16 p[kM][kPRow];            //  3,456 B
  alignas(16) float m[4][kM];               //    384 B (4-warp rowmax scratch — separate from l to avoid WAR race!)
  alignas(16) float l[4][kM];               //    384 B (4-warp rowsum scratch)
};

}  // namespace

// ---------------------------------------------------------------------------
// 1. SWA Verify Split-KV Attention Kernel (SM120, 3 CTAs/SM)
// Grid: [num_splits, 8 (kv_heads), seqs], Block: [128, 1, 1]
// If ring_splits < num_splits (e.g. ring_splits=8, num_splits=9), split==ring_splits
// is dedicated exclusively to the 8-key KCur/VCur micro-tile (eliminates straggler!).
// If ring_splits == num_splits, split==num_splits-1 handles both its ring tiles and cur.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads, 3) kern_attn_swa_verify_split(
    const bf16* __restrict__ Q,
    const uint8_t* __restrict__ SeqState,
    const uint8_t* __restrict__ KCur,
    const uint8_t* __restrict__ VCur,
    const bf16* __restrict__ Rel,
    const int32_t* __restrict__ Lines,
    const int32_t* __restrict__ SeqLens,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    int num_tokens,
    float sm_scale,
    float v_scale,
    int num_splits,
    int ring_splits) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int req = blockIdx.z;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int q_start = req * kQlen;
  const int line = Lines[req];
  const int seqlen_k = SeqLens[req];

  if (line <= 0 || seqlen_k <= 0) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      LsePart[part] = kNegInf;
    }
    return;
  }

  const int past = seqlen_k - kQlen;
  const int k_min = (past - (kWindow - 1) > 0) ? (past - (kWindow - 1)) : 0;
  const int n_ring = (past > k_min) ? (past - k_min) : 0;
  const int n_ring_tiles = (n_ring + kBN - 1) / kBN;

  int tile_start = 0;
  int tile_count = 0;
  bool do_cur = false;
  if (split < ring_splits) {
    const int base_tiles = n_ring_tiles / ring_splits;
    const int rem_tiles = n_ring_tiles % ring_splits;
    tile_start = split * base_tiles + (split < rem_tiles ? split : rem_tiles);
    tile_count = base_tiles + (split < rem_tiles ? 1 : 0);
    do_cur = (ring_splits == num_splits) && (split == num_splits - 1);
  } else {
    do_cur = (split == num_splits - 1);
  }

  if (tile_count == 0 && !do_cur) {
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
      LsePart[part] = kNegInf;
    }
    return;
  }

  __shared__ SwaVerifySmem smem;

  const uint8_t* ring_k_base = SeqState
      + static_cast<int64_t>(line) * kSeqStateBytes
      + static_cast<int64_t>(kv_head) * kSwaHeadRingBytes;
  const uint8_t* ring_v_base = ring_k_base + kSwaHalfVisitBytes;

  // Prefetch first ring K tile into smem.k[0] BEFORE griddepcontrol.wait (SeqState untouched by prologue!).
  int k0_first = 0;
  bool full_first = false;
  if (tile_count > 0) {
    k0_first = k_min + tile_start * kBN;
    full_first = (k0_first + kBN <= past) && (past + 7 - k0_first < kWindow);
    cp_async_ring_tile(smem.k[0], ring_k_base, k0_first, past, tid, full_first);
    cp_async_commit();
  }

  // Wait for attn_prologue_swa_verify to finish writing Q, Rel, KCur, VCur.
  asm volatile("griddepcontrol.wait;" ::: "memory");

  // Immediately kick off Rel_0 cp.async (it==0) in parallel with Q load!
  if (tile_count > 0) {
    cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, k0_first, tid, full_first);
    cp_async_commit();
  }

  // Stage Q[24][128] bf16 (6 KB) into smem.k[1] (free until it==1!), then load into registers.
  bf16 (*smem_q)[kHeadDim] = reinterpret_cast<bf16 (*)[kHeadDim]>(&smem.k[1][0][0]);
#pragma unroll
  for (int step = 0; step < 3; ++step) {
    const int idx = tid + step * kThreads;
    const int m = idx >> 4;
    const int c16 = idx & 15;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const int col_bf16 = c16 << 3;
    const int sw_col = swiz_q_col(m, col_bf16);
    const bf16* q_src = Q + (static_cast<int64_t>(q_start + tok) * kHeads + head) * kHeadDim + col_bf16;
    *reinterpret_cast<uint4*>(&smem_q[m][sw_col]) = *reinterpret_cast<const uint4*>(q_src);
  }
  __syncthreads();

  uint32_t q_b0[8][3];
  uint32_t q_b1[8][3];
#pragma unroll
  for (int ds = 0; ds < 8; ++ds) {
    const int d_step = ds << 4;
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int q_m = mt * 8 + gid;
      const int q_c0 = swiz_q_col(q_m, d_step) + tig * 2;
      const int q_c1 = swiz_q_col(q_m, d_step + 8) + tig * 2;
      q_b0[ds][mt] = *reinterpret_cast<const uint32_t*>(&smem_q[q_m][q_c0]);
      q_b1[ds][mt] = *reinterpret_cast<const uint32_t*>(&smem_q[q_m][q_c1]);
    }
  }

  const float qk_scale = sm_scale * kLog2E;

  float m_i[3][2];
  float l_i[3][2];
  float acc[2][3][4];
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    m_i[mt][0] = kNegInf; m_i[mt][1] = kNegInf;
    l_i[mt][0] = 0.0f;    l_i[mt][1] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      acc[dt][mt][0] = 0.0f; acc[dt][mt][1] = 0.0f;
      acc[dt][mt][2] = 0.0f; acc[dt][mt][3] = 0.0f;
    }
  }

  int owned_tok[3][2];
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    const int m0 = mt * 8 + tig * 2;
    owned_tok[mt][0] = m0 / kGroup;
    owned_tok[mt][1] = (m0 + 1) / kGroup;
  }

  // ---------------- Main Ring Loop (BN=64, 3-sync overlapped K+Rel / V pipeline) ----------------
  for (int it = 0; it < tile_count; ++it) {
    const int cur = it & 1;
    const int nxt = cur ^ 1;
    const int k0 = k_min + (tile_start + it) * kBN;
    const bool is_full_interior = (k0 + kBN <= past) && (past + 7 - k0 < kWindow);

    // Issue V_it into smem.v while K_it + Rel_it finish and K_it @ Q^T runs!
    cp_async_ring_tile(smem.v, ring_v_base, k0, past, tid, is_full_interior);
    cp_async_commit();

    // Wait for BOTH K_it and Rel_it (V_it stays in flight!).
    cp_async_wait<1>();
    __syncthreads();

    // Phase 1: S^T = K_it @ Q^T (64 x 24), Q directly from registers!
    float s[3][4];
    compute_kq_tile_qreg(s, smem.k[cur], q_b0, q_b1, w, gid, tig);

    // Rel_it is ALREADY ready in smem.rel — zero wait, zero sync!
    const int k_local0 = w * 16 + gid;
    const int k_local1 = k_local0 + 8;
    const int k_abs0 = k0 + k_local0;
    const int k_abs1 = k0 + k_local1;
    const int rev0 = (kBN - 1) - k_local0;
    const int rev1 = (kBN - 1) - k_local1;

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const int tok0 = owned_tok[mt][0];
      const int tok1 = owned_tok[mt][1];
      const int d_lo0 = (past + tok0) - (k0 + kBN - 1);
      const int d_lo1 = (past + tok1) - (k0 + kBN - 1);
      const int sh0 = d_lo0 & 7;
      const int sh1 = d_lo1 & 7;

      if (is_full_interior) {
        const float r00 = __bfloat162float(smem.rel[m0][rev0 + sh0]);
        const float r01 = __bfloat162float(smem.rel[m1][rev0 + sh1]);
        const float r10 = __bfloat162float(smem.rel[m0][rev1 + sh0]);
        const float r11 = __bfloat162float(smem.rel[m1][rev1 + sh1]);
        s[mt][0] = __fmaf_rn(s[mt][0], qk_scale, r00 * kLog2E);
        s[mt][1] = __fmaf_rn(s[mt][1], qk_scale, r01 * kLog2E);
        s[mt][2] = __fmaf_rn(s[mt][2], qk_scale, r10 * kLog2E);
        s[mt][3] = __fmaf_rn(s[mt][3], qk_scale, r11 * kLog2E);
      } else {
        const int dist00 = (past + tok0) - k_abs0;
        const int dist01 = (past + tok1) - k_abs0;
        const int dist10 = (past + tok0) - k_abs1;
        const int dist11 = (past + tok1) - k_abs1;
        const bool v00 = (k_abs0 >= 0) && (k_abs0 < past) && (dist00 >= 0) && (dist00 < kWindow);
        const bool v01 = (k_abs0 >= 0) && (k_abs0 < past) && (dist01 >= 0) && (dist01 < kWindow);
        const bool v10 = (k_abs1 >= 0) && (k_abs1 < past) && (dist10 >= 0) && (dist10 < kWindow);
        const bool v11 = (k_abs1 >= 0) && (k_abs1 < past) && (dist11 >= 0) && (dist11 < kWindow);
        s[mt][0] = v00 ? __fmaf_rn(s[mt][0], qk_scale, __bfloat162float(smem.rel[m0][rev0 + sh0]) * kLog2E) : kNegInf;
        s[mt][1] = v01 ? __fmaf_rn(s[mt][1], qk_scale, __bfloat162float(smem.rel[m1][rev0 + sh1]) * kLog2E) : kNegInf;
        s[mt][2] = v10 ? __fmaf_rn(s[mt][2], qk_scale, __bfloat162float(smem.rel[m0][rev1 + sh0]) * kLog2E) : kNegInf;
        s[mt][3] = v11 ? __fmaf_rn(s[mt][3], qk_scale, __bfloat162float(smem.rel[m1][rev1 + sh1]) * kLog2E) : kNegInf;
      }

      const float wm0 = col_reduce_max8(fmaxf(s[mt][0], s[mt][2]));
      const float wm1 = col_reduce_max8(fmaxf(s[mt][1], s[mt][3]));
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    // Cross-warp rowmax (from smem.m), compute P -> smem.p, warp rowsum -> smem.l (disjoint from smem.m!).
    // Note: smem.rel reads are ALL complete after the __syncthreads() above!
    float alpha[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = fmaxf(fmaxf(smem.m[0][m0], smem.m[1][m0]), fmaxf(smem.m[2][m0], smem.m[3][m0]));
      const float tm1 = fmaxf(fmaxf(smem.m[0][m1], smem.m[1][m1]), fmaxf(smem.m[2][m1], smem.m[3][m1]));
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;
      alpha[mt][0] = ex2_approx(m_i[mt][0] - ms0);
      alpha[mt][1] = ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;

      const float p00 = ex2_approx(s[mt][0] - ms0);
      const float p01 = ex2_approx(s[mt][1] - ms1);
      const float p10 = ex2_approx(s[mt][2] - ms0);
      const float p11 = ex2_approx(s[mt][3] - ms1);

      smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
      smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
      smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
      smem.p[m1][k_local1] = __float2bfloat16_rn(p11);

      const float wl0 = col_reduce_sum8(p00 + p10);
      const float wl1 = col_reduce_sum8(p01 + p11);
      if (gid == 0) {
        smem.l[w][m0] = wl0;
        smem.l[w][m1] = wl1;
      }
    }

    // Issue BOTH K_{it+1} and Rel_{it+1} in ONE cp.async group while V_it finishes and V_it^T @ P_it^T runs!
    if (it + 1 < tile_count) {
      const int k0_nxt = k_min + (tile_start + it + 1) * kBN;
      const bool full_nxt = (k0_nxt + kBN <= past) && (past + 7 - k0_nxt < kWindow);
      cp_async_ring_tile(smem.k[nxt], ring_k_base, k0_nxt, past, tid, full_nxt);
      cp_async_rel_tile(smem.rel, Rel, q_start, kv_head, past, k0_nxt, tid, full_nxt);
      cp_async_commit();
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();

    // Update l_i and rescale acc by alpha.
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float sum0 = (smem.l[0][m0] + smem.l[1][m0]) + (smem.l[2][m0] + smem.l[3][m0]);
      const float sum1 = (smem.l[0][m1] + smem.l[1][m1]) + (smem.l[2][m1] + smem.l[3][m1]);
      l_i[mt][0] = __fmaf_rn(l_i[mt][0], alpha[mt][0], sum0);
      l_i[mt][1] = __fmaf_rn(l_i[mt][1], alpha[mt][1], sum1);
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        acc[dt][mt][0] *= alpha[mt][0];
        acc[dt][mt][1] *= alpha[mt][1];
        acc[dt][mt][2] *= alpha[mt][0];
        acc[dt][mt][3] *= alpha[mt][1];
      }
    }

    // Phase 2: O^T += V_it^T @ P_it^T (128 x 24).
    compute_vp_tile<4>(acc, smem.v, smem.p, w, gid, tig);

    // Ensure all warps finish reading smem.v before next iter overwrites it with V_{it+1}.
    if (it + 1 < tile_count || do_cur) {
      __syncthreads();
    }
  }

  // ---------------- 8-key Cur Micro-Tile ----------------
  if (do_cur) {
    const int r = tid >> 3;          // 0..15
    const int c16 = tid & 7;         // 0..7
    const int col_byte = c16 << 4;
    const int sw_col = col_byte ^ ((r & 7) << 4);
    if (r < kQlen) {
      const int k_real = past + r;
      const bool valid_k = (k_real >= 0 && k_real < seqlen_k);
      const int64_t cur_off = (static_cast<int64_t>(q_start + r) * kKvHeads + kv_head) * kHeadDim + col_byte;
      cp_async_cg_16_pred(&smem.k[0][r][sw_col], KCur + cur_off, valid_k ? 16 : 0);
      cp_async_cg_16_pred(&smem.v[r][sw_col], VCur + cur_off, valid_k ? 16 : 0);
    } else {
      *reinterpret_cast<uint4*>(&smem.k[0][r][sw_col]) = make_uint4(0u, 0u, 0u, 0u);
      *reinterpret_cast<uint4*>(&smem.v[r][sw_col]) = make_uint4(0u, 0u, 0u, 0u);
    }
    if (tid < kM) {
      const int tok = tid / kGroup;
      const int head = kv_head * kGroup + (tid - tok * kGroup);
      const int64_t row_off = static_cast<int64_t>(q_start + tok) * (kHeads * kRelExtent)
                            + static_cast<int64_t>(head) * kRelExtent;
      cp_async_cg_16(&smem.rel[tid][0], Rel + row_off);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();

    if (w == 0) {
      float s[3][4];
      compute_kq_tile_qreg(s, smem.k[0], q_b0, q_b1, 0, gid, tig);

      const int k_real = past + gid;
      const bool k_ok = (k_real >= 0 && k_real < seqlen_k);
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        const int m0 = mt * 8 + tig * 2;
        const int m1 = m0 + 1;
        const int d0 = owned_tok[mt][0] - gid;
        const int d1 = owned_tok[mt][1] - gid;
        const bool v0 = k_ok && (d0 >= 0);
        const bool v1 = k_ok && (d1 >= 0);
        const float r0 = v0 ? __bfloat162float(smem.rel[m0][d0]) : 0.0f;
        const float r1 = v1 ? __bfloat162float(smem.rel[m1][d1]) : 0.0f;
        const float s0 = v0 ? __fmaf_rn(s[mt][0], qk_scale, r0 * kLog2E) : kNegInf;
        const float s1 = v1 ? __fmaf_rn(s[mt][1], qk_scale, r1 * kLog2E) : kNegInf;

        const float tm0 = col_reduce_max8(s0);
        const float tm1 = col_reduce_max8(s1);
        const float mn0 = fmaxf(m_i[mt][0], tm0);
        const float mn1 = fmaxf(m_i[mt][1], tm1);
        const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
        const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;

        const float p0 = ex2_approx(s0 - ms0);
        const float p1 = ex2_approx(s1 - ms1);
        const float sum0 = col_reduce_sum8(p0);
        const float sum1 = col_reduce_sum8(p1);

        if (gid == 0) {
          smem.m[0][m0] = tm0;
          smem.m[0][m1] = tm1;
          smem.l[0][m0] = sum0;
          smem.l[0][m1] = sum1;
        }
        smem.p[m0][gid] = __float2bfloat16_rn(p0);
        smem.p[m1][gid] = __float2bfloat16_rn(p1);
        smem.p[m0][gid + 8] = __float2bfloat16_rn(0.0f);
        smem.p[m1][gid + 8] = __float2bfloat16_rn(0.0f);
      }
    }
    __syncthreads();

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = smem.m[0][m0];
      const float tm1 = smem.m[0][m1];
      const float sum0 = smem.l[0][m0];
      const float sum1 = smem.l[0][m1];
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kNegInf) ? 0.0f : mn1;
      const float a0 = ex2_approx(m_i[mt][0] - ms0);
      const float a1 = ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;
      l_i[mt][0] = __fmaf_rn(l_i[mt][0], a0, sum0);
      l_i[mt][1] = __fmaf_rn(l_i[mt][1], a1, sum1);
#pragma unroll
      for (int dt = 0; dt < 2; ++dt) {
        acc[dt][mt][0] *= a0;
        acc[dt][mt][1] *= a1;
        acc[dt][mt][2] *= a0;
        acc[dt][mt][3] *= a1;
      }
    }
    compute_vp_tile<1>(acc, smem.v, smem.p, w, gid, tig);
  }

  // Launch dependent combine kernel (preloads Gate/Sink and waits at gdc_wait).
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  // Stage normalized O into smem.k (16 KB -> float[24][128] = 12 KB) for coalesced float4 stores.
  __syncthreads();
  float (*smem_o)[kHeadDim] = reinterpret_cast<float (*)[kHeadDim]>(&smem.k[0][0][0]);

#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    const int m0 = mt * 8 + tig * 2;
    const int m1 = m0 + 1;
    const float inv0 = (l_i[mt][0] == 0.0f) ? 0.0f : (v_scale / l_i[mt][0]);
    const float inv1 = (l_i[mt][1] == 0.0f) ? 0.0f : (v_scale / l_i[mt][1]);
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      const int d0 = w * 32 + dt * 16 + gid;
      const int d1 = d0 + 8;
      smem_o[m0][d0] = acc[dt][mt][0] * inv0;
      smem_o[m1][d0] = acc[dt][mt][1] * inv1;
      smem_o[m0][d1] = acc[dt][mt][2] * inv0;
      smem_o[m1][d1] = acc[dt][mt][3] * inv1;
    }
    if (w == 0 && gid == 0) {
      const int tok0 = owned_tok[mt][0];
      const int tok1 = owned_tok[mt][1];
      const int h0 = kv_head * kGroup + (m0 - tok0 * kGroup);
      const int h1 = kv_head * kGroup + (m1 - tok1 * kGroup);
      const float lse0 = (l_i[mt][0] == 0.0f) ? kNegInf : (m_i[mt][0] + lg2_approx(l_i[mt][0]));
      const float lse1 = (l_i[mt][1] == 0.0f) ? kNegInf : (m_i[mt][1] + lg2_approx(l_i[mt][1]));
      LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok0)) * kHeads + h0] = lse0;
      LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok1)) * kHeads + h1] = lse1;
    }
  }
  __syncthreads();

#pragma unroll
  for (int step = 0; step < 6; ++step) {
    const int idx = tid + step * kThreads;
    const int m = idx >> 5;
    const int d4 = (idx & 31) << 2;
    const int tok = m / kGroup;
    const int head = kv_head * kGroup + (m - tok * kGroup);
    const int64_t part = (static_cast<int64_t>(split) * num_tokens + (q_start + tok)) * kHeads + head;
    *reinterpret_cast<float4*>(OPart + part * kHeadDim + d4) =
        *reinterpret_cast<const float4*>(&smem_o[m][d4]);
  }
}

// ---------------------------------------------------------------------------
// 2. Fast 1-Warp Float4 Split-KV Combine + Gate + Sink Epilogue (SM120)
// Grid: [num_tokens, 24 (heads), 1], Block: [32, 1, 1]
// ---------------------------------------------------------------------------
template <int NUM_SPLITS>
__device__ __forceinline__ void attn_verify_combine_impl(
    const float* __restrict__ OPart,
    const float* __restrict__ LsePart,
    const bf16* __restrict__ Gate,
    const bf16* __restrict__ Sink,
    bf16* __restrict__ Y,
    int num_tokens,
    int stride_gate,
    int stride_sink) {
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const int d4 = lane << 2;

  const uint2 gate_raw = *reinterpret_cast<const uint2*>(
      Gate + row * stride_gate + head * kHeadDim + d4);
  const bf16* g_bf = reinterpret_cast<const bf16*>(&gate_raw);
  const float g0 = __bfloat162float(g_bf[0]);
  const float g1 = __bfloat162float(g_bf[1]);
  const float g2 = __bfloat162float(g_bf[2]);
  const float g3 = __bfloat162float(g_bf[3]);
  const float sink_val = __bfloat162float(Sink[row * stride_sink + head]);

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int64_t base_row_head = row * kHeads + head;
  const int64_t split_stride = static_cast<int64_t>(num_tokens) * kHeads;

  float lse_s = (lane < NUM_SPLITS) ? LsePart[static_cast<int64_t>(lane) * split_stride + base_row_head] : kNegInf;

  float lse_max = lse_s;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    lse_max = fmaxf(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, off));
  }

  bf16* dst = Y + row * (kHeads * kHeadDim) + head * kHeadDim + d4;
  if (lse_max == kNegInf) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  float w_lane = (lane < NUM_SPLITS) ? ex2_approx(lse_s - lse_max) : 0.0f;
  float w_sum = w_lane;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
  }

  if (w_sum == 0.0f) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const float inv_w_sum = 1.0f / w_sum;
  const float w_norm_lane = w_lane * inv_w_sum;

  uint32_t active_mask = __ballot_sync(0xffffffffu, w_lane > 0.0f);

  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  while (active_mask != 0u) {
    const int s = __ffs(active_mask) - 1;
    active_mask &= (active_mask - 1u);
    const float ws = __shfl_sync(0xffffffffu, w_norm_lane, s);
    const int64_t part = static_cast<int64_t>(s) * split_stride + base_row_head;
    const float4 v = *reinterpret_cast<const float4*>(OPart + part * kHeadDim + d4);
    o0 = __fmaf_rn(ws, v.x, o0);
    o1 = __fmaf_rn(ws, v.y, o1);
    o2 = __fmaf_rn(ws, v.z, o2);
    o3 = __fmaf_rn(ws, v.w, o3);
  }

  const float lse_nat = (lg2_approx(w_sum) + lse_max) * kLn2;
  const float sink_scale = sigmoid_fast(lse_nat - sink_val);

  uint2 out_raw;
  bf16* out_bf = reinterpret_cast<bf16*>(&out_raw);
  out_bf[0] = __float2bfloat16_rn(o0 * sigmoid_fast(g0) * sink_scale);
  out_bf[1] = __float2bfloat16_rn(o1 * sigmoid_fast(g1) * sink_scale);
  out_bf[2] = __float2bfloat16_rn(o2 * sigmoid_fast(g2) * sink_scale);
  out_bf[3] = __float2bfloat16_rn(o3 * sigmoid_fast(g3) * sink_scale);
  *reinterpret_cast<uint2*>(dst) = out_raw;
}

#define DEFINE_COMBINE(S) \
  extern "C" __global__ void __launch_bounds__(32) kern_attn_verify_combine_s##S( \
      const float* __restrict__ OPart, const float* __restrict__ LsePart, \
      const bf16* __restrict__ Gate, const bf16* __restrict__ Sink, bf16* __restrict__ Y, \
      int num_tokens, int stride_gate, int stride_sink) { \
    attn_verify_combine_impl<S>(OPart, LsePart, Gate, Sink, Y, num_tokens, stride_gate, stride_sink); \
  }

DEFINE_COMBINE(4)
DEFINE_COMBINE(5)
DEFINE_COMBINE(8)
DEFINE_COMBINE(9)
DEFINE_COMBINE(16)
DEFINE_COMBINE(17)
DEFINE_COMBINE(32)
#undef DEFINE_COMBINE
