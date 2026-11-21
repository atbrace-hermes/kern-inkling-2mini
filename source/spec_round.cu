// SPDX-License-Identifier: Apache-2.0
// Adapted from kern/tools/kernels-src/spec_round.cu for SM120,
// plus SM120 Blackwell Tensor-Core Split-KV Paged FP8 Attention + PDL Combine for DSpark layer 1.

#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

extern "C" __global__ void kern_splice_draft(
    const long long* __restrict__ anchor, long long* __restrict__ ids, int block, long long mask) {
  const int i = blockIdx.x;
  const int t = threadIdx.x;
  if (t == 0) ids[(long long)i * block] = anchor[i];
  if (t < block - 1) ids[(long long)i * block + 1 + t] = mask;
}

extern "C" __global__ void kern_splice_verify(
    const long long* __restrict__ anchor, const long long* __restrict__ drafts,
    long long* __restrict__ ids, int block, int dstride) {
  const int i = blockIdx.x;
  const int t = threadIdx.x;
  if (t == 0) ids[(long long)i * block] = anchor[i];
  if (t < block - 1) ids[(long long)i * block + 1 + t] = drafts[(long long)i * dstride + t];
}

extern "C" __global__ void kern_spec_count(
    const long long* __restrict__ drafts, const long long* __restrict__ verify,
    int* __restrict__ nacc, int block, int dstride) {
  const int i = blockIdx.x;
  if (threadIdx.x != 0) return;
  const long long* d = drafts + (long long)i * dstride;
  const long long* v = verify + (long long)i * block;
  int a = 0;
  while (a < block - 1 && d[a] == v[a]) ++a;
  nacc[i] = a + 1;
}

extern "C" __global__ void kern_spec_lines(
    const int* __restrict__ line_in, const int* __restrict__ nacc, int* __restrict__ line_adv,
    int block, int rows, int seqs_max) {
  const int i = blockIdx.x;
  const int a = nacc[i] - 1;
  for (int r = threadIdx.x; r < rows; r += blockDim.x) {
    const long long cell = ((long long)r * seqs_max + i) * block;
    const int line = line_in[cell];
    for (int e = 0; e < block; ++e) line_adv[cell + e] = e == a ? line : 0;
  }
}

extern "C" __global__ void kern_ones_i32(int* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = 1;
}

// ============================================================================
// SM120 Blackwell Transposed-MMA Split-KV Paged FP8 Attention for DSpark Layer 1
// Bidirectional block attention: Qlen=8, H=24, HKV=8, G=3, M=24, D=128, PAGE=64
// - 1 full page (64 keys) per tile, 16B XOR-swizzled cp.async.cg double-buffered K + overlapped V
// - Q resident in registers (zero SMEM Q reads in hot loop)
// - Blackwell 1-instr cvt.rn.bf16x2.e4m3x2 (F2FP.BF16.E4M3)
// - 28.8 KB SMEM -> 3 CTAs/SM on SM120 (zero spills)
// - PDL-chained 1-warp float4 split-KV combine
// ============================================================================

namespace {

using bf16 = __nv_bfloat16;

constexpr int kDsHeads = 24;
constexpr int kDsGroup = 3;
constexpr int kDsHeadDim = 128;
constexpr int kDsQlen = 8;
constexpr int kDsM = kDsQlen * kDsGroup;  // 24
constexpr int kDsBN = 64;                 // == PAGE=64 (1 page per tile)
constexpr int kDsThreads = 128;           // 4 warps
constexpr int kDsPRow = 72;               // 72 bf16 = 144 B -> zero bank conflicts

constexpr float kDsLog2E = 1.4426950408889634f;
constexpr float kDsNegInf = -INFINITY;

__device__ __forceinline__ uint32_t ds_smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void ds_cp_async_cg_16(void* dst_smem, const void* src_gmem) {
  uint32_t dst = ds_smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst), "l"(src_gmem) : "memory");
}

__device__ __forceinline__ void ds_cp_async_cg_16_pred(void* dst_smem, const void* src_gmem, int src_size) {
  uint32_t dst = ds_smem_u32(dst_smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
               :: "r"(dst), "l"(src_gmem), "r"(src_size) : "memory");
}

__device__ __forceinline__ void ds_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
__device__ __forceinline__ void ds_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

__device__ __forceinline__ uint32_t ds_cvt_e4m3x2_to_bf16x2(uint16_t x) {
  uint32_t res;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(res) : "h"(x));
  return res;
}

__device__ __forceinline__ void ds_mma_m16n8k16_bf16(
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

__device__ __forceinline__ float ds_ex2_approx(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float ds_lg2_approx(float x) {
  float y;
  asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ int ds_swiz_q_col(int row, int col_bf16) {
  return col_bf16 ^ ((row & 7) << 3);
}

__device__ __forceinline__ float ds_col_reduce_max8(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  return v;
}

__device__ __forceinline__ float ds_col_reduce_sum8(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  return v;
}

// Load 64x128 FP8 page/ring tile into dst[64][128] with 16B XOR swizzle.
__device__ __forceinline__ void ds_cp_async_paged_tile(
    uint8_t (&dst)[kDsBN][kDsHeadDim],
    const uint8_t* __restrict__ page_base,
    int64_t kv_slot_stride,
    int k0,
    int first_key,
    int prefix,
    int tid,
    bool full_valid) {
  const int c16 = tid & 7;
  const int r0 = tid >> 3;
  const int col_byte = c16 << 4;
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    const int r = r0 + step * 16;
    const int swiz_col = col_byte ^ ((r & 7) << 4);
    const uint8_t* src = page_base + static_cast<int64_t>(r) * kv_slot_stride + col_byte;
    if (full_valid) {
      ds_cp_async_cg_16(&dst[r][swiz_col], src);
    } else {
      const int key = k0 + r;
      const int sz = (key >= first_key && key < prefix) ? 16 : 0;
      ds_cp_async_cg_16_pred(&dst[r][swiz_col], src, sz);
    }
  }
}

// S^T = K_fp8[64x128] @ Q^T[128x24] for warp w (keys w*16..w*16+15).
__device__ __forceinline__ void ds_compute_kq_fp8_qreg(
    float (&s)[3][4],
    const uint8_t (&smem_k)[kDsBN][kDsHeadDim],
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
    a[0] = ds_cvt_e4m3x2_to_bf16x2(raw0);
    a[1] = ds_cvt_e4m3x2_to_bf16x2(raw1);
    a[2] = ds_cvt_e4m3x2_to_bf16x2(raw2);
    a[3] = ds_cvt_e4m3x2_to_bf16x2(raw3);
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      ds_mma_m16n8k16_bf16(s[mt], a, q_b0[ds][mt], q_b1[ds][mt]);
    }
  }
}

// O^T += V_fp8^T[128x64] @ P^T[64x24] for warp w (dims w*32..w*32+31).
__device__ __forceinline__ void ds_compute_vp_fp8_tile(
    float (&acc)[2][3][4],
    const uint8_t (&smem_v)[kDsBN][kDsHeadDim],
    const bf16 (&smem_p)[kDsM][kDsPRow],
    int w,
    int gid,
    int tig) {
#pragma unroll
  for (int ks = 0; ks < 4; ++ks) {
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
      a[0] = ds_cvt_e4m3x2_to_bf16x2(v_lo0);
      a[1] = ds_cvt_e4m3x2_to_bf16x2(v_hi0);
      a[2] = ds_cvt_e4m3x2_to_bf16x2(v_lo1);
      a[3] = ds_cvt_e4m3x2_to_bf16x2(v_hi1);
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        ds_mma_m16n8k16_bf16(acc[dt][mt], a, b0[mt], b1[mt]);
      }
    }
  }
}

// Total static SMEM = 28,800 B -> 3 CTAs/SM = 89.5 KB <= 101,376 B on SM120!
struct DsparkPagedSmem {
  alignas(16) uint8_t k[2][kDsBN][kDsHeadDim];  // 16,384 B (k[1] stages Q; k[0..1] stages O at epilogue)
  alignas(16) uint8_t v[kDsBN][kDsHeadDim];     //  8,192 B (also holds live BF16 V)
  alignas(16) bf16 p[kDsM][kDsPRow];            //  3,456 B
  alignas(16) float m[4][kDsM];                 //    384 B
  alignas(16) float l[4][kDsM];                 //    384 B
};

}  // namespace

constexpr int kDsRingWindow = 1024;
constexpr int64_t kDsRingSlotStride = 2048;
constexpr int64_t kDsRingVOffset = 1024;

template <bool IS_RING>
__device__ __forceinline__ void dspark_attn_split_sm120_impl(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    const uint8_t* __restrict__ Cache,
    const int32_t* __restrict__ BlockTableOrLines,
    const int32_t* __restrict__ Lengths,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    float k_scale,
    float v_scale,
    float softmax_scale,
    int64_t stride_q_t,
    int64_t stride_k_t,
    int64_t stride_v_t,
    int64_t stride_bt_or_seq,
    int64_t kv_layer_offset,
    int64_t kv_page_stride,
    int64_t kv_slot_stride,
    int64_t kv_head_stride,
    int64_t kv_v_offset,
    int num_tokens,
    int num_splits) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int seq = blockIdx.z;
  const int tid = threadIdx.x;
  const int w = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  const int q_start = seq * kDsQlen;
  const int seqlen_k = Lengths[seq];

  auto store_empty_and_return = [&]() {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (tid < kDsM) {
      const int tok = tid / kDsGroup;
      const int head = kv_head * kDsGroup + (tid - tok * kDsGroup);
      const int token = q_start + tok;
      if (token < num_tokens) {
        const int64_t part = (static_cast<int64_t>(split) * num_tokens + token) * kDsHeads + head;
        LsePart[part] = kDsNegInf;
      }
    }
  };

  if (seqlen_k <= 0) {
    store_empty_and_return();
    return;
  }

  const int prefix = (seqlen_k > kDsQlen) ? (seqlen_k - kDsQlen) : 0;
  const int num_live = seqlen_k - prefix;
  int first_key = 0;
  int n_lo = 0;
  int n_hi = 0;
  int64_t ring_base_off = 0;
  if constexpr (IS_RING) {
    const int64_t line = static_cast<int64_t>(BlockTableOrLines[seq]);
    if (line > 0) {
      first_key = (prefix > kDsRingWindow) ? (prefix - kDsRingWindow) : 0;
      if (first_key < prefix) {
        n_lo = first_key >> 6;
        n_hi = (prefix + kDsBN - 1) >> 6;
      }
      ring_base_off = line * stride_bt_or_seq + static_cast<int64_t>(kv_head) * kDsHeadDim;
    } else {
      first_key = prefix;
    }
  } else {
    n_lo = 0;
    n_hi = (prefix + kDsBN - 1) / kDsBN;
  }

  const int n_tiles_total = n_hi - n_lo;
  const bool do_live = (split == num_splits - 1);

  const int base_tiles = n_tiles_total / num_splits;
  const int rem_tiles = n_tiles_total % num_splits;
  const int tile_start = n_lo + split * base_tiles + (split < rem_tiles ? split : rem_tiles);
  const int tile_count = base_tiles + (split < rem_tiles ? 1 : 0);

  if (tile_count == 0 && !do_live) {
    store_empty_and_return();
    return;
  }

  __shared__ DsparkPagedSmem smem;

  const int32_t* bt = IS_RING ? nullptr : (BlockTableOrLines + static_cast<int64_t>(seq) * stride_bt_or_seq);
  const int64_t kv_head_off = IS_RING ? ring_base_off : (kv_layer_offset + static_cast<int64_t>(kv_head) * kv_head_stride);
  const int64_t slot_stride = IS_RING ? kDsRingSlotStride : kv_slot_stride;
  const int64_t v_offset = IS_RING ? kDsRingVOffset : kv_v_offset;

  auto tile_k_ptr = [&](int tidx) -> const uint8_t* {
    if constexpr (IS_RING) {
      const int64_t ring_slot = static_cast<int64_t>((tidx * kDsBN) & (kDsRingWindow - 1));
      return Cache + kv_head_off + ring_slot * kDsRingSlotStride;
    } else {
      const int64_t page = static_cast<int64_t>(bt[tidx]);
      return Cache + kv_head_off + page * kv_page_stride;
    }
  };

  // Prefetch first K tile into smem.k[0] while staging Q into smem.k[1].
  if (tile_count > 0) {
    const int k0_first = tile_start * kDsBN;
    const uint8_t* p0_base = tile_k_ptr(tile_start);
    const bool full0 = (k0_first >= first_key) && (k0_first + kDsBN <= prefix);
    ds_cp_async_paged_tile(smem.k[0], p0_base, slot_stride, k0_first, first_key, prefix, tid, full0);
    ds_cp_async_commit();
  }

  // Stage Q[24][128] bf16 into smem.k[1] (6 KB <= 8 KB), then load into registers.
  bf16 (*smem_q)[kDsHeadDim] = reinterpret_cast<bf16 (*)[kDsHeadDim]>(&smem.k[1][0][0]);
#pragma unroll
  for (int step = 0; step < 3; ++step) {
    const int idx = tid + step * kDsThreads;
    const int m = idx >> 4;
    const int c16 = idx & 15;
    const int tok = m / kDsGroup;
    const int head = kv_head * kDsGroup + (m - tok * kDsGroup);
    const int col_bf16 = c16 << 3;
    const int sw_col = ds_swiz_q_col(m, col_bf16);
    const int token = q_start + tok;
    if (token < num_tokens) {
      const bf16* q_src = Q + static_cast<int64_t>(token) * stride_q_t + head * kDsHeadDim + col_bf16;
      *reinterpret_cast<uint4*>(&smem_q[m][sw_col]) = *reinterpret_cast<const uint4*>(q_src);
    } else {
      *reinterpret_cast<uint4*>(&smem_q[m][sw_col]) = make_uint4(0u, 0u, 0u, 0u);
    }
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
      const int q_c0 = ds_swiz_q_col(q_m, d_step) + tig * 2;
      const int q_c1 = ds_swiz_q_col(q_m, d_step + 8) + tig * 2;
      q_b0[ds][mt] = *reinterpret_cast<const uint32_t*>(&smem_q[q_m][q_c0]);
      q_b1[ds][mt] = *reinterpret_cast<const uint32_t*>(&smem_q[q_m][q_c1]);
    }
  }
  // Ensure all threads finish reading smem.k[1] (Q) before iter 0 prefetches K_1 into smem.k[1].
  __syncthreads();

  const float qk_scale_old = softmax_scale * k_scale * kDsLog2E;

  float m_i[3][2];
  float l_i[3][2];
  float acc[2][3][4];
#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    m_i[mt][0] = kDsNegInf; m_i[mt][1] = kDsNegInf;
    l_i[mt][0] = 0.0f;      l_i[mt][1] = 0.0f;
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      acc[dt][mt][0] = 0.0f; acc[dt][mt][1] = 0.0f;
      acc[dt][mt][2] = 0.0f; acc[dt][mt][3] = 0.0f;
    }
  }

  // ---------------- Main Prefix Loop (1 full 64-key page/ring tile per iter) ----------------
  for (int it = 0; it < tile_count; ++it) {
    const int cur = it & 1;
    const int nxt = cur ^ 1;
    const int page_idx = tile_start + it;
    const int k0 = page_idx * kDsBN;
    const bool is_full = (k0 >= first_key) && (k0 + kDsBN <= prefix);

    const uint8_t* v_base = tile_k_ptr(page_idx) + v_offset;

    // Issue V_it into smem.v while K_it finishes and K_it @ Q^T runs.
    ds_cp_async_paged_tile(smem.v, v_base, slot_stride, k0, first_key, prefix, tid, is_full);
    ds_cp_async_commit();

    // Wait for K_it (V_it stays in flight!).
    ds_cp_async_wait<1>();
    __syncthreads();

    // Phase 1: S^T = K_it @ Q^T (64 x 24).
    float s[3][4];
    ds_compute_kq_fp8_qreg(s, smem.k[cur], q_b0, q_b1, w, gid, tig);

    const int k_local0 = w * 16 + gid;
    const int k_local1 = k_local0 + 8;
    const bool ok0 = is_full || ((k0 + k_local0 >= first_key) && (k0 + k_local0 < prefix));
    const bool ok1 = is_full || ((k0 + k_local1 >= first_key) && (k0 + k_local1 < prefix));

#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      if (is_full) {
        s[mt][0] *= qk_scale_old;
        s[mt][1] *= qk_scale_old;
        s[mt][2] *= qk_scale_old;
        s[mt][3] *= qk_scale_old;
      } else {
        s[mt][0] = ok0 ? (s[mt][0] * qk_scale_old) : kDsNegInf;
        s[mt][1] = ok0 ? (s[mt][1] * qk_scale_old) : kDsNegInf;
        s[mt][2] = ok1 ? (s[mt][2] * qk_scale_old) : kDsNegInf;
        s[mt][3] = ok1 ? (s[mt][3] * qk_scale_old) : kDsNegInf;
      }
      const float wm0 = ds_col_reduce_max8(fmaxf(s[mt][0], s[mt][2]));
      const float wm1 = ds_col_reduce_max8(fmaxf(s[mt][1], s[mt][3]));
      if (gid == 0) {
        smem.m[w][m0] = wm0;
        smem.m[w][m1] = wm1;
      }
    }
    __syncthreads();

    float alpha[3][2];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m0 = mt * 8 + tig * 2;
      const int m1 = m0 + 1;
      const float tm0 = fmaxf(fmaxf(smem.m[0][m0], smem.m[1][m0]), fmaxf(smem.m[2][m0], smem.m[3][m0]));
      const float tm1 = fmaxf(fmaxf(smem.m[0][m1], smem.m[1][m1]), fmaxf(smem.m[2][m1], smem.m[3][m1]));
      const float mn0 = fmaxf(m_i[mt][0], tm0);
      const float mn1 = fmaxf(m_i[mt][1], tm1);
      const float ms0 = (mn0 == kDsNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kDsNegInf) ? 0.0f : mn1;
      alpha[mt][0] = ds_ex2_approx(m_i[mt][0] - ms0);
      alpha[mt][1] = ds_ex2_approx(m_i[mt][1] - ms1);
      m_i[mt][0] = mn0;
      m_i[mt][1] = mn1;

      const float p00 = ds_ex2_approx(s[mt][0] - ms0);
      const float p01 = ds_ex2_approx(s[mt][1] - ms1);
      const float p10 = ds_ex2_approx(s[mt][2] - ms0);
      const float p11 = ds_ex2_approx(s[mt][3] - ms1);

      smem.p[m0][k_local0] = __float2bfloat16_rn(p00);
      smem.p[m1][k_local0] = __float2bfloat16_rn(p01);
      smem.p[m0][k_local1] = __float2bfloat16_rn(p10);
      smem.p[m1][k_local1] = __float2bfloat16_rn(p11);

      const float wl0 = ds_col_reduce_sum8(p00 + p10);
      const float wl1 = ds_col_reduce_sum8(p01 + p11);
      if (gid == 0) {
        smem.l[w][m0] = wl0;
        smem.l[w][m1] = wl1;
      }
    }

    // Prefetch K_{it+1} into smem.k[nxt] while waiting for V_it.
    if (it + 1 < tile_count) {
      const int next_pidx = page_idx + 1;
      const int k0_nxt = next_pidx * kDsBN;
      const uint8_t* kn_base = tile_k_ptr(next_pidx);
      const bool full_nxt = (k0_nxt >= first_key) && (k0_nxt + kDsBN <= prefix);
      ds_cp_async_paged_tile(smem.k[nxt], kn_base, slot_stride, k0_nxt, first_key, prefix, tid, full_nxt);
      ds_cp_async_commit();
      ds_cp_async_wait<1>();
    } else {
      ds_cp_async_wait<0>();
    }
    __syncthreads();

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
    ds_compute_vp_fp8_tile(acc, smem.v, smem.p, w, gid, tig);

    if (it + 1 < tile_count || do_live) {
      __syncthreads();
    }
  }

  // Apply FP8 v_scale to prefix accumulator BEFORE merging unscaled BF16 live block.
  if (tile_count > 0 && v_scale != 1.0f) {
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        acc[dt][mt][0] *= v_scale;
        acc[dt][mt][1] *= v_scale;
        acc[dt][mt][2] *= v_scale;
        acc[dt][mt][3] *= v_scale;
      }
    }
  }

  // ---------------- Live BF16 Draft Block (1..8 tokens, bidirectional) ----------------
  if (do_live) {
    bf16 (*smem_kl)[kDsHeadDim] = reinterpret_cast<bf16 (*)[kDsHeadDim]>(&smem.k[0][0][0]);
    bf16 (*smem_vl)[kDsHeadDim] = reinterpret_cast<bf16 (*)[kDsHeadDim]>(&smem.v[0][0]);
#pragma unroll
    for (int step = 0; step < 2; ++step) {
      const int idx = tid + step * kDsThreads;
      const int r = idx >> 4;          // 0..15
      const int col_bf16 = (idx & 15) << 3;
      const int sw_col = ds_swiz_q_col(r, col_bf16);
      if (r < num_live) {
        const int64_t ltok = static_cast<int64_t>(q_start + r);
        const bf16* ks = K + ltok * stride_k_t + kv_head * kDsHeadDim + col_bf16;
        const bf16* vs = V + ltok * stride_v_t + kv_head * kDsHeadDim + col_bf16;
        *reinterpret_cast<uint4*>(&smem_kl[r][sw_col]) = *reinterpret_cast<const uint4*>(ks);
        *reinterpret_cast<uint4*>(&smem_vl[r][sw_col]) = *reinterpret_cast<const uint4*>(vs);
      } else {
        *reinterpret_cast<uint4*>(&smem_kl[r][sw_col]) = make_uint4(0u, 0u, 0u, 0u);
        *reinterpret_cast<uint4*>(&smem_vl[r][sw_col]) = make_uint4(0u, 0u, 0u, 0u);
      }
    }
    __syncthreads();

    const float qk_scale_live = softmax_scale * kDsLog2E;
    if (w == 0) {
      float s[3][4];
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        s[mt][0] = 0.0f; s[mt][1] = 0.0f; s[mt][2] = 0.0f; s[mt][3] = 0.0f;
      }
#pragma unroll
      for (int ds = 0; ds < 8; ++ds) {
        const int d_step = ds << 4;
        const int k_row0 = gid;
        const int k_row1 = gid + 8;
        uint32_t a[4];
        a[0] = *reinterpret_cast<const uint32_t*>(&smem_kl[k_row0][ds_swiz_q_col(k_row0, d_step) + tig * 2]);
        a[1] = *reinterpret_cast<const uint32_t*>(&smem_kl[k_row1][ds_swiz_q_col(k_row1, d_step) + tig * 2]);
        a[2] = *reinterpret_cast<const uint32_t*>(&smem_kl[k_row0][ds_swiz_q_col(k_row0, d_step + 8) + tig * 2]);
        a[3] = *reinterpret_cast<const uint32_t*>(&smem_kl[k_row1][ds_swiz_q_col(k_row1, d_step + 8) + tig * 2]);
#pragma unroll
        for (int mt = 0; mt < 3; ++mt) {
          ds_mma_m16n8k16_bf16(s[mt], a, q_b0[ds][mt], q_b1[ds][mt]);
        }
      }

      const bool k_ok = (gid < num_live);
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        const int m0 = mt * 8 + tig * 2;
        const int m1 = m0 + 1;
        const float s0 = k_ok ? (s[mt][0] * qk_scale_live) : kDsNegInf;
        const float s1 = k_ok ? (s[mt][1] * qk_scale_live) : kDsNegInf;
        const float tm0 = ds_col_reduce_max8(s0);
        const float tm1 = ds_col_reduce_max8(s1);
        const float mn0 = fmaxf(m_i[mt][0], tm0);
        const float mn1 = fmaxf(m_i[mt][1], tm1);
        const float ms0 = (mn0 == kDsNegInf) ? 0.0f : mn0;
        const float ms1 = (mn1 == kDsNegInf) ? 0.0f : mn1;
        const float p0 = ds_ex2_approx(s0 - ms0);
        const float p1 = ds_ex2_approx(s1 - ms1);
        const float sum0 = ds_col_reduce_sum8(p0);
        const float sum1 = ds_col_reduce_sum8(p1);
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
      const float ms0 = (mn0 == kDsNegInf) ? 0.0f : mn0;
      const float ms1 = (mn1 == kDsNegInf) ? 0.0f : mn1;
      const float a0 = ds_ex2_approx(m_i[mt][0] - ms0);
      const float a1 = ds_ex2_approx(m_i[mt][1] - ms1);
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

    // O^T += V_live_bf16^T[128x16] @ P_live^T[16x24] (1 K-step of 16).
    const int k0 = tig * 2;
    const int k1 = k0 + 1;
    const int k2 = k0 + 8;
    const int k3 = k2 + 1;
    uint32_t b0[3], b1[3];
#pragma unroll
    for (int mt = 0; mt < 3; ++mt) {
      const int m = mt * 8 + gid;
      b0[mt] = *reinterpret_cast<const uint32_t*>(&smem.p[m][k0]);
      b1[mt] = *reinterpret_cast<const uint32_t*>(&smem.p[m][k2]);
    }
#pragma unroll
    for (int dt = 0; dt < 2; ++dt) {
      const int d_base = w * 32 + dt * 16;
      const uint16_t lo_k0 = *reinterpret_cast<const uint16_t*>(&smem_vl[k0][ds_swiz_q_col(k0, d_base) + gid]);
      const uint16_t lo_k1 = *reinterpret_cast<const uint16_t*>(&smem_vl[k1][ds_swiz_q_col(k1, d_base) + gid]);
      const uint16_t hi_k0 = *reinterpret_cast<const uint16_t*>(&smem_vl[k0][ds_swiz_q_col(k0, d_base + 8) + gid]);
      const uint16_t hi_k1 = *reinterpret_cast<const uint16_t*>(&smem_vl[k1][ds_swiz_q_col(k1, d_base + 8) + gid]);
      const uint16_t lo_k2 = *reinterpret_cast<const uint16_t*>(&smem_vl[k2][ds_swiz_q_col(k2, d_base) + gid]);
      const uint16_t lo_k3 = *reinterpret_cast<const uint16_t*>(&smem_vl[k3][ds_swiz_q_col(k3, d_base) + gid]);
      const uint16_t hi_k2 = *reinterpret_cast<const uint16_t*>(&smem_vl[k2][ds_swiz_q_col(k2, d_base + 8) + gid]);
      const uint16_t hi_k3 = *reinterpret_cast<const uint16_t*>(&smem_vl[k3][ds_swiz_q_col(k3, d_base + 8) + gid]);
      uint32_t a[4];
      a[0] = static_cast<uint32_t>(lo_k0) | (static_cast<uint32_t>(lo_k1) << 16);
      a[1] = static_cast<uint32_t>(hi_k0) | (static_cast<uint32_t>(hi_k1) << 16);
      a[2] = static_cast<uint32_t>(lo_k2) | (static_cast<uint32_t>(lo_k3) << 16);
      a[3] = static_cast<uint32_t>(hi_k2) | (static_cast<uint32_t>(hi_k3) << 16);
#pragma unroll
      for (int mt = 0; mt < 3; ++mt) {
        ds_mma_m16n8k16_bf16(acc[dt][mt], a, b0[mt], b1[mt]);
      }
    }
  }

  // Launch dependent combine kernel (waits at griddepcontrol.wait).
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  __syncthreads();
  float (*smem_o)[kDsHeadDim] = reinterpret_cast<float (*)[kDsHeadDim]>(&smem.k[0][0][0]);

#pragma unroll
  for (int mt = 0; mt < 3; ++mt) {
    const int m0 = mt * 8 + tig * 2;
    const int m1 = m0 + 1;
    const float inv0 = (l_i[mt][0] == 0.0f) ? 0.0f : (1.0f / l_i[mt][0]);
    const float inv1 = (l_i[mt][1] == 0.0f) ? 0.0f : (1.0f / l_i[mt][1]);
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
      const int tok0 = m0 / kDsGroup;
      const int tok1 = m1 / kDsGroup;
      const int h0 = kv_head * kDsGroup + (m0 - tok0 * kDsGroup);
      const int h1 = kv_head * kDsGroup + (m1 - tok1 * kDsGroup);
      const float lse0 = (l_i[mt][0] == 0.0f) ? kDsNegInf : (m_i[mt][0] + ds_lg2_approx(l_i[mt][0]));
      const float lse1 = (l_i[mt][1] == 0.0f) ? kDsNegInf : (m_i[mt][1] + ds_lg2_approx(l_i[mt][1]));
      if (q_start + tok0 < num_tokens) {
        LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok0)) * kDsHeads + h0] = lse0;
      }
      if (q_start + tok1 < num_tokens) {
        LsePart[(static_cast<int64_t>(split) * num_tokens + (q_start + tok1)) * kDsHeads + h1] = lse1;
      }
    }
  }
  __syncthreads();

#pragma unroll
  for (int step = 0; step < 6; ++step) {
    const int idx = tid + step * kDsThreads;
    const int m = idx >> 5;
    const int d4 = (idx & 31) << 2;
    const int tok = m / kDsGroup;
    const int head = kv_head * kDsGroup + (m - tok * kDsGroup);
    const int token = q_start + tok;
    if (token < num_tokens) {
      const int64_t part = (static_cast<int64_t>(split) * num_tokens + token) * kDsHeads + head;
      *reinterpret_cast<float4*>(OPart + part * kDsHeadDim + d4) =
          *reinterpret_cast<const float4*>(&smem_o[m][d4]);
    }
  }
}

// Fast 1-warp float4 PDL split-KV combine for DSpark layer 1 (NUM_SPLITS <= 32).
// Grid: [num_tokens, 24 (heads), 1], Block: [32, 1, 1]
template <int NUM_SPLITS>
__device__ __forceinline__ void dspark_attn_paged_combine_impl(
    const float* __restrict__ OPart,
    const float* __restrict__ LsePart,
    bf16* __restrict__ Y,
    int num_tokens) {
  const int64_t row = blockIdx.x;
  const int head = blockIdx.y;
  const int lane = threadIdx.x;
  const int d4 = lane << 2;
  if (row >= num_tokens) return;

  asm volatile("griddepcontrol.wait;" ::: "memory");

  const int64_t base_row_head = row * kDsHeads + head;
  const int64_t split_stride = static_cast<int64_t>(num_tokens) * kDsHeads;

  const float lse_s = (lane < NUM_SPLITS)
      ? LsePart[static_cast<int64_t>(lane) * split_stride + base_row_head]
      : kDsNegInf;

  float lse_max = lse_s;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    lse_max = fmaxf(lse_max, __shfl_xor_sync(0xffffffffu, lse_max, off));
  }

  bf16* dst = Y + row * (kDsHeads * kDsHeadDim) + head * kDsHeadDim + d4;
  if (lse_max == kDsNegInf) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const float w_lane = (lane < NUM_SPLITS) ? ds_ex2_approx(lse_s - lse_max) : 0.0f;
  float w_sum = w_lane;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    w_sum += __shfl_xor_sync(0xffffffffu, w_sum, off);
  }
  if (w_sum == 0.0f) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(0u, 0u);
    return;
  }

  const float w_norm_lane = w_lane * (1.0f / w_sum);
  uint32_t active_mask = __ballot_sync(0xffffffffu, w_lane > 0.0f);

  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
  while (active_mask != 0u) {
    const int s = __ffs(active_mask) - 1;
    active_mask &= (active_mask - 1u);
    const float ws = __shfl_sync(0xffffffffu, w_norm_lane, s);
    const int64_t part = static_cast<int64_t>(s) * split_stride + base_row_head;
    const float4 v = *reinterpret_cast<const float4*>(OPart + part * kDsHeadDim + d4);
    o0 = __fmaf_rn(ws, v.x, o0);
    o1 = __fmaf_rn(ws, v.y, o1);
    o2 = __fmaf_rn(ws, v.z, o2);
    o3 = __fmaf_rn(ws, v.w, o3);
  }

  uint2 out_raw;
  bf16* out_bf = reinterpret_cast<bf16*>(&out_raw);
  out_bf[0] = __float2bfloat16_rn(o0);
  out_bf[1] = __float2bfloat16_rn(o1);
  out_bf[2] = __float2bfloat16_rn(o2);
  out_bf[3] = __float2bfloat16_rn(o3);
  *reinterpret_cast<uint2*>(dst) = out_raw;
}

#define DEFINE_DS_COMBINE(S) \
  extern "C" __global__ void __launch_bounds__(32) kern_dspark_attn_paged_combine_s##S( \
      const float* __restrict__ OPart, const float* __restrict__ LsePart, \
      bf16* __restrict__ Y, int num_tokens) { \
    dspark_attn_paged_combine_impl<S>(OPart, LsePart, Y, num_tokens); \
  }

DEFINE_DS_COMBINE(4)
DEFINE_DS_COMBINE(5)
DEFINE_DS_COMBINE(8)
DEFINE_DS_COMBINE(10)
DEFINE_DS_COMBINE(16)
DEFINE_DS_COMBINE(20)
DEFINE_DS_COMBINE(27)
DEFINE_DS_COMBINE(32)
#undef DEFINE_DS_COMBINE

// Grid: [num_splits, 8 (kv_heads), seqs], Block: [128, 1, 1]
extern "C" __global__ void __launch_bounds__(kDsThreads, 3) kern_dspark_attn_paged_split_sm120(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    const uint8_t* __restrict__ Cache,
    const int32_t* __restrict__ BlockTable,
    const int32_t* __restrict__ Lengths,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    float k_scale,
    float v_scale,
    float softmax_scale,
    int64_t stride_q_t,
    int64_t stride_k_t,
    int64_t stride_v_t,
    int64_t stride_bt,
    int64_t kv_layer_offset,
    int64_t kv_page_stride,
    int64_t kv_slot_stride,
    int64_t kv_head_stride,
    int64_t kv_v_offset,
    int num_tokens,
    int num_splits) {
  dspark_attn_split_sm120_impl<false>(
      Q, K, V, Cache, BlockTable, Lengths, OPart, LsePart,
      k_scale, v_scale, softmax_scale, stride_q_t, stride_k_t, stride_v_t,
      stride_bt, kv_layer_offset, kv_page_stride, kv_slot_stride, kv_head_stride,
      kv_v_offset, num_tokens, num_splits);
}

// Grid: [num_splits, 8 (kv_heads), seqs], Block: [128, 1, 1]
extern "C" __global__ void __launch_bounds__(kDsThreads, 3) kern_dspark_attn_ring_split_sm120(
    const bf16* __restrict__ Q,
    const bf16* __restrict__ K,
    const bf16* __restrict__ V,
    const uint8_t* __restrict__ Cache,
    const int32_t* __restrict__ Lines,
    const int32_t* __restrict__ Lengths,
    float* __restrict__ OPart,
    float* __restrict__ LsePart,
    float k_scale,
    float v_scale,
    float softmax_scale,
    int64_t stride_q_t,
    int64_t stride_k_t,
    int64_t stride_v_t,
    int64_t cache_seq_stride,
    int num_tokens,
    int num_splits) {
  dspark_attn_split_sm120_impl<true>(
      Q, K, V, Cache, Lines, Lengths, OPart, LsePart,
      k_scale, v_scale, softmax_scale, stride_q_t, stride_k_t, stride_v_t,
      cache_seq_stride, 0, 0, kDsRingSlotStride, 0, kDsRingVOffset,
      num_tokens, num_splits);
}
