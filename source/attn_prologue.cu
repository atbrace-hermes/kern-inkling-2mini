// SPDX-License-Identifier: Apache-2.0
// Attention prologue + FP8 E4M3 KV write + SWA ring commit + gate_sink (SM120).
// - Q L2 clip (q_l2_clip.scale [24, 128], thresh=22.627416997969522, eps=1e-6) + optional tau
// - R L2 clip (rel_l2_clip.scale [24, 16], thresh=8.0, eps=1e-6) + optional tau + unsheared rel_bias [T, 24, 1024]
// - K L2 clip (no learned scale, thresh=22.627416997969522, eps=1e-6) + FP8 E4M3 quant
// - V RMSNorm (no gamma, eps=1e-6) + FP8 E4M3 quant
// - Full KV write -> states.kv [pages, 9, 2, 8, 64, 128]
// - SWA KV write -> swa_k/swa_v (prefill/verify) or states.seq_state ring (decode)
// - SWA ring commits (prefill & speculative round)

#include <cuda_bf16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

constexpr int kPrologueThreads = 256;
constexpr int kHeads = 24;
constexpr int kKvHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kRelDim = 16;
constexpr int kRelExtent = 1024;

constexpr int64_t kKvPageBytes = 1179648LL;       // 9 * 2 * 8 * 64 * 128
constexpr int64_t kKvHalfPageBytes = 65536LL;     // 8 * 64 * 128
constexpr int64_t kKvHeadPageBytes = 8192LL;      // 64 * 128
constexpr int64_t kSeqStateBytes = 90005376LL;
constexpr int64_t kSwaTargetOffset = 1916928LL;
constexpr int64_t kSwaVisitBytes = 2097152LL;     // 2 * 8 * 1024 * 128
constexpr int64_t kSwaHalfVisitBytes = 1048576LL; // 8 * 1024 * 128
constexpr int64_t kSwaHeadRingBytes = 131072LL;   // 1024 * 128

constexpr float kQkClipThresh = 22.627416997969522f;
constexpr float kRelClipThresh = 8.0f;
constexpr float kEps = 1e-6f;
constexpr float kInvHeadDim = 0.0078125f;         // 1 / 128
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kLn2 = 0.6931471805599453f;

__device__ __forceinline__ float bf_round(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float rsqrt_approx(float x) {
  float y;
  asm("rsqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float lane_sum2(float v) {
  return __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ float lane_sum8(float v) {
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ float warp_sum32(float v) {
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 16));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
  return __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
}

__device__ __forceinline__ float clip_coeff(float sum_sq, float thresh, float eps, float max_coeff = 1.0f) {
  if (sum_sq == 0.0f) return max_coeff;
  if (!(sum_sq > 0.0f)) return sum_sq;  // propagate NaN
  if (!isfinite(sum_sq)) return 0.0f;   // +inf guard: avoid +inf * 0 = NaN
  float norm = __fmul_rn(sum_sq, rsqrt_approx(sum_sq));
  float coeff = __fdiv_rn(thresh, __fadd_rn(norm, eps));
  return coeff >= max_coeff ? max_coeff : coeff;
}

__device__ __forceinline__ unsigned short f32x2_to_e4m3x2_sat(float lo, float hi) {
  lo = fminf(fmaxf(lo, -448.0f), 448.0f);
  hi = fminf(fmaxf(hi, -448.0f), 448.0f);
  unsigned short d;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(d) : "f"(hi), "f"(lo));
  return d;
}

__device__ __forceinline__ uint32_t f32x4_to_e4m3x4_sat(const float (&v)[4], float inv_scale) {
  unsigned short lo = f32x2_to_e4m3x2_sat(__fmul_rn(v[0], inv_scale), __fmul_rn(v[1], inv_scale));
  unsigned short hi = f32x2_to_e4m3x2_sat(__fmul_rn(v[2], inv_scale), __fmul_rn(v[3], inv_scale));
  return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

__device__ __forceinline__ void load8_bf16(const bf16* p, float (&v)[8]) {
  uint4 raw = *reinterpret_cast<const uint4*>(p);
  const bf16* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) v[i] = __bfloat162float(h[i]);
}

__device__ __forceinline__ void store8_bf16(bf16* p, const float (&v)[8]) {
  uint4 raw;
  bf16* h = reinterpret_cast<bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 8; ++i) h[i] = __float2bfloat16_rn(v[i]);
  *reinterpret_cast<uint4*>(p) = raw;
}

__device__ __forceinline__ void load4_bf16(const bf16* p, float (&v)[4]) {
  uint2 raw = *reinterpret_cast<const uint2*>(p);
  const bf16* h = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
  for (int i = 0; i < 4; ++i) v[i] = __bfloat162float(h[i]);
}

__device__ __forceinline__ void store2_bf16(bf16* p, float a, float b) {
  *reinterpret_cast<__nv_bfloat162*>(p) = __floats2bfloat162_rn(a, b);
}

__device__ __forceinline__ uint32_t pack_bf16(bf16 lo, bf16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo))
      | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void mma16816(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
        "r"(b0), "r"(b1),
        "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
}

// Shared Q + R clip + unsheared rel_bias implementation for part >= 1.
template <bool HAS_TAU>
__device__ __forceinline__ void run_qr_parts(
    int row,
    int part,
    int t,
    bool row_valid,
    const bf16* __restrict__ q,
    const bf16* __restrict__ r,
    const bf16* __restrict__ q_l2_scale,
    const bf16* __restrict__ r_l2_scale,
    const bf16* __restrict__ rel_proj,
    const float* __restrict__ tau_buf,
    bf16* __restrict__ out_q,
    bf16* __restrict__ out_rel,
    int stride_q,
    int stride_r) {
  if (part == 1) {
    // 24 Q heads x 8 threads/head = 192 threads.
    const int head = t >> 3;
    const int l = t & 7;
    float q_scale[2][8];
    if (t < kHeads * 8) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        load8_bf16(q_l2_scale + head * kHeadDim + half * 64 + l * 8, q_scale[half]);
      }
    }

    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

    if (t < kHeads * 8) {
      bf16* dst = out_q + (int64_t)row * (kHeads * kHeadDim) + head * kHeadDim;
      if (!row_valid) {
        float z[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
        store8_bf16(dst + l * 8, z);
        store8_bf16(dst + 64 + l * 8, z);
        return;
      }
      float tau_val = 1.0f;
      if constexpr (HAS_TAU) tau_val = tau_buf[row];
      const bf16* src = q + (int64_t)row * stride_q + head * kHeadDim;
      float v[2][8];
      float acc = 0.0f;
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        load8_bf16(src + half * 64 + l * 8, v[half]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          v[half][j] = __fmul_rn(v[half][j], q_scale[half][j]);
          acc = __fmaf_rn(v[half][j], v[half][j], acc);
        }
      }
      float tot_q = lane_sum8(acc);
      float eff_q_eps = kEps;
      float max_q_coeff = 1.0f;
      const bool was_inf = isinf(tot_q);
      if (__builtin_expect(__any_sync(0xffffffffu, was_inf), 0)) {
        float acc_s = 0.0f;
        if (was_inf) {
#pragma unroll
          for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
              v[half][j] = __fmul_rn(v[half][j], 0x1.0p-70f);
              acc_s = __fmaf_rn(v[half][j], v[half][j], acc_s);
            }
          }
        }
        const float tot_q_s = lane_sum8(acc_s);
        if (was_inf) {
          tot_q = tot_q_s;
          eff_q_eps = __fmul_rn(kEps, 0x1.0p-70f);
          max_q_coeff = 0x1.0p+70f;
        }
      }
      float c = clip_coeff(tot_q, kQkClipThresh, eff_q_eps, max_q_coeff);
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        float y[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          float tmp = __fmul_rn(v[half][j], c);
          if constexpr (HAS_TAU) tmp = __fmul_rn(bf_round(tmp), tau_val);
          y[j] = tmp;
        }
        store8_bf16(dst + half * 64 + l * 8, y);
      }
    }
    return;
  }

  // part >= 2: 4 workers, each handles 256 columns of rel_proj [16, 1024].
  constexpr int kColsPerWorker = 256;
  const int worker = part - 2;
  const int col0 = worker * kColsPerWorker;

  __shared__ __align__(16) bf16 r_tile[32][kRelDim];
  __shared__ __align__(16) bf16 proj[kRelDim][kColsPerWorker + 8];
  __shared__ __align__(16) bf16 out_tile[kHeads][kColsPerWorker + 8];

  // Preload rel_proj [16, 256] slice (512 uint4s = 2 iterations for 256 threads).
  for (int i = t; i < kRelDim * (kColsPerWorker / 8); i += kPrologueThreads) {
    const int k = i >> 5;           // / 32
    const int c = (i & 31) << 3;    // * 8
    *reinterpret_cast<uint4*>(&proj[k][c]) =
        *reinterpret_cast<const uint4*>(rel_proj + k * kRelExtent + col0 + c);
  }

  // Preload r_l2_scale [24, 16]: 48 threads (2 threads per head, 8 dims each).
  const int r_head = t >> 1;
  const int r_l = t & 1;
  float r_scale[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (t < kHeads * 2) {
    load8_bf16(r_l2_scale + r_head * kRelDim + r_l * 8, r_scale);
  }
  if (t < 8 * 2) {
    // Zero pad rows 24..31 of r_tile for the second m16n8k16 tile.
    *reinterpret_cast<uint4*>(&r_tile[24 + (t >> 1)][(t & 1) * 8]) = make_uint4(0u, 0u, 0u, 0u);
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (!row_valid) {
    __syncthreads();
    bf16* rel_dst = out_rel + (int64_t)row * (kHeads * kRelExtent) + col0;
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    for (int i = t; i < kHeads * (kColsPerWorker / 8); i += kPrologueThreads) {
      const int h = i >> 5;
      const int c = (i & 31) << 3;
      *reinterpret_cast<uint4*>(rel_dst + h * kRelExtent + c) = z;
    }
    return;
  }

  float rv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  float acc = 0.0f;
  if (t < kHeads * 2) {
    load8_bf16(r + (int64_t)row * stride_r + r_head * kRelDim + r_l * 8, rv);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      rv[j] = __fmul_rn(rv[j], r_scale[j]);
      acc = __fmaf_rn(rv[j], rv[j], acc);
    }
  }
  // All threads in all warps execute lane_sum2 uniformly (no split-warp shfl deadlock).
  float tot_r = lane_sum2(acc);
  float eff_r_eps = kEps;
  float max_r_coeff = 1.0f;
  const bool was_inf = isinf(tot_r);
  if (__builtin_expect(__any_sync(0xffffffffu, was_inf), 0)) {
    float acc_s = 0.0f;
    if (was_inf && t < kHeads * 2) {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        rv[j] = __fmul_rn(rv[j], 0x1.0p-70f);
        acc_s = __fmaf_rn(rv[j], rv[j], acc_s);
      }
    }
    const float tot_r_s = lane_sum2(acc_s);
    if (was_inf) {
      tot_r = tot_r_s;
      eff_r_eps = __fmul_rn(kEps, 0x1.0p-70f);
      max_r_coeff = 0x1.0p+70f;
    }
  }
  float c = clip_coeff(tot_r, kRelClipThresh, eff_r_eps, max_r_coeff);
  if (t < kHeads * 2) {
    float tau_val = 1.0f;
    if constexpr (HAS_TAU) tau_val = tau_buf[row];
    float y[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      float tmp = __fmul_rn(rv[j], c);
      if constexpr (HAS_TAU) tmp = __fmul_rn(bf_round(tmp), tau_val);
      y[j] = tmp;
    }
    store8_bf16(&r_tile[r_head][r_l * 8], y);
  }
  __syncthreads();

  const int warp = t >> 5;
  const int lane = t & 31;
  const int gid = lane >> 2;
  const int tig = lane & 3;

  // Compute 24 x 256 via 2 M-tiles (0..15, 16..23) x 2 128-col steps x 2 N-tiles/warp (8 warps * 8 = 64 cols).
#pragma unroll
  for (int m_tile = 0; m_tile < 2; ++m_tile) {
    const int row_base = m_tile * 16;
    uint32_t a[4];
    a[0] = pack_bf16(r_tile[row_base + gid][tig * 2], r_tile[row_base + gid][tig * 2 + 1]);
    a[1] = pack_bf16(r_tile[row_base + gid + 8][tig * 2], r_tile[row_base + gid + 8][tig * 2 + 1]);
    a[2] = pack_bf16(r_tile[row_base + gid][tig * 2 + 8], r_tile[row_base + gid][tig * 2 + 9]);
    a[3] = pack_bf16(r_tile[row_base + gid + 8][tig * 2 + 8], r_tile[row_base + gid + 8][tig * 2 + 9]);

#pragma unroll
    for (int step = 0; step < 4; ++step) {
      const int n = (step * 8 + warp) * 8;
      const uint32_t b0 = pack_bf16(proj[tig * 2][n + gid], proj[tig * 2 + 1][n + gid]);
      const uint32_t b1 = pack_bf16(proj[tig * 2 + 8][n + gid], proj[tig * 2 + 9][n + gid]);
      float c4[4];
      mma16816(c4, a, b0, b1);
      store2_bf16(&out_tile[row_base + gid][n + tig * 2], c4[0], c4[1]);
      if (m_tile == 0) {
        store2_bf16(&out_tile[row_base + gid + 8][n + tig * 2], c4[2], c4[3]);
      }
    }
  }
  __syncthreads();

  bf16* rel_dst = out_rel + (int64_t)row * (kHeads * kRelExtent) + col0;
  for (int i = t; i < kHeads * (kColsPerWorker / 8); i += kPrologueThreads) {
    const int h = i >> 5;
    const int c = (i & 31) << 3;
    *reinterpret_cast<uint4*>(rel_dst + h * kRelExtent + c) =
        *reinterpret_cast<const uint4*>(&out_tile[h][c]);
  }
}

// Helper for part == 0: K L2 clip + V RMSNorm -> quantized FP8 E4M3 packed 4-byte outputs.
__device__ __forceinline__ void quant_kv_4(
    const bf16* __restrict__ k_row,
    const bf16* __restrict__ v_row,
    int h,
    int lane,
    float inv_ks,
    float inv_vs,
    uint32_t& k_q4,
    uint32_t& v_q4) {
  float kv[4], vv[4];
  load4_bf16(k_row + h * kHeadDim + lane * 4, kv);
  load4_bf16(v_row + h * kHeadDim + lane * 4, vv);

  float sum_k = 0.0f;
  float sum_v = 0.0f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    sum_k = __fmaf_rn(kv[i], kv[i], sum_k);
    sum_v = __fmaf_rn(vv[i], vv[i], sum_v);
  }
  float tot_k = warp_sum32(sum_k);
  float eff_k_eps = kEps;
  float max_k_coeff = 1.0f;
  if (__builtin_expect(isinf(tot_k), 0)) {
    float sum_k_s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      kv[i] = __fmul_rn(kv[i], 0x1.0p-70f);
      sum_k_s = __fmaf_rn(kv[i], kv[i], sum_k_s);
    }
    tot_k = warp_sum32(sum_k_s);
    eff_k_eps = __fmul_rn(kEps, 0x1.0p-70f);
    max_k_coeff = 0x1.0p+70f;
  }

  float tot_v = warp_sum32(sum_v);
  float eff_v_eps = kEps;
  if (__builtin_expect(isinf(tot_v), 0)) {
    float sum_v_s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      vv[i] = __fmul_rn(vv[i], 0x1.0p-70f);
      sum_v_s = __fmaf_rn(vv[i], vv[i], sum_v_s);
    }
    tot_v = warp_sum32(sum_v_s);
    eff_v_eps = 0x1.0p-140f;
  }

  const float ck = clip_coeff(tot_k, kQkClipThresh, eff_k_eps, max_k_coeff);
  const float rstd_v = rsqrt_approx(__fmaf_rn(tot_v, kInvHeadDim, eff_v_eps));

  float k_clipped[4], v_normed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    k_clipped[i] = __fmul_rn(kv[i], ck);
    v_normed[i] = __fmul_rn(vv[i], rstd_v);
  }
  k_q4 = f32x4_to_e4m3x4_sat(k_clipped, inv_ks);
  v_q4 = f32x4_to_e4m3x4_sat(v_normed, inv_vs);
}

// ---------------------------------------------------------------------------
// 1. Full Attention Prologue (Prefill: rows_per_seq=0, Decode/Verify: rows_per_seq=1 or 8)
// Grid: [tokens, 6, 1], Block: [256, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kPrologueThreads) kern_attn_prologue_full_prefill(
    const bf16* __restrict__ q,
    const bf16* __restrict__ r,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_l2_scale,
    const bf16* __restrict__ r_l2_scale,
    const bf16* __restrict__ rel_proj,
    const float* __restrict__ tau_buf,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int64_t* __restrict__ positions,
    const int* __restrict__ pages,
    void* __restrict__ kv_state,
    bf16* __restrict__ out_q,
    bf16* __restrict__ out_rel,
    int stride_q,
    int stride_r,
    int stride_k,
    int stride_v,
    int tokens) {
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int t = threadIdx.x;
  if (row >= tokens) return;

  if (part >= 1) {
    run_qr_parts<true>(
        row, part, t, true, q, r, q_l2_scale, r_l2_scale, rel_proj, tau_buf,
        out_q, out_rel, stride_q, stride_r);
    return;
  }

  const int64_t pos = positions[row];
  const int page = (pos >= 0) ? pages[pos >> 6] : -1;
  const float inv_ks = 1.0f / (*k_scale);
  const float inv_vs = 1.0f / (*v_scale);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (page < 0) return;
  const int h = t >> 5;
  const int lane = t & 31;
  uint32_t k_q4, v_q4;
  quant_kv_4(k + (int64_t)row * stride_k, v + (int64_t)row * stride_v, h, lane, inv_ks, inv_vs, k_q4, v_q4);

  const int slot_in_page = static_cast<int>(pos & 63LL);
  char* k_dst = static_cast<char*>(kv_state)
      + (int64_t)page * kKvPageBytes
      + (int64_t)h * kKvHeadPageBytes
      + (int64_t)slot_in_page * kHeadDim
      + lane * 4;
  *reinterpret_cast<uint32_t*>(k_dst) = k_q4;
  *reinterpret_cast<uint32_t*>(k_dst + kKvHalfPageBytes) = v_q4;
}

extern "C" __global__ void __launch_bounds__(kPrologueThreads) kern_attn_prologue_full_decode(
    const bf16* __restrict__ q,
    const bf16* __restrict__ r,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_l2_scale,
    const bf16* __restrict__ r_l2_scale,
    const bf16* __restrict__ rel_proj,
    const float* __restrict__ tau_buf,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int64_t* __restrict__ positions,
    const int* __restrict__ valid,
    const int* __restrict__ pages,
    void* __restrict__ kv_state,
    bf16* __restrict__ out_q,
    bf16* __restrict__ out_rel,
    int stride_q,
    int stride_r,
    int stride_k,
    int stride_v,
    int rows_per_seq,
    int page_columns) {
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int t = threadIdx.x;
  const bool row_valid = (valid[row] != 0);

  if (part >= 1) {
    run_qr_parts<true>(
        row, part, t, row_valid, q, r, q_l2_scale, r_l2_scale, rel_proj, tau_buf,
        out_q, out_rel, stride_q, stride_r);
    return;
  }

  const int64_t pos = positions[row];
  const int seq = row / rows_per_seq;
  const int64_t page_idx = pos >> 6;
  const int page = (row_valid && pos >= 0 && page_idx < page_columns) ? pages[(int64_t)seq * page_columns + page_idx] : -1;
  const float inv_ks = 1.0f / (*k_scale);
  const float inv_vs = 1.0f / (*v_scale);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (!row_valid || page < 0) return;
  const int h = t >> 5;
  const int lane = t & 31;
  uint32_t k_q4, v_q4;
  quant_kv_4(k + (int64_t)row * stride_k, v + (int64_t)row * stride_v, h, lane, inv_ks, inv_vs, k_q4, v_q4);

  const int slot_in_page = static_cast<int>(pos & 63LL);
  char* k_dst = static_cast<char*>(kv_state)
      + (int64_t)page * kKvPageBytes
      + (int64_t)h * kKvHeadPageBytes
      + (int64_t)slot_in_page * kHeadDim
      + lane * 4;
  *reinterpret_cast<uint32_t*>(k_dst) = k_q4;
  *reinterpret_cast<uint32_t*>(k_dst + kKvHalfPageBytes) = v_q4;
}

// ---------------------------------------------------------------------------
// 2. SWA Attention Prologue (no tau!)
// - Prefill: writes FP8 K/V to swa_k / swa_v [tokens, 8, 128]
// - Decode: writes FP8 K/V directly to states.seq_state SWA ring (pos % 1024)
// - Verify: writes FP8 K/V to verify_swa_k / verify_swa_v [seqs*8, 8, 128] slice
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kPrologueThreads) kern_attn_prologue_swa_prefill(
    const bf16* __restrict__ q,
    const bf16* __restrict__ r,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_l2_scale,
    const bf16* __restrict__ r_l2_scale,
    const bf16* __restrict__ rel_proj,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    bf16* __restrict__ out_q,
    bf16* __restrict__ out_rel,
    uint8_t* __restrict__ out_swa_k,
    uint8_t* __restrict__ out_swa_v,
    int stride_q,
    int stride_r,
    int stride_k,
    int stride_v,
    int tokens) {
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int t = threadIdx.x;
  if (row >= tokens) return;

  if (part >= 1) {
    run_qr_parts<false>(
        row, part, t, true, q, r, q_l2_scale, r_l2_scale, rel_proj, nullptr,
        out_q, out_rel, stride_q, stride_r);
    return;
  }

  const float inv_ks = 1.0f / (*k_scale);
  const float inv_vs = 1.0f / (*v_scale);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const int h = t >> 5;
  const int lane = t & 31;
  uint32_t k_q4, v_q4;
  quant_kv_4(k + (int64_t)row * stride_k, v + (int64_t)row * stride_v, h, lane, inv_ks, inv_vs, k_q4, v_q4);

  const int64_t off = (int64_t)row * (kKvHeads * kHeadDim) + h * kHeadDim + lane * 4;
  *reinterpret_cast<uint32_t*>(out_swa_k + off) = k_q4;
  *reinterpret_cast<uint32_t*>(out_swa_v + off) = v_q4;
}

extern "C" __global__ void __launch_bounds__(kPrologueThreads) kern_attn_prologue_swa_decode(
    const bf16* __restrict__ q,
    const bf16* __restrict__ r,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_l2_scale,
    const bf16* __restrict__ r_l2_scale,
    const bf16* __restrict__ rel_proj,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int64_t* __restrict__ positions,
    const int* __restrict__ valid,
    const int* __restrict__ lines,
    void* __restrict__ seq_state,
    bf16* __restrict__ out_q,
    bf16* __restrict__ out_rel,
    int stride_q,
    int stride_r,
    int stride_k,
    int stride_v,
    int64_t swa_visit_off) {
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int t = threadIdx.x;
  const int line = lines[row];
  const bool row_valid = (valid[row] != 0) && (line > 0);

  if (part >= 1) {
    run_qr_parts<false>(
        row, part, t, row_valid, q, r, q_l2_scale, r_l2_scale, rel_proj, nullptr,
        out_q, out_rel, stride_q, stride_r);
    return;
  }

  const int64_t pos = positions[row];
  const float inv_ks = 1.0f / (*k_scale);
  const float inv_vs = 1.0f / (*v_scale);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (!row_valid || pos < 0) return;
  const int h = t >> 5;
  const int lane = t & 31;
  uint32_t k_q4, v_q4;
  quant_kv_4(k + (int64_t)row * stride_k, v + (int64_t)row * stride_v, h, lane, inv_ks, inv_vs, k_q4, v_q4);

  const int ring_pos = static_cast<int>(pos & 1023LL);
  char* k_dst = static_cast<char*>(seq_state)
      + (int64_t)line * kSeqStateBytes
      + swa_visit_off
      + (int64_t)h * kSwaHeadRingBytes
      + (int64_t)ring_pos * kHeadDim
      + lane * 4;
  *reinterpret_cast<uint32_t*>(k_dst) = k_q4;
  *reinterpret_cast<uint32_t*>(k_dst + kSwaHalfVisitBytes) = v_q4;
}

extern "C" __global__ void __launch_bounds__(kPrologueThreads) kern_attn_prologue_swa_verify(
    const bf16* __restrict__ q,
    const bf16* __restrict__ r,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_l2_scale,
    const bf16* __restrict__ r_l2_scale,
    const bf16* __restrict__ rel_proj,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int* __restrict__ valid,
    const int* __restrict__ lines,
    bf16* __restrict__ out_q,
    bf16* __restrict__ out_rel,
    uint8_t* __restrict__ out_swa_k,
    uint8_t* __restrict__ out_swa_v,
    int stride_q,
    int stride_r,
    int stride_k,
    int stride_v) {
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int t = threadIdx.x;
  const int seq = row >> 3;
  const int line = lines[seq];
  const bool row_valid = (valid[row] != 0) && (line > 0);

  if (part >= 1) {
    run_qr_parts<false>(
        row, part, t, row_valid, q, r, q_l2_scale, r_l2_scale, rel_proj, nullptr,
        out_q, out_rel, stride_q, stride_r);
    return;
  }

  const float inv_ks = 1.0f / (*k_scale);
  const float inv_vs = 1.0f / (*v_scale);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const int h = t >> 5;
  const int lane = t & 31;
  const int64_t off = (int64_t)row * (kKvHeads * kHeadDim) + h * kHeadDim + lane * 4;
  if (!row_valid) {
    *reinterpret_cast<uint32_t*>(out_swa_k + off) = 0u;
    *reinterpret_cast<uint32_t*>(out_swa_v + off) = 0u;
    return;
  }
  uint32_t k_q4, v_q4;
  quant_kv_4(k + (int64_t)row * stride_k, v + (int64_t)row * stride_v, h, lane, inv_ks, inv_vs, k_q4, v_q4);
  *reinterpret_cast<uint32_t*>(out_swa_k + off) = k_q4;
  *reinterpret_cast<uint32_t*>(out_swa_v + off) = v_q4;
}

// ---------------------------------------------------------------------------
// 3. SWA Ring Commits (Prefill & Speculative Round)
// ---------------------------------------------------------------------------
// Grid: [tokens, 8, 1], Block: [32, 1, 1]
extern "C" __global__ void __launch_bounds__(32) kern_swa_ring_commit_prefill(
    const uint8_t* __restrict__ swa_k,
    const uint8_t* __restrict__ swa_v,
    const int64_t* __restrict__ positions,
    const int* __restrict__ lines,
    void* __restrict__ seq_state,
    int tokens,
    int64_t swa_visit_off) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int lane = threadIdx.x;
  if (t >= tokens || t < tokens - 1024) return;
  const int line = lines[0];
  if (line <= 0) return;
  const int64_t pos = positions[t];
  if (pos < 0) return;

  const int ring_pos = static_cast<int>(pos & 1023LL);
  const int64_t src_off = (int64_t)t * (kKvHeads * kHeadDim) + h * kHeadDim + lane * 4;
  char* k_dst = static_cast<char*>(seq_state)
      + (int64_t)line * kSeqStateBytes
      + swa_visit_off
      + (int64_t)h * kSwaHeadRingBytes
      + (int64_t)ring_pos * kHeadDim
      + lane * 4;

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  *reinterpret_cast<uint32_t*>(k_dst) = *reinterpret_cast<const uint32_t*>(swa_k + src_off);
  *reinterpret_cast<uint32_t*>(k_dst + kSwaHalfVisitBytes) = *reinterpret_cast<const uint32_t*>(swa_v + src_off);
}

// Single launch commits all 40 target SWA visits for accepted[seq] rows in round.
// Grid: [seqs, 40, 1], Block: [256, 1, 1]
extern "C" __global__ void __launch_bounds__(256) kern_spec_swa_commit_all(
    const uint8_t* __restrict__ verify_swa_k,
    const uint8_t* __restrict__ verify_swa_v,
    const int64_t* __restrict__ positions,
    const int* __restrict__ lines,
    const int* __restrict__ accepted,
    const int* __restrict__ valid,
    void* __restrict__ seq_state,
    int max_seqs) {
  const int seq = blockIdx.x;
  const int swa_idx = blockIdx.y;
  const int h = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const int line = lines[seq];
  int nacc = accepted[seq];
  const int vflag = valid[seq << 3];
  if (line <= 0 || nacc <= 0 || vflag == 0) return;
  if (nacc > 8) nacc = 8;

  char* visit_base = static_cast<char*>(seq_state)
      + (int64_t)line * kSeqStateBytes
      + kSwaTargetOffset
      + (int64_t)swa_idx * kSwaVisitBytes
      + (int64_t)h * kSwaHeadRingBytes
      + lane * 4;
  const int64_t slice_base = (int64_t)swa_idx * max_seqs * 8 * (kKvHeads * kHeadDim);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  for (int j = 0; j < nacc; ++j) {
    const int row = (seq << 3) + j;
    const int64_t pos = positions[row];
    if (pos < 0) continue;
    const int ring_pos = static_cast<int>(pos & 1023LL);
    const int64_t src_off = slice_base + (int64_t)row * (kKvHeads * kHeadDim) + h * kHeadDim + lane * 4;
    const int64_t dst_off = (int64_t)ring_pos * kHeadDim;
    *reinterpret_cast<uint32_t*>(visit_base + dst_off) =
        *reinterpret_cast<const uint32_t*>(verify_swa_k + src_off);
    *reinterpret_cast<uint32_t*>(visit_base + kSwaHalfVisitBytes + dst_off) =
        *reinterpret_cast<const uint32_t*>(verify_swa_v + src_off);
  }
}

// ---------------------------------------------------------------------------
// 4. Standalone Gate + Sink Kernel (optional fallback for MODE_OUT_LSE)
// Grid: [tokens, 1, 1], Block: [768, 1, 1] (24 * 128 / 4 = 768)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float sigmoid_approx(float x) {
  float e, r;
  float t = __fmul_rn(-x, kLog2E);
  asm("ex2.approx.f32 %0, %1;" : "=f"(e) : "f"(t));
  asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(__fadd_rn(e, 1.0f)));
  return r;
}

extern "C" __global__ void __launch_bounds__(768) kern_gate_sink_bf16(
    const float* __restrict__ attn_out,
    const bf16* __restrict__ gate,
    const float* __restrict__ lse,
    const bf16* __restrict__ sink,
    bf16* __restrict__ y,
    int stride_gate,
    int stride_sink,
    int lse_is_log2) {
  const int row = blockIdx.x;
  const int col = threadIdx.x * 4;
  const int head = col >> 7;
  const int d = col & 127;

  const uint2 g_raw = *reinterpret_cast<const uint2*>(gate + (int64_t)row * stride_gate + col);
  const float s = __bfloat162float(sink[(int64_t)row * stride_sink + head]);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  const float* xr = attn_out + (int64_t)row * (kHeads * kHeadDim) + col;
  float lse_nat = lse[(int64_t)row * kHeads + head];
  if (lse_is_log2) lse_nat = __fmul_rn(lse_nat, kLn2);
  const float scale = sigmoid_approx(__fsub_rn(lse_nat, s));
  const bf16* gh = reinterpret_cast<const bf16*>(&g_raw);
  alignas(8) bf16 out[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float g = sigmoid_approx(__bfloat162float(gh[i]));
    out[i] = __float2bfloat16_rn(__fmul_rn(__fmul_rn(xr[i], g), scale));
  }
  *reinterpret_cast<uint2*>(y + (int64_t)row * (kHeads * kHeadDim) + col) =
      *reinterpret_cast<const uint2*>(out);
  (void)d;
}

// ---------------------------------------------------------------------------
// 5. Derive Scalar Fill (used by derive program for k_scale / v_scale carry)
// Grid: [ceil_div(n, 32), 1, 1] or [1, 1, 1], Block: [1, 1, 1] or [32, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void kern_fill_f32(float* __restrict__ p, int n, float v) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}
