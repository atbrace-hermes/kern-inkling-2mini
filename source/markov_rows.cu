// SPDX-License-Identifier: Apache-2.0
// Adapted from kern/tools/kernels-src/markov_rows.cu for SM120.
// Includes SM120 Tensor-Core fused Markov fold_proj + fold_bias + base_logits + softcap + vocab mask + argmax
// with L2 eviction policy pinning (evict_last on first ~87 MiB of fold_weight, evict_first on tail & logits).
#include <cuda_bf16.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>

struct Pair {
  float v;
  int i;
};

__device__ static inline Pair pick(Pair a, Pair b) {
  return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

template <int BLOCK>
__device__ static inline Pair block_reduce(Pair best) {
  __shared__ float sv[BLOCK];
  __shared__ int si[BLOCK];
  sv[threadIdx.x] = best.v;
  si[threadIdx.x] = best.i;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      Pair o = {sv[threadIdx.x + stride], si[threadIdx.x + stride]};
      Pair m = pick({sv[threadIdx.x], si[threadIdx.x]}, o);
      sv[threadIdx.x] = m.v;
      si[threadIdx.x] = m.i;
    }
    __syncthreads();
  }
  return {sv[0], si[0]};
}

__device__ static __forceinline__ float tanh_approx_f32(float x) {
  float y;
  asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ static __forceinline__ void mma_m16n8k16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ static __forceinline__ uint64_t make_policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ static __forceinline__ uint64_t make_policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}

__device__ static __forceinline__ uint4 ld_u4_policy(const uint4* ptr, uint64_t policy) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0, %1, %2, %3}, [%4], %5;"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr), "l"(policy));
  return v;
}

__device__ static __forceinline__ float ld_bf16_evict_first(const __nv_bfloat16* ptr, uint64_t policy) {
  unsigned short raw;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.u16 %0, [%1], %2;"
               : "=h"(raw)
               : "l"(ptr), "l"(policy));
  __nv_bfloat16 b = *reinterpret_cast<const __nv_bfloat16*>(&raw);
  return __bfloat162float(b);
}

// ============================================================================
// Tensor-Core Fused Markov fold_proj + add_softcap_mask + argmax_partial (K = 256)
// Grid: [220, 1, 1] (2 CTAs/SM on 110 SMs), Block: [256, 1, 1] (8 warps)
// Each warp processes 16 vocab rows x up to 8 seqs in 16 mma.m16n8k16 instructions
// (zero warp shuffles in hot loop, latent [8, 256] held entirely in 32 B-fragment regs!).
// Pins first 196 CTAs (~87.0 MiB) of fold_weight in SM120 96 MiB L2 via evict_last;
// streams remaining ~10.7 MiB + fold_bias + base_logits via evict_first.
// ============================================================================

constexpr int kMarkovPinCtas = 196;  // 196/220 * 97.68 MiB = 87.0 MiB (< 96 MiB L2)
constexpr int kMarkovWarps = 8;
constexpr int kSmemStride = 72;      // 64 + 8 bf16 padding to eliminate bank conflicts

__device__ static __forceinline__ void markov_tc_chunk8(
    const __nv_bfloat16* __restrict__ latent,       // [n_s, 256]
    const __nv_bfloat16* __restrict__ fold_weight,  // [vocab_padded, 256]
    const __nv_bfloat16* __restrict__ fold_bias,    // [vocab_padded]
    const __nv_bfloat16* __restrict__ base_logits,  // [n_s, num_steps, vocab_padded]
    float* __restrict__ pmax,                       // [n_s, gridDim.x]
    int* __restrict__ pidx,                         // [n_s, gridDim.x]
    int n_s,                                        // 1..8
    int step,
    float softcap,
    int vocab_padded,
    int vocab_unpadded,
    int num_steps,
    __nv_bfloat16 s_w[kMarkovWarps][16][kSmemStride],
    float s_pv[kMarkovWarps][8],
    int s_pi[kMarkovWarps][8]) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7
  const int tid4 = lane & 3;   // 0..3
  const int cta = blockIdx.x;
  const int n_ctas = gridDim.x;

  const uint64_t pol_first = make_policy_evict_first();
  const uint64_t w_policy = (cta < kMarkovPinCtas) ? make_policy_evict_last() : pol_first;

  // Load entire latent[0..n_s-1, 0..255] directly into 32 B-fragment registers (0 spills!).
  uint32_t lb0[16], lb1[16];
  if (gid < n_s) {
    const __nv_bfloat16* lrow = latent + static_cast<long long>(gid) * 256;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      int kc = ks * 16;
      lb0[ks] = *reinterpret_cast<const uint32_t*>(lrow + kc + 2 * tid4);
      lb1[ks] = *reinterpret_cast<const uint32_t*>(lrow + kc + 8 + 2 * tid4);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      lb0[ks] = 0u;
      lb1[ks] = 0u;
    }
  }

  const int total_tiles = (vocab_unpadded + 15) >> 4;  // 16 vocab rows per tile
  const int t_lo = (cta * total_tiles) / n_ctas;
  const int t_hi = ((cta + 1) * total_tiles) / n_ctas;
  const float inv_softcap = 1.0f / softcap;

  const int s0 = 2 * tid4;      // 0, 2, 4, 6
  const int s1 = 2 * tid4 + 1;  // 1, 3, 5, 7
  const bool has_s0 = (s0 < n_s);
  const bool has_s1 = (s1 < n_s);
  const __nv_bfloat16* base_s0 =
      has_s0 ? (base_logits + (static_cast<long long>(s0) * num_steps + step) * vocab_padded) : nullptr;
  const __nv_bfloat16* base_s1 =
      has_s1 ? (base_logits + (static_cast<long long>(s1) * num_steps + step) * vocab_padded) : nullptr;

  Pair best0 = {-__builtin_inff(), INT_MAX};
  Pair best1 = {-__builtin_inff(), INT_MAX};

  // Lane load coordinates for 16 x 64 bf16 tile (8 lanes x 8 bf16 = 64 bf16 = 128B per row).
  const int lr = lane >> 3;        // 0..3
  const int lc = (lane & 7) * 8;   // 0, 8, ..., 56

  auto eval_and_pick = [&](Pair& best, float dot, float fb, float base, int v) {
    float sb = __bfloat162float(__float2bfloat16_rn(dot));
    float bias_bf = __bfloat162float(__float2bfloat16_rn(sb + fb));
    float x = __bfloat162float(__float2bfloat16_rn(base + bias_bf));
    float y = __bfloat162float(__float2bfloat16_rn(softcap * tanh_approx_f32(x * inv_softcap)));
    if (isnan(y)) y = -__builtin_inff();
    best = pick(best, {y, v});
  };

  for (int tile = t_lo + warp_id; tile < t_hi; tile += kMarkovWarps) {
    const int v0 = tile * 16;
    const bool full_tile = (v0 + 16 <= vocab_unpadded);

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

    // 4 K-chunks of 64 (total K = 256).
#pragma unroll
    for (int kc64 = 0; kc64 < 4; ++kc64) {
      const int k_base = kc64 * 64;
      // 4 coalesced 128-bit LDG.E.NA.128 loads per lane (total 16 x 64 bf16 = 2 KB).
      if (full_tile) {
#pragma unroll
        for (int rg = 0; rg < 4; ++rg) {
          int r = rg * 4 + lr;
          const uint4* wptr = reinterpret_cast<const uint4*>(
              fold_weight + static_cast<long long>(v0 + r) * 256 + k_base + lc);
          *reinterpret_cast<uint4*>(&s_w[warp_id][r][lc]) = ld_u4_policy(wptr, w_policy);
        }
      } else {
#pragma unroll
        for (int rg = 0; rg < 4; ++rg) {
          int r = rg * 4 + lr;
          if (v0 + r < vocab_unpadded) {
            const uint4* wptr = reinterpret_cast<const uint4*>(
                fold_weight + static_cast<long long>(v0 + r) * 256 + k_base + lc);
            *reinterpret_cast<uint4*>(&s_w[warp_id][r][lc]) = ld_u4_policy(wptr, w_policy);
          } else {
            *reinterpret_cast<uint4*>(&s_w[warp_id][r][lc]) = make_uint4(0u, 0u, 0u, 0u);
          }
        }
      }
      __syncwarp();

      // 4 Tensor Core m16n8k16 ops (zero bank conflicts via kSmemStride = 72).
#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        int lk = ks * 16;
        int gk = kc64 * 4 + ks;
        uint32_t a0 = *reinterpret_cast<const uint32_t*>(&s_w[warp_id][gid][lk + 2 * tid4]);
        uint32_t a1 = *reinterpret_cast<const uint32_t*>(&s_w[warp_id][gid + 8][lk + 2 * tid4]);
        uint32_t a2 = *reinterpret_cast<const uint32_t*>(&s_w[warp_id][gid][lk + 8 + 2 * tid4]);
        uint32_t a3 = *reinterpret_cast<const uint32_t*>(&s_w[warp_id][gid + 8][lk + 8 + 2 * tid4]);
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, lb0[gk], lb1[gk]);
      }
      __syncwarp();
    }

    // Fused epilogue directly from registers (no s_dot smem!).
    const int va = v0 + gid;
    const int vb = v0 + gid + 8;
    if (va < vocab_unpadded) {
      float fba = ld_bf16_evict_first(fold_bias + va, pol_first);
      if (has_s0) eval_and_pick(best0, d0, fba, ld_bf16_evict_first(base_s0 + va, pol_first), va);
      if (has_s1) eval_and_pick(best1, d1, fba, ld_bf16_evict_first(base_s1 + va, pol_first), va);
    }
    if (vb < vocab_unpadded) {
      float fbb = ld_bf16_evict_first(fold_bias + vb, pol_first);
      if (has_s0) eval_and_pick(best0, d2, fbb, ld_bf16_evict_first(base_s0 + vb, pol_first), vb);
      if (has_s1) eval_and_pick(best1, d3, fbb, ld_bf16_evict_first(base_s1 + vb, pol_first), vb);
    }
  }

  // Reduce best0 (seq 2*tid4) and best1 (seq 2*tid4+1) across the 8 gid lanes sharing tid4 (offsets 16, 8, 4).
#pragma unroll
  for (int off = 16; off >= 4; off >>= 1) {
    float ov0 = __shfl_down_sync(0xffffffffu, best0.v, off);
    int oi0 = __shfl_down_sync(0xffffffffu, best0.i, off);
    best0 = pick(best0, {ov0, oi0});
    float ov1 = __shfl_down_sync(0xffffffffu, best1.v, off);
    int oi1 = __shfl_down_sync(0xffffffffu, best1.i, off);
    best1 = pick(best1, {ov1, oi1});
  }
  if (gid == 0) {
    if (has_s0) {
      s_pv[warp_id][s0] = best0.v;
      s_pi[warp_id][s0] = best0.i;
    }
    if (has_s1) {
      s_pv[warp_id][s1] = best1.v;
      s_pi[warp_id][s1] = best1.i;
    }
  }
  __syncthreads();

  // Warp 0 reduces across the 8 warps for each seq 0..n_s-1.
  if (warp_id == 0) {
    for (int s = 0; s < n_s; ++s) {
      Pair p = (lane < kMarkovWarps) ? Pair{s_pv[lane][s], s_pi[lane][s]}
                                     : Pair{-__builtin_inff(), INT_MAX};
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, p.v, off);
        int oi = __shfl_down_sync(0xffffffffu, p.i, off);
        p = pick(p, {ov, oi});
      }
      if (lane == 0) {
        pmax[static_cast<long long>(s) * n_ctas + cta] = p.v;
        pidx[static_cast<long long>(s) * n_ctas + cta] = p.i;
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(256, 2) void kern_markov_fold_softcap_argmax_partial(
    const __nv_bfloat16* __restrict__ latent,
    const __nv_bfloat16* __restrict__ fold_weight,
    const __nv_bfloat16* __restrict__ fold_bias,
    const __nv_bfloat16* __restrict__ base_logits,
    float* __restrict__ pmax,
    int* __restrict__ pidx,
    int seqs,
    int step,
    float softcap,
    int vocab_padded,
    int vocab_unpadded,
    int num_steps) {
  __shared__ __align__(16) __nv_bfloat16 s_w[kMarkovWarps][16][kSmemStride];  // 18,432 B
  __shared__ float s_pv[kMarkovWarps][8];
  __shared__ int s_pi[kMarkovWarps][8];

  for (int s0 = 0; s0 < seqs; s0 += 8) {
    int n_s = min(8, seqs - s0);
    markov_tc_chunk8(
        latent + static_cast<long long>(s0) * 256,
        fold_weight,
        fold_bias,
        base_logits + static_cast<long long>(s0) * num_steps * vocab_padded,
        pmax + static_cast<long long>(s0) * gridDim.x,
        pidx + static_cast<long long>(s0) * gridDim.x,
        n_s, step, softcap, vocab_padded, vocab_unpadded, num_steps, s_w, s_pv, s_pi);
    if (s0 + 8 < seqs) __syncthreads();
  }
  asm volatile("griddepcontrol.launch_dependents;\n" ::);
}

// ============================================================================
// FP8 E4M3 per-row Tensor-Core Fused Markov fold_proj + add_softcap_mask + argmax_partial
// fold_weight_fp8 [vocab_padded, 256] is 49.08 MiB -> 100% resident in SM120 96 MiB L2!
// Converts FP8 -> BF16 via 1-cycle `cvt.rn.bf16x2.e4m3x2` and scales by fold_scale[v].
// ============================================================================
constexpr int kSmemStrideFp8 = 80;  // 64 + 16 B padding: 16B aligned & zero bank conflicts

__device__ static __forceinline__ uint32_t cvt_e4m3x2_to_bf16x2_mk(uint16_t raw) {
  uint32_t res;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(res) : "h"(raw));
  return res;
}

__device__ static __forceinline__ float ld_f32_evict_last(const float* ptr, uint64_t policy) {
  float v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::cache_hint.f32 %0, [%1], %2;"
               : "=f"(v)
               : "l"(ptr), "l"(policy));
  return v;
}

__device__ static __forceinline__ void markov_tc_chunk8_fp8(
    const __nv_bfloat16* __restrict__ latent,         // [n_s, 256]
    const uint8_t* __restrict__ fold_weight_fp8,      // [vocab_padded, 256]
    const float* __restrict__ fold_scale,             // [vocab_padded]
    const __nv_bfloat16* __restrict__ fold_bias,      // [vocab_padded]
    const __nv_bfloat16* __restrict__ base_logits,    // [n_s, num_steps, vocab_padded]
    float* __restrict__ pmax,                         // [n_s, gridDim.x]
    int* __restrict__ pidx,                           // [n_s, gridDim.x]
    int n_s,
    int step,
    float softcap,
    int vocab_padded,
    int vocab_unpadded,
    int num_steps,
    uint8_t s_w[kMarkovWarps][16][kSmemStrideFp8],
    float s_pv[kMarkovWarps][8],
    int s_pi[kMarkovWarps][8]) {
  const int tid = threadIdx.x;
  const int warp_id = tid >> 5;
  const int lane = tid & 31;
  const int gid = lane >> 2;   // 0..7
  const int tid4 = lane & 3;   // 0..3
  const int cta = blockIdx.x;
  const int n_ctas = gridDim.x;

  const uint64_t pol_first = make_policy_evict_first();
  const uint64_t pol_last = make_policy_evict_last();

  uint32_t lb0[16], lb1[16];
  if (gid < n_s) {
    const __nv_bfloat16* lrow = latent + static_cast<long long>(gid) * 256;
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      int kc = ks * 16;
      lb0[ks] = *reinterpret_cast<const uint32_t*>(lrow + kc + 2 * tid4);
      lb1[ks] = *reinterpret_cast<const uint32_t*>(lrow + kc + 8 + 2 * tid4);
    }
  } else {
#pragma unroll
    for (int ks = 0; ks < 16; ++ks) {
      lb0[ks] = 0u;
      lb1[ks] = 0u;
    }
  }

  const int total_tiles = (vocab_unpadded + 15) >> 4;
  const int t_lo = (cta * total_tiles) / n_ctas;
  const int t_hi = ((cta + 1) * total_tiles) / n_ctas;
  const float inv_softcap = 1.0f / softcap;

  const int s0 = 2 * tid4;
  const int s1 = 2 * tid4 + 1;
  const bool has_s0 = (s0 < n_s);
  const bool has_s1 = (s1 < n_s);
  const __nv_bfloat16* base_s0 =
      has_s0 ? (base_logits + (static_cast<long long>(s0) * num_steps + step) * vocab_padded) : nullptr;
  const __nv_bfloat16* base_s1 =
      has_s1 ? (base_logits + (static_cast<long long>(s1) * num_steps + step) * vocab_padded) : nullptr;

  Pair best0 = {-__builtin_inff(), INT_MAX};
  Pair best1 = {-__builtin_inff(), INT_MAX};

  // Lane load coordinates for 16 x 64 FP8 tile (4 lanes x 16B = 64B per row -> 2 loads/lane).
  const int lr = lane >> 2;        // 0..7
  const int lc = (lane & 3) * 16;  // 0, 16, 32, 48

  auto eval_and_pick = [&](Pair& best, float dot, float fb, float base, int v) {
    float sb = __bfloat162float(__float2bfloat16_rn(dot));
    float bias_bf = __bfloat162float(__float2bfloat16_rn(sb + fb));
    float x = __bfloat162float(__float2bfloat16_rn(base + bias_bf));
    float y = __bfloat162float(__float2bfloat16_rn(softcap * tanh_approx_f32(x * inv_softcap)));
    if (isnan(y)) y = -__builtin_inff();
    best = pick(best, {y, v});
  };

  for (int tile = t_lo + warp_id; tile < t_hi; tile += kMarkovWarps) {
    const int v0 = tile * 16;
    const bool full_tile = (v0 + 16 <= vocab_unpadded);

    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;

#pragma unroll
    for (int kc64 = 0; kc64 < 4; ++kc64) {
      const int k_base = kc64 * 64;
      if (full_tile) {
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
          int r = rg * 8 + lr;
          const uint4* wptr = reinterpret_cast<const uint4*>(
              fold_weight_fp8 + static_cast<long long>(v0 + r) * 256 + k_base + lc);
          *reinterpret_cast<uint4*>(&s_w[warp_id][r][lc]) = ld_u4_policy(wptr, pol_last);
        }
      } else {
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
          int r = rg * 8 + lr;
          if (v0 + r < vocab_unpadded) {
            const uint4* wptr = reinterpret_cast<const uint4*>(
                fold_weight_fp8 + static_cast<long long>(v0 + r) * 256 + k_base + lc);
            *reinterpret_cast<uint4*>(&s_w[warp_id][r][lc]) = ld_u4_policy(wptr, pol_last);
          } else {
            *reinterpret_cast<uint4*>(&s_w[warp_id][r][lc]) = make_uint4(0u, 0u, 0u, 0u);
          }
        }
      }
      __syncwarp();

#pragma unroll
      for (int ks = 0; ks < 4; ++ks) {
        int lk = ks * 16;
        int gk = kc64 * 4 + ks;
        uint32_t a0 = cvt_e4m3x2_to_bf16x2_mk(*reinterpret_cast<const uint16_t*>(&s_w[warp_id][gid][lk + 2 * tid4]));
        uint32_t a1 = cvt_e4m3x2_to_bf16x2_mk(*reinterpret_cast<const uint16_t*>(&s_w[warp_id][gid + 8][lk + 2 * tid4]));
        uint32_t a2 = cvt_e4m3x2_to_bf16x2_mk(*reinterpret_cast<const uint16_t*>(&s_w[warp_id][gid][lk + 8 + 2 * tid4]));
        uint32_t a3 = cvt_e4m3x2_to_bf16x2_mk(*reinterpret_cast<const uint16_t*>(&s_w[warp_id][gid + 8][lk + 8 + 2 * tid4]));
        mma_m16n8k16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, lb0[gk], lb1[gk]);
      }
      __syncwarp();
    }

    const int va = v0 + gid;
    const int vb = v0 + gid + 8;
    if (va < vocab_unpadded) {
      float sa = ld_f32_evict_last(fold_scale + va, pol_last);
      float fba = ld_bf16_evict_first(fold_bias + va, pol_first);
      if (has_s0) eval_and_pick(best0, d0 * sa, fba, ld_bf16_evict_first(base_s0 + va, pol_first), va);
      if (has_s1) eval_and_pick(best1, d1 * sa, fba, ld_bf16_evict_first(base_s1 + va, pol_first), va);
    }
    if (vb < vocab_unpadded) {
      float sb = ld_f32_evict_last(fold_scale + vb, pol_last);
      float fbb = ld_bf16_evict_first(fold_bias + vb, pol_first);
      if (has_s0) eval_and_pick(best0, d2 * sb, fbb, ld_bf16_evict_first(base_s0 + vb, pol_first), vb);
      if (has_s1) eval_and_pick(best1, d3 * sb, fbb, ld_bf16_evict_first(base_s1 + vb, pol_first), vb);
    }
  }

#pragma unroll
  for (int off = 16; off >= 4; off >>= 1) {
    float ov0 = __shfl_down_sync(0xffffffffu, best0.v, off);
    int oi0 = __shfl_down_sync(0xffffffffu, best0.i, off);
    best0 = pick(best0, {ov0, oi0});
    float ov1 = __shfl_down_sync(0xffffffffu, best1.v, off);
    int oi1 = __shfl_down_sync(0xffffffffu, best1.i, off);
    best1 = pick(best1, {ov1, oi1});
  }
  if (gid == 0) {
    if (has_s0) {
      s_pv[warp_id][s0] = best0.v;
      s_pi[warp_id][s0] = best0.i;
    }
    if (has_s1) {
      s_pv[warp_id][s1] = best1.v;
      s_pi[warp_id][s1] = best1.i;
    }
  }
  __syncthreads();

  if (warp_id == 0) {
    for (int s = 0; s < n_s; ++s) {
      Pair p = (lane < kMarkovWarps) ? Pair{s_pv[lane][s], s_pi[lane][s]}
                                     : Pair{-__builtin_inff(), INT_MAX};
#pragma unroll
      for (int off = 4; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, p.v, off);
        int oi = __shfl_down_sync(0xffffffffu, p.i, off);
        p = pick(p, {ov, oi});
      }
      if (lane == 0) {
        pmax[static_cast<long long>(s) * n_ctas + cta] = p.v;
        pidx[static_cast<long long>(s) * n_ctas + cta] = p.i;
      }
    }
  }
}

extern "C" __global__ __launch_bounds__(256, 2) void kern_markov_fold_fp8_softcap_argmax_partial(
    const __nv_bfloat16* __restrict__ latent,
    const uint8_t* __restrict__ fold_weight_fp8,
    const float* __restrict__ fold_scale,
    const __nv_bfloat16* __restrict__ fold_bias,
    const __nv_bfloat16* __restrict__ base_logits,
    float* __restrict__ pmax,
    int* __restrict__ pidx,
    int seqs,
    int step,
    float softcap,
    int vocab_padded,
    int vocab_unpadded,
    int num_steps) {
  __shared__ __align__(16) uint8_t s_w[kMarkovWarps][16][kSmemStrideFp8];  // 10,240 B
  __shared__ float s_pv[kMarkovWarps][8];
  __shared__ int s_pi[kMarkovWarps][8];

  for (int s0 = 0; s0 < seqs; s0 += 8) {
    int n_s = min(8, seqs - s0);
    markov_tc_chunk8_fp8(
        latent + static_cast<long long>(s0) * 256,
        fold_weight_fp8,
        fold_scale,
        fold_bias,
        base_logits + static_cast<long long>(s0) * num_steps * vocab_padded,
        pmax + static_cast<long long>(s0) * gridDim.x,
        pidx + static_cast<long long>(s0) * gridDim.x,
        n_s, step, softcap, vocab_padded, vocab_unpadded, num_steps, s_w, s_pv, s_pi);
    if (s0 + 8 < seqs) __syncthreads();
  }
  asm volatile("griddepcontrol.launch_dependents;\n" ::);
}

// PDL-chained final argmax over up to 256 partial CTAs (grid [seqs, 1, 1], block [256, 1, 1]).
extern "C" __global__ void kern_markov_argmax_final_pdl(
    const float* __restrict__ pmax,
    const int* __restrict__ pidx,
    long long* __restrict__ out,
    int out_stride,
    int nb) {
  asm volatile("griddepcontrol.wait;\n" ::);
  Pair best = {-__builtin_inff(), INT_MAX};
  if (threadIdx.x < nb) {
    best = {pmax[blockIdx.x * nb + threadIdx.x],
            pidx[blockIdx.x * nb + threadIdx.x]};
  }
  best = block_reduce<256>(best);
  if (threadIdx.x == 0) out[(long long)blockIdx.x * out_stride] = best.i;
}

// ============================================================================
// Legacy standalone entry points preserved for unit test ABI compatibility
// ============================================================================

// grid [rows], block 256: out[r] = table[ids[r * id_stride]]
extern "C" __global__ void kern_embedding_rows_i64_bf16(
    const long long* __restrict__ ids, int id_stride,
    const __nv_bfloat16* __restrict__ table, __nv_bfloat16* __restrict__ out,
    int hidden) {
  long long row = ids[(long long)blockIdx.x * id_stride];
  const __nv_bfloat16* src = table + row * hidden;
  __nv_bfloat16* dst = out + (long long)blockIdx.x * hidden;
  for (int j = threadIdx.x; j < hidden; j += blockDim.x) {
    dst[j] = src[j];
  }
}

// grid [rows, NB], block 1024
extern "C" __global__ void kern_argmax_rows_partial_bf16(
    const __nv_bfloat16* __restrict__ logits, long long row_stride,
    float* __restrict__ pmax, int* __restrict__ pidx, int n) {
  int nb = gridDim.y;
  int chunk = (n + nb - 1) / nb;
  int lo = blockIdx.y * chunk;
  int hi = min(lo + chunk, n);
  const __nv_bfloat16* row = logits + (long long)blockIdx.x * row_stride;
  Pair best = {-__builtin_inff(), INT_MAX};
  for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) {
    float v = __bfloat162float(row[j]);
    if (isnan(v)) v = -__builtin_inff();
    best = pick(best, {v, j});
  }
  best = block_reduce<1024>(best);
  if (threadIdx.x == 0) {
    pmax[blockIdx.x * nb + blockIdx.y] = best.v;
    pidx[blockIdx.x * nb + blockIdx.y] = best.i;
  }
}

// grid [rows], block 64 (= NB)
extern "C" __global__ void kern_argmax_rows_final_i64(
    const float* __restrict__ pmax, const int* __restrict__ pidx,
    long long* __restrict__ out, int out_stride, int nb) {
  Pair best = {-__builtin_inff(), INT_MAX};
  if (threadIdx.x < nb) {
    best = {pmax[blockIdx.x * nb + threadIdx.x],
            pidx[blockIdx.x * nb + threadIdx.x]};
  }
  best = block_reduce<64>(best);
  if (threadIdx.x == 0) out[(long long)blockIdx.x * out_stride] = best.i;
}
