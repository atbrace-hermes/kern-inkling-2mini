// SPDX-License-Identifier: Apache-2.0
// SM120 (sm_120a) small-M BF16 GEMV/GEMM kernels for Inkling-2-Mini linear projections
// (`m3_gemm_bf16`, M <= 32): C[M, N] = A[M, K] @ W[N, K]^T (row-major BF16, FP32 acc).
//
// Design:
// - 256-bit vectorized global loads (`ld.global.nc.L1::no_allocate.L2::256B.v8.b32` for W,
//   `ld.global.L1::evict_last.L2::evict_last.v8.b32` for A): 1 load = 16 BF16 = 32B,
//   4 lanes/row = 128B full cache line, feeding 4 `mma.sync.aligned.m16n8k16` ops directly
//   from registers with ZERO SMEM staging and ZERO `__syncthreads()` inside the K-loop.
// - Intra-CTA K-splitting across 8 warps (K_STEP = 512 BF16 = 1024B/row/iter -> 5 iters
//   for K=2304, 6 iters for K=3072) with software-pipelined W prefetch and a single
//   bank-conflict-free float4 SMEM cross-warp reduction at epilogue.
// - Programmatic Dependent Launch (PDL): immutable weight W prefetch precedes
//   `griddepcontrol.wait;`, and `griddepcontrol.launch_dependents;` overlaps consecutive
//   projections' launch + weight prefetch with the current kernel's epilogue/tail.
// - Dedicated M=1 warp-per-row (N>=512), 2-warp/row (128<=N<512), and CTA-per-row (N<128)
//   GEMV kernel for single-token decode (`decode-g1-r1`, `prefill-c1`).

#include <cuda_bf16.h>
#include <cstdint>

namespace {

using bf16 = __nv_bfloat16;

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

// Coherent global load for mutable activation A (NO .nc! A is written by prior kernels in the graph).
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

// Generic safe fallback for arbitrary (M, N, K) when alignment or K % 64 != 0 or M > M_CAP.
// Keep __noinline__ so fast path stays at minimum registers.
__device__ __noinline__ void fallback_gemm_bf16(
    const bf16* __restrict__ A,
    const bf16* __restrict__ W,
    bf16* __restrict__ C,
    int M, int N, int K) {
  pdl_wait();
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int stride = gridDim.x * blockDim.x;
  for (int n = tid; n < N; n += stride) {
    const bf16* w_row = W + static_cast<int64_t>(n) * K;
    for (int m = 0; m < M; ++m) {
      const bf16* a_row = A + static_cast<int64_t>(m) * K;
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        acc = fmaf(__bfloat162float(a_row[k]), __bfloat162float(w_row[k]), acc);
      }
      C[static_cast<int64_t>(m) * N + n] = __float2bfloat16_rn(acc);
    }
  }
  pdl_launch_dependents();
}

__device__ __forceinline__ bool is_v8_aligned(const bf16* A, const bf16* W, int K) {
  return ((K & 63) == 0)
      && ((reinterpret_cast<uintptr_t>(A) & 31u) == 0u)
      && ((reinterpret_cast<uintptr_t>(W) & 31u) == 0u);
}

// ---------------------------------------------------------------------------
// Direct-register 256-bit v8.b32 MMA kernel (X = m16, W = n8), 8 warps along K.
// - Adaptive BN: 8 for N>=1024 (q/gate/o/k/v, 6552, 8576) and N<128 (sink=24, sconv=32);
//   4 for 128<=N<1024 (r_proj=384 -> 96 CTAs, saturating both 110-SM and 188-SM GPUs).
// - Zero stack spills (`REG:64` for M<=8, `REG:80` for M<=16 & M<=32).
// ---------------------------------------------------------------------------
template <int M_TILES, bool M_LE_8, int BATCH_MT>
__device__ __forceinline__ void run_v8_mma_pdl(
    const bf16* __restrict__ A,
    const bf16* __restrict__ W,
    bf16* __restrict__ C,
    int M, int N, int K) {
  constexpr int M_CAP = M_LE_8 ? 8 : (M_TILES * 16);
  if (M <= 0 || N <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_v8_aligned(A, W, K) || M > M_CAP) {
    fallback_gemm_bf16(A, W, C, M, N, K);
    return;
  }

  constexpr int WARPS_K = 8;
  constexpr int K_STEP = 64 * WARPS_K;  // 512 BF16 = 1024 B/row/iter

  // BN=8 for large N (>=1024) and tiny N (<128, minimizes X L2 traffic on sink/sconv);
  // BN=4 for mid N (128..1023, e.g. r_proj=384 -> 96 CTAs across 110/188 SMs).
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
  const bf16* a_base = A + t * 16;

  for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
    const int n_base = tile * bn;
    const int n_row = n_base + g;
    const bool w_ok = (g < bn) && (n_row < N);
    const bf16* w_base = W + static_cast<int64_t>(n_row) * K + t * 16;

    // Prefetch first immutable weight chunk BEFORE waiting on previous kernel!
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
      for (int mb = 0; mb < M_TILES; mb += BATCH_MT) {
        u32x8 x0[BATCH_MT], x1[BATCH_MT];
#pragma unroll
        for (int j = 0; j < BATCH_MT; ++j) {
          const int mt = mb + j;
          const int m0 = mt * 16 + g;
          if (m0 < M) {
            x0[j] = ld_x_v8(a_base + static_cast<int64_t>(m0) * K + k);
          } else {
            x0[j] = {};
          }
          if constexpr (!M_LE_8) {
            const int m1 = m0 + 8;
            if (m1 < M) {
              x1[j] = ld_x_v8(a_base + static_cast<int64_t>(m1) * K + k);
            } else {
              x1[j] = {};
            }
          } else {
            x1[j] = {};
          }
        }
#pragma unroll
        for (int j = 0; j < BATCH_MT; ++j) {
          mma_v8_step(acc[mb + j], x0[j], x1[j], w_cur);
        }
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
      const bool c_align4 = ((N & 1) == 0) && ((reinterpret_cast<uintptr_t>(C) & 3u) == 0u);
      if (c0_ok) {
#pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
#pragma unroll
          for (int w = 0; w < WARPS_K - 1; ++w) {
            float4 v = s_red[w][mt][lane];
            acc[mt][0] += v.x;
            acc[mt][1] += v.y;
            acc[mt][2] += v.z;
            acc[mt][3] += v.w;
          }
          const int m0 = mt * 16 + g;
          if (m0 < M) {
            if (c1_ok && c_align4) {
              *reinterpret_cast<__nv_bfloat162*>(C + static_cast<int64_t>(m0) * N + col0) =
                  __floats2bfloat162_rn(acc[mt][0], acc[mt][1]);
            } else {
              C[static_cast<int64_t>(m0) * N + col0] = __float2bfloat16_rn(acc[mt][0]);
              if (c1_ok) C[static_cast<int64_t>(m0) * N + col1] = __float2bfloat16_rn(acc[mt][1]);
            }
          }
          if constexpr (!M_LE_8) {
            const int m1 = m0 + 8;
            if (m1 < M) {
              if (c1_ok && c_align4) {
                *reinterpret_cast<__nv_bfloat162*>(C + static_cast<int64_t>(m1) * N + col0) =
                    __floats2bfloat162_rn(acc[mt][2], acc[mt][3]);
              } else {
                C[static_cast<int64_t>(m1) * N + col0] = __float2bfloat16_rn(acc[mt][2]);
                if (c1_ok) C[static_cast<int64_t>(m1) * N + col1] = __float2bfloat16_rn(acc[mt][3]);
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
// M = 1 specialized 256-bit BF16 GEMV with PDL (decode-g1-r1 / prefill-c1).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void run_v8_gemv_m1_pdl(
    const bf16* __restrict__ A,
    const bf16* __restrict__ W,
    bf16* __restrict__ C,
    int M, int N, int K) {
  if (M <= 0 || N <= 0 || K <= 0) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }
  if (!is_v8_aligned(A, W, K) || M != 1) {
    fallback_gemm_bf16(A, W, C, M, N, K);
    return;
  }

  constexpr int WARPS = 8;
  constexpr int NUM_THREADS = 256;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;

  if (N < 128) {
    // 1 CTA per output row, all 256 threads collaborate across K (1 load/thread for K=2304!).
    __shared__ float s_red[WARPS];
    if (blockIdx.x >= N) {
      pdl_wait();
      pdl_launch_dependents();
      return;
    }
    for (int row = blockIdx.x; row < N; row += gridDim.x) {
      const bf16* w_row = W + static_cast<int64_t>(row) * K;
      const int k0 = tid * 16;
      u32x8 wv0 = (k0 < K) ? ld_w_v8(w_row + k0) : u32x8{};
      if (row == blockIdx.x) pdl_wait();

      float acc = 0.0f;
      for (int k = k0; k < K; k += NUM_THREADS * 16) {
        u32x8 wv = (k == k0) ? wv0 : ld_w_v8(w_row + k);
        u32x8 xv = ld_x_v8(A + k);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&wv.v[i]));
          float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.v[i]));
          acc = fmaf(wf.x, xf.x, acc);
          acc = fmaf(wf.y, xf.y, acc);
        }
      }
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
      if (lane == 0) s_red[warp] = acc;
      __syncthreads();
      if (row + gridDim.x >= N) pdl_launch_dependents();
      if (tid == 0) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) sum += s_red[w];
        C[row] = __float2bfloat16_rn(sum);
      }
      if (row + gridDim.x < N) __syncthreads();
    }
    return;
  }

  if (N < 512) {
    // 2 warps per output row -> 4 rows/CTA (r_proj=384 -> 96 CTAs across 110/188 SMs!).
    __shared__ float s_pair[4];
    const int sub_row = warp >> 1;   // 0..3
    const int sub_k = warp & 1;      // 0..1
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
      const bf16* w_row = W + static_cast<int64_t>(row) * K;
      u32x8 w_cur = (ok && k0 < K) ? ld_w_v8(w_row + k0) : u32x8{};
      if (tile_row == base_row) pdl_wait();

      float acc = 0.0f;
      if (ok) {
        for (int k = k0; k < K; k += 1024) {
          u32x8 w_nxt = (k + 1024 < K) ? ld_w_v8(w_row + k + 1024) : u32x8{};
          u32x8 xv = ld_x_v8(A + k);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w_cur.v[i]));
            float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.v[i]));
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
        C[row] = __float2bfloat16_rn(acc + s_pair[sub_row]);
      }
      if (tile_row + gridDim.x * 4 < N) __syncthreads();
    }
    return;
  }

  // N >= 512: 1 warp per output row (8 rows/CTA -> 1075 CTAs for N=8600, 819 for 6552, 128 for 1024).
  const int global_warp = blockIdx.x * WARPS + warp;
  const int total_warps = gridDim.x * WARPS;
  if (global_warp >= N) {
    pdl_wait();
    pdl_launch_dependents();
    return;
  }

  for (int row = global_warp; row < N; row += total_warps) {
    const bf16* w_row = W + static_cast<int64_t>(row) * K;
    const int k0 = lane * 16;
    u32x8 w_cur = (k0 < K) ? ld_w_v8(w_row + k0) : u32x8{};
    if (row == global_warp) pdl_wait();

    float acc = 0.0f;
    for (int k = k0; k < K; k += 512) {
      u32x8 w_nxt = (k + 512 < K) ? ld_w_v8(w_row + k + 512) : u32x8{};
      u32x8 xv = ld_x_v8(A + k);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        float2 wf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w_cur.v[i]));
        float2 xf = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&xv.v[i]));
        acc = fmaf(wf.x, xf.x, acc);
        acc = fmaf(wf.y, xf.y, acc);
      }
      w_cur = w_nxt;
    }
    if (row + total_warps >= N) pdl_launch_dependents();
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0) {
      C[row] = __float2bfloat16_rn(acc);
    }
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Single GEMV/GEMM entries: M=1, M in [2..8], [9..16], [17..32], [33..64].
// Grid: [1088, 1, 1], Block: [256, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 4) void kern_bf16_gemv_m1_sm120(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
  run_v8_gemv_m1_pdl(A, W, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 4) void kern_bf16_gemm_m8_sm120(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
  run_v8_mma_pdl<1, true, 1>(A, W, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 3) void kern_bf16_gemm_m16_sm120(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
  run_v8_mma_pdl<1, false, 1>(A, W, C, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 3) void kern_bf16_gemm_m32_sm120(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ W,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K) {
  run_v8_mma_pdl<2, false, 2>(A, W, C, M, N, K);
}

// ---------------------------------------------------------------------------
// Dual-stream paired GEMV/GEMM entries (e.g. k_proj + v_proj when x_k != x_v).
// Grid: [ceil_div(N, 8), 2, 1], Block: [256, 1, 1] (blockIdx.y=0 -> stream 0, y=1 -> stream 1).
// ---------------------------------------------------------------------------
extern "C" __global__ __launch_bounds__(256, 4) void kern_bf16_gemm_pair_m1_sm120(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ W0,
    __nv_bfloat16* __restrict__ C0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ W1,
    __nv_bfloat16* __restrict__ C1,
    int M, int N, int K) {
  const bool s1 = (blockIdx.y != 0);
  run_v8_gemv_m1_pdl(s1 ? A1 : A0, s1 ? W1 : W0, s1 ? C1 : C0, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 4) void kern_bf16_gemm_pair_m8_sm120(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ W0,
    __nv_bfloat16* __restrict__ C0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ W1,
    __nv_bfloat16* __restrict__ C1,
    int M, int N, int K) {
  const bool s1 = (blockIdx.y != 0);
  run_v8_mma_pdl<1, true, 1>(s1 ? A1 : A0, s1 ? W1 : W0, s1 ? C1 : C0, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 3) void kern_bf16_gemm_pair_m16_sm120(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ W0,
    __nv_bfloat16* __restrict__ C0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ W1,
    __nv_bfloat16* __restrict__ C1,
    int M, int N, int K) {
  const bool s1 = (blockIdx.y != 0);
  run_v8_mma_pdl<1, false, 1>(s1 ? A1 : A0, s1 ? W1 : W0, s1 ? C1 : C0, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 3) void kern_bf16_gemm_pair_m32_sm120(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ W0,
    __nv_bfloat16* __restrict__ C0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ W1,
    __nv_bfloat16* __restrict__ C1,
    int M, int N, int K) {
  const bool s1 = (blockIdx.y != 0);
  run_v8_mma_pdl<2, false, 2>(s1 ? A1 : A0, s1 ? W1 : W0, s1 ? C1 : C0, M, N, K);
}

extern "C" __global__ __launch_bounds__(256, 3) void kern_bf16_gemm_pair_m64_sm120(
    const __nv_bfloat16* __restrict__ A0,
    const __nv_bfloat16* __restrict__ W0,
    __nv_bfloat16* __restrict__ C0,
    const __nv_bfloat16* __restrict__ A1,
    const __nv_bfloat16* __restrict__ W1,
    __nv_bfloat16* __restrict__ C1,
    int M, int N, int K) {
  const bool s1 = (blockIdx.y != 0);
  run_v8_mma_pdl<4, false, 1>(s1 ? A1 : A0, s1 ? W1 : W0, s1 ? C1 : C0, M, N, K);
}
