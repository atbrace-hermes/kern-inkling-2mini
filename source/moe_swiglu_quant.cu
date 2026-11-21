// SPDX-License-Identifier: Apache-2.0
// SM120 MoE activation quantization, softcapped SwiGLU, and weight scale swizzle for Inkling-2-Mini:
//   - moe_swizzle_weight_scales (derive): [E, N, scale_k] row-major -> SM120 128x4 Sm1xxBlkScaledConfig
//   - moe_quant_nvfp4: fouroversix_mae (K=2304, block=16) + 128x4 swizzled scales + zero_tails
//   - moe_swiglu_quant_nvfp4: softcapped SwiGLU (gate=4.0, up=25.0) + fouroversix_mae (N=1152) + zero_tails
//   - moe_quant_mxfp8: UE8M0 block=32 MXFP8 (K=2304) + 128x4 swizzled scales + zero_tails
//   - moe_swiglu_quant_mxfp8: softcapped SwiGLU + UE8M0 block=32 MXFP8 (N=1152) + 128x4 scales + zero_tails

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kNumExperts = 128;
constexpr int kTopK = 8;
constexpr int kHidden = 2304;
constexpr int kInter = 1152;
constexpr float kGateCap = 4.0f;
constexpr float kUpCap = 25.0f;
constexpr float kFltMax = 3.4028234663852886e38f;

__device__ __forceinline__ float e4m3_round(float v, uint8_t& bits) {
  uint16_t pair;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
  bits = pair & 0xFF;
  uint32_t halves;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(halves) : "h"(pair));
  return __half2float(__ushort_as_half(halves & 0xFFFF));
}

__device__ __forceinline__ uint32_t e2m1_pack(float low, float high, float& rlow, float& rhigh) {
  uint32_t packed, halves;
  asm("{ .reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, %3, %2; "
      "cvt.u32.u8 %0, b; cvt.rn.f16x2.e2m1x2 %1, b; }"
      : "=r"(packed), "=r"(halves)
      : "f"(low), "f"(high));
  rlow = __half2float(__ushort_as_half(halves & 0xFFFF));
  rhigh = __half2float(__ushort_as_half(halves >> 16));
  return packed;
}

template <int E>
__device__ __forceinline__ void quantize_part(
    float const (&x)[E], float amax, float block_max, uint32_t& packed, uint8_t& scale) {
  float encoder = fminf(__fdiv_rn(1536.0f, amax), kFltMax);
  float decoder = __fdiv_rn(1.0f, encoder);
  float outer = __fmul_rn(amax, float(1.0 / 1536.0));
  uint8_t bits6, bits4;
  float scale6 = e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, float(1.0 / 6.0))), 448.0f), bits6);
  float scale4 = e4m3_round(fminf(__fmul_rn(block_max, __fmul_rn(encoder, 0.25f)), 448.0f), bits4);
  float inverse6 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale6, decoder)), kFltMax);
  float inverse4 = fminf(__fdiv_rn(1.0f, __fmul_rn(scale4, decoder)), kFltMax);
  uint32_t p6 = 0, p4 = 0;
  float e6[E], e4[E];
#pragma unroll
  for (int i = 0; i < E / 2; ++i) {
    float r0, r1;
    p6 |= e2m1_pack(__fmul_rn(x[2 * i], inverse6), __fmul_rn(x[2 * i + 1], inverse6), r0, r1) << (8 * i);
    e6[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale6), outer), x[2 * i]));
    e6[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale6), outer), x[2 * i + 1]));
    p4 |= e2m1_pack(__fmul_rn(x[2 * i], inverse4), __fmul_rn(x[2 * i + 1], inverse4), r0, r1) << (8 * i);
    e4[2 * i] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r0, scale4), outer), x[2 * i]));
    e4[2 * i + 1] = fabsf(__fsub_rn(__fmul_rn(__fmul_rn(r1, scale4), outer), x[2 * i + 1]));
  }
#pragma unroll
  for (int width = E / 2; width >= 1; width /= 2) {
#pragma unroll
    for (int i = 0; i < width; ++i) {
      e6[i] = __fadd_rn(e6[2 * i], e6[2 * i + 1]);
      e4[i] = __fadd_rn(e4[2 * i], e4[2 * i + 1]);
    }
  }
#pragma unroll
  for (int s = 1; s < 16 / E; s *= 2) {
    e6[0] = __fadd_rn(e6[0], __shfl_xor_sync(0xFFFFFFFFu, e6[0], s));
    e4[0] = __fadd_rn(e4[0], __shfl_xor_sync(0xFFFFFFFFu, e4[0], s));
  }
  bool choose4 = e4[0] < e6[0];
  packed = choose4 ? p4 : p6;
  scale = choose4 ? bits4 : bits6;
}

template <int E>
__device__ __forceinline__ float part_max(float const (&x)[E]) {
  float m = 0.0f;
#pragma unroll
  for (int i = 0; i < E; ++i) m = fmaxf(m, fabsf(x[i]));
#pragma unroll
  for (int s = 1; s < 16 / E; s *= 2) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, s));
  return m;
}

template <int kThreads>
__device__ __forceinline__ float block_max_reduce(float v, float* red) {
#pragma unroll
  for (int s = 16; s >= 1; s /= 2) v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, s));
  constexpr int kWarps = kThreads / 32;
  if (threadIdx.x % 32 == 0) red[threadIdx.x / 32] = v;
  __syncthreads();
  v = red[0];
#pragma unroll
  for (int w = 1; w < kWarps; ++w) v = fmaxf(v, red[w]);
  return v;
}

__device__ __forceinline__ int64_t scale_base(int const* offsets, int expert, int scale_k) {
  return int64_t(((offsets[expert] + expert * 127) / 128) * 128) * scale_k;
}

__device__ __forceinline__ int scale_offset(int local_r, int col, int scale_k) {
  return ((local_r / 128) * (scale_k / 4) + col / 4) * 512 + (local_r % 32) * 16 +
         ((local_r / 32) % 4) * 4 + col % 4;
}

__device__ __forceinline__ int row_expert(int const* offsets, int row) {
  int lo = 0, hi = kNumExperts;
  while (hi - lo > 1) {
    int mid = (lo + hi) / 2;
    if (offsets[mid] <= row)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

__device__ __forceinline__ void load_offsets(int const* offsets, int* s_offsets) {
  for (int i = threadIdx.x; i <= kNumExperts; i += blockDim.x) {
    s_offsets[i] = offsets[i];
  }
  __syncthreads();
}

__device__ __forceinline__ void zero_tails(
    int const* offsets, uint8_t* x_scales, int scale_k) {
  int const atoms = scale_k / 4;
  int const lane = threadIdx.x & 31;
  int const warps = blockDim.x >> 5;
  for (int item = blockIdx.x * warps + (threadIdx.x >> 5); item < kNumExperts * atoms;
       item += gridDim.x * warps) {
    int e = item / atoms;
    int count = offsets[e + 1] - offsets[e];
    int used = count & 127;
    if (used == 0) continue;
    uint32_t* atom = reinterpret_cast<uint32_t*>(
        x_scales + scale_base(offsets, e, scale_k) +
        (int64_t(count / 128) * atoms + (item % atoms)) * 512 + lane * 16);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      if (lane + 32 * i >= used) atom[i] = 0;
    }
  }
}

__device__ __forceinline__ float capped_swiglu(float gate, float up, float gamma) {
  float sigmoid;
  asm("{ .reg .f32 t; mul.rn.f32 t, %1, 0fBFB8AA3B; "
      "ex2.approx.ftz.f32 t, t; add.rn.f32 t, t, 0f3F800000; "
      "rcp.approx.ftz.f32 %0, t; }"
      : "=f"(sigmoid)
      : "f"(gate));
  float tg, tu;
  asm("tanh.approx.f32 %0, %1;" : "=f"(tg) : "f"(__fdiv_rn(gate, kGateCap)));
  gate = __fmul_rn(kGateCap, tg);
  asm("tanh.approx.f32 %0, %1;" : "=f"(tu) : "f"(__fdiv_rn(up, kUpCap)));
  up = __fmul_rn(kUpCap, tu);
  return __fmul_rn(__fmul_rn(__fmul_rn(gate, sigmoid), up), gamma);
}

__device__ __forceinline__ void mxfp8_half_block(
    float const (&x)[16], uint4& values, uint8_t& exponent_out) {
  float amax = 0.0f;
#pragma unroll
  for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
  amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, 1));
  float const min_scale = __int_as_float(0x00400000);
  float raw = __fmul_rn(amax, float(1.0 / 448.0));
  int exponent = ((__float_as_int(raw) + 0x007FFFFF) & 0x7F800000) >> 23;
  exponent = (raw <= min_scale) ? 0 : min(exponent, 254);
  if (raw != raw) exponent = 255;
  float scale = fmaxf(__int_as_float(exponent << 23), min_scale);
  float inverse = (exponent == 0) ? __int_as_float(0x7F000000) : __frcp_rn(scale);
  if (exponent == 255) inverse = 0.0f;
  uint16_t pairs[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float lo = __fmul_rn(x[2 * i], inverse);
    float hi = __fmul_rn(x[2 * i + 1], inverse);
    lo = fminf(fmaxf(lo, -448.0f), 448.0f);
    hi = fminf(fmaxf(hi, -448.0f), 448.0f);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pairs[i]) : "f"(hi), "f"(lo));
  }
  values = *reinterpret_cast<uint4*>(pairs);
  exponent_out = uint8_t(exponent);
}

}  // namespace

// Derive kernel: swizzle [experts, n, scale_k] row-major scales into SM120 128x4 layout.
// Grid: (ceil_div(experts * n * scale_k, 256), 1, 1), Block: (256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) moe_swizzle_weight_scales(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    int experts,
    int n,
    int scale_k) {
  int64_t idx = int64_t(blockIdx.x) * 256 + threadIdx.x;
  int64_t total = int64_t(experts) * n * scale_k;
  if (idx >= total) return;

  int c = int(idx % scale_k);
  int64_t row_idx = idx / scale_k;
  int r = int(row_idx % n);
  int e = int(row_idx / n);

  int64_t dst_idx = int64_t(e) * n * scale_k + scale_offset(r, c, scale_k);
  dst[dst_idx] = src[idx];
}

// Gather & quantize BF16 x [T, 2304] into NVFP4 [n, 1152] + 128x4 swizzled E4M3 scales + x_amax [n].
// K=2304 -> 144 blocks of 16, split=4 -> 576 threads (18 warps) per CTA, E=4 values/thread.
// Grid: (min(n, 2048), 1, 1), Block: (576, 1, 1)
extern "C" __global__ void __launch_bounds__(576) moe_quant_nvfp4(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ dst2src,
    const int* __restrict__ m_indptr,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    float* __restrict__ x_amax,
    int n) {
  constexpr int kThreads = 576;
  constexpr int kSplit = 4;
  constexpr int kE = 4;
  constexpr int kScaleK = kHidden / 16;  // 144

  __shared__ int s_offsets[kNumExperts + 1];
  __shared__ float red[2][kThreads / 32];

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  load_offsets(m_indptr, s_offsets);
  int const block = threadIdx.x / kSplit;
  int const part = threadIdx.x % kSplit;
  int const total_rows = s_offsets[kNumExperts];
  int parity = 0;

  for (int row = blockIdx.x; row < total_rows; row += gridDim.x, parity ^= 1) {
    float v[kE];
    int64_t source = dst2src[row] / kTopK;
    const __nv_bfloat16* src = x + source * kHidden + block * 16 + part * kE;
    uint2 raw = *reinterpret_cast<const uint2*>(src);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
#pragma unroll
    for (int i = 0; i < kE; ++i) v[i] = __bfloat162float(h[i]);

    float block_max = part_max<kE>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale;
    quantize_part<kE>(v, amax, block_max, packed, scale);

    if (threadIdx.x == 0) x_amax[row] = amax;
    int expert = row_expert(s_offsets, row);
    uint8_t* dst = values + int64_t(row) * (kHidden / 2) + block * 8 + part * 2;
    *reinterpret_cast<uint16_t*>(dst) = uint16_t(packed);
    if (part == 0) {
      x_scales[scale_base(s_offsets, expert, kScaleK) +
               scale_offset(row - s_offsets[expert], block, kScaleK)] = scale;
    }
  }
  zero_tails(s_offsets, x_scales, kScaleK);
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Softcapped SwiGLU (interleaved [gate, up]) + NVFP4 fouroversix_mae quantization (N=1152).
// N=1152 -> 72 blocks of 16, split=4 -> 288 threads (9 warps) per CTA, E=4 values/thread.
// Grid: (min(n, 2048), 1, 1), Block: (288, 1, 1)
extern "C" __global__ void __launch_bounds__(288) moe_swiglu_quant_nvfp4(
    const float* __restrict__ w13_f32,
    const float* __restrict__ x_amax,
    const float* __restrict__ w13_amax,
    const float* __restrict__ w13_denom,
    const float* __restrict__ gammas,
    const int* __restrict__ m_indptr,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    float* __restrict__ act_amax,
    int n) {
  constexpr int kThreads = 288;
  constexpr int kSplit = 4;
  constexpr int kE = 4;
  constexpr int kScaleK = kInter / 16;  // 72

  __shared__ int s_offsets[kNumExperts + 1];
  __shared__ float red[2][kThreads / 32];

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  load_offsets(m_indptr, s_offsets);
  int const block = threadIdx.x / kSplit;
  int const part = threadIdx.x % kSplit;
  float const rcp = __frcp_rn(__fmul_rn(1536.0f, *w13_denom));
  int const total_rows = s_offsets[kNumExperts];
  int parity = 0;

  for (int row = blockIdx.x; row < total_rows; row += gridDim.x, parity ^= 1) {
    int expert = row_expert(s_offsets, row);
    const float4* src =
        reinterpret_cast<const float4*>(w13_f32 + int64_t(row) * (2 * kInter) + block * 32 + part * 8);
    float4 raw[2] = {src[0], src[1]};
    const float* a = reinterpret_cast<const float*>(raw);
    float scale = __fmul_rn(__fmul_rn(x_amax[row], w13_amax[expert]), rcp);
    float gamma = gammas[row];

    float v[kE];
#pragma unroll
    for (int i = 0; i < kE; ++i) {
      float gate = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a[2 * i], scale)));
      float up = __bfloat162float(__float2bfloat16_rn(__fmul_rn(a[2 * i + 1], scale)));
      v[i] = __bfloat162float(__float2bfloat16_rn(capped_swiglu(gate, up, gamma)));
    }

    float block_max = part_max<kE>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale_bits;
    quantize_part<kE>(v, amax, block_max, packed, scale_bits);

    if (threadIdx.x == 0) act_amax[row] = amax;
    uint8_t* dst = values + int64_t(row) * (kInter / 2) + block * 8 + part * 2;
    *reinterpret_cast<uint16_t*>(dst) = uint16_t(packed);
    if (part == 0) {
      x_scales[scale_base(s_offsets, expert, kScaleK) +
               scale_offset(row - s_offsets[expert], block, kScaleK)] = scale_bits;
    }
  }
  zero_tails(s_offsets, x_scales, kScaleK);
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Softcapped SwiGLU (interleaved [gate, up] BF16, pre-scaled by NVFP4 EVT epilogue)
// + NVFP4 fouroversix_mae quantization (N=1152).
// N=1152 -> 72 blocks of 16, split=4 -> 288 threads (9 warps) per CTA, E=4 values/thread.
// Grid: (min(n, 2048), 1, 1), Block: (288, 1, 1)
extern "C" __global__ void __launch_bounds__(288) moe_swiglu_quant_nvfp4_bf16(
    const __nv_bfloat16* __restrict__ w13_bf16,
    const float* __restrict__ gammas,
    const int* __restrict__ m_indptr,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    float* __restrict__ act_amax,
    int n) {
  constexpr int kThreads = 288;
  constexpr int kSplit = 4;
  constexpr int kE = 4;
  constexpr int kScaleK = kInter / 16;  // 72

  __shared__ int s_offsets[kNumExperts + 1];
  __shared__ float red[2][kThreads / 32];

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  load_offsets(m_indptr, s_offsets);
  int const block = threadIdx.x / kSplit;
  int const part = threadIdx.x % kSplit;
  int const total_rows = s_offsets[kNumExperts];
  int parity = 0;

  for (int row = blockIdx.x; row < total_rows; row += gridDim.x, parity ^= 1) {
    uint4 raw = *reinterpret_cast<const uint4*>(
        w13_bf16 + int64_t(row) * (2 * kInter) + block * 32 + part * 8);
    const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(&raw);
    float gamma = gammas[row];

    float v[kE];
#pragma unroll
    for (int i = 0; i < kE; ++i) {
      float gate = __bfloat162float(h[2 * i]);
      float up = __bfloat162float(h[2 * i + 1]);
      v[i] = __bfloat162float(__float2bfloat16_rn(capped_swiglu(gate, up, gamma)));
    }

    float block_max = part_max<kE>(v);
    float amax = block_max_reduce<kThreads>(block_max, red[parity]);
    uint32_t packed;
    uint8_t scale_bits;
    quantize_part<kE>(v, amax, block_max, packed, scale_bits);

    if (threadIdx.x == 0) act_amax[row] = amax;
    uint8_t* dst = values + int64_t(row) * (kInter / 2) + block * 8 + part * 2;
    *reinterpret_cast<uint16_t*>(dst) = uint16_t(packed);
    if (part == 0) {
      int expert = row_expert(s_offsets, row);
      x_scales[scale_base(s_offsets, expert, kScaleK) +
               scale_offset(row - s_offsets[expert], block, kScaleK)] = scale_bits;
    }
  }
  zero_tails(s_offsets, x_scales, kScaleK);
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Gather & quantize BF16 x [T, 2304] into MXFP8 [n, 2304] + 128x4 swizzled UE8M0 scales.
// K=2304 -> 144 half-blocks of 16 -> 160 threads (5 warps), active block < 144.
// Grid: (min(n, 2048), 1, 1), Block: (160, 1, 1)
extern "C" __global__ void __launch_bounds__(160) moe_quant_mxfp8(
    const __nv_bfloat16* __restrict__ x,
    const int* __restrict__ dst2src,
    const int* __restrict__ m_indptr,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    int n) {
  constexpr int kScaleK = kHidden / 32;  // 72
  constexpr int kHalfBlocks = kHidden / 16;  // 144

  __shared__ int s_offsets[kNumExperts + 1];
#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  load_offsets(m_indptr, s_offsets);
  int const block = threadIdx.x;
  bool const active = (block < kHalfBlocks);
  int const total_rows = s_offsets[kNumExperts];

  for (int row = blockIdx.x; row < total_rows; row += gridDim.x) {
    float v[16];
    if (active) {
      int64_t source = dst2src[row] / kTopK;
      const uint4* src = reinterpret_cast<const uint4*>(x + source * kHidden + block * 16);
      uint4 raw[2] = {src[0], src[1]};
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(raw);
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = __bfloat162float(h[i]);
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    }
    uint4 packed;
    uint8_t exponent;
    mxfp8_half_block(v, packed, exponent);
    if (active) {
      *reinterpret_cast<uint4*>(values + int64_t(row) * kHidden + block * 16) = packed;
      if ((block & 1) == 0) {
        int expert = row_expert(s_offsets, row);
        x_scales[scale_base(s_offsets, expert, kScaleK) +
                 scale_offset(row - s_offsets[expert], block >> 1, kScaleK)] = exponent;
      }
    }
  }
  zero_tails(s_offsets, x_scales, kScaleK);
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

// Softcapped SwiGLU (interleaved [gate, up] BF16) + MXFP8 quantization (N=1152) + 128x4 scales.
// N=1152 -> 72 half-blocks of 16 -> 96 threads (3 warps), active block < 72.
// Grid: (min(n, 2048), 1, 1), Block: (96, 1, 1)
extern "C" __global__ void __launch_bounds__(96) moe_swiglu_quant_mxfp8(
    const __nv_bfloat16* __restrict__ w13_bf16,
    const float* __restrict__ gammas,
    const int* __restrict__ m_indptr,
    uint8_t* __restrict__ values,
    uint8_t* __restrict__ x_scales,
    int n) {
  constexpr int kScaleK = kInter / 32;  // 36
  constexpr int kHalfBlocks = kInter / 16;  // 72

  __shared__ int s_offsets[kNumExperts + 1];
#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
  load_offsets(m_indptr, s_offsets);
  int const block = threadIdx.x;
  bool const active = (block < kHalfBlocks);
  int const total_rows = s_offsets[kNumExperts];

  for (int row = blockIdx.x; row < total_rows; row += gridDim.x) {
    float v[16];
    if (active) {
      const uint4* src =
          reinterpret_cast<const uint4*>(w13_bf16 + int64_t(row) * (2 * kInter) + block * 32);
      uint4 raw[4] = {src[0], src[1], src[2], src[3]};
      const __nv_bfloat16* h = reinterpret_cast<const __nv_bfloat16*>(raw);
      float gamma = gammas[row];
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        v[i] = __bfloat162float(__float2bfloat16_rn(
            capped_swiglu(__bfloat162float(h[2 * i]), __bfloat162float(h[2 * i + 1]), gamma)));
      }
    } else {
#pragma unroll
      for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    }
    uint4 packed;
    uint8_t exponent;
    mxfp8_half_block(v, packed, exponent);
    if (active) {
      *reinterpret_cast<uint4*>(values + int64_t(row) * kInter + block * 16) = packed;
      if ((block & 1) == 0) {
        int expert = row_expert(s_offsets, row);
        x_scales[scale_base(s_offsets, expert, kScaleK) +
                 scale_offset(row - s_offsets[expert], block >> 1, kScaleK)] = exponent;
      }
    }
  }
  zero_tails(s_offsets, x_scales, kScaleK);
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
