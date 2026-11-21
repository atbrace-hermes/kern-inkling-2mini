// SPDX-License-Identifier: Apache-2.0
// SM120 MXFP8 Grouped GEMM adapter for Inkling-2-Mini MoE (Layers 20..23):
//   - SwapAB=true, TileM=128, TileN in {32, 64}, TileK=128
//   - ElementA = mx_float8_t<float_e4m3_t>, ElementSF = float_ue8m0_t (SFVecSize=32, 128x4 swizzled)
//   - DTypeOut = cutlass::bfloat16_t, alpha = 1.0f
//   - Compile with -DMOE_METADATA_PROBE as host binary to emit build/sm_120/moe_mxfp8_sm120.json
//     (zero GPU calls!).

#ifdef MOE_METADATA_PROBE
#define CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL 1
#include <cuda.h>
#include <cstdio>
#include <cstring>
#include <vector>
struct CapturedTmaCall {
  uint64_t magic;
  int format, rank, interleave, swizzle, l2_promotion, oob_fill;
  uint64_t global_dim[5];
  uint64_t global_strides[4];
  uint32_t box_dim[5];
  uint32_t elem_strides[5];
};
static std::vector<CapturedTmaCall> g_tma_calls;
extern "C" CUresult CUDAAPI cuTensorMapEncodeTiled(
    CUtensorMap* tensorMap,
    CUtensorMapDataType tensorDataType,
    cuuint32_t tensorRank,
    void* globalAddress,
    const cuuint64_t* globalDim,
    const cuuint64_t* globalStrides,
    const cuuint32_t* boxDim,
    const cuuint32_t* elementStrides,
    CUtensorMapInterleave interleave,
    CUtensorMapSwizzle swizzle,
    CUtensorMapL2promotion l2Promotion,
    CUtensorMapFloatOOBfill oobFill) {
  (void)globalAddress;
  CapturedTmaCall c{};
  c.magic = 0xCAFE000000000000ULL + g_tma_calls.size();
  c.format = int(tensorDataType);
  c.rank = int(tensorRank);
  c.interleave = int(interleave);
  c.swizzle = int(swizzle);
  c.l2_promotion = int(l2Promotion);
  c.oob_fill = int(oobFill);
  for (uint32_t i = 0; i < tensorRank && i < 5; ++i) {
    c.global_dim[i] = globalDim[i];
    c.box_dim[i] = boxDim[i];
    c.elem_strides[i] = elementStrides[i];
  }
  for (uint32_t i = 0; i + 1 < tensorRank && i < 4; ++i) {
    c.global_strides[i] = globalStrides[i];
  }
  g_tma_calls.push_back(c);
  std::memset(tensorMap, 0, sizeof(*tensorMap));
  std::memcpy(tensorMap, &c.magic, sizeof(c.magic));
  return CUDA_SUCCESS;
}
#endif

#include <cstdint>
#include <cuda_bf16.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace moe_mxfp8 {

template <class StrideIntT>
CUTLASS_HOST_DEVICE cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> make_packed_stride(
    cute::Stride<StrideIntT, cute::Int<1>, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL) {
  auto s_copy = s;
  cute::get<0>(s_copy) = static_cast<StrideIntT>(cute::get<1>(shape_MKL));
  return s_copy;
}

template <class StrideIntT>
CUTLASS_HOST_DEVICE cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> make_packed_stride(
    cute::Stride<cute::Int<1>, StrideIntT, cute::Int<0>> s, cute::Shape<int, int, int> shape_MKL) {
  auto s_copy = s;
  cute::get<1>(s_copy) = static_cast<StrideIntT>(cute::get<0>(shape_MKL));
  return s_copy;
}

}  // namespace moe_mxfp8

namespace moe_mxfp8 {

using namespace cute;

template <int TileN>
struct Mxfp8Traits {
  static constexpr int TileM = 128;
  static constexpr int TileK = 128;
  static constexpr bool SwapAB = true;

  using ElementA = cutlass::float_e4m3_t;
  using ElementSFA = cutlass::float_ue8m0_t;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16

  using ElementB = cutlass::float_e4m3_t;
  using ElementSFB = cutlass::float_ue8m0_t;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16

  using ElementD = cutlass::bfloat16_t;
  using ElementC = void;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

  using ElementAccumulator = float;
  using ElementCompute = float;

  using ElementAMainloop = cutlass::mx_float8_t<ElementA>;
  using ElementBMainloop = cutlass::mx_float8_t<ElementB>;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ThreadBlockShape,
      ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
      ElementCompute, ElementC, LayoutD*, AlignmentD, ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule>::CollectiveOp;

  using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementAMainloop, LayoutA*,
      AlignmentA, ElementBMainloop, LayoutB*, AlignmentB, ElementAccumulator, ThreadBlockShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      MainloopSchedule>::CollectiveOp;

  using BaseKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue,
                                           void>;
};

struct Mxfp8KernelN32 : public Mxfp8Traits<32>::BaseKernel {};
struct Mxfp8KernelN64 : public Mxfp8Traits<64>::BaseKernel {};

using TraitsRef = Mxfp8Traits<64>;
using ProblemShape = typename TraitsRef::ProblemShape;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
using StrideA = typename TraitsRef::BaseKernel::InternalStrideA;
using StrideB = typename TraitsRef::BaseKernel::InternalStrideB;
using StrideD = typename TraitsRef::BaseKernel::InternalStrideD;
using ScaleConfig = typename TraitsRef::BaseKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = typename TraitsRef::BaseKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename TraitsRef::BaseKernel::CollectiveMainloop::InternalLayoutSFB;

static_assert(std::is_same_v<typename Mxfp8Traits<32>::BaseKernel::InternalStrideA, StrideA>);
static_assert(std::is_same_v<typename Mxfp8Traits<32>::BaseKernel::InternalStrideB, StrideB>);
static_assert(std::is_same_v<typename Mxfp8Traits<32>::BaseKernel::InternalStrideD, StrideD>);
static_assert(
    std::is_same_v<typename Mxfp8Traits<32>::BaseKernel::CollectiveMainloop::InternalLayoutSFA,
                   LayoutSFA>);
static_assert(
    std::is_same_v<typename Mxfp8Traits<32>::BaseKernel::CollectiveMainloop::InternalLayoutSFB,
                   LayoutSFB>);
static_assert(TraitsRef::BaseKernel::CollectiveMainloop::TiledMma::SFVecSize == 32);
static_assert(Mxfp8KernelN32::SharedStorageSize <= 101376, "N32 exceeds SM120 99KiB SMEM");
static_assert(Mxfp8KernelN64::SharedStorageSize <= 101376, "N64 exceeds SM120 99KiB SMEM");

}  // namespace moe_mxfp8

#ifndef MOE_METADATA_PROBE

// Device arg-prep kernel for SwapAB=true MXFP8 grouped GEMM (SFVecSize=32).
extern "C" __global__ void __launch_bounds__(128) moe_mxfp8_prepare_args(
    const uint8_t* __restrict__ A_acts,
    const uint8_t* __restrict__ B_weights,
    const uint8_t* __restrict__ SFA_acts,
    const uint8_t* __restrict__ SFB_weights,
    __nv_bfloat16* __restrict__ D_out,
    const int* __restrict__ m_indptr,
    int n,
    int k,
    int num_groups,
    void* problem_sizes_out,
    void* A_ptr_out,
    void* B_ptr_out,
    void* SFA_ptr_out,
    void* SFB_ptr_out,
    void* D_ptr_out,
    void* stride_A_out,
    void* stride_B_out,
    void* stride_D_out,
    void* layout_SFA_out,
    void* layout_SFB_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  using ElementA = typename moe_mxfp8::TraitsRef::ElementA;      // float_e4m3_t (weights)
  using ElementB = typename moe_mxfp8::TraitsRef::ElementB;      // float_e4m3_t (acts)
  using ElementSFA = typename moe_mxfp8::TraitsRef::ElementSFA;  // float_ue8m0_t (weights)
  using ElementSFB = typename moe_mxfp8::TraitsRef::ElementSFB;  // float_ue8m0_t (acts)
  using ElementD = typename moe_mxfp8::TraitsRef::ElementD;      // cutlass::bfloat16_t

  constexpr size_t ScaleGranularity = 32;
  constexpr size_t alignment_swizzled_mn = 128;
  constexpr size_t alignment_swizzled_k = ScaleGranularity * 4;
  size_t sf_n = (size_t(n) + alignment_swizzled_mn - 1) / alignment_swizzled_mn * alignment_swizzled_mn;
  size_t swizzled_k = (size_t(k) + alignment_swizzled_k - 1) / alignment_swizzled_k * alignment_swizzled_k;
  size_t sf_k = swizzled_k / ScaleGranularity;

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  if (i < num_groups) {
    int m_offset = m_indptr[i];
    int m_offset_next = m_indptr[i + 1];
    size_t m = size_t(m_offset_next) - size_t(m_offset);
    size_t sf_m_offset =
        (size_t(m_offset) + size_t(i) * (alignment_swizzled_mn - 1)) / alignment_swizzled_mn *
        alignment_swizzled_mn;

    auto* problem_sizes = reinterpret_cast<moe_mxfp8::UnderlyingProblemShape*>(problem_sizes_out);
    auto** A_ptr = reinterpret_cast<const ElementA**>(A_ptr_out);
    auto** B_ptr = reinterpret_cast<const ElementB**>(B_ptr_out);
    auto** SFA_ptr = reinterpret_cast<const ElementSFA**>(SFA_ptr_out);
    auto** SFB_ptr = reinterpret_cast<const ElementSFB**>(SFB_ptr_out);
    auto** D_ptr = reinterpret_cast<ElementD**>(D_ptr_out);
    auto* stride_A = reinterpret_cast<moe_mxfp8::StrideA*>(stride_A_out);
    auto* stride_B = reinterpret_cast<moe_mxfp8::StrideB*>(stride_B_out);
    auto* stride_D = reinterpret_cast<moe_mxfp8::StrideD*>(stride_D_out);
    auto* layout_SFA = reinterpret_cast<moe_mxfp8::LayoutSFA*>(layout_SFA_out);
    auto* layout_SFB = reinterpret_cast<moe_mxfp8::LayoutSFB*>(layout_SFB_out);

    problem_sizes[i] = moe_mxfp8::UnderlyingProblemShape(n, int(m), k);
    stride_A[i] = moe_mxfp8::make_packed_stride(moe_mxfp8::StrideA{}, {n, k, 1});
    stride_B[i] = moe_mxfp8::make_packed_stride(moe_mxfp8::StrideB{}, {int(m), k, 1});
    stride_D[i] = moe_mxfp8::make_packed_stride(moe_mxfp8::StrideD{}, {n, int(m), 1});

    A_ptr[i] = reinterpret_cast<const ElementA*>(B_weights + size_t(i) * size_t(n) * size_t(k));
    B_ptr[i] = reinterpret_cast<const ElementB*>(A_acts + size_t(m_offset) * size_t(k));
    D_ptr[i] = reinterpret_cast<ElementD*>(D_out) + size_t(m_offset) * size_t(n);

    layout_SFA[i] = moe_mxfp8::ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(int(sf_n), int(m), int(swizzled_k), 1));
    SFA_ptr[i] = reinterpret_cast<const ElementSFA*>(SFB_weights + size_t(i) * sf_n * sf_k);

    layout_SFB[i] = moe_mxfp8::ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(int(sf_n), int(m), int(swizzled_k), 1));
    SFB_ptr[i] = reinterpret_cast<const ElementSFB*>(SFA_acts + sf_m_offset * sf_k);
  }

#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

namespace {

__device__ __forceinline__ uint4 mxfp8_ld_w_cs(const void* ptr) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(ptr));
  return v;
}

__device__ __forceinline__ uint32_t mxfp8_ld_sf_nc(const void* ptr) {
  uint32_t v;
  asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(v) : "l"(ptr));
  return v;
}

__device__ __forceinline__ void mxfp8_transpose_4x4(
    uint4 v, int tid_g, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  int odd = tid_g & 1;
  uint32_t p0 = __shfl_xor_sync(0xFFFFFFFFu, odd ? v.x : v.y, 1);
  uint32_t p1 = __shfl_xor_sync(0xFFFFFFFFu, odd ? v.z : v.w, 1);
  uint32_t u0 = odd ? p0 : v.x;
  uint32_t u1 = odd ? v.y : p0;
  uint32_t u2 = odd ? p1 : v.z;
  uint32_t u3 = odd ? v.w : p1;
  int hi = (tid_g >> 1) & 1;
  uint32_t q0 = __shfl_xor_sync(0xFFFFFFFFu, hi ? u0 : u2, 2);
  uint32_t q1 = __shfl_xor_sync(0xFFFFFFFFu, hi ? u1 : u3, 2);
  r0 = hi ? q0 : u0;
  r1 = hi ? q1 : u1;
  r2 = hi ? u2 : q0;
  r3 = hi ? u3 : q1;
}

__device__ __forceinline__ void mxfp8_mma_m16n8k32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    uint32_t sfa, uint32_t sfb, uint16_t bid) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
  uint16_t const zero = 0;
  asm volatile(
      "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
      "{%10}, {%11, %12}, {%13}, {%14, %15};\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
        "r"(sfa), "h"(bid), "h"(zero), "r"(sfb), "h"(bid), "h"(zero));
#endif
}

}  // namespace

// Custom SM120 low-M MXFP8 Grouped GEMM for round/decode (T <= 32):
//   - Single-launch persistent kernel (eliminates moe_mxfp8_prepare_args & TMA tensormap updates)
//   - Active-expert compaction directly from m_indptr (zero empty-expert iteration)
//   - Native SM120 block-scaled mma.m16n8k32 (SwapAB: 64 weight rows x up to 32 activation rows)
//   - kMaxRows = 32 guarantees 1.00x DRAM weight reads per active expert for all T <= 32
//   - 3x-unrolled K=384 streaming ld.global.cs.v4.u32 weight loads + L1/L2 cached UE8M0 scales
//   - On-the-fly L1/L2 activation loads keep REG <= 168 (3 CTAs/SM, STACK:0)
//   - Bit-exact BF16 output matching CUTLASS 128x32x128
// Grid: (sm_count * 4, 1, 1), Block: (128, 1, 1)
extern "C" __global__ void __launch_bounds__(128, 2) moe_mxfp8_gemm_lowm(
    const uint8_t* __restrict__ A_acts,
    const uint8_t* __restrict__ B_weights,
    const uint8_t* __restrict__ SFA_acts,
    const uint8_t* __restrict__ SFB_weights,
    __nv_bfloat16* __restrict__ D_out,
    const int* __restrict__ m_indptr,
    int n,
    int k,
    int num_groups) {
  constexpr int kWarps = 4;
  constexpr int kTileM = kWarps * 16;  // 64
  constexpr int kE = 128;
  constexpr int kMaxRows = 32;
  constexpr int kMaxN8 = kMaxRows / 8;  // 4
  constexpr int kUnrollK128 = 3;        // 3 * 128 = 384: exact divisor of K=2304 (6x) and K=1152 (3x)

  __shared__ int s_indptr[kE + 1];
  __shared__ int s_ntile_prefix[kE + 1];
  __shared__ __align__(16) __nv_bfloat16 s_out[kMaxRows][kTileM + 8];

  int const tid = threadIdx.x;
  int const warp = tid >> 5;
  int const lane = tid & 31;
  int const grp = lane >> 2;
  int const tid_g = lane & 3;
  bool const load_sfa = (tid_g < 2);

#if (__CUDA_ARCH__ >= 900)
  asm volatile("griddepcontrol.wait;" ::: "memory");
#endif

  for (int i = tid; i <= kE; i += kWarps * 32) {
    s_indptr[i] = (i <= num_groups) ? m_indptr[i] : m_indptr[num_groups];
  }
  __syncthreads();

  if (warp == 0) {
    int run = 0;
#pragma unroll
    for (int g = 0; g < 4; ++g) {
      int e = g * 32 + lane;
      int cnt = s_indptr[e + 1] - s_indptr[e];
      int nt = (cnt + kMaxRows - 1) >> 5;
      int x = nt;
#pragma unroll
      for (int s = 1; s < 32; s <<= 1) {
        int y = __shfl_up_sync(0xFFFFFFFFu, x, s);
        if (lane >= s) x += y;
      }
      s_ntile_prefix[e] = run + (x - nt);
      run += __shfl_sync(0xFFFFFFFFu, x, 31);
    }
    if (lane == 0) s_ntile_prefix[kE] = run;
  }
  __syncthreads();

  int const total_ntiles = s_ntile_prefix[kE];
  int const m_tiles_per_n = n / kTileM;
  int const total_tiles = total_ntiles * m_tiles_per_n;
  int const sf_n = ((n + 127) >> 7) << 7;
  int const sf_k = (((k + 127) >> 7) << 7) >> 5;  // k / 32
  int const sf_k_div4 = sf_k >> 2;                 // k / 128
  int const num_k128 = k >> 7;

  for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
    int const ntile_global = tile_id / m_tiles_per_n;
    int const m_tile = tile_id - ntile_global * m_tiles_per_n;

    int lo = 0, hi = kE;
    while (hi - lo > 1) {
      int mid = (lo + hi) >> 1;
      if (s_ntile_prefix[mid] <= ntile_global)
        lo = mid;
      else
        hi = mid;
    }
    int const e = lo;
    int const local_nt = ntile_global - s_ntile_prefix[e];
    int const m_off_e = s_indptr[e];
    int const cnt_e = s_indptr[e + 1] - m_off_e;
    int const row_start_local = local_nt * kMaxRows;
    int const rows_this_tile = min(kMaxRows, cnt_e - row_start_local);
    int const row_start_global = m_off_e + row_start_local;
    int const m_base = m_tile * kTileM + warp * 16;

    const uint8_t* W_e = B_weights + int64_t(e) * n * k;
    const uint8_t* SFW_e = SFB_weights + int64_t(e) * sf_n * sf_k;
    int64_t const sf_m_off_e = ((int64_t(m_off_e) + int64_t(e) * 127) >> 7) << 7;
    const uint8_t* SFA_e = SFA_acts + sf_m_off_e * sf_k;

    const uint8_t* W_row0 = W_e + int64_t(m_base + grp) * k + tid_g * 16;
    const uint8_t* W_row1 = W_e + int64_t(m_base + grp + 8) * k + tid_g * 16;
    int const sf_r = m_base + grp + (lane & 1) * 8;
    int64_t const sfa_base = (int64_t(sf_r >> 7) * sf_k_div4) * 512 +
                             (sf_r & 31) * 16 + ((sf_r >> 5) & 3) * 4;

    const uint8_t* B_tile_base = A_acts + int64_t(row_start_global) * k + tid_g * 16;
    int64_t const sfb_tile_off =
        (int64_t(row_start_local >> 7) * sf_k_div4) * 512 +
        ((row_start_local >> 5) & 3) * 4 +
        (row_start_local & 31) * 16 + grp * 16;
    const uint8_t* SFB_tile = SFA_e + sfb_tile_off;

    float acc[kMaxN8][4];
#pragma unroll
    for (int nb = 0; nb < kMaxN8; ++nb) {
      acc[nb][0] = 0.0f;
      acc[nb][1] = 0.0f;
      acc[nb][2] = 0.0f;
      acc[nb][3] = 0.0f;
    }

    for (int ks = 0; ks + kUnrollK128 <= num_k128; ks += kUnrollK128) {
      uint4 w0_lo[kUnrollK128], w0_hi[kUnrollK128], w1_lo[kUnrollK128], w1_hi[kUnrollK128];
      uint32_t sfa_buf[kUnrollK128];

#pragma unroll
      for (int u = 0; u < kUnrollK128; ++u) {
        int kb = (ks + u) * 128;
        int blk = ks + u;
        w0_lo[u] = mxfp8_ld_w_cs(W_row0 + kb);
        w0_hi[u] = mxfp8_ld_w_cs(W_row0 + kb + 64);
        w1_lo[u] = mxfp8_ld_w_cs(W_row1 + kb);
        w1_hi[u] = mxfp8_ld_w_cs(W_row1 + kb + 64);
        sfa_buf[u] = load_sfa ? mxfp8_ld_sf_nc(SFW_e + sfa_base + int64_t(blk) * 512) : 0u;
      }

#pragma unroll
      for (int u = 0; u < kUnrollK128; ++u) {
        int kb = (ks + u) * 128;
        int blk = ks + u;
        uint32_t sfa = __shfl_sync(0xFFFFFFFFu, sfa_buf[u], (lane & ~2));
        uint32_t a0_0, a2_0, a0_1, a2_1, a1_0, a3_0, a1_1, a3_1;
        uint32_t a0_2, a2_2, a0_3, a2_3, a1_2, a3_2, a1_3, a3_3;
        mxfp8_transpose_4x4(w0_lo[u], tid_g, a0_0, a2_0, a0_1, a2_1);
        mxfp8_transpose_4x4(w1_lo[u], tid_g, a1_0, a3_0, a1_1, a3_1);
        mxfp8_transpose_4x4(w0_hi[u], tid_g, a0_2, a2_2, a0_3, a2_3);
        mxfp8_transpose_4x4(w1_hi[u], tid_g, a1_2, a3_2, a1_3, a3_3);
#pragma unroll
        for (int nb = 0; nb < kMaxN8; ++nb) {
          if (nb == 0 || nb * 8 < rows_this_tile) {
            bool const valid = (nb * 8 + grp < rows_this_tile);
            const uint8_t* bptr = B_tile_base + int64_t(nb * 8 + grp) * k + kb;
            uint4 blo = valid ? *reinterpret_cast<const uint4*>(bptr) : make_uint4(0, 0, 0, 0);
            uint4 bhi = valid ? *reinterpret_cast<const uint4*>(bptr + 64) : make_uint4(0, 0, 0, 0);
            uint32_t sfb = valid ? mxfp8_ld_sf_nc(SFB_tile + nb * 128 + int64_t(blk) * 512) : 0u;
            uint32_t b0_0, b1_0, b0_1, b1_1;
            mxfp8_transpose_4x4(blo, tid_g, b0_0, b1_0, b0_1, b1_1);
            mxfp8_mma_m16n8k32(acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3],
                               a0_0, a1_0, a2_0, a3_0, b0_0, b1_0, sfa, sfb, 0);
            mxfp8_mma_m16n8k32(acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3],
                               a0_1, a1_1, a2_1, a3_1, b0_1, b1_1, sfa, sfb, 1);
            mxfp8_transpose_4x4(bhi, tid_g, b0_0, b1_0, b0_1, b1_1);
            mxfp8_mma_m16n8k32(acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3],
                               a0_2, a1_2, a2_2, a3_2, b0_0, b1_0, sfa, sfb, 2);
            mxfp8_mma_m16n8k32(acc[nb][0], acc[nb][1], acc[nb][2], acc[nb][3],
                               a0_3, a1_3, a2_3, a3_3, b0_1, b1_1, sfa, sfb, 3);
          }
        }
      }
    }

    int const n0 = tid_g * 2;
    int const n1 = tid_g * 2 + 1;
#pragma unroll
    for (int nb = 0; nb < kMaxN8; ++nb) {
      if (nb * 8 < rows_this_tile) {
        int r0 = nb * 8 + n0;
        int r1 = nb * 8 + n1;
        if (r0 < rows_this_tile) {
          s_out[r0][warp * 16 + grp] = __float2bfloat16_rn(acc[nb][0]);
          s_out[r0][warp * 16 + grp + 8] = __float2bfloat16_rn(acc[nb][2]);
        }
        if (r1 < rows_this_tile) {
          s_out[r1][warp * 16 + grp] = __float2bfloat16_rn(acc[nb][1]);
          s_out[r1][warp * 16 + grp + 8] = __float2bfloat16_rn(acc[nb][3]);
        }
      }
    }
    __syncthreads();

    constexpr int kU4PerRow = kTileM / 8;
    int const total_u4 = rows_this_tile * kU4PerRow;
    for (int idx = tid; idx < total_u4; idx += kWarps * 32) {
      int r = idx / kU4PerRow;
      int c = (idx - r * kU4PerRow) * 8;
      *reinterpret_cast<uint4*>(D_out + int64_t(row_start_global + r) * n + m_tile * kTileM + c) =
          *reinterpret_cast<const uint4*>(&s_out[r][c]);
    }
    __syncthreads();
  }
#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template __global__ void cutlass::device_kernel<moe_mxfp8::Mxfp8KernelN32>(
    CUTLASS_GRID_CONSTANT typename moe_mxfp8::Mxfp8KernelN32::Params const);
template __global__ void cutlass::device_kernel<moe_mxfp8::Mxfp8KernelN64>(
    CUTLASS_GRID_CONSTANT typename moe_mxfp8::Mxfp8KernelN64::Params const);

#else  // MOE_METADATA_PROBE

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>

template <class Kernel>
static void emit_variant(const char* name, int tile_n, int sm_count) {
  using Params = typename Kernel::Params;
  using Arguments = typename Kernel::Arguments;
  using CollectiveMainloop = typename Kernel::CollectiveMainloop;
  using CollectiveEpilogue = typename Kernel::CollectiveEpilogue;

  auto build_params = [&](int sm, uint8_t* ws_base) -> Params {
    Arguments args{
        cutlass::gemm::GemmUniversalMode::kGrouped,
        {128, nullptr, nullptr},
        {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
        {{}, nullptr, nullptr, nullptr, nullptr},
        cutlass::KernelHardwareInfo{0, sm, 0},
    };
    args.epilogue.thread.alpha = 1.0f;
    args.epilogue.thread.beta = 0.0f;
    return Kernel::to_underlying_arguments(args, ws_base);
  };

  auto paint_stack = [](uint8_t val) __attribute__((noinline)) {
    volatile uint8_t buf[16384];
    for (size_t i = 0; i < sizeof(buf); ++i) buf[i] = val;
  };

  alignas(128) uint8_t dummy_ws[65536] = {};
  g_tma_calls.clear();
  paint_stack(0xAA);
  Params p0 = build_params(sm_count, dummy_ws);
  std::vector<CapturedTmaCall> tmas = g_tma_calls;
  g_tma_calls.clear();
  paint_stack(0x55);
  Params p0_chk = build_params(sm_count, dummy_ws);
  g_tma_calls.clear();
  paint_stack(0xAA);
  Params p1 = build_params(sm_count + 1, dummy_ws);
  auto zero_tma = [](Params& p) {
    std::memset(&p.mainloop.tma_load_a.tma_desc_, 0, 128);
    std::memset(&p.mainloop.tma_load_b.tma_desc_, 0, 128);
    std::memset(&p.mainloop.tma_load_sfa.tma_desc_, 0, 128);
    std::memset(&p.mainloop.tma_load_sfb.tma_desc_, 0, 128);
    std::memset(&p.epilogue.tma_store_d.tma_desc_, 0, 128);
  };
  zero_tma(p0);
  zero_tma(p0_chk);
  zero_tma(p1);

  p0.mainloop.tensormaps = nullptr;
  p0.epilogue.tensormaps = nullptr;
  p0.workspace = nullptr;
  p0_chk.mainloop.tensormaps = nullptr;
  p0_chk.epilogue.tensormaps = nullptr;
  p0_chk.workspace = nullptr;
  p1.workspace = nullptr;

  auto off = [&](const void* field) -> size_t {
    return size_t(reinterpret_cast<const char*>(field) - reinterpret_cast<const char*>(&p0));
  };

  uint8_t* b0 = reinterpret_cast<uint8_t*>(&p0);
  const uint8_t* b0_chk = reinterpret_cast<const uint8_t*>(&p0_chk);
  const uint8_t* b1 = reinterpret_cast<const uint8_t*>(&p1);
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (b0[i] != b0_chk[i]) b0[i] = 0;
  }
  size_t off_sm = off(&p0.hw_info.sm_count);
  size_t off_bap = off(&p0.scheduler.blocks_across_problem_);
  size_t off_mtm = off(&p0.mainloop.tensormaps);
  size_t off_etm = off(&p0.epilogue.tensormaps);
  for (size_t i = 0; i < sizeof(Params); ++i) {
    if (b0_chk[i] != b0[i]) continue;  // was padding
    bool in_sm = (i >= off_sm && i < off_sm + sizeof(p0.hw_info.sm_count));
    bool in_bap = (i >= off_bap && i < off_bap + sizeof(p0.scheduler.blocks_across_problem_));
    bool in_mtm = (i >= off_mtm && i < off_mtm + sizeof(p0.mainloop.tensormaps));
    bool in_etm = (i >= off_etm && i < off_etm + sizeof(p0.epilogue.tensormaps));
    if (!in_sm && !in_bap && !in_mtm && !in_etm && b0[i] != b1[i]) {
      fprintf(stderr, "Unexpected sm_count dependency at byte %zu\n", i);
      abort();
    }
  }

  size_t epi_ws_bytes =
      CollectiveEpilogue::get_workspace_size(p0.problem_shape, {}, sm_count);
  size_t main_ws_bytes =
      CollectiveMainloop::get_workspace_size(p0.problem_shape, {}, sm_count);
  size_t epi_ws_per_sm =
      CollectiveEpilogue::get_workspace_size(p0.problem_shape, {}, 1);
  size_t main_ws_per_sm =
      CollectiveMainloop::get_workspace_size(p0.problem_shape, {}, 1);

  printf("\"%s\":{\n", name);
  printf("  \"tile_m\":128,\"tile_n\":%d,\"tile_k\":128,\n", tile_n);
  printf("  \"shared_mem\":%d,\"max_threads\":%u,\n",
         int(Kernel::SharedStorageSize), unsigned(Kernel::MaxThreadsPerBlock));
  printf("  \"sm_count_default\":%d,\n", sm_count);
  printf("  \"epi_ws_bytes\":%zu,\"main_ws_bytes\":%zu,\n", epi_ws_bytes, main_ws_bytes);
  printf("  \"epi_ws_per_sm\":%zu,\"main_ws_per_sm\":%zu,\n", epi_ws_per_sm, main_ws_per_sm);
  printf("  \"sizeof\":{\n");
  printf("    \"problem_shape\":%zu,\"stride_a\":%zu,\"stride_b\":%zu,\"stride_d\":%zu,\n",
         sizeof(moe_mxfp8::UnderlyingProblemShape), sizeof(moe_mxfp8::StrideA),
         sizeof(moe_mxfp8::StrideB), sizeof(moe_mxfp8::StrideD));
  printf("    \"layout_sfa\":%zu,\"layout_sfb\":%zu\n",
         sizeof(moe_mxfp8::LayoutSFA), sizeof(moe_mxfp8::LayoutSFB));
  printf("  },\n");
  printf("  \"abi\":{\n");
  printf("    \"size\":%zu,\n", sizeof(Params));
  printf("    \"offsets\":{\n");
  printf("      \"problem_shapes\":%zu,\n", off(&p0.problem_shape.problem_shapes));
  printf("      \"scheduler_problem_shapes\":%zu,\n",
         off(&p0.scheduler.problem_shapes_.problem_shapes));
  printf("      \"ptr_A\":%zu,\"dA\":%zu,\"ptr_B\":%zu,\"dB\":%zu,\n",
         off(&p0.mainloop.ptr_A), off(&p0.mainloop.dA),
         off(&p0.mainloop.ptr_B), off(&p0.mainloop.dB));
  printf("      \"ptr_SFA\":%zu,\"layout_SFA\":%zu,\"ptr_SFB\":%zu,\"layout_SFB\":%zu,\n",
         off(&p0.mainloop.ptr_SFA), off(&p0.mainloop.layout_SFA),
         off(&p0.mainloop.ptr_SFB), off(&p0.mainloop.layout_SFB));
  printf("      \"ptr_D\":%zu,\"dD\":%zu,\n",
         off(&p0.epilogue.ptr_D), off(&p0.epilogue.dD));
  printf("      \"mainloop_tensormaps\":%zu,\"epilogue_tensormaps\":%zu,\n", off_mtm, off_etm);
  printf("      \"tma_load_a\":%zu,\"tma_load_b\":%zu,\"tma_load_sfa\":%zu,\"tma_load_sfb\":%zu,\"tma_store_d\":%zu,\n",
         off(&p0.mainloop.tma_load_a.tma_desc_),
         off(&p0.mainloop.tma_load_b.tma_desc_),
         off(&p0.mainloop.tma_load_sfa.tma_desc_),
         off(&p0.mainloop.tma_load_sfb.tma_desc_),
         off(&p0.epilogue.tma_store_d.tma_desc_));
  printf("      \"hw_info_sm_count\":%zu,\"scheduler_blocks_across_problem\":%zu\n",
         off_sm, off_bap);
  printf("    },\n");
  printf("    \"tma_calls\":[");
  for (size_t idx = 0; idx < tmas.size(); ++idx) {
    const auto& c = tmas[idx];
    printf("%s{\"format\":%d,\"rank\":%d,\"interleave\":%d,\"swizzle\":%d,\"l2\":%d,\"oob\":%d,\"dims\":[",
           idx ? "," : "", c.format, c.rank, c.interleave, c.swizzle, c.l2_promotion, c.oob_fill);
    for (int d = 0; d < c.rank; ++d) printf("%s%llu", d ? "," : "", (unsigned long long)c.global_dim[d]);
    printf("],\"strides\":[");
    for (int d = 0; d + 1 < c.rank; ++d) printf("%s%llu", d ? "," : "", (unsigned long long)c.global_strides[d]);
    printf("],\"box\":[");
    for (int d = 0; d < c.rank; ++d) printf("%s%u", d ? "," : "", c.box_dim[d]);
    printf("]}");
  }
  printf("],\n");
  printf("    \"bytes\":[");
  for (size_t i = 0; i < sizeof(Params); ++i) {
    printf("%s%u", i ? "," : "", unsigned(b0[i]));
  }
  printf("]\n  }\n}");
}

int main(int argc, char** argv) {
  int sm_count = (argc > 1) ? atoi(argv[1]) : 110;
  printf("{\n");
  emit_variant<moe_mxfp8::Mxfp8KernelN32>("n32", 32, sm_count);
  printf(",\n");
  emit_variant<moe_mxfp8::Mxfp8KernelN64>("n64", 64, sm_count);
  printf("\n}\n");
  return 0;
}

#endif  // MOE_METADATA_PROBE
