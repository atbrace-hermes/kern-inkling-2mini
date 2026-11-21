// SPDX-License-Identifier: Apache-2.0
// SM120 NVFP4 Grouped GEMM adapter for Inkling-2-Mini MoE (Layers 0..19):
//   - SwapAB=true, TileM=128, TileN in {32, 64, 128}, TileK=128
//   - Fused EVT epilogue: D_bf16 = bf16(acc_f32 * (x_amax[row] * (w_amax[expert] / (1536 * w_denom))))
//     eliminating FP32 intermediate buffers (2x less output/input DRAM traffic, half epilogue SMEM,
//     +1 mainloop pipeline stage for N32/N64, and bit-exact BF16 output).
//   - Compile with -DMOE_METADATA_PROBE as host binary to emit build/sm_120/moe_nvfp4_sm120.json
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
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp>
#include <cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

namespace moe_nvfp4 {

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

}  // namespace moe_nvfp4

namespace moe_nvfp4 {

using namespace cute;

template <int TileN>
struct Nvfp4Traits {
  static constexpr int TileM = 128;
  static constexpr int TileK = 128;
  static constexpr bool SwapAB = true;

  using ElementA = cutlass::float_e2m1_t;
  using ElementSFA = cutlass::float_ue4m3_t;
  static constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 32

  using ElementB = cutlass::float_e2m1_t;
  using ElementSFB = cutlass::float_ue4m3_t;
  static constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 32

  using ElementD = cutlass::bfloat16_t;
  using ElementC = void;
  static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 8

  using ElementAccumulator = float;
  using ElementCompute = float;

  using ElementAMainloop = cutlass::nv_float4_t<ElementA>;
  using ElementBMainloop = cutlass::nv_float4_t<ElementB>;
  using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD = cutlass::layout::ColumnMajor;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;

  // Fused exact-order scaling: D_bf16 = bf16(acc * ((x_amax[row] * w_amax[expert]) * rcp))
  // Matches baseline `__fmul_rn(__fmul_rn(x_amax[row], w_amax[expert]), rcp)` bit-for-bit!
  using RowScaleNode = cutlass::epilogue::fusion::Sm90RowBroadcast<
      0, ThreadBlockShape, float*, float, Stride<_0, _1, _0>, 1, false>;
  using GroupAmaxNode = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
      float, Stride<_0, _0, _1>>;
  using RcpNode = cutlass::epilogue::fusion::Sm90ScalarBroadcast<
      float, Stride<_0, _0, _1>>;
  using AmaxMul = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
      RowScaleNode,
      GroupAmaxNode>;
  using ScaleMul = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>,
      AmaxMul,
      RcpNode>;
  using EpilogueEVT = cutlass::epilogue::fusion::Sm90EVT<
      cutlass::epilogue::fusion::Sm90Compute<
          cutlass::multiplies, ElementD, float, cutlass::FloatRoundStyle::round_to_nearest>,
      cutlass::epilogue::fusion::Sm90AccFetch,
      ScaleMul>;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ThreadBlockShape,
      ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
      ElementCompute, ElementC, LayoutD*, AlignmentD, ElementD, LayoutD*, AlignmentD,
      EpilogueSchedule, EpilogueEVT>::CollectiveOp;

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

struct Nvfp4KernelN32 : public Nvfp4Traits<32>::BaseKernel {};
struct Nvfp4KernelN64 : public Nvfp4Traits<64>::BaseKernel {};
struct Nvfp4KernelN128 : public Nvfp4Traits<128>::BaseKernel {};

using TraitsRef = Nvfp4Traits<64>;
using ProblemShape = typename TraitsRef::ProblemShape;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
using StrideA = typename TraitsRef::BaseKernel::InternalStrideA;
using StrideB = typename TraitsRef::BaseKernel::InternalStrideB;
using StrideD = typename TraitsRef::BaseKernel::InternalStrideD;
using ScaleConfig = typename TraitsRef::BaseKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using LayoutSFA = typename TraitsRef::BaseKernel::CollectiveMainloop::InternalLayoutSFA;
using LayoutSFB = typename TraitsRef::BaseKernel::CollectiveMainloop::InternalLayoutSFB;

static_assert(std::is_same_v<typename Nvfp4Traits<32>::BaseKernel::InternalStrideA, StrideA>);
static_assert(std::is_same_v<typename Nvfp4Traits<32>::BaseKernel::InternalStrideB, StrideB>);
static_assert(std::is_same_v<typename Nvfp4Traits<32>::BaseKernel::InternalStrideD, StrideD>);
static_assert(std::is_same_v<typename Nvfp4Traits<128>::BaseKernel::InternalStrideA, StrideA>);
static_assert(std::is_same_v<typename Nvfp4Traits<128>::BaseKernel::InternalStrideB, StrideB>);
static_assert(std::is_same_v<typename Nvfp4Traits<128>::BaseKernel::InternalStrideD, StrideD>);
static_assert(
    std::is_same_v<typename Nvfp4Traits<32>::BaseKernel::CollectiveMainloop::InternalLayoutSFA,
                   LayoutSFA>);
static_assert(
    std::is_same_v<typename Nvfp4Traits<32>::BaseKernel::CollectiveMainloop::InternalLayoutSFB,
                   LayoutSFB>);
static_assert(
    std::is_same_v<typename Nvfp4Traits<128>::BaseKernel::CollectiveMainloop::InternalLayoutSFA,
                   LayoutSFA>);
static_assert(
    std::is_same_v<typename Nvfp4Traits<128>::BaseKernel::CollectiveMainloop::InternalLayoutSFB,
                   LayoutSFB>);
static_assert(TraitsRef::BaseKernel::CollectiveMainloop::TiledMma::SFVecSize == 16);
static_assert(Nvfp4KernelN32::SharedStorageSize <= 101376, "N32 exceeds SM120 99KiB SMEM");
static_assert(Nvfp4KernelN64::SharedStorageSize <= 101376, "N64 exceeds SM120 99KiB SMEM");
static_assert(Nvfp4KernelN128::SharedStorageSize <= 101376, "N128 exceeds SM120 99KiB SMEM");

}  // namespace moe_nvfp4

#ifndef MOE_METADATA_PROBE

// Device arg-prep kernel for SwapAB=true NVFP4 grouped GEMM with fused BF16 scaling.
// Note: A_acts [m, k/2] u8, B_weights [E, n, k/2] u8, SFA_acts u8, SFB_weights u8,
//       D_out bf16 [m, n], x_amax f32 [m], w_amax f32 [E], w_denom f32 [1].
extern "C" __global__ void __launch_bounds__(128) moe_nvfp4_prepare_args(
    const uint8_t* __restrict__ A_acts,
    const uint8_t* __restrict__ B_weights,
    const uint8_t* __restrict__ SFA_acts,
    const uint8_t* __restrict__ SFB_weights,
    __nv_bfloat16* __restrict__ D_out,
    const int* __restrict__ m_indptr,
    const float* __restrict__ x_amax,
    const float* __restrict__ w_amax,
    const float* __restrict__ w_denom,
    int n,
    int k,
    int num_groups,
    void* problem_sizes_out,
    void* A_ptr_out,
    void* B_ptr_out,
    void* SFA_ptr_out,
    void* SFB_ptr_out,
    void* D_ptr_out,
    void* x_amax_ptr_out,
    float* group_scale_out,
    void* stride_A_out,
    void* stride_B_out,
    void* stride_D_out,
    void* layout_SFA_out,
    void* layout_SFB_out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  using ElementA = typename moe_nvfp4::TraitsRef::ElementA;      // float_e2m1_t (weights)
  using ElementB = typename moe_nvfp4::TraitsRef::ElementB;      // float_e2m1_t (acts)
  using ElementSFA = typename moe_nvfp4::TraitsRef::ElementSFA;  // float_ue4m3_t (weights)
  using ElementSFB = typename moe_nvfp4::TraitsRef::ElementSFB;  // float_ue4m3_t (acts)
  using ElementD = typename moe_nvfp4::TraitsRef::ElementD;      // cutlass::bfloat16_t

  constexpr size_t ScaleGranularity = 16;
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

    auto* problem_sizes = reinterpret_cast<moe_nvfp4::UnderlyingProblemShape*>(problem_sizes_out);
    auto** A_ptr = reinterpret_cast<const ElementA**>(A_ptr_out);
    auto** B_ptr = reinterpret_cast<const ElementB**>(B_ptr_out);
    auto** SFA_ptr = reinterpret_cast<const ElementSFA**>(SFA_ptr_out);
    auto** SFB_ptr = reinterpret_cast<const ElementSFB**>(SFB_ptr_out);
    auto** D_ptr = reinterpret_cast<ElementD**>(D_ptr_out);
    auto** x_amax_ptr = reinterpret_cast<const float**>(x_amax_ptr_out);
    auto* stride_A = reinterpret_cast<moe_nvfp4::StrideA*>(stride_A_out);
    auto* stride_B = reinterpret_cast<moe_nvfp4::StrideB*>(stride_B_out);
    auto* stride_D = reinterpret_cast<moe_nvfp4::StrideD*>(stride_D_out);
    auto* layout_SFA = reinterpret_cast<moe_nvfp4::LayoutSFA*>(layout_SFA_out);
    auto* layout_SFB = reinterpret_cast<moe_nvfp4::LayoutSFB*>(layout_SFB_out);

    problem_sizes[i] = moe_nvfp4::UnderlyingProblemShape(n, int(m), k);
    stride_A[i] = moe_nvfp4::make_packed_stride(moe_nvfp4::StrideA{}, {n, k, 1});
    stride_B[i] = moe_nvfp4::make_packed_stride(moe_nvfp4::StrideB{}, {int(m), k, 1});
    stride_D[i] = moe_nvfp4::make_packed_stride(moe_nvfp4::StrideD{}, {n, int(m), 1});

    // Packed 2x E2M1 per byte -> divide element offset by 2.
    A_ptr[i] = reinterpret_cast<const ElementA*>(B_weights + (size_t(i) * size_t(n) * size_t(k)) / 2);
    B_ptr[i] = reinterpret_cast<const ElementB*>(A_acts + (size_t(m_offset) * size_t(k)) / 2);
    D_ptr[i] = reinterpret_cast<ElementD*>(D_out) + size_t(m_offset) * size_t(n);

    x_amax_ptr[i] = (m > 0) ? (x_amax + m_offset) : x_amax;
    group_scale_out[i] = __frcp_rn(__fmul_rn(1536.0f, *w_denom));

    layout_SFA[i] = moe_nvfp4::ScaleConfig::tile_atom_to_shape_SFA(
        cute::make_shape(int(sf_n), int(m), int(swizzled_k), 1));
    SFA_ptr[i] = reinterpret_cast<const ElementSFA*>(SFB_weights + size_t(i) * sf_n * sf_k);

    layout_SFB[i] = moe_nvfp4::ScaleConfig::tile_atom_to_shape_SFB(
        cute::make_shape(int(sf_n), int(m), int(swizzled_k), 1));
    SFB_ptr[i] = reinterpret_cast<const ElementSFB*>(SFA_acts + sf_m_offset * sf_k);
  }

#if (__CUDA_ARCH__ >= 900)
  __syncthreads();
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template __global__ void cutlass::device_kernel<moe_nvfp4::Nvfp4KernelN32>(
    CUTLASS_GRID_CONSTANT typename moe_nvfp4::Nvfp4KernelN32::Params const);
template __global__ void cutlass::device_kernel<moe_nvfp4::Nvfp4KernelN64>(
    CUTLASS_GRID_CONSTANT typename moe_nvfp4::Nvfp4KernelN64::Params const);
template __global__ void cutlass::device_kernel<moe_nvfp4::Nvfp4KernelN128>(
    CUTLASS_GRID_CONSTANT typename moe_nvfp4::Nvfp4KernelN128::Params const);

#else  // MOE_METADATA_PROBE

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
    args.epilogue.thread.op_1.op_0.op_0.ptr_row = nullptr;
    args.epilogue.thread.op_1.op_0.op_0.dRow = cute::Stride<cute::_0, cute::_1, cute::_0>{};
    args.epilogue.thread.op_1.op_0.op_1.scalar_ptrs[0] = nullptr;
    args.epilogue.thread.op_1.op_0.op_1.dScalar[0] = cute::Stride<cute::_0, cute::_0, cute::_1>{};
    args.epilogue.thread.op_1.op_1.scalar_ptrs[0] = nullptr;
    args.epilogue.thread.op_1.op_1.dScalar[0] = cute::Stride<cute::_0, cute::_0, cute::_1>{};
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

  // Zero out workspace pointers in p0 before dumping template bytes
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

  // Clear uninitialized C++ struct padding bytes (those that track stack paint 0xAA vs 0x55)
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
         sizeof(moe_nvfp4::UnderlyingProblemShape), sizeof(moe_nvfp4::StrideA),
         sizeof(moe_nvfp4::StrideB), sizeof(moe_nvfp4::StrideD));
  printf("    \"layout_sfa\":%zu,\"layout_sfb\":%zu\n",
         sizeof(moe_nvfp4::LayoutSFA), sizeof(moe_nvfp4::LayoutSFB));
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
  printf("      \"evt_row_scale_ptrs\":%zu,\"evt_group_amax_ptr\":%zu,\"evt_rcp_ptr\":%zu,\n",
         off(&p0.epilogue.thread.op_1.op_0.op_0.ptr_row),
         off(&p0.epilogue.thread.op_1.op_0.op_1.scalar_ptrs[0]),
         off(&p0.epilogue.thread.op_1.op_1.scalar_ptrs[0]));
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
  emit_variant<moe_nvfp4::Nvfp4KernelN32>("n32", 32, sm_count);
  printf(",\n");
  emit_variant<moe_nvfp4::Nvfp4KernelN64>("n64", 64, sm_count);
  printf(",\n");
  emit_variant<moe_nvfp4::Nvfp4KernelN128>("n128", 128, sm_count);
  printf("\n}\n");
  return 0;
}

#endif  // MOE_METADATA_PROBE
