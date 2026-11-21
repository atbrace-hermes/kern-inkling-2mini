// SPDX-License-Identifier: Apache-2.0
// 4-stream short convolution (W=4) over states.seq_state for Inkling-2-Mini (SM120).
// Streams per logical visit (39,936 B = 3 chronological BF16 taps per stream):
//   0: k_sconv   (C=1024, byte offset     0, verify_raw col offset    0)
//   1: v_sconv   (C=1024, byte offset  6144, verify_raw col offset 1024)
//   2: attn_sconv(C=2304, byte offset 12288, verify_raw col offset 2048)
//   3: mlp_sconv (C=2304, byte offset 26112, verify_raw col offset 4352)

#include <cuda_bf16.h>
#include <cstdint>

using bf16 = __nv_bfloat16;

constexpr int kVec = 8;
constexpr int kThreads = 128;

constexpr int64_t kSeqStateBytes = 90005376LL;
constexpr int64_t kSconvVisitBytes = 39936LL;
constexpr int64_t kVerifyRawRowElems = 6656LL;

constexpr float kStaticScale = 0.4472135901451111f;   // 5^-0.5 rounded to fp32
constexpr float kDynamicScale = 0.3333333432674408f;  // 9^-0.5 rounded to fp32

struct Vec8 {
  uint4 raw;

  __device__ __forceinline__ void load(const bf16* p) {
    raw = *reinterpret_cast<const uint4*>(p);
  }
  __device__ __forceinline__ void store(bf16* p) const {
    *reinterpret_cast<uint4*>(p) = raw;
  }
  __device__ __forceinline__ void zero() {
    raw = make_uint4(0u, 0u, 0u, 0u);
  }
  __device__ __forceinline__ float operator[](int i) const {
    const bf16* h = reinterpret_cast<const bf16*>(&raw);
    return __bfloat162float(h[i]);
  }
  __device__ __forceinline__ void set(const float (&v)[kVec]) {
    bf16* h = reinterpret_cast<bf16*>(&raw);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      h[i] = __float2bfloat16_rn(v[i]);
    }
  }
};

__device__ __forceinline__ float round_bf16(float v) {
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ void load_static_w8(const bf16* w, int col, float (&wt)[4][kVec]) {
  // w is [C, 4] BF16 row-major. Load 8 * 4 = 32 BF16 = 4 uint4s.
  const uint4* w4 = reinterpret_cast<const uint4*>(w + (int64_t)col * 4);
#pragma unroll
  for (int chunk = 0; chunk < 4; ++chunk) {
    uint4 r = w4[chunk];
    const bf16* h = reinterpret_cast<const bf16*>(&r);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      int idx = chunk * 8 + j;
      wt[idx & 3][idx >> 2] = __bfloat162float(h[j]);
    }
  }
}

__device__ __forceinline__ void load_dynamic_w8(
    const bf16* static_w, const bf16* dyn, int row, int col, float (&wt)[4][kVec]) {
  // static_w is [4, 1024] BF16; dyn is [T, 8, 4] BF16.
  const int head = col >> 7;  // col / 128
  const bf16* dyn_row = dyn + (int64_t)row * 32 + head * 4;
#pragma unroll
  for (int lag = 0; lag < 4; ++lag) {
    Vec8 sw;
    sw.load(static_w + (int64_t)lag * 1024 + col);
    float d = __bfloat162float(dyn_row[lag]);
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      wt[lag][i] = __fadd_rn(sw[i], d);
    }
  }
}

__device__ __forceinline__ void eval_static_conv(
    const Vec8& cur,
    const Vec8 (&taps)[4],
    const float (&wt)[4][kVec],
    float (&y)[kVec]) {
  float acc[kVec] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    // Oldest tap first: k=0 -> lag=3, k=1 -> lag=2, k=2 -> lag=1, k=3 -> lag=0.
    int lag = 3 - k;
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      acc[i] = __fmaf_rn(taps[lag][i], wt[k][i], acc[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    y[i] = __fmul_rn(__fadd_rn(acc[i], cur[i]), kStaticScale);
  }
}

__device__ __forceinline__ void eval_dynamic_conv(
    const Vec8& cur,
    const Vec8 (&taps)[4],
    const float (&wt)[4][kVec],
    float (&y)[kVec]) {
  float acc[kVec];
#pragma unroll
  for (int lag = 0; lag < 4; ++lag) {
    // Newest tap first: lag = 0..3; lag==0 uses non-FMA __fmul_rn.
#pragma unroll
    for (int i = 0; i < kVec; ++i) {
      acc[i] = (lag == 0) ? __fmul_rn(taps[lag][i], wt[lag][i])
                          : __fmaf_rn(taps[lag][i], wt[lag][i], acc[i]);
    }
  }
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    y[i] = __fmul_rn(__fadd_rn(acc[i], cur[i]), kDynamicScale);
  }
}

__device__ __forceinline__ void apply_mlp_epilogue(
    const float (&y)[kVec],
    const bf16* res,
    const bf16* addend,
    int64_t row_off,
    int has_res,
    Vec8& out_vec) {
  Vec8 a;
  a.load(addend + row_off);
  Vec8 r;
  if (has_res) r.load(res + row_off);
  float out_f[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    float y_bf = round_bf16(y[i]);
    float acc = a[i];
    if (has_res) acc = __fadd_rn(r[i], acc);
    out_f[i] = __fadd_rn(acc, y_bf);
  }
  out_vec.set(out_f);
}

// ---------------------------------------------------------------------------
// 1. Static Prefill (read-only on seq_state)
// Grid: [tokens, ceil_div(C, 1024), 1], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_static_prefill(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    const void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    bf16* __restrict__ out,
    int C,
    int stride_x,
    int tokens,
    int64_t visit_stream_off) {
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  const int row = blockIdx.x;
  if (row >= tokens || col >= C) return;

  float wt[4][kVec];
  load_static_w8(w, col, wt);
  const int line = lines[0];
  const int64_t p = pos[row];
  const bf16* hist = (line > 0)
      ? reinterpret_cast<const bf16*>(static_cast<const char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_taps[4];
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (hist != nullptr && (p - lag >= 0) && (row - lag < 0)) {
      hist_taps[lag].load(hist + (int64_t)(3 + row - lag) * C + col);
    } else {
      hist_taps[lag].zero();
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  Vec8 taps[4];
  taps[0].load(x + (int64_t)row * stride_x + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else if (row - lag >= 0) {
      taps[lag].load(x + (int64_t)(row - lag) * stride_x + col);
    } else {
      taps[lag] = hist_taps[lag];
    }
  }

  float y[kVec];
  eval_static_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  out_vec.set(y);
  out_vec.store(out + (int64_t)row * C + col);
}

extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_static_mlp_prefill(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    const void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    bf16* __restrict__ out,
    const bf16* __restrict__ res,
    const bf16* __restrict__ addend,
    int C,
    int stride_x,
    int tokens,
    int64_t visit_stream_off,
    int has_res) {
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  const int row = blockIdx.x;
  if (row >= tokens || col >= C) return;

  float wt[4][kVec];
  load_static_w8(w, col, wt);
  const int line = lines[0];
  const int64_t p = pos[row];
  const bf16* hist = (line > 0)
      ? reinterpret_cast<const bf16*>(static_cast<const char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_taps[4];
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (hist != nullptr && (p - lag >= 0) && (row - lag < 0)) {
      hist_taps[lag].load(hist + (int64_t)(3 + row - lag) * C + col);
    } else {
      hist_taps[lag].zero();
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  Vec8 taps[4];
  taps[0].load(x + (int64_t)row * stride_x + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else if (row - lag >= 0) {
      taps[lag].load(x + (int64_t)(row - lag) * stride_x + col);
    } else {
      taps[lag] = hist_taps[lag];
    }
  }

  float y[kVec];
  eval_static_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  apply_mlp_epilogue(y, res, addend, (int64_t)row * C + col, has_res, out_vec);
  out_vec.store(out + (int64_t)row * C + col);
}

// ---------------------------------------------------------------------------
// 2. Prefill Commit (shifts last 3 raw rows into seq_state)
// Grid: [ceil_div(C, 1024), 1, 1], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_prefill_commit(
    const bf16* __restrict__ x,
    void* __restrict__ seq_state,
    const int* __restrict__ lines,
    int C,
    int stride_x,
    int tokens,
    int64_t visit_stream_off) {
  const int col = (blockIdx.x * kThreads + threadIdx.x) * kVec;
  if (col >= C || tokens <= 0) return;
  const int line = lines[0];
  if (line <= 0) return;

  bf16* hist = reinterpret_cast<bf16*>(
      static_cast<char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off);

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  Vec8 v[3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    int t = tokens - 3 + j;
    if (t < 0) {
      v[j].load(hist + (int64_t)(t + 3) * C + col);
    } else {
      v[j].load(x + (int64_t)t * stride_x + col);
    }
  }
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    v[j].store(hist + (int64_t)j * C + col);
  }
}

// ---------------------------------------------------------------------------
// 3. Static Decode / Decode-Batch (updates seq_state in place)
// Grid: [seqs, ceil_div(C, 1024), 1], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_static_decode(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    const int* __restrict__ valid,
    bf16* __restrict__ out,
    int C,
    int stride_x,
    int64_t visit_stream_off) {
  const int seq = blockIdx.x;
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (col >= C) return;

  const int line = lines[seq];
  const int vflag = valid[seq];
  float wt[4][kVec];
  if (line > 0 && vflag != 0) load_static_w8(w, col, wt);
  const int64_t p = pos[seq];
  bf16* hist = (line > 0)
      ? reinterpret_cast<bf16*>(static_cast<char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_raw[3];
  if (hist != nullptr && vflag != 0) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      hist_raw[j].load(hist + (int64_t)j * C + col);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 3; ++j) hist_raw[j].zero();
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (line <= 0 || vflag == 0) {
    Vec8 z;
    z.zero();
    z.store(out + (int64_t)seq * C + col);
    return;
  }

  Vec8 taps[4];
  taps[0].load(x + (int64_t)seq * stride_x + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else {
      taps[lag] = hist_raw[3 - lag];
    }
  }

  float y[kVec];
  eval_static_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  out_vec.set(y);
  out_vec.store(out + (int64_t)seq * C + col);

  // Shift chronological history [oldest, mid, newest].
  hist_raw[1].store(hist + 0LL * C + col);
  hist_raw[2].store(hist + 1LL * C + col);
  taps[0].store(hist + 2LL * C + col);
}

extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_static_mlp_decode(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    const int* __restrict__ valid,
    bf16* __restrict__ out,
    const bf16* __restrict__ res,
    const bf16* __restrict__ addend,
    int C,
    int stride_x,
    int64_t visit_stream_off,
    int has_res) {
  const int seq = blockIdx.x;
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (col >= C) return;

  const int line = lines[seq];
  const int vflag = valid[seq];
  float wt[4][kVec];
  if (line > 0 && vflag != 0) load_static_w8(w, col, wt);
  const int64_t p = pos[seq];
  bf16* hist = (line > 0)
      ? reinterpret_cast<bf16*>(static_cast<char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_raw[3];
  if (hist != nullptr && vflag != 0) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      hist_raw[j].load(hist + (int64_t)j * C + col);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 3; ++j) hist_raw[j].zero();
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (line <= 0 || vflag == 0) {
    Vec8 z;
    z.zero();
    z.store(out + (int64_t)seq * C + col);
    return;
  }

  Vec8 taps[4];
  taps[0].load(x + (int64_t)seq * stride_x + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else {
      taps[lag] = hist_raw[3 - lag];
    }
  }

  float y[kVec];
  eval_static_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  apply_mlp_epilogue(y, res, addend, (int64_t)seq * C + col, has_res, out_vec);
  out_vec.store(out + (int64_t)seq * C + col);

  hist_raw[1].store(hist + 0LL * C + col);
  hist_raw[2].store(hist + 1LL * C + col);
  taps[0].store(hist + 2LL * C + col);
}

// ---------------------------------------------------------------------------
// 4. Static Verify (read-only on seq_state, saves cur_x to verify_raw)
// Grid: [tokens, ceil_div(C, 1024), 1], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_static_verify(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    const void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    const int* __restrict__ valid,
    bf16* __restrict__ out,
    bf16* __restrict__ verify_raw,
    int C,
    int stride_x,
    int64_t visit_stream_off,
    int64_t visit_raw_base,
    int stream_raw_col_off) {
  const int row = blockIdx.x;
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (col >= C) return;

  const int seq = row >> 3;
  const int r = row & 7;
  const int bos = seq << 3;
  const int line = lines[seq];
  const int vflag = valid[row];

  float wt[4][kVec];
  if (line > 0 && vflag != 0) load_static_w8(w, col, wt);
  const int64_t p = pos[row];
  const bf16* hist = (line > 0)
      ? reinterpret_cast<const bf16*>(static_cast<const char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_taps[4];
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (hist != nullptr && vflag != 0 && (p - lag >= 0) && (r - lag < 0)) {
      hist_taps[lag].load(hist + (int64_t)(3 + r - lag) * C + col);
    } else {
      hist_taps[lag].zero();
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (line <= 0 || vflag == 0) {
    Vec8 z;
    z.zero();
    z.store(out + (int64_t)row * C + col);
    return;
  }

  Vec8 taps[4];
  taps[0].load(x + (int64_t)row * stride_x + col);
  taps[0].store(verify_raw + visit_raw_base + (int64_t)row * kVerifyRawRowElems + stream_raw_col_off + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else if (r - lag >= 0) {
      taps[lag].load(x + (int64_t)(bos + r - lag) * stride_x + col);
    } else {
      taps[lag] = hist_taps[lag];
    }
  }

  float y[kVec];
  eval_static_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  out_vec.set(y);
  out_vec.store(out + (int64_t)row * C + col);
}

extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_static_mlp_verify(
    const bf16* __restrict__ x,
    const bf16* __restrict__ w,
    const void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    const int* __restrict__ valid,
    bf16* __restrict__ out,
    bf16* __restrict__ verify_raw,
    const bf16* __restrict__ res,
    const bf16* __restrict__ addend,
    int C,
    int stride_x,
    int64_t visit_stream_off,
    int64_t visit_raw_base,
    int stream_raw_col_off,
    int has_res) {
  const int row = blockIdx.x;
  const int col = (blockIdx.y * kThreads + threadIdx.x) * kVec;
  if (col >= C) return;

  const int seq = row >> 3;
  const int r = row & 7;
  const int bos = seq << 3;
  const int line = lines[seq];
  const int vflag = valid[row];

  float wt[4][kVec];
  if (line > 0 && vflag != 0) load_static_w8(w, col, wt);
  const int64_t p = pos[row];
  const bf16* hist = (line > 0)
      ? reinterpret_cast<const bf16*>(static_cast<const char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_taps[4];
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (hist != nullptr && vflag != 0 && (p - lag >= 0) && (r - lag < 0)) {
      hist_taps[lag].load(hist + (int64_t)(3 + r - lag) * C + col);
    } else {
      hist_taps[lag].zero();
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (line <= 0 || vflag == 0) {
    Vec8 z;
    z.zero();
    z.store(out + (int64_t)row * C + col);
    return;
  }

  Vec8 taps[4];
  taps[0].load(x + (int64_t)row * stride_x + col);
  taps[0].store(verify_raw + visit_raw_base + (int64_t)row * kVerifyRawRowElems + stream_raw_col_off + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else if (r - lag >= 0) {
      taps[lag].load(x + (int64_t)(bos + r - lag) * stride_x + col);
    } else {
      taps[lag] = hist_taps[lag];
    }
  }

  float y[kVec];
  eval_static_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  apply_mlp_epilogue(y, res, addend, (int64_t)row * C + col, has_res, out_vec);
  out_vec.store(out + (int64_t)row * C + col);
}

// ---------------------------------------------------------------------------
// 5. Dynamic Prefill / Decode / Verify (C = 1024)
// Grid: [tokens_or_seqs, 1, 1], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_dynamic_prefill(
    const bf16* __restrict__ x,
    const bf16* __restrict__ static_w,
    const bf16* __restrict__ dyn,
    const void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    bf16* __restrict__ out,
    int stride_x,
    int tokens,
    int64_t visit_stream_off) {
  constexpr int C = 1024;
  const int row = blockIdx.x;
  const int col = threadIdx.x * kVec;
  if (row >= tokens || col >= C) return;

  const int line = lines[0];
  const int64_t p = pos[row];
  const bf16* hist = (line > 0)
      ? reinterpret_cast<const bf16*>(static_cast<const char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_taps[4];
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (hist != nullptr && (p - lag >= 0) && (row - lag < 0)) {
      hist_taps[lag].load(hist + (int64_t)(3 + row - lag) * C + col);
    } else {
      hist_taps[lag].zero();
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  float wt[4][kVec];
  load_dynamic_w8(static_w, dyn, row, col, wt);

  Vec8 taps[4];
  taps[0].load(x + (int64_t)row * stride_x + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else if (row - lag >= 0) {
      taps[lag].load(x + (int64_t)(row - lag) * stride_x + col);
    } else {
      taps[lag] = hist_taps[lag];
    }
  }

  float y[kVec];
  eval_dynamic_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  out_vec.set(y);
  out_vec.store(out + (int64_t)row * C + col);
}

extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_dynamic_decode(
    const bf16* __restrict__ x,
    const bf16* __restrict__ static_w,
    const bf16* __restrict__ dyn,
    void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    const int* __restrict__ valid,
    bf16* __restrict__ out,
    int stride_x,
    int64_t visit_stream_off) {
  constexpr int C = 1024;
  const int seq = blockIdx.x;
  const int col = threadIdx.x * kVec;
  if (col >= C) return;

  const int line = lines[seq];
  const int vflag = valid[seq];
  const int64_t p = pos[seq];
  bf16* hist = (line > 0)
      ? reinterpret_cast<bf16*>(static_cast<char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_raw[3];
  if (hist != nullptr && vflag != 0) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      hist_raw[j].load(hist + (int64_t)j * C + col);
    }
  } else {
#pragma unroll
    for (int j = 0; j < 3; ++j) hist_raw[j].zero();
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (line <= 0 || vflag == 0) {
    Vec8 z;
    z.zero();
    z.store(out + (int64_t)seq * C + col);
    return;
  }

  float wt[4][kVec];
  load_dynamic_w8(static_w, dyn, seq, col, wt);

  Vec8 taps[4];
  taps[0].load(x + (int64_t)seq * stride_x + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else {
      taps[lag] = hist_raw[3 - lag];
    }
  }

  float y[kVec];
  eval_dynamic_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  out_vec.set(y);
  out_vec.store(out + (int64_t)seq * C + col);

  hist_raw[1].store(hist + 0LL * C + col);
  hist_raw[2].store(hist + 1LL * C + col);
  taps[0].store(hist + 2LL * C + col);
}

extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_dynamic_verify(
    const bf16* __restrict__ x,
    const bf16* __restrict__ static_w,
    const bf16* __restrict__ dyn,
    const void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int64_t* __restrict__ pos,
    const int* __restrict__ valid,
    bf16* __restrict__ out,
    bf16* __restrict__ verify_raw,
    int stride_x,
    int64_t visit_stream_off,
    int64_t visit_raw_base,
    int stream_raw_col_off) {
  constexpr int C = 1024;
  const int row = blockIdx.x;
  const int col = threadIdx.x * kVec;
  if (col >= C) return;

  const int seq = row >> 3;
  const int r = row & 7;
  const int bos = seq << 3;
  const int line = lines[seq];
  const int vflag = valid[row];
  const int64_t p = pos[row];
  const bf16* hist = (line > 0)
      ? reinterpret_cast<const bf16*>(static_cast<const char*>(seq_state) + (int64_t)line * kSeqStateBytes + visit_stream_off)
      : nullptr;

  Vec8 hist_taps[4];
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (hist != nullptr && vflag != 0 && (p - lag >= 0) && (r - lag < 0)) {
      hist_taps[lag].load(hist + (int64_t)(3 + r - lag) * C + col);
    } else {
      hist_taps[lag].zero();
    }
  }

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

  if (line <= 0 || vflag == 0) {
    Vec8 z;
    z.zero();
    z.store(out + (int64_t)row * C + col);
    return;
  }

  float wt[4][kVec];
  load_dynamic_w8(static_w, dyn, row, col, wt);

  Vec8 taps[4];
  taps[0].load(x + (int64_t)row * stride_x + col);
  taps[0].store(verify_raw + visit_raw_base + (int64_t)row * kVerifyRawRowElems + stream_raw_col_off + col);
#pragma unroll
  for (int lag = 1; lag < 4; ++lag) {
    if (p - lag < 0) {
      taps[lag].zero();
    } else if (r - lag >= 0) {
      taps[lag].load(x + (int64_t)(bos + r - lag) * stride_x + col);
    } else {
      taps[lag] = hist_taps[lag];
    }
  }

  float y[kVec];
  eval_dynamic_conv(taps[0], taps, wt, y);
  Vec8 out_vec;
  out_vec.set(y);
  out_vec.store(out + (int64_t)row * C + col);
}

// ---------------------------------------------------------------------------
// 6. Speculative Commit All (48 visits x 4 streams in 1 launch)
// Grid: [seqs, 48, 3], Block: [128, 1, 1]
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(kThreads) kern_sconv_spec_commit_all(
    const bf16* __restrict__ verify_raw,
    void* __restrict__ seq_state,
    const int* __restrict__ lines,
    const int* __restrict__ accepted,
    const int* __restrict__ valid,
    int max_seqs) {
  const int seq = blockIdx.x;
  const int visit = blockIdx.y;
  const int col = (blockIdx.z * kThreads + threadIdx.x) * kVec;

  const int line = lines[seq];
  int nacc = accepted[seq];
  const int vflag = valid[seq << 3];
  if (line <= 0 || nacc <= 0 || vflag == 0) return;
  if (nacc > 8) nacc = 8;

  constexpr int kWidths[4] = {1024, 1024, 2304, 2304};
  constexpr int64_t kHistOff[4] = {0LL, 6144LL, 12288LL, 26112LL};
  constexpr int kRawOff[4] = {0, 1024, 2048, 4352};

  char* visit_state = static_cast<char*>(seq_state)
      + (int64_t)line * kSeqStateBytes
      + (int64_t)visit * kSconvVisitBytes;
  const int64_t visit_raw_base = (int64_t)visit * max_seqs * 8 * kVerifyRawRowElems;

  asm volatile("griddepcontrol.wait;" ::: "memory");
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");

#pragma unroll
  for (int s = 0; s < 4; ++s) {
    const int C = kWidths[s];
    if (col < C) {
      bf16* hist = reinterpret_cast<bf16*>(visit_state + kHistOff[s]);
      Vec8 v[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int src_r = nacc - 3 + j;
        if (src_r < 0) {
          v[j].load(hist + (int64_t)(src_r + 3) * C + col);
        } else {
          const int row = (seq << 3) + src_r;
          v[j].load(verify_raw + visit_raw_base + (int64_t)row * kVerifyRawRowElems + kRawOff[s] + col);
        }
      }
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        v[j].store(hist + (int64_t)j * C + col);
      }
    }
  }
}
