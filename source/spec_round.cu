// SPDX-License-Identifier: Apache-2.0
// Adapted from kern/tools/kernels-src/spec_round.cu for SM120.

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
