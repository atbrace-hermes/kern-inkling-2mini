// SPDX-License-Identifier: Apache-2.0
// Glue around FlashInfer sampling chain for BF16 and FP32 logits.
#include <cuda_bf16.h>

extern "C" __global__ void kern_sample_cast_bf16(const __nv_bfloat16* __restrict__ logits,
                                                 float* __restrict__ probs, int vocab) {
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < vocab) {
    size_t at = (size_t)blockIdx.x * vocab + i;
    probs[at] = __bfloat162float(logits[at]);
  }
}

extern "C" __global__ void kern_sample_copy_f32(const float* __restrict__ logits,
                                                float* __restrict__ probs, int vocab) {
  int i = blockIdx.y * blockDim.x + threadIdx.x;
  if (i < vocab) {
    size_t at = (size_t)blockIdx.x * vocab + i;
    probs[at] = logits[at];
  }
}

extern "C" __global__ void kern_sample_prep(const float* __restrict__ temperature, const int* __restrict__ top_k,
                                            const int* __restrict__ seq_len, float* __restrict__ chain_temperature,
                                            int* __restrict__ chain_top_k, unsigned long long* __restrict__ offset,
                                            unsigned* __restrict__ states, int seqs, int vocab, int state_words) {
  for (int i = threadIdx.x; i < seqs; i += blockDim.x) {
    float t = temperature[i];
    chain_temperature[i] = t > 0.f ? t : 1.f;
    int k = top_k[i];
    chain_top_k[i] = (k >= 1 && k < vocab) ? k : vocab;
    offset[i] = (unsigned long long)seq_len[i];
  }
  for (int i = threadIdx.x; i < state_words; i += blockDim.x) states[i] = 0;
}

extern "C" __global__ void kern_sample_select(const float* __restrict__ temperature, const int* __restrict__ drawn,
                                              const long long* __restrict__ argmax, const unsigned char* __restrict__ valid,
                                              long long* __restrict__ tokens,
                                              int seqs) {
  for (int i = threadIdx.x; i < seqs; i += blockDim.x) {
    if (temperature[i] > 0.f && !valid[i]) __trap();
    tokens[i] = temperature[i] > 0.f ? (long long)drawn[i] : argmax[i];
  }
}
