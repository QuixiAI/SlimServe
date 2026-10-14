// SPDX-License-Identifier: Apache-2.0
// Standalone, first-histogram-pass diagnostic. Not linked into serving.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "glm53f_warp_histogram.cuh"

namespace {

template <bool aggregate>
__device__ __forceinline__ void count_value(float value, int* histogram) {
  // Match sampler.cu extractBinIdx<0>, including signed zero and infinities.
  unsigned bits = __half_as_ushort(__float2half_rn(value));
  bits = (bits & 0x8000u) ? bits : (~bits & 0x7fffu);
  const int bin = bits >> 5;
  if constexpr (aggregate) {
    quixicore::warp_histogram_add(histogram, bin);
  } else {
    atomicAdd(histogram + bin, 1);
  }
}

template <bool aggregate>
__global__ __launch_bounds__(512) void histogram_first_pass(
    const float* logits, const int* lengths, int* output, int stride) {
  __shared__ int histogram[2048];
  const int row = blockIdx.x;
  const int n = lengths[row];  // Benchmark supplies 0 <= n <= stride.
  const float* source = logits + static_cast<size_t>(row) * stride;
  for (int bin = threadIdx.x; bin < 2048; bin += blockDim.x) {
    histogram[bin] = 0;
  }
  __syncthreads();
  // Same four-values-per-thread traversal as the aligned serving helper.
  const float4* vectors = reinterpret_cast<const float4*>(source);
  for (int i = threadIdx.x; i < n / 4; i += blockDim.x) {
    const float4 v = vectors[i];
    count_value<aggregate>(v.x, histogram);
    count_value<aggregate>(v.y, histogram);
    count_value<aggregate>(v.z, histogram);
    count_value<aggregate>(v.w, histogram);
  }
  const int tail = (n / 4) * 4 + threadIdx.x;
  if (tail < n) {
    count_value<aggregate>(source[tail], histogram);
  }
  __syncthreads();
  for (int bin = threadIdx.x; bin < 2048; bin += blockDim.x) {
    output[static_cast<size_t>(row) * 2048 + bin] = histogram[bin];
  }
}

}  // namespace

extern "C" int glm53f_histogram_launch(
    const float* logits, const int* lengths, int* output, int rows, int stride,
    int aggregate, cudaStream_t stream) {
  if (rows <= 0 || stride <= 0 || stride % 4 != 0 ||
      (aggregate != 0 && aggregate != 1)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (aggregate) {
    histogram_first_pass<true><<<rows, 512, 0, stream>>>(logits, lengths, output,
                                                       stride);
  } else {
    histogram_first_pass<false><<<rows, 512, 0, stream>>>(logits, lengths, output,
                                                        stride);
  }
  return static_cast<int>(cudaGetLastError());
}
