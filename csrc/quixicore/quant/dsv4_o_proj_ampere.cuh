// SPDX-License-Identifier: Apache-2.0
// Native DeepSeek-V4 Q8_0 attention output projection for NVIDIA Ampere.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace slimserve::dsv4_ampere {

template <typename scalar_t, typename position_t>
__global__ __launch_bounds__(32, 8) void inverse_rope_quant_q8_1(
    const scalar_t* __restrict__ input,
    const position_t* __restrict__ positions,
    const float* __restrict__ cos_sin,
    block_q8_1* __restrict__ output, const int tokens,
    const int local_groups, const int heads_per_group, const int head_dim,
    const int rope_dim, const int64_t cos_sin_stride) {
  const int block = blockIdx.x;
  const int token_group = blockIdx.y;
  const int token = token_group / local_groups;
  const int group = token_group - token * local_groups;
  const int lane = threadIdx.x;
  if (token >= tokens) return;

  const int group_dim = heads_per_group * head_dim;
  const int dim = block * QK8_1 + lane;
  const int head_in_group = dim / head_dim;
  const int channel = dim - head_in_group * head_dim;
  const int head = group * heads_per_group + head_in_group;
  const int64_t input_base =
      (int64_t(token) * local_groups * heads_per_group + head) * head_dim;

  float value;
  const int nope_dim = head_dim - rope_dim;
  if (channel < nope_dim) {
    value = float(input[input_base + channel]);
  } else {
    const int rope_channel = channel - nope_dim;
    const int pair = rope_channel >> 1;
    const int even_channel = nope_dim + 2 * pair;
    const float a = float(input[input_base + even_channel]);
    const float b = float(input[input_base + even_channel + 1]);
    const int64_t position = int64_t(positions[token]);
    const float cosine = cos_sin[position * cos_sin_stride + pair];
    const float sine =
        cos_sin[position * cos_sin_stride + rope_dim / 2 + pair];
    const float rotated =
        (rope_channel & 1) ? b * cosine - a * sine
                           : a * cosine + b * sine;
    // The reference path materializes inverse-RoPE in BF16 before GGUF
    // activation quantization. Preserve that boundary exactly.
    value = float(scalar_t(rotated));
  }

  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    sum += __shfl_xor_sync(0xffffffffu, sum, mask);
  }
  const float scale = amax / 127.0f;
  block_q8_1& quant =
      output[(int64_t(token_group) * group_dim) / QK8_1 + block];
  quant.qs[lane] =
      amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (lane == 0) quant.ds = __floats2half2_rn(scale, sum);
}

// This matches the existing one-vector Q8_0 MMVQ reduction order: two warps
// cooperate on two output rows, each thread accumulates every sixteenth Q8
// block, and warp 0 folds warp 1 before the XOR tree. Keeping that order makes
// the native grouped projection bit-identical to the ordinary GGUF Q8 path.
template <typename scalar_t>
__global__ __launch_bounds__(64, 8) void grouped_q8_0_q8_1_gemv(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ inputs,
    scalar_t* __restrict__ output, const int tokens, const int local_groups,
    const int rows_per_group, const int blocks_per_row) {
  constexpr int kRows = 2;
  constexpr int kBlocksPerIteration = 16;
  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int row0 = blockIdx.x * kRows;
  const int token_group = blockIdx.y;
  const int token = token_group / local_groups;
  const int group = token_group - token * local_groups;
  if (token >= tokens || row0 >= rows_per_group) return;

  const block_q8_0* group_weights =
      weights + int64_t(group) * rows_per_group * blocks_per_row;
  const block_q8_1* input =
      inputs + int64_t(token_group) * blocks_per_row;
  float accum[kRows] = {0.0f, 0.0f};
  const int iqs = 2 * (tid & 3);
  for (int block = tid >> 2; block < blocks_per_row;
       block += kBlocksPerIteration) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
      const int row = min(row0 + r, rows_per_group - 1);
      accum[r] += vec_dot_q8_0_q8_1(
          group_weights + int64_t(row) * blocks_per_row + block,
          input + block, iqs);
    }
  }

  __shared__ float warp_partials[kRows][32];
  if (warp == 1) {
#pragma unroll
    for (int r = 0; r < kRows; ++r) warp_partials[r][lane] = accum[r];
  }
  __syncthreads();
  if (warp != 0) return;

#pragma unroll
  for (int r = 0; r < kRows; ++r) {
    float value = accum[r] + warp_partials[r][lane];
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, mask);
    }
    if (lane == 0 && row0 + r < rows_per_group) {
      output[(int64_t(token) * local_groups + group) * rows_per_group +
             row0 + r] = scalar_t(value);
    }
  }
}

template <typename scalar_t, typename position_t>
inline void launch_inverse_rope_quant_q8_1(
    const scalar_t* input, const position_t* positions, const float* cos_sin,
    void* output, const int tokens, const int local_groups,
    const int heads_per_group, const int head_dim, const int rope_dim,
    const int64_t cos_sin_stride, cudaStream_t stream) {
  const int group_dim = heads_per_group * head_dim;
  inverse_rope_quant_q8_1<scalar_t, position_t>
      <<<dim3(group_dim / QK8_1, tokens * local_groups), 32, 0, stream>>>(
          input, positions, cos_sin, static_cast<block_q8_1*>(output), tokens,
          local_groups, heads_per_group, head_dim, rope_dim, cos_sin_stride);
}

template <typename scalar_t>
inline void launch_grouped_q8_0_q8_1_gemv(
    const void* weights, const void* inputs, scalar_t* output,
    const int tokens, const int local_groups, const int rows_per_group,
    const int blocks_per_row, cudaStream_t stream) {
  grouped_q8_0_q8_1_gemv<scalar_t>
      <<<dim3((rows_per_group + 1) / 2, tokens * local_groups), 64, 0,
         stream>>>(static_cast<const block_q8_0*>(weights),
                   static_cast<const block_q8_1*>(inputs), output, tokens,
                   local_groups, rows_per_group, blocks_per_row);
}

}  // namespace slimserve::dsv4_ampere
