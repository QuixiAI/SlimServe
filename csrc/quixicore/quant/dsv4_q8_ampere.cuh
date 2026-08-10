#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>


namespace slimserve::dsv4_ampere {

// GGUF Q8_0 interleaves a two-byte scale with 32 int8 codes. That 34-byte
// stride makes every other code block misaligned and prevents vector loads.
// The aligned representation is byte-neutral: all fp16 scales followed by all
// code bytes. The original packed tensor remains available for prefill MMQ.
static __global__ void repack_q8_0_aligned_kernel(
    const uint8_t* __restrict__ input, uint8_t* __restrict__ output,
    int64_t total_blocks) {
  const int64_t block =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= total_blocks) {
    return;
  }

  const uint8_t* src = input + block * 34;
  uint16_t scale;
  memcpy(&scale, src, sizeof(scale));
  reinterpret_cast<uint16_t*>(output)[block] = scale;

  uint8_t* codes = output + total_blocks * sizeof(uint16_t) + block * 32;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint32_t value;
    memcpy(&value, src + sizeof(uint16_t) + i * sizeof(uint32_t),
           sizeof(value));
    reinterpret_cast<uint32_t*>(codes)[i] = value;
  }
}

static inline void launch_repack_q8_0_aligned(const void* input, void* output,
                                               int64_t total_blocks,
                                               cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = static_cast<int>((total_blocks + threads - 1) / threads);
  repack_q8_0_aligned_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output),
      total_blocks);
}

template <typename scalar_t, int rows_per_cta, int tokens_per_cta>
static __global__ void aligned_q8_0_q8_1_gemv_kernel(
    const uint16_t* __restrict__ scales, const int8_t* __restrict__ codes,
    const block_q8_1* __restrict__ activation, scalar_t* __restrict__ output,
    int tokens, int rows, int blocks_per_row,
    int activation_blocks_per_token) {
  const int tid = 32 * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cta * blockIdx.x;
  const int token0 = tokens_per_cta * blockIdx.y;
  if (row0 >= rows) {
    return;
  }

  const block_q8_1* token_activation[tokens_per_cta];
#pragma unroll
  for (int j = 0; j < tokens_per_cta; ++j) {
    const int token = token0 + j < tokens ? token0 + j : tokens - 1;
    token_activation[j] =
        activation + static_cast<int64_t>(token) * activation_blocks_per_token;
  }
  float partial[tokens_per_cta][rows_per_cta] = {{0.0f}};
  const int code_index = 2 * (tid % 4);

  // This is the same two-warp block traversal and accumulation order as the
  // ordinary Q8_0 MMVQ kernel. Only the weight addresses differ.
  for (int block = tid / 4; block < blocks_per_row; block += 16) {
#pragma unroll
    for (int i = 0; i < rows_per_cta; ++i) {
      const int row = row0 + i < rows ? row0 + i : rows - 1;
      const int64_t weight_block =
          static_cast<int64_t>(row) * blocks_per_row + block;
      const int* weight_codes = reinterpret_cast<const int*>(
          codes + weight_block * 32);
      const int weight_code0 = weight_codes[code_index];
      const int weight_code1 = weight_codes[code_index + 1];
      const float weight_scale =
          __half2float(reinterpret_cast<const half*>(scales)[weight_block]);
#pragma unroll
      for (int j = 0; j < tokens_per_cta; ++j) {
        const block_q8_1& act = token_activation[j][block];
        const int* act_codes = reinterpret_cast<const int*>(act.qs);
        int dot = 0;
        dot = __dp4a(weight_code0, act_codes[code_index], dot);
        dot = __dp4a(weight_code1, act_codes[code_index + 1], dot);
        partial[j][i] += weight_scale * __low2float(act.ds) * dot;
      }
    }
  }

  __shared__ float warp_partial[tokens_per_cta][rows_per_cta][32];
  if (threadIdx.y == 1) {
#pragma unroll
    for (int j = 0; j < tokens_per_cta; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cta; ++i) {
        warp_partial[j][i][threadIdx.x] = partial[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y != 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < tokens_per_cta; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cta; ++i) {
      partial[j][i] += warp_partial[j][i][threadIdx.x];
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        partial[j][i] += __shfl_xor_sync(0xffffffffu, partial[j][i], mask);
      }
      if (threadIdx.x == 0 && token0 + j < tokens && row0 + i < rows) {
        output[static_cast<int64_t>(token0 + j) * rows + row0 + i] =
            partial[j][i];
      }
    }
  }
}

template <typename scalar_t, int rows_per_cta, int tokens_per_cta>
static inline void launch_aligned_q8_0_q8_1_gemv_tile(
    const uint16_t* scales, const int8_t* codes,
    const block_q8_1* activation, scalar_t* output, int tokens, int rows,
    int blocks_per_row, int activation_blocks_per_token,
    cudaStream_t stream) {
  const dim3 grid((rows + rows_per_cta - 1) / rows_per_cta,
                  (tokens + tokens_per_cta - 1) / tokens_per_cta, 1);
  const dim3 threads(32, 2, 1);
  aligned_q8_0_q8_1_gemv_kernel<scalar_t, rows_per_cta, tokens_per_cta>
      <<<grid, threads, 0, stream>>>(scales, codes, activation, output, tokens,
                                    rows, blocks_per_row,
                                    activation_blocks_per_token);
}

template <typename scalar_t, int rows_per_cta>
static inline void launch_aligned_q8_0_q8_1_gemv_tokens(
    const uint16_t* scales, const int8_t* codes,
    const block_q8_1* activation, scalar_t* output, int tokens, int rows,
    int blocks_per_row, int activation_blocks_per_token,
    cudaStream_t stream) {
  if (tokens <= 1) {
    launch_aligned_q8_0_q8_1_gemv_tile<scalar_t, rows_per_cta, 1>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  } else if (tokens <= 2) {
    launch_aligned_q8_0_q8_1_gemv_tile<scalar_t, rows_per_cta, 2>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  } else if (tokens <= 4) {
    launch_aligned_q8_0_q8_1_gemv_tile<scalar_t, rows_per_cta, 4>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  } else {
    launch_aligned_q8_0_q8_1_gemv_tile<scalar_t, rows_per_cta, 8>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  }
}

template <typename scalar_t>
static inline void launch_aligned_q8_0_q8_1_gemv(
    const void* aligned_weight, const void* quant_activation, scalar_t* output,
    int tokens, int rows, int blocks_per_row,
    int activation_blocks_per_token, int rows_per_cta, cudaStream_t stream) {
  const int64_t total_blocks = static_cast<int64_t>(rows) * blocks_per_row;
  const auto* scales = static_cast<const uint16_t*>(aligned_weight);
  const auto* codes = reinterpret_cast<const int8_t*>(scales + total_blocks);
  const auto* activation = static_cast<const block_q8_1*>(quant_activation);

  if (rows_per_cta >= 4) {
    launch_aligned_q8_0_q8_1_gemv_tokens<scalar_t, 4>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  } else if (rows_per_cta >= 2) {
    launch_aligned_q8_0_q8_1_gemv_tokens<scalar_t, 2>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  } else {
    launch_aligned_q8_0_q8_1_gemv_tokens<scalar_t, 1>(
        scales, codes, activation, output, tokens, rows, blocks_per_row,
        activation_blocks_per_token, stream);
  }
}

}  // namespace slimserve::dsv4_ampere
