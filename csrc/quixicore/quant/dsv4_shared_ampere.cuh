// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4 shared-expert decode kernels for NVIDIA Ampere.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <cstdlib>

namespace slimserve::dsv4_ampere {

// Decode normalization and Q8_1 emission share the same 4096-value input.
// Matching vLLM's 512-thread, vec8 CUB reduction preserves the BF16 RMSNorm
// output exactly; the second phase mirrors GGUF's warp Q8_1 quantizer.
template <typename scalar_t>
__global__ __launch_bounds__(512, 2) void rms_norm_q8_1_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    block_q8_1* __restrict__ quant_output,
    const float epsilon) {
  constexpr int HIDDEN = 4096;
  constexpr int THREADS = 512;
  constexpr int VEC = 8;
  constexpr int Q8_BLOCKS = HIDDEN / QK8_1;
  static_assert(THREADS * VEC == HIDDEN);

  const int token = blockIdx.x;
  const int tid = threadIdx.x;
  const scalar_t* input_row = input + int64_t(token) * HIDDEN;
  scalar_t* output_row = output + int64_t(token) * HIDDEN;
  block_q8_1* quant_row = quant_output + int64_t(token) * Q8_BLOCKS;

  float variance = 0.0f;
#pragma unroll
  for (int i = 0; i < VEC; ++i) {
    const float value = float(input_row[tid * VEC + i]);
    variance += value * value;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  __shared__ float inverse_rms;
  variance = BlockReduce(reduce_storage).Reduce(
      variance, CubAddOp{}, blockDim.x);
  if (tid == 0) {
    inverse_rms = rsqrtf(variance / float(HIDDEN) + epsilon);
  }
  __syncthreads();

  const int warp = tid >> 5;
  const int lane = tid & 31;
#pragma unroll
  for (int round = 0; round < VEC; ++round) {
    const int dim = round * THREADS + tid;
    const scalar_t normalized = scalar_t(
        float(input_row[dim]) * inverse_rms * float(weight[dim]));
    output_row[dim] = normalized;

    const int block = round * (THREADS / 32) + warp;
    const float value = float(normalized);
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float scale = amax / 127.0f;
    quant_row[block].qs[lane] =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    if (lane == 0) {
      quant_row[block].ds = __floats2half2_rn(scale, sum);
    }
  }
}

template <typename scalar_t>
inline void launch_rms_norm_q8_1(
    const scalar_t* input, const scalar_t* weight, scalar_t* output,
    void* quant_output, int tokens, float epsilon, cudaStream_t stream) {
  rms_norm_q8_1_kernel<scalar_t><<<tokens, 512, 0, stream>>>(
      input, weight, output, static_cast<block_q8_1*>(quant_output), epsilon);
}

__device__ __forceinline__ int32_t load_i8x4_unaligned(const int8_t* ptr) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(ptr);
  return int32_t(uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
                 (uint32_t(bytes[2]) << 16) |
                 (uint32_t(bytes[3]) << 24));
}

__device__ __forceinline__ int32_t dot_q8_blocks(const block_q8_0& weight,
                                                  const block_q8_1& input) {
  int32_t dot = 0;
#pragma unroll
  for (int i = 0; i < QK8_0; i += 4) {
    const int32_t w = load_i8x4_unaligned(weight.qs + i);
    const int32_t x = *reinterpret_cast<const int32_t*>(input.qs + i);
    dot = __dp4a(w, x, dot);
  }
  return dot;
}

template <typename scalar_t, int WARPS_PER_PAIR>
__global__ void q8_0_gate_up_swiglu_kernel(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ inputs, scalar_t* __restrict__ output,
    int blocks_per_row, int intermediate, int input_blocks_per_token,
    float swiglu_limit) {
  constexpr int kBlockWarps = 8;
  constexpr int kPairsPerBlock = kBlockWarps / WARPS_PER_PAIR;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = warp / WARPS_PER_PAIR;
  const int warp_in_pair = warp % WARPS_PER_PAIR;
  const int row = blockIdx.x * kPairsPerBlock + pair;
  const int token = blockIdx.y;
  if (row >= intermediate) {
    return;
  }

  const block_q8_0* gate = weights + int64_t(row) * blocks_per_row;
  const block_q8_0* up =
      weights + int64_t(intermediate + row) * blocks_per_row;
  const block_q8_1* input =
      inputs + int64_t(token) * input_blocks_per_token;
  float gate_acc = 0.0f;
  float up_acc = 0.0f;

  for (int block = warp_in_pair * 32 + lane; block < blocks_per_row;
       block += 32 * WARPS_PER_PAIR) {
    const block_q8_1 input_block = input[block];
    const float input_scale = __low2float(input_block.ds);
    const block_q8_0 gate_block = gate[block];
    const block_q8_0 up_block = up[block];
    gate_acc = fmaf(float(dot_q8_blocks(gate_block, input_block)),
                    __half2float(gate_block.d) * input_scale, gate_acc);
    up_acc = fmaf(float(dot_q8_blocks(up_block, input_block)),
                  __half2float(up_block.d) * input_scale, up_acc);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, offset);
    up_acc += __shfl_xor_sync(0xffffffffu, up_acc, offset);
  }
  __shared__ float gate_partials[kBlockWarps];
  __shared__ float up_partials[kBlockWarps];
  if (lane == 0) {
    gate_partials[warp] = gate_acc;
    up_partials[warp] = up_acc;
  }
  __syncthreads();
  if (lane != 0 || warp_in_pair != 0) {
    return;
  }

#pragma unroll
  for (int i = 1; i < WARPS_PER_PAIR; ++i) {
    gate_acc += gate_partials[warp + i];
    up_acc += up_partials[warp + i];
  }

  if (swiglu_limit > 0.0f) {
    gate_acc = fminf(gate_acc, swiglu_limit);
    up_acc = fminf(fmaxf(up_acc, -swiglu_limit), swiglu_limit);
  }
  const float silu = gate_acc / (1.0f + expf(-gate_acc));
  output[int64_t(token) * intermediate + row] =
      static_cast<scalar_t>(silu * up_acc);
}

template <typename scalar_t>
void launch_q8_0_gate_up_swiglu(const void* weights, const void* inputs,
                                scalar_t* output, int tokens,
                                int blocks_per_row, int intermediate,
                                int input_blocks_per_token,
                                float swiglu_limit, cudaStream_t stream) {
  constexpr int kBlockWarps = 8;
  int warps_per_pair = 4;
  if (const char* value = std::getenv("VLLM_DSV4_SHARED_Q8_WARPS_PER_PAIR")) {
    warps_per_pair = std::atoi(value);
  }
#define LAUNCH_SHARED_Q8(WARPS_PER_PAIR)                                      \
  do {                                                                        \
    constexpr int pairs_per_block = kBlockWarps / (WARPS_PER_PAIR);           \
    const dim3 grid((intermediate + pairs_per_block - 1) / pairs_per_block,   \
                    tokens, 1);                                               \
    q8_0_gate_up_swiglu_kernel<scalar_t, WARPS_PER_PAIR>                       \
        <<<grid, kBlockWarps * 32, 0, stream>>>(                               \
            static_cast<const block_q8_0*>(weights),                           \
            static_cast<const block_q8_1*>(inputs), output, blocks_per_row,   \
            intermediate, input_blocks_per_token, swiglu_limit);              \
  } while (0)
  if (warps_per_pair == 1) {
    LAUNCH_SHARED_Q8(1);
  } else if (warps_per_pair == 4) {
    LAUNCH_SHARED_Q8(4);
  } else if (warps_per_pair == 2) {
    LAUNCH_SHARED_Q8(2);
  } else {
    LAUNCH_SHARED_Q8(4);
  }
#undef LAUNCH_SHARED_Q8
}

// Decode-only persistent variant. A cooperative grid computes the gate/up
// rows, keeps the exact BF16 SwiGLU materialization used by the existing path,
// then packs those values to Q8_1 after a grid barrier. The down projection can
// consume the packed activation directly and avoids a separate quantization
// launch and intermediate reread.
template <typename scalar_t, int WARPS_PER_PAIR>
__global__ void q8_0_gate_up_swiglu_q8_1_kernel(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ inputs, scalar_t* __restrict__ output,
    block_q8_1* __restrict__ quant_output, int blocks_per_row,
    int intermediate, int input_blocks_per_token, float swiglu_limit) {
  constexpr int kBlockWarps = 8;
  constexpr int kPairsPerBlock = kBlockWarps / WARPS_PER_PAIR;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = warp / WARPS_PER_PAIR;
  const int warp_in_pair = warp % WARPS_PER_PAIR;
  const block_q8_1* input = inputs;

  __shared__ float gate_partials[kBlockWarps];
  __shared__ float up_partials[kBlockWarps];
  for (int row_base = blockIdx.x * kPairsPerBlock;
       row_base < intermediate;
       row_base += gridDim.x * kPairsPerBlock) {
    const int row = row_base + pair;
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    if (row < intermediate) {
      const block_q8_0* gate = weights + int64_t(row) * blocks_per_row;
      const block_q8_0* up =
          weights + int64_t(intermediate + row) * blocks_per_row;
      for (int block = warp_in_pair * 32 + lane; block < blocks_per_row;
           block += 32 * WARPS_PER_PAIR) {
        const block_q8_1 input_block = input[block];
        const float input_scale = __low2float(input_block.ds);
        const block_q8_0 gate_block = gate[block];
        const block_q8_0 up_block = up[block];
        gate_acc = fmaf(float(dot_q8_blocks(gate_block, input_block)),
                        __half2float(gate_block.d) * input_scale, gate_acc);
        up_acc = fmaf(float(dot_q8_blocks(up_block, input_block)),
                      __half2float(up_block.d) * input_scale, up_acc);
      }
    }

#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, offset);
      up_acc += __shfl_xor_sync(0xffffffffu, up_acc, offset);
    }
    if (lane == 0) {
      gate_partials[warp] = gate_acc;
      up_partials[warp] = up_acc;
    }
    __syncthreads();

    if (row < intermediate && lane == 0 && warp_in_pair == 0) {
#pragma unroll
      for (int i = 1; i < WARPS_PER_PAIR; ++i) {
        gate_acc += gate_partials[warp + i];
        up_acc += up_partials[warp + i];
      }
      if (swiglu_limit > 0.0f) {
        gate_acc = fminf(gate_acc, swiglu_limit);
        up_acc = fminf(fmaxf(up_acc, -swiglu_limit), swiglu_limit);
      }
      const float silu = gate_acc / (1.0f + expf(-gate_acc));
      output[row] = static_cast<scalar_t>(silu * up_acc);
    }
    // Protect the shared partial arrays before the next persistent row tile.
    __syncthreads();
  }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x >= (intermediate + QK8_1 - 1) / QK8_1 || warp != 0) {
    return;
  }
  const int dim = blockIdx.x * QK8_1 + lane;
  const float value = dim < intermediate ? float(output[dim]) : 0.0f;
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    sum += __shfl_xor_sync(0xffffffffu, sum, mask);
  }
  const float scale = amax / 127.0f;
  block_q8_1& quant = quant_output[blockIdx.x];
  quant.qs[lane] =
      amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  if (lane == 0) quant.ds = __floats2half2_rn(scale, sum);
}

template <typename scalar_t>
void launch_q8_0_gate_up_swiglu_q8_1(
    const void* weights, const void* inputs, scalar_t* output,
    void* quant_output, int blocks_per_row, int intermediate,
    int input_blocks_per_token, float swiglu_limit, cudaStream_t stream) {
  constexpr int kBlockWarps = 8;
  constexpr int WARPS_PER_PAIR = 4;
  constexpr int kPairsPerBlock = kBlockWarps / WARPS_PER_PAIR;
  auto kernel = q8_0_gate_up_swiglu_q8_1_kernel<scalar_t, WARPS_PER_PAIR>;
  int blocks_per_sm = 0;
  int sm_count = 0;
  int device = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, kBlockWarps * 32, 0);
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  const int row_blocks = (intermediate + kPairsPerBlock - 1) / kPairsPerBlock;
  const int resident_blocks = blocks_per_sm * sm_count;
  const int grid_blocks =
      row_blocks < resident_blocks ? row_blocks : resident_blocks;
  const auto* weight_ptr = static_cast<const block_q8_0*>(weights);
  const auto* input_ptr = static_cast<const block_q8_1*>(inputs);
  auto* quant_ptr = static_cast<block_q8_1*>(quant_output);
  void* args[] = {
      &weight_ptr, &input_ptr, &output,
      &quant_ptr, &blocks_per_row, &intermediate, &input_blocks_per_token,
      &swiglu_limit,
  };
  cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                              dim3(grid_blocks), dim3(kBlockWarps * 32), args,
                              0, stream);
}

}  // namespace slimserve::dsv4_ampere
