#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dsv4_projection {

constexpr int HIDDEN = 4096;
constexpr int THREADS = 256;
constexpr int VALUES_PER_LOAD = 8;

template <int TOKENS, typename OutputT>
__global__ __launch_bounds__(THREADS, 4) void bf16_fp32_gemv(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        OutputT* __restrict__ output, int rows) {
    constexpr int WARPS = THREADS / 32;
    constexpr int VALUES_PER_ITERATION = THREADS * VALUES_PER_LOAD;
    constexpr int ITERATIONS = HIDDEN / VALUES_PER_ITERATION;
    static_assert(HIDDEN % VALUES_PER_ITERATION == 0);

    const int row = int(blockIdx.x);
    if (row >= rows) return;
    const int tid = int(threadIdx.x);
    const __nv_bfloat16* weight_row = weight + size_t(row) * HIDDEN;
    float sums[TOKENS] = {};

#pragma unroll
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        const int column = iteration * VALUES_PER_ITERATION +
                           tid * VALUES_PER_LOAD;
        const uint4 weight_vector =
            *reinterpret_cast<const uint4*>(weight_row + column);
        const __nv_bfloat16* weight_values =
            reinterpret_cast<const __nv_bfloat16*>(&weight_vector);

#pragma unroll
        for (int token = 0; token < TOKENS; ++token) {
            const uint4 input_vector = *reinterpret_cast<const uint4*>(
                x + size_t(token) * HIDDEN + column);
            const __nv_bfloat16* input_values =
                reinterpret_cast<const __nv_bfloat16*>(&input_vector);
#pragma unroll
            for (int value = 0; value < VALUES_PER_LOAD; ++value) {
                sums[token] = fmaf(__bfloat162float(input_values[value]),
                                   __bfloat162float(weight_values[value]),
                                   sums[token]);
            }
        }
    }

    __shared__ float warp_sums[TOKENS][WARPS];
    const int lane = tid & 31;
    const int warp = tid >> 5;
#pragma unroll
    for (int token = 0; token < TOKENS; ++token) {
        float value = sums[token];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) warp_sums[token][warp] = value;
    }
    __syncthreads();

    if (warp != 0) return;
#pragma unroll
    for (int token = 0; token < TOKENS; ++token) {
        float value = lane < WARPS ? warp_sums[token][lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) output[size_t(token) * rows + row] = OutputT(value);
    }
}

template <int TOKENS, typename OutputT>
inline void launch_tokens(const __nv_bfloat16* x,
                          const __nv_bfloat16* weight, OutputT* output,
                          int rows, cudaStream_t stream) {
    bf16_fp32_gemv<TOKENS, OutputT><<<rows, THREADS, 0, stream>>>(
        x, weight, output, rows);
}

template <typename OutputT>
inline void launch(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                   OutputT* output, int tokens, int rows,
                   cudaStream_t stream) {
    switch (tokens) {
        case 1: launch_tokens<1>(x, weight, output, rows, stream); break;
        case 2: launch_tokens<2>(x, weight, output, rows, stream); break;
        case 3: launch_tokens<3>(x, weight, output, rows, stream); break;
        case 4: launch_tokens<4>(x, weight, output, rows, stream); break;
        case 5: launch_tokens<5>(x, weight, output, rows, stream); break;
        case 6: launch_tokens<6>(x, weight, output, rows, stream); break;
        case 7: launch_tokens<7>(x, weight, output, rows, stream); break;
        case 8: launch_tokens<8>(x, weight, output, rows, stream); break;
    }
}

}  // namespace dsv4_projection
