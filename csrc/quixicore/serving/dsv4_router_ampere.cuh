#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace dsv4_router {

constexpr int HIDDEN = 4096;
constexpr int EXPERTS = 256;
constexpr int VECTOR_ELEMENTS = 8;
constexpr int THREADS = 128;
constexpr int HASH_TOPK = 6;
constexpr int HASH_THREADS = HASH_TOPK * 32;

template <int TOKENS>
__global__ __launch_bounds__(THREADS) void bf16_fp32_gemv(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        float* __restrict__ output) {
    constexpr int WARPS = THREADS / 32;
    constexpr int VALUES_PER_ITERATION = THREADS * VECTOR_ELEMENTS;
    constexpr int ITERATIONS = HIDDEN / VALUES_PER_ITERATION;
    static_assert(HIDDEN % VALUES_PER_ITERATION == 0);

    const int expert = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const __nv_bfloat16* weight_row = weight + expert * HIDDEN;
    float sums[TOKENS] = {};

#pragma unroll
    for (int iteration = 0; iteration < ITERATIONS; ++iteration) {
        const int column = iteration * VALUES_PER_ITERATION +
                           tid * VECTOR_ELEMENTS;
        const uint4 weight_vector =
            *reinterpret_cast<const uint4*>(weight_row + column);
        const __nv_bfloat16* weight_values =
            reinterpret_cast<const __nv_bfloat16*>(&weight_vector);

#pragma unroll
        for (int token = 0; token < TOKENS; ++token) {
            const uint4 input_vector = *reinterpret_cast<const uint4*>(
                x + token * HIDDEN + column);
            const __nv_bfloat16* input_values =
                reinterpret_cast<const __nv_bfloat16*>(&input_vector);
#pragma unroll
            for (int value = 0; value < VECTOR_ELEMENTS; ++value) {
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
            value += __shfl_down_sync(0xffffffff, value, offset);
        }
        if (lane == 0) {
            warp_sums[token][warp] = value;
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int token = 0; token < TOKENS; ++token) {
            float value = lane < WARPS ? warp_sums[token][lane] : 0.0f;
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                value += __shfl_down_sync(0xffffffff, value, offset);
            }
            if (lane == 0) {
                output[token * EXPERTS + expert] = value;
            }
        }
    }
}

template <int TOKENS>
inline void launch_tokens(const __nv_bfloat16* x,
                          const __nv_bfloat16* weight, float* output,
                          cudaStream_t stream) {
    bf16_fp32_gemv<TOKENS><<<EXPERTS, THREADS, 0, stream>>>(
        x, weight, output);
}

inline void launch(const __nv_bfloat16* x, const __nv_bfloat16* weight,
                   float* output, int tokens, cudaStream_t stream) {
    switch (tokens) {
        case 1: launch_tokens<1>(x, weight, output, stream); break;
        case 2: launch_tokens<2>(x, weight, output, stream); break;
        case 3: launch_tokens<3>(x, weight, output, stream); break;
        case 4: launch_tokens<4>(x, weight, output, stream); break;
        case 5: launch_tokens<5>(x, weight, output, stream); break;
        case 6: launch_tokens<6>(x, weight, output, stream); break;
        case 7: launch_tokens<7>(x, weight, output, stream); break;
        case 8: launch_tokens<8>(x, weight, output, stream); break;
    }
}

__device__ __forceinline__ float sqrt_softplus(float value) {
    const float softplus = fmaxf(value, 0.0f) +
                           __logf(1.0f + __expf(-fabsf(value)));
    const float score = sqrtf(softplus);
    return isnan(score) ? 0.0f : score;
}

// Hash layers predetermine the six expert IDs from the token ID. Computing
// all 256 router rows is therefore wasted work. One warp evaluates each
// selected row, then warp 0 performs the exact six-way normalization and
// writes the routing result consumed by the native expert kernels.
//
// vocab_size bounds the tid2eid first dimension; a token ID outside
// [0, vocab_size) must not be dereferenced (the resulting expert would be
// garbage and the weight-row read can hit unmapped memory). Such tokens are
// treated as padding. When debug_slot is non-null, the first offender is
// recorded as {flag, token_index, raw_token_id, blockIdx} for diagnosis.
template <typename TokenIdT>
__global__ __launch_bounds__(HASH_THREADS) void bf16_hash_router(
        const __nv_bfloat16* __restrict__ x,
        const __nv_bfloat16* __restrict__ weight,
        const TokenIdT* __restrict__ input_ids,
        const int32_t* __restrict__ tid2eid,
        const bool* __restrict__ is_padding,
        float routed_scaling_factor,
        float* __restrict__ topk_weights,
        int32_t* __restrict__ topk_ids,
        int32_t vocab_size,
        int32_t* __restrict__ debug_slot) {
    const int token = int(blockIdx.x);
    const int warp = int(threadIdx.x) >> 5;
    const int lane = int(threadIdx.x) & 31;
    bool padding = is_padding != nullptr && is_padding[token];

    const int64_t raw_token_id = int64_t(input_ids[token]);
    const bool bad_token_id = raw_token_id < 0 || raw_token_id >= vocab_size;
    if (bad_token_id) {
        if (!padding && debug_slot != nullptr && threadIdx.x == 0) {
            if (atomicCAS(debug_slot, 0, 1) == 0) {
                debug_slot[1] = token;
                debug_slot[2] = int32_t(raw_token_id);
                debug_slot[3] = int32_t(raw_token_id >> 32);
            }
        }
        padding = true;
    }

    __shared__ float selected_weights[HASH_TOPK];
    __shared__ int32_t selected_ids[HASH_TOPK];

    if (warp < HASH_TOPK) {
        const int32_t token_id = padding ? 0 : int32_t(raw_token_id);
        const int32_t expert = tid2eid[size_t(token_id) * HASH_TOPK + warp];
        float sum = 0.0f;
        if (!padding) {
            const __nv_bfloat16* input_row = x + size_t(token) * HIDDEN;
            const __nv_bfloat16* weight_row =
                weight + size_t(expert) * HIDDEN;
#pragma unroll
            for (int column = lane * VECTOR_ELEMENTS; column < HIDDEN;
                 column += 32 * VECTOR_ELEMENTS) {
                const uint4 input_vector =
                    *reinterpret_cast<const uint4*>(input_row + column);
                const uint4 weight_vector =
                    *reinterpret_cast<const uint4*>(weight_row + column);
                const __nv_bfloat16* input_values =
                    reinterpret_cast<const __nv_bfloat16*>(&input_vector);
                const __nv_bfloat16* weight_values =
                    reinterpret_cast<const __nv_bfloat16*>(&weight_vector);
#pragma unroll
                for (int value = 0; value < VECTOR_ELEMENTS; ++value) {
                    sum = fmaf(__bfloat162float(input_values[value]),
                               __bfloat162float(weight_values[value]), sum);
                }
            }
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffffu, sum, offset);
            }
        }
        if (lane == 0) {
            selected_weights[warp] = padding ? 0.0f : sqrt_softplus(sum);
            selected_ids[warp] = padding ? -1 : expert;
        }
    }
    __syncthreads();

    if (warp == 0) {
        float score = lane < HASH_TOPK ? selected_weights[lane] : 0.0f;
        float score_sum = score;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            score_sum += __shfl_xor_sync(0xffffffffu, score_sum, mask);
        }
        if (lane < HASH_TOPK) {
            const int output = token * HASH_TOPK + lane;
            const float denominator = score_sum > 0.0f ? score_sum : 1.0f;
            topk_weights[output] =
                score * routed_scaling_factor / denominator;
            topk_ids[output] = selected_ids[lane];
        }
    }
}

template <typename TokenIdT>
inline void launch_hash(const __nv_bfloat16* x,
                        const __nv_bfloat16* weight,
                        const TokenIdT* input_ids,
                        const int32_t* tid2eid,
                        const bool* is_padding,
                        float routed_scaling_factor,
                        float* topk_weights,
                        int32_t* topk_ids,
                        int tokens,
                        int vocab_size,
                        int32_t* debug_slot,
                        cudaStream_t stream) {
    bf16_hash_router<TokenIdT><<<tokens, HASH_THREADS, 0, stream>>>(
        x, weight, input_ids, tid2eid, is_padding, routed_scaling_factor,
        topk_weights, topk_ids, vocab_size, debug_slot);
}

}  // namespace dsv4_router
