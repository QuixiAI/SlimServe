#pragma once

#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/std/functional>

namespace tms::dsv4_mhc {

constexpr int HC = 4;
constexpr int MIXES = 24;
constexpr int SPLITS = 32;
constexpr int THREADS = 256;

struct block_q8_1 {
    __half2 ds;
    int8_t qs[32];
};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float sigmoid(float value) {
    return 1.0f / (1.0f + expf(-value));
}

template <int NOUT, bool FUSED_POST, typename FnT = float>
__global__ void partials(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    const FnT* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    int hidden_size) {
    const int split = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int total = HC * hidden_size;

    __shared__ float mix_coeffs[HC + HC * HC];
    if constexpr (FUSED_POST) {
        if (tid < HC) {
            mix_coeffs[tid] = post[token * HC + tid];
        }
        if (tid < HC * HC) {
            mix_coeffs[HC + tid] = comb[token * HC * HC + tid];
        }
        __syncthreads();
    }

    float accum[NOUT];
#pragma unroll
    for (int output = 0; output < NOUT; ++output) {
        accum[output] = 0.0f;
    }
    float square_sum = 0.0f;

    for (int flat = split * THREADS + tid; flat < total;
         flat += SPLITS * THREADS) {
        float value;
        if constexpr (FUSED_POST) {
            const int stream = flat / hidden_size;
            const int dim = flat - stream * hidden_size;
            value = mix_coeffs[stream] * float(x[token * hidden_size + dim]);
#pragma unroll
            for (int input_stream = 0; input_stream < HC; ++input_stream) {
                value += mix_coeffs[HC + input_stream * HC + stream] *
                         float(residual[(token * HC + input_stream) * hidden_size + dim]);
            }
            const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
            residual_out[token * total + flat] = rounded;
            value = float(rounded);
        } else {
            value = float(residual[token * total + flat]);
        }
        square_sum += value * value;
#pragma unroll
        for (int output = 0; output < NOUT; ++output) {
            accum[output] += value * float(fn[output * total + flat]);
        }
    }

    __shared__ float warp_partials[THREADS / 32][NOUT + 1];
#pragma unroll
    for (int output = 0; output < NOUT; ++output) {
        const float sum = warp_sum(accum[output]);
        if (lane == 0) {
            warp_partials[warp][output] = sum;
        }
    }
    const float sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_partials[warp][NOUT] = sum;
    }
    __syncthreads();

    if (warp == 0) {
        for (int output = lane; output < NOUT + 1; output += 32) {
            float block_sum = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp) {
                block_sum += warp_partials[source_warp][output];
            }
            partial[(token * SPLITS + split) * (NOUT + 1) + output] = block_sum;
        }
    }
}

template <int NSPLITS>
__device__ __forceinline__ void finalize_pre_mix_block(
    float* partial,
    const float* scale,
    const float* base,
    float* post,
    float* comb,
    int token,
    int hidden_size,
    float rms_eps,
    float pre_eps,
    float sinkhorn_eps,
    float post_multiplier,
    int sinkhorn_repeat) {
    float* pre_mix = partial + token * NSPLITS * (MIXES + 1);
    const int lane = threadIdx.x;
    __shared__ float mixes[MIXES + 1];
    __shared__ float inverse_rms;

    if (lane < MIXES + 1) {
        float value = 0.0f;
#pragma unroll
        for (int split = 0; split < NSPLITS; ++split) {
            const float* source =
                partial + (token * NSPLITS + split) * (MIXES + 1);
            value += source[lane];
        }
        mixes[lane] = value;
        if (lane == MIXES) {
            inverse_rms =
                rsqrtf(value / float(HC * hidden_size) + rms_eps);
        }
    }
    __syncwarp();

    if (lane < HC) {
        pre_mix[lane] =
            sigmoid(mixes[lane] * inverse_rms * scale[0] + base[lane]) +
            pre_eps;
    } else if (lane < 2 * HC) {
        const int stream = lane - HC;
        post[token * HC + stream] =
            sigmoid(mixes[lane] * inverse_rms * scale[1] + base[lane]) *
            post_multiplier;
    }

    // Lanes 0..15 own the 4x4 Sinkhorn matrix in row-major order. Width-4
    // shuffles normalize rows; XOR 4/8 shuffles normalize columns.
    float matrix = 0.0f;
    if (lane < HC * HC) {
        const int index = 2 * HC + lane;
        matrix = mixes[index] * inverse_rms * scale[2] + base[index];
    }
    float row_max = matrix;
    row_max = fmaxf(row_max,
                    __shfl_xor_sync(0xffffffffu, row_max, 1, HC));
    row_max = fmaxf(row_max,
                    __shfl_xor_sync(0xffffffffu, row_max, 2, HC));
    if (lane < HC * HC) matrix = expf(matrix - row_max);
    float row_sum = matrix;
    row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, HC);
    row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, HC);
    if (lane < HC * HC) matrix = matrix / row_sum + sinkhorn_eps;

    for (int iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
        if (iteration > 0) {
            row_sum = matrix;
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, HC);
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, HC);
            if (lane < HC * HC) matrix /= row_sum + sinkhorn_eps;
        }
        float column_sum = matrix;
        column_sum += __shfl_xor_sync(0xffffffffu, column_sum, HC);
        column_sum += __shfl_xor_sync(0xffffffffu, column_sum, 2 * HC);
        if (lane < HC * HC) matrix /= column_sum + sinkhorn_eps;
    }
    if (lane < HC * HC) {
        comb[token * HC * HC + lane] = matrix;
    }
}

__global__ void finalize_pre_mix(
    float* partial,
    const float* scale,
    const float* base,
    float* post,
    float* comb,
    int hidden_size,
    float rms_eps,
    float pre_eps,
    float sinkhorn_eps,
    float post_multiplier,
    int sinkhorn_repeat) {
    finalize_pre_mix_block<SPLITS>(
        partial, scale, base, post, comb, blockIdx.x, hidden_size, rms_eps,
        pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_repeat);
}

template <int PARTIAL_WIDTH>
__global__ void apply_pre_mix(
    const float* partial,
    const __nv_bfloat16* residual,
    __nv_bfloat16* output,
    int hidden_size) {
    const int token = blockIdx.y;
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= hidden_size) {
        return;
    }
    const float* pre_mix = partial + token * SPLITS * PARTIAL_WIDTH;
    const int total = HC * hidden_size;
    float value = 0.0f;
#pragma unroll
    for (int stream = 0; stream < HC; ++stream) {
        value += pre_mix[stream] *
                 float(residual[token * total + stream * hidden_size + dim]);
    }
    output[token * hidden_size + dim] = __float2bfloat16_rn(value);
}

template <int PARTIAL_WIDTH, int HIDDEN_SIZE = 4096>
__global__ void apply_pre_mix_rms_norm(
    const float* partial,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* norm_weight,
    __nv_bfloat16* output,
    float norm_eps) {
    static_assert(HIDDEN_SIZE % THREADS == 0);
    constexpr int VALUES = HIDDEN_SIZE / THREADS;
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const float* pre_mix = partial + token * SPLITS * PARTIAL_WIDTH;
    const int total = HC * HIDDEN_SIZE;

    __nv_bfloat16 values[VALUES];
    float square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int dim = tid + i * THREADS;
        float value = 0.0f;
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
            value += pre_mix[stream] *
                     float(residual[token * total + stream * HIDDEN_SIZE + dim]);
        }
        values[i] = __float2bfloat16_rn(value);
        const float rounded = float(values[i]);
        square_sum += rounded * rounded;
    }

    __shared__ float warp_sums[THREADS / 32];
    __shared__ float inverse_rms;
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < THREADS / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            inverse_rms = rsqrtf(block_sum / float(HIDDEN_SIZE) + norm_eps);
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int dim = tid + i * THREADS;
        output[token * HIDDEN_SIZE + dim] = __float2bfloat16_rn(
            float(values[i]) * inverse_rms * float(norm_weight[dim]));
    }
}

template <bool FUSED_POST, bool RMS_NORM, int HIDDEN_SIZE = 4096,
          int NSPLITS = SPLITS, typename FnT = float>
__global__ void fused_pre_transition(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    const FnT* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    const float* scale,
    const float* base,
    float* next_post,
    float* next_comb,
    __nv_bfloat16* layer_input,
    const __nv_bfloat16* norm_weight,
    float rms_eps,
    float pre_eps,
    float sinkhorn_eps,
    float post_multiplier,
    int sinkhorn_repeat,
    float norm_eps) {
    static_assert(HIDDEN_SIZE % THREADS == 0);
    constexpr int NOUT = MIXES;
    constexpr int VALUES = HIDDEN_SIZE / THREADS;
    const int split = blockIdx.x;
    const int token = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    constexpr int TOTAL = HC * HIDDEN_SIZE;

    __shared__ float mix_coeffs[HC + HC * HC];
    if constexpr (FUSED_POST) {
        if (tid < HC) mix_coeffs[tid] = post_mix[token * HC + tid];
        if (tid < HC * HC)
            mix_coeffs[HC + tid] = comb_mix[token * HC * HC + tid];
        __syncthreads();
    }

    float accum[NOUT];
#pragma unroll
    for (int output = 0; output < NOUT; ++output) accum[output] = 0.0f;
    float square_sum = 0.0f;

    for (int flat = split * THREADS + tid; flat < TOTAL;
         flat += NSPLITS * THREADS) {
        float value;
        if constexpr (FUSED_POST) {
            const int stream = flat / HIDDEN_SIZE;
            const int dim = flat - stream * HIDDEN_SIZE;
            value = mix_coeffs[stream] * float(x[token * HIDDEN_SIZE + dim]);
#pragma unroll
            for (int input_stream = 0; input_stream < HC; ++input_stream) {
                value += mix_coeffs[HC + input_stream * HC + stream] *
                         float(residual[(token * HC + input_stream) *
                                            HIDDEN_SIZE +
                                        dim]);
            }
            const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
            residual_out[token * TOTAL + flat] = rounded;
            value = float(rounded);
        } else {
            value = float(residual[token * TOTAL + flat]);
        }
        square_sum += value * value;
#pragma unroll
        for (int output = 0; output < NOUT; ++output)
            accum[output] += value * float(fn[output * TOTAL + flat]);
    }

    __shared__ float warp_partials[THREADS / 32][NOUT + 1];
#pragma unroll
    for (int output = 0; output < NOUT; ++output) {
        const float sum = warp_sum(accum[output]);
        if (lane == 0) warp_partials[warp][output] = sum;
    }
    const float sum = warp_sum(square_sum);
    if (lane == 0) warp_partials[warp][NOUT] = sum;
    __syncthreads();

    if (warp == 0) {
        for (int output = lane; output < NOUT + 1; output += 32) {
            float block_sum = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < THREADS / 32;
                 ++source_warp)
                block_sum += warp_partials[source_warp][output];
            partial[(token * NSPLITS + split) * (NOUT + 1) + output] =
                block_sum;
        }
    }

    cooperative_groups::this_grid().sync();
    if (split != 0) return;

    finalize_pre_mix_block<NSPLITS>(
        partial, scale, base, next_post, next_comb, token, HIDDEN_SIZE,
        rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_repeat);
    __syncthreads();

    const float* pre = partial + token * NSPLITS * (NOUT + 1);
    const __nv_bfloat16* mixed_residual =
        FUSED_POST ? residual_out : residual;
    __nv_bfloat16 values[VALUES];
    float norm_square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int dim = tid + i * THREADS;
        float value = 0.0f;
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
            value += pre[stream] *
                     float(mixed_residual[(token * HC + stream) *
                                              HIDDEN_SIZE +
                                          dim]);
        }
        values[i] = __float2bfloat16_rn(value);
        if constexpr (RMS_NORM) {
            const float rounded = float(values[i]);
            norm_square_sum += rounded * rounded;
        }
    }

    if constexpr (RMS_NORM) {
        __shared__ float norm_warp_sums[THREADS / 32];
        __shared__ float norm_inverse_rms;
        norm_square_sum = warp_sum(norm_square_sum);
        if (lane == 0) norm_warp_sums[warp] = norm_square_sum;
        __syncthreads();
        if (warp == 0) {
            float block_sum =
                lane < THREADS / 32 ? norm_warp_sums[lane] : 0.0f;
            block_sum = warp_sum(block_sum);
            if (lane == 0)
                norm_inverse_rms =
                    rsqrtf(block_sum / float(HIDDEN_SIZE) + norm_eps);
        }
        __syncthreads();
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int dim = tid + i * THREADS;
            layer_input[token * HIDDEN_SIZE + dim] = __float2bfloat16_rn(
                float(values[i]) * norm_inverse_rms * float(norm_weight[dim]));
        }
    } else {
#pragma unroll
        for (int i = 0; i < VALUES; ++i) {
            const int dim = tid + i * THREADS;
            layer_input[token * HIDDEN_SIZE + dim] = values[i];
        }
    }
}

__global__ void post(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    __nv_bfloat16* output,
    int hidden_size) {
    const int token = blockIdx.y;
    const int dim = blockIdx.x * blockDim.x + threadIdx.x;
    if (dim >= hidden_size) {
        return;
    }
    const int total = HC * hidden_size;
#pragma unroll
    for (int stream = 0; stream < HC; ++stream) {
        float value = post_mix[token * HC + stream] * float(x[token * hidden_size + dim]);
#pragma unroll
        for (int input_stream = 0; input_stream < HC; ++input_stream) {
            value += comb_mix[(token * HC + input_stream) * HC + stream] *
                     float(residual[token * total + input_stream * hidden_size + dim]);
        }
        output[token * total + stream * hidden_size + dim] = __float2bfloat16_rn(value);
    }
}

__global__ void finalize_head_mix(
    float* partial,
    const float* scale,
    const float* base,
    int hidden_size,
    float rms_eps,
    float hc_eps) {
    const int token = blockIdx.x;
    float* gates = partial + token * SPLITS * (HC + 1);
    if (threadIdx.x == 0) {
        float mixes[HC] = {};
        float square_sum = 0.0f;
#pragma unroll
        for (int split = 0; split < SPLITS; ++split) {
            const float* source = partial + (token * SPLITS + split) * (HC + 1);
#pragma unroll
            for (int output_index = 0; output_index < HC; ++output_index) {
                mixes[output_index] += source[output_index];
            }
            square_sum += source[HC];
        }
        const float rms = rsqrtf(square_sum / float(HC * hidden_size) + rms_eps);
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
            gates[stream] = sigmoid(mixes[stream] * rms * scale[0] + base[stream]) + hc_eps;
        }
    }
}

}  // namespace tms::dsv4_mhc
