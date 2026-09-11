#pragma once

#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cuda/std/functional>
#include <type_traits>

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

// Prefill-shaped partials for the split path. `partials` above is shaped
// for decode: 32 x T blocks, two residual elements per thread, 24 fn loads
// and 24 FMAs per element and a 125-shuffle reduction per warp per token;
// at T = 7000 it runs the residual streams at 7% of HBM bandwidth. Here a
// block owns one split of one 32-token tile, where split s is the 128-dim
// slice [s * 128, s * 128 + 128) of all four streams (512 flats): the
// split's fn values are staged in shared memory once per tile, the tile's
// residual rows once (for the fused post-mix, each input element is read
// once and yields all four output streams), and each lane then runs whole
// 512-long dot products for one token and three mix rows. Same partial
// layout ([T][SPLITS][MIXES + 1]; the flat-to-split assignment differs from
// the strided one above), so finalize_pre_mix / apply_pre_mix are unchanged.
// residual_out is bit-identical to `partials` (same expression, same
// order); the mix sums differ only in fp32 summation order.
constexpr int PREFILL_TILE = 32;
constexpr int PREFILL_FLATS = 2 * THREADS;
constexpr int PREFILL_FN_STRIDE = PREFILL_FLATS;
constexpr int PREFILL_V_STRIDE = PREFILL_FLATS + 2;
constexpr size_t PREFILL_SMEM =
    size_t(MIXES) * PREFILL_FN_STRIDE * sizeof(float) +
    size_t(PREFILL_TILE) * PREFILL_V_STRIDE * sizeof(__nv_bfloat16);

template <bool FUSED_POST, int HIDDEN_SIZE, typename FnT, bool PAIRED_FN = false>
__global__ void __launch_bounds__(THREADS) partials_prefill(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    const FnT* __restrict__ fn,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ partial,
    int num_tokens) {
    constexpr int TOTAL = HC * HIDDEN_SIZE;
    constexpr int DIMS = HIDDEN_SIZE / SPLITS;
    static_assert(HC * DIMS == PREFILL_FLATS,
                  "partials_prefill: 32 splits of 4 x 128 flats");
    static_assert(DIMS % 64 == 0, "partials_prefill: 8-wide chunks per thread");
    static_assert(PREFILL_TILE == 32 && THREADS == 8 * 32 && MIXES == 24,
                  "partials_prefill: lane = token, warp = 3 mix rows");
    extern __shared__ __align__(16) unsigned char prefill_smem[];
    float* fn_tile = reinterpret_cast<float*>(prefill_smem);
    __nv_bfloat16* v_tile = reinterpret_cast<__nv_bfloat16*>(
        fn_tile + MIXES * PREFILL_FN_STRIDE);

    const int split = blockIdx.x;
    const int token0 = blockIdx.y * PREFILL_TILE;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int dim_split = split * DIMS;

    // Local flat l in [0, 512): stream l / 128, dim dim_split + l % 128.
    // The serving dispatcher enables paired staging only for aligned BF16 fn
    // on SM120. Other dtypes/devices and odd BF16 storage offsets stay scalar.
    // Pair adjacent BF16 loads and exact FP32 conversions without changing the
    // shared layout, arithmetic or reduction order. DIMS/strides and the
    // 16-byte shared base make every source pair and float2 store aligned.
    if constexpr (PAIRED_FN && std::is_same_v<FnT, __nv_bfloat16>) {
        static_assert(PREFILL_FN_STRIDE % 2 == 0 && DIMS % 2 == 0);
        for (int pair = tid; pair < MIXES * PREFILL_FLATS / 2; pair += THREADS) {
            const int i = 2 * pair;
            const int output = i / PREFILL_FLATS;
            const int l = i - output * PREFILL_FLATS;
            const int stream = l / DIMS;
            const auto packed = *reinterpret_cast<const __nv_bfloat162*>(
                fn + output * TOTAL + stream * HIDDEN_SIZE + dim_split + (l - stream * DIMS));
            *reinterpret_cast<float2*>(fn_tile + output * PREFILL_FN_STRIDE + l) =
                __bfloat1622float2(packed);
        }
    } else {
        for (int i = tid; i < MIXES * PREFILL_FLATS; i += THREADS) {
            const int output = i / PREFILL_FLATS;
            const int l = i - output * PREFILL_FLATS;
            const int stream = l / DIMS;
            fn_tile[output * PREFILL_FN_STRIDE + l] =
                float(fn[output * TOTAL + stream * HIDDEN_SIZE + dim_split + (l - stream * DIMS)]);
        }
    }
    // Staging: thread (t = tid / 8, g = tid % 8) owns token t and dims
    // dim_split + g * 16 .. + 16 as two 8-wide chunks; 16-byte loads, all of
    // a chunk's inputs in flight together. With the fused post-mix the five
    // input rows are read once and produce the four output streams.
    {
        constexpr int CHUNKS = DIMS / 64;
        const int t = tid >> 3;
        const int g = tid & 7;
        const int token = token0 + t;
        if (token < num_tokens) {
            __nv_bfloat16* v_row = v_tile + t * PREFILL_V_STRIDE;
            if constexpr (FUSED_POST) {
                float post_mix[HC];
                float comb_mix[HC * HC];
#pragma unroll
                for (int output_stream = 0; output_stream < HC; ++output_stream) {
                    post_mix[output_stream] = post[token * HC + output_stream];
                }
#pragma unroll
                for (int i = 0; i < HC * HC; ++i) {
                    comb_mix[i] = comb[token * HC * HC + i];
                }
#pragma unroll
                for (int c = 0; c < CHUNKS; ++c) {
                    const int dim = dim_split + g * (8 * CHUNKS) + c * 8;
                    const uint4 xv = *reinterpret_cast<const uint4*>(
                        x + size_t(token) * HIDDEN_SIZE + dim);
                    uint4 rv[HC];
#pragma unroll
                    for (int input_stream = 0; input_stream < HC; ++input_stream) {
                        rv[input_stream] = *reinterpret_cast<const uint4*>(
                            residual + (size_t(token) * HC + input_stream) * HIDDEN_SIZE + dim);
                    }
                    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
                    for (int output_stream = 0; output_stream < HC; ++output_stream) {
                        __align__(16) __nv_bfloat16 rounded[8];
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            float value = post_mix[output_stream] * float(xb[j]);
#pragma unroll
                            for (int input_stream = 0; input_stream < HC; ++input_stream) {
                                const __nv_bfloat16* rb =
                                    reinterpret_cast<const __nv_bfloat16*>(&rv[input_stream]);
                                value += comb_mix[input_stream * HC + output_stream] * float(rb[j]);
                            }
                            rounded[j] = __float2bfloat16_rn(value);
                        }
                        *reinterpret_cast<uint4*>(
                            residual_out + (size_t(token) * HC + output_stream) * HIDDEN_SIZE + dim) =
                            *reinterpret_cast<const uint4*>(rounded);
                        __nv_bfloat16* v_dst =
                            v_row + output_stream * DIMS + g * (8 * CHUNKS) + c * 8;
#pragma unroll
                        for (int j = 0; j < 8; j += 2) {
                            *reinterpret_cast<__nv_bfloat162*>(v_dst + j) =
                                *reinterpret_cast<const __nv_bfloat162*>(rounded + j);
                        }
                    }
                }
            } else {
#pragma unroll
                for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
                    for (int c = 0; c < CHUNKS; ++c) {
                        const int dim = dim_split + g * (8 * CHUNKS) + c * 8;
                        const uint4 rv = *reinterpret_cast<const uint4*>(
                            residual + (size_t(token) * HC + stream) * HIDDEN_SIZE + dim);
                        const __nv_bfloat16* rb = reinterpret_cast<const __nv_bfloat16*>(&rv);
                        __nv_bfloat16* v_dst = v_row + stream * DIMS + g * (8 * CHUNKS) + c * 8;
#pragma unroll
                        for (int j = 0; j < 8; j += 2) {
                            *reinterpret_cast<__nv_bfloat162*>(v_dst + j) =
                                *reinterpret_cast<const __nv_bfloat162*>(rb + j);
                        }
                    }
                }
            }
        }
    }
    __syncthreads();

    // lane = token of the tile; warp w = mix rows w, w + 8, w + 16; warp 0
    // also the square sum. v rows are padded by one word so the 32 lanes hit
    // 32 banks; the fn reads are warp-uniform broadcasts (no pad needed).
    const __nv_bfloat16* v_row = v_tile + lane * PREFILL_V_STRIDE;
    const float* fn0 = fn_tile + warp * PREFILL_FN_STRIDE;
    const float* fn1 = fn_tile + (warp + 8) * PREFILL_FN_STRIDE;
    const float* fn2 = fn_tile + (warp + 16) * PREFILL_FN_STRIDE;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, square_sum = 0.0f;
#pragma unroll 8
    for (int f = 0; f < PREFILL_FLATS; f += 2) {
        const float2 v = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(v_row + f));
        const float2 c0 = *reinterpret_cast<const float2*>(fn0 + f);
        const float2 c1 = *reinterpret_cast<const float2*>(fn1 + f);
        const float2 c2 = *reinterpret_cast<const float2*>(fn2 + f);
        acc0 += v.x * c0.x;
        acc0 += v.y * c0.y;
        acc1 += v.x * c1.x;
        acc1 += v.y * c1.y;
        acc2 += v.x * c2.x;
        acc2 += v.y * c2.y;
        if (warp == 0) {
            square_sum += v.x * v.x;
            square_sum += v.y * v.y;
        }
    }
    const int token = token0 + lane;
    if (token < num_tokens) {
        float* out = partial + (token * SPLITS + split) * (MIXES + 1);
        out[warp] = acc0;
        out[warp + 8] = acc1;
        out[warp + 16] = acc2;
        if (warp == 0) {
            out[MIXES] = square_sum;
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

// Phase 1 of the fused pre-transition: one block's slice of the 24 mix
// partials (and the fused post-mix residual write when FUSED_POST).
template <bool FUSED_POST, int HIDDEN_SIZE, int NSPLITS, typename FnT>
__device__ __forceinline__ void pre_transition_partials(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    const FnT* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    int split,
    int token) {
    constexpr int NOUT = MIXES;
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
}

// Phase 2, one block per token: finalize the mixes (Sinkhorn), apply the
// pre-mix to the (post-mixed) residual and write the layer input.
template <bool FUSED_POST, bool RMS_NORM, int HIDDEN_SIZE, int NSPLITS>
__device__ __forceinline__ void pre_transition_tail(
    const __nv_bfloat16* residual,
    const __nv_bfloat16* residual_out,
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
    float norm_eps,
    int token) {
    constexpr int NOUT = MIXES;
    constexpr int VALUES = HIDDEN_SIZE / THREADS;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

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

// Cooperative form: NSPLITS blocks per token, one grid sync, block 0 runs
// the tail. Needs cudaLaunchCooperativeKernel.
template <bool FUSED_POST, bool RMS_NORM, int HIDDEN_SIZE,
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
    const int split = blockIdx.x;
    const int token = blockIdx.y;
    pre_transition_partials<FUSED_POST, HIDDEN_SIZE, NSPLITS, FnT>(
        x, residual, post_mix, comb_mix, fn, residual_out, partial, split,
        token);

    cooperative_groups::this_grid().sync();
    if (split != 0) return;

    pre_transition_tail<FUSED_POST, RMS_NORM, HIDDEN_SIZE, NSPLITS>(
        residual, residual_out, partial, scale, base, next_post, next_comb,
        layer_input, norm_weight, rms_eps, pre_eps, sinkhorn_eps,
        post_multiplier, sinkhorn_repeat, norm_eps, token);
}

// Last-block form (2026-09-07, rtx6000): the same math with a regular
// launch. Every block publishes its partials, fences, and bumps `counter`;
// the block that observes NSPLITS - 1 is the last one, so the partials of
// all others are visible to it (threadFenceReduction pattern), and it runs
// the tail and resets the counter for the next launch on the stream. One
// token per launch (grid NSPLITS x 1); the counter is a persistent device
// int owned by the launcher. Removes the cooperative-launch requirement and
// the grid-wide barrier, which is what the 8.5 us per site was made of.
template <bool FUSED_POST, bool RMS_NORM, int HIDDEN_SIZE,
          int NSPLITS = SPLITS, typename FnT = float>
__global__ void fused_pre_transition_lastblock(
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
    float norm_eps,
    int* counter) {
    static_assert(HIDDEN_SIZE % THREADS == 0);
    const int split = blockIdx.x;
    constexpr int token = 0;
    pre_transition_partials<FUSED_POST, HIDDEN_SIZE, NSPLITS, FnT>(
        x, residual, post_mix, comb_mix, fn, residual_out, partial, split,
        token);

    // Every writer fences its own partials, the block joins, then one thread
    // takes the ticket: the last block observes all NSPLITS partials.
    __shared__ int is_last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        is_last = atomicAdd(counter, 1) == NSPLITS - 1;
    }
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    if (threadIdx.x == 0) *counter = 0;

    pre_transition_tail<FUSED_POST, RMS_NORM, HIDDEN_SIZE, NSPLITS>(
        residual, residual_out, partial, scale, base, next_post, next_comb,
        layer_input, norm_weight, rms_eps, pre_eps, sinkhorn_eps,
        post_multiplier, sinkhorn_repeat, norm_eps, token);
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
