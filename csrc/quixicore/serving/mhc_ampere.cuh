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

// partials_batched with the 24 outputs split across the 8 warps (3 each).
// partials_batched keeps TT x NOUT accumulators plus NOUT fn values per
// thread (~140 registers): one block per SM, so its fn loads (24 x 64 KiB
// per token tile, L2-resident) run latency-bound - 31 us at T=32, 98 us at
// T=128 on the GLM-5.3 TP4 x DP2 profile (2026-09-12; a transpose-reduce
// epilogue made it slower, so the epilogue was not the cost). Here the
// block's 2 x THREADS elements are mixed ONCE into shared memory (phase 1,
// also the residual_out write), then warp w streams fn rows [3w, 3w + 3)
// over the chunk with TT x 3 accumulators per lane (phase 2); the square
// sum rides on warp 0. Registers ~40, so 4-6 blocks per SM keep the L2
// loads in flight. Same per-element arithmetic and bf16 rounding as
// partials_batched; the fp32 partial sums group differently (finalize
// tolerates any grouping).
template <int NOUT, int TT, typename FnT = float>
__global__ void partials_batched_ws(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    const FnT* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    int hidden_size,
    int tokens) {
    constexpr int WARPS = THREADS / 32;
    constexpr int OPW = NOUT / WARPS;   // outputs per warp
    static_assert(NOUT % WARPS == 0, "outputs must split evenly across warps");
    constexpr int EPT = 2;              // elements per thread (hidden 4096 x 4 streams / 32 splits)
    constexpr int CHUNK = THREADS * EPT;
    const int split = blockIdx.x;
    const int token0 = blockIdx.y * TT;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int total = HC * hidden_size;
    __shared__ float mix_coeffs[TT][HC + HC * HC];
    __shared__ float values[TT][CHUNK];
    for (int i = tid; i < TT * (HC + HC * HC); i += THREADS) {
        const int t = i / (HC + HC * HC);
        const int j = i - t * (HC + HC * HC);
        const int token = token0 + t;
        float v = 0.0f;
        if (token < tokens) {
            v = (j < HC) ? post[token * HC + j] : comb[token * HC * HC + (j - HC)];
        }
        mix_coeffs[t][j] = v;
    }
    __syncthreads();
    // Phase 1: mix, round, write residual_out, stage values. Element set
    // per block = the strided set of partials_batched (split * THREADS +
    // k * SPLITS * THREADS + tid), so residual_out coverage is identical.
    // All EPT x TT x (1 + HC) loads are issued up front with clamped
    // indices (predicating them per token serialized the block on HBM
    // latency); the token bound only gates the store and the staged value.
    const int last_token = tokens - 1;
    __nv_bfloat16 xv[EPT][TT];
    __nv_bfloat16 rv[EPT][TT][HC];
    int flats[EPT];
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int flat = split * THREADS + e * SPLITS * THREADS + tid;
        flats[e] = flat;
        const int cflat = flat < total ? flat : 0;
        const int stream = cflat / hidden_size;
        const int dim = cflat - stream * hidden_size;
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = min(token0 + t, last_token);
            xv[e][t] = x[token * hidden_size + dim];
#pragma unroll
            for (int input_stream = 0; input_stream < HC; ++input_stream) {
                rv[e][t][input_stream] = residual[(token * HC + input_stream) * hidden_size + dim];
            }
        }
    }
#pragma unroll
    for (int e = 0; e < EPT; ++e) {
        const int flat = flats[e];
        const bool in = flat < total;
        const int stream = (in ? flat : 0) / hidden_size;
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = token0 + t;
            float value = 0.0f;
            if (in && token < tokens) {
                value = mix_coeffs[t][stream] * float(xv[e][t]);
#pragma unroll
                for (int input_stream = 0; input_stream < HC; ++input_stream) {
                    value += mix_coeffs[t][HC + input_stream * HC + stream] * float(rv[e][t][input_stream]);
                }
                const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
                residual_out[token * total + flat] = rounded;
                value = float(rounded);
            }
            values[t][e * THREADS + tid] = value;
        }
    }
    __syncthreads();
    // Phase 2: warp w owns outputs [w * OPW, (w + 1) * OPW).
    float acc[TT][OPW];
    float sq[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        sq[t] = 0.0f;
#pragma unroll
        for (int j = 0; j < OPW; ++j) acc[t][j] = 0.0f;
    }
    // Fully unrolled so all CHUNK / 32 x OPW fn loads are in flight at once
    // (a rolled loop serialized on L2 latency: 38 us at T=128).
#pragma unroll
    for (int it = 0; it < CHUNK / 32; ++it) {
        const int c = it * 32 + lane;
        const int e = c / THREADS;
        const int flat = split * THREADS + e * SPLITS * THREADS + (c - e * THREADS);
        const bool in = flat < total;
        float f[OPW];
#pragma unroll
        for (int j = 0; j < OPW; ++j) {
            f[j] = in ? float(fn[(warp * OPW + j) * total + flat]) : 0.0f;
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const float value = values[t][c];
#pragma unroll
            for (int j = 0; j < OPW; ++j) acc[t][j] += value * f[j];
            if (warp == 0) sq[t] += value * value;
        }
    }
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        const int token = token0 + t;
#pragma unroll
        for (int j = 0; j < OPW; ++j) {
            const float sum = warp_sum(acc[t][j]);
            if (lane == 0 && token < tokens) {
                partial[(token * SPLITS + split) * (NOUT + 1) + warp * OPW + j] = sum;
            }
        }
        if (warp == 0) {
            const float sum = warp_sum(sq[t]);
            if (lane == 0 && token < tokens) {
                partial[(token * SPLITS + split) * (NOUT + 1) + NOUT] = sum;
            }
        }
    }
}

// Batched form of `partials` for decode batches: one block owns (split,
// tile of TT tokens) and loads each fn column once for all TT tokens. The
// per-token kernel re-reads the whole fn matrix (NOUT x 4 x hidden fp32,
// 1.5 MiB) for every token, which is what made it 78 us at 32 tokens on
// the GLM-5.3 TP4 x DP2 profile (2026-09-11); with TT tokens per block the
// fn traffic drops by TT. Same arithmetic and rounding per token, same
// partial layout, so finalize_pre_mix is unchanged.
template <int NOUT, int TT, typename FnT = float>
__global__ void partials_batched(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    const FnT* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    int hidden_size,
    int tokens) {
    const int split = blockIdx.x;
    const int token0 = blockIdx.y * TT;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int total = HC * hidden_size;
    __shared__ float mix_coeffs[TT][HC + HC * HC];
    for (int i = tid; i < TT * (HC + HC * HC); i += THREADS) {
        const int t = i / (HC + HC * HC);
        const int j = i - t * (HC + HC * HC);
        const int token = token0 + t;
        float v = 0.0f;
        if (token < tokens) {
            v = (j < HC) ? post[token * HC + j] : comb[token * HC * HC + (j - HC)];
        }
        mix_coeffs[t][j] = v;
    }
    __syncthreads();
    float accum[TT][NOUT];
    float square_sum[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        square_sum[t] = 0.0f;
#pragma unroll
        for (int output = 0; output < NOUT; ++output) accum[t][output] = 0.0f;
    }
    for (int flat = split * THREADS + tid; flat < total;
         flat += SPLITS * THREADS) {
        const int stream = flat / hidden_size;
        const int dim = flat - stream * hidden_size;
        float f[NOUT];
#pragma unroll
        for (int output = 0; output < NOUT; ++output) {
            f[output] = float(fn[output * total + flat]);
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = token0 + t;
            if (token < tokens) {
                float value = mix_coeffs[t][stream] * float(x[token * hidden_size + dim]);
#pragma unroll
                for (int input_stream = 0; input_stream < HC; ++input_stream) {
                    value += mix_coeffs[t][HC + input_stream * HC + stream] *
                             float(residual[(token * HC + input_stream) * hidden_size + dim]);
                }
                const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
                residual_out[token * total + flat] = rounded;
                value = float(rounded);
                square_sum[t] += value * value;
#pragma unroll
                for (int output = 0; output < NOUT; ++output) {
                    accum[t][output] += value * f[output];
                }
            }
        }
    }
    __shared__ float warp_partials[THREADS / 32][NOUT + 1];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        const int token = token0 + t;
        if (token >= tokens) break;
#pragma unroll
        for (int output = 0; output < NOUT; ++output) {
            const float sum = warp_sum(accum[t][output]);
            if (lane == 0) warp_partials[warp][output] = sum;
        }
        const float sum = warp_sum(square_sum[t]);
        if (lane == 0) warp_partials[warp][NOUT] = sum;
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
        __syncthreads();
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

// partials_batched with the outputs split across two half-blocks: warps
// [0, THREADS/64) accumulate outputs [0, NOUT/2), the rest [NOUT/2, NOUT).
// Each thread then carries TT x NOUT/2 accumulators, so TT=8 fits the
// register budget of the TT=4 kernel above while halving the fn traffic
// again (fn is read once per 8-token tile). x/residual reads double (each
// half reads them), which is small next to fn at these T.
template <int NOUT, int TT, typename FnT = float>
__global__ void partials_batched_split(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post,
    const float* comb,
    const FnT* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    int hidden_size,
    int tokens) {
    static_assert(NOUT % 2 == 0 && THREADS % 64 == 0);
    constexpr int HALF = NOUT / 2;
    constexpr int HT = THREADS / 2;       // threads per output half
    const int split = blockIdx.x;
    const int token0 = blockIdx.y * TT;
    const int tid = threadIdx.x;
    const int half = tid / HT;            // which output half this thread serves
    const int htid = tid - half * HT;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int total = HC * hidden_size;
    __shared__ float mix_coeffs[TT][HC + HC * HC];
    for (int i = tid; i < TT * (HC + HC * HC); i += THREADS) {
        const int t = i / (HC + HC * HC);
        const int j = i - t * (HC + HC * HC);
        const int token = token0 + t;
        float v = 0.0f;
        if (token < tokens) {
            v = (j < HC) ? post[token * HC + j] : comb[token * HC * HC + (j - HC)];
        }
        mix_coeffs[t][j] = v;
    }
    __syncthreads();
    float accum[TT][HALF];
    float square_sum[TT];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        square_sum[t] = 0.0f;
#pragma unroll
        for (int o = 0; o < HALF; ++o) accum[t][o] = 0.0f;
    }
    // Both halves walk the same flat elements (each with HT threads), so the
    // element set per split matches partials_batched exactly.
    for (int flat = split * HT + htid; flat < total; flat += SPLITS * HT) {
        const int stream = flat / hidden_size;
        const int dim = flat - stream * hidden_size;
        float f[HALF];
#pragma unroll
        for (int o = 0; o < HALF; ++o) {
            f[o] = float(fn[(half * HALF + o) * total + flat]);
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            const int token = token0 + t;
            if (token < tokens) {
                float value = mix_coeffs[t][stream] * float(x[token * hidden_size + dim]);
#pragma unroll
                for (int input_stream = 0; input_stream < HC; ++input_stream) {
                    value += mix_coeffs[t][HC + input_stream * HC + stream] *
                             float(residual[(token * HC + input_stream) * hidden_size + dim]);
                }
                const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
                if (half == 0) residual_out[token * total + flat] = rounded;
                value = float(rounded);
                square_sum[t] += value * value;
#pragma unroll
                for (int o = 0; o < HALF; ++o) accum[t][o] += value * f[o];
            }
        }
    }
    __shared__ float warp_partials[THREADS / 32][HALF + 1];
#pragma unroll
    for (int t = 0; t < TT; ++t) {
        const int token = token0 + t;
        if (token >= tokens) break;
#pragma unroll
        for (int o = 0; o < HALF; ++o) {
            const float sum = warp_sum(accum[t][o]);
            if (lane == 0) warp_partials[warp][o] = sum;
        }
        const float sum = warp_sum(square_sum[t]);
        if (lane == 0) warp_partials[warp][HALF] = sum;
        __syncthreads();
        if (warp == 0) {
            // outputs [0, HALF) from the first half's warps; [HALF, NOUT)
            // from the second half's; the square sum from the first half.
            for (int o = lane; o < NOUT + 1; o += 32) {
                float block_sum = 0.0f;
                if (o < HALF) {
#pragma unroll
                    for (int w = 0; w < THREADS / 64; ++w) block_sum += warp_partials[w][o];
                } else if (o < NOUT) {
#pragma unroll
                    for (int w = THREADS / 64; w < THREADS / 32; ++w) block_sum += warp_partials[w][o - HALF];
                } else {
#pragma unroll
                    for (int w = 0; w < THREADS / 64; ++w) block_sum += warp_partials[w][HALF];
                }
                partial[(token * SPLITS + split) * (NOUT + 1) + o] = block_sum;
            }
        }
        __syncthreads();
    }
}

// finalize_pre_mix + apply_pre_mix_rms_norm in one launch, with a wider
// block. The two-kernel form ran T blocks of 32 threads (finalize) and T
// blocks of 256 threads (norm) - at T=32 that is 32 blocks on 108 SMs,
// occupancy-bound (5.4 + 10.1 us per site on the GLM-5.3 c32 profile,
// 2026-09-12). Warp 0 runs the finalize (warp-shuffle only, no block
// barrier inside), then all NT threads apply the coefficients and the
// RMS norm exactly as apply_pre_mix_rms_norm does.
template <int PARTIAL_WIDTH, int NT, int HIDDEN_SIZE = 4096>
__global__ void __launch_bounds__(NT)
finalize_apply_pre_mix_rms_norm(
    float* partial,
    const float* scale,
    const float* base,
    float* post,
    float* comb,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* norm_weight,
    __nv_bfloat16* output,
    int hidden_size,
    float rms_eps,
    float pre_eps,
    float sinkhorn_eps,
    float post_multiplier,
    int sinkhorn_repeat,
    float norm_eps) {
    static_assert(HIDDEN_SIZE % NT == 0);
    constexpr int VALUES = HIDDEN_SIZE / NT;
    const int token = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    if (warp == 0) {
        finalize_pre_mix_block<SPLITS>(
            partial, scale, base, post, comb, token, hidden_size, rms_eps,
            pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_repeat);
    }
    __syncthreads();
    const float* pre_mix = partial + token * SPLITS * PARTIAL_WIDTH;
    const int total = HC * HIDDEN_SIZE;
    __nv_bfloat16 values[VALUES];
    float square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int dim = tid + i * NT;
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
    __shared__ float warp_sums[NT / 32];
    __shared__ float inverse_rms;
    square_sum = warp_sum(square_sum);
    if (lane == 0) {
        warp_sums[warp] = square_sum;
    }
    __syncthreads();
    if (warp == 0) {
        float block_sum = lane < NT / 32 ? warp_sums[lane] : 0.0f;
        block_sum = warp_sum(block_sum);
        if (lane == 0) {
            inverse_rms = rsqrtf(block_sum / float(HIDDEN_SIZE) + norm_eps);
        }
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
        const int dim = tid + i * NT;
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
