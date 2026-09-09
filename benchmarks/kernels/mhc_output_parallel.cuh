#pragma once
// Isolated experimental mHC decode partials, outside the serving source tree.
// Reuse the existing FP32 weights, FP32 post-mix expression/BF16 rounding,
// cooperative synchronization and nonlinear tail. Each warp owns three of the
// 24 outputs and traverses the complete split, replacing 25 warp reductions
// per warp with three (plus the square sum in warp zero). This trades repeated
// cached residual/post-mix work for fewer shuffles; it needs measured validation.
#include "mhc_ampere.cuh"

namespace tms::dsv4_mhc {

template <bool FUSED_POST, int HIDDEN_SIZE, int NSPLITS>
__device__ __forceinline__ void output_parallel_partials(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    const float* fn,
    __nv_bfloat16* residual_out,
    float* partial,
    int split,
    int token) {
    static_assert(THREADS == 256 && MIXES == 24);
    static_assert((HC * HIDDEN_SIZE) % NSPLITS == 0);
    constexpr int TOTAL = HC * HIDDEN_SIZE;
    constexpr int FLATS = TOTAL / NSPLITS;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    __shared__ float coeffs[HC + HC * HC];
    if constexpr (FUSED_POST) {
        if (tid < HC) coeffs[tid] = post_mix[token * HC + tid];
        if (tid < HC * HC) coeffs[HC + tid] = comb_mix[token * HC * HC + tid];
        __syncthreads();
    }

    float accum[3] = {};
    float square_sum = 0.0f;
#pragma unroll
    for (int local = lane; local < FLATS; local += 32) {
        const int flat = split * FLATS + local;
        float value;
        if constexpr (FUSED_POST) {
            const int stream = flat / HIDDEN_SIZE;
            const int dim = flat % HIDDEN_SIZE;
            value = coeffs[stream] * float(x[token * HIDDEN_SIZE + dim]);
#pragma unroll
            for (int input = 0; input < HC; ++input) {
                value += coeffs[HC + input * HC + stream] *
                         float(residual[(token * HC + input) * HIDDEN_SIZE + dim]);
            }
            const __nv_bfloat16 rounded = __float2bfloat16_rn(value);
            if (warp == 0) residual_out[token * TOTAL + flat] = rounded;
            value = float(rounded);
        } else {
            value = float(residual[token * TOTAL + flat]);
        }
#pragma unroll
        for (int output = 0; output < 3; ++output) {
            accum[output] += value * fn[(warp * 3 + output) * TOTAL + flat];
        }
        if (warp == 0) square_sum += value * value;
    }
#pragma unroll
    for (int output = 0; output < 3; ++output) {
        const float sum = warp_sum(accum[output]);
        if (lane == 0) {
            partial[(token * NSPLITS + split) * (MIXES + 1) + warp * 3 + output] = sum;
        }
    }
    if (warp == 0) {
        square_sum = warp_sum(square_sum);
        if (lane == 0) partial[(token * NSPLITS + split) * (MIXES + 1) + MIXES] = square_sum;
    }
}

template <bool FUSED_POST, bool RMS_NORM, int NSPLITS = 64>
__global__ void fused_pre_transition_output_parallel(
    const __nv_bfloat16* x,
    const __nv_bfloat16* residual,
    const float* post_mix,
    const float* comb_mix,
    const float* fn,
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
    const int split = blockIdx.x;
    const int token = blockIdx.y;
    output_parallel_partials<FUSED_POST, 4096, NSPLITS>(
        x, residual, post_mix, comb_mix, fn, residual_out, partial, split, token);
    cooperative_groups::this_grid().sync();
    if (split != 0) return;
    pre_transition_tail<FUSED_POST, RMS_NORM, 4096, NSPLITS>(
        residual, residual_out, partial, scale, base, next_post, next_comb,
        layer_input, norm_weight, rms_eps, pre_eps, sinkhorn_eps,
        post_multiplier, sinkhorn_repeat, norm_eps, token);
}

}  // namespace tms::dsv4_mhc
