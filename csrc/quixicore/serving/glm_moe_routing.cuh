#pragma once
// Fused small-M MoE routing for GLM-5.3-Flash: sigmoid (or sqrt-softplus) scores,
// bias-only top-k selection (noaux_tc / grouped_topk with one expert group),
// renormalize + scaling, and the Marlin block alignment (moe_align_block_size
// semantics) in ONE block, for M x TOPK <= 128 assignments and E <= 512 experts.
// Replaces grouped_topk + moe_align_block_size (2 blocks) + count_and_sort + the
// fill kernel, four launches per MoE layer at decode. Developed and measured on
// sm_120 (2026-09-04): ids / weights / alignment identical to the reference path,
// 6.3 -> 3.7 us per layer at M=1, 6.8 -> 4.8 at M=8.
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cfloat>
#include <cstdint>

namespace tms::glm_route {

constexpr int THREADS = 512;
constexpr int MAX_TOKENS = 16;

enum Scoring : int { SIGMOID = 0, SQRT_SOFTPLUS = 1 };

__device__ __forceinline__ float apply_scoring(float x, int scoring) {
    if (scoring == SIGMOID) return 1.0f / (1.0f + expf(-x));
    // torch.nn.functional.softplus with beta=1, threshold=20, then sqrt.
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    return sqrtf(sp);
}

template <int E, int TOPK>
__global__ __launch_bounds__(THREADS) void route_align_kernel(
        const float* __restrict__ logits,      // [M, E]
        const float* __restrict__ bias,        // [E]
        float* __restrict__ topk_weights,      // [M, TOPK]
        int32_t* __restrict__ topk_ids,        // [M, TOPK]
        int32_t* __restrict__ sorted_token_ids,// [max_padded]
        int32_t* __restrict__ expert_ids,      // [max_blocks]
        int32_t* __restrict__ num_tokens_post_pad,
        int M, int scoring, float scaling, bool renormalize, int block_size,
        int max_padded, int max_blocks) {
    static_assert(E <= THREADS, "one thread per expert for the scan");
    __shared__ float choice[MAX_TOKENS][E];   // biased scores, -inf once selected
    __shared__ float score[MAX_TOKENS][E];    // unbiased scores (weights)
    __shared__ int counts[E];
    __shared__ int cursor[E];
    __shared__ int cumsum[E + 1];
    __shared__ typename cub::BlockScan<int, THREADS>::TempStorage scan_tmp;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int numel = M * TOPK;

    // A. scores for every (token, expert)
    for (int idx = tid; idx < M * E; idx += THREADS) {
        const int m = idx / E, e = idx - m * E;
        const float s = apply_scoring(logits[idx], scoring);
        score[m][e] = s;
        choice[m][e] = s + bias[e];
    }
    if (tid < E) { counts[tid] = 0; cursor[tid] = 0; }
    __syncthreads();

    // B. top-k per token, one warp per token: TOPK rounds of warp argmax.
    if (warp < M) {
        const int m = warp;
        float wsum = 0.0f;
        int sel[TOPK];
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float best = -FLT_MAX; int best_e = E;
            for (int e = lane; e < E; e += 32) {
                const float v = choice[m][e];
                if (v > best || (v == best && e < best_e)) { best = v; best_e = e; }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, best, off);
                const int oe = __shfl_xor_sync(0xffffffffu, best_e, off);
                if (ov > best || (ov == best && oe < best_e)) { best = ov; best_e = oe; }
            }
            if (lane == 0) choice[m][best_e] = -FLT_MAX;
            __syncwarp();
            sel[k] = best_e;
            wsum += score[m][best_e];
        }
        if (lane == 0) {
            const float inv = renormalize ? 1.0f / fmaxf(wsum, 1e-20f) : 1.0f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                topk_ids[m * TOPK + k] = sel[k];
                topk_weights[m * TOPK + k] = score[m][sel[k]] * inv * scaling;
                atomicAdd(&counts[sel[k]], 1);
            }
        }
    }
    __syncthreads();

    // C. alignment: per-expert padded counts, exclusive scan, block expert ids.
    int padded = 0;
    if (tid < E) padded = ((counts[tid] + block_size - 1) / block_size) * block_size;
    int excl = 0, total = 0;
    cub::BlockScan<int, THREADS>(scan_tmp).ExclusiveSum(padded, excl, total);
    if (tid < E) cumsum[tid] = excl;
    if (tid == 0) { cumsum[E] = total; num_tokens_post_pad[0] = total; }
    __syncthreads();
    for (int i = tid; i < max_padded; i += THREADS) sorted_token_ids[i] = numel;
    if (tid < E) {
        for (int i = cumsum[tid]; i < cumsum[tid + 1]; i += block_size) expert_ids[i / block_size] = tid;
    }
    for (int b = total / block_size + tid; b < max_blocks; b += THREADS) expert_ids[b] = -1;
    __syncthreads();
    // scatter the assignments (token*TOPK + k) into their expert's range
    for (int a = tid; a < numel; a += THREADS) {
        const int e = topk_ids[a];
        const int pos = cumsum[e] + atomicAdd(&cursor[e], 1);
        sorted_token_ids[pos] = a;
    }
}

}  // namespace tms::glm_route
