#pragma once
// Fused small-M MoE routing for GLM-5.3-Flash: sigmoid (or sqrt-softplus) scores,
// bias-only top-k selection (noaux_tc / grouped_topk with one expert group),
// renormalize + scaling, and the Marlin block alignment (moe_align_block_size
// semantics) in ONE block, for M x TOPK <= 128 assignments and E <= 512 experts.
// Replaces grouped_topk + moe_align_block_size + count_and_sort + the fill of
// sorted_token_ids: four launches per MoE layer at decode.
//
// The selection is total over non-finite inputs: vLLM's dummy runs (graph-capture
// warmups, the sampler warmup) feed NaN activations, so a NaN score ranks below
// every finite score, a selected expert is marked -inf, and ties go to the lowest
// index. Every expert id is therefore in [0, E) and the block layout is valid
// (valid entries first, then padding) whatever the input. Marlin takes the first
// num_valid entries of a block as tokens, so an id of E would alias the padding
// value numel as a row and touch one row past the activations.
//
// Assignments of one expert are scattered into its range with an atomic cursor,
// so their order within the expert's blocks is unspecified. Every assignment's
// result is written back to its own (token, k) row, so the routed output does
// not depend on that order.
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
    static_assert((E + 31) / 32 > TOPK, "every lane must keep an unselected candidate");
    __shared__ float choice[MAX_TOKENS][E];   // biased scores, -inf once selected
    __shared__ float score[MAX_TOKENS][E];    // unbiased scores (weights)
    __shared__ int counts[E];
    __shared__ int cursor[E];
    __shared__ int cumsum[E + 1];
    __shared__ int sel[MAX_TOKENS][TOPK];      // selected experts, always < E
    __shared__ typename cub::BlockScan<int, THREADS>::TempStorage scan_tmp;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int numel = M * TOPK;

    // A. scores for every (token, expert); a NaN choice ranks below every
    // finite one (-FLT_MAX) but above a selected expert (-inf).
    for (int idx = tid; idx < M * E; idx += THREADS) {
        const int m = idx / E, e = idx - m * E;
        const float s = apply_scoring(logits[idx], scoring);
        score[m][e] = s;
        const float c = s + bias[e];
        choice[m][e] = (c == c) ? c : -FLT_MAX;
    }
    if (tid < E) { counts[tid] = 0; cursor[tid] = 0; }
    __syncthreads();

    // B. top-k per token, one warp per token: TOPK rounds of warp argmax.
    // Each lane scans 9 experts of which at most TOPK are selected (-inf), so
    // every round has a candidate >= -FLT_MAX and yields an index in [0, E).
    if (warp < M) {
        const int m = warp;
        float wsum = 0.0f;
#pragma unroll
        for (int k = 0; k < TOPK; ++k) {
            float best = -INFINITY; int best_e = E;
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
            // Shuffle synchronization does not order shared-memory accesses.
            // Finish every lane's choice reads before lane0 marks the winner;
            // the following sync then publishes that write for the next round.
            __syncwarp();
            if (lane == 0) { choice[m][best_e] = -INFINITY; sel[m][k] = best_e; }
            __syncwarp();
            wsum += score[m][best_e];
        }
        if (lane == 0) {
            const float inv = renormalize ? 1.0f / fmaxf(wsum, 1e-20f) : 1.0f;
#pragma unroll
            for (int k = 0; k < TOPK; ++k) {
                const int e = sel[m][k];
                topk_ids[m * TOPK + k] = e;
                topk_weights[m * TOPK + k] = score[m][e] * inv * scaling;
                atomicAdd(&counts[e], 1);
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
    // D. scatter every assignment into its expert's range.
    for (int a = tid; a < numel; a += THREADS) {
        const int e = sel[a / TOPK][a - (a / TOPK) * TOPK];
        const int pos = cumsum[e] + atomicAdd(&cursor[e], 1);
        sorted_token_ids[pos] = a;
    }
}

}  // namespace tms::glm_route
