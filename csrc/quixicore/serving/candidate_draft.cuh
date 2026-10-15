// Candidate draft sampling for speculative decoding (F5a): the drafter's next token
// drawn from the batch's top-k candidates gathered across the tensor-parallel ranks
// instead of the full [T, vocab] logits, with the same draw as the full path.
//
// The full path (speculator.sample_draft): all-gather the [T, V] logits, temper,
// mask to the request's top-k (ties at the kth value kept) and nucleus (the native
// topk_topp_mask rule: double-precision budget (1 - p) * z over the kept set, the
// lowest-ID kth ties dropped first, a maximum always retained), write the processed
// row into the draft-logits buffer the block rejection sampler reads, and draw
// argmax_g (logit_g + Gumbel(seed, pos, g)) with the (seed, pos, token)-keyed noise of
// v2_gumbel_sample_k. Every token the mask can keep is in the union of the ranks'
// top-K windows (K >= the request's top-k), so the draw over the C = tp x K
// candidates is the same draw: the same kept set (tie groups that overflow a rank's
// window aside), the same processed row (-inf everywhere else) and the same noise
// per token id. One block of CAND_THREADS threads per token, C <= CAND_THREADS.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include "v2_sample_kernels.cuh"

namespace tmv2s {

constexpr int CAND_THREADS = 128;

template <typename VT>
__device__ __forceinline__ VT cand_noise(uint32_t gumbel_seed, int id) {
    if constexpr (sizeof(VT) == 8)
        return VT(tt_gumbel64(uint64_t(gumbel_seed), uint64_t(uint32_t(id))));
    else
        return VT(tt_gumbel32(uint64_t(gumbel_seed), uint64_t(uint32_t(id))));
}

__device__ __forceinline__ double cand_block_sum(double v, double* s_warp) {
    // 128 threads = 4 warps.
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    const double r = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    __syncthreads();
    return r;
}

__device__ __forceinline__ int cand_block_count(int v, int* s_warp) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_warp[warp] = v;
    __syncthreads();
    const int r = s_warp[0] + s_warp[1] + s_warp[2] + s_warp[3];
    __syncthreads();
    return r;
}

template <typename VT>
__global__ void __launch_bounds__(CAND_THREADS) v2_candidate_draft_k(
        const float* __restrict__ cand_logits,      // [T, C] untempered fp32 (-inf where absent)
        const int32_t* __restrict__ cand_ids,       // [T, C] global vocab ids
        int C,
        const int32_t* __restrict__ top_k,          // [max_num_reqs] (vocab_size = none)
        const float* __restrict__ top_p,            // [max_num_reqs] or null
        const int32_t* __restrict__ expanded_idx_mapping,  // [T] request state index, -1 padded
        const int64_t* __restrict__ seeds,          // [max_num_reqs]
        const int64_t* __restrict__ pos_ptr,        // [T] (the draft's salted position)
        const float* __restrict__ temperature,      // [max_num_reqs]
        int V,
        float* __restrict__ processed_logits,       // [max_num_reqs, pl_stride] or null
        int64_t pl_stride, const int64_t* __restrict__ pl_col, int per_token_col,
        int64_t* __restrict__ out) {                // [T]
    static_assert(CAND_THREADS == 128, "four warps");
    __shared__ float sv[CAND_THREADS];
    __shared__ int si[CAND_THREADS];
    __shared__ int s_rank[CAND_THREADS];
    __shared__ double s_dw[4];
    __shared__ int s_iw[4];
    __shared__ float s_kth;
    __shared__ VT s_best[4];
    __shared__ int s_bid[4];
    const int row = blockIdx.x, tid = threadIdx.x;
    const int req = expanded_idx_mapping[row];
    const bool valid_req = req >= 0;
    const float temp = valid_req ? temperature[req] : 0.0f;
    // ---- the candidate, tempered like _temperature_kernel (temp 0 / 1: untouched) ----
    float v = -INFINITY;
    int id = INT_MAX;
    if (tid < C) {
        v = cand_logits[int64_t(row) * C + tid];
        id = cand_ids[int64_t(row) * C + tid];
        if (v != v) v = -INFINITY;                    // NaN sanitizes like the full path
        if (temp != 0.0f && temp != 1.0f) v = tt_div(v, temp);
    }
    sv[tid] = v; si[tid] = id;
    __syncthreads();
    // ---- rank in the native cutoff's order, descending (value, id): among equal
    // values the higher id sorts first and is kept first; the block maximum ----
    int rank = 0;
    float vmax = -INFINITY;
    for (int j = 0; j < C; ++j) {
        const float ov = sv[j];
        vmax = fmaxf(vmax, ov);
        if (ov > v || (ov == v && si[j] > id)) ++rank;
    }
    s_rank[tid] = rank;
    // ---- top-k: the k-th value (ties at it kept) ----
    const int kreq = valid_req ? top_k[req] : V;
    const int k = min(max(kreq, 1), C);
    const bool no_top_k = kreq >= V;                 // vocab_size = no cutoff: every candidate counts
    if (tid < C && rank == k - 1) s_kth = v;
    __syncthreads();
    const float kth = no_top_k ? -INFINITY : s_kth;
    const bool in_topk = tid < C && v > -INFINITY && (no_top_k || v >= kth);
    const bool is_tie = in_topk && !no_top_k && v == kth;
    const bool above = in_topk && !is_tie;
    // ---- nucleus (the native cutoff rule, in double) ----
    const double e = above ? exp(double(v) - double(vmax)) : 0.0;
    const int n_above = cand_block_count(above ? 1 : 0, s_iw);
    const int ties = cand_block_count(is_tie ? 1 : 0, s_iw);
    const double tie_e = (ties > 0) ? exp(double(kth) - double(vmax)) : 0.0;
    const double tie_mass = double(ties) * tie_e;
    const double z = cand_block_sum(e, s_dw) + tie_mass;
    const double p = (top_p != nullptr && valid_req) ? double(top_p[req]) : 1.0;
    const double budget = (1.0 - p) * z;
    int drop = tie_e > 0.0 ? int(fmin(double(ties), floor(budget / tie_e))) : ties;
    drop = min(max(drop, 0), ties);
    if (n_above == 0) drop = min(drop, ties - 1);   // always retain a maximum
    bool kept;
    if (drop < ties) {
        // Cut at the kth value: the `drop` lowest-ID ties go, everything above stays.
        int tie_rank = 0;
        if (is_tie) {
            for (int j = 0; j < C; ++j)
                if (j != tid && sv[j] == kth && si[j] < id) ++tie_rank;
        }
        kept = above || (is_tie && tie_rank >= drop);
    } else {
        // The whole tie group goes; above it, a token stays when the mass of itself
        // and every token below it (plus the ties) exceeds the budget, the maximum always.
        double asc = e;
        if (above) {
            for (int j = 0; j < C; ++j)
                if (j != tid && s_rank[j] > rank) asc += (sv[j] > kth && sv[j] > -INFINITY) ? exp(double(sv[j]) - double(vmax)) : 0.0;
        }
        kept = above && (rank == 0 || asc + tie_mass > budget);
    }
    // ---- the processed row for the rejection sampler: -inf, then the kept candidates ----
    if (processed_logits != nullptr && valid_req) {
        int64_t col = 0;
        if (pl_col != nullptr) col = per_token_col ? pl_col[row] : pl_col[0];
        float* prow = processed_logits + int64_t(req) * pl_stride + col * int64_t(V);
        for (int g = tid; g < V; g += CAND_THREADS) prow[g] = -INFINITY;
        __syncthreads();
        if (kept) prow[id] = v;
    }
    // ---- the draw: argmax of tempered logit + (seed, pos, id)-keyed Gumbel noise ----
    uint32_t gumbel_seed = 0;
    if (temp != 0.0f) {
        const int64_t seed = valid_req ? seeds[req] : 0;
        gumbel_seed = tt_randint(uint64_t(seed), uint64_t(pos_ptr[row]));
    }
    VT score = VT(V2S_NEG_INF);
    int best_id = INT_MAX;
    if (kept) {
        score = VT(v);
        if (temp != 0.0f) score += cand_noise<VT>(gumbel_seed, id);
        best_id = id;
    }
    warp_bfly_argmax(score, best_id);
    const int lane = tid & 31, warp = tid >> 5;
    if (lane == 0) { s_best[warp] = score; s_bid[warp] = best_id; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < 4; ++w) argmax_combine(score, best_id, s_best[w], s_bid[w]);
        // An all -inf row (no kept candidate) degrades to the first candidate id, like
        // the full path's block base.
        out[row] = int64_t(best_id == INT_MAX ? si[0] : best_id);
    }
}

}  // namespace tmv2s
