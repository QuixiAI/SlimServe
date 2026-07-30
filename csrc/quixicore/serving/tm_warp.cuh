/**
 * @file
 * @brief Shared warp/block device helpers for the ThunderMittens-port kernels
 * (CUDA twins of TM's simd_* / masked_topk / threadgroup scan primitives).
 * Single source; the serving/decode/elementwise kernels all include this.
 *
 * Tie-breaking contract: argmax ties resolve toward the SMALLER id (numpy
 * semantics) - every sampler and topk consumer depends on it.
 */
#pragma once
#include <cstdint>

namespace tms {

__device__ __forceinline__ float warp_sum_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_max_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ float warp_min_f(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}
__device__ __forceinline__ int warp_sum_i32(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ void warp_argmax(float& best, int& bi) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, off);
        const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
}

/// K rounds of masked-argmax top-k over n scanned candidates (TM Family-A).
/// cand(idx, &id, &v, &valid) reads candidate idx; every lane ends with the
/// same chosen[]/chosen_val[]; empty rounds write id -1 / val neg_inf.
template <typename FN>
__device__ inline void masked_topk(FN&& cand, int n, int K, int lane, float neg_inf,
                                   int* chosen, float* chosen_val) {
    for (int kk = 0; kk < K; kk++) {
        float best = neg_inf;
        int bi = -1;
        for (int idx = lane; idx < n; idx += 32) {
            int id; float v; bool valid;
            cand(idx, id, v, valid);
            if (!valid) continue;
            bool taken = false;
            for (int m = 0; m < kk; m++) if (chosen[m] == id) taken = true;
            if (taken) continue;
            if (v > best || (v == best && id < bi)) { best = v; bi = id; }
        }
        int gid = (bi < 0) ? 0x7fffffff : bi;
        warp_argmax(best, gid);
        chosen[kk] = (best == neg_inf) ? -1 : gid;
        chosen_val[kk] = best;
    }
}

/// K rounds over per-lane LOCAL candidate sets (TM Family-B): each lane holds
/// nloc (val,id) slots; the winner's owning lane clears its slot; emit(kk,
/// gbest, gid) runs on ALL lanes each round (guard lane==0 inside if needed).
template <typename EMIT>
__device__ inline void masked_topk_local(float* mine_val, int* mine_id, bool* used, int nloc,
                                         int K, float neg_inf, EMIT&& emit) {
    for (int j = 0; j < nloc; j++) used[j] = false;
    for (int kk = 0; kk < K; kk++) {
        float best = neg_inf;
        int bi = -1, bl = -1;
        for (int j = 0; j < nloc; j++) {
            if (used[j]) continue;
            if (mine_val[j] > best || (mine_val[j] == best && mine_id[j] < bi)) {
                best = mine_val[j]; bi = mine_id[j]; bl = j;
            }
        }
        float gbest = best;
        int gid = (bi < 0) ? 0x7fffffff : bi;
        warp_argmax(gbest, gid);
        if (bl >= 0 && bi == gid) used[bl] = true;
        emit(kk, gbest, gid);
    }
}

/// Exclusive prefix sum of `val` across the block's first nthreads threads
/// (multiple of 32). sg_sums: shared scratch >= nthreads/32 ints. total = block
/// sum on every thread. Contains a __syncthreads (whole block must call).
__device__ inline int block_exclusive_scan_i32(int val, unsigned tid, unsigned nthreads,
                                               int* sg_sums, int& total) {
    const unsigned lane = tid % 32, sg = tid / 32, nsg = (nthreads + 31) / 32;
    int incl = val;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int up = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= unsigned(off)) incl += up;
    }
    if (lane == 31) sg_sums[sg] = incl;
    __syncthreads();
    int base = 0, t = 0;
    for (unsigned i = 0; i < nsg; i++) {
        const int s = sg_sums[i];
        if (i < sg) base += s;
        t += s;
    }
    total = t;
    return base + (incl - val);
}

/// Float INCLUSIVE prefix sum of `val` across the block's first nthreads
/// threads (multiple of 32). sg_sums: shared scratch >= nthreads/32 floats.
/// Returns the inclusive scan (running sum through and including this thread);
/// `total` = block sum on every thread. Contains a __syncthreads. For the
/// skew-transform sampler (CDF over a probability row).
__device__ inline float block_inclusive_scan_f(float val, unsigned tid, unsigned nthreads,
                                               float* sg_sums, float& total) {
    const unsigned lane = tid % 32, sg = tid / 32, nsg = (nthreads + 31) / 32;
    float incl = val;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const float up = __shfl_up_sync(0xffffffffu, incl, off);
        if (lane >= unsigned(off)) incl += up;
    }
    if (lane == 31) sg_sums[sg] = incl;
    __syncthreads();
    float base = 0.0f, t = 0.0f;
    for (unsigned i = 0; i < nsg; i++) {
        const float s = sg_sums[i];
        if (i < sg) base += s;
        t += s;
    }
    total = t;
    return base + incl;
}

}  // namespace tms
