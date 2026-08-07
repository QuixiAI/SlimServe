/**
 * @file
 * @brief Threadgroup parallel prefix-sum (P2).
 *
 * Single-tile exclusive scan across a threadgroup: per-simdgroup inclusive scan
 * (simd_prefix_inclusive_sum) plus a cross-simdgroup base offset combined through
 * threadgroup memory. Wrap it in a strided loop with a running prefix to scan
 * arbitrary-length per-row data (MoE expert offsets; nucleus cumulative mass).
 *
 * Free function (NOT a group<> member): included from ops/ops.metal, not from
 * group/group.metal (whose includes nest inside the group<> struct body).
 */

#pragma once

// SIMD_THREADS + <metal_stdlib> for standalone compile; pragma-once-safe in the umbrella
// (utils.metal is already included by common.metal before this file).
#include "../../common/utils.metal"

namespace mittens {

/**
 * @brief Exclusive prefix sum of `val` across the threadgroup's first `nthreads`
 *        threads (tid in [0,nthreads), nthreads a multiple of 32).
 *
 * @param sg_sums threadgroup scratch with >= nthreads/32 ints.
 * @param total   (out) sum of all vals — identical on every thread.
 * @return this thread's exclusive prefix (sum of vals from lower tids).
 */
static METAL_FUNC int threadgroup_exclusive_scan_i32(
        int val, uint tid, uint nthreads, threadgroup int *sg_sums, thread int &total) {
    const uint lane = tid % SIMD_THREADS;
    const uint sg   = tid / SIMD_THREADS;
    const uint nsg  = (nthreads + SIMD_THREADS - 1) / SIMD_THREADS;

    const int incl = metal::simd_prefix_inclusive_sum(val);   // inclusive within simdgroup
    if (lane == SIMD_THREADS - 1) {
        sg_sums[sg] = incl;                                   // this simdgroup's total
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    int base = 0, t = 0;
    for (uint i = 0; i < nsg; ++i) {
        const int s = sg_sums[i];
        if (i < sg) { base += s; }
        t += s;
    }
    total = t;
    return base + (incl - val);   // simdgroup base + (inclusive - own value)
}

} // namespace mittens
