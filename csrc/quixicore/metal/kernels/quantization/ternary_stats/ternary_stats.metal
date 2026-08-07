#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Ternary-health monitors over PACKED wq (train_plan §10.2 / moe_train_plan §6.2)
// — the step-0-mandated metrics without unpacking 14.5B params in PyTorch:
//
//   ternary_stats:   per-row {-1, 0, +1} code counts (the zero-code fraction is
//                    counts[:,1]/K — the decay-erosion alarm's numerator; callers
//                    reshape rows -> experts and reduce).
//   code_flip_count: per-row count of code positions that differ between two
//                    packs of the same layout (the code-flip rate's numerator;
//                    compare the eval-interval snapshot against the current pack).
//
// Input layout: the 10-byte `bitnet` block {half scale; uchar qs[8]}, rows of
// nblocks blocks. One simdgroup per row; grid (rows, 1, 1), 32 threads. The scale
// bytes are skipped — these kernels never touch float math at all.
// ---------------------------------------------------------------------------

kernel void ternary_stats(device const uchar *wq     [[buffer(0)]],
                          device uint        *counts [[buffer(1)]],   // (rows, 3): [-1, 0, +1]
                          constant int       &nblocks [[buffer(2)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    device const uchar *base = wq + (long)row * nblocks * 10;
    uint neg = 0, zero = 0, pos = 0;
    const int nbytes = nblocks * 8;
    for (int i = (int)lane; i < nbytes; i += 32) {
        const uchar b = base[(i / 8) * 10 + 2 + (i % 8)];
        #pragma clang loop unroll(full)
        for (int t = 0; t < 4; ++t) {
            const uint code = (b >> (2 * t)) & 3u;                    // {0,1,2} -> {-1,0,+1}
            neg  += uint(code == 0u);
            zero += uint(code == 1u);
            pos  += uint(code == 2u);
        }
    }
    neg = simd_sum(neg); zero = simd_sum(zero); pos = simd_sum(pos);
    if (lane == 0) {
        counts[(long)row * 3 + 0] = neg;
        counts[(long)row * 3 + 1] = zero;
        counts[(long)row * 3 + 2] = pos;
    }
}

kernel void code_flip_count(device const uchar *wq_a  [[buffer(0)]],
                            device const uchar *wq_b  [[buffer(1)]],
                            device uint        *flips [[buffer(2)]],   // (rows,)
                            constant int       &nblocks [[buffer(3)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    device const uchar *ba = wq_a + (long)row * nblocks * 10;
    device const uchar *bb = wq_b + (long)row * nblocks * 10;
    uint n = 0;
    const int nbytes = nblocks * 8;
    for (int i = (int)lane; i < nbytes; i += 32) {
        const int off = (i / 8) * 10 + 2 + (i % 8);
        const uchar x = ba[off] ^ bb[off];                            // 2-bit fields differ iff != 0
        #pragma clang loop unroll(full)
        for (int t = 0; t < 4; ++t) { n += uint(((x >> (2 * t)) & 3u) != 0u); }
    }
    n = simd_sum(n);
    if (lane == 0) { flips[row] = n; }
}
