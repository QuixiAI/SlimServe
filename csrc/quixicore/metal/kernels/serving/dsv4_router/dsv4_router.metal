// SPDX-License-Identifier: Apache-2.0
// DeepSeek-V4 MoE router top-k on Metal: collapses the eager
// _topk_softplus_sqrt_torch chain (softplus -> sqrt -> +bias -> topk ->
// gather -> renorm -> scale, ~11 MPS dispatches per MoE layer call) into
// one dispatch. scores = sqrt(softplus(logits)); selection on
// scores + bias (or expert ids from the hash table for hash-MoE layers);
// weights from the UNBIASED scores; optional renorm to sum 1 with the
// torch clamp(min=1e-20); final scale by routed_scaling_factor.
//
// gating must already hold softplus(logits): no Metal softplus formula
// matches MPS aten::softplus bitwise, so the host runs torch's softplus
// eagerly (one dispatch) and the kernel starts from the bitwise-identical
// output; every remaining op (sqrt, +bias, select, gather, renorm divide,
// scale) is correctly rounded and mirrored per op.
//
// Selection tie order is deterministic (choice desc, expert index asc);
// torch.topk's order is implementation-defined, so a tie may flip the
// sampled trajectory.
//
// One simdgroup per token; lane owns experts e = lane + 32*j. K <= 8.

#include <metal_stdlib>

// float -> order-preserving uint (handles negatives).
METAL_FUNC uint qc_f32_sortable(float v) {
    uint b = as_type<uint>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

kernel void dsv4_router_topk(
        device const float *gating     [[buffer(0)]],  // (T, E) fp32
        device const float *bias       [[buffer(1)]],  // (E,) fp32 (has_bias)
        device const int   *hash_table [[buffer(2)]],  // (vocab, K) (has_hash)
        device const int   *input_ids  [[buffer(3)]],  // (T,) (has_hash)
        device float       *out_w      [[buffer(4)]],  // (T, K) fp32
        device int         *out_ids    [[buffer(5)]],  // (T, K) int32
        constant int &num_experts      [[buffer(6)]],
        constant int &topk             [[buffer(7)]],
        constant int &has_bias         [[buffer(8)]],
        constant int &has_hash         [[buffer(9)]],
        constant int &renorm           [[buffer(10)]],
        constant float &scaling        [[buffer(11)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint laneId [[thread_index_in_simdgroup]]) {
    constexpr int MAX_PER_LANE = 32;  // E <= 1024
    constexpr int MAX_K = 8;
    const int token = blockIdx.x;
    device const float *g = gating + (long)token * num_experts;

    float win_s[MAX_K];
    int win_id[MAX_K];

    if (has_hash) {
        // Expert ids predetermined by the hash table; weights from the
        // unbiased scores. topk <= 8 scalar reads per lane, identical in
        // every lane so the renorm sum needs no broadcast.
        const long tok_id = input_ids[token];
        for (int k = 0; k < topk; ++k) {
            const int e = hash_table[tok_id * topk + k];
            win_id[k] = e;
            win_s[k] = metal::precise::sqrt(g[e]);
        }
    } else {
        // Scores for my experts, packed sortable keys on choice = s + bias.
        float s_l[MAX_PER_LANE];
        ulong key_l[MAX_PER_LANE];
        const int per_lane = (num_experts + 31) / 32;
        for (int j = 0; j < per_lane; ++j) {
            const int e = (int)laneId + 32 * j;
            if (e < num_experts) {
                const float s = metal::precise::sqrt(g[e]);
                const float c = has_bias ? (s + bias[e]) : s;
                s_l[j] = s;
                key_l[j] = ((ulong)qc_f32_sortable(c) << 32) |
                           (ulong)(0xFFFFFFFFu - (uint)e);
            } else {
                s_l[j] = 0.0f;
                key_l[j] = 0;
            }
        }
        for (int k = 0; k < topk; ++k) {
            ulong best = 0;
            int best_j = 0;
            for (int j = 0; j < per_lane; ++j) {
                if (key_l[j] > best) { best = key_l[j]; best_j = j; }
            }
            // simd_max has no 64-bit overload: reduce (hi, lo) in two
            // stages — lanes not holding the max hi drop out of the lo max.
            const uint hi = (uint)(best >> 32);
            const uint win_hi = metal::simd_max(hi);
            const uint lo = (hi == win_hi) ? (uint)(best & 0xFFFFFFFFu) : 0u;
            const uint win_lo = metal::simd_max(lo);
            const ulong winner = ((ulong)win_hi << 32) | (ulong)win_lo;
            const bool mine = (best == winner) && (winner != 0);
            win_s[k] = metal::simd_max(mine ? s_l[best_j] : -1.0f);
            win_id[k] = (int)(0xFFFFFFFFu - win_lo);
            if (mine) { key_l[best_j] = 0; }
        }
    }

    // Renorm + scale with torch's rounding points:
    //   w = w / clamp(sum, 1e-20); out = w * scaling
    // MPS reduces the K-wide sum as a pairwise tree (probed 2026-08-11:
    // ((w0+w1)+(w2+w3))+(w4+w5) is bitwise for K=6; sequential is 2 ulp
    // off) — mirror that exactly.
    float denom = 1.0f;
    if (renorm) {
        float acc[MAX_K];
        int n = topk;
        {
            #pragma clang fp reassociate(off) contract(off)
            for (int k = 0; k < topk; ++k) { acc[k] = win_s[k]; }
            while (n > 1) {
                const int nh = n / 2;
                for (int i = 0; i < nh; ++i) {
                    acc[i] = acc[2 * i] + acc[2 * i + 1];
                }
                if (n & 1) { acc[nh] = acc[n - 1]; }
                n = nh + (n & 1);
            }
        }
        denom = metal::max(acc[0], 1e-20f);
    }
    if ((int)laneId < topk) {
        // Fast-math folds divide(w, denom) * scaling into a reciprocal
        // multiply (2-4 ulp off torch's CR divide); pin the chain.
        #pragma clang fp reassociate(off) contract(off)
        const int k = (int)laneId;
        float w = win_s[k];
        if (renorm) { w = metal::precise::divide(w, denom); }
        out_w[(long)token * topk + k] = w * scaling;
        out_ids[(long)token * topk + k] = win_id[k];
    }
}
