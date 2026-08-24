// SPDX-License-Identifier: Apache-2.0
// MoE finalize (weighted expert-row sum) for the Metal GGUF grouped path:
// collapses the eager tail in _fused_moe_gguf's Metal branch —
//   reduced = (out.float() * topk_weights.unsqueeze(-1)).sum(dim=1)
//   out_hidden_states.copy_(reduced.to(dtype))
// (cast, mul, strided sum, cast, copy) — into one dispatch, with torch's
// exact rounding points: the fp32 product is rounded per element, then
// summed SEQUENTIALLY over the expert slots (MPS reduces the strided
// [T,K,D] dim=1 sum sequentially; the pairwise tree it uses for
// contiguous K-wide sums does NOT match here — contrast the router's
// renorm, which mirrors the pairwise tree), then rounded to the output
// dtype. One 256-thread threadgroup per token, d strided.

#include <metal_stdlib>

template <typename T>
inline void qc_moe_weighted_sum_body(
        device const T *x,       // (T, K, D) expert rows
        device const float *w,   // (T, K) fp32 weights
        device T *y,             // (T, D)
        int K, int D, uint token, uint tid) {
    device const T *xt = x + (ulong)token * K * D;
    device T *yt = y + (ulong)token * D;
    float wk[8];  // topk <= 8 host-checked
    for (int k = 0; k < K; ++k) { wk[k] = w[(ulong)token * K + k]; }
    // 256 = the launcher's threads-per-threadgroup (launch_qc_moe_weighted_sum).
    for (int d = (int)tid; d < D; d += 256) {
        float acc;
        {
            #pragma clang fp reassociate(off) contract(off)
            acc = float(xt[d]) * wk[0];
            for (int k = 1; k < K; ++k) {
                const float p = float(xt[(ulong)k * D + d]) * wk[k];
                acc += p;
            }
        }
        yt[d] = T(acc);
    }
}

kernel void qc_moe_weighted_sum_float16(
        device const half *x    [[buffer(0)]],
        device const float *w   [[buffer(1)]],
        device half *y          [[buffer(2)]],
        constant int &K         [[buffer(3)]],
        constant int &D         [[buffer(4)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    qc_moe_weighted_sum_body<half>(x, w, y, K, D, token, tid);
}

kernel void qc_moe_weighted_sum_bfloat16(
        device const bfloat *x [[buffer(0)]],
        device const float *w     [[buffer(1)]],
        device bfloat *y       [[buffer(2)]],
        constant int &K           [[buffer(3)]],
        constant int &D           [[buffer(4)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    qc_moe_weighted_sum_body<bfloat>(x, w, y, K, D, token, tid);
}
