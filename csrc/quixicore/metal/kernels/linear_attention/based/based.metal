#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Based: 2nd-order Taylor feature-map linear attention (causal), bf16. D_QK=16, D_VO=64.
//   phi(q).phi(k) = 1 + (q.k) + (q.k)^2/2   (the 2nd-order Taylor approximation of exp)
//   out_i = sum_{j<=i} [ 1 + x_ij + x_ij^2/2 ] * v_j,   x_ij = (q_i . k_j)/sqrt(D_QK)
// Materialized chunked form (one simdgroup per (batch,head,query-chunk), loops key-chunks <= query;
// same shape as lin_attn_causal/mamba2): A = 1 + x + x^2/2 built elementwise, causal-masked on the
// diagonal block, then O = A @ V. Unnormalized numerator (the "1" term contributes cumsum(V)), which
// matches the TK based kernel. q,k are D_QK-wide; v,o are D_VO-wide.
template <int DQK, int DVO>
kernel void based(device   bf16     *q [[buffer(0)]],
                  device   bf16     *k [[buffer(1)]],
                  device   bf16     *v [[buffer(2)]],
                  device   bf16     *o [[buffer(3)]],
                  constant unsigned &N [[buffer(4)]],
                  constant unsigned &H [[buffer(5)]],
                  uint3 blockIdx [[threadgroup_position_in_grid]],
                  uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(DQK == 16 && DVO == 64, "based currently supports D_QK=16, D_VO=64");
    using gl_qk = gl<bfloat, 1, -1, -1, DQK>;          // q,k : (B,H,N,16)
    using gl_vo = gl<bfloat, 1, -1, -1, DVO>;          // v,o : (B,H,N,64)
    gl_qk gq(q, nullptr, H, N, nullptr);
    gl_qk gk(k, nullptr, H, N, nullptr);
    gl_vo gv(v, nullptr, H, N, nullptr);
    gl_vo go(o, nullptr, H, N, nullptr);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qi = blockIdx.x;                          // this query chunk
    constexpr const float temp = 0.25f;                 // 1/sqrt(D_QK) = 1/sqrt(16)

    rt_bf<8, DQK> q_reg;
    load(q_reg, gq, {batch, head, qi, 0}, laneId);

    rt_fl<8, DVO> o_reg;
    zero(o_reg);

    for (int kj = 0; kj <= qi; kj++) {
        rt_bf<8, DQK, ducks::rt_layout::col> k_reg;      // col layout for Q @ K^T
        rt_bf<8, DVO> v_reg;
        load(k_reg, gk, {batch, head, kj, 0}, laneId);
        load(v_reg, gv, {batch, head, kj, 0}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);                 // x' = Q @ K^T  (over D_QK)
        mul(att, att, temp);                             // x = x'/sqrt(D_QK)

        rt_fl<8, 8> sq;
        copy(sq, att);
        mul(sq, sq, sq);                                 // x^2
        mul(sq, sq, 0.5f);                               // x^2/2
        add(att, att, sq);                               // x + x^2/2
        add(att, att, 1.0f);                             // 1 + x + x^2/2  (Taylor phi.phi)

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);    // future positions -> 0
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(o_reg, att_bf, v_reg, o_reg);             // O += A @ V
    }
    store(go, o_reg, {batch, head, qi, 0}, laneId);
}

#define instantiate_based(DQK, DVO)                                    \
  template [[host_name("based_" #DQK "_" #DVO)]] [[kernel]] void       \
  based<DQK, DVO>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device bf16 *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_based(16, 64);

}
