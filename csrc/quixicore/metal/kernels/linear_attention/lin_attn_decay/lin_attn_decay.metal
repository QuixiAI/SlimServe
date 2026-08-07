#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Decay / retention linear attention (RetNet / Lightning-Attention-2), bf16, D=64.
//   out_i = sum_{j<=i} exp(-slope*(i-j)) * (q_i . k_j) * v_j
// i.e. causal linear attention with a per-head EXPONENTIAL DECAY on the score by token distance —
// distinct from lin_attn_causal (no decay). Materialized chunked form (one simdgroup per
// (batch, head, query-chunk), loops key-chunks <= query): O = ((Q@K^T) (.) L (.) causal) @ V, with
// the decay matrix L[i,j] = exp(decay_i[i] - decay_j[j]) built by add_row/sub_col/exp — the same
// decay-tile mechanic as mamba2, here fed the geometric ramp decay = -slope*position (the tk wrapper
// builds `cl[b,h,n] = -slopes[h]*n` from the RetNet per-head slope; positions are implicit).
template <int D>
kernel void lin_attn_decay(device   bf16     *q  [[buffer(0)]],
                           device   bf16     *k  [[buffer(1)]],
                           device   bf16     *v  [[buffer(2)]],
                           device   float    *cl [[buffer(3)]],   // -slope*position ramp (B,H,N) fp32
                           device   bf16     *o  [[buffer(4)]],
                           constant unsigned &N  [[buffer(5)]],
                           constant unsigned &H  [[buffer(6)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "lin_attn_decay currently supports D=64");
    using gl_t  = gl<bfloat, 1, -1, -1, D>;            // q,k,v,o : (B,H,N,D)
    using gl_cl = gl<float, 1, -1, 1, -1>;             // decay ramp (B,H,N) viewed along cols
    gl_t  gq(q, nullptr, H, N, nullptr);
    gl_t  gk(k, nullptr, H, N, nullptr);
    gl_t  gv(v, nullptr, H, N, nullptr);
    gl_t  go(o, nullptr, H, N, nullptr);
    gl_cl gcl(cl, nullptr, H, nullptr, N);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int qi = blockIdx.x;                          // this query chunk

    rt_bf<8, D> q_reg;
    load(q_reg, gq, {batch, head, qi, 0}, laneId);
    typename rt_fl<8, 8>::col_vec decay_i;              // per query row  (-slope*pos)
    load(decay_i, gcl, {batch, head, 0, qi}, laneId);

    rt_fl<8, D> o_reg;
    zero(o_reg);

    for (int kj = 0; kj <= qi; kj++) {
        rt_bf<8, D, ducks::rt_layout::col> k_reg;        // col layout for Q @ K^T
        rt_bf<8, D> v_reg;
        load(k_reg, gk, {batch, head, kj, 0}, laneId);
        load(v_reg, gv, {batch, head, kj, 0}, laneId);
        typename rt_fl<8, 8>::row_vec decay_j;           // per key col
        load(decay_j, gcl, {batch, head, 0, kj}, laneId);

        rt_fl<8, 8> att;
        zero(att);
        mma_ABt(att, q_reg, k_reg, att);                 // Q @ K^T

        // L[i,j] = exp(decay_i[i] - decay_j[j]) = exp(-slope*(pos_i - pos_j))
        rt_fl<8, 8> decay;
        zero(decay);
        add_row(decay, decay, decay_i);
        sub_col(decay, decay, decay_j);
        exp(decay, decay);
        mul(att, att, decay);

        if (kj == qi) {
            float zero_fill = 0.0f;
            make_causal(att, att, laneId, zero_fill);    // future positions -> 0
        }

        rt_bf<8, 8> att_bf;
        copy(att_bf, att);
        mma_AB(o_reg, att_bf, v_reg, o_reg);
    }
    store(go, o_reg, {batch, head, qi, 0}, laneId);
}

#define instantiate_lin_attn_decay(D)                                  \
  template [[host_name("lin_attn_decay_" #D)]] [[kernel]] void         \
  lin_attn_decay<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device float *cl [[buffer(3)]], \
    device bf16 *o [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], constant unsigned &H [[buffer(6)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_lin_attn_decay(64);

}
