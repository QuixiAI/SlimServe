#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

#define PARAMS(T) \
    device T* q [[buffer(0)]], \
    device T* k [[buffer(1)]], \
    device T* v [[buffer(2)]], \
    device T* o [[buffer(3)]], \
    constant unsigned &H [[buffer(4)]], \
    constant unsigned &N [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint  laneId   [[thread_index_in_simdgroup]] \

namespace custom_ops {
struct subexp2 {;
    template<typename T> static METAL_FUNC T op(thread const T &a, thread const T &b) { return metal::exp2(a-b); }
};
}
    
template<typename RT, typename RV>
static METAL_FUNC typename metal::enable_if<ducks::is_register_tile<RT>() && ducks::is_register_vector<RV>(), void>::type
subexp2(thread RT &dst, thread const RT &src, thread const RV &row_values) {
    row_map<custom_ops::subexp2, RT, RV>(dst, src, row_values);
}
template<typename RV, typename U>
static METAL_FUNC typename metal::enable_if<ducks::is_register_vector<RV>(), void>::type
subexp2(thread RV &dst, thread const RV &lhs, thread const U &rhs) {
    bin_op<custom_ops::subexp2, RV>(dst, lhs, rhs);
}

constant constexpr const int TN = 8;
// TNQ = query rows per simdgroup. Default 8; a 16-row variant halves the number of passes
// over K/V (each pass streams the whole K/V) and is routed in for D=128 when N%16==0 — at
// (2,16,4096,128) the 8-row tile fell to 7.4 TFLOP/s from K/V re-read pressure.
// softcap: Gemma-2/3-style logit soft-capping, softcap*tanh(s/softcap); <=0 disables. The
// tanh is nonlinear, so with softcap active the log2(e) factor must NOT be folded into Q —
// q_mul stays the raw 1/sqrt(D) scale and the log2 conversion happens after the tanh.
// sinks: gpt-oss-style attention sinks — a per-head extra logit (value contributes nothing
// to O) added to the softmax denominator exactly once. Implemented by seeding the running
// row max with sinks[head]*log2e (so exp2(s - m) <= 1 is unconditionally stable) and adding
// exp2(s_sink - m) to norm_vec after the KV loop. `sinks` is only dereferenced when
// has_sink != 0 — the launcher binds q as a placeholder when absent (buffers can't be null).
template <int D, int TNQ = TN>
kernel void attn_fwd(device   bf16     *q [[buffer(0)]],
                     device   bf16     *k [[buffer(1)]],
                     device   bf16     *v [[buffer(2)]],
                     device   bf16     *o [[buffer(3)]],
                     constant unsigned &N [[buffer(4)]],
                     constant unsigned &H [[buffer(5)]],
                     constant float    &softcap [[buffer(6)]],
                     device const float *sinks  [[buffer(7)]],
                     constant int      &has_sink [[buffer(8)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_k(k, nullptr, H, N, nullptr);
    global_layout gl_v(v, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using st_qkv     = st_bf<TN, D>;
    using rt_qkv     = rt_bf<TNQ, D>;
    using rt_k_t     = rt_bf<TN, D, ducks::rt_layout::col>;
    using rt_att     = rt_fl<TNQ, TN>;
    using rt_att_mma = rt_bf<TNQ, TN>;
    using rt_o       = rt_fl<TNQ, D>;
    using rv_att     = typename rt_fl<TNQ, TN>::col_vec;

    const int block = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq = blockIdx.x;
    
    const int kv_blocks = N / st_qkv::rows;
    rt_qkv q_reg;
    rt_k_t k_reg;
    rt_bf<TN, D> v_reg;   // K/V tiles stay 8 rows regardless of the Q-tile height
    rt_att att_block;
    rt_o o_reg;
    rv_att max_vec_last;
    rv_att max_vec;
    rv_att norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    const bool capped = softcap > 0.0f;
    constexpr const float scale = (D == 128) ? 0.08838834764f : 0.125f;
    const float sink_l2 = (has_sink != 0) ? sinks[head] * 1.44269504089f : 0.0f;
    if (has_sink != 0) {                       // seed running max with the sink logit
        zero(max_vec);
        add(max_vec, max_vec, sink_l2);
    } else {
        neg_infty(max_vec);
    }
    zero(norm_vec);
    zero(o_reg);
    const bf16 q_mul = bf16(capped ? scale : scale * 1.44269504089f);
    mul(q_reg, q_reg, q_mul);
    #pragma clang loop unroll(full)
    for(auto kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (capped) {                          // s = softcap*tanh(s/softcap), then -> log2 domain
            mul(att_block, att_block, 1.0f / softcap);
            tanh(att_block, att_block);
            mul(att_block, att_block, softcap * 1.44269504089f);
        }
        copy(max_vec_last,  max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        
//        subexp2(max_vec_last, max_vec_last, max_vec);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
//        subexp2(att_block, att_block, max_vec);
        
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, gl_v, {block, head, kv_idx, 0}, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
    }
    if (has_sink != 0) {                       // denominator-only sink term, added exactly once
        rv_att sink_term;
        copy(sink_term, max_vec, laneId);
        mul(sink_term, sink_term, -1.0f);
        add(sink_term, sink_term, sink_l2);
        exp2(sink_term, sink_term);            // exp2(s_sink - m) <= 1 by the max seed
        add(norm_vec, norm_vec, sink_term);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}


#define instantiate_add_custom(D)                                \
  template [[host_name("attn_fwd_" #D)]] [[kernel]] void         \
  attn_fwd<D>(device   bf16     *q [[buffer(0)]], \
    device   bf16     *k [[buffer(1)]], \
    device   bf16     *v [[buffer(2)]], \
    device   bf16     *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], \
    constant unsigned &H [[buffer(5)]], \
    constant float    &softcap [[buffer(6)]], \
    device const float *sinks  [[buffer(7)]], \
    constant int      &has_sink [[buffer(8)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]); \

instantiate_add_custom(64);
instantiate_add_custom(128);

// 16-row Q tiles (routed in when N % 16 == 0): halves the passes over K/V.
#define instantiate_attn_fwd_q16(D)                                     \
  template [[host_name("attn_fwd_" #D "_q16")]] [[kernel]] void         \
  attn_fwd<D, 16>(device   bf16     *q [[buffer(0)]],                   \
                  device   bf16     *k [[buffer(1)]],                   \
                  device   bf16     *v [[buffer(2)]],                   \
                  device   bf16     *o [[buffer(3)]],                   \
                  constant unsigned &N [[buffer(4)]],                   \
                  constant unsigned &H [[buffer(5)]],                   \
                  constant float    &softcap [[buffer(6)]],             \
                  device const float *sinks  [[buffer(7)]],             \
                  constant int      &has_sink [[buffer(8)]],            \
                  uint3 blockIdx [[threadgroup_position_in_grid]],      \
                  uint laneId [[thread_index_in_simdgroup]]);

instantiate_attn_fwd_q16(128);

}
