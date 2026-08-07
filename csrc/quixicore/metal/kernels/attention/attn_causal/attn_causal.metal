#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Causal flash-attention forward (warp-level), bf16, D in {64,128}.
// Same online-softmax structure as attn_fwd, with causal masking:
//   - only iterate kv blocks kv_idx <= q_seq (later keys are fully masked),
//   - on the diagonal block (kv_idx == q_seq) apply make_causal (set the
//     strictly-upper-triangular scores to -inf before the softmax).
constant constexpr const int TNc = 8;

// softcap / sinks: same semantics + implementation as attn_fwd (see the comment there).
// Ordering matters vs masking: the softcap tanh applies to RAW scores BEFORE make_causal /
// make_windowed — capping after would compress the -1e30 mask to a finite -softcap and
// un-mask future positions.
template <int D>
kernel void attn_causal(device   bf16     *q [[buffer(0)]],
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
    using st_qkv     = st_bf<TNc, D>;
    using rt_qkv     = rt_bf<TNc, D>;
    using rt_k_t     = rt_bf<TNc, D, ducks::rt_layout::col>;
    using rt_att     = rt_fl<TNc, TNc>;
    using rt_o       = rt_fl<TNc, D>;
    using rv_att     = rt_fl<TNc, TNc>::col_vec;

    const int block = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq = blockIdx.x;

    rt_qkv q_reg;
    rt_k_t k_reg;
    rt_qkv v_reg;
    rt_att att_block;
    rt_o o_reg;
    rv_att max_vec_last;
    rv_att max_vec;
    rv_att norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    const bool capped = softcap > 0.0f;
    constexpr const float scale_c = (D == 128) ? 0.08838834764f : 0.125f;
    const float sink_l2 = (has_sink != 0) ? sinks[head] * 1.44269504089f : 0.0f;
    if (has_sink != 0) {
        zero(max_vec);
        add(max_vec, max_vec, sink_l2);
    } else {
        neg_infty(max_vec);
    }
    zero(norm_vec);
    zero(o_reg);
    const bf16 q_mul = bf16(capped ? scale_c : scale_c * 1.44269504089f);
    mul(q_reg, q_reg, q_mul);

    // Causal: only attend to key blocks at or before the query block.
    #pragma clang loop unroll(full)
    for(int kv_idx = 0; kv_idx <= q_seq; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (capped) {                          // BEFORE the mask (see header comment)
            mul(att_block, att_block, 1.0f / softcap);
            tanh(att_block, att_block);
            mul(att_block, att_block, softcap * 1.44269504089f);
        }
        if (kv_idx == q_seq) {
            // strictly-upper-triangular scores -> -inf (mask future positions)
            float neg_big = -1e30f;   // thread-space (make_causal takes thread const&)
            make_causal(att_block, att_block, laneId, neg_big);
        }
        copy(max_vec_last,  max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, gl_v, {block, head, kv_idx, 0}, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
    }
    if (has_sink != 0) {
        rv_att sink_term;
        copy(sink_term, max_vec, laneId);
        mul(sink_term, sink_term, -1.0f);
        add(sink_term, sink_term, sink_l2);
        exp2(sink_term, sink_term);
        add(norm_vec, norm_vec, sink_term);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}

#define instantiate_attn_causal(D)                               \
  template [[host_name("attn_causal_" #D)]] [[kernel]] void      \
  attn_causal<D>(device   bf16     *q [[buffer(0)]], \
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

instantiate_attn_causal(64);
instantiate_attn_causal(128);

// Sliding-window causal flash-attention forward (Mistral/Gemma-style local attention).
// window = W > 0: a query at position i attends keys j in [max(0, i-W+1), i] — the W most
// recent tokens including self (flash-attention window_size_left = W-1, right = 0).
// Structure = attn_causal with (a) the kv loop's LOWER bound clamped to the window
// (kj_lo = max(0, q0 - W + 1) / 8, clamped BEFORE dividing) and (b) make_windowed applied to
// the (at most two) boundary kv tiles the band edge cuts through: for the tile pair
// (q_seq, kv_idx) the element (r, c) is out-of-window iff (r - c) >= shift with
// shift = window - 8*(q_seq - kv_idx); shift >= 8 means fully in-window (no-op).
template <int D>
kernel void attn_window(device   bf16     *q [[buffer(0)]],
                        device   bf16     *k [[buffer(1)]],
                        device   bf16     *v [[buffer(2)]],
                        device   bf16     *o [[buffer(3)]],
                        constant unsigned &N [[buffer(4)]],
                        constant unsigned &H [[buffer(5)]],
                        constant int      &window [[buffer(6)]],
                        constant float    &softcap [[buffer(7)]],
                        device const float *sinks  [[buffer(8)]],
                        constant int      &has_sink [[buffer(9)]],
                        uint3 blockIdx [[threadgroup_position_in_grid]],
                        uint laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_k(k, nullptr, H, N, nullptr);
    global_layout gl_v(v, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using rt_qkv     = rt_bf<TNc, D>;
    using rt_k_t     = rt_bf<TNc, D, ducks::rt_layout::col>;
    using rt_att     = rt_fl<TNc, TNc>;
    using rt_o       = rt_fl<TNc, D>;
    using rv_att     = rt_fl<TNc, TNc>::col_vec;

    const int block = blockIdx.z;
    const int head = blockIdx.y;
    const int q_seq = blockIdx.x;

    rt_qkv q_reg;
    rt_k_t k_reg;
    rt_qkv v_reg;
    rt_att att_block;
    rt_o o_reg;
    rv_att max_vec_last;
    rv_att max_vec;
    rv_att norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    const bool capped = softcap > 0.0f;
    constexpr const float scale_w = (D == 128) ? 0.08838834764f : 0.125f;
    const float sink_l2 = (has_sink != 0) ? sinks[head] * 1.44269504089f : 0.0f;
    if (has_sink != 0) {
        zero(max_vec);
        add(max_vec, max_vec, sink_l2);
    } else {
        neg_infty(max_vec);
    }
    zero(norm_vec);
    zero(o_reg);
    const bf16 q_mul = bf16(capped ? scale_w : scale_w * 1.44269504089f);
    mul(q_reg, q_reg, q_mul);

    // oldest key any row of this q tile can reach is q_seq*8 - window + 1 (row 0's bound)
    const int kj_lo = (window > 0) ? metal::max(0, q_seq * 8 - window + 1) / 8 : 0;
    for(int kv_idx = kj_lo; kv_idx <= q_seq; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (capped) {                          // BEFORE the masks (see attn_causal comment)
            mul(att_block, att_block, 1.0f / softcap);
            tanh(att_block, att_block);
            mul(att_block, att_block, softcap * 1.44269504089f);
        }
        if (kv_idx == q_seq) {
            float neg_big = -1e30f;
            make_causal(att_block, att_block, laneId, neg_big);
        }
        if (window > 0) {
            const int shift = window - 8 * (q_seq - kv_idx);
            if (shift <= 7) {   // band edge cuts this tile
                float neg_big = -1e30f;
                make_windowed(att_block, att_block, laneId, shift, neg_big);
            }
        }
        copy(max_vec_last,  max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, gl_v, {block, head, kv_idx, 0}, laneId);
        mma_AB(o_reg, att_block, v_reg, o_reg);
    }
    if (has_sink != 0) {
        rv_att sink_term;
        copy(sink_term, max_vec, laneId);
        mul(sink_term, sink_term, -1.0f);
        add(sink_term, sink_term, sink_l2);
        exp2(sink_term, sink_term);
        add(norm_vec, norm_vec, sink_term);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}

#define instantiate_attn_window(D)                               \
  template [[host_name("attn_window_" #D)]] [[kernel]] void      \
  attn_window<D>(device   bf16     *q [[buffer(0)]], \
    device   bf16     *k [[buffer(1)]], \
    device   bf16     *v [[buffer(2)]], \
    device   bf16     *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], \
    constant unsigned &H [[buffer(5)]], \
    constant int      &window [[buffer(6)]], \
    constant float    &softcap [[buffer(7)]], \
    device const float *sinks  [[buffer(8)]], \
    constant int      &has_sink [[buffer(9)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]); \

instantiate_attn_window(64);
instantiate_attn_window(128);

}
