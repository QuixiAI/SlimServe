#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// FlashAttention-2 BACKWARD (dQ/dK/dV), bf16, D in {64,128}, non-causal + causal.
// Ported from TK attention/mha_h100.cu (bwd_attend_prep_ker + bwd_attend_ker). Forward exp2-domain:
// scores S2 = (scale*log2e) * Q·Kᵀ, L = rowmax(S2) + log2(rowsum exp2(S2-rowmax)). The backward
// recomputes P = exp2(S2 - L) (the softmax probabilities) and forms:
//   D_i = rowsum(dO_i ∘ O_i);  dP = dO·Vᵀ;  dS = P∘(dP - D_i);  dQ = scale·dS·K;  dK = scale·dSᵀ·Q;
//   dV = Pᵀ·dO.   (dS/dV/dK contract over the query index → swap_layout to feed mma_AtB.)
//
// Decomposed into three kernels (one simdgroup per 8-row block, no atomics):
//   attn_fwd_l   — forward that also writes L (B,H,N) for the backward.
//   attn_bwd_prep— D_i = rowsum(dO∘O)  (B,H,N).
//   attn_bwd_dq  — fix query block i, loop kv j, accumulate dQ_i.
//   attn_bwd_dkv — fix kv block j, loop query i, accumulate dK_j, dV_j.

constant constexpr const int TB = 8;

template <int D> constexpr float bwd_scale();
template <> constexpr float bwd_scale<64>()  { return 0.125f; }
template <> constexpr float bwd_scale<128>() { return 0.08838834764f; }

// ---------- forward + logsumexp L ----------
template <int D, bool CAUSAL>
kernel void attn_fwd_l(device   bf16     *q [[buffer(0)]],
                       device   bf16     *k [[buffer(1)]],
                       device   bf16     *v [[buffer(2)]],
                       device   bf16     *o [[buffer(3)]],
                       device   float    *L [[buffer(4)]],
                       constant unsigned &N [[buffer(5)]],
                       constant unsigned &H [[buffer(6)]],
                       uint3 blockIdx [[threadgroup_position_in_grid]],
                       uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    using gl_L = gl<float, 1, -1, 1, -1>;
    gl_t gl_q(q, nullptr, H, N, nullptr), gl_k(k, nullptr, H, N, nullptr);
    gl_t gl_v(v, nullptr, H, N, nullptr), gl_o(o, nullptr, H, N, nullptr);
    gl_L gl_l(L, nullptr, H, nullptr, N);

    const int block = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x;
    const int kv_last = CAUSAL ? q_seq : (int)(N / TB) - 1;
    rt_bf<TB, D> q_reg, v_reg;
    rt_bf<TB, D, ducks::rt_layout::col> k_reg;
    rt_fl<TB, TB> att_block;
    rt_fl<TB, D> o_reg;
    typename rt_fl<TB, TB>::col_vec max_vec_last, max_vec, norm_vec;

    load(q_reg, gl_q, {block, head, q_seq, 0}, laneId);
    neg_infty(max_vec); zero(norm_vec); zero(o_reg);
    constexpr const bf16 q_mul = (bf16)(bwd_scale<D>() * 1.44269504089f);
    mul(q_reg, q_reg, q_mul);
    for (int kv_idx = 0; kv_idx <= kv_last; kv_idx++) {
        load(k_reg, gl_k, {block, head, kv_idx, 0}, laneId);
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (CAUSAL && kv_idx == q_seq) { float nb = -1e30f; make_causal(att_block, att_block, laneId, nb); }
        copy(max_vec_last, max_vec, laneId);
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
    // L = max + log2(norm)   (log2-domain logsumexp, matching the exp2 forward)
    typename rt_fl<TB, TB>::col_vec lvec, lognorm;
    log(lognorm, norm_vec);
    mul(lognorm, lognorm, 1.44269504089f);
    add(lvec, max_vec, lognorm);
    store(gl_l, lvec, {block, head, 0, q_seq}, laneId);
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_seq, 0}, laneId);
}

// ---------- prep: D_i = rowsum(dO_i ∘ O_i) ----------
template <int D>
kernel void attn_bwd_prep(device   bf16     *o  [[buffer(0)]],
                          device   bf16     *ddo [[buffer(1)]],
                          device   float    *delta [[buffer(2)]],
                          constant unsigned &N [[buffer(3)]],
                          constant unsigned &H [[buffer(4)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    using gl_d = gl<float, 1, -1, 1, -1>;
    gl_t gl_o(o, nullptr, H, N, nullptr), gl_do(ddo, nullptr, H, N, nullptr);
    gl_d gl_delta(delta, nullptr, H, nullptr, N);

    const int block = blockIdx.z, head = blockIdx.y, q_seq = blockIdx.x;
    rt_bf<TB, D> o_bf, do_bf;
    load(o_bf, gl_o, {block, head, q_seq, 0}, laneId);
    load(do_bf, gl_do, {block, head, q_seq, 0}, laneId);
    rt_fl<TB, D> o_fl, do_fl;
    copy(o_fl, o_bf); copy(do_fl, do_bf);
    mul(o_fl, o_fl, do_fl);
    typename rt_fl<TB, D>::col_vec delta_vec;
    zero(delta_vec);
    row_sum(delta_vec, o_fl, delta_vec, laneId);
    store(gl_delta, delta_vec, {block, head, 0, q_seq}, laneId);
}

// ---------- dQ: fix query block i, loop kv j ----------
template <int D, bool CAUSAL>
kernel void attn_bwd_dq(device   bf16     *q  [[buffer(0)]],
                        device   bf16     *k  [[buffer(1)]],
                        device   bf16     *v  [[buffer(2)]],
                        device   bf16     *ddo [[buffer(3)]],
                        device   float    *L  [[buffer(4)]],
                        device   float    *delta [[buffer(5)]],
                        device   bf16     *dq [[buffer(6)]],
                        constant unsigned &N  [[buffer(7)]],
                        constant unsigned &H  [[buffer(8)]],
                        uint3 blockIdx [[threadgroup_position_in_grid]],
                        uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    using gl_v = gl<float, 1, -1, 1, -1>;
    gl_t gl_q(q, nullptr, H, N, nullptr), gl_k(k, nullptr, H, N, nullptr);
    gl_t gl_vv(v, nullptr, H, N, nullptr), gl_do(ddo, nullptr, H, N, nullptr), gl_dq(dq, nullptr, H, N, nullptr);
    gl_v gl_L(L, nullptr, H, nullptr, N), gl_del(delta, nullptr, H, nullptr, N);

    const int block = blockIdx.z, head = blockIdx.y, i = blockIdx.x;
    constexpr const float scale = bwd_scale<D>();
    constexpr const bf16 q_mul = (bf16)(scale * 1.44269504089f);

    rt_bf<TB, D> q_reg, do_reg;
    load(q_reg, gl_q, {block, head, i, 0}, laneId);
    load(do_reg, gl_do, {block, head, i, 0}, laneId);
    typename rt_fl<TB, TB>::col_vec L_i, del_i;
    load(L_i, gl_L, {block, head, 0, i}, laneId);
    load(del_i, gl_del, {block, head, 0, i}, laneId);

    rt_fl<TB, D> dq_reg;
    zero(dq_reg);
    const int kv_last = CAUSAL ? i : (int)(N / TB) - 1;
    for (int j = 0; j <= kv_last; j++) {
        rt_bf<TB, D, ducks::rt_layout::col> k_col, v_col;
        rt_bf<TB, D> k_row;
        load(k_row, gl_k, {block, head, j, 0}, laneId);
        swap_layout(k_col, k_row, laneId);
        load(v_col, gl_vv, {block, head, j, 0}, laneId);

        rt_fl<TB, TB> att;
        zero(att);
        mma_ABt(att, q_reg, k_col, att);          // Q_i·Kᵀ_j  (i×j)
        mul(att, att, (float)q_mul);
        sub_row(att, att, L_i);                    // - L_i (per query row)
        exp2(att, att);                            // P (i×j)
        if (CAUSAL && j == i) { float zf = 0.0f; make_causal(att, att, laneId, zf); }

        rt_fl<TB, TB> dP;
        zero(dP);
        mma_ABt(dP, do_reg, v_col, dP);            // dO_i·Vᵀ_j  (i×j)
        sub_row(dP, dP, del_i);                    // dP - D_i
        rt_fl<TB, TB> dS;
        mul(dS, att, dP);                          // P ∘ (dP - D_i)
        rt_bf<TB, TB> dS_bf;
        copy(dS_bf, dS);
        mma_AB(dq_reg, dS_bf, k_row, dq_reg);      // += dS·K_j  (i×D)
    }
    mul(dq_reg, dq_reg, scale);
    store(gl_dq, dq_reg, {block, head, i, 0}, laneId);
}

// ---------- dK, dV: fix kv block j, loop query i ----------
template <int D, bool CAUSAL>
kernel void attn_bwd_dkv(device   bf16     *q  [[buffer(0)]],
                         device   bf16     *k  [[buffer(1)]],
                         device   bf16     *v  [[buffer(2)]],
                         device   bf16     *dod [[buffer(3)]],
                         device   float    *L  [[buffer(4)]],
                         device   float    *delta [[buffer(5)]],
                         device   bf16     *dk [[buffer(6)]],
                         device   bf16     *dv [[buffer(7)]],
                         constant unsigned &N  [[buffer(8)]],
                         constant unsigned &H  [[buffer(9)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    using gl_v = gl<float, 1, -1, 1, -1>;
    gl_t gl_q(q, nullptr, H, N, nullptr), gl_k(k, nullptr, H, N, nullptr);
    gl_t gl_vv(v, nullptr, H, N, nullptr), gl_do(dod, nullptr, H, N, nullptr);
    gl_t gl_dk(dk, nullptr, H, N, nullptr), gl_dv(dv, nullptr, H, N, nullptr);
    gl_v gl_L(L, nullptr, H, nullptr, N), gl_del(delta, nullptr, H, nullptr, N);

    const int block = blockIdx.z, head = blockIdx.y, j = blockIdx.x;
    constexpr const float scale = bwd_scale<D>();
    constexpr const bf16 q_mul = (bf16)(scale * 1.44269504089f);

    rt_bf<TB, D, ducks::rt_layout::col> k_col, v_col;
    load(k_col, gl_k, {block, head, j, 0}, laneId);
    load(v_col, gl_vv, {block, head, j, 0}, laneId);

    rt_fl<TB, D> dk_reg, dv_reg;
    zero(dk_reg); zero(dv_reg);
    const int i_first = CAUSAL ? j : 0;
    const int q_blocks = N / TB;
    for (int i = i_first; i < q_blocks; i++) {
        rt_bf<TB, D> q_row, do_row;
        load(q_row, gl_q, {block, head, i, 0}, laneId);
        load(do_row, gl_do, {block, head, i, 0}, laneId);
        typename rt_fl<TB, TB>::col_vec L_i, del_i;
        load(L_i, gl_L, {block, head, 0, i}, laneId);
        load(del_i, gl_del, {block, head, 0, i}, laneId);

        rt_fl<TB, TB> att;
        zero(att);
        mma_ABt(att, q_row, k_col, att);           // Q_i·Kᵀ_j  (i×j)
        mul(att, att, (float)q_mul);
        sub_row(att, att, L_i);
        exp2(att, att);                            // P (i×j)
        if (CAUSAL && i == j) { float zf = 0.0f; make_causal(att, att, laneId, zf); }

        rt_bf<TB, TB> P_bf;
        copy(P_bf, att);
        rt_bf<TB, TB, ducks::rt_layout::col> P_col;
        swap_layout(P_col, P_bf, laneId);          // P as col layout (i-contraction)
        mma_AtB(dv_reg, P_col, do_row, dv_reg);    // dV_j += Pᵀ·dO_i  (j×D)

        rt_fl<TB, TB> dP;
        zero(dP);
        mma_ABt(dP, do_row, v_col, dP);            // dO_i·Vᵀ_j  (i×j)
        sub_row(dP, dP, del_i);
        rt_fl<TB, TB> dS;
        mul(dS, att, dP);                          // dS (i×j)
        rt_bf<TB, TB> dS_bf;
        copy(dS_bf, dS);
        rt_bf<TB, TB, ducks::rt_layout::col> dS_col;
        swap_layout(dS_col, dS_bf, laneId);
        mma_AtB(dk_reg, dS_col, q_row, dk_reg);    // dK_j += dSᵀ·Q_i  (j×D)
    }
    mul(dk_reg, dk_reg, scale);
    store(gl_dk, dk_reg, {block, head, j, 0}, laneId);
    store(gl_dv, dv_reg, {block, head, j, 0}, laneId);
}

#define inst_fwd_l(D, C, TAG)                                                  \
  template [[host_name("attn_fwd_l_" TAG "_" #D)]] [[kernel]] void attn_fwd_l<D, C>( \
    device bf16* q [[buffer(0)]], device bf16* k [[buffer(1)]], device bf16* v [[buffer(2)]], \
    device bf16* o [[buffer(3)]], device float* L [[buffer(4)]],               \
    constant unsigned& N [[buffer(5)]], constant unsigned& H [[buffer(6)]],    \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);
#define inst_prep(D)                                                           \
  template [[host_name("attn_bwd_prep_" #D)]] [[kernel]] void attn_bwd_prep<D>( \
    device bf16* o [[buffer(0)]], device bf16* ddo [[buffer(1)]], device float* delta [[buffer(2)]], \
    constant unsigned& N [[buffer(3)]], constant unsigned& H [[buffer(4)]],    \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);
#define inst_dq(D, C, TAG)                                                     \
  template [[host_name("attn_bwd_dq_" TAG "_" #D)]] [[kernel]] void attn_bwd_dq<D, C>( \
    device bf16* q [[buffer(0)]], device bf16* k [[buffer(1)]], device bf16* v [[buffer(2)]], \
    device bf16* ddo [[buffer(3)]], device float* L [[buffer(4)]], device float* delta [[buffer(5)]], \
    device bf16* dq [[buffer(6)]], constant unsigned& N [[buffer(7)]], constant unsigned& H [[buffer(8)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);
#define inst_dkv(D, C, TAG)                                                    \
  template [[host_name("attn_bwd_dkv_" TAG "_" #D)]] [[kernel]] void attn_bwd_dkv<D, C>( \
    device bf16* q [[buffer(0)]], device bf16* k [[buffer(1)]], device bf16* v [[buffer(2)]], \
    device bf16* dodv [[buffer(3)]], device float* L [[buffer(4)]], device float* delta [[buffer(5)]], \
    device bf16* dk [[buffer(6)]], device bf16* dv [[buffer(7)]], constant unsigned& N [[buffer(8)]], \
    constant unsigned& H [[buffer(9)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint laneId [[thread_index_in_simdgroup]]);

inst_fwd_l(64, false, "noncausal") inst_fwd_l(128, false, "noncausal")
inst_fwd_l(64, true, "causal")     inst_fwd_l(128, true, "causal")
inst_prep(64) inst_prep(128)
inst_dq(64, false, "noncausal") inst_dq(128, false, "noncausal")
inst_dq(64, true, "causal")     inst_dq(128, true, "causal")
inst_dkv(64, false, "noncausal") inst_dkv(128, false, "noncausal")
inst_dkv(64, true, "causal")     inst_dkv(128, true, "causal")

}
