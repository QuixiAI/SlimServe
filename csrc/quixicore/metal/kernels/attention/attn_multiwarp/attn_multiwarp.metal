#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Multi-warp flash-attention forward (non-causal), bf16, D in {64,128}.
//
// A threadgroup has N_WARPS simdgroups and processes N_WARPS query tiles (8 rows
// each). Each K/V block is loaded ONCE into threadgroup memory (cooperatively by
// all warps) and reused by every warp's query tile — amortizing the K/V global
// reads by N_WARPS. Each warp runs an independent online softmax for its query tile.
constant constexpr const int TNm = 8;
constant constexpr const int NUM_WARPS = 4;

template <int D>
kernel void attn_multiwarp(device   bf16     *q [[buffer(0)]],
                           device   bf16     *k [[buffer(1)]],
                           device   bf16     *v [[buffer(2)]],
                           device   bf16     *o [[buffer(3)]],
                           constant unsigned &N [[buffer(4)]],
                           constant unsigned &H [[buffer(5)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  tid  [[thread_index_in_threadgroup]],
                           uint  warp [[simdgroup_index_in_threadgroup]],
                           uint  laneId [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using G = group<NUM_WARPS>;
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q, nullptr, H, N, nullptr);
    global_layout gl_k(k, nullptr, H, N, nullptr);
    global_layout gl_v(v, nullptr, H, N, nullptr);
    global_layout gl_o(o, nullptr, H, N, nullptr);
    using st_kv      = st_bf<TNm, D>;
    using rt_qkv     = rt_bf<TNm, D>;
    using rt_k_t     = rt_bf<TNm, D, ducks::rt_layout::col>;
    using rt_att     = rt_fl<TNm, TNm>;
    using rt_o       = rt_fl<TNm, D>;
    using rv_att     = rt_fl<TNm, TNm>::col_vec;

    const int block = blockIdx.z;
    const int head = blockIdx.y;
    const int q_tile = blockIdx.x * NUM_WARPS + (int)warp;   // this warp's query tile

    threadgroup st_kv sK, sV;                                 // shared K/V block (all warps)
    const int kv_blocks = N / st_kv::rows;

    rt_qkv q_reg;
    rt_k_t k_reg;
    rt_qkv v_reg;
    rt_att att_block;
    rt_o o_reg;
    rv_att max_vec_last, max_vec, norm_vec;

    load(q_reg, gl_q, {block, head, q_tile, 0}, laneId);
    neg_infty(max_vec);
    zero(norm_vec);
    zero(o_reg);
    constexpr const bf16 q_mul = ((D == 128) ? 0.08838834764bf : 0.125bf) * 1.44269504089bf;
    mul(q_reg, q_reg, q_mul);

    for (int kv_idx = 0; kv_idx < kv_blocks; kv_idx++) {
        // cooperatively stage this K/V block into shared memory (shared by all warps)
        G::load(sK, gl_k, {block, head, kv_idx, 0}, tid);
        G::load(sV, gl_v, {block, head, kv_idx, 0}, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        load(k_reg, sK, laneId);            // shared -> register (col layout for K^T)
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        copy(max_vec_last, max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec);
        exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec);
        exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, sV, laneId);            // shared -> register (row layout for V)
        mma_AB(o_reg, att_block, v_reg, o_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);  // before sK/sV overwrite
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {block, head, q_tile, 0}, laneId);
}

#define instantiate_attn_multiwarp(D)                            \
  template [[host_name("attn_multiwarp_" #D)]] [[kernel]] void   \
  attn_multiwarp<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device bf16 *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint tid [[thread_index_in_threadgroup]], \
    uint warp [[simdgroup_index_in_threadgroup]], \
    uint laneId [[thread_index_in_simdgroup]]); \

instantiate_attn_multiwarp(64);
instantiate_attn_multiwarp(128);

}
