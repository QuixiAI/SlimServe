#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Hedgehog-style linear attention (non-causal), bf16, D=64.
// Applies a softmax-style feature map phi(x) = exp(x - rowmax(x)) to Q and K,
// then linear attention out = phi(Q) @ (phi(K)^T @ V). One simdgroup per
// (batch, head); same structure as linear_attn with the feature map added.
template <int D>
kernel void hedgehog(device   bf16     *q [[buffer(0)]],
                     device   bf16     *k [[buffer(1)]],
                     device   bf16     *v [[buffer(2)]],
                     device   bf16     *o [[buffer(3)]],
                     constant unsigned &N [[buffer(4)]],
                     constant unsigned &H [[buffer(5)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "hedgehog currently supports D=64");
    using gl_t = gl<bfloat, 1, -1, -1, D>;
    gl_t gq(q, nullptr, H, N, nullptr);
    gl_t gk(k, nullptr, H, N, nullptr);
    gl_t gv(v, nullptr, H, N, nullptr);
    gl_t go(o, nullptr, H, N, nullptr);

    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int blocks = N / 8;

    rt_fl<D, D> kv;
    zero(kv);
    // phase 1: KV = sum_j phi(k_j)^T v_j
    for (int kb = 0; kb < blocks; kb++) {
        rt_bf<8, D, ducks::rt_layout::col> k_reg;
        rt_bf<8, D> v_reg;
        load(k_reg, gk, {batch, head, kb, 0}, laneId);
        load(v_reg, gv, {batch, head, kb, 0}, laneId);
        // feature map phi(k) = exp(k - rowmax(k)) over the feature dim
        typename rt_bf<8, D, ducks::rt_layout::col>::col_vec kmax;
        neg_infty(kmax);
        row_max(kmax, k_reg, kmax, laneId);
        sub_row(k_reg, k_reg, kmax);
        exp(k_reg, k_reg);
        mma_AtB(kv, k_reg, v_reg, kv);
    }
    rt_bf<D, D> kv_bf;
    copy(kv_bf, kv);

    // phase 2: out_i = phi(q_i) @ KV
    for (int qb = 0; qb < blocks; qb++) {
        rt_bf<8, D> q_reg;
        load(q_reg, gq, {batch, head, qb, 0}, laneId);
        typename rt_bf<8, D>::col_vec qmax;
        neg_infty(qmax);
        row_max(qmax, q_reg, qmax, laneId);
        sub_row(q_reg, q_reg, qmax);
        exp(q_reg, q_reg);
        rt_fl<8, D> o_reg;
        zero(o_reg);
        mma_AB(o_reg, q_reg, kv_bf, o_reg);
        store(go, o_reg, {batch, head, qb, 0}, laneId);
    }
}

#define instantiate_hedgehog(D)                                  \
  template [[host_name("hedgehog_" #D)]] [[kernel]] void         \
  hedgehog<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device bf16 *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_hedgehog(64);

}
