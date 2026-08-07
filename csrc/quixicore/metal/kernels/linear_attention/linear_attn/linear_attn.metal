#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Non-causal linear attention (identity feature map), bf16, D=64.
//   KV = sum_j k_j^T v_j  (D x D),  out_i = q_i @ KV
// One simdgroup per (batch, head): phase 1 accumulates KV over key blocks
// (mma_AtB: K^T @ V), phase 2 computes Q @ KV per query block. Unnormalized
// (the feature-map normalization, if any, is applied outside).
template <int D>
kernel void linear_attn(device   bf16     *q [[buffer(0)]],
                        device   bf16     *k [[buffer(1)]],
                        device   bf16     *v [[buffer(2)]],
                        device   bf16     *o [[buffer(3)]],
                        constant unsigned &N [[buffer(4)]],
                        constant unsigned &H [[buffer(5)]],
                        uint3 blockIdx [[threadgroup_position_in_grid]],
                        uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64, "linear_attn currently supports D=64");
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
    // phase 1: KV = sum_j k_j^T v_j
    for (int kb = 0; kb < blocks; kb++) {
        rt_bf<8, D, ducks::rt_layout::col> k_reg;
        rt_bf<8, D> v_reg;
        load(k_reg, gk, {batch, head, kb, 0}, laneId);
        load(v_reg, gv, {batch, head, kb, 0}, laneId);
        mma_AtB(kv, k_reg, v_reg, kv);
    }
    rt_bf<D, D> kv_bf;
    copy(kv_bf, kv);   // fp32 accumulator -> bf16 for the second matmul

    // phase 2: out_i = q_i @ KV
    for (int qb = 0; qb < blocks; qb++) {
        rt_bf<8, D> q_reg;
        rt_fl<8, D> o_reg;
        load(q_reg, gq, {batch, head, qb, 0}, laneId);
        zero(o_reg);
        mma_AB(o_reg, q_reg, kv_bf, o_reg);
        store(go, o_reg, {batch, head, qb, 0}, laneId);
    }
}

#define instantiate_linear_attn(D)                                  \
  template [[host_name("linear_attn_" #D)]] [[kernel]] void         \
  linear_attn<D>(device bf16 *q [[buffer(0)]], device bf16 *k [[buffer(1)]], \
    device bf16 *v [[buffer(2)]], device bf16 *o [[buffer(3)]], \
    constant unsigned &N [[buffer(4)]], constant unsigned &H [[buffer(5)]], \
    uint3 blockIdx [[threadgroup_position_in_grid]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_linear_attn(64);

}
