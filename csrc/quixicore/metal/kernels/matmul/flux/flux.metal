#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Fused GEMM + epilogue (the Flux kernels). Built on the register-tile GEMM
// (cf. matmul_custom): accumulate A@B in a register tile, then apply the
// epilogue in-register before storing — no extra global round-trip.
//   flux_gelu: out = gelu(A@B + bias)
//   flux_gelu_erf: out = erf_gelu(A@B + bias)
//   flux_gate: out = (A@B + bias) * gate + residual
// A is (N,K), B is (K,M), out/residual are (N,M); bias/gate are (M,) per-column.

template<typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void flux_gelu(
    device   T*   D    [[buffer(0)]],
    device   T*   A    [[buffer(1)]],
    device   T*   B    [[buffer(2)]],
    device   T*   bias [[buffer(3)]],
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    using gl_mat = gl<T, 1, 1, -1, -1>;
    using gl_vec = gl<T, 1, 1, 1, -1>;
    gl_mat gl_a(A, nullptr, nullptr, N, K);
    gl_mat gl_b(B, nullptr, nullptr, K, M);
    gl_mat gl_d(D, nullptr, nullptr, N, M);
    gl_vec gl_bias(bias, nullptr, nullptr, nullptr, M);

    constexpr const int N_BE = N_BLOCK * TILE_DIM;
    constexpr const int M_BE = M_BLOCK * TILE_DIM;
    constexpr const int K_BE = K_BLOCK * TILE_DIM;
    rt<T, N_BE, K_BE> a_reg;
    rt<T, K_BE, M_BE> b_reg;
    rt<float, N_BE, M_BE> d_reg;
    zero(d_reg);

    const int OY = threadgroup_id.y;
    const int OX = threadgroup_id.x;
    #pragma clang loop unroll(full)
    for (int k = 0; k < K / K_BE; k++) {
        load(a_reg, gl_a, {0, 0, OY, k}, simd_lane_id);
        load(b_reg, gl_b, {0, 0, k, OX}, simd_lane_id);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
    }
    // epilogue: + bias (per output column), then GELU
    typename rt<float, N_BE, M_BE>::row_vec bias_vec;
    load(bias_vec, gl_bias, {0, 0, 0, OX}, simd_lane_id);
    add_col(d_reg, d_reg, bias_vec);
    gelu(d_reg, d_reg);
    store(gl_d, d_reg, {0, 0, OY, OX}, simd_lane_id);
}

template<typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void flux_gelu_erf(
    device T *D [[buffer(0)]],
    device T *A [[buffer(1)]],
    device T *B [[buffer(2)]],
    device T *bias [[buffer(3)]],
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
    using gl_mat = gl<T, 1, 1, -1, -1>;
    using gl_vec = gl<T, 1, 1, 1, -1>;
    gl_mat gl_a(A, nullptr, nullptr, N, K);
    gl_mat gl_b(B, nullptr, nullptr, K, M);
    gl_mat gl_d(D, nullptr, nullptr, N, M);
    gl_vec gl_bias(bias, nullptr, nullptr, nullptr, M);

    constexpr const int N_BE = N_BLOCK * TILE_DIM;
    constexpr const int M_BE = M_BLOCK * TILE_DIM;
    constexpr const int K_BE = K_BLOCK * TILE_DIM;
    rt<T, N_BE, K_BE> a_reg;
    rt<T, K_BE, M_BE> b_reg;
    rt<float, N_BE, M_BE> d_reg;
    zero(d_reg);
    const int OY = threadgroup_id.y;
    const int OX = threadgroup_id.x;
    #pragma clang loop unroll(full)
    for (int k = 0; k < K / K_BE; k++) {
        load(a_reg, gl_a, {0, 0, OY, k}, simd_lane_id);
        load(b_reg, gl_b, {0, 0, k, OX}, simd_lane_id);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
    }
    typename rt<float, N_BE, M_BE>::row_vec bias_vec;
    load(bias_vec, gl_bias, {0, 0, 0, OX}, simd_lane_id);
    add_col(d_reg, d_reg, bias_vec);
    gelu_erf(d_reg, d_reg);
    store(gl_d, d_reg, {0, 0, OY, OX}, simd_lane_id);
}

template<typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
kernel void flux_gate(
    device   T*   D        [[buffer(0)]],
    device   T*   A        [[buffer(1)]],
    device   T*   B        [[buffer(2)]],
    device   T*   bias     [[buffer(3)]],
    device   T*   gate     [[buffer(4)]],
    device   T*   residual [[buffer(5)]],
    const constant int &N [[buffer(6)]],
    const constant int &K [[buffer(7)]],
    const constant int &M [[buffer(8)]],
    uint3 threadgroup_id [[threadgroup_position_in_grid]],
    uint  simd_lane_id   [[thread_index_in_simdgroup]]) {
    using gl_mat = gl<T, 1, 1, -1, -1>;
    using gl_vec = gl<T, 1, 1, 1, -1>;
    gl_mat gl_a(A, nullptr, nullptr, N, K);
    gl_mat gl_b(B, nullptr, nullptr, K, M);
    gl_mat gl_d(D, nullptr, nullptr, N, M);
    gl_mat gl_r(residual, nullptr, nullptr, N, M);
    gl_vec gl_bias(bias, nullptr, nullptr, nullptr, M);
    gl_vec gl_gate(gate, nullptr, nullptr, nullptr, M);

    constexpr const int N_BE = N_BLOCK * TILE_DIM;
    constexpr const int M_BE = M_BLOCK * TILE_DIM;
    constexpr const int K_BE = K_BLOCK * TILE_DIM;
    rt<T, N_BE, K_BE> a_reg;
    rt<T, K_BE, M_BE> b_reg;
    rt<float, N_BE, M_BE> d_reg;
    rt<float, N_BE, M_BE> r_reg;
    zero(d_reg);

    const int OY = threadgroup_id.y;
    const int OX = threadgroup_id.x;
    #pragma clang loop unroll(full)
    for (int k = 0; k < K / K_BE; k++) {
        load(a_reg, gl_a, {0, 0, OY, k}, simd_lane_id);
        load(b_reg, gl_b, {0, 0, k, OX}, simd_lane_id);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
    }
    // epilogue: (A@B + bias) * gate + residual
    typename rt<float, N_BE, M_BE>::row_vec bias_vec, gate_vec;
    load(bias_vec, gl_bias, {0, 0, 0, OX}, simd_lane_id);
    load(gate_vec, gl_gate, {0, 0, 0, OX}, simd_lane_id);
    add_col(d_reg, d_reg, bias_vec);
    mul_col(d_reg, d_reg, gate_vec);
    load(r_reg, gl_r, {0, 0, OY, OX}, simd_lane_id);
    add(d_reg, d_reg, r_reg);
    store(gl_d, d_reg, {0, 0, OY, OX}, simd_lane_id);
}

#define instantiate_flux(type_name, T)                                       \
   template [[host_name("flux_gelu_" #type_name)]] [[kernel]]                \
   void flux_gelu<T, 4, 2, 4>(                                               \
     device T* D [[buffer(0)]], device T* A [[buffer(1)]], device T* B [[buffer(2)]], \
     device T* bias [[buffer(3)]], const constant int &N [[buffer(4)]],      \
     const constant int &K [[buffer(5)]], const constant int &M [[buffer(6)]], \
     uint3 threadgroup_id [[threadgroup_position_in_grid]],                  \
     uint simd_lane_id [[thread_index_in_simdgroup]]);                       \
   template [[host_name("flux_gelu_erf_" #type_name)]] [[kernel]]           \
   void flux_gelu_erf<T, 4, 2, 4>(                                           \
     device T* D [[buffer(0)]], device T* A [[buffer(1)]], device T* B [[buffer(2)]], \
     device T* bias [[buffer(3)]], const constant int &N [[buffer(4)]],      \
     const constant int &K [[buffer(5)]], const constant int &M [[buffer(6)]], \
     uint3 threadgroup_id [[threadgroup_position_in_grid]],                  \
     uint simd_lane_id [[thread_index_in_simdgroup]]);                       \
   template [[host_name("flux_gate_" #type_name)]] [[kernel]]                \
   void flux_gate<T, 4, 2, 4>(                                               \
     device T* D [[buffer(0)]], device T* A [[buffer(1)]], device T* B [[buffer(2)]], \
     device T* bias [[buffer(3)]], device T* gate [[buffer(4)]],             \
     device T* residual [[buffer(5)]], const constant int &N [[buffer(6)]],  \
     const constant int &K [[buffer(7)]], const constant int &M [[buffer(8)]], \
     uint3 threadgroup_id [[threadgroup_position_in_grid]],                  \
     uint simd_lane_id [[thread_index_in_simdgroup]]);

instantiate_flux(float32, float);
instantiate_flux(bfloat16, bf16);

}
