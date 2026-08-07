#include <metal_stdlib>
#include "tk.metal"

namespace mittens {
#define GEMM_PARAMS_DEF(T) \
    device T* D [[buffer(0)]], \
    device T* A [[buffer(1)]], \
    device T* B [[buffer(2)]], \
    const constant int &N [[buffer(3)]], \
    const constant int &K [[buffer(4)]], \
    const constant int &M [[buffer(5)]], \
    uint3 id [[thread_position_in_grid]], \
    uint3 threadgroup_id [[threadgroup_position_in_grid]], \
    uint simdgroup_id [[simdgroup_index_in_threadgroup]], \
    uint simd_lane_id [[thread_index_in_simdgroup]], \
    uint3 num_threadgroups [[threadgroups_per_grid]]


template<typename T, unsigned N_BLOCK, unsigned K_BLOCK, unsigned M_BLOCK>
//    kernel void matmul_naive() {
kernel void matmul_naive(GEMM_PARAMS_DEF(T)) {
    // A += 512;
    using global_layout = gl<T, 1, 1, -1, -1>;
    global_layout gl_a(A, nullptr, nullptr, N, K);
    global_layout gl_b(B, nullptr, nullptr, K, M);
    global_layout gl_d(D, nullptr, nullptr, N, M);
    constexpr const int N_BLOCK_ELEMS = N_BLOCK * TILE_DIM;
    constexpr const int M_BLOCK_ELEMS = M_BLOCK * TILE_DIM;
    constexpr const int K_BLOCK_ELEMS = K_BLOCK * TILE_DIM;
    rt<T, N_BLOCK_ELEMS, K_BLOCK_ELEMS> a_reg;
    rt<T, K_BLOCK_ELEMS, M_BLOCK_ELEMS> b_reg;
    rt<float, N_BLOCK_ELEMS, M_BLOCK_ELEMS> d_reg;
    zero(d_reg);
    // zero(b_reg);
    // zero(a_reg);
    
    const int OUTPUT_Y = threadgroup_id.y;
    const int OUTPUT_X = threadgroup_id.x;
    #pragma clang loop unroll(full)
    for (int k = 0; k < K / K_BLOCK_ELEMS; k++) {
        load(a_reg, gl_a, {0, 0, OUTPUT_Y, k}, simd_lane_id);
        load(b_reg, gl_b, {0, 0, k, OUTPUT_X}, simd_lane_id);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
    }
    store(gl_d, d_reg, {0, 0, OUTPUT_Y, OUTPUT_X}, simd_lane_id);
}

#define instantiate_matmul_custom(type_name, T) \
   template [[host_name("matmul_custom_" #type_name)]] [[kernel]] \
   void matmul_naive<T, 4, 2, 4>(GEMM_PARAMS_DEF(T)); \

instantiate_matmul_custom(float32, float);
// instantiate_matmul_custom(float16, half);
instantiate_matmul_custom(bfloat16, bf16);

}