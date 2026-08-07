#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized fused GEMM+GELU (Phase 6 retrofit: a real fused kernel taking quantized weights):
//   D = gelu(dequantize(Wq) @ X + bias)
// Dequant-direct-to-fragment (Marlin zero-shuffle): the weight block is dequantized straight into
// the simdgroup register fragment (dequant_into_register) — no threadgroup tile, no barrier — then
// flux's per-column-bias + GELU epilogue. Single simdgroup per 32x32 output tile. W (N,K) quantized
// blocks (format FMT); X (K,M), bias (M,), D (N,M) all half.
template<typename FMT>
kernel void qflux_gelu(
    device   half*  D    [[buffer(0)]],
    device   uchar* Wq   [[buffer(1)]],
    device   half*  X    [[buffer(2)]],
    device   half*  bias [[buffer(3)]],   // (M,) per-output-column
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int BN = 32, BK = 32, BM = 32;

    using gl_h = gl<half, 1, 1, -1, -1>;
    using gl_vec = gl<half, 1, 1, 1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M);
    gl_h gl_d(D, nullptr, nullptr, N, M);
    gl_vec gl_bias(bias, nullptr, nullptr, nullptr, M);

    rt<half, BN, BK> w_reg;
    rt<half, BK, BM> x_reg;
    rt<float, BN, BM> d_reg;
    zero(d_reg);

    const int by = tgid.y, bx = tgid.x;

    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_register<FMT>(w_reg, Wq, N, K, by, kb, lane);   // straight to fragment
        load(x_reg, gl_x, {0, 0, kb, bx}, lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
    }
    // epilogue: + bias (per output column), then GELU
    typename rt<float, BN, BM>::row_vec bias_vec;
    load(bias_vec, gl_bias, {0, 0, 0, bx}, lane);
    add_col(d_reg, d_reg, bias_vec);
    gelu(d_reg, d_reg);
    store(gl_d, d_reg, {0, 0, by, bx}, lane);
}

#define instantiate_qflux(name, FMT)                                          \
   template [[host_name(name)]] [[kernel]]                                    \
   void qflux_gelu<FMT>(                                                      \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     device half* bias [[buffer(3)]],                                         \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &M [[buffer(6)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qflux("qflux_gelu_q8_0", q8_0);
instantiate_qflux("qflux_gelu_q4_0", q4_0);
instantiate_qflux("qflux_gelu_q4_K", q4_K);
instantiate_qflux("qflux_gelu_kU4B8", kU4B8);
instantiate_qflux("qflux_gelu_kU4", kU4);
instantiate_qflux("qflux_gelu_fp8_e4m3", fp8_e4m3);
instantiate_qflux("qflux_gelu_fp4_e2m1", fp4_e2m1);
instantiate_qflux("qflux_gelu_mxfp8", mxfp8);
instantiate_qflux("qflux_gelu_nvfp4", nvfp4);
instantiate_qflux("qflux_gelu_mxfp4", mxfp4);
instantiate_qflux("qflux_gelu_bitnet", bitnet);
instantiate_qflux("qflux_gelu_iq4_nl", iq4_nl);
instantiate_qflux("qflux_gelu_iq4_xs", iq4_xs);
instantiate_qflux("qflux_gelu_iq2_xxs", iq2_xxs);
instantiate_qflux("qflux_gelu_iq2_xs", iq2_xs);
instantiate_qflux("qflux_gelu_iq3_xxs", iq3_xxs);
instantiate_qflux("qflux_gelu_iq1_s", iq1_s);
instantiate_qflux("qflux_gelu_q4_1", q4_1);
instantiate_qflux("qflux_gelu_q5_0", q5_0);
instantiate_qflux("qflux_gelu_q5_1", q5_1);
instantiate_qflux("qflux_gelu_q2_K", q2_K);
instantiate_qflux("qflux_gelu_q3_K", q3_K);
instantiate_qflux("qflux_gelu_q5_K", q5_K);
instantiate_qflux("qflux_gelu_q6_K", q6_K);
instantiate_qflux("qflux_gelu_e5m2", e5m2);
instantiate_qflux("qflux_gelu_fp8_block", fp8_block);
instantiate_qflux("qflux_gelu_mxfp6_e3m2", mxfp6_e3m2);
instantiate_qflux("qflux_gelu_mxfp6_e2m3", mxfp6_e2m3);
instantiate_qflux("qflux_gelu_hqq", hqq);

}
