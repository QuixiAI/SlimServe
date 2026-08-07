#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized GEMM (Marlin's method, dequant-to-shared):  D = W @ X
//   W (N,K) is quantized (packed blocks, format FMT); X (K,M) and D (N,M) are half.
// Mirrors gemm_staged, but the weight block is DEQUANTIZED into the shared half tile each
// K-step instead of byte-copied — then the existing shared->register load + simdgroup MMA
// run unchanged (fp32 accumulate). BK = FMT::block_k (= 32). Shapes need N%32, M%BM, K%BK.
template<typename FMT, int N_WARPS, int BM_PER_WARP>
kernel void qgemm(
    device   half*  D  [[buffer(0)]],   // (N, M) output
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   half*  X  [[buffer(2)]],   // (K, M) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using G = group<N_WARPS>;
    constexpr const int BN = 32;
    constexpr const int BK = 32;             // MMA K-step (decoupled from FMT::block_k)

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M);
    gl_h gl_d(D, nullptr, nullptr, N, M);

    threadgroup st<half, BN, BK> sW;        // dequantized weight block, shared by all warps
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM_PER_WARP> x_reg;
    rt<float, BN, BM_PER_WARP> d_reg;
    zero(d_reg);

    const int by = tgid.y;                              // output row block (BN weight rows)
    const int bx = tgid.x;                              // output col block (BM cols)
    const int col_block = bx * N_WARPS + (int)warp;     // this warp's BM_PER_WARP column block

    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_shared<FMT, BN, BK>(sW, Wq, N, K, by, kb, G::GROUP_THREADS, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(w_reg, sW, lane);                           // shared -> register (dequantized W)
        load(x_reg, gl_x, {0, 0, kb, col_block}, lane);  // this warp's X columns
        mma_AB(d_reg, w_reg, x_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);  // before sW is overwritten
    }
    store(gl_d, d_reg, {0, 0, by, col_block}, lane);
}

#define instantiate_qgemm(name, FMT, NW, BMPW)                                \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm<FMT, NW, BMPW>(                                                 \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm("qgemm_q8_0", q8_0, 2, 16);
instantiate_qgemm("qgemm_q4_0", q4_0, 2, 16);
instantiate_qgemm("qgemm_q4_K", q4_K, 2, 16);
instantiate_qgemm("qgemm_kU4B8", kU4B8, 2, 16);
instantiate_qgemm("qgemm_kU4", kU4, 2, 16);
instantiate_qgemm("qgemm_fp8_e4m3", fp8_e4m3, 2, 16);
instantiate_qgemm("qgemm_fp4_e2m1", fp4_e2m1, 2, 16);
instantiate_qgemm("qgemm_mxfp8", mxfp8, 2, 32);
instantiate_qgemm("qgemm_nvfp4", nvfp4, 2, 16);
instantiate_qgemm("qgemm_mxfp4", mxfp4, 2, 16);
instantiate_qgemm("qgemm_bitnet", bitnet, 2, 16);
instantiate_qgemm("qgemm_tq2_0", tq2_0, 2, 16);
instantiate_qgemm("qgemm_iq4_nl", iq4_nl, 2, 16);
instantiate_qgemm("qgemm_iq4_xs", iq4_xs, 2, 16);
instantiate_qgemm("qgemm_iq2_xxs", iq2_xxs, 2, 16);
instantiate_qgemm("qgemm_iq2_xs", iq2_xs, 2, 16);
instantiate_qgemm("qgemm_iq3_xxs", iq3_xxs, 2, 16);
instantiate_qgemm("qgemm_iq1_s", iq1_s, 2, 16);
instantiate_qgemm("qgemm_q4_1", q4_1, 2, 16);
instantiate_qgemm("qgemm_q5_0", q5_0, 2, 16);
instantiate_qgemm("qgemm_q5_1", q5_1, 2, 16);
instantiate_qgemm("qgemm_q2_K", q2_K, 2, 16);
instantiate_qgemm("qgemm_q3_K", q3_K, 2, 16);
instantiate_qgemm("qgemm_q5_K", q5_K, 2, 16);
instantiate_qgemm("qgemm_q6_K", q6_K, 2, 16);
instantiate_qgemm("qgemm_e5m2", e5m2, 2, 16);
instantiate_qgemm("qgemm_fp8_block", fp8_block, 2, 16);
instantiate_qgemm("qgemm_mxfp6_e3m2", mxfp6_e3m2, 2, 16);
instantiate_qgemm("qgemm_mxfp6_e2m3", mxfp6_e2m3, 2, 16);
instantiate_qgemm("qgemm_hqq", hqq, 2, 16);

// ---- qgemm_frag: dequant-direct-to-fragment (Marlin zero-shuffle). Single simdgroup per
// (32x32) output tile; the weight block is dequantized straight into the register fragment
// (dequant_into_register) — no threadgroup tile, no barrier. -----
template<typename FMT>
kernel void qgemm_frag(
    device   half*  D  [[buffer(0)]],
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int BN = 32, BK = 32, BM = 32;
    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M);
    gl_h gl_d(D, nullptr, nullptr, N, M);
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM> x_reg;
    rt<float, BN, BM> d_reg;
    zero(d_reg);
    const int by = tgid.y, bx = tgid.x;
    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_register<FMT>(w_reg, Wq, N, K, by, kb, lane);  // straight to fragment
        load(x_reg, gl_x, {0, 0, kb, bx}, lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
    }
    store(gl_d, d_reg, {0, 0, by, bx}, lane);
}

#define instantiate_qgemm_frag(name, FMT)                                     \
   template [[host_name(name)]] [[kernel]] void qgemm_frag<FMT>(             \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_frag("qgemm_frag_q8_0", q8_0);
instantiate_qgemm_frag("qgemm_frag_q4_0", q4_0);
instantiate_qgemm_frag("qgemm_frag_q4_K", q4_K);
instantiate_qgemm_frag("qgemm_frag_kU4B8", kU4B8);
instantiate_qgemm_frag("qgemm_frag_kU4", kU4);
instantiate_qgemm_frag("qgemm_frag_fp8_e4m3", fp8_e4m3);
instantiate_qgemm_frag("qgemm_frag_fp4_e2m1", fp4_e2m1);
instantiate_qgemm_frag("qgemm_frag_mxfp8", mxfp8);
instantiate_qgemm_frag("qgemm_frag_nvfp4", nvfp4);
instantiate_qgemm_frag("qgemm_frag_mxfp4", mxfp4);
instantiate_qgemm_frag("qgemm_frag_bitnet", bitnet);
instantiate_qgemm_frag("qgemm_frag_tq2_0", tq2_0);
instantiate_qgemm_frag("qgemm_frag_iq4_nl", iq4_nl);
instantiate_qgemm_frag("qgemm_frag_iq4_xs", iq4_xs);
instantiate_qgemm_frag("qgemm_frag_iq2_xxs", iq2_xxs);
instantiate_qgemm_frag("qgemm_frag_iq2_xs", iq2_xs);
instantiate_qgemm_frag("qgemm_frag_iq3_xxs", iq3_xxs);
instantiate_qgemm_frag("qgemm_frag_iq1_s", iq1_s);
instantiate_qgemm_frag("qgemm_frag_q4_1", q4_1);
instantiate_qgemm_frag("qgemm_frag_q5_0", q5_0);
instantiate_qgemm_frag("qgemm_frag_q5_1", q5_1);
instantiate_qgemm_frag("qgemm_frag_q2_K", q2_K);
instantiate_qgemm_frag("qgemm_frag_q3_K", q3_K);
instantiate_qgemm_frag("qgemm_frag_q5_K", q5_K);
instantiate_qgemm_frag("qgemm_frag_q6_K", q6_K);
instantiate_qgemm_frag("qgemm_frag_e5m2", e5m2);
instantiate_qgemm_frag("qgemm_frag_fp8_block", fp8_block);
instantiate_qgemm_frag("qgemm_frag_mxfp6_e3m2", mxfp6_e3m2);
instantiate_qgemm_frag("qgemm_frag_mxfp6_e2m3", mxfp6_e2m3);
instantiate_qgemm_frag("qgemm_frag_hqq", hqq);

// ---- qgemm_actorder: GPTQ act-order with an IN-KERNEL g_idx gather. The weight is quantized in
// permuted (K) order (groups contiguous); the X K-rows are gathered by `perm` during the X load
// (fused into the fragment fill — no materialized permuted-X copy). Mirrors qgemm_frag otherwise. ----
template<typename FMT>
kernel void qgemm_actorder(
    device   half*  D    [[buffer(0)]],
    device   uchar* Wq   [[buffer(1)]],
    device   half*  X    [[buffer(2)]],
    device   int*   perm [[buffer(3)]],   // (K,) g_idx-derived permutation of the K axis
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int BN = 32, BK = 32, BM = 32;
    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_d(D, nullptr, nullptr, N, M);
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM> x_reg;
    rt<float, BN, BM> d_reg;
    zero(d_reg);
    const int by = tgid.y, bx = tgid.x;
    const int qid = (int)lane / 4;
    const int simd_y = (qid & 4) + ((int)lane / 2) % 4;
    const int simd_x = (qid & 2) * 2 + ((int)lane % 2) * 2;
    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_register<FMT>(w_reg, Wq, N, K, by, kb, lane);
        #pragma clang loop unroll(full)
        for (int i = 0; i < x_reg.height; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < x_reg.width; j++) {
                const int kr = perm[kb * BK + i * mittens::TILE_DIM + simd_y];   // gathered K row
                const int col = bx * BM + j * mittens::TILE_DIM + simd_x;
                x_reg.tiles[i][j].data.thread_elements()[0] = X[(uint)kr * M + col];
                x_reg.tiles[i][j].data.thread_elements()[1] = X[(uint)kr * M + col + 1];
            }
        }
        mma_AB(d_reg, w_reg, x_reg, d_reg);
    }
    store(gl_d, d_reg, {0, 0, by, bx}, lane);
}

#define instantiate_qgemm_actorder(name, FMT)                                 \
   template [[host_name(name)]] [[kernel]] void qgemm_actorder<FMT>(          \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     device int* perm [[buffer(3)]],                                          \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &M [[buffer(6)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_actorder("qgemm_actorder_kU4B8", kU4B8);
instantiate_qgemm_actorder("qgemm_actorder_kU4", kU4);
instantiate_qgemm_actorder("qgemm_actorder_q4_0", q4_0);

// ---- qgemm_blockscale: fp8_block with a storage-optimal SEPARATE 2D scale buffer. Codes-only
// weights (fp8_raw) are dequantized, then scaled by the (128-row x 128-col) tile scale from
// scale2d (N/128 x K/128) — no per-row scale replication. -----
template<typename FMT>
kernel void qgemm_blockscale(
    device   half*  D       [[buffer(0)]],
    device   uchar* Wq      [[buffer(1)]],
    device   half*  X       [[buffer(2)]],
    device   half*  scale2d [[buffer(3)]],   // (N/128, K/128) per-tile fp16 scale
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int BN = 32, BK = 32, BM = 32;
    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M);
    gl_h gl_d(D, nullptr, nullptr, N, M);
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM> x_reg;
    rt<float, BN, BM> d_reg;
    zero(d_reg);
    const int by = tgid.y, bx = tgid.x, KT = K / 128;          // 128-tile = 4 of the 32-wide blocks
    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_register<FMT>(w_reg, Wq, N, K, by, kb, lane);
        const half sc = scale2d[(by / 4) * KT + (kb / 4)];     // thread-space tile scale
        mul(w_reg, w_reg, sc);
        load(x_reg, gl_x, {0, 0, kb, bx}, lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
    }
    store(gl_d, d_reg, {0, 0, by, bx}, lane);
}

template [[host_name("qgemm_blockscale_fp8_raw")]] [[kernel]] void qgemm_blockscale<fp8_raw>(
    device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]],
    device half* scale2d [[buffer(3)]],
    const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

// ---- qgemm_fp8_scaled: BOTH operands fp8 e4m3 (codes only), rank-1 scaled. W is fp8 (N,K), X is fp8
// (K,M); both are decoded to half into the fragment, a standard half MMA, then the output is scaled by
// w_scale[n] (per output channel) * a_scale[m] (per token) — the SmoothQuant/fp8-scaled epilogue. The
// scale is applied in the d_reg epilogue (per-element row/col index) to avoid tile-vector orientation. ----
kernel void qgemm_fp8_scaled(
    device   half*  D       [[buffer(0)]],
    device   uchar* Wq      [[buffer(1)]],   // (N, K) fp8 e4m3 codes
    device   uchar* Xq      [[buffer(2)]],   // (K, M) fp8 e4m3 codes
    device   half*  w_scale [[buffer(3)]],   // (N,) per output channel
    device   half*  a_scale [[buffer(4)]],   // (M,) per token
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    const constant int &M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int BN = 32, BK = 32, BM = 32;
    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_d(D, nullptr, nullptr, N, M);
    rt<half, BN, BK> w_reg;
    rt<half, BK, BM> x_reg;
    rt<float, BN, BM> d_reg;
    zero(d_reg);
    const int by = tgid.y, bx = tgid.x;
    const int qid = (int)lane / 4;
    const int simd_y = (qid & 4) + ((int)lane / 2) % 4;
    const int simd_x = (qid & 2) * 2 + ((int)lane % 2) * 2;
    for (int kb = 0; kb < K / BK; kb++) {
        dequant_into_register<fp8_raw>(w_reg, Wq, N, K, by, kb, lane);   // W fp8 -> half fragment
        #pragma clang loop unroll(full)
        for (int i = 0; i < x_reg.height; i++) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < x_reg.width; j++) {
                const int kr = kb * BK + i * mittens::TILE_DIM + simd_y;
                const int col = bx * BM + j * mittens::TILE_DIM + simd_x;
                x_reg.tiles[i][j].data.thread_elements()[0] = tk_e4m3_decode(Xq[(uint)kr * M + col]);
                x_reg.tiles[i][j].data.thread_elements()[1] = tk_e4m3_decode(Xq[(uint)kr * M + col + 1]);
            }
        }
        mma_AB(d_reg, w_reg, x_reg, d_reg);
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < d_reg.height; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < d_reg.width; j++) {
            const int row = by * BN + i * mittens::TILE_DIM + simd_y;
            const int col = bx * BM + j * mittens::TILE_DIM + simd_x;
            const float ws = (float)w_scale[row];
            d_reg.tiles[i][j].data.thread_elements()[0] *= ws * (float)a_scale[col];
            d_reg.tiles[i][j].data.thread_elements()[1] *= ws * (float)a_scale[col + 1];
        }
    }
    store(gl_d, d_reg, {0, 0, by, bx}, lane);
}
// non-template kernel: Metal symbol is the namespaced "mittens::qgemm_fp8_scaled" (see launcher).

}
