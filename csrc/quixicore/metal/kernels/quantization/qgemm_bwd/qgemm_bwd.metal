#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// ---------------------------------------------------------------------------
// ACADEMIC: ternary-operand backward GEMM — grad_x (M,K) = grad_y (M,N) @ W_deq (N,K)
// with W read directly from packed bitnet blocks (dequant-to-shared, no w_deq
// materialization).
//
// train_plan §4 argues this kernel shouldn't exist: grad_y is dense, so the ternary
// operand only saves weight bytes (0.31 vs 2 B/weight), and at training M the GEMM is
// compute-bound on the MMA — the dense path loses nothing. It is built anyway to put a
// NUMBER on that argument (perf note), and because the contraction is a neat exercise:
// the backward contracts over N while bitnet packs along K, but a 32-wide K-tile is
// exactly one packed block per row, so the qgemm dequant machinery reuses cleanly with
// the WARPS SPLITTING M (they share one dequantized W tile) instead of splitting K.
//
//   grid (K/32, M/(N_WARPS*BM_PER_WARP)); each threadgroup owns a (M-tile, K-tile)
//   output block and loops nb over N/32: dequant W(nb, kx) once into shared, each
//   warp MMAs its own grad_y row-tile against it.
// ---------------------------------------------------------------------------

template<typename FMT, int N_WARPS, int BM_PER_WARP>
kernel void qgemm_bwd(
    device   half*  D  [[buffer(0)]],   // (M, K) grad_x
    device   half*  G  [[buffer(1)]],   // (M, N) grad_y
    device   uchar* Wq [[buffer(2)]],   // (N, K/block_k) packed blocks
    const constant int &M [[buffer(3)]],
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using Grp = group<N_WARPS>;
    constexpr const int BN = 32;        // contraction step over N
    constexpr const int BK = 32;        // output K-tile (= one packed block per row)

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_g(G, nullptr, nullptr, M, N);
    gl_h gl_d(D, nullptr, nullptr, M, K);

    threadgroup st<half, BN, BK> sW;    // dequantized W (n-rows x k-cols), shared by all warps
    rt<half, BM_PER_WARP, BN> g_reg;
    rt<half, BN, BK> w_reg;
    rt<float, BM_PER_WARP, BK> d_reg;
    zero(d_reg);

    const int kx = tgid.x;                                        // output col block over K
    const int row_block = tgid.y * N_WARPS + (int)warp;           // this warp's M-rows / BM

    for (int nb = 0; nb < N / BN; nb++) {
        dequant_into_shared<FMT, BN, BK>(sW, Wq, N, K, nb, kx, Grp::GROUP_THREADS, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        load(w_reg, sW, lane);                                    // (BN x BK) dequantized
        load(g_reg, gl_g, {0, 0, row_block, nb}, lane);           // (BMPW x BN) grad_y tile
        mma_AB(d_reg, g_reg, w_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    store(gl_d, d_reg, {0, 0, row_block, kx}, lane);
}

#define instantiate_qgemm_bwd(name, FMT, NW, BMPW)                            \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_bwd<FMT, NW, BMPW>(                                             \
     device half* D [[buffer(0)]], device half* G [[buffer(1)]],              \
     device uchar* Wq [[buffer(2)]],                                          \
     const constant int &M [[buffer(3)]], const constant int &N [[buffer(4)]],\
     const constant int &K [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     uint tid [[thread_index_in_threadgroup]],                                \
     uint warp [[simdgroup_index_in_threadgroup]],                            \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_bwd("qgemm_bwd_bitnet", bitnet, 2, 16);

}
