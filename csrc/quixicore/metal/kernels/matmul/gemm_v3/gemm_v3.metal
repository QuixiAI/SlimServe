#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// ---------------------------------------------------------------------------
// ACADEMIC: gemm_v3 — an honest attempt to close the gap to MPS's GEMM at training
// shapes, measured and recorded either way (perf note).
//
// D = A @ B, A (N,K), B (K,M). Four warps in a 2x2 arrangement over a 64x64 output
// block; BOTH operands are staged through shared memory (gemm_staged stages only A;
// each of its warps re-reads B from global), so every staged element is reused by
// two warps. Per K-step: 4 cooperative tile loads (sA0/sA1 rows, sB0/sB1 cols),
// barrier, each warp MMAs its (row, col) tile pair. Separate tiles instead of one
// 64-wide tile because st_subtile/subtile_inplace is not implemented in this port.
// Shapes: N % 64, M % 64, K % 32 (the bench pads or picks conforming shapes).
//
// Prior art in this tree says the big-tile direction loses on Apple GPUs (see
// gemm_staged's note: BM=128 measured -20..26%; no async copies to hide staging).
// v3 differs by staging B too and keeping the tile square-ish; the measurement
// decides.
// ---------------------------------------------------------------------------

template<typename T>
kernel void gemm_v3(
    device   T*   D [[buffer(0)]],
    device   T*   A [[buffer(1)]],
    device   T*   B [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using G = group<4>;
    constexpr const int BK = 32;

    using gl_mat = gl<T, 1, 1, -1, -1>;
    gl_mat gl_a(A, nullptr, nullptr, N, K);
    gl_mat gl_b(B, nullptr, nullptr, K, M);
    gl_mat gl_d(D, nullptr, nullptr, N, M);

    threadgroup st<T, 32, BK> sA0, sA1;     // two 32-row slices of the 64-row A block
    threadgroup st<T, BK, 32> sB0, sB1;     // two 32-col slices of the 64-col B block
    rt<T, 32, BK> a_reg;
    rt<T, BK, 32> b_reg;
    rt<float, 32, 32> d_reg;
    zero(d_reg);

    const int wy = (int)warp >> 1;          // warp's row half (0/1)
    const int wx = (int)warp & 1;           // warp's col half (0/1)
    const int by = tgid.y * 2;              // A row-block base (32-row units)
    const int bx = tgid.x * 2;              // B col-block base (32-col units)

    for (int kb = 0; kb < K / BK; kb++) {
        G::load(sA0, gl_a, {0, 0, by,     kb}, tid);
        G::load(sA1, gl_a, {0, 0, by + 1, kb}, tid);
        G::load(sB0, gl_b, {0, 0, kb, bx    }, tid);
        G::load(sB1, gl_b, {0, 0, kb, bx + 1}, tid);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (wy == 0) load(a_reg, sA0, lane); else load(a_reg, sA1, lane);
        if (wx == 0) load(b_reg, sB0, lane); else load(b_reg, sB1, lane);
        mma_AB(d_reg, a_reg, b_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    store(gl_d, d_reg, {0, 0, by + wy, bx + wx}, lane);
}

#define instantiate_gemm_v3(name, T)                                          \
   template [[host_name(name)]] [[kernel]]                                    \
   void gemm_v3<T>(                                                           \
     device T* D [[buffer(0)]], device T* A [[buffer(1)]], device T* B [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     uint tid [[thread_index_in_threadgroup]],                                \
     uint warp [[simdgroup_index_in_threadgroup]],                            \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_gemm_v3("gemm_v3_float32", float);
instantiate_gemm_v3("gemm_v3_bfloat16", bf16);

}
