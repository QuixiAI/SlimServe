#include <metal_stdlib>
#if defined(__HAVE_TENSOR__)
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif
#include "tk.metal"

namespace mittens {

// Bench-only pseudo-format: unquantized half weights behind the FMT
// contract. Splits "dequant-ALU-bound" from "structure-bound": if this runs
// at its own (4x larger) bandwidth floor, the tile flow is healthy and the
// quant kernels are decode-limited; if it is equally far off, the structure
// (barriers, staging, occupancy) is the wall.
struct f16_raw {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 64;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        return ((device const half*)base)[col];
    }
};

// Small-M quantized GEMM (speculative verify / draft blocks / small decode
// batches):  D = dequant(W) @ X,  W (N,K) packed blocks, X (K,M) half, D (N,M)
// half, fp32 accumulate. M is padded to M_PAD=32 by the host.
//
// Differs from qgemm (which splits the M columns across warps) where small M
// leaves warps idle on padding: here EVERY warp computes all M_PAD columns for
// its own RPW weight rows, so all streamed weight bytes feed useful MMA work.
// W is dequantized into a warp-private shared slice guarded only by
// simdgroup_barrier (no cross-warp traffic). Both X tiles and W slices are
// double-buffered and staged one K-step ahead of the MMA that consumes them
// (software pipeline), so each step costs a single threadgroup_barrier and
// the next step's device loads are in flight behind the current step's math.
// Small RPW keeps the launch wide: N/RPW warps, vs the ~10% occupancy a
// 16-row warp leaves on a 40-core M-series part.
//
// Grid: (1, N / (N_WARPS*RPW), 1), N_WARPS*32 threads.
// Shapes: N % (N_WARPS*RPW) == 0, K % 32 == 0, M == 32 (padded).
// SPLIT_K > 1 raises threadgroup count on deep-K shapes (grid.z slices the
// K-walk); each slice writes float partials to D (then shaped (SPLIT_K, N,
// M_PAD)) and qgemm_sm_reduce folds them. Deterministic: no atomics.
template<typename FMT, int N_WARPS, int RPW, int SPLIT_K, int BK = 32>
kernel void qgemm_sm(
    device   uchar* Dv [[buffer(0)]],   // (N, M_PAD) half, or split-K float partials
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   half*  X  [[buffer(2)]],   // (K, M_PAD) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    using G = group<N_WARPS>;
    constexpr const int M_PAD = 32;

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M_PAD);

    threadgroup st<half, BK, M_PAD> sX[2];          // double-buffered X K-tile
    threadgroup st<half, RPW, BK> sW[2][N_WARPS];   // warp-private dequant slices

    rt<half, RPW, BK> w_reg;
    rt<half, BK, M_PAD> x_reg;
    rt<float, RPW, M_PAD> d_reg;
    zero(d_reg);

    // this warp's global row block, in RPW units
    const int rb = tgid.y * N_WARPS + (int)warp;
    // this threadgroup's K-slice, in BK steps
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    // prologue: stage the slice's first step
    G::load(sX[kb_beg & 1], gl_x, {0, 0, kb_beg, 0}, tid);
    dequant_into_shared<FMT, RPW, BK>(sW[kb_beg & 1][warp], Wq, N, K, rb,
                                      kb_beg, 32, lane);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        const int nxt = cur ^ 1;
        // stage kb+1 first so its device loads overlap this step's math
        if (kb + 1 < kb_end) {
            G::load(sX[nxt], gl_x, {0, 0, kb + 1, 0}, tid);
            dequant_into_shared<FMT, RPW, BK>(sW[nxt][warp], Wq, N, K, rb,
                                              kb + 1, 32, lane);
        }
        load(w_reg, sW[cur][warp], lane);
        load(x_reg, sX[cur], lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
        // one barrier closes the step: staging of kb+1 is complete everywhere,
        // and every thread is done reading the `cur` tiles that kb+2 reuses
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (SPLIT_K == 1) {
        gl_h gl_d((device half*)Dv, nullptr, nullptr, N, M_PAD);
        store(gl_d, d_reg, {0, 0, rb, 0}, lane);
    } else {
        using gl_f = gl<float, 1, 1, -1, -1>;
        gl_f gl_p((device float*)Dv + (size_t)tgid.z * N * M_PAD, nullptr,
                  nullptr, N, M_PAD);
        store(gl_p, d_reg, {0, 0, rb, 0}, lane);
    }
}

// ---- paired-plane dequant (Marlin lesson, Metal form) ----------------------
// In q4_K/q5_K a quant byte's two nibbles feed columns `pos` and `pos+32`,
// which both land inside one BK=64 tile (one whole 64-col chunk of the
// 256-weight superblock). Each lane owns one 8-byte run of one row: two
// vector plane-splits, hardware uchar4->half4 converts, one half4 fma per 4
// weights, and four half4 stores at two row-major addresses -- ~2.4 ops per
// weight vs ~12 for the per-element span path. Requires RPW=8, BK=64,
// 32 lanes; kb64 is the K-step index in 64-column units.
template<typename FMT>
METAL_FUNC void dequant_into_shared_paired(threadgroup st<half, 8, 64>& dst,
                                           device const uchar* Wq, int N,
                                           int K, int rb, int kb64, uint lane);

template<>
METAL_FUNC void dequant_into_shared_paired<q4_K>(
        threadgroup st<half, 8, 64>& dst, device const uchar* Wq, int N,
        int K, int rb, int kb64, uint lane) {
    const int row = (int)lane >> 2;       // 8 rows x 4 byte-runs
    const int j0  = ((int)lane & 3) * 8;  // qs byte offset within the chunk
    const int grow = rb * 8 + row;
    const int gk = kb64 * 64;
    const int blk = gk >> 8;
    const int chunk = (gk & 255) >> 6;
    device const uchar* base =
        Wq + (size_t)(grow * (K >> 8) + blk) * q4_K::block_bytes;
    const half d    = ((device const half*)base)[0];
    const half dmin = ((device const half*)base)[1];
    device const uchar* scales = base + 4;
    // both sub-block scale pairs of this chunk (6-bit unpack, once per lane)
    const int sub0 = chunk * 2;
    uchar sc0, m0, sc1, m1;
    if (sub0 < 4) {
        sc0 = scales[sub0] & 63;     m0 = scales[sub0 + 4] & 63;
        sc1 = scales[sub0 + 1] & 63; m1 = scales[sub0 + 5] & 63;
    } else {
        sc0 = (scales[sub0 + 4] & 0x0F) | ((scales[sub0 - 4] >> 6) << 4);
        m0  = (scales[sub0 + 4] >> 4)   | ((scales[sub0]     >> 6) << 4);
        sc1 = (scales[sub0 + 5] & 0x0F) | ((scales[sub0 - 3] >> 6) << 4);
        m1  = (scales[sub0 + 5] >> 4)   | ((scales[sub0 + 1] >> 6) << 4);
    }
    const half dl0 = d * half(sc0), ml0 = dmin * half(m0);
    const half dl1 = d * half(sc1), ml1 = dmin * half(m1);
    const packed_uint2 qw =
        *(device const packed_uint2*)(base + 16 + chunk * 32 + j0);
    const uint lox = qw.x & 0x0F0F0F0Fu, loy = qw.y & 0x0F0F0F0Fu;
    const uint hix = (qw.x >> 4) & 0x0F0F0F0Fu, hiy = (qw.y >> 4) & 0x0F0F0F0Fu;
    threadgroup half4* plo = (threadgroup half4*)&dst[int2(row, j0)];
    threadgroup half4* phi = (threadgroup half4*)&dst[int2(row, 32 + j0)];
    plo[0] = half4(as_type<uchar4>(lox)) * dl0 - ml0;
    plo[1] = half4(as_type<uchar4>(loy)) * dl0 - ml0;
    phi[0] = half4(as_type<uchar4>(hix)) * dl1 - ml1;
    phi[1] = half4(as_type<uchar4>(hiy)) * dl1 - ml1;
}

template<>
METAL_FUNC void dequant_into_shared_paired<q5_K>(
        threadgroup st<half, 8, 64>& dst, device const uchar* Wq, int N,
        int K, int rb, int kb64, uint lane) {
    const int row = (int)lane >> 2;
    const int j0  = ((int)lane & 3) * 8;
    const int grow = rb * 8 + row;
    const int gk = kb64 * 64;
    const int blk = gk >> 8;
    const int chunk = (gk & 255) >> 6;
    device const uchar* base =
        Wq + (size_t)(grow * (K >> 8) + blk) * q5_K::block_bytes;
    const half d    = ((device const half*)base)[0];
    const half dmin = ((device const half*)(base + 2))[0];
    device const uchar* sca = base + 4;
    const int is0 = chunk * 2;
    int sc0, mn0, sc1, mn1;
    if (is0 < 4) {
        sc0 = sca[is0] & 63;     mn0 = sca[is0 + 4] & 63;
        sc1 = sca[is0 + 1] & 63; mn1 = sca[is0 + 5] & 63;
    } else {
        sc0 = (sca[is0 + 4] & 0x0F) | ((sca[is0 - 4] >> 6) << 4);
        mn0 = (sca[is0 + 4] >> 4)   | ((sca[is0]     >> 6) << 4);
        sc1 = (sca[is0 + 5] & 0x0F) | ((sca[is0 - 3] >> 6) << 4);
        mn1 = (sca[is0 + 5] >> 4)   | ((sca[is0 + 1] >> 6) << 4);
    }
    const half dl0 = d * half(sc0), ml0 = dmin * half(mn0);
    const half dl1 = d * half(sc1), ml1 = dmin * half(mn1);
    const packed_uint2 qw =
        *(device const packed_uint2*)(base + 48 + chunk * 32 + j0);
    const packed_uint2 hw = *(device const packed_uint2*)(base + 16 + j0);
    // 5th bit: qh bit is0 (lo plane) / is0+1 (hi plane), +16 on the nibble
    const uint vlx = (qw.x & 0x0F0F0F0Fu) | (((hw.x >> is0) & 0x01010101u) << 4);
    const uint vly = (qw.y & 0x0F0F0F0Fu) | (((hw.y >> is0) & 0x01010101u) << 4);
    const uint vhx =
        ((qw.x >> 4) & 0x0F0F0F0Fu) | (((hw.x >> (is0 + 1)) & 0x01010101u) << 4);
    const uint vhy =
        ((qw.y >> 4) & 0x0F0F0F0Fu) | (((hw.y >> (is0 + 1)) & 0x01010101u) << 4);
    threadgroup half4* plo = (threadgroup half4*)&dst[int2(row, j0)];
    threadgroup half4* phi = (threadgroup half4*)&dst[int2(row, 32 + j0)];
    plo[0] = half4(as_type<uchar4>(vlx)) * dl0 - ml0;
    plo[1] = half4(as_type<uchar4>(vly)) * dl0 - ml0;
    phi[0] = half4(as_type<uchar4>(vhx)) * dl1 - ml1;
    phi[1] = half4(as_type<uchar4>(vhy)) * dl1 - ml1;
}

// qgemm_sm body at BK=64 with the paired-plane dequant (q4_K/q5_K only).
// Same pipeline, split-K partials and reduce as qgemm_sm. N_WARPS scales how
// many 8-row warps share one staged X tile: X re-staging traffic is
// 2*M_PAD*N*K/(8*N_WARPS) bytes total -- ~7x the weight bytes at 2 warps on
// the big muse shapes -- so wider threadgroups directly cut the actual wall.
template<typename FMT, int N_WARPS>
kernel void qgemm_sm_p(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],   // (K, 32) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int RPW     = 8;
    constexpr const int SPLIT_K = 4;
    constexpr const int BK      = 64;
    constexpr const int M_PAD   = 32;
    using G = group<N_WARPS>;

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M_PAD);

    threadgroup st<half, BK, M_PAD> sX[2];
    threadgroup st<half, RPW, BK> sW[2][N_WARPS];

    rt<half, RPW, BK> w_reg;
    rt<half, BK, M_PAD> x_reg;
    rt<float, RPW, M_PAD> d_reg;
    zero(d_reg);

    const int rb = tgid.y * N_WARPS + (int)warp;
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    G::load(sX[kb_beg & 1], gl_x, {0, 0, kb_beg, 0}, tid);
    dequant_into_shared_paired<FMT>(sW[kb_beg & 1][warp], Wq, N, K, rb,
                                    kb_beg, lane);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        const int nxt = cur ^ 1;
        if (kb + 1 < kb_end) {
            G::load(sX[nxt], gl_x, {0, 0, kb + 1, 0}, tid);
            dequant_into_shared_paired<FMT>(sW[nxt][warp], Wq, N, K, rb,
                                            kb + 1, lane);
        }
        load(w_reg, sW[cur][warp], lane);
        load(x_reg, sX[cur], lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    using gl_f = gl<float, 1, 1, -1, -1>;
    gl_f gl_p(P + (size_t)tgid.z * N * M_PAD, nullptr, nullptr, N, M_PAD);
    store(gl_p, d_reg, {0, 0, rb, 0}, lane);
}

#define instantiate_qgemm_sm_p(name, FMT, NW)                                 \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm_p<FMT, NW>(                                                  \
     device float* P [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm_p("qgemm_sm_p_q4_K", q4_K, 2);
instantiate_qgemm_sm_p("qgemm_sm_p_q5_K", q5_K, 2);
instantiate_qgemm_sm_p("qgemm_sm_p4_q4_K", q4_K, 4);
instantiate_qgemm_sm_p("qgemm_sm_p4_q5_K", q5_K, 4);
instantiate_qgemm_sm_p("qgemm_sm_p8_q4_K", q4_K, 8);
instantiate_qgemm_sm_p("qgemm_sm_p8_q5_K", q5_K, 8);

#if defined(__HAVE_TENSOR__)
// ---- tensor-ops verify GEMM (M5 GPU neural accelerators) -------------------
// Same staging pipeline as qgemm_sm_p (paired-plane dequant for q4_K/q5_K,
// span dequant for q6_K) with the per-warp simdgroup MMA replaced by one
// cooperative 32x64 @ 64xM_PAD matmul2d per K-step. Measured on the down
// shape: chained simdgroup MMA 6.4 TFLOPS vs tensor ops 55 TFLOPS at this
// exact tile shape (perf/results/2026-08-14/qgemm-sm-profile/), which moves
// MMA off the critical path -- the kernel becomes staging-bound. The four
// 8x64 warp slices are contiguous in threadgroup memory, so the combined
// 32x64 A tile needs no staging changes. Split-K float partials and the
// deterministic reduce are unchanged (greedy verify stays bit-reproducible).

template<typename FMT>
METAL_FUNC void dequant_paired_raw(threadgroup st<half, 8, 64>& dst,
                                   device const uchar* Wq, int N, int K,
                                   int rb, int kb64, uint lane);

template<>
METAL_FUNC void dequant_paired_raw<q4_K>(
        threadgroup st<half, 8, 64>& dst, device const uchar* Wq, int N,
        int K, int rb, int kb64, uint lane) {
    const int row = (int)lane >> 2;
    const int j0  = ((int)lane & 3) * 8;
    const int grow = rb * 8 + row;
    const int gk = kb64 * 64;
    const int blk = gk >> 8;
    const int chunk = (gk & 255) >> 6;
    device const uchar* base =
        Wq + (size_t)(grow * (K >> 8) + blk) * q4_K::block_bytes;
    const half d    = ((device const half*)base)[0];
    const half dmin = ((device const half*)base)[1];
    const uchar s0  = (base + 4)[chunk * 2];
    const uchar s1  = (base + 4)[chunk * 2 + 4];
    const packed_uint2 qw =
        *(device const packed_uint2*)(base + 16 + chunk * 32 + j0);
    threadgroup half4* plo = (threadgroup half4*)&dst[int2(row, j0)];
    threadgroup half4* phi = (threadgroup half4*)&dst[int2(row, 32 + j0)];
    plo[0] = half4(as_type<uchar4>(qw.x)) * d;
    plo[1] = half4(as_type<uchar4>(qw.y)) * dmin;
    phi[0] = half4(as_type<uchar4>(qw.x)) + half(s0);
    phi[1] = half4(as_type<uchar4>(qw.y)) + half(s1);
}

template<>
METAL_FUNC void dequant_paired_raw<q5_K>(
        threadgroup st<half, 8, 64>& dst, device const uchar* Wq, int N,
        int K, int rb, int kb64, uint lane) {
    const int row = (int)lane >> 2;
    const int j0  = ((int)lane & 3) * 8;
    const int grow = rb * 8 + row;
    const int gk = kb64 * 64;
    const int blk = gk >> 8;
    const int chunk = (gk & 255) >> 6;
    device const uchar* base =
        Wq + (size_t)(grow * (K >> 8) + blk) * q5_K::block_bytes;
    const half d    = ((device const half*)base)[0];
    const half dmin = ((device const half*)(base + 2))[0];
    const uchar s0  = (base + 4)[chunk * 2];
    const packed_uint2 qw =
        *(device const packed_uint2*)(base + 48 + chunk * 32 + j0);
    const packed_uint2 hw = *(device const packed_uint2*)(base + 16 + j0);
    threadgroup half4* plo = (threadgroup half4*)&dst[int2(row, j0)];
    threadgroup half4* phi = (threadgroup half4*)&dst[int2(row, 32 + j0)];
    plo[0] = half4(as_type<uchar4>(qw.x)) * d;
    plo[1] = half4(as_type<uchar4>(qw.y)) * dmin;
    phi[0] = half4(as_type<uchar4>(hw.x)) + half(s0);
    phi[1] = half4(as_type<uchar4>(hw.y));
}

template<typename FMT, bool PAIRED>
struct sm_t_stager {
    static METAL_FUNC void stage(threadgroup st<half, 8, 64>& dst,
                                 device uchar* Wq, int N, int K, int rb,
                                 int kb, uint lane) {
        dequant_into_shared<FMT, 8, 64>(dst, Wq, N, K, rb, kb, 32, lane);
    }
};
template<typename FMT>
struct sm_t_stager<FMT, true> {
    static METAL_FUNC void stage(threadgroup st<half, 8, 64>& dst,
                                 device uchar* Wq, int N, int K, int rb,
                                 int kb, uint lane) {
        dequant_into_shared_paired<FMT>(dst, Wq, N, K, rb, kb, lane);
    }
};

template<typename FMT, bool PAIRED>
kernel void qgemm_sm_t(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],   // (K, 32) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int N_WARPS = 4;
    constexpr const int RPW     = 8;
    constexpr const int SPLIT_K = 4;
    constexpr const int BK      = 64;
    constexpr const int M_PAD   = 32;
    constexpr const int ROWS    = N_WARPS * RPW;
    using G = group<N_WARPS>;

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M_PAD);

    threadgroup st<half, BK, M_PAD> sX[2];
    threadgroup st<half, RPW, BK> sW[2][N_WARPS];

    const int rb = tgid.y * N_WARPS + (int)warp;
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    constexpr auto t_desc = mpp::tensor_ops::matmul2d_descriptor(
        ROWS, M_PAD, BK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<t_desc, metal::execution_simdgroups<N_WARPS>> op;

    using tg_a = metal::tensor<threadgroup half,
                               metal::extents<int32_t, BK, ROWS>,
                               metal::tensor_inline>;
    using tg_b = metal::tensor<threadgroup half,
                               metal::extents<int32_t, M_PAD, BK>,
                               metal::tensor_inline>;
    tg_a tA0((threadgroup half*)&sW[0][0], metal::extents<int32_t, BK, ROWS>());
    tg_a tA1((threadgroup half*)&sW[1][0], metal::extents<int32_t, BK, ROWS>());
    tg_b tB0((threadgroup half*)&sX[0], metal::extents<int32_t, M_PAD, BK>());
    tg_b tB1((threadgroup half*)&sX[1], metal::extents<int32_t, M_PAD, BK>());

    auto cT = op.template get_destination_cooperative_tensor<tg_a, tg_b,
                                                             float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    G::load(sX[kb_beg & 1], gl_x, {0, 0, kb_beg, 0}, tid);
    sm_t_stager<FMT, PAIRED>::stage(sW[kb_beg & 1][warp], Wq, N, K, rb,
                                    kb_beg, lane);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        if (kb + 1 < kb_end) {
            G::load(sX[cur ^ 1], gl_x, {0, 0, kb + 1, 0}, tid);
            sm_t_stager<FMT, PAIRED>::stage(sW[cur ^ 1][warp], Wq, N, K, rb,
                                            kb + 1, lane);
        }
        if (cur == 0) {
            op.run(tA0, tB0, cT);
        } else {
            op.run(tA1, tB1, cT);
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    using dev_c = metal::tensor<device float,
                                metal::extents<int32_t, M_PAD, ROWS>,
                                metal::tensor_inline>;
    dev_c tC(P + (size_t)tgid.z * N * M_PAD + (size_t)tgid.y * ROWS * M_PAD,
             metal::extents<int32_t, M_PAD, ROWS>());
    cT.store(tC);
}

// Device-X flavor: X is small enough to stay L2-resident (<= ~426 KB on the
// muse shapes), so skip staging it to threadgroup memory and hand matmul2d a
// device tensor slice per K-step -- the op streams its own operands. Halves
// threadgroup memory (8 KB -> better occupancy) and drops the X-stage work
// and half the barrier pressure.
template<typename FMT, bool PAIRED, int N_WARPS = 4, int ABLATE = 0,
         int SPLIT_K = 4>
kernel void qgemm_sm_t2(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],   // (K, 32) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int RPW     = 8;
    constexpr const int BK      = 64;
    constexpr const int M_PAD   = 32;
    constexpr const int ROWS    = N_WARPS * RPW;

    threadgroup st<half, RPW, BK> sW[2][N_WARPS];

    const int rb = tgid.y * N_WARPS + (int)warp;
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    constexpr auto t_desc = mpp::tensor_ops::matmul2d_descriptor(
        ROWS, M_PAD, BK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<t_desc, metal::execution_simdgroups<N_WARPS>> op;

    using tg_a = metal::tensor<threadgroup half,
                               metal::extents<int32_t, BK, ROWS>,
                               metal::tensor_inline>;
    using dev_b = metal::tensor<device half,
                                metal::extents<int32_t, M_PAD, BK>,
                                metal::tensor_inline>;
    tg_a tA0((threadgroup half*)&sW[0][0], metal::extents<int32_t, BK, ROWS>());
    tg_a tA1((threadgroup half*)&sW[1][0], metal::extents<int32_t, BK, ROWS>());

    auto cT = op.template get_destination_cooperative_tensor<tg_a, dev_b,
                                                             float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    sm_t_stager<FMT, PAIRED>::stage(sW[kb_beg & 1][warp], Wq, N, K, rb,
                                    kb_beg, lane);
    if (ABLATE == 1) {
        sm_t_stager<FMT, PAIRED>::stage(
            sW[(kb_beg & 1) ^ 1][warp], Wq, N, K, rb,
            metal::min(kb_beg + 1, steps_total - 1), lane);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        if (kb + 1 < kb_end && ABLATE != 1) {
            if (ABLATE == 3) {
                dequant_paired_raw<FMT>(sW[cur ^ 1][warp], Wq, N, K, rb,
                                        kb + 1, lane);
            } else {
                sm_t_stager<FMT, PAIRED>::stage(sW[cur ^ 1][warp], Wq, N, K,
                                                rb, kb + 1, lane);
            }
        }
        dev_b tB(X + (size_t)kb * BK * M_PAD,
                 metal::extents<int32_t, M_PAD, BK>());
        if (ABLATE != 2) {
            if (cur == 0) {
                op.run(tA0, tB, cT);
            } else {
                op.run(tA1, tB, cT);
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (ABLATE == 2) {
        // tail run keeps the staged tiles live at 1/steps of the cost
        dev_b tBt(X, metal::extents<int32_t, M_PAD, BK>());
        if ((kb_end & 1) == 0) {
            op.run(tA0, tBt, cT);
        } else {
            op.run(tA1, tBt, cT);
        }
    }

    // NOTE: manual element readout of the destination cooperative tensor is
    // layout-unsafe for execution_simdgroups<8> (half the slots are invalid
    // and some valid slots hold cross-simdgroup PARTIAL sums that only
    // combine inside cT.store -- measured via the coord_dump probe). Always
    // store through cT.store; SPLIT_K == 1 writes one float slice (z = 0)
    // and the host folds it with the SK=1 reduce (a float->half cast).
    using dev_c = metal::tensor<device float,
                                metal::extents<int32_t, M_PAD, ROWS>,
                                metal::tensor_inline>;
    dev_c tC(P + (size_t)tgid.z * N * M_PAD + (size_t)tgid.y * ROWS * M_PAD,
             metal::extents<int32_t, M_PAD, ROWS>());
    cT.store(tC);
}

// ---- uint4-native q4_K GEMM (probe-validated 1.19x floor) ------------------
// Weights repacked at load time to plain uint4, tile-major
// [n_tile=64][K][64] (packed formats forbid strided tensors), with
// per-32-group scale/min half planes SC/MN (K/32, N). matmul2d streams the
// packed B operand itself; A is row-major X^T (M_PAD=32, K) -- the serving
// layout, no transpose. Per group g: multiply into a tmp cooperative
// tensor, then cMain += sc[g,n]*tmp - mn[g,n]*xs[g,m], where xs is the
// per-group X column sum (rank-1 min-term; qgemm_sm_u4_xsum below).
// Deterministic split-K float partials, folded by qgemm_sm_reduce.
kernel void qgemm_sm_u4_xsum(
    device   half* XS [[buffer(0)]],   // (K/32, 32)
    device const half* X [[buffer(1)]],   // (32, K) row-major X^T
    const constant int &K [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
    const int g = (int)gid >> 5;
    const int m = (int)gid & 31;
    if (g >= K / 32) return;
    float acc = 0.0f;
    for (int i = 0; i < 32; i++) acc += (float)X[(size_t)m * K + g * 32 + i];
    XS[(size_t)g * 32 + m] = half(acc);
}

kernel void qgemm_sm_u4(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device const uchar* Wu [[buffer(1)]],  // tile-major packed uint4
    device const half*  X  [[buffer(2)]],  // (32, K) row-major X^T
    device const half*  SC [[buffer(3)]],  // (K/32, N)
    device const half*  MN [[buffer(4)]],  // (K/32, N)
    device const half*  XS [[buffer(5)]],  // (K/32, 32)
    const constant int &N [[buffer(6)]],
    const constant int &K [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
    constexpr const int NT      = 64;
    constexpr const int SPLIT_K = 4;
    constexpr const int M_PAD   = 32;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        M_PAD, NT, 32, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    const int n0 = (int)tgid.y * NT;
    const int groups_total = K / 32;
    const int chunk = (groups_total + SPLIT_K - 1) / SPLIT_K;
    const int g_beg = (int)tgid.z * chunk;
    const int g_end = metal::min(groups_total, g_beg + chunk);

    using t_a = metal::tensor<device half, metal::extents<int32_t, 32, M_PAD>,
                              metal::tensor_inline>;
    using t_b = metal::tensor<device metal::uint4b_format,
                              metal::extents<int32_t, NT, 32>,
                              metal::tensor_inline>;
    auto cTmp = op.template get_destination_cooperative_tensor<t_a, t_b,
                                                               float>();
    auto cMain = op.template get_destination_cooperative_tensor<t_a, t_b,
                                                                float>();
    constexpr int CAP = 16;  // 32*64 / 128 threads
    int32_t n_of[CAP];
    int32_t m_of[CAP];
    const uint16_t cap = cMain.get_capacity();
    for (uint16_t i = 0; i < cap && i < CAP; ++i) {
        if (cMain.is_valid_element(i)) {
            auto idx = cMain.get_multidimensional_index(i);
            n_of[i] = idx[0];
            m_of[i] = idx[1];
            cMain[i] = 0.0f;
        }
    }
    device const uchar* wtile = Wu + (size_t)tgid.y * ((size_t)K * NT / 2);
    for (int g = g_beg; g < g_end; ++g) {
        t_a tA((device half*)(X + g * 32),
               metal::extents<int32_t, 32, M_PAD>(),
               metal::array<int32_t, 2>{1, K});
        t_b tB((device uchar*)(wtile + (size_t)g * (32 * NT / 2)),
               metal::extents<int32_t, NT, 32>());
        op.run(tA, tB, cTmp);
        device const half* sc_row = SC + (size_t)g * N + n0;
        device const half* mn_row = MN + (size_t)g * N + n0;
        device const half* xs_row = XS + (size_t)g * 32;
        for (uint16_t i = 0; i < cap && i < CAP; ++i) {
            if (cMain.is_valid_element(i)) {
                cMain[i] += (float)sc_row[n_of[i]] * cTmp[i] -
                            (float)mn_row[n_of[i]] * (float)xs_row[m_of[i]];
            }
        }
    }
    // scatter store into the (N, 32) m-contiguous partials layout
    device float* pz = P + (size_t)tgid.z * N * M_PAD;
    for (uint16_t i = 0; i < cap && i < CAP; ++i) {
        if (cMain.is_valid_element(i)) {
            pz[(size_t)(n0 + n_of[i]) * M_PAD + m_of[i]] = cMain[i];
        }
    }
}

// Glue-free u4 flavor: A is the ORIGINAL (M, K) bf16 activation tensor
// (dynamic m extent -- matmul2d bounds-checks the edge), output goes
// through qgemm_sm_reduce_rm straight to (M, N) bf16. Kills the per-call
// pad/cast/transpose torch glue on both sides of the op.
kernel void qgemm_sm_u4b_xsum(
    device   half* XS [[buffer(0)]],   // (K/32, 32)
    device const bfloat* X [[buffer(1)]],  // (M, K) bf16 row-major
    const constant int &K [[buffer(2)]],
    const constant int &M [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    const int g = (int)gid >> 5;
    const int m = (int)gid & 31;
    if (g >= K / 32) return;
    float acc = 0.0f;
    if (m < M) {
        for (int i = 0; i < 32; i++)
            acc += (float)X[(size_t)m * K + g * 32 + i];
    }
    XS[(size_t)g * 32 + m] = half(acc);
}

kernel void qgemm_sm_u4b(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device const uchar* Wu [[buffer(1)]],  // tile-major packed uint4
    device const bfloat* X [[buffer(2)]],  // (M, K) bf16 row-major
    device const half*  SC [[buffer(3)]],  // (K/32, N)
    device const half*  MN [[buffer(4)]],  // (K/32, N)
    device const half*  XS [[buffer(5)]],  // (K/32, 32)
    const constant int &N [[buffer(6)]],
    const constant int &K [[buffer(7)]],
    const constant int &M [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
    constexpr const int NT      = 64;
    constexpr const int SPLIT_K = 4;
    constexpr const int M_PAD   = 32;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        M_PAD, NT, 32, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    const int n0 = (int)tgid.y * NT;
    const int groups_total = K / 32;
    const int chunk = (groups_total + SPLIT_K - 1) / SPLIT_K;
    const int g_beg = (int)tgid.z * chunk;
    const int g_end = metal::min(groups_total, g_beg + chunk);

    using t_a = metal::tensor<device bfloat, metal::dextents<int32_t, 2>,
                              metal::tensor_inline>;
    using t_b = metal::tensor<device metal::uint4b_format,
                              metal::extents<int32_t, NT, 32>,
                              metal::tensor_inline>;
    auto cTmp = op.template get_destination_cooperative_tensor<t_a, t_b,
                                                               float>();
    auto cMain = op.template get_destination_cooperative_tensor<t_a, t_b,
                                                                float>();
    constexpr int CAP = 16;
    int32_t n_of[CAP];
    int32_t m_of[CAP];
    const uint16_t cap = cMain.get_capacity();
    for (uint16_t i = 0; i < cap && i < CAP; ++i) {
        if (cMain.is_valid_element(i)) {
            auto idx = cMain.get_multidimensional_index(i);
            n_of[i] = idx[0];
            m_of[i] = idx[1];
            cMain[i] = 0.0f;
        }
    }
    device const uchar* wtile = Wu + (size_t)tgid.y * ((size_t)K * NT / 2);
    for (int g = g_beg; g < g_end; ++g) {
        t_a tA((device bfloat*)(X + g * 32),
               metal::dextents<int32_t, 2>(32, M),
               metal::array<int32_t, 2>{1, K});
        t_b tB((device uchar*)(wtile + (size_t)g * (32 * NT / 2)),
               metal::extents<int32_t, NT, 32>());
        op.run(tA, tB, cTmp);
        device const half* sc_row = SC + (size_t)g * N + n0;
        device const half* mn_row = MN + (size_t)g * N + n0;
        device const half* xs_row = XS + (size_t)g * 32;
        for (uint16_t i = 0; i < cap && i < CAP; ++i) {
            if (cMain.is_valid_element(i)) {
                cMain[i] += (float)sc_row[n_of[i]] * cTmp[i] -
                            (float)mn_row[n_of[i]] * (float)xs_row[m_of[i]];
            }
        }
    }
    device float* pz = P + (size_t)tgid.z * N * M_PAD;
    for (uint16_t i = 0; i < cap && i < CAP; ++i) {
        if (cMain.is_valid_element(i)) {
            pz[(size_t)(n0 + n_of[i]) * M_PAD + m_of[i]] = cMain[i];
        }
    }
}

// ---- int8-native q6_K GEMM -------------------------------------------------
// q6_K is symmetric (w = d*sc16*(q-32)), so the -32 folds into the int8
// repack at load time: Wq8 is tile-major int8 ([n_tile=64][K][64], each
// 16-k group a contiguous 1 KB tile -- strided rows measured 17% slower
// than v15; contiguous tiles are how the op wants to stream) and SC is a
// (K/16, N) half plane of d*sc16. No min-term. Per 16-k group: multiply
// into a tmp cooperative tensor, cMain += sc[g,n]*tmp.
kernel void qgemm_sm_u8(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device const char*  Wq8 [[buffer(1)]],  // tile-major int8
    device const half*  X   [[buffer(2)]],  // (32, K) row-major X^T
    device const half*  SC  [[buffer(3)]],  // (K/16, N)
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
    constexpr const int NT      = 64;
    constexpr const int SPLIT_K = 4;
    constexpr const int M_PAD   = 32;
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        M_PAD, NT, 16, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;

    const int n0 = (int)tgid.y * NT;
    const int groups_total = K / 16;
    const int chunk = (groups_total + SPLIT_K - 1) / SPLIT_K;
    const int g_beg = (int)tgid.z * chunk;
    const int g_end = metal::min(groups_total, g_beg + chunk);

    using t_a = metal::tensor<device half, metal::extents<int32_t, 16, M_PAD>,
                              metal::tensor_inline>;
    using t_b = metal::tensor<device int8_t, metal::extents<int32_t, NT, 16>,
                              metal::tensor_inline>;
    auto cTmp = op.template get_destination_cooperative_tensor<t_a, t_b,
                                                               float>();
    auto cMain = op.template get_destination_cooperative_tensor<t_a, t_b,
                                                                float>();
    constexpr int CAP = 16;  // 32*64 / 128 threads
    int32_t n_of[CAP];
    int32_t m_of[CAP];
    const uint16_t cap = cMain.get_capacity();
    for (uint16_t i = 0; i < cap && i < CAP; ++i) {
        if (cMain.is_valid_element(i)) {
            auto idx = cMain.get_multidimensional_index(i);
            n_of[i] = idx[0];
            m_of[i] = idx[1];
            cMain[i] = 0.0f;
        }
    }
    device const char* wtile = Wq8 + (size_t)tgid.y * ((size_t)K * NT);
    for (int g = g_beg; g < g_end; ++g) {
        t_a tA((device half*)(X + g * 16),
               metal::extents<int32_t, 16, M_PAD>(),
               metal::array<int32_t, 2>{1, K});
        t_b tB((device int8_t*)(wtile + (size_t)g * 16 * NT),
               metal::extents<int32_t, NT, 16>());
        op.run(tA, tB, cTmp);
        device const half* sc_row = SC + (size_t)g * N + n0;
        for (uint16_t i = 0; i < cap && i < CAP; ++i) {
            if (cMain.is_valid_element(i)) {
                cMain[i] += (float)sc_row[n_of[i]] * cTmp[i];
            }
        }
    }
    device float* pz = P + (size_t)tgid.z * N * M_PAD;
    for (uint16_t i = 0; i < cap && i < CAP; ++i) {
        if (cMain.is_valid_element(i)) {
            pz[(size_t)(n0 + n_of[i]) * M_PAD + m_of[i]] = cMain[i];
        }
    }
}

#define instantiate_qgemm_sm_t2(name, FMT, PAIRED, NW, ...)                   \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm_t2<FMT, PAIRED, NW, ##__VA_ARGS__>(                          \
     device float* P [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm_t2("qgemm_sm_t2_q4_K", q4_K, true, 4);
instantiate_qgemm_sm_t2("qgemm_sm_t2_q5_K", q5_K, true, 4);
instantiate_qgemm_sm_t2("qgemm_sm_t2_q6_K", q6_K, false, 4);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8_q4_K", q4_K, true, 8);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8_q5_K", q5_K, true, 8);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8_q6_K", q6_K, false, 8);
// bench-only ablations of the 8-warp q5_K kernel (wide-shape wall hunt):
// a1 = W staged once, a2 = MMA (and its device-X stream) off, a3 = dequant
// ALU replaced by raw splat (same loads)
instantiate_qgemm_sm_t2("qgemm_sm_t2w8a1_q5_K", q5_K, true, 8, 1);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8a2_q5_K", q5_K, true, 8, 2);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8a3_q5_K", q5_K, true, 8, 3);
// split-K 1 direct-store flavor for wide shapes (enough threadgroups
// without K-splitting; deletes ~20 MB of partials traffic + the reduce)
instantiate_qgemm_sm_t2("qgemm_sm_t2w8s1_q5_K", q5_K, true, 8, 0, 1);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8s1_q4_K", q4_K, true, 8, 0, 1);
instantiate_qgemm_sm_t2("qgemm_sm_t2w8s1_q6_K", q6_K, false, 8, 0, 1);

#define instantiate_qgemm_sm_t(name, FMT, PAIRED)                             \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm_t<FMT, PAIRED>(                                              \
     device float* P [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm_t("qgemm_sm_t_q4_K", q4_K, true);
instantiate_qgemm_sm_t("qgemm_sm_t_q5_K", q5_K, true);
instantiate_qgemm_sm_t("qgemm_sm_t_q6_K", q6_K, false);
#endif  // __HAVE_TENSOR__

// ---- bench-only stage ablations (per-stage measurement) --------------------
// Dequant-off loader for the ablation kernel: identical device loads (block
// header, scale byte, 8-byte qs run at the same addresses) but the 6-bit
// scale unpack, nibble plane splits, and per-plane fma are dropped. Each
// loaded value is folded into a stored half so nothing is DCE'd. The delta
// vs dequant_into_shared_paired is the exposed decode-ALU cost.
// (primary template declared above the tensor-ops kernels)

// Weight-stream-only probe: the paired loader's exact device addresses
// (header, scale byte, qs run), checksummed with minimal ALU. Measures the
// achievable stream rate of this access pattern alone.
template<typename FMT>
METAL_FUNC float stream_weights_only(device const uchar* Wq, int N, int K,
                                     int rb, int kb64, uint lane);

template<>
METAL_FUNC float stream_weights_only<q4_K>(
        device const uchar* Wq, int N, int K, int rb, int kb64, uint lane) {
    const int row = (int)lane >> 2;
    const int j0  = ((int)lane & 3) * 8;
    const int grow = rb * 8 + row;
    const int gk = kb64 * 64;
    const int blk = gk >> 8;
    const int chunk = (gk & 255) >> 6;
    device const uchar* base =
        Wq + (size_t)(grow * (K >> 8) + blk) * q4_K::block_bytes;
    const half d   = ((device const half*)base)[0];
    const uchar s0 = (base + 4)[chunk * 2];
    const packed_uint2 qw =
        *(device const packed_uint2*)(base + 16 + chunk * 32 + j0);
    return (float)d + (float)s0 + (float)(qw.x ^ qw.y);
}


// Stage-ablation copies of qgemm_sm_p (BENCH-ONLY; parity is intentionally
// broken for every variant except 6). One stage is removed per variant so the
// full kernel's time decomposes into measured, not inferred, per-stage terms
// (full kernel = qgemm_sm_p4, host variant 12; ablations are 21..27):
//   A=1: X staged in prologue only          -> delta = X re-staging cost
//   A=2: dequant ALU dropped, loads kept    -> delta = decode-ALU cost
//   A=3: MMA + fragment loads dropped       -> delta = MMA + fragment cost
//   A=4: weights staged in prologue only    -> delta = weight stream + decode
//   A=5: weight stream only, no X/MMA/barrier -> pure stream rate floor
//   A=6: threadgroup -> simdgroup barrier   -> delta = barrier latency
//   A=7: fragment loads hoisted, MMA kept   -> delta = fragment-load cost
template<typename FMT, int N_WARPS, int ABLATE>
kernel void qgemm_sm_pa(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],   // (K, 32) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int RPW     = 8;
    constexpr const int SPLIT_K = 4;
    constexpr const int BK      = 64;
    constexpr const int M_PAD   = 32;
    using G = group<N_WARPS>;

    using gl_h = gl<half, 1, 1, -1, -1>;
    gl_h gl_x(X, nullptr, nullptr, K, M_PAD);

    threadgroup st<half, BK, M_PAD> sX[2];
    threadgroup st<half, RPW, BK> sW[2][N_WARPS];

    rt<half, RPW, BK> w_reg;
    rt<half, BK, M_PAD> x_reg;
    rt<float, RPW, M_PAD> d_reg;
    zero(d_reg);

    const int rb = tgid.y * N_WARPS + (int)warp;
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    if (ABLATE == 5) {
        float acc = 0.0f;
        for (int kb = kb_beg; kb < kb_end; kb++) {
            acc += stream_weights_only<FMT>(Wq, N, K, rb, kb, lane);
        }
        // write/read the threadgroup tiles so their allocation (and the
        // kernel's occupancy shape) survives; fold everything into P
        sX[0][int2((int)tid % BK, (int)tid / BK)] = half(acc);
        sW[0][warp][int2((int)lane % RPW, (int)lane)] = half(acc);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        const float keep = acc +
            (float)sX[0][int2((int)(tid ^ 1) % BK, (int)(tid ^ 1) / BK)] +
            (float)sW[0][warp][int2((int)(lane ^ 1) % RPW, (int)(lane ^ 1))];
        P[(size_t)tgid.z * N * M_PAD + (size_t)rb * RPW * M_PAD + (int)lane] =
            keep;
        return;
    }

    // prologue: stage once; the "staged once" ablations fill BOTH double
    // buffers with real data so the loop reads valid halves (no NaN skew)
    G::load(sX[kb_beg & 1], gl_x, {0, 0, kb_beg, 0}, tid);
    if (ABLATE == 1) {
        G::load(sX[(kb_beg & 1) ^ 1], gl_x,
                {0, 0, metal::min(kb_beg + 1, steps_total - 1), 0}, tid);
    }
    if (ABLATE == 2) {
        dequant_paired_raw<FMT>(sW[kb_beg & 1][warp], Wq, N, K, rb, kb_beg,
                                lane);
    } else {
        dequant_into_shared_paired<FMT>(sW[kb_beg & 1][warp], Wq, N, K, rb,
                                        kb_beg, lane);
    }
    if (ABLATE == 4) {
        dequant_into_shared_paired<FMT>(
            sW[(kb_beg & 1) ^ 1][warp], Wq, N, K, rb,
            metal::min(kb_beg + 1, steps_total - 1), lane);
    }
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    if (ABLATE == 7) {
        load(w_reg, sW[kb_beg & 1][warp], lane);
        load(x_reg, sX[kb_beg & 1], lane);
    }

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        const int nxt = cur ^ 1;
        if (kb + 1 < kb_end) {
            if (ABLATE != 1) {
                G::load(sX[nxt], gl_x, {0, 0, kb + 1, 0}, tid);
            }
            if (ABLATE != 4) {
                if (ABLATE == 2) {
                    dequant_paired_raw<FMT>(sW[nxt][warp], Wq, N, K, rb,
                                            kb + 1, lane);
                } else {
                    dequant_into_shared_paired<FMT>(sW[nxt][warp], Wq, N, K,
                                                    rb, kb + 1, lane);
                }
            }
        }
        if (ABLATE != 3 && ABLATE != 7) {
            load(w_reg, sW[cur][warp], lane);
            load(x_reg, sX[cur], lane);
        }
        if (ABLATE != 3) {
            mma_AB(d_reg, w_reg, x_reg, d_reg);
        }
        if (ABLATE == 6) {
            simdgroup_barrier(metal::mem_flags::mem_threadgroup);
        } else {
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }
    if (ABLATE == 3 || ABLATE == 7) {
        // one tail MMA from freshly loaded fragments keeps every staged tile
        // live (runtime buffer index defeats store elimination) at 1/steps
        // of the per-step cost being ablated
        load(w_reg, sW[kb_end & 1][warp], lane);
        load(x_reg, sX[kb_end & 1], lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
    }
    using gl_f = gl<float, 1, 1, -1, -1>;
    gl_f gl_p(P + (size_t)tgid.z * N * M_PAD, nullptr, nullptr, N, M_PAD);
    store(gl_p, d_reg, {0, 0, rb, 0}, lane);
}

#define instantiate_qgemm_sm_pa(name, FMT, NW, A)                             \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm_pa<FMT, NW, A>(                                              \
     device float* P [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm_pa("qgemm_sm_p4a1_q4_K", q4_K, 4, 1);
instantiate_qgemm_sm_pa("qgemm_sm_p4a2_q4_K", q4_K, 4, 2);
instantiate_qgemm_sm_pa("qgemm_sm_p4a3_q4_K", q4_K, 4, 3);
instantiate_qgemm_sm_pa("qgemm_sm_p4a4_q4_K", q4_K, 4, 4);
instantiate_qgemm_sm_pa("qgemm_sm_p4a5_q4_K", q4_K, 4, 5);
instantiate_qgemm_sm_pa("qgemm_sm_p4a6_q4_K", q4_K, 4, 6);
instantiate_qgemm_sm_pa("qgemm_sm_p4a7_q4_K", q4_K, 4, 7);

// BK=128 flavor of the q5_K paired-plane loader: same math as
// dequant_into_shared_paired<q5_K>, destination is an st<8,128> tile and
// col_base selects which 64-chunk half of the step this call fills. Each
// step covers two chunks, so each lane reads 16 contiguous qs bytes per
// step (vs 8 at BK=64) -- the GEMV kernels prove this layout streams at
// ~floor when reads are long and row-sequential.
METAL_FUNC void dequant_paired_q5k_bk128(
        threadgroup st<half, 8, 128>& dst, device const uchar* Wq, int N,
        int K, int rb, int kb64, int col_base, uint lane) {
    const int row = (int)lane >> 2;
    const int j0  = ((int)lane & 3) * 8;
    const int grow = rb * 8 + row;
    const int gk = kb64 * 64;
    const int blk = gk >> 8;
    const int chunk = (gk & 255) >> 6;
    device const uchar* base =
        Wq + (size_t)(grow * (K >> 8) + blk) * q5_K::block_bytes;
    const half d    = ((device const half*)base)[0];
    const half dmin = ((device const half*)(base + 2))[0];
    device const uchar* sca = base + 4;
    const int is0 = chunk * 2;
    int sc0, mn0, sc1, mn1;
    if (is0 < 4) {
        sc0 = sca[is0] & 63;     mn0 = sca[is0 + 4] & 63;
        sc1 = sca[is0 + 1] & 63; mn1 = sca[is0 + 5] & 63;
    } else {
        sc0 = (sca[is0 + 4] & 0x0F) | ((sca[is0 - 4] >> 6) << 4);
        mn0 = (sca[is0 + 4] >> 4)   | ((sca[is0]     >> 6) << 4);
        sc1 = (sca[is0 + 5] & 0x0F) | ((sca[is0 - 3] >> 6) << 4);
        mn1 = (sca[is0 + 5] >> 4)   | ((sca[is0 + 1] >> 6) << 4);
    }
    const half dl0 = d * half(sc0), ml0 = dmin * half(mn0);
    const half dl1 = d * half(sc1), ml1 = dmin * half(mn1);
    const packed_uint2 qw =
        *(device const packed_uint2*)(base + 48 + chunk * 32 + j0);
    const packed_uint2 hw = *(device const packed_uint2*)(base + 16 + j0);
    const uint vlx = (qw.x & 0x0F0F0F0Fu) | (((hw.x >> is0) & 0x01010101u) << 4);
    const uint vly = (qw.y & 0x0F0F0F0Fu) | (((hw.y >> is0) & 0x01010101u) << 4);
    const uint vhx =
        ((qw.x >> 4) & 0x0F0F0F0Fu) | (((hw.x >> (is0 + 1)) & 0x01010101u) << 4);
    const uint vhy =
        ((qw.y >> 4) & 0x0F0F0F0Fu) | (((hw.y >> (is0 + 1)) & 0x01010101u) << 4);
    threadgroup half4* plo = (threadgroup half4*)&dst[int2(row, col_base + j0)];
    threadgroup half4* phi =
        (threadgroup half4*)&dst[int2(row, col_base + 32 + j0)];
    plo[0] = half4(as_type<uchar4>(vlx)) * dl0 - ml0;
    plo[1] = half4(as_type<uchar4>(vly)) * dl0 - ml0;
    phi[0] = half4(as_type<uchar4>(vhx)) * dl1 - ml1;
    phi[1] = half4(as_type<uchar4>(vhy)) * dl1 - ml1;
}

// BK=128 device-X tensor kernel for q5_K (gate/up): halves the K-steps and
// barriers, doubles per-lane contiguous qs reads. Same split-K partials.
[[host_name("qgemm_sm_t4_q5_K")]]
kernel void qgemm_sm_t4_q5_K(
    device   float* P  [[buffer(0)]],   // (4, N, 32) float partials
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],   // (K, 32) activations
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int N_WARPS = 4;
    constexpr const int RPW     = 8;
    constexpr const int SPLIT_K = 4;
    constexpr const int BK      = 128;
    constexpr const int M_PAD   = 32;
    constexpr const int ROWS    = N_WARPS * RPW;

    threadgroup st<half, RPW, BK> sW[2][N_WARPS];

    const int rb = tgid.y * N_WARPS + (int)warp;
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    constexpr auto t_desc = mpp::tensor_ops::matmul2d_descriptor(
        ROWS, M_PAD, BK, false, false, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<t_desc, metal::execution_simdgroups<N_WARPS>> op;

    using tg_a = metal::tensor<threadgroup half,
                               metal::extents<int32_t, BK, ROWS>,
                               metal::tensor_inline>;
    using dev_b = metal::tensor<device half,
                                metal::extents<int32_t, M_PAD, BK>,
                                metal::tensor_inline>;
    tg_a tA0((threadgroup half*)&sW[0][0], metal::extents<int32_t, BK, ROWS>());
    tg_a tA1((threadgroup half*)&sW[1][0], metal::extents<int32_t, BK, ROWS>());

    auto cT = op.template get_destination_cooperative_tensor<tg_a, dev_b,
                                                             float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    }

    dequant_paired_q5k_bk128(sW[kb_beg & 1][warp], Wq, N, K, rb, 2 * kb_beg,
                             0, lane);
    dequant_paired_q5k_bk128(sW[kb_beg & 1][warp], Wq, N, K, rb,
                             2 * kb_beg + 1, 64, lane);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        if (kb + 1 < kb_end) {
            dequant_paired_q5k_bk128(sW[cur ^ 1][warp], Wq, N, K, rb,
                                     2 * (kb + 1), 0, lane);
            dequant_paired_q5k_bk128(sW[cur ^ 1][warp], Wq, N, K, rb,
                                     2 * (kb + 1) + 1, 64, lane);
        }
        dev_b tB(X + (size_t)kb * BK * M_PAD,
                 metal::extents<int32_t, M_PAD, BK>());
        if (cur == 0) {
            op.run(tA0, tB, cT);
        } else {
            op.run(tA1, tB, cT);
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    using dev_c = metal::tensor<device float,
                                metal::extents<int32_t, M_PAD, ROWS>,
                                metal::tensor_inline>;
    dev_c tC(P + (size_t)tgid.z * N * M_PAD + (size_t)tgid.y * ROWS * M_PAD,
             metal::extents<int32_t, M_PAD, ROWS>());
    cT.store(tC);
}

// Cooperative transposing bf16 -> half X stage for qgemm_sm_rm:
// sX[k][m] = X[m, kb*BK + k]; rows >= M stage zeros.
template<int BK, int M_PAD, int GROUP_THREADS>
METAL_FUNC void qgemm_sm_rm_stage_x(threadgroup st<half, BK, M_PAD>& sX,
                                    device const ushort* X, int K, int M,
                                    int kb, uint tid) {
    // ushort4 spans: one vector load covers 4 consecutive k of one X row
    constexpr int SPANS_PER_ROW = BK / 4;
    for (int s = (int)tid; s < M_PAD * SPANS_PER_ROW; s += GROUP_THREADS) {
        const int mcol = s / SPANS_PER_ROW;
        const int kcol = (s % SPANS_PER_ROW) * 4;
        if (mcol < M) {
            const ushort4 u = *(device const ushort4*)(
                X + (size_t)mcol * K + kb * BK + kcol);
            sX[int2(kcol + 0, mcol)] = half(as_type<float>(uint(u.x) << 16));
            sX[int2(kcol + 1, mcol)] = half(as_type<float>(uint(u.y) << 16));
            sX[int2(kcol + 2, mcol)] = half(as_type<float>(uint(u.z) << 16));
            sX[int2(kcol + 3, mcol)] = half(as_type<float>(uint(u.w) << 16));
        } else {
            sX[int2(kcol + 0, mcol)] = half(0.0f);
            sX[int2(kcol + 1, mcol)] = half(0.0f);
            sX[int2(kcol + 2, mcol)] = half(0.0f);
            sX[int2(kcol + 3, mcol)] = half(0.0f);
        }
    }
}

// Row-major variant for the fused muse_step verify: X is (M, K) bf16 exactly
// as the serving scratch holds it (no host transpose/pad/cast), D is (M, N)
// bf16. Fixed at the best measured config: 2 warps x 8 rows, split-K 4 float
// partials (fold with qgemm_sm_reduce_rm). The X K-tile transposes during
// cooperative staging; rows >= M stage zeros so the MMA columns are inert.
template<typename FMT>
kernel void qgemm_sm_rm(
    device   float* P  [[buffer(0)]],   // (4, N, M_PAD) float partials
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device const ushort* X [[buffer(2)]],   // (M, K) bf16 activations, row-major
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    const constant int &M [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  warp [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr const int N_WARPS = 2;
    constexpr const int RPW     = 8;
    constexpr const int SPLIT_K = 4;
    constexpr const int BK      = 32;
    constexpr const int M_PAD   = 32;
    constexpr const int GROUP_THREADS = N_WARPS * 32;

    threadgroup st<half, BK, M_PAD> sX[2];
    threadgroup st<half, RPW, BK> sW[2][N_WARPS];

    rt<half, RPW, BK> w_reg;
    rt<half, BK, M_PAD> x_reg;
    rt<float, RPW, M_PAD> d_reg;
    zero(d_reg);

    const int rb = tgid.y * N_WARPS + (int)warp;
    const int steps_total = K / BK;
    const int chunk = (steps_total + SPLIT_K - 1) / SPLIT_K;
    const int kb_beg = (int)tgid.z * chunk;
    const int kb_end = metal::min(steps_total, kb_beg + chunk);

    qgemm_sm_rm_stage_x<BK, M_PAD, GROUP_THREADS>(sX[kb_beg & 1], X, K, M,
                                                   kb_beg, tid);
    dequant_into_shared<FMT, RPW, BK>(sW[kb_beg & 1][warp], Wq, N, K, rb,
                                      kb_beg, 32, lane);
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int kb = kb_beg; kb < kb_end; kb++) {
        const int cur = kb & 1;
        const int nxt = cur ^ 1;
        if (kb + 1 < kb_end) {
            qgemm_sm_rm_stage_x<BK, M_PAD, GROUP_THREADS>(sX[nxt], X, K, M,
                                                          kb + 1, tid);
            dequant_into_shared<FMT, RPW, BK>(sW[nxt][warp], Wq, N, K, rb,
                                              kb + 1, 32, lane);
        }
        load(w_reg, sW[cur][warp], lane);
        load(x_reg, sX[cur], lane);
        mma_AB(d_reg, w_reg, x_reg, d_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    using gl_f = gl<float, 1, 1, -1, -1>;
    gl_f gl_p(P + (size_t)tgid.z * N * M_PAD, nullptr, nullptr, N, M_PAD);
    store(gl_p, d_reg, {0, 0, rb, 0}, lane);
}

// Fold qgemm_sm_rm partials (SK, N, 32) into bf16 (M, N), transposed store.
// gid is laid out m-major so the D writes coalesce along N.
kernel void qgemm_sm_reduce_rm(
    device   ushort* D [[buffer(0)]],   // (M, N) bf16
    device   float*  P [[buffer(1)]],
    const constant int &N [[buffer(2)]],
    const constant int &M [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr const int M_PAD = 32;
    constexpr const int SK    = 4;
    if ((int)gid >= M * N) return;
    const int mcol = (int)gid / N;
    const int n    = (int)gid % N;
    float acc = 0.0f;
    for (int z = 0; z < SK; z++) {
        acc += P[((size_t)z * N + n) * M_PAD + mcol];
    }
    const uint bits = as_type<uint>(acc);
    const uint rounded = bits + 0x7FFFu + ((bits >> 16) & 1u);
    D[(size_t)mcol * N + n] = ushort(rounded >> 16);
}

// Fold split-K float partials (SK, N, M_PAD) into the half output (N, M_PAD).
kernel void qgemm_sm_reduce(
    device   half*  D  [[buffer(0)]],
    device   float* P  [[buffer(1)]],
    const constant int &N  [[buffer(2)]],
    const constant int &SK [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
    constexpr const int M_PAD = 32;
    if ((int)gid >= N * M_PAD) return;
    float acc = 0.0f;
    for (int z = 0; z < SK; z++) acc += P[(size_t)z * N * M_PAD + gid];
    D[gid] = half(acc);
}

#define instantiate_qgemm_sm(name, FMT, NW, RPW, SK)                          \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm<FMT, NW, RPW, SK, 32>(                                       \
     device uchar* Dv [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm("qgemm_sm_q4_0", q4_0, 4, 16, 1);
instantiate_qgemm_sm("qgemm_sm_q8_0", q8_0, 4, 16, 1);
instantiate_qgemm_sm("qgemm_sm_q4_K", q4_K, 4, 16, 1);
instantiate_qgemm_sm("qgemm_sm_q5_K", q5_K, 4, 16, 1);
instantiate_qgemm_sm("qgemm_sm_q6_K", q6_K, 4, 16, 1);

instantiate_qgemm_sm("qgemm_sm_w2_q4_0", q4_0, 2, 16, 1);
instantiate_qgemm_sm("qgemm_sm_w2_q8_0", q8_0, 2, 16, 1);
instantiate_qgemm_sm("qgemm_sm_w2_q4_K", q4_K, 2, 16, 1);
instantiate_qgemm_sm("qgemm_sm_w2_q5_K", q5_K, 2, 16, 1);
instantiate_qgemm_sm("qgemm_sm_w2_q6_K", q6_K, 2, 16, 1);

instantiate_qgemm_sm("qgemm_sm_r8_q4_0", q4_0, 2, 8, 1);
instantiate_qgemm_sm("qgemm_sm_r8_q8_0", q8_0, 2, 8, 1);
instantiate_qgemm_sm("qgemm_sm_r8_q4_K", q4_K, 2, 8, 1);
instantiate_qgemm_sm("qgemm_sm_r8_q5_K", q5_K, 2, 8, 1);
instantiate_qgemm_sm("qgemm_sm_r8_q6_K", q6_K, 2, 8, 1);

instantiate_qgemm_sm("qgemm_sm_r8sk4_q4_0", q4_0, 2, 8, 4);
instantiate_qgemm_sm("qgemm_sm_r8sk4_q8_0", q8_0, 2, 8, 4);
instantiate_qgemm_sm("qgemm_sm_r8sk4_q4_K", q4_K, 2, 8, 4);
instantiate_qgemm_sm("qgemm_sm_r8sk4_q5_K", q5_K, 2, 8, 4);
instantiate_qgemm_sm("qgemm_sm_r8sk4_q6_K", q6_K, 2, 8, 4);
instantiate_qgemm_sm("qgemm_sm_r8sk4_f16_raw", f16_raw, 2, 8, 4);

#define instantiate_qgemm_sm_bk64(name, FMT)                                  \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm<FMT, 2, 8, 4, 64>(                                           \
     device uchar* Dv [[buffer(0)]], device uchar* Wq [[buffer(1)]], device half* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm_bk64("qgemm_sm_bk64_q4_0", q4_0);
instantiate_qgemm_sm_bk64("qgemm_sm_bk64_q8_0", q8_0);
instantiate_qgemm_sm_bk64("qgemm_sm_bk64_q4_K", q4_K);
instantiate_qgemm_sm_bk64("qgemm_sm_bk64_q5_K", q5_K);
instantiate_qgemm_sm_bk64("qgemm_sm_bk64_q6_K", q6_K);

#define instantiate_qgemm_sm_rm(name, FMT)                                    \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_sm_rm<FMT>(                                                     \
     device float* P [[buffer(0)]], device uchar* Wq [[buffer(1)]], device const ushort* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint tid [[thread_index_in_threadgroup]],                               \
     uint warp [[simdgroup_index_in_threadgroup]],                           \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_sm_rm("qgemm_sm_rm_q4_0", q4_0);
instantiate_qgemm_sm_rm("qgemm_sm_rm_q8_0", q8_0);
instantiate_qgemm_sm_rm("qgemm_sm_rm_q4_K", q4_K);
instantiate_qgemm_sm_rm("qgemm_sm_rm_q5_K", q5_K);
instantiate_qgemm_sm_rm("qgemm_sm_rm_q6_K", q6_K);

}
