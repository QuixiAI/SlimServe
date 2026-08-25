#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Quantized GEMV (batch-1 decode):  d = dequantize(W) @ x,  W (N,K) quantized blocks, x (K,1).
// No MMA — one simdgroup (32 lanes) per output row, walked block-major: each lane owns an
// 8-col contiguous span inside a block (block_k/8 lanes cover a block; the simdgroup covers
// 32/(block_k/8) blocks per iteration). The span keeps the block-scale reads CSE-able, kills
// the per-element div/mod of the old strided walk, and lets X load as half4. This is the
// memory-bound decode path where shrinking the weight bytes (4-8x) is the real Apple win.
template<typename FMT, typename T>
kernel void qgemv(
    device   T*     D  [[buffer(0)]],   // (N, 1) output
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   T*     X  [[buffer(2)]],   // (K, 1) activation vector
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;                          // one threadgroup (simdgroup) per output row
    const int bpr = K / FMT::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * FMT::block_bytes;

    constexpr int CPL = 8;                           // contiguous cols per lane
    constexpr int LPB = FMT::block_k / CPL;          // lanes per block (2..32)
    constexpr int BPI = 32 / LPB;                    // blocks per simdgroup iteration (1..16)
    const int b_off = (int)lane / LPB;
    const int col0  = ((int)lane % LPB) * CPL;

    float acc = 0.0f;
    for (int kb = b_off; kb < bpr; kb += BPI) {
        device const uchar* base = row_base + (uint)kb * FMT::block_bytes;
        const int x_base = kb * FMT::block_k + col0;
        half w[8];
        tk_dequant8<FMT>(base, col0, w);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) acc += float(w[i]) * float(X[x_base + i]);
    }
    acc = metal::simd_sum(acc);                      // reduce the dot across the 32 lanes
    if (lane == 0) D[row] = T(acc);
}

// q4_K decode GEMV in the llama.cpp mul_mv layout (kernel_mul_mv_q4_K_f32
// port: 2 rows per simdgroup, 2 simdgroups per threadgroup, 4 blocks in
// flight on the K axis). The one-simdgroup-per-row walk above has BPI=1 for
// 256-wide blocks -- the whole simdgroup advances through a single 144-byte
// block per iteration -- and measured weight bandwidth collapses to 180-290
// GB/s at the Qwen3.8 MLP shapes vs ~500+ for the BPI=8 q8_0 walk. This
// kernel restores in-flight blocks (ib += 4), amortizes the activation loads
// across both rows from registers (yl/yh + per-sub-block activation sums
// loaded once per block phase), and keeps the nibbles integer inside the
// inner product: masked-ushort accumulate into four fp32 accumulators,
// renormalized once by 1/256 and 1/16, with the 6-bit sub-block scale/min
// factored out per (block, row). No dequantized span is ever materialized,
// which is what retires the register-pressure objection recorded against the
// two-rows-per-simdgroup experiment in launch_qgemv's geometry note.
// NUMERICS: fp32 y*nibble accumulation with factored scales has no
// per-element half rounding, so outputs are NOT bit-identical to qgemv<q4_K>
// (they are closer to the exact dequant-dot). The host routes only
// N % 4 == 0 && K % 256 == 0 here: tail rows would read past the end of the
// weight buffer, and MPS does not fault on out-of-bounds reads.
template<typename T>
kernel void qgemv_q4k_nr(
    device   T*     D  [[buffer(0)]],   // (N, 1) output
    device   uchar* Wq [[buffer(1)]],   // (N, K/256 * 144B) q4_K blocks
    device   T*     X  [[buffer(2)]],   // (K, 1) activation vector
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 2;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = lane / 8;          // 0..3: block phase on the K axis
    const short it = lane % 8;          // 0..7
    const short iq = it / 4;            // 0..1: 64-col half of the low 128
    const short ir = it % 4;            // 0..3: 8-col span within 32

    const int nb = K / 256;             // q4_K blocks per row
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const ulong row_bytes = (ulong)nb * 144;
    device const uchar* rows_base = Wq + (ulong)first_row * row_bytes;

    float yl[16];                       // y at cols 64*iq + 8*ir (+0, +32)
    float yh[16];                       // same span in the high 128 cols
    float sumf[NR] = {0.f, 0.f};

    device const T* y4 = X + ix * 256 + 64 * iq + 8 * ir;

    ushort sc16[4];
    thread const uchar* sc8 = (thread const uchar*)sc16;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = float(y4[i +   0]); sumy[0] += yl[i + 0];
            yl[i + 8] = float(y4[i +  32]); sumy[1] += yl[i + 8];
            yh[i + 0] = float(y4[i + 128]); sumy[2] += yh[i + 0];
            yh[i + 8] = float(y4[i + 160]); sumy[3] += yh[i + 8];
        }

        device const uchar* blk = rows_base + (ulong)ib * 144;
        #pragma clang loop unroll(full)
        for (short row = 0; row < NR; ++row) {
            device const uchar* b = blk + (ulong)row * row_bytes;
            device const half*   dh = (device const half*)b;
            device const ushort* sc = (device const ushort*)(b + 4) + iq;
            device const ushort* q1 = (device const ushort*)(b + 16)
                                      + 16 * iq + 4 * ir;
            device const ushort* q2 = q1 + 32;

            // All eight 6-bit (scale, min) pairs this lane needs, via the
            // GGUF get_scale_min_k4 packing, extracted as two masked ushorts
            // per half: sc8[0,1]/[4,5] scales, sc8[2,3]/[6,7] mins.
            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }
            const float dall = float(dh[0]);
            const float dmin = float(dh[1]);
            sumf[row] +=
                dall * ((acc1[0] + 1.f / 256.f * acc1[1]) * sc8[0] +
                        (acc1[2] + 1.f / 256.f * acc1[3]) * sc8[1] * 1.f / 16.f +
                        (acc2[0] + 1.f / 256.f * acc2[1]) * sc8[4] +
                        (acc2[2] + 1.f / 256.f * acc2[3]) * sc8[5] * 1.f / 16.f) -
                dmin * (sumy[0] * sc8[2] + sumy[1] * sc8[3] +
                        sumy[2] * sc8[6] + sumy[3] * sc8[7]);
        }
        y4 += 4 * 256;
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < NR; ++row) {
        const float tot = metal::simd_sum(sumf[row]);
        if (lane == 0) D[first_row + row] = T(tot);
    }
}

// Multi-batch twin of qgemv_q4k_nr for the even 2..8-row decode/verify
// widths: grid.y indexes COLUMN PAIRS (dispatch (N/4, M/2)), and each
// threadgroup runs the batch-1 lane geometry, block phasing, and factored
// scales over its two columns with the weight block's scale words and
// nibble ushorts shared from registers. Per-(row, column) FP chain is the
// batch-1 kernel's, so every output row is bit-identical to a looped
// qgemv_q4k_nr launch. The generic qgemv_mm walk measured 91/81 GB/s at
// the M=8 MLP shapes (the BPI=1 collapse), which is what kept c4/c8 flat
// when the batch-1 kernel landed.
//
// WHY COLUMN PAIRS ON THE GRID -- each alternative measured at the M=8 MLP
// shapes (see notebook UPDATE 11):
// - All M columns in one threadgroup, column loop UNROLLED (array staging,
//   fused scalars, literal-index macro chains, scheduling fences): 12-14
//   GB/s at M=8, ~90 at M=4 vs 379 at M=2 -- superlinear in M, identical
//   pipeline register footprints for M=4/8 (maxTotalThreadsPerThreadgroup
//   384 for both), independent of the X addresses read (all columns
//   reading row 0 reproduced it). That is instruction-cache thrash of the
//   unrolled block-loop body, not register spill or memory divergence.
// - All M columns, column loop ROLLED with threadgroup-memory accumulators
//   (register arrays under a rolled loop spill to stack; the TG slots were
//   lane-private and bank-conflict-free): only parity with the generic
//   walk (95/88 GB/s at M=8) -- the small body works, but occupancy and
//   the fold's TG RMW latency give back the weight-stationarity win.
// - Sequential per-pair K sweeps inside one threadgroup: break-even (96
//   GB/s at M=8); a sweep's weight working set does not stay
//   cache-resident, so every sweep repays device bandwidth.
// Splitting the pairs across grid.y instead nominally re-reads the weight
// bytes M/2 times, but threadgroups with equal tgid.x and different
// tgid.y are resident together, so the re-reads land in cache.
// The scale bytes are extracted arithmetically (& 0xFF / >> 8) rather than
// through the batch-1 kernel's `thread uchar*` view of sc16: same values
// and FP order, no thread-address aliasing to defeat promotion.
// Host guards match the batch-1 route (N % 4 == 0, K % 256 == 0, M even):
// tail rows would read past the end of the weight buffer, and MPS does
// not fault on out-of-bounds reads.
template<typename T>
kernel void qgemv_q4k_nr_mb(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   uchar* Wq [[buffer(1)]],   // (N, K/256 * 144B) q4_K blocks
    device   T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 2;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const short ix = lane / 8;          // 0..3: block phase on the K axis
    const short it = lane % 8;          // 0..7
    const short iq = it / 4;            // 0..1: 64-col half of the low 128
    const short ir = it % 4;            // 0..3: 8-col span within 32

    const int nb = K / 256;             // q4_K blocks per row
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const int first_col = int(tgid.y) * 2;
    const ulong row_bytes = (ulong)nb * 144;
    device const uchar* rows_base = Wq + (ulong)first_row * row_bytes;

    float sumf[NR][2] = {};

    for (int ib = ix; ib < nb; ib += 4) {
        // Weight registers for both rows, read once per block per pair.
        float  dallv[NR], dminv[NR];
        ushort sc16v[NR][4];
        ushort q1v[NR][4], q2v[NR][4];
        device const uchar* blk = rows_base + (ulong)ib * 144;
        #pragma clang loop unroll(full)
        for (short row = 0; row < NR; ++row) {
            device const uchar* b = blk + (ulong)row * row_bytes;
            device const half*   dh = (device const half*)b;
            device const ushort* sc = (device const ushort*)(b + 4) + iq;
            device const ushort* q1 = (device const ushort*)(b + 16)
                                      + 16 * iq + 4 * ir;
            dallv[row] = float(dh[0]);
            dminv[row] = float(dh[1]);
            sc16v[row][0] = sc[0] & kmask1;
            sc16v[row][1] = sc[2] & kmask1;
            sc16v[row][2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16v[row][3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                q1v[row][i] = q1[i];
                q2v[row][i] = q1[i + 32];
            }
        }

        // Two columns per threadgroup, register accumulators, per-column
        // fused y stage. Per-accumulator FP order matches the batch-1
        // kernel exactly.
        const int ycol = ib * 256 + 64 * iq + 8 * ir;
        #pragma clang loop unroll(full)
        for (short mi = 0; mi < 2; ++mi) {
            device const T* y4 = X + (long)(first_col + mi) * K + ycol;
            // Each 8-wide span rides two vec4 loads instead of 8 scalar
            // bf16 loads. Values and accumulate order are unchanged.
            using T4 = metal::vec<T, 4>;
            device const T4* yv = (device const T4*)y4;
            const float4 ya0 = float4(yv[0]),  ya1 = float4(yv[1]);
            const float4 yb0 = float4(yv[8]),  yb1 = float4(yv[9]);
            const float4 yc0 = float4(yv[32]), yc1 = float4(yv[33]);
            const float4 yd0 = float4(yv[40]), yd1 = float4(yv[41]);
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            float4 acc1[NR];
            float4 acc2[NR];
            #pragma clang loop unroll(full)
            for (short row = 0; row < NR; ++row) {
                acc1[row] = {0.f, 0.f, 0.f, 0.f};
                acc2[row] = {0.f, 0.f, 0.f, 0.f};
            }
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const float yl0 = (i < 2) ? ya0[2 * i + 0] : ya1[2 * i - 4];
                const float yl1 = (i < 2) ? ya0[2 * i + 1] : ya1[2 * i - 3];
                const float yl8 = (i < 2) ? yb0[2 * i + 0] : yb1[2 * i - 4];
                const float yl9 = (i < 2) ? yb0[2 * i + 1] : yb1[2 * i - 3];
                const float yh0 = (i < 2) ? yc0[2 * i + 0] : yc1[2 * i - 4];
                const float yh1 = (i < 2) ? yc0[2 * i + 1] : yc1[2 * i - 3];
                const float yh8 = (i < 2) ? yd0[2 * i + 0] : yd1[2 * i - 4];
                const float yh9 = (i < 2) ? yd0[2 * i + 1] : yd1[2 * i - 3];
                sumy[0] += yl0; sumy[0] += yl1;
                sumy[1] += yl8; sumy[1] += yl9;
                sumy[2] += yh0; sumy[2] += yh1;
                sumy[3] += yh8; sumy[3] += yh9;
                #pragma clang loop unroll(full)
                for (short row = 0; row < NR; ++row) {
                    acc1[row][0] += yl0 * (q1v[row][i] & 0x000F);
                    acc1[row][1] += yl1 * (q1v[row][i] & 0x0F00);
                    acc1[row][2] += yl8 * (q1v[row][i] & 0x00F0);
                    acc1[row][3] += yl9 * (q1v[row][i] & 0xF000);
                    acc2[row][0] += yh0 * (q2v[row][i] & 0x000F);
                    acc2[row][1] += yh1 * (q2v[row][i] & 0x0F00);
                    acc2[row][2] += yh8 * (q2v[row][i] & 0x00F0);
                    acc2[row][3] += yh9 * (q2v[row][i] & 0xF000);
                }
            }
            #pragma clang loop unroll(full)
            for (short row = 0; row < NR; ++row) {
                sumf[row][mi] +=
                    dallv[row] *
                        ((acc1[row][0] + 1.f / 256.f * acc1[row][1]) *
                             float(sc16v[row][0] & 0xFF) +
                         (acc1[row][2] + 1.f / 256.f * acc1[row][3]) *
                             float(sc16v[row][0] >> 8) * 1.f / 16.f +
                         (acc2[row][0] + 1.f / 256.f * acc2[row][1]) *
                             float(sc16v[row][2] & 0xFF) +
                         (acc2[row][2] + 1.f / 256.f * acc2[row][3]) *
                             float(sc16v[row][2] >> 8) * 1.f / 16.f) -
                    dminv[row] *
                        (sumy[0] * float(sc16v[row][1] & 0xFF) +
                         sumy[1] * float(sc16v[row][1] >> 8) +
                         sumy[2] * float(sc16v[row][3] & 0xFF) +
                         sumy[3] * float(sc16v[row][3] >> 8));
            }
        }
    }

    #pragma clang loop unroll(full)
    for (short mi = 0; mi < 2; ++mi)
        #pragma clang loop unroll(full)
        for (short row = 0; row < NR; ++row) {
            const float tot = metal::simd_sum(sumf[row][mi]);
            if (lane == 0)
                D[(long)(first_col + mi) * N + first_row + row] = T(tot);
        }
}

// Weight-stationary multi-row GEMV:  D (M,N) = X (M,K) @ dequantize(W)^T.
// Same block-major walk and lane geometry as qgemv, but each dequantized
// 8-wide weight span is applied to all M activation rows from registers, so
// the weight bytes -- the memory-bound term at decode -- are read once for
// the whole row block instead of once per row. M is a compile-time template
// parameter so the accumulator array unrolls into registers. Serves the
// small-M band (speculative verify/draft blocks, batched decode) where the
// per-row qgemv is linear in M and the fragment-path GEMM is ~4-5x off
// weight bandwidth.
template<typename FMT, typename T, int M>
kernel void qgemv_mm(
    device   T*     D  [[buffer(0)]],   // (M, N) output
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   T*     X  [[buffer(2)]],   // (M, K) activation rows
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / FMT::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * FMT::block_bytes;

    constexpr int CPL = 8;
    constexpr int LPB = FMT::block_k / CPL;
    constexpr int BPI = 32 / LPB;
    const int b_off = (int)lane / LPB;
    const int col0  = ((int)lane % LPB) * CPL;

    using T4 = metal::vec<T, 4>;
    float acc[M];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int kb = b_off; kb < bpr; kb += BPI) {
        device const uchar* base = row_base + (uint)kb * FMT::block_bytes;
        const int x_base = kb * FMT::block_k + col0;
        half w[8];
        tk_dequant8<FMT>(base, col0, w);
        // The span stays in two float4 registers across all M rows, and X
        // rides two vec4 loads per row: the load-issue rate, not weight
        // bandwidth, is what bounds this loop as M grows.
        const float4 w_lo = float4(float(w[0]), float(w[1]),
                                   float(w[2]), float(w[3]));
        const float4 w_hi = float4(float(w[4]), float(w[5]),
                                   float(w[6]), float(w[7]));
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            device const T4* xv =
                (device const T4*)(X + (long)m * K + x_base);
            acc[m] += metal::dot(w_lo, float4(xv[0]))
                    + metal::dot(w_hi, float4(xv[1]));
        }
    }
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) {
        const float r = metal::simd_sum(acc[m]);
        if (lane == 0) D[(long)m * N + row] = T(r);
    }
}

// Device-selected grouped GEMV for GGUF MoE weights. One dispatch consumes
// the complete routing table and avoids synchronizing every expert id back to
// the host. Wq is [experts, N, packed-K], X is [tokens, K], and D is emitted
// in flat [token, route, N] order for the following activation/down pass.
template<typename FMT, typename T>
kernel void qgemv_moe(
    device T *D [[buffer(0)]],
    device const uchar *Wq [[buffer(1)]],
    device const T *X [[buffer(2)]],
    device const int *topk_ids [[buffer(3)]],
    constant int &N [[buffer(4)]],
    constant int &K [[buffer(5)]],
    constant int &topk [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(tgid.x);
    const int route = int(tgid.y);
    const int token = route / topk;
    const int expert = topk_ids[route];
    if (expert < 0) {
        if (lane == 0) D[(long)route * N + row] = T(0);
        return;
    }

    const int bpr = K / FMT::block_k;
    device const uchar *row_base =
        Wq + ((long)expert * N + row) * bpr * FMT::block_bytes;
    device const T *x = X + (long)token * K;
    constexpr int CPL = 8;
    constexpr int LPB = FMT::block_k / CPL;
    constexpr int BPI = 32 / LPB;
    const int b_off = int(lane) / LPB;
    const int col0 = (int(lane) % LPB) * CPL;

    float acc = 0.0f;
    for (int kb = b_off; kb < bpr; kb += BPI) {
        device const uchar *base =
            row_base + (long)kb * FMT::block_bytes;
        const int x_base = kb * FMT::block_k + col0;
        half w[8];
        tk_dequant8<FMT>(base, col0, w);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            acc += float(w[i]) * float(x[x_base + i]);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) D[(long)route * N + row] = T(acc);
}

// Multi-row MoE GEMV for IQ2_XXS (the DSV4 routed gate|up format), shaped
// after the ds4/llama.cpp mul_mv_id kernels: NSG simdgroups per
// threadgroup, NR0 rows per simdgroup, the lane's 32-weight activation
// group loaded into registers ONCE and reused across the NR0 rows, and the
// 2 KiB sign/grid codebooks staged in threadgroup memory (divergent
// indexed loads from the constant address space serialize; threadgroup
// memory is banked). The fp32 accumulation chain applies the block scale
// per 32-group and the global 0.25 at the store, which reorders the
// reduction vs the one-simdgroup-per-row qgemv_moe — ULP-level output
// changes. Weights stay AoS (see the note at the instantiations).
template<typename T, int NSG, int NR0>
kernel void qgemv_moe_mr_iq2_xxs(
    device T *D [[buffer(0)]],              // (tokens*topk, N)
    device const uchar *Wq [[buffer(1)]],   // (E, N, K/256 * 66)
    device const T *X [[buffer(2)]],        // (tokens, K)
    device const int *topk_ids [[buffer(3)]],
    constant int &N [[buffer(4)]],
    constant int &K [[buffer(5)]],
    constant int &topk [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong svalues[256];
    threadgroup uchar ssigns[128];
    {
        const int tid = int(sgitg) * 32 + int(lane);
        for (int i = tid; i < 256; i += NSG * 32) svalues[i] = iq2xxs_grid[i];
        for (int i = tid; i < 128; i += NSG * 32) ssigns[i] = ksigns_iq2xs[i];
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    const int slot = int(tgid.y);
    const int token = slot / topk;
    const int expert = topk_ids[slot];
    const int first_row = (int(tgid.x) * NSG + int(sgitg)) * NR0;
    if (first_row >= N) return;
    device T *out = D + (long)slot * N;
    if (expert < 0) {
        if (lane == 0) {
            for (int r = first_row; r < first_row + NR0 && r < N; ++r) {
                out[r] = T(0);
            }
        }
        return;
    }

    const int bpr = K / 256;                 // 66-byte iq2_xxs superblocks
    const int nb32 = bpr * 8;                // 32-weight groups per row
    const long row_bytes = (long)bpr * 66;
    device const uchar *w_base =
        Wq + ((long)expert * N + first_row) * row_bytes;
    device const T *x = X + (long)token * K;

    float yl[32];
    float sumf[NR0];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.0f;

    for (int ib32 = int(lane); ib32 < nb32; ib32 += 32) {
        const int ibl = ib32 >> 3;           // superblock index
        const int ib = ib32 & 7;             // 32-group within superblock
        device const metal::vec<T, 4> *y4 =
            (device const metal::vec<T, 4> *)(x + 32 * ib32);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            const metal::vec<T, 4> v = y4[i];
            yl[4 * i + 0] = float(v.x);
            yl[4 * i + 1] = float(v.y);
            yl[4 * i + 2] = float(v.z);
            yl[4 * i + 3] = float(v.w);
        }
        for (short row = 0; row < NR0; ++row) {
            float sum = 0.0f;
            device const uchar *b =
                w_base + (long)ibl * 66 + (long)row * row_bytes;
            const float db = float(((device const half *)b)[0]);
            device const ushort *q2 =
                (device const ushort *)(b + 2) + 4 * ib;
            device const uchar *aux8 = (device const uchar *)q2;
            const uint aux32 = (uint)q2[2] | ((uint)q2[3] << 16);
            const float d = db * (0.5f + float(aux32 >> 28));
            #pragma clang loop unroll(full)
            for (short l = 0; l < 4; ++l) {
                const threadgroup uchar *grid =
                    (const threadgroup uchar *)(svalues + aux8[l]);
                const uchar signs = ssigns[(aux32 >> (7 * l)) & 127];
                #pragma clang loop unroll(full)
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8 * l + j] * float(grid[j]) *
                           ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
            sumf[row] += d * sum;
        }
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < NR0; ++row) {
        const int r = first_row + row;
        if (r < N) {
            const float s = metal::simd_sum(sumf[row]);
            if (lane == 0) out[r] = T(s * 0.25f);
        }
    }
}

#define instantiate_qgemv_moe_mr(name, T, NSG, NR0)                          \
   template [[host_name(name)]] [[kernel]]                                   \
   void qgemv_moe_mr_iq2_xxs<T, NSG, NR0>(                                   \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &topk [[buffer(6)]],                                  \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_moe_mr("qgemv_iq2_xxs_moe_mr", half, 2, 4)
instantiate_qgemv_moe_mr("qgemv_iq2_xxs_moe_mr_bfloat16", bf16, 2, 4)

// No iq2_xxs SoA twins: the production layout is AoS by measurement (both
// the plane split and the A100-style pairing measured slower here — see
// optimization_status 2026-08-12/13 and the note in gguf/fused_moe.py).

// Multi-row MoE GEMV for IQ2_XXS with the SwiGLU epilogue fused in: each
// simdgroup owns NPAIR gate rows AND their matching up rows (gate = rows
// [0, N/2), up = rows [N/2, N) of the combined gate|up expert tensor), so
// after the dots it can emit act = silu(clamp?(g)) * clamp?(u) directly —
// no intermediate (slots, N) tensor and no separate qc_swiglu dispatch.
// BIT-EXACT vs the two-step path by construction: the accumulators round
// to T with the same *0.25 store expression the plain kernel uses, then
// the clamp/silu chain mirrors qc_swiglu (oai_form 0) exactly.
template<typename T, int NSG, int NPAIR>
kernel void qgemv_moe_mr_iq2_xxs_swiglu(
    device T *D [[buffer(0)]],              // (tokens*topk, N/2) act output
    device const uchar *Wq [[buffer(1)]],   // (E, N, K/256 * 66)
    device const T *X [[buffer(2)]],        // (tokens, K)
    device const int *topk_ids [[buffer(3)]],
    constant int &N [[buffer(4)]],          // FULL gate|up row count
    constant int &K [[buffer(5)]],
    constant int &topk [[buffer(6)]],
    constant int &has_clamp [[buffer(7)]],
    constant float &limit [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NROWS = 2 * NPAIR;
    threadgroup ulong svalues[256];
    threadgroup uchar ssigns[128];
    {
        const int tid = int(sgitg) * 32 + int(lane);
        for (int i = tid; i < 256; i += NSG * 32) svalues[i] = iq2xxs_grid[i];
        for (int i = tid; i < 128; i += NSG * 32) ssigns[i] = ksigns_iq2xs[i];
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    const int slot = int(tgid.y);
    const int token = slot / topk;
    const int expert = topk_ids[slot];
    const int Nh = N / 2;
    const int first = (int(tgid.x) * NSG + int(sgitg)) * NPAIR;
    if (first >= Nh) return;
    device T *out = D + (long)slot * Nh;
    if (expert < 0) {
        if (lane == 0) {
            for (int j = 0; j < NPAIR && first + j < Nh; ++j) {
                out[first + j] = T(0);
            }
        }
        return;
    }

    const int bpr = K / 256;
    const int nb32 = bpr * 8;
    const long row_bytes = (long)bpr * 66;
    device const T *x = X + (long)token * K;
    // Row r of sumf: r < NPAIR -> gate row (first+r); else up row
    // (Nh+first+r-NPAIR).
    device const uchar *rbase[NROWS];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NPAIR; ++r) {
        rbase[r] = Wq + ((long)expert * N + first + r) * row_bytes;
        rbase[NPAIR + r] =
            Wq + ((long)expert * N + Nh + first + r) * row_bytes;
    }

    float yl[32];
    float sumf[NROWS];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NROWS; ++r) sumf[r] = 0.0f;

    for (int ib32 = int(lane); ib32 < nb32; ib32 += 32) {
        const int ibl = ib32 >> 3;
        const int ib = ib32 & 7;
        device const metal::vec<T, 4> *y4 =
            (device const metal::vec<T, 4> *)(x + 32 * ib32);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            const metal::vec<T, 4> v = y4[i];
            yl[4 * i + 0] = float(v.x);
            yl[4 * i + 1] = float(v.y);
            yl[4 * i + 2] = float(v.z);
            yl[4 * i + 3] = float(v.w);
        }
        #pragma clang loop unroll(full)
        for (short row = 0; row < NROWS; ++row) {
            device const uchar *b = rbase[row] + (long)ibl * 66;
            const float db = float(((device const half *)b)[0]);
            device const ushort *q2 =
                (device const ushort *)(b + 2) + 4 * ib;
            device const uchar *aux8 = (device const uchar *)q2;
            const uint aux32 = (uint)q2[2] | ((uint)q2[3] << 16);
            const float d = db * (0.5f + float(aux32 >> 28));
            float sum = 0.0f;
            #pragma clang loop unroll(full)
            for (short l = 0; l < 4; ++l) {
                const threadgroup uchar *grid =
                    (const threadgroup uchar *)(svalues + aux8[l]);
                const uchar signs = ssigns[(aux32 >> (7 * l)) & 127];
                #pragma clang loop unroll(full)
                for (short j = 0; j < 8; ++j) {
                    sum += yl[8 * l + j] * float(grid[j]) *
                           ((signs & kmask_iq2xs[j]) ? -1.0f : 1.0f);
                }
            }
            sumf[row] += d * sum;
        }
    }

    float sums[NROWS];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NROWS; ++r) sums[r] = metal::simd_sum(sumf[r]);
    if (lane == 0) {
        #pragma clang fp reassociate(off) contract(off)
        for (short j = 0; j < NPAIR; ++j) {
            if (first + j >= Nh) break;
            // Same store rounding as the plain kernel (T(s*0.25)), then
            // the exact qc_swiglu oai_form-0 chain.
            T g = T(sums[j] * 0.25f);
            T u = T(sums[NPAIR + j] * 0.25f);
            if (has_clamp) {
                const T lim = T(limit);
                const T nlim = T(-limit);
                g = (g > lim) ? lim : g;
                u = (u > lim) ? lim : ((u < nlim) ? nlim : u);
            }
            const T s = T(metal::precise::divide(
                float(g), 1.0f + metal::precise::exp(-float(g))));
            out[first + j] = s * u;
        }
    }
}

#define instantiate_qgemv_moe_mr_swiglu(name, T, NSG, NPAIR)                 \
   template [[host_name(name)]] [[kernel]]                                   \
   void qgemv_moe_mr_iq2_xxs_swiglu<T, NSG, NPAIR>(                          \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &topk [[buffer(6)]],                                  \
     const constant int &has_clamp [[buffer(7)]],                             \
     const constant float &limit [[buffer(8)]],                               \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_moe_mr_swiglu("qgemv_iq2_xxs_moe_mr_swiglu", half, 2, 2)
instantiate_qgemv_moe_mr_swiglu("qgemv_iq2_xxs_moe_mr_swiglu_bfloat16", bf16, 2, 2)

// No expert-grouped swiglu twin: measured negative, 123 -> 230 ms/step
// (optimization_status 2026-08-13, expert-grouped w13 entry).

// Multi-row MoE GEMV for Q2_K (the DSV4 routed down-projection format),
// same NSG x NR0 shape as the iq2_xxs twin. Ported from the ds4/llama.cpp
// mul_mv q2_K walk: lane = (ix, iq, ir) covers 4 superblocks in flight,
// the 2-bit extraction folds into the FMA as integer masks (0x0003 low
// byte / 0x0300 high byte with the 1/256 correction, per-quarter 1/4,
// 1/16, 1/64 post-scales), and dmin applies to per-quarter activation
// sums. fp32 chain; no codebooks, so no threadgroup staging. Reduction
// order differs from qgemv_moe -> ULP-level output changes. SOA=true reads
// the load-time repacked per-expert planes
// [N*nb x 64B qs | N*nb x 16B scales | N*nb x 4B (d,dmin)]
// (byte-neutral): aligned 8-byte qs/scale words and one packed half2
// (d,dmin) load replace the ~10 narrow unaligned loads of the 84-byte AoS
// stride. Same bytes, same arithmetic order -> bit-exact.
template<typename T, int NSG, int NR0, bool SOA = false>
kernel void qgemv_moe_mr_q2_K(
    device T *D [[buffer(0)]],              // (tokens*topk, N)
    device const uchar *Wq [[buffer(1)]],   // (E, N, K/256 * 84)
    device const T *X [[buffer(2)]],        // (tokens, K)
    device const int *topk_ids [[buffer(3)]],
    constant int &N [[buffer(4)]],
    constant int &K [[buffer(5)]],
    constant int &topk [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const int slot = int(tgid.y);
    const int token = slot / topk;
    const int expert = topk_ids[slot];
    const int first_row = (int(tgid.x) * NSG + int(sgitg)) * NR0;
    if (first_row >= N) return;
    device T *out = D + (long)slot * N;
    if (expert < 0) {
        if (lane == 0) {
            for (int r = first_row; r < first_row + NR0 && r < N; ++r) {
                out[r] = T(0);
            }
        }
        return;
    }

    const int nb = K / 256;                  // 84-byte q2_K superblocks
    const long row_bytes = (long)nb * 84;
    device const uchar *w_base = nullptr;
    device const uchar *qs_base = nullptr;
    device const uchar *sc_base = nullptr;
    device const uchar *dm_base = nullptr;
    if constexpr (SOA) {
        device const uchar *ebase =
            Wq + (long)expert * ((long)N * nb * 84);
        qs_base = ebase + (long)first_row * nb * 64;
        sc_base = ebase + (long)N * nb * 64 + (long)first_row * nb * 16;
        dm_base = ebase + (long)N * nb * 80 + (long)first_row * nb * 4;
    } else {
        w_base = Wq + ((long)expert * N + first_row) * row_bytes;
    }
    device const T *x = X + (long)token * K;

    float yl[32];
    float sumf[NR0];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NR0; ++r) sumf[r] = 0.0f;

    const short ix = short(lane) / 8;        // 0..3: superblock in flight
    const short it = short(lane) % 8;
    const short iq = it / 4;                 // 0/1: which 128-weight half
    const short ir = it % 4;                 // 0..3: 8-span within quarter
    const short is = (8 * ir) / 16;          // 0/1: scale byte parity

    device const T *y4 = x + ix * 256 + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = float(y4[i + 0]);  sumy[0] += yl[i + 0];
            yl[i + 8] = float(y4[i + 32]); sumy[1] += yl[i + 8];
            yl[i + 16] = float(y4[i + 64]); sumy[2] += yl[i + 16];
            yl[i + 24] = float(y4[i + 96]); sumy[3] += yl[i + 24];
        }

        if constexpr (SOA) {
            // The SoA and AoS branches MUST keep byte-identical load types
            // and expression text (only the plane bases/strides differ):
            // the q2_K dall/dmin expression gives the compiler
            // fp-contraction freedom, and a divergent load shape flips
            // one-ULP contraction choices vs the AoS binary. The SoA gain
            // is plane locality, not load merging.
            device const uchar *sc = sc_base + (long)ib * 16 + 8 * iq + is;
            device const ushort *qs =
                (device const ushort *)(qs_base + (long)ib * 64) +
                16 * iq + 4 * ir;
            device const half *dh =
                (device const half *)(dm_base + (long)ib * 4);

            for (short row = 0; row < NR0; ++row) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                #pragma clang loop unroll(full)
                for (short i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i + 0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i + 1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i + 8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i + 9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }
                const float dall = float(dh[0]);
                const float dmin = float(dh[1]) * (1.f / 16.f);
                sumf[row] +=
                    dall * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f / 1.f) +
                            (acc1[1] + (1.f / 256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f / 4.f) +
                            (acc1[2] + (1.f / 256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f / 16.f) +
                            (acc1[3] + (1.f / 256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f / 64.f)) -
                    dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                            sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += nb * 32;
                sc += nb * 16;
                dh += nb * 2;
            }
        } else {
            device const uchar *b = w_base + (long)ib * 84;
            device const uchar *sc = b + 8 * iq + is;
            device const ushort *qs = (device const ushort *)(b + 16) + 16 * iq + 4 * ir;
            device const half *dh = (device const half *)(b + 80);

            for (short row = 0; row < NR0; ++row) {
                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                #pragma clang loop unroll(full)
                for (short i = 0; i < 8; i += 2) {
                    acc1[0] += yl[i + 0] * (qs[i / 2] & 0x0003);
                    acc2[0] += yl[i + 1] * (qs[i / 2] & 0x0300);
                    acc1[1] += yl[i + 8] * (qs[i / 2] & 0x000c);
                    acc2[1] += yl[i + 9] * (qs[i / 2] & 0x0c00);
                    acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                    acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                    acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                    acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                }
                const float dall = float(dh[0]);
                const float dmin = float(dh[1]) * (1.f / 16.f);
                sumf[row] +=
                    dall * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f / 1.f) +
                            (acc1[1] + (1.f / 256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f / 4.f) +
                            (acc1[2] + (1.f / 256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f / 16.f) +
                            (acc1[3] + (1.f / 256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f / 64.f)) -
                    dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                            sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                qs += row_bytes / 2;
                sc += row_bytes;
                dh += row_bytes / 2;
            }
        }

        y4 += 4 * 256;
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < NR0; ++row) {
        const int r = first_row + row;
        if (r < N) {
            const float s = metal::simd_sum(sumf[row]);
            if (lane == 0) out[r] = T(s);
        }
    }
}

#define instantiate_qgemv_moe_mr_q2k(name, T, NSG, NR0)                      \
   template [[host_name(name)]] [[kernel]]                                   \
   void qgemv_moe_mr_q2_K<T, NSG, NR0>(                                      \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &topk [[buffer(6)]],                                  \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

// fp16 uses the 4x8 geometry, bf16 the 2x4 (see launch_qgemv_moe_mr); all
// geometries are bit-exact, only these two are instantiated.
instantiate_qgemv_moe_mr_q2k("qgemv_q2_K_moe_mr_bfloat16", bf16, 2, 4)
instantiate_qgemv_moe_mr_q2k("qgemv_q2_K_moe_mr_g48", half, 4, 8)

#define instantiate_qgemv_moe_mr_q2k_soa(name, T, NSG, NR0)                   \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_moe_mr_q2_K<T, NSG, NR0, true>(                                 \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &topk [[buffer(6)]],                                  \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_moe_mr_q2k_soa("qgemv_q2_K_moe_mr_soa_bfloat16", bf16, 2, 4)
instantiate_qgemv_moe_mr_q2k_soa("qgemv_q2_K_moe_mr_g48_soa", half, 4, 8)

// Sum-folded Q2_K MoE GEMV: one threadgroup column per TOKEN (not slot).
// Loops the token's topk expert slots; per slot runs the per-row dot walk
// with the same text, types, and lane mapping as qgemv_moe_mr_q2_K on the
// slot's activation row, rounds each expert's simd_sum result to T (the
// rounding the (tokens*topk, N) intermediate store used to apply, T(0)
// for expert < 0), then applies qc_moe_weighted_sum's exact sequential
// fp32 reduce (reassociate/contract off, slot-ascending) and stores
// T(acc). Collapses down-GEMV + intermediate round-trip + weighted-sum
// dispatch into one kernel. Bit-exactness vs the unfused chain is
// oracle-gated: identical walk text, but the enclosing slot loop can in
// principle move fp-contraction choices (see the SoA load-shape lesson).
template<typename T, int NSG, int NR0, bool SOA = false>
kernel void qgemv_moe_mr_q2_K_sum(
    device T *D [[buffer(0)]],                // (tokens, N)
    device const uchar *Wq [[buffer(1)]],     // (E, N, K/256 * 84)
    device const T *X [[buffer(2)]],          // (tokens*topk, K)
    device const int *topk_ids [[buffer(3)]], // (tokens, topk)
    device const float *topk_w [[buffer(4)]], // (tokens, topk) fp32
    constant int &N [[buffer(5)]],
    constant int &K [[buffer(6)]],
    constant int &topk [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
    const int token = int(tgid.y);
    const int first_row = (int(tgid.x) * NSG + int(sgitg)) * NR0;
    if (first_row >= N) return;
    device T *out = D + (long)token * N;

    const int nb = K / 256;                  // 84-byte q2_K superblocks
    const long row_bytes = (long)nb * 84;

    const short ix = short(lane) / 8;        // 0..3: superblock in flight
    const short it = short(lane) % 8;
    const short iq = it / 4;                 // 0/1: which 128-weight half
    const short ir = it % 4;                 // 0..3: 8-span within quarter
    const short is = (8 * ir) / 16;          // 0/1: scale byte parity

    float acc[NR0];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NR0; ++r) acc[r] = 0.0f;

    for (int k = 0; k < topk; ++k) {
        const int slot = token * topk + k;
        const int expert = topk_ids[slot];
        const float wk = topk_w[slot];
        float hk[NR0];
        if (expert < 0) {
            #pragma clang loop unroll(full)
            for (short r = 0; r < NR0; ++r) hk[r] = 0.0f;
        } else {
            device const uchar *w_base = nullptr;
            device const uchar *qs_base = nullptr;
            device const uchar *sc_base = nullptr;
            device const uchar *dm_base = nullptr;
            if constexpr (SOA) {
                device const uchar *ebase =
                    Wq + (long)expert * ((long)N * nb * 84);
                qs_base = ebase + (long)first_row * nb * 64;
                sc_base = ebase + (long)N * nb * 64 + (long)first_row * nb * 16;
                dm_base = ebase + (long)N * nb * 80 + (long)first_row * nb * 4;
            } else {
                w_base = Wq + ((long)expert * N + first_row) * row_bytes;
            }
            device const T *x = X + (long)slot * K;

            float yl[32];
            float sumf[NR0];
            #pragma clang loop unroll(full)
            for (short r = 0; r < NR0; ++r) sumf[r] = 0.0f;

            device const T *y4 = x + ix * 256 + 128 * iq + 8 * ir;

            for (int ib = ix; ib < nb; ib += 4) {
                float4 sumy = {0.f, 0.f, 0.f, 0.f};
                #pragma clang loop unroll(full)
                for (short i = 0; i < 8; ++i) {
                    yl[i + 0] = float(y4[i + 0]);  sumy[0] += yl[i + 0];
                    yl[i + 8] = float(y4[i + 32]); sumy[1] += yl[i + 8];
                    yl[i + 16] = float(y4[i + 64]); sumy[2] += yl[i + 16];
                    yl[i + 24] = float(y4[i + 96]); sumy[3] += yl[i + 24];
                }

                if constexpr (SOA) {
                    device const uchar *sc = sc_base + (long)ib * 16 + 8 * iq + is;
                    device const ushort *qs =
                        (device const ushort *)(qs_base + (long)ib * 64) +
                        16 * iq + 4 * ir;
                    device const half *dh =
                        (device const half *)(dm_base + (long)ib * 4);

                    for (short row = 0; row < NR0; ++row) {
                        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                        #pragma clang loop unroll(full)
                        for (short i = 0; i < 8; i += 2) {
                            acc1[0] += yl[i + 0] * (qs[i / 2] & 0x0003);
                            acc2[0] += yl[i + 1] * (qs[i / 2] & 0x0300);
                            acc1[1] += yl[i + 8] * (qs[i / 2] & 0x000c);
                            acc2[1] += yl[i + 9] * (qs[i / 2] & 0x0c00);
                            acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                            acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                            acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                            acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                        }
                        const float dall = float(dh[0]);
                        const float dmin = float(dh[1]) * (1.f / 16.f);
                        sumf[row] +=
                            dall * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f / 1.f) +
                                    (acc1[1] + (1.f / 256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f / 4.f) +
                                    (acc1[2] + (1.f / 256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f / 16.f) +
                                    (acc1[3] + (1.f / 256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f / 64.f)) -
                            dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                    sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                        qs += nb * 32;
                        sc += nb * 16;
                        dh += nb * 2;
                    }
                } else {
                    device const uchar *b = w_base + (long)ib * 84;
                    device const uchar *sc = b + 8 * iq + is;
                    device const ushort *qs = (device const ushort *)(b + 16) + 16 * iq + 4 * ir;
                    device const half *dh = (device const half *)(b + 80);

                    for (short row = 0; row < NR0; ++row) {
                        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                        #pragma clang loop unroll(full)
                        for (short i = 0; i < 8; i += 2) {
                            acc1[0] += yl[i + 0] * (qs[i / 2] & 0x0003);
                            acc2[0] += yl[i + 1] * (qs[i / 2] & 0x0300);
                            acc1[1] += yl[i + 8] * (qs[i / 2] & 0x000c);
                            acc2[1] += yl[i + 9] * (qs[i / 2] & 0x0c00);
                            acc1[2] += yl[i + 16] * (qs[i / 2] & 0x0030);
                            acc2[2] += yl[i + 17] * (qs[i / 2] & 0x3000);
                            acc1[3] += yl[i + 24] * (qs[i / 2] & 0x00c0);
                            acc2[3] += yl[i + 25] * (qs[i / 2] & 0xc000);
                        }
                        const float dall = float(dh[0]);
                        const float dmin = float(dh[1]) * (1.f / 16.f);
                        sumf[row] +=
                            dall * ((acc1[0] + (1.f / 256.f) * acc2[0]) * (sc[0] & 0xF) * (1.f / 1.f) +
                                    (acc1[1] + (1.f / 256.f) * acc2[1]) * (sc[2] & 0xF) * (1.f / 4.f) +
                                    (acc1[2] + (1.f / 256.f) * acc2[2]) * (sc[4] & 0xF) * (1.f / 16.f) +
                                    (acc1[3] + (1.f / 256.f) * acc2[3]) * (sc[6] & 0xF) * (1.f / 64.f)) -
                            dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) +
                                    sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

                        qs += row_bytes / 2;
                        sc += row_bytes;
                        dh += row_bytes / 2;
                    }
                }

                y4 += 4 * 256;
            }

            #pragma clang loop unroll(full)
            for (short row = 0; row < NR0; ++row) {
                const int r = first_row + row;
                if (r < N) {
                    const float s = metal::simd_sum(sumf[row]);
                    hk[row] = float(T(s));
                } else {
                    hk[row] = 0.0f;
                }
            }
        }

        {
            #pragma clang fp reassociate(off) contract(off)
            if (k == 0) {
                #pragma clang loop unroll(full)
                for (short r = 0; r < NR0; ++r) acc[r] = hk[r] * wk;
            } else {
                #pragma clang loop unroll(full)
                for (short r = 0; r < NR0; ++r) {
                    const float p = hk[r] * wk;
                    acc[r] += p;
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (short row = 0; row < NR0; ++row) {
        const int r = first_row + row;
        if (r < N && lane == 0) out[r] = T(acc[row]);
    }
}

#define instantiate_qgemv_moe_mr_q2k_sum(name, T, NSG, NR0)                   \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_moe_mr_q2_K_sum<T, NSG, NR0>(                                   \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     device const float* topk_w [[buffer(4)]],                                \
     const constant int &N [[buffer(5)]], const constant int &K [[buffer(6)]], \
     const constant int &topk [[buffer(7)]],                                  \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_moe_mr_q2k_sum("qgemv_q2_K_moe_mr_sum_bfloat16", bf16, 2, 4)
instantiate_qgemv_moe_mr_q2k_sum("qgemv_q2_K_moe_mr_sum_g48", half, 4, 8)

#define instantiate_qgemv_moe_mr_q2k_sum_soa(name, T, NSG, NR0)               \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_moe_mr_q2_K_sum<T, NSG, NR0, true>(                             \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     device const float* topk_w [[buffer(4)]],                                \
     const constant int &N [[buffer(5)]], const constant int &K [[buffer(6)]], \
     const constant int &topk [[buffer(7)]],                                  \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_moe_mr_q2k_sum_soa("qgemv_q2_K_moe_mr_sum_soa_bfloat16", bf16, 2, 4)
instantiate_qgemv_moe_mr_q2k_sum_soa("qgemv_q2_K_moe_mr_sum_g48_soa", half, 4, 8)

// Multi-batch weight-stationary GEMV for the 2..8-row decode/verify widths.
// One simdgroup per output row decodes each weight block ONCE and accumulates
// into M register accumulators, so weight traffic stays at the batch-1 cost
// instead of scaling linearly with M (the host used to loop the batch-1
// kernel per row). M is a template parameter: a runtime-M accumulator array
// gets runtime-indexed and spilled off registers. The per-row walk and FMA
// order are IDENTICAL to qgemv, so each output
// row is bit-identical to the looped batch-1 launch.
template<typename FMT, typename T, int M>
kernel void qgemv_mb(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   uchar* Wq [[buffer(1)]],   // (N, K/block_k) packed weight blocks
    device   T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / FMT::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * FMT::block_bytes;

    constexpr int CPL = 8;
    constexpr int LPB = FMT::block_k / CPL;
    constexpr int BPI = 32 / LPB;
    const int b_off = (int)lane / LPB;
    const int col0  = ((int)lane % LPB) * CPL;

    float acc[M];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    for (int kb = b_off; kb < bpr; kb += BPI) {
        device const uchar* base = row_base + (uint)kb * FMT::block_bytes;
        const int x_base = kb * FMT::block_k + col0;
        half w[8];
        tk_dequant8<FMT>(base, col0, w);
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            device const T* xm = X + (long)m * K + x_base;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) acc[m] += float(w[i]) * float(xm[i]);
        }
    }
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) {
        const float s = metal::simd_sum(acc[m]);
        if (lane == 0) D[(long)m * N + row] = T(s);
    }
}

[[host_name("qgemv_q8_0")]]
kernel void qgemv_q8_0_fast(
    device   half*  D  [[buffer(0)]],
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / q8_0::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * q8_0::block_bytes;

    const int block_offset = (int)(lane >> 2);       // 8 q8_0 blocks per simdgroup iteration
    const int chunk = (int)(lane & 3);               // 8 contiguous int8 values within the block

    float acc = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 8) {
        device const uchar* block = row_base + (uint)kb * q8_0::block_bytes;
        const float d = float(((device const half*)block)[0]);
        device const char* qs = (device const char*)(block + 2 + chunk * 8);
        const int x0 = (kb << 5) + chunk * 8;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            acc += d * float(qs[i]) * float(X[x0 + i]);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) D[row] = half(acc);
}

[[host_name("qgemv_q4_0")]]
kernel void qgemv_q4_0_fast(
    device   half*  D  [[buffer(0)]],
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / q4_0::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * q4_0::block_bytes;

    const int block_offset = (int)(lane >> 1);       // 16 q4_0 blocks per simdgroup iteration
    const int byte_start = (int)(lane & 1) * 8;      // each lane handles 8 packed bytes = 16 weights

    float acc = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 16) {
        device const uchar* block = row_base + (uint)kb * q4_0::block_bytes;
        const float d = float(((device const half*)block)[0]);
        device const uchar* qs = block + 2 + byte_start;
        const int x0 = (kb << 5) + byte_start;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = qs[i];
            acc += d * float((int)(packed & 0x0F) - 8) * float(X[x0 + i]);
            acc += d * float((int)(packed >> 4) - 8) * float(X[x0 + i + 16]);
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) D[row] = half(acc);
}

// Compressed-tensors FP8-per-channel W8A16 GEMV (Qwen3.8 NVFP4-checkpoint
// FP8 side: attn qkvo, GDN qkv/z/out, lm_head, mlp 56-63). Unlike every GGUF
// kernel above, the weight is NOT a block format: planar row-major e4m3
// bytes (N, K) straight from the checkpoint, with one float scale per output
// row in a separate buffer. Geometry is qgemv_q4k_nr's (2 simdgroups x 2
// rows, dispatch (N/4, 1)): each lane owns 16 contiguous bytes per row per
// iteration (one uint4 load), X is staged to registers once and shared
// across both rows, accumulation fp32, and the per-row scale is applied
// once in the epilogue. Decode is the select-free v6 bit-pattern form
// (the nvfp4_planar E2M1 lesson ported to E4M3): per uint of 4 bytes, two
// half2 patterns — even bytes ((w<<7)&0x3F803F80)|((w<<8)&0x80008000),
// odd bytes ((w>>1)&0x3F803F80)|(w&0x80008000) — drop the e4m3 exp/mant
// into the half field positions, giving exactly value/2^8 (bias 7 vs 15),
// subnormals included: float(as_type<half>(..)) is a convert, not
// arithmetic, so offline-compile FTZ cannot flush it. The 2^8 rebias is
// folded into the per-row scale epilogue; power-of-two scaling commutes
// with IEEE rounding at every FMA and the epilogue rounds once from the
// same real product, so outputs are bit-identical to the tk_e4m3_decode
// form (NaN codes 0x7F/0xFF decode to +-480 in both; checkpoint has
// none). Host guards: N % 4 == 0, K % 16 == 0, contiguous row-major
// weight.
template<typename T>
kernel void qgemv_fp8ch(
    device   T*     D  [[buffer(0)]],   // (N, 1) output
    device   const uchar* Wq [[buffer(1)]],   // (N, K) e4m3 bytes, row-major
    device   const T*     X  [[buffer(2)]],   // (K, 1) activation vector
    device   const float* WS [[buffer(3)]],   // (N,) per-channel scales
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 2;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    device const uchar* w0 = Wq + (ulong)first_row * (ulong)K;
    device const uchar* w1 = w0 + (ulong)K;
    float sumf[NR] = {0.f, 0.f};
    for (int c = int(lane) * 16; c + 16 <= K; c += 32 * 16) {
        float xs[16];
        device const metal::vec<T, 4>* xp =
            (device const metal::vec<T, 4>*)(X + c);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 4; ++i) {
            const metal::vec<T, 4> xv = xp[i];
            xs[4 * i + 0] = float(xv.x);
            xs[4 * i + 1] = float(xv.y);
            xs[4 * i + 2] = float(xv.z);
            xs[4 * i + 3] = float(xv.w);
        }
        const uint4 a = *(device const uint4*)(w0 + c);
        const uint4 b = *(device const uint4*)(w1 + c);
        float r0 = 0.f, r1 = 0.f;
        #pragma clang loop unroll(full)
        for (short u = 0; u < 4; ++u) {
            const uint av = a[u];
            const uint bv = b[u];
            // bytes j=0,2 in .x/.y of the even pair; j=1,3 in the odd pair
            const half2 ae = as_type<half2>(((av << 7) & 0x3F803F80u) |
                                            ((av << 8) & 0x80008000u));
            const half2 ao = as_type<half2>(((av >> 1) & 0x3F803F80u) |
                                            (av & 0x80008000u));
            const half2 be = as_type<half2>(((bv << 7) & 0x3F803F80u) |
                                            ((bv << 8) & 0x80008000u));
            const half2 bo = as_type<half2>(((bv >> 1) & 0x3F803F80u) |
                                            (bv & 0x80008000u));
            r0 += xs[4 * u + 0] * float(ae.x);
            r1 += xs[4 * u + 0] * float(be.x);
            r0 += xs[4 * u + 1] * float(ao.x);
            r1 += xs[4 * u + 1] * float(bo.x);
            r0 += xs[4 * u + 2] * float(ae.y);
            r1 += xs[4 * u + 2] * float(be.y);
            r0 += xs[4 * u + 3] * float(ao.y);
            r1 += xs[4 * u + 3] * float(bo.y);
        }
        sumf[0] += r0;
        sumf[1] += r1;
    }
    #pragma clang loop unroll(full)
    for (short row = 0; row < NR; ++row) {
        const float tot = metal::simd_sum(sumf[row]);
        // 256 = the folded 2^8 decode rebias (exact: power-of-two scale)
        if (lane == 0)
            D[first_row + row] = T(tot * (256.0f * WS[first_row + row]));
    }
}

// Weight-stationary column-pair batch twin of qgemv_fp8ch (the q4_K
// qgemv_q4k_nr_mb pattern): grid.y indexes COLUMN PAIRS (dispatch
// (N/4, M/2)), each threadgroup runs the batch-1 lane geometry over two
// activation columns with the e4m3 weight bytes loaded and decoded ONCE
// per iteration. The per-(row, column) FP chain matches the batch-1
// kernel, so every output row is bit-identical to a looped qgemv_fp8ch
// launch. Threadgroups with equal tgid.x and different tgid.y are
// resident together, so the nominal M/2 weight re-reads land in cache.
// Host guards: batch-1's plus M even and X/D contiguous row-major
// (M, K) / (M, N).
template<typename T>
kernel void qgemv_fp8ch_mb(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   const uchar* Wq [[buffer(1)]],   // (N, K) e4m3 bytes, row-major
    device   const T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    device   const float* WS [[buffer(3)]],   // (N,) per-channel scales
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 2;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const int first_col = int(tgid.y) * 2;
    device const uchar* w0 = Wq + (ulong)first_row * (ulong)K;
    device const uchar* w1 = w0 + (ulong)K;
    float sumf[NR][2] = {};
    for (int c = int(lane) * 16; c + 16 <= K; c += 32 * 16) {
        float xs[2][16];
        #pragma clang loop unroll(full)
        for (short mi = 0; mi < 2; ++mi) {
            device const metal::vec<T, 4>* xp = (device const metal::vec<T, 4>*)
                (X + (long)(first_col + mi) * K + c);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const metal::vec<T, 4> xv = xp[i];
                xs[mi][4 * i + 0] = float(xv.x);
                xs[mi][4 * i + 1] = float(xv.y);
                xs[mi][4 * i + 2] = float(xv.z);
                xs[mi][4 * i + 3] = float(xv.w);
            }
        }
        const uint4 a = *(device const uint4*)(w0 + c);
        const uint4 b = *(device const uint4*)(w1 + c);
        float r[NR][2] = {};
        #pragma clang loop unroll(full)
        for (short u = 0; u < 4; ++u) {
            const uint av = a[u];
            const uint bv = b[u];
            // select-free decode, identical to batch-1 (bit-identity contract)
            const half2 ae = as_type<half2>(((av << 7) & 0x3F803F80u) |
                                            ((av << 8) & 0x80008000u));
            const half2 ao = as_type<half2>(((av >> 1) & 0x3F803F80u) |
                                            (av & 0x80008000u));
            const half2 be = as_type<half2>(((bv << 7) & 0x3F803F80u) |
                                            ((bv << 8) & 0x80008000u));
            const half2 bo = as_type<half2>(((bv >> 1) & 0x3F803F80u) |
                                            (bv & 0x80008000u));
            const float w0j[4] = {float(ae.x), float(ao.x),
                                  float(ae.y), float(ao.y)};
            const float w1j[4] = {float(be.x), float(bo.x),
                                  float(be.y), float(bo.y)};
            #pragma clang loop unroll(full)
            for (short j = 0; j < 4; ++j) {
                #pragma clang loop unroll(full)
                for (short mi = 0; mi < 2; ++mi) {
                    r[0][mi] += xs[mi][4 * u + j] * w0j[j];
                    r[1][mi] += xs[mi][4 * u + j] * w1j[j];
                }
            }
        }
        #pragma clang loop unroll(full)
        for (short row = 0; row < NR; ++row)
            #pragma clang loop unroll(full)
            for (short mi = 0; mi < 2; ++mi) sumf[row][mi] += r[row][mi];
    }
    #pragma clang loop unroll(full)
    for (short mi = 0; mi < 2; ++mi)
        #pragma clang loop unroll(full)
        for (short row = 0; row < NR; ++row) {
            const float tot = metal::simd_sum(sumf[row][mi]);
            // 256 = the folded 2^8 decode rebias (exact: power-of-two scale)
            if (lane == 0)
                D[(long)(first_col + mi) * N + first_row + row] =
                    T(tot * (256.0f * WS[first_row + row]));
        }
}

// Compressed-tensors NVFP4 W4A16 GEMV over the checkpoint's PLANAR layout
// (Qwen3.8 NVFP4 side: mlp gate/up/down, layers 0-55). Three separate
// buffers, no repack: weight_packed (N, K/2) e2m1 nibble pairs (low nibble
// = even column), weight_scale (N, K/16) raw e4m3 bytes, and the fp32
// per-tensor global multiplier. NOT the interleaved 9-byte `nvfp4` struct
// the generic qgemv template reads. Structure per the ggml-mxfp4 x MLX
// fp_qmv precedent: each lane owns one 16-value group per iteration (one
// 8-byte load), unscaled FMA tree within the group, ONE scale multiply per
// group hoisted onto the partial sum, fp32 cross-group accumulation, and
// the global scale applied once per output element. Row geometry matches
// qgemv_q4k_nr (2 simdgroups x 2 rows, dispatch (N/4, 1)); X staged to
// registers once per iteration and shared across both rows. Host guards:
// N % 4 == 0, K % 16 == 0, contiguous row-major buffers.
template<typename T>
kernel void qgemv_nvfp4_planar(
    device   T*     D  [[buffer(0)]],   // (N, 1) output
    device   const uchar* Wq [[buffer(1)]],   // (N, K/2) packed e2m1
    device   const T*     X  [[buffer(2)]],   // (K, 1) activation vector
    device   const uchar* WS [[buffer(3)]],   // (N, K/16) e4m3 group scales
    device   const float* GS [[buffer(4)]],   // (1,) global multiplier
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 2;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const int groups = K / 16;
    const ulong wrow = (ulong)K / 2;
    device const uchar* w0 = Wq + (ulong)first_row * wrow;
    device const uchar* s0 = WS + (ulong)first_row * (ulong)groups;
    // Select-free E2M1 decode, half2-vectorized (UPDATE 18 pattern): four
    // half2 bit-pattern constructs per uint decode all 8 nibbles with no
    // byte extraction — [ee][m] lands at half bits 11..9, sign at 15, and
    // the CONVERT to fp32 is exact on subnormal halfs (no arithmetic, so
    // offline-compile FTZ cannot flush the 0.5 code). ~2.5 int ops + 1
    // convert + 1 FMA per value. The uniform 2^14 rebias and the group
    // scale's own 2^8 e4m3 rebias fold into one 2^22 constant on the
    // per-group multiply (exact: powers of two).
    // Variants measured at the MLP shapes and rejected: tk_e2m1_decode
    // select chain 260/323 GB/s (ALU-bound); data-dependent simd_shuffle
    // register LUT and dynamic constant-table gathers both ~135 (dynamic
    // indexing serializes on M1); two-groups-per-lane uint4 loads with
    // vec4 X staging ~135 (unrolled-body register pressure, the UPDATE 11
    // i-cache/occupancy class); per-byte scalar nibble constructs 529/452
    // (byte extraction + 2 scalar constructs ~5.5 ops/value — UPDATE 19).
    float sumf[NR] = {0.f, 0.f};
    for (int g = int(lane); g < groups; g += 32) {
        float xs[16];
        device const metal::vec<T, 4>* xp4 =
            (device const metal::vec<T, 4>*)(X + g * 16);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 4; ++i) {
            const metal::vec<T, 4> x4 = xp4[i];
            xs[4 * i + 0] = float(x4[0]);
            xs[4 * i + 1] = float(x4[1]);
            xs[4 * i + 2] = float(x4[2]);
            xs[4 * i + 3] = float(x4[3]);
        }
        #pragma clang loop unroll(full)
        for (short r = 0; r < NR; ++r) {
            const uint2 p = *(device const uint2*)(w0 + (ulong)r * wrow + g * 8);
            float gsum = 0.f;
            #pragma clang loop unroll(full)
            for (short u = 0; u < 2; ++u) {
                const uint v = (u == 0) ? p.x : p.y;
                const short col = 8 * u;
                // Four half2s decode all 8 nibbles of the uint (no byte
                // extraction): lanes are (byte0, byte2) / (byte1, byte3).
                const half2 le = as_type<half2>(((v << 9) & 0x0E000E00u) |
                                                ((v << 12) & 0x80008000u));
                const half2 lo = as_type<half2>(((v << 1) & 0x0E000E00u) |
                                                ((v << 4) & 0x80008000u));
                const half2 he = as_type<half2>(((v << 5) & 0x0E000E00u) |
                                                ((v << 8) & 0x80008000u));
                const half2 ho = as_type<half2>(((v >> 3) & 0x0E000E00u) |
                                                (v & 0x80008000u));
                gsum += xs[col + 0] * float(le.x);
                gsum += xs[col + 1] * float(he.x);
                gsum += xs[col + 2] * float(lo.x);
                gsum += xs[col + 3] * float(ho.x);
                gsum += xs[col + 4] * float(le.y);
                gsum += xs[col + 5] * float(he.y);
                gsum += xs[col + 6] * float(lo.y);
                gsum += xs[col + 7] * float(ho.y);
            }
            // Group scale via the UPDATE 18 select-free e4m3 pattern;
            // 2^22 = 2^14 (E2M1 rebias) * 2^8 (E4M3 rebias), exact.
            const uint sb = s0[(ulong)r * groups + g];
            const half sh = as_type<half>(ushort(((sb & 0x7F) << 7) |
                                                 ((sb & 0x80) << 8)));
            sumf[r] += gsum * (4194304.0f * float(sh));
        }
    }
    #pragma clang loop unroll(full)
    for (short row = 0; row < NR; ++row) {
        const float tot = metal::simd_sum(sumf[row]);
        if (lane == 0) D[first_row + row] = T(tot * GS[0]);
    }
}

// Weight-stationary column-pair batch twin of qgemv_nvfp4_planar (same
// grid.y column-pair pattern as qgemv_fp8ch_mb / qgemv_q4k_nr_mb): the
// packed nibbles and the per-group e4m3 scale are decoded ONCE per
// (row, group) and applied to both activation columns from registers.
// The per-(row, column) FP chain matches the batch-1 kernel, so every
// output row is bit-identical to a looped qgemv_nvfp4_planar launch.
// Host guards: batch-1's plus M even and X/D contiguous row-major
// (M, K) / (M, N).
template<typename T>
kernel void qgemv_nvfp4_planar_mb(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   const uchar* Wq [[buffer(1)]],   // (N, K/2) packed e2m1
    device   const T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    device   const uchar* WS [[buffer(3)]],   // (N, K/16) e4m3 group scales
    device   const float* GS [[buffer(4)]],   // (1,) global multiplier
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 2;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const int first_col = int(tgid.y) * 2;
    const int groups = K / 16;
    const ulong wrow = (ulong)K / 2;
    device const uchar* w0 = Wq + (ulong)first_row * wrow;
    device const uchar* s0 = WS + (ulong)first_row * (ulong)groups;
    float sumf[NR][2] = {};
    for (int g = int(lane); g < groups; g += 32) {
        float xs[2][16];
        #pragma clang loop unroll(full)
        for (short mi = 0; mi < 2; ++mi) {
            device const metal::vec<T, 4>* xp4 = (device const metal::vec<T, 4>*)(
                X + (long)(first_col + mi) * K + g * 16);
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const metal::vec<T, 4> x4 = xp4[i];
                xs[mi][4 * i + 0] = float(x4[0]);
                xs[mi][4 * i + 1] = float(x4[1]);
                xs[mi][4 * i + 2] = float(x4[2]);
                xs[mi][4 * i + 3] = float(x4[3]);
            }
        }
        #pragma clang loop unroll(full)
        for (short r = 0; r < NR; ++r) {
            const uint2 p = *(device const uint2*)(w0 + (ulong)r * wrow + g * 8);
            float gsum[2] = {0.f, 0.f};
            #pragma clang loop unroll(full)
            for (short u = 0; u < 2; ++u) {
                const uint v = (u == 0) ? p.x : p.y;
                const short col = 8 * u;
                // identical constructs to batch-1 (bit-identity contract)
                const half2 le = as_type<half2>(((v << 9) & 0x0E000E00u) |
                                                ((v << 12) & 0x80008000u));
                const half2 lo = as_type<half2>(((v << 1) & 0x0E000E00u) |
                                                ((v << 4) & 0x80008000u));
                const half2 he = as_type<half2>(((v << 5) & 0x0E000E00u) |
                                                ((v << 8) & 0x80008000u));
                const half2 ho = as_type<half2>(((v >> 3) & 0x0E000E00u) |
                                                (v & 0x80008000u));
                const float wj[8] = {float(le.x), float(he.x),
                                     float(lo.x), float(ho.x),
                                     float(le.y), float(he.y),
                                     float(lo.y), float(ho.y)};
                #pragma clang loop unroll(full)
                for (short j = 0; j < 8; ++j) {
                    #pragma clang loop unroll(full)
                    for (short mi = 0; mi < 2; ++mi)
                        gsum[mi] += xs[mi][col + j] * wj[j];
                }
            }
            const uint sb = s0[(ulong)r * groups + g];
            const half sh = as_type<half>(ushort(((sb & 0x7F) << 7) |
                                                 ((sb & 0x80) << 8)));
            const float sc = 4194304.0f * float(sh);
            #pragma clang loop unroll(full)
            for (short mi = 0; mi < 2; ++mi) sumf[r][mi] += gsum[mi] * sc;
        }
    }
    #pragma clang loop unroll(full)
    for (short mi = 0; mi < 2; ++mi)
        #pragma clang loop unroll(full)
        for (short row = 0; row < NR; ++row) {
            const float tot = metal::simd_sum(sumf[row][mi]);
            if (lane == 0)
                D[(long)(first_col + mi) * N + first_row + row] =
                    T(tot * GS[0]);
        }
}

// mv_ext-class batch GEMV for fp8ch (the llama.cpp kernel_mul_mv_ext
// precedent, built there for speculative-decode batch sizes): NR=4 weight
// rows per simdgroup x R1 activation columns per pass, weights decoded
// ONCE per thread and applied to all R1 columns, X read per column via
// device vec4 loads (L1-served) instead of register staging — the
// structural dodge around the UPDATE 11 register-blowup dead end. No
// threadgroup memory, no barriers. dispatch (N/8, ceil(M/R1)), 64
// threads; weights re-read ceil(M/R1) times.
// N5b variant study (clean box, qkv 12288x5120 @ M=8, GB/s counting the
// quantized bytes once): mb column pairs 228 (the shipped route);
// simdgroup_matrix GEMM 64x32 tiles 153-199 across five staging/layout
// variants (the MAC phase itself is issue-bound at ~2.3e12 FMA/s — M1's
// simdgroup_matrix is cooperative lane math, not a tensor core, so at
// M<=8 the FMA work is the wall, not bandwidth); NR=2 x R1=8 single-pass
// 290; NR=4 x R1=4 two-pass 381 GB/s WINNER (X loads/converts amortized
// over 4 rows lift the FMA issue rate to ~3.0e12/s). M=4 712 GB/s
// (+58% vs mb), lm_head M<=8 ~2x vs both mb and the dense fallback.
// Per-(row, column) chunk partials keep the batch-1 FP chain order, so
// every output row is bit-identical to a looped qgemv_fp8ch launch.
// Host guards: batch-1's plus N % 8 == 0 and X/D contiguous row-major
// (M, K) / (M, N); any M >= 1 (pad columns clamp-read, store-guarded).
template<typename T, short R1>
kernel void qgemv_fp8ch_mv4r(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   const uchar* Wq [[buffer(1)]],   // (N, K) e4m3 bytes, row-major
    device   const T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    device   const float* WS [[buffer(3)]],   // (N,) per-channel scales
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 4;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const int col0 = int(tgid.y) * R1;
    device const uchar* wr[NR];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r)
        wr[r] = Wq + (ulong)(first_row + r) * (ulong)K;
    float sumf[NR][R1];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r)
        #pragma clang loop unroll(full)
        for (short j = 0; j < R1; ++j) sumf[r][j] = 0.f;
    // pad columns clamp to the last live row; their stores are guarded
    ulong xoff[R1];
    #pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) {
        const int col = (col0 + j < M) ? (col0 + j) : (M - 1);
        xoff[j] = (ulong)col * (ulong)K;
    }
    for (int c = int(lane) * 16; c + 16 <= K; c += 32 * 16) {
        // decode all NR rows once (select-free e4m3, UPDATE 18), laid out
        // in the batch-1 xs order: [4u+0]=ae.x [4u+1]=ao.x [4u+2]=ae.y
        // [4u+3]=ao.y
        float w[NR][16];
        #pragma clang loop unroll(full)
        for (short r = 0; r < NR; ++r) {
            const uint4 a = *(device const uint4*)(wr[r] + c);
            #pragma clang loop unroll(full)
            for (short u = 0; u < 4; ++u) {
                const uint av = a[u];
                const half2 ae = as_type<half2>(((av << 7) & 0x3F803F80u) |
                                                ((av << 8) & 0x80008000u));
                const half2 ao = as_type<half2>(((av >> 1) & 0x3F803F80u) |
                                                (av & 0x80008000u));
                w[r][4 * u + 0] = float(ae.x);
                w[r][4 * u + 1] = float(ao.x);
                w[r][4 * u + 2] = float(ae.y);
                w[r][4 * u + 3] = float(ao.y);
            }
        }
        #pragma clang loop unroll(full)
        for (short j = 0; j < R1; ++j) {
            device const metal::vec<T, 4>* xp =
                (device const metal::vec<T, 4>*)(X + xoff[j] + c);
            float xs[16];
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const metal::vec<T, 4> xv = xp[i];
                xs[4 * i + 0] = float(xv.x);
                xs[4 * i + 1] = float(xv.y);
                xs[4 * i + 2] = float(xv.z);
                xs[4 * i + 3] = float(xv.w);
            }
            // per-chunk partials with the rows INTERLEAVED per element —
            // the batch-1/mb chain structure. A clean per-row 16-FMA
            // reduction lets fast-math re-tree the sum and breaks the
            // bit-identity contract with the looped kernel (measured:
            // nvfp4's gsum form survives, this fp8ch form only survives
            // interleaved).
            float rr[NR];
            #pragma clang loop unroll(full)
            for (short r = 0; r < NR; ++r) rr[r] = 0.f;
            #pragma clang loop unroll(full)
            for (short i = 0; i < 16; ++i)
                #pragma clang loop unroll(full)
                for (short r = 0; r < NR; ++r) rr[r] += xs[i] * w[r][i];
            #pragma clang loop unroll(full)
            for (short r = 0; r < NR; ++r) sumf[r][j] += rr[r];
        }
    }
    #pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j)
        #pragma clang loop unroll(full)
        for (short r = 0; r < NR; ++r) {
            const float tot = metal::simd_sum(sumf[r][j]);
            // 256 = the folded 2^8 decode rebias (exact: power-of-two scale)
            if (lane == 0 && col0 + j < M)
                D[(ulong)(col0 + j) * (ulong)N + first_row + r] =
                    T(tot * (256.0f * WS[first_row + r]));
        }
}

// mv_ext-class batch twin for the planar NVFP4 layout: NR=4 rows x R1=2
// columns per pass (R1=2 measured best for this format — the heavier
// per-group decode + scale hoist leaves less FMA headroom than fp8ch:
// gate_up M=8 mv4r2 0.560 ms vs mb 0.670, M=4 0.285 vs 0.340; down M=8
// 0.277 vs 0.367 — and R1=4 regresses on register pressure). The
// per-(row, column) chain is the batch-1 chain (unscaled 16-FMA group
// dot, one gsum * (2^22 * scale) per group, fp32 cross-group), so rows
// are bit-identical to a looped qgemv_nvfp4_planar launch. Host guards:
// batch-1's plus N % 8 == 0 and contiguous row-major X/D; any M >= 1.
// dispatch (N/8, ceil(M/R1)), 64 threads.
template<typename T, short R1>
kernel void qgemv_nvfp4_mv4r(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   const uchar* Wq [[buffer(1)]],   // (N, K/2) packed e2m1
    device   const T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    device   const uchar* WS [[buffer(3)]],   // (N, K/16) e4m3 group scales
    device   const float* GS [[buffer(4)]],   // (1,) global multiplier
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    const constant int &M [[buffer(7)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr short NR = 4;             // output rows per simdgroup
    constexpr short NSG = 2;            // simdgroups per threadgroup
    const int first_row = (int(tgid.x) * NSG + sgitg) * NR;
    const int col0 = int(tgid.y) * R1;
    const int groups = K / 16;
    const ulong wrow = (ulong)K / 2;
    device const uchar* w0 = Wq + (ulong)first_row * wrow;
    device const uchar* s0 = WS + (ulong)first_row * (ulong)groups;
    float sumf[NR][R1];
    #pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r)
        #pragma clang loop unroll(full)
        for (short j = 0; j < R1; ++j) sumf[r][j] = 0.f;
    ulong xoff[R1];
    #pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) {
        const int col = (col0 + j < M) ? (col0 + j) : (M - 1);
        xoff[j] = (ulong)col * (ulong)K;
    }
    for (int g = int(lane); g < groups; g += 32) {
        // decode all NR rows' nibbles unscaled (identical constructs to
        // batch-1) + hoist each row's 2^22-folded group scale once
        float w[NR][16];
        float sc[NR];
        #pragma clang loop unroll(full)
        for (short r = 0; r < NR; ++r) {
            const uint2 p = *(device const uint2*)(w0 + (ulong)r * wrow + g * 8);
            #pragma clang loop unroll(full)
            for (short u = 0; u < 2; ++u) {
                const uint v = (u == 0) ? p.x : p.y;
                const short col = 8 * u;
                const half2 le = as_type<half2>(((v << 9) & 0x0E000E00u) |
                                                ((v << 12) & 0x80008000u));
                const half2 lo = as_type<half2>(((v << 1) & 0x0E000E00u) |
                                                ((v << 4) & 0x80008000u));
                const half2 he = as_type<half2>(((v << 5) & 0x0E000E00u) |
                                                ((v << 8) & 0x80008000u));
                const half2 ho = as_type<half2>(((v >> 3) & 0x0E000E00u) |
                                                (v & 0x80008000u));
                w[r][col + 0] = float(le.x);
                w[r][col + 1] = float(he.x);
                w[r][col + 2] = float(lo.x);
                w[r][col + 3] = float(ho.x);
                w[r][col + 4] = float(le.y);
                w[r][col + 5] = float(he.y);
                w[r][col + 6] = float(lo.y);
                w[r][col + 7] = float(ho.y);
            }
            const uint sb = s0[(ulong)r * groups + g];
            const half sh = as_type<half>(ushort(((sb & 0x7F) << 7) |
                                                 ((sb & 0x80) << 8)));
            sc[r] = 4194304.0f * float(sh);
        }
        #pragma clang loop unroll(full)
        for (short j = 0; j < R1; ++j) {
            device const metal::vec<T, 4>* xp4 =
                (device const metal::vec<T, 4>*)(X + xoff[j] + g * 16);
            float xs[16];
            #pragma clang loop unroll(full)
            for (short i = 0; i < 4; ++i) {
                const metal::vec<T, 4> x4 = xp4[i];
                xs[4 * i + 0] = float(x4[0]);
                xs[4 * i + 1] = float(x4[1]);
                xs[4 * i + 2] = float(x4[2]);
                xs[4 * i + 3] = float(x4[3]);
            }
            #pragma clang loop unroll(full)
            for (short r = 0; r < NR; ++r) {
                float gsum = 0.f;
                #pragma clang loop unroll(full)
                for (short i = 0; i < 16; ++i) gsum += xs[i] * w[r][i];
                sumf[r][j] += gsum * sc[r];
            }
        }
    }
    #pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j)
        #pragma clang loop unroll(full)
        for (short r = 0; r < NR; ++r) {
            const float tot = metal::simd_sum(sumf[r][j]);
            if (lane == 0 && col0 + j < M)
                D[(ulong)(col0 + j) * (ulong)N + first_row + r] =
                    T(tot * GS[0]);
        }
}

// Multi-batch twin of qgemv_q8_0_fast: same 8-blocks-per-iteration walk and
// per-row FMA order (bit-identical rows to the looped fast kernel), with the
// block decode amortized over M activation rows. M is compile-time so the
// accumulators stay in registers.
template<int M>
kernel void qgemv_q8_0_mb_fast(
    device   half*  D  [[buffer(0)]],   // (M, N)
    device   uchar* Wq [[buffer(1)]],
    device   half*  X  [[buffer(2)]],   // (M, K)
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / q8_0::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * q8_0::block_bytes;

    const int block_offset = (int)(lane >> 2);
    const int chunk = (int)(lane & 3);

    float acc[M];
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 8) {
        device const uchar* block = row_base + (uint)kb * q8_0::block_bytes;
        const float d = float(((device const half*)block)[0]);
        device const char* qs = (device const char*)(block + 2 + chunk * 8);
        const int x0 = (kb << 5) + chunk * 8;
        #pragma clang loop unroll(full)
        for (int m = 0; m < M; ++m) {
            // Bit-compat with the compiled batch-1 fast kernel requires the
            // exact same FMA chain per row; fast-math reassociation across
            // the unrolled m/i nest changes rounding on scattered rows —
            // do not hoist d*float(qs[i]) (see optimization_status
            // 2026-08-11).
            #pragma clang fp reassociate(off)
            device const half* xm = X + (long)m * K + x0;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                acc[m] += d * float(qs[i]) * float(xm[i]);
            }
        }
    }
    #pragma clang loop unroll(full)
    for (int m = 0; m < M; ++m) {
        const float s = metal::simd_sum(acc[m]);
        if (lane == 0) D[(long)m * N + row] = half(s);
    }
}

#define instantiate_qgemv_q8_0_mb(M)                                          \
   template [[host_name("qgemv_q8_0_mb" #M)]] [[kernel]]                      \
   void qgemv_q8_0_mb_fast<M>(                                                \
     device half* D [[buffer(0)]], device uchar* Wq [[buffer(1)]],            \
     device half* X [[buffer(2)]],                                            \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]],\
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_q8_0_mb(2)
instantiate_qgemv_q8_0_mb(3)
instantiate_qgemv_q8_0_mb(4)
instantiate_qgemv_q8_0_mb(5)
instantiate_qgemv_q8_0_mb(6)
instantiate_qgemv_q8_0_mb(7)
instantiate_qgemv_q8_0_mb(8)

// MXFP4 whole-block decode. A single lane consumes all 32 weights behind one
// E8M0 scale, reducing scale expansion from four times per block in the generic
// 8-value-span geometry to once. Adjacent lanes still walk adjacent 17-byte
// blocks and adjacent 32-value activation spans.
[[host_name("qgemv_mxfp4")]]
kernel void qgemv_mxfp4_fast(
    device half *D [[buffer(0)]],
    device const uchar *Wq [[buffer(1)]],
    device const half *X [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(tgid.x);
    const int blocks_per_row = K / mxfp4::block_k;
    device const uchar *row_base =
        Wq + (uint)(row * blocks_per_row) * mxfp4::block_bytes;
    float accumulator = 0.0f;
    for (int block = int(lane); block < blocks_per_row; block += 32) {
        device const uchar *base =
            row_base + (uint)block * mxfp4::block_bytes;
        const half scale = tk_e8m0_decode(base[0]);
        device const uchar *codes = base + 1;
        const int input_base = block * mxfp4::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            const half low = scale * tk_e2m1_decode(uint(packed & 0x0f));
            const half high = scale * tk_e2m1_decode(uint(packed >> 4));
            accumulator += float(low) * float(X[input_base + i]);
            accumulator += float(high) * float(X[input_base + i + 16]);
        }
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) D[row] = half(accumulator);
}

// MXFP8 whole-block decode. One lane consumes all 32 E4M3 codes behind
// an E8M0 scale, replacing the generic four-lanes-per-block schedule.
[[host_name("qgemv_mxfp8")]]
kernel void qgemv_mxfp8_fast(
    device half *D [[buffer(0)]],
    device const uchar *Wq [[buffer(1)]],
    device const half *X [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(tgid.x);
    const int blocks_per_row = K / mxfp8::block_k;
    device const uchar *row_base =
        Wq + (uint)(row * blocks_per_row) * mxfp8::block_bytes;
    float accumulator = 0.0f;
    for (int block = int(lane); block < blocks_per_row; block += 32) {
        device const uchar *base =
            row_base + (uint)block * mxfp8::block_bytes;
        const half scale = metal::exp2(half((int)base[0] - 127));
        device const uchar *codes = base + 1;
        const int input_base = block * mxfp8::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < mxfp8::block_k; ++i) {
            const half value = scale * tk_e4m3_decode(codes[i]);
            accumulator += float(value) * float(X[input_base + i]);
        }
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) D[row] = half(accumulator);
}

// The f32 decode specializations preserve f32 activations and output instead
// of routing through the fp16 decode contract. They are intentionally limited
// to the q4_0 and q6_K GGUF layouts.
[[host_name("qgemv_q4_0_float32")]]
kernel void qgemv_q4_0_float32(
    device float *D [[buffer(0)]],
    device const uchar *Wq [[buffer(1)]],
    device const float *X [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(tgid.x);
    const int bpr = K / q4_0::block_k;
    device const uchar *row_base = Wq + (uint)(row * bpr) * q4_0::block_bytes;
    const int block_offset = int(lane >> 1);
    const int byte_start = int(lane & 1) * 8;
    float acc = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 16) {
        device const uchar *block = row_base + (uint)kb * q4_0::block_bytes;
        const float scale = float(((device const half *)block)[0]);
        device const uchar *qs = block + 2 + byte_start;
        const int x0 = (kb << 5) + byte_start;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = qs[i];
            acc += scale * float(int(packed & 0x0f) - 8) * X[x0 + i];
            acc += scale * float(int(packed >> 4) - 8) * X[x0 + i + 16];
        }
    }
    acc = metal::simd_sum(acc);
    if (lane == 0) D[row] = acc;
}

METAL_FUNC float qgemv_f16_bits_to_f32(ushort value) {
    const uint sign = uint(value & 0x8000) << 16;
    const uint exponent = (value >> 10) & 0x1f;
    const uint mantissa = value & 0x03ff;
    if (exponent == 0) {
        const float magnitude = metal::ldexp(float(mantissa), -24);
        return sign != 0 ? -magnitude : magnitude;
    }
    if (exponent == 31) {
        return as_type<float>(sign | 0x7f800000 | (mantissa << 13));
    }
    return as_type<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

[[host_name("qgemv_q6_K_float32")]]
kernel void qgemv_q6_K_float32(
    device float *D [[buffer(0)]],
    device const uchar *Wq [[buffer(1)]],
    device const float *X [[buffer(2)]],
    const constant int &N [[buffer(3)]],
    const constant int &K [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int row = int(tgid.x);
    const int blocks_per_row = K / 256;
    device const uchar *row_weights = Wq + (long)row * blocks_per_row * 210;
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        device const uchar *block = row_weights + (long)block_index * 210;
        device const uchar *ql = block;
        device const uchar *qh = block + 128;
        device const char *scales = (device const char *)(block + 192);
        const ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
        const float d = qgemv_f16_bits_to_f32(d_bits);
        for (uint chunk = 0; chunk < 2; ++chunk) {
            for (uint group = 0; group < 4; ++group) {
                const uchar ql_byte = ql[chunk * 64 + lane + 32 * (group & 1)];
                const uint nibble = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0x0f);
                const uint high = (qh[chunk * 32 + lane] >> (2 * group)) & 3;
                const int quant = int(nibble | (high << 4)) - 32;
                const uint scale_index = chunk * 8 + (lane >> 4) + group * 2;
                const uint column =
                    block_index * 256 + chunk * 128 + group * 32 + lane;
                sum += d * float(int(scales[scale_index])) * float(quant) * X[column];
            }
        }
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) D[row] = sum;
}

#define instantiate_qgemv(name, FMT, T)                                       \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv<FMT, T>(                                                        \
     device T* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device T* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint lane [[thread_index_in_simdgroup]]);

// Pipeline naming: qgemv_<fmt>_mb<M> for half, qgemv_<fmt>_mb<M>_bfloat16
// for bf16 (M is part of the host name so the accumulators stay compile-time).
#define instantiate_qgemv_mb_one(suffix, name, FMT, T, M)                     \
   template [[host_name(name #M suffix)]] [[kernel]]                          \
   void qgemv_mb<FMT, T, M>(                                                  \
     device T* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device T* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     uint lane [[thread_index_in_simdgroup]]);

#define instantiate_qgemv_mb(suffix, name, FMT, T)                            \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 2)                          \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 3)                          \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 4)                          \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 5)                          \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 6)                          \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 7)                          \
   instantiate_qgemv_mb_one(suffix, name, FMT, T, 8)

#define instantiate_qgemv_moe(name, FMT, T)                                  \
   template [[host_name(name)]] [[kernel]]                                   \
   void qgemv_moe<FMT, T>(                                                   \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const int* topk_ids [[buffer(3)]], \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]], \
     const constant int &topk [[buffer(6)]],                                  \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     uint lane [[thread_index_in_simdgroup]]);

// Full-weight dequant to fp16: packed (N, K/bk, bytes) -> W (N, K) half. Flat, one thread per
// 8-col span (tk_dequant8 + two half4 stores). Backs the k-quant PREFILL route: the 256-superblock
// formats' fragment-path in-GEMM dequant measured 2-2.3x slower than dequantize-then-mx.matmul,
// so qgemm routes them here for M >= 64.
template<typename FMT>
kernel void qdequant_fp16(
    device half*  W  [[buffer(0)]],   // (N, K) output
    device uchar* Wq [[buffer(1)]],   // (N, K/block_k, block_bytes)
    const constant int &N [[buffer(2)]],
    const constant int &K [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    const int spans_per_row = K / 8;
    const uint total = (uint)N * (uint)spans_per_row;
    if (tid >= total) return;
    const int row  = (int)(tid / spans_per_row);
    const int col0 = (int)(tid % spans_per_row) * 8;
    const int bpr = K / FMT::block_k;
    const int blk = col0 / FMT::block_k;
    const int cib = col0 % FMT::block_k;
    device const uchar* base = Wq + ((uint)row * bpr + blk) * FMT::block_bytes;
    half w[8];
    tk_dequant8<FMT>(base, cib, w);
    device half4* dst = (device half4*)(W + (long)row * K + col0);
    dst[0] = half4(w[0], w[1], w[2], w[3]);
    dst[1] = half4(w[4], w[5], w[6], w[7]);
}

#define instantiate_qdequant(name, FMT)                                       \
   template [[host_name(name)]] [[kernel]]                                    \
   void qdequant_fp16<FMT>(                                                   \
     device half* W [[buffer(0)]], device uchar* Wq [[buffer(1)]],            \
     const constant int &N [[buffer(2)]], const constant int &K [[buffer(3)]],\
     uint tid [[thread_position_in_grid]]);

instantiate_qdequant("qdequant_q4_K", q4_K);
instantiate_qdequant("qdequant_q5_K", q5_K);
instantiate_qdequant("qdequant_q6_K", q6_K);
instantiate_qdequant("qdequant_q2_K", q2_K);
instantiate_qdequant("qdequant_q3_K", q3_K);
instantiate_qdequant("qdequant_iq4_xs", iq4_xs);
instantiate_qdequant("qdequant_iq2_xxs", iq2_xxs);
instantiate_qdequant("qdequant_iq2_xs", iq2_xs);
instantiate_qdequant("qdequant_iq3_xxs", iq3_xxs);
instantiate_qdequant("qdequant_iq1_s", iq1_s);
instantiate_qdequant("qdequant_tq2_0", tq2_0);

#define instantiate_qgemv_mm(name, FMT, T, MROWS)                            \
   template [[host_name(name)]] [[kernel]]                                   \
   void qgemv_mm<FMT, T, MROWS>(                                             \
     device T* D [[buffer(0)]], device uchar* Wq [[buffer(1)]],              \
     device T* X [[buffer(2)]],                                              \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     uint lane [[thread_index_in_simdgroup]]);

#define instantiate_qgemv_mm_rows(name, FMT, T)                              \
    instantiate_qgemv_mm(name "_2",  FMT, T, 2);                             \
    instantiate_qgemv_mm(name "_4",  FMT, T, 4);                             \
    instantiate_qgemv_mm(name "_8",  FMT, T, 8);                             \
    instantiate_qgemv_mm(name "_16", FMT, T, 16);                            \
    instantiate_qgemv_mm(name "_17", FMT, T, 17);

#define instantiate_qgemv_mm_format(name, FMT)                               \
    instantiate_qgemv_mm_rows(name, FMT, half);                              \
    instantiate_qgemv_mm_rows(name "_bfloat16", FMT, bf16);

instantiate_qgemv_mm_format("qgemv_mm_q4_0", q4_0);
instantiate_qgemv_mm_format("qgemv_mm_q8_0", q8_0);
instantiate_qgemv_mm_format("qgemv_mm_q4_K", q4_K);
instantiate_qgemv_mm_format("qgemv_mm_q5_K", q5_K);
instantiate_qgemv_mm_format("qgemv_mm_q6_K", q6_K);

// NR-layout q4_K batch twin (see qgemv_q4k_nr_mb): grid.y carries the
// column-pair index, so one instantiation per dtype serves every even
// batch width. The 16/17-row kMMRows chunks stay on qgemv_mm until the
// odd-width remainder handling is worth it.
#define instantiate_qgemv_q4k_nr_mb(name, T)                                 \
   template [[host_name(name)]] [[kernel]]                                   \
   void qgemv_q4k_nr_mb<T>(                                                  \
     device T* D [[buffer(0)]], device uchar* Wq [[buffer(1)]],              \
     device T* X [[buffer(2)]],                                              \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                            \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                        \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_q4k_nr_mb("qgemv_mm_q4_K_nr", half);
instantiate_qgemv_q4k_nr_mb("qgemv_mm_q4_K_nr_bfloat16", bf16);

#define instantiate_qgemv_format(name, FMT)                                  \
    instantiate_qgemv(name, FMT, half);                                      \
    instantiate_qgemv(name "_bfloat16", FMT, bf16);                         \
    instantiate_qgemv_moe(name "_moe", FMT, half);                          \
    instantiate_qgemv_moe(name "_moe_bfloat16", FMT, bf16);

// q8_0/q4_0 retain their specialized half decode kernels. BF16 and MoE use
// the generic typed implementation, whose names do not collide with them.
instantiate_qgemv("qgemv_q8_0_bfloat16", q8_0, bf16);
instantiate_qgemv("qgemv_q4_0_bfloat16", q4_0, bf16);
instantiate_qgemv_moe("qgemv_q8_0_moe", q8_0, half);
instantiate_qgemv_moe("qgemv_q8_0_moe_bfloat16", q8_0, bf16);
instantiate_qgemv_moe("qgemv_q4_0_moe", q4_0, half);
instantiate_qgemv_moe("qgemv_q4_0_moe_bfloat16", q4_0, bf16);
instantiate_qgemv("qgemv_q8_0_small", q8_0, half);
instantiate_qgemv("qgemv_q4_0_small", q4_0, half);

// Multi-batch (M<=8) weight-stationary variants for the serving-path
// decode/verify widths. q8_0 fp16 uses the specialized twin above; the rest
// share the generic template (same walk as their batch-1 instantiations).
instantiate_qgemv_mb("_bfloat16", "qgemv_q8_0_mb", q8_0, bf16);
instantiate_qgemv_mb("", "qgemv_q2_K_mb", q2_K, half);
instantiate_qgemv_mb("_bfloat16", "qgemv_q2_K_mb", q2_K, bf16);
instantiate_qgemv_mb("", "qgemv_iq2_xxs_mb", iq2_xxs, half);
instantiate_qgemv_mb("_bfloat16", "qgemv_iq2_xxs_mb", iq2_xxs, bf16);
instantiate_qgemv_mb("", "qgemv_q6_K_mb", q6_K, half);
instantiate_qgemv_mb("_bfloat16", "qgemv_q6_K_mb", q6_K, bf16);

instantiate_qgemv_format("qgemv_q4_K", q4_K);

// llama.cpp-layout q4_K decode GEMV (batch-1 fast path; see qgemv_q4k_nr).
#define instantiate_qgemv_q4k_nr(name, T)                                     \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_q4k_nr<T>(                                                      \
     device T* D [[buffer(0)]], device uchar* Wq [[buffer(1)]], device T* X [[buffer(2)]], \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_q4k_nr("qgemv_q4_K_nr", half);
instantiate_qgemv_q4k_nr("qgemv_q4_K_nr_bfloat16", bf16);

// Compressed-tensors FP8-per-channel W8A16 GEMV (planar rows + scale buffer).
#define instantiate_qgemv_fp8ch(name, T)                                      \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_fp8ch<T>(                                                       \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const float* WS [[buffer(3)]],   \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]],\
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_fp8ch("qgemv_fp8ch", half);
instantiate_qgemv_fp8ch("qgemv_fp8ch_bfloat16", bf16);

// Column-pair batch twin (grid (N/4, M/2)); same bindings as batch-1.
#define instantiate_qgemv_fp8ch_mb(name, T)                                   \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_fp8ch_mb<T>(                                                    \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const float* WS [[buffer(3)]],   \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]],\
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_fp8ch_mb("qgemv_fp8ch_mb", half);
instantiate_qgemv_fp8ch_mb("qgemv_fp8ch_mb_bfloat16", bf16);

// Compressed-tensors NVFP4 W4A16 GEMV over the planar checkpoint layout.
#define instantiate_qgemv_nvfp4_planar(name, T)                               \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_nvfp4_planar<T>(                                                \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],   \
     device const float* GS [[buffer(4)]],                                    \
     const constant int &N [[buffer(5)]], const constant int &K [[buffer(6)]],\
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_nvfp4_planar("qgemv_nvfp4_planar", half);
instantiate_qgemv_nvfp4_planar("qgemv_nvfp4_planar_bfloat16", bf16);

// Column-pair batch twin (grid (N/4, M/2)); same bindings as batch-1.
#define instantiate_qgemv_nvfp4_planar_mb(name, T)                            \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_nvfp4_planar_mb<T>(                                             \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],   \
     device const float* GS [[buffer(4)]],                                    \
     const constant int &N [[buffer(5)]], const constant int &K [[buffer(6)]],\
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_nvfp4_planar_mb("qgemv_nvfp4_planar_mb", half);
instantiate_qgemv_nvfp4_planar_mb("qgemv_nvfp4_planar_mb_bfloat16", bf16);

// mv_ext batch twin (grid (N/8, ceil(M/4))); batch-1 bindings + M @6.
#define instantiate_qgemv_fp8ch_mv4r(name, T, R1)                             \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_fp8ch_mv4r<T, R1>(                                              \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const float* WS [[buffer(3)]],   \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]],\
     const constant int &M [[buffer(6)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_fp8ch_mv4r("qgemv_fp8ch_mv4r", half, 4);
instantiate_qgemv_fp8ch_mv4r("qgemv_fp8ch_mv4r_bfloat16", bf16, 4);

// mv_ext batch twin (grid (N/8, ceil(M/2))); batch-1 bindings + M @7.
#define instantiate_qgemv_nvfp4_mv4r(name, T, R1)                             \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemv_nvfp4_mv4r<T, R1>(                                              \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],   \
     device const float* GS [[buffer(4)]],                                    \
     const constant int &N [[buffer(5)]], const constant int &K [[buffer(6)]],\
     const constant int &M [[buffer(7)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

instantiate_qgemv_nvfp4_mv4r("qgemv_nvfp4_mv4r", half, 2);
instantiate_qgemv_nvfp4_mv4r("qgemv_nvfp4_mv4r_bfloat16", bf16, 2);
instantiate_qgemv_format("qgemv_kU4B8", kU4B8);
instantiate_qgemv_format("qgemv_kU4", kU4);
instantiate_qgemv_format("qgemv_fp8_e4m3", fp8_e4m3);
instantiate_qgemv_format("qgemv_fp4_e2m1", fp4_e2m1);
instantiate_qgemv_format("qgemv_nvfp4", nvfp4);
instantiate_qgemv_format("qgemv_bitnet", bitnet);
instantiate_qgemv_format("qgemv_tq2_0", tq2_0);
instantiate_qgemv_format("qgemv_iq4_nl", iq4_nl);
instantiate_qgemv_format("qgemv_iq4_xs", iq4_xs);
instantiate_qgemv_format("qgemv_iq2_xxs", iq2_xxs);
instantiate_qgemv_format("qgemv_iq2_xs", iq2_xs);
instantiate_qgemv_format("qgemv_iq3_xxs", iq3_xxs);
instantiate_qgemv_format("qgemv_iq1_s", iq1_s);
instantiate_qgemv_format("qgemv_q4_1", q4_1);
instantiate_qgemv_format("qgemv_q5_0", q5_0);
instantiate_qgemv_format("qgemv_q5_1", q5_1);
instantiate_qgemv_format("qgemv_q2_K", q2_K);
instantiate_qgemv_format("qgemv_q3_K", q3_K);
instantiate_qgemv_format("qgemv_q5_K", q5_K);
instantiate_qgemv_format("qgemv_q6_K", q6_K);
instantiate_qgemv_format("qgemv_e5m2", e5m2);
instantiate_qgemv_format("qgemv_fp8_block", fp8_block);
instantiate_qgemv_format("qgemv_mxfp6_e3m2", mxfp6_e3m2);
instantiate_qgemv_format("qgemv_mxfp6_e2m3", mxfp6_e2m3);
instantiate_qgemv_format("qgemv_hqq", hqq);

}
