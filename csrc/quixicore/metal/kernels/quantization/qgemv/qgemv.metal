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
instantiate_qdequant("qdequant_iq2_s", iq2_s);
instantiate_qdequant("qdequant_iq3_s", iq3_s);
instantiate_qdequant("qdequant_iq1_m", iq1_m);
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
// Qwen3.8 UD-Q2_K_XL verify band: the i-quant / K-quant formats that carry
// the MLP and GDN projections. Their single-row qgemv is dequant-ALU-bound,
// so the weight-stationary row block amortizes the decode over M rows
// instead of re-running it (and re-streaming the bytes) once per row.
instantiate_qgemv_mm_format("qgemv_mm_q2_K", q2_K);
instantiate_qgemv_mm_format("qgemv_mm_q3_K", q3_K);
instantiate_qgemv_mm_format("qgemv_mm_iq1_s", iq1_s);
instantiate_qgemv_mm_format("qgemv_mm_iq1_m", iq1_m);
instantiate_qgemv_mm_format("qgemv_mm_iq2_xxs", iq2_xxs);
instantiate_qgemv_mm_format("qgemv_mm_iq2_xs", iq2_xs);
instantiate_qgemv_mm_format("qgemv_mm_iq2_s", iq2_s);
instantiate_qgemv_mm_format("qgemv_mm_iq3_xxs", iq3_xxs);
instantiate_qgemv_mm_format("qgemv_mm_iq3_s", iq3_s);
instantiate_qgemv_mm_format("qgemv_mm_iq4_xs", iq4_xs);

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
instantiate_qgemv_format("qgemv_iq2_s", iq2_s);
instantiate_qgemv_format("qgemv_iq3_s", iq3_s);
instantiate_qgemv_format("qgemv_iq1_m", iq1_m);
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
