#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Small-M weight-stationary MMA GEMM, q8_0 / q4_K x bf16, row-major in and out:
//   D[m][n] = sum_k X[m][k] * W[n][k]
//   W (N, K/32 * 34 B) q8_0 blocks, X (M, K) bf16, D (M, N) bf16 rows LDD apart.
//
// The 9..32-row dense band (2026-09-17, perf/results/2026-09-15/glm53f-q2-conc/):
// the tile-library qgemm_sm ran at 110-125 GB/s on q8_0 and its no-dequant
// probe at the same rate (latency-bound on its structure), and the NR walk
// does not widen past 4 rows per lane (NB=8/16 spill). Here the
// accumulators live in float fragments (2 registers per lane per 8 columns
// of M) so occupancy stays high; a shared X K-tile is staged once per
// threadgroup per step with 8-byte loads (bf16 -> half is a 16-bit shift);
// each lane dequantizes 8 weights into 8 threadgroup halves; the weight
// tile is loaded TRANSPOSED as the B operand so the product lands
// row-major (M, N). Tiles are double-buffered (one barrier per step) and K
// is split across grid.z into fp32 partials (qgemm_mma_reduce folds them)
// so short-N / long-K shapes still fill the GPU. MT is M padded to 8.
//
// Numerics: bf16 X and q8_0 W both go through half operands (the X cast is
// what qgemm_sm already did in torch on this band); fp32 accumulation over
// K. Not bit-identical to the NR walk.
// RPS = weight rows per simdgroup (8: one n-fragment; 16: two, so each A
// fragment load feeds twice the MMAs), BK = K per stage (32 or 64: fewer
// barriers, more bytes in flight per lane). (8, 32) is the 2026-09-17
// original, bit for bit; the wider variants were added the same day for
// the 16..32-row band (see perf/optimization_status.md).
template<int FMT, int MT, int RPS, int BK>
kernel void qgemm_mma(
    device   float* P  [[buffer(0)]],   // (KS, M, N) fp32 partials
    device   const uchar* Wq [[buffer(1)]],   // (N, ...) q8_0 (34 B / 32) or q4_K (144 B / 256) blocks
    device   const bf16*  X  [[buffer(2)]],   // (M, K) bf16 row-major
    const constant int &N   [[buffer(3)]],
    const constant int &K   [[buffer(4)]],
    const constant int &M   [[buffer(5)]],
    const constant int &KS  [[buffer(6)]],   // K slices (grid.z)
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
    constexpr int NSG = 4;              // simdgroups per threadgroup
    constexpr int RT  = RPS / 8;        // n fragments per simdgroup
    constexpr int BB  = BK / 32;        // q8_0 blocks (q4_K sub-blocks) per stage
    constexpr int MF  = MT / 8;         // m fragments
    constexpr int AS  = BK + 8;         // padded row strides (halves)
    constexpr int XS  = BK + 8;

    threadgroup half  sA[2][NSG][RPS * AS];   // per-simdgroup dequantized weights [RPS n][BK k]
    threadgroup half  sX[2][MT * XS];         // shared activations [MT m][BK k]
    threadgroup float sC[NSG][64];            // per-simdgroup store staging (8 m x 8 n)

    const int n0 = int(tgid.x) * (NSG * RPS) + int(sgitg) * RPS;
    const int nb = K / BK;              // stages over K
    const int per = (nb + KS - 1) / KS;
    const int ib0 = int(tgid.z) * per;
    const int ib1 = metal::min(nb, ib0 + per);
    const ulong row_bytes = (FMT == 12) ? (ulong)(K / 256) * 144 : (ulong)(K / 32) * 34;
    const short r = lane >> 2;          // weight row within the simdgroup's first 8
    const short q = lane & 3;           // 8-weight quarter of a 32-block

    // X staging: thread -> (m, 4-wide k chunk); MT*BK/4 chunks over 128 threads.
    constexpr int XCH = MT * (BK / 4);

    metal::simdgroup_float8x8 acc[RT][MF];
    #pragma clang loop unroll(full)
    for (int rt = 0; rt < RT; ++rt) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < MF; ++i) acc[rt][i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }

    // Stage K-range `ib` (BK wide) into tile buffer `buf` (a macro: Metal has no lambdas).
    // One lane dequantizes 8 weights of each (row r + 8*rt, 32-block bb).
#define QGEMM_MMA_STAGE(ib_, buf_)                                                   \
    {                                                                                 \
        for (int c = tid; c < XCH; c += NSG * 32) {                                   \
            const int m = c / (BK / 4);                                               \
            const int kk = (c - m * (BK / 4)) * 4;                                    \
            metal::half4 v = metal::half4(0.0h);                                      \
            if (m < M) {                                                              \
                const uint2 raw = *(device const uint2*)(X + (ulong)m * K + (ib_) * BK + kk); \
                v = metal::half4(as_type<float>(raw.x << 16),                  \
                                 as_type<float>(raw.x & 0xFFFF0000u),          \
                                 as_type<float>(raw.y << 16),                  \
                                 as_type<float>(raw.y & 0xFFFF0000u));         \
            }                                                                         \
            *(threadgroup metal::half4*)(sX[(buf_)] + m * XS + kk) = v;               \
        }                                                                             \
        _Pragma("clang loop unroll(full)")                                            \
        for (int rt = 0; rt < RT; ++rt) {                                             \
            device const uchar* wrow = Wq + (ulong)(n0 + r + 8 * rt) * row_bytes;     \
            _Pragma("clang loop unroll(full)")                                        \
            for (int bb = 0; bb < BB; ++bb) {                                         \
                const int b32 = (ib_) * BB + bb;   /* 32-wide block index over K */   \
                threadgroup metal::half4* ap = (threadgroup metal::half4*)             \
                    (sA[(buf_)][sgitg] + (r + 8 * rt) * AS + bb * 32 + q * 8);        \
                if (FMT == 12) {                                                      \
                    /* q4_K: block of 256 = 8 sub-blocks of 32; sub-block (b32 & 7), */ \
                    /* 6-bit scale/min pairs packed in 12 bytes, nibbles low-then-  */ \
                    /* high per 64-value pair of sub-blocks.                        */ \
                    const int blk = b32 >> 3, sub = b32 & 7;                          \
                    device const uchar* bp = wrow + (ulong)blk * 144;                 \
                    /* one 16-byte header load: d, dmin, 12 scale bytes */            \
                    const uint4 hdr = *(device const uint4*)bp;                       \
                    const float d = float(as_type<half>(ushort(hdr.x & 0xFFFFu)));    \
                    const float dmin = float(as_type<half>(ushort(hdr.x >> 16)));     \
                    const uint bsh = uint(sub & 3) * 8u;                              \
                    uint s8, m8;                                                      \
                    if (sub < 4) {   /* sc[sub], sc[sub+4] */                         \
                        s8 = (hdr.y >> bsh) & 63u; m8 = (hdr.z >> bsh) & 63u;         \
                    } else {         /* sc[sub+4] low/high nibble + top bits of sc[sub-4], sc[sub] */\
                        const uint b8 = (hdr.w >> bsh) & 0xFFu;                       \
                        s8 = (b8 & 0x0Fu) | ((((hdr.y >> bsh) >> 6) & 3u) << 4);      \
                        m8 = (b8 >> 4) | ((((hdr.z >> bsh) >> 6) & 3u) << 4);         \
                    }                                                                 \
                    const float dsc = d * float(s8), dmm = dmin * float(m8);          \
                    const uint2 raw = *(device const uint2*)(bp + 16 + (sub >> 1) * 32 + q * 8);\
                    const uint sh = uint(sub & 1) * 4u;                               \
                    /* 4 nibbles -> uchar4 -> float4 in one convert each */           \
                    const metal::float4 lo = metal::float4(as_type<metal::uchar4>((raw.x >> sh) & 0x0F0F0F0Fu));\
                    const metal::float4 hi = metal::float4(as_type<metal::uchar4>((raw.y >> sh) & 0x0F0F0F0Fu));\
                    ap[0] = metal::half4(lo * dsc - dmm);                             \
                    ap[1] = metal::half4(hi * dsc - dmm);                             \
                } else {                                                              \
                    device const uchar* bp = wrow + (ulong)b32 * 34;                  \
                    const float d = float(((device const half*)bp)[0]);              \
                    const metal::char4 q0 = *(device const metal::char4*)(bp + 2 + q * 8); \
                    const metal::char4 q1 = *(device const metal::char4*)(bp + 2 + q * 8 + 4); \
                    ap[0] = metal::half4(metal::float4(q0) * d);                      \
                    ap[1] = metal::half4(metal::float4(q1) * d);                      \
                }                                                                     \
            }                                                                         \
        }                                                                             \
    }

    if (ib0 < ib1) QGEMM_MMA_STAGE(ib0, 0);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    for (int ib = ib0; ib < ib1; ++ib) {
        const int cur = (ib - ib0) & 1;
        if (ib + 1 < ib1) QGEMM_MMA_STAGE(ib + 1, cur ^ 1);   // overlaps this step's MMA
        #pragma clang loop unroll(full)
        for (int kf = 0; kf < BK / 8; ++kf) {
            metal::simdgroup_half8x8 b[RT];    // W^T (k x n): sA is [n][k], loaded transposed
            #pragma clang loop unroll(full)
            for (int rt = 0; rt < RT; ++rt) {
                metal::simdgroup_load(b[rt], sA[cur][sgitg] + rt * 8 * AS + kf * 8, AS, metal::ulong2(0, 0), true);
            }
            #pragma clang loop unroll(full)
            for (int mf = 0; mf < MF; ++mf) {
                metal::simdgroup_half8x8 a;    // X (m x k)
                metal::simdgroup_load(a, sX[cur] + mf * 8 * XS + kf * 8, XS, metal::ulong2(0, 0), false);
                #pragma clang loop unroll(full)
                for (int rt = 0; rt < RT; ++rt) {
                    metal::simdgroup_multiply_accumulate(acc[rt][mf], a, b[rt], acc[rt][mf]);
                }
            }
        }
        // one barrier: the next step's staging is complete, this step's tiles are free
        metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

#undef QGEMM_MMA_STAGE
    // acc[rt][mf] holds (8 m x 8 n) for m = mf*8.., n = n0 + rt*8..: stage
    // through threadgroup memory, then two fp32 partial stores per lane.
    device float* Pz = P + (ulong)tgid.z * M * N;
    #pragma clang loop unroll(full)
    for (int rt = 0; rt < RT; ++rt) {
        #pragma clang loop unroll(full)
        for (int mf = 0; mf < MF; ++mf) {
            metal::simdgroup_store(acc[rt][mf], sC[sgitg], 8, metal::ulong2(0, 0), false);
            metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
            const int m = mf * 8 + (lane >> 2);
            const int nn = (lane & 3) * 2;
            if (m < M) {
                *(device metal::float2*)(Pz + (ulong)m * N + n0 + rt * 8 + nn) =
                    *(threadgroup metal::float2*)(sC[sgitg] + (lane >> 2) * 8 + nn);
            }
            metal::simdgroup_barrier(metal::mem_flags::mem_threadgroup);
        }
    }
}

// D[m][n] = bf16(sum_s P[s][m][n]); rows of D are LDD apart. One thread per
// 4 columns.
[[host_name("qgemm_mma_reduce")]] kernel void qgemm_mma_reduce(
    device   bf16*  D  [[buffer(0)]],
    device   const float* P [[buffer(1)]],
    const constant int &N   [[buffer(2)]],
    const constant int &M   [[buffer(3)]],
    const constant int &KS  [[buffer(4)]],
    const constant int &LDD [[buffer(5)]],
    uint gid [[thread_position_in_grid]]) {
    const uint n4 = uint(N) / 4;
    const uint m = gid / n4;
    if (m >= uint(M)) return;
    const uint n = (gid - m * n4) * 4;
    metal::float4 acc = *(device const metal::float4*)(P + (ulong)m * N + n);
    for (int s = 1; s < KS; ++s) {
        acc += *(device const metal::float4*)(P + ((ulong)s * M + m) * N + n);
    }
    device bf16* dp = D + (ulong)m * LDD + n;
    dp[0] = bf16(acc.x); dp[1] = bf16(acc.y); dp[2] = bf16(acc.z); dp[3] = bf16(acc.w);
}

#define instantiate_qgemm_mma(name, FMT, MT, RPS, BK)                        \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_mma<FMT, MT, RPS, BK>(                                          \
     device float* P [[buffer(0)]], device const uchar* Wq [[buffer(1)]],     \
     device const bf16* X [[buffer(2)]],                                      \
     const constant int &N [[buffer(3)]], const constant int &K [[buffer(4)]], \
     const constant int &M [[buffer(5)]], const constant int &KS [[buffer(6)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort tid [[thread_index_in_threadgroup]],                              \
     ushort sgitg [[simdgroup_index_in_threadgroup]],                         \
     ushort lane [[thread_index_in_simdgroup]]);

#define instantiate_qgemm_mma_variants(fmtname, FMT, MT)                                     \
   instantiate_qgemm_mma("qgemm_mma_" fmtname "_m" #MT "_r8k32_bfloat16", FMT, MT, 8, 32)   \
   instantiate_qgemm_mma("qgemm_mma_" fmtname "_m" #MT "_r16k32_bfloat16", FMT, MT, 16, 32) \
   instantiate_qgemm_mma("qgemm_mma_" fmtname "_m" #MT "_r8k64_bfloat16", FMT, MT, 8, 64)   \
   instantiate_qgemm_mma("qgemm_mma_" fmtname "_m" #MT "_r16k64_bfloat16", FMT, MT, 16, 64)

instantiate_qgemm_mma_variants("q8_0", 8, 8)
instantiate_qgemm_mma_variants("q8_0", 8, 16)
instantiate_qgemm_mma_variants("q8_0", 8, 24)
instantiate_qgemm_mma_variants("q8_0", 8, 32)
instantiate_qgemm_mma_variants("q4_K", 12, 8)
instantiate_qgemm_mma_variants("q4_K", 12, 16)
instantiate_qgemm_mma_variants("q4_K", 12, 24)
instantiate_qgemm_mma_variants("q4_K", 12, 32)

}  // namespace mittens
