/**
 * @file
 * @brief Q2_K weight-only matmul for NVIDIA Ampere (SM 8.0+), int8-activation route.
 *
 * Design, budget analysis and measurements: perf/a100_glm52_design.md.
 *
 * Why this exists: the stock GGUF Q2_K path holds 12-32% of DRAM roofline because
 * it (a) has no async copy and only one outstanding weight load per warp, (b)
 * re-reads the weight tile every MMQ_X=4 tokens, (c) recomputes the
 * `m_j * sum(x)` rank-1 term per output row via dp4a, and (d) re-decodes scales
 * per output element. This kernel keeps GGUF's proven integer math and fixes all
 * four.
 *
 * Route selection (measured on A100, see the design doc):
 *   integer dp4a + int32 IMAD scale/min reduction   4975 Gweight/s   <-- this
 *   fp16 lop3 + Marlin scale_and_sub, native scales 4346 Gweight/s
 *   fp16 lop3 + pre-expanded fp16 scale pairs       1054 Gweight/s
 * Marlin's "fold the min into the fp16 fragment for free" is correct at its
 * group_size=128, but Q2_K's group is 16, so the fp16 scale decode runs 8x more
 * often and the integer route wins. Do not switch to fp16 without re-measuring
 * at group_size=16.
 *
 * The factorization, per 16-element sub-block j (4-bit scale sc_j and min m_j,
 * superblock fp16 d/dmin):
 *   dot(w,x) = d * SUM_j sc_j * (q.x)_j  -  dmin * SUM_j m_j * (SUM x)_j
 * `(SUM x)_j` is activation-only, so q2k_quant_a8 computes it once and it is
 * reused across every output row and every expert. That removes the 8-of-16
 * wasted dp4a in the stock path.
 *
 * Layout: a load-time, byte-neutral repack splits native 84-byte q2_K blocks into
 * three aligned planes so every warp load is coalesced and stride arithmetic is a
 * power of two. Total stays 84 B / 256 weights = 2.625 bpw:
 *   qp [N][K/16] uint32   quants, one word = one 16-element sub-block  (K/4  B/row)
 *        Within a word, q for element l sits at bit 8*(l/4) + 2*(l%4), so the four
 *        dp4a operands are (w >> 2p) & 0x03030303 for p = 0..3, and byte b of
 *        operand p holds l = 4b + p. q2k_quant_a8 stores activations in the same
 *        interleave, so no runtime shuffle is needed.
 *   sp [N][K/16] uint8    sc | m nibble pair per sub-block             (K/16 B/row)
 *   dp [N][K/256] half2   (d, dmin) per superblock                     (K/64 B/row)
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace tmq_a100 {

// ---------------------------------------------------------------- cp.async (sm80)
// Provided for staged variants; the GEMV below gets its memory-level parallelism
// from explicit unrolling instead, which keeps the register budget for accumulators.
__device__ __forceinline__ void cp_async16(void* smem, const void* glob) {
    uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    // .cg bypasses L1 and needs no extra registers (Ampere tuning guide 1.4.1.2).
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(glob));
}
__device__ __forceinline__ void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ------------------------------------------------------------------------ repack
// Native q2_K block (84 B, matches quant_formats_tables.cuh::q2_K and llama.cpp):
//   uint8 scales[16];  uint8 qs[64];  half d;  half dmin;
// Native column -> value mapping (from q2_K::dequant):
//   chunk = col>>7, pos = col&127, sidx = pos>>5, sub = (pos>>4)&1, l = pos&15
//   is = chunk*8 + sidx*2 + sub
//   q  = (qs[chunk*32 + sub*16 + l] >> (2*sidx)) & 3
//   v  = d*(scales[is]&0xF)*q - dmin*(scales[is]>>4)
__global__ void q2k_repack(const uint8_t* __restrict__ src, uint32_t* __restrict__ qp,
                           uint8_t* __restrict__ sp, half2* __restrict__ dp,
                           int N, int K) {
    const int nsb = K / 256;
    const long total = (long)N * nsb;
    for (long t = blockIdx.x * (long)blockDim.x + threadIdx.x; t < total;
         t += (long)gridDim.x * blockDim.x) {
        const int n = (int)(t / nsb), sb = (int)(t % nsb);
        const uint8_t* b = src + ((size_t)n * nsb + sb) * 84;
        const uint8_t* scales = b;
        const uint8_t* qs = b + 16;

        dp[(size_t)n * nsb + sb] =
            __halves2half2(*reinterpret_cast<const __half*>(b + 80),
                           *reinterpret_cast<const __half*>(b + 82));

        for (int is = 0; is < 16; ++is) {
            const int chunk = is >> 3, rem = is & 7, sidx = rem >> 1, sub = rem & 1;
            uint32_t w = 0;
            for (int l = 0; l < 16; ++l) {
                const uint32_t q = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3u;
                // Bit position chosen so (w >> 2p) & 0x03030303 yields, in its four
                // bytes, logical elements l = p, 4+p, 8+p, 12+p.
                w |= q << (8 * (l >> 2) + 2 * (l & 3));
            }
            const size_t o = (size_t)n * (K / 16) + sb * 16 + is;
            qp[o] = w;
            sp[o] = scales[is];
        }
    }
}

// Global column covered by (superblock sb, sub-block is, element l).
__host__ __device__ __forceinline__ int q2k_col_of(int sb, int is, int l) {
    const int chunk = is >> 3, rem = is & 7, sidx = rem >> 1, sub = rem & 1;
    return sb * 256 + chunk * 128 + sidx * 32 + sub * 16 + l;
}

// ------------------------------------------------------- activation quantization
// int8 with a per-SUPERBLOCK (256) scale + per-16 sums for the min term.
//   xq   [M][K]       int8   packed interleave, k in repacked order
//   xs   [M][K/256]   half   superblock scale
//   xsum [M][K/16]    int32  sum of the 16 int8 codes  <-- the amortized SUM x
// The scale group is 256, not q8_1's 32, so that `g` is constant across the four
// sub-blocks a lane batches in the GEMV; that lets sc_j*A_j and m_j*S_j
// accumulate in int32 and cuts the float epilogue by 4x (design doc 2.5b).
// Activations are the more precise side (int8 vs 2-bit weights), so the coarser
// group costs little -- the harness checks it against an fp64 reference.
__global__ void q2k_quant_a8(const __half* __restrict__ X, int8_t* __restrict__ xq,
                             __half* __restrict__ xs, int32_t* __restrict__ xsum,
                             int M, int K) {
    const int m = blockIdx.y;
    if (m >= M) return;
    const int nsb = K / 256;
    for (int sb = blockIdx.x * blockDim.x + threadIdx.x; sb < nsb;
         sb += gridDim.x * blockDim.x) {
        float amax = 0.f;
        for (int is = 0; is < 16; ++is)
            for (int l = 0; l < 16; ++l)
                amax = fmaxf(amax, fabsf(__half2float(
                                       X[(size_t)m * K + q2k_col_of(sb, is, l)])));
        const float d = amax / 127.f;
        const float id = d > 0.f ? 1.f / d : 0.f;
        xs[(size_t)m * nsb + sb] = __float2half(d);
        for (int is = 0; is < 16; ++is) {
            int s = 0;
            for (int l = 0; l < 16; ++l) {
                const int c = q2k_col_of(sb, is, l);
                const int q = __float2int_rn(__half2float(X[(size_t)m * K + c]) * id);
                const int8_t qc = (int8_t)max(-127, min(127, q));
                // same interleave as the packed quants: pos = (l%4)*4 + l/4
                xq[(size_t)m * K + (size_t)(sb * 16 + is) * 16 + (l & 3) * 4 +
                   (l >> 2)] = qc;
                s += qc;
            }
            xsum[(size_t)m * (K / 16) + sb * 16 + is] = s;
        }
    }
}

// smem bytes needed by q2k_gemv_a8<M_MAX, NR, KC, ...>
template <int M_MAX, int KC>
__host__ __device__ constexpr size_t q2k_gemv_smem() {
    // 4 operand planes of uint32 [p][m][KC/16] + sums + scales
    return (size_t)4 * M_MAX * (KC / 16) * 4 + (size_t)M_MAX * (KC / 16) * 4 +
           (size_t)M_MAX * (KC / 256 + 1) * 2;
}

// ------------------------------------------------------------------------- GEMV
// Three structural fixes over the stock GGUF path, in order of measured impact:
//   1. UNROLL independent weight loads are issued before any is consumed, so each
//      warp keeps UNROLL*128 B in flight. With one dependent load per warp the
//      loop is latency-bound (measured 16% of roofline); this supplies the
//      memory-level parallelism cp.async would otherwise provide.
//   2. Activations for a k-chunk are staged in shared memory once per block, so
//      they are never re-read from DRAM per output row. (Stock MMQ_X = 4
//      re-streams weights every 4 tokens; here the amortization factor is the
//      block's whole row count.)
//   3. A lane owns one 16-element sub-block, so 32 lanes read 128 B contiguous
//      per step and march sequentially along a row.
// d/dmin are per superblock, so a lane applies them itself; nothing accumulates
// across a superblock boundary.
// Tile defaults from the measured (NR, QB) sweep on A100: NR=4, QB=2 is the
// best balance across M (877/747/428/195 GB/s at M=1/2/4/8). Larger NR*QB
// looks better on paper but blows the register budget: NR=8,QB=4 costs 127
// registers -> 2 blocks/SM and drops M=2 by 37%. Use NR=2 for pure M=1
// (1081 GB/s = 61% of the stream ceiling).
template <int M_MAX, int NR = 4, int KC = 4096, int QB = 2>
__global__ __launch_bounds__(256) void q2k_gemv_a8(
    const uint32_t* __restrict__ qp, const uint8_t* __restrict__ sp,
    const half2* __restrict__ dp, const int8_t* __restrict__ xq,
    const __half* __restrict__ xs, const int32_t* __restrict__ xsum,
    float* __restrict__ Y, int M, int N, int K) {
    extern __shared__ unsigned char smem[];
    // Activations live as 4 separate uint32 operand planes indexed by (p, m, jl).
    // Lanes hold consecutive jl, so each plane read is stride-1 and bank-conflict
    // free. A 16-byte-contiguous-per-sub-block layout instead gives lanes stride 4
    // words = a 4-way conflict, which measured 22% vs 60%+ of roofline.
    const int NJC = KC / 16;
    uint32_t* sxp = reinterpret_cast<uint32_t*>(smem);            // [4][M][NJC]
    int32_t* ssum = reinterpret_cast<int32_t*>(sxp + (size_t)4 * M_MAX * NJC);
    __half* sxs = reinterpret_cast<__half*>(ssum + (size_t)M_MAX * NJC);

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nbase = (blockIdx.x * nwarp + warp) * NR;

    const int nj = K / 16, nsb = K / 256;

    float acc[NR][M_MAX];
#pragma unroll
    for (int r = 0; r < NR; ++r)
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) acc[r][m] = 0.f;

    for (int kc = 0; kc < K; kc += KC) {
        const int klen = min(KC, K - kc);
        __syncthreads();
        for (int m = 0; m < M; ++m) {
            for (int t = threadIdx.x; t < klen / 16; t += blockDim.x) {
                const uint32_t* src = reinterpret_cast<const uint32_t*>(
                    xq + (size_t)m * K + (size_t)(kc / 16 + t) * 16);
#pragma unroll
                for (int p = 0; p < 4; ++p)
                    sxp[((size_t)p * M_MAX + m) * NJC + t] = src[p];
            }
            for (int t = threadIdx.x; t < klen / 16; t += blockDim.x)
                ssum[(size_t)m * NJC + t] = xsum[(size_t)m * nj + kc / 16 + t];
            for (int t = threadIdx.x; t < klen / 256; t += blockDim.x)
                sxs[(size_t)m * (KC / 256 + 1) + t] =
                    xs[(size_t)m * nsb + kc / 256 + t];
        }
        __syncthreads();

        // A lane batches QB = 4 consecutive sub-blocks: one 16-byte vectorized
        // load, and all four fall inside one superblock (16 sub-blocks), so d,
        // dmin and the activation scale g are constant across them. sc_j*A_j and
        // m_j*S_j therefore accumulate in int32 and only ONE float epilogue runs
        // per 4 sub-blocks instead of per sub-block.
        const int jlo = kc / 16, jhi = jlo + klen / 16;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int n = nbase + r;
            if (n >= N) break;
            for (int jb = jlo + lane * QB; jb < jhi; jb += 32 * QB) {
                // QB consecutive sub-blocks; the compiler vectorizes these into a
                // single 16 B load when QB == 4 and jb is 4-aligned (it is: jlo is a
                // multiple of 256 and jb = jlo + lane*QB).
                uint32_t wv[QB], scv[QB];
#pragma unroll
                for (int i = 0; i < QB; ++i) {
                    wv[i] = qp[(size_t)n * nj + jb + i];
                    scv[i] = sp[(size_t)n * nj + jb + i];
                }
                const float2 dm = __half22float2(dp[(size_t)n * nsb + (jb >> 4)]);
                const int jl = jb - jlo;
#pragma unroll
                for (int m = 0; m < M_MAX; ++m) {
                    if (m >= M) break;
                    int Ai = 0, Bi = 0;
#pragma unroll
                    for (int i = 0; i < QB; ++i) {
                        const uint32_t w = wv[i];
                        const uint32_t sc = scv[i];
                        const size_t xb = (size_t)m * NJC + jl + i;
                        int A = 0;
                        A = __dp4a((int)((w >> 0) & 0x03030303u),
                                   (int)sxp[0 * M_MAX * NJC + xb], A);
                        A = __dp4a((int)((w >> 2) & 0x03030303u),
                                   (int)sxp[1 * M_MAX * NJC + xb], A);
                        A = __dp4a((int)((w >> 4) & 0x03030303u),
                                   (int)sxp[2 * M_MAX * NJC + xb], A);
                        A = __dp4a((int)((w >> 6) & 0x03030303u),
                                   (int)sxp[3 * M_MAX * NJC + xb], A);
                        Ai += (int)(sc & 0xFu) * A;              // int32 IMAD
                        Bi += (int)(sc >> 4) * ssum[xb];         // int32 IMAD
                    }
                    const float g = __half2float(
                        sxs[(size_t)m * (KC / 256 + 1) + (jl >> 4)]);
                    acc[r][m] += g * (dm.x * (float)Ai - dm.y * (float)Bi);
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int n = nbase + r;
        if (n >= N) break;
#pragma unroll
        for (int m = 0; m < M_MAX; ++m) {
            if (m >= M) break;
            float v = acc[r][m];
#pragma unroll
            for (int off = 16; off; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
            if (lane == 0) Y[(size_t)m * N + n] = v;
        }
    }
}

// Reference dequant from the repacked planes (host-side oracle).
inline float q2k_ref_weight(const uint32_t* qp, const uint8_t* sp, const half2* dp,
                            int n, int K, int j, int l) {
    const size_t o = (size_t)n * (K / 16) + j;
    const uint32_t w = qp[o];
    const uint8_t sc = sp[o];
    const __half* d2 =
        reinterpret_cast<const __half*>(&dp[(size_t)n * (K / 256) + (j >> 4)]);
    const float d = __half2float(d2[0]), dmin = __half2float(d2[1]);
    const int q = (w >> (8 * (l >> 2) + 2 * (l & 3))) & 3;
    return d * (float)(sc & 0xF) * (float)q - dmin * (float)(sc >> 4);
}

}  // namespace tmq_a100
