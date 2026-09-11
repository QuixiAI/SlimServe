#pragma once
// SPDX-License-Identifier: Apache-2.0
// Phase 4.1 rejected diagnostic: cross-item staging improves this prototype
// but loses to retained Marlin. No serving dispatcher selects it.
// Derived from the recorded 2026-09-08 planar NVFP4 prototype. Fixed NT16,
// WARPS8, K512, stages4. PIPELINE keeps its ring alive between grid-stride
// items and separates reduction scratch; false is the draining control.
// No precision, quant or repack change between those two implementations.
// Prototype: decode-shaped NVFP4 expert GEMM on tensor cores, M <= 16 rows per expert, weights in the
// checkpoint's planar layout (no repack): w [E, N, K/2] uint8 (two e2m1 per byte, low nibble = even k),
// scale [E, N, K/16] e4m3 (one per 16-wide group), gscale [E] fp32. Work item = (expert slot, NT rows);
// blocks walk work items grid-stride (persistent when the grid is smaller than the item count). The rows
// of an expert are gathered from x by a per-slot token list; only valid rows are loaded (rows >= M hold
// stale smem and their mma rows are never stored). Same cp.async / ldmatrix / mma pipeline as
// csrc/quixicore/serving/fp8_decode_gemm.cuh; a k16 mma step covers exactly one scale group, so the B
// fragment is dequantised as bf16(scale * e2m1) from one byte pair per lane.
//   out[pair0[s] + i, n] = bf16(gscale[e] * sum_k x[rows[s][i], k] * scale[e, n, k/16] * w[e, n, k])
#include <cuda_fp16.h>
#include "quixicore/serving/bf16_decode_gemm.cuh"

namespace tms::nvfp4_decode_sm120 {
using tms::decode_gemm::MT;
using tms::decode_gemm::smem_u32;
using tms::decode_gemm::cp_async16;
using tms::decode_gemm::cp_async_commit;
using tms::decode_gemm::cp_async_wait;
using tms::decode_gemm::mma_bf16_16816;

constexpr int G = 16;        // scale group along k
constexpr int WPAD = 16;     // bytes of padding per packed weight row in smem
constexpr int SPAD = 16;     // bytes of padding per scale row in smem

// Bit-level dequant (the Marlin e2m1 trick). Weight words are layout-only repacked so that lane q of an
// n8 tile reads one 32-bit word per k32 segment holding its eight nibbles: bits [15:12] k=2q, [31:28] 2q+1,
// [11:8] 2q+8, [27:24] 2q+9, [7:4] 16+2q, [23:20] 17+2q, [3:0] 24+2q, [19:16] 25+2q. Placing a nibble's
// (e1 e0 m) at bf16 bits [8:6] and its sign at bit 15 gives raw = true * 2^-126 exactly (e = 0 lands on the
// bf16 subnormal m/2 * 2^-126). The e4m3 group scale is turned into bf16 with its exponent field offset by
// SBIAS so that raw * s' = true * s * 2^-8 (normal bf16, exact: <= 7 significant bits); the epilogue
// multiplies by gscale * 2^8. Marlin relies on the same subnormal-raw times bf16-scale product.
constexpr int SBIAS = 238;   // bf16 exponent field = e4m3 exponent + 238: s' = s * 2^(238 - 120)
constexpr float EPI = 256.0f;
__device__ __forceinline__ uint32_t e2m1x2_raw_bf16x2(uint32_t q) {   // nibbles at [15:12] and [31:28]
    return (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
}
__device__ __forceinline__ uint32_t e4m3_scale_bf16x2(uint32_t b) {   // one e4m3 byte -> broadcast bf16x2
    uint32_t h = ((b & 0x80u) << 8) | (((b & 0x7Fu) << 4) + (uint32_t(SBIAS) << 7));
    // The exponent-offset shortcut only applies to normal E4M3 scales.
    // Subnormal scales are mantissa * 2^-9, then biased by 2^118.
    if ((b & 0x78u) == 0) {
        float value = float(b & 7u) * 0x1p109f;
        if (b & 0x80u) value = -value;
        h = __bfloat16_as_ushort(__float2bfloat16_rn(value));
    }
    return h | (h << 16);
}
__device__ __forceinline__ uint32_t hmul2_bf16(uint32_t a, uint32_t b) {
    uint32_t d;
    asm("mul.rn.bf16x2 %0, %1, %2;\n" : "=r"(d) : "r"(a), "r"(b));
    return d;
}

template <int NT, int WARPS, int KCHUNK, int STAGES>
struct Cfg {
    static constexpr int THREADS = WARPS * 32;
    static constexpr int WROW = KCHUNK / 2 + WPAD;         // bytes
    static constexpr int W_BYTES = NT * WROW;
    static constexpr int SROW = KCHUNK / G + SPAD;         // bytes
    static constexpr int S_BYTES = NT * SROW;
    static constexpr int STAGE_BYTES = W_BYTES + S_BYTES;
    static constexpr int RED_BYTES = WARPS * MT * NT * 4;
    static constexpr int SMEM_BYTES = STAGES * STAGE_BYTES > RED_BYTES ? STAGES * STAGE_BYTES : RED_BYTES;
    static constexpr int KSTEPS = KCHUNK / 32;             // k32 segments per chunk (two mma k-steps each)
    static constexpr int NTILES = NT / 8;
    static constexpr int WVEC = KCHUNK / 32;               // 16-byte vectors per packed weight row per chunk
    static constexpr int SVEC = KCHUNK / 256;              // 16-byte vectors per scale row per chunk
    static_assert(KCHUNK % 256 == 0, "scale rows load in 16-byte vectors");
    static_assert(KSTEPS % WARPS == 0, "warps split the k32 segments of a chunk evenly");
    static_assert(NT % 8 == 0 && STAGES >= 2, "n8 tiles, double buffering at least");
    static_assert(STAGE_BYTES % 16 == 0 && WROW % 16 == 0 && SROW % 16 == 0, "16-byte aligned");
};

template <int NT, int WARPS, int KCHUNK, int STAGES, bool PIPELINE, int FIXED_K>
__global__ void __launch_bounds__(WARPS * 32) nvfp4_moe_gemm_kernel(
        const __nv_bfloat16* __restrict__ x,      // [T, K]
        const uint8_t* __restrict__ w,            // [E, N, K/2]
        const uint8_t* __restrict__ sc,           // [E, N, K/16] e4m3
        const float* __restrict__ gs,             // [E]
        const int* __restrict__ eid,              // [S]
        const int* __restrict__ rows,             // [S, MT] token index or -1
        const int* __restrict__ cnt,              // [S]
        const int* __restrict__ pair0,            // [S]
        __nv_bfloat16* __restrict__ out,          // [P, N]
        int S, int N, int unused_k, int mode) {
    constexpr int K = FIXED_K;           // mode 1: stream only (no mma)
    using C = Cfg<NT, WARPS, KCHUNK, STAGES>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int ntiles_n = (N + NT - 1) / NT;
    const int total = ntiles_n * S;
    const int nchunks = K / KCHUNK;
    const int bn = lane >> 2, bq = lane & 3;   // B fragment: row bn of each n8 tile; k pairs (2bq, 2bq+1) and (+8)

    auto stage_w = [&](int s) { return smem_raw + s * C::STAGE_BYTES; };
    auto stage_s = [&](int s) { return smem_raw + s * C::STAGE_BYTES + C::W_BYTES; };
    // A fragments come straight from x (L2-resident): lane holds rows r0 and r0+8 of the mma tile.
    const int r0 = lane >> 2, q2 = (lane & 3) * 2;

    int timeline = 0;
    for (int item = blockIdx.x; item < total; item += gridDim.x, timeline += nchunks) {
        const int slot = item / ntiles_n;
        const int n0 = (item - slot * ntiles_n) * NT;
        const int e = eid[slot];
        const int M = cnt[slot];
        const int* rowlist = rows + slot * MT;
        const __nv_bfloat16* xa = r0 < M ? x + size_t(rowlist[r0]) * K + q2 : nullptr;
        const __nv_bfloat16* xb = r0 + 8 < M ? x + size_t(rowlist[r0 + 8]) * K + q2 : nullptr;

        auto load_chunk = [&](int c, int s) {
            int target_n = n0, target_e = e;
            if constexpr (PIPELINE) {
                // c may reach a later grid-stride item. Keep issuing through
                // expert boundaries without waiting for this item's epilogue.
                const int target = item + (c / nchunks) * gridDim.x;
                if (target >= total) return;
                const int target_slot = target / ntiles_n;
                target_n = (target - target_slot * ntiles_n) * NT;
                target_e = eid[target_slot];
                c %= nchunks;
            }
            const int k0 = c * KCHUNK;
            const uint8_t* we = w + size_t(target_e) * N * (K / 2);
            const uint8_t* se = sc + size_t(target_e) * N * (K / G);
            unsigned char* ws = stage_w(s);
            unsigned char* ss = stage_s(s);
            for (int i = tid; i < NT * C::WVEC; i += C::THREADS) {
                const int r = i / C::WVEC, v = i - r * C::WVEC;
                const int n = min(target_n + r, N - 1);
                cp_async16(smem_u32(ws + r * C::WROW + v * 16),
                           we + size_t(n) * (K / 2) + k0 / 2 + v * 16, true);
            }
            for (int i = tid; i < NT * C::SVEC; i += C::THREADS) {
                const int r = i / C::SVEC, v = i - r * C::SVEC;
                const int n = min(target_n + r, N - 1);
                cp_async16(smem_u32(ss + r * C::SROW + v * 16),
                           se + size_t(n) * (K / G) + k0 / G + v * 16, true);
            }
        };

        float acc[C::NTILES][4];
#pragma unroll
        for (int j = 0; j < C::NTILES; ++j)
#pragma unroll
            for (int q = 0; q < 4; ++q) acc[j][q] = 0.0f;

        if (!PIPELINE || timeline == 0) {
#pragma unroll
            for (int c = 0; c < STAGES - 1; ++c) {
                if (PIPELINE || c < nchunks) load_chunk(c, c);
                cp_async_commit();
            }
        }

        for (int c = 0; c < nchunks; ++c) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            {
                const int cn = c + STAGES - 1;
                if (PIPELINE || cn < nchunks)
                    load_chunk(cn, (PIPELINE ? timeline + cn : cn) % STAGES);
                cp_async_commit();
            }
            const int s = (PIPELINE ? timeline + c : c) % STAGES;
            const unsigned char* ws = stage_w(s);
            const unsigned char* ss = stage_s(s);
            const int kc0 = c * KCHUNK;
            if (mode == 1) continue;
#pragma unroll
            for (int step = warp; step < C::KSTEPS; step += WARPS) {
                const int k0 = step * 32;
                uint32_t a0[4], a1[4];
                a0[0] = xa ? __ldg(reinterpret_cast<const unsigned int*>(xa + kc0 + k0)) : 0u;
                a0[1] = xb ? __ldg(reinterpret_cast<const unsigned int*>(xb + kc0 + k0)) : 0u;
                a0[2] = xa ? __ldg(reinterpret_cast<const unsigned int*>(xa + kc0 + k0 + 8)) : 0u;
                a0[3] = xb ? __ldg(reinterpret_cast<const unsigned int*>(xb + kc0 + k0 + 8)) : 0u;
                a1[0] = xa ? __ldg(reinterpret_cast<const unsigned int*>(xa + kc0 + k0 + 16)) : 0u;
                a1[1] = xb ? __ldg(reinterpret_cast<const unsigned int*>(xb + kc0 + k0 + 16)) : 0u;
                a1[2] = xa ? __ldg(reinterpret_cast<const unsigned int*>(xa + kc0 + k0 + 24)) : 0u;
                a1[3] = xb ? __ldg(reinterpret_cast<const unsigned int*>(xb + kc0 + k0 + 24)) : 0u;
#pragma unroll
                for (int j = 0; j < C::NTILES; ++j) {
                    const int r = 8 * j + bn;
                    uint32_t q = *reinterpret_cast<const uint32_t*>(ws + r * C::WROW + step * 16 + bq * 4);
                    const uint32_t sv = *reinterpret_cast<const uint16_t*>(ss + r * C::SROW + step * 2);
                    const uint32_t s0 = e4m3_scale_bf16x2(sv & 0xFFu), s1 = e4m3_scale_bf16x2(sv >> 8);
                    const uint32_t b00 = hmul2_bf16(e2m1x2_raw_bf16x2(q), s0);
                    const uint32_t b01 = hmul2_bf16(e2m1x2_raw_bf16x2(q << 4), s0);
                    const uint32_t b10 = hmul2_bf16(e2m1x2_raw_bf16x2(q << 8), s1);
                    const uint32_t b11 = hmul2_bf16(e2m1x2_raw_bf16x2(q << 12), s1);
                    mma_bf16_16816(acc[j], a0, b00, b01);
                    mma_bf16_16816(acc[j], a1, b10, b11);
                }
            }
        }

        if constexpr (!PIPELINE) cp_async_wait<0>();
        __syncthreads();
        // Live prefetched weight tiles must never alias reduction scratch.
        float* red = reinterpret_cast<float*>(
            smem_raw + (PIPELINE ? STAGES * C::STAGE_BYTES : 0));
        {
            float* mine = red + warp * (MT * NT);
            const int r0 = lane >> 2, cc = (lane & 3) * 2;
#pragma unroll
            for (int j = 0; j < C::NTILES; ++j) {
                const int n = 8 * j + cc;
                mine[r0 * NT + n] = acc[j][0];
                mine[r0 * NT + n + 1] = acc[j][1];
                mine[(r0 + 8) * NT + n] = acc[j][2];
                mine[(r0 + 8) * NT + n + 1] = acc[j][3];
            }
        }
        __syncthreads();
        const float g = gs[e] * EPI;
        const int p0 = pair0[slot];
        for (int i = tid; i < M * NT; i += C::THREADS) {
            const int m = i / NT, n = i - m * NT;
            if (n0 + n >= N) continue;
            float v = 0.0f;
#pragma unroll
            for (int wv = 0; wv < WARPS; ++wv) v += red[wv * (MT * NT) + m * NT + n];
            out[size_t(p0 + m) * N + n0 + n] = __float2bfloat16_rn(v * g);
        }
        __syncthreads();  // Finish reduction reads before its next use.
    }
    cp_async_wait<0>();
    __syncthreads();
}
}  // namespace tms::nvfp4_decode_sm120
