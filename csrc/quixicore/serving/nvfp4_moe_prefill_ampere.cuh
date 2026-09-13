// SPDX-License-Identifier: Apache-2.0
// NVFP4 (e2m1 weights, e4m3 group-16 scales, fp32 global scale) grouped MoE
// GEMM for PREFILL-sized batches on Ampere, reading the Marlin-packed expert
// weights in place.
//
// Why a second kernel next to Marlin: at 8K-token prefill chunks each expert
// sees ~226 rows per rank. Marlin's 64-row m-tile re-dequantizes every B
// fragment per 64 rows inside each warp, its 4-row-block accumulator tile
// pins 255 registers (8 warps per SM) and the w2 GEMM (K = 512) is dominated
// by its split-K reduction: ncu at M = 8128 showed the tensor pipe at 54%
// (w13) and 34% (w2) of peak with DRAM at 15-17% (2026-09-13).
//
// This kernel: CTA tile 128 x 128 x 64, 8 warps (4 along M x 2 along N, warp
// tile 32 x 64), bf16 mma.m16n8k16 with fp32 accumulation. Per stage the 256
// threads dequantize the packed 64k x 128n tile ONCE - each thread turns its
// int4 of Marlin words (exactly the words one Marlin lane would own) into the
// four scaled bf16x2 mma B fragments and writes them fragment-major to shared
// memory - and every warp then streams its B fragments with one 16 B LDS per
// n16 block per k16. The numerics are Marlin's: the e2m1 -> bf16 bit trick
// (value * 2^-126), the e4m3 scale decode (s * sf * 2^7), the fragment
// multiply in bf16, and the epilogue's fp32 global-scale (gs * 2^119 / sf),
// so the products are bit-identical to Marlin's and only the fp32 summation
// order differs.
//
// Layout contract (csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu,
// tests/glm5_next/test_nvfp4_marlin_layout.py): per expert the packed weight is
// int32 [K/16][N*2]; the 16k x 64n tile (kt, nt) is the 128 words at
// ((kt * N/64) + nt) * 128 and lane L owns words 4L..4L+3 (word j = n16
// block j); nibble i of a word is k = 16kt + 8(i%2) + 2(L%4) + i/4,
// n = 64nt + 16j + 8((i%4)/2) + L/4. Scales are uint8 [K/16][N] with the
// 64-wide Marlin scale permutation and the [0,2,1,3] byte pre-swizzle, so
// the 8 bytes at [g][64nt + 8(L/4)] decode straight into the four
// (frag_b0, frag_b1) scale pairs of lane L.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tms::nvfp4moe {

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int THREADS = 256;
constexpr int WM = 4, WN = 2;            // warp grid
constexpr int WROWS = BM / WM;           // 32 rows per warp
constexpr int WCOLS = BN / WN;           // 64 cols per warp
constexpr int MT = WROWS / 16;           // 2 m16 tiles per warp
constexpr int NT = WCOLS / 8;            // 8 n8 tiles per warp
constexpr int LDA = BK + 8;              // bf16 elements per A smem row (144 B)
constexpr int A_STAGE_BYTES = BM * LDA * 2;                 // 18432
constexpr int BF_STAGE_BYTES = (BK / 16) * (BN / 16) * 32 * 16;  // 16384 (4 kt x 8 j x 32 lanes x 16 B)
constexpr int BR_STAGE_BYTES = (BK / 16) * (BN / 64) * 128 * 4;  // 4096 packed words
constexpr int S_STAGE_BYTES = (BK / 16) * BN;                     // 512 scale bytes
constexpr int STAGE_BYTES = A_STAGE_BYTES + BF_STAGE_BYTES + BR_STAGE_BYTES + S_STAGE_BYTES;
constexpr int META_BYTES = BM * 4 + BM * 4;                      // sorted ids + row scales
template <int STAGES>
constexpr int smem_bytes() { return STAGES * STAGE_BYTES + META_BYTES; }

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Marlin's e2m1 -> bf16 "skip_flop" dequant: each nibble becomes
// value * 2^-126 (the sign lands on bit 15, e1 e0 m0 on bits 8..6).
// Out1 takes nibbles 3 and 7, Out2 (after << 4) nibbles 2 and 6; with the
// caller's << 8 pre-shift that yields frag[0] = (n0, n4), frag[1] = (n1, n5)
// for the low word and (n2, n6), (n3, n7) for the high one.
__device__ __forceinline__ void dequant_e2m1_bf16(unsigned q, unsigned& lo, unsigned& hi) {
    const unsigned out1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    q <<= 4;
    const unsigned out2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    lo = out2;   // frag_b[0]
    hi = out1;   // frag_b[1]
}
// Marlin's e4m3 scale bytes -> bf16x2 pair (bytes 0/2 -> first, 1/3 -> second).
__device__ __forceinline__ void dequant_e4m3_scales_bf16(unsigned q, unsigned& first, unsigned& second) {
    const unsigned out1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    q <<= 8;
    const unsigned out2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    first = out2;    // frag_s[0]
    second = out1;   // frag_s[1]
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
    return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ unsigned bf16_bcast(unsigned packed, int half) {
    const unsigned v = half ? (packed >> 16) : (packed & 0xFFFFu);
    return v | (v << 16);
}

// One expert-block of 128 sorted rows x 128 output columns.
//   A [rows][lda] bf16 (w13: hidden states, row = sorted_id / top_k;
//                       w2: intermediate, top_k = 1)
//   B int32 [E][K/16][N*2], S uint8 [E][K/16][N], G fp32 [E]
//   sorted_ids int32 (moe_align_block_size at block 128; padding = M_topk)
//   expert_ids int32 per 128-row block, num_post_padded int32[1]
//   C [M_topk][ldc] bf16 (row = sorted id); mul_topk folds topk_weights[sid]
//   into the row scale (Marlin's w2 contract).
template <int STAGES>
__global__ void __launch_bounds__(THREADS, STAGES == 2 ? 2 : 1) nvfp4_moe_gemm_kernel(
        const __nv_bfloat16* __restrict__ A, int lda,
        const int32_t* __restrict__ B, const uint8_t* __restrict__ S, const float* __restrict__ G,
        const int32_t* __restrict__ sorted_ids, const int32_t* __restrict__ expert_ids,
        const int32_t* __restrict__ num_post_padded, const float* __restrict__ topk_weights,
        __nv_bfloat16* __restrict__ C, int ldc, int M_topk, int top_k, int N, int K, int mul_topk) {
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int blk = blockIdx.y;
    if (blk * BM >= num_post_padded[0]) return;
    const int e = expert_ids[blk];
    if (e < 0) return;
    __nv_bfloat16* a_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    unsigned char* bf_smem = smem_raw + STAGES * A_STAGE_BYTES;
    unsigned char* br_smem = bf_smem + STAGES * BF_STAGE_BYTES;
    unsigned char* s_smem = br_smem + STAGES * BR_STAGE_BYTES;
    int* ids_smem = reinterpret_cast<int*>(s_smem + STAGES * S_STAGE_BYTES);
    float* rs_smem = reinterpret_cast<float*>(ids_smem + BM);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WM, wn = warp / WM;
    const int n0 = blockIdx.x * BN;
    const float gs = G[e];
    if (tid < BM) {
        const int sid = sorted_ids[blk * BM + tid];
        const bool ok = sid < M_topk;
        ids_smem[tid] = ok ? sid : -1;
        float rsc = gs;
        if (mul_topk && ok) rsc *= topk_weights[sid];
        rs_smem[tid] = rsc;
    }
    __syncthreads();

    const size_t NQ64 = N / 64;
    const int32_t* Be = B + size_t(e) * (K / 16) * (size_t(N) * 2);
    const uint8_t* Se = S + size_t(e) * (K / 16) * size_t(N);

    // Stage loads: A rows (4 x 16 B per thread), packed B (1 x 16 B), scales (threads 0..31).
    auto load_stage = [&](int s, int kk) {
        __nv_bfloat16* as = a_smem + s * BM * LDA;
#pragma unroll
        for (int i = 0; i < (BM * BK / 8) / THREADS; ++i) {
            const int c = tid + i * THREADS;
            const int r = c >> 3, c8 = (c & 7) * 8;
            const int sid = ids_smem[r];
            const bool ok = sid >= 0;
            const __nv_bfloat16* src = A + (ok ? size_t(sid / top_k) * lda : 0) + kk + c8;
            cp_async_16(as + r * LDA + c8, src, ok);
        }
        {
            const int kt = tid >> 6, nt = (tid >> 5) & 1, L = tid & 31;
            const int32_t* src = Be + ((size_t(kk / 16 + kt) * NQ64) + (n0 / 64 + nt)) * 128 + 4 * L;
            cp_async_16(br_smem + s * BR_STAGE_BYTES + tid * 16, src, true);
        }
        if (tid < 32) {
            const int kt = tid >> 3, off = (tid & 7) * 16;
            const uint8_t* src = Se + size_t(kk / 16 + kt) * N + n0 + off;
            cp_async_16(s_smem + s * S_STAGE_BYTES + kt * BN + off, src, true);
        }
    };
    // Dequantize the staged packed tile into fragment-major bf16 fragments.
    auto dequant_stage = [&](int s) {
        const int kt = tid >> 6, nt = (tid >> 5) & 1, L = tid & 31;
        const uint4 w = *reinterpret_cast<const uint4*>(br_smem + s * BR_STAGE_BYTES + tid * 16);
        const uint2 sw = *reinterpret_cast<const uint2*>(s_smem + s * S_STAGE_BYTES + kt * BN + 64 * nt + 8 * (L >> 2));
        unsigned fs[4];
        dequant_e4m3_scales_bf16(sw.x, fs[0], fs[1]);
        dequant_e4m3_scales_bf16(sw.y, fs[2], fs[3]);
        const unsigned words[4] = {w.x, w.y, w.z, w.w};
        uint4* dst = reinterpret_cast<uint4*>(bf_smem + s * BF_STAGE_BYTES) + (kt * (BN / 16) + nt * 4) * 32 + L;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned b00, b01, b10, b11;
            dequant_e2m1_bf16(words[j] << 8, b00, b01);   // frag_b0: n = 16j + L/4
            dequant_e2m1_bf16(words[j], b10, b11);        // frag_b1: n = 16j + 8 + L/4
            const unsigned s0 = bf16_bcast(fs[j], 0), s1 = bf16_bcast(fs[j], 1);
            b00 = bf16x2_mul(b00, s0); b01 = bf16x2_mul(b01, s0);
            b10 = bf16x2_mul(b10, s1); b11 = bf16x2_mul(b11, s1);
            dst[j * 32] = make_uint4(b00, b01, b10, b11);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.0f;

    const int steps = K / BK;
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < steps) load_stage(s, s * BK);
        cp_async_commit();
    }
    for (int step = 0; step < steps; ++step) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nxt = step + STAGES - 1;
            if (nxt < steps) load_stage(nxt % STAGES, nxt * BK);
            cp_async_commit();
        }
        const int cur = step % STAGES;
        dequant_stage(cur);
        __syncthreads();
        const __nv_bfloat16* as = a_smem + cur * BM * LDA;
        const uint4* bfs = reinterpret_cast<const uint4*>(bf_smem + cur * BF_STAGE_BYTES);
        unsigned a[2][MT][4];
        uint4 b[2][NT / 2];
        auto load_k16 = [&](int buf, int kk) {
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                const int r = wm * WROWS + i * 16 + (lane & 15);
                const int c = kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a[buf][i], as + r * LDA + c);
            }
#pragma unroll
            for (int j = 0; j < NT / 2; ++j)
                b[buf][j] = bfs[(kk * (BN / 16) + wn * (NT / 2) + j) * 32 + lane];
        };
        load_k16(0, 0);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            const int c = kk & 1;
            if (kk + 1 < BK / 16) load_k16(c ^ 1, kk + 1);
#pragma unroll
            for (int j = 0; j < NT / 2; ++j) {
#pragma unroll
                for (int i = 0; i < MT; ++i) {
                    mma_bf16_16816(acc[i][2 * j], a[c][i], b[c][j].x, b[c][j].y);
                    mma_bf16_16816(acc[i][2 * j + 1], a[c][i], b[c][j].z, b[c][j].w);
                }
            }
        }
    }
    cp_async_wait<0>();

    const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
        const int r0 = wm * WROWS + i * 16 + g;
        const int sid0 = ids_smem[r0], sid1 = ids_smem[r0 + 8];
        const float sc0 = rs_smem[r0], sc1 = rs_smem[r0 + 8];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int n = n0 + wn * WCOLS + j * 8 + t2;
            if (sid0 >= 0)
                *reinterpret_cast<__nv_bfloat162*>(C + size_t(sid0) * ldc + n) =
                    __floats2bfloat162_rn(acc[i][j][0] * sc0, acc[i][j][1] * sc0);
            if (sid1 >= 0)
                *reinterpret_cast<__nv_bfloat162*>(C + size_t(sid1) * ldc + n) =
                    __floats2bfloat162_rn(acc[i][j][2] * sc1, acc[i][j][3] * sc1);
        }
    }
}

}  // namespace tms::nvfp4moe
