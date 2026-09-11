#pragma once
// SPDX-License-Identifier: Apache-2.0
// Phase 4.2 candidate: paired K128 BF16 projections from the strided merged
// KDA input, producing [2, M, 2048] without input copies. Same cp.async,
// ldmatrix, FP32 MMA accumulation and reduction as bf16_decode_gemm.cuh.
// Parked after a 23-25% isolated win: only ~20 us across 34 layers, not yet
// worth a serving qualification. No serving dispatcher selects this candidate.
#include "quixicore/serving/bf16_decode_gemm.cuh"

namespace tms::kda_fg_b_decode {
using namespace tms::decode_gemm;
__global__ void __launch_bounds__(256) kda_fg_b_decode_kernel(
        const __nv_bfloat16* __restrict__ x,     // [M, K]
        const __nv_bfloat16* __restrict__ w,     // [N, K]
        __nv_bfloat16* __restrict__ out,                  // [M, N]
        int M, int x_stride) {
    constexpr int NT = 16, WARPS = 8, KCHUNK = 128, STAGES = 2;
    constexpr int N = 2048, K = 128;
    const int projection = blockIdx.y;
    x += projection * K;
    w += projection * N * K;
    out += projection * M * N;
    using C = Cfg<NT, WARPS, KCHUNK, STAGES>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n0 = blockIdx.x * NT;
    const int nchunks = K / KCHUNK;

    auto stage_x = [&](int s) { return smem + s * C::STAGE; };
    auto stage_w = [&](int s) { return smem + s * C::STAGE + C::X_TILE; };

    // Issue the cp.async loads of chunk c into stage s.
    auto load_chunk = [&](int c, int s) {
        const int k0 = c * KCHUNK;
        __nv_bfloat16* xs = stage_x(s);
        __nv_bfloat16* ws = stage_w(s);
        for (int i = tid; i < MT * C::KVEC; i += C::THREADS) {
            const int m = i / C::KVEC, v = i - m * C::KVEC;
            const bool valid = m < M;
            const __nv_bfloat16* src = x + size_t(valid ? m : 0) * x_stride + k0 + v * 8;
            cp_async16(smem_u32(xs + m * C::ROW + v * 8), src, valid);
        }
        for (int i = tid; i < NT * C::KVEC; i += C::THREADS) {
            const int r = i / C::KVEC, v = i - r * C::KVEC;
            const int n = min(n0 + r, N - 1);
            cp_async16(smem_u32(ws + r * C::ROW + v * 8), w + size_t(n) * K + k0 + v * 8, true);
        }
    };

    float acc[C::NTILES][4];
#pragma unroll
    for (int j = 0; j < C::NTILES; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    // ldmatrix addressing: lane supplies the row address of matrix (lane / 8).
    const int mat = lane >> 3, mrow = lane & 7;
    // A (x) tile: a0 = rows 0-7 / k 0-7, a1 = rows 8-15 / k 0-7, a2 = rows 0-7 / k 8-15, a3 = rows 8-15 / k 8-15.
    const int a_row = mrow + 8 * (mat & 1), a_col = 8 * (mat >> 1);
    // B (w) tile pair: r0 = tile 2p k 0-7, r1 = tile 2p k 8-15, r2 = tile 2p+1 k 0-7, r3 = tile 2p+1 k 8-15.
    const int b_row = mrow + 8 * (mat >> 1), b_col = 8 * (mat & 1);

    // Prologue: STAGES-1 chunks in flight.
#pragma unroll
    for (int c = 0; c < STAGES - 1; ++c) {
        if (c < nchunks) load_chunk(c, c);
        cp_async_commit();
    }

    for (int c = 0; c < nchunks; ++c) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();   // chunk c landed for everyone; stage (c-1)%STAGES is free
        {
            const int cn = c + STAGES - 1;
            if (cn < nchunks) load_chunk(cn, cn % STAGES);
            cp_async_commit();
        }
        const int s = c % STAGES;
        const uint32_t xs = smem_u32(stage_x(s));
        const uint32_t ws = smem_u32(stage_w(s));
#pragma unroll
        for (int step = warp; step < C::KSTEPS; step += WARPS) {
            const int k0 = step * 16;
            uint32_t a[4];
            ldmatrix_x4(a, xs + (a_row * C::ROW + k0 + a_col) * 2);
#pragma unroll
            for (int p = 0; p < C::NTILES / 2; ++p) {
                uint32_t b[4];
                ldmatrix_x4(b, ws + ((16 * p + b_row) * C::ROW + k0 + b_col) * 2);
                mma_bf16_16816(acc[2 * p], a, b[0], b[1]);
                mma_bf16_16816(acc[2 * p + 1], a, b[2], b[3]);
            }
            if constexpr (C::NTILES % 2 == 1) {
                // Last single n8 tile: x2 with lanes 0-15 addressing (k 0-7, k 8-15).
                uint32_t b[2];
                const int r = mrow, col = 8 * (mat & 1);
                ldmatrix_x2(b, ws + ((16 * (C::NTILES / 2) + r) * C::ROW + k0 + col) * 2);
                mma_bf16_16816(acc[C::NTILES - 1], a, b[0], b[1]);
            }
        }
    }

    // Cross-warp reduction of the K partials through shared memory.
    cp_async_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem_raw);   // [WARPS][MT][NT]
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
    for (int i = tid; i < MT * NT; i += C::THREADS) {
        const int m = i / NT, n = i - m * NT;
        if (m >= M || n0 + n >= N) continue;
        float v = 0.0f;
#pragma unroll
        for (int wv = 0; wv < WARPS; ++wv) v += red[wv * (MT * NT) + i];
        out[size_t(m) * N + n0 + n] = __float2bfloat16_rn(v);
    }
}

}  // namespace tms::kda_fg_b_decode
