/**
 * @file
 * @brief DSA lightning-indexer MQA logits (fp8 K), native CUDA.
 *
 * Replaces the DeepGEMM `fp8_fp4_mqa_logits` path, which is sm90+ only, and the
 * pure-torch fallback used below it. Ampere has no fp8 tensor cores and Triton
 * refuses `fp8e4nv` under sm89, but e4m3 is just bits: decode in software and
 * let the multiply ride fp32. Nothing here is arch-gated.
 *
 *   score[h,m,n] = (SUM_d q[m,h,d] * k[n,d]) * kscale[n]
 *   logits[m,n]  = SUM_h relu(score[h,m,n]) * weights[m,h]
 *   masked to ks[m] <= n < ke[m], else -inf
 *
 * The torch reference materializes `score` as an [H,M,N] fp32 tensor -- 8.6 GB
 * at H=64, M=2048, N=4096 -- which is why it is unusable at real context
 * lengths. This kernel keeps the h-reduction in registers and never forms it.
 *
 * Layout note: q is decoded once per block into a D-major shared tile
 * `q_f[d][h]`, so lane h reads consecutive addresses (bank-conflict free). An
 * h-major tile would put every lane on the same bank.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "quant_formats.cuh"   // tmq::e4m3_decode

namespace tms {

// One block owns a single query row m and a tile of n. blockDim.x == H.
template <int D, int NTILE>
__global__ void indexer_mqa_logits_fp8(
    const uint8_t* __restrict__ q,        // [M, H, D] e4m3
    const uint8_t* __restrict__ k,        // [N, D]    e4m3
    const float* __restrict__ kscale,     // [N]
    const float* __restrict__ weights,    // [M, H]
    const int* __restrict__ ks,           // [M]
    const int* __restrict__ ke,           // [M]
    float* __restrict__ logits,           // [M, N]
    int M, int N, int H) {
    extern __shared__ float smem[];
    float* q_f = smem;                    // [D][H], D-major
    float* k_f = smem + (size_t)D * H;    // [D]
    float* red = k_f + D;                 // [blockDim.x / 32]

    const int m = blockIdx.y;
    const int n0 = blockIdx.x * NTILE;
    if (m >= M || n0 >= N) return;

    const int h = threadIdx.x;
    const int lane = h & 31, warp = h >> 5, nwarps = blockDim.x >> 5;

    // Decode this row's q once; reused across every n in the tile.
    for (int idx = threadIdx.x; idx < D * H; idx += blockDim.x) {
        const int hh = idx % H, dd = idx / H;
        q_f[dd * H + hh] =
            tmq::e4m3_decode(q[((size_t)m * H + hh) * D + dd]);
    }

    const float w = (h < H) ? weights[(size_t)m * H + h] : 0.0f;
    const int lo = ks[m], hi = ke[m];
    __syncthreads();

    for (int t = 0; t < NTILE; ++t) {
        const int n = n0 + t;
        if (n >= N) break;

        for (int d = threadIdx.x; d < D; d += blockDim.x)
            k_f[d] = tmq::e4m3_decode(k[(size_t)n * D + d]);
        __syncthreads();

        float acc = 0.0f;
        if (h < H) {
#pragma unroll 8
            for (int d = 0; d < D; ++d) acc += q_f[d * H + h] * k_f[d];
            acc = fmaxf(acc * kscale[n], 0.0f) * w;   // relu then weight
        }

        // sum over heads
#pragma unroll
        for (int off = 16; off; off >>= 1)
            acc += __shfl_xor_sync(0xffffffffu, acc, off);
        if (lane == 0) red[warp] = acc;
        __syncthreads();
        if (threadIdx.x == 0) {
            float s = 0.0f;
            for (int i = 0; i < nwarps; ++i) s += red[i];
            logits[(size_t)m * N + n] =
                (n >= lo && n < hi) ? s : -INFINITY;
        }
        __syncthreads();
    }
}

// smem bytes for indexer_mqa_logits_fp8<D, NTILE> at the given H and blockDim
__host__ __device__ constexpr size_t indexer_mqa_logits_smem(int D, int H,
                                                             int threads) {
    return ((size_t)D * H + D + (threads / 32)) * sizeof(float);
}

}  // namespace tms
