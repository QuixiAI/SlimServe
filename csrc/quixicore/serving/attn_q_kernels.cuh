#pragma once
// Quantized-KV attention (prefill), CUDA/SM86 port of ThunderMittens kernels/attn_q.
// Q stays fp16; K and V arrive BLOCKWISE-QUANTIZED (quant.py quantize_kv layout:
// (B, H, N, D/block_k, block_bytes) uint8) and are dequantized on the fly inside
// the attention loop - the KV cache memory win without a separate dequant pass.
// Formats: any tmq format with block_k <= D (TM ships q8_0/q4_0/fp8_e4m3).
//
// Correctness-first shape: one warp per query row, online softmax over keys
// (causal or full), per-lane d-strided dequant dot. (TM's 8-row tile/mma version
// is the perf pass; this matches the thrice-validated decode-kernel shape.)
//
// Build:
//   /usr/local/cuda/bin/nvcc attn_q.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -I../quant -o attn_q.out
#include "quant_formats.cuh"
#include "tm_warp.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace tmq;

namespace tms {


// grid (N, H, B), one warp per query row
template <typename FMT, int D, bool CAUSAL>
__global__ void attn_q(const half* q, const uint8_t* Kq, const uint8_t* Vq, half* out,
                       int N, int num_heads, float scale) {
    constexpr int VPL = D / 32;
    constexpr int BPR = D / FMT::block_k;          // quant blocks per (token) row
    const int qi = blockIdx.x, head = blockIdx.y, batch = blockIdx.z, lane = threadIdx.x;
    const int64_t bh = (int64_t(batch) * num_heads + head);
    const int64_t q_base = (bh * N + qi) * D;
    const uint8_t* Krow0 = Kq + bh * N * BPR * FMT::block_bytes;
    const uint8_t* Vrow0 = Vq + bh * N * BPR * FMT::block_bytes;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = -3.4028234663852886e38f, l = 0.0f;

    const int t_end = CAUSAL ? qi + 1 : N;
    for (int t = 0; t < t_end; t++) {
        const uint8_t* krow = Krow0 + int64_t(t) * BPR * FMT::block_bytes;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            partial += qv[i] * FMT::dequant(krow + (d / FMT::block_k) * FMT::block_bytes, d % FMT::block_k);
        }
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        const uint8_t* vrow = Vrow0 + int64_t(t) * BPR * FMT::block_bytes;
        #pragma unroll
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            acc[i] = acc[i] * alpha + beta * FMT::dequant(vrow + (d / FMT::block_k) * FMT::block_bytes, d % FMT::block_k);
        }
        l = l * alpha + beta;
        m = nm;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[q_base + lane + 32 * i] = (l == 0.0f) ? __float2half(0.0f) : __float2half(acc[i] / l);
}

}  // namespace tms

