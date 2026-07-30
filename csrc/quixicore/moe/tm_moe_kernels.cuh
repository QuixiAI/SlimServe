/**
 * @file
 * @brief CUDA/SM86 port of ThunderMittens' MoE family (moe.metal): routing
 * top-k, permute pipeline (histogram/scan/scatter), 32-row padded schedule
 * (vLLM moe_align pattern), gather, grouped expert GEMMs (square, rect, fused
 * SwiGLU) and the atomic-free weighted finalize.
 *
 * Faithful math/layout port. The grouped GEMMs here are scalar-FMA fp32
 * kernels (one 256-thread block per 32x32 output tile, expert_of_tile<0
 * early-exit) — correctness-first; the perf pass routes them through the
 * 44-TFLOP mma GEMM. Reuses tm_warp.cuh masked_topk (smaller-id ties) and
 * block_exclusive_scan_i32 — the same helpers the samplers/varlen kernels
 * were validated with.
 *
 * Layout contract (identical to TM):
 *   topk_ids/weights (T, K); routing row r in [0,T*K) = (token r/K, slot r%K)
 *   offsets (E+1) exclusive; sorted_row_idx (TK); inv_idx (TK)
 *   off_pad (E+1) 32-padded; expert_of_tile (max_tiles) -1 sentinel;
 *   gather_idx (total_pad_max) -1 -> zero row; inv_pad (TK)
 *   W (E, H, H) / W1 (E, H, 2*inter) [gate|up] / W2 (E, inter, H)
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "../serving/tm_warp.cuh"

namespace tmoe {

using tms::masked_topk;
using tms::block_exclusive_scan_i32;

#define MOE_NEG_INF (-3.4028234663852886e38f)
#define MOE_MAX_K 16
#define MOE_SCAN_NT 256

template <typename T> __device__ __forceinline__ float mtf(T v);
template <> __device__ __forceinline__ float mtf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float mtf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float mtf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T mft(float v);
template <> __device__ __forceinline__ float          mft<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         mft<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  mft<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Top-k experts per token (K masked-argmax rounds) + renormalized softmax
// weights over the selected logits (the Mixtral rule). One warp per token.
template <typename T>
__global__ void moe_route_topk(const T* __restrict__ logits, int* __restrict__ topk_ids,
                               float* __restrict__ topk_weights, int E, int K) {
    const long base = (long)blockIdx.x * E;
    const int lane = threadIdx.x;
    int chosen_id[MOE_MAX_K];
    float chosen_logit[MOE_MAX_K];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = idx; v = mtf(logits[base + idx]); valid = true;
    };
    masked_topk(cand, E, K, lane, MOE_NEG_INF, chosen_id, chosen_logit);

    float m = MOE_NEG_INF;
    for (int k = 0; k < K; ++k) m = fmaxf(m, chosen_logit[k]);
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) sum += expf(chosen_logit[k] - m);
    const float inv = 1.0f / sum;
    if (lane == 0) {
        const long ob = (long)blockIdx.x * K;
        for (int k = 0; k < K; ++k) {
            topk_ids[ob + k] = chosen_id[k];
            topk_weights[ob + k] = expf(chosen_logit[k] - m) * inv;
        }
    }
}

__global__ void moe_histogram(const int* __restrict__ topk_ids, int* __restrict__ counts,
                              int TK) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TK) return;
    atomicAdd(counts + topk_ids[tid], 1);
}

// Exclusive prefix sum of counts -> offsets (E+1) + scatter cursor seed.
// One block of MOE_SCAN_NT threads; running prefix across tiles supports any E.
__global__ void moe_scan_offsets(const int* __restrict__ counts, int* __restrict__ offsets,
                                 int* __restrict__ cursor, int E) {
    __shared__ int sg_sums[MOE_SCAN_NT / 32];
    __shared__ int running;
    const int tid = threadIdx.x;
    if (tid == 0) running = 0;
    __syncthreads();
    for (int b = 0; b < E; b += MOE_SCAN_NT) {
        const int e = b + tid;
        const int v = (e < E) ? counts[e] : 0;
        int total;
        const int excl = block_exclusive_scan_i32(v, tid, MOE_SCAN_NT, sg_sums, total);
        if (e < E) {
            offsets[e] = running + excl;
            cursor[e] = running + excl;
        }
        __syncthreads();
        if (tid == 0) running += total;
        __syncthreads();
    }
    if (tid == 0) offsets[E] = running;
}

__global__ void moe_scatter(const int* __restrict__ topk_ids, int* __restrict__ cursor,
                            int* __restrict__ sorted_row_idx, int* __restrict__ inv_idx,
                            int TK) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TK) return;
    const int pos = atomicAdd(cursor + topk_ids[tid], 1);
    sorted_row_idx[pos] = tid;
    inv_idx[tid] = pos;
}

// Padded schedule: off_pad = exclusive scan of ceil32(counts); expert_of_tile
// via binary search over off_pad (-1 beyond the real total); gather_idx -> -1.
// Single block.
__global__ void moe_pad_offsets(const int* __restrict__ offsets, int* __restrict__ off_pad,
                                int* __restrict__ expert_of_tile, int* __restrict__ gather_idx,
                                int E, int max_tiles, int total_pad_max) {
    __shared__ int sg_sums[MOE_SCAN_NT / 32];
    __shared__ int running;
    const int tid = threadIdx.x;
    if (tid == 0) running = 0;
    __syncthreads();
    for (int b = 0; b < E; b += MOE_SCAN_NT) {
        const int e = b + tid;
        const int count = (e < E) ? (offsets[e + 1] - offsets[e]) : 0;
        const int padded = ((count + 31) / 32) * 32;
        int total;
        const int excl = block_exclusive_scan_i32(padded, tid, MOE_SCAN_NT, sg_sums, total);
        if (e < E) off_pad[e] = running + excl;
        __syncthreads();
        if (tid == 0) running += total;
        __syncthreads();
    }
    if (tid == 0) off_pad[E] = running;
    __syncthreads();

    const int total_pad = off_pad[E];
    for (int t = tid; t < max_tiles; t += MOE_SCAN_NT) {
        const int pos = t * 32;
        if (pos >= total_pad) { expert_of_tile[t] = -1; continue; }
        int lo = 0, hi = E;              // largest e with off_pad[e] <= pos
        while (hi - lo > 1) {
            const int mid = (lo + hi) / 2;
            if (off_pad[mid] <= pos) lo = mid; else hi = mid;
        }
        expert_of_tile[t] = lo;
    }
    for (int p = tid; p < total_pad_max; p += MOE_SCAN_NT) gather_idx[p] = -1;
}

// Compact position p -> padded position; record gather (padpos -> token) and
// inv_pad (routing row -> padpos, what finalize reads).
__global__ void moe_pad_scatter(const int* __restrict__ sorted_row_idx,
                                const int* __restrict__ offsets, const int* __restrict__ off_pad,
                                int* __restrict__ gather_idx, int* __restrict__ inv_pad,
                                int TK, int E, int K) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= TK) return;
    int lo = 0, hi = E;                  // expert whose compact segment contains p
    while (hi - lo > 1) {
        const int mid = (lo + hi) / 2;
        if (offsets[mid] <= p) lo = mid; else hi = mid;
    }
    const int padpos = off_pad[lo] + (p - offsets[lo]);
    const int r = sorted_row_idx[p];
    gather_idx[padpos] = r / K;
    inv_pad[r] = padpos;
}

// permuted_input[p, :] = x[gather_idx[p], :] (zeros for pad rows).
// One 128-thread block per padded row.
template <typename T>
__global__ void moe_gather(const T* __restrict__ x, const int* __restrict__ gather_idx,
                           T* __restrict__ out, int H) {
    const long p = blockIdx.x;
    const int src = gather_idx[p];
    T* dst = out + p * H;
    if (src < 0) {
        for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = mft<T>(0.0f);
        return;
    }
    const T* row = x + (long)src * H;
    for (int i = threadIdx.x; i < H; i += blockDim.x) dst[i] = row[i];
}

// ---- grouped (segmented) expert GEMMs, one 256-thread block per 32x32 output
// tile; every tile belongs to exactly one expert (32-row padding), tiles with
// expert_of_tile < 0 exit (their rows are never read downstream). ----
template <typename T>
__global__ void moe_grouped_gemm_rect(T* __restrict__ out, const T* __restrict__ A,
                                      const T* __restrict__ W,
                                      const int* __restrict__ expert_of_tile,
                                      int total_rows, int K_dim, int N_out) {
    const int OY = blockIdx.y, OX = blockIdx.x;
    const int e = expert_of_tile[OY];
    if (e < 0) return;
    const T* We = W + (long)e * K_dim * N_out;
    // thread owns 4 of the tile's 1024 elements
    for (int t = threadIdx.x; t < 32 * 32; t += blockDim.x) {
        const int r = OY * 32 + t / 32;
        const int c = OX * 32 + t % 32;
        if (r >= total_rows || c >= N_out) continue;
        float acc = 0.0f;
        const T* arow = A + (long)r * K_dim;
        for (int k = 0; k < K_dim; ++k)
            acc += mtf(arow[k]) * mtf(We[(long)k * N_out + c]);
        out[(long)r * N_out + c] = mft<T>(acc);
    }
}

// TM's square moe_grouped_gemm (W (E,H,H)) is the K_dim==N_out==H case of
// rect; hosts dispatch moe_grouped_gemm_rect for both.

// Fused SiLU-GLU GEMM1: out = silu(A @ W1_gate) * (A @ W1_up); W1[e] is
// (H, 2*inter) laid out [gate | up].
template <typename T>
__global__ void moe_grouped_gemm_swiglu(T* __restrict__ out, const T* __restrict__ A,
                                        const T* __restrict__ W1,
                                        const int* __restrict__ expert_of_tile,
                                        int total_rows, int H, int inter) {
    const int OY = blockIdx.y, OX = blockIdx.x;
    const int e = expert_of_tile[OY];
    if (e < 0) return;
    const T* We = W1 + (long)e * H * (2 * inter);
    for (int t = threadIdx.x; t < 32 * 32; t += blockDim.x) {
        const int r = OY * 32 + t / 32;
        const int c = OX * 32 + t % 32;
        if (r >= total_rows || c >= inter) continue;
        float gate = 0.0f, up = 0.0f;
        const T* arow = A + (long)r * H;
        for (int k = 0; k < H; ++k) {
            const float a = mtf(arow[k]);
            gate += a * mtf(We[(long)k * (2 * inter) + c]);
            up   += a * mtf(We[(long)k * (2 * inter) + inter + c]);
        }
        const float s = gate / (1.0f + expf(-gate));   // silu
        out[(long)r * inter + c] = mft<T>(s * up);
    }
}

// Per-token weighted k-way reduce of the (permuted-order) expert outputs via
// the inverse map — no atomics. One warp per token. inv is inv_idx (compact)
// or inv_pad (padded schedule); expert_out rows indexed by it either way.
template <typename T>
__global__ void moe_finalize(const T* __restrict__ expert_out, const int* __restrict__ inv,
                             const float* __restrict__ topk_weights, T* __restrict__ out,
                             int K, int Hdim) {
    const int token = blockIdx.x;
    const long wbase = (long)token * K;
    const long obase = (long)token * Hdim;
    for (int h = threadIdx.x; h < Hdim; h += 32) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            const int pos = inv[token * K + k];
            acc += topk_weights[wbase + k] * mtf(expert_out[(long)pos * Hdim + h]);
        }
        out[obase + h] = mft<T>(acc);
    }
}

} // namespace tmoe
