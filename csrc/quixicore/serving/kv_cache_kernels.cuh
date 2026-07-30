#pragma once
// Paged KV cache machinery + fused v1 decode attention, CUDA/SM86 port of
// ThunderMittens kernels/kv_cache (core set; fp8 variants + beam kernels follow).
//
// Paged layout (the canonical one all serving kernels share, vLLM-compatible):
//   key_cache/value_cache: (num_blocks, block_size, num_kv_heads, D) contiguous
//   block_table: (batch, max_blocks) int32, block < 0 = unmapped (skipped)
//   slot_mapping: (num_tokens,) int64, slot < 0 = padding (skipped)
//   slot -> address: ((block*block_size + offset)*num_kv_heads + kv_head)*D
//
// paged_attention: one warp per (query head, batch); lane owns D/32 elements;
// online softmax over the context; fused GQA + ALiBi + sliding window +
// block-sparse mask (mask shares the block_table layout).
//
// Build (test harness):
//   /usr/local/cuda/bin/nvcc kv_cache.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o kv_cache.out
#include <cuda_fp16.h>
#include "tm_warp.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {


template <typename T>
__global__ void kv_cache_zero(T* key_cache, T* value_cache, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    key_cache[i] = T(0);
    value_cache[i] = T(0);
}

// one block per token; threads stride the H_KV*D row
template <typename T>
__global__ void kv_cache_scatter(const T* key, const T* value, const int64_t* slot_mapping,
                                 T* key_cache, T* value_cache,
                                 int num_heads, int head_size, int block_size) {
    const int token = blockIdx.x;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    const int64_t block = slot / block_size, off = slot % block_size;
    const int row_elems = num_heads * head_size;
    const int64_t src = int64_t(token) * row_elems;
    const int64_t dst = (block * block_size + off) * row_elems;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        key_cache[dst + i] = key[src + i];
        value_cache[dst + i] = value[src + i];
    }
}

// paged -> packed; binary search cu_seq_lens for the owning batch
template <typename T>
__global__ void kv_cache_gather(const T* key_cache, const T* value_cache,
                                T* key_out, T* value_out,
                                const int* block_table, const int* cu_seq_lens,
                                int num_tokens, int num_seqs, int block_size,
                                int bt_stride, int num_heads, int head_size) {
    const int token = blockIdx.x;
    if (token >= num_tokens) return;
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= token) lo = mid; else hi = mid - 1;
    }
    const int batch = lo, local = token - cu_seq_lens[batch];
    const int col = local / block_size, slot = local % block_size;
    const int block = block_table[batch * bt_stride + col];
    const int row_elems = num_heads * head_size;
    const int64_t out = int64_t(token) * row_elems;
    if (block < 0) {
        for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
            key_out[out + i] = T(0);
            value_out[out + i] = T(0);
        }
        return;
    }
    const int64_t src = (int64_t(block) * block_size + slot) * row_elems;
    for (int i = threadIdx.x; i < row_elems; i += blockDim.x) {
        key_out[out + i] = key_cache[src + i];
        value_out[out + i] = value_cache[src + i];
    }
}

template <typename T>
__global__ void kv_cache_clone(const T* key_cache, const T* value_cache,
                               T* key_out, T* value_out, size_t n) {
    const size_t i = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 4;
    if (i + 4 <= n) {
        *reinterpret_cast<float2*>(key_out + i)   = *reinterpret_cast<const float2*>(key_cache + i);
        *reinterpret_cast<float2*>(value_out + i) = *reinterpret_cast<const float2*>(value_cache + i);
    } else {
        for (size_t j = i; j < n; j++) { key_out[j] = key_cache[j]; value_out[j] = value_cache[j]; }
    }
}

// src->dst block copies: reads ORIGINAL, writes CLONE (race-free beam-reorder chains)
template <typename T>
__global__ void kv_cache_copy_blocks(const T* key_src, const T* value_src,
                                     T* key_dst, T* value_dst,
                                     const int64_t* block_mapping,   // (num_pairs, 2), -1 sentinel
                                     int block_elems) {
    const int pair = blockIdx.x;
    const int64_t src = block_mapping[2 * pair], dst = block_mapping[2 * pair + 1];
    if (src < 0 || dst < 0) return;
    const int64_t s = src * block_elems, d = dst * block_elems;
    for (int i = threadIdx.x; i < block_elems; i += blockDim.x) {
        key_dst[d + i] = key_src[s + i];
        value_dst[d + i] = value_src[s + i];
    }
}

// ---- fused v1 decode: one warp per (head, batch); GQA + ALiBi + window + block-sparse ----
template <typename T, int D>
__global__ void paged_attention(const T* q, const T* key_cache, const T* value_cache,
                                const int* block_table, const int* context_lens, T* out,
                                int block_size, int bt_stride, float scale,
                                int num_heads, int num_kv_heads,
                                const float* alibi_slopes, int use_alibi,
                                const int* block_mask, int use_mask, int window) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int64_t row = (int64_t(batch) * num_heads + head) * D;
    const int t_start = (window > 0) ? max(0, context_len - window) : 0;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) {
        qv[i] = float(q[row + lane + 32 * i]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = t_start; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        if (use_mask && block_mask[batch * bt_stride + col] == 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D + int64_t(kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * float(key_cache[base + lane + 32 * i]);
        float score = warp_sum_f(partial) * scale;
        if (use_alibi) score += alibi_slopes[head] * float(t - context_len + 1);
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * float(value_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[row + lane + 32 * i] = (l == 0.0f) ? T(0) : T(acc[i] / l);
}

// GQA KV-reuse staged decode (TM kv_cache.metal 410-489, flashinfer decode
// structure): one block per (kv_head, batch) with group_size warps — one query
// head each. Every KV token vector is staged into shared memory ONCE and
// reused by all query heads of the kv_head, amortizing cache bandwidth by
// group_size. Math is identical (same op order) to paged_attention above, so
// outputs are bit-for-bit equal; only the traffic shape differs. Launch:
// grid (num_kv_heads, B), block 32*group_size.
template <typename T, int D>
__global__ void paged_attention_gqa_staged(const T* q, const T* key_cache, const T* value_cache,
                                           const int* block_table, const int* context_lens,
                                           T* out, int block_size, int bt_stride, float scale,
                                           int num_heads, int num_kv_heads) {
    constexpr int VPL = D / 32;
    __shared__ float sh_k[D], sh_v[D];
    const int kv_head = blockIdx.x, batch = blockIdx.y;
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const int group_size = num_heads / num_kv_heads;
    const int head = kv_head * group_size + warp;
    const int context_len = context_lens[batch];
    const int64_t row = (int64_t(batch) * num_heads + head) * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) {
        qv[i] = float(q[row + lane + 32 * i]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        __syncthreads();                                    // prior iter done reading sh_k/sh_v
        if (block >= 0) {
            const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D
                               + int64_t(kv_head) * D;
            for (int idx = threadIdx.x; idx < D; idx += blockDim.x) {
                sh_k[idx] = float(key_cache[base + idx]);
                sh_v[idx] = float(value_cache[base + idx]);
            }
        }
        __syncthreads();
        if (block < 0) continue;

        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * sh_k[lane + 32 * i];
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++) acc[i] = acc[i] * alpha + beta * sh_v[lane + 32 * i];
        l = l * alpha + beta;
        m = nm;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[row + lane + 32 * i] = (l == 0.0f) ? T(0) : T(acc[i] / l);
}

// Dense sliding-window causal attention (TM attn_causal.metal attn_window,
// Mistral/Gemma local attention): query i attends keys j in [max(0, i-W+1), i]
// (window_size_left = W-1, right = 0); window <= 0 means full causal. Layout
// (B, H, N, D) contiguous, one warp per (b, h, i) query row, online softmax —
// the warp-scalar twin of TM's tile kernel (make_windowed band edge == the
// loop bounds here). The TK mha_ampere tile version is a perf-pass item.
template <typename T, int D>
__global__ void attn_window(const T* q, const T* k, const T* v, T* out,
                            int N, float scale, int window) {
    constexpr int VPL = D / 32;
    const int i = blockIdx.x;                       // query position
    const int bh = blockIdx.y;                      // fused (batch, head)
    const int lane = threadIdx.x;
    const int64_t base = int64_t(bh) * N * D;
    const int j_lo = (window > 0) ? max(0, i - window + 1) : 0;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int d = 0; d < VPL; d++) {
        qv[d] = float(q[base + int64_t(i) * D + lane + 32 * d]);
        acc[d] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;
    for (int j = j_lo; j <= i; j++) {
        float partial = 0.0f;
        #pragma unroll
        for (int d = 0; d < VPL; d++)
            partial += qv[d] * float(k[base + int64_t(j) * D + lane + 32 * d]);
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int d = 0; d < VPL; d++)
            acc[d] = acc[d] * alpha + beta * float(v[base + int64_t(j) * D + lane + 32 * d]);
        l = l * alpha + beta;
        m = nm;
    }
    #pragma unroll
    for (int d = 0; d < VPL; d++)
        out[base + int64_t(i) * D + lane + 32 * d] = (l == 0.0f) ? T(0) : T(acc[d] / l);
}

}  // namespace tms

