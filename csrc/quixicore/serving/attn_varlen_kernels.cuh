#pragma once
// Varlen/paged prefill attention + device-side scheduling, CUDA/SM86 port of
// ThunderMittens kernels/attn_varlen:
//   block_exclusive_scan_i32 - single-block chunked exclusive prefix scan (the
//     primitive W5's spec_compact and MoE offsets reuse)
//   varlen_build_worklist    - device cu_seqlens -> qlens, pad offsets, and an
//     8-row tile worklist (tile_seq/tile_local0, -1 sentinels; grid sized from a
//     host upper bound because a grid can't be sized from device data)
//   varlen_q_pad_gather / varlen_o_regather - packed (total_q,H,D) <-> head-major
//     padded (H,total_padded,D)
//   attn_varlen_prefill      - causal ragged prefill reading K/V straight from the
//     paged cache, with prefix (context_len >= q_len) and GQA. One block per
//     worklist tile, 8 warps = 8 query rows (warp-per-row; TM's 8-row mma tiling
//     is the perf pass).
//
// Build:
//   /usr/local/cuda/bin/nvcc attn_varlen.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o attn_varlen.out
#include <cuda_fp16.h>
#include "tm_warp.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {


__global__ void varlen_build_worklist(const int* cu_seqlens, int* qlens, int* pad_off,
                                      int* tile_seq, int* tile_local0, int* n_tiles,
                                      int B, int max_tiles) {
    __shared__ int sg_sums[8];
    const unsigned tid = threadIdx.x, nthreads = blockDim.x;
    const int chunk = (B + int(nthreads) - 1) / int(nthreads);
    int lo = int(tid) * chunk; if (lo > B) lo = B;
    int hi = lo + chunk;       if (hi > B) hi = B;
    int local_tiles = 0, local_pad = 0;
    for (int b = lo; b < hi; b++) {
        const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
        qlens[b] = qlen;
        const int nt = (qlen + 7) / 8;
        local_tiles += nt;
        local_pad += nt * 8;
    }
    int total_tiles = 0;
    const int base_tile = block_exclusive_scan_i32(local_tiles, tid, nthreads, sg_sums, total_tiles);
    __syncthreads();                               // reuse sg_sums for scan #2
    int total_padded = 0;
    const int base_pad = block_exclusive_scan_i32(local_pad, tid, nthreads, sg_sums, total_padded);
    if (tid == 0) { pad_off[B] = total_padded; n_tiles[0] = total_tiles; }
    int run_tile = base_tile, run_pad = base_pad;
    for (int b = lo; b < hi; b++) {
        const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
        const int nt = (qlen + 7) / 8;
        pad_off[b] = run_pad;
        for (int t = 0; t < nt; t++) {
            tile_seq[run_tile + t] = b;
            tile_local0[run_tile + t] = t * 8;
        }
        run_tile += nt;
        run_pad += nt * 8;
    }
    for (int i = int(tid); i < max_tiles; i += int(nthreads))
        if (i >= total_tiles) { tile_seq[i] = -1; tile_local0[i] = 0; }
}

// packed (total_q, H, D) -> head-major padded (H, total_padded, D); 1 block per (padded pos)
template <typename T, int D>
__global__ void varlen_q_pad_gather(const T* q, T* q_hm, const int* cu_seqlens,
                                    const int* pad_off, int B, int H, int total_padded) {
    const int p = blockIdx.x;
    // find batch owning padded position p
    int lo = 0, hi = B;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (pad_off[mid] <= p) lo = mid; else hi = mid - 1;
    }
    const int b = lo, local = p - pad_off[b];
    const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
    for (int hd = threadIdx.x; hd < H * D; hd += blockDim.x) {
        const int h = hd / D, d = hd % D;
        const T v = (local < qlen) ? q[(int64_t(cu_seqlens[b] + local) * H + h) * D + d] : T(0);
        q_hm[(int64_t(h) * total_padded + p) * D + d] = v;
    }
}

// inverse: head-major padded -> packed
template <typename T, int D>
__global__ void varlen_o_regather(const T* o_hm, T* o, const int* cu_seqlens,
                                  const int* pad_off, int B, int H, int total_padded) {
    const int p = blockIdx.x;
    int lo = 0, hi = B;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (pad_off[mid] <= p) lo = mid; else hi = mid - 1;
    }
    const int b = lo, local = p - pad_off[b];
    const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
    if (local >= qlen) return;
    for (int hd = threadIdx.x; hd < H * D; hd += blockDim.x) {
        const int h = hd / D, d = hd % D;
        o[(int64_t(cu_seqlens[b] + local) * H + h) * D + d] =
            o_hm[(int64_t(h) * total_padded + p) * D + d];
    }
}

// causal ragged prefill over the paged cache. grid (max_tiles, H); block = 8 warps,
// warp w = query row tile_local0[gx] + w; query at local ql attends kv [0, past + ql].
template <typename T, int D>
__global__ void attn_varlen_prefill(const T* q, const T* key_cache, const T* value_cache,
                                    const int* block_table, const int* context_lens,
                                    const int* cu_seqlens, const int* qlens,
                                    const int* tile_seq, const int* tile_local0,
                                    T* out, int block_size, int bt_stride, float scale,
                                    int num_heads, int num_kv_heads) {
    constexpr int VPL = D / 32;
    const int gx = blockIdx.x, head = blockIdx.y;
    const int b = tile_seq[gx];
    if (b < 0) return;                            // sentinel tile
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int ql = tile_local0[gx] + warp;        // local query row
    if (ql >= qlens[b]) return;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int past = context_lens[b] - qlens[b];  // prefix length (context >= qlen)
    const int t_end = past + ql + 1;              // causal bound in KV positions

    const int64_t q_row = (int64_t(cu_seqlens[b] + ql) * num_heads + head) * D;
    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_row + lane + 32 * i]); acc[i] = 0.0f; }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < t_end; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[b * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D + int64_t(kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * float(key_cache[base + lane + 32 * i]);
        const float score = warp_sum_f(partial) * scale;
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
        out[q_row + lane + 32 * i] = (l == 0.0f) ? T(0) : T(acc[i] / l);
}

}  // namespace tms

