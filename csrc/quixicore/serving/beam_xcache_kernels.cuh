#pragma once
// Beam-search KV reorder builders + vLLM x-packed cache decode, CUDA/SM86 port
// of the remaining ThunderMittens kv_cache kernels:
//   beam_build_copy_pairs  - device-side (src,dst) block-pair list for the beam KV
//                            reorder (atomic-free, fixed shape, -1 sentinels)
//   beam_remap_block_table - zero-copy reorder: child rows point at parent blocks
//   kv_cache_scales        - per-tensor absmax/240 K,V scales (shared-mem reduce)
//   paged_attention_xcache - decode reading vLLM's x-packed layout directly:
//                            key (nb, HKV, D/x, bs, x), value (nb, HKV, D, bs)
//
// Build:
//   /usr/local/cuda/bin/nvcc beam_xcache.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o beam_xcache.out
#include <cuda_fp16.h>
#include "tm_warp.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {


__global__ void beam_build_copy_pairs(const int* parent_beam, const int* block_table,
                                      const int* seq_lens, int64_t* pairs,
                                      int BM, int max_blocks, int block_size, int n_slots) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_slots) return;
    const int gb = gid / max_blocks, c = gid % max_blocks;
    const int b = gb / BM, k = gb % BM;
    int64_t src = -1, dst = -1;
    const int p = parent_beam[b * BM + k];
    if (p != k) {                                   // beam kept its own history -> no move
        const int nblk = (seq_lens[gb] + block_size - 1) / block_size;
        if (c < nblk) {
            const int s = block_table[(b * BM + p) * max_blocks + c];
            const int d = block_table[gb * max_blocks + c];
            if (s >= 0 && d >= 0) { src = s; dst = d; }
        }
    }
    pairs[2 * int64_t(gid)] = src;
    pairs[2 * int64_t(gid) + 1] = dst;
}

__global__ void beam_remap_block_table(const int* block_table, const int* parent_beam,
                                       int* new_block_table, int BM, int max_blocks) {
    const int row = blockIdx.x;
    const int b = row / BM;
    const int parent = parent_beam[row];
    const int64_t dst = int64_t(row) * max_blocks;
    const int64_t src = int64_t(b * BM + parent) * max_blocks;
    for (int c = threadIdx.x; c < max_blocks; c += blockDim.x)
        new_block_table[dst + c] = block_table[src + c];
}

template <typename T>
__global__ void kv_cache_scales(const T* key, const T* value,
                                float* key_scale, float* value_scale, size_t n) {
    __shared__ float sk[256], sv[256];
    const int tid = threadIdx.x;
    float km = 0.0f, vm = 0.0f;
    for (size_t i = tid; i < n; i += 256) {
        km = fmaxf(km, fabsf(float(key[i])));
        vm = fmaxf(vm, fabsf(float(value[i])));
    }
    sk[tid] = km; sv[tid] = vm;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
        if (tid < s) { sk[tid] = fmaxf(sk[tid], sk[tid + s]); sv[tid] = fmaxf(sv[tid], sv[tid + s]); }
        __syncthreads();
    }
    if (tid == 0) { key_scale[0] = sk[0] / 240.0f; value_scale[0] = sv[0] / 240.0f; }
}

// decode over vLLM's x-packed layout (read a vLLM cache directly, no repack)
template <typename T, int D>
__global__ void paged_attention_xcache(const T* q, const T* key_cache, const T* value_cache,
                                       const int* block_table, const int* context_lens, T* out,
                                       int block_size, int bt_stride, float scale,
                                       int num_heads, int num_kv_heads, int x) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int64_t row = (int64_t(batch) * num_heads + head) * D;
    const int dh = D / x;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[row + lane + 32 * i]); acc[i] = 0.0f; }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t kv_hbase = int64_t(block) * num_kv_heads + kv_head;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            const int64_t kidx = ((kv_hbase * dh + d / x) * block_size + slot) * x + d % x;
            partial += qv[i] * float(key_cache[kidx]);
        }
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            const int64_t vidx = (kv_hbase * D + d) * block_size + slot;
            acc[i] = acc[i] * alpha + beta * float(value_cache[vidx]);
        }
        l = l * alpha + beta;
        m = nm;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[row + lane + 32 * i] = (l == 0.0f) ? T(0) : T(acc[i] / l);
}

}  // namespace tms

