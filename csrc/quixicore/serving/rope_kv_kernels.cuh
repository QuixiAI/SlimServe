#pragma once
// Fused RoPE + paged-KV insert, CUDA/SM86 port of ThunderMittens kernels/rope_kv.
// Split-half / GPT-NeoX RoPE (matches mx.fast.rope(traditional=False)):
//   ko1 = k1*cos - k2*sin;  ko2 = k2*cos + k1*sin   (halves k1=k[:D/2], k2=k[D/2:])
// One warp per (token, kv_head) row; rotated K + unrotated V go straight to the
// paged cache slot (fusing rotary + two scatters). Variants: rope_kv_insert_norm
// (RMSNorm over the K head dim first; gemma flag -> (1+w)) and rope_q (Q path,
// contiguous output, optional norm).
//
// Build:
//   /usr/local/cuda/bin/nvcc rope_kv.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -o rope_kv.out
#include <cuda_fp16.h>
#include "tm_warp.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {


// optional fused RMSNorm over the full D of one K row (lane-strided), then split-half RoPE,
// then paged insert. norm_weight == nullptr disables the norm. Covers all three TM kernels.
template <typename T, int D, bool NORM>
__global__ void rope_kv_insert(const T* k, const T* v, const T* cosb, const T* sinb,
                               const int* positions, const int64_t* slot_mapping,
                               T* key_cache, T* value_cache, const T* norm_weight,
                               int num_kv_heads, int block_size, int gemma, float eps) {
    constexpr int D2 = D / 2;
    const int row = blockIdx.x;                 // (token, kv_head) flattened
    const int token = row / num_kv_heads;
    const int kv_head = row % num_kv_heads;
    const int lane = threadIdx.x;

    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    const int64_t dst = ((slot / block_size) * block_size + slot % block_size) * num_kv_heads + kv_head;
    const int pos = positions[token];

    const T* krow = k + int64_t(row) * D;
    const T* vrow = v + int64_t(row) * D;
    T* kc = key_cache + dst * D;
    T* vc = value_cache + dst * D;

    float kv_[ (D + 31) / 32 ];                 // this lane's K elements (full D)
    #pragma unroll
    for (int i = 0; i < D / 32; i++) kv_[i] = float(krow[lane + 32 * i]);

    if constexpr (NORM) {
        float ss = 0;
        #pragma unroll
        for (int i = 0; i < D / 32; i++) ss += kv_[i] * kv_[i];
        ss = warp_sum_f(ss);
        const float rms = rsqrtf(ss / D + eps);
        #pragma unroll
        for (int i = 0; i < D / 32; i++) {
            const float w = float(norm_weight[lane + 32 * i]);
            kv_[i] *= rms * (gemma ? (1.0f + w) : w);
        }
    }

    // split-half rotate: lane owns positions (lane + 32i) of each half.
    // kv_[i] holds k[lane+32i]; the first D2/32 entries are half-1 iff lane+32i < D2 -
    // with D2 a multiple of 32, entries i < D2/32 are half 1, the rest half 2.
    constexpr int H = D2 / 32;
    #pragma unroll
    for (int i = 0; i < H; i++) {
        const int d = lane + 32 * i;            // index within the half
        const float c = float(cosb[int64_t(pos) * D2 + d]);
        const float s = float(sinb[int64_t(pos) * D2 + d]);
        const float k1 = kv_[i], k2 = kv_[i + H];
        kc[d]      = T(k1 * c - k2 * s);
        kc[d + D2] = T(k2 * c + k1 * s);
    }
    #pragma unroll
    for (int i = 0; i < D / 32; i++) vc[lane + 32 * i] = vrow[lane + 32 * i];
}

// Q companion: rotate (optionally norm) and write contiguous q_out (Q is not paged)
template <typename T, int D, bool NORM>
__global__ void rope_q(const T* q, const T* cosb, const T* sinb, const int* positions,
                       T* q_out, const T* norm_weight, int num_heads, int gemma, float eps) {
    constexpr int D2 = D / 2;
    const int row = blockIdx.x;                 // (token, head) flattened
    const int token = row / num_heads;
    const int lane = threadIdx.x;
    const int pos = positions[token];
    const T* qrow = q + int64_t(row) * D;
    T* orow = q_out + int64_t(row) * D;

    float qv[(D + 31) / 32];
    #pragma unroll
    for (int i = 0; i < D / 32; i++) qv[i] = float(qrow[lane + 32 * i]);
    if constexpr (NORM) {
        float ss = 0;
        #pragma unroll
        for (int i = 0; i < D / 32; i++) ss += qv[i] * qv[i];
        ss = warp_sum_f(ss);
        const float rms = rsqrtf(ss / D + eps);
        #pragma unroll
        for (int i = 0; i < D / 32; i++) {
            const float w = float(norm_weight[lane + 32 * i]);
            qv[i] *= rms * (gemma ? (1.0f + w) : w);
        }
    }
    constexpr int H = D2 / 32;
    #pragma unroll
    for (int i = 0; i < H; i++) {
        const int d = lane + 32 * i;
        const float c = float(cosb[int64_t(pos) * D2 + d]);
        const float s = float(sinb[int64_t(pos) * D2 + d]);
        const float k1 = qv[i], k2 = qv[i + H];
        orow[d]      = T(k1 * c - k2 * s);
        orow[d + D2] = T(k2 * c + k1 * s);
    }
}

}  // namespace tms

