#pragma once
// Partitioned paged decode attention (vLLM v2 shape) + cascade/shared-prefix,
// CUDA/SM86 port of ThunderMittens kernels/paged_attn_v2.
//
// Each (head, batch) query splits across num_partitions KV slices:
//   partition: local online-softmax over [p*PS, min((p+1)*PS, ctx)) ->
//     max_logits (B,H,P), exp_sums (B,H,P), tmp_out (B,H,P,D) locally normalized
//   reduce: m* = max_p m_p; out = sum_p tmp_out_p * S_p e^{m_p-m*} / (sum + 1e-6)
// fp8 partition dequantizes e4m3 codes on read (per-KV-head scales); partials
// stay fp32 so the reduce is format-agnostic. Cascade prefix emits the SAME
// partial layout from a shared contiguous prefix, so prefix ++ suffix partials
// concatenate along P and fold through the same reduce (flashinfer merge_states).
// The reduce is also instantiated at D=512 for MLA.
//
// Build:
//   /usr/local/cuda/bin/nvcc paged_attn_v2.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -I../quant -o paged_attn_v2.out
#include "quant_formats.cuh"   // e4m3_decode
#include "tm_warp.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

namespace tms {

#define NEG_INF (-3.4028234663852886e38f)


template <typename T, int D>
__global__ void paged_attention_partition(
        const T* q, const T* key_cache, const T* value_cache,
        const int* block_table, const int* context_lens,
        float* tmp_out, float* max_logits, float* exp_sums,
        int block_size, int bt_stride, float scale,
        int num_heads, int num_kv_heads, int num_partitions, int partition_size, int window) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int start = part * partition_size;
    const int end = min(start + partition_size, context_len);
    const int t_start = (window > 0) ? max(start, context_len - window) : start;

    const int64_t q_base = (int64_t(batch) * num_heads + head) * D;
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t out_base = stat * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = NEG_INF, l = 0.0f;

    for (int t = t_start; t < end; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
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
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? NEG_INF : m;
        exp_sums[stat] = l;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        tmp_out[out_base + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

// fp8 caches: uint8 e4m3 codes, per-KV-head scales; identical math otherwise
template <typename T, int D>
__global__ void paged_attention_partition_fp8(
        const T* q, const uint8_t* key_cache, const uint8_t* value_cache,
        const int* block_table, const int* context_lens,
        const float* k_scale, const float* v_scale,
        float* tmp_out, float* max_logits, float* exp_sums,
        int block_size, int bt_stride, float scale,
        int num_heads, int num_kv_heads, int num_partitions, int partition_size, int window) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int start = part * partition_size;
    const int end = min(start + partition_size, context_len);
    const int t_start = (window > 0) ? max(start, context_len - window) : start;
    const float ks = k_scale[kv_head], vs = v_scale[kv_head];

    const int64_t q_base = (int64_t(batch) * num_heads + head) * D;
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t out_base = stat * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = NEG_INF, l = 0.0f;

    for (int t = t_start; t < end; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * num_kv_heads * D + int64_t(kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            partial += qv[i] * ks * tmq::e4m3_decode(key_cache[base + lane + 32 * i]);
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * vs * tmq::e4m3_decode(value_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? NEG_INF : m;
        exp_sums[stat] = l;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        tmp_out[out_base + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

// cascade / shared-prefix: contiguous prefix_k/v (prefix_len, H_KV, D), same partial layout
template <typename T, int D>
__global__ void cascade_prefix_partition(
        const T* q, const T* prefix_k, const T* prefix_v,
        float* tmp_out, float* max_logits, float* exp_sums,
        float scale, int num_heads, int num_kv_heads,
        int prefix_len, int num_partitions, int partition_size) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int start = part * partition_size;
    const int end = min(start + partition_size, prefix_len);

    const int64_t q_base = (int64_t(batch) * num_heads + head) * D;
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t out_base = stat * D;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }
    float m = NEG_INF, l = 0.0f;

    for (int t = start; t < end; t++) {
        const int64_t base = (int64_t(t) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) partial += qv[i] * float(prefix_k[base + lane + 32 * i]);
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++)
            acc[i] = acc[i] * alpha + beta * float(prefix_v[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? NEG_INF : m;
        exp_sums[stat] = l;
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        tmp_out[out_base + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
}

template <typename T, int D>
__global__ void paged_attention_reduce(
        const float* tmp_out, const float* max_logits, const float* exp_sums,
        T* out, int num_heads, int num_partitions) {
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int64_t base = (int64_t(batch) * num_heads + head) * num_partitions;

    float gm = NEG_INF;
    for (int p = 0; p < num_partitions; p++) gm = fmaxf(gm, max_logits[base + p]);
    float gden = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        const float mp = max_logits[base + p];
        if (mp == NEG_INF) continue;
        gden += exp_sums[base + p] * expf(mp - gm);
    }
    const float inv = 1.0f / (gden + 1e-6f);

    float acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) acc[i] = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        const float mp = max_logits[base + p];
        if (mp == NEG_INF) continue;
        const float r = exp_sums[base + p] * expf(mp - gm);
        const int64_t ob = (base + p) * D;
        #pragma unroll
        for (int i = 0; i < VPL; i++) acc[i] += tmp_out[ob + lane + 32 * i] * r;
    }
    const int64_t out_base = (int64_t(batch) * num_heads + head) * D;
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[out_base + lane + 32 * i] = (gm == NEG_INF) ? T(0) : T(acc[i] * inv);
}

// explicit D=512 reduce instantiation (MLA reuses it)
template __global__ void paged_attention_reduce<__nv_bfloat16, 512>(
    const float*, const float*, const float*, __nv_bfloat16*, int, int);

}  // namespace tms

