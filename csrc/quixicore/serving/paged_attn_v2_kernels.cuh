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
        // !(mp > NEG_INF) skips empty partitions AND degrades a NaN partial
        // stat to "empty" instead of poisoning the head (fmaxf already drops
        // NaN from gm, so without this the NaN re-enters via expf).
        if (!(mp > NEG_INF)) continue;
        gden += exp_sums[base + p] * expf(mp - gm);
    }
    const float inv = 1.0f / (gden + 1e-6f);

    float acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) acc[i] = 0.0f;
    for (int p = 0; p < num_partitions; p++) {
        const float mp = max_logits[base + p];
        if (!(mp > NEG_INF)) continue;
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

// Decode-specialized reducer for a large partition axis. The original reducer
// intentionally uses one warp, which leaves Ampere almost idle when DSV4's
// short split-K partitions produce O(100) partials per head.
template <typename T, int D, int WARPS>
__global__ void paged_attention_reduce_multiwarp(
        const float* tmp_out, const float* max_logits, const float* exp_sums,
        T* out, int num_heads, int num_partitions) {
    static_assert(D % 32 == 0, "D must be a multiple of a warp");
    constexpr int VPL = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int thread = threadIdx.x;
    const int64_t base = (int64_t(batch) * num_heads + head) * num_partitions;

    extern __shared__ float partition_weights[];
    __shared__ float warp_max[WARPS];
    __shared__ float warp_den[WARPS];
    __shared__ float warp_out[WARPS][D];

    float local_max = NEG_INF;
    for (int p = thread; p < num_partitions; p += WARPS * 32) {
        local_max = fmaxf(local_max, max_logits[base + p]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(
            local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    if (lane == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();

    if (warp == 0) {
        float value = lane < WARPS ? warp_max[lane] : NEG_INF;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(
                value, __shfl_down_sync(0xffffffffu, value, offset));
        }
        if (lane == 0) {
            warp_max[0] = value;
        }
    }
    __syncthreads();
    const float global_max = warp_max[0];

    float local_den = 0.0f;
    for (int p = thread; p < num_partitions; p += WARPS * 32) {
        const float partition_max = max_logits[base + p];
        const float weight = !(partition_max > NEG_INF)
            ? 0.0f
            : exp_sums[base + p] * expf(partition_max - global_max);
        partition_weights[p] = weight;
        local_den += weight;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_den += __shfl_down_sync(0xffffffffu, local_den, offset);
    }
    if (lane == 0) {
        warp_den[warp] = local_den;
    }
    __syncthreads();
    if (warp == 0) {
        float value = lane < WARPS ? warp_den[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) {
            warp_den[0] = value;
        }
    }
    __syncthreads();

    float accum[VPL];
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        accum[i] = 0.0f;
    }
    for (int p = warp; p < num_partitions; p += WARPS) {
        const float weight = partition_weights[p];
        if (weight == 0.0f) {
            continue;
        }
        const int64_t partition_base = (base + p) * D;
#pragma unroll
        for (int i = 0; i < VPL; ++i) {
            accum[i] += tmp_out[partition_base + lane + 32 * i] * weight;
        }
    }
#pragma unroll
    for (int i = 0; i < VPL; ++i) {
        warp_out[warp][lane + 32 * i] = accum[i];
    }
    __syncthreads();

    if (warp == 0) {
        const float inverse = 1.0f / (warp_den[0] + 1e-6f);
        const int64_t output_base = (int64_t(batch) * num_heads + head) * D;
#pragma unroll
        for (int i = 0; i < VPL; ++i) {
            float value = 0.0f;
#pragma unroll
            for (int source_warp = 0; source_warp < WARPS; ++source_warp) {
                value += warp_out[source_warp][lane + 32 * i];
            }
            out[output_base + lane + 32 * i] =
                global_max == NEG_INF ? T(0) : T(value * inverse);
        }
    }
}

// DSV4 decode owns exactly one thread per MLA value channel. Partition weights
// are computed once by the block, then every partial output element is read and
// accumulated exactly once. This avoids the multi-warp reducer's replicated
// 512-channel accumulators and shared-memory cross-warp epilogue.
template <typename T, int D>
__global__ __launch_bounds__(D, 2) void paged_attention_reduce_channels(
        const float* tmp_out, const float* max_logits, const float* exp_sums,
        T* out, int num_heads, int num_partitions) {
    static_assert(D == 512, "channel reducer is specialized for DSV4 MLA");
    constexpr int WARPS = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y;
    const int thread = threadIdx.x;
    const int warp = thread >> 5, lane = thread & 31;
    const int64_t base = (int64_t(batch) * num_heads + head) * num_partitions;

    extern __shared__ float partition_weights[];
    __shared__ float warp_stats[WARPS];

    float local_max = NEG_INF;
    for (int p = thread; p < num_partitions; p += D) {
        local_max = fmaxf(local_max, max_logits[base + p]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_max = fmaxf(
            local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    }
    if (lane == 0) warp_stats[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float value = lane < WARPS ? warp_stats[lane] : NEG_INF;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value = fmaxf(
                value, __shfl_down_sync(0xffffffffu, value, offset));
        }
        if (lane == 0) warp_stats[0] = value;
    }
    __syncthreads();
    const float global_max = warp_stats[0];

    float local_den = 0.0f;
    for (int p = thread; p < num_partitions; p += D) {
        const float partition_max = max_logits[base + p];
        const float weight = !(partition_max > NEG_INF)
            ? 0.0f
            : exp_sums[base + p] * expf(partition_max - global_max);
        partition_weights[p] = weight;
        local_den += weight;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_den += __shfl_down_sync(0xffffffffu, local_den, offset);
    }
    if (lane == 0) warp_stats[warp] = local_den;
    __syncthreads();
    if (warp == 0) {
        float value = lane < WARPS ? warp_stats[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            value += __shfl_down_sync(0xffffffffu, value, offset);
        }
        if (lane == 0) warp_stats[0] = value;
    }
    __syncthreads();

    float value = 0.0f;
    for (int p = 0; p < num_partitions; ++p) {
        value += tmp_out[(base + p) * D + thread] * partition_weights[p];
    }
    const int64_t output_base = (int64_t(batch) * num_heads + head) * D;
    out[output_base + thread] = global_max == NEG_INF
        ? T(0)
        : T(value / (warp_stats[0] + 1e-6f));
}

// DSV4 keeps fixed-width CUDA-graph buffers for each sparse source. At short
// contexts the ratio-128 source can expose 1024 logical partition slots while
// only a few contain tokens. Preserve the canonical slot/order mapping, but
// visit only the active prefix of each source in the value reduction.
template <typename T, int D>
__global__ __launch_bounds__(D, 2) void dsv4_attention_reduce_active_channels(
        const float* tmp_out, const float* max_logits, const float* exp_sums,
        const int* main_lengths, const int* extra_lengths, const float* sink,
        T* out,
        int num_heads, int main_partitions, int extra_partitions,
        int total_partitions) {
    static_assert(D == 512, "active channel reducer is specialized for DSV4 MLA");
    constexpr int WARPS = D / 32;
    const int head = blockIdx.x, batch = blockIdx.y;
    const int thread = threadIdx.x;
    const int warp = thread >> 5, lane = thread & 31;
    const int64_t base =
        (int64_t(batch) * num_heads + head) * total_partitions;
    const int main_active = min(main_partitions, max(main_lengths[batch], 0));
    const int extra_active = min(extra_partitions, max(extra_lengths[batch], 0));
    const int extra_begin = main_partitions;
    const int extra_end = extra_begin + extra_active;
    const int sink_owner = (main_partitions + extra_partitions) % D;

    extern __shared__ float partition_weights[];
    __shared__ float warp_stats[WARPS];

    float local_max = NEG_INF;
    for (int p = thread; p < main_active; p += D)
        local_max = fmaxf(local_max, max_logits[base + p]);
    int p = thread;
    if (p < extra_begin)
        p += ((extra_begin - p + D - 1) / D) * D;
    for (; p < extra_end; p += D)
        local_max = fmaxf(local_max, max_logits[base + p]);
    if (sink != nullptr) local_max = fmaxf(local_max, sink[head]);
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_max = fmaxf(
            local_max, __shfl_down_sync(0xffffffffu, local_max, offset));
    if (lane == 0) warp_stats[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float value = lane < WARPS ? warp_stats[lane] : NEG_INF;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value = fmaxf(
                value, __shfl_down_sync(0xffffffffu, value, offset));
        if (lane == 0) warp_stats[0] = value;
    }
    __syncthreads();
    const float global_max = warp_stats[0];

    float local_den = 0.0f;
    for (int partition = thread; partition < main_active; partition += D) {
        const float partition_max = max_logits[base + partition];
        const float weight = !(partition_max > NEG_INF)
            ? 0.0f
            : exp_sums[base + partition] * expf(partition_max - global_max);
        partition_weights[partition] = weight;
        local_den += weight;
    }
    p = thread;
    if (p < extra_begin)
        p += ((extra_begin - p + D - 1) / D) * D;
    for (; p < extra_end; p += D) {
        const float partition_max = max_logits[base + p];
        const float weight = !(partition_max > NEG_INF)
            ? 0.0f
            : exp_sums[base + p] * expf(partition_max - global_max);
        partition_weights[p] = weight;
        local_den += weight;
    }
    if (sink != nullptr && thread == sink_owner) {
        const float sink_logit = sink[head];
        if (!(isinf(sink_logit) && sink_logit < 0.0f))
            local_den += expf(sink_logit - global_max);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        local_den += __shfl_down_sync(0xffffffffu, local_den, offset);
    if (lane == 0) warp_stats[warp] = local_den;
    __syncthreads();
    if (warp == 0) {
        float value = lane < WARPS ? warp_stats[lane] : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (lane == 0) warp_stats[0] = value;
    }
    __syncthreads();

    float value = 0.0f;
    for (int partition = 0; partition < main_active; ++partition)
        value += tmp_out[(base + partition) * D + thread] *
                 partition_weights[partition];
    for (int partition = extra_begin; partition < extra_end; ++partition)
        value += tmp_out[(base + partition) * D + thread] *
                 partition_weights[partition];
    const int64_t output_base = (int64_t(batch) * num_heads + head) * D;
    out[output_base + thread] = global_max == NEG_INF
        ? T(0)
        : T(value / (warp_stats[0] + 1e-6f));
}

// explicit D=512 reduce instantiation (MLA reuses it)
template __global__ void paged_attention_reduce<__nv_bfloat16, 512>(
    const float*, const float*, const float*, __nv_bfloat16*, int, int);

}  // namespace tms
