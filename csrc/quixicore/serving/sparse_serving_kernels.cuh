/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's sparse-attention + serving extras
 * (attention.metal / cache/tau.metal / cache/gather_kv_cache.metal): 2-way
 * LSE merge of attention partials, the tau/temperature Q/V gate, the DeepSeek
 * lightning-indexer fp8 K-quant-and-cache, and the MInference vertical/slash
 * sparse-index builder.
 */
#pragma once
#include "quant_formats.cuh"             // e4m3_encode / e4m3_decode
#include "tm_warp.cuh"                   // warp_max_f
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tmsp {

template <typename T> __device__ __forceinline__ float pf(T v);
template <> __device__ __forceinline__ float pf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float pf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float pf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T fp_(float v);
template <> __device__ __forceinline__ float          fp_<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         fp_<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  fp_<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// ---- 2-way LSE merge of attention partials (MetalForge merge_attn_states):
// output = softmax-weighted combine of prefix/suffix by their log-sum-exps;
// output_lse = log(exp(p)+exp(s)). Layout (tokens, heads, head_size); lse
// (tokens, heads). One thread per (token, head, hidden). Tokens beyond the
// prefix copy the suffix. Complements paged_attention_reduce (N-way partition
// combine); this is the chunked-prefill / cascade 2-way merge. ----
template <typename T>
__global__ void merge_attn_states(T* __restrict__ output, float* __restrict__ output_lse,
        const T* __restrict__ prefix_out, const float* __restrict__ prefix_lse,
        const T* __restrict__ suffix_out, const float* __restrict__ suffix_lse,
        int num_heads, int head_size, int num_tokens, int prefix_num_tokens) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)num_tokens * num_heads * head_size;
    if (idx >= total) return;
    const int hidden = idx % head_size;
    const int head = (idx / head_size) % num_heads;
    const int token = idx / ((long)head_size * num_heads);
    const long lse_idx = (long)token * num_heads + head;

    if (token >= prefix_num_tokens) {                 // suffix-only tail
        output[idx] = suffix_out[idx];
        if (hidden == 0) output_lse[lse_idx] = suffix_lse[lse_idx];
        return;
    }
    const float pl = prefix_lse[lse_idx], sl = suffix_lse[lse_idx];
    const float m = fmaxf(pl, sl);
    if (isinf(m)) {
        output[idx] = prefix_out[idx];
        if (hidden == 0) output_lse[lse_idx] = m;
        return;
    }
    const float pse = expf(pl - m), sse = expf(sl - m), z = pse + sse;
    output[idx] = fp_<T>(pf(prefix_out[idx]) * (pse / z) + pf(suffix_out[idx]) * (sse / z));
    if (hidden == 0) output_lse[lse_idx] = logf(z) + m;
}

// ---- fp8-output 2-way LSE merge (MetalForge merge_attn_states_fp8): same
// combine as merge_attn_states but writes e4m3 codes scaled by 1/output_scale.
// Layout matches merge_attn_states (tokens, heads, head_size); lse (tokens,
// heads) stays fp32. One thread per (token, head, hidden). ----
template <typename T>
__global__ void merge_attn_states_fp8(uint8_t* __restrict__ output, float* __restrict__ output_lse,
        const T* __restrict__ prefix_out, const float* __restrict__ prefix_lse,
        const T* __restrict__ suffix_out, const float* __restrict__ suffix_lse,
        int num_heads, int head_size, int num_tokens, int prefix_num_tokens, float output_scale) {
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = (long)num_tokens * num_heads * head_size;
    if (idx >= total) return;
    const int hidden = idx % head_size;
    const int head = (idx / head_size) % num_heads;
    const int token = idx / ((long)head_size * num_heads);
    const long lse_idx = (long)token * num_heads + head;
    const float scale_inv = 1.0f / output_scale;

    if (token >= prefix_num_tokens) {                 // suffix-only tail
        output[idx] = tmq::e4m3_encode(pf(suffix_out[idx]) * scale_inv);
        if (hidden == 0) output_lse[lse_idx] = suffix_lse[lse_idx];
        return;
    }
    const float pl = prefix_lse[lse_idx], sl = suffix_lse[lse_idx];
    const float m = fmaxf(pl, sl);
    if (isinf(m)) {
        output[idx] = tmq::e4m3_encode(pf(prefix_out[idx]) * scale_inv);
        if (hidden == 0) output_lse[lse_idx] = m;
        return;
    }
    const float pse = expf(pl - m), sse = expf(sl - m), z = pse + sse;
    const float v = pf(prefix_out[idx]) * (pse / z) + pf(suffix_out[idx]) * (sse / z);
    output[idx] = tmq::e4m3_encode(v * scale_inv);
    if (hidden == 0) output_lse[lse_idx] = logf(z) + m;
}

// ---- tau/temperature Q/V gate (MetalForge tau_tail): packed qkv (tokens,
// 3*q_dim); scales the Q slice by tanh(tok_q_lin)+tau_pos[pos,head] and the V
// slice by tanh(tok_v_lin)+tau_pos (K untouched). tok_qv_lin (tokens, 2*heads).
// One thread per (token, head, dim). ----
template <typename T>
__global__ void tau_tail(T* __restrict__ qkv, const T* __restrict__ tok_qv_lin,
        const T* __restrict__ tau_pos_table, const int64_t* __restrict__ positions,
        int elements, int n_heads, int head_dim, int q_dim) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= elements) return;
    const int dim = tid % head_dim;
    const int head = (tid / head_dim) % n_heads;
    const int tok = tid / ((long)head_dim * n_heads);
    const float tq = tanhf(pf(tok_qv_lin[(long)tok * 2 * n_heads + head]));
    const float tv = tanhf(pf(tok_qv_lin[(long)tok * 2 * n_heads + n_heads + head]));
    const float tp = pf(tau_pos_table[positions[tok] * (long)n_heads + head]);
    const long qi = (long)tok * 3 * q_dim + (long)head * head_dim + dim;
    const long vi = (long)tok * 3 * q_dim + 2L * q_dim + (long)head * head_dim + dim;
    qkv[qi] = fp_<T>(pf(qkv[qi]) * (tq + tp));
    qkv[vi] = fp_<T>(pf(qkv[vi]) * (tv + tp));
}

// ---- DeepSeek lightning-indexer K-quant-and-cache (MetalForge
// indexer_k_quant_and_cache): quantize each K row to fp8 e4m3 in quant_block_size
// chunks; per-block scale = max(|k|)/448 (optional UE8M0 pow2 round); write fp8
// codes then the fp32 scale after the head_dim data region. slot<0 skips. One
// warp per (token, qblock). Reuses the mla_kv_insert_fp8 e4m3+per-block idiom. ----
template <typename T>
__global__ void indexer_k_quant_and_cache(const T* __restrict__ k, uint8_t* __restrict__ kv_cache,
        const int64_t* __restrict__ slot_mapping, int num_tokens, int head_dim,
        int quant_block_size, int cache_block_size, int cache_stride, int use_ue8m0) {
    const int token = blockIdx.x, qblock = blockIdx.y, lane = threadIdx.x;
    const int start = qblock * quant_block_size;
    if (token >= num_tokens || start >= head_dim) return;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;

    float amax = 0.0f;
    for (int i = lane; i < quant_block_size && start + i < head_dim; i += 32)
        amax = fmaxf(amax, fabsf(pf(k[(long)token * head_dim + start + i])));
    amax = tms::warp_max_f(amax);
    float scale = fmaxf(amax, 1e-4f) / 448.0f;
    if (use_ue8m0) scale = exp2f(ceilf(log2f(scale)));

    const long block = slot / cache_block_size, off = slot - block * (long)cache_block_size;
    const long token_base = block * cache_block_size * (long)cache_stride + off * (long)head_dim;
    for (int i = lane; i < quant_block_size && start + i < head_dim; i += 32) {
        const float scaled = pf(k[(long)token * head_dim + start + i]) / scale;
        kv_cache[token_base + start + i] = (scaled == 0.0f) ? uint8_t(0) : tmq::e4m3_encode(scaled);
    }
    if (lane == 0) {
        const long scale_off = block * cache_block_size * (long)cache_stride
                             + (long)cache_block_size * head_dim
                             + (off * (long)head_dim + start) * 4 / quant_block_size;
        reinterpret_cast<float*>(kv_cache + scale_off)[0] = scale;
    }
}

// ---- lightning-indexer gather (MetalForge cp_gather_indexer_k_quant_cache):
// the inverse of indexer_k_quant_and_cache. For each cached token, binary-search
// its batch in cu_seq_lens, walk block_table to the paged slot, and copy the fp8
// K codes + the per-block scale out to a dense (num_tokens, head_dim) destination.
// One warp per (token, qblock); dst_scale is fp32 at token_stride/quant_block_size
// granularity. ----
static __global__ void cp_gather_indexer_k_quant_cache(const uint8_t* __restrict__ kv_cache,
        uint8_t* __restrict__ dst_k, uint8_t* __restrict__ dst_scale,
        const int* __restrict__ block_table, const int* __restrict__ cu_seq_lens,
        int batch_size, long token_stride, int head_dim, long block_stride,
        int cache_block_size, int num_blocks, int num_tokens, int quant_block_size) {
    const int token = blockIdx.x, qblock = blockIdx.y, lane = threadIdx.x, threads = blockDim.x;
    const int start = qblock * quant_block_size;
    if (token >= num_tokens || start >= head_dim) return;

    int batch = -1, lo = 0, hi = batch_size - 1;      // batch owning this token
    while (lo <= hi) {
        const int mid = (lo + hi) >> 1;
        if (cu_seq_lens[mid + 1] <= token) lo = mid + 1;
        else if (cu_seq_lens[mid] > token) hi = mid - 1;
        else { batch = mid; break; }
    }
    if (batch < 0) return;

    const int inbatch = token - cu_seq_lens[batch];
    const int bti = inbatch / cache_block_size;
    if (bti >= num_blocks) return;
    const int block = block_table[batch * num_blocks + bti];
    const int slot = inbatch - bti * cache_block_size;
    const long cache_off = (long)block * block_stride + (long)slot * head_dim + start;
    const long dst_off = (long)token * token_stride + start;

    for (int i = lane; i < quant_block_size && start + i < head_dim; i += threads)
        dst_k[dst_off + i] = kv_cache[cache_off + i];
    if (lane == 0) {
        const long src_scale_off = (long)block * block_stride + (long)cache_block_size * head_dim
                                 + ((long)slot * head_dim + start) * 4 / quant_block_size;
        const float scale = *reinterpret_cast<const float*>(kv_cache + src_scale_off);
        reinterpret_cast<float*>(dst_scale)[((long)token * token_stride + start) / quant_block_size] = scale;
    }
}

// ---- MInference vertical/slash sparse-index builder
// (convert_vertical_slash_indexes): per (head, batch, M-block), a serial
// two-pointer merge of the vertical-index and slash/diagonal-index lists into
// dense KV block offsets (block_offset/block_count) + leftover sparse columns
// (column_index/column_count). causal offset = kv_seqlen - q_seqlen. One thread
// per M-block. Faithful control-flow port (correctness, not math). ----
__device__ __forceinline__ int vs_save_blocks(int* block_offset, int rs, int re, int bs,
                                              int in_cnt, int kv_seqlen) {
    if (rs >= kv_seqlen) return in_cnt;
    if (re > kv_seqlen) re = kv_seqlen;
    int c = in_cnt;
    for (int idx = rs; idx < re; idx += bs) block_offset[c++] = idx;
    return c;
}
static __global__ void convert_vertical_slash_indexes(
        const int* __restrict__ q_seqlens, const int* __restrict__ kv_seqlens,
        const int* __restrict__ vertical_indexes, const int* __restrict__ slash_indexes,
        int* __restrict__ block_count, int* __restrict__ block_offset,
        int* __restrict__ column_count, int* __restrict__ column_index,
        int num_heads, int num_rows, int nnz_v, int nnz_s,
        int block_size_m, int block_size_n, int causal) {
    const int head = blockIdx.x, batch = blockIdx.y;
    const int block_idx_m = blockIdx.z * 64 + threadIdx.x;
    const int q_seqlen = q_seqlens[batch], kv_seqlen = kv_seqlens[batch];
    const int start_m = block_idx_m * block_size_m;
    if (block_idx_m >= num_rows || start_m >= q_seqlen) return;
    const int end_m = start_m + block_size_m;
    const int* row_v = vertical_indexes + (long)(batch * num_heads + head) * nnz_v;
    const int* row_s = slash_indexes + (long)(batch * num_heads + head) * nnz_s;
    const int row_off = (batch * num_heads + head) * num_rows + block_idx_m;
    int* rbc = block_count + row_off;
    int* rbo = block_offset + (long)row_off * nnz_s;
    int* rcc = column_count + row_off;
    int* rci = column_index + (long)row_off * nnz_v;

    bool has_slash = true;
    int tmp_col = 0, tmp_blk = 0, s = 0, v = 0;
    int v_idx = row_v[v++], s_idx = row_s[s++];
    const int offset = kv_seqlen - q_seqlen;
    if (causal) {
        while (s_idx >= end_m + offset && s < nnz_s) s_idx = row_s[s++];
        if (s_idx > end_m + offset) has_slash = false;
        s_idx = max(offset + end_m - s_idx, block_size_m);
    } else {
        while (s_idx >= end_m + kv_seqlen && s < nnz_s) s_idx = row_s[s++];
        if (s_idx > end_m + kv_seqlen) has_slash = false;
        s_idx = max(kv_seqlen + end_m - s_idx, block_size_m);
    }
    int rs = s_idx - block_size_m, re = s_idx;
    if (!has_slash) {
        if (causal) { rs = offset + end_m; re = offset + end_m + block_size_n; }
        else        { rs = kv_seqlen;      re = kv_seqlen + block_size_n; }
    }
    bool slash_finished = false;
    while (true) {
        if (v_idx < re) {
            if (v_idx < rs) rci[tmp_col++] = v_idx;
            if (v < nnz_v) v_idx = row_v[v++];
            else v_idx = causal ? (end_m + block_size_n + offset) : (end_m + block_size_n + kv_seqlen);
        } else {
            if ((s < nnz_s && causal) || (s < nnz_s && !causal && row_s[s] >= start_m)) {
                if (causal) s_idx = max(offset + end_m - row_s[s++], block_size_m);
                else        s_idx = max(kv_seqlen + end_m - row_s[s++], block_size_m);
            } else {
                if (v == nnz_v || (v_idx > rs && causal)) {
                    if (v == nnz_v && !causal && v_idx < kv_seqlen) rci[tmp_col++] = v_idx;
                    tmp_blk = vs_save_blocks(rbo, rs, re, block_size_n, tmp_blk, kv_seqlen);
                    break;
                } else {
                    if (causal) { rs = offset + end_m; re = offset + end_m + block_size_n; }
                    else {
                        tmp_blk = vs_save_blocks(rbo, rs, re, block_size_n, tmp_blk, kv_seqlen);
                        rs = kv_seqlen; re = kv_seqlen + block_size_n;
                    }
                    slash_finished = true;
                }
            }
            if (!slash_finished) {
                if (s_idx > re + block_size_m) {
                    tmp_blk = vs_save_blocks(rbo, rs, re, block_size_n, tmp_blk, kv_seqlen);
                    rs = s_idx - block_size_m; re = s_idx;
                } else if (s_idx > re) re += block_size_m;
            }
        }
    }
    rbc[0] = tmp_blk;
    rcc[0] = tmp_col;
}

} // namespace tmsp
