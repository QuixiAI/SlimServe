/**
 * @file
 * @brief Sparse-MLA indexer index arithmetic, native HIP (gfx942).
 *
 * These replace two hand-written Triton kernels on the ROCm sparse-MLA path
 * (vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py). Unlike the
 * quixicore serving kernels, these have no CUDA counterpart: the ROCm sparse
 * backend is the only consumer, so the Triton kernel is the reference and
 * bitwise equality with it is the bar. Both are pure integer index arithmetic
 * over num_tokens -- the motivation is removing Triton from the serving path,
 * not throughput.
 */
#pragma once
#include <cstdint>

namespace qcrocm {

// Maps request-local top-k token positions to global paged-KV slots.
//
// Semantics are taken from the Triton kernel body, not its docstring: an
// invalid token (< 0) or an out-of-range block id yields **0**, not -1. The
// docstring above the Triton wrapper claims -1; the `tl.where` it describes
// writes 0, and downstream consumers depend on the code's behaviour.
//
//   req_id        [num_tokens]        int32
//   block_table   [num_reqs, max_blk] int32
//   token_indices [num_tokens, topk]  int32
//   cu_seqlens    [num_tokens + 1]    int32
//   out           ragged, packed at cu_seqlens[token]
__global__ void convert_req_index_to_global_index(
    const int* __restrict__ req_id, const int* __restrict__ block_table,
    const int* __restrict__ token_indices, const int* __restrict__ cu_seqlens,
    int* __restrict__ out, int max_num_blocks_per_req, int block_size, int topk,
    long bt_stride0, long bt_stride1, long ti_stride0, long ti_stride1) {
    const int token_id = blockIdx.x;
    const int seq_start = cu_seqlens[token_id];
    const int seq_end = cu_seqlens[token_id + 1];
    const int req = req_id[token_id];

    for (int col = threadIdx.x; col < topk; col += blockDim.x) {
        // The Triton store mask is (seq_start + col) < seq_end, so a row
        // contributes only its first (seq_end - seq_start) columns.
        if (seq_start + col >= seq_end) continue;
        const int tok =
            token_indices[(long)token_id * ti_stride0 + (long)col * ti_stride1];
        const int block_id = tok / block_size;
        const int inblock_off = tok % block_size;
        const bool valid_block =
            block_id < max_num_blocks_per_req && block_id >= 0;
        int val = 0;
        if (tok >= 0 && valid_block) {
            const int base = block_table[(long)req * bt_stride0 +
                                         (long)block_id * bt_stride1];
            val = base * block_size + inblock_off;
        }
        out[seq_start + col] = val;
    }
}

// Per-query-token sparse KV length: min(context_start + offset + 1, topk).
// `out` is zero-initialized by the caller and rows with seq_len == 0 are left
// untouched, matching the Triton early return.
__global__ void generate_sparse_seqlen(const int* __restrict__ seq_lens,
                                       const int* __restrict__ cu_query_lens,
                                       int* __restrict__ out, int topk_token) {
    const int seq_id = blockIdx.x;
    const int query_start = cu_query_lens[seq_id];
    const int query_end = cu_query_lens[seq_id + 1];
    const int query_len = query_end - query_start;
    const int seq_len = seq_lens[seq_id];
    if (seq_len == 0 || query_len <= 0) return;

    const int context_start_point = seq_len - query_len;
    for (int off = threadIdx.x; off < query_len; off += blockDim.x) {
        const int sparse_seqlen = context_start_point + off;
        out[query_start + off] =
            (sparse_seqlen + 1 < topk_token) ? (sparse_seqlen + 1) : topk_token;
    }
}


// Per-token fp8 quantize of the indexer K vector, scattered into the paged
// cache. One block per token, one lane per head-dim element.
//
// scale = max(1e-4, amax) / FP8_MAX, so |val / scale| <= FP8_MAX by
// construction and the conversion can never overflow. That matters: within
// range, ROCm Triton's float->fp8 cast and HIP's agree bitwise (measured over
// 131577 samples including exact midpoint ties), but they disagree *outside*
// it -- Triton saturates to max-finite where HIP emits NaN. This kernel stays
// inside the agreeing domain.
//
// The amax reduction is a max, which is order-independent and exact, so the
// reduction tree shape is free here (unlike the sampler's fp32 sums).
//
//   k          [num_tokens, head_dim]  bf16/fp16/fp32
//   cache      fp8, paged; SHUFFLE tiles when block_size > 1
//   scale_out  fp32, [num_blocks, block_size]
template <typename T, typename FP8_T>
__global__ void indexer_k_quant_and_cache(
    const T* __restrict__ k, FP8_T* __restrict__ kv_cache,
    float* __restrict__ kv_cache_scale, const long* __restrict__ slot_mapping,
    long scale_stride, long value_stride, int block_size, int head_dim,
    int block_tile_size, int head_tile_size, float fp8_max, int shuffle) {
    const int tid = blockIdx.x;
    const long slot_id = slot_mapping[tid];
    if (slot_id < 0) return;

    const T* src = k + (long)tid * head_dim;

    // amax over the row.
    float local = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x)
        local = fmaxf(local, fabsf((float)src[i]));
    __shared__ float s_amax[64];
    const int nwarps = (blockDim.x + 63) / 64;
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    for (int off = 32; off > 0; off >>= 1)
        local = fmaxf(local, __shfl_xor(local, off, 64));
    if (lane == 0) s_amax[warp] = local;
    __syncthreads();
    float amax = s_amax[0];
    for (int w = 1; w < nwarps; ++w) amax = fmaxf(amax, s_amax[w]);

    const float scale = fmaxf(1e-4f, amax) / fp8_max;

    const long block_id = slot_id / block_size;
    const int block_offset = (int)(slot_id % block_size);
    const int tile_block_id = block_offset / block_tile_size;
    const int tile_block_offset = block_offset % block_tile_size;

    FP8_T* dst;
    if (shuffle)
        dst = kv_cache + block_id * value_stride +
              (long)tile_block_id * block_tile_size * head_dim +
              (long)tile_block_offset * head_tile_size;
    else
        dst = kv_cache + block_id * value_stride + (long)block_offset * head_dim;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const int tile_offset =
            shuffle ? (i / head_tile_size) * block_tile_size * head_tile_size +
                          (i % head_tile_size)
                    : i;
        dst[tile_offset] = (FP8_T)((float)src[i] / scale);
    }
    if (threadIdx.x == 0)
        kv_cache_scale[block_id * scale_stride + block_offset] = scale;
}


// Gather quantized indexer K (and its scale) out of the paged cache into a
// contiguous prefill workspace -- the inverse of indexer_k_quant_and_cache.
//
// Two asymmetries are carried over deliberately from the Triton kernel:
//   * an invalid *token* returns early, leaving both outputs untouched;
//   * an invalid *block* still writes the scale (as 0.0) but leaves the value
//     row untouched, because the value store is masked while the scale store
//     is not.
// The source load is likewise unmasked in the original -- safe only because
// the block id is clamped to 0 first -- and that clamp is reproduced here.
template <typename FP8_T>
__global__ void cp_gather_indexer_quant_cache(
    const FP8_T* __restrict__ kv_cache, const float* __restrict__ kv_cache_scale,
    FP8_T* __restrict__ k_fp8, float* __restrict__ k_scale,
    const int* __restrict__ block_table, const int* __restrict__ cu_seqlen,
    const int* __restrict__ token_to_seq, int block_size,
    long block_table_stride, long kv_cache_stride, long kv_cache_scale_stride,
    int head_dim, int block_tile_size, int head_tile_size, int num_tokens,
    int num_batches, int block_table_width, int num_blocks, int shuffle) {
    const int tid = blockIdx.x;
    if (tid >= num_tokens) return;

    const int batch_id = token_to_seq[tid];
    const bool valid_batch = batch_id >= 0 && batch_id < num_batches;
    const int safe_batch = valid_batch ? batch_id : 0;
    const int batch_start = valid_batch ? cu_seqlen[safe_batch] : 0;
    const int batch_end = valid_batch ? cu_seqlen[safe_batch + 1] : 0;
    const int batch_offset = tid - batch_start;
    if (!(valid_batch && tid >= batch_start && tid < batch_end)) return;

    const int block_table_id = batch_offset / block_size;
    const int block_offset = batch_offset % block_size;
    const bool valid_bt = block_table_id >= 0 &&
                          block_table_id < block_table_width &&
                          block_offset >= 0 && block_offset < block_size;
    const int safe_bt = valid_bt ? block_table_id : 0;
    const int block_id =
        valid_bt ? block_table[(long)safe_batch * block_table_stride + safe_bt]
                 : -1;
    const bool valid_block = valid_bt && block_id >= 0 && block_id < num_blocks;
    const long safe_block = valid_block ? (long)block_id : 0L;
    const int safe_off = valid_block ? block_offset : 0;

    long src;
    if (shuffle)
        src = safe_block * kv_cache_stride +
              (long)(safe_off / block_tile_size) * head_dim * block_tile_size +
              (long)(safe_off % block_tile_size) * head_tile_size;
    else
        src = safe_block * kv_cache_stride + (long)safe_off * head_dim;

    if (threadIdx.x == 0)
        k_scale[tid] =
            valid_block
                ? kv_cache_scale[safe_block * kv_cache_scale_stride + safe_off]
                : 0.0f;
    if (!valid_block) return;

    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const int to = shuffle ? (i / head_tile_size) * head_tile_size *
                                        block_tile_size +
                                    (i % head_tile_size)
                               : i;
        k_fp8[(long)tid * head_dim + i] = kv_cache[src + to];
    }
}

}  // namespace qcrocm
