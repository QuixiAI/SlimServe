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

}  // namespace qcrocm
