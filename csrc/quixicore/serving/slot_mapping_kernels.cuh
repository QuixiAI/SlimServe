/**
 * @file
 * @brief Slot-mapping and indexer metadata, native CUDA.
 *
 * These replace hand-written Triton kernels on the serving path. They are index
 * arithmetic over num_reqs, not compute -- the motivation is removing the Triton
 * dependency (and its architecture gates, which cost real time on Ampere via
 * `fp8e4nv`), not throughput.
 *
 * Semantics mirror vllm/v1/worker/block_table.py::_compute_slot_mapping_kernel
 * exactly, including the context-parallel interleave and the PAD_ID fill used to
 * keep CUDA-graph replay shapes stable.
 */
#pragma once
#include <cstdint>

namespace tms {

// One block per request; the final block pads [num_tokens, max_num_tokens).
//   query_start_loc [num_reqs+1] int32
//   positions       [num_tokens] int64
//   block_table     [max_num_reqs, block_table_stride] int32
//   slot_mapping    [max_num_tokens] int64
__global__ void compute_slot_mapping(
    long num_tokens, long max_num_tokens,
    const int* __restrict__ query_start_loc,
    const long* __restrict__ positions,
    const int* __restrict__ block_table, int block_table_stride, int block_size,
    long* __restrict__ slot_mapping,
    int kv_cache_block_size, int blocks_per_kv_block,
    int cp_world, int cp_rank, int cp_interleave, long pad_id) {
    const int req = blockIdx.x;
    const int nthr = blockDim.x;

    if (req == gridDim.x - 1) {
        // Pad the tail so CUDA-graph replay sees a fixed-size buffer.
        for (long i = num_tokens + threadIdx.x; i < max_num_tokens; i += nthr)
            slot_mapping[i] = pad_id;
        return;
    }

    const long start = (long)query_start_loc[req];
    const long end = (long)query_start_loc[req + 1];
    const long virtual_block_size = (long)kv_cache_block_size * cp_world;
    const long row_offset = (long)req * block_table_stride;

    for (long t = start + threadIdx.x; t < end; t += nthr) {
        const long pos = positions[t];
        const long vbi = pos / virtual_block_size;
        const long vbo = pos - vbi * virtual_block_size;
        const bool is_local = ((vbo / cp_interleave) % cp_world) == cp_rank;
        const long lbo = (vbo / ((long)cp_world * cp_interleave)) * cp_interleave +
                         (vbo % cp_interleave);
        long slot = pad_id;
        if (is_local) {
            const long bi = vbi * blocks_per_kv_block + lbo / block_size;
            const long bn = (long)block_table[row_offset + bi];
            slot = bn * block_size + (lbo % block_size);
        }
        slot_mapping[t] = slot;
    }
}

// Native replacement for the DSA indexer's Triton metadata kernel
// (vllm/v1/attention/backends/mla/indexer.py::kernel). One block per request.
// Writes, per query token in [query_slice_start, query_slice_stop):
//   cu_ks[out_pos] = row_start
//   cu_ke[out_pos] = row_start + per-token compressed context length
// and fills token_to_seq over each request's compressed span.
// DCP_WORLD > 1 applies the interleave-aware per-rank context length, matching
// get_dcp_local_seq_lens.
__global__ void indexer_metadata(
    const int* __restrict__ query_start_loc,
    const int* __restrict__ uncompressed_seq_lens,
    const int* __restrict__ cu_compressed_seq_lens,
    const int* __restrict__ row_start_cu_compressed_seq_lens,
    int* __restrict__ token_to_seq,
    int* __restrict__ cu_ks,
    int* __restrict__ cu_ke,
    int query_slice_start, int query_slice_stop,
    int dcp_rank, int dcp_world, int dcp_interleave, int compress_ratio) {
    const int b = blockIdx.x;
    const int nthr = blockDim.x;

    const int query_start = query_start_loc[b];
    const int query_len = query_start_loc[b + 1] - query_start;
    const int seq_start = cu_compressed_seq_lens[b];
    const int compressed_seq_len = cu_compressed_seq_lens[b + 1] - seq_start;
    const int row_start = row_start_cu_compressed_seq_lens[b];
    const int start_pos = uncompressed_seq_lens[b] - query_len;

    for (int off = threadIdx.x; off < query_len; off += nthr) {
        const int abs_pos = query_start + off;
        if (abs_pos < query_slice_start || abs_pos >= query_slice_stop) continue;
        const int out_pos = abs_pos - query_slice_start;
        cu_ks[out_pos] = row_start;

        int len_per_token = (start_pos + 1 + off) / compress_ratio;
        if (dcp_world > 1) {
            const int base =
                (len_per_token / dcp_interleave / dcp_world) * dcp_interleave;
            int rem = len_per_token - base * dcp_world;
            rem = min(max(rem - dcp_rank * dcp_interleave, 0), dcp_interleave);
            len_per_token = base + rem;
        }
        cu_ke[out_pos] = row_start + len_per_token;
    }

    for (int off = threadIdx.x; off < compressed_seq_len; off += nthr)
        token_to_seq[seq_start + off] = b;
}

}  // namespace tms
