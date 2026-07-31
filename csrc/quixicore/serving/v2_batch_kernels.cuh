/**
 * @file
 * @brief V2 model-runner batch-prep kernels, native CUDA.
 *
 * Index-arithmetic kernels replacing the Triton batch-preparation kernels of
 * the V2 GPU worker (vllm/v1/worker/gpu/{buffer_utils,input_batch,block_table}.py,
 * spec_decode/dflash/speculator.py and the DSA indexer decode expansion). Like
 * slot_mapping_kernels.cuh, the motivation is removing Triton from the serving
 * path, not throughput: one block per request, semantics mirror the Triton
 * source exactly (including integer truncations on stores and CUDA-graph
 * padding sections).
 */
#pragma once
#include <cstdint>

namespace tms {

// buffer_utils.py::_apply_write_kernel, MULTI_GROUP=False. One block per
// staged write; T is the tensor dtype (int32/int64/fp32 all reduce to a
// bit-exact element copy).
template <typename T>
__global__ void apply_write(
    T* __restrict__ out, long row_stride,
    const int* __restrict__ write_indices,
    const int* __restrict__ write_starts,
    const T* __restrict__ write_contents,
    const int* __restrict__ write_cu_lens) {
    const int w = blockIdx.x;
    const int row = write_indices[w];
    const int start = write_starts[w];
    const int cu_start = w > 0 ? write_cu_lens[w - 1] : 0;
    const int len = write_cu_lens[w] - cu_start;
    T* row_ptr = out + (long)row * row_stride + start;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        row_ptr[i] = write_contents[cu_start + i];
}

// buffer_utils.py::_apply_write_kernel, MULTI_GROUP=True. Each write resolves
// its own destination tensor through a pointer/stride table (int32 elements,
// as used by the fused block-table writer).
__global__ void apply_write_multi(
    const unsigned long long* __restrict__ out_ptrs,
    const long* __restrict__ out_strides,
    const int* __restrict__ write_indices,
    const int* __restrict__ write_starts,
    const int* __restrict__ write_contents,
    const int* __restrict__ write_cu_lens,
    const int* __restrict__ write_group_ids) {
    const int w = blockIdx.x;
    const int row = write_indices[w];
    const int start = write_starts[w];
    const int cu_start = w > 0 ? write_cu_lens[w - 1] : 0;
    const int len = write_cu_lens[w] - cu_start;
    const int g = write_group_ids[w];
    int* row_ptr = reinterpret_cast<int*>(out_ptrs[g]) +
                   (long)row * out_strides[g] + start;
    for (int i = threadIdx.x; i < len; i += blockDim.x)
        row_ptr[i] = write_contents[cu_start + i];
}

// input_batch.py::_prepare_pos_seq_lens_kernel. Grid is num_reqs + 1; the
// extra block zero-pads seq_lens[num_reqs:max_num_reqs] for full CUDA graphs.
__global__ void prepare_pos_seq_lens(
    long* __restrict__ pos, int* __restrict__ seq_lens,
    const int* __restrict__ idx_mapping,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ num_computed_tokens, int max_num_reqs) {
    const int req = blockIdx.x;
    const int num_reqs = gridDim.x - 1;
    if (req == num_reqs) {
        for (int i = num_reqs + threadIdx.x; i < max_num_reqs; i += blockDim.x)
            seq_lens[i] = 0;
        return;
    }
    const int req_state_idx = idx_mapping[req];
    const int num_computed = num_computed_tokens[req_state_idx];
    const int start = query_start_loc[req];
    const int query_len = query_start_loc[req + 1] - start;
    if (threadIdx.x == 0) seq_lens[req] = num_computed + query_len;
    for (int i = threadIdx.x; i < query_len; i += blockDim.x)
        pos[start + i] = num_computed + i;
}

// input_batch.py::_prepare_prefill_inputs_kernel. One block per request;
// no-op for requests that are past their prefill.
__global__ void prepare_prefill_inputs(
    int* __restrict__ input_ids, int* __restrict__ next_prefill_tokens,
    const int* __restrict__ idx_mapping,
    const int* __restrict__ query_start_loc,
    const int* __restrict__ all_token_ids, long all_token_ids_stride,
    const int* __restrict__ prefill_lens,
    const int* __restrict__ num_computed_tokens) {
    const int b = blockIdx.x;
    const int req_state_idx = idx_mapping[b];
    const int prefill_len = prefill_lens[req_state_idx];
    const int num_computed = num_computed_tokens[req_state_idx];
    if (num_computed >= prefill_len) return;

    const int query_start = query_start_loc[b];
    const int query_len = query_start_loc[b + 1] - query_start;
    const int* req_ptr = all_token_ids + (long)req_state_idx * all_token_ids_stride;
    for (int i = threadIdx.x; i < query_len; i += blockDim.x)
        input_ids[query_start + i] = req_ptr[num_computed + i];

    const int next_pos = num_computed + query_len;
    if (threadIdx.x == 0 && next_pos < prefill_len)
        next_prefill_tokens[req_state_idx] = req_ptr[next_pos];
}

// input_batch.py::_combine_sampled_and_draft_tokens_kernel. Token stores
// truncate int64 token ids to the int32 input_ids buffer, as Triton does.
__global__ void combine_sampled_and_draft_tokens(
    int* __restrict__ input_ids, const int* __restrict__ idx_mapping,
    const long* __restrict__ last_sampled_tokens,
    const int* __restrict__ query_start_loc, const int* __restrict__ seq_lens,
    const int* __restrict__ prefill_len,
    const long* __restrict__ draft_tokens, long draft_tokens_stride,
    const int* __restrict__ cu_num_logits, long* __restrict__ logits_indices,
    int num_new_sampled_tokens) {
    const int b = blockIdx.x;
    const int req_state_idx = idx_mapping[b];
    const int logits_start_idx = cu_num_logits[b];
    const int num_logits = cu_num_logits[b + 1] - logits_start_idx;
    const int num_draft = num_logits - num_new_sampled_tokens;
    const int query_end = query_start_loc[b + 1];
    const int logits_start = query_end - num_logits;
    for (int i = threadIdx.x; i < num_logits; i += blockDim.x)
        logits_indices[logits_start_idx + i] = logits_start + i;

    const int seq_len = seq_lens[b];
    const int pl = prefill_len[req_state_idx];
    if (seq_len <= pl) return;  // Prefill: no sampled or draft tokens.

    // Keep prompt-tail slots intact; only rewrite generated-token slots.
    if (threadIdx.x == 0 && num_new_sampled_tokens > 0 &&
        seq_len - num_logits >= pl)
        input_ids[logits_start] = (int)last_sampled_tokens[req_state_idx];

    for (int i = threadIdx.x; i < num_draft; i += blockDim.x)
        input_ids[query_end - num_draft + i] =
            (int)draft_tokens[(long)req_state_idx * draft_tokens_stride + i];
}

// input_batch.py::_get_num_sampled_and_rejected_kernel. One thread per req.
__global__ void get_num_sampled_and_rejected(
    int* __restrict__ num_sampled, int* __restrict__ num_rejected,
    const int* __restrict__ seq_lens, const int* __restrict__ cu_num_logits,
    const int* __restrict__ idx_mapping, const int* __restrict__ prefill_len,
    int num_reqs) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_reqs) return;
    const int req_state_idx = idx_mapping[b];
    const bool chunked = seq_lens[b] < prefill_len[req_state_idx];
    const int ns = chunked ? 0 : num_sampled[b];
    num_sampled[b] = ns;
    const int num_logits = cu_num_logits[b + 1] - cu_num_logits[b];
    num_rejected[b] = chunked ? 0 : num_logits - ns;
}

// input_batch.py::_post_update_kernel. One thread per request (the Triton
// kernel is a serial per-request program); nullable pointers mirror the
// optional output_bin_counts / query_start_loc arguments.
__global__ void post_update(
    const int* __restrict__ idx_mapping, int* __restrict__ num_computed_tokens,
    long* __restrict__ last_sampled_tokens,
    int* __restrict__ output_bin_counts, long output_bin_counts_stride,
    const long* __restrict__ sampled_tokens, long sampled_tokens_stride,
    const int* __restrict__ num_sampled, const int* __restrict__ num_rejected,
    const int* __restrict__ query_start_loc,
    int* __restrict__ all_token_ids, long all_token_ids_stride,
    int* __restrict__ total_len, int num_reqs) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= num_reqs) return;
    const int req_state_idx = idx_mapping[r];
    if (req_state_idx < 0) return;  // Filter rows with negative index entries.

    const int tlen = total_len[req_state_idx];
    const int ns = num_sampled[r];
    if (ns > 0) {
        last_sampled_tokens[req_state_idx] =
            sampled_tokens[(long)r * sampled_tokens_stride + ns - 1];
        total_len[req_state_idx] = tlen + ns;
    }
    for (int i = 0; i < ns; i++) {
        const long token_id = sampled_tokens[(long)r * sampled_tokens_stride + i];
        all_token_ids[(long)req_state_idx * all_token_ids_stride + tlen + i] =
            (int)token_id;
        if (output_bin_counts != nullptr)
            output_bin_counts[(long)req_state_idx * output_bin_counts_stride +
                              token_id] += 1;
    }
    const int query_len =
        query_start_loc ? query_start_loc[r + 1] - query_start_loc[r] : 0;
    const int delta = query_len - num_rejected[r];
    if (delta != 0) num_computed_tokens[req_state_idx] += delta;
}

// input_batch.py::_post_update_num_computed_tokens_kernel.
__global__ void post_update_num_computed_tokens(
    const int* __restrict__ idx_mapping, int* __restrict__ num_computed_tokens,
    const int* __restrict__ query_start_loc, int num_reqs) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= num_reqs) return;
    const int query_len = query_start_loc[b + 1] - query_start_loc[b];
    num_computed_tokens[idx_mapping[b]] += query_len;
}

// input_batch.py::_expand_idx_mapping_kernel. One block per request.
__global__ void expand_idx_mapping(
    const int* __restrict__ idx_mapping, int* __restrict__ expanded_idx_mapping,
    int* __restrict__ expanded_local_pos,
    const int* __restrict__ cu_num_logits) {
    const int r = blockIdx.x;
    const int start = cu_num_logits[r];
    const int num_tokens = cu_num_logits[r + 1] - start;
    const int req_state_idx = idx_mapping[r];
    for (int i = threadIdx.x; i < num_tokens; i += blockDim.x) {
        expanded_idx_mapping[start + i] = req_state_idx;
        expanded_local_pos[start + i] = i;
    }
}

// block_table.py (gpu worker)::_gather_block_tables_kernel. Grid is
// (num_kv_cache_groups, num_reqs_padded); padded rows are zeroed in full
// (stride == max_num_blocks), valid rows copy only their live prefix.
__global__ void gather_block_tables(
    const int* __restrict__ idx_mapping,
    const unsigned long long* __restrict__ src_ptrs,
    const unsigned long long* __restrict__ dst_ptrs,
    const long* __restrict__ strides, const int* __restrict__ num_blocks,
    long num_blocks_stride, int num_reqs) {
    const int g = blockIdx.x;
    const int b = blockIdx.y;
    const long stride = strides[g];
    int* dst = reinterpret_cast<int*>(dst_ptrs[g]) + (long)b * stride;
    if (b >= num_reqs) {
        for (long i = threadIdx.x; i < stride; i += blockDim.x) dst[i] = 0;
        return;
    }
    const int req_idx = idx_mapping[b];
    const int nb = num_blocks[(long)g * num_blocks_stride + req_idx];
    const int* src =
        reinterpret_cast<const int*>(src_ptrs[g]) + (long)req_idx * stride;
    for (int i = threadIdx.x; i < nb; i += blockDim.x) dst[i] = src[i];
}

// block_table.py (gpu worker)::_compute_slot_mappings_kernel. Grid is
// (num_kv_cache_groups, num_reqs + 1); the final row pads
// [num_tokens, max_num_tokens) with pad_id per group for CUDA graphs.
__global__ void compute_slot_mappings(
    long max_num_tokens, const int* __restrict__ idx_mapping,
    const int* __restrict__ query_start_loc, const long* __restrict__ pos,
    const unsigned long long* __restrict__ block_table_ptrs,
    const long* __restrict__ block_table_strides,
    const int* __restrict__ block_sizes, long* __restrict__ slot_mappings,
    long slot_mappings_stride, int cp_rank, int cp_size, int cp_interleave,
    long pad_id) {
    const int g = blockIdx.x;
    const int b = blockIdx.y;
    long* slot_mapping = slot_mappings + (long)g * slot_mappings_stride;
    if (b == gridDim.y - 1) {
        // Start from the actual token count to overwrite stale slots left by
        // previous chunks during chunked prefill.
        const long actual_num_tokens = (long)query_start_loc[b];
        for (long i = actual_num_tokens + threadIdx.x; i < max_num_tokens;
             i += blockDim.x)
            slot_mapping[i] = pad_id;
        return;
    }
    const int* block_table = reinterpret_cast<const int*>(block_table_ptrs[g]);
    const long bt_stride = block_table_strides[g];
    const long block_size = (long)block_sizes[g];
    const int req_state_idx = idx_mapping[b];
    const int start = query_start_loc[b];
    const int end = query_start_loc[b + 1];
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        const long p = pos[i];
        const long block_idx = p / (block_size * cp_size);
        const long block_off = p % (block_size * cp_size);
        const long bn =
            (long)block_table[(long)req_state_idx * bt_stride + block_idx];
        long slot;
        if (cp_size == 1) {
            slot = bn * block_size + block_off;
        } else {
            const bool is_local =
                (block_off / cp_interleave) % cp_size == cp_rank;
            const long rounds = block_off / ((long)cp_interleave * cp_size);
            const long local_off =
                rounds * cp_interleave + block_off % cp_interleave;
            slot = is_local ? bn * block_size + local_off : pad_id;
        }
        slot_mapping[i] = slot;
    }
}

// mla/indexer.py::_prepare_uniform_decode_kernel. One block per expanded
// decode token; copies the full expanded block-table row width from the
// source row exactly as the Triton kernel does.
__global__ void prepare_uniform_decode(
    const int* __restrict__ seq_lens, int* __restrict__ decode_seq_lens,
    const int* __restrict__ block_table, long block_table_stride,
    int* __restrict__ expanded_block_table, long expanded_bt_stride,
    int* __restrict__ decode_lens, int max_decode_len) {
    const int idx = blockIdx.x;
    const int req = idx / max_decode_len;
    const int local_idx = idx % max_decode_len;
    if (threadIdx.x == 0) {
        decode_seq_lens[idx] = seq_lens[req] - max_decode_len + local_idx + 1;
        decode_lens[idx] = 1;  // All reqs now have decode_len = 1.
    }
    const int* src = block_table + (long)req * block_table_stride;
    int* dst = expanded_block_table + (long)idx * expanded_bt_stride;
    for (long i = threadIdx.x; i < expanded_bt_stride; i += blockDim.x)
        dst[i] = src[i];
}

// spec_decode/dflash/speculator.py::_prepare_dflash_inputs_kernel. One block
// per request covering [0, max_tokens_per_req); the last request's block also
// runs the CUDA-graph padding sections after a barrier, matching the Triton
// program order (padding wins over any overlapping earlier store).
__global__ void prepare_dflash_inputs(
    int* __restrict__ out_input_ids, long* __restrict__ out_query_positions,
    int* __restrict__ out_query_start_loc, int* __restrict__ out_seq_lens,
    long* __restrict__ out_query_slot_mapping,
    long* __restrict__ out_context_positions,
    long* __restrict__ out_context_slot_mapping,
    long* __restrict__ out_sample_indices, long* __restrict__ out_sample_pos,
    int* __restrict__ out_sample_idx_mapping,
    const long* __restrict__ target_positions,
    const int* __restrict__ target_query_start_loc,
    const int* __restrict__ idx_mapping, const long* __restrict__ last_sampled,
    const int* __restrict__ next_prefill_tokens,
    const int* __restrict__ num_sampled, const int* __restrict__ num_rejected,
    const int* __restrict__ block_table, long block_table_stride,
    int parallel_drafting_token_id, int block_size, int num_query_per_req,
    int num_speculative_steps, int max_num_reqs, long max_num_tokens,
    long max_model_len, int sample_from_anchor, long pad_slot_id,
    int max_tokens_per_req) {
    const int req_idx = blockIdx.x;
    const int num_reqs = gridDim.x;
    const int req_state_idx = idx_mapping[req_idx];

    const int ctx_start = target_query_start_loc[req_idx];
    const int ctx_end = target_query_start_loc[req_idx + 1];
    const int num_ctx = ctx_end - ctx_start;
    const int valid_ctx_end = ctx_end - num_rejected[req_idx];

    // Chunked prefilling has no sampled token; splice in the next prefill one.
    const int bonus_token = num_sampled[req_idx] > 0
                                ? (int)last_sampled[req_state_idx]
                                : next_prefill_tokens[req_state_idx];
    const long last_valid_pos = target_positions[valid_ctx_end - 1];
    const long query_base = (long)req_idx * num_query_per_req;
    const int sample_off = sample_from_anchor ? 0 : 1;
    const long bt_row = (long)req_idx * block_table_stride;

    for (int j = threadIdx.x; j < max_tokens_per_req; j += blockDim.x) {
        if (j < num_ctx) {
            const long ctx_pos = target_positions[ctx_start + j];
            long bn = ctx_pos / block_size;
            bn = min(bn, block_table_stride - 1);
            const long block_id = (long)block_table[bt_row + bn];
            out_context_positions[ctx_start + j] = ctx_pos;
            out_context_slot_mapping[ctx_start + j] =
                block_id * block_size + ctx_pos % block_size;
        } else if (j < num_ctx + num_query_per_req) {
            const int query_off = j - num_ctx;
            const long query_pos = last_valid_pos + 1 + query_off;
            const long query_idx = query_base + query_off;
            long bn = query_pos / block_size;
            bn = min(bn, block_table_stride - 1);
            const long block_id = (long)block_table[bt_row + bn];
            out_input_ids[query_idx] =
                query_off == 0 ? bonus_token : parallel_drafting_token_id;
            out_query_positions[query_idx] = min(query_pos, max_model_len - 1);
            out_query_slot_mapping[query_idx] =
                block_id * block_size + query_pos % block_size;
            // SAMPLE_FROM_ANCHOR (DSpark): every query position predicts the
            // NEXT token. Otherwise the anchor is the bonus token and only
            // mask tokens at offsets > 0 are sampled, each at its own position.
            if (query_off >= sample_off) {
                const long si = (long)req_idx * num_speculative_steps +
                                (query_off - sample_off);
                out_sample_indices[si] = query_idx;
                out_sample_pos[si] =
                    sample_from_anchor ? query_pos + 1 : query_pos;
                out_sample_idx_mapping[si] = req_state_idx;
            }
        }
    }
    if (threadIdx.x == 0) {
        out_query_start_loc[req_idx] = (int)query_base;
        // seq_lens is the absolute length the draft attention reads up to
        // (context + query), not the count of accepted tokens this step.
        out_seq_lens[req_idx] = (int)(last_valid_pos + 1 + num_query_per_req);
    }
    if (req_idx != num_reqs - 1) return;

    // Pad per-request buffers to max_num_reqs for CUDA graph safety. The
    // barrier keeps Triton's in-program store order: padding overwrites any
    // overlapping store from the loops above.
    __syncthreads();
    const long last_query_end = (long)num_reqs * num_query_per_req;
    for (int i = num_reqs + threadIdx.x; i < max_num_reqs + 1; i += blockDim.x)
        out_query_start_loc[i] = (int)last_query_end;
    for (int i = num_reqs + threadIdx.x; i < max_num_reqs; i += blockDim.x)
        out_seq_lens[i] = 0;
    // Padded sample slots point at query index 0 (a valid row) so CG replay
    // never reads OOB; padded idx mappings are -1 and ignored when sampling.
    const long pad_start = (long)num_reqs * num_speculative_steps;
    const long pad_end = (long)max_num_reqs * num_speculative_steps;
    for (long i = pad_start + threadIdx.x; i < pad_end; i += blockDim.x) {
        out_sample_indices[i] = 0;
        out_sample_pos[i] = 0;
        out_sample_idx_mapping[i] = -1;
    }
    // PAD slots (no K/V write) for CG replay sizes above the request count.
    for (long i = last_query_end + threadIdx.x; i < max_num_tokens;
         i += blockDim.x)
        out_query_slot_mapping[i] = pad_slot_id;
}

}  // namespace tms
