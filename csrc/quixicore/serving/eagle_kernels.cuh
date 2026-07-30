/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's EAGLE speculative-decode core
 * (spec_decode.metal): rejection verification (greedy + random) with the
 * externalized-uniform / precomputed-recovered convention, the residual
 * recovered-token argmax, and the eagle_prepare_inputs index math. These
 * complement our spec_verify_linear (which draws its own Gumbel); the EAGLE
 * variants take pre-drawn uniforms and a separate recovered-token pass so a
 * host reference is exact. Metadata kernels are 1 thread/request; the recovered
 * sampler is one warp per draft token (argmax over the residual distribution,
 * smaller-id ties — reuses warp_argmax).
 */
#pragma once
#include "tm_warp.cuh"                   // warp_argmax
#include <cstdint>

namespace tmeg {

// Greedy verify: write target_argmax per draft pos until draft != target, then
// (if none rejected) the bonus token. cu_num_draft_tokens is a prefix sum.
static __global__ void rejection_greedy_sample(int64_t* out, const int64_t* target_argmax,
        const int64_t* draft_token_ids, const int64_t* bonus_token_ids,
        const int* cu_num_draft_tokens, const uint8_t* is_greedy, int has_is_greedy,
        int batch, int out_stride, int bonus_stride) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= batch) return;
    if (has_is_greedy && !is_greedy[req]) return;
    const long start = req == 0 ? 0 : cu_num_draft_tokens[req - 1];
    const long end = cu_num_draft_tokens[req];
    const long nd = end - start;
    int64_t* row = out + (long)req * out_stride;
    bool rejected = false;
    for (long p = 0; p < nd; ++p) {
        const int64_t tgt = target_argmax[start + p];
        row[p] = tgt;
        if (draft_token_ids[start + p] != tgt) { rejected = true; break; }
    }
    if (!rejected) row[nd] = bonus_token_ids[(long)req * bonus_stride];
}

// Random verify: accept draft iff p_target/q_draft >= uniform_probs[token]
// (q=1 if no_draft_probs); else emit the precomputed recovered token and stop.
static __global__ void rejection_random_sample(int64_t* out, const float* target_probs,
        const float* draft_probs, const int64_t* draft_token_ids, const int64_t* bonus_token_ids,
        const int64_t* recovered_token_ids, const float* uniform_probs,
        const int* cu_num_draft_tokens, const uint8_t* is_greedy, int has_is_greedy,
        int no_draft_probs, int batch, int out_stride, int bonus_stride,
        int target_stride, int draft_stride) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= batch) return;
    if (has_is_greedy && is_greedy[req]) return;
    const long start = req == 0 ? 0 : cu_num_draft_tokens[req - 1];
    const long end = cu_num_draft_tokens[req];
    const long nd = end - start;
    int64_t* row = out + (long)req * out_stride;
    bool rejected = false;
    for (long p = 0; p < nd; ++p) {
        const long tok = start + p;
        const int64_t did = draft_token_ids[tok];
        const float pt = target_probs[tok * target_stride + did];
        const float q = no_draft_probs ? 1.0f : draft_probs[tok * draft_stride + did];
        const float ratio = q > 0.0f ? pt / q : 0.0f;
        if (ratio >= uniform_probs[tok]) row[p] = did;
        else { row[p] = recovered_token_ids[tok]; rejected = true; break; }
    }
    if (!rejected) row[nd] = bonus_token_ids[(long)req * bonus_stride];
}

// Residual recovered-token argmax: per draft token, over vocab
// prob = (no_draft ? (v==draft?0:p_t) : max(p_t-p_d,0)) * inv_q[v]; argmax
// (smaller id ties). One warp per draft token; binary-search the request.
static __global__ void sample_recovered_tokens(int64_t* out, const float* target_probs,
        const float* draft_probs, const int64_t* draft_token_ids, const float* inv_q,
        const int* cu_num_draft_tokens, int no_draft_probs, int batch, int total_draft,
        int vocab, int target_stride, int draft_stride, int inv_q_stride) {
    const int tok = blockIdx.x, lane = threadIdx.x;
    if (tok >= total_draft) return;
    int lo = 0, hi = batch;                           // request that owns this token
    while (lo < hi) { const int mid = lo + ((hi - lo) >> 1);
        if (cu_num_draft_tokens[mid] <= tok) lo = mid + 1; else hi = mid; }
    const int req = lo;
    const int64_t did = draft_token_ids[tok];
    const float* tp = target_probs + (long)tok * target_stride;
    const float* dp = no_draft_probs ? draft_probs : draft_probs + (long)tok * draft_stride;
    const float* iq = inv_q + (long)req * inv_q_stride;

    float best = -1.0f; int bi = vocab;
    for (int v = lane; v < vocab; v += 32) {
        float prob = tp[v];
        if (no_draft_probs) { if (v == did) prob = 0.0f; }
        else { const float d = prob - dp[v]; prob = d > 0.0f ? d : 0.0f; }
        const float val = prob * iq[v];
        if (val > best || (val == best && v < bi)) { best = val; bi = v; }
    }
    tms::warp_argmax(best, bi);
    if (lane == 0) out[tok] = bi;
}

// eagle_prepare_inputs_padded: per request index math for the padded draft
// layout. rejected = num_draft>0 ? num_draft+1-valid : 0; sample index = q_last
// - rejected; also writes num_rejected_tokens.
static __global__ void eagle_prepare_inputs_padded(int* token_indices_to_sample,
        int* num_rejected_tokens, const int* cu_num_draft_tokens,
        const int* valid_sampled_tokens_count, const int* query_start_loc, int num_reqs) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= num_reqs) return;
    const long start = req == 0 ? 0 : cu_num_draft_tokens[req - 1];
    const long nd = cu_num_draft_tokens[req] - start;
    const long valid = valid_sampled_tokens_count[req];
    const long rej = nd > 0 ? nd + 1 - valid : 0;
    const int q_last = query_start_loc[req + 1] - 1;
    token_indices_to_sample[req] = q_last - int(rej);
    num_rejected_tokens[req] = int(rej);
}

// eagle_prepare_next_token_padded: pick the next accepted token per request.
// Scan the per-request sampled row for the last in-vocab (!= -1) token; if the
// request is discarded, or none valid, fall back to backup_next_token_ids.
// Also emits the valid-token count. One thread per request.
static __global__ void eagle_prepare_next_token_padded(int64_t* next_token_ids,
        int64_t* valid_sampled_tokens_count, const int64_t* sampled_token_ids,
        const uint8_t* discard_request_mask, const int64_t* backup_next_token_ids,
        int64_t vocab_size, int num_sampled_tokens_per_req, int num_reqs, int64_t sampled_stride) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= num_reqs) return;
    const int64_t* row = sampled_token_ids + (long)req * sampled_stride;
    int64_t valid = 0, last_valid = -1;
    for (int pos = 0; pos < num_sampled_tokens_per_req; ++pos) {
        const int64_t tok = row[pos];
        if (tok != -1 && tok < vocab_size) { valid += 1; last_valid = tok; }
    }
    if (discard_request_mask[req] != 0) {
        next_token_ids[req] = backup_next_token_ids[req];
        valid_sampled_tokens_count[req] = 0;
    } else {
        next_token_ids[req] = valid > 0 ? last_valid : backup_next_token_ids[req];
        valid_sampled_tokens_count[req] = valid;
    }
}

// eagle_step_slot_mapping_metadata: advance one decode step. Clamp position+1 to
// max_model_len (overflow -> pad), resolve the paged slot through block_table,
// and bump seq_lens. Requests past batch_size are padding (slot = pad_id). One
// thread per (input) request.
static __global__ void eagle_step_slot_mapping_metadata(int* seq_lens,
        int64_t* out_clamped_positions, int64_t* out_slot_mapping,
        const int64_t* positions, const int* block_table,
        int64_t block_size, int64_t max_model_len, int64_t pad_id,
        int batch_size, int input_batch_size, int64_t block_table_stride, int64_t n_blocks_per_req) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= input_batch_size) return;
    if (req >= batch_size) { out_slot_mapping[req] = pad_id; return; }

    const int64_t new_position = positions[req] + 1;
    const bool exceeds = new_position >= max_model_len;
    const int64_t clamped = exceeds ? 0 : new_position;
    out_clamped_positions[req] = clamped;

    int64_t block_number = clamped / block_size;
    block_number = min(block_number, n_blocks_per_req - 1);
    const int block_id = block_table[(long)req * block_table_stride + block_number];
    const int64_t slot = (int64_t)block_id * block_size + (clamped % block_size);
    out_slot_mapping[req] = exceeds ? pad_id : slot;

    const int seq_len = seq_lens[req];
    seq_lens[req] = exceeds ? 1 : min(seq_len + 1, int(max_model_len));
}

// eagle_expand_int64: broadcast input[req] across its [cu_num_tokens[req-1],
// cu_num_tokens[req]) output span, optionally remapping a sentinel value. One
// thread per request (serial fill of the span).
static __global__ void eagle_expand_int64(int64_t* output, const int64_t* input,
        const int64_t* cu_num_tokens, int64_t replace_from, int64_t replace_to, int batch_size) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= batch_size) return;
    const long start = req == 0 ? 0 : cu_num_tokens[req - 1];
    const long end = cu_num_tokens[req];
    int64_t value = input[req];
    if (value == replace_from) value = replace_to;
    for (long i = start; i < end; ++i) output[i] = value;
}

// copy_and_expand_eagle_inputs: build the padded EAGLE draft input layout per
// request. Each output slot is one of {valid target token, bonus token, parallel
// -drafting placeholder, rejected}; positions, rejected/masked masks, the
// new-token index map, and (when shifting) the hidden-state map are filled to
// match. One thread per request (serial over the request's output span).
static __global__ void copy_and_expand_eagle_inputs(const int64_t* target_token_ids,
        const int64_t* target_positions, const int64_t* next_token_ids,
        int64_t* out_input_ids, int64_t* out_positions, uint8_t* out_is_rejected_token_mask,
        uint8_t* out_is_masked_token_mask, int* out_new_token_indices, int* out_hidden_state_mapping,
        const int* query_start_loc, const int* query_end_loc, int64_t padding_token_id,
        int64_t parallel_drafting_token_id, int64_t total_input_tokens,
        int64_t num_padding_slots_per_request, int shift_input_ids, int num_reqs) {
    const int req = blockIdx.x * blockDim.x + threadIdx.x;
    if (req >= num_reqs) return;
    const int q_start = query_start_loc[req];
    const int next_q_start = query_start_loc[req + 1];
    const int q_end = query_end_loc[req];
    const bool shift = shift_input_ids != 0;
    const long num_valid = shift ? (q_end - q_start) : (q_end - q_start + 1);
    const long input_offset = shift ? 1 : 0;
    const long out_start = q_start + (long)req * (num_padding_slots_per_request - (shift ? 1 : 0));
    const long num_rejected = next_q_start - q_end - 1;
    const long total_output = num_valid + num_padding_slots_per_request + num_rejected;
    const long start_pos = target_positions[q_start];
    const long bonus_token = next_token_ids[req];

    for (long j = 0; j < total_output; ++j) {
        const long out_idx = out_start + j;
        const bool is_valid = j < num_valid;
        const bool is_bonus = j == num_valid;
        const bool is_parallel = (j > num_valid) && (j < num_valid + num_padding_slots_per_request);
        const bool is_rejected = j >= num_valid + num_padding_slots_per_request;
        const long in_idx = min((long)q_start + input_offset + j, total_input_tokens - 1);

        int64_t token_id = padding_token_id;
        if (is_valid) token_id = target_token_ids[in_idx];
        else if (is_bonus) token_id = bonus_token;
        else if (is_parallel) token_id = parallel_drafting_token_id;

        out_input_ids[out_idx] = token_id;
        out_positions[out_idx] = is_rejected ? 0 : (start_pos + j);
        out_is_rejected_token_mask[out_idx] = is_rejected ? 1 : 0;
        out_is_masked_token_mask[out_idx] = is_parallel ? 1 : 0;

        if (is_bonus || is_parallel) {
            const long local_idx = j - num_valid;
            const long new_token_idx = (long)req * num_padding_slots_per_request + local_idx;
            out_new_token_indices[new_token_idx] = int(out_idx);
        }
    }
    if (shift) {
        const long n_input = next_q_start - q_start;
        for (long j = 0; j < n_input; ++j)
            out_hidden_state_mapping[q_start + j] = int(out_start + j);
    }
}

} // namespace tmeg
