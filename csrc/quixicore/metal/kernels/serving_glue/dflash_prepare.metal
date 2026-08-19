// Native DFlash input prep (Metal port of the Triton
// _prepare_dflash_inputs_kernel). Replaces the MPS fallback in
// vllm/v1/worker/gpu/spec_decode/dflash/speculator.py -- a per-request
// Python loop with two GPU->CPU syncs and ~25 small dispatches per step
// (~5 ms of the drafter's propose time). One dispatch, no syncs; the
// semantics mirror the Triton kernel line for line so greedy output stays
// token-identical.
//
// Grid: (ceil(max_tokens_per_req / 256), num_reqs, 1), 256 threads.
#include <metal_stdlib>

namespace mittens {

kernel void prepare_dflash_inputs(
    device int*        out_input_ids [[buffer(0)]],   // int32 (max_num_tokens)
    device long*       out_positions [[buffer(1)]],   // int64
    device int*        out_qsl       [[buffer(2)]],   // int32 (max_num_reqs+1)
    device int*        out_seq_lens  [[buffer(3)]],   // int32
    device long*       out_qslot     [[buffer(4)]],   // int64
    device long*       out_ctx_pos   [[buffer(5)]],   // int64
    device long*       out_ctx_slot  [[buffer(6)]],   // int64
    device long*       out_samp_idx  [[buffer(7)]],   // int64
    device long*       out_samp_pos  [[buffer(8)]],   // int64
    device int*        out_samp_map  [[buffer(9)]],   // int32
    device const long* tgt_pos       [[buffer(10)]],  // int64
    device const int*  tgt_qsl       [[buffer(11)]],  // int32
    device const int*  idx_mapping   [[buffer(12)]],  // int32
    device const long* last_sampled  [[buffer(13)]],  // int64 (R, 1)
    device const int*  next_prefill  [[buffer(14)]],  // int32
    device const int*  num_sampled   [[buffer(15)]],  // int32
    device const int*  num_rejected  [[buffer(16)]],  // int32
    device const int*  block_table   [[buffer(17)]],  // int32
    const constant int& bt_stride       [[buffer(18)]],
    const constant int& pdt_id          [[buffer(19)]],
    const constant int& block_size      [[buffer(20)]],
    const constant int& nq_per_req      [[buffer(21)]],
    const constant int& n_spec_steps    [[buffer(22)]],
    const constant int& max_num_reqs    [[buffer(23)]],
    const constant int& max_num_tokens  [[buffer(24)]],
    const constant int& max_model_len   [[buffer(25)]],
    const constant int& sample_anchor   [[buffer(26)]],
    const constant int& pad_slot_id     [[buffer(27)]],
    const constant int& num_reqs        [[buffer(28)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
    const int req_idx = (int)tgid.y;
    const int req_state_idx = idx_mapping[req_idx];
    const int ctx_start = tgt_qsl[req_idx];
    const int ctx_end = tgt_qsl[req_idx + 1];
    const int num_ctx = ctx_end - ctx_start;
    const int valid_ctx_end = ctx_end - num_rejected[req_idx];
    const int bonus = num_sampled[req_idx] > 0
                          ? (int)last_sampled[req_state_idx]
                          : next_prefill[req_state_idx];
    const long last_valid_pos = tgt_pos[valid_ctx_end - 1];
    const int query_base = req_idx * nq_per_req;

    const int j = (int)(tgid.x * 256 + tid);
    if (j < num_ctx) {
        const long cpos = tgt_pos[ctx_start + j];
        long cbn = metal::min(cpos / block_size, (long)(bt_stride - 1));
        const long cbid = (long)block_table[req_idx * bt_stride + (int)cbn];
        out_ctx_pos[ctx_start + j] = cpos;
        out_ctx_slot[ctx_start + j] = cbid * block_size + (cpos % block_size);
    } else if (j < num_ctx + nq_per_req) {
        const int qoff = j - num_ctx;
        const long qpos = last_valid_pos + 1 + qoff;
        const int qidx = query_base + qoff;
        out_input_ids[qidx] = (qoff == 0) ? bonus : pdt_id;
        long qbn = metal::min(qpos / block_size, (long)(bt_stride - 1));
        const long qbid = (long)block_table[req_idx * bt_stride + (int)qbn];
        out_qslot[qidx] = qbid * block_size + (qpos % block_size);
        out_positions[qidx] = metal::min(qpos, (long)(max_model_len - 1));
        const int soff = sample_anchor ? 0 : 1;
        if (qoff >= soff) {
            const int sidx = req_idx * n_spec_steps + (qoff - soff);
            out_samp_idx[sidx] = qidx;
            out_samp_pos[sidx] = sample_anchor ? qpos + 1 : qpos;
            out_samp_map[sidx] = req_state_idx;
        }
    }
    if (tgid.x == 0 && tid == 0) {
        out_qsl[req_idx] = query_base;
        out_seq_lens[req_idx] = (int)(last_valid_pos + 1) + nq_per_req;
    }
    if (tgid.x == 0 && req_idx == num_reqs - 1) {
        // pad per-request buffers (CUDA-graph-safety mirror of the Triton
        // tail), parallelized over this threadgroup's 256 threads
        const int last_query_end = num_reqs * nq_per_req;
        for (int i = num_reqs + (int)tid; i < max_num_reqs + 1; i += 256)
            out_qsl[i] = last_query_end;
        for (int i = num_reqs + (int)tid; i < max_num_reqs; i += 256)
            out_seq_lens[i] = 0;
        const int pad_start = num_reqs * n_spec_steps;
        const int pad_end = max_num_reqs * n_spec_steps;
        for (int i = pad_start + (int)tid; i < pad_end; i += 256) {
            out_samp_idx[i] = 0;
            out_samp_pos[i] = 0;
            out_samp_map[i] = -1;
        }
        for (int i = last_query_end + (int)tid; i < max_num_tokens; i += 256)
            out_qslot[i] = (long)pad_slot_id;
    }
}

}  // namespace mittens
