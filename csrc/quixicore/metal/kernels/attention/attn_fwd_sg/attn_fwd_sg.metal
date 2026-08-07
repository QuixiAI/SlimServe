#include <metal_stdlib>

using namespace metal;

namespace mittens {

// ---------------------------------------------------------------------------
// attn_fwd_sg_d256: simdgroup_matrix (MMA) flash attention, head-dim 256, GQA,
// f16 KV. An alternate attention forward built on raw simdgroup_float8x8 tiles
// (Metal 3 8x8 cooperative matrices) instead of QuixiCore's TK register tiles —
// the head_dim=256 shape is too wide for the TK D in {64,128} attn_fwd path.
//
// Bidirectional (non-causal) attention with an optional symmetric sliding window.
// GQA: Hq query heads share Hkv KV heads (group G = Hq/Hkv); f32 Q/O, f16 K/V.
// Layout is token-major (T, H, 256): q/o (T, Hq, 256) f32, k/v (T, Hkv, 256) f16.
// Q is scaled by `scale` on load (pass 1/sqrt(256) for standard attention).
//
// One threadgroup (4 simdgroups / 128 lanes) owns 8 query rows of one head; grid
// (ceil(T/8), Hq, 1). Per 32-key block: 4 simdgroups compute the 8x32 QK^T tile
// (each owns 8 keys), a warp-parallel online softmax rescales the running output,
// and the 4 simdgroups split the 256 output columns for the P·V update. window
// == 0 is full attention; window > 0 keeps keys within window/2 of the query.
// ---------------------------------------------------------------------------
kernel void attn_fwd_sg_d256(
    device const float *q     [[buffer(0)]],   // (T, Hq, 256)  f32
    device const half  *k     [[buffer(1)]],   // (T, Hkv, 256) f16
    device const half  *v     [[buffer(2)]],   // (T, Hkv, 256) f16
    device float       *o     [[buffer(3)]],   // (T, Hq, 256)  f32
    constant uint  &n_tokens  [[buffer(4)]],
    constant uint  &window    [[buffer(5)]],   // 0 = full; else symmetric half-width*2
    constant float &scale     [[buffer(6)]],   // Q pre-scale (e.g. 1/sqrt(256))
    constant uint  &Hq        [[buffer(7)]],
    constant uint  &Hkv       [[buffer(8)]],
    uint3  group     [[threadgroup_position_in_grid]],
    ushort lane      [[thread_index_in_simdgroup]],
    ushort simdgroup [[simdgroup_index_in_threadgroup]],
    ushort tid       [[thread_index_in_threadgroup]]) {
    constexpr uint queries_per_group = 8;
    constexpr uint keys_per_group = 32;
    constexpr uint head_dim = 256;
    constexpr uint simdgroups = 4;

    threadgroup half  q_tile[queries_per_group * head_dim];
    threadgroup float scores[queries_per_group * keys_per_group];
    threadgroup float accum[queries_per_group * head_dim];
    threadgroup float row_max[queries_per_group];
    threadgroup float row_sum[queries_per_group];

    const uint query_start = group.x * queries_per_group;
    const uint head = group.y;
    const uint group_size = Hq / Hkv;         // GQA group
    const uint kv_head = head / group_size;
    const uint q_stride = Hq * head_dim;       // row stride of q/o
    const uint kv_stride = Hkv * head_dim;     // row stride of k/v
    const uint q_head_off = head * head_dim;
    const uint kv_head_off = kv_head * head_dim;

    for (uint item = tid; item < queries_per_group * head_dim;
         item += 32 * simdgroups) {
        const uint local_query = item / head_dim;
        const uint dim = item % head_dim;
        const uint query = query_start + local_query;
        q_tile[item] = query < n_tokens
            ? half(q[query * q_stride + q_head_off + dim] * scale)
            : 0.0h;
        accum[item] = 0.0f;
    }
    if (tid < queries_per_group) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint key_begin = 0;
    uint key_end = n_tokens;
    if (window != 0) {
        const uint half_window = window / 2;
        key_begin = query_start > half_window ? query_start - half_window : 0;
        key_begin = (key_begin / keys_per_group) * keys_per_group;
        key_end = min(n_tokens, query_start + queries_per_group + half_window);
    }

    for (uint key_start = key_begin; key_start < key_end;
         key_start += keys_per_group) {
        // QK^T: this simdgroup owns 8 keys (key_start + 8*simdgroup), all 8 queries.
        simdgroup_float8x8 qk = make_filled_simdgroup_matrix<float, 8>(0.0f);
#pragma unroll(8)
        for (ushort pair = 0; pair < head_dim / 16; pair++) {
            simdgroup_half8x8 mq[2];
            simdgroup_half8x8 mk[2];
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mq[0], q_tile + 16 * pair, head_dim);
            simdgroup_load(mq[1], q_tile + 16 * pair + 8, head_dim);
            device const half *key_ptr =
                k + (key_start + 8 * simdgroup) * kv_stride + kv_head_off + 16 * pair;
            simdgroup_load(mk[0], key_ptr, kv_stride, 0, true);
            simdgroup_load(mk[1], key_ptr + 8, kv_stride, 0, true);
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_multiply_accumulate(qk, mq[0], mk[0], qk);
            simdgroup_multiply_accumulate(qk, mq[1], mk[1], qk);
        }
        simdgroup_store(qk, scores + 8 * simdgroup, keys_per_group, 0, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax: each simdgroup owns 2 query rows, rescales its running output.
#pragma clang loop unroll(full)
        for (ushort local = 0; local < 2; local++) {
            const ushort row = 2 * simdgroup + local;
            const uint query = query_start + row;
            const uint key_index = key_start + lane;
            const uint half_window = window / 2;
            const bool valid = query < n_tokens && key_index < n_tokens &&
                (window == 0 ||
                 (key_index + half_window >= query && key_index <= query + half_window));
            const float score = valid ? scores[row * keys_per_group + lane] : -INFINITY;
            if (query < n_tokens) {
                const float previous_max = row_max[row];
                const float next_max = max(previous_max, simd_max(score));
                const float alpha = exp(previous_max - next_max);
                const float probability = valid ? exp(score - next_max) : 0.0f;
                const float next_sum = row_sum[row] * alpha + simd_sum(probability);
                scores[row * keys_per_group + lane] = probability;
                for (uint dim = lane; dim < head_dim; dim += 32) {
                    accum[row * head_dim + dim] *= alpha;
                }
                if (lane == 0) {
                    row_max[row] = next_max;
                    row_sum[row] = next_sum;
                }
            } else {
                scores[row * keys_per_group + lane] = 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // P·V: 4 simdgroups split the 256 output dims; each owns a 64-col stripe.
#pragma clang loop unroll(full)
        for (ushort tile = 0; tile < 8; tile++) {
            const ushort dim = simdgroup * 64 + tile * 8;
            simdgroup_float8x8 out_matrix;
            simdgroup_load(out_matrix, accum + dim, head_dim, 0, false);
#pragma clang loop unroll(full)
            for (ushort key_tile = 0; key_tile < 4; key_tile++) {
                simdgroup_float8x8 probability_matrix;
                simdgroup_half8x8 value_matrix;
                simdgroup_load(probability_matrix, scores + key_tile * 8,
                               keys_per_group, 0, false);
                device const half *value_ptr =
                    v + (key_start + key_tile * 8) * kv_stride + kv_head_off + dim;
                simdgroup_load(value_matrix, value_ptr, kv_stride, 0, false);
                simdgroup_barrier(mem_flags::mem_none);
                simdgroup_multiply_accumulate(out_matrix, probability_matrix,
                                              value_matrix, out_matrix);
            }
            simdgroup_store(out_matrix, accum + dim, head_dim, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint item = tid; item < queries_per_group * head_dim;
         item += 32 * simdgroups) {
        const uint local_query = item / head_dim;
        const uint dim = item % head_dim;
        const uint query = query_start + local_query;
        if (query < n_tokens) {
            const float denom = row_sum[local_query];
            const float rescale = denom == 0.0f ? 0.0f : 1.0f / denom;
            o[query * q_stride + q_head_off + dim] = accum[item] * rescale;
        }
    }
}

}  // namespace mittens
