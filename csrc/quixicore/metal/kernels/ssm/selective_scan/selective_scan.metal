#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// ---------------------------------------------------------------------------
// Mamba-1 (S6) selective scan forward — sequential in time, parallel over the state
// dimension. One threadgroup per (batch, dim) channel, one thread per state index
// (dstate <= 256); fp32 state always, io templated over float/half/bf16.
//
//   delta' = softplus?(delta + delta_bias)
//   h_n    = exp(delta' * A[d,n]) * h_n + B[g,n,t] * delta' * u[d,t]
//   y[d,t] = D[d]*u[d,t] + sum_n h_n * C[g,n,t]     (thread 0 reduces)
//   y     *= silu(z)                                 (optional gate)
//
// Layouts (channel-major, the mamba conv convention): u/delta/z/out (batch, dim, seqlen);
// B/C (batch, n_groups, dstate, seqlen); A (dim, dstate) f32; D/delta_bias (dim,) f32;
// state (batch, dim, dstate) f32 in/out. Optional D/delta_bias/z: bind a 1-elem dummy
// with the has_* flag 0. The varlen variant flattens the batch into total_tokens
// (u/delta/z/out (dim, total_tokens); B/C (n_groups, dstate, total_tokens)) with
// query_start_loc (B+1), per-request paged state slots via cache_indices, and
// has_initial_state gating the state load (fresh prefill starts at h = 0).
// Port of metal-forge selective_scan.metal (vLLM mamba semantics), with tk-native bf16
// instead of the reference's uint16 bit-twiddling.
// ---------------------------------------------------------------------------

constant int SSCAN_MAX_DSTATE = 256;

METAL_FUNC float sscan_softplus(float x) {
    return x <= 20.0f ? metal::log(1.0f + metal::exp(x)) : x;
}

METAL_FUNC void sscan_store_partial(threadgroup float* partial, uint thread_idx, float v) {
    const float simd_total = metal::simd_sum(v);
    if ((thread_idx & 31u) == 0u) {
        partial[thread_idx >> 5] = simd_total;
    }
}

METAL_FUNC float sscan_reduce_partials(threadgroup const float* partial, int dstate) {
    if (dstate <= 32)  { return partial[0]; }
    if (dstate <= 64)  { return partial[0] + partial[1]; }
    if (dstate <= 128) { return partial[0] + partial[1] + partial[2] + partial[3]; }
    float total = 0.0f;
    const int groups = (dstate + 31) / 32;
    for (int i = 0; i < groups; ++i) { total += partial[i]; }
    return total;
}

template <typename T>
kernel void selective_scan_fwd_dense(
    const device T *u            [[buffer(0)]],
    const device T *delta        [[buffer(1)]],
    const device float *A        [[buffer(2)]],
    const device T *B            [[buffer(3)]],
    const device T *C            [[buffer(4)]],
    const device float *D        [[buffer(5)]],
    const device float *delta_bias [[buffer(6)]],
    const device T *z            [[buffer(7)]],
    device T *out                [[buffer(8)]],
    device float *state          [[buffer(9)]],
    constant int &batch          [[buffer(10)]],
    constant int &dim            [[buffer(11)]],
    constant int &seqlen         [[buffer(12)]],
    constant int &dstate         [[buffer(13)]],
    constant int &n_groups       [[buffer(14)]],
    constant int &has_d          [[buffer(15)]],
    constant int &has_delta_bias [[buffer(16)]],
    constant int &has_z          [[buffer(17)]],
    constant int &delta_softplus [[buffer(18)]],
    uint3 row [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]) {
    threadgroup float partial[SSCAN_MAX_DSTATE / 32];

    const uint thread_idx = thread_pos.x;
    const int batch_idx = int(row.x);
    const int dim_idx = int(row.y);
    if (batch_idx >= batch || dim_idx >= dim || int(thread_idx) >= SSCAN_MAX_DSTATE) {
        return;
    }

    const int group_idx = dim_idx / (dim / n_groups);
    const int state_idx = int(thread_idx);
    const bool active_state = state_idx < dstate;

    const long ud_base = ((long)batch_idx * dim + dim_idx) * seqlen;
    const long bc_base = ((long)batch_idx * n_groups + group_idx) * dstate * seqlen;
    const long state_base = ((long)batch_idx * dim + dim_idx) * dstate;

    float running = active_state ? state[state_base + state_idx] : 0.0f;
    const float d_val = has_d != 0 ? D[dim_idx] : 0.0f;
    const float bias_val = has_delta_bias != 0 ? delta_bias[dim_idx] : 0.0f;
    const float a_val = active_state ? A[(long)dim_idx * dstate + state_idx] : 0.0f;

    for (int t = 0; t < seqlen; ++t) {
        const long token_idx = ud_base + t;
        const float u_val = float(u[token_idx]);
        float delta_val = float(delta[token_idx]) + bias_val;
        if (delta_softplus != 0) {
            delta_val = sscan_softplus(delta_val);
        }
        float contribution = 0.0f;
        if (active_state) {
            const long bc_idx = bc_base + (long)state_idx * seqlen + t;
            running = metal::exp(delta_val * a_val) * running
                      + float(B[bc_idx]) * delta_val * u_val;
            contribution = running * float(C[bc_idx]);
        }
        sscan_store_partial(partial, thread_idx, contribution);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (thread_idx == 0) {
            float sum = d_val * u_val + sscan_reduce_partials(partial, dstate);
            if (has_z != 0) {
                const float z_val = float(z[token_idx]);
                sum *= z_val / (1.0f + metal::exp(-z_val));
            }
            out[token_idx] = T(sum);
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    if (active_state) {
        state[state_base + state_idx] = running;
    }
}

template <typename T>
kernel void selective_scan_fwd_varlen(
    const device T *u            [[buffer(0)]],
    const device T *delta        [[buffer(1)]],
    const device float *A        [[buffer(2)]],
    const device T *B            [[buffer(3)]],
    const device T *C            [[buffer(4)]],
    const device float *D        [[buffer(5)]],
    const device float *delta_bias [[buffer(6)]],
    const device T *z            [[buffer(7)]],
    const device int *query_start_loc [[buffer(8)]],
    const device int *cache_indices   [[buffer(9)]],
    const device uchar *has_initial_state [[buffer(10)]],
    device T *out                [[buffer(11)]],
    device float *state          [[buffer(12)]],
    constant int &batch          [[buffer(13)]],
    constant int &dim            [[buffer(14)]],
    constant int &total_tokens   [[buffer(15)]],
    constant int &dstate         [[buffer(16)]],
    constant int &n_groups       [[buffer(17)]],
    constant int &has_d          [[buffer(18)]],
    constant int &has_delta_bias [[buffer(19)]],
    constant int &has_z          [[buffer(20)]],
    constant int &delta_softplus [[buffer(21)]],
    constant int &use_cache_indices [[buffer(22)]],
    constant int &use_has_initial_state [[buffer(23)]],
    constant int &null_block_id  [[buffer(24)]],
    uint3 row [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]) {
    threadgroup float partial[SSCAN_MAX_DSTATE / 32];

    const uint thread_idx = thread_pos.x;
    const int batch_idx = int(row.x);
    const int dim_idx = int(row.y);
    if (batch_idx >= batch || dim_idx >= dim || int(thread_idx) >= SSCAN_MAX_DSTATE) {
        return;
    }

    const int cache_idx = use_cache_indices != 0 ? cache_indices[batch_idx] : batch_idx;
    if (use_cache_indices != 0 && cache_idx == null_block_id) {
        return;
    }
    const int seq_start = query_start_loc[batch_idx];
    const int seq_end = query_start_loc[batch_idx + 1];
    if (seq_start < 0 || seq_end < seq_start || seq_end > total_tokens) {
        return;
    }

    const int group_idx = dim_idx / (dim / n_groups);
    const int state_idx = int(thread_idx);
    const bool active_state = state_idx < dstate;

    // varlen layouts drop the batch axis: token position indexes the flattened axis
    const long ud_base = (long)dim_idx * total_tokens;
    const long bc_base = (long)group_idx * dstate * total_tokens;
    const long state_base = ((long)cache_idx * dim + dim_idx) * dstate;

    const bool load_initial = use_has_initial_state != 0 && has_initial_state[batch_idx] != 0;
    float running = (active_state && load_initial) ? state[state_base + state_idx] : 0.0f;
    const float d_val = has_d != 0 ? D[dim_idx] : 0.0f;
    const float bias_val = has_delta_bias != 0 ? delta_bias[dim_idx] : 0.0f;
    const float a_val = active_state ? A[(long)dim_idx * dstate + state_idx] : 0.0f;

    for (int t = seq_start; t < seq_end; ++t) {
        const long token_idx = ud_base + t;
        const float u_val = float(u[token_idx]);
        float delta_val = float(delta[token_idx]) + bias_val;
        if (delta_softplus != 0) {
            delta_val = sscan_softplus(delta_val);
        }
        float contribution = 0.0f;
        if (active_state) {
            const long bc_idx = bc_base + (long)state_idx * total_tokens + t;
            running = metal::exp(delta_val * a_val) * running
                      + float(B[bc_idx]) * delta_val * u_val;
            contribution = running * float(C[bc_idx]);
        }
        sscan_store_partial(partial, thread_idx, contribution);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        if (thread_idx == 0) {
            float sum = d_val * u_val + sscan_reduce_partials(partial, dstate);
            if (has_z != 0) {
                const float z_val = float(z[token_idx]);
                sum *= z_val / (1.0f + metal::exp(-z_val));
            }
            out[token_idx] = T(sum);
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }

    if (active_state) {
        state[state_base + state_idx] = running;
    }
}

// varlen + automatic prefix caching (APC): identical S6 recurrence, but the running state is
// checkpointed into PAGED state blocks at chunk boundaries and the initial state is loaded
// from a (possibly cached) prefix block. Buffer table transcribed 1:1 from metal-forge's
// selective_scan_fwd_varlen_apc_state_float32_typed so the paged-KV scheduler drops in
// unchanged. use_chunk_metadata=0 falls back to fixed block_size chunking.
template <typename T>
kernel void selective_scan_fwd_varlen_apc(
    const device T *u            [[buffer(0)]],
    const device T *delta        [[buffer(1)]],
    const device float *A        [[buffer(2)]],
    const device T *B            [[buffer(3)]],
    const device T *C            [[buffer(4)]],
    const device float *D        [[buffer(5)]],
    const device float *delta_bias [[buffer(6)]],
    const device T *z            [[buffer(7)]],
    const device int *query_start_loc [[buffer(8)]],
    const device int *cache_indices   [[buffer(9)]],
    const device uchar *has_initial_state [[buffer(10)]],
    device T *out                [[buffer(11)]],
    device float *state          [[buffer(12)]],
    const device int *block_idx_first_scheduled_token [[buffer(13)]],
    const device int *block_idx_last_scheduled_token  [[buffer(14)]],
    const device int *initial_state_idx [[buffer(15)]],
    const device int *cu_chunk_seqlen   [[buffer(16)]],
    const device int *last_chunk_indices [[buffer(17)]],
    constant int &batch          [[buffer(18)]],
    constant int &dim            [[buffer(19)]],
    constant int &total_tokens   [[buffer(20)]],
    constant int &dstate         [[buffer(21)]],
    constant int &n_groups       [[buffer(22)]],
    constant int &has_d          [[buffer(23)]],
    constant int &has_delta_bias [[buffer(24)]],
    constant int &has_z          [[buffer(25)]],
    constant int &delta_softplus [[buffer(26)]],
    constant int &null_block_id  [[buffer(27)]],
    constant int &block_size     [[buffer(28)]],
    constant int &cache_indices_stride [[buffer(29)]],
    constant int &use_chunk_metadata   [[buffer(30)]],
    uint3 row [[threadgroup_position_in_grid]],
    uint3 thread_pos [[thread_position_in_threadgroup]]) {
    threadgroup float partial[SSCAN_MAX_DSTATE / 32];

    const uint thread_idx = thread_pos.x;
    const int batch_idx = int(row.x);
    const int dim_idx = int(row.y);
    if (batch_idx >= batch || dim_idx >= dim || int(thread_idx) >= SSCAN_MAX_DSTATE) {
        return;
    }

    const int state_idx = int(thread_idx);
    const bool active_state = state_idx < dstate;
    const int initial_block_idx = initial_state_idx[batch_idx];
    const int initial_slot =
        cache_indices[batch_idx * cache_indices_stride + initial_block_idx];
    if (initial_slot == null_block_id) {
        return;
    }

    const int seq_start = query_start_loc[batch_idx];
    const int seq_end = query_start_loc[batch_idx + 1];
    if (seq_start < 0 || seq_end < seq_start || seq_end > total_tokens) {
        return;
    }
    const int seqlen = seq_end - seq_start;
    if (seqlen <= 0) {
        return;
    }

    const int group_idx = dim_idx / (dim / n_groups);
    const long ud_base = (long)dim_idx * total_tokens;
    const long bc_base = (long)group_idx * dstate * total_tokens;
    const long initial_state_base = ((long)initial_slot * dim + dim_idx) * dstate;

    const bool load_initial = has_initial_state[batch_idx] != 0;
    float running = (active_state && load_initial) ? state[initial_state_base + state_idx] : 0.0f;
    const float d_val = has_d != 0 ? D[dim_idx] : 0.0f;
    const float bias_val = has_delta_bias != 0 ? delta_bias[dim_idx] : 0.0f;
    const float a_val = active_state ? A[(long)dim_idx * dstate + state_idx] : 0.0f;

    const int safe_block_size = metal::max(block_size, 1);
    int first_chunk_idx = 0;
    int n_chunks = (seqlen + safe_block_size - 1) / safe_block_size;
    int current_position = 0;
    if (use_chunk_metadata != 0) {
        const int last_chunk_idx = last_chunk_indices[batch_idx];
        first_chunk_idx = batch_idx == 0 ? 0 : last_chunk_indices[batch_idx - 1] + 1;
        n_chunks = last_chunk_idx - first_chunk_idx + 1;
        const int first_chunk_tokens =
            cu_chunk_seqlen[first_chunk_idx + 1] - cu_chunk_seqlen[first_chunk_idx];
        const int chunk_start_offset =
            (n_chunks > 1 && first_chunk_tokens < safe_block_size)
                ? (safe_block_size - first_chunk_tokens) : 0;
        current_position =
            block_idx_first_scheduled_token[batch_idx] * safe_block_size + chunk_start_offset;
    }

    int tokens_processed = 0;
    for (int chunk = 0; chunk < n_chunks && tokens_processed < seqlen; ++chunk) {
        int chunk_tokens = metal::min(safe_block_size, seqlen - tokens_processed);
        if (use_chunk_metadata != 0) {
            chunk_tokens = cu_chunk_seqlen[first_chunk_idx + chunk + 1] -
                           cu_chunk_seqlen[first_chunk_idx + chunk];
            chunk_tokens = metal::min(chunk_tokens, seqlen - tokens_processed);
        }
        if (chunk_tokens <= 0) {
            break;
        }
        const int chunk_start = seq_start + tokens_processed;
        for (int offset = 0; offset < chunk_tokens; ++offset) {
            const int t = chunk_start + offset;
            const long token_idx = ud_base + t;
            const float u_val = float(u[token_idx]);
            float delta_val = float(delta[token_idx]) + bias_val;
            if (delta_softplus != 0) {
                delta_val = sscan_softplus(delta_val);
            }
            float contribution = 0.0f;
            if (active_state) {
                const long bc_idx = bc_base + (long)state_idx * total_tokens + t;
                running = metal::exp(delta_val * a_val) * running
                          + float(B[bc_idx]) * delta_val * u_val;
                contribution = running * float(C[bc_idx]);
            }
            sscan_store_partial(partial, thread_idx, contribution);
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
            if (thread_idx == 0) {
                float sum = d_val * u_val + sscan_reduce_partials(partial, dstate);
                if (has_z != 0) {
                    const float z_val = float(z[token_idx]);
                    sum *= z_val / (1.0f + metal::exp(-z_val));
                }
                out[token_idx] = T(sum);
            }
            threadgroup_barrier(metal::mem_flags::mem_threadgroup);
        }

        // checkpoint the running state into the paged block that owns this chunk boundary
        if (active_state) {
            int state_block_idx;
            if (chunk == n_chunks - 1) {
                state_block_idx = block_idx_last_scheduled_token[batch_idx];
            } else {
                state_block_idx = (current_position + chunk_tokens - 1) / safe_block_size;
            }
            const int state_slot =
                cache_indices[batch_idx * cache_indices_stride + state_block_idx];
            if (state_slot != null_block_id) {
                const long sb = ((long)state_slot * dim + dim_idx) * dstate;
                state[sb + state_idx] = running;
            }
        }
        tokens_processed += chunk_tokens;
        current_position += chunk_tokens;
    }
}

#define instantiate_selective_scan(type_name, T)                                       \
  template [[host_name("selective_scan_dense_" #type_name)]] [[kernel]] void           \
  selective_scan_fwd_dense<T>(                                                          \
      const device T *u [[buffer(0)]], const device T *delta [[buffer(1)]],            \
      const device float *A [[buffer(2)]], const device T *B [[buffer(3)]],            \
      const device T *C [[buffer(4)]], const device float *D [[buffer(5)]],            \
      const device float *delta_bias [[buffer(6)]], const device T *z [[buffer(7)]],   \
      device T *out [[buffer(8)]], device float *state [[buffer(9)]],                  \
      constant int &batch [[buffer(10)]], constant int &dim [[buffer(11)]],            \
      constant int &seqlen [[buffer(12)]], constant int &dstate [[buffer(13)]],        \
      constant int &n_groups [[buffer(14)]], constant int &has_d [[buffer(15)]],       \
      constant int &has_delta_bias [[buffer(16)]], constant int &has_z [[buffer(17)]], \
      constant int &delta_softplus [[buffer(18)]],                                     \
      uint3 row [[threadgroup_position_in_grid]],                                      \
      uint3 thread_pos [[thread_position_in_threadgroup]]);                            \
  template [[host_name("selective_scan_varlen_" #type_name)]] [[kernel]] void          \
  selective_scan_fwd_varlen<T>(                                                         \
      const device T *u [[buffer(0)]], const device T *delta [[buffer(1)]],            \
      const device float *A [[buffer(2)]], const device T *B [[buffer(3)]],            \
      const device T *C [[buffer(4)]], const device float *D [[buffer(5)]],            \
      const device float *delta_bias [[buffer(6)]], const device T *z [[buffer(7)]],   \
      const device int *query_start_loc [[buffer(8)]],                                 \
      const device int *cache_indices [[buffer(9)]],                                   \
      const device uchar *has_initial_state [[buffer(10)]],                            \
      device T *out [[buffer(11)]], device float *state [[buffer(12)]],                \
      constant int &batch [[buffer(13)]], constant int &dim [[buffer(14)]],            \
      constant int &total_tokens [[buffer(15)]], constant int &dstate [[buffer(16)]],  \
      constant int &n_groups [[buffer(17)]], constant int &has_d [[buffer(18)]],       \
      constant int &has_delta_bias [[buffer(19)]], constant int &has_z [[buffer(20)]], \
      constant int &delta_softplus [[buffer(21)]],                                     \
      constant int &use_cache_indices [[buffer(22)]],                                  \
      constant int &use_has_initial_state [[buffer(23)]],                              \
      constant int &null_block_id [[buffer(24)]],                                      \
      uint3 row [[threadgroup_position_in_grid]],                                      \
      uint3 thread_pos [[thread_position_in_threadgroup]]);                            \
  template [[host_name("selective_scan_varlen_apc_" #type_name)]] [[kernel]] void      \
  selective_scan_fwd_varlen_apc<T>(                                                    \
      const device T *u [[buffer(0)]], const device T *delta [[buffer(1)]],            \
      const device float *A [[buffer(2)]], const device T *B [[buffer(3)]],            \
      const device T *C [[buffer(4)]], const device float *D [[buffer(5)]],            \
      const device float *delta_bias [[buffer(6)]], const device T *z [[buffer(7)]],   \
      const device int *query_start_loc [[buffer(8)]],                                 \
      const device int *cache_indices [[buffer(9)]],                                   \
      const device uchar *has_initial_state [[buffer(10)]],                            \
      device T *out [[buffer(11)]], device float *state [[buffer(12)]],                \
      const device int *block_idx_first_scheduled_token [[buffer(13)]],                \
      const device int *block_idx_last_scheduled_token [[buffer(14)]],                 \
      const device int *initial_state_idx [[buffer(15)]],                              \
      const device int *cu_chunk_seqlen [[buffer(16)]],                                \
      const device int *last_chunk_indices [[buffer(17)]],                             \
      constant int &batch [[buffer(18)]], constant int &dim [[buffer(19)]],            \
      constant int &total_tokens [[buffer(20)]], constant int &dstate [[buffer(21)]],  \
      constant int &n_groups [[buffer(22)]], constant int &has_d [[buffer(23)]],       \
      constant int &has_delta_bias [[buffer(24)]], constant int &has_z [[buffer(25)]], \
      constant int &delta_softplus [[buffer(26)]],                                     \
      constant int &null_block_id [[buffer(27)]],                                      \
      constant int &block_size [[buffer(28)]],                                         \
      constant int &cache_indices_stride [[buffer(29)]],                               \
      constant int &use_chunk_metadata [[buffer(30)]],                                 \
      uint3 row [[threadgroup_position_in_grid]],                                      \
      uint3 thread_pos [[thread_position_in_threadgroup]]);

instantiate_selective_scan(float32, float)
instantiate_selective_scan(float16, half)
instantiate_selective_scan(bfloat16, bf16)

// fp32 pool clone (grid-stride): the MLX (functional) path copies the incoming state pool
// into the output array, then the scan kernel updates only its request's slots in place —
// untouched slots keep their prior values, matching the persistent-pool semantics.
kernel void sscan_pool_clone(const device float *src [[buffer(0)]],
                             device float *dst       [[buffer(1)]],
                             constant uint &n        [[buffer(2)]],
                             uint tid [[thread_position_in_grid]],
                             uint nthreads [[threads_per_grid]]) {
    for (uint i = tid; i < n; i += nthreads) {
        dst[i] = src[i];
    }
}

}
