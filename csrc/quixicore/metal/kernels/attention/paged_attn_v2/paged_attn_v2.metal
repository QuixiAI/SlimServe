#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Long-context paged decode attention, two-kernel partition/reduce (vLLM v2).
//
// Each (head, batch) query is split across `num_partitions` KV-sequence slices
// of `partition_size` tokens, so no single threadgroup walks the whole context.
//
//   partition: local online-softmax over slice [p*PS, min((p+1)*PS, ctx_len))
//              -> max_logits[b,h,p] = m_p (local max), exp_sums[b,h,p] = S_p,
//                 tmp_out[b,h,p,:] = (sum_j e^{l_j-m_p} v_j) / S_p   (locally normalized)
//   reduce:    m* = max_p m_p ;  rescale_p = S_p * exp(m_p - m*)
//              out = sum_p tmp_out[p] * rescale_p / (sum_p rescale_p + 1e-6)
//
// That recovers the exact global softmax. GQA/MQA: kv_head = head/(H/H_KV).
// Caches are (num_blocks, block_size, num_kv_heads, D); q/out are (B,H,D); D∈{64,128}.
// Partials are fp32 for a numerically clean merge. One simdgroup (32 lanes) per block.
// ---------------------------------------------------------------------------

constant float NEG_INF = -3.4028234663852886e38f;

template <typename T, int D>
kernel void paged_attention_partition(
    device const T *q [[buffer(0)]],
    device const T *key_cache [[buffer(1)]],
    device const T *value_cache [[buffer(2)]],
    device const int *block_table [[buffer(3)]],
    device const int *context_lens [[buffer(4)]],
    device float *tmp_out [[buffer(5)]],      // (B, H, P, D)
    device float *max_logits [[buffer(6)]],   // (B, H, P)
    device float *exp_sums [[buffer(7)]],     // (B, H, P)
    constant int &block_size [[buffer(8)]],
    constant int &block_table_stride [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    constant int &num_heads [[buffer(11)]],
    constant int &num_kv_heads [[buffer(12)]],
    constant int &num_partitions [[buffer(13)]],
    constant int &partition_size [[buffer(14)]],
    constant int &window [[buffer(15)]],      // >0 = sliding window
    constant float &softcap [[buffer(16)]],   // >0 = Gemma-style logit soft-capping
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part = (int)tgid.z;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int start = part * partition_size;
    const int end = min(start + partition_size, context_len);
    // Sliding window: raise this partition's lower bound to the window start (end unchanged).
    const int t_start = (window > 0) ? max(start, context_len - window) : start;
    const bool capped = softcap > 0.0f;

    const long q_base = ((long)batch * num_heads + head) * D;
    const long stat_idx = ((long)batch * num_heads + head) * num_partitions + part;
    const long out_base = stat_idx * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[q_base + d]);
        acc[i] = 0.0f;
    }

    float m = NEG_INF, l = 0.0f;
    for (int t = t_start; t < end; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) {
            continue;
        }
        const long cache_base =
            (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            partial += qv[i] * float(key_cache[cache_base + d]);
        }
        float score = simd_sum(partial) * scale;
        if (capped) score = softcap * metal::tanh(score / softcap);   // natural domain here
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            acc[i] = acc[i] * alpha + beta * float(value_cache[cache_base + d]);
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (lane == 0) {
        max_logits[stat_idx] = l == 0.0f ? NEG_INF : m;
        exp_sums[stat_idx] = l;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        tmp_out[out_base + d] = l == 0.0f ? 0.0f : (acc[i] / l);
    }
}

// fp8 partition: identical online-softmax, but the caches hold uint8 (e4m3/e5m2) codes
// dequantized on read with per-head k_scale/v_scale. tmp_out/max_logits/exp_sums stay fp32
// so the existing (format-agnostic) reduce kernel merges the partitions unchanged.
template <typename T, int D, int FMT>
kernel void paged_attention_partition_fp8(
    device const T *q [[buffer(0)]],
    device const uchar *key_cache [[buffer(1)]],
    device const uchar *value_cache [[buffer(2)]],
    device const int *block_table [[buffer(3)]],
    device const int *context_lens [[buffer(4)]],
    device float *tmp_out [[buffer(5)]],
    device float *max_logits [[buffer(6)]],
    device float *exp_sums [[buffer(7)]],
    constant int &block_size [[buffer(8)]],
    constant int &block_table_stride [[buffer(9)]],
    constant float &scale [[buffer(10)]],
    constant int &num_heads [[buffer(11)]],
    constant int &num_kv_heads [[buffer(12)]],
    constant int &num_partitions [[buffer(13)]],
    constant int &partition_size [[buffer(14)]],
    device const float *k_scale [[buffer(15)]],
    device const float *v_scale [[buffer(16)]],
    constant int &fmt [[buffer(17)]],       // 0 = e4m3, 1 = e5m2
    constant int &window [[buffer(18)]],    // >0 = sliding window
    constant float &softcap [[buffer(19)]], // >0 = Gemma-style logit soft-capping
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part = (int)tgid.z;
    const int kv_head = head / (num_heads / num_kv_heads);
    const float ks = k_scale[kv_head], vs = v_scale[kv_head];
    const float score_scale = scale * ks;
    const int context_len = context_lens[batch];
    const int start = part * partition_size;
    const int end = min(start + partition_size, context_len);
    const int t_start = (window > 0) ? max(start, context_len - window) : start;
    const bool capped = softcap > 0.0f;

    const long q_base = ((long)batch * num_heads + head) * D;
    const long stat_idx = ((long)batch * num_heads + head) * num_partitions + part;
    const long out_base = stat_idx * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[q_base + d]);
        acc[i] = 0.0f;
    }

    float m = NEG_INF, l = 0.0f;
    for (int t = t_start; t < end; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) {
            continue;
        }
        const long cache_base =
            (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const uchar kc = key_cache[cache_base + d];
            const float kdec = FMT == 1 ? float(tk_e5m2_decode(kc)) : float(tk_e4m3_decode(kc));
            partial += qv[i] * kdec;
        }
        float score = simd_sum(partial) * score_scale;
        if (capped) score = softcap * metal::tanh(score / softcap);
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const uchar vc = value_cache[cache_base + d];
            const float vdec = FMT == 1 ? float(tk_e5m2_decode(vc)) : float(tk_e4m3_decode(vc));
            acc[i] = acc[i] * alpha + beta * vdec;
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (lane == 0) {
        max_logits[stat_idx] = l == 0.0f ? NEG_INF : m;
        exp_sums[stat_idx] = l;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        tmp_out[out_base + d] = l == 0.0f ? 0.0f : (acc[i] / l) * vs;
    }
}

// The attention sink lands HERE, not in the partition kernels: it must enter the softmax
// denominator exactly once per (batch, head), and only the reduce sees all partitions.
// The sink's value row contributes nothing to the output accumulation.
template <typename T, int D>
kernel void paged_attention_reduce(
    device const float *tmp_out [[buffer(0)]],    // (B, H, P, D)
    device const float *max_logits [[buffer(1)]], // (B, H, P)
    device const float *exp_sums [[buffer(2)]],   // (B, H, P)
    device T *out [[buffer(3)]],                   // (B, H, D)
    constant int &num_heads [[buffer(4)]],
    constant int &num_partitions [[buffer(5)]],
    device const float *sinks [[buffer(6)]],       // per-head; read only when has_sink
    constant int &has_sink [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const long base = ((long)batch * num_heads + head) * num_partitions;

    float gm = NEG_INF;
    for (int p = 0; p < num_partitions; ++p) {
        gm = max(gm, max_logits[base + p]);
    }
    const float sink = (has_sink != 0) ? sinks[head] : NEG_INF;
    if (has_sink != 0) gm = max(gm, sink);
    float gden = 0.0f;
    for (int p = 0; p < num_partitions; ++p) {
        const float mp = max_logits[base + p];
        if (mp == NEG_INF) {
            continue;
        }
        gden += exp_sums[base + p] * exp(mp - gm);
    }
    if (has_sink != 0) gden += exp(sink - gm);     // exactly once, across all partitions
    const float inv = 1.0f / (gden + 1e-6f);

    float acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        acc[i] = 0.0f;
    }
    for (int p = 0; p < num_partitions; ++p) {
        const float mp = max_logits[base + p];
        if (mp == NEG_INF) {
            continue;
        }
        const float r = exp_sums[base + p] * exp(mp - gm);
        const long ob = (base + p) * D;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            acc[i] += tmp_out[ob + d] * r;
        }
    }
    const long out_base = ((long)batch * num_heads + head) * D;
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[out_base + d] = gm == NEG_INF ? T(0) : T(acc[i] * inv);
    }
}

// ---------------------------------------------------------------------------
// Cascade / shared-prefix attention: the "prefix" partition. All (batch, head) queries attend a
// single SHARED, CONTIGUOUS prefix KV (prefix_k/prefix_v: (prefix_len, H_KV, D)) — the common
// system prompt for a batch of requests. It emits the same locally-normalized (m, l, o) partials
// as paged_attention_partition, in the identical (B,H,P,D)/(B,H,P) layout, so the two levels'
// partials can be concatenated along P and folded by the SAME paged_attention_reduce — the
// mergeable-attention-state (log-sum-exp) merge, with no math change. Host composition:
//   prefix partials (this kernel) ++ suffix partials (paged_attention_partition, per-request paged
//   KV) -> concat along P -> paged_attention_reduce.  Ref: flashinfer cascade.py merge_states.
// ---------------------------------------------------------------------------
template <typename T, int D>
kernel void cascade_prefix_partition(
    device const T *q [[buffer(0)]],           // (B, H, D)
    device const T *prefix_k [[buffer(1)]],    // (prefix_len, H_KV, D)  shared across the batch
    device const T *prefix_v [[buffer(2)]],
    device float *tmp_out [[buffer(3)]],       // (B, H, Pp, D)
    device float *max_logits [[buffer(4)]],    // (B, H, Pp)
    device float *exp_sums [[buffer(5)]],      // (B, H, Pp)
    constant int &prefix_len [[buffer(6)]],
    constant float &scale [[buffer(7)]],
    constant int &num_heads [[buffer(8)]],
    constant int &num_kv_heads [[buffer(9)]],
    constant int &num_partitions [[buffer(10)]],
    constant int &partition_size [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part = (int)tgid.z;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int start = part * partition_size;
    const int end = min(start + partition_size, prefix_len);

    const long q_base = ((long)batch * num_heads + head) * D;
    const long stat_idx = ((long)batch * num_heads + head) * num_partitions + part;
    const long out_base = stat_idx * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[q_base + d]);
        acc[i] = 0.0f;
    }

    float m = NEG_INF, l = 0.0f;
    for (int t = start; t < end; ++t) {
        const long cache_base = ((long)t * num_kv_heads + kv_head) * D;   // contiguous, no paging
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            partial += qv[i] * float(prefix_k[cache_base + d]);
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            acc[i] = acc[i] * alpha + beta * float(prefix_v[cache_base + d]);
        }
        l = l * alpha + beta;
        m = new_m;
    }

    if (lane == 0) {
        max_logits[stat_idx] = l == 0.0f ? NEG_INF : m;
        exp_sums[stat_idx] = l;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        tmp_out[out_base + d] = l == 0.0f ? 0.0f : (acc[i] / l);
    }
}

// fp8 cascade prefix partition: same as cascade_prefix_partition, but the SHARED contiguous prefix
// KV is uint8 fp8 (e4m3/e5m2) dequantized on read with per-kv-head scales (mirrors
// paged_attention_partition_fp8). Emits the same (tmp_out, max_logits, exp_sums) partials so it
// concatenates with the (bf16 or fp8) suffix partials into the shared reduce.
template <typename T, int D, int FMT>
kernel void cascade_prefix_partition_fp8(
    device const T     *q          [[buffer(0)]],   // (B, H, D)
    device const uchar *prefix_k   [[buffer(1)]],   // (prefix_len, H_KV, D) uint8 fp8
    device const uchar *prefix_v   [[buffer(2)]],
    device float       *tmp_out    [[buffer(3)]],
    device float       *max_logits [[buffer(4)]],
    device float       *exp_sums   [[buffer(5)]],
    constant int   &prefix_len     [[buffer(6)]],
    constant float &scale          [[buffer(7)]],
    constant int   &num_heads      [[buffer(8)]],
    constant int   &num_kv_heads   [[buffer(9)]],
    constant int   &num_partitions [[buffer(10)]],
    constant int   &partition_size [[buffer(11)]],
    device const float *k_scale    [[buffer(12)]],
    device const float *v_scale    [[buffer(13)]],
    constant int   &fmt            [[buffer(14)]],   // 0 = e4m3, 1 = e5m2
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;
    const int head = (int)tgid.x, batch = (int)tgid.y, part = (int)tgid.z;
    const int kv_head = head / (num_heads / num_kv_heads);
    const float ks = k_scale[kv_head], vs = v_scale[kv_head];
    const float score_scale = scale * ks;
    const int start = part * partition_size;
    const int end = min(start + partition_size, prefix_len);
    const long q_base = ((long)batch * num_heads + head) * D;
    const long stat_idx = ((long)batch * num_heads + head) * num_partitions + part;
    const long out_base = stat_idx * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[q_base + d]);
        acc[i] = 0.0f;
    }
    float m = NEG_INF, l = 0.0f;
    for (int t = start; t < end; ++t) {
        const long cache_base = ((long)t * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const uchar kc = prefix_k[cache_base + d];
            const float kdec = FMT == 1 ? float(tk_e5m2_decode(kc)) : float(tk_e4m3_decode(kc));
            partial += qv[i] * kdec;
        }
        const float score = simd_sum(partial) * score_scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const uchar vc = prefix_v[cache_base + d];
            const float vdec = FMT == 1 ? float(tk_e5m2_decode(vc)) : float(tk_e4m3_decode(vc));
            acc[i] = acc[i] * alpha + beta * vdec;
        }
        l = l * alpha + beta;
        m = new_m;
    }
    if (lane == 0) {
        max_logits[stat_idx] = l == 0.0f ? NEG_INF : m;
        exp_sums[stat_idx] = l;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        tmp_out[out_base + d] = l == 0.0f ? 0.0f : (acc[i] / l) * vs;
    }
}

#define instantiate_paged_v2(type_name, T, DVAL)                              \
  template [[host_name("paged_attention_partition_" #type_name "_" #DVAL)]]   \
  [[kernel]] void paged_attention_partition<T, DVAL>(                         \
      device const T *q [[buffer(0)]],                                        \
      device const T *key_cache [[buffer(1)]],                                \
      device const T *value_cache [[buffer(2)]],                              \
      device const int *block_table [[buffer(3)]],                            \
      device const int *context_lens [[buffer(4)]],                           \
      device float *tmp_out [[buffer(5)]],                                    \
      device float *max_logits [[buffer(6)]],                                 \
      device float *exp_sums [[buffer(7)]],                                   \
      constant int &block_size [[buffer(8)]],                                 \
      constant int &block_table_stride [[buffer(9)]],                         \
      constant float &scale [[buffer(10)]],                                   \
      constant int &num_heads [[buffer(11)]],                                 \
      constant int &num_kv_heads [[buffer(12)]],                              \
      constant int &num_partitions [[buffer(13)]],                            \
      constant int &partition_size [[buffer(14)]],                            \
      constant int &window [[buffer(15)]],                                    \
      constant float &softcap [[buffer(16)]],                                 \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);                               \
  template [[host_name("paged_attention_reduce_" #type_name "_" #DVAL)]]      \
  [[kernel]] void paged_attention_reduce<T, DVAL>(                            \
      device const float *tmp_out [[buffer(0)]],                              \
      device const float *max_logits [[buffer(1)]],                           \
      device const float *exp_sums [[buffer(2)]],                             \
      device T *out [[buffer(3)]],                                            \
      constant int &num_heads [[buffer(4)]],                                  \
      constant int &num_partitions [[buffer(5)]],                             \
      device const float *sinks [[buffer(6)]],                                \
      constant int &has_sink [[buffer(7)]],                                   \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);                               \
  template [[host_name("cascade_prefix_partition_" #type_name "_" #DVAL)]]    \
  [[kernel]] void cascade_prefix_partition<T, DVAL>(                          \
      device const T *q [[buffer(0)]],                                        \
      device const T *prefix_k [[buffer(1)]],                                 \
      device const T *prefix_v [[buffer(2)]],                                 \
      device float *tmp_out [[buffer(3)]],                                    \
      device float *max_logits [[buffer(4)]],                                 \
      device float *exp_sums [[buffer(5)]],                                   \
      constant int &prefix_len [[buffer(6)]],                                 \
      constant float &scale [[buffer(7)]],                                    \
      constant int &num_heads [[buffer(8)]],                                  \
      constant int &num_kv_heads [[buffer(9)]],                               \
      constant int &num_partitions [[buffer(10)]],                            \
      constant int &partition_size [[buffer(11)]],                            \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_paged_v2_fp8(fmt_name, FMT, type_name, T, DVAL)           \
  template [[host_name("paged_attention_partition_fp8_" #fmt_name "_" #type_name "_" #DVAL)]] \
  [[kernel]] void paged_attention_partition_fp8<T, DVAL, FMT>(                 \
      device const T *q [[buffer(0)]],                                        \
      device const uchar *key_cache [[buffer(1)]],                            \
      device const uchar *value_cache [[buffer(2)]],                          \
      device const int *block_table [[buffer(3)]],                            \
      device const int *context_lens [[buffer(4)]],                           \
      device float *tmp_out [[buffer(5)]],                                    \
      device float *max_logits [[buffer(6)]],                                 \
      device float *exp_sums [[buffer(7)]],                                   \
      constant int &block_size [[buffer(8)]],                                 \
      constant int &block_table_stride [[buffer(9)]],                         \
      constant float &scale [[buffer(10)]],                                   \
      constant int &num_heads [[buffer(11)]],                                 \
      constant int &num_kv_heads [[buffer(12)]],                              \
      constant int &num_partitions [[buffer(13)]],                            \
      constant int &partition_size [[buffer(14)]],                            \
      device const float *k_scale [[buffer(15)]],                             \
      device const float *v_scale [[buffer(16)]],                             \
      constant int &fmt [[buffer(17)]],                                       \
      constant int &window [[buffer(18)]],                                    \
      constant float &softcap [[buffer(19)]],                                 \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);                               \
  template [[host_name("cascade_prefix_partition_fp8_" #fmt_name "_" #type_name "_" #DVAL)]] \
  [[kernel]] void cascade_prefix_partition_fp8<T, DVAL, FMT>(                  \
      device const T *q [[buffer(0)]],                                        \
      device const uchar *prefix_k [[buffer(1)]],                             \
      device const uchar *prefix_v [[buffer(2)]],                             \
      device float *tmp_out [[buffer(3)]],                                    \
      device float *max_logits [[buffer(4)]],                                 \
      device float *exp_sums [[buffer(5)]],                                   \
      constant int &prefix_len [[buffer(6)]],                                 \
      constant float &scale [[buffer(7)]],                                    \
      constant int &num_heads [[buffer(8)]],                                  \
      constant int &num_kv_heads [[buffer(9)]],                               \
      constant int &num_partitions [[buffer(10)]],                            \
      constant int &partition_size [[buffer(11)]],                            \
      device const float *k_scale [[buffer(12)]],                             \
      device const float *v_scale [[buffer(13)]],                             \
      constant int &fmt [[buffer(14)]],                                       \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_paged_v2(float32, float, 64)
instantiate_paged_v2(float32, float, 128)
instantiate_paged_v2(float16, half, 64)
instantiate_paged_v2(float16, half, 128)
instantiate_paged_v2(bfloat16, bf16, 64)
instantiate_paged_v2(bfloat16, bf16, 128)

// reduce-only instantiation at D=512: consumed by mla_decode_partition (MLA latent decode
// emits paged-v2-style partials over the 512-wide latent).
template [[host_name("paged_attention_reduce_bfloat16_512")]]
[[kernel]] void paged_attention_reduce<bf16, 512>(
    device const float *tmp_out [[buffer(0)]],
    device const float *max_logits [[buffer(1)]],
    device const float *exp_sums [[buffer(2)]],
    device bf16 *out [[buffer(3)]],
    constant int &num_heads [[buffer(4)]],
    constant int &num_partitions [[buffer(5)]],
    device const float *sinks [[buffer(6)]],
    constant int &has_sink [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]);
instantiate_paged_v2_fp8(e4m3, 0, float32, float, 64)
instantiate_paged_v2_fp8(e4m3, 0, float32, float, 128)
instantiate_paged_v2_fp8(e4m3, 0, float16, half, 64)
instantiate_paged_v2_fp8(e4m3, 0, float16, half, 128)
instantiate_paged_v2_fp8(e4m3, 0, bfloat16, bf16, 64)
instantiate_paged_v2_fp8(e4m3, 0, bfloat16, bf16, 128)
instantiate_paged_v2_fp8(e5m2, 1, float32, float, 64)
instantiate_paged_v2_fp8(e5m2, 1, float32, float, 128)
instantiate_paged_v2_fp8(e5m2, 1, float16, half, 64)
instantiate_paged_v2_fp8(e5m2, 1, float16, half, 128)
instantiate_paged_v2_fp8(e5m2, 1, bfloat16, bf16, 64)
instantiate_paged_v2_fp8(e5m2, 1, bfloat16, bf16, 128)
