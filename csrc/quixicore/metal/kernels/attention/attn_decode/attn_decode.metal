#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Batch-1 GQA decode: one new-token query against a dense floating-point KV
// cache, using online softmax.
//
//   q (Hq, D) · kc/vc (Tk, Hkv, D) -> out (Hq, D);  head h reads kv head h/rep.
//
// One simdgroup per query head; lanes stride D (D <= 128 -> <= 4 regs/lane);
// fp32 accumulators throughout. grid (Hq, 1, 1), 32 threads.
// ---------------------------------------------------------------------------

constant float ATTN_NEG_INF = -3.4028234663852886e38f;

template <typename T>
kernel void attn_decode(device const T *q   [[buffer(0)]],
                        device const T *kc  [[buffer(1)]],
                        device const T *vc  [[buffer(2)]],
                        device T       *out [[buffer(3)]],
                        constant int &Tk  [[buffer(4)]],
                        constant int &Hq  [[buffer(5)]],
                        constant int &Hkv [[buffer(6)]],
                        constant int &D   [[buffer(7)]],
                        uint head [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const int rep = Hq / Hkv;
    const int hk = (int)head / rep;
    const float scale = metal::rsqrt(float(D));
    const int per = (D + 31) / 32;                   // regs per lane (<= 4 at D <= 128)

    float qreg[4], oacc[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int d = j * 32 + (int)lane;
        qreg[j] = (j < per && d < D) ? float(q[(long)head * D + d]) : 0.0f;
        oacc[j] = 0.0f;
    }

    float m = ATTN_NEG_INF, l = 0.0f;
    for (int t = 0; t < Tk; ++t) {
        const long kvbase = ((long)t * Hkv + hk) * D;
        float dot = 0.0f;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + (int)lane;
            if (d < D) { dot += qreg[j] * float(kc[kvbase + d]); }
        }
        dot = metal::simd_sum(dot) * scale;          // identical on every lane
        const float nm   = max(m, dot);
        const float corr = exp(m - nm);
        const float p    = exp(dot - nm);
        l = l * corr + p;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + (int)lane;
            if (d < D) { oacc[j] = oacc[j] * corr + p * float(vc[kvbase + d]); }
        }
        m = nm;
    }
    const float invl = 1.0f / l;
    for (int j = 0; j < per; ++j) {
        const int d = j * 32 + (int)lane;
        if (d < D) { out[(long)head * D + d] = T(oacc[j] * invl); }
    }
}

#define instantiate_attn_decode(type_name, T)                                   \
  template [[host_name("attn_decode_" #type_name)]] [[kernel]] void             \
  attn_decode<T>(device const T *q [[buffer(0)]],                               \
                 device const T *kc [[buffer(1)]],                              \
                 device const T *vc [[buffer(2)]],                              \
                 device T *out [[buffer(3)]],                                   \
                 constant int &Tk [[buffer(4)]],                                \
                 constant int &Hq [[buffer(5)]],                                \
                 constant int &Hkv [[buffer(6)]],                               \
                 constant int &D [[buffer(7)]],                                 \
                 uint head [[threadgroup_position_in_grid]],                    \
                 uint lane [[thread_index_in_simdgroup]]);

instantiate_attn_decode(float32, float)
instantiate_attn_decode(float16, half)
instantiate_attn_decode(bfloat16, bf16)

// Head-major batched decode. q is (B,Hq,D), while K/V use the preallocated
// decoder-cache layout (B,Hkv,cache_T,D), avoiding a history
// transpose or copy before every decode dispatch.
template <typename T>
kernel void attn_decode_bh(device const T *q [[buffer(0)]],
                           device const T *kc [[buffer(1)]],
                           device const T *vc [[buffer(2)]],
                           device T *out [[buffer(3)]],
                           constant int &Tk [[buffer(4)]],
                           constant int &Hq [[buffer(5)]],
                           constant int &Hkv [[buffer(6)]],
                           constant int &D [[buffer(7)]],
                           constant int &cache_T [[buffer(8)]],
                           uint2 pos [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const int head = int(pos.x);
    const int batch = int(pos.y);
    const int rep = Hq / Hkv;
    const int hk = head / rep;
    const float scale = metal::rsqrt(float(D));
    const int per = (D + 31) / 32;
    const long qbase = ((long)batch * Hq + head) * D;

    float qreg[4], oacc[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int d = j * 32 + int(lane);
        qreg[j] = (j < per && d < D) ? float(q[qbase + d]) : 0.0f;
        oacc[j] = 0.0f;
    }

    float maximum = ATTN_NEG_INF, denominator = 0.0f;
    for (int token = 0; token < Tk; ++token) {
        const long kvbase = (((long)batch * Hkv + hk) * cache_T + token) * D;
        float score = 0.0f;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) score += qreg[j] * float(kc[kvbase + d]);
        }
        score = metal::simd_sum(score) * scale;
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) {
                oacc[j] = oacc[j] * correction + probability * float(vc[kvbase + d]);
            }
        }
        maximum = next_maximum;
    }

    const float inverse = 1.0f / denominator;
    for (int j = 0; j < per; ++j) {
        const int d = j * 32 + int(lane);
        if (d < D) out[qbase + d] = T(oacc[j] * inverse);
    }
}

#define instantiate_attn_decode_bh_single(type_name, T)                       \
  template [[host_name("attn_decode_bh_" #type_name)]] [[kernel]] void       \
  attn_decode_bh<T>(device const T *q [[buffer(0)]],                          \
                    device const T *kc [[buffer(1)]],                         \
                    device const T *vc [[buffer(2)]],                         \
                    device T *out [[buffer(3)]],                              \
                    constant int &Tk [[buffer(4)]],                           \
                    constant int &Hq [[buffer(5)]],                           \
                    constant int &Hkv [[buffer(6)]],                          \
                    constant int &D [[buffer(7)]],                            \
                    constant int &cache_T [[buffer(8)]],                      \
                    uint2 pos [[threadgroup_position_in_grid]],               \
                    uint lane [[thread_index_in_simdgroup]]);

instantiate_attn_decode_bh_single(float32, float)
instantiate_attn_decode_bh_single(float16, half)
instantiate_attn_decode_bh_single(bfloat16, bf16)

// Long contexts are split across a compile-time number of simdgroups. Their
// online-softmax states are merged once in threadgroup memory.
template <typename T, int PARTITIONS>
kernel void attn_decode_bh_partitioned(device const T *q [[buffer(0)]],
                           device const T *kc [[buffer(1)]],
                           device const T *vc [[buffer(2)]],
                           device T *out [[buffer(3)]],
                           constant int &Tk [[buffer(4)]],
                           constant int &Hq [[buffer(5)]],
                           constant int &Hkv [[buffer(6)]],
                           constant int &D [[buffer(7)]],
                           constant int &cache_T [[buffer(8)]],
                           uint3 pos [[threadgroup_position_in_grid]],
                           uint simd_index [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float warp_maximum[PARTITIONS];
    threadgroup float warp_denominator[PARTITIONS];
    threadgroup float warp_output[PARTITIONS * 128];

    const int head = int(pos.x);
    const int batch = int(pos.y);
    const int rep = Hq / Hkv;
    const int hk = head / rep;
    const float scale = metal::rsqrt(float(D));
    const int per = (D + 31) / 32;
    const long qbase = ((long)batch * Hq + head) * D;

    float qreg[4], oacc[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        const int d = j * 32 + int(lane);
        qreg[j] = (j < per && d < D) ? float(q[qbase + d]) : 0.0f;
        oacc[j] = 0.0f;
    }

    float maximum = ATTN_NEG_INF, denominator = 0.0f;
    for (int token = int(simd_index); token < Tk; token += PARTITIONS) {
        const long kvbase = (((long)batch * Hkv + hk) * cache_T + token) * D;
        float score = 0.0f;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) score += qreg[j] * float(kc[kvbase + d]);
        }
        score = metal::simd_sum(score) * scale;
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) {
                oacc[j] = oacc[j] * correction + probability * float(vc[kvbase + d]);
            }
        }
        maximum = next_maximum;
    }

    if (lane == 0) {
        warp_maximum[simd_index] = maximum;
        warp_denominator[simd_index] = denominator;
    }
    for (int j = 0; j < per; ++j) {
        const int d = j * 32 + int(lane);
        if (d < D) warp_output[simd_index * 128 + d] = oacc[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0) {
        float global_maximum = ATTN_NEG_INF;
        for (int warp = 0; warp < PARTITIONS; ++warp) {
            if (warp_denominator[warp] > 0.0f) {
                global_maximum = max(global_maximum, warp_maximum[warp]);
            }
        }
        float global_denominator = 0.0f;
        for (int warp = 0; warp < PARTITIONS; ++warp) {
            if (warp_denominator[warp] > 0.0f) {
                global_denominator += warp_denominator[warp] *
                    exp(warp_maximum[warp] - global_maximum);
            }
        }
        for (int j = 0; j < per; ++j) {
            const int d = j * 32 + int(lane);
            if (d < D) {
                float numerator = 0.0f;
                for (int warp = 0; warp < PARTITIONS; ++warp) {
                    if (warp_denominator[warp] > 0.0f) {
                        numerator += warp_output[warp * 128 + d] *
                            exp(warp_maximum[warp] - global_maximum);
                    }
                }
                out[qbase + d] = T(numerator / global_denominator);
            }
        }
    }
}

#define instantiate_attn_decode_bh_partitioned(type_name, T, partitions)       \
  template [[host_name("attn_decode_bh_partitioned_" #partitions "_"        \
                       #type_name)]] [[kernel]] void                          \
  attn_decode_bh_partitioned<T, partitions>(                                  \
                    device const T *q [[buffer(0)]],                          \
                    device const T *kc [[buffer(1)]],                         \
                    device const T *vc [[buffer(2)]],                         \
                    device T *out [[buffer(3)]],                              \
                    constant int &Tk [[buffer(4)]],                           \
                    constant int &Hq [[buffer(5)]],                           \
                    constant int &Hkv [[buffer(6)]],                          \
                    constant int &D [[buffer(7)]],                            \
                    constant int &cache_T [[buffer(8)]],                      \
                    uint3 pos [[threadgroup_position_in_grid]],               \
                    uint simd_index [[simdgroup_index_in_threadgroup]],       \
                    uint lane [[thread_index_in_simdgroup]]);

#define instantiate_attn_decode_bh_partitions(type_name, T)                   \
  instantiate_attn_decode_bh_partitioned(type_name, T, 8)                    \
  instantiate_attn_decode_bh_partitioned(type_name, T, 32)

instantiate_attn_decode_bh_partitions(float32, float)
instantiate_attn_decode_bh_partitions(float16, half)
instantiate_attn_decode_bh_partitions(bfloat16, bf16)

// Functional decode step over a head-major cache.  The host clones the input
// cache first; this dispatch optionally RMS-normalizes Q/K, applies split-half
// RoPE, appends the rotated K and raw V, and attends through the appended token.
// SIMD groups partition each context and merge online-softmax state.
template <typename T>
kernel void decode_cache_attention(
    device const T *q [[buffer(0)]],
    device const T *new_k [[buffer(1)]],
    device const T *new_v [[buffer(2)]],
    device const T *cos_table [[buffer(3)]],
    device const T *sin_table [[buffer(4)]],
    device const int *positions [[buffer(5)]],
    device const int *context_lengths [[buffer(6)]],
    device const T *q_weight [[buffer(7)]],
    device const T *k_weight [[buffer(8)]],
    device T *key_cache [[buffer(9)]],
    device T *value_cache [[buffer(10)]],
    device T *output [[buffer(11)]],
    constant int &batch_size [[buffer(12)]],
    constant int &heads_q [[buffer(13)]],
    constant int &heads_kv [[buffer(14)]],
    constant int &cache_length [[buffer(15)]],
    constant int &dimension [[buffer(16)]],
    constant float &epsilon [[buffer(17)]],
    constant int &do_q_norm [[buffer(18)]],
    constant int &do_k_norm [[buffer(19)]],
    constant int &gemma [[buffer(20)]],
    constant float &softmax_scale [[buffer(21)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint simd_index [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 threads [[threads_per_threadgroup]]) {
    threadgroup float warp_maximum[32];
    threadgroup float warp_denominator[32];
    threadgroup float warp_output[32 * 128];

    const int head = int(group.x);
    const int batch = int(group.y);
    if (batch >= batch_size || head >= heads_q) return;
    const int repetitions = heads_q / heads_kv;
    const int kv_head = head / repetitions;
    const int context = context_lengths[batch];
    const int position = positions[batch];
    const int half_dimension = dimension / 2;
    const int values_per_lane = dimension / 32;
    const int active_simdgroups = int(threads.x / 32);
    const long q_base = ((long)batch * heads_q + head) * dimension;
    const long new_k_base = ((long)batch * heads_kv + kv_head) * dimension;
    const long rope_base = (long)position * half_dimension;

    float q_sumsq = 0.0f, k_sumsq = 0.0f;
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = j * 32 + int(lane);
        const float qv = float(q[q_base + d]);
        const float kv = float(new_k[new_k_base + d]);
        q_sumsq += qv * qv;
        k_sumsq += kv * kv;
    }
    q_sumsq = metal::simd_sum(q_sumsq);
    k_sumsq = metal::simd_sum(k_sumsq);
    const float q_inverse = do_q_norm != 0
        ? metal::rsqrt(q_sumsq / float(dimension) + epsilon) : 1.0f;
    const float k_inverse = do_k_norm != 0
        ? metal::rsqrt(k_sumsq / float(dimension) + epsilon) : 1.0f;

    float q_rotated[4], k_rotated[4], v_values[4], output_accumulator[4];
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        q_rotated[j] = 0.0f;
        k_rotated[j] = 0.0f;
        v_values[j] = 0.0f;
        output_accumulator[j] = 0.0f;
    }
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = j * 32 + int(lane);
        const int pair_d = d < half_dimension ? d + half_dimension : d - half_dimension;
        const int rope_d = d < half_dimension ? d : d - half_dimension;
        float qv = float(q[q_base + d]);
        float qp = float(q[q_base + pair_d]);
        float kv = float(new_k[new_k_base + d]);
        float kp = float(new_k[new_k_base + pair_d]);
        if (do_q_norm != 0) {
            const float qw = float(q_weight[d]) + (gemma != 0 ? 1.0f : 0.0f);
            const float qpw = float(q_weight[pair_d]) + (gemma != 0 ? 1.0f : 0.0f);
            qv *= q_inverse * qw;
            qp *= q_inverse * qpw;
        }
        if (do_k_norm != 0) {
            const float kw = float(k_weight[d]) + (gemma != 0 ? 1.0f : 0.0f);
            const float kpw = float(k_weight[pair_d]) + (gemma != 0 ? 1.0f : 0.0f);
            kv *= k_inverse * kw;
            kp *= k_inverse * kpw;
        }
        const float cosine = float(cos_table[rope_base + rope_d]);
        const float sine = float(sin_table[rope_base + rope_d]);
        q_rotated[j] = d < half_dimension
            ? qv * cosine - qp * sine : qv * cosine + qp * sine;
        k_rotated[j] = d < half_dimension
            ? kv * cosine - kp * sine : kv * cosine + kp * sine;
        v_values[j] = float(new_v[new_k_base + d]);
    }

    // Exactly one query group writes each (batch, kv_head) append row. Other
    // query heads consume their register copy of the same new K/V, avoiding a
    // cross-threadgroup ordering dependency.
    if (simd_index == 0 && head % repetitions == 0 &&
        context >= 0 && context < cache_length) {
        const long cache_base =
            (((long)batch * heads_kv + kv_head) * cache_length + context) * dimension;
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = j * 32 + int(lane);
            key_cache[cache_base + d] = T(k_rotated[j]);
            value_cache[cache_base + d] = T(v_values[j]);
        }
    }

    const float scale = softmax_scale > 0.0f
        ? softmax_scale : metal::rsqrt(float(dimension));
    float maximum = ATTN_NEG_INF;
    float denominator = 0.0f;
    for (int token = int(simd_index); token <= context;
         token += active_simdgroups) {
        float score = 0.0f;
        const bool appended = token == context;
        const long cache_base =
            (((long)batch * heads_kv + kv_head) * cache_length + token) * dimension;
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = j * 32 + int(lane);
            const float key_value = appended
                ? k_rotated[j] : float(key_cache[cache_base + d]);
            score += q_rotated[j] * key_value;
        }
        score = metal::simd_sum(score) * scale;
        const float next_maximum = max(maximum, score);
        const float correction = exp(maximum - next_maximum);
        const float probability = exp(score - next_maximum);
        denominator = denominator * correction + probability;
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = j * 32 + int(lane);
            const float value = appended
                ? v_values[j] : float(value_cache[cache_base + d]);
            output_accumulator[j] =
                output_accumulator[j] * correction + probability * value;
        }
        maximum = next_maximum;
    }
    if (lane == 0) {
        warp_maximum[simd_index] = maximum;
        warp_denominator[simd_index] = denominator;
    }
    for (int j = 0; j < values_per_lane; ++j) {
        const int d = j * 32 + int(lane);
        warp_output[simd_index * 128 + d] = output_accumulator[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_index == 0) {
        float global_maximum = ATTN_NEG_INF;
        for (int warp = 0; warp < active_simdgroups; ++warp) {
            if (warp_denominator[warp] > 0.0f) {
                global_maximum = max(global_maximum, warp_maximum[warp]);
            }
        }
        float global_denominator = 0.0f;
        for (int warp = 0; warp < active_simdgroups; ++warp) {
            if (warp_denominator[warp] > 0.0f) {
                global_denominator += warp_denominator[warp] *
                    exp(warp_maximum[warp] - global_maximum);
            }
        }
        const long output_base = ((long)batch * heads_q + head) * dimension;
        for (int j = 0; j < values_per_lane; ++j) {
            const int d = j * 32 + int(lane);
            float numerator = 0.0f;
            for (int warp = 0; warp < active_simdgroups; ++warp) {
                if (warp_denominator[warp] > 0.0f) {
                    numerator += warp_output[warp * 128 + d] *
                        exp(warp_maximum[warp] - global_maximum);
                }
            }
            output[output_base + d] = T(numerator / global_denominator);
        }
    }
}

#define instantiate_decode_cache_attention(type_name, T)                     \
  template [[host_name("decode_cache_attention_" #type_name)]] [[kernel]]   \
  void decode_cache_attention<T>(                                            \
    device const T *q [[buffer(0)]], device const T *new_k [[buffer(1)]],    \
    device const T *new_v [[buffer(2)]], device const T *cos_table [[buffer(3)]], \
    device const T *sin_table [[buffer(4)]], device const int *positions [[buffer(5)]], \
    device const int *context_lengths [[buffer(6)]],                         \
    device const T *q_weight [[buffer(7)]], device const T *k_weight [[buffer(8)]], \
    device T *key_cache [[buffer(9)]], device T *value_cache [[buffer(10)]], \
    device T *output [[buffer(11)]], constant int &batch_size [[buffer(12)]], \
    constant int &heads_q [[buffer(13)]], constant int &heads_kv [[buffer(14)]], \
    constant int &cache_length [[buffer(15)]], constant int &dimension [[buffer(16)]], \
    constant float &epsilon [[buffer(17)]], constant int &do_q_norm [[buffer(18)]], \
    constant int &do_k_norm [[buffer(19)]], constant int &gemma [[buffer(20)]], \
    constant float &softmax_scale [[buffer(21)]],                            \
    uint3 group [[threadgroup_position_in_grid]],                            \
    uint simd_index [[simdgroup_index_in_threadgroup]],                      \
    uint lane [[thread_index_in_simdgroup]],                                 \
    uint3 threads [[threads_per_threadgroup]]);

instantiate_decode_cache_attention(float32, float)
instantiate_decode_cache_attention(float16, half)
instantiate_decode_cache_attention(bfloat16, bf16)
