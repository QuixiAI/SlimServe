#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

template <typename T, int D>
kernel void cross_attention(
    device const T *q [[buffer(0)]], device const T *k [[buffer(1)]],
    device const T *v [[buffer(2)]], device const int *key_lengths [[buffer(3)]],
    device const float *bias [[buffer(4)]], device T *out [[buffer(5)]],
    constant int &query_length [[buffer(6)]], constant int &key_length [[buffer(7)]],
    constant int &query_heads [[buffer(8)]], constant int &kv_heads [[buffer(9)]],
    constant float &scale [[buffer(10)]], constant float &softcap [[buffer(11)]],
    constant int &has_bias [[buffer(12)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int VALUES = D / 32;
  const int tq = int(group.x), hq = int(group.y), batch = int(group.z);
  const int hk = hq / (query_heads / kv_heads);
  const int valid_keys = metal::clamp(key_lengths[batch], 0, key_length);
  const long q_base = (((long)batch * query_heads + hq) * query_length + tq) * D;
  float row_max = -INFINITY;
  for (int tk = 0; tk < valid_keys; ++tk) {
    const long k_base = (((long)batch * kv_heads + hk) * key_length + tk) * D;
    float score = 0.0f;
    for (int d = int(lane); d < D; d += 32) score += float(q[q_base + d]) * float(k[k_base + d]);
    score = metal::simd_sum(score) * scale;
    if (has_bias != 0) {
      score += bias[(((long)batch * query_heads + hq) * query_length + tq) * key_length + tk];
    }
    if (softcap > 0.0f) score = softcap * metal::tanh(score / softcap);
    row_max = metal::max(row_max, score);
  }
  float acc[VALUES];
  #pragma clang loop unroll(full)
  for (int i = 0; i < VALUES; ++i) acc[i] = 0.0f;
  float denominator = 0.0f;
  for (int tk = 0; tk < valid_keys; ++tk) {
    const long kv_base = (((long)batch * kv_heads + hk) * key_length + tk) * D;
    float score = 0.0f;
    for (int d = int(lane); d < D; d += 32) score += float(q[q_base + d]) * float(k[kv_base + d]);
    score = metal::simd_sum(score) * scale;
    if (has_bias != 0) {
      score += bias[(((long)batch * query_heads + hq) * query_length + tq) * key_length + tk];
    }
    if (softcap > 0.0f) score = softcap * metal::tanh(score / softcap);
    const float probability = metal::exp(score - row_max);
    denominator += probability;
    #pragma clang loop unroll(full)
    for (int i = 0; i < VALUES; ++i) {
      acc[i] += probability * float(v[kv_base + int(lane) + i * 32]);
    }
  }
  const long out_base = q_base;
  const float inverse = valid_keys > 0 ? 1.0f / denominator : 0.0f;
  #pragma clang loop unroll(full)
  for (int i = 0; i < VALUES; ++i) out[out_base + int(lane) + i * 32] = T(acc[i] * inverse);
}

#define instantiate_cross(type_name, T, DVAL)                                    \
  template [[host_name("cross_attention_D" #DVAL "_" #type_name)]] [[kernel]] void\
  cross_attention<T, DVAL>(device const T *q [[buffer(0)]], device const T *k [[buffer(1)]],\
    device const T *v [[buffer(2)]], device const int *key_lengths [[buffer(3)]],\
    device const float *bias [[buffer(4)]], device T *out [[buffer(5)]],           \
    constant int &query_length [[buffer(6)]], constant int &key_length [[buffer(7)]],\
    constant int &query_heads [[buffer(8)]], constant int &kv_heads [[buffer(9)]],\
    constant float &scale [[buffer(10)]], constant float &softcap [[buffer(11)]], \
    constant int &has_bias [[buffer(12)]], uint3 group [[threadgroup_position_in_grid]],\
    uint lane [[thread_index_in_simdgroup]]);

#define instantiate_cross_type(type_name, T) \
  instantiate_cross(type_name, T, 64)         \
  instantiate_cross(type_name, T, 128)        \
  instantiate_cross(type_name, T, 256)

instantiate_cross_type(float32, float)
instantiate_cross_type(float16, half)
instantiate_cross_type(bfloat16, bf16)

}  // namespace mittens
