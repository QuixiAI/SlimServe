#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

template <typename T, int D>
kernel void audio_relative_attention_kernel(
    device const T *q [[buffer(0)]], device const T *k [[buffer(1)]],
    device const T *v [[buffer(2)]], device const T *relative_k [[buffer(3)]],
    device const float *per_dim_scale [[buffer(4)]],
    device const int *lengths [[buffer(5)]], device T *out [[buffer(6)]],
    constant int &length [[buffer(7)]], constant int &heads [[buffer(8)]],
    constant int &relative_positions [[buffer(9)]],
    constant int &chunk_size [[buffer(10)]], constant int &left_context [[buffer(11)]],
    constant int &right_context [[buffer(12)]], constant float &q_scale [[buffer(13)]],
    constant float &k_scale [[buffer(14)]], constant float &softcap [[buffer(15)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constexpr int VALUES = D / 32;
  const int t = int(group.x), head = int(group.y), batch = int(group.z);
  const int valid_length = metal::clamp(lengths[batch], 0, length);
  const long row = ((long)batch * length + t) * heads + head;
  const long q_base = row * D;
  if (t >= valid_length) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < VALUES; ++i)
      out[q_base + int(lane) + i * 32] = T(0.0f);
    return;
  }

  const int query_in_chunk = t % chunk_size;
  const int block_start = (t / chunk_size) * chunk_size;
  const int context_start = block_start - (left_context - 1);
  const int context_length = chunk_size + left_context - 1 + right_context;
  float qv[VALUES];
  #pragma clang loop unroll(full)
  for (int i = 0; i < VALUES; ++i) {
    const int d = int(lane) + i * 32;
    const float raw_scale = per_dim_scale[d];
    const float learned_scale = metal::max(raw_scale, 0.0f) +
                                metal::log(1.0f + metal::exp(-metal::abs(raw_scale)));
    qv[i] = float(q[q_base + d]) * q_scale * learned_scale;
  }

  float row_max = -INFINITY;
  for (int ci = 0; ci < context_length; ++ci) {
    const int key_t = context_start + ci;
    if (key_t < 0 || key_t >= valid_length) continue;
    const long kv_base = (((long)batch * length + key_t) * heads + head) * D;
    const int relative_index = ci - query_in_chunk;
    const bool has_relative = relative_index >= 0 && relative_index < relative_positions;
    const long relative_base = ((long)relative_index * heads + head) * D;
    float score = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < VALUES; ++i) {
      const int d = int(lane) + i * 32;
      score += qv[i] * float(k[kv_base + d]) * k_scale;
      if (has_relative) score += qv[i] * float(relative_k[relative_base + d]);
    }
    score = metal::simd_sum(score);
    if (softcap > 0.0f) score = softcap * metal::tanh(score / softcap);
    row_max = metal::max(row_max, score);
  }

  float acc[VALUES];
  #pragma clang loop unroll(full)
  for (int i = 0; i < VALUES; ++i) acc[i] = 0.0f;
  float denominator = 0.0f;
  for (int ci = 0; ci < context_length; ++ci) {
    const int key_t = context_start + ci;
    if (key_t < 0 || key_t >= valid_length) continue;
    const long kv_base = (((long)batch * length + key_t) * heads + head) * D;
    const int relative_index = ci - query_in_chunk;
    const bool has_relative = relative_index >= 0 && relative_index < relative_positions;
    const long relative_base = ((long)relative_index * heads + head) * D;
    float score = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < VALUES; ++i) {
      const int d = int(lane) + i * 32;
      score += qv[i] * float(k[kv_base + d]) * k_scale;
      if (has_relative) score += qv[i] * float(relative_k[relative_base + d]);
    }
    score = metal::simd_sum(score);
    if (softcap > 0.0f) score = softcap * metal::tanh(score / softcap);
    const float probability = metal::exp(score - row_max);
    denominator += probability;
    #pragma clang loop unroll(full)
    for (int i = 0; i < VALUES; ++i)
      acc[i] += probability * float(v[kv_base + int(lane) + i * 32]);
  }
  const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
  #pragma clang loop unroll(full)
  for (int i = 0; i < VALUES; ++i)
    out[q_base + int(lane) + i * 32] = T(acc[i] * inverse);
}

#define instantiate_audio_relative(type_name, T, DVAL)                         \
  template [[host_name("audio_relative_attention_D" #DVAL "_" #type_name)]] [[kernel]] void\
  audio_relative_attention_kernel<T, DVAL>(device const T *q [[buffer(0)]],     \
    device const T *k [[buffer(1)]], device const T *v [[buffer(2)]],           \
    device const T *relative_k [[buffer(3)]], device const float *per_dim_scale [[buffer(4)]],\
    device const int *lengths [[buffer(5)]], device T *out [[buffer(6)]],       \
    constant int &length [[buffer(7)]], constant int &heads [[buffer(8)]],      \
    constant int &relative_positions [[buffer(9)]], constant int &chunk_size [[buffer(10)]],\
    constant int &left_context [[buffer(11)]], constant int &right_context [[buffer(12)]],\
    constant float &q_scale [[buffer(13)]], constant float &k_scale [[buffer(14)]],\
    constant float &softcap [[buffer(15)]], uint3 group [[threadgroup_position_in_grid]],\
    uint lane [[thread_index_in_simdgroup]]);

#define instantiate_audio_relative_type(type_name, T) \
  instantiate_audio_relative(type_name, T, 64)        \
  instantiate_audio_relative(type_name, T, 128)       \
  instantiate_audio_relative(type_name, T, 256)

instantiate_audio_relative_type(float32, float)
instantiate_audio_relative_type(float16, half)
instantiate_audio_relative_type(bfloat16, bf16)

}  // namespace mittens
