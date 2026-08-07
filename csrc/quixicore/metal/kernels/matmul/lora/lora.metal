#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// Decode/small-batch LoRA path. One threadgroup owns one row so the F16 low-rank
// projection can stay in threadgroup memory and the scale/base add can be fused
// into the up projection. The adapter contract deliberately rounds both
// projection outputs to F16, matching the two F16 framework GEMMs used by the
// prefill route while accumulating every dot product in FP32.
template <typename T>
kernel void lora_apply_direct_f16(
    device const T *x [[buffer(0)]],
    device const half *A [[buffer(1)]],
    device const half *B [[buffer(2)]],
    device const T *base [[buffer(3)]],
    device T *out [[buffer(4)]],
    constant int &input_dim [[buffer(5)]],
    constant int &output_dim [[buffer(6)]],
    constant int &rank [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant int &has_base [[buffer(9)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  threadgroup half low[256];
  const int row = int(group);
  const long x_base = (long)row * input_dim;

  // Eight SIMD groups partition the adapter rows. Each dot uses the same
  // SIMD reduction as the former two-dispatch implementation.
  for (int r = int(simdgroup); r < rank; r += 8) {
    float sum = 0.0f;
    const long a_base = (long)r * input_dim;
    for (int k = int(lane); k < input_dim; k += 32) {
      sum += float(x[x_base + k]) * float(A[a_base + k]);
    }
    sum = metal::simd_sum(sum);
    if (lane == 0) {
      low[r] = half(sum);
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (int output = int(tid); output < output_dim; output += 256) {
    float sum = 0.0f;
    const long b_base = (long)output * rank;
    for (int r = 0; r < rank; ++r) {
      sum += float(low[r]) * float(B[b_base + r]);
    }
    const float rounded_delta = float(half(sum));
    const long index = (long)row * output_dim + output;
    const float base_value = has_base != 0 ? float(base[index]) : 0.0f;
    out[index] = T(base_value + scale * rounded_delta);
  }
}

#define instantiate_lora(type_name, T)                                        \
  template [[host_name("lora_apply_direct_f16_" #type_name)]] [[kernel]] void\
  lora_apply_direct_f16<T>(device const T *x [[buffer(0)]],                    \
      device const half *A [[buffer(1)]], device const half *B [[buffer(2)]], \
      device const T *base [[buffer(3)]], device T *out [[buffer(4)]],        \
      constant int &input_dim [[buffer(5)]],                                 \
      constant int &output_dim [[buffer(6)]], constant int &rank [[buffer(7)]],\
      constant float &scale [[buffer(8)]], constant int &has_base [[buffer(9)]],\
      uint group [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint lane [[thread_index_in_simdgroup]],                                \
      uint simdgroup [[simdgroup_index_in_threadgroup]]);

instantiate_lora(float32, float)
instantiate_lora(float16, half)
instantiate_lora(bfloat16, bf16)

}  // namespace mittens
