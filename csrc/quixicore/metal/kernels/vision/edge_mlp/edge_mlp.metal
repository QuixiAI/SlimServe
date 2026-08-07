#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Fixed 512->256->7 pairwise MLP, factorized as
//   left[b,i]  = hidden[b,i] @ W[:,:256]^T
//   right[b,j] = hidden[b,j] @ W[:,256:]^T + bias
//   out[b,:,i,j] = W2 @ GELU(left[b,i] + right[b,j]) + bias2.
// The first projection is linear in the two pair inputs, so computing each
// side once reduces that stage from O(L^2) matrix-vector products to O(L).
// Both partials use the public dtype so the combine kernel retains the legacy
// separate left/right rounding points.
template <typename T>
kernel void edge_mlp_project_256(
    device const T *hidden [[buffer(0)]],
    device const T *first_weight [[buffer(1)]],
    device const T *first_bias [[buffer(2)]],
    device T *left_output [[buffer(3)]],
    device T *right_output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &length [[buffer(6)]],
    uint row [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]]) {
    if (row >= uint(batch * length)) return;
    const int feature = int(thread_id);
    const long hidden_base = (long)row * 256;
    const long weight_base = (long)feature * 512;
    device const metal::vec<T, 4> *hidden4 =
        (device const metal::vec<T, 4> *)(hidden + hidden_base);
    device const metal::vec<T, 4> *left_weight4 =
        (device const metal::vec<T, 4> *)(first_weight + weight_base);
    device const metal::vec<T, 4> *right_weight4 =
        (device const metal::vec<T, 4> *)(first_weight + weight_base + 256);
    float left = 0.0f;
    float right = float(first_bias[feature]);
    #pragma clang loop unroll(full)
    for (int dimension4 = 0; dimension4 < 64; ++dimension4) {
        const float4 input = float4(hidden4[dimension4]);
        left += dot(input, float4(left_weight4[dimension4]));
        right += dot(input, float4(right_weight4[dimension4]));
    }
    left_output[hidden_base + feature] = T(left);
    right_output[hidden_base + feature] = T(right);
}

template <typename T>
kernel void edge_mlp_combine_256x7(
    device const T *left_partial [[buffer(0)]],
    device const T *right_partial [[buffer(1)]],
    device const T *second_weight [[buffer(2)]],
    device const T *second_bias [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &length [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint thread_id [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]) {
    const int batch_index = int(group.y);
    const int pair = int(group.x);
    if (batch_index >= batch || pair >= length * length) return;
    const int left_index = pair / length;
    const int right_index = pair - left_index * length;
    const int feature = int(thread_id);
    const long left_base = ((long)batch_index * length + left_index) * 256;
    const long right_base = ((long)batch_index * length + right_index) * 256;
    threadgroup float activation[256];
    const T joined = T(float(left_partial[left_base + feature]) +
                       float(right_partial[right_base + feature]));
    activation[feature] = float(T(glu_gelu_erf(float(joined))));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simdgroup_id < 7) {
        const int edge_class = int(simdgroup_id);
        float accumulator = 0.0f;
        for (int index = int(lane); index < 256; index += 32) {
            accumulator += activation[index] *
                           float(second_weight[edge_class * 256 + index]);
        }
        accumulator = metal::simd_sum(accumulator);
        if (lane == 0) {
            const long destination =
                ((long)batch_index * 7 + edge_class) * length * length + pair;
            output[destination] = T(accumulator + float(second_bias[edge_class]));
        }
    }
}

#define instantiate_edge_mlp_project(type_name, T)                            \
  template [[host_name("edge_mlp_project_256_" #type_name)]] [[kernel]]     \
  void edge_mlp_project_256<T>(device const T *hidden [[buffer(0)]],          \
    device const T *first_weight [[buffer(1)]],                               \
    device const T *first_bias [[buffer(2)]],                                 \
    device T *left_output [[buffer(3)]], device T *right_output [[buffer(4)]], \
    constant int &batch [[buffer(5)]], constant int &length [[buffer(6)]],    \
    uint row [[threadgroup_position_in_grid]],                                \
    uint thread_id [[thread_index_in_threadgroup]]);                          \
                                                                               \
  template [[host_name("edge_mlp_combine_256x7_" #type_name)]] [[kernel]]  \
  void edge_mlp_combine_256x7<T>(device const T *left_partial [[buffer(0)]],  \
    device const T *right_partial [[buffer(1)]],                              \
    device const T *second_weight [[buffer(2)]],                              \
    device const T *second_bias [[buffer(3)]], device T *output [[buffer(4)]], \
    constant int &batch [[buffer(5)]], constant int &length [[buffer(6)]],    \
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint thread_id [[thread_index_in_threadgroup]],                           \
    uint lane [[thread_index_in_simdgroup]],                                  \
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]);

instantiate_edge_mlp_project(float32, float)
instantiate_edge_mlp_project(bfloat16, bf16)

} // namespace mittens
