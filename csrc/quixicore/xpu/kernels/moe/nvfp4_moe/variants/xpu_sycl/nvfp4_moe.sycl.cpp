// Decode-oriented fused and split routed MoE for ModelOpt NVFP4 weights.

#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"

#include <cstdint>
#include <vector>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kSubgroups = 8;
constexpr int kWG = kSG * kSubgroups;
// Preserve enough independent output work-groups to fill one B60 while each
// group amortizes its local SwiGLU activation across several output rows.
constexpr std::size_t kTargetOutputWorkGroups = 128;
using WVec = sycl::vec<std::uint32_t, 4>;

inline sycl::vec<float, 2> decode_e2m1_pair(std::uint32_t word) {
  const std::uint32_t nibbles = word & 0x000f000fu;
  const std::uint32_t halves = ((nibbles & 0x00080008u) << 12) | ((nibbles & 0x00070007u) << 9);
  return sycl::bit_cast<sycl::vec<sycl::half, 2>>(halves).template convert<float>();
}

inline float decode_e4m3_raw(std::uint8_t value) {
  const auto bits = static_cast<std::uint16_t>(((value & 0x80u) << 8) | ((value & 0x7fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

template <typename T>
inline float nvfp4_row_dot(const sycl::sub_group &subgroup, const std::uint8_t *weight,
                           const std::uint8_t *block_scales, float global_scale, const T *input,
                           std::size_t K) {
  const WVec *packed = reinterpret_cast<const WVec *>(weight);
  const int lane = static_cast<int>(subgroup.get_local_linear_id());
  float partial = 0.0f;
  for (std::size_t chunk = lane; chunk < K / 32; chunk += kSG) {
    const WVec values = packed[chunk];
    const float scale0 = decode_e4m3_raw(block_scales[2 * chunk]) * global_scale;
    const float scale1 = decode_e4m3_raw(block_scales[2 * chunk + 1]) * global_scale;
    const T *x = input + chunk * 32;
    float words[4];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      const std::uint32_t word = values[wi];
      const sycl::vec<T, 8> xv =
          *reinterpret_cast<const sycl::vec<T, 8> *>(x + wi * 8);
      float low = 0.0f;
      float high = 0.0f;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const sycl::vec<float, 2> decoded = decode_e2m1_pair(word >> (4 * pair));
        low += decoded[0] * static_cast<float>(xv[pair]);
        high += decoded[1] * static_cast<float>(xv[pair + 4]);
      }
      words[wi] = low + high;
    }
    partial += (words[0] + words[1]) * scale0 + (words[2] + words[3]) * scale1;
  }
  return sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
}

// Gate row i and up row i consume the same hidden vector. Decode the pair in
// one subgroup pass so every hidden load and chunk-index calculation is shared.
template <typename T>
inline sycl::vec<float, 2>
nvfp4_row_dot_pair(const sycl::sub_group &subgroup, const std::uint8_t *gate_weight,
                   const std::uint8_t *gate_scales, const std::uint8_t *up_weight,
                   const std::uint8_t *up_scales, float global_scale, const T *input,
                   std::size_t K) {
  const WVec *gate_packed = reinterpret_cast<const WVec *>(gate_weight);
  const WVec *up_packed = reinterpret_cast<const WVec *>(up_weight);
  const int lane = static_cast<int>(subgroup.get_local_linear_id());
  float gate_partial = 0.0f;
  float up_partial = 0.0f;
  for (std::size_t chunk = lane; chunk < K / 32; chunk += kSG) {
    const WVec gate_values = gate_packed[chunk];
    const WVec up_values = up_packed[chunk];
    const float gate_scale0 = decode_e4m3_raw(gate_scales[2 * chunk]) * global_scale;
    const float gate_scale1 = decode_e4m3_raw(gate_scales[2 * chunk + 1]) * global_scale;
    const float up_scale0 = decode_e4m3_raw(up_scales[2 * chunk]) * global_scale;
    const float up_scale1 = decode_e4m3_raw(up_scales[2 * chunk + 1]) * global_scale;
    const T *x = input + chunk * 32;
    float gate_words[4];
    float up_words[4];
#pragma unroll
    for (int wi = 0; wi < 4; ++wi) {
      const std::uint32_t gate_word = gate_values[wi];
      const std::uint32_t up_word = up_values[wi];
      const sycl::vec<T, 8> xv =
          *reinterpret_cast<const sycl::vec<T, 8> *>(x + wi * 8);
      float gate_low = 0.0f;
      float gate_high = 0.0f;
      float up_low = 0.0f;
      float up_high = 0.0f;
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const sycl::vec<float, 2> gate = decode_e2m1_pair(gate_word >> (4 * pair));
        const sycl::vec<float, 2> up = decode_e2m1_pair(up_word >> (4 * pair));
        const float x_low = static_cast<float>(xv[pair]);
        const float x_high = static_cast<float>(xv[pair + 4]);
        gate_low += gate[0] * x_low;
        gate_high += gate[1] * x_high;
        up_low += up[0] * x_low;
        up_high += up[1] * x_high;
      }
      gate_words[wi] = gate_low + gate_high;
      up_words[wi] = up_low + up_high;
    }
    gate_partial += (gate_words[0] + gate_words[1]) * gate_scale0 +
                    (gate_words[2] + gate_words[3]) * gate_scale1;
    up_partial += (up_words[0] + up_words[1]) * up_scale0 +
                  (up_words[2] + up_words[3]) * up_scale1;
  }
  return {sycl::reduce_over_group(subgroup, gate_partial, sycl::plus<float>()),
          sycl::reduce_over_group(subgroup, up_partial, sycl::plus<float>())};
}

inline float silu(float value) {
  return value / (1.0f + sycl::exp(-value));
}

template <typename T> class Nvfp4MoeFusedKernel;

template <typename T>
sycl::event
fused_typed(sycl::queue &q, const T *hidden, const int *topk_ids, const float *topk_weights,
            const std::uint8_t *w13, const std::uint8_t *w13_scales, const float *w13_global_scales,
            const std::uint8_t *w2, const std::uint8_t *w2_scales, const float *w2_global_scales,
            float *output, std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
            std::size_t I, bool multiply_router_weight, const sycl::event &output_ready) {
  const std::size_t two_i = 2 * I;
  const std::size_t w13_expert_stride = two_i * (K / 2);
  const std::size_t s13_expert_stride = two_i * (K / 16);
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  const sycl::nd_range<1> range(sycl::range<1>(M * top_k * kWG), sycl::range<1>(kWG));

  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(output_ready);
    sycl::local_accessor<float, 1> gate_up(sycl::range<1>(two_i), handler);
    sycl::local_accessor<float, 1> activated(sycl::range<1>(I), handler);
    handler.parallel_for<Nvfp4MoeFusedKernel<T>>(
        range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_group(0);
          const std::size_t token = pair / top_k;
          const int expert = topk_ids[pair];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;

          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const int thread = static_cast<int>(item.get_local_linear_id());
          const std::size_t expert_index = static_cast<std::size_t>(expert);
          const float w13_global = w13_global_scales[expert_index] * 4194304.0f;
          const std::uint8_t *expert_w13 = w13 + expert_index * w13_expert_stride;
          const std::uint8_t *expert_s13 = w13_scales + expert_index * s13_expert_stride;

          for (std::size_t row = subgroup_id; row < I; row += kSubgroups) {
            const sycl::vec<float, 2> values = nvfp4_row_dot_pair(
                subgroup, expert_w13 + row * (K / 2), expert_s13 + row * (K / 16),
                expert_w13 + (row + I) * (K / 2), expert_s13 + (row + I) * (K / 16), w13_global,
                hidden + token * K, K);
            if (lane == 0) {
              gate_up[row] = values[0];
              gate_up[row + I] = values[1];
            }
          }
          sycl::group_barrier(item.get_group());

          for (std::size_t i = thread; i < I; i += kWG) {
            activated[i] = silu(gate_up[i]) * gate_up[i + I];
          }
          sycl::group_barrier(item.get_group());

          const float router_weight = multiply_router_weight ? topk_weights[pair] : 1.0f;
          const float w2_global = w2_global_scales[expert_index] * 4194304.0f;
          const std::uint8_t *expert_w2 = w2 + expert_index * w2_expert_stride;
          const std::uint8_t *expert_s2 = w2_scales + expert_index * s2_expert_stride;
          const float *activation = &activated[0];
          for (std::size_t row = subgroup_id; row < K; row += kSubgroups) {
            const float value = nvfp4_row_dot(subgroup, expert_w2 + row * (I / 2),
                                              expert_s2 + row * (I / 16), w2_global, activation, I);
            if (lane == 0) {
              sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  accumulator(output[token * K + row]);
              accumulator.fetch_add(router_weight * value);
            }
          }
        });
  });
}

template <typename T> class Nvfp4MoeGateUpKernel;

template <int RowsPerSubgroup> class Nvfp4MoeOutputKernel;

template <typename T>
sycl::event gate_up_typed(sycl::queue &q, const T *hidden, const int *topk_ids,
                          const std::uint8_t *w13, const std::uint8_t *w13_scales,
                          const float *w13_global_scales, float *scratch, std::size_t pairs,
                          std::size_t E, std::size_t top_k, std::size_t K, std::size_t I) {
  const std::size_t two_i = 2 * I;
  const std::size_t w13_expert_stride = two_i * (K / 2);
  const std::size_t s13_expert_stride = two_i * (K / 16);
  const std::size_t row_tiles = (I + kSubgroups - 1) / kSubgroups;
  const sycl::nd_range<2> range(sycl::range<2>(pairs, row_tiles * kWG), sycl::range<2>(1, kWG));
  return q.parallel_for<Nvfp4MoeGateUpKernel<T>>(
      range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
        const std::size_t pair = item.get_global_id(0);
        const int expert = topk_ids[pair];
        if (expert < 0 || static_cast<std::size_t>(expert) >= E)
          return;
        const std::size_t token = pair / top_k;
        const sycl::sub_group subgroup = item.get_sub_group();
        const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
        const int lane = static_cast<int>(subgroup.get_local_linear_id());
        const std::size_t row = item.get_group(1) * kSubgroups + subgroup_id;
        if (row >= I)
          return;
        const std::size_t expert_index = static_cast<std::size_t>(expert);
        const float global = w13_global_scales[expert_index] * 4194304.0f;
        const std::uint8_t *expert_w13 = w13 + expert_index * w13_expert_stride;
        const std::uint8_t *expert_s13 = w13_scales + expert_index * s13_expert_stride;
        const sycl::vec<float, 2> values = nvfp4_row_dot_pair(
            subgroup, expert_w13 + row * (K / 2), expert_s13 + row * (K / 16),
            expert_w13 + (row + I) * (K / 2), expert_s13 + (row + I) * (K / 16), global,
            hidden + token * K, K);
        if (lane == 0) {
          scratch[pair * two_i + row] = values[0];
          scratch[pair * two_i + row + I] = values[1];
        }
      });
}

template <int RowsPerSubgroup>
sycl::event output_typed(sycl::queue &q, const int *topk_ids, const float *topk_weights,
                         const std::uint8_t *w2, const std::uint8_t *w2_scales,
                         const float *w2_global_scales, const float *scratch, float *output,
                         std::size_t pairs, std::size_t E, std::size_t top_k, std::size_t K,
                         std::size_t I, bool multiply_router_weight,
                         const sycl::event &gate_up_ready, const sycl::event &output_ready) {
  const std::size_t two_i = 2 * I;
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  constexpr std::size_t kRowsPerWorkGroup = kSubgroups * RowsPerSubgroup;
  const std::size_t row_tiles = (K + kRowsPerWorkGroup - 1) / kRowsPerWorkGroup;
  const sycl::nd_range<2> range(sycl::range<2>(pairs, row_tiles * kWG), sycl::range<2>(1, kWG));
  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(std::vector<sycl::event>{gate_up_ready, output_ready});
    sycl::local_accessor<float, 1> activated(sycl::range<1>(I), handler);
    handler.parallel_for<Nvfp4MoeOutputKernel<RowsPerSubgroup>>(
        range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_global_id(0);
          const int expert = topk_ids[pair];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;
          const std::size_t token = pair / top_k;
          const int thread = static_cast<int>(item.get_local_linear_id());
          const float *gate_up = scratch + pair * two_i;
          for (std::size_t i = static_cast<std::size_t>(thread); i < I; i += kWG)
            activated[i] = silu(gate_up[i]) * gate_up[i + I];
          sycl::group_barrier(item.get_group());

          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const std::size_t expert_index = static_cast<std::size_t>(expert);
          const float global = w2_global_scales[expert_index] * 4194304.0f;
          const float router_weight = multiply_router_weight ? topk_weights[pair] : 1.0f;
          const std::size_t first_row =
              item.get_group(1) * kRowsPerWorkGroup + static_cast<std::size_t>(subgroup_id);
          // The activation and barrier above are paid once per work-group. Each
          // subgroup then consumes multiple rows without changing row ownership.
          for (int row_in_subgroup = 0; row_in_subgroup < RowsPerSubgroup;
               ++row_in_subgroup) {
            const std::size_t row =
                first_row + static_cast<std::size_t>(row_in_subgroup) * kSubgroups;
            if (row >= K)
              break;
            const float value = nvfp4_row_dot(
                subgroup, w2 + expert_index * w2_expert_stride + row * (I / 2),
                w2_scales + expert_index * s2_expert_stride + row * (I / 16), global,
                &activated[0], I);
            if (lane == 0) {
              sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  accumulator(output[token * K + row]);
              accumulator.fetch_add(router_weight * value);
            }
          }
        });
  });
}

template <typename T>
sycl::event dispatch_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w13, const void *w13_scales,
                           const float *w13_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *out_f32, std::size_t M,
                           std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                           bool multiply_router_weight, const sycl::event &output_ready) {
  return fused_typed(q, static_cast<const T *>(hidden), topk_ids, topk_weights,
                     static_cast<const std::uint8_t *>(w13),
                     static_cast<const std::uint8_t *>(w13_scales), w13_global_scales,
                     static_cast<const std::uint8_t *>(w2),
                     static_cast<const std::uint8_t *>(w2_scales), w2_global_scales, out_f32, M, E,
                     top_k, K, I, multiply_router_weight, output_ready);
}

template <typename T>
sycl::event dispatch_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w13, const void *w13_scales,
                           const float *w13_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *scratch_f32, float *out_f32,
                           std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
                           std::size_t I, bool multiply_router_weight,
                           const sycl::event &output_ready) {
  const std::size_t pairs = M * top_k;
  const sycl::event gate_up_ready = gate_up_typed(
      q, static_cast<const T *>(hidden), topk_ids, static_cast<const std::uint8_t *>(w13),
      static_cast<const std::uint8_t *>(w13_scales), w13_global_scales, scratch_f32, pairs, E,
      top_k, K, I);
  const std::size_t baseline_output_work_groups = pairs * ((K + kSubgroups - 1) / kSubgroups);
  // Pick the largest measured row tile that leaves at least the target number
  // of work-groups. Small K/top-k shapes retain the original one-row mapping.
  const auto launch_output = [&]<int RowsPerSubgroup>() {
    return output_typed<RowsPerSubgroup>(
        q, topk_ids, topk_weights, static_cast<const std::uint8_t *>(w2),
        static_cast<const std::uint8_t *>(w2_scales), w2_global_scales, scratch_f32, out_f32, pairs,
        E, top_k, K, I, multiply_router_weight, gate_up_ready, output_ready);
  };
  if (baseline_output_work_groups >= kTargetOutputWorkGroups * 16)
    return launch_output.template operator()<16>();
  if (baseline_output_work_groups >= kTargetOutputWorkGroups * 8)
    return launch_output.template operator()<8>();
  if (baseline_output_work_groups >= kTargetOutputWorkGroups * 4)
    return launch_output.template operator()<4>();
  if (baseline_output_work_groups >= kTargetOutputWorkGroups * 2)
    return launch_output.template operator()<2>();
  return launch_output.template operator()<1>();
}

// ---- ReLU²-ungated MoE (NemotronH) ----
// Mirrors the SwiGLU kernels but: w1 is a SINGLE up-projection [E, I, K/2]
// (not gate+up [E, 2I, K/2]), and the activation is squared-ReLU with no gate
// multiply: a[i] = relu(g[i])^2. Activations stay 16-bit (W4A16). Ported back
// from the vLLM XPU serving prototype (vllm-xpu-kernels commit dffcab7),
// which itself originated from this file's SwiGLU kernels.

template <typename T> class Nvfp4MoeRelu2FusedKernel;

template <typename T>
sycl::event relu2_fused_typed(sycl::queue &q, const T *hidden, const int *topk_ids,
                              const float *topk_weights, const std::uint8_t *w1,
                              const std::uint8_t *w1_scales, const float *w1_global_scales,
                              const std::uint8_t *w2, const std::uint8_t *w2_scales,
                              const float *w2_global_scales, float *output, std::size_t M,
                              std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                              bool multiply_router_weight, const sycl::event &output_ready) {
  const std::size_t w1_expert_stride = I * (K / 2);
  const std::size_t s1_expert_stride = I * (K / 16);
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  const sycl::nd_range<1> range(sycl::range<1>(M * top_k * kWG), sycl::range<1>(kWG));

  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(output_ready);
    sycl::local_accessor<float, 1> up(sycl::range<1>(I), handler);
    sycl::local_accessor<float, 1> activated(sycl::range<1>(I), handler);
    handler.parallel_for<Nvfp4MoeRelu2FusedKernel<T>>(
        range, [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_group(0);
          const std::size_t token = pair / top_k;
          const int expert = topk_ids[pair];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;

          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const int thread = static_cast<int>(item.get_local_linear_id());
          const std::size_t expert_index = static_cast<std::size_t>(expert);
          const float w1_global = w1_global_scales[expert_index] * 4194304.0f;
          const std::uint8_t *expert_w1 = w1 + expert_index * w1_expert_stride;
          const std::uint8_t *expert_s1 = w1_scales + expert_index * s1_expert_stride;

          for (std::size_t row = subgroup_id; row < I; row += kSubgroups) {
            const float value =
                nvfp4_row_dot(subgroup, expert_w1 + row * (K / 2), expert_s1 + row * (K / 16),
                              w1_global, hidden + token * K, K);
            if (lane == 0)
              up[row] = value;
          }
          sycl::group_barrier(item.get_group());

          for (std::size_t i = thread; i < I; i += kWG) {
            const float rectified = sycl::fmax(up[i], 0.0f);
            activated[i] = rectified * rectified;
          }
          sycl::group_barrier(item.get_group());

          const float router_weight = multiply_router_weight ? topk_weights[pair] : 1.0f;
          const float w2_global = w2_global_scales[expert_index] * 4194304.0f;
          const std::uint8_t *expert_w2 = w2 + expert_index * w2_expert_stride;
          const std::uint8_t *expert_s2 = w2_scales + expert_index * s2_expert_stride;
          const float *activation = &activated[0];
          for (std::size_t row = subgroup_id; row < K; row += kSubgroups) {
            const float value = nvfp4_row_dot(subgroup, expert_w2 + row * (I / 2),
                                              expert_s2 + row * (I / 16), w2_global, activation, I);
            if (lane == 0) {
              sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  accumulator(output[token * K + row]);
              accumulator.fetch_add(router_weight * value);
            }
          }
        });
  });
}

template <typename T> class Nvfp4MoeRelu2UpKernel;
class Nvfp4MoeRelu2OutputKernel;

template <typename T>
sycl::event relu2_up_typed(sycl::queue &q, const T *hidden, const int *topk_ids,
                           const std::uint8_t *w1, const std::uint8_t *w1_scales,
                           const float *w1_global_scales, float *scratch, std::size_t pairs,
                           std::size_t top_k, std::size_t E, std::size_t K, std::size_t I) {
  const std::size_t w1_expert_stride = I * (K / 2);
  const std::size_t s1_expert_stride = I * (K / 16);
  const std::size_t row_tiles = (I + kSubgroups - 1) / kSubgroups;
  const sycl::nd_range<2> range(sycl::range<2>(pairs, row_tiles * kWG), sycl::range<2>(1, kWG));
  return q.parallel_for<Nvfp4MoeRelu2UpKernel<T>>(
      range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
        const std::size_t pair = item.get_global_id(0);
        const int expert = topk_ids[pair];
        if (expert < 0 || static_cast<std::size_t>(expert) >= E)
          return;
        const std::size_t token = pair / top_k;
        const sycl::sub_group subgroup = item.get_sub_group();
        const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
        const int lane = static_cast<int>(subgroup.get_local_linear_id());
        const std::size_t row = item.get_group(1) * kSubgroups + subgroup_id;
        if (row >= I)
          return;
        const std::size_t expert_index = static_cast<std::size_t>(expert);
        const float w1_global = w1_global_scales[expert_index] * 4194304.0f;
        const float value = nvfp4_row_dot(
            subgroup, w1 + expert_index * w1_expert_stride + row * (K / 2),
            w1_scales + expert_index * s1_expert_stride + row * (K / 16), w1_global,
            hidden + token * K, K);
        if (lane == 0)
          scratch[pair * I + row] = value;
      });
}

inline sycl::event relu2_output_typed(sycl::queue &q, const int *topk_ids,
                                      const float *topk_weights, const std::uint8_t *w2,
                                      const std::uint8_t *w2_scales,
                                      const float *w2_global_scales, const float *scratch,
                                      float *output, std::size_t pairs, std::size_t top_k,
                                      std::size_t E, std::size_t K, std::size_t I,
                                      bool multiply_router_weight,
                                      const sycl::event &up_ready,
                                      const sycl::event &output_ready) {
  const std::size_t w2_expert_stride = K * (I / 2);
  const std::size_t s2_expert_stride = K * (I / 16);
  const std::size_t row_tiles = (K + kSubgroups - 1) / kSubgroups;
  const sycl::nd_range<2> range(sycl::range<2>(pairs, row_tiles * kWG), sycl::range<2>(1, kWG));
  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(std::vector<sycl::event>{up_ready, output_ready});
    sycl::local_accessor<float, 1> activated(sycl::range<1>(I), handler);
    handler.parallel_for<Nvfp4MoeRelu2OutputKernel>(
        range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t pair = item.get_global_id(0);
          const int expert = topk_ids[pair];
          if (expert < 0 || static_cast<std::size_t>(expert) >= E)
            return;
          const std::size_t token = pair / top_k;
          const float router_weight = multiply_router_weight ? topk_weights[pair] : 1.0f;
          const sycl::sub_group subgroup = item.get_sub_group();
          const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
          const int lane = static_cast<int>(subgroup.get_local_linear_id());
          const int thread = static_cast<int>(item.get_local_linear_id());
          const float *up_row = scratch + pair * I;
          for (std::size_t i = thread; i < I; i += kWG) {
            const float rectified = sycl::fmax(up_row[i], 0.0f);
            activated[i] = rectified * rectified;
          }
          sycl::group_barrier(item.get_group());
          const std::size_t row = item.get_group(1) * kSubgroups + subgroup_id;
          if (row >= K)
            return;
          const std::size_t expert_index = static_cast<std::size_t>(expert);
          const float w2_global = w2_global_scales[expert_index] * 4194304.0f;
          const float value = nvfp4_row_dot(
              subgroup, w2 + expert_index * w2_expert_stride + row * (I / 2),
              w2_scales + expert_index * s2_expert_stride + row * (I / 16), w2_global,
              &activated[0], I);
          if (lane == 0) {
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                accumulator(output[token * K + row]);
            accumulator.fetch_add(router_weight * value);
          }
        });
  });
}

template <typename T>
sycl::event dispatch_relu2_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w1, const void *w1_scales,
                                 const float *w1_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *scratch_f32, float *out_f32, std::size_t M, std::size_t E,
                                 std::size_t top_k, std::size_t K, std::size_t I,
                                 bool multiply_router_weight, const sycl::event &output_ready) {
  const std::size_t pairs = M * top_k;
  const sycl::event up_ready = relu2_up_typed(
      q, static_cast<const T *>(hidden), topk_ids, static_cast<const std::uint8_t *>(w1),
      static_cast<const std::uint8_t *>(w1_scales), w1_global_scales, scratch_f32, pairs, top_k, E,
      K, I);
  return relu2_output_typed(q, topk_ids, topk_weights, static_cast<const std::uint8_t *>(w2),
                            static_cast<const std::uint8_t *>(w2_scales), w2_global_scales,
                            scratch_f32, out_f32, pairs, top_k, E, K, I, multiply_router_weight,
                            up_ready, output_ready);
}

} // namespace

sycl::event nvfp4_moe_fused_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *out_f32, std::size_t M, std::size_t E, std::size_t top_k,
                                 std::size_t K, std::size_t I, bool multiply_router_weight,
                                 DType act_dt, const sycl::event &output_ready) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_fused<float>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                 w13_global_scales, w2, w2_scales, w2_global_scales, out_f32, M, E,
                                 top_k, K, I, multiply_router_weight, output_ready);
  case DType::f16:
    return dispatch_fused<half_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, out_f32, M, E,
                                  top_k, K, I, multiply_router_weight, output_ready);
  case DType::bf16:
    return dispatch_fused<bf16_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, out_f32, M, E,
                                  top_k, K, I, multiply_router_weight, output_ready);
  }
  return {};
}

sycl::event nvfp4_moe_split_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                 const float *topk_weights, const void *w13, const void *w13_scales,
                                 const float *w13_global_scales, const void *w2,
                                 const void *w2_scales, const float *w2_global_scales,
                                 float *scratch_f32, float *out_f32, std::size_t M, std::size_t E,
                                 std::size_t top_k, std::size_t K, std::size_t I,
                                 bool multiply_router_weight, DType act_dt,
                                 const sycl::event &output_ready) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_split<float>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                 w13_global_scales, w2, w2_scales, w2_global_scales, scratch_f32,
                                 out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready);
  case DType::f16:
    return dispatch_split<half_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, scratch_f32,
                                  out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready);
  case DType::bf16:
    return dispatch_split<bf16_t>(q, hidden, topk_ids, topk_weights, w13, w13_scales,
                                  w13_global_scales, w2, w2_scales, w2_global_scales, scratch_f32,
                                  out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready);
  }
  return {};
}

sycl::event nvfp4_moe_relu2_fused_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                       const float *topk_weights, const void *w1,
                                       const void *w1_scales, const float *w1_global_scales,
                                       const void *w2, const void *w2_scales,
                                       const float *w2_global_scales, float *out_f32,
                                       std::size_t M, std::size_t E, std::size_t top_k,
                                       std::size_t K, std::size_t I, bool multiply_router_weight,
                                       DType act_dt, const sycl::event &output_ready) {
#define QX_R2F(T)                                                                                  \
  return relu2_fused_typed<T>(q, static_cast<const T *>(hidden), topk_ids, topk_weights,           \
                              static_cast<const std::uint8_t *>(w1),                               \
                              static_cast<const std::uint8_t *>(w1_scales), w1_global_scales,      \
                              static_cast<const std::uint8_t *>(w2),                               \
                              static_cast<const std::uint8_t *>(w2_scales), w2_global_scales,      \
                              out_f32, M, E, top_k, K, I, multiply_router_weight, output_ready)
  switch (act_dt) {
  case DType::f32:
    QX_R2F(float);
  case DType::f16:
    QX_R2F(half_t);
  case DType::bf16:
    QX_R2F(bf16_t);
  }
#undef QX_R2F
  return {};
}

sycl::event nvfp4_moe_relu2_split_sycl(sycl::queue &q, const void *hidden, const int *topk_ids,
                                       const float *topk_weights, const void *w1,
                                       const void *w1_scales, const float *w1_global_scales,
                                       const void *w2, const void *w2_scales,
                                       const float *w2_global_scales, float *scratch_f32,
                                       float *out_f32, std::size_t M, std::size_t E,
                                       std::size_t top_k, std::size_t K, std::size_t I,
                                       bool multiply_router_weight, DType act_dt,
                                       const sycl::event &output_ready) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_relu2_split<float>(q, hidden, topk_ids, topk_weights, w1, w1_scales,
                                       w1_global_scales, w2, w2_scales, w2_global_scales,
                                       scratch_f32, out_f32, M, E, top_k, K, I,
                                       multiply_router_weight, output_ready);
  case DType::f16:
    return dispatch_relu2_split<half_t>(q, hidden, topk_ids, topk_weights, w1, w1_scales,
                                        w1_global_scales, w2, w2_scales, w2_global_scales,
                                        scratch_f32, out_f32, M, E, top_k, K, I,
                                        multiply_router_weight, output_ready);
  case DType::bf16:
    return dispatch_relu2_split<bf16_t>(q, hidden, topk_ids, topk_weights, w1, w1_scales,
                                        w1_global_scales, w2, w2_scales, w2_global_scales,
                                        scratch_f32, out_f32, M, E, top_k, K, I,
                                        multiply_router_weight, output_ready);
  }
  return {};
}

} // namespace quixicore::xpu::kernels
