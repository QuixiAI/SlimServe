// Qwen3.5/Qwen3.6 non-interleaved Gated DeltaNet decode state update.

#include "linear_attention/qwen_gdn_decode/qwen_gdn_kernel.hpp"

#include <cstdint>

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kQueryHeads = 16;
constexpr std::size_t kValueHeads = 32;
constexpr std::size_t kHeadDim = 128;
constexpr std::size_t kQueryDim = kQueryHeads * kHeadDim;
constexpr std::size_t kKeyDim = kQueryHeads * kHeadDim;
constexpr std::size_t kValueDim = kValueHeads * kHeadDim;
constexpr std::size_t kConvDim = kQueryDim + kKeyDim + kValueDim;
constexpr std::size_t kQkvzDim = kConvDim + kValueDim;
constexpr std::size_t kBaDim = 2 * kValueHeads;

inline float silu(float value) {
  return value / (1.0f + sycl::exp(-value));
}

inline float softplus(float value) {
  return value <= 20.0f ? sycl::log(1.0f + sycl::exp(value)) : value;
}

template <typename T> class QwenGdnConvKernel;

template <typename T, typename StateT, typename BiasT> class QwenGdnRecurrentKernel;

template <typename T>
sycl::event conv_typed(sycl::queue &q, const T *projected_qkvz, const T *conv_weight,
                       const T *conv_bias, const int *state_indices, T *conv_state,
                       bool conv_dim_first, T *mixed_qkv, T *z_out, std::size_t batch,
                       std::size_t state_slots) {
  const std::size_t elements = batch * kConvDim;
  const sycl::nd_range<1> range(sycl::range<1>(((elements + 255) / 256) * 256),
                                sycl::range<1>(256));
  return q.parallel_for<QwenGdnConvKernel<T>>(range, [=](sycl::nd_item<1> item) {
    const std::size_t index = item.get_global_id(0);
    if (index >= elements)
      return;
    const std::size_t batch_index = index / kConvDim;
    const std::size_t channel = index % kConvDim;
    if (channel >= kConvDim - kValueDim) {
      const std::size_t value_channel = channel - (kConvDim - kValueDim);
      z_out[batch_index * kValueDim + value_channel] =
          projected_qkvz[batch_index * kQkvzDim + kConvDim + value_channel];
    }
    const int state_index = state_indices[batch_index];
    if (state_index < 0 || static_cast<std::size_t>(state_index) >= state_slots) {
      mixed_qkv[index] = T(0);
      return;
    }

    const T input = projected_qkvz[batch_index * kQkvzDim + channel];
    T *state = conv_state + static_cast<std::size_t>(state_index) * 3 * kConvDim;
    const std::size_t state_offset = conv_dim_first ? channel * 3 : channel;
    const std::size_t state_stride = conv_dim_first ? 1 : kConvDim;
    const float state0 = static_cast<float>(state[state_offset]);
    const float state1 = static_cast<float>(state[state_offset + state_stride]);
    const float state2 = static_cast<float>(state[state_offset + 2 * state_stride]);

    state[state_offset] = static_cast<T>(state1);
    state[state_offset + state_stride] = static_cast<T>(state2);
    state[state_offset + 2 * state_stride] = input;

    const T *weight = conv_weight + channel * 4;
    float value = conv_bias == nullptr ? 0.0f : static_cast<float>(conv_bias[channel]);
    value += state0 * static_cast<float>(weight[0]);
    value += state1 * static_cast<float>(weight[1]);
    value += state2 * static_cast<float>(weight[2]);
    value += static_cast<float>(input) * static_cast<float>(weight[3]);
    mixed_qkv[index] = static_cast<T>(silu(value));

  });
}

template <typename T, typename StateT, typename BiasT>
sycl::event recurrent_typed(sycl::queue &q, const T *mixed_qkv, const T *projected_ba,
                            const float *A_log, const BiasT *dt_bias, const int *state_indices,
                            StateT *ssm_state, T *core_out, std::size_t batch,
                            std::size_t state_slots, const sycl::event &conv_ready) {
  const sycl::nd_range<2> range(sycl::range<2>(batch * kValueHeads, kHeadDim),
                                sycl::range<2>(1, kHeadDim));
  return q.submit([&](sycl::handler &handler) {
    handler.depends_on(conv_ready);
    sycl::local_accessor<float, 1> query(sycl::range<1>(kHeadDim), handler);
    sycl::local_accessor<float, 1> key(sycl::range<1>(kHeadDim), handler);
    handler.parallel_for<QwenGdnRecurrentKernel<T, StateT, BiasT>>(
        range, [=](sycl::nd_item<2> item) {
          const std::size_t packed_head = item.get_global_id(0);
          const std::size_t lane = item.get_local_id(1);
          const std::size_t batch_index = packed_head / kValueHeads;
          const std::size_t value_head = packed_head % kValueHeads;
          const std::size_t query_head = value_head / (kValueHeads / kQueryHeads);
          const int state_index = state_indices[batch_index];
          if (state_index < 0 || static_cast<std::size_t>(state_index) >= state_slots) {
            core_out[packed_head * kHeadDim + lane] = T(0);
            return;
          }

          const T *query_row = mixed_qkv + batch_index * kConvDim + query_head * kHeadDim;
          const T *key_row = mixed_qkv + batch_index * kConvDim + kQueryDim + query_head * kHeadDim;
          const T *value_row =
              mixed_qkv + batch_index * kConvDim + kQueryDim + kKeyDim + value_head * kHeadDim;
          StateT *state_row =
              ssm_state +
              (((static_cast<std::size_t>(state_index) * kValueHeads + value_head) * kHeadDim +
                lane) *
               kHeadDim);

          const sycl::group<2> group = item.get_group();
          const float query_lane = static_cast<float>(query_row[lane]);
          const float key_lane = static_cast<float>(key_row[lane]);
          const float query_norm =
              sycl::reduce_over_group(group, query_lane * query_lane, sycl::plus<float>()) + 1e-6f;
          const float key_norm =
              sycl::reduce_over_group(group, key_lane * key_lane, sycl::plus<float>()) + 1e-6f;
          query[lane] = query_lane * sycl::rsqrt(query_norm) * 0.08838834764831845f;
          key[lane] = key_lane * sycl::rsqrt(key_norm);
          sycl::group_barrier(group);

          const float a =
              static_cast<float>(projected_ba[batch_index * kBaDim + kValueHeads + value_head]);
          const float b = static_cast<float>(projected_ba[batch_index * kBaDim + value_head]);
          const float log_decay =
              -sycl::exp(A_log[value_head]) * softplus(a + static_cast<float>(dt_bias[value_head]));
          const float decay = sycl::exp(log_decay);
          const float beta = 1.0f / (1.0f + sycl::exp(-b));

          float prediction = 0.0f;
          for (std::size_t k = 0; k < kHeadDim; ++k) {
            prediction += static_cast<float>(state_row[k]) * decay * key[k];
          }
          const float delta = (static_cast<float>(value_row[lane]) - prediction) * beta;
          float output = 0.0f;
          for (std::size_t k = 0; k < kHeadDim; ++k) {
            const float updated = static_cast<float>(state_row[k]) * decay + delta * key[k];
            state_row[k] = static_cast<StateT>(updated);
            output += updated * query[k];
          }
          core_out[packed_head * kHeadDim + lane] = static_cast<T>(output);
        });
  });
}

template <typename T, typename StateT>
sycl::event dispatch_bias(sycl::queue &q, const T *mixed_qkv, const T *projected_ba,
                          const float *A_log, const void *dt_bias, const int *state_indices,
                          StateT *ssm_state, T *core_out, std::size_t batch,
                          std::size_t state_slots, DType dt_bias_dt,
                          const sycl::event &conv_ready) {
  switch (dt_bias_dt) {
  case DType::f32:
    return recurrent_typed(q, mixed_qkv, projected_ba, A_log, static_cast<const float *>(dt_bias),
                           state_indices, ssm_state, core_out, batch, state_slots, conv_ready);
  case DType::f16:
    return recurrent_typed(q, mixed_qkv, projected_ba, A_log, static_cast<const half_t *>(dt_bias),
                           state_indices, ssm_state, core_out, batch, state_slots, conv_ready);
  case DType::bf16:
    return recurrent_typed(q, mixed_qkv, projected_ba, A_log, static_cast<const bf16_t *>(dt_bias),
                           state_indices, ssm_state, core_out, batch, state_slots, conv_ready);
  }
  return {};
}

template <typename T>
sycl::event dispatch_state(sycl::queue &q, const T *mixed_qkv, const T *projected_ba,
                           const float *A_log, const void *dt_bias, const int *state_indices,
                           void *ssm_state, T *core_out, std::size_t batch, std::size_t state_slots,
                           DType state_dt, DType dt_bias_dt, const sycl::event &conv_ready) {
  switch (state_dt) {
  case DType::f32:
    return dispatch_bias(q, mixed_qkv, projected_ba, A_log, dt_bias, state_indices,
                         static_cast<float *>(ssm_state), core_out, batch, state_slots, dt_bias_dt,
                         conv_ready);
  case DType::f16:
    return dispatch_bias(q, mixed_qkv, projected_ba, A_log, dt_bias, state_indices,
                         static_cast<half_t *>(ssm_state), core_out, batch, state_slots, dt_bias_dt,
                         conv_ready);
  case DType::bf16:
    return dispatch_bias(q, mixed_qkv, projected_ba, A_log, dt_bias, state_indices,
                         static_cast<bf16_t *>(ssm_state), core_out, batch, state_slots, dt_bias_dt,
                         conv_ready);
  }
  return {};
}

template <typename T>
sycl::event dispatch_activation(sycl::queue &q, const void *projected_qkvz,
                                const void *projected_ba, void *conv_state, void *ssm_state,
                                const void *conv_weight, const void *conv_bias, const float *A_log,
                                const void *dt_bias, const int *state_indices, void *mixed_qkv,
                                void *core_out, void *z_out, std::size_t batch,
                                std::size_t state_slots, bool conv_dim_first, DType state_dt,
                                DType dt_bias_dt) {
  const sycl::event conv_ready = conv_typed(
      q, static_cast<const T *>(projected_qkvz), static_cast<const T *>(conv_weight),
      static_cast<const T *>(conv_bias), state_indices, static_cast<T *>(conv_state),
      conv_dim_first, static_cast<T *>(mixed_qkv), static_cast<T *>(z_out), batch, state_slots);
  return dispatch_state(q, static_cast<const T *>(mixed_qkv), static_cast<const T *>(projected_ba),
                        A_log, dt_bias, state_indices, ssm_state, static_cast<T *>(core_out), batch,
                        state_slots, state_dt, dt_bias_dt, conv_ready);
}

} // namespace

sycl::event qwen_gdn_decode_sycl(sycl::queue &q, const void *projected_qkvz,
                                 const void *projected_ba, void *conv_state, void *ssm_state,
                                 const void *conv_weight, const void *conv_bias, const float *A_log,
                                 const void *dt_bias, const int *state_indices, void *mixed_qkv,
                                 void *core_out, void *z_out, std::size_t batch,
                                 std::size_t state_slots, bool conv_dim_first, DType act_dt,
                                 DType state_dt, DType dt_bias_dt) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_activation<float>(q, projected_qkvz, projected_ba, conv_state, ssm_state,
                                      conv_weight, conv_bias, A_log, dt_bias, state_indices,
                                      mixed_qkv, core_out, z_out, batch, state_slots,
                                      conv_dim_first, state_dt, dt_bias_dt);
  case DType::f16:
    return dispatch_activation<half_t>(q, projected_qkvz, projected_ba, conv_state, ssm_state,
                                       conv_weight, conv_bias, A_log, dt_bias, state_indices,
                                       mixed_qkv, core_out, z_out, batch, state_slots,
                                       conv_dim_first, state_dt, dt_bias_dt);
  case DType::bf16:
    return dispatch_activation<bf16_t>(q, projected_qkvz, projected_ba, conv_state, ssm_state,
                                       conv_weight, conv_bias, A_log, dt_bias, state_indices,
                                       mixed_qkv, core_out, z_out, batch, state_slots,
                                       conv_dim_first, state_dt, dt_bias_dt);
  }
  return {};
}

} // namespace quixicore::xpu::kernels
