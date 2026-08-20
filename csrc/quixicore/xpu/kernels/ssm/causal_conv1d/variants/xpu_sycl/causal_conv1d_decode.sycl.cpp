// Depthwise causal conv1d decode update (Mamba-2 / NemotronH conv stage).
//
// Per (batch, channel): a kernel-tap causal correlation over the rolling
// conv-state window, optional SiLU, then shift the window left by one and
// append the new sample. One work-item per (batch, channel); fp32 math:
//   window = [state[0], .., state[L-1], x]        L = state_len = kernel-1
//   out    = silu?(bias + sum_k window[k] * weight[c,k])
//   state <- window[1:]
// State is read+written in place at indices[b]; invalid (null) slots emit
// zeros and touch no state.
//
// Adapted from the QuixiAI MIT decode kernel set proven in vLLM XPU serving
// (vllm-xpu-kernels csrc/xpu/sycl/decode/mamba2_conv1d_decode_kernel.hpp,
// commit dffcab7), re-homed on the QuixiCore DType/ops ABI.

#include "ssm/causal_conv1d/causal_conv1d_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

inline constexpr std::size_t kMaxTaps = 8;  // conv kernel is 4 for Mamba-2

inline float conv1d_silu(float x) { return x / (1.0f + sycl::exp(-x)); }

template <typename T, typename StateT>
class CausalConv1dDecodeKernel;

// conv_state: [nslots, dim, state_len] StateT, element strides cs0..cs2
// x, out:     [batch, dim] T
// weight:     [dim, kernel] T ; bias: [dim] T (nullptr = none)
// indices:    [batch] int32 (cache slot per row; <0 => null)
template <typename T, typename StateT>
sycl::event causal_conv1d_decode_typed(
    sycl::queue& q, StateT* conv_state, const T* x, const T* weight,
    const T* bias, const std::int32_t* indices, T* out, bool silu,
    std::size_t batch, std::size_t dim, std::size_t state_len,
    std::size_t kernel, std::size_t nslots, std::int64_t cs0, std::int64_t cs1,
    std::int64_t cs2) {
  constexpr std::size_t kLocal = 256;
  const std::size_t total = batch * dim;
  const std::size_t nwg = (total + kLocal - 1) / kLocal;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kLocal),
                              sycl::range<1>(kLocal));
  return q.submit([&](sycl::handler& hnd) {
    hnd.parallel_for<CausalConv1dDecodeKernel<T, StateT>>(
        ndr, [=](sycl::nd_item<1> it) {
          const std::size_t gid = it.get_global_id(0);
          if (gid >= total) return;
          const std::size_t b = gid / dim;
          const std::size_t c = gid - b * dim;

          T* out_ptr = out + b * dim + c;
          const std::int32_t slot = indices[b];
          if (slot < 0 || static_cast<std::size_t>(slot) >= nslots) {
            *out_ptr = T(0);
            return;
          }

          StateT* srow = conv_state + static_cast<std::int64_t>(slot) * cs0 +
                         static_cast<std::int64_t>(c) * cs1;
          const T* wrow = weight + c * kernel;

          float window[kMaxTaps];
          for (std::size_t k = 0; k < state_len; ++k) {
            window[k] =
                static_cast<float>(srow[static_cast<std::int64_t>(k) * cs2]);
          }
          window[state_len] = static_cast<float>(x[b * dim + c]);

          float acc = bias != nullptr ? static_cast<float>(bias[c]) : 0.0f;
          for (std::size_t k = 0; k < kernel; ++k) {
            acc += window[k] * static_cast<float>(wrow[k]);
          }
          if (silu) acc = conv1d_silu(acc);
          *out_ptr = static_cast<T>(acc);

          for (std::size_t k = 0; k < state_len; ++k) {
            srow[static_cast<std::int64_t>(k) * cs2] =
                static_cast<StateT>(window[k + 1]);
          }
        });
  });
}

}  // namespace

sycl::event causal_conv1d_decode_sycl(
    sycl::queue& q, void* conv_state, const void* x, const void* weight,
    const void* bias, const std::int32_t* indices, void* out, bool silu,
    std::size_t batch, std::size_t dim, std::size_t state_len,
    std::size_t kernel, std::size_t nslots, std::int64_t cs0, std::int64_t cs1,
    std::int64_t cs2, DType act_dt, DType state_dt) {
  if (kernel > kMaxTaps || kernel != state_len + 1) {
    return {};  // outside the register-window envelope: reject, don't corrupt
  }
#define QX_C1_DISP(T, ST)                                                      \
  return causal_conv1d_decode_typed<T, ST>(                                    \
      q, static_cast<ST*>(conv_state), static_cast<const T*>(x),               \
      static_cast<const T*>(weight), static_cast<const T*>(bias), indices,     \
      static_cast<T*>(out), silu, batch, dim, state_len, kernel, nslots, cs0,  \
      cs1, cs2)
#define QX_C1_BY_STATE(T)          \
  switch (state_dt) {              \
    case DType::f32:               \
      QX_C1_DISP(T, float);        \
    case DType::f16:               \
      QX_C1_DISP(T, half_t);       \
    case DType::bf16:              \
      QX_C1_DISP(T, bf16_t);       \
  }
  switch (act_dt) {
    case DType::f32:
      QX_C1_BY_STATE(float);
      break;
    case DType::f16:
      QX_C1_BY_STATE(half_t);
      break;
    case DType::bf16:
      QX_C1_BY_STATE(bf16_t);
      break;
  }
#undef QX_C1_BY_STATE
#undef QX_C1_DISP
  return {};
}

}  // namespace quixicore::xpu::kernels
