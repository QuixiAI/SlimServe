// Varlen depthwise causal conv1d prefill (Mamba-2 / NemotronH conv stage).
//
// Two kernels chained by an explicit event dependency:
//   (A) output: one work-item per (token, channel). kernel-tap causal sum over
//       the last `kernel` samples of [initial_state | seq_x] ending at the
//       token, drawing the earliest taps from the per-slot conv_state window
//       when in reach of the sequence start; optional SiLU. Never writes
//       conv_state.
//   (B) state write-back: one work-item per (seq, channel). Gathers the last
//       state_len samples of the effective sequence into registers FIRST (they
//       may alias the destination window), then writes conv_state[slot].
//       depends_on(A) so every read of the old window precedes these writes.
// Invalid (null) slots emit zero output and touch no state; empty sequences
// leave their slot untouched. fp32 math.
//
// Adapted from the QuixiAI MIT decode kernel set proven in vLLM XPU serving
// (vllm-xpu-kernels csrc/xpu/sycl/decode/mamba2_conv1d_prefill_kernel.hpp,
// commit dffcab7), re-homed on the QuixiCore DType/ops ABI.

#include "ssm/causal_conv1d/causal_conv1d_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

inline constexpr std::size_t kMaxPrefillTaps = 8;

inline float cp_silu(float x) { return x / (1.0f + sycl::exp(-x)); }

template <typename T, typename StateT>
class CausalConv1dPrefillOutKernel;
template <typename T, typename StateT>
class CausalConv1dPrefillStateKernel;

// Locate the sequence owning packed token `tok`: qsl[s] <= tok < qsl[s+1].
inline std::size_t cp_find_seq(const std::int32_t* qsl, std::size_t batch,
                               std::size_t tok) {
  for (std::size_t s = 0; s < batch; ++s) {
    if (tok < static_cast<std::size_t>(qsl[s + 1])) return s;
  }
  return batch - 1;
}

// conv_state: [nslots, dim, state_len] StateT, strides cs0..cs2 (written in B)
// x, out:     [dim, total_tokens] T, strides xs0/xs1 and os0/os1 (dim-major)
// weight:     [dim, kernel] T ; bias: [dim] T (nullptr = none)
// cu_seqlens: [batch+1] int32 ; indices: [batch] int32 (<0 => null)
// has_init:   [batch] bool (nullptr = none)
template <typename T, typename StateT>
sycl::event causal_conv1d_prefill_typed(
    sycl::queue& q, StateT* conv_state, const T* x, const T* weight,
    const T* bias, const std::int32_t* qsl, const std::int32_t* indices,
    const bool* has_init, T* out, bool silu, std::size_t total_tokens,
    std::size_t batch, std::size_t dim, std::size_t state_len,
    std::size_t kernel, std::size_t nslots, std::int64_t xs0, std::int64_t xs1,
    std::int64_t os0, std::int64_t os1, std::int64_t cs0, std::int64_t cs1,
    std::int64_t cs2) {
  constexpr std::size_t kLocal = 256;

  const std::size_t total_a = total_tokens * dim;
  const std::size_t nwg_a = (total_a + kLocal - 1) / kLocal;
  sycl::event ev_a = q.submit([&](sycl::handler& hnd) {
    hnd.parallel_for<CausalConv1dPrefillOutKernel<T, StateT>>(
        sycl::nd_range<1>(sycl::range<1>(nwg_a * kLocal),
                          sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          const std::size_t gid = it.get_global_id(0);
          if (gid >= total_a) return;
          const std::size_t tok = gid / dim;
          const std::size_t c = gid - tok * dim;

          const std::size_t s = cp_find_seq(qsl, batch, tok);
          const std::size_t bos = static_cast<std::size_t>(qsl[s]);
          const std::size_t t = tok - bos;
          const std::int32_t slot = indices[s];

          T* out_ptr = out + static_cast<std::int64_t>(c) * os0 +
                       static_cast<std::int64_t>(tok) * os1;
          if (slot < 0 || static_cast<std::size_t>(slot) >= nslots) {
            *out_ptr = T(0);
            return;
          }
          const bool use_init = has_init != nullptr && has_init[s];

          const T* wrow = weight + c * kernel;
          const StateT* srow = conv_state +
                               static_cast<std::int64_t>(slot) * cs0 +
                               static_cast<std::int64_t>(c) * cs1;

          float acc = bias != nullptr ? static_cast<float>(bias[c]) : 0.0f;
          for (std::size_t k = 0; k < kernel; ++k) {
            const std::int64_t src = static_cast<std::int64_t>(t) -
                                     static_cast<std::int64_t>(kernel - 1) +
                                     static_cast<std::int64_t>(k);
            float val;
            if (src >= 0) {
              val = static_cast<float>(
                  x[static_cast<std::int64_t>(c) * xs0 +
                    (static_cast<std::int64_t>(bos) + src) * xs1]);
            } else if (use_init) {
              const std::int64_t si = static_cast<std::int64_t>(state_len) + src;
              val = static_cast<float>(srow[si * cs2]);
            } else {
              val = 0.0f;
            }
            acc += val * static_cast<float>(wrow[k]);
          }
          if (silu) acc = cp_silu(acc);
          *out_ptr = static_cast<T>(acc);
        });
  });

  const std::size_t total_b = batch * dim;
  const std::size_t nwg_b = (total_b + kLocal - 1) / kLocal;
  return q.submit([&](sycl::handler& hnd) {
    hnd.depends_on(ev_a);
    hnd.parallel_for<CausalConv1dPrefillStateKernel<T, StateT>>(
        sycl::nd_range<1>(sycl::range<1>(nwg_b * kLocal),
                          sycl::range<1>(kLocal)),
        [=](sycl::nd_item<1> it) {
          const std::size_t gid = it.get_global_id(0);
          if (gid >= total_b) return;
          const std::size_t s = gid / dim;
          const std::size_t c = gid - s * dim;

          const std::int32_t slot = indices[s];
          if (slot < 0 || static_cast<std::size_t>(slot) >= nslots) return;
          const std::size_t bos = static_cast<std::size_t>(qsl[s]);
          const std::size_t eos = static_cast<std::size_t>(qsl[s + 1]);
          if (eos <= bos) return;
          const std::size_t seqlen = eos - bos;
          const bool use_init = has_init != nullptr && has_init[s];

          StateT* srow = conv_state +
                         static_cast<std::int64_t>(slot) * cs0 +
                         static_cast<std::int64_t>(c) * cs1;

          float win[kMaxPrefillTaps];
          for (std::size_t m = 0; m < state_len; ++m) {
            const std::int64_t src = static_cast<std::int64_t>(seqlen) -
                                     static_cast<std::int64_t>(state_len) +
                                     static_cast<std::int64_t>(m);
            if (src >= 0) {
              win[m] = static_cast<float>(
                  x[static_cast<std::int64_t>(c) * xs0 +
                    (static_cast<std::int64_t>(bos) + src) * xs1]);
            } else if (use_init) {
              const std::int64_t si = static_cast<std::int64_t>(state_len) + src;
              win[m] = static_cast<float>(srow[si * cs2]);
            } else {
              win[m] = 0.0f;
            }
          }
          for (std::size_t m = 0; m < state_len; ++m) {
            srow[static_cast<std::int64_t>(m) * cs2] =
                static_cast<StateT>(win[m]);
          }
        });
  });
}

}  // namespace

sycl::event causal_conv1d_prefill_sycl(
    sycl::queue& q, void* conv_state, const void* x, const void* weight,
    const void* bias, const std::int32_t* cu_seqlens,
    const std::int32_t* indices, const bool* has_init, void* out, bool silu,
    std::size_t total_tokens, std::size_t batch, std::size_t dim,
    std::size_t state_len, std::size_t kernel, std::size_t nslots,
    std::int64_t xs0, std::int64_t xs1, std::int64_t os0, std::int64_t os1,
    std::int64_t cs0, std::int64_t cs1, std::int64_t cs2, DType act_dt,
    DType state_dt) {
  if (kernel > kMaxPrefillTaps || kernel != state_len + 1) {
    return {};  // outside the register-window envelope: reject, don't corrupt
  }
#define QX_CP_DISP(T, ST)                                                      \
  return causal_conv1d_prefill_typed<T, ST>(                                   \
      q, static_cast<ST*>(conv_state), static_cast<const T*>(x),               \
      static_cast<const T*>(weight), static_cast<const T*>(bias), cu_seqlens,  \
      indices, has_init, static_cast<T*>(out), silu, total_tokens, batch, dim, \
      state_len, kernel, nslots, xs0, xs1, os0, os1, cs0, cs1, cs2)
#define QX_CP_BY_STATE(T)          \
  switch (state_dt) {              \
    case DType::f32:               \
      QX_CP_DISP(T, float);        \
    case DType::f16:               \
      QX_CP_DISP(T, half_t);       \
    case DType::bf16:              \
      QX_CP_DISP(T, bf16_t);       \
  }
  switch (act_dt) {
    case DType::f32:
      QX_CP_BY_STATE(float);
      break;
    case DType::f16:
      QX_CP_BY_STATE(half_t);
      break;
    case DType::bf16:
      QX_CP_BY_STATE(bf16_t);
      break;
  }
#undef QX_CP_BY_STATE
#undef QX_CP_DISP
  return {};
}

}  // namespace quixicore::xpu::kernels
