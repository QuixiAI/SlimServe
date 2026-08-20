// Varlen gated delta rule (Gated DeltaNet / Qwen3.5-3.6 family), native SYCL
// — exact recurrence, prefill + decode in one call (contract
// gated_linear_attention / gdn_recurrence coverage beyond the shape-locked
// qwen_gdn_decode).
//
// Per (sequence, v head), sequentially over the sequence's tokens:
//   beta = sigmoid(b[t,hv])
//   g    = exp(-exp(A_log[hv]) * softplus(a[t,hv] + dt_bias[hv]))
//   S   *= g
//   kv_d = S[d,:] . k_t          (k, q pre-L2-normalized by gdn_l2norm_qk)
//   delta_d = (v[t,hv,d] - kv_d) * beta
//   S[d,:] += k_t * delta_d
//   out[t,hv,d] = S[d,:] . q_t
// One work-group per (seq, v head), one lane per v dim; each lane's state
// row S[d, 0..dk) lives in registers; q/k are staged per token in SLM. The
// GQA map is hv -> hk = hv / (Hv/Hk). State is read from state_indices[seq]
// (zero when has_initial_state says so or the slot is null) and written back
// in place after the last token. The exact per-token order (decay, then kv
// read, then rank-1 update, then output) matches the vLLM general kernel.
// The width-4 conv stage is NOT here: causal_conv1d_prefill/decode already
// cover it — token-major mixed_qkv is just strides (xs0=1, xs1=dim).
//
// Semantics from vllm-xpu-kernels csrc/xpu/gdn_attn/gated_delta_rule.hpp
// (Apache; translated). The chunked-DPAS pipeline is the deferred perf
// variant (>=2.5x gate, caller workspaces).

#include "linear_attention/gated_delta_rule/gated_delta_rule_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kMaxDk = 128;

inline float gdr_sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float gdr_softplus(float x) {
  return x <= 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
}

template <typename T>
class GdnL2NormKernel;
template <typename T, typename StateT>
class GatedDeltaRuleKernel;

template <typename T>
sycl::event l2norm_typed(sycl::queue& q, T* qk, std::size_t rows,
                         std::size_t dk, float eps) {
  return q.parallel_for<GdnL2NormKernel<T>>(
      sycl::range<1>(rows), [=](sycl::id<1> idx) {
        T* v = qk + idx[0] * dk;
        float ss = 0.0f;
        for (std::size_t i = 0; i < dk; ++i) {
          const float f = static_cast<float>(v[i]);
          ss += f * f;
        }
        const float inv = sycl::rsqrt(ss + eps);
        for (std::size_t i = 0; i < dk; ++i)
          v[i] = static_cast<T>(static_cast<float>(v[i]) * inv);
      });
}

template <typename T, typename StateT>
sycl::event gdr_typed(sycl::queue& q, const T* Q, const T* K, const T* V,
                      const float* b, const float* a, const float* A_log,
                      const float* dt_bias, StateT* ssm_state, T* core_out,
                      const std::int32_t* cu_seqlens,
                      const std::int32_t* state_indices,
                      const bool* has_initial_state, std::size_t batch,
                      std::size_t Hk, std::size_t dk, std::size_t Hv,
                      std::size_t dv, std::size_t nslots) {
  const std::size_t vpk = Hv / Hk;  // v heads per k head
  const sycl::nd_range<2> ndr(sycl::range<2>(batch * Hv, dv),
                              sycl::range<2>(1, dv));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> qs(sycl::range<1>(dk), h);
    sycl::local_accessor<float, 1> ks(sycl::range<1>(dk), h);
    h.parallel_for<GatedDeltaRuleKernel<T, StateT>>(
        ndr, [=](sycl::nd_item<2> it) {
          const std::size_t shv = it.get_group(0);
          const std::size_t s = shv / Hv;
          const std::size_t hv = shv % Hv;
          const std::size_t hk = hv / vpk;
          const std::size_t d = it.get_local_id(1);

          const std::int32_t slot = state_indices[s];
          const bool slot_ok =
              slot >= 0 && static_cast<std::size_t>(slot) < nslots;
          const std::size_t lo = static_cast<std::size_t>(cu_seqlens[s]);
          const std::size_t hi = static_cast<std::size_t>(cu_seqlens[s + 1]);

          float state[kMaxDk];
          StateT* srow =
              slot_ok ? ssm_state + ((static_cast<std::size_t>(slot) * Hv +
                                      hv) *
                                         dv +
                                     d) *
                                        dk
                      : nullptr;
          const bool use_init =
              slot_ok &&
              (has_initial_state == nullptr || has_initial_state[s]);
          for (std::size_t i = 0; i < dk; ++i)
            state[i] = use_init ? static_cast<float>(srow[i]) : 0.0f;

          const float neg_A = -sycl::exp(A_log[hv]);
          const float bias = dt_bias != nullptr ? dt_bias[hv] : 0.0f;

          for (std::size_t t = lo; t < hi; ++t) {
            // Stage this token's normalized q/k for the whole group.
            for (std::size_t i = d; i < dk; i += dv) {
              qs[i] = static_cast<float>(Q[(t * Hk + hk) * dk + i]);
              ks[i] = static_cast<float>(K[(t * Hk + hk) * dk + i]);
            }
            sycl::group_barrier(it.get_group());

            const float beta = gdr_sigmoid(b[t * Hv + hv]);
            const float g =
                sycl::exp(neg_A * gdr_softplus(a[t * Hv + hv] + bias));

            float kv = 0.0f;
            for (std::size_t i = 0; i < dk; ++i) {
              state[i] *= g;
              kv += state[i] * ks[i];
            }
            const float delta =
                (static_cast<float>(V[(t * Hv + hv) * dv + d]) - kv) * beta;
            float out = 0.0f;
            for (std::size_t i = 0; i < dk; ++i) {
              state[i] += ks[i] * delta;
              out += state[i] * qs[i];
            }
            if (slot_ok) {
              core_out[(t * Hv + hv) * dv + d] = static_cast<T>(out);
            } else {
              core_out[(t * Hv + hv) * dv + d] = T(0.0f);
            }
            sycl::group_barrier(it.get_group());
          }

          if (slot_ok && hi > lo) {
            for (std::size_t i = 0; i < dk; ++i)
              srow[i] = static_cast<StateT>(state[i]);
          }
        });
  });
}

}  // namespace

sycl::event gdn_l2norm_qk_sycl(sycl::queue& q, void* qk, std::size_t tokens,
                               std::size_t heads, std::size_t dk, float eps,
                               DType dt) {
  const std::size_t rows = tokens * heads;
  switch (dt) {
    case DType::f32:
      return l2norm_typed(q, static_cast<float*>(qk), rows, dk, eps);
    case DType::f16:
      return l2norm_typed(q, static_cast<half_t*>(qk), rows, dk, eps);
    case DType::bf16:
      return l2norm_typed(q, static_cast<bf16_t*>(qk), rows, dk, eps);
  }
  return {};
}

sycl::event gated_delta_rule_varlen_sycl(
    sycl::queue& q, const void* Q, const void* K, const void* V,
    const float* b, const float* a, const float* A_log, const float* dt_bias,
    void* ssm_state, void* core_out, const std::int32_t* cu_seqlens,
    const std::int32_t* state_indices, const bool* has_initial_state,
    std::size_t batch, std::size_t Hk, std::size_t dk, std::size_t Hv,
    std::size_t dv, std::size_t nslots, DType act_dt, DType state_dt) {
  if (dk > kMaxDk || Hv % Hk != 0) return {};  // reject, don't corrupt
#define QX_GDR(T, ST)                                                          \
  return gdr_typed<T, ST>(q, static_cast<const T*>(Q),                         \
                          static_cast<const T*>(K), static_cast<const T*>(V),  \
                          b, a, A_log, dt_bias, static_cast<ST*>(ssm_state),   \
                          static_cast<T*>(core_out), cu_seqlens,               \
                          state_indices, has_initial_state, batch, Hk, dk, Hv, \
                          dv, nslots)
#define QX_GDR_ST(T)          \
  switch (state_dt) {         \
    case DType::f32:          \
      QX_GDR(T, float);       \
    case DType::f16:          \
      QX_GDR(T, half_t);      \
    case DType::bf16:         \
      QX_GDR(T, bf16_t);      \
  }
  switch (act_dt) {
    case DType::f32:
      QX_GDR_ST(float);
      break;
    case DType::f16:
      QX_GDR_ST(half_t);
      break;
    case DType::bf16:
      QX_GDR_ST(bf16_t);
      break;
  }
#undef QX_GDR_ST
#undef QX_GDR
  return {};
}

}  // namespace quixicore::xpu::kernels
