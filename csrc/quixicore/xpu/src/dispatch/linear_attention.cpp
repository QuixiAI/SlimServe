// Dispatch layer for the linear_attention family (native only).

#include "quixicore/xpu/ops.hpp"

#include "linear_attention/gated_delta_rule/gated_delta_rule_kernel.hpp"

#include "linear_attention/linear_attn/linear_attn_kernel.hpp"
#include "linear_attention/qwen_gdn_decode/qwen_gdn_kernel.hpp"

namespace quixicore::xpu::ops {

void linear_attn(sycl::queue& q, const void* Q, const void* K, const void* V,
                 void* O, std::size_t n_heads, std::size_t seq, std::size_t dim,
                 DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::linear_attn_sycl(q, Q, K, V, O, n_heads, seq, dim, dt);
  if (blocking) ev.wait();
}

void qwen_gdn_decode(sycl::queue &q, const void *projected_qkvz, const void *projected_ba,
                     void *conv_state, void *ssm_state, const void *conv_weight,
                     const void *conv_bias, const float *A_log, const void *dt_bias,
                     const int *state_indices, void *mixed_qkv, void *core_out, void *z_out,
                     std::size_t batch, std::size_t state_slots, bool conv_dim_first, DType act_dt,
                     DType state_dt, DType dt_bias_dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event event = kernels::qwen_gdn_decode_sycl(
      q, projected_qkvz, projected_ba, conv_state, ssm_state, conv_weight, conv_bias, A_log,
      dt_bias, state_indices, mixed_qkv, core_out, z_out, batch, state_slots, conv_dim_first,
      act_dt, state_dt, dt_bias_dt);
  if (blocking)
    event.wait();
}

void gdn_l2norm_qk(sycl::queue& q, void* qk, std::size_t tokens,
                   std::size_t heads, std::size_t dk, float eps, DType dt,
                   Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::gdn_l2norm_qk_sycl(q, qk, tokens, heads, dk, eps, dt);
  if (blocking) ev.wait();
}

void gated_delta_rule_varlen(
    sycl::queue& q, const void* Q, const void* K, const void* V,
    const float* b, const float* a, const float* A_log, const float* dt_bias,
    void* ssm_state, void* core_out, const std::int32_t* cu_seqlens,
    const std::int32_t* state_indices, const bool* has_initial_state,
    std::size_t batch, std::size_t Hk, std::size_t dk, std::size_t Hv,
    std::size_t dv, std::size_t nslots, DType act_dt, DType state_dt,
    Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::gated_delta_rule_varlen_sycl(
      q, Q, K, V, b, a, A_log, dt_bias, ssm_state, core_out, cu_seqlens,
      state_indices, has_initial_state, batch, Hk, dk, Hv, dv, nslots, act_dt,
      state_dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
