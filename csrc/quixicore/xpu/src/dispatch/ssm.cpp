// Dispatch layer for the ssm family (native only — sequential scan).

#include "quixicore/xpu/ops.hpp"

#include "ssm/causal_conv1d/causal_conv1d_kernel.hpp"
#include "ssm/dsv4_hc/dsv4_hc_kernel.hpp"
#include "ssm/selective_scan/selective_scan_kernel.hpp"
#include "ssm/ssd/ssd_kernel.hpp"

namespace quixicore::xpu::ops {

void selective_scan(sycl::queue& q, const void* u, const void* delta,
                    const void* A, const void* B, const void* C, const void* D,
                    void* y, std::size_t n_chan, std::size_t seq,
                    std::size_t state, DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::selective_scan_sycl(q, u, delta, A, B, C, D, y,
                                                n_chan, seq, state, dt);
  if (blocking) ev.wait();
}

void ssd_decode(sycl::queue& q, void* state, const void* x, const float* dt_raw,
                const float* A, const void* B, const void* C, const float* D,
                const float* dt_bias, const std::int32_t* src_indices,
                const std::int32_t* dst_indices, void* out, bool dt_softplus,
                std::size_t batch, std::size_t nheads, std::size_t headdim,
                std::size_t dstate, std::size_t ngroups, std::size_t nslots,
                std::int64_t s0, std::int64_t s1, std::int64_t s2,
                std::int64_t s3, DType act_dt, DType state_dt, Variant variant,
                bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::ssd_decode_sycl(
      q, state, x, dt_raw, A, B, C, D, dt_bias, src_indices, dst_indices, out,
      dt_softplus, batch, nheads, headdim, dstate, ngroups, nslots, s0, s1, s2,
      s3, act_dt, state_dt);
  if (blocking) ev.wait();
}

void ssd_prefill(sycl::queue& q, const void* x, const float* dt_raw,
                 const float* A, const void* B, const void* C, const float* D,
                 const float* dt_bias, const std::int32_t* cu_seqlens,
                 const void* initial_states, void* out, void* varlen_states,
                 bool dt_softplus, float dt_lo, float dt_hi, std::size_t batch,
                 std::size_t nheads, std::size_t headdim, std::size_t dstate,
                 std::size_t ngroups, DType act_dt, DType state_dt,
                 Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::ssd_prefill_sycl(
      q, x, dt_raw, A, B, C, D, dt_bias, cu_seqlens, initial_states, out,
      varlen_states, dt_softplus, dt_lo, dt_hi, batch, nheads, headdim, dstate,
      ngroups, act_dt, state_dt);
  if (blocking) ev.wait();
}

void causal_conv1d_decode(sycl::queue& q, void* conv_state, const void* x,
                          const void* weight, const void* bias,
                          const std::int32_t* indices, void* out, bool silu,
                          std::size_t batch, std::size_t dim,
                          std::size_t state_len, std::size_t kernel,
                          std::size_t nslots, std::int64_t cs0,
                          std::int64_t cs1, std::int64_t cs2, DType act_dt,
                          DType state_dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::causal_conv1d_decode_sycl(
      q, conv_state, x, weight, bias, indices, out, silu, batch, dim, state_len,
      kernel, nslots, cs0, cs1, cs2, act_dt, state_dt);
  if (blocking) ev.wait();
}

void causal_conv1d_prefill(sycl::queue& q, void* conv_state, const void* x,
                           const void* weight, const void* bias,
                           const std::int32_t* cu_seqlens,
                           const std::int32_t* indices, const bool* has_init,
                           void* out, bool silu, std::size_t total_tokens,
                           std::size_t batch, std::size_t dim,
                           std::size_t state_len, std::size_t kernel,
                           std::size_t nslots, std::int64_t xs0,
                           std::int64_t xs1, std::int64_t os0, std::int64_t os1,
                           std::int64_t cs0, std::int64_t cs1, std::int64_t cs2,
                           DType act_dt, DType state_dt, Variant variant,
                           bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::causal_conv1d_prefill_sycl(
      q, conv_state, x, weight, bias, cu_seqlens, indices, has_init, out, silu,
      total_tokens, batch, dim, state_len, kernel, nslots, xs0, xs1, os0, os1,
      cs0, cs1, cs2, act_dt, state_dt);
  if (blocking) ev.wait();
}

void dsv4_hc_post(sycl::queue& q, const float* comb_res_mix,
                  const void* residual, const float* post_mix, const void* x,
                  void* out, std::size_t tokens, std::size_t n_streams,
                  std::size_t hidden, DType dt, Variant variant,
                  bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::dsv4_hc_post_sycl(q, comb_res_mix, residual,
                                              post_mix, x, out, tokens,
                                              n_streams, hidden, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
