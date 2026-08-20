#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// In-place L2 normalization of q/k head vectors (see ops::gdn_l2norm_qk).
sycl::event gdn_l2norm_qk_sycl(sycl::queue& q, void* qk, std::size_t tokens,
                               std::size_t heads, std::size_t dk, float eps,
                               DType dt);

// Varlen gated delta rule (exact recurrence; see
// ops::gated_delta_rule_varlen).
sycl::event gated_delta_rule_varlen_sycl(
    sycl::queue& q, const void* Q, const void* K, const void* V,
    const float* b, const float* a, const float* A_log, const float* dt_bias,
    void* ssm_state, void* core_out, const std::int32_t* cu_seqlens,
    const std::int32_t* state_indices, const bool* has_initial_state,
    std::size_t batch, std::size_t Hk, std::size_t dk, std::size_t Hv,
    std::size_t dv, std::size_t nslots, DType act_dt, DType state_dt);

}  // namespace quixicore::xpu::kernels
