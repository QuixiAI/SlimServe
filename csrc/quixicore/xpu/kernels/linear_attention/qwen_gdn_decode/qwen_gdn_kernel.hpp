#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event qwen_gdn_decode_sycl(sycl::queue &q, const void *projected_qkvz,
                                 const void *projected_ba, void *conv_state, void *ssm_state,
                                 const void *conv_weight, const void *conv_bias, const float *A_log,
                                 const void *dt_bias, const int *state_indices, void *mixed_qkv,
                                 void *core_out, void *z_out, std::size_t batch,
                                 std::size_t state_slots, bool conv_dim_first, DType act_dt,
                                 DType state_dt, DType dt_bias_dt);

} // namespace quixicore::xpu::kernels
