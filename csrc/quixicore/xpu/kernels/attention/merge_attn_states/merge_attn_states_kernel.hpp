#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// LSE-weighted merge of two partial attention results (see
// ops::merge_attn_states).
sycl::event merge_attn_states_sycl(sycl::queue& q, const void* out_a,
                                   const float* lse_a, const void* out_b,
                                   const float* lse_b, void* out,
                                   float* lse_out, std::size_t rows,
                                   std::size_t d, DType dt);

}  // namespace quixicore::xpu::kernels
