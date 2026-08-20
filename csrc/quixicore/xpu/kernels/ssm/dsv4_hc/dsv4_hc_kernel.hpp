#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// DeepSeek-V4 manifold hyper-connections, post stage (see ops::dsv4_hc_post).
sycl::event dsv4_hc_post_sycl(sycl::queue& q, const float* comb_res_mix,
                              const void* residual, const float* post_mix,
                              const void* x, void* out, std::size_t tokens,
                              std::size_t n_streams, std::size_t hidden,
                              DType dt);

}  // namespace quixicore::xpu::kernels
