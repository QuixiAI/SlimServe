#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event gelu_backward_sycl(sycl::queue& q, const void* grad_out,
                               const void* x, void* grad_in, std::size_t n,
                               DType dt, bool tanh_approx);

}  // namespace quixicore::xpu::kernels
