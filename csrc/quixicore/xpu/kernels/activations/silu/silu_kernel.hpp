#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event silu_sycl(sycl::queue& q, const void* in, void* out, std::size_t n,
                      DType dt);

}  // namespace quixicore::xpu::kernels
