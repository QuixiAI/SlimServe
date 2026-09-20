#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event quantize_int4_group_sycl(sycl::queue& q, const void* w,
                                     void* w_packed, void* scales, std::size_t N,
                                     std::size_t K, std::size_t group, DType dt);

}  // namespace quixicore::xpu::kernels
