#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event qgemv_int4_sycl(sycl::queue& q, const void* w_packed,
                            const void* scales, const void* x, void* y,
                            std::size_t N, std::size_t K, std::size_t group,
                            DType act_dt);

}  // namespace quixicore::xpu::kernels
