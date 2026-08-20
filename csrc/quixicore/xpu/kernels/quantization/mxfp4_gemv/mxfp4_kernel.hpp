#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event mxfp4_gemv_sycl(sycl::queue& q, const void* w_packed,
                            const void* block_scales, const void* x, void* y,
                            std::size_t N, std::size_t K, DType act_dt);

}  // namespace quixicore::xpu::kernels
