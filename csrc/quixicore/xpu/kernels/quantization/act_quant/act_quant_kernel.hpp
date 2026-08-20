#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event act_quant_int8_sycl(sycl::queue& q, const void* x, signed char* q_out,
                                float* scale, std::size_t rows, std::size_t dim,
                                DType dt);

}  // namespace quixicore::xpu::kernels
