#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event argmax_sycl(sycl::queue& q, const void* logits, int* out,
                        std::size_t rows, std::size_t vocab, DType dt);

}  // namespace quixicore::xpu::kernels
