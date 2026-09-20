#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event linear_attn_sycl(sycl::queue& q, const void* Q, const void* K,
                             const void* V, void* O, std::size_t n_heads,
                             std::size_t seq, std::size_t dim, DType dt);

}  // namespace quixicore::xpu::kernels
