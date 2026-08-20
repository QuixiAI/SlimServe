#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event rope_sycl(sycl::queue& q, const void* x, void* out,
                      std::size_t tokens, std::size_t n_heads,
                      std::size_t head_dim, float base, std::size_t pos0,
                      DType dt);

}  // namespace quixicore::xpu::kernels
