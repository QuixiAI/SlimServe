#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Bias corrections bc1 = 1 - beta1^step, bc2 = 1 - beta2^step are precomputed by
// the dispatch layer and passed in.
sycl::event adamw_sycl(sycl::queue& q, void* p, const void* g, void* m, void* v,
                       std::size_t n, float lr, float beta1, float beta2,
                       float eps, float weight_decay, float bc1, float bc2,
                       DType dt);

}  // namespace quixicore::xpu::kernels
