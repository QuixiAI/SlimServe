#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Top-k / top-p probability renormalization (see ops::top_k_renorm /
// ops::top_p_renorm).
sycl::event top_k_renorm_sycl(sycl::queue& q, const void* probs, void* out,
                              std::size_t rows, std::size_t vocab, int k,
                              DType dt);

sycl::event top_p_renorm_sycl(sycl::queue& q, const void* probs, void* out,
                              std::size_t rows, std::size_t vocab, float top_p,
                              DType dt);

}  // namespace quixicore::xpu::kernels
