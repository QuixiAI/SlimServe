#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event dropout_sycl(sycl::queue& q, const void* in, void* out,
                         std::size_t n, float p, std::uint32_t seed, DType dt);

sycl::event cross_entropy_sycl(sycl::queue& q, const void* logits,
                               const int* target, float* loss, std::size_t rows,
                               std::size_t vocab, DType dt);

sycl::event hadamard_sycl(sycl::queue& q, const void* in, void* out,
                          std::size_t rows, std::size_t n, DType dt);

}  // namespace quixicore::xpu::kernels
