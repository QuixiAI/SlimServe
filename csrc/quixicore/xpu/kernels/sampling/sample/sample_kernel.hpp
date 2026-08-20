#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event sample_categorical_sycl(sycl::queue& q, const void* logits, int* out,
                                    std::size_t rows, std::size_t vocab,
                                    float temperature, std::uint32_t seed, DType dt);

sycl::event top_k_sample_sycl(sycl::queue& q, const void* logits, int* out,
                              std::size_t rows, std::size_t vocab, int k,
                              float temperature, std::uint32_t seed, DType dt);

}  // namespace quixicore::xpu::kernels
