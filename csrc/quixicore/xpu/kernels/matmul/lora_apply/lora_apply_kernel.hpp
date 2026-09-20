#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Batched-gather LoRA matvecs (see ops::lora_shrink / ops::lora_expand).
sycl::event lora_shrink_sycl(sycl::queue& q, const void* in, const void* w,
                             const std::int32_t* lora_idx, float* out,
                             std::size_t batch, std::size_t hidden,
                             std::size_t rank, std::size_t n_loras,
                             float scale, DType dt);

sycl::event lora_expand_sycl(sycl::queue& q, const float* in, const void* w,
                             const std::int32_t* lora_idx, void* out,
                             std::size_t batch, std::size_t rank,
                             std::size_t out_dim, std::size_t n_loras,
                             std::size_t out_offset, std::size_t out_stride,
                             int accumulate, DType dt);

}  // namespace quixicore::xpu::kernels
