#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// DeepSeek-style fp8 MQA indexer logits (see ops::mqa_logits).
sycl::event mqa_logits_sycl(sycl::queue& q, const std::uint8_t* q_fp8,
                            const std::uint8_t* kv_fp8, const float* kv_scales,
                            const float* head_weights, const std::int32_t* ks,
                            const std::int32_t* ke, float* logits,
                            std::size_t S, std::size_t H, std::size_t D,
                            std::size_t Skv);

}  // namespace quixicore::xpu::kernels
