#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Expert-sorted permute + inverse weighted reduce (see ops::moe_permute /
// ops::moe_unpermute_weighted_reduce).
sycl::event moe_permute_sycl(sycl::queue& q, const void* hidden,
                             const int* topk_ids, void* permuted,
                             std::int32_t* rows_per_expert,
                             std::int32_t* row_map, std::int32_t* cursors,
                             std::size_t n_tokens, std::size_t top_k,
                             std::size_t hidden_dim, std::size_t n_experts,
                             DType dt);

sycl::event moe_unpermute_weighted_reduce_sycl(
    sycl::queue& q, const void* permuted, const std::int32_t* row_map,
    const float* topk_weights, void* out, std::size_t n_tokens,
    std::size_t top_k, std::size_t hidden_dim, DType dt);

}  // namespace quixicore::xpu::kernels
