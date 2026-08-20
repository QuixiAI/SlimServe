#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Multimodal / positioned rotary embedding (see ops::mrope and
// ops::rotary_positioned). sections == nullptr means one section covering
// the whole rotary space (plain positioned RoPE).
sycl::event mrope_sycl(sycl::queue& q, void* query, void* key,
                       const void* cos_sin_cache,
                       const std::int64_t* positions,
                       const std::int32_t* sections, std::size_t n_sections,
                       std::size_t tokens, std::size_t n_heads,
                       std::size_t n_kv_heads, std::size_t head_size,
                       std::size_t rot_dim, int neox, DType dt);

}  // namespace quixicore::xpu::kernels
