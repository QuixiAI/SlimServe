#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Depthwise causal conv1d decode update. Shapes, the kernel<=8 envelope, and
// slot semantics are documented on ops::causal_conv1d_decode.
sycl::event causal_conv1d_decode_sycl(
    sycl::queue& q, void* conv_state, const void* x, const void* weight,
    const void* bias, const std::int32_t* indices, void* out, bool silu,
    std::size_t batch, std::size_t dim, std::size_t state_len,
    std::size_t kernel, std::size_t nslots, std::int64_t cs0, std::int64_t cs1,
    std::int64_t cs2, DType act_dt, DType state_dt);

// Varlen depthwise causal conv1d prefill (two chained kernels: output pass,
// then state write-back). Shapes and layout are documented on
// ops::causal_conv1d_prefill. The returned event is the state write-back's;
// it transitively covers the output pass.
sycl::event causal_conv1d_prefill_sycl(
    sycl::queue& q, void* conv_state, const void* x, const void* weight,
    const void* bias, const std::int32_t* cu_seqlens,
    const std::int32_t* indices, const bool* has_init, void* out, bool silu,
    std::size_t total_tokens, std::size_t batch, std::size_t dim,
    std::size_t state_len, std::size_t kernel, std::size_t nslots,
    std::int64_t xs0, std::int64_t xs1, std::int64_t os0, std::int64_t os1,
    std::int64_t cs0, std::int64_t cs1, std::int64_t cs2, DType act_dt,
    DType state_dt);

}  // namespace quixicore::xpu::kernels
