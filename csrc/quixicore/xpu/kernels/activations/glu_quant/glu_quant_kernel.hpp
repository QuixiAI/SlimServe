#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Fused SwiGLU + activation quantization (see ops::glu_quant).
// mode: 0 = per-group dynamic fp8 (e4m3, fp32 scales), 1 = mxfp4.
sycl::event glu_quant_sycl(sycl::queue& q, const void* x, std::uint8_t* out_q,
                           float* out_scales, std::size_t rows, std::size_t d,
                           std::size_t group, int mode, DType dt);

}  // namespace quixicore::xpu::kernels
