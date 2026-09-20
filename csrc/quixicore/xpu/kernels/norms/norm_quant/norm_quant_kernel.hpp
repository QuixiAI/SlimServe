#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Fused RMSNorm + quantization (see ops::norm_quant for the contract).
// mode: 0 = static fp8 (e4m3, caller device-scalar scale), 1 = dynamic
// per-token fp8 (scale out per row), 2 = mxfp4 (per-32-group power-of-two
// fp32 scale + packed e2m1 nibbles).
sycl::event norm_quant_sycl(sycl::queue& q, const void* x, void* residual,
                            const void* weight, std::uint8_t* out_q,
                            const float* static_scale, float* out_scales,
                            std::size_t rows, std::size_t hidden, float eps,
                            int mode, DType dt);

}  // namespace quixicore::xpu::kernels
