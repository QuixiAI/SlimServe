#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Segmented per-expert GEMM with fused weight dequantization (see
// ops::moe_grouped_qgemm). fmt: 0 = w16 (act dtype), 1 = int4 group,
// 2 = nvfp4 (e2m1 + linear e4m3 16-block scales + per-expert global).
sycl::event moe_grouped_qgemm_sycl(sycl::queue& q, const void* A,
                                   const void* W, const void* scales,
                                   const float* global_scales, void* C,
                                   const std::int32_t* rows_per_expert,
                                   std::size_t M_total, std::size_t N,
                                   std::size_t K, std::size_t E,
                                   std::size_t group, int fmt, DType act_dt);

}  // namespace quixicore::xpu::kernels
