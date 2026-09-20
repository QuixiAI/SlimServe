#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event nvfp4_gemv_sycl(sycl::queue& q, const void* w_packed,
                            const void* block_scales, float global_scale,
                            const void* x, void* y, std::size_t N, std::size_t K,
                            DType act_dt);

sycl::event nvfp4_gemm_sycl(sycl::queue &q, const void *w_packed, const void *block_scales,
                            float global_scale, const void *x, void *y, std::size_t M,
                            std::size_t N, std::size_t K, DType act_dt);

// Experimental decode-once kernel retained for controlled M-sweep benchmarks.
// The public route uses `nvfp4_gemm_sycl` because this regressed at M=1,4,8.
sycl::event nvfp4_gemm_mtiled_sycl(sycl::queue &q, const void *w_packed, const void *block_scales,
                                   float global_scale, const void *x, void *y, std::size_t M,
                                   std::size_t N, std::size_t K, DType act_dt);

}  // namespace quixicore::xpu::kernels
