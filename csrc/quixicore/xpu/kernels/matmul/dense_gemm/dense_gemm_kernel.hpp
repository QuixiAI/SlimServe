#pragma once

// Dense GEMM: C[M,N] = A[M,K] * B[K,N], all row-major, fp32 accumulation.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event dense_gemm_sycl(sycl::queue& q, const void* a, const void* b,
                            void* c, std::size_t M, std::size_t N, std::size_t K,
                            DType dt);

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
sycl::event dense_gemm_onednn(sycl::queue& q, const void* a, const void* b,
                              void* c, std::size_t M, std::size_t N,
                              std::size_t K, DType dt);
#endif

}  // namespace quixicore::xpu::kernels
