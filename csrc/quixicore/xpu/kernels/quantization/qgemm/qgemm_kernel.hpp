#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event qgemm_int8_sycl(sycl::queue& q, const void* a, const void* b,
                            const void* a_scale, const void* b_scale, void* c,
                            std::size_t M, std::size_t N, std::size_t K,
                            DType out_dt);

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
sycl::event qgemm_int8_onednn(sycl::queue& q, const void* a, const void* b,
                              const void* a_scale, const void* b_scale, void* c,
                              std::size_t M, std::size_t N, std::size_t K,
                              DType out_dt);
#endif

}  // namespace quixicore::xpu::kernels
