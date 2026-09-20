#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event selective_scan_sycl(sycl::queue& q, const void* u, const void* delta,
                                const void* A, const void* B, const void* C,
                                const void* D, void* y, std::size_t n_chan,
                                std::size_t seq, std::size_t state, DType dt);

}  // namespace quixicore::xpu::kernels
