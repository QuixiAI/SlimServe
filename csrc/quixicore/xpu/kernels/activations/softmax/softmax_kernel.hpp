#pragma once

// Internal variant entry points for softmax. Included by the dispatch layer and
// the benchmark harness via the ops target's private kernels/ include path.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event softmax_sycl(sycl::queue& q, const void* x, void* out,
                         std::size_t rows, std::size_t dim, DType dt);

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
sycl::event softmax_onednn(sycl::queue& q, const void* x, void* out,
                           std::size_t rows, std::size_t dim, DType dt);
#endif

}  // namespace quixicore::xpu::kernels
