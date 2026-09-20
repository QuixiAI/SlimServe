#pragma once

// Internal variant entry points for the GELU op. Not part of the public API;
// included by the dispatch layer and the benchmark harness (via the ops target's
// private `kernels/` include path). Each entry submits the kernel and returns
// its sycl::event without waiting, so callers own synchronization and the perf
// harness can read profiling timestamps.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Native SYCL GELU. Always available.
sycl::event gelu_sycl(sycl::queue& q, const void* in, void* out, std::size_t n,
                      DType dt, bool tanh_approx);

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
// Vendor GELU via oneDNN eltwise, executed on the caller's queue through SYCL
// interop. Compiled only when oneDNN is found at configure time.
sycl::event gelu_onednn(sycl::queue& q, const void* in, void* out, std::size_t n,
                        DType dt, bool tanh_approx);
#endif

}  // namespace quixicore::xpu::kernels
