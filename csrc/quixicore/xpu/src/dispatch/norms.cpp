// Dispatch layer for the norms family (native SYCL only): routes the public ops
// ABI to the kernel and applies blocking.

#include "quixicore/xpu/ops.hpp"

#include "norms/norms_kernel.hpp"

namespace quixicore::xpu::ops {

void rms_norm(sycl::queue& q, const void* x, const void* weight, void* out,
              std::size_t rows, std::size_t dim, float eps, DType dt,
              Variant variant, bool blocking) {
  // No oneDNN RMSNorm primitive; every variant resolves to the native path.
  (void)variant;
  sycl::event ev = kernels::rms_norm_sycl(q, x, weight, out, rows, dim, eps, dt);
  if (blocking) ev.wait();
}

void fused_add_rms_norm(sycl::queue &q, const void *x, void *residual, const void *weight,
                        void *out, std::size_t rows, std::size_t dim, float eps, DType dt,
                        Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev =
      kernels::fused_add_rms_norm_sycl(q, x, residual, weight, out, rows, dim, eps, dt);
  if (blocking)
    ev.wait();
}

}  // namespace quixicore::xpu::ops
