// Dispatch layer for the optimizers family.

#include <cmath>

#include "quixicore/xpu/ops.hpp"

#include "optimizers/adamw/adamw_kernel.hpp"

namespace quixicore::xpu::ops {

void adamw(sycl::queue& q, void* p, const void* g, void* m, void* v,
           std::size_t n, float lr, float beta1, float beta2, float eps,
           float weight_decay, int step, DType dt, Variant variant,
           bool blocking) {
  (void)variant;  // native only
  const float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
  const float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
  sycl::event ev = kernels::adamw_sycl(q, p, g, m, v, n, lr, beta1, beta2, eps,
                                       weight_decay, bc1, bc2, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
