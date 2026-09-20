// Dispatch layer for the utils family (native only).

#include "quixicore/xpu/ops.hpp"

#include "utils/utils_kernel.hpp"

namespace quixicore::xpu::ops {

void dropout(sycl::queue& q, const void* in, void* out, std::size_t n, float p,
             std::uint32_t seed, DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::dropout_sycl(q, in, out, n, p, seed, dt);
  if (blocking) ev.wait();
}

void cross_entropy(sycl::queue& q, const void* logits, const int* target,
                   float* loss, std::size_t rows, std::size_t vocab, DType dt,
                   Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::cross_entropy_sycl(q, logits, target, loss, rows, vocab, dt);
  if (blocking) ev.wait();
}

void hadamard(sycl::queue& q, const void* in, void* out, std::size_t rows,
              std::size_t n, DType dt, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::hadamard_sycl(q, in, out, rows, n, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
