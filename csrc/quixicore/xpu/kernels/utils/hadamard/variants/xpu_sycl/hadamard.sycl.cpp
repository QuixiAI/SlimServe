// Fast Walsh-Hadamard transform (unnormalized), native SYCL. One work-group per
// row loads the row into SLM (fp32), runs log2(n) butterfly passes, writes back.
// Used by quantization rotations (QuIP#/QuaRot-style). n is a power of two.

#include "utils/utils_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kMaxN = 2048;
constexpr std::size_t kThreads = 256;

template <typename T>
sycl::event hadamard_typed(sycl::queue& q, const T* in, T* out,
                           std::size_t rows, std::size_t n) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kThreads),
                              sycl::range<1>(kThreads));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> sm(sycl::range<1>(kMaxN), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      const std::size_t r = it.get_group(0);
      const std::size_t lid = it.get_local_id(0);
      const T* row = in + r * n;
      T* orow = out + r * n;

      for (std::size_t i = lid; i < n; i += kThreads) sm[i] = static_cast<float>(row[i]);
      it.barrier(sycl::access::fence_space::local_space);

      const std::size_t npairs = n / 2;
      for (std::size_t s = 1; s < n; s <<= 1) {
        for (std::size_t p = lid; p < npairs; p += kThreads) {
          const std::size_t base = (p / s) * (2 * s) + (p % s);
          const float a = sm[base];
          const float b = sm[base + s];
          sm[base] = a + b;
          sm[base + s] = a - b;
        }
        it.barrier(sycl::access::fence_space::local_space);
      }

      for (std::size_t i = lid; i < n; i += kThreads) orow[i] = static_cast<T>(sm[i]);
    });
  });
}

}  // namespace

sycl::event hadamard_sycl(sycl::queue& q, const void* in, void* out,
                          std::size_t rows, std::size_t n, DType dt) {
  switch (dt) {
    case DType::f32:
      return hadamard_typed(q, static_cast<const float*>(in), static_cast<float*>(out), rows, n);
    case DType::f16:
      return hadamard_typed(q, static_cast<const half_t*>(in), static_cast<half_t*>(out), rows, n);
    case DType::bf16:
      return hadamard_typed(q, static_cast<const bf16_t*>(in), static_cast<bf16_t*>(out), rows, n);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
