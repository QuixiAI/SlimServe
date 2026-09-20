// Fused AdamW update, native SYCL. In-place on params/moments; compute in fp32.
// One work-item per element.

#include "optimizers/adamw/adamw_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

template <typename T>
sycl::event adamw_typed(sycl::queue& q, T* p, const T* g, T* m, T* v,
                        std::size_t n, float lr, float b1, float b2, float eps,
                        float wd, float bc1, float bc2) {
  return q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
    const std::size_t i = idx[0];
    const float grad = static_cast<float>(g[i]);
    const float mi = b1 * static_cast<float>(m[i]) + (1.0f - b1) * grad;
    const float vi = b2 * static_cast<float>(v[i]) + (1.0f - b2) * grad * grad;
    const float mhat = mi / bc1;
    const float vhat = vi / bc2;
    float pi = static_cast<float>(p[i]);
    pi -= lr * (mhat / (sycl::sqrt(vhat) + eps) + wd * pi);
    m[i] = static_cast<T>(mi);
    v[i] = static_cast<T>(vi);
    p[i] = static_cast<T>(pi);
  });
}

}  // namespace

sycl::event adamw_sycl(sycl::queue& q, void* p, const void* g, void* m, void* v,
                       std::size_t n, float lr, float b1, float b2, float eps,
                       float wd, float bc1, float bc2, DType dt) {
  switch (dt) {
    case DType::f32:
      return adamw_typed(q, static_cast<float*>(p), static_cast<const float*>(g),
                         static_cast<float*>(m), static_cast<float*>(v), n, lr, b1,
                         b2, eps, wd, bc1, bc2);
    case DType::f16:
      return adamw_typed(q, static_cast<half_t*>(p), static_cast<const half_t*>(g),
                         static_cast<half_t*>(m), static_cast<half_t*>(v), n, lr, b1,
                         b2, eps, wd, bc1, bc2);
    case DType::bf16:
      return adamw_typed(q, static_cast<bf16_t*>(p), static_cast<const bf16_t*>(g),
                         static_cast<bf16_t*>(m), static_cast<bf16_t*>(v), n, lr, b1,
                         b2, eps, wd, bc1, bc2);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
