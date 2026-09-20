// SiLU / swish, native SYCL variant. Elementwise, vectorized by default via the
// shared vec_unary helper.

#include "activations/silu/silu_kernel.hpp"

#include "common/vec_map.hpp"

namespace quixicore::xpu::kernels {

sycl::event silu_sycl(sycl::queue& q, const void* in, void* out, std::size_t n,
                      DType dt) {
  auto f = [](float x) { return detail::siluf(x); };
  switch (dt) {
    case DType::f32:
      return detail::vec_unary(q, static_cast<const float*>(in),
                               static_cast<float*>(out), n, f);
    case DType::f16:
      return detail::vec_unary(q, static_cast<const half_t*>(in),
                               static_cast<half_t*>(out), n, f);
    case DType::bf16:
      return detail::vec_unary(q, static_cast<const bf16_t*>(in),
                               static_cast<bf16_t*>(out), n, f);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
