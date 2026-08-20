// GELU backward, native SYCL variant. grad_in = grad_out * gelu'(x), elementwise
// and vectorized via the shared vec_binary helper.

#include "activations/gelu_backward/gelu_backward_kernel.hpp"

#include "common/vec_map.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr float kSqrt2OverPi = 0.7978845608028654f;
constexpr float kInvSqrt2 = 0.7071067811865476f;
constexpr float kInvSqrt2Pi = 0.3989422804014327f;  // 1/sqrt(2*pi)

// d/dx GELU(x).
inline float gelu_grad(float x, bool tanh_approx) {
  if (tanh_approx) {
    const float u = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    const float t = sycl::tanh(u);
    const float du = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * du;
  }
  const float cdf = 0.5f * (1.0f + sycl::erf(x * kInvSqrt2));
  const float pdf = kInvSqrt2Pi * sycl::exp(-0.5f * x * x);
  return cdf + x * pdf;
}

}  // namespace

sycl::event gelu_backward_sycl(sycl::queue& q, const void* grad_out,
                               const void* x, void* grad_in, std::size_t n,
                               DType dt, bool tanh_approx) {
  auto f = [tanh_approx](float g, float xv) { return g * gelu_grad(xv, tanh_approx); };
  switch (dt) {
    case DType::f32:
      return detail::vec_binary(q, static_cast<const float*>(grad_out),
                                static_cast<const float*>(x),
                                static_cast<float*>(grad_in), n, f);
    case DType::f16:
      return detail::vec_binary(q, static_cast<const half_t*>(grad_out),
                                static_cast<const half_t*>(x),
                                static_cast<half_t*>(grad_in), n, f);
    case DType::bf16:
      return detail::vec_binary(q, static_cast<const bf16_t*>(grad_out),
                                static_cast<const bf16_t*>(x),
                                static_cast<bf16_t*>(grad_in), n, f);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
