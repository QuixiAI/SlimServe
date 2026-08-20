// Per-token symmetric int8 activation quantization, native SYCL. One work-group
// per row: reduce the row |max|, derive scale = max/127, then round each element.
// Produces the int8 activation + per-row scale consumed by qgemm_int8 (w8a8).

#include "quantization/act_quant/act_quant_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

template <typename T>
sycl::event aq_typed(sycl::queue& q, const T* x, signed char* qo, float* scale,
                     std::size_t rows, std::size_t dim) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t r = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* row = x + r * dim;
    signed char* orow = qo + r * dim;

    float amax = 0.0f;
    for (std::size_t j = lid; j < dim; j += kRowThreads)
      amax = sycl::max(amax, sycl::fabs(static_cast<float>(row[j])));
    amax = sycl::reduce_over_group(it.get_group(), amax, sycl::maximum<float>());

    const float s = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
    const float inv = 1.0f / s;
    if (lid == 0) scale[r] = s;
    for (std::size_t j = lid; j < dim; j += kRowThreads) {
      int v = static_cast<int>(sycl::round(static_cast<float>(row[j]) * inv));
      v = sycl::max(-127, sycl::min(127, v));
      orow[j] = static_cast<signed char>(v);
    }
  });
}

}  // namespace

sycl::event act_quant_int8_sycl(sycl::queue& q, const void* x, signed char* q_out,
                                float* scale, std::size_t rows, std::size_t dim,
                                DType dt) {
  switch (dt) {
    case DType::f32:
      return aq_typed(q, static_cast<const float*>(x), q_out, scale, rows, dim);
    case DType::f16:
      return aq_typed(q, static_cast<const half_t*>(x), q_out, scale, rows, dim);
    case DType::bf16:
      return aq_typed(q, static_cast<const bf16_t*>(x), q_out, scale, rows, dim);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
