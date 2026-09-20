// Per-row cross-entropy from logits, native SYCL. One work-group per row: reduce
// the row max and sum-of-exp (fp32), then loss = (max + log sumexp) -
// logits[target]. Same row-reduction shape as softmax.

#include "utils/utils_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

template <typename T>
sycl::event ce_typed(sycl::queue& q, const T* logits, const int* target,
                     float* loss, std::size_t rows, std::size_t vocab) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t r = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* row = logits + r * vocab;
    const auto grp = it.get_group();

    float pmax = -std::numeric_limits<float>::infinity();
    for (std::size_t j = lid; j < vocab; j += kRowThreads)
      pmax = sycl::max(pmax, static_cast<float>(row[j]));
    const float m = sycl::reduce_over_group(grp, pmax, sycl::maximum<float>());

    float psum = 0.0f;
    for (std::size_t j = lid; j < vocab; j += kRowThreads)
      psum += sycl::exp(static_cast<float>(row[j]) - m);
    const float sum = sycl::reduce_over_group(grp, psum, sycl::plus<float>());

    if (lid == 0) {
      const int t = target[r];
      const float lt = static_cast<float>(row[t]);
      loss[r] = (m + sycl::log(sum)) - lt;
    }
  });
}

}  // namespace

sycl::event cross_entropy_sycl(sycl::queue& q, const void* logits,
                               const int* target, float* loss, std::size_t rows,
                               std::size_t vocab, DType dt) {
  switch (dt) {
    case DType::f32:
      return ce_typed(q, static_cast<const float*>(logits), target, loss, rows, vocab);
    case DType::f16:
      return ce_typed(q, static_cast<const half_t*>(logits), target, loss, rows, vocab);
    case DType::bf16:
      return ce_typed(q, static_cast<const bf16_t*>(logits), target, loss, rows, vocab);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
