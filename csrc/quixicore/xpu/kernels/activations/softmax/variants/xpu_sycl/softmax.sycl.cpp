// Numerically stable softmax over the last axis, native SYCL variant.
//
// One work-group per row, two fp32 reductions: row max (for stability), then the
// sum of exp(x - max). A final pass writes exp(x - max) / sum. Deterministic
// within a fixed launch geometry.

#include <limits>

#include "activations/softmax/softmax_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

// 16-byte coalesced vector loads (see perf/optimization_status.md 2026-07-06):
// the wide-load trick that lifted bf16/f16 norms ~2x applies here too.
template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 2 ? 8 : 4;
}

template <typename T>
sycl::event softmax_typed(sycl::queue& q, const T* x, T* out, std::size_t rows,
                          std::size_t dim) {
  constexpr int V = vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const std::size_t nvec = dim / V;
  const std::size_t tail0 = nvec * V;
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t row = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* xr = x + row * dim;
    T* outr = out + row * dim;
    const auto grp = it.get_group();
    const Vec* xv = reinterpret_cast<const Vec*>(xr);

    float pmax = -std::numeric_limits<float>::infinity();
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec v = xv[vi];
#pragma unroll
      for (int k = 0; k < V; ++k) pmax = sycl::max(pmax, static_cast<float>(v[k]));
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      pmax = sycl::max(pmax, static_cast<float>(xr[i]));
    }
    const float row_max =
        sycl::reduce_over_group(grp, pmax, sycl::maximum<float>());

    float psum = 0.0f;
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec v = xv[vi];
#pragma unroll
      for (int k = 0; k < V; ++k) psum += sycl::exp(static_cast<float>(v[k]) - row_max);
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      psum += sycl::exp(static_cast<float>(xr[i]) - row_max);
    }
    const float row_sum =
        sycl::reduce_over_group(grp, psum, sycl::plus<float>());
    const float inv = 1.0f / row_sum;

    Vec* ov = reinterpret_cast<Vec*>(outr);
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec v = xv[vi];
      Vec ovec;
#pragma unroll
      for (int k = 0; k < V; ++k) {
        ovec[k] = static_cast<T>(sycl::exp(static_cast<float>(v[k]) - row_max) * inv);
      }
      ov[vi] = ovec;
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      outr[i] = static_cast<T>(sycl::exp(static_cast<float>(xr[i]) - row_max) * inv);
    }
  });
}

}  // namespace

sycl::event softmax_sycl(sycl::queue& q, const void* x, void* out,
                         std::size_t rows, std::size_t dim, DType dt) {
  switch (dt) {
    case DType::f32:
      return softmax_typed(q, static_cast<const float*>(x),
                           static_cast<float*>(out), rows, dim);
    case DType::f16:
      return softmax_typed(q, static_cast<const half_t*>(x),
                           static_cast<half_t*>(out), rows, dim);
    case DType::bf16:
      return softmax_typed(q, static_cast<const bf16_t*>(x),
                           static_cast<bf16_t*>(out), rows, dim);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
