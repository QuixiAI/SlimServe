// LayerNorm over the last axis, native SYCL variant.
//
// One work-group per row. Two fp32 reductions (sum and sum of squares) give the
// row mean and variance in a single pass over the input, then rescale with the
// affine weight/bias. `bias` may be null.

#include "norms/norms_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

// V*sizeof(T) = 16-byte vector loads; adjacent threads read adjacent vectors
// (coalesced + wide). Same trick that lifted bf16/f16 rms_norm ~2x -- see
// perf/optimization_status.md 2026-07-06.
template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 2 ? 8 : 4;
}

template <typename T>
sycl::event layernorm_typed(sycl::queue& q, const T* x, const T* w, const T* b,
                            T* out, std::size_t rows, std::size_t dim,
                            float eps) {
  const bool has_bias = (b != nullptr);
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
    const Vec* xv = reinterpret_cast<const Vec*>(xr);

    float psum = 0.0f;
    float psumsq = 0.0f;
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec v = xv[vi];
#pragma unroll
      for (int k = 0; k < V; ++k) {
        const float f = static_cast<float>(v[k]);
        psum += f;
        psumsq += f * f;
      }
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      const float f = static_cast<float>(xr[i]);
      psum += f;
      psumsq += f * f;
    }
    const auto grp = it.get_group();
    const float sum = sycl::reduce_over_group(grp, psum, sycl::plus<float>());
    const float sumsq = sycl::reduce_over_group(grp, psumsq, sycl::plus<float>());

    const float inv_dim = 1.0f / static_cast<float>(dim);
    const float mean = sum * inv_dim;
    const float var = sumsq * inv_dim - mean * mean;
    const float inv = sycl::rsqrt(var + eps);

    const Vec* wv = reinterpret_cast<const Vec*>(w);
    const Vec* bv = reinterpret_cast<const Vec*>(b);
    Vec* ov = reinterpret_cast<Vec*>(outr);
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec xvec = xv[vi];
      const Vec wvec = wv[vi];
      const Vec bvec = has_bias ? bv[vi] : Vec{};
      Vec ovec;
#pragma unroll
      for (int k = 0; k < V; ++k) {
        float y = (static_cast<float>(xvec[k]) - mean) * inv *
                  static_cast<float>(wvec[k]);
        if (has_bias) y += static_cast<float>(bvec[k]);
        ovec[k] = static_cast<T>(y);
      }
      ov[vi] = ovec;
    }
    for (std::size_t i = tail0 + lid; i < dim; i += kRowThreads) {
      float y = (static_cast<float>(xr[i]) - mean) * inv * static_cast<float>(w[i]);
      if (has_bias) y += static_cast<float>(b[i]);
      outr[i] = static_cast<T>(y);
    }
  });
}

}  // namespace

sycl::event layernorm_sycl(sycl::queue& q, const void* x, const void* weight,
                           const void* bias, void* out, std::size_t rows,
                           std::size_t dim, float eps, DType dt) {
  switch (dt) {
    case DType::f32:
      return layernorm_typed(q, static_cast<const float*>(x),
                             static_cast<const float*>(weight),
                             static_cast<const float*>(bias),
                             static_cast<float*>(out), rows, dim, eps);
    case DType::f16:
      return layernorm_typed(q, static_cast<const half_t*>(x),
                             static_cast<const half_t*>(weight),
                             static_cast<const half_t*>(bias),
                             static_cast<half_t*>(out), rows, dim, eps);
    case DType::bf16:
      return layernorm_typed(q, static_cast<const bf16_t*>(x),
                             static_cast<const bf16_t*>(weight),
                             static_cast<const bf16_t*>(bias),
                             static_cast<bf16_t*>(out), rows, dim, eps);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
