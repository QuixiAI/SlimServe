#pragma once

// Shared elementwise kernel helpers for the XPU backend.
//
// Every memory-bound elementwise op should build on these so it is vectorized by
// default: each work-item reads/writes one 16-byte sycl::vec (V*sizeof(T)=16, so
// V=8 for bf16/f16, V=4 for f32) and adjacent work-items touch adjacent vectors
// (coalesced + wide). This is the pattern that made bf16/f16 row kernels ~2x
// faster (perf/optimization_status.md 2026-07-06); new kernels start there.
//
// Compute is done in fp32: functors take/return float regardless of storage T.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels::detail {

template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 2 ? 8 : 4;
}

// out[i] = f(float(in[i])), i in [0, n). f: float -> float.
template <typename T, typename F>
sycl::event vec_unary(sycl::queue& q, const T* in, T* out, std::size_t n, F f) {
  constexpr int V = vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const std::size_t nvec = n / V;
  const std::size_t tail0 = nvec * V;
  const Vec* iv = reinterpret_cast<const Vec*>(in);
  Vec* ov = reinterpret_cast<Vec*>(out);
  return q.parallel_for(sycl::range<1>(nvec + (tail0 < n ? 1 : 0)),
                        [=](sycl::id<1> idx) {
    const std::size_t vi = idx[0];
    if (vi < nvec) {
      const Vec v = iv[vi];
      Vec o;
#pragma unroll
      for (int k = 0; k < V; ++k) o[k] = static_cast<T>(f(static_cast<float>(v[k])));
      ov[vi] = o;
    } else {
      for (std::size_t i = tail0; i < n; ++i)
        out[i] = static_cast<T>(f(static_cast<float>(in[i])));
    }
  });
}

// out[i] = f(float(a[i]), float(b[i])). f: (float, float) -> float.
template <typename T, typename F>
sycl::event vec_binary(sycl::queue& q, const T* a, const T* b, T* out,
                       std::size_t n, F f) {
  constexpr int V = vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const std::size_t nvec = n / V;
  const std::size_t tail0 = nvec * V;
  const Vec* av = reinterpret_cast<const Vec*>(a);
  const Vec* bv = reinterpret_cast<const Vec*>(b);
  Vec* ov = reinterpret_cast<Vec*>(out);
  return q.parallel_for(sycl::range<1>(nvec + (tail0 < n ? 1 : 0)),
                        [=](sycl::id<1> idx) {
    const std::size_t vi = idx[0];
    if (vi < nvec) {
      const Vec va = av[vi];
      const Vec vb = bv[vi];
      Vec o;
#pragma unroll
      for (int k = 0; k < V; ++k)
        o[k] = static_cast<T>(f(static_cast<float>(va[k]), static_cast<float>(vb[k])));
      ov[vi] = o;
    } else {
      for (std::size_t i = tail0; i < n; ++i)
        out[i] = static_cast<T>(f(static_cast<float>(a[i]), static_cast<float>(b[i])));
    }
  });
}

// Small fp32 activation primitives shared by several ops.
inline float sigmoidf(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline float siluf(float x) { return x * sigmoidf(x); }
inline float geluf(float x) {
  return 0.5f * x * (1.0f + sycl::erf(x * 0.7071067811865476f));
}
inline float reluf(float x) { return x > 0.0f ? x : 0.0f; }

}  // namespace quixicore::xpu::kernels::detail
