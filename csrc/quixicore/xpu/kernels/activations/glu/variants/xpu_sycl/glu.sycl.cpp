// Gated linear unit variants, native SYCL. Input [rows, 2*d] (gate half then
// value half), output [rows, d]: out = act(gate) * value. One work-group per
// row; vectorized when d is a multiple of the vector width (else scalar, which
// also keeps the value-half pointer correctly aligned).

#include "activations/glu/glu_kernel.hpp"

#include "common/vec_map.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

// tanh GELU approximation (matches embeddinggemma.c ei_gelu_tanh); distinct from
// detail::geluf, which is the erf form used by glu mode=geglu.
inline float gelu_tanh(float x) {
  constexpr float a = 0.044715f;
  constexpr float s = 0.79788456080286535587989211986876f;
  return 0.5f * x * (1.0f + sycl::tanh(s * x * (1.0f + a * x * x)));
}

inline float act(float x, int mode) {
  switch (mode) {
    case 0:
      return detail::siluf(x);
    case 1:
      return detail::geluf(x);
    case 2:
      return detail::reluf(x);
    default:
      return detail::sigmoidf(x);
  }
}

template <typename T>
sycl::event glu_typed(sycl::queue& q, const T* x, T* out, std::size_t rows,
                      std::size_t d, int mode) {
  constexpr int V = detail::vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const std::size_t nvec = (d % V == 0) ? d / V : 0;  // 0 => fully scalar (alignment)
  const std::size_t tail0 = nvec * V;
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) {
    const std::size_t row = it.get_group(0);
    const std::size_t lid = it.get_local_id(0);
    const T* gate = x + row * 2 * d;
    const T* val = gate + d;
    T* outr = out + row * d;

    const Vec* gv = reinterpret_cast<const Vec*>(gate);
    const Vec* vv = reinterpret_cast<const Vec*>(val);
    Vec* ov = reinterpret_cast<Vec*>(outr);
    for (std::size_t vi = lid; vi < nvec; vi += kRowThreads) {
      const Vec g = gv[vi];
      const Vec v = vv[vi];
      Vec o;
#pragma unroll
      for (int k = 0; k < V; ++k)
        o[k] = static_cast<T>(act(static_cast<float>(g[k]), mode) *
                              static_cast<float>(v[k]));
      ov[vi] = o;
    }
    for (std::size_t i = tail0 + lid; i < d; i += kRowThreads) {
      outr[i] = static_cast<T>(act(static_cast<float>(gate[i]), mode) *
                               static_cast<float>(val[i]));
    }
  });
}

// GEGLU -> f16: gelu_tanh(gate) * value, converted to f16. One work-item per
// output element ([rows, d] range avoids the div/mod of a flat id); reads the
// two dt input halves, computes in fp32, stores f16. Mirrors the
// embeddinggemma.c launch_up_gate_gelu_half epilogue (following glus [gate,
// value] half order rather than the source [up, gate]; the product is identical).
template <typename T>
sycl::event glu_gelu_f16_typed(sycl::queue& q, const T* x, half_t* out,
                               std::size_t rows, std::size_t d) {
  return q.parallel_for(sycl::range<2>(rows, d), [=](sycl::id<2> idx) {
    const std::size_t row = idx[0];
    const std::size_t col = idx[1];
    const T* gate = x + row * 2 * d;
    const T* val = gate + d;
    out[row * d + col] = static_cast<half_t>(
        gelu_tanh(static_cast<float>(gate[col])) * static_cast<float>(val[col]));
  });
}

}  // namespace

sycl::event glu_sycl(sycl::queue& q, const void* x, void* out, std::size_t rows,
                     std::size_t d, DType dt, int mode) {
  switch (dt) {
    case DType::f32:
      return glu_typed(q, static_cast<const float*>(x), static_cast<float*>(out),
                       rows, d, mode);
    case DType::f16:
      return glu_typed(q, static_cast<const half_t*>(x), static_cast<half_t*>(out),
                       rows, d, mode);
    case DType::bf16:
      return glu_typed(q, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(out),
                       rows, d, mode);
  }
  return {};
}


sycl::event glu_gelu_f16_sycl(sycl::queue& q, const void* x, void* out,
                              std::size_t rows, std::size_t d, DType dt) {
  auto* o = static_cast<half_t*>(out);
  switch (dt) {
    case DType::f32:
      return glu_gelu_f16_typed(q, static_cast<const float*>(x), o, rows, d);
    case DType::f16:
      return glu_gelu_f16_typed(q, static_cast<const half_t*>(x), o, rows, d);
    case DType::bf16:
      return glu_gelu_f16_typed(q, static_cast<const bf16_t*>(x), o, rows, d);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
