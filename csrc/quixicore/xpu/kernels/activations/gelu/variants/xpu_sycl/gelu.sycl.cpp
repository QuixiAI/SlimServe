// GELU activation, native SYCL variant (Battlemage / Xe2 and portable).
//
// Elementwise, memory-bound. This is the reference kernel for the XPU backend:
// it establishes the variant-entry -> dispatch -> ops-ABI pattern that every
// later operation copies. Compute is done in fp32 regardless of storage dtype.
//
// Not blindly ported from Metal: no threadgroup staging, no simdgroup tricks --
// a flat parallel_for is the right shape for an elementwise op on Xe. Launch
// geometry / vectorization tuning is a later perf lever, recorded in
// perf/optimization_status.md before any speedup is claimed.

#include "activations/gelu/gelu_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// sqrt(2/pi), used by the tanh approximation.
constexpr float kSqrt2OverPi = 0.7978845608028654f;
// 1/sqrt(2), used by the exact erf form.
constexpr float kInvSqrt2 = 0.7071067811865476f;

inline float gelu_scalar(float x, bool tanh_approx) {
  if (tanh_approx) {
    const float inner = kSqrt2OverPi * (x + 0.044715f * x * x * x);
    return 0.5f * x * (1.0f + sycl::tanh(inner));
  }
  return 0.5f * x * (1.0f + sycl::erf(x * kInvSqrt2));
}

// 16-byte coalesced vector loads (V*sizeof(T) = 16): each work-item reads one
// wide vector and adjacent items read adjacent vectors. This beat the earlier
// coalesced-grid-stride scalar mapping, especially for bf16/f16 where the scalar
// path was narrow-transaction-bound (perf/optimization_status.md 2026-07-06).
template <typename T>
constexpr int vec_width() {
  return sizeof(T) == 2 ? 8 : 4;
}

template <typename T>
sycl::event gelu_typed(sycl::queue& q, const T* in, T* out, std::size_t n,
                       bool tanh_approx) {
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
      for (int k = 0; k < V; ++k) {
        o[k] = static_cast<T>(gelu_scalar(static_cast<float>(v[k]), tanh_approx));
      }
      ov[vi] = o;
    } else {
      // single trailing work-item mops up the [tail0, n) remainder
      for (std::size_t i = tail0; i < n; ++i) {
        out[i] = static_cast<T>(gelu_scalar(static_cast<float>(in[i]), tanh_approx));
      }
    }
  });
}

}  // namespace

sycl::event gelu_sycl(sycl::queue& q, const void* in, void* out, std::size_t n,
                      DType dt, bool tanh_approx) {
  switch (dt) {
    case DType::f32:
      return gelu_typed(q, static_cast<const float*>(in),
                        static_cast<float*>(out), n, tanh_approx);
    case DType::f16:
      return gelu_typed(q, static_cast<const half_t*>(in),
                        static_cast<half_t*>(out), n, tanh_approx);
    case DType::bf16:
      return gelu_typed(q, static_cast<const bf16_t*>(in),
                        static_cast<bf16_t*>(out), n, tanh_approx);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
