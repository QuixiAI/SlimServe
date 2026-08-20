// Inverted dropout, native SYCL. Counter-based RNG (rng.hpp) gives a stateless
// deterministic mask per element; kept elements are scaled by 1/(1-p).

#include "utils/utils_kernel.hpp"

#include "common/rng.hpp"
#include "common/vec_map.hpp"

namespace quixicore::xpu::kernels {
namespace {

// Vectorized: each work-item handles V contiguous elements via 16-byte
// coalesced load/store (was scalar per-element: 144 GB/s). The per-element RNG
// index (base+k) is preserved, so the dropout mask is bit-identical.
template <typename T>
sycl::event dropout_typed(sycl::queue& q, const T* in, T* out, std::size_t n,
                          float p, std::uint32_t seed) {
  constexpr int V = detail::vec_width<T>();
  using Vec = sycl::vec<T, V>;
  const float scale = 1.0f / (1.0f - p);
  const std::size_t nvec = n / V;
  const std::size_t tail0 = nvec * V;
  const Vec* iv = reinterpret_cast<const Vec*>(in);
  Vec* ov = reinterpret_cast<Vec*>(out);
  return q.parallel_for(sycl::range<1>(nvec + (tail0 < n ? 1 : 0)), [=](sycl::id<1> idx) {
    const std::size_t vi = idx[0];
    if (vi < nvec) {
      const Vec cin = iv[vi];
      Vec cout;
      const std::size_t base = vi * V;
#pragma unroll
      for (int k = 0; k < V; ++k) {
        const float u = detail::uniform01(seed, base + k);
        cout[k] = (u < p) ? static_cast<T>(0.0f)
                          : static_cast<T>(static_cast<float>(cin[k]) * scale);
      }
      ov[vi] = cout;
    } else {
      for (std::size_t i = tail0; i < n; ++i) {
        const float u = detail::uniform01(seed, i);
        out[i] = (u < p) ? static_cast<T>(0.0f)
                         : static_cast<T>(static_cast<float>(in[i]) * scale);
      }
    }
  });
}

}  // namespace

sycl::event dropout_sycl(sycl::queue& q, const void* in, void* out,
                         std::size_t n, float p, std::uint32_t seed, DType dt) {
  switch (dt) {
    case DType::f32:
      return dropout_typed(q, static_cast<const float*>(in), static_cast<float*>(out), n, p, seed);
    case DType::f16:
      return dropout_typed(q, static_cast<const half_t*>(in), static_cast<half_t*>(out), n, p, seed);
    case DType::bf16:
      return dropout_typed(q, static_cast<const bf16_t*>(in), static_cast<bf16_t*>(out), n, p, seed);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
