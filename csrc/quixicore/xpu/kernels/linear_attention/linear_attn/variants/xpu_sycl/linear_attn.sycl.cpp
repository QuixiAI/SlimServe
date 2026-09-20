// Non-causal linear attention, native SYCL. One work-group per head. Phase 1
// builds the (dim x dim) KV state and the (dim) normalizer z in SLM by summing
// over the sequence; phase 2 applies O[t] = (Q[t] @ KV) / (Q[t] . z). This is
// the linear-attention identity O = Q (K^T V) with associativity exploited so
// cost is O(seq * dim^2) instead of O(seq^2 * dim). dim <= 64 for the SLM path.

#include "linear_attention/linear_attn/linear_attn_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kThreads = 256;
constexpr std::size_t kMaxDim = 64;

template <typename T>
sycl::event la_typed(sycl::queue& q, const T* Q, const T* K, const T* V, T* O,
                     std::size_t n_heads, std::size_t seq, std::size_t dim) {
  const sycl::nd_range<1> ndr(sycl::range<1>(n_heads * kThreads),
                              sycl::range<1>(kThreads));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> KV(sycl::range<1>(kMaxDim * kMaxDim), h);
    sycl::local_accessor<float, 1> z(sycl::range<1>(kMaxDim), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      const std::size_t head = it.get_group(0);
      const std::size_t lid = it.get_local_id(0);
      const std::size_t base = head * seq * dim;

      // Phase 1: KV[i,j] = sum_t K[t,i]*V[t,j]; z[i] = sum_t K[t,i].
      for (std::size_t e = lid; e < dim * dim; e += kThreads) {
        const std::size_t i = e / dim, j = e % dim;
        float acc = 0.0f;
        for (std::size_t t = 0; t < seq; ++t)
          acc += static_cast<float>(K[base + t * dim + i]) *
                 static_cast<float>(V[base + t * dim + j]);
        KV[e] = acc;
      }
      for (std::size_t i = lid; i < dim; i += kThreads) {
        float acc = 0.0f;
        for (std::size_t t = 0; t < seq; ++t) acc += static_cast<float>(K[base + t * dim + i]);
        z[i] = acc;
      }
      it.barrier(sycl::access::fence_space::local_space);

      // Phase 2: O[t,j] = (sum_i Q[t,i]*KV[i,j]) / (sum_i Q[t,i]*z[i] + eps).
      for (std::size_t e = lid; e < seq * dim; e += kThreads) {
        const std::size_t t = e / dim, j = e % dim;
        float num = 0.0f, den = 0.0f;
        for (std::size_t i = 0; i < dim; ++i) {
          const float qi = static_cast<float>(Q[base + t * dim + i]);
          num += qi * KV[i * dim + j];
          den += qi * z[i];
        }
        O[base + t * dim + j] = static_cast<T>(num / (den + 1e-6f));
      }
    });
  });
}

}  // namespace

sycl::event linear_attn_sycl(sycl::queue& q, const void* Q, const void* K,
                             const void* V, void* O, std::size_t n_heads,
                             std::size_t seq, std::size_t dim, DType dt) {
  switch (dt) {
    case DType::f32:
      return la_typed(q, static_cast<const float*>(Q), static_cast<const float*>(K),
                      static_cast<const float*>(V), static_cast<float*>(O), n_heads, seq, dim);
    case DType::f16:
      return la_typed(q, static_cast<const half_t*>(Q), static_cast<const half_t*>(K),
                      static_cast<const half_t*>(V), static_cast<half_t*>(O), n_heads, seq, dim);
    case DType::bf16:
      return la_typed(q, static_cast<const bf16_t*>(Q), static_cast<const bf16_t*>(K),
                      static_cast<const bf16_t*>(V), static_cast<bf16_t*>(O), n_heads, seq, dim);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
