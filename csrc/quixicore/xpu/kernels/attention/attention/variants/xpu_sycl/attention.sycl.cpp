// Flash-style scaled dot-product attention, native SYCL. One work-item per
// (head, query): it streams the keys with the online-softmax recurrence (running
// max m, running denom l, running weighted-value acc[d]) so the seq_k x seq_k
// score matrix is never materialized — the defining property of flash attention.
// GQA maps q head -> kv head. Causal masks keys past the query (end-aligned).
//
// This is the correctness-first shape: per-query registers hold acc[d] (d<=128).
// A tiled work-group/joint_matrix variant (SLM K/V tiles, subgroup dot) is the
// throughput optimization, deferred to the attention depth wave.

#include "attention/attention/attention_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxD = 128;

template <typename T>
sycl::event attn_typed(sycl::queue& q, const T* Q, const T* K, const T* V, T* O,
                       std::size_t n_heads, std::size_t n_kv_heads,
                       std::size_t seq_q, std::size_t seq_k, std::size_t d,
                       bool causal) {
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));
  const std::size_t gqa = n_heads / n_kv_heads;      // q heads per kv head
  const std::size_t delta = seq_k - seq_q;           // causal end-alignment
  return q.parallel_for(sycl::range<1>(n_heads * seq_q), [=](sycl::id<1> idx) {
    const std::size_t h = idx[0] / seq_q;
    const std::size_t qi = idx[0] % seq_q;
    const std::size_t kvh = h / gqa;
    const T* qrow = Q + (h * seq_q + qi) * d;
    const T* kbase = K + kvh * seq_k * d;
    const T* vbase = V + kvh * seq_k * d;

    float qreg[kMaxD];
    for (std::size_t j = 0; j < d; ++j) qreg[j] = static_cast<float>(qrow[j]);

    const std::size_t last = causal ? (qi + delta) : (seq_k - 1);
    float m = -std::numeric_limits<float>::infinity();
    float l = 0.0f;
    float acc[kMaxD];
    for (std::size_t j = 0; j < d; ++j) acc[j] = 0.0f;

    for (std::size_t ki = 0; ki <= last && ki < seq_k; ++ki) {
      const T* krow = kbase + ki * d;
      float score = 0.0f;
      for (std::size_t j = 0; j < d; ++j) score += qreg[j] * static_cast<float>(krow[j]);
      score *= scale;

      const float m_new = sycl::max(m, score);
      const float corr = sycl::exp(m - m_new);
      const float p = sycl::exp(score - m_new);
      l = l * corr + p;
      const T* vrow = vbase + ki * d;
      for (std::size_t j = 0; j < d; ++j) acc[j] = acc[j] * corr + p * static_cast<float>(vrow[j]);
      m = m_new;
    }

    T* orow = O + (h * seq_q + qi) * d;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (std::size_t j = 0; j < d; ++j) orow[j] = static_cast<T>(acc[j] * inv);
  });
}

}  // namespace

sycl::event attention_sycl(sycl::queue& q, const void* Q, const void* K,
                           const void* V, void* O, std::size_t n_heads,
                           std::size_t n_kv_heads, std::size_t seq_q,
                           std::size_t seq_k, std::size_t d, bool causal,
                           DType dt) {
  switch (dt) {
    case DType::f32:
      return attn_typed(q, static_cast<const float*>(Q), static_cast<const float*>(K),
                        static_cast<const float*>(V), static_cast<float*>(O),
                        n_heads, n_kv_heads, seq_q, seq_k, d, causal);
    case DType::f16:
      return attn_typed(q, static_cast<const half_t*>(Q), static_cast<const half_t*>(K),
                        static_cast<const half_t*>(V), static_cast<half_t*>(O),
                        n_heads, n_kv_heads, seq_q, seq_k, d, causal);
    case DType::bf16:
      return attn_typed(q, static_cast<const bf16_t*>(Q), static_cast<const bf16_t*>(K),
                        static_cast<const bf16_t*>(V), static_cast<bf16_t*>(O),
                        n_heads, n_kv_heads, seq_q, seq_k, d, causal);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
