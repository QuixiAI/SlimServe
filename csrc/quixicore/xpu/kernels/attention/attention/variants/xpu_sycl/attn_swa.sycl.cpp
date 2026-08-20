// Symmetric sliding-window attention (SWA), flash-style native SYCL. Same
// online-softmax recurrence and per-(head, query) work-item layout as
// attention_sycl (running max m, running denom l, running weighted-value
// acc[d]; the seq_k x seq_k score matrix is never materialized), but instead of
// a causal prefix each query attends a SYMMETRIC band of keys centered on its
// own position:
//
//   keys ki in [center - window/2, center + window/2]  (clamped to [0, seq_k)),
//
// where center = qi + (seq_k - seq_q) end-aligns the query into the key axis
// (center == qi for self-attention, seq_q == seq_k). window == 0 means dense
// (attend all keys) -- byte-identical to attention_sycl with causal == false.
//
// This is the mask EmbeddingGemma's local encoder layers use in its alternating
// global/local stack: unlike causal sliding-window attention the band looks BOTH
// forward and backward within +-window/2, so it is genuinely symmetric, not a
// prefix. Ported from embeddinggemma.c engine_xpu.cpp launch_attention /
// launch_attention_softmax_banded (the `window` branch), CPU fp64 oracle
// kernels.c ei_attention_mha_range. The QK dot-product and PV weighted-sum are
// the same math as the dense flash kernel; only the key range is banded, so the
// long-sequence win is doing O(window) work per query instead of O(seq_k). A
// tiled joint_matrix banded variant (the launch_banded_tensor_attention GEMM
// path) is the throughput optimization, deferred to the attention depth wave.
//
// Shape: symmetric sliding-window, GQA, D<=256.

#include "attention/attention/attention_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxD = 256;

template <typename T>
sycl::event attn_swa_typed(sycl::queue& q, const T* Q, const T* K, const T* V,
                           T* O, std::size_t n_heads, std::size_t n_kv_heads,
                           std::size_t seq_q, std::size_t seq_k, std::size_t d,
                           std::size_t window) {
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));
  const std::size_t gqa = n_heads / n_kv_heads;      // q heads per kv head
  const std::size_t delta = seq_k - seq_q;           // end-align query -> key axis
  const std::size_t half = window / 2;               // symmetric half-band
  return q.parallel_for(sycl::range<1>(n_heads * seq_q), [=](sycl::id<1> idx) {
    const std::size_t h = idx[0] / seq_q;
    const std::size_t qi = idx[0] % seq_q;
    const std::size_t kvh = h / gqa;
    const T* qrow = Q + (h * seq_q + qi) * d;
    const T* kbase = K + kvh * seq_k * d;
    const T* vbase = V + kvh * seq_k * d;

    float qreg[kMaxD];
    for (std::size_t j = 0; j < d; ++j) qreg[j] = static_cast<float>(qrow[j]);

    // Symmetric band [first, last) centered on the query's end-aligned position.
    // window == 0 -> dense (whole key axis). Matches ei_attention_mha_range.
    const std::size_t center = qi + delta;
    std::size_t first = 0, last = seq_k;
    if (window != 0) {
      first = (center > half) ? (center - half) : 0;
      const std::size_t cand = center + half + 1;
      last = (cand < seq_k) ? cand : seq_k;
    }

    float m = -std::numeric_limits<float>::infinity();
    float l = 0.0f;
    float acc[kMaxD];
    for (std::size_t j = 0; j < d; ++j) acc[j] = 0.0f;

    for (std::size_t ki = first; ki < last; ++ki) {
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

sycl::event attn_swa_sycl(sycl::queue& q, const void* Q, const void* K,
                          const void* V, void* O, std::size_t n_heads,
                          std::size_t n_kv_heads, std::size_t seq_q,
                          std::size_t seq_k, std::size_t d, std::size_t window,
                          DType dt) {
  switch (dt) {
    case DType::f32:
      return attn_swa_typed(q, static_cast<const float*>(Q), static_cast<const float*>(K),
                            static_cast<const float*>(V), static_cast<float*>(O),
                            n_heads, n_kv_heads, seq_q, seq_k, d, window);
    case DType::f16:
      return attn_swa_typed(q, static_cast<const half_t*>(Q), static_cast<const half_t*>(K),
                            static_cast<const half_t*>(V), static_cast<half_t*>(O),
                            n_heads, n_kv_heads, seq_q, seq_k, d, window);
    case DType::bf16:
      return attn_swa_typed(q, static_cast<const bf16_t*>(Q), static_cast<const bf16_t*>(K),
                            static_cast<const bf16_t*>(V), static_cast<bf16_t*>(O),
                            n_heads, n_kv_heads, seq_q, seq_k, d, window);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
