// Flash-style scaled dot-product attention with a fused f16 context store,
// native SYCL. Identical online-softmax math and per-query register layout to
// attention_sycl (running max m, running denom l, running weighted-value
// acc[d]; the seq_k x seq_k score matrix is never materialized), but the
// epilogue writes the output TWICE: once in the compute/storage dtype T (O) and
// once as a rounded half copy (O_f16) in the same pass.
//
// Shape: online-attention + fused f16 context store. This folds the ctx->f16
// convert that a downstream attention-output projection GEMM would otherwise
// need into the attention epilogue, eliminating a standalone O-sized read+write
// convert kernel (and its launch) between attention and the projection. Ported
// from embeddinggemma.c's EI_XPU_FUSE_ATTN_OUTPUT_HALF online-attention
// epilogue, where the fused f16 store let the attn-output GEMM read the half
// context directly. acc[j]*inv is computed once and consumed by both stores, so
// O and O_f16 are the exact-same fp32 result rounded to their storage types --
// O_f16 is bit-identical to static_cast<half>(O) when T==f32/f16.

#include "attention/attention/attention_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxD = 128;

template <typename T>
sycl::event attn_f16ctx_typed(sycl::queue& q, const T* Q, const T* K, const T* V,
                              T* O, half_t* O16, std::size_t n_heads,
                              std::size_t n_kv_heads, std::size_t seq_q,
                              std::size_t seq_k, std::size_t d, bool causal) {
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
    half_t* orow16 = O16 + (h * seq_q + qi) * d;
    const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    for (std::size_t j = 0; j < d; ++j) {
      const float res = acc[j] * inv;
      orow[j] = static_cast<T>(res);
      orow16[j] = static_cast<half_t>(res);  // fused ctx->f16 store
    }
  });
}

}  // namespace

sycl::event attention_f16ctx_sycl(sycl::queue& q, const void* Q, const void* K,
                                  const void* V, void* O, void* O_f16,
                                  std::size_t n_heads, std::size_t n_kv_heads,
                                  std::size_t seq_q, std::size_t seq_k,
                                  std::size_t d, bool causal, DType dt) {
  half_t* o16 = static_cast<half_t*>(O_f16);
  switch (dt) {
    case DType::f32:
      return attn_f16ctx_typed(q, static_cast<const float*>(Q), static_cast<const float*>(K),
                               static_cast<const float*>(V), static_cast<float*>(O), o16,
                               n_heads, n_kv_heads, seq_q, seq_k, d, causal);
    case DType::f16:
      return attn_f16ctx_typed(q, static_cast<const half_t*>(Q), static_cast<const half_t*>(K),
                               static_cast<const half_t*>(V), static_cast<half_t*>(O), o16,
                               n_heads, n_kv_heads, seq_q, seq_k, d, causal);
    case DType::bf16:
      return attn_f16ctx_typed(q, static_cast<const bf16_t*>(Q), static_cast<const bf16_t*>(K),
                               static_cast<const bf16_t*>(V), static_cast<bf16_t*>(O), o16,
                               n_heads, n_kv_heads, seq_q, seq_k, d, causal);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
