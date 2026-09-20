// Varlen paged/dense prefill FMHA forward, native SYCL (contract
// paged_attention_advanced + mixed_prefill_decode_attention).
//
// One work-item per (packed q token, head), house flash-recurrence shape:
// stream the key range with online softmax, resolving keys through the block
// table when paged (page_size is runtime) or through the contiguous
// [B, max_seqlen_k, n_kv_heads, d] layout when block_table is null. Causal
// masking is end-aligned (query qi attends keys <= qi + seq_k - seq_q);
// window_left/right bound the band on either side (-1 = unbounded), which
// with causal=false also expresses the symmetric-window shape. A per-batch
// is_prefill mask lets decode rows share the launch (mixed batches): masked
// sequences are skipped untouched. Optional LSE output (m + log l, sink
// included) feeds merge_attn_states-style consumers; optional attention
// sinks join the denominator. fp8 KV via the shared exact codecs.
// Graph-capture-safe. A q-block/DPAS-tiled variant is the recorded
// throughput lever (prefill is compute-bound; this shape closes the
// contract first).
//
// Semantics from vllm-xpu-kernels csrc/xpu/attn/xe_2/chunk_prefill (Apache,
// cutlass; translated).

#include "attention/paged_attention/paged_attention_kernel.hpp"

#include <limits>

#include "common/quant_codecs.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxD = 256;

namespace qc = quixicore::xpu::qcodec;

template <typename T, int KvDt>
inline float kv_elem(const void* cache, std::size_t idx, float scale) {
  if constexpr (KvDt == 0) {
    return static_cast<float>(static_cast<const T*>(cache)[idx]);
  } else if constexpr (KvDt == 1) {
    return qc::fp8_e4m3(static_cast<const std::uint8_t*>(cache)[idx]) * scale;
  } else {
    return qc::fp8_e5m2(static_cast<const std::uint8_t*>(cache)[idx]) * scale;
  }
}

template <typename T, int KvDt>
class PagedPrefillKernel;

template <typename T, int KvDt>
sycl::event prefill_typed(
    sycl::queue& q, const T* Q, const void* k_cache, const void* v_cache, T* O,
    float* lse, const std::int32_t* block_table,
    const std::int32_t* cu_seqlens_q, const std::int32_t* cu_seqlens_k,
    const std::uint8_t* is_prefill, std::size_t batch, std::size_t n_heads,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t max_pages, std::size_t page_stride_elems,
    std::size_t max_seqlen_k, float sm_scale, bool causal, int window_left,
    int window_right, const float* sinks, const float* k_scale,
    const float* v_scale, std::size_t total_q) {
  const std::size_t gqa = n_heads / n_kv_heads;
  const float neg_inf = -std::numeric_limits<float>::infinity();
  return q.parallel_for<PagedPrefillKernel<T, KvDt>>(
      sycl::range<1>(total_q * n_heads), [=](sycl::id<1> idx) {
        const std::size_t tok = idx[0] / n_heads;
        const std::size_t h = idx[0] % n_heads;
        const std::size_t kvh = h / gqa;
        // Locate the owning sequence (batch is small; linear scan).
        std::size_t b = 0;
        while (b + 1 < batch &&
               tok >= static_cast<std::size_t>(cu_seqlens_q[b + 1]))
          ++b;
        if (is_prefill != nullptr && is_prefill[b] == 0) return;
        const std::size_t q_lo = static_cast<std::size_t>(cu_seqlens_q[b]);
        const std::size_t seq_q =
            static_cast<std::size_t>(cu_seqlens_q[b + 1]) - q_lo;
        const std::size_t k_start = static_cast<std::size_t>(cu_seqlens_k[b]);
        const std::size_t seq_k =
            static_cast<std::size_t>(cu_seqlens_k[b + 1]) - k_start;
        const std::size_t qi = tok - q_lo;
        const float kscale = k_scale != nullptr ? *k_scale : 1.0f;
        const float vscale = v_scale != nullptr ? *v_scale : 1.0f;

        // Band [lo, hi] over this sequence's keys.
        const std::int64_t center =
            static_cast<std::int64_t>(qi) +
            static_cast<std::int64_t>(seq_k) - static_cast<std::int64_t>(seq_q);
        std::int64_t lo64 = 0;
        std::int64_t hi64 = static_cast<std::int64_t>(seq_k) - 1;
        if (causal) hi64 = center < hi64 ? center : hi64;
        if (window_left >= 0) {
          const std::int64_t l2 = center - window_left;
          lo64 = l2 > lo64 ? l2 : lo64;
        }
        if (window_right >= 0 && !causal) {
          const std::int64_t r2 = center + window_right;
          hi64 = r2 < hi64 ? r2 : hi64;
        }

        float qreg[kMaxD];
        const T* qrow = Q + (tok * n_heads + h) * d;
        for (std::size_t j = 0; j < d; ++j)
          qreg[j] = static_cast<float>(qrow[j]);

        float m = neg_inf, l = 0.0f;
        float acc[kMaxD];
        for (std::size_t j = 0; j < d; ++j) acc[j] = 0.0f;

        for (std::int64_t k = lo64; k <= hi64; ++k) {
          std::size_t base;
          if (block_table != nullptr) {
            const std::int32_t page =
                block_table[b * max_pages +
                            static_cast<std::size_t>(k) / page_size];
            if (page < 0) continue;
            base = static_cast<std::size_t>(page) * page_stride_elems +
                   ((static_cast<std::size_t>(k) % page_size) * n_kv_heads +
                    kvh) *
                       d;
          } else {
            base = ((b * max_seqlen_k + static_cast<std::size_t>(k)) *
                        n_kv_heads +
                    kvh) *
                   d;
            (void)k_start;
          }
          float score = 0.0f;
          for (std::size_t j = 0; j < d; ++j)
            score += qreg[j] * kv_elem<T, KvDt>(k_cache, base + j, kscale);
          score *= sm_scale;
          const float m_new = sycl::fmax(m, score);
          const float corr = sycl::exp(m - m_new);
          const float p = sycl::exp(score - m_new);
          l = l * corr + p;
          for (std::size_t j = 0; j < d; ++j)
            acc[j] =
                acc[j] * corr + p * kv_elem<T, KvDt>(v_cache, base + j, vscale);
          m = m_new;
        }

        if (sinks != nullptr && m > neg_inf) l += sycl::exp(sinks[h] - m);

        T* orow = O + (tok * n_heads + h) * d;
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (std::size_t j = 0; j < d; ++j)
          orow[j] = static_cast<T>(acc[j] * inv);
        if (lse != nullptr)
          lse[tok * n_heads + h] = m > neg_inf ? m + sycl::log(l) : neg_inf;
      });
}

template <typename T>
sycl::event prefill_kv(sycl::queue& q, const T* Q, const void* kc,
                       const void* vc, T* O, float* lse,
                       const std::int32_t* bt, const std::int32_t* cq,
                       const std::int32_t* ck, const std::uint8_t* pf,
                       std::size_t B, std::size_t H, std::size_t Hkv,
                       std::size_t d, std::size_t ps, std::size_t mp,
                       std::size_t pse, std::size_t msk, float sc, bool causal,
                       int wl, int wr, const float* sk, const float* ksc,
                       const float* vsc, std::size_t total_q, int kv_dt) {
  switch (kv_dt) {
    case 1:
      return prefill_typed<T, 1>(q, Q, kc, vc, O, lse, bt, cq, ck, pf, B, H,
                                 Hkv, d, ps, mp, pse, msk, sc, causal, wl, wr,
                                 sk, ksc, vsc, total_q);
    case 2:
      return prefill_typed<T, 2>(q, Q, kc, vc, O, lse, bt, cq, ck, pf, B, H,
                                 Hkv, d, ps, mp, pse, msk, sc, causal, wl, wr,
                                 sk, ksc, vsc, total_q);
    default:
      return prefill_typed<T, 0>(q, Q, kc, vc, O, lse, bt, cq, ck, pf, B, H,
                                 Hkv, d, ps, mp, pse, msk, sc, causal, wl, wr,
                                 sk, ksc, vsc, total_q);
  }
}

}  // namespace

sycl::event paged_attention_prefill_sycl(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* lse, const std::int32_t* block_table,
    const std::int32_t* cu_seqlens_q, const std::int32_t* cu_seqlens_k,
    const std::uint8_t* is_prefill, std::size_t total_q, std::size_t batch,
    std::size_t n_heads, std::size_t n_kv_heads, std::size_t d,
    std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, std::size_t max_seqlen_k, float sm_scale,
    bool causal, int window_left, int window_right, const float* sinks,
    const float* k_scale, const float* v_scale, DType dt, int kv_dt) {
  if (d > kMaxD || total_q == 0) return {};  // reject, don't corrupt
  switch (dt) {
    case DType::f32:
      return prefill_kv(q, static_cast<const float*>(Q), k_cache, v_cache,
                        static_cast<float*>(O), lse, block_table, cu_seqlens_q,
                        cu_seqlens_k, is_prefill, batch, n_heads, n_kv_heads,
                        d, page_size, max_pages, page_stride_elems,
                        max_seqlen_k, sm_scale, causal, window_left,
                        window_right, sinks, k_scale, v_scale, total_q, kv_dt);
    case DType::f16:
      return prefill_kv(q, static_cast<const half_t*>(Q), k_cache, v_cache,
                        static_cast<half_t*>(O), lse, block_table,
                        cu_seqlens_q, cu_seqlens_k, is_prefill, batch, n_heads,
                        n_kv_heads, d, page_size, max_pages, page_stride_elems,
                        max_seqlen_k, sm_scale, causal, window_left,
                        window_right, sinks, k_scale, v_scale, total_q, kv_dt);
    case DType::bf16:
      return prefill_kv(q, static_cast<const bf16_t*>(Q), k_cache, v_cache,
                        static_cast<bf16_t*>(O), lse, block_table,
                        cu_seqlens_q, cu_seqlens_k, is_prefill, batch, n_heads,
                        n_kv_heads, d, page_size, max_pages, page_stride_elems,
                        max_seqlen_k, sm_scale, causal, window_left,
                        window_right, sinks, k_scale, v_scale, total_q, kv_dt);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
