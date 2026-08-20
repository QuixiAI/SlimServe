// Split-KV paged attention decode, native SYCL — the paged KV-cache decode
// the backend was missing (contract decode_cache_attention).
//
// Correctness-first geometry matching the house dense flash kernel: one
// work-item per (sequence, q head, kv split) runs the online-softmax
// recurrence over its split's key range, resolving each key through the
// block table (page = block_table[b, k / page_size], element offset
// (k % page_size) * n_kv_heads * d — page_size is RUNTIME, deleting the
// build-time tuple matrix of the cutlass source). Per-split fp32 partials
// (weighted-value accumulator, exp sum, running max) land in CALLER-OWNED
// max-shaped workspaces; a second chained submission LSE-merges the splits
// per (sequence, head), adds the optional attention-sink term to the
// denominator, and stores O. No host sync, no allocation — graph-capture-
// safe. fp8 KV decodes through the shared exact codecs with device-scalar
// k/v scales.
//
// Semantics from vllm-xpu-kernels csrc/xpu/attn/xe_2/paged_decode (Apache,
// cutlass; translated). A subgroup/DPAS-tiled variant is the recorded
// throughput lever, shared with mqa_logits.

#include "attention/paged_attention/paged_attention_kernel.hpp"

#include <limits>

#include "common/quant_codecs.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxD = 256;

namespace qc = quixicore::xpu::qcodec;

template <typename T, int KvDt>
inline float kv_at(const void* cache, std::size_t idx, float scale) {
  if constexpr (KvDt == 0) {
    return static_cast<float>(static_cast<const T*>(cache)[idx]);
  } else if constexpr (KvDt == 1) {
    return qc::fp8_e4m3(static_cast<const std::uint8_t*>(cache)[idx]) * scale;
  } else {
    return qc::fp8_e5m2(static_cast<const std::uint8_t*>(cache)[idx]) * scale;
  }
}

template <typename T, int KvDt>
class PagedDecodePartialKernel;
template <typename T, int KvDt>
class PagedDecodeReduceKernel;

template <typename T, int KvDt>
sycl::event decode_typed(sycl::queue& q, const T* Q, const void* k_cache,
                         const void* v_cache, T* O, float* tmp_out,
                         float* exp_sums, float* max_logits,
                         const std::int32_t* block_table,
                         const std::int32_t* seq_lens, std::size_t batch,
                         std::size_t n_heads, std::size_t n_kv_heads,
                         std::size_t d, std::size_t page_size,
                         std::size_t max_pages, std::size_t page_stride_elems,
                         int num_kv_splits, float sm_scale, int window_left,
                         const float* sinks, const float* k_scale,
                         const float* v_scale) {
  const std::size_t gqa = n_heads / n_kv_heads;
  const std::size_t splits = static_cast<std::size_t>(num_kv_splits);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  constexpr int kSG = 16;
  sycl::event partials = q.submit([&](sycl::handler& h1) {
    h1.parallel_for<PagedDecodePartialKernel<T, KvDt>>(
        sycl::nd_range<1>(sycl::range<1>(batch * n_heads * splits * kSG),
                          sycl::range<1>(kSG)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
          const std::size_t grp = it.get_group(0);
          const std::size_t sp = grp % splits;
          const std::size_t bh = grp / splits;
          const std::size_t b = bh / n_heads;
          const std::size_t h = bh % n_heads;
          const std::size_t kvh = h / gqa;
          const int lane = static_cast<int>(it.get_local_id(0));
          const sycl::sub_group sg = it.get_sub_group();
          const std::size_t ctx = static_cast<std::size_t>(seq_lens[b]);
          const float kscale = k_scale != nullptr ? *k_scale : 1.0f;
          const float vscale = v_scale != nullptr ? *v_scale : 1.0f;

          std::size_t lo = 0;
          if (window_left >= 0) {
            const std::size_t w = static_cast<std::size_t>(window_left) + 1;
            lo = ctx > w ? ctx - w : 0;
          }
          const std::size_t span = ctx > lo ? ctx - lo : 0;
          const std::size_t per = (span + splits - 1) / splits;
          const std::size_t k0 = lo + sp * per;
          const std::size_t k1 = sycl::min(k0 + per, ctx);

          // Lane-interleaved register slices: lane owns elements
          // j = lane + i*kSG, so row reads are coalesced across the subgroup.
          constexpr int kPerLane = kMaxD / kSG;
          const int nsl = static_cast<int>(d) / kSG;
          float qreg[kPerLane];
          const T* qrow = Q + (b * n_heads + h) * d;
          for (int i = 0; i < nsl; ++i)
            qreg[i] = static_cast<float>(qrow[lane + i * kSG]);

          float m = neg_inf, l = 0.0f;
          float acc[kPerLane];
          for (int i = 0; i < nsl; ++i) acc[i] = 0.0f;

          for (std::size_t k = k0; k < k1; ++k) {
            const std::int32_t page =
                block_table[b * max_pages + k / page_size];
            if (page < 0) continue;
            const std::size_t base =
                static_cast<std::size_t>(page) * page_stride_elems +
                ((k % page_size) * n_kv_heads + kvh) * d;
            float partial = 0.0f;
            for (int i = 0; i < nsl; ++i)
              partial += qreg[i] *
                         kv_at<T, KvDt>(k_cache, base + lane + i * kSG, kscale);
            const float score =
                sycl::reduce_over_group(sg, partial, sycl::plus<float>()) *
                sm_scale;
            const float m_new = sycl::fmax(m, score);
            const float corr = sycl::exp(m - m_new);
            const float p = sycl::exp(score - m_new);
            l = l * corr + p;
            for (int i = 0; i < nsl; ++i)
              acc[i] = acc[i] * corr +
                       p * kv_at<T, KvDt>(v_cache, base + lane + i * kSG,
                                          vscale);
            m = m_new;
          }

          const std::size_t widx = (b * n_heads + h) * splits + sp;
          if (lane == 0) {
            exp_sums[widx] = l;
            max_logits[widx] = m;
          }
          float* dst = tmp_out + widx * d;
          for (int i = 0; i < nsl; ++i) dst[lane + i * kSG] = acc[i];
        });
  });

  return q.submit([&](sycl::handler& h2) {
    h2.depends_on(partials);
    h2.parallel_for<PagedDecodeReduceKernel<T, KvDt>>(
        sycl::range<1>(batch * n_heads), [=](sycl::id<1> idx) {
          const std::size_t b = idx[0] / n_heads;
          const std::size_t h = idx[0] % n_heads;
          const std::size_t base = (b * n_heads + h) * splits;
          float M = neg_inf;
          for (std::size_t sp = 0; sp < splits; ++sp)
            M = sycl::fmax(M, max_logits[base + sp]);
          float denom = 0.0f;
          if (sinks != nullptr && M > neg_inf)
            denom += sycl::exp(sinks[h] - M);
          float acc[kMaxD];
          for (std::size_t j = 0; j < d; ++j) acc[j] = 0.0f;
          for (std::size_t sp = 0; sp < splits; ++sp) {
            const float msp = max_logits[base + sp];
            if (!(msp > neg_inf)) continue;  // empty split
            const float w = sycl::exp(msp - M);
            denom += exp_sums[base + sp] * w;
            const float* src = tmp_out + (base + sp) * d;
            for (std::size_t j = 0; j < d; ++j) acc[j] += src[j] * w;
          }
          T* orow = static_cast<T*>(O) + (b * n_heads + h) * d;
          const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
          for (std::size_t j = 0; j < d; ++j)
            orow[j] = static_cast<T>(acc[j] * inv);
        });
  });
}

template <typename T>
sycl::event decode_kv(sycl::queue& q, const T* Q, const void* kc,
                      const void* vc, T* O, float* t, float* e, float* m,
                      const std::int32_t* bt, const std::int32_t* sl,
                      std::size_t B, std::size_t H, std::size_t Hkv,
                      std::size_t d, std::size_t ps, std::size_t mp,
                      std::size_t pse, int ns, float sc, int wl,
                      const float* sk, const float* ksc, const float* vsc,
                      int kv_dt) {
  switch (kv_dt) {
    case 1:
      return decode_typed<T, 1>(q, Q, kc, vc, O, t, e, m, bt, sl, B, H, Hkv, d,
                                ps, mp, pse, ns, sc, wl, sk, ksc, vsc);
    case 2:
      return decode_typed<T, 2>(q, Q, kc, vc, O, t, e, m, bt, sl, B, H, Hkv, d,
                                ps, mp, pse, ns, sc, wl, sk, ksc, vsc);
    default:
      return decode_typed<T, 0>(q, Q, kc, vc, O, t, e, m, bt, sl, B, H, Hkv, d,
                                ps, mp, pse, ns, sc, wl, sk, ksc, vsc);
  }
}

}  // namespace

sycl::event paged_attention_decode_sycl(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* tmp_out, float* exp_sums, float* max_logits,
    const std::int32_t* block_table, const std::int32_t* seq_lens,
    std::size_t batch, std::size_t n_heads, std::size_t n_kv_heads,
    std::size_t d, std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, int num_kv_splits, float sm_scale,
    int window_left, const float* sinks, const float* k_scale,
    const float* v_scale, DType dt, int kv_dt) {
  if (d > kMaxD || d % 16 != 0 || num_kv_splits < 1) return {};  // reject, don't corrupt
  switch (dt) {
    case DType::f32:
      return decode_kv(q, static_cast<const float*>(Q), k_cache, v_cache,
                       static_cast<float*>(O), tmp_out, exp_sums, max_logits,
                       block_table, seq_lens, batch, n_heads, n_kv_heads, d,
                       page_size, max_pages, page_stride_elems, num_kv_splits,
                       sm_scale, window_left, sinks, k_scale, v_scale, kv_dt);
    case DType::f16:
      return decode_kv(q, static_cast<const half_t*>(Q), k_cache, v_cache,
                       static_cast<half_t*>(O), tmp_out, exp_sums, max_logits,
                       block_table, seq_lens, batch, n_heads, n_kv_heads, d,
                       page_size, max_pages, page_stride_elems, num_kv_splits,
                       sm_scale, window_left, sinks, k_scale, v_scale, kv_dt);
    case DType::bf16:
      return decode_kv(q, static_cast<const bf16_t*>(Q), k_cache, v_cache,
                       static_cast<bf16_t*>(O), tmp_out, exp_sums, max_logits,
                       block_table, seq_lens, batch, n_heads, n_kv_heads, d,
                       page_size, max_pages, page_stride_elems, num_kv_splits,
                       sm_scale, window_left, sinks, k_scale, v_scale, kv_dt);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
