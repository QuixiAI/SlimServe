// Dispatch layer for the serving family (native only — index copies).

#include "quixicore/xpu/ops.hpp"

#include "serving/mqa_logits/mqa_logits_kernel.hpp"

#include "serving/serving_kernel.hpp"

namespace quixicore::xpu::ops {

void embedding_lookup(sycl::queue& q, const void* table, const int* ids,
                      void* out, std::size_t n, std::size_t dim, DType dt,
                      Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::embedding_lookup_sycl(q, table, ids, out, n, dim, dt);
  if (blocking) ev.wait();
}

void kv_cache_scatter(sycl::queue& q, void* cache, const void* src,
                      const int* slots, std::size_t n, std::size_t row, DType dt,
                      Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::kv_cache_scatter_sycl(q, cache, src, slots, n, row, dt);
  if (blocking) ev.wait();
}

void kv_cache_gather(sycl::queue& q, const void* cache, const int* idx,
                     void* out, std::size_t n, std::size_t row, DType dt,
                     Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::kv_cache_gather_sycl(q, cache, idx, out, n, row, dt);
  if (blocking) ev.wait();
}

void pool_mean_rms_l2(sycl::queue& q, const void* x, const void* weight,
                      const int* offsets, void* out, std::size_t batch,
                      std::size_t dim, float eps, DType dt, Variant variant,
                      bool blocking) {
  (void)variant;
  sycl::event ev =
      kernels::pool_mean_rms_l2_sycl(q, x, weight, offsets, out, batch, dim, eps, dt);
  if (blocking) ev.wait();
}

void mqa_logits(sycl::queue& q, const std::uint8_t* q_fp8,
                const std::uint8_t* kv_fp8, const float* kv_scales,
                const float* head_weights, const std::int32_t* ks,
                const std::int32_t* ke, float* logits, std::size_t S,
                std::size_t H, std::size_t D, std::size_t Skv, Variant variant,
                bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::mqa_logits_sycl(q, q_fp8, kv_fp8, kv_scales,
                                            head_weights, ks, ke, logits, S, H,
                                            D, Skv);
  if (blocking) ev.wait();
}

void kv_cache_scatter_paged(
    sycl::queue& q, const void* key, const void* value, void* k_cache,
    void* v_cache, const std::int64_t* slot_mapping, std::size_t n_tokens,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    KvCacheDType kv_dt, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::kv_cache_scatter_paged_sycl(
      q, key, value, k_cache, v_cache, slot_mapping, n_tokens, n_kv_heads, d,
      page_size, page_stride_elems, k_scale, v_scale,
      kv_dt == KvCacheDType::same ? 0 : 1, dt);
  if (blocking) ev.wait();
}

void kv_cache_gather_paged(
    sycl::queue& q, const void* k_cache, const void* v_cache, void* k_out,
    void* v_out, const std::int64_t* slots, std::size_t n,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    KvCacheDType kv_dt, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::kv_cache_gather_paged_sycl(
      q, k_cache, v_cache, k_out, v_out, slots, n, n_kv_heads, d, page_size,
      page_stride_elems, k_scale, v_scale,
      kv_dt == KvCacheDType::same ? 0 : 1, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
