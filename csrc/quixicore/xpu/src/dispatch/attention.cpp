// Dispatch layer for the attention family.

#include "quixicore/xpu/ops.hpp"

#include "attention/attention/attention_kernel.hpp"
#include "attention/merge_attn_states/merge_attn_states_kernel.hpp"
#include "attention/mrope/mrope_kernel.hpp"
#include "attention/paged_attention/paged_attention_kernel.hpp"
#include "attention/rope/rope_kernel.hpp"

namespace quixicore::xpu::ops {

void attention(sycl::queue& q, const void* Q, const void* K, const void* V,
               void* O, std::size_t n_heads, std::size_t n_kv_heads,
               std::size_t seq_q, std::size_t seq_k, std::size_t d, bool causal,
               DType dt, Variant variant, bool blocking) {
  (void)variant;  // native flash; oneDNN-Graph SDPA vendor variant deferred
  sycl::event ev = kernels::attention_sycl(q, Q, K, V, O, n_heads, n_kv_heads,
                                           seq_q, seq_k, d, causal, dt);
  if (blocking) ev.wait();
}

void attention_f16ctx(sycl::queue& q, const void* Q, const void* K,
                      const void* V, void* O, void* O_f16, std::size_t n_heads,
                      std::size_t n_kv_heads, std::size_t seq_q,
                      std::size_t seq_k, std::size_t d, bool causal, DType dt,
                      Variant variant, bool blocking) {
  (void)variant;  // native flash + fused f16 store
  sycl::event ev =
      kernels::attention_f16ctx_sycl(q, Q, K, V, O, O_f16, n_heads, n_kv_heads,
                                     seq_q, seq_k, d, causal, dt);
  if (blocking) ev.wait();
}

void attn_swa(sycl::queue& q, const void* Q, const void* K, const void* V,
              void* O, std::size_t n_heads, std::size_t n_kv_heads,
              std::size_t seq_q, std::size_t seq_k, std::size_t d,
              std::size_t window, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native flash + symmetric sliding-window band mask
  sycl::event ev = kernels::attn_swa_sycl(q, Q, K, V, O, n_heads, n_kv_heads,
                                          seq_q, seq_k, d, window, dt);
  if (blocking) ev.wait();
}

void rope(sycl::queue& q, const void* x, void* out, std::size_t tokens,
          std::size_t n_heads, std::size_t head_dim, float base,
          std::size_t pos0, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev =
      kernels::rope_sycl(q, x, out, tokens, n_heads, head_dim, base, pos0, dt);
  if (blocking) ev.wait();
}

void paged_attention_decode(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* tmp_out, float* exp_sums, float* max_logits,
    const std::int32_t* block_table, const std::int32_t* seq_lens,
    std::size_t batch, std::size_t n_heads, std::size_t n_kv_heads,
    std::size_t d, std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, int num_kv_splits, float sm_scale,
    int window_left, const float* sinks, const float* k_scale,
    const float* v_scale, DType dt, KvCacheDType kv_dt, Variant variant,
    bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::paged_attention_decode_sycl(
      q, Q, k_cache, v_cache, O, tmp_out, exp_sums, max_logits, block_table,
      seq_lens, batch, n_heads, n_kv_heads, d, page_size, max_pages,
      page_stride_elems, num_kv_splits, sm_scale, window_left, sinks, k_scale,
      v_scale, dt, static_cast<int>(kv_dt));
  if (blocking) ev.wait();
}

void paged_attention_prefill(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* lse, const std::int32_t* block_table,
    const std::int32_t* cu_seqlens_q, const std::int32_t* cu_seqlens_k,
    const std::uint8_t* is_prefill, std::size_t total_q, std::size_t batch,
    std::size_t n_heads, std::size_t n_kv_heads, std::size_t d,
    std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, std::size_t max_seqlen_k, float sm_scale,
    bool causal, int window_left, int window_right, const float* sinks,
    const float* k_scale, const float* v_scale, DType dt, KvCacheDType kv_dt,
    Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::paged_attention_prefill_sycl(
      q, Q, k_cache, v_cache, O, lse, block_table, cu_seqlens_q, cu_seqlens_k,
      is_prefill, total_q, batch, n_heads, n_kv_heads, d, page_size, max_pages,
      page_stride_elems, max_seqlen_k, sm_scale, causal, window_left,
      window_right, sinks, k_scale, v_scale, dt, static_cast<int>(kv_dt));
  if (blocking) ev.wait();
}

void mrope(sycl::queue& q, void* query, void* key, const void* cos_sin_cache,
           const std::int64_t* positions, const std::int32_t* sections,
           std::size_t n_sections, std::size_t tokens, std::size_t n_heads,
           std::size_t n_kv_heads, std::size_t head_size, std::size_t rot_dim,
           bool neox, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::mrope_sycl(q, query, key, cos_sin_cache, positions,
                                       sections, n_sections, tokens, n_heads,
                                       n_kv_heads, head_size, rot_dim,
                                       neox ? 1 : 0, dt);
  if (blocking) ev.wait();
}

void rotary_positioned(sycl::queue& q, void* query, void* key,
                       const void* cos_sin_cache,
                       const std::int64_t* positions, std::size_t tokens,
                       std::size_t n_heads, std::size_t n_kv_heads,
                       std::size_t head_size, std::size_t rot_dim, bool neox,
                       DType dt, Variant variant, bool blocking) {
  (void)variant;  // single-section mrope
  sycl::event ev = kernels::mrope_sycl(q, query, key, cos_sin_cache, positions,
                                       nullptr, 1, tokens, n_heads, n_kv_heads,
                                       head_size, rot_dim, neox ? 1 : 0, dt);
  if (blocking) ev.wait();
}

void merge_attn_states(sycl::queue& q, const void* out_a, const float* lse_a,
                       const void* out_b, const float* lse_b, void* out,
                       float* lse_out, std::size_t rows, std::size_t d,
                       DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::merge_attn_states_sycl(q, out_a, lse_a, out_b,
                                                   lse_b, out, lse_out, rows,
                                                   d, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
