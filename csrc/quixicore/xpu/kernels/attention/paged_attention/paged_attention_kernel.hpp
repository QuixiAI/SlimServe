#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// KV-cache element encoding for the paged attention ops (see ops.hpp).
// 0 = same as act dtype, 1 = fp8 e4m3, 2 = fp8 e5m2.

// Split-KV paged decode: partials + LSE-merge reduction, two chained
// submissions. See ops::paged_attention_decode.
sycl::event paged_attention_decode_sycl(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* tmp_out, float* exp_sums, float* max_logits,
    const std::int32_t* block_table, const std::int32_t* seq_lens,
    std::size_t batch, std::size_t n_heads, std::size_t n_kv_heads,
    std::size_t d, std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, int num_kv_splits, float sm_scale,
    int window_left, const float* sinks, const float* k_scale,
    const float* v_scale, DType dt, int kv_dt);

// Varlen paged/dense prefill FMHA forward. See ops::paged_attention_prefill.
sycl::event paged_attention_prefill_sycl(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* lse, const std::int32_t* block_table,
    const std::int32_t* cu_seqlens_q, const std::int32_t* cu_seqlens_k,
    const std::uint8_t* is_prefill, std::size_t total_q, std::size_t batch,
    std::size_t n_heads, std::size_t n_kv_heads, std::size_t d,
    std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, std::size_t max_seqlen_k, float sm_scale,
    bool causal, int window_left, int window_right, const float* sinks,
    const float* k_scale, const float* v_scale, DType dt, int kv_dt);

}  // namespace quixicore::xpu::kernels
