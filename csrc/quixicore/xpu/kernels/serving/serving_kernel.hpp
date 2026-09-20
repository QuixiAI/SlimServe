#pragma once

// Internal entry points for the serving family (embedding lookup, KV-cache
// scatter/gather). All are indexed row copies; dtype-agnostic (copied by byte
// width) so one kernel covers f32/f16/bf16.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event embedding_lookup_sycl(sycl::queue& q, const void* table,
                                  const int* ids, void* out, std::size_t n,
                                  std::size_t dim, DType dt);

sycl::event kv_cache_scatter_sycl(sycl::queue& q, void* cache, const void* src,
                                  const int* slots, std::size_t n,
                                  std::size_t row, DType dt);

sycl::event kv_cache_gather_sycl(sycl::queue& q, const void* cache,
                                 const int* idx, void* out, std::size_t n,
                                 std::size_t row, DType dt);

// Sentence-embedding pooling head: masked mean-pool over each sequence's tokens
// with a per-token RMSNorm (learned weight) folded in, then L2-normalize the
// pooled vector. Fuses the three passes (per-token RMSNorm -> masked mean ->
// L2) into one kernel that reads the token embeddings once and writes one
// vector per sequence.
//
// `x` is [total_tokens, dim] row-major; sequence s owns token rows
// [offsets[s], offsets[s+1]). `weight` is [dim], `out` is [batch, dim], all
// dtype dt; `offsets` is [batch+1] int32 (monotic, non-negative). For sequence
// s with token range [a, b):
//   r_t   = x[t] * rsqrt(mean_d(x[t,d]^2) + eps) * weight   (RMSNorm, per token)
//   m     = (1/(b-a)) * sum_{t in [a,b)} r_t                (masked mean)
//   out[s]= m * rsqrt(sum_d m[d]^2)                         (L2; 0-vector passes)
// `dim` is the shape key (256/512/768/1024); one subgroup owns each sequence's
// dim-vector. fp32 accumulation. An empty sequence (b==a) yields a zero vector.
// Paged KV-cache write (see ops::kv_cache_scatter_paged).
sycl::event kv_cache_scatter_paged_sycl(
    sycl::queue& q, const void* key, const void* value, void* k_cache,
    void* v_cache, const std::int64_t* slot_mapping, std::size_t n_tokens,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    int fp8, DType dt);

// Paged KV-cache gather + optional fp8 dequant (see ops::kv_cache_gather_paged).
sycl::event kv_cache_gather_paged_sycl(
    sycl::queue& q, const void* k_cache, const void* v_cache, void* k_out,
    void* v_out, const std::int64_t* slots, std::size_t n,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    int fp8, DType dt);

sycl::event pool_mean_rms_l2_sycl(sycl::queue& q, const void* x,
                                  const void* weight, const int* offsets,
                                  void* out, std::size_t batch, std::size_t dim,
                                  float eps, DType dt);

}  // namespace quixicore::xpu::kernels
