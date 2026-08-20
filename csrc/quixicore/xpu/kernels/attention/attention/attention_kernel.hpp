#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

sycl::event attention_sycl(sycl::queue& q, const void* Q, const void* K,
                           const void* V, void* O, std::size_t n_heads,
                           std::size_t n_kv_heads, std::size_t seq_q,
                           std::size_t seq_k, std::size_t d, bool causal,
                           DType dt);

// Same flash SDPA, but the epilogue additionally writes a rounded half copy of
// the output to O_f16 (fused ctx->f16 context store). O is dtype dt; O_f16 is
// always f16. Shape: online-attention + fused f16 context store.
sycl::event attention_f16ctx_sycl(sycl::queue& q, const void* Q, const void* K,
                                  const void* V, void* O, void* O_f16,
                                  std::size_t n_heads, std::size_t n_kv_heads,
                                  std::size_t seq_q, std::size_t seq_k,
                                  std::size_t d, bool causal, DType dt);

// Same flash online-softmax as attention_sycl, but each query attends a
// SYMMETRIC band of `window` keys centered on its own (end-aligned) position
// instead of a causal prefix: keys in [center - window/2, center + window/2]
// clamped to [0, seq_k), center = qi + (seq_k - seq_q). window == 0 is dense
// (identical to attention_sycl non-causal). Shape: symmetric sliding-window,
// GQA, D<=256.
sycl::event attn_swa_sycl(sycl::queue& q, const void* Q, const void* K,
                          const void* V, void* O, std::size_t n_heads,
                          std::size_t n_kv_heads, std::size_t seq_q,
                          std::size_t seq_k, std::size_t d, std::size_t window,
                          DType dt);

}  // namespace quixicore::xpu::kernels
