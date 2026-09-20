#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Fused per-head QK-norm + RoPE (+ optional f16 convert). For every (token,
// head) of Q and K: RMS-normalize the head-dim vector by its learned weight,
// scale the query heads by `query_scale` (key heads by 1), then apply NeoX
// half-split RoPE using position (pos0 + token). Q is [tokens, n_head, head_dim]
// and K is [tokens, n_head_kv, head_dim] row-major, both dtype dt, updated in
// place. `q_weight` / `k_weight` are [head_dim] of dtype dt. If `Q_f16` /
// `K_f16` are non-null (always f16, same layout) the rotated result is also
// written there (the fused ctx->f16 convert a downstream QK GEMM needs). One
// subgroup owns each (token, head); fp32 accumulation. Collapses the per-head
// rms_norm(Q) + rms_norm(K) + query-scale + rope(Q) + rope(K) chain into one
// launch. Shape: per-head RMSNorm + query-scale + RoPE.
sycl::event qk_norm_rope_sycl(sycl::queue& q, void* Q, void* K,
                              const void* q_weight, const void* k_weight,
                              void* Q_f16, void* K_f16, std::size_t tokens,
                              std::size_t n_head, std::size_t n_head_kv,
                              std::size_t head_dim, float base, std::size_t pos0,
                              float query_scale, float eps, DType dt);

}  // namespace quixicore::xpu::kernels
