#pragma once

// Internal variant entry points for the norms family (rms_norm, layernorm).
// Included by the dispatch layer and the benchmark harness via the ops target's
// private kernels/ include path. Each entry submits and returns its sycl::event
// without waiting.

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// Native SYCL. Always available.
sycl::event rms_norm_sycl(sycl::queue& q, const void* x, const void* weight,
                          void* out, std::size_t rows, std::size_t dim,
                          float eps, DType dt);

sycl::event fused_add_rms_norm_sycl(sycl::queue &q, const void *x, void *residual,
                                    const void *weight, void *out, std::size_t rows,
                                    std::size_t dim, float eps, DType dt);


// Fused residual-add + double RMSNorm with f16 convert. Extends
// fused_add_rms_norm to the transformer layer boundary: RMS-normalize the
// sublayer output `projection` by `post_weight`, add it into the `residual`
// stream (updated in place), then RMS-normalize the updated residual by
// `next_weight` for the next layer and write it to `next_out` as f16. Per row
// over `dim`, fp32 accumulation:
//   pinv     = rsqrt(mean_d(projection^2) + eps)
//   residual = residual + projection * post_weight * pinv
//   rinv     = rsqrt(mean_d(residual^2) + eps)
//   next_out = f16(residual * next_weight * rinv)
// `projection`, `post_weight`, `residual`, `next_weight` are dtype dt;
// `next_out` is always f16. Collapses the post-norm, residual add, next
// pre-norm, and f16 convert into one launch (~2 launches/layer saved). Shape:
// residual-add + double RMSNorm -> f16.
sycl::event rms_residual_next_sycl(sycl::queue& q, const void* projection,
                                   const void* post_weight, void* residual,
                                   const void* next_weight, void* next_out,
                                   std::size_t rows, std::size_t dim, float eps,
                                   DType dt);

sycl::event layernorm_sycl(sycl::queue& q, const void* x, const void* weight,
                           const void* bias, void* out, std::size_t rows,
                           std::size_t dim, float eps, DType dt);

#if defined(QUIXICORE_XPU_HAS_ONEDNN)
// Vendor LayerNorm via oneDNN layer_normalization. (oneDNN has no dedicated
// RMSNorm primitive, so rms_norm ships SYCL-only and Variant::vendor falls back
// to SYCL for it.)
sycl::event layernorm_onednn(sycl::queue& q, const void* x, const void* weight,
                             const void* bias, void* out, std::size_t rows,
                             std::size_t dim, float eps, DType dt);
#endif

// Gated group-RMSNorm (Mamba-2 mixer norm); see ops::group_rms_norm_gated.
sycl::event group_rms_norm_gated_sycl(sycl::queue& q, const void* x,
                                      const void* gate, const void* weight,
                                      void* out, std::size_t rows,
                                      std::size_t hidden, std::size_t n_groups,
                                      float eps, bool rms_norm, DType dt);

}  // namespace quixicore::xpu::kernels
