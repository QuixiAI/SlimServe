// Dispatch layer for the norms family: routes the public ops ABI to the selected
// implementation variant (native SYCL or vendor oneDNN) and applies blocking.

#include "quixicore/xpu/ops.hpp"

#include "norms/norms_kernel.hpp"
#include "norms/norm_quant/norm_quant_kernel.hpp"
#include "norms/qk_norm_rope/qk_norm_rope_kernel.hpp"

namespace quixicore::xpu::ops {

void rms_norm(sycl::queue& q, const void* x, const void* weight, void* out,
              std::size_t rows, std::size_t dim, float eps, DType dt,
              Variant variant, bool blocking) {
  // No oneDNN RMSNorm primitive; every variant resolves to the native path.
  (void)variant;
  sycl::event ev = kernels::rms_norm_sycl(q, x, weight, out, rows, dim, eps, dt);
  if (blocking) ev.wait();
}

void fused_add_rms_norm(sycl::queue &q, const void *x, void *residual, const void *weight,
                        void *out, std::size_t rows, std::size_t dim, float eps, DType dt,
                        Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev =
      kernels::fused_add_rms_norm_sycl(q, x, residual, weight, out, rows, dim, eps, dt);
  if (blocking)
    ev.wait();
}


void rms_residual_next(sycl::queue &q, const void *projection, const void *post_weight,
                       void *residual, const void *next_weight, void *next_out,
                       std::size_t rows, std::size_t dim, float eps, DType dt,
                       Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::rms_residual_next_sycl(
      q, projection, post_weight, residual, next_weight, next_out, rows, dim, eps, dt);
  if (blocking)
    ev.wait();
}

void layernorm(sycl::queue& q, const void* x, const void* weight,
               const void* bias, void* out, std::size_t rows, std::size_t dim,
               float eps, DType dt, Variant variant, bool blocking) {
  // Data-driven best routing (perf/optimization_status.md 2026-07-06, B60).
  // After the 16-byte vector-load pass, native SYCL wins layernorm at ALL dtypes
  // (f32 393 vs 244, bf16 388 vs 333) -- this overturned the pre-vectorization
  // "route bf16 -> vendor" call. best == sycl for every dtype now.
  if (variant == Variant::best) {
    variant = Variant::sycl;
  }
  const Variant v = resolve_variant(variant);
  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::layernorm_onednn(q, x, weight, bias, out, rows, dim, eps, dt);
      break;
#endif
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::layernorm_sycl(q, x, weight, bias, out, rows, dim, eps, dt);
      break;
  }
  if (blocking) ev.wait();
}

void norm_quant(sycl::queue& q, const void* x, void* residual,
                const void* weight, std::uint8_t* out_q,
                const float* static_scale, float* out_scales, std::size_t rows,
                std::size_t hidden, float eps, NormQuantMode mode, DType dt,
                Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::norm_quant_sycl(q, x, residual, weight, out_q,
                                            static_scale, out_scales, rows,
                                            hidden, eps,
                                            static_cast<int>(mode), dt);
  if (blocking) ev.wait();
}

void group_rms_norm_gated(sycl::queue& q, const void* x, const void* gate,
                          const void* weight, void* out, std::size_t rows,
                          std::size_t hidden, std::size_t n_groups, float eps,
                          bool rms_norm, DType dt, Variant variant,
                          bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::group_rms_norm_gated_sycl(
      q, x, gate, weight, out, rows, hidden, n_groups, eps, rms_norm, dt);
  if (blocking) ev.wait();
}

void qk_norm_rope(sycl::queue& q, void* Q, void* K, const void* q_weight,
                  const void* k_weight, void* Q_f16, void* K_f16,
                  std::size_t tokens, std::size_t n_head, std::size_t n_head_kv,
                  std::size_t head_dim, float base, std::size_t pos0,
                  float query_scale, float eps, DType dt, Variant variant,
                  bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::qk_norm_rope_sycl(
      q, Q, K, q_weight, k_weight, Q_f16, K_f16, tokens, n_head, n_head_kv,
      head_dim, base, pos0, query_scale, eps, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
