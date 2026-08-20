// Dispatch layer for activation ops: routes the public ops ABI to the selected
// implementation variant (native SYCL or vendor oneDNN) and applies blocking
// semantics. The XPU analogue of the Metal backend's `tk::launch_*` layer.

#include "quixicore/xpu/ops.hpp"

#include "activations/gelu/gelu_kernel.hpp"
#include "activations/gelu_backward/gelu_backward_kernel.hpp"
#include "activations/glu/glu_kernel.hpp"
#include "activations/glu_quant/glu_quant_kernel.hpp"
#include "activations/silu/silu_kernel.hpp"
#include "activations/softmax/softmax_kernel.hpp"

namespace quixicore::xpu::ops {

void gelu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          GeluApprox approx, Variant variant, bool blocking) {
  const bool tanh_approx = (approx == GeluApprox::tanh);
  const Variant v = resolve_variant(variant);

  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::gelu_onednn(q, in, out, n, dt, tanh_approx);
      break;
#endif
      // falls through to sycl when vendor is not compiled in
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::gelu_sycl(q, in, out, n, dt, tanh_approx);
      break;
  }

  if (blocking) {
    ev.wait();
  }
}

void softmax(sycl::queue& q, const void* x, void* out, std::size_t rows,
             std::size_t dim, DType dt, Variant variant, bool blocking) {
  // Data-driven best routing (perf/optimization_status.md 2026-07-06, B60).
  // Native SYCL wins f32 softmax (389 vs 223 GB/s). For bf16, the vector-load
  // pass narrowed the gap (195 -> 302) but oneDNN (348) still wins -- softmax is
  // exp()-bound, unlike layernorm where native now wins bf16 too. Route bf16/f16
  // to vendor.
  if (variant == Variant::best) {
    variant = (dt == DType::f32) ? Variant::sycl : Variant::vendor;
  }
  const Variant v = resolve_variant(variant);
  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::softmax_onednn(q, x, out, rows, dim, dt);
      break;
#endif
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::softmax_sycl(q, x, out, rows, dim, dt);
      break;
  }
  if (blocking) ev.wait();
}

void silu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          Variant variant, bool blocking) {
  (void)variant;  // native only for now
  sycl::event ev = kernels::silu_sycl(q, in, out, n, dt);
  if (blocking) ev.wait();
}

void gelu_backward(sycl::queue& q, const void* grad_out, const void* x,
                   void* grad_in, std::size_t n, DType dt, GeluApprox approx,
                   Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::gelu_backward_sycl(q, grad_out, x, grad_in, n, dt,
                                               approx == GeluApprox::tanh);
  if (blocking) ev.wait();
}

void glu(sycl::queue& q, const void* x, void* out, std::size_t rows,
         std::size_t d, DType dt, GluMode mode, Variant variant, bool blocking) {
  (void)variant;
  sycl::event ev = kernels::glu_sycl(q, x, out, rows, d, dt, static_cast<int>(mode));
  if (blocking) ev.wait();
}


void glu_gelu_f16(sycl::queue& q, const void* x, void* out, std::size_t rows,
                  std::size_t d, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::glu_gelu_f16_sycl(q, x, out, rows, d, dt);
  if (blocking) ev.wait();
}

void glu_quant(sycl::queue& q, const void* x, std::uint8_t* out_q,
               float* out_scales, std::size_t rows, std::size_t d,
               std::size_t group, GluQuantMode mode, DType dt, Variant variant,
               bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::glu_quant_sycl(q, x, out_q, out_scales, rows, d,
                                           group, static_cast<int>(mode), dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
