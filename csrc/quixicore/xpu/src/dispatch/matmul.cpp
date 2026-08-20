// Dispatch layer for the matmul family. Default routing prefers the vendor
// (oneDNN, XMX-backed) path when available, since the native SYCL variant is an
// untuned SLM-tiled baseline; callers can force Variant::sycl.

#include "quixicore/xpu/ops.hpp"

#include "matmul/lora_apply/lora_apply_kernel.hpp"

#include "matmul/dense_gemm/dense_gemm_kernel.hpp"

namespace quixicore::xpu::ops {

void dense_gemm(sycl::queue& q, const void* a, const void* b, void* c,
                std::size_t M, std::size_t N, std::size_t K, DType dt,
                Variant variant, bool blocking) {
  // GEMM is compute-bound; the oneDNN XMX path far outperforms the untuned
  // native tile kernel, so `best` -> vendor when oneDNN is present.
  if (variant == Variant::best) {
    variant = variant_available(Variant::vendor) ? Variant::vendor : Variant::sycl;
  }
  const Variant v = resolve_variant(variant);

  sycl::event ev;
  switch (v) {
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      ev = kernels::dense_gemm_onednn(q, a, b, c, M, N, K, dt);
      break;
#endif
    case Variant::sycl:
    case Variant::best:
    default:
      ev = kernels::dense_gemm_sycl(q, a, b, c, M, N, K, dt);
      break;
  }
  if (blocking) ev.wait();
}

void lora_shrink(sycl::queue& q, const void* in, const void* w,
                 const std::int32_t* lora_idx, float* out, std::size_t batch,
                 std::size_t hidden, std::size_t rank, std::size_t n_loras,
                 float scale, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::lora_shrink_sycl(q, in, w, lora_idx, out, batch,
                                             hidden, rank, n_loras, scale, dt);
  if (blocking) ev.wait();
}

void lora_expand(sycl::queue& q, const float* in, const void* w,
                 const std::int32_t* lora_idx, void* out, std::size_t batch,
                 std::size_t rank, std::size_t out_dim, std::size_t n_loras,
                 std::size_t out_offset, std::size_t out_stride,
                 bool accumulate, DType dt, Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::lora_expand_sycl(q, in, w, lora_idx, out, batch,
                                             rank, out_dim, n_loras,
                                             out_offset, out_stride,
                                             accumulate ? 1 : 0, dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
