#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// fp8 GEMV (M=1 fp8_gemm): y[N] = x_fp8[K] @ B_fp8[K,N] * scale, native SYCL
// decode (both fp8 kinds are bit-casts away from f16 — no vendor lib needed).
// kind: 0 = e4m3, 1 = e5m2.
sycl::event fp8_gemv_sycl(sycl::queue& q, const void* x_fp8, const void* b_fp8,
                          void* y, std::size_t N, std::size_t K, int kind,
                          float scale, DType out_dt);

sycl::event fp8_gemm_w8a16_sycl(sycl::queue &q, const void *activations, const void *weight_fp8,
                                const float *weight_scale, bool per_channel, void *out,
                                std::size_t M, std::size_t N, std::size_t K, int kind,
                                DType act_dt);

// kind: 0 = e4m3, 1 = e5m2.
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
// fp8 GEMM via oneDNN. Returns false (and leaves an unspecified event) if the
// device/library cannot build an fp8 matmul primitive; the caller reports it.
bool fp8_gemm_onednn(sycl::queue& q, const void* a, const void* b, void* c,
                     std::size_t M, std::size_t N, std::size_t K, int kind,
                     float scale, DType out_dt);

bool fp8_gemm_w8a16_onednn(sycl::queue &q, const void *activations, const void *weight_fp8,
                           const float *weight_scale, bool per_channel, void *out, std::size_t M,
                           std::size_t N, std::size_t K, int kind, DType act_dt);

// Codec helpers (oneDNN reorder) so tests need no hand-written fp8 encoder:
// f32 -> fp8 (out is 1 byte/elem) and fp8 -> f32.
void fp8_from_f32(sycl::queue& q, const float* in, void* out_fp8, std::size_t n, int kind);
void fp8_to_f32(sycl::queue& q, const void* in_fp8, float* out, std::size_t n, int kind);
bool fp8_supported();
#endif

}  // namespace quixicore::xpu::kernels
