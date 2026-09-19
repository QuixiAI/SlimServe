#pragma once

// QuixiCore XPU op ABI.
//
// This is the framework-agnostic launch surface for the backend: every op is a
// free function that takes a sycl::queue plus raw device pointers and shape
// metadata. It is the XPU analogue of the Metal backend's `tk::launch_*` layer.
//
// Both callers use the SAME entry points:
//   * the native C++ helpers (which allocate USM and manage a queue), and
//   * the PyTorch-XPU binding (which passes a Torch tensor's data pointer and
//     the tensor's own sycl::queue).
//
// Contract names and semantics are shared with the other QuixiCore backends;
// Intel-specific layout / subgroup / XMX choices stay inside the kernel
// variants under kernels/<family>/<operation>/variants/.
//
// Requires a SYCL toolchain; only compiled under QUIXICORE_XPU_ENABLE_SYCL.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"
#include "quixicore/xpu/variants.hpp"

namespace quixicore::xpu::ops {

// ----------------------------------------------------------------------------
// quantization
// ----------------------------------------------------------------------------

// GGUF routed GEMV / dequantize on ggml type ids (8 = Q8_0, 10 = Q2_K,
// 16 = IQ2_XXS), the GGUF serving formats. `w` is [E, N, row] on-disk
// blocks (E = 1 for a dense matrix); y[r, :] = W[expert_ids[r]] . x[r / top_k]
// for r < R (expert_ids null => dense, top_k = 1; negative id => zero row).
// `iq2xxs_grid_dev` is the 256-entry uint64 iq2xxs grid resident on the
// device (uploaded from tm_xpu/iq2xxs_grid_table.hpp), read only for
// IQ2_XXS. Native only.
bool gguf_routed_supports(int ggml_type);
void gguf_routed_gemv(sycl::queue& q, const void* w, const void* x, void* y,
                      const std::int32_t* expert_ids, std::size_t R,
                      std::size_t N, std::size_t K, std::size_t top_k,
                      int ggml_type, DType act_dt,
                      const std::uint64_t* iq2xxs_grid_dev, bool blocking = true);
void gguf_dequantize(sycl::queue& q, const void* w, void* out, std::size_t N,
                     std::size_t K, int ggml_type, DType out_dt,
                     const std::uint64_t* iq2xxs_grid_dev, bool blocking = true);
void q8_split_repack(sycl::queue& q, const void* raw, void* split,
                     std::size_t N, std::size_t K, bool blocking = true);
void q8_split_gemv(sycl::queue& q, const void* split, const void* x, void* y,
                   std::size_t R, std::size_t N, std::size_t K, DType act_dt,
                   bool blocking = true);
void q8_split_dequantize(sycl::queue& q, const void* split, void* out,
                         std::size_t N, std::size_t K, DType out_dt,
                         bool blocking = true);

// ----------------------------------------------------------------------------
// norms
// ----------------------------------------------------------------------------

// RMSNorm over the last axis of a [rows, dim] row-major tensor:
//   out[r, i] = x[r, i] * rsqrt(mean_i(x[r, :]^2) + eps) * weight[i]
// `x`, `out` are device pointers of dtype `dt` ([rows*dim]); `weight` is [dim]
// of dtype `dt`. Reduction accumulates in fp32. Deterministic family.
void rms_norm(sycl::queue& q, const void* x, const void* weight, void* out,
              std::size_t rows, std::size_t dim, float eps, DType dt,
              Variant variant = Variant::sycl, bool blocking = true);

// Fused residual add + RMSNorm. For each row, normalize the unrounded fp32 sum
// of x and residual, update residual in place with the storage-dtype sum, and
// write the normalized/weighted result to out. `out` must not alias `residual`.
void fused_add_rms_norm(sycl::queue &q, const void *x, void *residual, const void *weight,
                        void *out, std::size_t rows, std::size_t dim, float eps, DType dt,
                        Variant variant = Variant::sycl, bool blocking = true);

}  // namespace quixicore::xpu::ops
