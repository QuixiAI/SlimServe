#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// type: 0 = q8_0 (34-byte block), 1 = q4_0 (18-byte block).
sycl::event gguf_gemv_sycl(sycl::queue& q, const void* w_blocks, const void* x,
                           void* y, std::size_t N, std::size_t K, int type,
                           DType act_dt);

}  // namespace quixicore::xpu::kernels

// ---- SlimServe DeepSeek-V4 GGUF path (gguf_routed.sycl.cpp) ----------------
// ggml type ids (not the GgufType enum above): 8 = Q8_0, 10 = Q2_K, 16 = IQ2_XXS.
namespace quixicore::xpu::kernels {

bool gguf_routed_supports(int ggml_type);

// y[r, :] = W[expert_ids ? expert_ids[r] : 0][:, :] . x[r / top_k, :]
// W is [E, N, row_bytes] (E = 1 for a dense matrix), x [R / top_k, K], y [R, N].
// Negative expert ids write zero rows. `iq2xxs_grid_dev` is the 256-entry
// uint64 iq2xxs grid on the device (only read for IQ2_XXS).
sycl::event gguf_routed_gemv_sycl(sycl::queue& q, const void* w, const void* x,
                                  void* y, const std::int32_t* expert_ids,
                                  std::size_t R, std::size_t N, std::size_t K,
                                  std::size_t top_k, int ggml_type, DType act_dt,
                                  const std::uint64_t* iq2xxs_grid_dev);

// out[n, k] = decode(W[n, k]), out dtype out_dt.
sycl::event gguf_dequantize_sycl(sycl::queue& q, const void* w, void* out,
                                 std::size_t N, std::size_t K, int ggml_type,
                                 DType out_dt, const std::uint64_t* iq2xxs_grid_dev);

}  // namespace quixicore::xpu::kernels
