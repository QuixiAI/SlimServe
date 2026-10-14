// Dispatch layer for the quantization family.

#include <stdexcept>

#include "quixicore/xpu/ops.hpp"

#include "quantization/gguf_gemv/gguf_kernel.hpp"

namespace quixicore::xpu::ops {

bool gguf_routed_supports(int ggml_type) {
  return kernels::gguf_routed_supports(ggml_type);
}

void gguf_routed_gemv(sycl::queue& q, const void* w, const void* x, void* y,
                      const std::int32_t* expert_ids, std::size_t R,
                      std::size_t N, std::size_t K, std::size_t top_k,
                      int ggml_type, DType act_dt,
                      const std::uint64_t* iq2xxs_grid_dev, bool blocking) {
  if (!kernels::gguf_routed_supports(ggml_type))
    throw std::invalid_argument("gguf_routed_gemv: unsupported ggml type");
  sycl::event ev = kernels::gguf_routed_gemv_sycl(
      q, w, x, y, expert_ids, R, N, K, top_k, ggml_type, act_dt, iq2xxs_grid_dev);
  if (blocking) ev.wait();
}

void gguf_dequantize(sycl::queue& q, const void* w, void* out, std::size_t N,
                     std::size_t K, int ggml_type, DType out_dt,
                     const std::uint64_t* iq2xxs_grid_dev, bool blocking) {
  if (!kernels::gguf_routed_supports(ggml_type))
    throw std::invalid_argument("gguf_dequantize: unsupported ggml type");
  sycl::event ev = kernels::gguf_dequantize_sycl(q, w, out, N, K, ggml_type,
                                                 out_dt, iq2xxs_grid_dev);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
