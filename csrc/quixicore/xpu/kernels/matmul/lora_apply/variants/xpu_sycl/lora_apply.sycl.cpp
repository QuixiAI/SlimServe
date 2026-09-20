// Batched-gather LoRA matvecs (BGMV) — contract lora_apply. Each token row
// selects its adapter by lora_idx[b] (-1 = no adapter: shrink writes zeros,
// expand leaves/keeps the base output).
//   shrink: out[b, r] = scale * sum_h in[b, h] * A[idx[b], r, h]   (out f32)
//   expand: out[b, out_offset + j] (+)= sum_r in[b, r] * B[idx[b], j, r]
// expand's slice offset/stride cover the fused-QKV split-destination form
// (bgmv_expand_slice); `accumulate` adds into the existing output (the
// LoRA-on-base pattern). fp32 math; one work-item per output element
// (rank is small, hidden loops are the shrink cost — subgroup-cooperative
// shrink is the recorded lever).
//
// Semantics from vllm-xpu-kernels csrc/xpu/lora (Apache; translated).

#include "matmul/lora_apply/lora_apply_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

template <typename T>
class LoraShrinkKernel;
template <typename T>
class LoraExpandKernel;

template <typename T>
sycl::event shrink_typed(sycl::queue& q, const T* in, const T* w,
                         const std::int32_t* lora_idx, float* out,
                         std::size_t batch, std::size_t hidden,
                         std::size_t rank, std::size_t n_loras, float scale) {
  return q.parallel_for<LoraShrinkKernel<T>>(
      sycl::range<1>(batch * rank), [=](sycl::id<1> idx) {
        const std::size_t b = idx[0] / rank;
        const std::size_t r = idx[0] % rank;
        const std::int32_t l = lora_idx[b];
        if (l < 0 || static_cast<std::size_t>(l) >= n_loras) {
          out[idx[0]] = 0.0f;
          return;
        }
        const T* wrow = w + (static_cast<std::size_t>(l) * rank + r) * hidden;
        const T* xrow = in + b * hidden;
        float acc = 0.0f;
        for (std::size_t h = 0; h < hidden; ++h)
          acc += static_cast<float>(xrow[h]) * static_cast<float>(wrow[h]);
        out[idx[0]] = acc * scale;
      });
}

template <typename T>
sycl::event expand_typed(sycl::queue& q, const float* in, const T* w,
                         const std::int32_t* lora_idx, T* out,
                         std::size_t batch, std::size_t rank,
                         std::size_t out_dim, std::size_t n_loras,
                         std::size_t out_offset, std::size_t out_stride,
                         bool accumulate) {
  return q.parallel_for<LoraExpandKernel<T>>(
      sycl::range<1>(batch * out_dim), [=](sycl::id<1> idx) {
        const std::size_t b = idx[0] / out_dim;
        const std::size_t j = idx[0] % out_dim;
        const std::int32_t l = lora_idx[b];
        T* dst = out + b * out_stride + out_offset + j;
        if (l < 0 || static_cast<std::size_t>(l) >= n_loras) return;
        const T* wrow = w + (static_cast<std::size_t>(l) * out_dim + j) * rank;
        const float* xrow = in + b * rank;
        float acc = accumulate ? static_cast<float>(*dst) : 0.0f;
        for (std::size_t r = 0; r < rank; ++r)
          acc += xrow[r] * static_cast<float>(wrow[r]);
        *dst = static_cast<T>(acc);
      });
}

}  // namespace

sycl::event lora_shrink_sycl(sycl::queue& q, const void* in, const void* w,
                             const std::int32_t* lora_idx, float* out,
                             std::size_t batch, std::size_t hidden,
                             std::size_t rank, std::size_t n_loras,
                             float scale, DType dt) {
  switch (dt) {
    case DType::f32:
      return shrink_typed(q, static_cast<const float*>(in),
                          static_cast<const float*>(w), lora_idx, out, batch,
                          hidden, rank, n_loras, scale);
    case DType::f16:
      return shrink_typed(q, static_cast<const half_t*>(in),
                          static_cast<const half_t*>(w), lora_idx, out, batch,
                          hidden, rank, n_loras, scale);
    case DType::bf16:
      return shrink_typed(q, static_cast<const bf16_t*>(in),
                          static_cast<const bf16_t*>(w), lora_idx, out, batch,
                          hidden, rank, n_loras, scale);
  }
  return {};
}

sycl::event lora_expand_sycl(sycl::queue& q, const float* in, const void* w,
                             const std::int32_t* lora_idx, void* out,
                             std::size_t batch, std::size_t rank,
                             std::size_t out_dim, std::size_t n_loras,
                             std::size_t out_offset, std::size_t out_stride,
                             int accumulate, DType dt) {
  switch (dt) {
    case DType::f32:
      return expand_typed(q, in, static_cast<const float*>(w), lora_idx,
                          static_cast<float*>(out), batch, rank, out_dim,
                          n_loras, out_offset, out_stride, accumulate != 0);
    case DType::f16:
      return expand_typed(q, in, static_cast<const half_t*>(w), lora_idx,
                          static_cast<half_t*>(out), batch, rank, out_dim,
                          n_loras, out_offset, out_stride, accumulate != 0);
    case DType::bf16:
      return expand_typed(q, in, static_cast<const bf16_t*>(w), lora_idx,
                          static_cast<bf16_t*>(out), batch, rank, out_dim,
                          n_loras, out_offset, out_stride, accumulate != 0);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
