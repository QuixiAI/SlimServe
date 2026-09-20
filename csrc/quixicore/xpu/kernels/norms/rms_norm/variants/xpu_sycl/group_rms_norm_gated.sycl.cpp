// Gated group-RMSNorm (the Mamba-2 mixer output norm), native SYCL variant.
//
// y = x * silu(gate) in fp32, then RMS-normalize each of n_groups contiguous
// slices of the hidden dim independently, round to the storage dtype, and
// multiply by the learned weight (matching the torch reference's rounding
// order: x.to(dtype) THEN weight). rms_norm=false skips normalization and
// weight entirely (out = x * silu(gate)). Single-device semantics: the
// tensor-parallel n_groups==1 cross-rank variance reduction is
// integration-owned. One work-group per (row, group), reduce_over_group fp32
// accumulation; gated values are recomputed on the output pass rather than
// staged (correctness-first; SLM staging is a recorded perf lever).
//
// Semantics from vLLM's Mixer2RMSNormGated.forward_native (the path that runs
// eager on XPU today); independently expressed.

#include "norms/norms_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kGroupThreads = 256;

inline float grng_silu(float v) { return v / (1.0f + sycl::exp(-v)); }

template <typename T>
class GroupRmsNormGatedKernel;

template <typename T>
sycl::event group_rms_norm_gated_typed(sycl::queue& q, const T* x,
                                       const T* gate, const T* w, T* out,
                                       std::size_t rows, std::size_t hidden,
                                       std::size_t n_groups, float eps,
                                       bool rms_norm) {
  const std::size_t group_size = hidden / n_groups;
  const sycl::nd_range<2> ndr(
      sycl::range<2>(rows, n_groups * kGroupThreads),
      sycl::range<2>(1, kGroupThreads));
  return q.parallel_for<GroupRmsNormGatedKernel<T>>(
      ndr, [=](sycl::nd_item<2> it) {
        const std::size_t row = it.get_group(0);
        const std::size_t g = it.get_group(1);
        const std::size_t lid = it.get_local_id(1);
        const std::size_t base = row * hidden + g * group_size;

        if (!rms_norm) {
          for (std::size_t i = lid; i < group_size; i += kGroupThreads) {
            const float y = static_cast<float>(x[base + i]) *
                            grng_silu(static_cast<float>(gate[base + i]));
            out[base + i] = static_cast<T>(y);
          }
          return;
        }

        float partial = 0.0f;
        for (std::size_t i = lid; i < group_size; i += kGroupThreads) {
          const float y = static_cast<float>(x[base + i]) *
                          grng_silu(static_cast<float>(gate[base + i]));
          partial += y * y;
        }
        const float sumsq = sycl::reduce_over_group(
            it.get_group(), partial, sycl::plus<float>());
        const float inv =
            sycl::rsqrt(sumsq / static_cast<float>(group_size) + eps);

        for (std::size_t i = lid; i < group_size; i += kGroupThreads) {
          const float y = static_cast<float>(x[base + i]) *
                          grng_silu(static_cast<float>(gate[base + i]));
          const float rounded = static_cast<float>(static_cast<T>(y * inv));
          out[base + i] = static_cast<T>(
              static_cast<float>(w[g * group_size + i]) * rounded);
        }
      });
}

}  // namespace

sycl::event group_rms_norm_gated_sycl(sycl::queue& q, const void* x,
                                      const void* gate, const void* weight,
                                      void* out, std::size_t rows,
                                      std::size_t hidden, std::size_t n_groups,
                                      float eps, bool rms_norm, DType dt) {
  switch (dt) {
    case DType::f32:
      return group_rms_norm_gated_typed(
          q, static_cast<const float*>(x), static_cast<const float*>(gate),
          static_cast<const float*>(weight), static_cast<float*>(out), rows,
          hidden, n_groups, eps, rms_norm);
    case DType::f16:
      return group_rms_norm_gated_typed(
          q, static_cast<const half_t*>(x), static_cast<const half_t*>(gate),
          static_cast<const half_t*>(weight), static_cast<half_t*>(out), rows,
          hidden, n_groups, eps, rms_norm);
    case DType::bf16:
      return group_rms_norm_gated_typed(
          q, static_cast<const bf16_t*>(x), static_cast<const bf16_t*>(gate),
          static_cast<const bf16_t*>(weight), static_cast<bf16_t*>(out), rows,
          hidden, n_groups, eps, rms_norm);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
