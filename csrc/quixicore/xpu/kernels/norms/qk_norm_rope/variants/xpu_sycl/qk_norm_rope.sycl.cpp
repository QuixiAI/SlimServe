// Fused per-head QK-norm + RoPE (+ optional f16 convert), native SYCL.
//
// Ported from embeddinggemma.c src/engine_xpu.cpp launch_qk_norm_rope: per-head
// RMSNorm of Q and K by a learned weight, a query-only scale, and NeoX
// half-split RoPE, in one launch. One subgroup owns each (token, head); the
// subgroup reduces sum(x^2) for the RMS scale (fp32), then each lane rotates its
// strided pairs (i, i+head_dim/2). The RoPE angle is computed on the fly from
// `base` and a contiguous position (pos0 + token), matching rope.sycl.cpp so the
// unfused A/B baseline composes rms_norm + rope directly. Weight and inverse-RMS
// (with the query scale folded in) are applied before the rotation exactly as
// the reference does. The richer launch_qkv_epilogue (which also encodes the
// model's fused-QKV memory layout) was intentionally NOT ported.

#include "norms/qk_norm_rope/qk_norm_rope_kernel.hpp"

#include <cmath>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSub = 16;  // Arc subgroup width used across the backend.

template <typename T>
sycl::event qk_norm_rope_typed(sycl::queue& q, T* Q, T* K, const T* q_weight,
                               const T* k_weight, half_t* Q_f16, half_t* K_f16,
                               std::size_t tokens, std::size_t n_head,
                               std::size_t n_head_kv, std::size_t head_dim,
                               float base, std::size_t pos0, float query_scale,
                               float eps) {
  const std::size_t combined = n_head + n_head_kv;
  const std::size_t tasks = tokens * combined;
  const std::size_t half = head_dim / 2;
  // freq = base^(-2i/head_dim) = exp2(i * kc); kc folds log2(base) host-side
  // (matches rope.sycl.cpp -- one exp2 instead of pow's exp+log pair).
  const float kc = -2.0f * std::log2(base) / static_cast<float>(head_dim);
  return q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(tasks * kSub), sycl::range<1>(kSub)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSub)]] {
        const sycl::sub_group sg = item.get_sub_group();
        const std::size_t task = item.get_group(0);
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const std::size_t token = task / combined;
        const std::size_t ch = task - token * combined;
        const bool is_key = ch >= n_head;
        const std::size_t head = is_key ? ch - n_head : ch;
        const std::size_t heads = is_key ? n_head_kv : n_head;
        T* values = is_key ? K : Q;
        half_t* half_out = is_key ? K_f16 : Q_f16;
        const T* weight = is_key ? k_weight : q_weight;
        const std::size_t bufbase = (token * heads + head) * head_dim;

        // RMS over the head-dim vector (subgroup reduction), query scale folded.
        float ss = 0.0f;
        for (std::size_t d = lane; d < head_dim; d += kSub) {
          const float v = static_cast<float>(values[bufbase + d]);
          ss = sycl::fma(v, v, ss);
        }
        ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
        const float scale = is_key ? 1.0f : query_scale;
        const float inv =
            sycl::rsqrt(ss / static_cast<float>(head_dim) + eps) * scale;

        // NeoX half-split RoPE on the normalized (weighted, scaled) values.
        const float pos = static_cast<float>(pos0 + token);
        for (std::size_t i = lane; i < half; i += kSub) {
          const float freq = sycl::exp2(static_cast<float>(i) * kc);
          const float angle = pos * freq;
          float c;
          const float s = sycl::sincos(angle, sycl::private_ptr<float>(&c));
          const float x0 =
              static_cast<float>(values[bufbase + i]) *
              static_cast<float>(weight[i]) * inv;
          const float x1 =
              static_cast<float>(values[bufbase + i + half]) *
              static_cast<float>(weight[i + half]) * inv;
          const float r0 = x0 * c - x1 * s;
          const float r1 = x0 * s + x1 * c;
          values[bufbase + i] = static_cast<T>(r0);
          values[bufbase + i + half] = static_cast<T>(r1);
          if (half_out != nullptr) {
            half_out[bufbase + i] = static_cast<half_t>(r0);
            half_out[bufbase + i + half] = static_cast<half_t>(r1);
          }
        }
      });
}

}  // namespace

sycl::event qk_norm_rope_sycl(sycl::queue& q, void* Q, void* K,
                              const void* q_weight, const void* k_weight,
                              void* Q_f16, void* K_f16, std::size_t tokens,
                              std::size_t n_head, std::size_t n_head_kv,
                              std::size_t head_dim, float base, std::size_t pos0,
                              float query_scale, float eps, DType dt) {
  auto* qh = static_cast<half_t*>(Q_f16);
  auto* kh = static_cast<half_t*>(K_f16);
  switch (dt) {
    case DType::f32:
      return qk_norm_rope_typed(
          q, static_cast<float*>(Q), static_cast<float*>(K),
          static_cast<const float*>(q_weight), static_cast<const float*>(k_weight),
          qh, kh, tokens, n_head, n_head_kv, head_dim, base, pos0, query_scale, eps);
    case DType::f16:
      return qk_norm_rope_typed(
          q, static_cast<half_t*>(Q), static_cast<half_t*>(K),
          static_cast<const half_t*>(q_weight), static_cast<const half_t*>(k_weight),
          qh, kh, tokens, n_head, n_head_kv, head_dim, base, pos0, query_scale, eps);
    case DType::bf16:
      return qk_norm_rope_typed(
          q, static_cast<bf16_t*>(Q), static_cast<bf16_t*>(K),
          static_cast<const bf16_t*>(q_weight), static_cast<const bf16_t*>(k_weight),
          qh, kh, tokens, n_head, n_head_kv, head_dim, base, pos0, query_scale, eps);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
