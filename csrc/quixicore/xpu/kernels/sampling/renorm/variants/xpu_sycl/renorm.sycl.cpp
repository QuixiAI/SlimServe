// Top-k and top-p (nucleus) probability renormalization — contract
// top_k_renorm / top_p_renorm. Threshold-based formulation (the
// FlashInfer-style shape vLLM's renorm kernels use):
//   top-k: find the k-th largest probability by iterative masked max
//          (k <= 64), keep p >= threshold (ties at the pivot INCLUDED),
//          zero the rest, renormalize by the kept sum.
//   top-p: binary-search the largest threshold t whose kept mass
//          sum(p_i : p_i >= t) still reaches top_p (32 fp32 halvings),
//          then keep/renormalize as above — equivalent to the minimal
//          sorted-prefix definition with ties included.
// One work-item per row, sequential vocab scans — correctness-first and
// exactly replicable by the host oracle; a subgroup-cooperative variant is
// the recorded lever. Inputs are probabilities (>= 0); rows with zero kept
// mass pass through unchanged.

#include "sampling/renorm/renorm_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxK = 64;

template <typename T>
class TopKRenormKernel;
template <typename T>
class TopPRenormKernel;

template <typename T>
sycl::event topk_typed(sycl::queue& q, const T* probs, T* out,
                       std::size_t rows, std::size_t vocab, int k) {
  return q.parallel_for<TopKRenormKernel<T>>(
      sycl::range<1>(rows), [=](sycl::id<1> idx) {
        const T* row = probs + idx[0] * vocab;
        T* orow = out + idx[0] * vocab;
        // k-th largest via iterative masked max.
        float thresh = 0.0f;
        float prev = 3.4e38f;
        for (int i = 0; i < k; ++i) {
          float best = -1.0f;
          for (std::size_t v = 0; v < vocab; ++v) {
            const float p = static_cast<float>(row[v]);
            if (p < prev && p > best) best = p;
          }
          if (best < 0.0f) break;  // fewer than k distinct values
          thresh = best;
          prev = best;
        }
        float kept = 0.0f;
        for (std::size_t v = 0; v < vocab; ++v) {
          const float p = static_cast<float>(row[v]);
          if (p >= thresh) kept += p;
        }
        const float inv = kept > 0.0f ? 1.0f / kept : 0.0f;
        for (std::size_t v = 0; v < vocab; ++v) {
          const float p = static_cast<float>(row[v]);
          orow[v] = kept > 0.0f
                        ? static_cast<T>(p >= thresh ? p * inv : 0.0f)
                        : row[v];
        }
      });
}

template <typename T>
sycl::event topp_typed(sycl::queue& q, const T* probs, T* out,
                       std::size_t rows, std::size_t vocab, float top_p) {
  return q.parallel_for<TopPRenormKernel<T>>(
      sycl::range<1>(rows), [=](sycl::id<1> idx) {
        const T* row = probs + idx[0] * vocab;
        T* orow = out + idx[0] * vocab;
        float pmax = 0.0f;
        for (std::size_t v = 0; v < vocab; ++v)
          pmax = sycl::fmax(pmax, static_cast<float>(row[v]));
        // Largest threshold whose kept mass still reaches top_p.
        float lo = 0.0f, hi = pmax;
        for (int it = 0; it < 32; ++it) {
          const float mid = 0.5f * (lo + hi);
          float mass = 0.0f;
          for (std::size_t v = 0; v < vocab; ++v) {
            const float p = static_cast<float>(row[v]);
            if (p >= mid) mass += p;
          }
          if (mass >= top_p) lo = mid; else hi = mid;
        }
        float kept = 0.0f;
        for (std::size_t v = 0; v < vocab; ++v) {
          const float p = static_cast<float>(row[v]);
          if (p >= lo) kept += p;
        }
        const float inv = kept > 0.0f ? 1.0f / kept : 0.0f;
        for (std::size_t v = 0; v < vocab; ++v) {
          const float p = static_cast<float>(row[v]);
          orow[v] = kept > 0.0f
                        ? static_cast<T>(p >= lo ? p * inv : 0.0f)
                        : row[v];
        }
      });
}

}  // namespace

sycl::event top_k_renorm_sycl(sycl::queue& q, const void* probs, void* out,
                              std::size_t rows, std::size_t vocab, int k,
                              DType dt) {
  if (k < 1 || k > kMaxK) return {};  // reject, don't corrupt
  switch (dt) {
    case DType::f32:
      return topk_typed(q, static_cast<const float*>(probs),
                        static_cast<float*>(out), rows, vocab, k);
    case DType::f16:
      return topk_typed(q, static_cast<const half_t*>(probs),
                        static_cast<half_t*>(out), rows, vocab, k);
    case DType::bf16:
      return topk_typed(q, static_cast<const bf16_t*>(probs),
                        static_cast<bf16_t*>(out), rows, vocab, k);
  }
  return {};
}

sycl::event top_p_renorm_sycl(sycl::queue& q, const void* probs, void* out,
                              std::size_t rows, std::size_t vocab, float top_p,
                              DType dt) {
  switch (dt) {
    case DType::f32:
      return topp_typed(q, static_cast<const float*>(probs),
                        static_cast<float*>(out), rows, vocab, top_p);
    case DType::f16:
      return topp_typed(q, static_cast<const half_t*>(probs),
                        static_cast<half_t*>(out), rows, vocab, top_p);
    case DType::bf16:
      return topp_typed(q, static_cast<const bf16_t*>(probs),
                        static_cast<bf16_t*>(out), rows, vocab, top_p);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
