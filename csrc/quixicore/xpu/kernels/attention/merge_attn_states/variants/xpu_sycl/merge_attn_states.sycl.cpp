// LSE-weighted merge of two partial attention results (split-KV / prefix +
// suffix decomposition; the consumer of paged_attention_prefill's LSE
// output). Per (row, channel):
//   m  = max(lse_a, lse_b)
//   wa = exp(lse_a - m), wb = exp(lse_b - m)
//   out = (wa * out_a + wb * out_b) / (wa + wb)
//   lse_out = m + log(wa + wb)                       (optional)
// A partition with lse == -inf contributes zero weight — the empty-partition
// guard whose absence caused NaNs in the CUDA lineage of this op. All-empty
// rows produce zero output and -inf lse.
//
// Semantics per vllm-xpu-kernels csrc/attention/merge_attn_states.cpp
// (Apache; translated).

#include "attention/merge_attn_states/merge_attn_states_kernel.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

template <typename T>
class MergeAttnStatesKernel;

template <typename T>
sycl::event merge_typed(sycl::queue& q, const T* out_a, const float* lse_a,
                        const T* out_b, const float* lse_b, T* out,
                        float* lse_out, std::size_t rows, std::size_t d) {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  return q.parallel_for<MergeAttnStatesKernel<T>>(
      sycl::range<1>(rows * d), [=](sycl::id<1> idx) {
        const std::size_t r = idx[0] / d;
        const float la = lse_a[r];
        const float lb = lse_b[r];
        const float m = sycl::fmax(la, lb);
        const bool empty = !(m > neg_inf);
        const float wa = (empty || !(la > neg_inf)) ? 0.0f : sycl::exp(la - m);
        const float wb = (empty || !(lb > neg_inf)) ? 0.0f : sycl::exp(lb - m);
        const float denom = wa + wb;
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        out[idx[0]] = static_cast<T>(
            (wa * static_cast<float>(out_a[idx[0]]) +
             wb * static_cast<float>(out_b[idx[0]])) *
            inv);
        if (lse_out != nullptr && idx[0] % d == 0) {
          lse_out[r] = empty ? neg_inf : m + sycl::log(denom);
        }
      });
}

}  // namespace

sycl::event merge_attn_states_sycl(sycl::queue& q, const void* out_a,
                                   const float* lse_a, const void* out_b,
                                   const float* lse_b, void* out,
                                   float* lse_out, std::size_t rows,
                                   std::size_t d, DType dt) {
  switch (dt) {
    case DType::f32:
      return merge_typed(q, static_cast<const float*>(out_a), lse_a,
                         static_cast<const float*>(out_b), lse_b,
                         static_cast<float*>(out), lse_out, rows, d);
    case DType::f16:
      return merge_typed(q, static_cast<const half_t*>(out_a), lse_a,
                         static_cast<const half_t*>(out_b), lse_b,
                         static_cast<half_t*>(out), lse_out, rows, d);
    case DType::bf16:
      return merge_typed(q, static_cast<const bf16_t*>(out_a), lse_a,
                         static_cast<const bf16_t*>(out_b), lse_b,
                         static_cast<bf16_t*>(out), lse_out, rows, d);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
