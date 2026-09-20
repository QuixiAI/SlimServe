// DeepSeek-V4 manifold hyper-connections, post stage — contract dsv4_hc_post.
//
//   out[t, o, h] = sum_i comb_res_mix[t, i, o] * residual[t, i, h]
//                + post_mix[t, o] * x[t, h]
//
// residual/out are [tokens, n_streams, hidden] of dtype dt (n_streams = the
// hyper-connection count, 4 for DSV4), x [tokens, hidden], comb_res_mix
// [tokens, n_streams, n_streams] and post_mix [tokens, n_streams] fp32. One
// work-item per (t, o, h), fp32 accumulation over the small stream axis.
//
// Semantics from vllm-xpu-kernels csrc/xpu/mhc/xe_2/mhc_post.cpp (Apache;
// translated — the math is the file's own header formula). The pre and comb
// stages are deferred; see the D-wave ledger in perf/optimization_status.md.

#include "ssm/dsv4_hc/dsv4_hc_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kMaxStreams = 8;

template <typename T>
class Dsv4HcPostKernel;

template <typename T>
sycl::event post_typed(sycl::queue& q, const float* comb, const T* residual,
                       const float* post_mix, const T* x, T* out,
                       std::size_t tokens, std::size_t n_streams,
                       std::size_t hidden) {
  return q.parallel_for<Dsv4HcPostKernel<T>>(
      sycl::range<1>(tokens * n_streams * hidden), [=](sycl::id<1> idx) {
        const std::size_t h = idx[0] % hidden;
        const std::size_t to = idx[0] / hidden;
        const std::size_t t = to / n_streams;
        const std::size_t o = to % n_streams;
        float acc = post_mix[t * n_streams + o] *
                    static_cast<float>(x[t * hidden + h]);
        for (std::size_t i = 0; i < n_streams; ++i) {
          acc += comb[(t * n_streams + i) * n_streams + o] *
                 static_cast<float>(
                     residual[(t * n_streams + i) * hidden + h]);
        }
        out[idx[0]] = static_cast<T>(acc);
      });
}

}  // namespace

sycl::event dsv4_hc_post_sycl(sycl::queue& q, const float* comb_res_mix,
                              const void* residual, const float* post_mix,
                              const void* x, void* out, std::size_t tokens,
                              std::size_t n_streams, std::size_t hidden,
                              DType dt) {
  if (n_streams == 0 || n_streams > kMaxStreams) return {};
  switch (dt) {
    case DType::f32:
      return post_typed(q, comb_res_mix, static_cast<const float*>(residual),
                        post_mix, static_cast<const float*>(x),
                        static_cast<float*>(out), tokens, n_streams, hidden);
    case DType::f16:
      return post_typed(q, comb_res_mix, static_cast<const half_t*>(residual),
                        post_mix, static_cast<const half_t*>(x),
                        static_cast<half_t*>(out), tokens, n_streams, hidden);
    case DType::bf16:
      return post_typed(q, comb_res_mix, static_cast<const bf16_t*>(residual),
                        post_mix, static_cast<const bf16_t*>(x),
                        static_cast<bf16_t*>(out), tokens, n_streams, hidden);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
