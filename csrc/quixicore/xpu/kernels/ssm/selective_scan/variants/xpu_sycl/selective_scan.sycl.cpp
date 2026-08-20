// Mamba selective scan (S6) forward, native SYCL. One work-item per channel; the
// state vector h[state] lives in registers and the scan runs sequentially over
// the sequence (the recurrence is inherently serial in t). Parallelism comes
// from the many channels running independent scans. fp32 recurrence.
//
// This is the least GPU-friendly kernel shape (sequential dependence); a chunked
// parallel scan (ssd_chunk_*) is the optimization, deferred to the ssm depth wave.

#include "ssm/selective_scan/selective_scan_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxState = 16;

template <typename T>
sycl::event scan_typed(sycl::queue& q, const T* u, const T* delta, const T* A,
                       const T* B, const T* C, const T* D, T* y,
                       std::size_t n_chan, std::size_t seq, std::size_t state) {
  return q.parallel_for(sycl::range<1>(n_chan), [=](sycl::id<1> idx) {
    const std::size_t c = idx[0];
    float h[kMaxState];
    for (std::size_t s = 0; s < state; ++s) h[s] = 0.0f;
    const float Dc = static_cast<float>(D[c]);

    for (std::size_t t = 0; t < seq; ++t) {
      const float dt_ = static_cast<float>(delta[c * seq + t]);
      const float ut = static_cast<float>(u[c * seq + t]);
      float yt = 0.0f;
      for (std::size_t s = 0; s < state; ++s) {
        const float dA = sycl::exp(dt_ * static_cast<float>(A[c * state + s]));
        const float dBu = dt_ * static_cast<float>(B[t * state + s]) * ut;
        h[s] = dA * h[s] + dBu;
        yt += static_cast<float>(C[t * state + s]) * h[s];
      }
      y[c * seq + t] = static_cast<T>(yt + Dc * ut);
    }
  });
}

}  // namespace

sycl::event selective_scan_sycl(sycl::queue& q, const void* u, const void* delta,
                                const void* A, const void* B, const void* C,
                                const void* D, void* y, std::size_t n_chan,
                                std::size_t seq, std::size_t state, DType dt) {
  switch (dt) {
    case DType::f32:
      return scan_typed(q, static_cast<const float*>(u), static_cast<const float*>(delta),
                        static_cast<const float*>(A), static_cast<const float*>(B),
                        static_cast<const float*>(C), static_cast<const float*>(D),
                        static_cast<float*>(y), n_chan, seq, state);
    case DType::f16:
      return scan_typed(q, static_cast<const half_t*>(u), static_cast<const half_t*>(delta),
                        static_cast<const half_t*>(A), static_cast<const half_t*>(B),
                        static_cast<const half_t*>(C), static_cast<const half_t*>(D),
                        static_cast<half_t*>(y), n_chan, seq, state);
    case DType::bf16:
      return scan_typed(q, static_cast<const bf16_t*>(u), static_cast<const bf16_t*>(delta),
                        static_cast<const bf16_t*>(A), static_cast<const bf16_t*>(B),
                        static_cast<const bf16_t*>(C), static_cast<const bf16_t*>(D),
                        static_cast<bf16_t*>(y), n_chan, seq, state);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
