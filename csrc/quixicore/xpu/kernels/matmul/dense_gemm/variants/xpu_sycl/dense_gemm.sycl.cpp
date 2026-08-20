// Dense GEMM, native SYCL variant: shared-memory tiled, fp32 accumulation.
//
// A straightforward 16x16 SLM-tiled kernel -- correct and a reasonable native
// baseline, NOT yet tuned (no XMX/DPAS joint_matrix, no register blocking). The
// fast native path via joint_matrix is a later optimization pass; the oneDNN
// vendor variant already provides an XMX-backed path. Per the breadth-first
// directive, this ships as the native variant with perf deferred.

#include "matmul/dense_gemm/dense_gemm_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int TS = 16;  // tile size

template <typename T>
sycl::event gemm_typed(sycl::queue& q, const T* A, const T* B, T* C,
                       std::size_t M, std::size_t N, std::size_t K) {
  const std::size_t Mg = ((M + TS - 1) / TS) * TS;
  const std::size_t Ng = ((N + TS - 1) / TS) * TS;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 2> As({TS, TS}, h);
    sycl::local_accessor<float, 2> Bs({TS, TS}, h);
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(Mg, Ng), sycl::range<2>(TS, TS)),
        [=](sycl::nd_item<2> it) {
          const std::size_t row = it.get_global_id(0);
          const std::size_t col = it.get_global_id(1);
          const int lr = static_cast<int>(it.get_local_id(0));
          const int lc = static_cast<int>(it.get_local_id(1));

          float acc = 0.0f;
          const std::size_t ntiles = (K + TS - 1) / TS;
          for (std::size_t t = 0; t < ntiles; ++t) {
            const std::size_t kA = t * TS + lc;
            const std::size_t kB = t * TS + lr;
            As[lr][lc] = (row < M && kA < K) ? static_cast<float>(A[row * K + kA]) : 0.0f;
            Bs[lr][lc] = (kB < K && col < N) ? static_cast<float>(B[kB * N + col]) : 0.0f;
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int k = 0; k < TS; ++k) acc += As[lr][k] * Bs[k][lc];
            it.barrier(sycl::access::fence_space::local_space);
          }
          if (row < M && col < N) C[row * N + col] = static_cast<T>(acc);
        });
  });
}

}  // namespace

sycl::event dense_gemm_sycl(sycl::queue& q, const void* a, const void* b,
                            void* c, std::size_t M, std::size_t N, std::size_t K,
                            DType dt) {
  switch (dt) {
    case DType::f32:
      return gemm_typed(q, static_cast<const float*>(a),
                        static_cast<const float*>(b), static_cast<float*>(c), M, N, K);
    case DType::f16:
      return gemm_typed(q, static_cast<const half_t*>(a),
                        static_cast<const half_t*>(b), static_cast<half_t*>(c), M, N, K);
    case DType::bf16:
      return gemm_typed(q, static_cast<const bf16_t*>(a),
                        static_cast<const bf16_t*>(b), static_cast<bf16_t*>(c), M, N, K);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
