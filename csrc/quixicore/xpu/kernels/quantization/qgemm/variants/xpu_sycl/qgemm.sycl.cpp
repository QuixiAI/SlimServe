// int8 w8a8 GEMM, native SYCL: 16x16 SLM-tiled, int32 accumulation, then apply
// per-row (activation) and per-col (weight) fp32 scales. Correct, untuned
// baseline (no DPAS joint_matrix yet -- that is the flagged native optimization;
// the oneDNN vendor variant already provides the XMX int8 path).

#include "quantization/qgemm/qgemm_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int TS = 16;

template <typename OT>
sycl::event qgemm_typed(sycl::queue& q, const std::int8_t* A, const std::int8_t* B,
                        const float* as, const float* bs, OT* C, std::size_t M,
                        std::size_t N, std::size_t K) {
  const std::size_t Mg = ((M + TS - 1) / TS) * TS;
  const std::size_t Ng = ((N + TS - 1) / TS) * TS;
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<std::int8_t, 2> As({TS, TS}, h);
    sycl::local_accessor<std::int8_t, 2> Bs({TS, TS}, h);
    h.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(Mg, Ng), sycl::range<2>(TS, TS)),
        [=](sycl::nd_item<2> it) {
          const std::size_t row = it.get_global_id(0);
          const std::size_t col = it.get_global_id(1);
          const int lr = static_cast<int>(it.get_local_id(0));
          const int lc = static_cast<int>(it.get_local_id(1));

          int acc = 0;
          const std::size_t ntiles = (K + TS - 1) / TS;
          for (std::size_t t = 0; t < ntiles; ++t) {
            const std::size_t kA = t * TS + lc;
            const std::size_t kB = t * TS + lr;
            As[lr][lc] = (row < M && kA < K) ? A[row * K + kA] : std::int8_t{0};
            Bs[lr][lc] = (kB < K && col < N) ? B[kB * N + col] : std::int8_t{0};
            it.barrier(sycl::access::fence_space::local_space);
#pragma unroll
            for (int k = 0; k < TS; ++k)
              acc += static_cast<int>(As[lr][k]) * static_cast<int>(Bs[k][lc]);
            it.barrier(sycl::access::fence_space::local_space);
          }
          if (row < M && col < N)
            C[row * N + col] = static_cast<OT>(static_cast<float>(acc) * as[row] * bs[col]);
        });
  });
}

}  // namespace

sycl::event qgemm_int8_sycl(sycl::queue& q, const void* a, const void* b,
                            const void* a_scale, const void* b_scale, void* c,
                            std::size_t M, std::size_t N, std::size_t K,
                            DType out_dt) {
  const auto* A = static_cast<const std::int8_t*>(a);
  const auto* B = static_cast<const std::int8_t*>(b);
  const auto* as = static_cast<const float*>(a_scale);
  const auto* bs = static_cast<const float*>(b_scale);
  switch (out_dt) {
    case DType::f32:
      return qgemm_typed(q, A, B, as, bs, static_cast<float*>(c), M, N, K);
    case DType::f16:
      return qgemm_typed(q, A, B, as, bs, static_cast<half_t*>(c), M, N, K);
    case DType::bf16:
      return qgemm_typed(q, A, B, as, bs, static_cast<bf16_t*>(c), M, N, K);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
