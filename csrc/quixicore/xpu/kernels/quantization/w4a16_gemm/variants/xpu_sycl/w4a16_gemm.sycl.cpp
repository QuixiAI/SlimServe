// W4A16 GEMM (int4 group-quantized weight x f16/bf16 activation) on the Intel
// Xe tensor engine (DPAS) via the SYCL joint_matrix extension.
//
//   C[M,N] = A[M,K] . dequant(W)^T,   W is [N,K] int4 group-quant.
//
// This is the genuinely missing int4-weight x 16-bit-activation shape on the
// box: it currently has int4 only as a batch-1 GEMV (qgemv_int4) and an int8
// w8a8 GEMM whose native variant is a scalar SLM tile (no DPAS joint_matrix).
// Ported in spirit from embeddinggemma.c's EI_XPU_XE2_W4 path
// (engine_xpu_w4.cpp launch_w4 -> xe_gemm_4bits with w4a16_policy_m_{8,16,32}),
// but written natively against SYCL joint_matrix -- NOT against that path's
// cutlass-sycl / cute (xe_gemm_4bits, XE_DPAS_TT) headers, so it pulls in no
// external GEMM library. A bf16 joint_matrix MMA was verified to compile and run
// bit-exact on the Arc Pro B60 with this oneAPI (worst_abs=0) before this kernel
// was built -- the feasibility probe that unblocked the native route.
//
// Tiling: one subgroup (SG=16 lanes) owns one C output tile of TM x TN = 8 x 16.
// It loops K in TK=16 steps; per step it stages the A tile and the DEQUANTIZED
// weight tile into SLM (both zero-padded on the M/N/K edges, so any shape is
// correct), then joint_matrix_load + joint_matrix_mad accumulates in fp32. The
// weight tile is transposed during dequant so B is [TK,TN] row-major (bt[kk][nn]
// = dequant(W[n0+nn][k0+kk])), matching the joint_matrix use::b row-major load.
// The int4 encoding is identical to qgemv_int4: W packed 2 nibbles/byte (low
// nibble = even k, high = odd k, signed two's-complement), f16 scales [N,K/group],
// dequant(W)[n,k] = s4(nibble) * scales[n, k/group].
//
// Shape: int4-weight x f16/bf16-activation DPAS GEMM (small-M decode), D-tiled.

#include "quantization/w4a16_gemm/w4a16_gemm_kernel.hpp"

#include <sycl/ext/oneapi/matrix/matrix.hpp>

namespace quixicore::xpu::kernels {
namespace {

using namespace sycl::ext::oneapi::experimental::matrix;

constexpr int TM = 8;   // DPAS tile rows (M)
constexpr int TN = 16;  // DPAS tile cols (N)
constexpr int TK = 16;  // DPAS tile depth (K)
constexpr int SG = 16;  // subgroup width

// Sign-extend a 4-bit two's-complement nibble (matches qgemv_int4::s4).
inline int s4(int nib) { return nib >= 8 ? nib - 16 : nib; }

template <typename T>
sycl::event w4a16_typed(sycl::queue& q, const T* A, const std::uint8_t* W,
                        const half_t* scales, T* C, std::size_t M, std::size_t N,
                        std::size_t K, std::size_t group) {
  const std::size_t bpr = K / 2;      // packed weight bytes per row
  const std::size_t gpr = K / group;  // scale groups per weight row
  const std::size_t mtiles = (M + TM - 1) / TM;
  const std::size_t ntiles = (N + TN - 1) / TN;
  const std::size_t ktiles = (K + TK - 1) / TK;
  const sycl::range<2> global(mtiles, ntiles * SG);  // one subgroup per tile
  const sycl::range<2> local(1, SG);
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<T, 1> As(sycl::range<1>(TM * TK), h);
    sycl::local_accessor<T, 1> Bs(sycl::range<1>(TK * TN), h);
    sycl::local_accessor<float, 1> Cs(sycl::range<1>(TM * TN), h);
    h.parallel_for(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
          const std::size_t m0 = it.get_group(0) * TM;
          const std::size_t n0 = it.get_group(1) * TN;
          const int lane = static_cast<int>(it.get_local_id(1));
          const sycl::sub_group sg = it.get_sub_group();

          joint_matrix<sycl::sub_group, T, use::a, TM, TK, layout::row_major> ma;
          joint_matrix<sycl::sub_group, T, use::b, TK, TN, layout::row_major> mb;
          joint_matrix<sycl::sub_group, float, use::accumulator, TM, TN> mc;
          joint_matrix_fill(sg, mc, 0.0f);

          for (std::size_t kt = 0; kt < ktiles; ++kt) {
            const std::size_t k0 = kt * TK;
            // Stage A tile [TM][TK] into SLM (zero-pad OOB rows/cols).
            for (int e = lane; e < TM * TK; e += SG) {
              const std::size_t gm = m0 + e / TK, gk = k0 + e % TK;
              As[e] = (gm < M && gk < K) ? A[gm * K + gk] : T(0.0f);
            }
            // Stage DEQUANTIZED weight tile [TK][TN] into SLM, transposed so
            // bt[kk][nn] = dequant(W[n0+nn][k0+kk]) (joint_matrix use::b layout).
            for (int e = lane; e < TK * TN; e += SG) {
              const std::size_t gk = k0 + e / TN, gn = n0 + e % TN;
              T val = T(0.0f);
              if (gk < K && gn < N) {
                const std::uint8_t byte = W[gn * bpr + gk / 2];
                const int nib = (gk & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                const float sc = static_cast<float>(scales[gn * gpr + gk / group]);
                val = static_cast<T>(static_cast<float>(s4(nib)) * sc);
              }
              Bs[e] = val;
            }
            it.barrier(sycl::access::fence_space::local_space);
            joint_matrix_load(sg, ma,
                              As.template get_multi_ptr<sycl::access::decorated::no>(), TK);
            joint_matrix_load(sg, mb,
                              Bs.template get_multi_ptr<sycl::access::decorated::no>(), TN);
            joint_matrix_mad(sg, mc, ma, mb, mc);
            it.barrier(sycl::access::fence_space::local_space);
          }
          // Store the fp32 accumulator to SLM, then masked global write in act_dt.
          joint_matrix_store(sg, mc,
                             Cs.template get_multi_ptr<sycl::access::decorated::no>(),
                             TN, layout::row_major);
          it.barrier(sycl::access::fence_space::local_space);
          for (int e = lane; e < TM * TN; e += SG) {
            const std::size_t gm = m0 + e / TN, gn = n0 + e % TN;
            if (gm < M && gn < N) C[gm * N + gn] = static_cast<T>(Cs[e]);
          }
        });
  });
}

}  // namespace

sycl::event w4a16_gemm_sycl(sycl::queue& q, const void* A, const void* w_packed,
                            const void* scales, void* C, std::size_t M,
                            std::size_t N, std::size_t K, std::size_t group,
                            DType act_dt) {
  const auto* W = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const half_t*>(scales);
  switch (act_dt) {
    case DType::f16:
      return w4a16_typed(q, static_cast<const half_t*>(A), W, s,
                         static_cast<half_t*>(C), M, N, K, group);
    case DType::bf16:
      return w4a16_typed(q, static_cast<const bf16_t*>(A), W, s,
                         static_cast<bf16_t*>(C), M, N, K, group);
    case DType::f32:
      // a16 means 16-bit activation; f32 activation is out of contract.
      return {};
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
