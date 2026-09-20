// Segmented per-expert GEMM with fused weight dequant, on the native DPAS
// building block (kernels/common/xmx_tile.hpp) — the cutlass-free grouped
// MoE GEMM (contract grouped_gemm / moe_grouped_qgemm).
//
//   C[r, :] = A[r, :] . dequant(W[expert(r)])^T for rows sorted by expert;
//   rows_per_expert [E] (device int32, zeros allowed) segments A/C.
//
// Graph-safe segmentation: a static worst-case grid of ceil(M/TM)+E m-tiles
// x ceil(N/TN) n-tiles is launched; every subgroup walks the per-expert
// prefix (E <= 256 ints read through SLM once per work-group) to map its
// linear m-tile to (expert, expert-local row range) and exits when past the
// real tile count — the host never reads rows_per_expert (no sync, no
// allocation). Weight formats are stage-lambda decoders:
//   w16:   W [E, N, K] act dtype
//   int4:  W [E, N, K/2] u8 (low nibble = even k), f16 scales [E, N, K/group]
//   nvfp4: W [E, N, K/2] u8 e2m1, u8 e4m3 scales [E, N, K/16] LINEAR
//          (de-swizzled at load), per-expert f32 global scale applied in the
//          fp32 epilogue (exact — a per-tensor scalar factors out).
// Semantics from the vllm-xpu-kernels cutlass grouped GEMM + its local NVFP4
// graft (translated); the w4a16_gemm kernel is the tiling precedent.

#include "moe/grouped_qgemm/grouped_qgemm_kernel.hpp"

#include "common/quant_codecs.hpp"
#include "common/xmx_tile.hpp"

namespace quixicore::xpu::kernels {
namespace {

namespace xmx = quixicore::xpu::xmx;
namespace qc = quixicore::xpu::qcodec;

constexpr std::size_t kMaxExperts = 256;

// e2m1 nibble decode via the shared codec table.
inline float nv_nibble(std::uint8_t byte, std::size_t k) {
  return qc::e2m1((k & 1) ? static_cast<std::uint8_t>(byte >> 4)
                          : static_cast<std::uint8_t>(byte & 0x0f));
}

template <typename T, int Fmt>
class GroupedQgemmKernel;

template <typename T, int Fmt>
sycl::event grouped_typed(sycl::queue& q, const T* A, const void* W,
                          const void* scales, const float* global_scales, T* C,
                          const std::int32_t* rows_per_expert,
                          std::size_t M_total, std::size_t N, std::size_t K,
                          std::size_t E, std::size_t group) {
  const std::size_t mtiles_max = (M_total + xmx::kTM - 1) / xmx::kTM + E;
  const std::size_t ntiles = (N + xmx::kTN - 1) / xmx::kTN;
  const std::size_t ktiles = K / xmx::kTK;
  const std::size_t w_estride =
      Fmt == 0 ? N * K : N * (K / 2);  // elements (w16) or bytes (packed)
  const std::size_t s_estride =
      Fmt == 1 ? N * (K / group) : (Fmt == 2 ? N * (K / 16) : 0);
  const sycl::range<2> global(mtiles_max, ntiles * xmx::kSG);
  const sycl::range<2> local(1, xmx::kSG);
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<T, 1> As(sycl::range<1>(xmx::kTM * xmx::kTK), h);
    sycl::local_accessor<T, 1> Bs(sycl::range<1>(xmx::kTK * xmx::kTN), h);
    sycl::local_accessor<float, 1> Cs(sycl::range<1>(xmx::kTM * xmx::kTN), h);
    sycl::local_accessor<std::int32_t, 1> seg(sycl::range<1>(kMaxExperts + 1),
                                              h);
    h.parallel_for<GroupedQgemmKernel<T, Fmt>>(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(xmx::kSG)]] {
          const int lane = static_cast<int>(it.get_local_id(1));
          const sycl::sub_group sg = it.get_sub_group();
          // Cooperative prefix of rows_per_expert into SLM: seg[e] = first
          // GLOBAL m-tile index of expert e; each expert's tiles cover its
          // rows padded to TM (the +E grid slack).
          if (lane == 0) {
            std::int32_t tile0 = 0;
            for (std::size_t e = 0; e < E; ++e) {
              seg[e] = tile0;
              tile0 += (rows_per_expert[e] + xmx::kTM - 1) / xmx::kTM;
            }
            seg[E] = tile0;
          }
          sycl::group_barrier(it.get_group());

          const std::int32_t mt =
              static_cast<std::int32_t>(it.get_group(0));
          if (mt >= seg[E]) return;  // grid slack past the real tile count
          // Locate the owning expert (E small; linear walk).
          std::size_t e = 0;
          while (e + 1 < E && mt >= seg[e + 1]) ++e;
          // Expert-local and global row ranges.
          std::int32_t row0 = 0;
          for (std::size_t ee = 0; ee < e; ++ee) row0 += rows_per_expert[ee];
          const std::int32_t rlocal0 = (mt - seg[e]) * xmx::kTM;
          const std::int32_t rows_e = rows_per_expert[e];
          const std::size_t col0 = it.get_group(1) * xmx::kTN;

          const std::uint8_t* Wq = static_cast<const std::uint8_t*>(W);
          const T* W16 = static_cast<const T*>(W);
          const sycl::half* s16 = static_cast<const sycl::half*>(scales);
          const std::uint8_t* s8 = static_cast<const std::uint8_t*>(scales);

          xmx::MatC acc;
          sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sg, acc,
                                                                     0.0f);
          for (std::size_t kt = 0; kt < ktiles; ++kt) {
            const std::size_t k0 = kt * xmx::kTK;
            xmx::stage_tile<xmx::kTM, xmx::kTK>(As, lane, [=](int r, int c) {
              const std::int32_t rl = rlocal0 + r;
              return rl < rows_e
                         ? A[(static_cast<std::size_t>(row0 + rl)) * K + k0 +
                             static_cast<std::size_t>(c)]
                         : T(0.0f);
            });
            xmx::stage_tile<xmx::kTK, xmx::kTN>(Bs, lane, [=](int r, int c) {
              const std::size_t n = col0 + static_cast<std::size_t>(c);
              if (n >= N) return T(0.0f);
              const std::size_t k = k0 + static_cast<std::size_t>(r);
              if constexpr (Fmt == 0) {
                return W16[e * w_estride + n * K + k];
              } else if constexpr (Fmt == 1) {
                const std::uint8_t byte = Wq[e * w_estride + n * (K / 2) + k / 2];
                const int nib = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
                const float sc = static_cast<float>(
                    s16[e * s_estride + n * (K / group) + k / group]);
                return static_cast<T>(static_cast<float>(qc::s4(nib)) * sc);
              } else {
                const std::uint8_t byte = Wq[e * w_estride + n * (K / 2) + k / 2];
                const float sc =
                    qc::fp8_e4m3(s8[e * s_estride + n * (K / 16) + k / 16]);
                return static_cast<T>(nv_nibble(byte, k) * sc);
              }
            });
            sycl::group_barrier(it.get_group());
            xmx::mad_step(sg, As, Bs, acc);
            sycl::group_barrier(it.get_group());
          }

          const float gscale =
              Fmt == 2 && global_scales != nullptr ? global_scales[e] : 1.0f;
          xmx::store_tile(it.get_group(), sg, Cs, lane, acc,
                          [=](int r, int c, float v) {
                            const std::int32_t rl = rlocal0 + r;
                            const std::size_t n =
                                col0 + static_cast<std::size_t>(c);
                            if (rl < rows_e && n < N) {
                              C[(static_cast<std::size_t>(row0 + rl)) * N + n] =
                                  static_cast<T>(v * gscale);
                            }
                          });
        });
  });
}

}  // namespace

sycl::event moe_grouped_qgemm_sycl(sycl::queue& q, const void* A,
                                   const void* W, const void* scales,
                                   const float* global_scales, void* C,
                                   const std::int32_t* rows_per_expert,
                                   std::size_t M_total, std::size_t N,
                                   std::size_t K, std::size_t E,
                                   std::size_t group, int fmt, DType act_dt) {
  if (E > kMaxExperts || K % xmx::kTK != 0 ||
      (fmt == 1 && (group == 0 || K % group != 0))) {
    return {};  // reject, don't corrupt
  }
#define QX_GG(T, F)                                                           \
  return grouped_typed<T, F>(q, static_cast<const T*>(A), W, scales,          \
                             global_scales, static_cast<T*>(C),               \
                             rows_per_expert, M_total, N, K, E, group)
  switch (act_dt) {
    case DType::f16:
      if (fmt == 0) QX_GG(half_t, 0);
      if (fmt == 1) QX_GG(half_t, 1);
      QX_GG(half_t, 2);
    case DType::bf16:
      if (fmt == 0) QX_GG(bf16_t, 0);
      if (fmt == 1) QX_GG(bf16_t, 1);
      QX_GG(bf16_t, 2);
    case DType::f32:
      return {};  // DPAS operands are 16-bit; f32 activations unsupported
  }
#undef QX_GG
  return {};
}

}  // namespace quixicore::xpu::kernels
