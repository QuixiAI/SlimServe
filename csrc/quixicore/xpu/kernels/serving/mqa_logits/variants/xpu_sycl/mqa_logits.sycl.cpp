// DeepSeek-style fp8 MQA indexer logits on the Xe tensor engine — the
// proving-ground consumer of the shared xmx_tile DPAS building block.
//
//   logits[s, kv] = sum_h w[s,h] * relu( (q[s,h,:] . kv[kv,:]) * kv_scale[kv] )
//   masked to -inf outside [ks[s], ke[s])
//
// One subgroup per (query token, 16-wide kv tile). Heads ride the DPAS M axis
// in 8-row tiles (rows past H are zero-staged and carry zero weight); D is
// walked in 16-deep steps. The e4m3 inputs are decoded to bf16 in the
// stage-tile lambdas (this generation has no fp8 MMA — the cutlass source
// converts before DPAS too, so nothing is lost natively). After each head
// tile the accumulator spill hands lane c its column: relu + kv_scale +
// weighted reduce happen in registers, since the 8x16 spill walk visits a
// fixed column per lane.
//
// Semantics from vllm-xpu-kernels csrc/xpu/mqa_logits (Apache, cutlass);
// independently expressed against raw joint_matrix.

#include "serving/mqa_logits/mqa_logits_kernel.hpp"

#include "common/quant_codecs.hpp"
#include "common/xmx_tile.hpp"

namespace quixicore::xpu::kernels {
namespace {

namespace xmx = quixicore::xpu::xmx;
namespace qcodec = quixicore::xpu::qcodec;

class MqaLogitsKernel;

}  // namespace

sycl::event mqa_logits_sycl(sycl::queue& q, const std::uint8_t* q_fp8,
                            const std::uint8_t* kv_fp8, const float* kv_scales,
                            const float* head_weights, const std::int32_t* ks,
                            const std::int32_t* ke, float* logits,
                            std::size_t S, std::size_t H, std::size_t D,
                            std::size_t Skv) {
  if (D % xmx::kTK != 0) return {};  // depth must tile; reject, don't corrupt
  const std::size_t kv_tiles = (Skv + xmx::kTN - 1) / xmx::kTN;
  const std::size_t head_tiles = (H + xmx::kTM - 1) / xmx::kTM;
  const std::size_t k_steps = D / xmx::kTK;
  const sycl::range<2> global(S, kv_tiles * xmx::kSG);
  const sycl::range<2> local(1, xmx::kSG);
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<bf16_t, 1> As(sycl::range<1>(xmx::kTM * xmx::kTK), h);
    sycl::local_accessor<bf16_t, 1> Bs(sycl::range<1>(xmx::kTK * xmx::kTN), h);
    sycl::local_accessor<float, 1> Cs(sycl::range<1>(xmx::kTM * xmx::kTN), h);
    h.parallel_for<MqaLogitsKernel>(
        sycl::nd_range<2>(global, local),
        [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(xmx::kSG)]] {
          const std::size_t s = it.get_group(0);
          const std::size_t kv0 = it.get_group(1) * xmx::kTN;
          const int lane = static_cast<int>(it.get_local_id(1));
          const sycl::sub_group sg = it.get_sub_group();

          float colsum = 0.0f;
          for (std::size_t ht = 0; ht < head_tiles; ++ht) {
            const std::size_t h0 = ht * xmx::kTM;
            xmx::MatC acc;
            sycl::ext::oneapi::experimental::matrix::joint_matrix_fill(sg, acc,
                                                                       0.0f);
            for (std::size_t kt = 0; kt < k_steps; ++kt) {
              const std::size_t k0 = kt * xmx::kTK;
              xmx::stage_tile<xmx::kTM, xmx::kTK>(
                  As, lane, [=](int r, int c) {
                    const std::size_t hh = h0 + static_cast<std::size_t>(r);
                    return hh < H ? static_cast<bf16_t>(qcodec::fp8_e4m3(
                                        q_fp8[(s * H + hh) * D + k0 + c]))
                                  : bf16_t(0.0f);
                  });
              xmx::stage_tile<xmx::kTK, xmx::kTN>(
                  Bs, lane, [=](int r, int c) {
                    const std::size_t kv = kv0 + static_cast<std::size_t>(c);
                    return kv < Skv ? static_cast<bf16_t>(qcodec::fp8_e4m3(
                                          kv_fp8[kv * D + k0 + r]))
                                    : bf16_t(0.0f);
                  });
              sycl::group_barrier(it.get_group());
              xmx::mad_step(sg, As, Bs, acc);
              sycl::group_barrier(it.get_group());
            }
            const std::size_t kv_lane = kv0 + static_cast<std::size_t>(lane);
            const float kscale = kv_lane < Skv ? kv_scales[kv_lane] : 0.0f;
            xmx::store_tile(it.get_group(), sg, Cs, lane,
                            acc, [&](int r, int c, float v) {
                              // spill walk: c == lane for every visit
                              const std::size_t hh =
                                  h0 + static_cast<std::size_t>(r);
                              if (hh < H) {
                                colsum += head_weights[s * H + hh] *
                                          sycl::fmax(v * kscale, 0.0f);
                              }
                              (void)c;
                            });
            sycl::group_barrier(it.get_group());
          }

          const std::size_t kv_lane = kv0 + static_cast<std::size_t>(lane);
          if (kv_lane < Skv) {
            const bool in_band =
                static_cast<std::int64_t>(kv_lane) >=
                    static_cast<std::int64_t>(ks[s]) &&
                static_cast<std::int64_t>(kv_lane) <
                    static_cast<std::int64_t>(ke[s]);
            logits[s * Skv + kv_lane] =
                in_band ? colsum
                        : -sycl::bit_cast<float>(0x7f800000u);  // -inf
          }
        });
  });
}

}  // namespace quixicore::xpu::kernels
