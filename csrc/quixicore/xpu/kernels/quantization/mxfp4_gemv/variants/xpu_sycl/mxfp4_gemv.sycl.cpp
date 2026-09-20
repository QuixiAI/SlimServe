// mxfp4 GEMV, native SYCL decoder. e2m1 (fp4) elements + e8m0 power-of-two block
// scale (block 32). Reuses the tuned qgemv structure: one 32-wide subgroup per
// output row, 16-byte wide loads. Conveniently, a 16-byte chunk is exactly 32
// nibbles = one mxfp4 block, so each chunk applies exactly one e8m0 scale.
//
// mxfp4 is a data encoding, not silicon: the e2m1 magnitudes are a fixed 8-entry
// table and the block scale is a power of two. This decodes natively on Intel.

#include "quantization/mxfp4_gemv/mxfp4_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;

// e2m1 magnitude table (bits e1 e0 m0): {0,.5,1,1.5,2,3,4,6}.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

using WVec = sycl::vec<std::uint32_t, 4>;  // 16 bytes = 32 fp4 = one mx block

template <typename T>
sycl::event mxfp4_typed(sycl::queue& q, const std::uint8_t* w,
                        const std::uint8_t* bscale, const T* x, T* y,
                        std::size_t N, std::size_t K) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t blocks_per_row = K / 32;      // == nchunks
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;

    const WVec* wvrow = reinterpret_cast<const WVec*>(w + n * bytes_per_row);
    const std::uint8_t* srow = bscale + n * blocks_per_row;

    // Decode is ALU/latency-bound (was 18% of BW roofline). One e8m0 scale per
    // block (= per chunk), so hoist it to a single mul per chunk; vectorize the
    // activation read (sycl::vec<T,8>); accumulate each word independently for ILP.
    float acc = 0.0f;
    for (std::size_t c = lane; c < blocks_per_row; c += kSG) {
      const WVec chunk = wvrow[c];
      const float scale = sycl::exp2(static_cast<float>(srow[c]) - 127.0f);
      const T* xc = x + c * 32;
      float aw[4];
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        const std::uint32_t word = chunk[wi];
        const sycl::vec<T, 8> xv = *reinterpret_cast<const sycl::vec<T, 8>*>(xc + wi * 8);
        float a = 0.0f;
#pragma unroll
        for (int nib = 0; nib < 8; ++nib) {
          const int n4 = (word >> (nib * 4)) & 0xF;
          float val = kE2M1[n4 & 7];
          if (n4 & 8) val = -val;
          a += val * static_cast<float>(xv[nib]);
        }
        aw[wi] = a;
      }
      acc += ((aw[0] + aw[1]) + (aw[2] + aw[3])) * scale;
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

}  // namespace

sycl::event mxfp4_gemv_sycl(sycl::queue& q, const void* w_packed,
                            const void* block_scales, const void* x, void* y,
                            std::size_t N, std::size_t K, DType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const std::uint8_t*>(block_scales);
  switch (act_dt) {
    case DType::f32:
      return mxfp4_typed(q, w, s, static_cast<const float*>(x), static_cast<float*>(y), N, K);
    case DType::f16:
      return mxfp4_typed(q, w, s, static_cast<const half_t*>(x), static_cast<half_t*>(y), N, K);
    case DType::bf16:
      return mxfp4_typed(q, w, s, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(y), N, K);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
