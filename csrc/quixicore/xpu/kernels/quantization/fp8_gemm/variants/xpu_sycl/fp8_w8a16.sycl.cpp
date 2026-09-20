// W8A16 decode GEMV for checkpoint-native fp8 weights in [N,K] layout.

#include "quantization/fp8_gemm/fp8_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;

using U8x16 = sycl::vec<std::uint8_t, 16>;

template <int Kind> inline float decode_fp8(std::uint8_t value) {
  if constexpr (Kind == 1) {
    return static_cast<float>(sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(value << 8)));
  }
  const auto bits = static_cast<std::uint16_t>(((value & 0x80u) << 8) | ((value & 0x7fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

template <typename T, int Kind> class Fp8W8A16Kernel;

template <typename T, int Kind>
sycl::event fp8_w8a16_typed(sycl::queue &q, const T *activations, const std::uint8_t *weight,
                            const float *weight_scale, bool per_channel, T *out, std::size_t M,
                            std::size_t N, std::size_t K) {
  const std::size_t chunks = K / 16;
  const std::size_t groups = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<2> range(sycl::range<2>(M, groups * kWG), sycl::range<2>(1, kWG));
  const float compensation = Kind == 0 ? 256.0f : 1.0f;
  return q.parallel_for<Fp8W8A16Kernel<T, Kind>>(
      range, [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(kSG)]] {
        const std::size_t m = item.get_global_id(0);
        const sycl::sub_group subgroup = item.get_sub_group();
        const int subgroup_id = static_cast<int>(subgroup.get_group_linear_id());
        const int lane = static_cast<int>(subgroup.get_local_linear_id());
        const std::size_t n = item.get_group(1) * kRowsPerWG + subgroup_id;
        if (n >= N)
          return;

        const T *x = activations + m * K;
        const std::uint8_t *w = weight + n * K;
        float partial = 0.0f;
        for (std::size_t chunk = lane; chunk < chunks; chunk += kSG) {
          const U8x16 packed = *reinterpret_cast<const U8x16 *>(w + chunk * 16);
          const sycl::vec<T, 8> x0 = *reinterpret_cast<const sycl::vec<T, 8> *>(x + chunk * 16);
          const sycl::vec<T, 8> x1 = *reinterpret_cast<const sycl::vec<T, 8> *>(x + chunk * 16 + 8);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            partial += decode_fp8<Kind>(packed[i]) * static_cast<float>(x0[i]);
          }
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            partial += decode_fp8<Kind>(packed[i + 8]) * static_cast<float>(x1[i]);
          }
        }
        const float sum = sycl::reduce_over_group(subgroup, partial, sycl::plus<float>());
        if (lane == 0) {
          const float scale = (per_channel ? weight_scale[n] : weight_scale[0]) * compensation;
          out[m * N + n] = static_cast<T>(sum * scale);
        }
      });
}

template <typename T>
sycl::event dispatch_kind(sycl::queue &q, const void *activations, const void *weight_fp8,
                          const float *weight_scale, bool per_channel, void *out, std::size_t M,
                          std::size_t N, std::size_t K, int kind) {
  const auto *x = static_cast<const T *>(activations);
  const auto *w = static_cast<const std::uint8_t *>(weight_fp8);
  auto *y = static_cast<T *>(out);
  if (kind == 1) {
    return fp8_w8a16_typed<T, 1>(q, x, w, weight_scale, per_channel, y, M, N, K);
  }
  return fp8_w8a16_typed<T, 0>(q, x, w, weight_scale, per_channel, y, M, N, K);
}

} // namespace

sycl::event fp8_gemm_w8a16_sycl(sycl::queue &q, const void *activations, const void *weight_fp8,
                                const float *weight_scale, bool per_channel, void *out,
                                std::size_t M, std::size_t N, std::size_t K, int kind,
                                DType act_dt) {
  switch (act_dt) {
  case DType::f32:
    return dispatch_kind<float>(q, activations, weight_fp8, weight_scale, per_channel, out, M, N, K,
                                kind);
  case DType::f16:
    return dispatch_kind<half_t>(q, activations, weight_fp8, weight_scale, per_channel, out, M, N,
                                 K, kind);
  case DType::bf16:
    return dispatch_kind<bf16_t>(q, activations, weight_fp8, weight_scale, per_channel, out, M, N,
                                 K, kind);
  }
  return {};
}

} // namespace quixicore::xpu::kernels
