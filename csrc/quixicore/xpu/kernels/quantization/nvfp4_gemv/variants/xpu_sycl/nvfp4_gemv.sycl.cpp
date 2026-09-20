// nvfp4 GEMV, native SYCL decoder. e2m1 (fp4) elements + e4m3 (fp8) block scale
// per 16 + a per-tensor fp32 global scale. Same subgroup-per-row / 16-byte-load
// structure as mxfp4; a 16-byte chunk (32 fp4) spans TWO 16-element blocks, so
// two e4m3 scales per chunk. Both fields bit-relocate exactly into f16 (no LUT,
// no ldexp) -- nvfp4 is a data encoding, decoded natively here.

#include "quantization/nvfp4_gemv/nvfp4_kernel.hpp"

#include <stdexcept>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;

using WVec = sycl::vec<std::uint32_t, 4>;  // 16 bytes = 32 fp4 = two nvfp4 blocks

// Bit-relocation decodes (the fp8_gemv trick, extended to fp4). Both fields of
// the nvfp4 format are bit-casts away from f16, so the hot loop needs no LUT,
// no sign select, and no ldexp:
//   e2m1 nibble s.e1e0.m -> f16 (s<<15)|(e<<10)|(m<<9) = value * 2^-14
//     (exact incl. the subnormal: m*0.5 -> f16 subnormal m*2^-15)
//   e4m3 byte -> f16 ((b&0x80)<<8)|((b&0x7f)<<7)       = value * 2^-8
// Every product therefore carries a uniform 2^-22, compensated by ONE host-side
// multiply folded into the global scale (see nvfp4_gemv_sycl).

// Pair decode: nibbles j and j+4 of a word (pass word>>4j), two f16 patterns
// built in one masked-shift pass, one vector convert -> 2 floats.
inline sycl::vec<float, 2> e2m1_dec2(std::uint32_t w) {
  const std::uint32_t t = w & 0x000F000Fu;
  const std::uint32_t h2 = ((t & 0x00080008u) << 12) | ((t & 0x00070007u) << 9);
  return sycl::bit_cast<sycl::vec<sycl::half, 2>>(h2).template convert<float>();
}

inline float e4m3_dec_raw(std::uint8_t b) {
  const auto h = static_cast<std::uint16_t>(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
  return static_cast<float>(sycl::bit_cast<sycl::half>(h));
}

template <typename T>
sycl::event nvfp4_typed(sycl::queue& q, const std::uint8_t* w,
                        const std::uint8_t* bscale, float gscale, const T* x,
                        T* y, std::size_t N, std::size_t K) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t blocks_per_row = K / 16;     // e4m3 scales per row
  const std::size_t nchunks = bytes_per_row / 16;  // 16-byte chunks (2 blocks each)
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

    // Decode is ALU/latency-bound (was 17% of BW roofline). All decode is
    // bit-relocation into f16 (see e2m1_dec2): 4 masked-shift pair passes per
    // word instead of 8 LUT+sign selects; the two block scales stay hoisted
    // (one mul per 16-elem block) and each word accumulates independently (ILP).
    // gscale arrives pre-multiplied by 2^22 to undo the raw-decode factors.
    float acc = 0.0f;
    for (std::size_t c = lane; c < nchunks; c += kSG) {
      const WVec chunk = wvrow[c];
      // Two 16-element blocks in this chunk -> two e4m3 scales.
      const float s0 = e4m3_dec_raw(srow[2 * c]) * gscale;
      const float s1 = e4m3_dec_raw(srow[2 * c + 1]) * gscale;
      const T* xc = x + c * 32;
      float aw[4];
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        const std::uint32_t word = chunk[wi];
        const sycl::vec<T, 8> xv = *reinterpret_cast<const sycl::vec<T, 8>*>(xc + wi * 8);
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int p = 0; p < 4; ++p) {  // pair p decodes nibbles p and p+4
          const sycl::vec<float, 2> v = e2m1_dec2(word >> (4 * p));
          a0 += v[0] * static_cast<float>(xv[p]);
          a1 += v[1] * static_cast<float>(xv[p + 4]);
        }
        aw[wi] = a0 + a1;
      }
      acc += (aw[0] + aw[1]) * s0 + (aw[2] + aw[3]) * s1;
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

constexpr std::size_t kMaxM = 8;

template <typename T> class Nvfp4GemmMKernel;

template <typename T>
sycl::event nvfp4_mtiled_typed(sycl::queue &q, const std::uint8_t *w, const std::uint8_t *bscale,
                               float gscale, const T *x, T *y, std::size_t M, std::size_t N,
                               std::size_t K) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t blocks_per_row = K / 16;
  const std::size_t nchunks = bytes_per_row / 16;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for<Nvfp4GemmMKernel<T>>(
      ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
        if (n >= N)
          return;

        const WVec *wvrow = reinterpret_cast<const WVec *>(w + n * bytes_per_row);
        const std::uint8_t *srow = bscale + n * blocks_per_row;
        float acc[kMaxM] = {};
        for (std::size_t c = lane; c < nchunks; c += kSG) {
          const WVec chunk = wvrow[c];
          const float s0 = e4m3_dec_raw(srow[2 * c]) * gscale;
          const float s1 = e4m3_dec_raw(srow[2 * c + 1]) * gscale;
          float decoded[32];
#pragma unroll
          for (int wi = 0; wi < 4; ++wi) {
            const std::uint32_t word = chunk[wi];
            const float scale = wi < 2 ? s0 : s1;
#pragma unroll
            for (int p = 0; p < 4; ++p) {
              const sycl::vec<float, 2> value = e2m1_dec2(word >> (4 * p));
              decoded[wi * 8 + p] = value[0] * scale;
              decoded[wi * 8 + p + 4] = value[1] * scale;
            }
          }
          const std::size_t base = c * 32;
          for (std::size_t m = 0; m < M; ++m) {
            const T *xrow = x + m * K + base;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; ++i) {
              dot += decoded[i] * static_cast<float>(xrow[i]);
            }
            acc[m] += dot;
          }
        }
        for (std::size_t m = 0; m < M; ++m) {
          const float sum = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
          if (lane == 0)
            y[m * N + n] = static_cast<T>(sum);
        }
      });
}

}  // namespace

sycl::event nvfp4_gemv_sycl(sycl::queue& q, const void* w_packed,
                            const void* block_scales, float global_scale,
                            const void* x, void* y, std::size_t N, std::size_t K,
                            DType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const std::uint8_t*>(block_scales);
  // Raw bit-relocation decodes carry 2^-14 (e2m1) * 2^-8 (e4m3 scale) per
  // product; compensate once here instead of per element on the device.
  global_scale *= 4194304.0f;  // 2^22
  switch (act_dt) {
    case DType::f32:
      return nvfp4_typed(q, w, s, global_scale, static_cast<const float*>(x), static_cast<float*>(y), N, K);
    case DType::f16:
      return nvfp4_typed(q, w, s, global_scale, static_cast<const half_t*>(x), static_cast<half_t*>(y), N, K);
    case DType::bf16:
      return nvfp4_typed(q, w, s, global_scale, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(y), N, K);
  }
  return {};
}

sycl::event nvfp4_gemm_sycl(sycl::queue &q, const void *w_packed, const void *block_scales,
                            float global_scale, const void *x, void *y, std::size_t M,
                            std::size_t N, std::size_t K, DType act_dt) {
  const auto *w = static_cast<const std::uint8_t *>(w_packed);
  const auto *s = static_cast<const std::uint8_t *>(block_scales);
  global_scale *= 4194304.0f;
  sycl::event last;
  switch (act_dt) {
  case DType::f32:
    for (std::size_t m = 0; m < M; ++m) {
      last = nvfp4_typed(q, w, s, global_scale, static_cast<const float *>(x) + m * K,
                         static_cast<float *>(y) + m * N, N, K);
    }
    break;
  case DType::f16:
    for (std::size_t m = 0; m < M; ++m) {
      last = nvfp4_typed(q, w, s, global_scale, static_cast<const half_t *>(x) + m * K,
                         static_cast<half_t *>(y) + m * N, N, K);
    }
    break;
  case DType::bf16:
    for (std::size_t m = 0; m < M; ++m) {
      last = nvfp4_typed(q, w, s, global_scale, static_cast<const bf16_t *>(x) + m * K,
                         static_cast<bf16_t *>(y) + m * N, N, K);
    }
    break;
  }
  return last;
}

sycl::event nvfp4_gemm_mtiled_sycl(sycl::queue &q, const void *w_packed, const void *block_scales,
                                   float global_scale, const void *x, void *y, std::size_t M,
                                   std::size_t N, std::size_t K, DType act_dt) {
  if (M == 0 || M > kMaxM) {
    throw std::invalid_argument("nvfp4 M-tiled kernel requires 1 <= M <= 8");
  }
  const auto *w = static_cast<const std::uint8_t *>(w_packed);
  const auto *s = static_cast<const std::uint8_t *>(block_scales);
  global_scale *= 4194304.0f;
  switch (act_dt) {
  case DType::f32:
    return nvfp4_mtiled_typed(q, w, s, global_scale, static_cast<const float *>(x),
                              static_cast<float *>(y), M, N, K);
  case DType::f16:
    return nvfp4_mtiled_typed(q, w, s, global_scale, static_cast<const half_t *>(x),
                              static_cast<half_t *>(y), M, N, K);
  case DType::bf16:
    return nvfp4_mtiled_typed(q, w, s, global_scale, static_cast<const bf16_t *>(x),
                              static_cast<bf16_t *>(y), M, N, K);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
