// Fused SwiGLU + activation quantization, native SYCL variant.
//
// Input x is [rows, 2*d] (gate half then value half, the glu layout);
// y = silu(gate) * value in fp32, then quantized per mode:
//   group_fp8: per-`group` absmax -> fp32 scale = max(absmax/448, floor),
//              e4m3 codes (the DeepSeek-style block activation quant)
//   mxfp4:     per-32-group fp32 power-of-two scale + packed e2m1 nibbles
// One work-group per row, one subgroup per quant group (strided); the gated
// value is recomputed after the absmax pass rather than staged. Encode steps
// are the shared integer-exact codecs. Chains match vllm-xpu-kernels
// fused_silu_mul_{block,mxfp4}_quant (Apache; translated).

#include "activations/glu_quant/glu_quant_kernel.hpp"
#include "quantization/turboquant/turboquant_codec.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0f / (448.0f * 512.0f);
constexpr float kFp4Max = 6.0f;
constexpr float kMxEps = 1e-10f;

namespace codec = quixicore::xpu::turboquant_codec;

inline std::uint8_t e2m1_encode(float x) {
  const float a = sycl::fabs(x);
  const int code = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) +
                   (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  return static_cast<std::uint8_t>(code | (x < 0.0f ? 0x8 : 0x0));
}

template <typename T, int Mode>
class GluQuantKernel;

template <typename T, int Mode>
sycl::event glu_quant_typed(sycl::queue& q, const T* x, std::uint8_t* out_q,
                            float* out_scales, std::size_t rows, std::size_t d,
                            std::size_t group) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.parallel_for<GluQuantKernel<T, Mode>>(
      ndr, [=](sycl::nd_item<1> it) {
        const std::size_t row = it.get_group(0);
        const T* gate = x + row * 2 * d;
        const T* val = gate + d;
        const sycl::sub_group sg = it.get_sub_group();
        const std::size_t sg_size = sg.get_local_linear_range();
        const std::size_t nsg = kRowThreads / sg_size;
        const std::size_t sg_id = it.get_local_linear_id() / sg_size;
        const std::size_t lane = sg.get_local_linear_id();
        const std::size_t ngroups = d / group;

        auto y_at = [=](std::size_t i) {
          const float g = static_cast<float>(gate[i]);
          return (g / (1.0f + sycl::exp(-g))) * static_cast<float>(val[i]);
        };

        for (std::size_t gi = sg_id; gi < ngroups; gi += nsg) {
          float amax = Mode == 1 ? kMxEps : 0.0f;
          for (std::size_t j = lane; j < group; j += sg_size)
            amax = sycl::fmax(amax, sycl::fabs(y_at(gi * group + j)));
          amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
          if constexpr (Mode == 0) {
            const float scale = sycl::fmax(amax / kFp8Max, kMinScale);
            if (lane == 0) out_scales[row * ngroups + gi] = scale;
            const float inv = 1.0f / scale;
            for (std::size_t j = lane; j < group; j += sg_size) {
              const std::size_t i = gi * group + j;
              const float qv = sycl::fmax(
                  sycl::fmin(y_at(i) * inv, kFp8Max), -kFp8Max);
              out_q[row * d + i] = codec::e4m3_encode(qv);
            }
          } else {
            const float ys = sycl::exp2(
                sycl::ceil(sycl::log2(sycl::fmax(amax / kFp4Max, kMxEps))));
            if (lane == 0) out_scales[row * ngroups + gi] = ys;
            const float inv = 1.0f / ys;
            if (lane < 32) {
              const std::size_t i = gi * 32 + lane;
              const float sv = sycl::fmax(
                  -kFp4Max, sycl::fmin(y_at(i) * inv, kFp4Max));
              const std::uint8_t code = e2m1_encode(sv);
              const std::uint8_t next = static_cast<std::uint8_t>(
                  sycl::shift_group_left(sg, static_cast<std::uint32_t>(code),
                                         1u));
              if ((lane & 1u) == 0u) {
                out_q[(row * d + i) / 2] = static_cast<std::uint8_t>(
                    ((next & 0x0fu) << 4) | (code & 0x0fu));
              }
            }
          }
        }
      });
}

}  // namespace

sycl::event glu_quant_sycl(sycl::queue& q, const void* x, std::uint8_t* out_q,
                           float* out_scales, std::size_t rows, std::size_t d,
                           std::size_t group, int mode, DType dt) {
  const std::size_t g = mode == 1 ? 32 : group;
  if (g == 0 || d % g != 0 || (mode == 1 && g != 32)) return {};
#define QX_GQ(T, M)                                                          \
  return glu_quant_typed<T, M>(q, static_cast<const T*>(x), out_q,           \
                               out_scales, rows, d, g)
  switch (dt) {
    case DType::f32:
      if (mode == 0) QX_GQ(float, 0);
      QX_GQ(float, 1);
    case DType::f16:
      if (mode == 0) QX_GQ(half_t, 0);
      QX_GQ(half_t, 1);
    case DType::bf16:
      if (mode == 0) QX_GQ(bf16_t, 0);
      QX_GQ(bf16_t, 1);
  }
#undef QX_GQ
  return {};
}

}  // namespace quixicore::xpu::kernels
