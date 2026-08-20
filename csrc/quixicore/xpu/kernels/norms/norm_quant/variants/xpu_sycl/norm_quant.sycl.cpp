// Fused RMSNorm + activation quantization, native SYCL variant.
//
// One work-group per token row. Pass 1 optionally adds the residual IN PLACE
// (variance uses the dtype-rounded read-back so it matches what later passes
// read — the vLLM layernorm_quant invariant), then reduces sum(x^2). The
// quantization epilogue is per mode:
//   static_fp8:  out = e4m3(clamp(norm_x / *static_scale, +-448))
//   dynamic_fp8: per-row absmax -> scale = max(absmax/448, kMinScale),
//                written to out_scales[row], then as static
//   mxfp4:       per-32-group absmax -> fp32 power-of-two scale
//                exp2(ceil(log2(max(absmax/6, eps32)))) in out_scales,
//                e2m1 midpoint-bucketize codes packed two per byte
//                (element 2i low nibble)
// The e4m3 encoder is the shared integer-exact codec (turboquant_codec), so
// the pure quantization step is deterministic; the norm/absmax reductions are
// tree-ordered fp32 and tolerance-checked via decoded outputs.
//
// Quant chains match vllm-xpu-kernels csrc/layernorm_quant.cpp +
// csrc/quantization/fp4/mxfp4_quant.h (Apache; translated, not imported).

#include "norms/norm_quant/norm_quant_kernel.hpp"
#include "quantization/turboquant/turboquant_codec.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;
constexpr float kFp8Max = 448.0f;
constexpr float kMinScale = 1.0f / (448.0f * 512.0f);
constexpr float kFp4Max = 6.0f;
constexpr float kMxEps = 1e-10f;

namespace codec = quixicore::xpu::turboquant_codec;

// e2m1 code via midpoint compares (strictly-greater: lower wins ties).
inline std::uint8_t e2m1_encode(float x) {
  const float a = sycl::fabs(x);
  const int code = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) +
                   (a > 2.5f) + (a > 3.5f) + (a > 5.0f);
  return static_cast<std::uint8_t>(code |
                                   (x < 0.0f ? 0x8 : 0x0));
}

template <typename T, int Mode, bool HasResidual>
class NormQuantKernel;

template <typename T, int Mode, bool HasResidual>
sycl::event norm_quant_typed(sycl::queue& q, const T* x, T* residual,
                             const T* w, std::uint8_t* out_q,
                             const float* static_scale, float* out_scales,
                             std::size_t rows, std::size_t hidden, float eps) {
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> slm(sycl::range<1>(2), h);
    h.parallel_for<NormQuantKernel<T, Mode, HasResidual>>(
        ndr, [=](sycl::nd_item<1> it) {
          const std::size_t row = it.get_group(0);
          const std::size_t lid = it.get_local_id(0);
          const T* xr = x + row * hidden;
          T* rr = HasResidual ? residual + row * hidden : nullptr;
          std::uint8_t* outr =
              out_q + row * (Mode == 2 ? hidden / 2 : hidden);

          float partial = 0.0f;
          for (std::size_t i = lid; i < hidden; i += kRowThreads) {
            float v = static_cast<float>(xr[i]);
            if constexpr (HasResidual) {
              v += static_cast<float>(rr[i]);
              rr[i] = static_cast<T>(v);
              v = static_cast<float>(rr[i]);  // rounded read-back
            }
            partial += v * v;
          }
          const float sumsq = sycl::reduce_over_group(
              it.get_group(), partial, sycl::plus<float>());
          if (lid == 0)
            slm[0] = sycl::rsqrt(sumsq / static_cast<float>(hidden) + eps);
          sycl::group_barrier(it.get_group());
          const float inv_rms = slm[0];

          auto norm_at = [=](std::size_t i) {
            const float v = HasResidual ? static_cast<float>(rr[i])
                                        : static_cast<float>(xr[i]);
            return v * inv_rms * static_cast<float>(w[i]);
          };

          if constexpr (Mode == 0 || Mode == 1) {
            float scale;
            if constexpr (Mode == 0) {
              scale = *static_scale;
            } else {
              float amax = 0.0f;
              for (std::size_t i = lid; i < hidden; i += kRowThreads)
                amax = sycl::fmax(amax, sycl::fabs(norm_at(i)));
              amax = sycl::reduce_over_group(it.get_group(), amax,
                                             sycl::maximum<float>());
              if (lid == 0) {
                slm[1] = sycl::fmax(amax / kFp8Max, kMinScale);
                out_scales[row] = slm[1];
              }
              sycl::group_barrier(it.get_group());
              scale = slm[1];
            }
            const float inv_scale = 1.0f / scale;
            for (std::size_t i = lid; i < hidden; i += kRowThreads) {
              const float qv = sycl::fmax(
                  sycl::fmin(norm_at(i) * inv_scale, kFp8Max), -kFp8Max);
              outr[i] = codec::e4m3_encode(qv);
            }
          } else {
            // mxfp4: subgroup per 32-group; two nibbles packed via shuffle.
            const sycl::sub_group sg = it.get_sub_group();
            const std::size_t sg_size = sg.get_local_linear_range();
            const std::size_t nsg = kRowThreads / sg_size;
            const std::size_t sg_id = it.get_local_linear_id() / sg_size;
            const std::size_t lane = sg.get_local_linear_id();
            const std::size_t ngroups = hidden / 32;
            for (std::size_t g = sg_id; g < ngroups; g += nsg) {
              float amax = kMxEps;
              if (lane < 32) {
                const float v = norm_at(g * 32 + lane);
                amax = sycl::fmax(amax, sycl::fabs(v));
              }
              amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
              const float ys = sycl::exp2(sycl::ceil(
                  sycl::log2(sycl::fmax(amax / kFp4Max, kMxEps))));
              if (lane == 0) out_scales[row * ngroups + g] = ys;
              const float inv = 1.0f / ys;
              if (lane < 32) {
                const float sv = sycl::fmax(
                    -kFp4Max,
                    sycl::fmin(norm_at(g * 32 + lane) * inv, kFp4Max));
                const std::uint8_t code = e2m1_encode(sv);
                const std::uint8_t next = static_cast<std::uint8_t>(
                    sycl::shift_group_left(sg, static_cast<std::uint32_t>(code),
                                           1u));
                if ((lane & 1u) == 0u) {
                  outr[(g * 32 + lane) / 2] = static_cast<std::uint8_t>(
                      ((next & 0x0fu) << 4) | (code & 0x0fu));
                }
              }
            }
          }
        });
  });
}

template <typename T>
sycl::event dispatch_mode(sycl::queue& q, const T* x, T* residual, const T* w,
                          std::uint8_t* out_q, const float* static_scale,
                          float* out_scales, std::size_t rows,
                          std::size_t hidden, float eps, int mode) {
  const bool has_res = residual != nullptr;
#define QX_NQ(M, R)                                                       \
  return norm_quant_typed<T, M, R>(q, x, residual, w, out_q, static_scale, \
                                   out_scales, rows, hidden, eps)
  switch (mode) {
    case 0:
      if (has_res) QX_NQ(0, true);
      QX_NQ(0, false);
    case 1:
      if (has_res) QX_NQ(1, true);
      QX_NQ(1, false);
    case 2:
      if (has_res) QX_NQ(2, true);
      QX_NQ(2, false);
  }
#undef QX_NQ
  return {};
}

}  // namespace

sycl::event norm_quant_sycl(sycl::queue& q, const void* x, void* residual,
                            const void* weight, std::uint8_t* out_q,
                            const float* static_scale, float* out_scales,
                            std::size_t rows, std::size_t hidden, float eps,
                            int mode, DType dt) {
  if (mode == 2 && (hidden % 32 != 0)) return {};  // mxfp4 needs whole groups
  switch (dt) {
    case DType::f32:
      return dispatch_mode(q, static_cast<const float*>(x),
                           static_cast<float*>(residual),
                           static_cast<const float*>(weight), out_q,
                           static_scale, out_scales, rows, hidden, eps, mode);
    case DType::f16:
      return dispatch_mode(q, static_cast<const half_t*>(x),
                           static_cast<half_t*>(residual),
                           static_cast<const half_t*>(weight), out_q,
                           static_scale, out_scales, rows, hidden, eps, mode);
    case DType::bf16:
      return dispatch_mode(q, static_cast<const bf16_t*>(x),
                           static_cast<bf16_t*>(residual),
                           static_cast<const bf16_t*>(weight), out_q,
                           static_scale, out_scales, rows, hidden, eps, mode);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
