// fp8 GEMV (the M=1 row of fp8_gemm), native SYCL decoder. At batch 1 the op is
// weight-memory-bound and the oneDNN fp8 matmul route measured 3% of the B60's
// bandwidth roofline there — so decode-on-the-fly is the fast path, exactly as
// with int4/mxfp4/nvfp4. fp8 needs no LUT at all: both encodings are bit-casts
// away from f16.
//
//   e5m2 IS a truncated f16:            half(b << 8)                  (exact)
//   e4m3 maps into the f16 grid:        half(((b&0x80)<<8)|((b&0x7f)<<7)) * 256
//     - exponent field lands on f16's with bias 15 vs 7 -> a fixed *2^8;
//       exact for normals AND subnormals (e4m3 subnormal m*2^-9 becomes the f16
//       subnormal (m<<7)/2^24 = m*2^-17 = (m*2^-9)*2^-8). The only non-value is
//       e4m3 NaN (0x7F/0xFF), which decodes to +-480; weights are never NaN.
//
// Layout note: fp8_gemm's B is [K, N] (activations-row-major), so unlike the
// [N, K] quant GEMVs the coalescing axis is N: each thread owns 8 adjacent
// output columns (one 8-byte load per k-row), and K is split into slabs across
// work-groups for occupancy (N/8 threads alone cannot fill 160 XVEs). Partials
// land in an f32 scratch and a small second kernel reduces slabs -> y.

#include "quantization/fp8_gemm/fp8_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr int kCols = 8;    // output columns per thread (one 8-byte weight load)
constexpr int kWG = 256;
constexpr std::size_t kSlabs = 32;  // K split; threads = slabs * N/8 (64 measured worse)

// kind: 0 = e4m3, 1 = e5m2 (matches ops::Fp8Kind).
//
// Decode is deliberately UNSCALED for e4m3: the bit-relocation alone yields
// value * 2^-8 as an f16 (exact, incl. subnormals: m*2^-9 -> f16 subnormal
// m*2^-17 = 128m ulps). Both x and B carry the same 2^-8, so every product is
// off by exactly 2^-16 and ONE multiply in the reduce kernel compensates —
// zero per-element scale muls in the hot loop (this was the e4m3-vs-e5m2 gap).
template <int KIND>
inline float fp8_dec_raw(std::uint32_t b) {
  if constexpr (KIND == 1) {
    return static_cast<float>(
        sycl::bit_cast<sycl::half>(static_cast<std::uint16_t>(b << 8)));
  } else {
    const auto h = static_cast<std::uint16_t>(((b & 0x80u) << 8) | ((b & 0x7Fu) << 7));
    return static_cast<float>(sycl::bit_cast<sycl::half>(h));
  }
}

// Pair decode: relocate the two EVEN (or odd-shifted) bytes of a u32 into two
// f16 lanes with one masked-shift pass, then one vector convert -> 2 floats.
template <int KIND>
inline sycl::vec<float, 2> fp8_dec2(std::uint32_t two) {
  std::uint32_t h2;
  if constexpr (KIND == 1)
    h2 = (two & 0x00FF00FFu) << 8;
  else
    h2 = ((two & 0x00800080u) << 8) | ((two & 0x007F007Fu) << 7);
  return sycl::bit_cast<sycl::vec<sycl::half, 2>>(h2).template convert<float>();
}

inline std::uint32_t rd_u32(const std::uint8_t* p) {
  return *reinterpret_cast<const std::uint32_t*>(p);
}
inline sycl::vec<std::uint32_t, 2> rd_u32x2(const std::uint8_t* p) {
  return *reinterpret_cast<const sycl::vec<std::uint32_t, 2>*>(p);
}

template <int KIND>
sycl::event partial_launch(sycl::queue& q, const std::uint8_t* x,
                           const std::uint8_t* B, float* part, std::size_t N,
                           std::size_t K, std::size_t S, std::size_t slab) {
  const std::size_t tn = (N + kCols - 1) / kCols;
  const std::size_t tn_pad = ((tn + kWG - 1) / kWG) * kWG;
  const bool aligned = (N % kCols == 0);
  const sycl::nd_range<2> ndr(sycl::range<2>(S, tn_pad), sycl::range<2>(1, kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<2> it) {
    const std::size_t s = it.get_global_id(0);
    const std::size_t n0 = it.get_global_id(1) * kCols;
    if (n0 >= N) return;
    const std::size_t k0 = s * slab;
    const std::size_t kend = sycl::min(k0 + slab, K);
    float a[kCols] = {};
    if (aligned) {
      // Fast path: one u32 x-load per 4 k-rows, one 8-byte weight load per
      // row. Lane-adjacent threads read adjacent 8-byte spans of the same B
      // row. All decodes go through the 2-at-a-time relocation (fp8_dec2):
      // for a u32 word, the even bytes (0,2) come out of one pass and the odd
      // bytes (1,3) out of a second on word>>8.
      std::size_t k = k0;  // slab is a multiple of 4, so x + k stays aligned
      for (; k + 4 <= kend; k += 4) {
        const std::uint32_t xw = rd_u32(x + k);
        const sycl::vec<float, 2> xe = fp8_dec2<KIND>(xw);        // rows k, k+2
        const sycl::vec<float, 2> xo = fp8_dec2<KIND>(xw >> 8);   // rows k+1, k+3
        const float xf4[4] = {xe[0], xo[0], xe[1], xo[1]};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          const float xf = xf4[j];
          const sycl::vec<std::uint32_t, 2> bw = rd_u32x2(B + (k + j) * N + n0);
#pragma unroll
          for (int w = 0; w < 2; ++w) {
            const sycl::vec<float, 2> fe = fp8_dec2<KIND>(bw[w]);       // cols 0,2
            const sycl::vec<float, 2> fo = fp8_dec2<KIND>(bw[w] >> 8);  // cols 1,3
            a[4 * w + 0] += xf * fe[0];
            a[4 * w + 1] += xf * fo[0];
            a[4 * w + 2] += xf * fe[1];
            a[4 * w + 3] += xf * fo[1];
          }
        }
      }
      for (; k < kend; ++k) {  // K % 4 tail
        const float xf = fp8_dec_raw<KIND>(x[k]);
        const std::uint8_t* br = B + k * N + n0;
#pragma unroll
        for (int j = 0; j < kCols; ++j) a[j] += xf * fp8_dec_raw<KIND>(br[j]);
      }
    } else {  // ragged N: byte loads with per-column guards
      for (std::size_t k = k0; k < kend; ++k) {
        const float xf = fp8_dec_raw<KIND>(x[k]);
        const std::uint8_t* br = B + k * N;
        for (int j = 0; j < kCols && n0 + j < N; ++j)
          a[j] += xf * fp8_dec_raw<KIND>(br[n0 + j]);
      }
    }
    float* pr = part + s * N + n0;
    for (int j = 0; j < kCols && n0 + j < N; ++j) pr[j] = a[j];
  });
}

template <typename T>
sycl::event reduce_launch(sycl::queue& q, const float* part, T* y, std::size_t N,
                          std::size_t S, float scale, sycl::event dep) {
  const std::size_t npad = ((N + kWG - 1) / kWG) * kWG;
  const sycl::nd_range<1> ndr{sycl::range<1>{npad}, sycl::range<1>{kWG}};
  return q.submit([&](sycl::handler& h) {
    h.depends_on(dep);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      const std::size_t n = it.get_global_id(0);
      if (n >= N) return;
      float acc = 0.0f;
      for (std::size_t s = 0; s < S; ++s) acc += part[s * N + n];
      y[n] = static_cast<T>(acc * scale);
    });
  });
}

}  // namespace

sycl::event fp8_gemv_sycl(sycl::queue& q, const void* x_fp8, const void* b_fp8,
                          void* y, std::size_t N, std::size_t K, int kind,
                          float scale, DType out_dt) {
  const auto* x = static_cast<const std::uint8_t*>(x_fp8);
  const auto* B = static_cast<const std::uint8_t*>(b_fp8);
  std::size_t slab = (K + kSlabs - 1) / kSlabs;
  slab = ((slab + 3) / 4) * 4;                  // keep 4-byte x loads aligned
  const std::size_t S = (K + slab - 1) / slab;  // slabs actually used
  // e4m3 decodes at 2^-8 per operand (see fp8_dec_raw); compensate once here.
  if (kind == 0) scale *= 65536.0f;
  float* part = sycl::malloc_device<float>(S * N, q);
  sycl::event p = (kind == 1) ? partial_launch<1>(q, x, B, part, N, K, S, slab)
                              : partial_launch<0>(q, x, B, part, N, K, S, slab);
  sycl::event r;
  switch (out_dt) {
    case DType::f32:
      r = reduce_launch(q, part, static_cast<float*>(y), N, S, scale, p);
      break;
    case DType::f16:
      r = reduce_launch(q, part, static_cast<half_t*>(y), N, S, scale, p);
      break;
    case DType::bf16:
      r = reduce_launch(q, part, static_cast<bf16_t*>(y), N, S, scale, p);
      break;
  }
  return q.submit([&](sycl::handler& h) {  // free scratch off the critical path
    h.depends_on(r);
    h.host_task([part, q]() mutable { sycl::free(part, q); });
  });
}

}  // namespace quixicore::xpu::kernels
