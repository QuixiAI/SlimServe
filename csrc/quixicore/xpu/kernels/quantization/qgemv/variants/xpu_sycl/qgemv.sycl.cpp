// int4 group-quantized GEMV, native SYCL. Dequant-on-the-fly in the style of
// Marlin (unpack + fused scale in registers, no separate dequant pass) and the
// Metal qgemv (branchless nibble decode, one scale load per weight byte). The
// point: at batch 1 this is weight-memory-bound, and int4 weights are 4x smaller
// than fp16, so a correct decoder should BEAT an fp16 GEMV -- the truism that
// "4-bit dequant isn't worth it" is one to test on B60, not assume.
//
// One work-group per output row; each thread walks a strided set of weight bytes
// (2 nibbles = 2 output-dim contributions), accumulates in fp32, group-reduces.

#include "quantization/qgemv/qgemv_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// One subgroup per output row (Metal "one simdgroup per row"); several rows per
// work-group for occupancy. 32-wide subgroup, 8 rows per work-group.
constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;  // 256

// Sign-extend a 4-bit two's-complement nibble to int.
inline int s4(int nib) { return nib >= 8 ? nib - 16 : nib; }

// Wide-load decode: each thread reads a 16-byte chunk (uint4 = 32 int4 weights)
// per step, unpacks in registers, and applies one fp16 scale per chunk (valid
// when group % 32 == 0, so a 32-k chunk stays within one group). Adjacent
// threads read adjacent 16-byte chunks -> coalesced + wide, which is what makes
// the decode weight-memory-bound (the naive 1-byte-per-thread version was not).
// A scalar tail handles any remainder.
using WVec = sycl::vec<std::uint32_t, 4>;  // 16 bytes = 32 nibbles

template <typename T>
sycl::event qgemv_typed(sycl::queue& q, const std::uint8_t* w, const half_t* scales,
                        const T* x, T* y, std::size_t N, std::size_t K,
                        std::size_t group) {
  const std::size_t bytes_per_row = K / 2;
  const std::size_t groups_per_row = K / group;
  const std::size_t nchunks = bytes_per_row / 16;   // whole 16-byte chunks
  const std::size_t tail_b0 = nchunks * 16;         // first scalar-tail byte
  const bool fast_scale = (group % 32 == 0);
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());  // 0..kRowsPerWG-1
    const int lane = static_cast<int>(sg.get_local_linear_id());  // 0..kSG-1
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;  // subgroup-uniform (row depends only on sgid): safe

    const std::uint8_t* wrow = w + n * bytes_per_row;
    const half_t* srow = scales + n * groups_per_row;
    const WVec* wvrow = reinterpret_cast<const WVec*>(wrow);

    // Decode is ALU/latency-bound, not weight-memory-bound (measured: was 25%
    // of BW roofline). Two fixes: (1) hoist the per-group scale OUT of the nibble
    // loop -- one mul per 8-nibble word instead of two muls per nibble; (2)
    // vectorize the activation read (one sycl::vec<T,8> vs 8 scalar loads), and
    // accumulate each word into its own register so the 4 words are independent
    // (breaks the serial acc dependency chain -> more ILP).
    float acc = 0.0f;
    for (std::size_t c = lane; c < nchunks; c += kSG) {
      const WVec chunk = wvrow[c];
      const std::size_t kbase = c * 32;
      const T* xc = x + kbase;
      const float sc = fast_scale ? static_cast<float>(srow[kbase / group]) : 0.0f;
      float aw[4];
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        const std::uint32_t word = chunk[wi];
        const float s = fast_scale ? sc : static_cast<float>(srow[(kbase + wi * 8) / group]);
        const sycl::vec<T, 8> xv = *reinterpret_cast<const sycl::vec<T, 8>*>(xc + wi * 8);
        float a = 0.0f;
#pragma unroll
        for (int nib = 0; nib < 8; ++nib)
          a += static_cast<float>(s4((word >> (nib * 4)) & 0xF)) * static_cast<float>(xv[nib]);
        aw[wi] = a * s;
      }
      acc += (aw[0] + aw[1]) + (aw[2] + aw[3]);
    }
    // scalar tail (remaining bytes past the last full 16-byte chunk)
    for (std::size_t b = tail_b0 + lane; b < bytes_per_row; b += kSG) {
      const std::uint8_t byte = wrow[b];
      const std::size_t k0 = 2 * b;
      const float sc = static_cast<float>(srow[k0 / group]);
      acc += static_cast<float>(s4(byte & 0xF)) * sc * static_cast<float>(x[k0]);
      acc += static_cast<float>(s4((byte >> 4) & 0xF)) * sc * static_cast<float>(x[k0 + 1]);
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

}  // namespace

sycl::event qgemv_int4_sycl(sycl::queue& q, const void* w_packed,
                            const void* scales, const void* x, void* y,
                            std::size_t N, std::size_t K, std::size_t group,
                            DType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_packed);
  const auto* s = static_cast<const half_t*>(scales);
  switch (act_dt) {
    case DType::f32:
      return qgemv_typed(q, w, s, static_cast<const float*>(x),
                         static_cast<float*>(y), N, K, group);
    case DType::f16:
      return qgemv_typed(q, w, s, static_cast<const half_t*>(x),
                         static_cast<half_t*>(y), N, K, group);
    case DType::bf16:
      return qgemv_typed(q, w, s, static_cast<const bf16_t*>(x),
                         static_cast<bf16_t*>(y), N, K, group);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
