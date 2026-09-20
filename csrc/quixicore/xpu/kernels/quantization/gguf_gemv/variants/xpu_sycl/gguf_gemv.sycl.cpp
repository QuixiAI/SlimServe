// GGUF q8_0 / q4_0 GEMV, native SYCL decoder from the authentic llama.cpp block
// layout: each 32-element block is { fp16 d; quants } stored consecutively along
// a row. q8_0: 32 int8 (w = int8*d). q4_0: 16 packed bytes, low nibbles are
// elements 0..15 and high nibbles 16..31, w = (nibble-8)*d.
//
// One 32-wide subgroup per row; each lane decodes whole blocks (strided). The
// interleaved on-disk layout is not GPU-coalescing-friendly (odd 34/18-byte
// strides), so this is correctness-first; a repack to a GPU layout is the
// optimization. GGUF k-quants are data encodings -> decode natively on Intel.

#include "quantization/gguf_gemv/gguf_kernel.hpp"
#include "quantization/gguf_gemv/gguf_iq_tables.hpp"

namespace quixicore::xpu::kernels {

// Lazy per-process device copies of the (small, read-only) i-quant tables. First
// use uploads; later calls reuse. Leaks at exit (fine for a table cache).
template <typename E>
const E* dev_table(sycl::queue& q, const E* host, std::size_t n, const E** slot) {
  if (!*slot) {
    E* d = sycl::malloc_device<E>(n, q);
    q.memcpy(d, host, n * sizeof(E)).wait();
    *slot = d;
  }
  return *slot;
}

namespace {

constexpr int kSG = 32;
constexpr int kRowsPerWG = 8;
constexpr int kWG = kSG * kRowsPerWG;

inline float load_half(const std::uint8_t* p) {
  const std::uint16_t bits = static_cast<std::uint16_t>(p[0]) |
                             (static_cast<std::uint16_t>(p[1]) << 8);
  return static_cast<float>(sycl::bit_cast<sycl::half>(bits));
}

template <typename T, int TYPE>
sycl::event gguf_typed(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                       std::size_t N, std::size_t K) {
  constexpr int BLOCK_BYTES = (TYPE == 0) ? 34 : 18;  // q8_0 : q4_0
  const std::size_t blocks_per_row = K / 32;
  const std::size_t row_bytes = blocks_per_row * BLOCK_BYTES;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < blocks_per_row; b += kSG) {
      const std::uint8_t* blk = wrow + b * BLOCK_BYTES;
      const float d = load_half(blk);
      const std::uint8_t* qs = blk + 2;
      const std::size_t kbase = b * 32;
      if (TYPE == 0) {  // q8_0
#pragma unroll
        for (int i = 0; i < 32; ++i)
          acc += static_cast<float>(static_cast<std::int8_t>(qs[i])) * d *
                 static_cast<float>(x[kbase + i]);
      } else {  // q4_0
#pragma unroll
        for (int i = 0; i < 16; ++i) {
          const int lo = (qs[i] & 0xF) - 8;
          const int hi = (qs[i] >> 4) - 8;
          acc += static_cast<float>(lo) * d * static_cast<float>(x[kbase + i]);
          acc += static_cast<float>(hi) * d * static_cast<float>(x[kbase + 16 + i]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q6_K: 256-element super-block (210 bytes): ql[128], qh[64], int8
// scales[16], fp16 d. 6-bit quant = 4 low bits (ql) + 2 high bits (qh),
// recentred by -32. Follows ggml dequantize_row_q6_K exactly. One 32-wide
// subgroup per row; each lane decodes whole super-blocks.
template <typename T>
sycl::event gguf_q6k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 210;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 210;
      const std::uint8_t* ql = blk;
      const std::uint8_t* qh = blk + 128;
      const std::int8_t* sc = reinterpret_cast<const std::int8_t*>(blk + 192);
      const float d = load_half(blk + 208);
      const std::size_t kbase = b * 256;
      for (int half = 0; half < 2; ++half) {
        const std::uint8_t* qlh = ql + half * 64;
        const std::uint8_t* qhh = qh + half * 32;
        const std::int8_t* sch = sc + half * 8;
        const std::size_t yoff = kbase + static_cast<std::size_t>(half) * 128;
        for (int l = 0; l < 32; ++l) {
          const int is = l / 16;
          const int q1 = static_cast<int>((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
          const int q2 = static_cast<int>((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
          const int q3 = static_cast<int>((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
          const int q4 = static_cast<int>((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
          acc += d * static_cast<float>(sch[is + 0]) * q1 * static_cast<float>(x[yoff + l + 0]);
          acc += d * static_cast<float>(sch[is + 2]) * q2 * static_cast<float>(x[yoff + l + 32]);
          acc += d * static_cast<float>(sch[is + 4]) * q3 * static_cast<float>(x[yoff + l + 64]);
          acc += d * static_cast<float>(sch[is + 6]) * q4 * static_cast<float>(x[yoff + l + 96]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// ggml get_scale_min_k4: unpack the j-th 6-bit sub-scale (sc) and sub-min (m)
// from the 12-byte packed scales array of q4_K/q5_K.
inline void scale_min_k4(int j, const std::uint8_t* s, int& sc, int& m) {
  if (j < 4) { sc = s[j] & 63; m = s[j + 4] & 63; }
  else {
    sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
    m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
  }
}

// GGUF q4_K: 256-element super-block (144 bytes): fp16 d, fp16 dmin,
// scales[12] (8 packed 6-bit scale+min pairs), qs[128]. 8 sub-blocks of 32,
// w = d*sc*(q&0xF or q>>4) - dmin*m. Follows ggml dequantize_row_q4_K.
template <typename T>
sycl::event gguf_q4k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 144;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 144;
      const float d = load_half(blk);
      const float dmin = load_half(blk + 2);
      const std::uint8_t* sc = blk + 4;
      const std::uint8_t* qs = blk + 16;
      const std::size_t kbase = b * 256;
      for (int iter = 0; iter < 4; ++iter) {
        int s1, m1i, s2, m2i;
        scale_min_k4(iter * 2 + 0, sc, s1, m1i);
        scale_min_k4(iter * 2 + 1, sc, s2, m2i);
        const float d1 = d * s1, mm1 = dmin * m1i;
        const float d2 = d * s2, mm2 = dmin * m2i;
        const std::uint8_t* qq = qs + iter * 32;
        const std::size_t yb = kbase + iter * 64;
        for (int l = 0; l < 32; ++l) {
          acc += (d1 * (qq[l] & 0xF) - mm1) * static_cast<float>(x[yb + l]);
          acc += (d2 * (qq[l] >> 4) - mm2) * static_cast<float>(x[yb + 32 + l]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q5_K: 176-byte super-block (fp16 d + dmin, scales[12], qh[32], qs[128]).
// Like q4_K but each 4-bit quant gains a 5th bit from qh (bit shifts by 2 per
// 64-block). Follows ggml dequantize_row_q5_K.
template <typename T>
sycl::event gguf_q5k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 176;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const int sgid = static_cast<int>(sg.get_group_linear_id());
    const int lane = static_cast<int>(sg.get_local_linear_id());
    const std::size_t n = it.get_group(0) * kRowsPerWG + sgid;
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 176;
      const float d = load_half(blk);
      const float dmin = load_half(blk + 2);
      const std::uint8_t* sc = blk + 4;
      const std::uint8_t* qh = blk + 16;
      const std::uint8_t* qs = blk + 48;
      const std::size_t kbase = b * 256;
      for (int iter = 0; iter < 4; ++iter) {
        int s1, m1i, s2, m2i;
        scale_min_k4(iter * 2 + 0, sc, s1, m1i);
        scale_min_k4(iter * 2 + 1, sc, s2, m2i);
        const float d1 = d * s1, mm1 = dmin * m1i, d2 = d * s2, mm2 = dmin * m2i;
        const std::uint8_t* ql = qs + iter * 32;
        const int u1 = 1 << (2 * iter), u2 = 2 << (2 * iter);
        const std::size_t yb = kbase + iter * 64;
        for (int l = 0; l < 32; ++l) {
          const float q1 = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
          const float q2 = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
          acc += (d1 * q1 - mm1) * static_cast<float>(x[yb + l]);
          acc += (d2 * q2 - mm2) * static_cast<float>(x[yb + 32 + l]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q2_K: 84-byte super-block (scales[16], qs[64], fp16 d, fp16 dmin). 16
// sub-blocks of 16, 2-bit quants; each scales byte = 4-bit scale (low) + 4-bit
// min (high). w = d*sc*q - dmin*m. Follows ggml dequantize_row_q2_K.
template <typename T>
sycl::event gguf_q2k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 84;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 84;
      const std::uint8_t* scales = blk;
      const std::uint8_t* qs = blk + 16;
      const float d = load_half(blk + 80);
      const float dmin = load_half(blk + 82);
      const std::size_t kbase = b * 256;
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t* qb = qs + h * 32;
        for (int j = 0; j < 4; ++j) {
          const int shift = 2 * j;
          const std::uint8_t sc1 = scales[h * 8 + j * 2];
          const std::uint8_t sc2 = scales[h * 8 + j * 2 + 1];
          const float dl1 = d * (sc1 & 0xF), ml1 = dmin * (sc1 >> 4);
          const float dl2 = d * (sc2 & 0xF), ml2 = dmin * (sc2 >> 4);
          const std::size_t yb = kbase + h * 128 + j * 32;
          for (int l = 0; l < 16; ++l) {
            acc += (dl1 * ((qb[l] >> shift) & 3) - ml1) * static_cast<float>(x[yb + l]);
            acc += (dl2 * ((qb[l + 16] >> shift) & 3) - ml2) * static_cast<float>(x[yb + 16 + l]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF q3_K: 110-byte super-block (hmask[32], qs[64], scales[12], fp16 d). 16
// sub-blocks of 16, 3-bit quant = 2 low bits (qs) + 1 inverted high bit (hmask);
// 6-bit scales packed via a custom kmask bit-shuffle. Follows ggml
// dequantize_row_q3_K exactly (the fiddliest k-quant).
template <typename T>
sycl::event gguf_q3k_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                           T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256;
  const std::size_t row_bytes = sblocks * 110;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    constexpr std::uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;

    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 110;
      const std::uint8_t* hm = blk;
      const std::uint8_t* qs = blk + 32;
      const std::uint8_t* sc = blk + 96;
      const float d = load_half(blk + 108);
      // unpack the 16 int8 scales (ggml kmask shuffle)
      auto rd32 = [&](int o) {
        return static_cast<std::uint32_t>(sc[o]) | (static_cast<std::uint32_t>(sc[o + 1]) << 8) |
               (static_cast<std::uint32_t>(sc[o + 2]) << 16) | (static_cast<std::uint32_t>(sc[o + 3]) << 24);
      };
      const std::uint32_t a0 = rd32(0), a1 = rd32(4), tmp = rd32(8);
      std::uint32_t aux[4];
      aux[2] = ((a0 >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
      aux[3] = ((a1 >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
      aux[0] = (a0 & kmask2) | (((tmp >> 0) & kmask1) << 4);
      aux[1] = (a1 & kmask2) | (((tmp >> 2) & kmask1) << 4);
      auto scl = [&](int i) { return static_cast<int>(static_cast<std::int8_t>((aux[i >> 2] >> (8 * (i & 3))) & 0xFF)); };

      const std::size_t kbase = b * 256;
      for (int h = 0; h < 2; ++h) {
        const std::uint8_t* qb = qs + h * 32;
        for (int j = 0; j < 4; ++j) {
          const int shift = 2 * j;
          const int is = h * 8 + j * 2;
          const std::uint8_t mbit = static_cast<std::uint8_t>(1u << (h * 4 + j));
          const float dl1 = d * (scl(is) - 32), dl2 = d * (scl(is + 1) - 32);
          const std::size_t yb = kbase + h * 128 + j * 32;
          for (int l = 0; l < 16; ++l) {
            const int v1 = static_cast<int>((qb[l] >> shift) & 3) - ((hm[l] & mbit) ? 0 : 4);
            const int v2 = static_cast<int>((qb[l + 16] >> shift) & 3) - ((hm[l + 16] & mbit) ? 0 : 4);
            acc += dl1 * v1 * static_cast<float>(x[yb + l]);
            acc += dl2 * v2 * static_cast<float>(x[yb + 16 + l]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF IQ4_NL: q4_0 footprint (18-byte block: fp16 d + qs[16]) but the 4-bit
// index maps through a fixed 16-entry non-linear codebook instead of a linear
// scale. QK=32, w = d * kvalues_iq4nl[nibble]. Follows ggml dequantize_row_iq4_nl.
constexpr int kIQ4NL[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                            1, 13, 25, 38, 53, 69, 89, 113};

template <typename T>
sycl::event gguf_iq4nl_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                             T* y, std::size_t N, std::size_t K) {
  const std::size_t blocks_per_row = K / 32;
  const std::size_t row_bytes = blocks_per_row * 18;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < blocks_per_row; b += kSG) {
      const std::uint8_t* blk = wrow + b * 18;
      const float d = load_half(blk);
      const std::uint8_t* qs = blk + 2;
      const std::size_t kbase = b * 32;
      for (int j = 0; j < 16; ++j) {
        acc += d * kIQ4NL[qs[j] & 0xF] * static_cast<float>(x[kbase + j]);
        acc += d * kIQ4NL[qs[j] >> 4] * static_cast<float>(x[kbase + 16 + j]);
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

inline std::uint32_t rd_u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

// GGUF legacy 32-element blocks: q4_1 (20B, affine d/m), q5_0 (22B, 5-bit signed
// -16), q5_1 (24B, 5-bit affine). TYPE: 8=q4_1, 9=q5_0, 10=q5_1.
template <typename T, int TYPE>
sycl::event gguf_legacy_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                              T* y, std::size_t N, std::size_t K) {
  constexpr int BB = (TYPE == 8) ? 20 : (TYPE == 9) ? 22 : 24;
  const std::size_t bpr = K / 32, row_bytes = bpr * BB;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < bpr; b += kSG) {
      const std::uint8_t* blk = wrow + b * BB;
      const float d = load_half(blk);
      const std::size_t kbase = b * 32;
      if (TYPE == 8) {  // q4_1
        const float m = load_half(blk + 2);
        const std::uint8_t* qs = blk + 4;
        for (int j = 0; j < 16; ++j) {
          acc += ((qs[j] & 0xF) * d + m) * static_cast<float>(x[kbase + j]);
          acc += ((qs[j] >> 4) * d + m) * static_cast<float>(x[kbase + 16 + j]);
        }
      } else {  // q5_0 / q5_1
        const bool affine = (TYPE == 10);
        const float m = affine ? load_half(blk + 2) : 0.0f;
        const std::uint8_t* qh_p = blk + (affine ? 4 : 2);
        const std::uint8_t* qs = blk + (affine ? 8 : 6);
        const std::uint32_t qh = rd_u32(qh_p);
        for (int j = 0; j < 16; ++j) {
          const int xh0 = ((qh >> (j + 0)) << 4) & 0x10;
          const int xh1 = ((qh >> (j + 12))) & 0x10;
          int x0 = (qs[j] & 0xF) | xh0;
          int x1 = (qs[j] >> 4) | xh1;
          if (!affine) { x0 -= 16; x1 -= 16; }
          acc += (x0 * d + m) * static_cast<float>(x[kbase + j]);
          acc += (x1 * d + m) * static_cast<float>(x[kbase + 16 + j]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// GGUF IQ4_XS: 136-byte super-block (fp16 d, uint16 scales_h, scales_l[4],
// qs[128]); 8 sub-blocks of 32; 6-bit per-sub scale split across scales_l/h;
// values via the iq4_nl codebook. Follows ggml dequantize_row_iq4_xs.
template <typename T>
sycl::event gguf_iq4xs_typed(sycl::queue& q, const std::uint8_t* w, const T* x,
                             T* y, std::size_t N, std::size_t K) {
  const std::size_t sblocks = K / 256, row_bytes = sblocks * 136;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  const sycl::nd_range<1> ndr(sycl::range<1>(nwg * kWG), sycl::range<1>(kWG));
  return q.parallel_for(ndr, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < sblocks; b += kSG) {
      const std::uint8_t* blk = wrow + b * 136;
      const float d = load_half(blk);
      const std::uint32_t scales_h = static_cast<std::uint32_t>(blk[2]) | (static_cast<std::uint32_t>(blk[3]) << 8);
      const std::uint8_t* scales_l = blk + 4;
      const std::uint8_t* qs = blk + 8;
      const std::size_t kbase = b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls = ((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scales_h >> (2 * ib)) & 3) << 4);
        const float dl = d * (ls - 32);
        const std::uint8_t* qq = qs + ib * 16;
        const std::size_t yb = kbase + ib * 32;
        for (int j = 0; j < 16; ++j) {
          acc += dl * kIQ4NL[qq[j] & 0xF] * static_cast<float>(x[yb + j]);
          acc += dl * kIQ4NL[qq[j] >> 4] * static_cast<float>(x[yb + 16 + j]);
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// Table cache slots for this TU.
const std::uint64_t* g_iq2xxs = nullptr;
const std::uint64_t* g_iq2xs = nullptr;
const std::uint32_t* g_iq3xxs = nullptr;
const std::uint8_t* g_ksigns = nullptr;
const std::uint8_t* g_kmask = nullptr;
const std::uint64_t* g_iq1s = nullptr;

// IQ1_S (50B: fp16 d + qs[32] + qh[uint16*8]). 11-bit grid index (qs + 3 bits
// from qh), per-group scale dl and +/-IQ1S_DELTA offset. Follows ggml
// dequantize_row_iq1_s (iq1s_grid is uint64 = 8 int8).
template <typename T>
sycl::event iq1s_typed(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                       std::size_t N, std::size_t K, const std::uint64_t* grid) {
  constexpr float kDelta = 0.125f;
  const std::size_t sb = K / 256, row_bytes = sb * 50;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  return q.parallel_for(sycl::nd_range<1>(nwg * kWG, kWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < sb; b += kSG) {
      const std::uint8_t* blk = wrow + b * 50;
      const float d = load_half(blk);
      const std::uint8_t* qs = blk + 2;
      const std::uint8_t* qhb = blk + 34;
      const std::size_t kbase = b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint16_t qh = static_cast<std::uint16_t>(qhb[2 * ib]) | (static_cast<std::uint16_t>(qhb[2 * ib + 1]) << 8);
        const float dl = d * (2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) ? -kDelta : kDelta;
        for (int l = 0; l < 4; ++l) {
          const std::uint32_t gi = qs[4 * ib + l] | (((qh >> (3 * l)) & 7) << 8);
          const std::uint64_t g = grid[gi];
          const std::size_t yb = kbase + ib * 32 + l * 8;
          for (int j = 0; j < 8; ++j) {
            const int gv = static_cast<int>(static_cast<std::int8_t>((g >> (8 * j)) & 0xFF));
            acc += dl * (gv + delta) * static_cast<float>(x[yb + j]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// IQ2_XXS (66B: fp16 d + qs[uint16*32]). Per 32-group: 4 grid lookups from
// iq2xxs_grid[aux8[l]] (uint64 = 8 int8), signs from ksigns, db scale from the
// top nibble of the second word. Follows ggml dequantize_row_iq2_xxs.
template <typename T>
sycl::event iq2xxs_typed(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                         std::size_t N, std::size_t K, const std::uint64_t* grid,
                         const std::uint8_t* ksigns, const std::uint8_t* kmask) {
  const std::size_t sb = K / 256, row_bytes = sb * 66;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  return q.parallel_for(sycl::nd_range<1>(nwg * kWG, kWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < sb; b += kSG) {
      const std::uint8_t* blk = wrow + b * 66;
      const float d = load_half(blk);
      const std::uint8_t* qs = blk + 2;
      const std::size_t kbase = b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint8_t* aux = qs + 8 * ib;
        const std::uint32_t a1 = rd_u32(aux + 4);
        const float db = d * (0.5f + (a1 >> 28)) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          const std::uint64_t g = grid[aux[l]];
          const std::uint8_t signs = ksigns[(a1 >> (7 * l)) & 127];
          const std::size_t yb = kbase + ib * 32 + l * 8;
          for (int j = 0; j < 8; ++j) {
            const int gv = static_cast<int>(static_cast<std::int8_t>((g >> (8 * j)) & 0xFF));
            const float s = (signs & kmask[j]) ? -1.0f : 1.0f;
            acc += db * gv * s * static_cast<float>(x[yb + j]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// IQ2_XS (74B: fp16 d + qs[uint16*32] + scales[8]). grid index = q2[l] & 511,
// sign = q2[l] >> 9; sub-scales from the scales byte nibbles.
template <typename T>
sycl::event iq2xs_typed(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                        std::size_t N, std::size_t K, const std::uint64_t* grid,
                        const std::uint8_t* ksigns, const std::uint8_t* kmask) {
  const std::size_t sb = K / 256, row_bytes = sb * 74;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  return q.parallel_for(sycl::nd_range<1>(nwg * kWG, kWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < sb; b += kSG) {
      const std::uint8_t* blk = wrow + b * 74;
      const float d = load_half(blk);
      const std::uint8_t* qsb = blk + 2;
      const std::uint8_t* scales = blk + 66;
      const std::size_t kbase = b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const int ls1 = scales[ib] & 0xF, ls2 = scales[ib] >> 4;
        const float db1 = d * (0.5f + ls1) * 0.25f, db2 = d * (0.5f + ls2) * 0.25f;
        for (int l = 0; l < 4; ++l) {
          const std::uint8_t* p = qsb + 8 * ib + 2 * l;
          const std::uint16_t q2 = static_cast<std::uint16_t>(p[0]) | (static_cast<std::uint16_t>(p[1]) << 8);
          const std::uint64_t g = grid[q2 & 511];
          const std::uint8_t signs = ksigns[q2 >> 9];
          const float db = (l < 2) ? db1 : db2;
          const std::size_t yb = kbase + ib * 32 + l * 8;
          for (int j = 0; j < 8; ++j) {
            const int gv = static_cast<int>(static_cast<std::int8_t>((g >> (8 * j)) & 0xFF));
            const float s = (signs & kmask[j]) ? -1.0f : 1.0f;
            acc += db * gv * s * static_cast<float>(x[yb + j]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

// IQ3_XXS (98B: fp16 d + qs[96] = 64 grid-index bytes + 32 scale/sign bytes).
// iq3xxs_grid is uint32 (4 int8 each); two grid lookups per l. Follows ggml.
template <typename T>
sycl::event iq3xxs_typed(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                         std::size_t N, std::size_t K, const std::uint32_t* grid,
                         const std::uint8_t* ksigns, const std::uint8_t* kmask) {
  const std::size_t sb = K / 256, row_bytes = sb * 98;
  const std::size_t nwg = (N + kRowsPerWG - 1) / kRowsPerWG;
  return q.parallel_for(sycl::nd_range<1>(nwg * kWG, kWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(kSG)]] {
    const sycl::sub_group sg = it.get_sub_group();
    const std::size_t n = it.get_group(0) * kRowsPerWG + sg.get_group_linear_id();
    const int lane = static_cast<int>(sg.get_local_linear_id());
    if (n >= N) return;
    const std::uint8_t* wrow = w + n * row_bytes;
    float acc = 0.0f;
    for (std::size_t b = lane; b < sb; b += kSG) {
      const std::uint8_t* blk = wrow + b * 98;
      const float d = load_half(blk);
      const std::uint8_t* q3 = blk + 2;          // 64 grid-index bytes
      const std::uint8_t* gas = blk + 2 + 64;    // 32 scale/sign bytes
      const std::size_t kbase = b * 256;
      for (int ib = 0; ib < 8; ++ib) {
        const std::uint32_t a = rd_u32(gas + 4 * ib);
        const float db = d * (0.5f + (a >> 28)) * 0.5f;
        const std::uint8_t* q3b = q3 + 8 * ib;
        for (int l = 0; l < 4; ++l) {
          const std::uint8_t signs = ksigns[(a >> (7 * l)) & 127];
          const std::uint32_t g1 = grid[q3b[2 * l + 0]];
          const std::uint32_t g2 = grid[q3b[2 * l + 1]];
          const std::size_t yb = kbase + ib * 32 + l * 8;
          for (int j = 0; j < 4; ++j) {
            const int v1 = static_cast<int>(static_cast<std::int8_t>((g1 >> (8 * j)) & 0xFF));
            const int v2 = static_cast<int>(static_cast<std::int8_t>((g2 >> (8 * j)) & 0xFF));
            const float s1 = (signs & kmask[j + 0]) ? -1.0f : 1.0f;
            const float s2 = (signs & kmask[j + 4]) ? -1.0f : 1.0f;
            acc += db * v1 * s1 * static_cast<float>(x[yb + j]);
            acc += db * v2 * s2 * static_cast<float>(x[yb + 4 + j]);
          }
        }
      }
    }
    const float sum = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
    if (lane == 0) y[n] = static_cast<T>(sum);
  });
}

template <typename T>
sycl::event dispatch_type(sycl::queue& q, const std::uint8_t* w, const T* x, T* y,
                          std::size_t N, std::size_t K, int type) {
  if (type == 12 || type == 13 || type == 14) {
    const auto* ks = dev_table(q, ksigns_iq2xs, 128, &g_ksigns);
    const auto* km = dev_table(q, kmask_iq2xs, 8, &g_kmask);
    if (type == 12) return iq2xxs_typed<T>(q, w, x, y, N, K, dev_table(q, iq2xxs_grid, 256, &g_iq2xxs), ks, km);
    if (type == 13) return iq2xs_typed<T>(q, w, x, y, N, K, dev_table(q, iq2xs_grid, 512, &g_iq2xs), ks, km);
    return iq3xxs_typed<T>(q, w, x, y, N, K, dev_table(q, iq3xxs_grid, 256, &g_iq3xxs), ks, km);
  }
  if (type == 15) return iq1s_typed<T>(q, w, x, y, N, K, dev_table(q, iq1s_grid, 2048, &g_iq1s));
  if (type == 7) return gguf_iq4nl_typed<T>(q, w, x, y, N, K);
  if (type == 8) return gguf_legacy_typed<T, 8>(q, w, x, y, N, K);
  if (type == 9) return gguf_legacy_typed<T, 9>(q, w, x, y, N, K);
  if (type == 10) return gguf_legacy_typed<T, 10>(q, w, x, y, N, K);
  if (type == 11) return gguf_iq4xs_typed<T>(q, w, x, y, N, K);
  if (type == 2) return gguf_q6k_typed<T>(q, w, x, y, N, K);
  if (type == 3) return gguf_q4k_typed<T>(q, w, x, y, N, K);
  if (type == 4) return gguf_q5k_typed<T>(q, w, x, y, N, K);
  if (type == 5) return gguf_q2k_typed<T>(q, w, x, y, N, K);
  if (type == 6) return gguf_q3k_typed<T>(q, w, x, y, N, K);
  return type == 0 ? gguf_typed<T, 0>(q, w, x, y, N, K)
                   : gguf_typed<T, 1>(q, w, x, y, N, K);
}

}  // namespace

sycl::event gguf_gemv_sycl(sycl::queue& q, const void* w_blocks, const void* x,
                           void* y, std::size_t N, std::size_t K, int type,
                           DType act_dt) {
  const auto* w = static_cast<const std::uint8_t*>(w_blocks);
  switch (act_dt) {
    case DType::f32:
      return dispatch_type(q, w, static_cast<const float*>(x), static_cast<float*>(y), N, K, type);
    case DType::f16:
      return dispatch_type(q, w, static_cast<const half_t*>(x), static_cast<half_t*>(y), N, K, type);
    case DType::bf16:
      return dispatch_type(q, w, static_cast<const bf16_t*>(x), static_cast<bf16_t*>(y), N, K, type);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
