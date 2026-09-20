#pragma once

// TurboQuant codec core (format version 2, specs/formats/turboquant.md),
// shared VERBATIM between the SYCL kernels and the host test oracle: the same
// inline functions compile for device and host, so "codes match the reference
// bit for bit" holds by construction. Only operations with identical
// correctly-rounded semantics on both sides are used (fp32 +-*/ and sqrt,
// floor, fp32<->fp16 conversions, integer bit ops).
//
// Precision chain (the load-bearing part):
//   keys (bits < 8): u = sign .* x (fp32) -> unnormalized FWHT (fp32,
//     iterative butterflies in ascending span order) -> * 1/sqrt(head_size)
//     (fp32) -> per-32-group RMS accumulated sequentially in fp32, rounded to
//     fp16 (a zero RMS becomes fp16(1) so all-zero groups encode
//     deterministically) -> element / fp16-RMS (fp32 divide by the rounded
//     value) -> code = count of strictly-greater midpoints (lower centroid
//     wins equality). Decode is the same transform applied to
//     centroid*fp16-RMS (FWHT is its own inverse up to head_size, which the
//     1/sqrt factors absorb), then the sign flip.
//   keys (bits == 8): unrotated saturating-RTNE e4m3 byte per element; no
//     signs, transform, scales, or centroid table.
//   values: per-32-group min/max in fp32 (sequential), scale =
//     fp16(range/(2^bits-1)) (zero range -> fp16(1)), zero = fp16(min/scale),
//     code = clamp(round-half-away(v/scale - zero)); decode
//     (code + zero) * scale. value_signed with bits == 8 interprets the byte
//     as two's-complement with clamp [-128,127]; other signed widths keep the
//     unsigned grid per the spec.
//   packing: LSB-first contiguous bit stream, element i at bit i*bits, row =
//     ceil(head_size*bits/8) bytes, unused high bits zero.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

namespace quixicore::xpu::turboquant_codec {

inline constexpr std::size_t kGroup = 32;
inline constexpr std::size_t kMaxHead = 256;

using half = sycl::half;

inline float round_half_away(float t) {
  return t >= 0.0f ? sycl::floor(t + 0.5f) : sycl::ceil(t - 0.5f);
}

// Manual RTNE fp32<->fp16 conversions. The device optimizer is entitled to
// elide a float->half->float round-trip (it did: the fp16-rounded scale
// reached the divide unrounded, ~1000 ULP off the host result), so the codec
// never uses compiler half conversions — the fp16 grid is applied through
// these integer implementations, identical on host and device by
// construction. sycl::half remains the storage type only.
inline std::uint16_t f32_to_f16_bits(float x) {
  std::uint32_t b = sycl::bit_cast<std::uint32_t>(x);
  const std::uint32_t sign = (b >> 16) & 0x8000u;
  b &= 0x7fffffffu;
  if (b >= 0x7f800000u)
    return static_cast<std::uint16_t>(
        sign | 0x7c00u | (b > 0x7f800000u ? 0x200u : 0u));
  const int exp32 = static_cast<int>(b >> 23) - 127;
  if (exp32 >= -14) {
    const std::uint32_t mant = b & 0x7fffffu;
    std::uint32_t keep = mant >> 13;
    const std::uint32_t rest = mant & 0x1fffu;
    std::uint32_t e = static_cast<std::uint32_t>(exp32 + 15);
    if (rest > 0x1000u || (rest == 0x1000u && (keep & 1u))) {
      if (++keep == 0x400u) {
        keep = 0;
        ++e;
      }
    }
    if (e >= 31u) return static_cast<std::uint16_t>(sign | 0x7c00u);
    return static_cast<std::uint16_t>(sign | (e << 10) | keep);
  }
  const int shift = -exp32 - 1;  // sig24 * 2^(exp32+1) -> subnormal grid
  if (shift >= 25) return static_cast<std::uint16_t>(sign);
  const std::uint32_t sig = 0x800000u | (b & 0x7fffffu);
  std::uint32_t m = sig >> shift;
  const std::uint32_t rem = sig & ((1u << shift) - 1u);
  const std::uint32_t halfway = 1u << (shift - 1);
  if (rem > halfway || (rem == halfway && (m & 1u))) ++m;
  return static_cast<std::uint16_t>(sign | m);
}

inline float f16_bits_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t e = (h >> 10) & 0x1fu;
  const std::uint32_t m = h & 0x3ffu;
  if (e == 0) {
    if (m == 0) return sycl::bit_cast<float>(sign);
    const float v = static_cast<float>(m) * 5.9604644775390625e-08f;  // 2^-24
    return (h & 0x8000u) ? -v : v;
  }
  if (e == 31u)
    return sycl::bit_cast<float>(sign | 0x7f800000u | (m << 13));
  return sycl::bit_cast<float>(sign | ((e + 112u) << 23) | (m << 13));
}

inline half f16_round(float x) {
  return sycl::bit_cast<half>(f32_to_f16_bits(x));
}
inline float f16_value(half h) {
  return f16_bits_to_f32(sycl::bit_cast<std::uint16_t>(h));
}

// Saturating RTNE fp32 -> e4m3 (bias 7, max finite 448, 0x7f = NaN).
inline std::uint8_t e4m3_encode(float x) {
  const std::uint32_t bits = sycl::bit_cast<std::uint32_t>(x);
  const std::uint8_t sign = static_cast<std::uint8_t>((bits >> 24) & 0x80u);
  const std::uint32_t abs_bits = bits & 0x7fffffffu;
  if (abs_bits > 0x7f800000u) return sign | 0x7fu;  // NaN
  const float ax = sycl::bit_cast<float>(abs_bits);
  if (ax > 448.0f) return sign | 0x7eu;  // saturate (incl. +inf)
  if ((abs_bits >> 23) == 0u) return sign;  // fp32 zero/subnormal -> 0
  const int e32 = static_cast<int>(abs_bits >> 23) - 127;
  if (e32 >= -6) {
    std::uint32_t mant = abs_bits & 0x7fffffu;
    std::uint32_t keep = mant >> 20;
    const std::uint32_t rest = mant & 0xfffffu;
    if (rest > 0x80000u || (rest == 0x80000u && (keep & 1u))) ++keep;
    int exp4 = e32 + 7;
    if (keep == 8u) {
      keep = 0u;
      ++exp4;
    }
    if (exp4 > 15 || (exp4 == 15 && keep == 7u)) return sign | 0x7eu;
    return sign |
           static_cast<std::uint8_t>((static_cast<std::uint32_t>(exp4) << 3) |
                                     keep);
  }
  // Subnormal target: value = k * 2^-9, k in [0,7], RTNE.
  const float t = ax * 512.0f;
  std::uint32_t k = static_cast<std::uint32_t>(t);
  const float frac = t - static_cast<float>(k);
  if (frac > 0.5f || (frac == 0.5f && (k & 1u))) ++k;
  if (k >= 8u) return sign | static_cast<std::uint8_t>(1u << 3);
  return sign | static_cast<std::uint8_t>(k);
}

inline float e4m3_decode(std::uint8_t b) {
  const float sign = (b & 0x80u) ? -1.0f : 1.0f;
  const int exp4 = (b >> 3) & 0xf;
  const int mant = b & 7;
  if (exp4 == 15 && mant == 7)
    return sign * sycl::bit_cast<float>(0x7fc00000u);  // NaN
  if (exp4 == 0) return sign * (static_cast<float>(mant) / 512.0f);
  const float scale = sycl::bit_cast<float>(
      static_cast<std::uint32_t>(exp4 - 10 + 127) << 23);
  return sign * static_cast<float>(8 + mant) * scale;
}

inline std::size_t row_bytes(std::size_t head_size, int bits) {
  return (head_size * static_cast<std::size_t>(bits) + 7u) / 8u;
}

inline void pack_codes(const std::uint8_t* codes, std::size_t n, int bits,
                       std::uint8_t* out) {
  const std::size_t nbytes = row_bytes(n, bits);
  for (std::size_t b = 0; b < nbytes; ++b) out[b] = 0u;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t bit = i * static_cast<std::size_t>(bits);
    std::uint32_t v = codes[i] & ((1u << bits) - 1u);
    if (bits == 8) v = codes[i];
    out[bit >> 3] |= static_cast<std::uint8_t>(v << (bit & 7u));
    const std::size_t spill = (bit & 7u) + static_cast<std::size_t>(bits);
    if (spill > 8u)
      out[(bit >> 3) + 1] |= static_cast<std::uint8_t>(v >> (8u - (bit & 7u)));
  }
}

inline std::uint8_t unpack_code(const std::uint8_t* row, std::size_t i,
                                int bits) {
  const std::size_t bit = i * static_cast<std::size_t>(bits);
  std::uint32_t v = static_cast<std::uint32_t>(row[bit >> 3]) >> (bit & 7u);
  const std::size_t spill = (bit & 7u) + static_cast<std::size_t>(bits);
  if (spill > 8u)
    v |= static_cast<std::uint32_t>(row[(bit >> 3) + 1]) << (8u - (bit & 7u));
  if (bits == 8) return static_cast<std::uint8_t>(v);
  return static_cast<std::uint8_t>(v & ((1u << bits) - 1u));
}

// Unnormalized FWHT, iterative ascending-span butterflies (self-inverse up to
// a factor of n). n is a power of two.
inline void fwht(float* u, std::size_t n) {
  for (std::size_t len = 1; len < n; len <<= 1) {
    for (std::size_t i = 0; i < n; i += len << 1) {
      for (std::size_t j = i; j < i + len; ++j) {
        const float a = u[j];
        const float b = u[j + len];
        u[j] = a + b;
        u[j + len] = a - b;
      }
    }
  }
}

// ---- Keys ----

template <typename T>
inline void encode_key_row(const T* key, std::uint8_t* row,
                           half* group_scales, const float* centroids,
                           const float* signs, std::size_t head_size,
                           int key_bits, float* scratch /* [head_size] */) {
  if (key_bits == 8) {
    for (std::size_t i = 0; i < head_size; ++i)
      row[i] = e4m3_encode(static_cast<float>(key[i]));
    return;
  }
  const float inv_root = 1.0f / sycl::sqrt(static_cast<float>(head_size));
  for (std::size_t i = 0; i < head_size; ++i)
    scratch[i] = static_cast<float>(key[i]) * signs[i];
  fwht(scratch, head_size);
  for (std::size_t i = 0; i < head_size; ++i) scratch[i] *= inv_root;

  const int n_mid = (1 << key_bits) - 1;
  std::uint8_t codes[kMaxHead];
  for (std::size_t g = 0; g < head_size / kGroup; ++g) {
    float ss = 0.0f;
    for (std::size_t j = 0; j < kGroup; ++j) {
      const float v = scratch[g * kGroup + j];
      ss += v * v;
    }
    half rms = f16_round(sycl::sqrt(ss / static_cast<float>(kGroup)));
    if (f16_value(rms) == 0.0f) rms = f16_round(1.0f);
    group_scales[g] = rms;
    const float inv = f16_value(rms);
    for (std::size_t j = 0; j < kGroup; ++j) {
      const float z = scratch[g * kGroup + j] / inv;
      std::uint32_t code = 0;
      for (int m = 0; m < n_mid; ++m) {
        const float mid = 0.5f * (centroids[m] + centroids[m + 1]);
        code += z > mid ? 1u : 0u;  // lower centroid wins equality
      }
      codes[g * kGroup + j] = static_cast<std::uint8_t>(code);
    }
  }
  pack_codes(codes, head_size, key_bits, row);
}

inline void decode_key_row(const std::uint8_t* row, const half* group_scales,
                           const float* centroids, const float* signs,
                           std::size_t head_size, int key_bits, float* out,
                           float* scratch /* [head_size] */) {
  if (key_bits == 8) {
    for (std::size_t i = 0; i < head_size; ++i)
      out[i] = e4m3_decode(row[i]);
    return;
  }
  const float inv_root = 1.0f / sycl::sqrt(static_cast<float>(head_size));
  for (std::size_t g = 0; g < head_size / kGroup; ++g) {
    const float rms = f16_value(group_scales[g]);
    for (std::size_t j = 0; j < kGroup; ++j) {
      const std::size_t i = g * kGroup + j;
      scratch[i] = centroids[unpack_code(row, i, key_bits)] * rms;
    }
  }
  fwht(scratch, head_size);
  for (std::size_t i = 0; i < head_size; ++i)
    out[i] = scratch[i] * inv_root * signs[i];
}

// ---- Values ----

template <typename T>
inline void encode_value_row(const T* value, std::uint8_t* row,
                             half* group_scales, half* group_zeros,
                             std::size_t head_size, int value_bits,
                             bool value_signed) {
  const bool twos = value_signed && value_bits == 8;
  const float cmax = twos ? 127.0f
                          : static_cast<float>((1u << value_bits) - 1u);
  const float cmin = twos ? -128.0f : 0.0f;
  std::uint8_t codes[kMaxHead];
  for (std::size_t g = 0; g < head_size / kGroup; ++g) {
    float lo = static_cast<float>(value[g * kGroup]);
    float hi = lo;
    for (std::size_t j = 1; j < kGroup; ++j) {
      const float v = static_cast<float>(value[g * kGroup + j]);
      lo = v < lo ? v : lo;
      hi = v > hi ? v : hi;
    }
    half scale = f16_round((hi - lo) / (cmax - cmin));
    half zero;
    if (f16_value(scale) == 0.0f) {
      scale = f16_round(1.0f);
      zero = f16_round(lo);
    } else {
      zero = f16_round(lo / f16_value(scale) - cmin);
    }
    group_scales[g] = scale;
    group_zeros[g] = zero;
    const float s = f16_value(scale);
    const float z = f16_value(zero);
    for (std::size_t j = 0; j < kGroup; ++j) {
      const float v = static_cast<float>(value[g * kGroup + j]);
      float c = round_half_away(v / s - z);
      c = c < cmin ? cmin : (c > cmax ? cmax : c);
      codes[g * kGroup + j] = twos
          ? static_cast<std::uint8_t>(static_cast<std::int8_t>(c))
          : static_cast<std::uint8_t>(c);
    }
  }
  pack_codes(codes, head_size, value_bits, row);
}

inline void decode_value_row(const std::uint8_t* row,
                             const half* group_scales, const half* group_zeros,
                             std::size_t head_size, int value_bits,
                             bool value_signed, float* out) {
  const bool twos = value_signed && value_bits == 8;
  for (std::size_t g = 0; g < head_size / kGroup; ++g) {
    const float s = f16_value(group_scales[g]);
    const float z = f16_value(group_zeros[g]);
    for (std::size_t j = 0; j < kGroup; ++j) {
      const std::size_t i = g * kGroup + j;
      const std::uint8_t raw = unpack_code(row, i, value_bits);
      const float code = twos
          ? static_cast<float>(static_cast<std::int8_t>(raw))
          : static_cast<float>(raw);
      out[i] = (code + z) * s;
    }
  }
}

}  // namespace quixicore::xpu::turboquant_codec
