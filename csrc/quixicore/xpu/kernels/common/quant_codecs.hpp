#pragma once

// Consolidated element decoders for the quantized formats the backend speaks,
// for NEW consumers (xmx-based kernels, stage-tile dequant lambdas). Existing
// shipped kernels keep their local copies until a bit-identical-regression
// refactor moves them over — decoder identity is part of each format's
// correctness contract, so consolidation is deliberate, not drive-by.
//
// All functions are deterministic host/device-shared code (integer bit
// manipulation + exact fp32 arithmetic only).

#include <cstdint>

#include <sycl/sycl.hpp>

namespace quixicore::xpu::qcodec {

// int4 two's-complement nibble (the qgemv_int4 / w4a16 encoding).
inline int s4(int nibble) { return nibble >= 8 ? nibble - 16 : nibble; }

// fp8 e4m3 (OCP, bias 7, max 448) — exact decode.
inline float fp8_e4m3(std::uint8_t b) {
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

// fp8 e5m2 (bias 15) — exact decode via bit relocation into fp16.
inline float fp8_e5m2(std::uint8_t b) {
  const std::uint16_t h = static_cast<std::uint16_t>(b) << 8;
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t e = (h >> 10) & 0x1fu;
  const std::uint32_t m = h & 0x3ffu;
  if (e == 0) {
    if (m == 0) return sycl::bit_cast<float>(sign);
    const float v = static_cast<float>(m) * 5.9604644775390625e-08f;
    return (h & 0x8000u) ? -v : v;
  }
  if (e == 31u) return sycl::bit_cast<float>(sign | 0x7f800000u | (m << 13));
  return sycl::bit_cast<float>(sign | ((e + 112u) << 23) | (m << 13));
}

// fp4 e2m1 nibble (nvfp4/mxfp4 element).
inline float e2m1(std::uint8_t nibble) {
  constexpr float kMag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float m = kMag[nibble & 0x7u];
  return (nibble & 0x8u) ? -m : m;
}

// e8m0 power-of-two scale (mxfp4 block scale): 2^(b-127); b==0 -> 2^-127.
inline float e8m0(std::uint8_t b) {
  if (b == 0) return sycl::bit_cast<float>(0x00400000u) * 0.5f;  // 2^-127
  if (b == 0xffu) return sycl::bit_cast<float>(0x7fc00000u);     // NaN
  return sycl::bit_cast<float>(static_cast<std::uint32_t>(b) << 23);
}

}  // namespace quixicore::xpu::qcodec
