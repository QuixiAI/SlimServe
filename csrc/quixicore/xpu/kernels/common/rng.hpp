#pragma once

// Counter-based (stateless) RNG for the XPU backend. Deterministic uniform in
// [0,1) from a (seed, index) pair — no per-thread state, reproducible, and
// order-independent. Good enough for dropout masks and categorical sampling
// (not cryptographic). Used by utils/dropout and the sampling family.

#include <cstdint>

namespace quixicore::xpu::kernels::detail {

// PCG-style output hash on a 32-bit word.
inline std::uint32_t pcg_hash(std::uint32_t x) {
  x = x * 747796405u + 2891336453u;
  const std::uint32_t w = ((x >> ((x >> 28) + 4)) ^ x) * 277803737u;
  return (w >> 22) ^ w;
}

// Uniform float in [0,1) for the given stream `seed` and 64-bit counter `idx`.
inline float uniform01(std::uint32_t seed, std::uint64_t idx) {
  const std::uint32_t lo = static_cast<std::uint32_t>(idx);
  const std::uint32_t hi = static_cast<std::uint32_t>(idx >> 32);
  const std::uint32_t h = pcg_hash(lo ^ (seed * 2654435761u) ^ pcg_hash(hi));
  return static_cast<float>(h >> 8) * (1.0f / 16777216.0f);  // 24-bit mantissa
}

}  // namespace quixicore::xpu::kernels::detail
