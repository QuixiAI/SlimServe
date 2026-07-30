/**
 * @file
 * @brief CUDA port of ThunderMittens' counter-based RNG (include/common/rng.metal).
 *
 * THE CONTRACT: bit-exact with TM's Metal version and its numpy reproduction —
 * every sampling/dropout/spec-decode test derives its oracle by replaying this
 * exact integer finalizer on the host. Do not "improve" it.
 *
 * Stateless: uniform is a pure function of (seed, a, b). numpy reproduction:
 *   x = (seed*0x9E3779B9 + a*0x85EBCA77 + b*0xC2B2AE3D) & 0xFFFFFFFF
 *   x ^= x>>16; x = (x*0x7FEB352D) & 0xFFFFFFFF
 *   x ^= x>>15; x = (x*0x846CA68B) & 0xFFFFFFFF
 *   x ^= x>>16
 *   u = (x >> 8) * (1/16777216)
 */
#pragma once
#include <cstdint>

namespace tmq {

__host__ __device__ __forceinline__ float rng_uniform(uint32_t seed, uint32_t a, uint32_t b) {
    uint32_t x = seed * 0x9E3779B9u + a * 0x85EBCA77u + b * 0xC2B2AE3Du;
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return float(x >> 8) * (1.0f / 16777216.0f);   // 24-bit mantissa -> [0,1)
}

__host__ __device__ __forceinline__ float rng_gumbel(uint32_t seed, uint32_t a, uint32_t b) {
    const float u = fmaxf(rng_uniform(seed, a, b), 1e-20f);
    return -logf(-logf(u));
}

}  // namespace tmq
