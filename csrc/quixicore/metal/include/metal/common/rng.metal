/**
 * @file
 * @brief Counter-based hash RNG (P4) — reproducible, NOT cryptographic.
 *
 * Stateless: the uniform is a pure function of a (seed, a, b) counter, so a GPU
 * draw is exactly reproducible on the host (the same integer finalizer in numpy)
 * — giving stochastic kernels an exact, deterministic oracle. Used by the sampling
 * kernels (Gumbel-max categorical / top-k / top-p).
 */

#pragma once

#include <metal_stdlib>   // standalone compile (in the umbrella, utils.metal pulls this first)

namespace mittens {

/** Uniform in [0,1) from a Murmur3-style integer finalizer over a mixed counter. */
static METAL_FUNC float rng_uniform(uint seed, uint a, uint b) {
    uint x = seed * 0x9E3779B9u + a * 0x85EBCA77u + b * 0xC2B2AE3Du;
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return float(x >> 8) * (1.0f / 16777216.0f);   // 24-bit mantissa -> [0,1)
}

/** Gumbel(0,1) noise from the same stream: g = -log(-log(u)). */
static METAL_FUNC float rng_gumbel(uint seed, uint a, uint b) {
    const float u = metal::max(rng_uniform(seed, a, b), 1e-20f);
    return -metal::log(-metal::log(u));
}

} // namespace mittens
