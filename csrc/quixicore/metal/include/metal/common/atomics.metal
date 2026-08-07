/**
 * @file
 * @brief Device atomic helpers (P3) — the codebase's only atomics.
 *
 * Thin wrappers over relaxed device atomics used by the histogram/scatter steps
 * (MoE routing, sampling penalties). Also provides a float atomic-max via the
 * order-preserving uint bit-mapping (sign-flip trick), for future reductions.
 */

#pragma once

#include <metal_stdlib>   // standalone compile (in the umbrella, utils.metal pulls this first)

namespace mittens {

/** counts[idx] += v  (relaxed device atomic). */
static METAL_FUNC void atomic_add(device metal::atomic_int *counts, int idx, int v) {
    metal::atomic_fetch_add_explicit(&counts[idx], v, metal::memory_order_relaxed);
}

/** Returns the previous value of counts[idx], then increments it (relaxed). */
static METAL_FUNC int atomic_fetch_inc(device metal::atomic_int *counts, int idx) {
    return metal::atomic_fetch_add_explicit(&counts[idx], 1, metal::memory_order_relaxed);
}

/** Map a float to a uint whose unsigned order matches the float's order
 *  (flip the sign bit for positives; flip all bits for negatives). */
static METAL_FUNC uint float_to_orderable_uint(float f) {
    uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static METAL_FUNC float orderable_uint_to_float(uint u) {
    // Atomic max scratch buffers are cleared with the shared integer-zero
    // kernel. Treat that raw-zero initializer as +0.0 when no positive value
    // reached the accumulator; no finite float maps to orderable uint zero.
    if (u == 0) { return 0.0f; }
    uint f = (u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u;
    return as_type<float>(f);
}

/** p = max(p, v)  for a float stored in a device atomic_uint via the mapping above. */
static METAL_FUNC void atomic_max_float(device metal::atomic_uint *p, float v) {
    metal::atomic_fetch_max_explicit(p, float_to_orderable_uint(v), metal::memory_order_relaxed);
}

/** table[idx] += v  (relaxed device atomic float). Metal 3.1 native float atomics — used by the
 *  gradient scatter-add paths (embedding backward and any dW/dtable accumulation). idx is `long`
 *  because gradient tables (vocab*D) can exceed int range. */
static METAL_FUNC void atomic_add_float(device metal::atomic_float *table, long idx, float v) {
    metal::atomic_fetch_add_explicit(&table[idx], v, metal::memory_order_relaxed);
}

} // namespace mittens
