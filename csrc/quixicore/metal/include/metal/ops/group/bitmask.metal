/**
 * @file
 * @brief Packed allow-bitmask token test (structured-output / grammar masking).
 *
 * A row of the bitmask is ceil(V/32) uint32 words; bit v of the row (word v>>5, bit v&31) is 1 iff
 * token v is allowed. Free function (mirrors scan.metal): included from ops/ops.metal, not from
 * group/group.metal (whose includes nest inside the group<> struct body).
 */

#pragma once

#include "../../common/utils.metal"   // standalone compile; pragma-once-safe in the umbrella

namespace mittens {

/** @brief Is token `v` allowed by the packed allow-bitmask row `bitmask` (ceil(V/32) uint words)? */
static METAL_FUNC bool token_allowed(device const uint *bitmask, int v) {
    return ((bitmask[v >> 5] >> (v & 31)) & 1u) != 0u;
}

} // namespace mittens
