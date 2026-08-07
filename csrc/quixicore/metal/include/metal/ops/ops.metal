#pragma once
#include "group/group.metal"
#include "group/scan.metal"    // P2 — free-function threadgroup parallel scan
#include "group/bitmask.metal" // packed allow-bitmask token test (grammar masking)
#include "warp/warp.metal"
#include "group/topk.metal"    // masked-simd_argmax top-k helper (needs warp.metal's simd_argmax)
