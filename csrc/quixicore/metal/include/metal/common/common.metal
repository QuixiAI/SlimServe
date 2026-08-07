/**
 * @file
 * @brief A collection of common resources on which Thundermittens depends.
 */


#pragma once
#include "base_types.metal"
#include "base_ops.metal"
#include "utils.metal"
#include "atomics.metal"   // P3 — device atomics (after utils: metal_stdlib is available)
#include "rng.metal"       // P4 — counter-based hash RNG
#include "glu_eval.metal"  // shared gated-activation forward (glu.metal + act_quant.metal)
