#pragma once

// Implementation-variant selector for QuixiCore XPU ops.
//
// The XPU backend ships two co-equal implementations of each op, analogous to
// how the Metal backend ships two co-equal framework bindings (PyTorch + MLX):
//
//   * Variant::sycl   -- native hand-written SYCL (joint_matrix/DPAS, subgroups)
//   * Variant::vendor -- Intel vendor library path (oneDNN / oneMKL / XeTLA)
//
// Both are real, tested, and benchmarked; callers pick. `Variant::best` lets the
// dispatch layer choose the empirically faster path for the given op/shape once
// perf data exists (until then it maps to sycl).
//
// A vendor variant may not be compiled into a given build (e.g. oneDNN absent);
// use variant_available() / resolve_variant() to query and fall back.

#include <cstdint>

namespace quixicore::xpu {

enum class Variant : std::uint8_t {
  sycl,
  vendor,
  best,
};

// "sycl" / "vendor" / "best".
const char* variant_name(Variant v) noexcept;

// True if the given variant is compiled into this build. `sycl` is always
// available; `vendor` depends on QUIXICORE_XPU_HAS_ONEDNN at build time.
bool variant_available(Variant v) noexcept;

// Resolve `best` (and any unavailable request) to a concrete, available variant.
// Never returns `best`. Falls back to `sycl` when a vendor path is unavailable.
Variant resolve_variant(Variant requested) noexcept;

}  // namespace quixicore::xpu
