#include "quixicore/xpu/variants.hpp"

namespace quixicore::xpu {

const char* variant_name(const Variant v) noexcept {
  switch (v) {
    case Variant::sycl:
      return "sycl";
    case Variant::vendor:
      return "vendor";
    case Variant::best:
      return "best";
  }
  return "unknown";
}

bool variant_available(const Variant v) noexcept {
  switch (v) {
    case Variant::sycl:
      return true;
    case Variant::vendor:
#if defined(QUIXICORE_XPU_HAS_ONEDNN)
      return true;
#else
      return false;
#endif
    case Variant::best:
      return true;
  }
  return false;
}

Variant resolve_variant(const Variant requested) noexcept {
  switch (requested) {
    case Variant::sycl:
      return Variant::sycl;
    case Variant::vendor:
      return variant_available(Variant::vendor) ? Variant::vendor : Variant::sycl;
    case Variant::best:
      // No per-shape perf model yet: prefer native SYCL. Once optimization runs
      // land, this is where the empirically faster variant is chosen.
      return Variant::sycl;
  }
  return Variant::sycl;
}

}  // namespace quixicore::xpu
