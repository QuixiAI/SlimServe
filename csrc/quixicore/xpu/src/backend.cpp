#include "quixicore/xpu/backend.hpp"

namespace quixicore::xpu {

const BackendInfo& runtime_backend_info() noexcept {
  return kBackendInfo;
}

bool is_contract_kernel_family(const std::string_view family) noexcept {
  for (const auto known_family : kContractKernelFamilies) {
    if (family == known_family) {
      return true;
    }
  }
  return false;
}

std::string_view status_for_kernel_family(const std::string_view family) noexcept {
  if (is_contract_kernel_family(family)) {
    return "planned";
  }
  return "unknown";
}

}  // namespace quixicore::xpu
