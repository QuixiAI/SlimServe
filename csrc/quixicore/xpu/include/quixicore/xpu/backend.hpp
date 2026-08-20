#pragma once

#include <array>
#include <string_view>

namespace quixicore::xpu {

struct BackendInfo {
  std::string_view backend;
  std::string_view name;
  std::string_view repo;
  std::string_view umbrella;
  std::string_view contract;
  std::string_view status;
  std::array<std::string_view, 3> targets;
  std::array<std::string_view, 3> integrations;
};

inline constexpr BackendInfo kBackendInfo{
    "xpu",
    "QuixiCore XPU",
    "QuixiAI/QuixiCore-XPU",
    "QuixiAI/QuixiCore",
    "v0.1",
    "planned",
    {"intel_arc", "intel_data_center_gpu", "future_xpu"},
    {"oneAPI", "SYCL", "Level Zero"},
};

inline constexpr std::array<std::string_view, 16> kContractKernelFamilies{
    "norms",
    "softmax",
    "activations",
    "causal_attention",
    "paged_attention",
    "mla_decode",
    "quant_gemv",
    "quant_gemm",
    "quantized_lm_head",
    "sampling",
    "beam_search",
    "speculative_decode",
    "mamba_ssd",
    "moe_routing",
    "grouped_moe_gemm",
    "optimizers",
};

const BackendInfo& runtime_backend_info() noexcept;
bool is_contract_kernel_family(std::string_view family) noexcept;
std::string_view status_for_kernel_family(std::string_view family) noexcept;

}  // namespace quixicore::xpu
