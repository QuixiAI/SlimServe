// SPDX-License-Identifier: Apache-2.0
// GLM53 NVFP4/BF16 TP4 prefill: all eight warps span N, with no cross-warp
// K reduction. Three prefetch stages keep the 64x512 tile within SM120's
// 100 KiB shared-memory budget. Decode keeps the existing Marlin schedules.
#include "kernel.h"
#include "marlin_template.h"

#include <cstdlib>
#include <cstring>

namespace MARLIN_NAMESPACE_NAME {

cudaError_t launch_glm53_sm120_prefill(MARLIN_KERNEL_PARAMS, int sms,
                                      cudaStream_t stream) {
  // Shared layout from marlin_template.h: max(B/reduction,bias overlay),
  // BF16 A stages, byte-wide FP8 scales, and padded assignment metadata.
  // The kernel also has 1024 bytes of static shared memory.
  constexpr int smem = 64 * (512 + 8) * 2 + 3 * 64 * 64 * 2 +
                       3 * (64 / 16) * 512 + 64 * 16;
  static_assert(smem == 98304 && smem + 1024 <= 102400);
  auto kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                       vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                       256, 4, 32, 4, false, 3, 1, false>;
  static const bool fixed_shapes = [] {
    const char* value = std::getenv("VLLM_GLM53_MARLIN_PREFILL_FIXED");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  // ops.cu has already checked both shapes and every specialized policy.
  // The flag keeps a same-binary control available for serving measurements.
  if (fixed_shapes) {
    if (top_k == 8) {
      kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                      vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                      256, 4, 32, 4, false, 3, 1, false, 1>;
    } else {
      kernel = Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                      vllm::kBFloat16.id(), vllm::kFE4M3fn.id(),
                      256, 4, 32, 4, false, 3, 1, false, 2>;
    }
  }
  auto error = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  if (error != cudaSuccess) return error;
  kernel<<<sms, 256, smem, stream>>>(
      A, B, C, C_tmp, b_bias_ptr, a_scales_ptr, scales_ptr, global_scale_ptr,
      zp_ptr, g_idx, sorted_token_ids_ptr, expert_ids_ptr,
      num_tokens_past_padded_ptr, topk_weights_ptr, top_k, mul_topk_weights,
      num_groups, prob_m, prob_n, prob_k, locks, has_bias, use_atomic_add,
      use_fp32_reduce);
  return cudaGetLastError();
}

}  // namespace MARLIN_NAMESPACE_NAME
