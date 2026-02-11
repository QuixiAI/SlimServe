// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * HIP MoE Marlin GEMM entry point for CDNA/CDNA2/CDNA3.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <cstdint>
#include <cstdlib>

#include "core/scalar_type.hpp"
#include "moe/marlin_moe_hip_kernel.hip"

namespace marlin_moe_hip {

bool is_cdna3_arch(const std::string& arch) {
  return arch.rfind("gfx94", 0) == 0 || arch.rfind("gfx95", 0) == 0;
}

bool is_cdna_arch(const std::string& arch) {
  return arch.rfind("gfx90", 0) == 0 || is_cdna3_arch(arch);
}

// 256 threads = 4 waves. With thread_k_blocks=4 => wave_groups=1.
// No pipeline needed; each wave handles a k-slice, all reduce in shmem.
static constexpr int kMoeStages = 1;

#define CALL_MOE_IF(NUM_BITS, MOE_BLOCK_SIZE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    num_bits == NUM_BITS && moe_block_size == MOE_BLOCK_SIZE && \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && \
    thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS \
  ) { \
    constexpr int kShared = moe_shared_mem_bytes<NUM_BITS, MOE_BLOCK_SIZE>(); \
    (void)hipFuncSetAttribute( \
      (const void*)MarlinMoE<scalar_t, NUM_BITS, kIsFp4, kScaleE4M3, MOE_THREADS, \
                             THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                             kMoeStages, MOE_BLOCK_SIZE, GROUP_BLOCKS>, \
      hipFuncAttributeMaxDynamicSharedMemorySize, \
      kShared \
    ); \
    hipLaunchKernelGGL( \
      (MarlinMoE<scalar_t, NUM_BITS, kIsFp4, kScaleE4M3, MOE_THREADS, \
                 THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, \
                 kMoeStages, MOE_BLOCK_SIZE, GROUP_BLOCKS>), \
      dim3(blocks), dim3(MOE_THREADS), kShared, stream, \
      A_ptr, B_ptr, C_ptr, b_bias_ptr, a_scales_ptr, s_ptr, global_scale_ptr, \
      g_idx_ptr, b_zeros_ptr, \
      sorted_token_ids_ptr, expert_ids_ptr, num_tokens_past_padded_ptr, \
      topk_weights_ptr, top_k, mul_topk_weights, num_experts, \
      prob_m, prob_n, prob_k, num_groups, \
      locks, has_bias, has_act_order, has_zp, is_zp_float, a_is_fp8, a_fp8_is_fnuz, b_is_fp8, b_fp8_is_fnuz \
    ); \
  }

// Expand all supported (MOE_BLOCK_SIZE, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS)
// combos for a given NUM_BITS.
// thread_n_blocks=4 (64 cols) is default; thread_n_blocks=2 (32 cols) doubles blocks
// for better CU saturation on small M.
// thread_m_blocks = moe_block_size / 16.
// thread_k_blocks controls how many waves cooperate on the K dimension.
// With 256 threads = 4 waves:
//   thread_k_blocks=4 => 1 wave-group per workgroup (K-sliced), sequential tiles
//   thread_k_blocks=2 => 2 wave-groups per workgroup (more N parallelism)
//   thread_k_blocks=1 => 4 wave-groups per workgroup (max N parallelism)
#define CALL_MOE_CONFIGS_N(NUM_BITS, THREAD_N_BLOCKS) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 4, -1) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 4, 2) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 4, 4) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 4, 8) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 4, 16) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 2, -1) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 2, 2) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 2, 4) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 2, 8) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 2, 16) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 1, -1) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 1, 2) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 1, 4) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 1, 8) \
  CALL_MOE_IF(NUM_BITS, 8, 1, THREAD_N_BLOCKS, 1, 16) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 4, -1) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 4, 2) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 4, 4) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 4, 8) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 4, 16) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 2, -1) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 2, 2) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 2, 4) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 2, 8) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 2, 16) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 1, -1) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 1, 2) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 1, 4) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 1, 8) \
  CALL_MOE_IF(NUM_BITS, 16, 1, THREAD_N_BLOCKS, 1, 16) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 4, -1) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 4, 2) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 4, 4) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 4, 8) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 4, 16) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 2, -1) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 2, 2) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 2, 4) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 2, 8) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 2, 16) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 1, -1) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 1, 2) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 1, 4) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 1, 8) \
  CALL_MOE_IF(NUM_BITS, 32, 2, THREAD_N_BLOCKS, 1, 16) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 4, -1) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 4, 2) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 4, 4) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 4, 8) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 4, 16) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 2, -1) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 2, 2) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 2, 4) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 2, 8) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 2, 16) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 1, -1) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 1, 2) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 1, 4) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 1, 8) \
  CALL_MOE_IF(NUM_BITS, 48, 3, THREAD_N_BLOCKS, 1, 16) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 4, -1) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 4, 2) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 4, 4) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 4, 8) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 4, 16) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 2, -1) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 2, 2) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 2, 4) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 2, 8) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 2, 16) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 1, -1) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 1, 2) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 1, 4) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 1, 8) \
  CALL_MOE_IF(NUM_BITS, 64, 4, THREAD_N_BLOCKS, 1, 16)

#define CALL_MOE_CONFIGS(NUM_BITS) \
  CALL_MOE_CONFIGS_N(NUM_BITS, 4) \
  CALL_MOE_CONFIGS_N(NUM_BITS, 2)

template <typename scalar_t, bool kIsFp4, bool kScaleE4M3>
void moe_marlin_hip_gemm_impl(
    const void* A,
    const void* B,
    void* C,
    const void* b_bias,
    const float* a_scales,
    void* s,
    const void* global_scale,
    const int* g_idx,
    const void* b_zeros,
    const int32_t* sorted_token_ids,
    const int32_t* expert_ids,
    const int32_t* num_tokens_past_padded,
    const float* topk_weights,
    int top_k,
    bool mul_topk_weights,
    int num_experts,
    int prob_m,
    int prob_n,
    int prob_k,
    int num_groups,
    void* workspace,
    int dev,
    hipStream_t stream,
    int num_bits,
    int moe_block_size,
    int max_num_tokens_padded,
    int group_blocks,
    int sms,
    bool has_bias,
    bool has_act_order,
    bool has_zp,
    bool is_zp_float,
    bool a_is_fp8,
    bool a_fp8_is_fnuz,
    bool b_is_fp8,
    bool b_fp8_is_fnuz) {

  const void* A_ptr = A;
  const int4* B_ptr = reinterpret_cast<const int4*>(B);
  int4* C_ptr = reinterpret_cast<int4*>(C);
  const scalar_t* b_bias_ptr = reinterpret_cast<const scalar_t*>(b_bias);
  const float* a_scales_ptr = a_scales;
  const void* s_ptr = s;
  const scalar_t* global_scale_ptr = reinterpret_cast<const scalar_t*>(global_scale);
  const int* g_idx_ptr = g_idx;
  const void* b_zeros_ptr = b_zeros;
  const int32_t* sorted_token_ids_ptr = sorted_token_ids;
  const int32_t* expert_ids_ptr = expert_ids;
  const int32_t* num_tokens_past_padded_ptr = num_tokens_past_padded;
  const float* topk_weights_ptr = topk_weights;
  int* locks = reinterpret_cast<int*>(workspace);

  // FP8 MFMA uses K=32 instructions on CDNA3+ when both A and B are FP8.
  // INT4 MFMA also uses K=32 (2 × FP16 MFMA K=16) when b_is_fp8 is set
  // (repurposed as "b_mfma_tiled" flag for INT4 weights).
  // Otherwise we use the existing K=16 MFMA path.
  int k_step = 16;
  if (a_is_fp8 && b_is_fp8 && a_fp8_is_fnuz && b_fp8_is_fnuz && !has_zp &&
      !has_act_order) {
    k_step = 32;
  }
  // W4A8 MFMA: INT4 weights in MFMA-tiled format with FP8 activations.
  // INT4 decoded to FP8 in registers, uses FP8 MFMA K=32 instruction.
  if (num_bits == 4 && b_fp8_is_fnuz && a_is_fp8 && !has_zp &&
      !has_act_order) {
    k_step = 32;
  }
  // INT4 MFMA: b_fp8_is_fnuz is repurposed to mean "MFMA-tiled format"
  // for INT4 weights. When set, use K=32 (2 × FP16 MFMA K=16) and
  // thread_k_blocks=1 to enable the register-only fast path.
  if (num_bits == 4 && b_fp8_is_fnuz && !has_zp && !has_act_order &&
      !a_is_fp8) {
    k_step = 32;
  }

  // Thread config
  int thread_m_blocks = (moe_block_size + 15) / 16;
  int thread_n_blocks = 4;  // default: 64 output columns per workgroup
  int thread_k_blocks = 4;  // default: 4 waves, all as k-slices

  // For FP8 MFMA (W8A8), use thread_k_blocks=1 (all 4 waves as independent
  // N-tile processors). This eliminates K-reduction overhead and enables the
  // register-only fast path for all MoE block sizes. With thread_k_blocks=1
  // and thread_n_blocks=4, each wave handles one 16-col N-tile independently.
  if (k_step == 32) {
    thread_k_blocks = 1;
  }

  // Constrain thread_k_blocks by group_blocks if grouped
  if (group_blocks > 0 && group_blocks < thread_k_blocks) {
    thread_k_blocks = group_blocks;
  }

  // Ensure prob_k is divisible
  int thread_k = thread_k_blocks * k_step;
  if (prob_k % thread_k != 0) {
    // Try a few sane candidates (kept small to limit kernel variants).
    // Prefer higher K parallelism if the requested config doesn't divide prob_k.
    thread_k_blocks = 2;
    thread_k = thread_k_blocks * k_step;
    if (prob_k % thread_k != 0) {
      thread_k_blocks = 4;
      thread_k = thread_k_blocks * k_step;
      if (prob_k % thread_k != 0) {
        thread_k_blocks = 1;
        thread_k = thread_k_blocks * k_step;
      }
    }
  }

  // Use host-known upper bound for grid size. The kernel reads the actual
  // num_tokens_past_padded from device memory and early-exits excess blocks.
  // This avoids hipMemcpyAsync + hipStreamSynchronize which break CUDA graphs.
  //
  // Tight bound: at most min(prob_m * top_k, num_experts) experts have tokens,
  // and each expert has at most ceil(prob_m / moe_block_size) blocks.
  // For batch=1 decode with top_k=8, num_experts=128: 8 blocks vs 128 from the
  // naive sorted_token_ids.size(0) / moe_block_size bound.
  int naive_max = (max_num_tokens_padded + moe_block_size - 1) / moe_block_size;
  int experts_with_tokens = std::min(prob_m * top_k, num_experts);
  int blocks_per_expert = (prob_m + moe_block_size - 1) / moe_block_size;
  int tight_max = experts_with_tokens * blocks_per_expert;
  int max_moe_blocks = std::min(naive_max, tight_max);

  // Select thread_n_blocks: use 2 (32 cols) instead of 4 (64 cols) when it
  // improves CU saturation. More blocks = better memory-level parallelism.
  int n_tiles_4 = prob_n / (16 * 4);
  if (max_moe_blocks * n_tiles_4 < sms && prob_n % 32 == 0) {
    // For the FP8 MFMA fast path (k_step=32, thread_k_blocks=1), there
    // are no __syncthreads barriers, so the tiles%wave_groups constraint
    // doesn't apply. Idle waves simply return.
    bool fast_path_handles = (k_step == 32 && thread_k_blocks == 1);
    if (fast_path_handles) {
      thread_n_blocks = 2;
    } else {
      constexpr int waves_per_block = 4;  // 256 threads / 64
      int wave_groups = waves_per_block / thread_k_blocks;
      int tiles = thread_m_blocks * 2;
      if (tiles % wave_groups == 0) {
        thread_n_blocks = 2;
      }
    }
  }

  // Debug/tuning overrides (used for local kernel exploration).
  // Example: VLLM_MARLIN_MOE_THREAD_N_BLOCKS=2 VLLM_MARLIN_MOE_THREAD_K_BLOCKS=2
  if (const char* v = std::getenv("VLLM_MARLIN_MOE_THREAD_N_BLOCKS")) {
    int forced = std::atoi(v);
    if (forced == 2 || forced == 4) {
      thread_n_blocks = forced;
    }
  }
  if (const char* v = std::getenv("VLLM_MARLIN_MOE_THREAD_K_BLOCKS")) {
    int forced = std::atoi(v);
    if (forced == 1 || forced == 2 || forced == 4) {
      thread_k_blocks = forced;
    }
  }
  // Re-apply constraints after overrides.
  if (group_blocks > 0 && group_blocks < thread_k_blocks) {
    thread_k_blocks = group_blocks;
  }
  {
    // Validate tile-to-wave-group mapping to avoid divergent __syncthreads()
    // deadlocks in the kernel. Skip for FP8 MFMA fast path which has no
    // barriers (idle waves simply return early).
    bool fast_path = (k_step == 32 && thread_k_blocks == 1);
    if (!fast_path) {
      constexpr int waves_per_block = 4;  // 256 threads / 64
      int wave_groups = waves_per_block / thread_k_blocks;
      int tiles = thread_m_blocks * thread_n_blocks;
      TORCH_CHECK(tiles % wave_groups == 0,
                  "Invalid MoE config: tiles=", tiles,
                  " not divisible by wave_groups=", wave_groups,
                  " (thread_m_blocks=", thread_m_blocks,
                  ", thread_n_blocks=", thread_n_blocks,
                  ", thread_k_blocks=", thread_k_blocks, ")");
    }
  }
  TORCH_CHECK(prob_k % (thread_k_blocks * k_step) == 0,
              "prob_k=", prob_k, " not divisible by ", thread_k_blocks * k_step,
              " (thread_k_blocks=", thread_k_blocks, ")");

  // Ensure prob_n is divisible by thread_n_blocks * 16
  TORCH_CHECK(prob_n % (thread_n_blocks * 16) == 0,
              "prob_n=", prob_n, " not divisible by ", thread_n_blocks * 16);

  int n_tiles = prob_n / (16 * thread_n_blocks);
  int blocks = max_moe_blocks * n_tiles;

  if (blocks == 0) return;

  // Dispatch kernel
  if (false) {
  }
  CALL_MOE_CONFIGS(4)
  CALL_MOE_CONFIGS(8)
  else {
    TORCH_CHECK(false, "Unsupported MoE config: num_bits=", num_bits,
                ", moe_block_size=", moe_block_size,
                ", thread_m_blocks=", thread_m_blocks,
                ", thread_n_blocks=", thread_n_blocks,
                ", thread_k_blocks=", thread_k_blocks,
                ", group_blocks=", group_blocks);
  }
}

#undef CALL_MOE_IF
#undef CALL_MOE_CONFIGS_N
#undef CALL_MOE_CONFIGS

}  // namespace marlin_moe_hip

torch::Tensor moe_wna16_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none,
    torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales_or_none,
    std::optional<torch::Tensor> const& global_scale_or_none,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace,
    torch::Tensor& sorted_token_ids,
    torch::Tensor& expert_ids,
    torch::Tensor& num_tokens_past_padded,
    torch::Tensor& topk_weights,
    int64_t moe_block_size, int64_t top_k, bool mul_topk_weights,
    vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool is_k_full,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float,
    bool b_fp8_is_fnuz,
    int64_t thread_k, int64_t thread_n, int64_t blocks_per_sm) {

  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string arch = dprops->gcnArchName;
  TORCH_CHECK(marlin_moe_hip::is_cdna_arch(arch),
              "moe_wna16_marlin_gemm (ROCm) requires CDNA-class GPU, got ", arch);

  // Determine types
  bool is_fp8 = b_type_id == vllm::kFE4M3fn.id();
  bool is_fp4 = b_type_id == vllm::kFE2M1f.id();
  int num_bits = 0;
  if (b_type_id == vllm::kU4B8.id() || b_type_id == vllm::kS4.id() ||
      b_type_id == vllm::kU4.id()) {
    num_bits = 4;
  } else if (b_type_id == vllm::kU8B128.id() || b_type_id == vllm::kU8.id()) {
    num_bits = 8;
  } else if (is_fp8) {
    num_bits = 8;
  } else if (is_fp4) {
    num_bits = 4;
  } else {
    TORCH_CHECK(false, "moe_wna16_marlin_gemm (ROCm) supports int4/int8/FP8 weights.");
  }

  int num_experts = b_q_weight.size(0);

  // Validate inputs
  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0)=", a.size(0),
              ", size_m=", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1)=", a.size(1),
              ", size_k=", size_k);
  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  TORCH_CHECK(moe_block_size == 8 || moe_block_size == 16 ||
              moe_block_size == 32 || moe_block_size == 48 ||
              moe_block_size == 64,
              "moe_block_size must be 8, 16, 32, 48, or 64, got ", moe_block_size);

  // Determine output dtype
  auto c_dtype = a.dtype();
  if (a.scalar_type() != at::ScalarType::Half &&
      a.scalar_type() != at::ScalarType::BFloat16) {
    c_dtype = b_scales.dtype();
    if (c_or_none.has_value()) {
      c_dtype = c_or_none.value().dtype();
    }
    // For FP8/INT8 activations with FP4/FP8 scales, default to BF16 output
    if (c_dtype != torch::kHalf && c_dtype != torch::kBFloat16) {
      c_dtype = torch::kBFloat16;
    }
  }

  // Allocate or use provided output
  torch::Tensor c;
  auto options = torch::TensorOptions().dtype(c_dtype).device(a.device());
  if (c_or_none.has_value()) {
    c = c_or_none.value();
    TORCH_CHECK(c.device().is_cuda(), "c is not on GPU");
    TORCH_CHECK(c.is_contiguous(), "c is not contiguous");
  } else {
    c = torch::zeros({size_m * top_k, size_n}, options);
  }

  // Get optional tensors
  auto options_fp32 = torch::TensorOptions().dtype(at::kFloat).device(a.device());
  auto options_int = torch::TensorOptions().dtype(at::kInt).device(a.device());

  torch::Tensor b_zeros = b_zeros_or_none.has_value()
      ? b_zeros_or_none.value() : torch::empty({0}, a.options());
  torch::Tensor a_scales = a_scales_or_none.has_value()
      ? a_scales_or_none.value() : torch::empty({0}, options_fp32);
  torch::Tensor global_scale = global_scale_or_none.has_value()
      ? global_scale_or_none.value() : torch::empty({0}, options);
  torch::Tensor g_idx = g_idx_or_none.has_value()
      ? g_idx_or_none.value() : torch::empty({0}, options_int);
  torch::Tensor b_bias = b_bias_or_none.has_value()
      ? b_bias_or_none.value() : torch::empty({0}, options);

  // Calculate group info
  int num_groups = b_scales.size(1);
  int group_size = size_k / num_groups;
  int group_blocks = (num_groups == 1) ? -1 : group_size / 16;

  bool has_act_order = g_idx_or_none.has_value() && g_idx_or_none.value().numel() > 0;
  bool has_zp = b_zeros_or_none.has_value() && b_zeros_or_none.value().numel() > 0;
  bool has_bias = b_bias_or_none.has_value() && b_bias_or_none.value().numel() > 0;
  bool a_is_fp8 = (a.scalar_type() == at::ScalarType::Float8_e4m3fn ||
                   a.scalar_type() == at::ScalarType::Float8_e4m3fnuz);
  bool a_fp8_is_fnuz = marlin_moe_hip::is_cdna3_arch(arch);

  int dev = a.get_device();
  hipStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int sms = 0;
  hipDeviceGetAttribute(&sms, hipDeviceAttributeMultiprocessorCount, dev);

  // Host-known upper bound for num_tokens_past_padded, avoids device->host
  // copy that would break CUDA graph capture.
  int max_num_tokens_padded = static_cast<int>(sorted_token_ids.size(0));

  // Dispatch based on output dtype
  auto dispatch = [&](auto scalar_tag) {
    using scalar_t = decltype(scalar_tag);
    if (is_fp4) {
      bool scale_e4m3 = (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn);
      if (scale_e4m3) {
        marlin_moe_hip::moe_marlin_hip_gemm_impl<scalar_t, true, true>(
            a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
            has_bias ? b_bias.data_ptr() : nullptr,
            a_scales.numel() > 0 ? a_scales.data_ptr<float>() : nullptr,
            b_scales.data_ptr(),
            global_scale.numel() > 0 ? global_scale.data_ptr() : nullptr,
            g_idx.numel() > 0 ? g_idx.data_ptr<int>() : nullptr,
            has_zp ? b_zeros.data_ptr() : nullptr,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens_past_padded.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            top_k, mul_topk_weights, num_experts,
            size_m, size_n, size_k, num_groups,
            workspace.data_ptr(), dev, stream,
            num_bits, moe_block_size, max_num_tokens_padded,
            group_blocks, sms,
            has_bias, has_act_order, has_zp, is_zp_float,
            a_is_fp8, a_fp8_is_fnuz, is_fp8, b_fp8_is_fnuz);
      } else {
        marlin_moe_hip::moe_marlin_hip_gemm_impl<scalar_t, true, false>(
            a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
            has_bias ? b_bias.data_ptr() : nullptr,
            a_scales.numel() > 0 ? a_scales.data_ptr<float>() : nullptr,
            b_scales.data_ptr(),
            global_scale.numel() > 0 ? global_scale.data_ptr() : nullptr,
            g_idx.numel() > 0 ? g_idx.data_ptr<int>() : nullptr,
            has_zp ? b_zeros.data_ptr() : nullptr,
            sorted_token_ids.data_ptr<int32_t>(),
            expert_ids.data_ptr<int32_t>(),
            num_tokens_past_padded.data_ptr<int32_t>(),
            topk_weights.data_ptr<float>(),
            top_k, mul_topk_weights, num_experts,
            size_m, size_n, size_k, num_groups,
            workspace.data_ptr(), dev, stream,
            num_bits, moe_block_size, max_num_tokens_padded,
            group_blocks, sms,
            has_bias, has_act_order, has_zp, is_zp_float,
            a_is_fp8, a_fp8_is_fnuz, is_fp8, b_fp8_is_fnuz);
      }
    } else {
      marlin_moe_hip::moe_marlin_hip_gemm_impl<scalar_t, false, false>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          has_bias ? b_bias.data_ptr() : nullptr,
          a_scales.numel() > 0 ? a_scales.data_ptr<float>() : nullptr,
          b_scales.data_ptr(),
          global_scale.numel() > 0 ? global_scale.data_ptr() : nullptr,
          g_idx.numel() > 0 ? g_idx.data_ptr<int>() : nullptr,
          has_zp ? b_zeros.data_ptr() : nullptr,
          sorted_token_ids.data_ptr<int32_t>(),
          expert_ids.data_ptr<int32_t>(),
          num_tokens_past_padded.data_ptr<int32_t>(),
          topk_weights.data_ptr<float>(),
          top_k, mul_topk_weights, num_experts,
          size_m, size_n, size_k, num_groups,
          workspace.data_ptr(), dev, stream,
          num_bits, moe_block_size, max_num_tokens_padded,
          group_blocks, sms,
          has_bias, has_act_order, has_zp, is_zp_float,
          a_is_fp8, a_fp8_is_fnuz, is_fp8, b_fp8_is_fnuz);
    }
  };

  if (c.scalar_type() == at::ScalarType::Half) {
    dispatch(__half{});
  } else {
    dispatch(__hip_bfloat16{});
  }

  return c;
}
