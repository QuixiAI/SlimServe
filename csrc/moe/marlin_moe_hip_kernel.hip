// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * HIP MoE Marlin kernel for CDNA/CDNA2/CDNA3.
 *
 * Adapts the dense gptq_marlin_hip_kernel structure for MoE routing.
 * Each workgroup processes one (moe_block, n_tile) pair.
 * A rows are loaded via sorted_token_ids indirection.
 * B/scales/zeros are per-expert.
 * TopK weights are applied during output.
 */

#ifndef VLLM_MARLIN_MOE_HIP_KERNEL_HIP
#define VLLM_MARLIN_MOE_HIP_KERNEL_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <stdint.h>
#include <type_traits>

#include "quantization/gptq_marlin/marlin_constants.h"
#include "quantization/gptq_marlin/marlin_perm.h"

#if defined(__has_builtin)
  #if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16)
    #define VLLM_HAS_MFMA_BF16 1
  #else
    #define VLLM_HAS_MFMA_BF16 0
  #endif
#else
  #define VLLM_HAS_MFMA_BF16 0
#endif

// FP8 MFMA is available on CDNA3+ (e.g., MI300X gfx942).
#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define VLLM_MOE_HAS_MFMA_FP8 1
#else
  #define VLLM_MOE_HAS_MFMA_FP8 0
#endif

#include "quantization/gptq_marlin/marlin_hip_common.h"

namespace marlin_moe_hip {

// NOTE: The FP8 MFMA builtins take packed FP8 bytes in a 64-bit register (long).
// The exact byte ordering follows the hardware MFMA operand mapping. We pack
// FP8 values in increasing K order into the 64-bit register.
__device__ __forceinline__ float4_t mfma_16x16x32_fp8(long a, long b,
                                                      float4_t acc) {
#if VLLM_MOE_HAS_MFMA_FP8
  return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
#else
  (void)a;
  (void)b;
  return acc;
#endif
}

template <int num_bits>
struct MarlinIntTraits {
  static constexpr int kPackFactor = 32 / num_bits;
  static constexpr int kBTileWords = (16 * 64) / kPackFactor;
  static constexpr int kBTileRows = kBTileWords / kBTileCols;
  static constexpr int kBTileWordsLds = kBTileRows * kBTileStride;
  static constexpr int kLaneBWords = 16 / kPackFactor;
  static constexpr uint32_t kMask = (1u << num_bits) - 1u;
  static constexpr int kZero = 1 << (num_bits - 1);
};

template <typename scalar_t, int num_bits>
__device__ __forceinline__ scalar_t int_to_scalar(uint32_t v) {
  float f = static_cast<float>(static_cast<int>(v) -
                               MarlinIntTraits<num_bits>::kZero);
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half(f);
  } else {
    return __float2bfloat16(f);
  }
}

template <bool kFnuz>
__device__ __forceinline__ float fp8_to_float(uint8_t v) {
  return ::internal::cast_from_f8<float, kFnuz>(
      static_cast<__hip_fp8_storage_t>(v), 3, 4);
}

__device__ __constant__ float kFp4Lut[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                            2.0f, 3.0f, 4.0f, 6.0f};

template <typename scalar_t>
__device__ __forceinline__ scalar_t fp4_to_scalar(uint32_t v) {
  float val = kFp4Lut[v & 0x7];
  if (v & 0x8) val = -val;
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half(val);
  } else {
    return __float2bfloat16(val);
  }
}

__device__ __forceinline__ float int8_to_float(int8_t v) {
  return static_cast<float>(v);
}

__device__ __forceinline__ int moe_permute_scale_col(int n, bool grouped) {
  if (grouped) {
    int block = n >> 6;
    int in_block = n & 63;
    int perm_idx = static_cast<int>(marlin::k_scale_perm_inv[in_block]);
    return block * 64 + perm_idx;
  }
  int block = n >> 5;
  int in_block = n & 31;
  int perm_idx = static_cast<int>(marlin::k_scale_perm_single_inv[in_block]);
  return block * 32 + perm_idx;
}

template <int num_bits>
__device__ __forceinline__ uint32_t moe_load_zp(const uint32_t* zp,
                                                int group, int n,
                                                int n_cols, bool grouped) {
  int col_perm = moe_permute_scale_col(n, grouped);
  int pack_factor = 32 / num_bits;
  int pack_idx = col_perm / pack_factor;
  int nibble = col_perm % pack_factor;
  int stride = n_cols / pack_factor;
  int row = grouped ? group : 0;
  uint32_t packed = zp[row * stride + pack_idx];
  return (packed >> (nibble * num_bits)) & ((1u << num_bits) - 1u);
}

template <typename scalar_t>
__device__ __forceinline__ float moe_load_zp_float(const scalar_t* zp,
                                                    int group, int n,
                                                    int n_cols, bool grouped) {
  if (grouped) {
    int block = n >> 6;
    int in_block = n & 63;
    int perm_idx = static_cast<int>(marlin::k_scale_perm_inv[in_block]);
    int idx = group * n_cols + block * 64 + perm_idx;
    return scalar_to_float(zp[idx]);
  }
  int block = n >> 5;
  int in_block = n & 31;
  int perm_idx = static_cast<int>(marlin::k_scale_perm_single_inv[in_block]);
  int idx = block * 32 + perm_idx;
  return scalar_to_float(zp[idx]);
}

template <bool kScaleE4M3>
__device__ inline float moe_load_scale_fp4(const uint8_t* s,
                                            int group, int n,
                                            int n_cols, bool grouped) {
  int idx = 0;
  if (grouped) {
    int block = n >> 6;
    int in_block = n & 63;
    int perm_idx = static_cast<int>(marlin::k_scale_perm_inv[in_block]);
    idx = group * n_cols + block * 64 + perm_idx;
  } else {
    int block = n >> 5;
    int in_block = n & 31;
    int perm_idx = static_cast<int>(marlin::k_scale_perm_single_inv[in_block]);
    idx = block * 32 + perm_idx;
  }
  uint8_t raw = s[idx];
  if constexpr (kScaleE4M3) {
    return fp8_to_float<false>(raw);
  } else {
    union { uint32_t u; float f; } bits;
    bits.u = static_cast<uint32_t>(raw) << 23;
    return bits.f;
  }
}

template <typename scalar_t, bool kIsFp4, bool kScaleE4M3>
__device__ inline float moe_load_scale_any(const void* s,
                                            int group, int n,
                                            int n_cols, bool grouped) {
  if constexpr (kIsFp4) {
    return moe_load_scale_fp4<kScaleE4M3>(
        reinterpret_cast<const uint8_t*>(s), group, n, n_cols, grouped);
  } else {
    const scalar_t* s_t = reinterpret_cast<const scalar_t*>(s);
    return scalar_to_float(load_scale(s_t, group, n, n_cols, grouped));
  }
}

// Direct linear scale access for MFMA-tiled B format.
// Scales are stored in natural [num_groups, N] layout per expert,
// without the Marlin permutation (k_scale_perm_inv lookup).
template <typename scalar_t>
__device__ __forceinline__ float moe_load_scale_linear(
    const void* s, int group, int col, int n_cols, bool grouped) {
  const scalar_t* s_t = reinterpret_cast<const scalar_t*>(s);
  if (grouped) {
    return scalar_to_float(s_t[group * n_cols + col]);
  }
  return scalar_to_float(s_t[col]);
}

// INT4 → FP8 decode for W4A8 MFMA path.
// Converts 8 INT4 nibbles (packed in uint32) to 8 FP8 bytes (packed in long).
// All integers -8..+7 are exactly representable in e4m3fnuz.
#if VLLM_MOE_HAS_MFMA_FP8
__device__ __forceinline__ long moe_decode_int4_to_fp8x8_fnuz(uint32_t packed) {
  float vals[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    vals[i] = (float)((int)((packed >> (i * 4)) & 0xF) - 8);

  uint32_t lo = 0, hi = 0;
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(vals[0], vals[1], lo, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(vals[2], vals[3], lo, true);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(vals[4], vals[5], hi, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(vals[6], vals[7], hi, true);
  return (long)lo | ((long)hi << 32);
}
#endif

/*
 * MoE Marlin kernel.
 *
 * Grid: (num_moe_blocks * n_tiles) workgroups
 * Each workgroup processes:
 *   - One MoE block (moe_block_size tokens from sorted_token_ids)
 *   - One N-tile group (thread_n_blocks * 16 output columns)
 *   - Full K reduction (all k-tiles)
 *
 * Structure mirrors the dense MarlinInt4 kernel but with:
 *   - A rows loaded via sorted_token_ids indirection
 *   - B/scales/zeros offset per-expert
 *   - TopK weights applied during output
 */
template <
  typename scalar_t,
  const int num_bits,
  const bool kIsFp4,
  const bool kScaleE4M3,
  const int threads,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int thread_k_blocks,
  const int stages,
  const int moe_block_size,
  const int group_blocks = -1>
__global__ __launch_bounds__(threads, 2) void MarlinMoE(
  const void* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const scalar_t* __restrict__ b_bias,
  const float* __restrict__ a_scales,
  const void* __restrict__ s,
  const scalar_t* __restrict__ global_scale,
  const int* __restrict__ g_idx,
  const void* __restrict__ b_zeros,
  const int32_t* __restrict__ sorted_token_ids,
  const int32_t* __restrict__ expert_ids,
  const int32_t* __restrict__ num_tokens_past_padded,
  const float* __restrict__ topk_weights,
  int  top_k,
  bool mul_topk_weights,
  int  num_experts,
  int  prob_m,
  int  prob_n,
  int  prob_k,
  int  num_groups,
  int* locks,
  bool has_bias,
  bool has_act_order,
  bool has_zp,
  bool is_zp_float,
  bool a_is_fp8,
  bool a_fp8_is_fnuz,
  bool b_is_fp8,
  bool b_fp8_is_fnuz) {

  using Traits = MarlinIntTraits<num_bits>;

  int total_tokens_padded = num_tokens_past_padded[0];
  int num_moe_blocks = (total_tokens_padded + moe_block_size - 1) / moe_block_size;
  int n_tiles = prob_n / (16 * thread_n_blocks);

  // XCD-aware block scheduling for MI300X (8 XCDs, ~38 CUs each).
  // Swizzle so that blocks serving the same expert land on the same XCD
  // for better L2 cache sharing of weight data.
  int wgid = blockIdx.x;
  int total_blocks = num_moe_blocks * n_tiles;
  if (total_blocks > 8) {
    constexpr int NUM_XCDS = 8;
    int group = wgid % NUM_XCDS;
    int local_id = wgid / NUM_XCDS;
    int blocks_per_xcd = total_blocks / NUM_XCDS;
    int extra = total_blocks % NUM_XCDS;
    int offset = group * blocks_per_xcd + (group < extra ? group : extra);
    wgid = offset + local_id;
    if (wgid >= total_blocks) return;
  }
  int moe_block_id = wgid / n_tiles;
  int n_tile_id = wgid % n_tiles;

  if (moe_block_id >= num_moe_blocks) return;

  int expert_id = expert_ids[moe_block_id];
  if (expert_id < 0 || expert_id >= num_experts) return;

  // Calculate per-expert offsets
  // B shape: [num_experts, K/16, N_words_per_row] where N_words_per_row = prob_n * 16 / pack_factor / 4 (in int4)
  int64_t expert_b_offset = (int64_t)expert_id * prob_k * prob_n / (Traits::kPackFactor * 4);
  const int4* B_expert = B + expert_b_offset;
  const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B_expert);

  // Scales shape: [num_experts, num_groups, N_scale_cols]
  int scale_elem_size = kIsFp4 ? 1 : sizeof(scalar_t);
  int64_t expert_scale_offset = (int64_t)expert_id * num_groups * prob_n;
  const void* s_expert = reinterpret_cast<const uint8_t*>(s) +
                          expert_scale_offset * scale_elem_size;
  const scalar_t* s_h = reinterpret_cast<const scalar_t*>(s_expert);

  // Zero-points (per-expert)
  const uint32_t* zp_u32 = nullptr;
  const scalar_t* zp_f = nullptr;
  if (has_zp) {
    if (is_zp_float) {
      zp_f = reinterpret_cast<const scalar_t*>(b_zeros) +
             (int64_t)expert_id * num_groups * prob_n;
    } else {
      zp_u32 = reinterpret_cast<const uint32_t*>(b_zeros) +
               (int64_t)expert_id * num_groups * (prob_n / Traits::kPackFactor);
    }
  }

  // g_idx (per-expert, for act_order)
  const int* g_idx_expert = nullptr;
  if (has_act_order && g_idx != nullptr) {
    g_idx_expert = g_idx + (int64_t)expert_id * prob_k;
  }

  // Bias (per-expert)
  const scalar_t* bias_expert = nullptr;
  if (has_bias && b_bias != nullptr) {
    bias_expert = b_bias + (int64_t)expert_id * prob_n;
  }

  // Global scale (per-expert for FP4)
  float global_scale_val = 1.0f;
  if constexpr (kIsFp4 && kScaleE4M3) {
    if (global_scale != nullptr) {
      global_scale_val = scalar_to_float(global_scale[expert_id]);
    }
  }

  // MoE block token range
  int token_start = moe_block_id * moe_block_size;
  int prob_m_top_k = prob_m * top_k;

  // Wave/lane setup (same as dense kernel)
  const int wave_size = 64;
  const int wave_id = threadIdx.x / wave_size;
  const int lane = threadIdx.x % wave_size;
  const int waves_per_block = threads / wave_size;
  const int wave_group = wave_id / thread_k_blocks;
  const int k_slice = wave_id % thread_k_blocks;
  const int wave_groups = waves_per_block / thread_k_blocks;
  const int lane_row = lane & 15;
  const int lane_col = lane_row;
  const int lane_k_block = (lane >> 4) * 4;
  const int lane_a_idx = lane * 4;
  const int lane_arow = lane_a_idx >> 4;
  const int lane_acol = lane_a_idx & 15;
  const int prob_n_words = prob_n * Traits::kLaneBWords;

  // Shared memory layout:
  //   [sorted_ids: moe_block_size ints]
  //   [topk_weights: moe_block_size floats]
  //   [A tiles: waves_per_block * kATileElems * sizeof(scalar_t)]
  //   [B tiles: waves_per_block * kBTileWordsLds * sizeof(uint32_t)]
  //   [C tiles: waves_per_block * kTileElems * sizeof(float)]
  extern __shared__ uint8_t shmem_raw[];

  int32_t* sh_sorted_ids = reinterpret_cast<int32_t*>(shmem_raw);
  float* sh_topk_wts = reinterpret_cast<float*>(sh_sorted_ids + moe_block_size);
  scalar_t* sh_a = reinterpret_cast<scalar_t*>(
      reinterpret_cast<uint8_t*>(sh_topk_wts + moe_block_size));
  uint32_t* sh_b = reinterpret_cast<uint32_t*>(
      sh_a + waves_per_block * kATileElems);
  // For FP8 MFMA (K=32), we need two 16-K B tiles resident at once.
  // We reserve space for 2 B tiles for all 8-bit kernels (int8/fp8).
  constexpr int kBTileFactor = (num_bits == 8) ? 2 : 1;
  float* sh_c = reinterpret_cast<float*>(
      sh_b + waves_per_block * Traits::kBTileWordsLds * kBTileFactor);

  // Load sorted_token_ids and topk_weights for this MoE block into shared mem
  if (threadIdx.x < moe_block_size) {
    int idx = sorted_token_ids[token_start + threadIdx.x];
    sh_sorted_ids[threadIdx.x] = idx;
    if (mul_topk_weights && topk_weights != nullptr) {
      sh_topk_wts[threadIdx.x] = (idx < prob_m_top_k) ? topk_weights[idx] : 0.0f;
    } else {
      sh_topk_wts[threadIdx.x] = 1.0f;
    }
  }
  __syncthreads();

  // Count valid tokens in this MoE block
  int block_num_valid = moe_block_size;
  for (int i = 0; i < moe_block_size; ++i) {
    if (sh_sorted_ids[i] >= prob_m_top_k) {
      block_num_valid = i;
      break;
    }
  }

  // Early exit for empty MoE blocks (no valid tokens).
  // Critical for batch=1 decode: with 128 experts and top_k=8, 120/128
  // blocks are empty. Without this, 93% of workgroups run the full K-loop
  // (64 iterations of B loads + MFMA) only to discard the result.
  if (block_num_valid == 0) return;

  const bool grouped = (group_blocks != -1);
  const int tiles = thread_m_blocks * thread_n_blocks;
  const int a_tile_base = wave_id * kATileElems;
  const int b_tile_base = wave_id * Traits::kBTileWordsLds;
  bool use_fp8_mfma = false;
  bool use_int4_mfma = false;
  bool use_w4a8_mfma = false;
#if VLLM_MOE_HAS_MFMA_FP8
  // Enable FP8 MFMA (W8A8) when:
  // - Activations A are FP8 with per-token scales
  // - Weights B are FP8 in the platform-native encoding (FNUZ on CDNA3)
  // This avoids dequant-to-16b and uses the native FP8 MFMA instructions.
  //
  // NOTE: act_order changes how scales are indexed; keep MFMA disabled for now.
  if constexpr (num_bits == 8 && !kIsFp4) {
    use_fp8_mfma = (a_scales != nullptr) && a_is_fp8 && b_is_fp8 &&
                   a_fp8_is_fnuz && b_fp8_is_fnuz && !has_zp && !has_act_order;
  }
  // Enable W4A8 MFMA when:
  // - Weights are INT4 in MFMA-tiled format (b_fp8_is_fnuz = "b_mfma_tiled")
  // - Activations are FP8 with per-token scales
  // - INT4 weights decoded to FP8 in registers, uses FP8 MFMA instruction
  if constexpr (num_bits == 4 && !kIsFp4) {
    use_w4a8_mfma = b_fp8_is_fnuz && (a_scales != nullptr) && a_is_fp8 &&
                    !has_zp && !has_act_order;
  }
  // Enable INT4 MFMA (W4A16) when:
  // - Weights are INT4 (not FP4) in MFMA-tiled format
  // - b_fp8_is_fnuz is repurposed as "b_mfma_tiled" flag for INT4
  // - No zero-points (symmetric quant only)
  // - No act_order, activations are BF16/FP16 (not FP8)
  if constexpr (num_bits == 4 && !kIsFp4) {
    use_int4_mfma = b_fp8_is_fnuz && !has_zp && !has_act_order && !a_is_fp8;
  }
#endif
  const int k_step = (use_fp8_mfma || use_int4_mfma || use_w4a8_mfma) ? 32 : 16;
  int k_tiles = prob_k / (k_step * thread_k_blocks);

  // Fast path: FP8 MFMA (W8A8) with thread_k_blocks=1.
  // Works for all moe_block_sizes (thread_m_blocks 1-4).
  //
  // Key optimizations vs the generic path:
  // 1. Each wave handles one 16-col N tile (tile_n = wave_id).
  // 2. A+B loaded directly from global to registers (no LDS for either).
  // 3. MFMA-tiled B format: weights stored as [N/16, K/32, 4halves, 16cols, 8k]
  //    enabling a single coalesced 8-byte load per lane.
  // 4. Software-pipelined K-loop: next iter's loads overlap with current compute.
  // 5. B-data reuse: load B once, compute MFMA for all M-tiles. With
  //    thread_m_blocks=4, this gives 4 MFMAs (384 cycles) per B load, fully
  //    hiding the ~300-cycle memory latency.
  // 6. No __syncthreads, no s_waitcnt, no byte extraction in the inner loop.
  if (use_fp8_mfma) {
    if constexpr ((thread_n_blocks == 2 || thread_n_blocks == 4) &&
                  thread_k_blocks == 1) {
      const int tile_n = wave_id;  // 0..thread_n_blocks-1
      // When thread_n_blocks < 4, excess waves have no N tile to process.
      if (tile_n >= thread_n_blocks) return;
      const int col_base =
          n_tile_id * thread_n_blocks * 16 + tile_n * 16;

      // Accumulators for each M-tile (B data is reused across M-tiles).
      float4_t c_reg[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

      const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A);

      // MFMA register mapping: lane l provides 8 FP8 values from
      // A[l%16][(l/16)*8..+7] and B[(l/16)*8..+7][l%16].
      const int lane_k_fp8 = (lane >> 4) * 8;

      // Precompute A fragment addresses for each M-tile.
      int64_t a_row_base[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        const int local_m = tm * 16 + lane_row;
        const int tid =
            (local_m < moe_block_size) ? sh_sorted_ids[local_m] : prob_m_top_k;
        const int orig_row = tid / top_k;
        const bool valid = (tid < prob_m_top_k && orig_row < prob_m);
        a_row_base[tm] = valid ? (int64_t)orig_row * prob_k : 0;
      }

      // MFMA-tiled B access (N-first layout).
      const uint8_t* B_expert_u8 = reinterpret_cast<const uint8_t*>(B_expert);
      const int n_tile = n_tile_id * thread_n_blocks + tile_n;
      const int lane_half = lane >> 4;
      const int lane_n_col = lane & 15;
      const int b_lane_off = lane_half * 128 + lane_n_col * 8;

      const uint8_t* b_base = B_expert_u8
          + (int64_t)n_tile * k_tiles * 512 + b_lane_off;

      // Prologue: load A (all M-tiles) and B for K-iteration 0.
      long a_cur[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        a_cur[tm] = *reinterpret_cast<const long*>(
            A_u8 + a_row_base[tm] + lane_k_fp8);
      long b_cur = *reinterpret_cast<const long*>(b_base);

      int a_k_next = k_step;
      const uint8_t* b_ptr = b_base + 512;

      if constexpr (group_blocks != -1) {
        // Block-wise quantization: outer loop over scale groups.
        constexpr int kFp8KStep = 32;
        constexpr int iters_per_group = group_blocks * 16 / kFp8KStep;
        const int num_groups_k = k_tiles / iters_per_group;

        for (int g = 0; g < num_groups_k; ++g) {
          float b_scale = moe_load_scale_linear<scalar_t>(
              s_expert, g, col_base + lane_col, prob_n, true);

          float4_t tmp[thread_m_blocks];
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            tmp[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

          #pragma unroll
          for (int sub = 0; sub < iters_per_group; ++sub) {
            const int iter = g * iters_per_group + sub;
            const bool is_last = (iter == k_tiles - 1);

            // Prefetch next A (all M-tiles) and B.
            long a_nxt[thread_m_blocks];
            long b_nxt = 0L;
            if (!is_last) {
              #pragma unroll
              for (int tm = 0; tm < thread_m_blocks; ++tm)
                a_nxt[tm] = *reinterpret_cast<const long*>(
                    A_u8 + a_row_base[tm] + lane_k_fp8 + a_k_next);
              b_nxt = *reinterpret_cast<const long*>(b_ptr);
              a_k_next += k_step;
              b_ptr += 512;
            }

            // Compute MFMA for all M-tiles, reusing B.
            #pragma unroll
            for (int tm = 0; tm < thread_m_blocks; ++tm)
              tmp[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, tmp[tm]);

            // Advance pipeline.
            #pragma unroll
            for (int tm = 0; tm < thread_m_blocks; ++tm)
              a_cur[tm] = a_nxt[tm];
            b_cur = b_nxt;
          }

          // Apply per-group scale.
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
              c_reg[tm][i] += tmp[tm][i] * b_scale;
          }
        }
      } else {
        // Channel-wise: accumulate directly into c_reg.
        for (int iter = 0; iter < k_tiles - 1; ++iter) {
          long a_nxt[thread_m_blocks];
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            a_nxt[tm] = *reinterpret_cast<const long*>(
                A_u8 + a_row_base[tm] + lane_k_fp8 + a_k_next);
          long b_nxt = *reinterpret_cast<const long*>(b_ptr);
          a_k_next += k_step;
          b_ptr += 512;

          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);

          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            a_cur[tm] = a_nxt[tm];
          b_cur = b_nxt;
        }
        // Final iteration.
        #pragma unroll
        for (int tm = 0; tm < thread_m_blocks; ++tm)
          c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);
      }

      // Direct writeback (thread_k_blocks=1, no K-reduction needed).
      scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        const int row_base = tm * 16;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int c_row = (lane >> 4) * 4 + i;
          int c_col_local = lane & 15;
          int local_row = row_base + c_row;
          int col = col_base + c_col_local;

          if (local_row < block_num_valid && col < prob_n) {
            float sum = c_reg[tm][i];

            if constexpr (group_blocks == -1) {
              sum *= moe_load_scale_linear<scalar_t>(
                  s_expert, 0, col, prob_n, false);
            }

            int sorted_id = sh_sorted_ids[local_row];
            int a_row = sorted_id / top_k;
            sum *= (a_row < prob_m) ? a_scales[a_row] : 0.0f;
            sum *= sh_topk_wts[local_row];

            if (has_bias && bias_expert != nullptr) {
              sum += scalar_to_float(bias_expert[col]);
            }

            C_s[(int64_t)sorted_id * prob_n + col] =
                float_to_scalar<scalar_t>(sum);
          }
        }
      }

      return;
    }
  }

  // Fast path: W4A8 MFMA (INT4 weights → FP8 MFMA) with thread_k_blocks=1.
  // Same structure as FP8 MFMA fast path but:
  // - B tile is 256 bytes (INT4, half of FP8's 512) per K=32 tile
  // - B loaded as uint32_t → moe_decode_int4_to_fp8x8_fnuz() → long
  // - 1 FP8 MFMA per K=32 tile (vs 2 FP16 MFMAs for W4A16)
  // - Writeback: weight_scale × a_scale × topk_weight
#if VLLM_MOE_HAS_MFMA_FP8
  if (use_w4a8_mfma) {
    if constexpr ((thread_n_blocks == 2 || thread_n_blocks == 4) &&
                  thread_k_blocks == 1 && num_bits == 4 && !kIsFp4) {
      const int tile_n = wave_id;
      if (tile_n >= thread_n_blocks) return;
      const int col_base =
          n_tile_id * thread_n_blocks * 16 + tile_n * 16;

      float4_t c_reg[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

      const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A);
      const int lane_k_fp8 = (lane >> 4) * 8;

      // Precompute A row addresses for each M-tile.
      int64_t a_row_base[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        const int local_m = tm * 16 + lane_row;
        const int tid =
            (local_m < moe_block_size) ? sh_sorted_ids[local_m] : prob_m_top_k;
        const int orig_row = tid / top_k;
        const bool valid = (tid < prob_m_top_k && orig_row < prob_m);
        a_row_base[tm] = valid ? (int64_t)orig_row * prob_k : 0;
      }

      // W4A8 B tile addressing: [N/16, K/32, 4halves, 16cols] of uint32
      // 256 bytes per tile.
      const uint8_t* B_expert_u8 = reinterpret_cast<const uint8_t*>(B_expert);
      const int n_tile = n_tile_id * thread_n_blocks + tile_n;
      const int lane_group = lane >> 4;
      const int b_lane_off = lane_group * 64 + lane_col * 4;

      const uint8_t* b_base = B_expert_u8
          + (int64_t)n_tile * k_tiles * 256 + b_lane_off;

      // Prologue: load A (all M-tiles) and B for K-iteration 0.
      long a_cur[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        a_cur[tm] = *reinterpret_cast<const long*>(
            A_u8 + a_row_base[tm] + lane_k_fp8);
      uint32_t b_raw = *reinterpret_cast<const uint32_t*>(b_base);
      long b_cur = moe_decode_int4_to_fp8x8_fnuz(b_raw);

      int a_k_next = k_step;
      const uint8_t* b_ptr = b_base + 256;

      if constexpr (group_blocks != -1) {
        constexpr int kFp8KStep = 32;
        constexpr int iters_per_group = group_blocks * 16 / kFp8KStep;
        const int num_groups_k = k_tiles / iters_per_group;

        for (int g = 0; g < num_groups_k; ++g) {
          float b_scale = moe_load_scale_linear<scalar_t>(
              s_expert, g, col_base + lane_col, prob_n, true);

          float4_t tmp[thread_m_blocks];
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            tmp[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

          #pragma unroll
          for (int sub = 0; sub < iters_per_group; ++sub) {
            const int iter = g * iters_per_group + sub;
            const bool is_last = (iter == k_tiles - 1);

            long a_nxt[thread_m_blocks];
            long b_nxt = 0L;
            if (!is_last) {
              #pragma unroll
              for (int tm = 0; tm < thread_m_blocks; ++tm)
                a_nxt[tm] = *reinterpret_cast<const long*>(
                    A_u8 + a_row_base[tm] + lane_k_fp8 + a_k_next);
              uint32_t b_raw_nxt = *reinterpret_cast<const uint32_t*>(b_ptr);
              b_nxt = moe_decode_int4_to_fp8x8_fnuz(b_raw_nxt);
              a_k_next += k_step;
              b_ptr += 256;
            }

            #pragma unroll
            for (int tm = 0; tm < thread_m_blocks; ++tm)
              tmp[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, tmp[tm]);

            #pragma unroll
            for (int tm = 0; tm < thread_m_blocks; ++tm)
              a_cur[tm] = a_nxt[tm];
            b_cur = b_nxt;
          }

          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
              c_reg[tm][i] += tmp[tm][i] * b_scale;
          }
        }
      } else {
        // Channel-wise: accumulate directly into c_reg.
        for (int iter = 0; iter < k_tiles - 1; ++iter) {
          long a_nxt[thread_m_blocks];
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            a_nxt[tm] = *reinterpret_cast<const long*>(
                A_u8 + a_row_base[tm] + lane_k_fp8 + a_k_next);
          uint32_t b_raw_nxt = *reinterpret_cast<const uint32_t*>(b_ptr);
          long b_nxt = moe_decode_int4_to_fp8x8_fnuz(b_raw_nxt);
          a_k_next += k_step;
          b_ptr += 256;

          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);

          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            a_cur[tm] = a_nxt[tm];
          b_cur = b_nxt;
        }
        // Final iteration.
        #pragma unroll
        for (int tm = 0; tm < thread_m_blocks; ++tm)
          c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);
      }

      // Direct writeback: weight_scale × a_scale × topk_weight.
      scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        const int row_base = tm * 16;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int c_row = (lane >> 4) * 4 + i;
          int c_col_local = lane & 15;
          int local_row = row_base + c_row;
          int col = col_base + c_col_local;

          if (local_row < block_num_valid && col < prob_n) {
            float sum = c_reg[tm][i];

            if constexpr (group_blocks == -1) {
              sum *= moe_load_scale_linear<scalar_t>(
                  s_expert, 0, col, prob_n, false);
            }

            int sorted_id = sh_sorted_ids[local_row];
            int a_row = sorted_id / top_k;
            sum *= (a_row < prob_m) ? a_scales[a_row] : 0.0f;

            sum *= sh_topk_wts[local_row];

            if (has_bias && bias_expert != nullptr) {
              sum += scalar_to_float(bias_expert[col]);
            }

            C_s[(int64_t)sorted_id * prob_n + col] =
                float_to_scalar<scalar_t>(sum);
          }
        }
      }

      return;
    }
  }
#endif  // VLLM_MOE_HAS_MFMA_FP8 (W4A8)

  // Fast path: INT4 MFMA (W4A16) with thread_k_blocks=1.
  // Works for all moe_block_sizes (thread_m_blocks 1-4).
  //
  // Same structure as FP8 MFMA fast path but:
  // - A loaded as BF16/FP16 (not FP8)
  // - B tile is 256 bytes (half FP8's 512 bytes) per K=32 tile
  // - Single uint32 load per lane (8 nibbles → decode_int4_to_pack4)
  // - 2 FP16 MFMAs per K=32 tile
  if (use_int4_mfma) {
    if constexpr ((thread_n_blocks == 2 || thread_n_blocks == 4) &&
                  thread_k_blocks == 1 && num_bits == 4 && !kIsFp4) {
      const int tile_n = wave_id;
      if (tile_n >= thread_n_blocks) return;
      const int col_base =
          n_tile_id * thread_n_blocks * 16 + tile_n * 16;

      float4_t c_reg[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

      const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A);
      const int lane_group = lane >> 4;  // 0..3
      const int lane_k_a = lane_group * 4;

      // Precompute A row addresses for each M-tile.
      int64_t a_row_base[thread_m_blocks];
      bool row_valid[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        const int local_m = tm * 16 + lane_row;
        const int tid =
            (local_m < moe_block_size) ? sh_sorted_ids[local_m] : prob_m_top_k;
        const int orig_row = tid / top_k;
        const bool valid = (tid < prob_m_top_k && orig_row < prob_m);
        a_row_base[tm] = valid ? (int64_t)orig_row * prob_k : 0;
        row_valid[tm] = valid;
      }

      // MFMA-tiled B access (INT4, N-first layout).
      // Layout: [N/16, K/32, 4groups, 16cols] of uint32, 256 bytes per tile.
      const uint8_t* B_expert_u8 = reinterpret_cast<const uint8_t*>(B_expert);
      const int n_tile = n_tile_id * thread_n_blocks + tile_n;
      const int b_lane_off = lane_group * 64 + lane_col * 4;

      const uint8_t* b_base = B_expert_u8
          + (int64_t)n_tile * k_tiles * 256 + b_lane_off;

      // Prologue: load B for K-iteration 0.
      uint32_t b_raw = *reinterpret_cast<const uint32_t*>(b_base);
      const uint8_t* b_ptr = b_base + 256;
      int a_k = 0;

      if constexpr (group_blocks != -1) {
        constexpr int kKStep = 32;
        constexpr int iters_per_group = group_blocks * 16 / kKStep;
        const int num_groups_k = k_tiles / iters_per_group;

        for (int g = 0; g < num_groups_k; ++g) {
          float b_scale = moe_load_scale_linear<scalar_t>(
              s_expert, g, col_base + lane_col, prob_n, true);

          float4_t tmp[thread_m_blocks];
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm)
            tmp[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

          #pragma unroll
          for (int sub = 0; sub < iters_per_group; ++sub) {
            const int iter = g * iters_per_group + sub;
            const bool is_last = (iter == k_tiles - 1);

            // Decode INT4 to two pack4_t.
            pack4_t<scalar_t> pack_b0, pack_b1;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              pack_b0.v[i] = float_to_scalar<scalar_t>(
                  (float)((int)((b_raw >> (i * 4)) & 0xF) - 8));
              pack_b1.v[i] = float_to_scalar<scalar_t>(
                  (float)((int)((b_raw >> ((4 + i) * 4)) & 0xF) - 8));
            }

            // Prefetch next B.
            uint32_t b_nxt = 0;
            if (!is_last) {
              b_nxt = *reinterpret_cast<const uint32_t*>(b_ptr);
              b_ptr += 256;
            }

            // Compute MFMA for all M-tiles, reusing B.
            __builtin_amdgcn_s_setprio(1);
            #pragma unroll
            for (int tm = 0; tm < thread_m_blocks; ++tm) {
              pack4_t<scalar_t> pack_a0;
              if (row_valid[tm]) {
                pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                    &A_s[a_row_base[tm] + a_k + lane_k_a]);
              } else {
                pack_a0.u = {0, 0};
              }
              tmp[tm] = mfma_compute(pack_a0, pack_b0, tmp[tm]);

              pack4_t<scalar_t> pack_a1;
              if (row_valid[tm]) {
                pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                    &A_s[a_row_base[tm] + a_k + 16 + lane_k_a]);
              } else {
                pack_a1.u = {0, 0};
              }
              tmp[tm] = mfma_compute(pack_a1, pack_b1, tmp[tm]);
            }
            __builtin_amdgcn_s_setprio(0);

            b_raw = b_nxt;
            a_k += 32;
          }

          // Apply per-group scale.
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            #pragma unroll
            for (int i = 0; i < 4; ++i)
              c_reg[tm][i] += tmp[tm][i] * b_scale;
          }
        }
      } else {
        // Channel-wise: accumulate directly into c_reg.
        for (int iter = 0; iter < k_tiles - 1; ++iter) {
          pack4_t<scalar_t> pack_b0, pack_b1;
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            pack_b0.v[i] = float_to_scalar<scalar_t>(
                (float)((int)((b_raw >> (i * 4)) & 0xF) - 8));
            pack_b1.v[i] = float_to_scalar<scalar_t>(
                (float)((int)((b_raw >> ((4 + i) * 4)) & 0xF) - 8));
          }

          uint32_t b_nxt = *reinterpret_cast<const uint32_t*>(b_ptr);
          b_ptr += 256;

          __builtin_amdgcn_s_setprio(1);
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            pack4_t<scalar_t> pack_a0;
            if (row_valid[tm]) {
              pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                  &A_s[a_row_base[tm] + a_k + lane_k_a]);
            } else {
              pack_a0.u = {0, 0};
            }
            c_reg[tm] = mfma_compute(pack_a0, pack_b0, c_reg[tm]);

            pack4_t<scalar_t> pack_a1;
            if (row_valid[tm]) {
              pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                  &A_s[a_row_base[tm] + a_k + 16 + lane_k_a]);
            } else {
              pack_a1.u = {0, 0};
            }
            c_reg[tm] = mfma_compute(pack_a1, pack_b1, c_reg[tm]);
          }
          __builtin_amdgcn_s_setprio(0);

          b_raw = b_nxt;
          a_k += 32;
        }
        // Final iteration.
        {
          pack4_t<scalar_t> pack_b0, pack_b1;
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            pack_b0.v[i] = float_to_scalar<scalar_t>(
                (float)((int)((b_raw >> (i * 4)) & 0xF) - 8));
            pack_b1.v[i] = float_to_scalar<scalar_t>(
                (float)((int)((b_raw >> ((4 + i) * 4)) & 0xF) - 8));
          }

          __builtin_amdgcn_s_setprio(1);
          #pragma unroll
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            pack4_t<scalar_t> pack_a0;
            if (row_valid[tm]) {
              pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                  &A_s[a_row_base[tm] + a_k + lane_k_a]);
            } else {
              pack_a0.u = {0, 0};
            }
            c_reg[tm] = mfma_compute(pack_a0, pack_b0, c_reg[tm]);

            pack4_t<scalar_t> pack_a1;
            if (row_valid[tm]) {
              pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                  &A_s[a_row_base[tm] + a_k + 16 + lane_k_a]);
            } else {
              pack_a1.u = {0, 0};
            }
            c_reg[tm] = mfma_compute(pack_a1, pack_b1, c_reg[tm]);
          }
          __builtin_amdgcn_s_setprio(0);
        }
      }

      // Direct writeback (no activation scale for W4A16).
      scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        const int row_base = tm * 16;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int c_row = (lane >> 4) * 4 + i;
          int c_col_local = lane & 15;
          int local_row = row_base + c_row;
          int col = col_base + c_col_local;

          if (local_row < block_num_valid && col < prob_n) {
            float sum = c_reg[tm][i];

            if constexpr (group_blocks == -1) {
              sum *= moe_load_scale_linear<scalar_t>(
                  s_expert, 0, col, prob_n, false);
            }

            sum *= sh_topk_wts[local_row];

            if (has_bias && bias_expert != nullptr) {
              sum += scalar_to_float(bias_expert[col]);
            }

            int sorted_id = sh_sorted_ids[local_row];
            C_s[(int64_t)sorted_id * prob_n + col] =
                float_to_scalar<scalar_t>(sum);
          }
        }
      }

      return;
    }
  }

  // Process tiles - each tile is a 16x16 MFMA operation
  for (int tile = 0; tile < tiles; ++tile) {
    const int tile_m = tile / thread_n_blocks;
    const int tile_n = tile % thread_n_blocks;
    const int row_base = tile_m * 16;  // row within MoE block
    const int col_base = n_tile_id * thread_n_blocks * 16 + tile_n * 16;
    const int n_block = col_base >> 4;
    const int block_group = n_block >> 2;
    const int block_offset = n_block & 3;
    const int b_group_offset = block_group * kBTileWords;

    // Only the owning wave_group processes this tile
    if (tile % wave_groups != wave_group) continue;

    float4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};
    // Cache grouped scales across K-iterations when group_size > thread_k.
    // This is especially important for FP8 MFMA where (group, col) scale stays
    // constant for multiple iters when thread_k_blocks < 4.
    [[maybe_unused]] int cached_group = -1;
    [[maybe_unused]] float cached_b_scale = 1.0f;

    for (int iter = 0; iter < k_tiles; ++iter) {
      int k_base = iter * thread_k_blocks * k_step;
      int k0 = k_base + k_slice * k_step;

      // -- Load A tile (with MoE indirection via sorted_token_ids) --
      scalar_t* a_tile = sh_a + a_tile_base;
      uint32_t* b_tile0 = sh_b + b_tile_base;
      uint32_t* b_tile1 = nullptr;
      if (use_fp8_mfma) {
        // Second 16-K tile lives after the first set of per-wave tiles.
        b_tile1 = sh_b + waves_per_block * Traits::kBTileWordsLds + b_tile_base;
      }

      int local_m = row_base + lane_arow;  // row within MoE block
      int token_id = (local_m < moe_block_size) ? sh_sorted_ids[local_m] : prob_m_top_k;
      int orig_row = token_id / top_k;
      int k_a = k0 + lane_acol;

      if (use_fp8_mfma) {
        // FP8 MFMA path: store a 16x32 FP8 tile in LDS (row-major, stride=32).
        // We intentionally avoid dequantizing to scalar_t here.
        uint8_t* a_tile8 = reinterpret_cast<uint8_t*>(a_tile);
        const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A);
        const int lane_a_idx8 = lane * 8;
        const int lane_arow8 = lane_a_idx8 >> 5;  // /32
        const int lane_acol8 = lane_a_idx8 & 31;  // %32
        const int local_m8 = row_base + lane_arow8;
        const int token_id8 =
            (local_m8 < moe_block_size) ? sh_sorted_ids[local_m8] : prob_m_top_k;
        const int orig_row8 = token_id8 / top_k;
        const int k_a8 = k0 + lane_acol8;
        uint64_t a_pack = 0;
        if (token_id8 < prob_m_top_k && orig_row8 < prob_m) {
          if (k_a8 + 7 < prob_k) {
            a_pack = *reinterpret_cast<const uint64_t*>(
                A_u8 + (int64_t)orig_row8 * prob_k + k_a8);
          } else {
            // K tail (should be rare for LLM dims, but keep correct).
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
              uint8_t v = (k_a8 + i < prob_k)
                              ? A_u8[(int64_t)orig_row8 * prob_k + (k_a8 + i)]
                              : 0;
              a_pack |= (uint64_t)v << (i * 8);
            }
          }
        }
        *reinterpret_cast<uint64_t*>(a_tile8 + lane_arow8 * 32 + lane_acol8) =
            a_pack;
      } else if (a_scales == nullptr) {
        // FP16/BF16 activation path
        uint32x2_t a_vec = {0u, 0u};
        if (token_id < prob_m_top_k && orig_row < prob_m) {
          const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A);
          const uint32x2_t* a_src = reinterpret_cast<const uint32x2_t*>(
              A_s + (int64_t)orig_row * prob_k + k_a);
          a_vec = *a_src;
        }
        *reinterpret_cast<uint32x2_t*>(
            a_tile + lane_arow * kATileStride + lane_acol) = a_vec;
      } else {
        // FP8/INT8 activation path
        scalar_t a_vals[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_vals[i] = float_to_scalar<scalar_t>(0.0f);
        }
        if (token_id < prob_m_top_k && orig_row < prob_m) {
          float scale = a_scales[orig_row];
          if (a_is_fp8) {
            const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A);
            uint32_t packed = *reinterpret_cast<const uint32_t*>(
                A_u8 + (int64_t)orig_row * prob_k + k_a);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              uint8_t byte = static_cast<uint8_t>((packed >> (i * 8)) & 0xFF);
              float val = a_fp8_is_fnuz ? fp8_to_float<true>(byte)
                                        : fp8_to_float<false>(byte);
              a_vals[i] = float_to_scalar<scalar_t>(val * scale);
            }
          } else {
            const int8_t* A_i8 = reinterpret_cast<const int8_t*>(A);
            uint32_t packed = *reinterpret_cast<const uint32_t*>(
                A_i8 + (int64_t)orig_row * prob_k + k_a);
            const int8_t* vals = reinterpret_cast<const int8_t*>(&packed);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              a_vals[i] = float_to_scalar<scalar_t>(int8_to_float(vals[i]) * scale);
            }
          }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          a_tile[lane_arow * kATileStride + (lane_acol + i)] = a_vals[i];
        }
      }

      // -- Load B tile (from expert's weights) --
      if (use_fp8_mfma) {
        // MFMA-tiled format: B is loaded directly from global memory during
        // the compute phase below. No LDS staging needed for B.
      } else {
        const uint32_t* b_src = B_u32 + (k0 >> 4) * prob_n_words + b_group_offset;
        if constexpr (num_bits == 8) {
          // Each 16-col block_offset only touches 1/4 of the 64-col tile.
          // Avoid loading unused words to reduce bandwidth.
          int q_index = ((lane >> 1) * 8) + (lane & 1) + (block_offset << 1);
          b_tile0[b_lds_index(q_index)] = b_src[q_index];
        } else {
          // num_bits == 4
          if (lane < 32) {
            int q_index = lane * 4 + block_offset;
            b_tile0[b_lds_index(q_index)] = b_src[q_index];
          }
        }
      }

      // -- Dequant B and compute MFMA --
      if (use_fp8_mfma) {
        // Read A fragment from LDS.
        const uint8_t* a_tile8 = reinterpret_cast<const uint8_t*>(a_tile);
        const int lane_k_fp8 = (lane >> 4) * 8;  // 0,8,16,24 within 32-K tile
        long a_frag = *reinterpret_cast<const long*>(
            a_tile8 + lane_row * 32 + lane_k_fp8);

        // Load B from MFMA-tiled global memory (1 coalesced 8-byte load).
        const uint8_t* B_exp_u8 = reinterpret_cast<const uint8_t*>(B_expert);
        int n_tile_g = col_base >> 4;
        int k_tile_g = k0 >> 5;  // k0 / 32
        // N-first layout: tile at (n_tile, k_tile) * 512
        long b_frag = *reinterpret_cast<const long*>(
            B_exp_u8 + ((int64_t)n_tile_g * k_tiles + k_tile_g) * 512
            + (lane >> 4) * 128 + (lane & 15) * 8);

        asm volatile("s_waitcnt lgkmcnt(0)");
        __builtin_amdgcn_s_setprio(1);
        if constexpr (group_blocks != -1) {
          int group = k0 / (group_blocks * 16);
          if (group != cached_group) {
            cached_b_scale = moe_load_scale_linear<scalar_t>(
                s_expert, group, col_base + lane_col, prob_n, true);
            cached_group = group;
          }
          float b_scale = cached_b_scale;
          float4_t tmp = {0.0f, 0.0f, 0.0f, 0.0f};
          tmp = mfma_16x16x32_fp8(a_frag, b_frag, tmp);
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            c_reg[i] += tmp[i] * b_scale;
          }
        } else {
          c_reg = mfma_16x16x32_fp8(a_frag, b_frag, c_reg);
        }
        __builtin_amdgcn_s_setprio(0);
        continue;
      }

      scalar_t a_reg[4];
      scalar_t b_reg[4];
      float scale = 1.0f;
      float zp_val = 0.0f;
      int group = 0;

      if (grouped) {
        group = (k0 + lane_k_block) / (group_blocks * 16);
      }
      if (has_act_order && g_idx_expert != nullptr) {
        group = g_idx_expert[k0 + lane_k_block];
      }
      if (grouped) {
        scale = moe_load_scale_any<scalar_t, kIsFp4, kScaleE4M3>(
            s_expert, group, col_base + lane_col, prob_n, true);
        if constexpr (kIsFp4 && kScaleE4M3) {
          scale *= global_scale_val;
        }
      }
      if (has_zp) {
        if (is_zp_float) {
          zp_val = moe_load_zp_float<scalar_t>(
              zp_f, group, col_base + lane_col, prob_n, grouped);
        } else {
          zp_val = static_cast<float>(
              moe_load_zp<num_bits>(zp_u32, group,
                                    col_base + lane_col, prob_n, grouped));
        }
      }

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        a_reg[i] = a_tile[lane_row * kATileStride + (lane_k_block + i)];
        int k_in = lane_k_block + i;
        int idx_before_perm = block_offset * 256 + k_in * 16 + lane_col;
        int idx_after_perm = static_cast<int>(marlin::k_perm[idx_before_perm]);
        int q_index = idx_after_perm / Traits::kPackFactor;
        int nibble = idx_after_perm % Traits::kPackFactor;
        uint32_t packed = b_tile0[b_lds_index(q_index)];
        uint32_t q = (packed >> (nibble * num_bits)) & Traits::kMask;
        scalar_t val;
        if constexpr (kIsFp4) {
          val = fp4_to_scalar<scalar_t>(q);
        } else {
          if (has_zp) {
            float val_f = static_cast<float>(q) - zp_val;
            val = float_to_scalar<scalar_t>(val_f);
          } else if (b_is_fp8) {
            // FP8 weights: on ROCm the platform-native dtype is often fnuz.
            // Decode based on how weights were normalized during loading.
            float val_f = b_fp8_is_fnuz
                              ? fp8_to_float<true>(static_cast<uint8_t>(q))
                              : fp8_to_float<false>(static_cast<uint8_t>(q));
            val = float_to_scalar<scalar_t>(val_f);
          } else {
            val = int_to_scalar<scalar_t, num_bits>(q);
          }
        }
        if (grouped)
          val = float_to_scalar<scalar_t>(scalar_to_float(val) * scale);
        b_reg[i] = val;
      }

      pack4_t<scalar_t> pack_a{};
      pack4_t<scalar_t> pack_b{};
      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        pack_a.v[i] = a_reg[i];
        pack_b.v[i] = b_reg[i];
      }

      asm volatile("s_waitcnt lgkmcnt(0)");
      __builtin_amdgcn_s_setprio(1);
      c_reg = mfma_compute(pack_a, pack_b, c_reg);
      __builtin_amdgcn_s_setprio(0);

    }  // k_tiles loop

    // -- Store partial results to shared memory --
    float* c_tile = sh_c + wave_id * 16 * 16;
    int c_row_base = (lane >> 4) * 4;
    int c_col = lane & 15;
    #pragma unroll
    for (int i = 0; i < 4; ++i)
      c_tile[(c_row_base + i) * 16 + c_col] = c_reg[i];

    __syncthreads();

    // -- Reduce across k-slices and write to global C --
    if (tile % wave_groups == wave_group && k_slice == 0) {
      scalar_t* C_s = reinterpret_cast<scalar_t*>(C);

      #pragma unroll
      for (int i = 0; i < 4; ++i) {
        int c_row = (lane >> 4) * 4 + i;
        int c_col_local = lane & 15;
        int local_row = row_base + c_row;
        int col = col_base + c_col_local;

        if (local_row < block_num_valid && col < prob_n) {
          // Reduce across k-slices
          float sum = 0.0f;
          #pragma unroll
          for (int ks = 0; ks < thread_k_blocks; ++ks) {
            int wave = wave_group * thread_k_blocks + ks;
            sum += sh_c[wave * 16 * 16 + c_row * 16 + c_col_local];
          }

          // Apply channel scale if not grouped
          if (!grouped) {
            if (use_fp8_mfma) {
              // MFMA-tiled path: linear scale layout (no Marlin permutation)
              sum *= moe_load_scale_linear<scalar_t>(s_expert, 0, col, prob_n, false);
            } else {
              scalar_t scale_val = load_scale(s_h, 0, col, prob_n, false);
              sum *= scalar_to_float(scale_val);
            }
          }

          // For FP8 MFMA (W8A8), apply activation scale after the dot product.
          if (use_fp8_mfma) {
            int sorted_id = sh_sorted_ids[local_row];
            int a_row = sorted_id / top_k;
            sum *= (a_row < prob_m) ? a_scales[a_row] : 0.0f;
          }

          // Apply topk weight
          float tw = sh_topk_wts[local_row];
          sum *= tw;

          // Apply bias
          if (has_bias && bias_expert != nullptr) {
            sum += scalar_to_float(bias_expert[col]);
          }

          // Write to output at sorted_token_id position
          int sorted_id = sh_sorted_ids[local_row];
          C_s[(int64_t)sorted_id * prob_n + col] =
              float_to_scalar<scalar_t>(sum);
        }
      }
    }
    __syncthreads();
  }  // tiles loop
}

// Constants
#ifndef VLLM_MARLIN_MOE_HIP_CONSTS
#define VLLM_MARLIN_MOE_HIP_CONSTS
static constexpr int MOE_THREADS = 256;
static constexpr int MOE_WAVES_PER_BLOCK = MOE_THREADS / 64;
#endif

template <int num_bits, int moe_block_size>
constexpr int moe_shared_mem_bytes() {
  // sorted_ids + topk_weights + A tiles + B tiles + C tiles
  constexpr int kBTileFactor = (num_bits == 8) ? 2 : 1;
  return moe_block_size * sizeof(int32_t) +
         moe_block_size * sizeof(float) +
         MOE_WAVES_PER_BLOCK * kATileElems * sizeof(__half) +
         MOE_WAVES_PER_BLOCK * kBTileFactor *
             MarlinIntTraits<num_bits>::kBTileWordsLds *
             sizeof(uint32_t) +
         MOE_WAVES_PER_BLOCK * kTileElems * sizeof(float);
}

}  // namespace marlin_moe_hip

#endif  // VLLM_MARLIN_MOE_HIP_KERNEL_HIP
