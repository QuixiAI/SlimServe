// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * Native FP8 MFMA dense Marlin kernel for CDNA3+ (MI300X gfx942/gfx950).
 *
 * W8A8: FP8 activations + FP8 weights with native mfma_f32_16x16x32_fp8_fp8.
 * Ported from the MoE kernel (marlin_moe_hip_kernel.hip) FP8 MFMA fast path,
 * adapted for dense (non-MoE) GEMM.
 *
 * Key differences from MoE version:
 *   - No sorted_token_ids, topk_weights, or expert routing
 *   - Simple M-parallel grid: each workgroup gets (m_slice, n_tile)
 *   - A loaded directly from contiguous row-major layout
 *   - Single B/scales tensor (no per-expert offset)
 *
 * Architecture: 256 threads (4 waves), thread_k_blocks=1 always.
 * Each wave handles one 16-col N-tile (tile_n = wave_id).
 * B-data reuse: load B once, compute MFMA for all thread_m_blocks M-tiles.
 */

#ifndef VLLM_FP8_MARLIN_MFMA_HIP_KERNEL_HIP
#define VLLM_FP8_MARLIN_MFMA_HIP_KERNEL_HIP

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include <stdint.h>
#include <type_traits>

#include "quantization/gptq_marlin/marlin_constants.h"
#include "quantization/gptq_marlin/marlin_perm.h"

// FP8 MFMA is available on CDNA3+ (e.g., MI300X gfx942).
#if defined(__HIPCC__) && (defined(__gfx942__) || defined(__gfx950__))
  #define VLLM_DENSE_HAS_MFMA_FP8 1
#else
  #define VLLM_DENSE_HAS_MFMA_FP8 0
#endif

#if defined(__has_builtin)
  #if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16)
    #define VLLM_HAS_MFMA_BF16 1
  #else
    #define VLLM_HAS_MFMA_BF16 0
  #endif
#else
  #define VLLM_HAS_MFMA_BF16 0
#endif

#include "quantization/gptq_marlin/marlin_hip_common.h"

namespace dense_fp8_mfma {

__device__ __forceinline__ float4_t mfma_16x16x32_fp8(long a, long b,
                                                      float4_t acc) {
#if VLLM_DENSE_HAS_MFMA_FP8
  return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
#else
  (void)a;
  (void)b;
  return acc;
#endif
}

// Direct linear scale access for MFMA-tiled B format.
// Scales are stored in natural [num_groups, N] layout,
// without the Marlin permutation (k_scale_perm_inv lookup).
template <typename scalar_t>
__device__ __forceinline__ float load_scale_linear(
    const scalar_t* s, int group, int col, int n_cols, bool grouped) {
  if (grouped) {
    return scalar_to_float(s[group * n_cols + col]);
  }
  return scalar_to_float(s[col]);
}

/*
 * Dense FP8 MFMA kernel.
 *
 * Grid: ceildiv(M, 16*thread_m_blocks) * (N / (16*thread_n_blocks)) workgroups.
 * Each workgroup processes:
 *   - thread_m_blocks * 16 rows of output
 *   - thread_n_blocks * 16 columns of output (1 wave per N-tile)
 *   - Full K reduction (all k-tiles), thread_k_blocks=1
 *
 * Template params:
 *   scalar_t: output type (FP16 or BF16)
 *   thread_m_blocks: 1-4 (M-tiles per workgroup)
 *   thread_n_blocks: 2 or 4 (N-tiles, one per wave)
 *   group_blocks: -1 (channel-wise) or 2,4,8,16 (group-wise)
 */
template <
  typename scalar_t,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int group_blocks = -1>
__global__ __launch_bounds__(256, 2) void DenseFp8Mfma(
  const uint8_t* __restrict__ A,        // FP8 activations [M, K]
  const uint8_t* __restrict__ B,        // MFMA-tiled FP8 weights
  scalar_t*      __restrict__ C,        // Output [M, N]
  const scalar_t* __restrict__ s,       // Weight scales [num_groups, N]
  const float*   __restrict__ a_scales, // Per-token activation scales [M]
  int  prob_m,
  int  prob_n,
  int  prob_k) {

  constexpr int wave_size = 64;
  constexpr int k_step = 32;  // FP8 MFMA: 16x16x32

  const int wave_id = threadIdx.x / wave_size;
  const int lane = threadIdx.x % wave_size;
  const int lane_row = lane & 15;
  const int lane_col = lane_row;

  // Each wave handles one 16-col N-tile.
  const int tile_n = wave_id;  // 0..thread_n_blocks-1
  if (tile_n >= thread_n_blocks) return;

  // Grid mapping: blockIdx.x -> (m_slice_id, n_group_id)
  const int n_tiles = prob_n / (16 * thread_n_blocks);
  const int k_tiles = prob_k / k_step;

  // XCD-aware block scheduling for MI300X (8 XCDs).
  int m_blocks = ceildiv(prob_m, 16 * thread_m_blocks);
  int total_blocks = m_blocks * n_tiles;

  int wgid = blockIdx.x;
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

  int m_slice_id = wgid / n_tiles;
  int n_tile_id = wgid % n_tiles;

  int m_start = m_slice_id * 16 * thread_m_blocks;
  if (m_start >= prob_m) return;

  int col_base = n_tile_id * thread_n_blocks * 16 + tile_n * 16;

  // Accumulators for each M-tile (B data is reused across M-tiles).
  float4_t c_reg[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm)
    c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

  // MFMA register mapping: lane l provides 8 FP8 values from
  // A[l%16][(l/16)*8..+7] and B[(l/16)*8..+7][l%16].
  const int lane_k_fp8 = (lane >> 4) * 8;

  // Precompute A row addresses for each M-tile.
  int64_t a_row_base[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    const int row = m_start + tm * 16 + lane_row;
    const bool valid = (row < prob_m);
    a_row_base[tm] = valid ? (int64_t)row * prob_k : 0;
  }
  // Validity mask for boundary checking on writeback.
  bool row_valid[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    row_valid[tm] = (m_start + tm * 16 + lane_row < prob_m);
  }

  // MFMA-tiled B access (N-first layout).
  // Layout: [N/16, K/32, 4_halves, 16_cols, 8_k_per_half]
  // 512 bytes per (N_tile, K_tile) pair.
  const int n_tile = n_tile_id * thread_n_blocks + tile_n;
  const int lane_half = lane >> 4;
  const int lane_n_col = lane & 15;
  const int b_lane_off = lane_half * 128 + lane_n_col * 8;

  const uint8_t* b_base = B + (int64_t)n_tile * k_tiles * 512 + b_lane_off;

  // Prologue: load A (all M-tiles) and B for K-iteration 0.
  long a_cur[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    if (row_valid[tm]) {
      a_cur[tm] = *reinterpret_cast<const long*>(
          A + a_row_base[tm] + lane_k_fp8);
    } else {
      a_cur[tm] = 0L;
    }
  }
  long b_cur = *reinterpret_cast<const long*>(b_base);

  int a_k_next = k_step;
  const uint8_t* b_ptr = b_base + 512;

  if constexpr (group_blocks != -1) {
    // Block-wise quantization: outer loop over scale groups.
    constexpr int kFp8KStep = 32;
    constexpr int iters_per_group = group_blocks * 16 / kFp8KStep;
    const int num_groups_k = k_tiles / iters_per_group;

    for (int g = 0; g < num_groups_k; ++g) {
      float b_scale = load_scale_linear<scalar_t>(
          s, g, col_base + lane_col, prob_n, true);

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
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            if (row_valid[tm]) {
              a_nxt[tm] = *reinterpret_cast<const long*>(
                  A + a_row_base[tm] + lane_k_fp8 + a_k_next);
            } else {
              a_nxt[tm] = 0L;
            }
          }
          b_nxt = *reinterpret_cast<const long*>(b_ptr);
          a_k_next += k_step;
          b_ptr += 512;
        }

        // Compute MFMA for all M-tiles, reusing B.
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int tm = 0; tm < thread_m_blocks; ++tm)
          tmp[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, tmp[tm]);
        __builtin_amdgcn_s_setprio(0);

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
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        if (row_valid[tm]) {
          a_nxt[tm] = *reinterpret_cast<const long*>(
              A + a_row_base[tm] + lane_k_fp8 + a_k_next);
        } else {
          a_nxt[tm] = 0L;
        }
      }
      long b_nxt = *reinterpret_cast<const long*>(b_ptr);
      a_k_next += k_step;
      b_ptr += 512;

      __builtin_amdgcn_s_setprio(1);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);
      __builtin_amdgcn_s_setprio(0);

      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        a_cur[tm] = a_nxt[tm];
      b_cur = b_nxt;
    }
    // Final iteration.
    __builtin_amdgcn_s_setprio(1);
    #pragma unroll
    for (int tm = 0; tm < thread_m_blocks; ++tm)
      c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);
    __builtin_amdgcn_s_setprio(0);
  }

  // Direct writeback (thread_k_blocks=1, no K-reduction needed).
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int c_row = (lane >> 4) * 4 + i;
      int c_col_local = lane & 15;
      int row = m_start + tm * 16 + c_row;
      int col = col_base + c_col_local;

      if (row < prob_m && col < prob_n) {
        float sum = c_reg[tm][i];

        // Apply weight scale.
        if constexpr (group_blocks == -1) {
          sum *= load_scale_linear<scalar_t>(s, 0, col, prob_n, false);
        }

        // Apply per-token activation scale.
        sum *= a_scales[row];

        C[(int64_t)row * prob_n + col] = float_to_scalar<scalar_t>(sum);
      }
    }
  }
}

// Dispatch macros.
constexpr int MFMA_THREADS = 256;

#define DENSE_FP8_CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && \
    thread_n_blocks == THREAD_N_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    hipLaunchKernelGGL( \
      (DenseFp8Mfma<scalar_t, THREAD_M_BLOCKS, THREAD_N_BLOCKS, GROUP_BLOCKS>), \
      dim3(grid_blocks), dim3(MFMA_THREADS), 0, stream, \
      A_u8, B_u8, C_s, s_ptr, a_scales_ptr, \
      prob_m, prob_n, prob_k); \
  }

template <typename scalar_t>
int dense_fp8_mfma_gemm(
  const void* A,          // FP8 activations [M, K]
  const void* B,          // MFMA-tiled FP8 weights
        void* C,          // Output [M, N]
  const void* s,          // Weight scales [num_groups, N]
  const void* a_scales,   // Per-token FP32 activation scales [M]
  int prob_m,
  int prob_n,
  int prob_k,
  int sms,
  hipStream_t stream,
  int group_blocks
) {
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  // thread_k_blocks is always 1 for FP8 MFMA.
  int tot_m_blocks = ceildiv(prob_m, 16);
  int thread_m_blocks = tot_m_blocks;
  if (thread_m_blocks > 4) thread_m_blocks = 4;
  int n_tiles_4 = prob_n / (16 * 4);
  int m_slices = ceildiv(prob_m, 16 * thread_m_blocks);
  int grid_4 = m_slices * n_tiles_4;
  int thread_n_blocks = (grid_4 < sms && prob_n % 32 == 0) ? 2 : 4;

  // Validate N divisibility.
  if (prob_n % (16 * thread_n_blocks) != 0) {
    // Fall back to smaller N tile.
    if (thread_n_blocks == 4 && prob_n % 32 == 0) {
      thread_n_blocks = 2;
    } else {
      return ERR_PROB_SHAPE;
    }
  }
  if (prob_k % 32 != 0)
    return ERR_PROB_SHAPE;

  int n_tiles = prob_n / (16 * thread_n_blocks);
  int grid_blocks = m_slices * n_tiles;

  const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A);
  const uint8_t* B_u8 = reinterpret_cast<const uint8_t*>(B);
  scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
  const scalar_t* s_ptr = reinterpret_cast<const scalar_t*>(s);
  const float* a_scales_ptr = reinterpret_cast<const float*>(a_scales);

  if (false) {
  }
  // thread_m_blocks x thread_n_blocks x group_blocks configurations
  // Channel-wise (group_blocks=-1)
  DENSE_FP8_CALL_IF(1, 2, -1)
  DENSE_FP8_CALL_IF(2, 2, -1)
  DENSE_FP8_CALL_IF(3, 2, -1)
  DENSE_FP8_CALL_IF(4, 2, -1)
  DENSE_FP8_CALL_IF(1, 4, -1)
  DENSE_FP8_CALL_IF(2, 4, -1)
  DENSE_FP8_CALL_IF(3, 4, -1)
  DENSE_FP8_CALL_IF(4, 4, -1)
  // Group-wise (group_blocks=2,4,8,16)
  DENSE_FP8_CALL_IF(1, 2, 2)
  DENSE_FP8_CALL_IF(2, 2, 2)
  DENSE_FP8_CALL_IF(3, 2, 2)
  DENSE_FP8_CALL_IF(4, 2, 2)
  DENSE_FP8_CALL_IF(1, 4, 2)
  DENSE_FP8_CALL_IF(2, 4, 2)
  DENSE_FP8_CALL_IF(3, 4, 2)
  DENSE_FP8_CALL_IF(4, 4, 2)
  DENSE_FP8_CALL_IF(1, 2, 4)
  DENSE_FP8_CALL_IF(2, 2, 4)
  DENSE_FP8_CALL_IF(3, 2, 4)
  DENSE_FP8_CALL_IF(4, 2, 4)
  DENSE_FP8_CALL_IF(1, 4, 4)
  DENSE_FP8_CALL_IF(2, 4, 4)
  DENSE_FP8_CALL_IF(3, 4, 4)
  DENSE_FP8_CALL_IF(4, 4, 4)
  DENSE_FP8_CALL_IF(1, 2, 8)
  DENSE_FP8_CALL_IF(2, 2, 8)
  DENSE_FP8_CALL_IF(3, 2, 8)
  DENSE_FP8_CALL_IF(4, 2, 8)
  DENSE_FP8_CALL_IF(1, 4, 8)
  DENSE_FP8_CALL_IF(2, 4, 8)
  DENSE_FP8_CALL_IF(3, 4, 8)
  DENSE_FP8_CALL_IF(4, 4, 8)
  DENSE_FP8_CALL_IF(1, 2, 16)
  DENSE_FP8_CALL_IF(2, 2, 16)
  DENSE_FP8_CALL_IF(3, 2, 16)
  DENSE_FP8_CALL_IF(4, 2, 16)
  DENSE_FP8_CALL_IF(1, 4, 16)
  DENSE_FP8_CALL_IF(2, 4, 16)
  DENSE_FP8_CALL_IF(3, 4, 16)
  DENSE_FP8_CALL_IF(4, 4, 16)
  else {
    return ERR_KERN_SHAPE;
  }

  return 0;
}

#undef DENSE_FP8_CALL_IF

// ============================================================
// W8A16 Dense MFMA kernel (BF16/FP16 activations, FP8 weights)
// ============================================================
//
// Same MFMA-tiled B format as DenseFp8Mfma, but activations are
// BF16/FP16 (not quantized to FP8).  Each K=32 B-tile is processed
// by 2 FP16/BF16 MFMAs of 16x16x16 each, with FP8 B-data decoded
// to scalar_t (BF16/FP16) in registers before each MFMA.
//
// Expected performance: within ~5% of W8A8 for decode (M=1) since
// both paths are B-bandwidth-bound (512 bytes per K=32 tile).

// FP8 → pack4_t<scalar_t> decoding for weight dequantization.
//
// On gfx942/gfx950: use hardware v_cvt_f32_fp8 instruction (1 cycle per byte).
// The native FP8 format on gfx942 is e4m3fnuz, matching MI300X weights.
//
// On other platforms: software fallback via cast_from_f8 (~25 instr per byte).
#if VLLM_DENSE_HAS_MFMA_FP8

template <typename scalar_t>
__device__ __forceinline__ pack4_t<scalar_t> decode_fp8_to_pack4(
    uint32_t raw, bool /* fnuz — gfx942 native format is always fnuz */) {
  pack4_t<scalar_t> p;
  p.v[0] = float_to_scalar<scalar_t>(__builtin_amdgcn_cvt_f32_fp8(raw, 0));
  p.v[1] = float_to_scalar<scalar_t>(__builtin_amdgcn_cvt_f32_fp8(raw, 1));
  p.v[2] = float_to_scalar<scalar_t>(__builtin_amdgcn_cvt_f32_fp8(raw, 2));
  p.v[3] = float_to_scalar<scalar_t>(__builtin_amdgcn_cvt_f32_fp8(raw, 3));
  return p;
}

#else  // Software fallback for non-CDNA3 platforms.

template <bool kFnuz>
__device__ __forceinline__ float fp8_to_float_w8a16(uint8_t v) {
  return ::internal::cast_from_f8<float, kFnuz>(
      static_cast<__hip_fp8_storage_t>(v), 3, 4);
}

template <typename scalar_t>
__device__ __forceinline__ pack4_t<scalar_t> decode_fp8_to_pack4(
    uint32_t raw, bool fnuz) {
  pack4_t<scalar_t> p;
  if (fnuz) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint8_t byte = (raw >> (i * 8)) & 0xFF;
      p.v[i] = float_to_scalar<scalar_t>(fp8_to_float_w8a16<true>(byte));
    }
  } else {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint8_t byte = (raw >> (i * 8)) & 0xFF;
      p.v[i] = float_to_scalar<scalar_t>(fp8_to_float_w8a16<false>(byte));
    }
  }
  return p;
}

#endif  // VLLM_DENSE_HAS_MFMA_FP8

/*
 * Dense W8A16 MFMA kernel.
 *
 * Grid: ceildiv(M, 16*thread_m_blocks) * (N / (16*thread_n_blocks)) workgroups.
 * Each workgroup processes:
 *   - thread_m_blocks * 16 rows of output
 *   - thread_n_blocks * 16 columns of output (1 wave per N-tile)
 *   - Full K reduction (all k-tiles), thread_k_blocks=1
 *
 * B is in MFMA-tiled FP8 format [N/16, K/32, 4halves, 16cols, 8k].
 * For each K=32 B-tile, 2 FP16/BF16 MFMAs are issued (K=0..15, K=16..31).
 * B-data is loaded once and reused across all thread_m_blocks M-tiles.
 */
template <
  typename scalar_t,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int group_blocks = -1>
__global__ __launch_bounds__(256, 2) void DenseW8A16Mfma(
  const scalar_t* __restrict__ A,     // BF16/FP16 activations [M, K]
  const uint8_t*  __restrict__ B,     // MFMA-tiled FP8 weights
  scalar_t*       __restrict__ C,     // Output [M, N]
  const scalar_t* __restrict__ s,     // Weight scales [num_groups, N]
  bool b_fp8_is_fnuz,
  int prob_m, int prob_n, int prob_k) {

  constexpr int wave_size = 64;
  constexpr int k_step = 32;  // 2 FP16 MFMAs per K=32 B-tile

  const int wave_id = threadIdx.x / wave_size;
  const int lane = threadIdx.x % wave_size;
  const int lane_row = lane & 15;
  const int lane_col = lane_row;
  const int lane_group = lane >> 4;  // 0..3

  const int tile_n = wave_id;
  if (tile_n >= thread_n_blocks) return;

  // Grid mapping (same as FP8 kernel).
  const int n_tiles = prob_n / (16 * thread_n_blocks);
  const int k_tiles = prob_k / k_step;

  int m_blocks = ceildiv(prob_m, 16 * thread_m_blocks);
  int total_blocks = m_blocks * n_tiles;

  // XCD-aware block scheduling for MI300X (8 XCDs).
  int wgid = blockIdx.x;
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

  int m_slice_id = wgid / n_tiles;
  int n_tile_id = wgid % n_tiles;

  int m_start = m_slice_id * 16 * thread_m_blocks;
  if (m_start >= prob_m) return;

  int col_base = n_tile_id * thread_n_blocks * 16 + tile_n * 16;

  // Accumulators for each M-tile (B data is reused across M-tiles).
  float4_t c_reg[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm)
    c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

  // A row addressing (in scalar_t elements).
  int64_t a_row_base[thread_m_blocks];
  bool row_valid[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    const int row = m_start + tm * 16 + lane_row;
    row_valid[tm] = (row < prob_m);
    a_row_base[tm] = row_valid[tm] ? (int64_t)row * prob_k : 0;
  }

  // B tile addressing (MFMA-tiled format, N-first).
  // Layout: [N/16, K/32, 4halves, 16cols, 8k_per_half]
  // Half h covers K[h*8..h*8+7].
  // For FP16 MFMA, lane l needs B[(l/16)*4..(l/16)*4+3, l%16]:
  //   group g = lane/16, MFMA0 offset = (g/2)*128 + col*8 + (g%2)*4
  //   MFMA1 offset = (2+g/2)*128 + col*8 + (g%2)*4
  const int n_tile = n_tile_id * thread_n_blocks + tile_n;
  const int b_off_mfma0 = (lane_group / 2) * 128 + lane_col * 8
                         + (lane_group % 2) * 4;
  const int b_off_mfma1 = (2 + lane_group / 2) * 128 + lane_col * 8
                         + (lane_group % 2) * 4;
  const uint8_t* b_tile_base = B + (int64_t)n_tile * k_tiles * 512;

  // FP16 MFMA A lane K offset: 4 scalar_t at (lane/16)*4.
  const int lane_k_a = lane_group * 4;

  // Prologue: load B for first K-tile.
  uint32_t b_raw0 = *reinterpret_cast<const uint32_t*>(
      b_tile_base + b_off_mfma0);
  uint32_t b_raw1 = *reinterpret_cast<const uint32_t*>(
      b_tile_base + b_off_mfma1);
  const uint8_t* b_ptr = b_tile_base + 512;
  int a_k = 0;

  if constexpr (group_blocks != -1) {
    // Group-wise quantization: accumulate within group, then scale.
    constexpr int iters_per_group = group_blocks * 16 / k_step;
    const int num_groups_k = k_tiles / iters_per_group;

    for (int g = 0; g < num_groups_k; ++g) {
      float b_scale = load_scale_linear<scalar_t>(
          s, g, col_base + lane_col, prob_n, true);

      float4_t tmp[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        tmp[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

      #pragma unroll
      for (int sub = 0; sub < iters_per_group; ++sub) {
        const int iter = g * iters_per_group + sub;
        const bool is_last = (iter == k_tiles - 1);

        // Decode current B to scalar_t.
        pack4_t<scalar_t> pack_b0 =
            decode_fp8_to_pack4<scalar_t>(b_raw0, b_fp8_is_fnuz);
        pack4_t<scalar_t> pack_b1 =
            decode_fp8_to_pack4<scalar_t>(b_raw1, b_fp8_is_fnuz);

        // Prefetch next B.
        uint32_t b_nxt0 = 0, b_nxt1 = 0;
        if (!is_last) {
          b_nxt0 = *reinterpret_cast<const uint32_t*>(b_ptr + b_off_mfma0);
          b_nxt1 = *reinterpret_cast<const uint32_t*>(b_ptr + b_off_mfma1);
          b_ptr += 512;
        }

        // Compute for all M-tiles, reusing B.
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int tm = 0; tm < thread_m_blocks; ++tm) {
          // MFMA0 (K=a_k..a_k+15)
          pack4_t<scalar_t> pack_a0;
          if (row_valid[tm]) {
            pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                &A[a_row_base[tm] + a_k + lane_k_a]);
          } else {
            pack_a0.u = {0, 0};
          }
          tmp[tm] = mfma_compute(pack_a0, pack_b0, tmp[tm]);

          // MFMA1 (K=a_k+16..a_k+31)
          pack4_t<scalar_t> pack_a1;
          if (row_valid[tm]) {
            pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                &A[a_row_base[tm] + a_k + 16 + lane_k_a]);
          } else {
            pack_a1.u = {0, 0};
          }
          tmp[tm] = mfma_compute(pack_a1, pack_b1, tmp[tm]);
        }
        __builtin_amdgcn_s_setprio(0);

        b_raw0 = b_nxt0;
        b_raw1 = b_nxt1;
        a_k += k_step;
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
      pack4_t<scalar_t> pack_b0 =
          decode_fp8_to_pack4<scalar_t>(b_raw0, b_fp8_is_fnuz);
      pack4_t<scalar_t> pack_b1 =
          decode_fp8_to_pack4<scalar_t>(b_raw1, b_fp8_is_fnuz);

      uint32_t b_nxt0 = *reinterpret_cast<const uint32_t*>(
          b_ptr + b_off_mfma0);
      uint32_t b_nxt1 = *reinterpret_cast<const uint32_t*>(
          b_ptr + b_off_mfma1);
      b_ptr += 512;

      __builtin_amdgcn_s_setprio(1);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        pack4_t<scalar_t> pack_a0;
        if (row_valid[tm]) {
          pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + lane_k_a]);
        } else {
          pack_a0.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a0, pack_b0, c_reg[tm]);

        pack4_t<scalar_t> pack_a1;
        if (row_valid[tm]) {
          pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + 16 + lane_k_a]);
        } else {
          pack_a1.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a1, pack_b1, c_reg[tm]);
      }
      __builtin_amdgcn_s_setprio(0);

      b_raw0 = b_nxt0;
      b_raw1 = b_nxt1;
      a_k += k_step;
    }

    // Final K-tile.
    {
      pack4_t<scalar_t> pack_b0 =
          decode_fp8_to_pack4<scalar_t>(b_raw0, b_fp8_is_fnuz);
      pack4_t<scalar_t> pack_b1 =
          decode_fp8_to_pack4<scalar_t>(b_raw1, b_fp8_is_fnuz);

      __builtin_amdgcn_s_setprio(1);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        pack4_t<scalar_t> pack_a0;
        if (row_valid[tm]) {
          pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + lane_k_a]);
        } else {
          pack_a0.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a0, pack_b0, c_reg[tm]);

        pack4_t<scalar_t> pack_a1;
        if (row_valid[tm]) {
          pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + 16 + lane_k_a]);
        } else {
          pack_a1.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a1, pack_b1, c_reg[tm]);
      }
      __builtin_amdgcn_s_setprio(0);
    }
  }

  // Direct writeback (weight scale only, no activation scale).
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int c_row = (lane >> 4) * 4 + i;
      int c_col_local = lane & 15;
      int row = m_start + tm * 16 + c_row;
      int col = col_base + c_col_local;

      if (row < prob_m && col < prob_n) {
        float sum = c_reg[tm][i];

        if constexpr (group_blocks == -1) {
          sum *= load_scale_linear<scalar_t>(s, 0, col, prob_n, false);
        }

        C[(int64_t)row * prob_n + col] = float_to_scalar<scalar_t>(sum);
      }
    }
  }
}

// Dispatch macros for W8A16 MFMA.
#define DENSE_W8A16_CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && \
    thread_n_blocks == THREAD_N_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    hipLaunchKernelGGL( \
      (DenseW8A16Mfma<scalar_t, THREAD_M_BLOCKS, THREAD_N_BLOCKS, \
                       GROUP_BLOCKS>), \
      dim3(grid_blocks), dim3(MFMA_THREADS), 0, stream, \
      A_s, B_u8, C_s, s_ptr, fp8_is_fnuz, \
      prob_m, prob_n, prob_k); \
  }

template <typename scalar_t>
int dense_w8a16_mfma_gemm(
  const void* A,          // BF16/FP16 activations [M, K]
  const void* B,          // MFMA-tiled FP8 weights
        void* C,          // Output [M, N]
  const void* s,          // Weight scales [num_groups, N]
  bool fp8_is_fnuz,
  int prob_m,
  int prob_n,
  int prob_k,
  int sms,
  hipStream_t stream,
  int group_blocks
) {
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  int tot_m_blocks = ceildiv(prob_m, 16);
  int thread_m_blocks = tot_m_blocks;
  if (thread_m_blocks > 4) thread_m_blocks = 4;
  int n_tiles_4 = prob_n / (16 * 4);
  int m_slices = ceildiv(prob_m, 16 * thread_m_blocks);
  int grid_4 = m_slices * n_tiles_4;
  int thread_n_blocks = (grid_4 < sms && prob_n % 32 == 0) ? 2 : 4;

  if (prob_n % (16 * thread_n_blocks) != 0) {
    if (thread_n_blocks == 4 && prob_n % 32 == 0) {
      thread_n_blocks = 2;
    } else {
      return ERR_PROB_SHAPE;
    }
  }
  if (prob_k % 32 != 0)
    return ERR_PROB_SHAPE;

  int n_tiles = prob_n / (16 * thread_n_blocks);
  int grid_blocks = m_slices * n_tiles;

  const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A);
  const uint8_t* B_u8 = reinterpret_cast<const uint8_t*>(B);
  scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
  const scalar_t* s_ptr = reinterpret_cast<const scalar_t*>(s);

  if (false) {
  }
  // Channel-wise (group_blocks=-1)
  DENSE_W8A16_CALL_IF(1, 2, -1)
  DENSE_W8A16_CALL_IF(2, 2, -1)
  DENSE_W8A16_CALL_IF(3, 2, -1)
  DENSE_W8A16_CALL_IF(4, 2, -1)
  DENSE_W8A16_CALL_IF(1, 4, -1)
  DENSE_W8A16_CALL_IF(2, 4, -1)
  DENSE_W8A16_CALL_IF(3, 4, -1)
  DENSE_W8A16_CALL_IF(4, 4, -1)
  // Group-wise (group_blocks=2,4,8,16)
  DENSE_W8A16_CALL_IF(1, 2, 2)
  DENSE_W8A16_CALL_IF(2, 2, 2)
  DENSE_W8A16_CALL_IF(3, 2, 2)
  DENSE_W8A16_CALL_IF(4, 2, 2)
  DENSE_W8A16_CALL_IF(1, 4, 2)
  DENSE_W8A16_CALL_IF(2, 4, 2)
  DENSE_W8A16_CALL_IF(3, 4, 2)
  DENSE_W8A16_CALL_IF(4, 4, 2)
  DENSE_W8A16_CALL_IF(1, 2, 4)
  DENSE_W8A16_CALL_IF(2, 2, 4)
  DENSE_W8A16_CALL_IF(3, 2, 4)
  DENSE_W8A16_CALL_IF(4, 2, 4)
  DENSE_W8A16_CALL_IF(1, 4, 4)
  DENSE_W8A16_CALL_IF(2, 4, 4)
  DENSE_W8A16_CALL_IF(3, 4, 4)
  DENSE_W8A16_CALL_IF(4, 4, 4)
  DENSE_W8A16_CALL_IF(1, 2, 8)
  DENSE_W8A16_CALL_IF(2, 2, 8)
  DENSE_W8A16_CALL_IF(3, 2, 8)
  DENSE_W8A16_CALL_IF(4, 2, 8)
  DENSE_W8A16_CALL_IF(1, 4, 8)
  DENSE_W8A16_CALL_IF(2, 4, 8)
  DENSE_W8A16_CALL_IF(3, 4, 8)
  DENSE_W8A16_CALL_IF(4, 4, 8)
  DENSE_W8A16_CALL_IF(1, 2, 16)
  DENSE_W8A16_CALL_IF(2, 2, 16)
  DENSE_W8A16_CALL_IF(3, 2, 16)
  DENSE_W8A16_CALL_IF(4, 2, 16)
  DENSE_W8A16_CALL_IF(1, 4, 16)
  DENSE_W8A16_CALL_IF(2, 4, 16)
  DENSE_W8A16_CALL_IF(3, 4, 16)
  DENSE_W8A16_CALL_IF(4, 4, 16)
  else {
    return ERR_KERN_SHAPE;
  }

  return 0;
}

#undef DENSE_W8A16_CALL_IF

// ============================================================
// W4A16 Dense MFMA kernel (BF16/FP16 activations, INT4 weights)
// ============================================================
//
// Same MFMA-tiled structure as DenseW8A16Mfma, but B tiles are half
// the size (256 bytes vs 512 bytes per K=32 tile) because INT4
// packs 8 nibbles into a single uint32.
//
// B tile layout: [N/16, K/32, 4groups, 16cols] of uint32
// Each uint32 holds 8 INT4 nibbles:
//   nibbles[0..3] (lower 16 bits) → MFMA0: K = group*4 + {0,1,2,3}
//   nibbles[4..7] (upper 16 bits) → MFMA1: K = 16 + group*4 + {0,1,2,3}
//
// Lane addressing: offset = lane_group * 64 + lane_col * 4
// Single uint32 load per lane per K=32 tile (vs two loads for FP8 W8A16).

template <typename scalar_t>
__device__ __forceinline__ void decode_int4_to_pack4(
    uint32_t packed,
    pack4_t<scalar_t>& pack_b0,   // MFMA0 (lower 4 nibbles)
    pack4_t<scalar_t>& pack_b1) { // MFMA1 (upper 4 nibbles)
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    pack_b0.v[i] = float_to_scalar<scalar_t>(
        (float)((int)((packed >> (i * 4)) & 0xF) - 8));
    pack_b1.v[i] = float_to_scalar<scalar_t>(
        (float)((int)((packed >> ((4 + i) * 4)) & 0xF) - 8));
  }
}

/*
 * Dense W4A16 MFMA kernel.
 *
 * Grid: ceildiv(M, 16*thread_m_blocks) * (N / (16*thread_n_blocks)) workgroups.
 * Each workgroup processes:
 *   - thread_m_blocks * 16 rows of output
 *   - thread_n_blocks * 16 columns of output (1 wave per N-tile)
 *   - Full K reduction (all k-tiles), thread_k_blocks=1
 *
 * B is in MFMA-tiled INT4 format [N/16, K/32, 4groups, 16cols] of uint32.
 * For each K=32 B-tile, 2 FP16/BF16 MFMAs are issued (K=0..15, K=16..31).
 * B-data is loaded once and reused across all thread_m_blocks M-tiles.
 */
template <
  typename scalar_t,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int group_blocks = -1>
__global__ __launch_bounds__(256, 2) void DenseW4A16Mfma(
  const scalar_t* __restrict__ A,     // BF16/FP16 activations [M, K]
  const uint8_t*  __restrict__ B,     // MFMA-tiled INT4 weights
  scalar_t*       __restrict__ C,     // Output [M, N]
  const scalar_t* __restrict__ s,     // Weight scales [num_groups, N]
  int prob_m, int prob_n, int prob_k) {

  constexpr int wave_size = 64;
  constexpr int k_step = 32;  // 2 FP16 MFMAs per K=32 B-tile

  const int wave_id = threadIdx.x / wave_size;
  const int lane = threadIdx.x % wave_size;
  const int lane_row = lane & 15;
  const int lane_col = lane_row;
  const int lane_group = lane >> 4;  // 0..3

  const int tile_n = wave_id;
  if (tile_n >= thread_n_blocks) return;

  // Grid mapping (same as FP8 kernel).
  const int n_tiles = prob_n / (16 * thread_n_blocks);
  const int k_tiles = prob_k / k_step;

  int m_blocks = ceildiv(prob_m, 16 * thread_m_blocks);
  int total_blocks = m_blocks * n_tiles;

  // XCD-aware block scheduling for MI300X (8 XCDs).
  int wgid = blockIdx.x;
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

  int m_slice_id = wgid / n_tiles;
  int n_tile_id = wgid % n_tiles;

  int m_start = m_slice_id * 16 * thread_m_blocks;
  if (m_start >= prob_m) return;

  int col_base = n_tile_id * thread_n_blocks * 16 + tile_n * 16;

  // Accumulators for each M-tile (B data is reused across M-tiles).
  float4_t c_reg[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm)
    c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

  // A row addressing (in scalar_t elements).
  int64_t a_row_base[thread_m_blocks];
  bool row_valid[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    const int row = m_start + tm * 16 + lane_row;
    row_valid[tm] = (row < prob_m);
    a_row_base[tm] = row_valid[tm] ? (int64_t)row * prob_k : 0;
  }

  // B tile addressing (MFMA-tiled INT4 format, N-first).
  // Layout: [N/16, K/32, 4groups, 16cols] of uint32
  // 256 bytes per (N_tile, K_tile) pair.
  // Lane addressing: offset = lane_group * 64 + lane_col * 4
  const int n_tile = n_tile_id * thread_n_blocks + tile_n;
  const int b_lane_off = lane_group * 64 + lane_col * 4;
  const uint8_t* b_tile_base = B + (int64_t)n_tile * k_tiles * 256;

  // FP16 MFMA A lane K offset: 4 scalar_t at (lane/16)*4.
  const int lane_k_a = lane_group * 4;

  // Prologue: load B for first K-tile.
  uint32_t b_raw = *reinterpret_cast<const uint32_t*>(
      b_tile_base + b_lane_off);
  const uint8_t* b_ptr = b_tile_base + 256;
  int a_k = 0;

  if constexpr (group_blocks != -1) {
    // Group-wise quantization: accumulate within group, then scale.
    constexpr int iters_per_group = group_blocks * 16 / k_step;
    const int num_groups_k = k_tiles / iters_per_group;

    for (int g = 0; g < num_groups_k; ++g) {
      float b_scale = load_scale_linear<scalar_t>(
          s, g, col_base + lane_col, prob_n, true);

      float4_t tmp[thread_m_blocks];
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        tmp[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

      #pragma unroll
      for (int sub = 0; sub < iters_per_group; ++sub) {
        const int iter = g * iters_per_group + sub;
        const bool is_last = (iter == k_tiles - 1);

        // Decode current B to scalar_t.
        pack4_t<scalar_t> pack_b0, pack_b1;
        decode_int4_to_pack4<scalar_t>(b_raw, pack_b0, pack_b1);

        // Prefetch next B.
        uint32_t b_nxt = 0;
        if (!is_last) {
          b_nxt = *reinterpret_cast<const uint32_t*>(b_ptr + b_lane_off);
          b_ptr += 256;
        }

        // Compute for all M-tiles, reusing B.
        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int tm = 0; tm < thread_m_blocks; ++tm) {
          // MFMA0 (K=a_k..a_k+15)
          pack4_t<scalar_t> pack_a0;
          if (row_valid[tm]) {
            pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                &A[a_row_base[tm] + a_k + lane_k_a]);
          } else {
            pack_a0.u = {0, 0};
          }
          tmp[tm] = mfma_compute(pack_a0, pack_b0, tmp[tm]);

          // MFMA1 (K=a_k+16..a_k+31)
          pack4_t<scalar_t> pack_a1;
          if (row_valid[tm]) {
            pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
                &A[a_row_base[tm] + a_k + 16 + lane_k_a]);
          } else {
            pack_a1.u = {0, 0};
          }
          tmp[tm] = mfma_compute(pack_a1, pack_b1, tmp[tm]);
        }
        __builtin_amdgcn_s_setprio(0);

        b_raw = b_nxt;
        a_k += k_step;
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
      decode_int4_to_pack4<scalar_t>(b_raw, pack_b0, pack_b1);

      uint32_t b_nxt = *reinterpret_cast<const uint32_t*>(
          b_ptr + b_lane_off);
      b_ptr += 256;

      __builtin_amdgcn_s_setprio(1);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        pack4_t<scalar_t> pack_a0;
        if (row_valid[tm]) {
          pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + lane_k_a]);
        } else {
          pack_a0.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a0, pack_b0, c_reg[tm]);

        pack4_t<scalar_t> pack_a1;
        if (row_valid[tm]) {
          pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + 16 + lane_k_a]);
        } else {
          pack_a1.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a1, pack_b1, c_reg[tm]);
      }
      __builtin_amdgcn_s_setprio(0);

      b_raw = b_nxt;
      a_k += k_step;
    }

    // Final K-tile.
    {
      pack4_t<scalar_t> pack_b0, pack_b1;
      decode_int4_to_pack4<scalar_t>(b_raw, pack_b0, pack_b1);

      __builtin_amdgcn_s_setprio(1);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        pack4_t<scalar_t> pack_a0;
        if (row_valid[tm]) {
          pack_a0 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + lane_k_a]);
        } else {
          pack_a0.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a0, pack_b0, c_reg[tm]);

        pack4_t<scalar_t> pack_a1;
        if (row_valid[tm]) {
          pack_a1 = *reinterpret_cast<const pack4_t<scalar_t>*>(
              &A[a_row_base[tm] + a_k + 16 + lane_k_a]);
        } else {
          pack_a1.u = {0, 0};
        }
        c_reg[tm] = mfma_compute(pack_a1, pack_b1, c_reg[tm]);
      }
      __builtin_amdgcn_s_setprio(0);
    }
  }

  // Direct writeback (weight scale only, no activation scale).
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int c_row = (lane >> 4) * 4 + i;
      int c_col_local = lane & 15;
      int row = m_start + tm * 16 + c_row;
      int col = col_base + c_col_local;

      if (row < prob_m && col < prob_n) {
        float sum = c_reg[tm][i];

        if constexpr (group_blocks == -1) {
          sum *= load_scale_linear<scalar_t>(s, 0, col, prob_n, false);
        }

        C[(int64_t)row * prob_n + col] = float_to_scalar<scalar_t>(sum);
      }
    }
  }
}

// Dispatch macros for W4A16 MFMA.
#define DENSE_W4A16_CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && \
    thread_n_blocks == THREAD_N_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    hipLaunchKernelGGL( \
      (DenseW4A16Mfma<scalar_t, THREAD_M_BLOCKS, THREAD_N_BLOCKS, \
                       GROUP_BLOCKS>), \
      dim3(grid_blocks), dim3(MFMA_THREADS), 0, stream, \
      A_s, B_u8, C_s, s_ptr, \
      prob_m, prob_n, prob_k); \
  }

template <typename scalar_t>
int dense_w4a16_mfma_gemm(
  const void* A,          // BF16/FP16 activations [M, K]
  const void* B,          // MFMA-tiled INT4 weights
        void* C,          // Output [M, N]
  const void* s,          // Weight scales [num_groups, N]
  int prob_m,
  int prob_n,
  int prob_k,
  int sms,
  hipStream_t stream,
  int group_blocks
) {
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  int tot_m_blocks = ceildiv(prob_m, 16);
  int thread_m_blocks = tot_m_blocks;
  if (thread_m_blocks > 4) thread_m_blocks = 4;
  int n_tiles_4 = prob_n / (16 * 4);
  int m_slices = ceildiv(prob_m, 16 * thread_m_blocks);
  int grid_4 = m_slices * n_tiles_4;
  int thread_n_blocks = (grid_4 < sms && prob_n % 32 == 0) ? 2 : 4;

  if (prob_n % (16 * thread_n_blocks) != 0) {
    if (thread_n_blocks == 4 && prob_n % 32 == 0) {
      thread_n_blocks = 2;
    } else {
      return ERR_PROB_SHAPE;
    }
  }
  if (prob_k % 32 != 0)
    return ERR_PROB_SHAPE;

  int n_tiles = prob_n / (16 * thread_n_blocks);
  int grid_blocks = m_slices * n_tiles;

  const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A);
  const uint8_t* B_u8 = reinterpret_cast<const uint8_t*>(B);
  scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
  const scalar_t* s_ptr = reinterpret_cast<const scalar_t*>(s);

  if (false) {
  }
  // Channel-wise (group_blocks=-1)
  DENSE_W4A16_CALL_IF(1, 2, -1)
  DENSE_W4A16_CALL_IF(2, 2, -1)
  DENSE_W4A16_CALL_IF(3, 2, -1)
  DENSE_W4A16_CALL_IF(4, 2, -1)
  DENSE_W4A16_CALL_IF(1, 4, -1)
  DENSE_W4A16_CALL_IF(2, 4, -1)
  DENSE_W4A16_CALL_IF(3, 4, -1)
  DENSE_W4A16_CALL_IF(4, 4, -1)
  // Group-wise (group_blocks=2,4,8)
  DENSE_W4A16_CALL_IF(1, 2, 2)
  DENSE_W4A16_CALL_IF(2, 2, 2)
  DENSE_W4A16_CALL_IF(3, 2, 2)
  DENSE_W4A16_CALL_IF(4, 2, 2)
  DENSE_W4A16_CALL_IF(1, 4, 2)
  DENSE_W4A16_CALL_IF(2, 4, 2)
  DENSE_W4A16_CALL_IF(3, 4, 2)
  DENSE_W4A16_CALL_IF(4, 4, 2)
  DENSE_W4A16_CALL_IF(1, 2, 4)
  DENSE_W4A16_CALL_IF(2, 2, 4)
  DENSE_W4A16_CALL_IF(3, 2, 4)
  DENSE_W4A16_CALL_IF(4, 2, 4)
  DENSE_W4A16_CALL_IF(1, 4, 4)
  DENSE_W4A16_CALL_IF(2, 4, 4)
  DENSE_W4A16_CALL_IF(3, 4, 4)
  DENSE_W4A16_CALL_IF(4, 4, 4)
  DENSE_W4A16_CALL_IF(1, 2, 8)
  DENSE_W4A16_CALL_IF(2, 2, 8)
  DENSE_W4A16_CALL_IF(3, 2, 8)
  DENSE_W4A16_CALL_IF(4, 2, 8)
  DENSE_W4A16_CALL_IF(1, 4, 8)
  DENSE_W4A16_CALL_IF(2, 4, 8)
  DENSE_W4A16_CALL_IF(3, 4, 8)
  DENSE_W4A16_CALL_IF(4, 4, 8)
  else {
    return ERR_KERN_SHAPE;
  }

  return 0;
}

#undef DENSE_W4A16_CALL_IF

// ============================================================
// W4A8 Dense MFMA kernel (FP8 activations, INT4 weights via FP8 MFMA)
// ============================================================
//
// INT4 weights are decoded to FP8 in registers, then processed with
// mfma_f32_16x16x32_fp8 — 1 MFMA per K=32 tile (vs 2 FP16 MFMAs for W4A16).
// Half the B bandwidth (256 vs 512 bytes/tile) AND half the compute cycles.
//
// B tile layout (W4A8): [N/16, K/32, 4halves, 16cols] of uint32
// Each uint32 holds 8 INT4 nibbles for K[8h+0..8h+7].
// Lane addressing: half*64 + col*4 (identical to FP8 MFMA B mapping).
//
// INT4→FP8 decode: 8 nibbles → 8 floats → 8 FP8 bytes via cvt_pk_fp8_f32.
// All integers -8..+7 are exactly representable in e4m3fnuz.

#if VLLM_DENSE_HAS_MFMA_FP8

__device__ __forceinline__ long decode_int4_to_fp8x8_fnuz(uint32_t packed) {
  // Extract 8 nibbles to floats, then pack to FP8 via hardware intrinsic.
  float vals[8];
  #pragma unroll
  for (int i = 0; i < 8; ++i)
    vals[i] = (float)((int)((packed >> (i * 4)) & 0xF) - 8);

  // __builtin_amdgcn_cvt_pk_fp8_f32 packs 2 floats → 2 FP8 bytes.
  // word=false: write to bytes [0:1], word=true: write to bytes [2:3].
  uint32_t lo = 0, hi = 0;
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(vals[0], vals[1], lo, false);
  lo = __builtin_amdgcn_cvt_pk_fp8_f32(vals[2], vals[3], lo, true);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(vals[4], vals[5], hi, false);
  hi = __builtin_amdgcn_cvt_pk_fp8_f32(vals[6], vals[7], hi, true);
  return (long)lo | ((long)hi << 32);
}

#endif  // VLLM_DENSE_HAS_MFMA_FP8

/*
 * Dense W4A8 MFMA kernel.
 *
 * Same structure as DenseFp8Mfma (W8A8) with these differences:
 *   - B tile is 256 bytes (half of FP8's 512) per K=32 tile
 *   - B loaded as uint32_t (4 bytes), decoded to long (8 FP8 bytes)
 *   - B lane offset: lane_group*64 + col*4 (vs lane_half*128 + col*8)
 *   - Same FP8 MFMA instruction, same writeback (weight_scale × a_scale)
 */
template <
  typename scalar_t,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int group_blocks = -1>
__global__ __launch_bounds__(256, 2) void DenseW4A8Mfma(
  const uint8_t* __restrict__ A,        // FP8 activations [M, K]
  const uint8_t* __restrict__ B,        // MFMA-tiled INT4 weights (W4A8 format)
  scalar_t*      __restrict__ C,        // Output [M, N]
  const scalar_t* __restrict__ s,       // Weight scales [num_groups, N]
  const float*   __restrict__ a_scales, // Per-token activation scales [M]
  int  prob_m,
  int  prob_n,
  int  prob_k) {

  constexpr int wave_size = 64;
  constexpr int k_step = 32;  // FP8 MFMA: 16x16x32

  const int wave_id = threadIdx.x / wave_size;
  const int lane = threadIdx.x % wave_size;
  const int lane_row = lane & 15;
  const int lane_col = lane_row;

  const int tile_n = wave_id;
  if (tile_n >= thread_n_blocks) return;

  const int n_tiles = prob_n / (16 * thread_n_blocks);
  const int k_tiles = prob_k / k_step;

  int m_blocks = ceildiv(prob_m, 16 * thread_m_blocks);
  int total_blocks = m_blocks * n_tiles;

  // XCD-aware block scheduling for MI300X (8 XCDs).
  int wgid = blockIdx.x;
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

  int m_slice_id = wgid / n_tiles;
  int n_tile_id = wgid % n_tiles;

  int m_start = m_slice_id * 16 * thread_m_blocks;
  if (m_start >= prob_m) return;

  int col_base = n_tile_id * thread_n_blocks * 16 + tile_n * 16;

  float4_t c_reg[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm)
    c_reg[tm] = {0.0f, 0.0f, 0.0f, 0.0f};

  // MFMA register mapping for FP8: lane l provides 8 FP8 values from
  // A[l%16][(l/16)*8..+7] and B[(l/16)*8..+7][l%16].
  const int lane_k_fp8 = (lane >> 4) * 8;

  int64_t a_row_base[thread_m_blocks];
  bool row_valid[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    const int row = m_start + tm * 16 + lane_row;
    row_valid[tm] = (row < prob_m);
    a_row_base[tm] = row_valid[tm] ? (int64_t)row * prob_k : 0;
  }

  // W4A8 B tile addressing: [N/16, K/32, 4halves, 16cols] of uint32
  // 256 bytes per (N_tile, K_tile) pair.
  const int n_tile = n_tile_id * thread_n_blocks + tile_n;
  const int lane_group = lane >> 4;  // 0..3 (maps to half)
  const int lane_n_col = lane & 15;
  const int b_lane_off = lane_group * 64 + lane_n_col * 4;

  const uint8_t* b_base = B + (int64_t)n_tile * k_tiles * 256 + b_lane_off;

  // Prologue: load A (all M-tiles) and B for K-iteration 0.
  long a_cur[thread_m_blocks];
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    if (row_valid[tm]) {
      a_cur[tm] = *reinterpret_cast<const long*>(
          A + a_row_base[tm] + lane_k_fp8);
    } else {
      a_cur[tm] = 0L;
    }
  }
#if VLLM_DENSE_HAS_MFMA_FP8
  uint32_t b_raw = *reinterpret_cast<const uint32_t*>(b_base);
  long b_cur = decode_int4_to_fp8x8_fnuz(b_raw);
#else
  long b_cur = 0L;
#endif

  int a_k_next = k_step;
  const uint8_t* b_ptr = b_base + 256;

  if constexpr (group_blocks != -1) {
    constexpr int kFp8KStep = 32;
    constexpr int iters_per_group = group_blocks * 16 / kFp8KStep;
    const int num_groups_k = k_tiles / iters_per_group;

    for (int g = 0; g < num_groups_k; ++g) {
      float b_scale = load_scale_linear<scalar_t>(
          s, g, col_base + lane_col, prob_n, true);

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
          for (int tm = 0; tm < thread_m_blocks; ++tm) {
            if (row_valid[tm]) {
              a_nxt[tm] = *reinterpret_cast<const long*>(
                  A + a_row_base[tm] + lane_k_fp8 + a_k_next);
            } else {
              a_nxt[tm] = 0L;
            }
          }
#if VLLM_DENSE_HAS_MFMA_FP8
          uint32_t b_raw_nxt = *reinterpret_cast<const uint32_t*>(b_ptr);
          b_nxt = decode_int4_to_fp8x8_fnuz(b_raw_nxt);
#endif
          a_k_next += k_step;
          b_ptr += 256;
        }

        __builtin_amdgcn_s_setprio(1);
        #pragma unroll
        for (int tm = 0; tm < thread_m_blocks; ++tm)
          tmp[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, tmp[tm]);
        __builtin_amdgcn_s_setprio(0);

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
      for (int tm = 0; tm < thread_m_blocks; ++tm) {
        if (row_valid[tm]) {
          a_nxt[tm] = *reinterpret_cast<const long*>(
              A + a_row_base[tm] + lane_k_fp8 + a_k_next);
        } else {
          a_nxt[tm] = 0L;
        }
      }
#if VLLM_DENSE_HAS_MFMA_FP8
      uint32_t b_raw_nxt = *reinterpret_cast<const uint32_t*>(b_ptr);
      long b_nxt = decode_int4_to_fp8x8_fnuz(b_raw_nxt);
#else
      long b_nxt = 0L;
#endif
      a_k_next += k_step;
      b_ptr += 256;

      __builtin_amdgcn_s_setprio(1);
      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);
      __builtin_amdgcn_s_setprio(0);

      #pragma unroll
      for (int tm = 0; tm < thread_m_blocks; ++tm)
        a_cur[tm] = a_nxt[tm];
      b_cur = b_nxt;
    }
    // Final iteration.
    __builtin_amdgcn_s_setprio(1);
    #pragma unroll
    for (int tm = 0; tm < thread_m_blocks; ++tm)
      c_reg[tm] = mfma_16x16x32_fp8(a_cur[tm], b_cur, c_reg[tm]);
    __builtin_amdgcn_s_setprio(0);
  }

  // Direct writeback: weight_scale × a_scale.
  #pragma unroll
  for (int tm = 0; tm < thread_m_blocks; ++tm) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      int c_row = (lane >> 4) * 4 + i;
      int c_col_local = lane & 15;
      int row = m_start + tm * 16 + c_row;
      int col = col_base + c_col_local;

      if (row < prob_m && col < prob_n) {
        float sum = c_reg[tm][i];

        if constexpr (group_blocks == -1) {
          sum *= load_scale_linear<scalar_t>(s, 0, col, prob_n, false);
        }

        sum *= a_scales[row];

        C[(int64_t)row * prob_n + col] = float_to_scalar<scalar_t>(sum);
      }
    }
  }
}

// Dispatch macros for W4A8 MFMA.
#define DENSE_W4A8_CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && \
    thread_n_blocks == THREAD_N_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    hipLaunchKernelGGL( \
      (DenseW4A8Mfma<scalar_t, THREAD_M_BLOCKS, THREAD_N_BLOCKS, \
                      GROUP_BLOCKS>), \
      dim3(grid_blocks), dim3(MFMA_THREADS), 0, stream, \
      A_u8, B_u8, C_s, s_ptr, a_scales_ptr, \
      prob_m, prob_n, prob_k); \
  }

template <typename scalar_t>
int dense_w4a8_mfma_gemm(
  const void* A,          // FP8 activations [M, K]
  const void* B,          // MFMA-tiled INT4 weights (W4A8 format)
        void* C,          // Output [M, N]
  const void* s,          // Weight scales [num_groups, N]
  const void* a_scales,   // Per-token FP32 activation scales [M]
  int prob_m,
  int prob_n,
  int prob_k,
  int sms,
  hipStream_t stream,
  int group_blocks
) {
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  int tot_m_blocks = ceildiv(prob_m, 16);
  int thread_m_blocks = tot_m_blocks;
  if (thread_m_blocks > 4) thread_m_blocks = 4;
  int n_tiles_4 = prob_n / (16 * 4);
  int m_slices = ceildiv(prob_m, 16 * thread_m_blocks);
  int grid_4 = m_slices * n_tiles_4;
  int thread_n_blocks = (grid_4 < sms && prob_n % 32 == 0) ? 2 : 4;

  if (prob_n % (16 * thread_n_blocks) != 0) {
    if (thread_n_blocks == 4 && prob_n % 32 == 0) {
      thread_n_blocks = 2;
    } else {
      return ERR_PROB_SHAPE;
    }
  }
  if (prob_k % 32 != 0)
    return ERR_PROB_SHAPE;

  int n_tiles = prob_n / (16 * thread_n_blocks);
  int grid_blocks = m_slices * n_tiles;

  const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A);
  const uint8_t* B_u8 = reinterpret_cast<const uint8_t*>(B);
  scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
  const scalar_t* s_ptr = reinterpret_cast<const scalar_t*>(s);
  const float* a_scales_ptr = reinterpret_cast<const float*>(a_scales);

  if (false) {
  }
  // Channel-wise (group_blocks=-1)
  DENSE_W4A8_CALL_IF(1, 2, -1)
  DENSE_W4A8_CALL_IF(2, 2, -1)
  DENSE_W4A8_CALL_IF(3, 2, -1)
  DENSE_W4A8_CALL_IF(4, 2, -1)
  DENSE_W4A8_CALL_IF(1, 4, -1)
  DENSE_W4A8_CALL_IF(2, 4, -1)
  DENSE_W4A8_CALL_IF(3, 4, -1)
  DENSE_W4A8_CALL_IF(4, 4, -1)
  // Group-wise (group_blocks=2,4,8)
  DENSE_W4A8_CALL_IF(1, 2, 2)
  DENSE_W4A8_CALL_IF(2, 2, 2)
  DENSE_W4A8_CALL_IF(3, 2, 2)
  DENSE_W4A8_CALL_IF(4, 2, 2)
  DENSE_W4A8_CALL_IF(1, 4, 2)
  DENSE_W4A8_CALL_IF(2, 4, 2)
  DENSE_W4A8_CALL_IF(3, 4, 2)
  DENSE_W4A8_CALL_IF(4, 4, 2)
  DENSE_W4A8_CALL_IF(1, 2, 4)
  DENSE_W4A8_CALL_IF(2, 2, 4)
  DENSE_W4A8_CALL_IF(3, 2, 4)
  DENSE_W4A8_CALL_IF(4, 2, 4)
  DENSE_W4A8_CALL_IF(1, 4, 4)
  DENSE_W4A8_CALL_IF(2, 4, 4)
  DENSE_W4A8_CALL_IF(3, 4, 4)
  DENSE_W4A8_CALL_IF(4, 4, 4)
  DENSE_W4A8_CALL_IF(1, 2, 8)
  DENSE_W4A8_CALL_IF(2, 2, 8)
  DENSE_W4A8_CALL_IF(3, 2, 8)
  DENSE_W4A8_CALL_IF(4, 2, 8)
  DENSE_W4A8_CALL_IF(1, 4, 8)
  DENSE_W4A8_CALL_IF(2, 4, 8)
  DENSE_W4A8_CALL_IF(3, 4, 8)
  DENSE_W4A8_CALL_IF(4, 4, 8)
  else {
    return ERR_KERN_SHAPE;
  }

  return 0;
}

#undef DENSE_W4A8_CALL_IF

}  // namespace dense_fp8_mfma

#endif  // VLLM_FP8_MARLIN_MFMA_HIP_KERNEL_HIP
