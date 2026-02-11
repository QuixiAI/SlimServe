// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
/*
 * HIP INT4 Marlin kernel for CDNA/CDNA2/CDNA3.
 *
 * Mirrors CUDA Marlin weight packing and scale permutation, and
 * dequantizes INT4 weights on the fly to FP16/BF16.
 */

#ifndef VLLM_GPTQ_MARLIN_HIP_KERNEL_HIP
#define VLLM_GPTQ_MARLIN_HIP_KERNEL_HIP

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

#include "quantization/gptq_marlin/marlin_hip_common.h"

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

__device__ __forceinline__ float int8_to_float(int8_t v) {
  return static_cast<float>(v);
}

template <typename scalar_t, int num_bits>
__device__ __forceinline__ scalar_t int_to_scalar(uint32_t v) {
  static_assert(std::is_same_v<scalar_t, __half> ||
                    std::is_same_v<scalar_t, __hip_bfloat16>,
                "int_to_scalar only supports fp16/bf16 outputs");
  float f = static_cast<float>(static_cast<int>(v) -
                               MarlinIntTraits<num_bits>::kZero);
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half(f);
  } else {
    return __float2bfloat16(f);
  }
}

__device__ __constant__ float kFp4Lut[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                            2.0f, 3.0f, 4.0f, 6.0f};

template <typename scalar_t>
__device__ __forceinline__ scalar_t fp4_to_scalar(uint32_t v) {
  float val = kFp4Lut[v & 0x7];
  if (v & 0x8) {
    val = -val;
  }
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half(val);
  } else {
    return __float2bfloat16(val);
  }
}

template <bool kFnuz>
__device__ __forceinline__ float fp8_to_float(uint8_t v) {
  return ::internal::cast_from_f8<float, kFnuz>(
      static_cast<__hip_fp8_storage_t>(v), 3, 4);
}

__device__ __forceinline__ float e8m0_to_float(uint8_t v) {
  union {
    uint32_t u;
    float f;
  } bits;
  bits.u = static_cast<uint32_t>(v) << 23;
  return bits.f;
}

template <typename scalar_t>
__device__ __forceinline__ void load_a_fragment(
    scalar_t out[4],
    const void* A_base,
    int m,
    int k_a,
    int prob_k,
    int prob_m,
    const float* a_scales,
    bool a_is_fp8,
    bool a_fp8_is_fnuz) {
  #pragma unroll
  for (int i = 0; i < 4; ++i) {
    out[i] = float_to_scalar<scalar_t>(0.0f);
  }
  if (m >= prob_m) {
    return;
  }
  if (a_scales == nullptr) {
    const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A_base);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      out[i] = A_s[m * prob_k + k_a + i];
    }
    return;
  }
  float scale = a_scales[m];
  if (a_is_fp8) {
    const uint8_t* A_u8 = reinterpret_cast<const uint8_t*>(A_base);
    uint32_t packed = 0;
    const uint8_t* src = A_u8 + m * prob_k + k_a;
    packed = *reinterpret_cast<const uint32_t*>(src);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      uint8_t byte = static_cast<uint8_t>((packed >> (i * 8)) & 0xFF);
      float val = a_fp8_is_fnuz ? fp8_to_float<true>(byte)
                                : fp8_to_float<false>(byte);
      out[i] = float_to_scalar<scalar_t>(val * scale);
    }
  } else {
    const int8_t* A_i8 = reinterpret_cast<const int8_t*>(A_base);
    const int8_t* src = A_i8 + m * prob_k + k_a;
    uint32_t packed = *reinterpret_cast<const uint32_t*>(src);
    const int8_t* vals = reinterpret_cast<const int8_t*>(&packed);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      float val = int8_to_float(vals[i]);
      out[i] = float_to_scalar<scalar_t>(val * scale);
    }
  }
}

__device__ __forceinline__ int permute_scale_col(int n, bool grouped) {
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
__device__ __forceinline__ uint32_t load_zp(const uint32_t* zp,
                                            int group,
                                            int n,
                                            int n_cols,
                                            bool grouped) {
  int col_perm = permute_scale_col(n, grouped);
  int pack_factor = 32 / num_bits;
  int pack_idx = col_perm / pack_factor;
  int nibble = col_perm % pack_factor;
  int stride = n_cols / pack_factor;
  int row = grouped ? group : 0;
  uint32_t packed = zp[row * stride + pack_idx];
  return (packed >> (nibble * num_bits)) & ((1u << num_bits) - 1u);
}

template <typename scalar_t>
__device__ __forceinline__ float load_zp_float(const scalar_t* zp,
                                               int group,
                                               int n,
                                               int n_cols,
                                               bool grouped) {
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
__device__ inline float load_scale_fp4(const uint8_t* s,
                                       int group,
                                       int n,
                                       int n_cols,
                                       bool grouped) {
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
    return e8m0_to_float(raw);
  }
}

template <typename scalar_t, bool kIsFp4, bool kScaleE4M3>
__device__ inline float load_scale_any(const void* s,
                                       int group,
                                       int n,
                                       int n_cols,
                                       bool grouped) {
  if constexpr (kIsFp4) {
    const uint8_t* s_u8 = reinterpret_cast<const uint8_t*>(s);
    return load_scale_fp4<kScaleE4M3>(s_u8, group, n, n_cols, grouped);
  } else {
    const scalar_t* s_t = reinterpret_cast<const scalar_t*>(s);
    return scalar_to_float(load_scale(s_t, group, n, n_cols, grouped));
  }
}


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
  const int group_blocks = -1>
__global__ __launch_bounds__(threads, 2) void MarlinInt4(
  const void* __restrict__ A,
  const int4* __restrict__ B,
        int4* __restrict__ C,
  const scalar_t* __restrict__ b_bias,
  const float* __restrict__ a_scales,
  const void* __restrict__ s,
  const scalar_t* __restrict__ global_scale,
  const int* __restrict__ g_idx,
  const void* __restrict__ b_zeros,
  int  prob_m,
  int  prob_n,
  int  prob_k,
  int* locks,
  bool has_bias,
  bool has_act_order,
  bool has_zp,
  bool is_zp_float,
  bool a_is_fp8,
  bool a_fp8_is_fnuz) {
  using Traits = MarlinIntTraits<num_bits>;
  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
  if (group_blocks != -1)
    iters = (group_blocks / thread_k_blocks) *
            ceildiv(iters, (group_blocks / thread_k_blocks));

  int wgid = blockIdx.x;
  int slice_row = (iters * wgid) % k_tiles;
  int slice_col_par = (iters * wgid) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters;
  int slice_count = 0;
  int slice_idx;
  float global_scale_val = 1.0f;
  const uint8_t* A_bytes = reinterpret_cast<const uint8_t*>(A);
  int a_elem_bytes = (a_scales != nullptr) ? 1 : sizeof(scalar_t);
  int a_row_bytes = prob_k * a_elem_bytes;
  const uint32_t* b_zeros_u32 = reinterpret_cast<const uint32_t*>(b_zeros);
  const scalar_t* b_zeros_f = reinterpret_cast<const scalar_t*>(b_zeros);
  if constexpr (kIsFp4 && kScaleE4M3) {
    if (global_scale != nullptr) {
      global_scale_val = scalar_to_float(global_scale[0]);
    }
  }

  if (slice_col_par >= n_tiles) {
    A_bytes += (slice_col_par / n_tiles) * 16 * thread_m_blocks * a_row_bytes;
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  auto init_slice = [&] () {
    slice_iters = iters * (wgid + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles)
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * wgid - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A_bytes += 16 * thread_m_blocks * a_row_bytes;
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  if (slice_iters == 0)
    return;

  const uint32_t* B_u32 = reinterpret_cast<const uint32_t*>(B);
  const scalar_t* s_h = reinterpret_cast<const scalar_t*>(s);

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
  const int lane_b_idx = lane * Traits::kLaneBWords;
  const int prob_n_words = prob_n * Traits::kLaneBWords;

  extern __shared__ uint8_t shmem_raw[];
  scalar_t* sh_a = reinterpret_cast<scalar_t*>(shmem_raw);
  uint32_t* sh_b = reinterpret_cast<uint32_t*>(
    sh_a + waves_per_block * stages * kATileElems
  );
  float* sh_c = reinterpret_cast<float*>(
    sh_b + waves_per_block * stages * Traits::kBTileWordsLds
  );

  const int tiles = thread_m_blocks * thread_n_blocks;
  const bool grouped = (group_blocks != -1);
  const int a_tile_base = (wave_id * stages) * kATileElems;
  const int b_tile_base = (wave_id * stages) * Traits::kBTileWordsLds;
  constexpr bool kUsePipeline = (stages > 1) && (waves_per_block > thread_k_blocks);

  __shared__ int stage_ready_iter[stages];
  __shared__ int stage_done_iter[stages];
  __shared__ int stage_ready_count[stages];
  __shared__ int stage_done_count[stages];

  while (slice_iters) {
    const void* A_base = reinterpret_cast<const void*>(A_bytes);
    scalar_t* C_s = reinterpret_cast<scalar_t*>(C);
    bool last = slice_idx == slice_count - 1;
    for (int tile = 0; tile < tiles; ++tile) {
      const int compute_group = tile % wave_groups;
      const int tile_m = tile / thread_n_blocks;
      const int tile_n = tile % thread_n_blocks;
      const int row_base = tile_m * 16;
      const int col_base = (slice_col * thread_n_blocks + tile_n) * 16;
      const int n_block = col_base >> 4;
      const int block_group = n_block >> 2;
      const int block_offset = n_block & 3;
      const int b_group_offset = block_group * kBTileWords;

      if constexpr (kUsePipeline) {
        const int producer_group = (compute_group + 1) % wave_groups;
        const bool do_producer = (wave_group == producer_group);
        const bool do_consumer = (wave_group == compute_group);

        if (threadIdx.x < stages) {
          stage_ready_iter[threadIdx.x] = -1;
          stage_done_iter[threadIdx.x] = threadIdx.x - stages;
          stage_ready_count[threadIdx.x] = 0;
          stage_done_count[threadIdx.x] = 0;
        }
        __syncthreads();

        if (do_producer) {
          for (int iter = 0; iter < slice_iters; ++iter) {
            int stage = iter % stages;
            wait_for_iter(&stage_done_iter[stage], iter - stages, lane);

            int k_tile = slice_row + iter;
            int k_base = k_tile * thread_k_blocks * 16;
            int k0 = k_base + k_slice * 16;

            int compute_wave = compute_group * thread_k_blocks + k_slice;
            scalar_t* a_tile = sh_a + (compute_wave * stages + stage) * kATileElems;
            uint32_t* b_tile =
                sh_b + (compute_wave * stages + stage) *
                           Traits::kBTileWordsLds;

            int m = row_base + lane_arow;
            int k_a = k0 + lane_acol;
            if (a_scales == nullptr) {
              uint32x2_t a_vec = {0u, 0u};
              if (m < prob_m) {
                const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A_base);
                const uint32x2_t* a_src =
                    reinterpret_cast<const uint32x2_t*>(
                        A_s + m * prob_k + k_a);
                a_vec = *a_src;
              }
              *reinterpret_cast<uint32x2_t*>(
                  a_tile + lane_arow * kATileStride + lane_acol) = a_vec;
            } else {
              scalar_t a_vals[4];
              load_a_fragment(a_vals, A_base, m, k_a, prob_k, prob_m, a_scales,
                              a_is_fp8, a_fp8_is_fnuz);
              #pragma unroll
              for (int i = 0; i < 4; ++i) {
                a_tile[lane_arow * kATileStride + (lane_acol + i)] = a_vals[i];
              }
            }
            const uint32_t* b_src = B_u32 + (k0 >> 4) * prob_n_words + b_group_offset;
            #pragma unroll
            for (int i = 0; i < Traits::kLaneBWords; ++i) {
              b_tile[b_lds_index(lane_b_idx + i)] = b_src[lane_b_idx + i];
            }

            __threadfence_block();
            __builtin_amdgcn_sched_barrier(0);
            if (lane == 0) {
              int count = atomicAdd(&stage_ready_count[stage], 1) + 1;
              if (count == thread_k_blocks) {
                stage_ready_count[stage] = 0;
                __atomic_store_n(&stage_ready_iter[stage], iter,
                                 __ATOMIC_RELEASE);
              }
            }
          }
        }

        if (do_consumer) {
          float4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};
          for (int iter = 0; iter < slice_iters; ++iter) {
            int stage = iter % stages;
            wait_for_iter(&stage_ready_iter[stage], iter, lane);

            int k_tile = slice_row + iter;
            int k_base = k_tile * thread_k_blocks * 16;
            int k0 = k_base + k_slice * 16;

            scalar_t* a_tile = sh_a + (a_tile_base + stage * kATileElems);
            uint32_t* b_tile =
                sh_b + (b_tile_base + stage * Traits::kBTileWordsLds);

            scalar_t a_reg[4];
            scalar_t b_reg[4];
            float scale = 1.0f;
            float zp_val = 0.0f;
            int group = 0;
            if (grouped) {
              group = (k0 + lane_k_block) / (group_blocks * 16);
            }
            if (has_act_order) {
              group = g_idx[k0 + lane_k_block];
            }
            if (grouped) {
              scale =
                  load_scale_any<scalar_t, kIsFp4, kScaleE4M3>(
                      s, group, col_base + lane_col, prob_n, true);
              if constexpr (kIsFp4 && kScaleE4M3) {
                scale *= global_scale_val;
              }
            }
            if (has_zp) {
              if (is_zp_float) {
                zp_val = load_zp_float<scalar_t>(
                    b_zeros_f, group, col_base + lane_col, prob_n, grouped);
              } else {
                zp_val = static_cast<float>(
                    load_zp<num_bits>(b_zeros_u32, group,
                                      col_base + lane_col, prob_n, grouped));
              }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              a_reg[i] = a_tile[lane_row * kATileStride + (lane_k_block + i)];
              int k_in = lane_k_block + i;
              int idx_before_perm =
                  block_offset * 256 + k_in * 16 + lane_col;
              int idx_after_perm = static_cast<int>(marlin::k_perm[idx_before_perm]);
              int idx_in_row = idx_after_perm;
              int q_index = idx_in_row / Traits::kPackFactor;
              int nibble = idx_in_row % Traits::kPackFactor;
              uint32_t packed = b_tile[b_lds_index(q_index)];
              uint32_t q = (packed >> (nibble * num_bits)) & Traits::kMask;
              scalar_t val;
              if constexpr (kIsFp4) {
                val = fp4_to_scalar<scalar_t>(q);
              } else {
                if (has_zp) {
                  float val_f = static_cast<float>(q) - zp_val;
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

            __threadfence_block();
            __builtin_amdgcn_sched_barrier(0);
            if (lane == 0) {
              int count = atomicAdd(&stage_done_count[stage], 1) + 1;
              if (count == thread_k_blocks) {
                stage_done_count[stage] = 0;
                __atomic_store_n(&stage_done_iter[stage], iter,
                                 __ATOMIC_RELEASE);
              }
            }
          }

          float* c_tile = sh_c + wave_id * 16 * 16;
          int c_row_base = (lane >> 4) * 4;
          int c_col = lane & 15;
          #pragma unroll
          for (int i = 0; i < 4; ++i)
            c_tile[(c_row_base + i) * 16 + c_col] = c_reg[i];
        }
      } else if (tile % wave_groups == wave_group) {
        float4_t c_reg = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int iter = 0; iter < slice_iters; ++iter) {
          int k_tile = slice_row + iter;
          int k_base = k_tile * thread_k_blocks * 16;
          int k0 = k_base + k_slice * 16;

          scalar_t* a_tile = sh_a + a_tile_base;
          uint32_t* b_tile = sh_b + b_tile_base;
          int m = row_base + lane_arow;
          int k_a = k0 + lane_acol;
          if (a_scales == nullptr) {
            uint32x2_t a_vec = {0u, 0u};
            if (m < prob_m) {
              const scalar_t* A_s = reinterpret_cast<const scalar_t*>(A_base);
              const uint32x2_t* a_src =
                  reinterpret_cast<const uint32x2_t*>(
                      A_s + m * prob_k + k_a);
              a_vec = *a_src;
            }
            *reinterpret_cast<uint32x2_t*>(
                a_tile + lane_arow * kATileStride + lane_acol) = a_vec;
          } else {
            scalar_t a_vals[4];
            load_a_fragment(a_vals, A_base, m, k_a, prob_k, prob_m, a_scales,
                            a_is_fp8, a_fp8_is_fnuz);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
              a_tile[lane_arow * kATileStride + (lane_acol + i)] = a_vals[i];
            }
          }
          const uint32_t* b_src = B_u32 + (k0 >> 4) * prob_n_words + b_group_offset;
          #pragma unroll
          for (int i = 0; i < Traits::kLaneBWords; ++i) {
            b_tile[b_lds_index(lane_b_idx + i)] = b_src[lane_b_idx + i];
          }

          if constexpr (waves_per_block == thread_k_blocks) {
            __syncthreads();
          }

          scalar_t a_reg[4];
          scalar_t b_reg[4];
          float scale = 1.0f;
          float zp_val = 0.0f;
          int group = 0;
          if (grouped) {
            group = (k0 + lane_k_block) / (group_blocks * 16);
          }
          if (has_act_order) {
            group = g_idx[k0 + lane_k_block];
          }
          if (grouped) {
            scale =
                load_scale_any<scalar_t, kIsFp4, kScaleE4M3>(
                    s, group, col_base + lane_col, prob_n, true);
            if constexpr (kIsFp4 && kScaleE4M3) {
              scale *= global_scale_val;
            }
          }
          if (has_zp) {
            if (is_zp_float) {
              zp_val = load_zp_float<scalar_t>(
                  b_zeros_f, group, col_base + lane_col, prob_n, grouped);
            } else {
              zp_val = static_cast<float>(
                  load_zp<num_bits>(b_zeros_u32, group,
                                    col_base + lane_col, prob_n, grouped));
            }
          }
          #pragma unroll
          for (int i = 0; i < 4; ++i) {
            a_reg[i] = a_tile[lane_row * kATileStride + (lane_k_block + i)];
            int k_in = lane_k_block + i;
            int idx_before_perm = block_offset * 256 + k_in * 16 + lane_col;
            int idx_after_perm = static_cast<int>(marlin::k_perm[idx_before_perm]);
            int idx_in_row = idx_after_perm;
            int q_index = idx_in_row / Traits::kPackFactor;
            int nibble = idx_in_row % Traits::kPackFactor;
            uint32_t packed = b_tile[b_lds_index(q_index)];
            uint32_t q = (packed >> (nibble * num_bits)) & Traits::kMask;
            scalar_t val;
            if constexpr (kIsFp4) {
              val = fp4_to_scalar<scalar_t>(q);
            } else {
              if (has_zp) {
                float val_f = static_cast<float>(q) - zp_val;
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

          if constexpr (waves_per_block == thread_k_blocks) {
            __syncthreads();
          }
        }

        float* c_tile = sh_c + wave_id * 16 * 16;
        int c_row_base = (lane >> 4) * 4;
        int c_col = lane & 15;
        #pragma unroll
        for (int i = 0; i < 4; ++i)
          c_tile[(c_row_base + i) * 16 + c_col] = c_reg[i];
      }
      __syncthreads();
      if (slice_count > 1)
        barrier_acquire(&locks[slice_col], slice_idx);

      if (tile % wave_groups == wave_group && k_slice == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
          int c_row = (lane >> 4) * 4 + i;
          int c_col = lane & 15;
          int row = row_base + c_row;
          int col = col_base + c_col;
          int idx = c_row * 16 + c_col;
          if (row < prob_m && col < prob_n) {
            float sum = 0.0f;
            #pragma unroll
            for (int ks = 0; ks < thread_k_blocks; ++ks) {
              int wave = wave_group * thread_k_blocks + ks;
              sum += sh_c[wave * 16 * 16 + idx];
            }

            scalar_t* c_ptr = C_s + row * prob_n + col;
            if (slice_count > 1 && slice_idx != 0)
              sum += scalar_to_float(*c_ptr);
            if (!grouped && last) {
              scalar_t scale_val = load_scale(s_h, 0, col, prob_n, false);
              sum *= scalar_to_float(scale_val);
            }
            if (has_bias) {
              sum += scalar_to_float(b_bias[col]);
            }
            *c_ptr = float_to_scalar<scalar_t>(sum);
          }
        }
      }

      if (slice_count > 1)
        barrier_release(&locks[slice_col], last);
      __syncthreads();
    }

    slice_iters = 0;
    slice_row = 0;
    slice_col_par++;
    slice_col++;
    init_slice();
  }
}

#ifndef MARLIN_STAGES
#define MARLIN_STAGES 2
#endif

#ifndef VLLM_MARLIN_HIP_KERNEL_CONSTS
#define VLLM_MARLIN_HIP_KERNEL_CONSTS
const int THREADS = 512;
const int STAGES = MARLIN_STAGES;
const int WAVES_PER_BLOCK = THREADS / 64;
#endif
template <int num_bits>
constexpr int shared_mem_bytes() {
  return WAVES_PER_BLOCK * STAGES * kATileElems * sizeof(__half) +
         WAVES_PER_BLOCK * STAGES *
             MarlinIntTraits<num_bits>::kBTileWordsLds * sizeof(uint32_t) +
         WAVES_PER_BLOCK * kTileElems * sizeof(float);
}

#define CALL_IF(NUM_BITS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS && \
    group_blocks == GROUP_BLOCKS \
  ) { \
    constexpr int kShared = shared_mem_bytes<NUM_BITS>(); \
    hipFuncSetAttribute( \
      (const void*)MarlinInt4<scalar_t, NUM_BITS, kIsFp4, kScaleE4M3, THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>, \
      hipFuncAttributeMaxDynamicSharedMemorySize, \
      kShared \
    ); \
    hipLaunchKernelGGL( \
      (MarlinInt4<scalar_t, NUM_BITS, kIsFp4, kScaleE4M3, THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES, GROUP_BLOCKS>), \
      dim3(blocks), dim3(THREADS), kShared, stream, \
      A_ptr, B_ptr, C_ptr, b_bias_ptr, a_scales_ptr, s_ptr, global_scale_ptr, \
      g_idx_ptr, b_zeros_ptr, \
      prob_m, prob_n, prob_k, \
      locks, has_bias, has_act_order, has_zp, is_zp_float, a_is_fp8, a_fp8_is_fnuz \
    ); \
  }

template <typename scalar_t, bool kIsFp4, bool kScaleE4M3>
int gptq_marlin_hip_gemm_impl(
  const void* A,
  const void* B,
        void* C,
  const void* b_bias,
  const float* a_scales,
        void* s,
        const void* global_scale,
  const int* g_idx,
  const void* b_zeros,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int dev,
  hipStream_t stream,
  int thread_k,
  int thread_n,
  int sms,
  int num_bits,
  int group_blocks,
  int max_par,
  bool has_bias,
  bool has_act_order,
  bool has_zp,
  bool is_zp_float,
  bool a_is_fp8,
  bool a_fp8_is_fnuz
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1) {
    hipDeviceProp_t props{};
    hipGetDeviceProperties(&props, dev);
    sms = props.multiProcessorCount;
  }
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
    if (prob_n % thread_n != 0) {
      if (prob_n % 128 == 0) {
        thread_n = 128;
      } else if (prob_n % 64 == 0) {
        thread_n = 64;
      }
    }
  }

  if (prob_k < thread_k) {
    thread_k = prob_k;
  } else if (prob_k % thread_k != 0) {
    if (prob_k % 64 == 0) {
      thread_k = 64;
    } else if (prob_k % 32 == 0) {
      thread_k = 32;
    }
  }

  if (group_blocks != -1) {
    int desired_thread_k = group_blocks * 16;
    if (thread_k > desired_thread_k) {
      thread_k = desired_thread_k;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int blocks = sms;

  if (group_blocks != -1) {
    if (group_blocks < thread_k_blocks ||
        (group_blocks % thread_k_blocks) != 0) {
      return ERR_KERN_SHAPE;
    }
  }

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0)
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const int4* A_ptr = (const int4*) A;
  const int4* B_ptr = (const int4*) B;
  int4* C_ptr = (int4*) C;
  const scalar_t* b_bias_ptr =
      reinterpret_cast<const scalar_t*>(b_bias);
  const float* a_scales_ptr = a_scales;
  const void* s_ptr = s;
  const scalar_t* global_scale_ptr =
      reinterpret_cast<const scalar_t*>(global_scale);
  const int* g_idx_ptr = g_idx;
  const void* b_zeros_ptr = b_zeros;

  int* locks = (int*) workspace;

  int ret = 0;
  int a_vec_div = (a_scales_ptr != nullptr) ? 16 : 8;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }

#define CALL_THREAD_CONFIGS(NUM_BITS, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  8,  8, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  4,  8, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  4,  8, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  4,  8, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  4,  8, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  8,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  8,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  8,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  8,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  4,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  4,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  4,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  4,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  8,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  8,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  8,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  8,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  4,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  4,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  4,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  4,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1, 16,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2, 16,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3, 16,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4, 16,  1, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1, 16,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2, 16,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3, 16,  4, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4, 16,  4, GROUP_BLOCKS)

#define CALL_THREAD_CONFIGS_SMALLK(NUM_BITS, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  8,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  8,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  8,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  8,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1,  4,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2,  4,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3,  4,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4,  4,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 1, 16,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 2, 16,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 3, 16,  2, GROUP_BLOCKS) \
  CALL_IF(NUM_BITS, 4, 16,  2, GROUP_BLOCKS)

    if (num_bits == 4) {
      if (false) {
      }
      CALL_THREAD_CONFIGS(4, -1)
      CALL_THREAD_CONFIGS(4, 1)
      CALL_THREAD_CONFIGS_SMALLK(4, 2)
      CALL_THREAD_CONFIGS(4, 4)
      CALL_THREAD_CONFIGS(4, 8)
      CALL_THREAD_CONFIGS(4, 16)
      else {
        ret = ERR_KERN_SHAPE;
      }
    } else if (num_bits == 8) {
      if (false) {
      }
      CALL_THREAD_CONFIGS(8, -1)
      CALL_THREAD_CONFIGS_SMALLK(8, 2)
      CALL_THREAD_CONFIGS(8, 4)
      CALL_THREAD_CONFIGS(8, 8)
      CALL_THREAD_CONFIGS(8, 16)
      else {
        ret = ERR_KERN_SHAPE;
      }
    } else {
      ret = ERR_KERN_SHAPE;
    }

#undef CALL_THREAD_CONFIGS
#undef CALL_THREAD_CONFIGS_SMALLK

    A_ptr += 16 * thread_m_blocks * prob_k / a_vec_div * par;
    C_ptr += 16 * thread_m_blocks * prob_n / 8 * par;
  }
  return ret;
}

template <typename scalar_t>
int gptq_marlin_hip_gemm(
  const void* A,
  const void* B,
        void* C,
        const void* b_bias,
        const float* a_scales,
        void* s,
        const void* b_zeros,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int dev,
  hipStream_t stream,
  int thread_k,
  int thread_n,
  int sms,
  int num_bits,
  int group_blocks,
  int max_par,
  bool has_bias,
  bool has_act_order,
  bool has_zp,
  bool is_zp_float,
  bool a_is_fp8,
  bool a_fp8_is_fnuz,
  const int* g_idx
) {
  return gptq_marlin_hip_gemm_impl<scalar_t, false, false>(
      A, B, C, b_bias, a_scales, s, nullptr, g_idx, b_zeros, prob_m, prob_n,
      prob_k, workspace, dev, stream, thread_k, thread_n, sms, num_bits,
      group_blocks, max_par, has_bias, has_act_order, has_zp, is_zp_float,
      a_is_fp8, a_fp8_is_fnuz);
}

template <typename scalar_t, bool kScaleE4M3>
int fp4_marlin_hip_gemm(
  const void* A,
  const void* B,
        void* C,
        const void* b_bias,
        const float* a_scales,
        void* s,
        const void* global_scale,
        const void* b_zeros,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int dev,
  hipStream_t stream,
  int thread_k,
  int thread_n,
  int sms,
  int group_blocks,
  int max_par,
  bool has_bias,
  bool has_act_order,
  bool has_zp,
  bool is_zp_float,
  bool a_is_fp8,
  bool a_fp8_is_fnuz,
  const int* g_idx
) {
  return gptq_marlin_hip_gemm_impl<scalar_t, true, kScaleE4M3>(
      A, B, C, b_bias, a_scales, s, global_scale, g_idx, b_zeros, prob_m,
      prob_n, prob_k, workspace, dev, stream, thread_k, thread_n, sms, 4,
      group_blocks, max_par, has_bias, has_act_order, has_zp, is_zp_float,
      a_is_fp8, a_fp8_is_fnuz);
}

#undef CALL_IF

#endif  // VLLM_GPTQ_MARLIN_HIP_KERNEL_HIP
