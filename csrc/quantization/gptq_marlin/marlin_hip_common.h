// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#ifndef VLLM_MARLIN_HIP_COMMON_H
#define VLLM_MARLIN_HIP_COMMON_H

#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <type_traits>

#include "quantization/gptq_marlin/marlin_constants.h"
#include "quantization/gptq_marlin/marlin_perm.h"

#ifndef VLLM_HAS_MFMA_BF16
#define VLLM_HAS_MFMA_BF16 0
#endif

constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

using float2_t = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using float4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using uint32x2_t =
    __attribute__((__vector_size__(2 * sizeof(uint32_t)))) uint32_t;

template <typename scalar_t>
__device__ __forceinline__ float scalar_to_float(scalar_t v);

template <>
__device__ __forceinline__ float scalar_to_float<__half>(__half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float scalar_to_float<__hip_bfloat16>(
    __hip_bfloat16 v) {
  return static_cast<float>(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t float_to_scalar(float v);

template <>
__device__ __forceinline__ __half float_to_scalar<__half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ __hip_bfloat16 float_to_scalar<__hip_bfloat16>(
    float v) {
  return __float2bfloat16(v);
}

__device__ __forceinline__ int b_lds_index(int idx) {
  int row = idx >> 4;
  int col = idx & 15;
  return row * kBTileStride + col;
}

template <typename scalar_t>
__device__ inline scalar_t load_scale(
    const scalar_t* s,
    int group,
    int n,
    int n_cols,
    bool grouped) {
  if (grouped) {
    int block = n >> 6;
    int in_block = n & 63;
    int perm_idx = static_cast<int>(marlin::k_scale_perm_inv[in_block]);
    int idx = group * n_cols + block * 64 + perm_idx;
    return s[idx];
  }
  int block = n >> 5;
  int in_block = n & 31;
  int perm_idx = static_cast<int>(marlin::k_scale_perm_single_inv[in_block]);
  int idx = block * 32 + perm_idx;
  return s[idx];
}

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    while (__hip_atomic_load(lock, __ATOMIC_ACQUIRE,
                             __HIP_MEMORY_SCOPE_AGENT) != count) {
    }
  }
  __syncthreads();
}

__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      __hip_atomic_store(lock, 0, __ATOMIC_RELEASE,
                         __HIP_MEMORY_SCOPE_AGENT);
      return;
    }
    __hip_atomic_fetch_add(lock, 1, __ATOMIC_RELEASE,
                           __HIP_MEMORY_SCOPE_AGENT);
  }
}

__device__ __forceinline__ void wait_for_iter(int* iter_ptr, int target,
                                              int lane) {
  unsigned go = 0;
  do {
    if (lane == 0) {
      int cur = __atomic_load_n(iter_ptr, __ATOMIC_ACQUIRE);
      go = static_cast<unsigned>(cur >= target);
    }
    go = __builtin_amdgcn_readfirstlane(go);
    if (!go)
      __builtin_amdgcn_s_sleep(0);
  } while (!go);
}

template <typename scalar_t>
__device__ __forceinline__ float4_t mfma_16x16x16(float2_t a, float2_t b,
                                                  float4_t acc);

template <>
__device__ __forceinline__ float4_t mfma_16x16x16<__half>(
    float2_t a, float2_t b, float4_t acc) {
  return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}

#if VLLM_HAS_MFMA_BF16
template <>
__device__ __forceinline__ float4_t mfma_16x16x16<__hip_bfloat16>(
    float2_t a, float2_t b, float4_t acc) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16(a, b, acc, 0, 0, 0);
}
#endif

template <typename scalar_t>
union pack4_t {
  scalar_t v[4];
  uint32x2_t u;
  float2_t f;
};

template <typename scalar_t>
__device__ __forceinline__ float4_t mfma_compute(pack4_t<scalar_t>& pack_a,
                                                 pack4_t<scalar_t>& pack_b,
                                                 float4_t acc) {
  if constexpr (std::is_same_v<scalar_t, __hip_bfloat16>) {
#if VLLM_HAS_MFMA_BF16
    return mfma_16x16x16<scalar_t>(pack_a.f, pack_b.f, acc);
#else
    pack4_t<__half> pack_a_f16{};
    pack4_t<__half> pack_b_f16{};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
      pack_a_f16.v[i] = __float2half(scalar_to_float(pack_a.v[i]));
      pack_b_f16.v[i] = __float2half(scalar_to_float(pack_b.v[i]));
    }
    return mfma_16x16x16<__half>(pack_a_f16.f, pack_b_f16.f, acc);
#endif
  } else {
    return mfma_16x16x16<scalar_t>(pack_a.f, pack_b.f, acc);
  }
}

#endif  // VLLM_MARLIN_HIP_COMMON_H
