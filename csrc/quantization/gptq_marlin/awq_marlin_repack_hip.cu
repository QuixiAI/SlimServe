// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_runtime.h>

#include "core/registration.h"

namespace {

constexpr int kTileSize = 16;
constexpr int kTileK = 16;
constexpr int kTileN = 64;
constexpr int kRepackStages = 8;
constexpr int kRepackThreads = 256;

constexpr int ceildiv(int a, int b) { return (a + b - 1) / b; }

__device__ __forceinline__ void cp_async4(int4* dst, const int4* src) {
  *dst = *src;
}

__device__ __forceinline__ void cp_async_fence() {}

template <int n>
__device__ __forceinline__ void cp_async_wait() {}

template <int const num_threads, int const num_bits, bool is_a_8bit>
__global__ void awq_marlin_repack_kernel(
    const uint32_t* __restrict__ b_q_weight_ptr,
    uint32_t* __restrict__ out_ptr,
    int size_k, int size_n) {
  constexpr int pack_factor = 32 / num_bits;

  constexpr int target_tile_n_size = kTileN / (is_a_8bit ? 2 : 1);
  constexpr int target_tile_k_size = kTileK * (is_a_8bit ? 2 : 1);
  int k_tiles = size_k / target_tile_k_size;
  int n_tiles = size_n / target_tile_n_size;
  int block_k_tiles = ceildiv(k_tiles, gridDim.x);

  int start_k_tile = blockIdx.x * block_k_tiles;
  if (start_k_tile >= k_tiles) {
    return;
  }

  int finish_k_tile = min(start_k_tile + block_k_tiles, k_tiles);

  auto wait_for_stage = [&]() {
    cp_async_wait<kRepackStages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int tile_n_ints = target_tile_n_size / pack_factor;

  constexpr int stage_n_threads = tile_n_ints / 4;
  constexpr int stage_k_threads = target_tile_k_size;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;
    int first_n_packed = first_n / pack_factor;

    int4* sh_ptr = sh + stage_size * pipe;

    if (threadIdx.x < stage_size) {
      auto k_id = threadIdx.x / stage_n_threads;
      auto n_id = threadIdx.x % stage_n_threads;

      int first_k = k_tile_id * target_tile_k_size;

      cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                reinterpret_cast<const int4*>(
                    &(b_q_weight_ptr[(first_k + k_id) * (size_n / pack_factor) +
                                     first_n_packed + (n_id * 4)])));
    }

    cp_async_fence();
  };

  auto repack_tile = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      return;
    }

    int warp_id = threadIdx.x / 32;
    int th_id = threadIdx.x % 32;

    if (warp_id >= 4) {
      return;
    }

    int tc_col = th_id / 4;
    int tc_row = (th_id % 4) * (is_a_8bit ? 4 : 2);

    constexpr int tc_offsets[4] = {0, 1, 8, 9};

    int cur_n = (warp_id / (is_a_8bit ? 2 : 1)) * 16 + tc_col;
    int cur_n_packed = cur_n / pack_factor;
    int cur_n_pos = cur_n % pack_factor;

    constexpr int sh_stride = tile_n_ints;
    constexpr uint32_t mask = (1u << num_bits) - 1u;

    int4* sh_stage_ptr = sh + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    int cur_n_pos_unpacked;
    if constexpr (num_bits == 4) {
      constexpr int undo_pack[8] = {0, 4, 1, 5, 2, 6, 3, 7};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    } else {
      constexpr int undo_pack[4] = {0, 2, 1, 3};
      cur_n_pos_unpacked = undo_pack[cur_n_pos];
    }

    uint32_t vals[8];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      if constexpr (is_a_8bit) {
        int cur_elem = tc_row + i;

        uint32_t packed_src_0 =
            sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) * (warp_id % 2) +
                             sh_stride * cur_elem];
        uint32_t packed_src_1 =
            sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) * (warp_id % 2) +
                             sh_stride * (cur_elem + 16)];

        vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
        vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
      } else {
        int cur_elem = tc_row + tc_offsets[i];

        uint32_t packed_src_0 =
            sh_stage_int_ptr[cur_n_packed + sh_stride * cur_elem];
        uint32_t packed_src_1 = sh_stage_int_ptr[cur_n_packed + (8 / pack_factor) +
                                            sh_stride * cur_elem];

        vals[i] = (packed_src_0 >> (cur_n_pos_unpacked * num_bits)) & mask;
        vals[4 + i] = (packed_src_1 >> (cur_n_pos_unpacked * num_bits)) & mask;
      }
    }

    constexpr int tile_size =
        target_tile_k_size * target_tile_n_size / pack_factor;
    int out_offset = (k_tile_id * n_tiles + n_tile_id) * tile_size;

    if constexpr (!is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 2, 4, 6, 1, 3, 5, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else if constexpr (is_a_8bit && num_bits == 4) {
      int pack_idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};

      uint32_t res = 0;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        res |= vals[pack_idx[i]] << (i * 4);
      }

      out_ptr[out_offset + th_id * 4 + warp_id] = res;

    } else {
      constexpr int pack_idx[4] = {0, 2, 1, 3};

      uint32_t res1 = 0;
      uint32_t res2 = 0;
#pragma unroll
      for (int i = 0; i < 4; i++) {
        const int ii = is_a_8bit ? i : pack_idx[i];
        res1 |= vals[ii] << (i * 8);
        res2 |= vals[4 + ii] << (i * 8);
      }

      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 0] = res1;
      out_ptr[out_offset + th_id * 8 + (warp_id * 2) + 1] = res2;
    }
  };

  auto start_pipes = [&](int k_tile_id, int n_tile_id) {
#pragma unroll
    for (int pipe = 0; pipe < kRepackStages - 1; pipe++) {
      fetch_to_shared(pipe, k_tile_id, n_tile_id + pipe);
    }

    wait_for_stage();
  };

#pragma unroll
  for (int k_tile_id = start_k_tile; k_tile_id < finish_k_tile; k_tile_id++) {
    int n_tile_id = 0;

    start_pipes(k_tile_id, n_tile_id);

    while (n_tile_id < n_tiles) {
#pragma unroll
      for (int pipe = 0; pipe < kRepackStages; pipe++) {
        fetch_to_shared((pipe + kRepackStages - 1) % kRepackStages, k_tile_id,
                        n_tile_id + pipe + kRepackStages - 1);
        repack_tile(pipe, k_tile_id, n_tile_id + pipe);
        wait_for_stage();
      }
      n_tile_id += kRepackStages;
    }
  }
}

template <int num_bits, bool is_a_8bit>
constexpr int repack_shared_bytes() {
  constexpr int pack_factor = 32 / num_bits;
  constexpr int target_tile_n_size = kTileN / (is_a_8bit ? 2 : 1);
  constexpr int target_tile_k_size = kTileK * (is_a_8bit ? 2 : 1);
  constexpr int tile_n_ints = target_tile_n_size / pack_factor;
  constexpr int stage_n_threads = tile_n_ints / 4;
  constexpr int stage_k_threads = target_tile_k_size;
  constexpr int stage_size = stage_k_threads * stage_n_threads;
  constexpr int sh_int4 = stage_size * kRepackStages;
  return static_cast<int>(sh_int4 * sizeof(int4));
}

}  // namespace

#define CALL_IF(NUM_BITS, IS_A_8BIT)                                         \
  else if (num_bits == NUM_BITS && is_a_8bit == IS_A_8BIT) {                 \
    constexpr int kShared = repack_shared_bytes<NUM_BITS, IS_A_8BIT>();      \
    hipFuncSetAttribute(                                                     \
        (const void*)awq_marlin_repack_kernel<kRepackThreads, NUM_BITS,       \
                                               IS_A_8BIT>,                   \
        hipFuncAttributeMaxDynamicSharedMemorySize, kShared);                \
    hipLaunchKernelGGL(                                                      \
        (awq_marlin_repack_kernel<kRepackThreads, NUM_BITS, IS_A_8BIT>),      \
        dim3(blocks), dim3(kRepackThreads), kShared, stream,                 \
        b_q_weight_ptr, out_ptr, static_cast<int>(size_k),                   \
        static_cast<int>(size_n));                                           \
  }

torch::Tensor awq_marlin_repack(torch::Tensor& b_q_weight, int64_t size_k,
                                int64_t size_n, int64_t num_bits,
                                bool is_a_8bit) {
  TORCH_CHECK(size_k % kTileK == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", kTileK);
  TORCH_CHECK(size_n % kTileN == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", kTileN);

  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int pack_factor = 32 / num_bits;

  TORCH_CHECK(b_q_weight.size(0) == size_k,
              "b_q_weight.size(0) = ", b_q_weight.size(0),
              " is not size_k = ", size_k);
  TORCH_CHECK((size_n / pack_factor) == b_q_weight.size(1),
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not size_n/pack_factor = ", size_n / pack_factor);

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_q_weight.dtype() == at::kInt,
              "b_q_weight type is not kInt");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto options = torch::TensorOptions()
                     .dtype(b_q_weight.dtype())
                     .device(b_q_weight.device());
  torch::Tensor out = torch::empty(
      {size_k / kTileSize, size_n * kTileSize / pack_factor}, options);

  const uint32_t* b_q_weight_ptr =
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  int dev = b_q_weight.get_device();
  hipStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int blocks = 0;
  hipDeviceGetAttribute(&blocks, hipDeviceAttributeMultiprocessorCount, dev);

  if (false) {
  }
  CALL_IF(4, false)
  CALL_IF(8, false)
  CALL_IF(4, true)
  CALL_IF(8, true)
  else {
    TORCH_CHECK(false, "Unsupported repack config: num_bits = ", num_bits,
                ", is_a_8bit = ", is_a_8bit);
  }

  return out;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("awq_marlin_repack", &awq_marlin_repack);
}
