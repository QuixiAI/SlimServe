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

__device__ __forceinline__ void cp_async4_pred(int4* dst,
                                               const int4* src,
                                               bool pred) {
  if (pred) {
    *dst = *src;
  }
}

__device__ __forceinline__ void cp_async_fence() {}

template <int n>
__device__ __forceinline__ void cp_async_wait() {}


template <int const num_threads, int const num_bits, bool const has_perm,
          bool is_a_8bit>
__global__ void gptq_marlin_repack_kernel(
    const uint32_t* __restrict__ b_q_weight_ptr,
    const uint32_t* __restrict__ perm_ptr, uint32_t* __restrict__ out_ptr,
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

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    cp_async_wait<kRepackStages - 2>();
    __syncthreads();
  };

  extern __shared__ int4 sh[];

  constexpr int perm_size = target_tile_k_size / 4;

  int4* sh_perm_ptr = sh;
  int4* sh_pipe_ptr = sh_perm_ptr;
  if constexpr (has_perm) {
    sh_pipe_ptr += perm_size;
  }

  constexpr int tile_ints = target_tile_k_size / pack_factor;

  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? target_tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;

  auto load_perm_to_shared = [&](int k_tile_id) {
    int first_k_int4 = (k_tile_id * target_tile_k_size) / 4;

    const int4* perm_int4_ptr = reinterpret_cast<const int4*>(perm_ptr);

    if (threadIdx.x < perm_size) {
      sh_perm_ptr[threadIdx.x] = perm_int4_ptr[first_k_int4 + threadIdx.x];
    }
    __syncthreads();
  };

  auto fetch_to_shared = [&](int pipe, int k_tile_id, int n_tile_id) {
    if (n_tile_id >= n_tiles) {
      cp_async_fence();
      return;
    }

    int first_n = n_tile_id * target_tile_n_size;

    int4* sh_ptr = sh_pipe_ptr + stage_size * pipe;

    if constexpr (has_perm) {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        const uint32_t* sh_perm_int_ptr =
            reinterpret_cast<const uint32_t*>(sh_perm_ptr);

        int src_k = sh_perm_int_ptr[k_id];
        int src_k_packed = src_k / pack_factor;

        cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                  reinterpret_cast<const int4*>(
                      &(b_q_weight_ptr[src_k_packed * size_n + first_n +
                                        (n_id * 4)])));
      }

    } else {
      if (threadIdx.x < stage_size) {
        auto k_id = threadIdx.x / stage_n_threads;
        auto n_id = threadIdx.x % stage_n_threads;

        int first_k = k_tile_id * target_tile_k_size;
        int first_k_packed = first_k / pack_factor;

        cp_async4(&sh_ptr[k_id * stage_n_threads + n_id],
                  reinterpret_cast<const int4*>(
                      &(b_q_weight_ptr[(first_k_packed + k_id) * size_n +
                                       first_n + (n_id * 4)])));
      }
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

    constexpr int sh_stride = target_tile_n_size;
    constexpr uint32_t mask = (1u << num_bits) - 1u;

    int4* sh_stage_ptr = sh_pipe_ptr + stage_size * pipe;
    uint32_t* sh_stage_int_ptr = reinterpret_cast<uint32_t*>(sh_stage_ptr);

    uint32_t* sh_perm_int_ptr = reinterpret_cast<uint32_t*>(sh_perm_ptr);

    uint32_t vals[8];

    if constexpr (has_perm) {
      static_assert(!is_a_8bit, "perm path does not support a8");
      for (int i = 0; i < 4; i++) {
        int k_idx = tc_row + tc_offsets[i];

        uint32_t src_k = sh_perm_int_ptr[k_idx];
        uint32_t src_k_pos = src_k % pack_factor;

        uint32_t b1_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n];
        uint32_t b1_cur_val = (b1_val >> (src_k_pos * num_bits)) & mask;

        uint32_t b2_val = sh_stage_int_ptr[k_idx * sh_stride + cur_n + 8];
        uint32_t b2_cur_val = (b2_val >> (src_k_pos * num_bits)) & mask;

        vals[i] = b1_cur_val;
        vals[4 + i] = b2_cur_val;
      }

    } else {
      uint32_t b1_vals[tile_ints];
      uint32_t b2_vals[tile_ints];

#pragma unroll
      for (int i = 0; i < tile_ints; i++) {
        if constexpr (is_a_8bit) {
          b1_vals[i] =
              sh_stage_int_ptr[cur_n + sh_stride * i + (warp_id % 2) * 8];
        } else {
          b1_vals[i] = sh_stage_int_ptr[cur_n + sh_stride * i];
          b2_vals[i] = sh_stage_int_ptr[cur_n + 8 + sh_stride * i];
        }
      }

#pragma unroll
      for (int i = 0; i < 4; i++) {
        int cur_elem = tc_row + (is_a_8bit ? i : tc_offsets[i]);
        int cur_int = cur_elem / pack_factor;
        int cur_pos = cur_elem % pack_factor;

        vals[i] = (b1_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        if constexpr (is_a_8bit) {
          vals[4 + i] = (b1_vals[cur_int + tile_ints / 2] >>
                         (cur_pos * num_bits)) &
                        mask;
        } else {
          vals[4 + i] = (b2_vals[cur_int] >> (cur_pos * num_bits)) & mask;
        }
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

    if constexpr (has_perm) {
      load_perm_to_shared(k_tile_id);
    }

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

template <int num_bits, bool has_perm, bool is_a_8bit>
constexpr int repack_shared_bytes() {
  constexpr int pack_factor = 32 / num_bits;
  constexpr int target_tile_n_size = kTileN / (is_a_8bit ? 2 : 1);
  constexpr int target_tile_k_size = kTileK * (is_a_8bit ? 2 : 1);
  constexpr int perm_size = has_perm ? (target_tile_k_size / 4) : 0;
  constexpr int tile_ints = target_tile_k_size / pack_factor;
  constexpr int stage_n_threads = target_tile_n_size / 4;
  constexpr int stage_k_threads = has_perm ? target_tile_k_size : tile_ints;
  constexpr int stage_size = stage_k_threads * stage_n_threads;
  constexpr int sh_int4 = perm_size + stage_size * kRepackStages;
  return static_cast<int>(sh_int4 * sizeof(int4));
}

}  // namespace

#define CALL_IF(NUM_BITS, HAS_PERM, IS_A_8BIT)                                \
  else if (num_bits == NUM_BITS && has_perm == HAS_PERM &&                    \
           is_a_8bit == IS_A_8BIT) {                                          \
    constexpr int kShared = repack_shared_bytes<NUM_BITS, HAS_PERM, IS_A_8BIT>(); \
    hipFuncSetAttribute(                                                      \
        (const void*)gptq_marlin_repack_kernel<kRepackThreads, NUM_BITS,       \
                                               HAS_PERM, IS_A_8BIT>,          \
        hipFuncAttributeMaxDynamicSharedMemorySize, kShared);                 \
    hipLaunchKernelGGL(                                                       \
        (gptq_marlin_repack_kernel<kRepackThreads, NUM_BITS, HAS_PERM,         \
                                   IS_A_8BIT>),                               \
        dim3(blocks), dim3(kRepackThreads), kShared, stream,                  \
        b_q_weight_ptr, perm_ptr, out_ptr, static_cast<int>(size_k),          \
        static_cast<int>(size_n));                                            \
  }

torch::Tensor gptq_marlin_repack(torch::Tensor& b_q_weight, torch::Tensor& perm,
                                 int64_t size_k, int64_t size_n,
                                 int64_t num_bits, bool is_a_8bit) {
  TORCH_CHECK(size_k % kTileK == 0, "size_k = ", size_k,
              " is not divisible by tile_k_size = ", kTileK);
  TORCH_CHECK(size_n % kTileN == 0, "size_n = ", size_n,
              " is not divisible by tile_n_size = ", kTileN);

  TORCH_CHECK(num_bits == 4 || num_bits == 8,
              "num_bits must be 4 or 8. Got = ", num_bits);
  int const pack_factor = 32 / num_bits;

  TORCH_CHECK((size_k / pack_factor) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", pack_factor = ", pack_factor);
  TORCH_CHECK(b_q_weight.size(1) == size_n,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not size_n = ", size_n);

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_q_weight.dtype() == at::kInt,
              "b_q_weight type is not kInt");

  TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
  TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");
  TORCH_CHECK(perm.dtype() == at::kInt, "perm type is not at::kInt");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto options = torch::TensorOptions()
                     .dtype(b_q_weight.dtype())
                     .device(b_q_weight.device());
  torch::Tensor out = torch::empty(
      {size_k / kTileSize, size_n * kTileSize / pack_factor}, options);

  bool has_perm = perm.size(0) != 0;

  const uint32_t* b_q_weight_ptr =
      reinterpret_cast<const uint32_t*>(b_q_weight.data_ptr());
  const uint32_t* perm_ptr = reinterpret_cast<const uint32_t*>(perm.data_ptr());
  uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

  int dev = b_q_weight.get_device();
  hipStream_t stream = at::cuda::getCurrentCUDAStream(dev);
  int blocks = 0;
  hipDeviceGetAttribute(&blocks, hipDeviceAttributeMultiprocessorCount, dev);

  if (false) {
  }
  CALL_IF(4, false, false)
  CALL_IF(4, true, false)
  CALL_IF(8, false, false)
  CALL_IF(8, true, false)

  CALL_IF(4, false, true)
  CALL_IF(8, false, true)

  else {
    TORCH_CHECK(false, "Unsupported repack config: num_bits = ", num_bits,
                ", has_perm = ", has_perm, ", is_a_8bit = ", is_a_8bit);
  }

  return out;
}

TORCH_LIBRARY_IMPL_EXPAND(TORCH_EXTENSION_NAME, CUDA, m) {
  m.impl("gptq_marlin_repack", &gptq_marlin_repack);
}
