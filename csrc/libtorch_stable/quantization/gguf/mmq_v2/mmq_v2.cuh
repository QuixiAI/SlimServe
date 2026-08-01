// MIT License
//
// Copyright (c) 2023-2024 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Adapted from llama.cpp ggml/src/ggml-cuda/{mmq,mmq-load-tiles,mmq-vec-dot,
// quantize}.cu[h] at commit 9b2a088819cda774bdbf713168ee1eee8498cda5.
//
// Scope: Q8_0 x q8_1 matmul for CUDA sm80+ only. Reduced to a single tile
// config family (I=128, 8 warps, 256 K per iteration) with the plain tiled grid
// (no stream-k) and the D4 activation scale layout. The ROCm build never sees
// this file.

#pragma once

#ifndef USE_ROCM

  #include <cuda_fp16.h>
  #include <cuda_runtime.h>

  #include "mma_int8.cuh"

namespace vllm_mmq_v2 {

constexpr int WARP_SIZE_V2 = 32;
constexpr int MMQ_TILE_NE_K = 32;
// block_q8_1_mmq is 32 int32 of quants + 4 int32 of scales.
constexpr int MMQ_TILE_Y_K = MMQ_TILE_NE_K + MMQ_TILE_NE_K / QI8_1;
constexpr int MMQ_ITER_K = 256;
constexpr int MMQ_NTHREADS = 256;
constexpr int MMQ_NWARPS = MMQ_NTHREADS / WARP_SIZE_V2;
constexpr int MMQ_I = 128;
// 2*32 quant words + 2*32/QI8_0 scale words + 4 padding; %8==4 keeps every
// ldmatrix address 16 byte aligned.
constexpr int MMQ_SRAM_STRIDE_Q8_0 = 2 * MMQ_TILE_NE_K + 2 * MMQ_TILE_NE_K / QI8_0 + 4;

// One block covers 128 contiguous values of a single column.
constexpr int QK8_1_MMQ = 4 * QK8_1;

struct block_q8_1_mmq {
  float d4[4];  // one scale per 32 values
  int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq) == 144, "unexpected block_q8_1_mmq size");

constexpr int MMQ_Y_INTS = sizeof(block_q8_1_mmq) / sizeof(int);  // 36

static constexpr __host__ __device__ int mmq_rows_per_warp(int J) {
  return (J >= 48 && J % 16 == 0) ? 32 : 16;
}

static constexpr __host__ __device__ int mmq_pad(int x, int n) { return ((x + n - 1) / n) * n; }

// Shared memory: J ids + the y tile + the x tile.
static constexpr __host__ __device__ int mmq_nbytes_shared(int J) {
  return J * (int)sizeof(int) +
         mmq_pad(J * MMQ_TILE_Y_K, MMQ_NTHREADS) * (int)sizeof(int) +
         MMQ_I * MMQ_SRAM_STRIDE_Q8_0 * (int)sizeof(int);
}

// ---------------------------------------------------------------- activations
//
// Layout differs from the legacy quantize_row_q8_1: values are grouped 128 per
// block (not 32), and the column index is the fastest-varying dimension, so the
// J columns of a tile are contiguous and the y tile is one flat copy.
// Element (column j, k-block kb) lives at block index kb*ncols + j.
template <typename scalar_t>
static __global__ void quantize_mmq_q8_1_d4(const scalar_t* __restrict__ x,
                                            void* __restrict__ vy,
                                            const int ncols, const int kx) {
  const int j = blockIdx.y;
  const int kb = blockIdx.x;
  const int t = threadIdx.x;  // 32 threads, 4 values each

  const int k0 = kb * QK8_1_MMQ + 4 * t;

  float v[4];
  #pragma unroll
  for (int l = 0; l < 4; ++l) {
    const int k = k0 + l;
    v[l] = k < kx ? static_cast<float>(x[(size_t)j * kx + k]) : 0.0f;
  }

  float amax = fabsf(v[0]);
  #pragma unroll
  for (int l = 1; l < 4; ++l) {
    amax = fmaxf(amax, fabsf(v[l]));
  }
  // one scale per 32 values == 8 threads
  #pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, mask, 32));
  }

  const float d = amax / 127.0f;
  const float id = amax > 0.0f ? 127.0f / amax : 0.0f;

  block_q8_1_mmq* y = (block_q8_1_mmq*)vy;
  block_q8_1_mmq* yb = &y[(size_t)kb * ncols + j];

  char4 q;
  q.x = (int8_t)roundf(v[0] * id);
  q.y = (int8_t)roundf(v[1] * id);
  q.z = (int8_t)roundf(v[2] * id);
  q.w = (int8_t)roundf(v[3] * id);
  *((char4*)&yb->qs[4 * t]) = q;

  if (t % 8 == 0) {
    yb->d4[t / 8] = d;
  }
}

template <typename scalar_t>
static void quantize_mmq_q8_1_d4_cuda(const scalar_t* x, void* vy,
                                      const int ncols, const int kx,
                                      cudaStream_t stream) {
  const dim3 nb(kx / QK8_1_MMQ, ncols, 1);
  quantize_mmq_q8_1_d4<scalar_t><<<nb, WARP_SIZE_V2, 0, stream>>>(x, vy, ncols,
                                                                  kx);
}

// ----------------------------------------------------------------- load tiles
template <int J, bool fallback>
static __device__ __forceinline__ void load_tiles_q8_0(
    const char* __restrict__ x, int* __restrict__ x_tile, const int kbx0,
    const int i_max, const int stride) {
  constexpr int sram_stride = MMQ_SRAM_STRIDE_Q8_0;

  int* x_qs = (int*)x_tile;
  float* x_df = (float*)(x_tile + 2 * MMQ_TILE_NE_K);

  const int txi = threadIdx.x;
  const int kbx = txi / QI8_0;
  const int kqsx = txi % QI8_0;

  #pragma unroll
  for (int i0 = 0; i0 < MMQ_I; i0 += MMQ_NWARPS) {
    int i = i0 + threadIdx.y;
    if (fallback) {
      i = min(i, i_max);
    }

    const block_q8_0* bxi = (const block_q8_0*)x + kbx0 + i * stride + kbx;

    x_qs[i * sram_stride + 0 + txi] = get_int_b2(bxi[0].qs, kqsx);
    x_qs[i * sram_stride + MMQ_TILE_NE_K + txi] =
        get_int_b2(bxi[MMQ_TILE_NE_K / QI8_0].qs, kqsx);
  }

  constexpr int blocks_per_tile_x_row = 2 * MMQ_TILE_NE_K / QI8_0;  // 8
  constexpr int rows_per_warp = WARP_SIZE_V2 / blocks_per_tile_x_row;
  const int kbxd = threadIdx.x % blocks_per_tile_x_row;

  #pragma unroll
  for (int i0 = 0; i0 < MMQ_I; i0 += MMQ_NWARPS * rows_per_warp) {
    int i = i0 + threadIdx.y * rows_per_warp +
            threadIdx.x / blocks_per_tile_x_row;
    if (fallback) {
      i = min(i, i_max);
    }

    const block_q8_0* bxi = (const block_q8_0*)x + kbx0 + i * stride + kbxd;
    x_df[i * sram_stride + kbxd] = __half2float(bxi->d);
  }
}

// ------------------------------------------------------------------- vec dot
template <int J>
static __device__ __forceinline__ void vec_dot_q8_0_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ sum, const int k00) {
  typedef tile<16, 8> tile_A;
  typedef tile<8, 8> tile_B;
  typedef tile<16, 8> tile_C;

  constexpr int sram_stride = MMQ_SRAM_STRIDE_Q8_0;
  constexpr int rows_per_warp = mmq_rows_per_warp(J);
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

  const int* x_qs = (const int*)x;
  const float* x_df = (const float*)x_qs + 2 * MMQ_TILE_NE_K;
  const int* y_qs = (const int*)y + 4;
  const float* y_df = (const float*)y;

  tile_A A[ntx][MMQ_TILE_NE_K / QI8_0];
  float dA[ntx][tile_C::ne / 2][MMQ_TILE_NE_K / QI8_0];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

  #pragma unroll
  for (int n = 0; n < ntx; ++n) {
  #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
      load_ldmatrix(A[n][k01 / QI8_0],
                    x_qs + (i0 + n * tile_A::I) * sram_stride + k00 + k01,
                    sram_stride);
    }

  #pragma unroll
    for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);
  #pragma unroll
      for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        dA[n][l][k01 / QI8_0] = x_df[i * sram_stride + (k00 + k01) / QI8_0];
      }
    }
  }

  #pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
  #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
      tile_B B;
      float dB[tile_C::ne / 2];

      // load_generic beats load_ldmatrix for the B fragment here
      load_generic(B, y_qs + j0 * MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

  #pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * MMQ_TILE_Y_K + k01 / QI8_1];
      }

  #pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01 / QI8_0], B);

  #pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
              C.x[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
        }
      }
    }
  }
}

// ----------------------------------------------------------------- write back
// dst_f32 != nullptr selects the split-K path: partial sums are accumulated
// into an fp32 scratch buffer instead of being stored to dst.
template <typename scalar_t, int J, bool fallback>
static __device__ __forceinline__ void write_back_mma(
    const float* __restrict__ sum, scalar_t* __restrict__ dst,
    float* __restrict__ dst_f32, const int stride, const int i_max,
    const int j_max) {
  typedef tile<16, 8> tile_C;

  constexpr int rows_per_warp = mmq_rows_per_warp(J);
  constexpr int ntx = rows_per_warp / tile_C::I;

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);

  #pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
  #pragma unroll
    for (int n = 0; n < ntx; ++n) {
  #pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j > j_max) {
          continue;
        }
        const int i = i0 + n * tile_C::I + tile_C::get_i(l);
        if (fallback && i > i_max) {
          continue;
        }
        const float v = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
        if (dst_f32 != nullptr) {
          atomicAdd(&dst_f32[(size_t)j * stride + i], v);
        } else {
          dst[(size_t)j * stride + i] = (scalar_t)v;
        }
      }
    }
  }
}

template <typename scalar_t>
static __global__ void mmq_v2_cast_f32(const float* __restrict__ src,
                                       scalar_t* __restrict__ dst,
                                       const int64_t n) {
  const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    dst[i] = (scalar_t)src[i];
  }
}

// --------------------------------------------------------------- main kernel
template <typename scalar_t, int J, bool fallback>
__launch_bounds__(MMQ_NTHREADS, 1) static __global__
    void mul_mat_q_v2(const char* __restrict__ x, const int* __restrict__ y,
                      scalar_t* __restrict__ dst, float* __restrict__ dst_f32,
                      const int nrows_x, const int ncols_dst,
                      const int stride_row_x, const int ncols_y,
                      const int stride_col_dst, const int blocks_per_ne00,
                      const int nsplit) {
  constexpr int blocks_per_iter = MMQ_ITER_K / QK8_0;

  extern __shared__ int mmq_v2_smem[];
  int* tile_y = mmq_v2_smem;
  int* tile_x = tile_y + mmq_pad(J * MMQ_TILE_Y_K, MMQ_NTHREADS);

  const int jt = blockIdx.y;
  const int it = blockIdx.x;
  const int zt = blockIdx.z;

  // split-K: each z slice owns a whole number of 256-element iterations
  const int niter = blocks_per_ne00 / blocks_per_iter;
  const int kb_beg = (zt * niter / nsplit) * blocks_per_iter;
  const int kb_end = zt == nsplit - 1
                         ? blocks_per_ne00
                         : ((zt + 1) * niter / nsplit) * blocks_per_iter;

  const int tile_x_max_i = nrows_x - it * MMQ_I - 1;
  const int tile_y_max_j = ncols_dst - jt * J - 1;

  const int offset_x = it * MMQ_I * stride_row_x;
  const int offset_y = jt * J * MMQ_Y_INTS;

  const int* yt = y + offset_y;
  const size_t dst_off = (size_t)jt * J * stride_col_dst + it * MMQ_I;
  scalar_t* dstt = dst + dst_off;
  float* dstt_f32 = dst_f32 != nullptr ? dst_f32 + dst_off : nullptr;

  float sum[J * MMQ_I / (MMQ_NWARPS * WARP_SIZE_V2)] = {0.0f};

  for (int kb0 = kb_beg; kb0 < kb_end; kb0 += blocks_per_iter) {
    load_tiles_q8_0<J, fallback>(x, tile_x, offset_x + kb0, tile_x_max_i,
                                 stride_row_x);
    {
      const int* by0 = yt + (size_t)ncols_y * (kb0 * QK8_0 / QK8_1_MMQ) * MMQ_Y_INTS;
  #pragma unroll
      for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
        const int l = l0 + threadIdx.y * WARP_SIZE_V2 + threadIdx.x;
        if (l0 + MMQ_NTHREADS <= J * MMQ_TILE_Y_K || l < J * MMQ_TILE_Y_K) {
          tile_y[l] = by0[l];
        }
      }
    }

    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y, sum, 0);
    __syncthreads();

    {
      const int* by0 =
          yt + (size_t)ncols_y * ((kb0 * QK8_0 / QK8_1_MMQ) * MMQ_Y_INTS + MMQ_Y_INTS);
  #pragma unroll
      for (int l0 = 0; l0 < J * MMQ_TILE_Y_K; l0 += MMQ_NTHREADS) {
        const int l = l0 + threadIdx.y * WARP_SIZE_V2 + threadIdx.x;
        if (l0 + MMQ_NTHREADS <= J * MMQ_TILE_Y_K || l < J * MMQ_TILE_Y_K) {
          tile_y[l] = by0[l];
        }
      }
    }

    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y, sum, MMQ_TILE_NE_K);
    __syncthreads();
  }

  write_back_mma<scalar_t, J, fallback>(sum, dstt, dstt_f32, stride_col_dst,
                                        tile_x_max_i, tile_y_max_j);
}

// -------------------------------------------------------------------- launch
// Without stream-k the plain tiled grid can leave most of the chip idle at low
// batch (nty*ntx tiles only), so split K across blockIdx.z to fill it.
static inline int mmq_v2_nsplit(int nty, int ntx, int ncols_x, int nsm) {
  const int niter = ncols_x / MMQ_ITER_K;
  const int tiles = nty * ntx;
  int nsplit = (nsm + tiles - 1) / tiles;
  nsplit = min(nsplit, niter);
  return max(nsplit, 1);
}

template <typename scalar_t, int J, bool fallback>
static void launch_mul_mat_q_v2(const void* vx, const void* vy, scalar_t* dst,
                                float* dst_f32, const int ncols_x,
                                const int nrows_x, const int ncols_dst,
                                const int nrows_dst, const int nsplit,
                                cudaStream_t stream) {
  constexpr int nbytes_shared = mmq_nbytes_shared(J);

  // J=128 needs more than the default 48 KiB. The opt-in is per device, so
  // track it per device rather than once per process.
  if (nbytes_shared > 48 * 1024) {
    static unsigned int smem_done = 0;
    int device = 0;
    cudaGetDevice(&device);
    if (device < 32 && !(smem_done & (1u << device))) {
      cudaFuncSetAttribute((const void*)mul_mat_q_v2<scalar_t, J, fallback>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           nbytes_shared);
      smem_done |= 1u << device;
    }
  }

  const int nty = (nrows_x + MMQ_I - 1) / MMQ_I;
  const int ntx = (ncols_dst + J - 1) / J;
  const dim3 block_nums(nty, ntx, nsplit);
  const dim3 block_dims(WARP_SIZE_V2, MMQ_NWARPS, 1);

  mul_mat_q_v2<scalar_t, J, fallback>
      <<<block_nums, block_dims, nbytes_shared, stream>>>(
          (const char*)vx, (const int*)vy, dst, nsplit > 1 ? dst_f32 : nullptr,
          nrows_x, ncols_dst, ncols_x / QK8_0, ncols_dst, nrows_dst,
          ncols_x / QK8_0, nsplit);
}

  #define VLLM_MMQ_V2_DISPATCH_J(J)                                          \
    launch_mul_mat_q_v2<scalar_t, J, fallback>(                              \
        vx, vy, dst, dst_f32, ncols_x, nrows_x, ncols_dst, nrows_dst,        \
        mmq_v2_nsplit((nrows_x + MMQ_I - 1) / MMQ_I, (ncols_dst + J - 1) / J, \
                      ncols_x, nsm),                                         \
        stream)

template <typename scalar_t, bool fallback>
static void mul_mat_q_v2_switch_J(const void* vx, const void* vy, scalar_t* dst,
                                  float* dst_f32, const int ncols_x,
                                  const int nrows_x, const int ncols_dst,
                                  const int nrows_dst, const int nsm,
                                  cudaStream_t stream) {
  if (ncols_dst <= 8) {
    VLLM_MMQ_V2_DISPATCH_J(8);
  } else if (ncols_dst <= 16) {
    VLLM_MMQ_V2_DISPATCH_J(16);
  } else if (ncols_dst <= 32) {
    VLLM_MMQ_V2_DISPATCH_J(32);
  } else if (ncols_dst <= 64) {
    VLLM_MMQ_V2_DISPATCH_J(64);
  } else {
    VLLM_MMQ_V2_DISPATCH_J(128);
  }
}

  #undef VLLM_MMQ_V2_DISPATCH_J

// x: [nrows_x, ncols_x] q8_0 weights, y: block_q8_1_mmq activations,
// dst: [ncols_dst, nrows_x] with row stride nrows_dst. dst_f32 is an
// ncols_dst*nrows_dst fp32 scratch buffer, only touched when split-K is used;
// it must be zeroed by the caller.
template <typename scalar_t>
static void ggml_mul_mat_q8_0_q8_1_v2_cuda(const void* vx, const void* vy,
                                           scalar_t* dst, float* dst_f32,
                                           const int ncols_x, const int nrows_x,
                                           const int ncols_dst,
                                           const int nrows_dst, const int nsm,
                                           cudaStream_t stream) {
  if (nrows_x % MMQ_I == 0) {
    mul_mat_q_v2_switch_J<scalar_t, false>(vx, vy, dst, dst_f32, ncols_x,
                                           nrows_x, ncols_dst, nrows_dst, nsm,
                                           stream);
  } else {
    mul_mat_q_v2_switch_J<scalar_t, true>(vx, vy, dst, dst_f32, ncols_x, nrows_x,
                                          ncols_dst, nrows_dst, nsm, stream);
  }
}

// True when the launcher will use split-K and therefore needs the fp32 scratch.
static inline bool mmq_v2_needs_scratch(int ncols_x, int nrows_x, int ncols_dst,
                                        int nsm) {
  const int J = ncols_dst <= 8    ? 8
                : ncols_dst <= 16 ? 16
                : ncols_dst <= 32 ? 32
                : ncols_dst <= 64 ? 64
                                  : 128;
  const int nty = (nrows_x + MMQ_I - 1) / MMQ_I;
  const int ntx = (ncols_dst + J - 1) / J;
  return mmq_v2_nsplit(nty, ntx, ncols_x, nsm) > 1;
}

template <typename scalar_t>
static void mmq_v2_finalize(const float* dst_f32, scalar_t* dst, int64_t n,
                            cudaStream_t stream) {
  const int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  mmq_v2_cast_f32<scalar_t><<<blocks, threads, 0, stream>>>(dst_f32, dst, n);
}

// Number of int32 the activation buffer needs, including the tail slack the
// tile copy may over-read.
static inline int64_t mmq_v2_y_ints(int64_t ncols_dst, int64_t ncols_x) {
  return (ncols_x / QK8_1_MMQ) * ncols_dst * MMQ_Y_INTS + 128 * MMQ_Y_INTS;
}

// The port only covers K that is a whole number of 256-element iterations;
// load_tiles_q8_0 reads up to 8 q8_0 blocks past kbx0.
static inline bool mmq_v2_supported(int64_t ncols_x, int64_t nrows_x) {
  return ncols_x % MMQ_ITER_K == 0 && nrows_x > 0;
}

}  // namespace vllm_mmq_v2

#endif  // USE_ROCM
