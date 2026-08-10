#pragma once

#ifndef USE_ROCM

// Grouped tensor-core MoE GEMM for MXFP4 experts on sm80+.
//
// The dp4a grouped tile (moe.cuh moe_mxfp4) is MOE_X=4 columns wide on CUDA,
// so prefill re-streams every expert's full weight tile per 4 routed rows;
// measured e2e that leaves the MXFP4 profiles at ~25% of the hybrid quant's
// throughput at c8 when activated-byte parity predicts ~59%. This kernel
// gives the wide path real weight reuse: 128 expert rows x 64 routed columns
// per tile, K in 256-value iterations, int8 mma.sync via the mmq_v2
// machinery.
//
// The trick that keeps it small: e2m1 decode goes through the same 2x-value
// int8 table as the fused decode path, and the decoded tile is written in
// exactly the shared-memory layout of the dense Q8_0 MMQ v2 kernel (64 quant
// ints + 8 fp32 block scales per row, stride 76). vec_dot_q8_0_q8_1_mma then
// runs unmodified; only the tile loader (MXFP4 -> int8 smem), the activation
// gather (per sorted_token_ids, like moe.cuh moe_q), and the write-back
// (scatter to routed columns, skip padding slots) are MoE/MXFP4 specific.
// The 0.5 factor of the 2x table is folded into the e8m0 block scale.
//
// Weights are read in either layout: raw GGUF AoS (17-byte blocks: scale
// byte + 16 code bytes) or the byte-neutral SoA split (scales | 16B-aligned
// codes) produced by ggml_dsv4_repack_mxfp4.
//
// Column tiles follow moe_align_block_size metadata with block size 64
// (ggml_moe_get_block_size(39) must agree). Padding slots hold
// sorted_token_ids >= ncols_dst: their gather is zeroed and the write-back
// drops them; the mma still computes them, which at DSV4 prefill widths is a
// few percent of the tile.

#include "../../libtorch_stable/quantization/gguf/mmq_v2/mmq_v2.cuh"
#include "dsv4_mxfp4_moe_ampere.cuh"

namespace slimserve::dsv4_ampere {

constexpr int MXMMQ_J = 64;    // routed columns per tile (= align block size)
constexpr int MXMMQ_I = 128;   // expert rows per tile
constexpr int MXMMQ_NWARPS = 8;
constexpr int MXMMQ_NTHREADS = MXMMQ_NWARPS * 32;
constexpr int MXMMQ_ITER_K = 256;                 // values per K iteration
constexpr int MXMMQ_BLOCKS_PER_ITER = MXMMQ_ITER_K / 32;  // 8 MXFP4 blocks
// Must equal MMQ_SRAM_STRIDE_Q8_0: 64 quant ints + 8 scale floats + 4 pad.
constexpr int MXMMQ_X_STRIDE = 2 * 32 + 8 + 4;
// Per-column ints in the y tile: 4 scale floats + 32 quant ints (one span of
// 128 activation values). Must equal MMQ_TILE_Y_K.
constexpr int MXMMQ_Y_STRIDE = 32 + 4;

static_assert(MXMMQ_X_STRIDE == vllm_mmq_v2::MMQ_SRAM_STRIDE_Q8_0,
              "x tile layout must match the dense v2 vec dot");
static_assert(MXMMQ_Y_STRIDE == vllm_mmq_v2::MMQ_TILE_Y_K,
              "y tile layout must match the dense v2 vec dot");

// Decode one iteration's weight tile into the q8_0-mmq shared layout.
// 128 rows x 8 blocks = 1024 (row, block) pairs; 256 threads, 4 rounds.
template <bool REPACKED, bool ROW_CLAMP>
__device__ __forceinline__ void mxmmq_load_x(
    const char* __restrict__ expert_base, int* __restrict__ x_tile,
    const int64_t nblocks, const int blocks_per_row, const int kb0,
    const int row_x0, const int i_max) {
  float* x_df = reinterpret_cast<float*>(x_tile + 2 * 32);
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int r = 0; r < MXMMQ_I * MXMMQ_BLOCKS_PER_ITER / MXMMQ_NTHREADS; ++r) {
    const int pair = tid + r * MXMMQ_NTHREADS;
    const int i = pair / MXMMQ_BLOCKS_PER_ITER;
    const int kb = pair % MXMMQ_BLOCKS_PER_ITER;
    int row = row_x0 + i;
    if (ROW_CLAMP) row = min(row, i_max);
    uint4 codes;
    uint8_t scale;
    mxfp4_load_block<REPACKED>(expert_base, nblocks,
                               int64_t(row) * blocks_per_row + kb0 + kb, codes,
                               scale);
    int lo, hi;
    int* qs = x_tile + i * MXMMQ_X_STRIDE + kb * 8;
    mxfp4_expand8((int)codes.x, lo, hi);
    qs[0] = lo;
    qs[4] = hi;
    mxfp4_expand8((int)codes.y, lo, hi);
    qs[1] = lo;
    qs[5] = hi;
    mxfp4_expand8((int)codes.z, lo, hi);
    qs[2] = lo;
    qs[6] = hi;
    mxfp4_expand8((int)codes.w, lo, hi);
    qs[3] = lo;
    qs[7] = hi;
    x_df[i * MXMMQ_X_STRIDE + kb] = mxfp4_scale_to_fp32(scale) * 0.5f;
  }
}

// Gather one 128-value activation span (4 block_q8_1 blocks) for each of the
// tile's 64 columns into the block_q8_1_mmq-shaped y tile.
__device__ __forceinline__ void mxmmq_load_y(
    const block_q8_1* __restrict__ y, int* __restrict__ tile_y,
    const int* __restrict__ token_offs, const int q8b0,
    const int blocks_per_col_y, const int ncols_dst, const int top_k) {
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int l0 = 0; l0 < MXMMQ_J * MXMMQ_Y_STRIDE; l0 += MXMMQ_NTHREADS) {
    const int l = l0 + tid;
    if (l >= MXMMQ_J * MXMMQ_Y_STRIDE) break;
    const int c = l / MXMMQ_Y_STRIDE;
    const int m = l % MXMMQ_Y_STRIDE;
    const int id = token_offs[c];
    if (id >= ncols_dst) {
      // Padding slot: zero the scales so discarded columns stay finite.
      if (m < 4) tile_y[l] = 0;
      continue;
    }
    const block_q8_1* col = y + int64_t(id / top_k) * blocks_per_col_y + q8b0;
    if (m < 4) {
      const float d = __low2float(col[m].ds);
      tile_y[l] = __float_as_int(d);
    } else {
      const int qi = m - 4;
      tile_y[l] =
          reinterpret_cast<const int*>(col[qi / 8].qs)[qi % 8];
    }
  }
}

template <typename scalar_t, bool REPACKED, bool ROW_CLAMP>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_mxfp4_mmq_v2(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded,
    const int64_t exp_stride_bytes, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int ncols_pad) {
  using namespace vllm_mmq_v2;

  const int blocks_per_row = ncols_x / 32;
  const int blocks_per_col_y = nrows_y / 32;
  const int ncols_dst = ncols_y * top_k;
  const int row_x0 = blockIdx.x * MXMMQ_I;
  const int col0 = blockIdx.y * MXMMQ_J;
  const int i_max = nrows_x - 1;
  const int tid = threadIdx.y * 32 + threadIdx.x;

  __shared__ int token_offs[MXMMQ_J];
  // ldmatrix requires 16-byte-aligned addresses; the row stride (76 ints =
  // 304 bytes) and the k offsets (multiples of 32 bytes) preserve base
  // alignment, so aligning the arrays is sufficient.
  __shared__ __align__(16) int tile_y[MXMMQ_J * MXMMQ_Y_STRIDE];
  __shared__ __align__(16) int tile_x[MXMMQ_I * MXMMQ_X_STRIDE];

  if (tid < MXMMQ_J) {
    // The sorted array length need not be a multiple of the tile width;
    // clamped reads land on padding slots which the write-back drops.
    token_offs[tid] = sorted_token_ids[min(col0 + tid, ncols_pad - 1)];
  }
  __syncthreads();

  const int exp_idx = expert_ids[blockIdx.y];
  if (exp_idx < 0) {
    // Callers no longer pre-fill dst; zero this tile's real columns.
    for (int j = tid; j < MXMMQ_J; j += MXMMQ_NTHREADS) {
      const int col_dst = token_offs[j];
      if (col_dst >= ncols_dst) continue;
      for (int i = 0; i < MXMMQ_I; ++i) {
        const int row = row_x0 + i;
        if (row < nrows_dst)
          dst[int64_t(col_dst) * nrows_dst + row] = scalar_t(0);
      }
    }
    return;
  }
  if (col0 >= num_tokens_post_padded[0]) return;

  const char* expert_base =
      reinterpret_cast<const char*>(vx) + int64_t(exp_idx) * exp_stride_bytes;
  const int64_t nblocks = exp_stride_bytes / 17;
  const block_q8_1* y = reinterpret_cast<const block_q8_1*>(vy);

  float sum[MXMMQ_J * MXMMQ_I / (MXMMQ_NTHREADS)] = {0.0f};

  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MXMMQ_BLOCKS_PER_ITER) {
    mxmmq_load_x<REPACKED, ROW_CLAMP>(expert_base, tile_x, nblocks,
                                      blocks_per_row, kb0, row_x0, i_max);
    mxmmq_load_y(y, tile_y, token_offs, kb0, blocks_per_col_y, ncols_dst,
                 top_k);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<MXMMQ_J>(tile_x, tile_y, sum, 0);
    __syncthreads();
    mxmmq_load_y(y, tile_y, token_offs, kb0 + 4, blocks_per_col_y, ncols_dst,
                 top_k);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<MXMMQ_J>(tile_x, tile_y, sum, 32);
    __syncthreads();
  }

  // Write back with the dense v2 fragment mapping, scattered to routed
  // columns; padding slots and row overhang are dropped.
  typedef tile<16, 8> tile_C;
  constexpr int rows_per_warp = mmq_rows_per_warp(MXMMQ_J);
  constexpr int ntx = rows_per_warp / tile_C::I;
  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

#pragma unroll
  for (int j0 = 0; j0 < MXMMQ_J; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j =
            j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        const int col_dst = token_offs[j];
        if (col_dst >= ncols_dst) continue;
        const int row = row_x0 + i0 + n * tile_C::I + tile_C::get_i(l);
        if (ROW_CLAMP && row >= nrows_dst) continue;
        dst[int64_t(col_dst) * nrows_dst + row] =
            scalar_t(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

// True when this kernel can serve the shape: K a whole number of 256-value
// iterations (the loader has no partial-iteration path).
inline bool moe_mxfp4_mmq_v2_supported(int64_t ncols_x) {
  return ncols_x % MXMMQ_ITER_K == 0;
}

template <typename scalar_t>
inline void launch_moe_mxfp4_mmq_v2(
    const void* quant_x, const void* weights, scalar_t* dst,
    const int* sorted_token_ids, const int* expert_ids,
    const int* num_tokens_post_padded, const int64_t exp_stride_bytes,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k, const int ncols_pad,
    const bool repacked, cudaStream_t stream) {
  const dim3 grid((nrows_x + MXMMQ_I - 1) / MXMMQ_I,
                  (ncols_pad + MXMMQ_J - 1) / MXMMQ_J);
  const dim3 block(32, MXMMQ_NWARPS);
  const bool row_clamp = (nrows_x % MXMMQ_I) != 0;
#define LAUNCH_MXMMQ(REPACKED, ROW_CLAMP)                                    \
  moe_mxfp4_mmq_v2<scalar_t, REPACKED, ROW_CLAMP>                            \
      <<<grid, block, 0, stream>>>(                                          \
          weights, quant_x, dst, sorted_token_ids, expert_ids,               \
          num_tokens_post_padded, exp_stride_bytes, ncols_x, nrows_x,        \
          ncols_y, nrows_y, nrows_dst, top_k, ncols_pad)
  if (repacked) {
    if (row_clamp) LAUNCH_MXMMQ(true, true);
    else LAUNCH_MXMMQ(true, false);
  } else {
    if (row_clamp) LAUNCH_MXMMQ(false, true);
    else LAUNCH_MXMMQ(false, false);
  }
#undef LAUNCH_MXMMQ
}

}  // namespace slimserve::dsv4_ampere

#endif  // USE_ROCM
