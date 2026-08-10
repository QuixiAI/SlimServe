#pragma once

#ifndef USE_ROCM

// Segmented (permutation-based) MXFP4 MoE pipeline for sm80+, replacing the
// moe_align padded-metadata route for the DSV4 wide path.
//
// Design ported from the QuixiCore-XPU grouped_qgemm review (2026-08-10):
// routes are grouped per expert by a device-side count/prefix/scatter chain
// (no host sync, no sorted+padded arrays), and the GEMM launches a STATIC
// worst-case grid of ceil(M/J)+E column tiles; every block rebuilds a small
// per-expert prefix table from rows_per_expert in shared memory and maps its
// linear tile index to (expert, local column range), exiting early past the
// real tile count. The grid depends only on (M, E), so the whole pipeline is
// CUDA-graph-capture-safe with varying routing.
//
// The W1 kernel additionally fuses the between-GEMM elementwise work (the
// XPU glu_quant idea): its epilogue spills the fp32 accumulator tile to the
// (dead) weight-staging smem region, applies SwiGLU + the route weight, and emits
// the Q8_1 mid activation directly -- the [routes, 2I] half intermediate,
// the separate activation pass, and the separate quantize pass all
// disappear. Pairing gate row r with up row I+r inside one tile is arranged
// by the row map: tile row i < 64 -> gate row g0+i, else up row I+g0+i-64.
//
// W2 consumes the Q8_1 mid (route-indexed, route weight already folded, the
// decode-path convention) through the same segmented gather and writes
// per-route output rows; a small deterministic reduce sums each token's
// routes in fixed j order (no atomics -- the XPU review flagged relaxed
// atomic accumulation as run-to-run nondeterministic).
//
// The mma stage reuses vec_dot_q8_0_q8_1_mma via the same smem layout as
// dsv4_mxfp4_mmq_ampere.cuh. J is a template axis: 64 for prefill widths,
// 16 for small routed counts where a 64-wide tile is mostly masked slack
// (per-expert tile tails are masked, not padded slots, but the mma quantum
// is still J columns -- J=16 quarters that waste).

#include "dsv4_mxfp4_mmq_ampere.cuh"

namespace slimserve::dsv4_ampere {

constexpr int SEG_MAX_EXPERTS = 256;

// ------------------------------------------------------------ perm metadata
// rows_per_expert/cursors must be zeroed before the histogram (launcher does
// a cudaMemsetAsync). Invalid expert ids (<0 or >=experts) are skipped, so
// perm slots cover only valid routes; the reduce re-checks validity.
static __global__ void seg_histogram(const int* __restrict__ topk_ids,
                                     int* __restrict__ rows_per_expert,
                                     const int routes, const int experts) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= routes) return;
  const int e = topk_ids[r];
  if (e < 0 || e >= experts) return;
  atomicAdd(&rows_per_expert[e], 1);
}

// Exclusive prefix into cursors plus the row/tile prefix tables the GEMM
// blocks consume (built once here instead of per-block; single block,
// serial, E <= 256).
static __global__ void seg_prefix(const int* __restrict__ rows_per_expert,
                                  int* __restrict__ cursors,
                                  int* __restrict__ rowseg,
                                  int* __restrict__ tseg, const int experts,
                                  const int J) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int rows = 0, tiles = 0;
  for (int e = 0; e < experts; ++e) {
    cursors[e] = rows;
    rowseg[e] = rows;
    tseg[e] = tiles;
    const int rpe = rows_per_expert[e];
    rows += rpe;
    tiles += (rpe + J - 1) / J;
  }
  rowseg[experts] = rows;
  tseg[experts] = tiles;
}

static __global__ void seg_scatter(const int* __restrict__ topk_ids,
                                   int* __restrict__ cursors,
                                   int* __restrict__ perm_ids,
                                   const int routes, const int experts) {
  const int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= routes) return;
  const int e = topk_ids[r];
  if (e < 0 || e >= experts) return;
  const int slot = atomicAdd(&cursors[e], 1);
  perm_ids[slot] = r;
}

// Cooperatively load the precomputed row/tile prefix tables into scratch
// shared memory (the caller lends its weight-staging tile, dead until the K
// loop) and map this block's linear column tile to (expert, slot range).
// Returns false when the tile is past the real tile count (grid slack).
// scratch must hold 2*(experts+1) ints; the caller must __syncthreads()
// after this returns true, before overwriting the scratch.
template <int J>
__device__ __forceinline__ bool seg_locate(
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const int experts, const int tile, int* __restrict__ scratch, int& expert,
    int& slot0, int& ncols) {
  int* rowseg = scratch;
  int* tseg = scratch + experts + 1;
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int e = tid; e <= experts; e += MXMMQ_NTHREADS) {
    rowseg[e] = g_rowseg[e];
    tseg[e] = g_tseg[e];
  }
  __syncthreads();
  if (tile >= tseg[experts]) return false;
  // Warp-uniform linear walk (E <= 256; every thread does the same walk).
  int e = 0;
  while (tseg[e + 1] <= tile) ++e;
  const int local_tile = tile - tseg[e];
  const int rows_e = rowseg[e + 1] - rowseg[e];
  expert = e;
  slot0 = rowseg[e] + local_tile * J;
  ncols = min(J, rows_e - local_tile * J);
  return true;
}

// Gather one 128-value activation span for the tile's J columns, reading
// token activations through perm_ids (route -> token). Pad columns zeroed.
template <int J>
__device__ __forceinline__ void seg_load_y_tokens(
    const block_q8_1* __restrict__ y, int* __restrict__ tile_y,
    const int* __restrict__ token_routes, const int ncols, const int q8b0,
    const int blocks_per_col_y, const int top_k) {
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int l = tid; l < J * MXMMQ_Y_STRIDE; l += MXMMQ_NTHREADS) {
    const int c = l / MXMMQ_Y_STRIDE;
    const int m = l % MXMMQ_Y_STRIDE;
    if (c >= ncols) {
      if (m < 4) tile_y[l] = 0;
      continue;
    }
    const block_q8_1* col =
        y + int64_t(token_routes[c] / top_k) * blocks_per_col_y + q8b0;
    if (m < 4) {
      tile_y[l] = __float_as_int(__low2float(col[m].ds));
    } else {
      const int qi = m - 4;
      tile_y[l] = reinterpret_cast<const int*>(col[qi / 8].qs)[qi % 8];
    }
  }
}

// Same, but the y rows are the Q8_1 mid activation indexed directly by route.
template <int J>
__device__ __forceinline__ void seg_load_y_mid(
    const block_q8_1* __restrict__ mid, int* __restrict__ tile_y,
    const int* __restrict__ token_routes, const int ncols, const int q8b0,
    const int blocks_per_col_y) {
  const int tid = threadIdx.y * 32 + threadIdx.x;
  for (int l = tid; l < J * MXMMQ_Y_STRIDE; l += MXMMQ_NTHREADS) {
    const int c = l / MXMMQ_Y_STRIDE;
    const int m = l % MXMMQ_Y_STRIDE;
    if (c >= ncols) {
      if (m < 4) tile_y[l] = 0;
      continue;
    }
    const block_q8_1* col =
        mid + int64_t(token_routes[c]) * blocks_per_col_y + q8b0;
    if (m < 4) {
      tile_y[l] = __float_as_int(__low2float(col[m].ds));
    } else {
      const int qi = m - 4;
      tile_y[l] = reinterpret_cast<const int*>(col[qi / 8].qs)[qi % 8];
    }
  }
}

// W1 weight loader over the paired row map: tile row i < 64 -> gate row
// g0 + i, else up row I + g0 + (i - 64). g0 + 63 < I is guaranteed by
// I % 64 == 0, so no clamp is needed.
template <bool REPACKED>
__device__ __forceinline__ void seg_load_x_w1(
    const char* __restrict__ expert_base, int* __restrict__ x_tile,
    const int64_t nblocks, const int blocks_per_row, const int kb0,
    const int g0, const int intermediate) {
  float* x_df = reinterpret_cast<float*>(x_tile + 2 * 32);
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int r = 0; r < MXMMQ_I * MXMMQ_BLOCKS_PER_ITER / MXMMQ_NTHREADS; ++r) {
    const int pair = tid + r * MXMMQ_NTHREADS;
    const int i = pair / MXMMQ_BLOCKS_PER_ITER;
    const int kb = pair % MXMMQ_BLOCKS_PER_ITER;
    const int row = (i < 64) ? (g0 + i) : (intermediate + g0 + (i - 64));
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

// ------------------------------------------------------------------- W1
// Grid: (intermediate/64 row-pair tiles, ceil(M/J)+E column tiles).
// Epilogue: spill C to smem, SwiGLU + route weight, emit Q8_1 mid blocks
// (mid[route, I] in 32-value blocks along I; this tile owns blocks
// rt*2 and rt*2+1 of each column's route).
template <int J, bool REPACKED>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_mxfp4_seg_w1(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ mid, const int* __restrict__ perm_ids,
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const float* __restrict__ route_weights, const int64_t exp_stride_bytes,
    const int hidden, const int input_token_blocks, const int intermediate,
    const int experts, const int top_k, const float swiglu_limit) {
  using namespace vllm_mmq_v2;

  __shared__ int token_routes[J];
  __shared__ __align__(16) int tile_y[J * MXMMQ_Y_STRIDE];
  __shared__ __align__(16) int tile_x[MXMMQ_I * MXMMQ_X_STRIDE];
  static_assert(2 * (SEG_MAX_EXPERTS + 1) <= MXMMQ_I * MXMMQ_X_STRIDE,
                "seg tables borrow the weight-staging tile");

  int expert, slot0, ncols;
  if (!seg_locate<J>(g_rowseg, g_tseg, experts, blockIdx.y, tile_x, expert,
                     slot0, ncols))
    return;

  const int tid = threadIdx.y * 32 + threadIdx.x;
  if (tid < J) {
    token_routes[tid] = tid < ncols ? perm_ids[slot0 + tid] : -1;
  }
  __syncthreads();

  const char* expert_base =
      reinterpret_cast<const char*>(weights) +
      int64_t(expert) * exp_stride_bytes;
  const int64_t nblocks = exp_stride_bytes / 17;
  const int blocks_per_row = hidden / 32;
  const int g0 = blockIdx.x * 64;

  float sum[J * MXMMQ_I / MXMMQ_NTHREADS] = {0.0f};

  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MXMMQ_BLOCKS_PER_ITER) {
    seg_load_x_w1<REPACKED>(expert_base, tile_x, nblocks, blocks_per_row, kb0,
                            g0, intermediate);
    seg_load_y_tokens<J>(input, tile_y, token_routes, ncols, kb0,
                         input_token_blocks, top_k);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y, sum, 0);
    __syncthreads();
    seg_load_y_tokens<J>(input, tile_y, token_routes, ncols, kb0 + 4,
                         input_token_blocks, top_k);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y, sum, 32);
    __syncthreads();
  }

  // Spill C into the dead weight-staging smem: C[i][c] at tile_x[i*J + c],
  // fp32 (128*J*4 bytes <= the tile_x region for J <= 76).
  static_assert(J <= MXMMQ_X_STRIDE, "C spill must fit in tile_x");
  float* c_spill = reinterpret_cast<float*>(tile_x);
  {
    typedef tile<16, 8> tile_C;
    constexpr int rows_per_warp = mmq_rows_per_warp(J);
    constexpr int ntx = rows_per_warp / tile_C::I;
    const int i0 = (threadIdx.y / ntx) * rows_per_warp;
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
      for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          const int j =
              j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
          const int i = i0 + n * tile_C::I + tile_C::get_i(l);
          c_spill[i * J + j] = sum[(j0 / tile_C::J + n) * tile_C::ne + l];
        }
      }
    }
  }
  __syncthreads();

  // SwiGLU + route weight + Q8_1 emission. Warp tasks: (column, block b) with
  // b in {0,1} covering mid rows [b*32, b*32+32) of this tile's 64 pairs.
  const int mid_blocks_per_route = intermediate / 32;
  for (int task = threadIdx.y; task < ncols * 2; task += MXMMQ_NWARPS) {
    const int c = task >> 1;
    const int b = task & 1;
    const int m_local = b * 32 + threadIdx.x;
    float gate = c_spill[m_local * J + c];
    float up = c_spill[(64 + m_local) * J + c];
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    const int route = token_routes[c];
    float value =
        (gate / (1.0f + expf(-gate))) * up * route_weights[route];
    if (!isfinite(value)) value = 0.0f;

    float amax = fabsf(value);
    float vsum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      vsum += __shfl_xor_sync(0xffffffffu, vsum, mask);
    }
    const float scale = amax / 127.0f;
    const int8_t quant =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    block_q8_1* out = mid + int64_t(route) * mid_blocks_per_route +
                      (int64_t(blockIdx.x) * 2 + b);
    out->qs[threadIdx.x] = quant;
    if (threadIdx.x == 0) out->ds = __floats2half2_rn(scale, vsum);
  }
}

// ------------------------------------------------------------------- W2
// Standard segmented tile over K = intermediate, y = Q8_1 mid (route rows),
// output per-route rows [routes, out_row] in the activation dtype.
template <typename scalar_t, int J, bool REPACKED>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_mxfp4_seg_w2(
    const void* __restrict__ weights, const block_q8_1* __restrict__ mid,
    scalar_t* __restrict__ w2out, const int* __restrict__ perm_ids,
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const int64_t exp_stride_bytes, const int intermediate,
    const int out_row, const int experts) {
  using namespace vllm_mmq_v2;

  __shared__ int token_routes[J];
  __shared__ __align__(16) int tile_y[J * MXMMQ_Y_STRIDE];
  __shared__ __align__(16) int tile_x[MXMMQ_I * MXMMQ_X_STRIDE];
  static_assert(2 * (SEG_MAX_EXPERTS + 1) <= MXMMQ_I * MXMMQ_X_STRIDE,
                "seg tables borrow the weight-staging tile");

  int expert, slot0, ncols;
  if (!seg_locate<J>(g_rowseg, g_tseg, experts, blockIdx.y, tile_x, expert,
                     slot0, ncols))
    return;

  const int tid = threadIdx.y * 32 + threadIdx.x;
  if (tid < J) {
    token_routes[tid] = tid < ncols ? perm_ids[slot0 + tid] : -1;
  }
  __syncthreads();

  const char* expert_base =
      reinterpret_cast<const char*>(weights) +
      int64_t(expert) * exp_stride_bytes;
  const int64_t nblocks = exp_stride_bytes / 17;
  const int blocks_per_row = intermediate / 32;
  const int row_x0 = blockIdx.x * MXMMQ_I;

  float sum[J * MXMMQ_I / MXMMQ_NTHREADS] = {0.0f};

  for (int kb0 = 0; kb0 < blocks_per_row; kb0 += MXMMQ_BLOCKS_PER_ITER) {
    mxmmq_load_x<REPACKED, false>(expert_base, tile_x, nblocks,
                                  blocks_per_row, kb0, row_x0, out_row - 1);
    seg_load_y_mid<J>(mid, tile_y, token_routes, ncols, kb0, blocks_per_row);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y, sum, 0);
    __syncthreads();
    seg_load_y_mid<J>(mid, tile_y, token_routes, ncols, kb0 + 4,
                      blocks_per_row);
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y, sum, 32);
    __syncthreads();
  }

  typedef tile<16, 8> tile_C;
  constexpr int rows_per_warp = mmq_rows_per_warp(J);
  constexpr int ntx = rows_per_warp / tile_C::I;
  const int i0 = (threadIdx.y / ntx) * rows_per_warp;
#pragma unroll
  for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j =
            j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);
        if (j >= ncols) continue;
        const int row = row_x0 + i0 + n * tile_C::I + tile_C::get_i(l);
        w2out[int64_t(token_routes[j]) * out_row + row] =
            scalar_t(sum[(j0 / tile_C::J + n) * tile_C::ne + l]);
      }
    }
  }
}

// Deterministic unpermute-reduce: out[t, h] = sum_j w2out[t*top_k+j, h] over
// valid routes, fixed j order. Route weights were folded into the mid.
template <typename scalar_t>
static __global__ void seg_reduce(const scalar_t* __restrict__ w2out,
                                  const int* __restrict__ topk_ids,
                                  scalar_t* __restrict__ out,
                                  const int out_row, const int top_k,
                                  const int experts) {
  const int t = blockIdx.x;
  for (int h = threadIdx.x; h < out_row; h += blockDim.x) {
    float acc = 0.0f;
    for (int j = 0; j < top_k; ++j) {
      const int route = t * top_k + j;
      const int e = topk_ids[route];
      if (e < 0 || e >= experts) continue;
      acc += float(w2out[int64_t(route) * out_row + h]);
    }
    out[int64_t(t) * out_row + h] = scalar_t(acc);
  }
}

// ------------------------------------------------------------------ launch
inline int seg_col_tiles(const int routes, const int experts, const int J) {
  return (routes + J - 1) / J + experts;
}

// meta scratch layout (ints): rows_per_expert[E] | cursors[E] |
// rowseg[E+1] | tseg[E+1] | perm_ids[routes].
inline int64_t seg_meta_ints(const int experts, const int routes) {
  return int64_t(2) * experts + 2 * (experts + 1) + routes;
}

template <typename scalar_t>
inline void launch_moe_mxfp4_seg(
    const void* quant_x, const void* w1, const void* w2, void* mid,
    scalar_t* w2out, scalar_t* out, const int* topk_ids,
    const float* route_weights, int* meta, const int64_t w1_stride_bytes,
    const int64_t w2_stride_bytes, const int hidden,
    const int input_token_blocks, const int intermediate, const int out_row,
    const int tokens, const int top_k, const int experts,
    const float swiglu_limit, const bool use_j16, const bool w1_repacked,
    const bool w2_repacked, cudaStream_t stream) {
  const int routes = tokens * top_k;
  int* rows_per_expert = meta;
  int* cursors = meta + experts;
  int* rowseg = meta + 2 * experts;
  int* tseg = rowseg + experts + 1;
  int* perm_ids = tseg + experts + 1;
  cudaMemsetAsync(rows_per_expert, 0, experts * sizeof(int), stream);
  {
    const int threads = 256;
    const int blocks = (routes + threads - 1) / threads;
    seg_histogram<<<blocks, threads, 0, stream>>>(topk_ids, rows_per_expert,
                                                  routes, experts);
    seg_prefix<<<1, 1, 0, stream>>>(rows_per_expert, cursors, rowseg, tseg,
                                    experts, use_j16 ? 16 : 64);
    seg_scatter<<<blocks, threads, 0, stream>>>(topk_ids, cursors, perm_ids,
                                                routes, experts);
  }
  const auto in = static_cast<const block_q8_1*>(quant_x);
  const auto mid_blocks = static_cast<block_q8_1*>(mid);
  const dim3 block(32, MXMMQ_NWARPS);
#define LAUNCH_SEG(J, R1, R2)                                                \
  do {                                                                       \
    const dim3 g1(intermediate / 64, seg_col_tiles(routes, experts, J));     \
    moe_mxfp4_seg_w1<J, R1><<<g1, block, 0, stream>>>(                       \
        w1, in, mid_blocks, perm_ids, rowseg, tseg, route_weights,           \
        w1_stride_bytes, hidden, input_token_blocks, intermediate, experts,  \
        top_k, swiglu_limit);                                                \
    const dim3 g2((out_row + MXMMQ_I - 1) / MXMMQ_I,                         \
                  seg_col_tiles(routes, experts, J));                        \
    moe_mxfp4_seg_w2<scalar_t, J, R2><<<g2, block, 0, stream>>>(             \
        w2, mid_blocks, w2out, perm_ids, rowseg, tseg, w2_stride_bytes,      \
        intermediate, out_row, experts);                                     \
  } while (0)
#define LAUNCH_SEG_J(J)                                                      \
  do {                                                                       \
    if (w1_repacked && w2_repacked) LAUNCH_SEG(J, true, true);               \
    else if (w1_repacked) LAUNCH_SEG(J, true, false);                        \
    else if (w2_repacked) LAUNCH_SEG(J, false, true);                        \
    else LAUNCH_SEG(J, false, false);                                        \
  } while (0)
  if (use_j16) LAUNCH_SEG_J(16);
  else LAUNCH_SEG_J(64);
#undef LAUNCH_SEG_J
#undef LAUNCH_SEG
  seg_reduce<scalar_t><<<tokens, 256, 0, stream>>>(w2out, topk_ids, out,
                                                   out_row, top_k, experts);
}

}  // namespace slimserve::dsv4_ampere

#endif  // USE_ROCM
