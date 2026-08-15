#pragma once

#ifndef USE_ROCM

// Segmented tensor-core MoE pipeline for the DSV4 hybrid quant pair
// (IQ2_XXS gate/up, Q2_K down) on sm80+ -- the 55-of-61-layer path of the
// Q4K-tail artifact. Same architecture as dsv4_mxfp4_seg_ampere.cuh
// (device-side route grouping, static worst-case grids, fused
// SwiGLU+route-weight+Q8_1 W1 epilogue, deterministic reduce); only the
// weight decode differs:
//
//  * W1 reads the load-time PAIRED IQ2_XXS repack (half2 (gate_d, up_d)
//    plane + uint4 {gate.x, gate.y, up.x, up.y} code plane, one pair per
//    (mid_row, superblock-group)). Codes expand through the iq2xxs grid
//    with in-register sign unpack -- the fused decode path's math -- into
//    the dense Q8_0 MMQ v2 tile layout (int8 codes + per-32 fp32 scale
//    with the 0.125 grid factor folded in), so vec_dot_q8_0_q8_1_mma runs
//    unmodified.
//  * W2 reads the three-plane Q2_K repack (packed 2-bit words | scale
//    bytes | half2 (d, dmin)). Q2_K carries per-16 scales AND mins, so its
//    vec dot runs on m16n8k16 fragments with a per-16 half2 (d*sc,
//    dmin*m) plane, and the min term is synthesized with a ones-matrix
//    mma per 16-value group (the upstream llama.cpp Turing Q2_K recipe,
//    with the ones-mma replacing their special d2s6 y layout so plain
//    block_q8_1 y tiles keep working).

#include "dsv4_mxfp4_seg_ampere.cuh"

namespace slimserve::dsv4_ampere {

// Q2_K x-tile stride: 64 quant ints + 16 half2 (16 ints) + 4 pad; %8==4
// keeps ldmatrix-compatible alignment parity with the other tiles.
constexpr int Q2K_X_STRIDE = 64 + 16 + 4;

// Spread one byte holding four 2-bit values (bits 0,2,4,6) into four int8
// lanes.
__device__ __forceinline__ int q2k_spread_byte(uint32_t b) {
  return int((b & 0x3u) | ((b & 0xCu) << 6) | ((b & 0x30u) << 12) |
             ((b & 0xC0u) << 18));
}

// ------------------------------------------------------------- W1 (IQ2_XXS)
// Per iteration each row consumes one 256-value superblock: 8 groups of 32.
// Paired repack: pair_block = mid_row * blocks_per_row + superblock.
__device__ __forceinline__ void seg_load_x_w1_iq2(
    const char* __restrict__ expert_base, int* __restrict__ x_tile,
    const int blocks_per_row, const int superblock, const int g0,
    const int intermediate) {
  float* x_df = reinterpret_cast<float*>(x_tile + 2 * 32);
  const int paired_blocks = intermediate * blocks_per_row;
  const half2* aligned_d = reinterpret_cast<const half2*>(expert_base);
  const uint4* aligned_q = reinterpret_cast<const uint4*>(
      expert_base + int64_t(paired_blocks) * sizeof(half2));
  const int tid = threadIdx.y * 32 + threadIdx.x;
#pragma unroll
  for (int r = 0; r < MXMMQ_I * 8 / MXMMQ_NTHREADS; ++r) {
    const int pair = tid + r * MXMMQ_NTHREADS;
    const int i = pair / 8;
    const int g = pair % 8;
    const bool is_up = i >= 64;
    const int mid_row = g0 + (is_up ? i - 64 : i);
    const int64_t pair_block = int64_t(mid_row) * blocks_per_row + superblock;
    const uint4 code = aligned_q[pair_block * 8 + g];
    const half2 d2 = aligned_d[pair_block];
    uint32_t code_x = is_up ? code.z : code.x;
    uint32_t code_y = is_up ? code.w : code.y;
    const float d = __half2float(is_up ? __high2half(d2) : __low2half(d2));

    int* qs = x_tile + i * MXMMQ_X_STRIDE + g * 8;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const uint8_t grid_index = uint8_t(code_x >> (8 * part));
      const uint2 grid = *reinterpret_cast<const uint2*>(
          &iq2xxs_grid[grid_index]);
      const uint32_t signs = iq2_xxs_unpack_signs(uint8_t(code_y));
      const uint32_t signs0 = __vcmpne4(signs & 0x08040201u, 0);
      const uint32_t signs1 = __vcmpne4(signs & 0x80402010u, 0);
      qs[2 * part] = int(__vsub4(grid.x ^ signs0, signs0));
      qs[2 * part + 1] = int(__vsub4(grid.y ^ signs1, signs1));
      code_y >>= 7;
    }
    x_df[i * MXMMQ_X_STRIDE + g] = d * float(2u * code_y + 1u) * 0.125f;
  }
}

template <int J>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_iq2_seg_w1(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ mid, const int* __restrict__ perm_ids,
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const float* __restrict__ route_weights, const int64_t exp_stride_bytes,
    const int hidden, const int input_token_blocks, const int intermediate,
    const int experts, const int top_k, const float swiglu_limit) {
  using namespace vllm_mmq_v2;

  extern __shared__ int iq2_w1_smem[];
  int* token_routes = iq2_w1_smem;
  int* tile_y0 = token_routes + J;
  int* tile_y1 = tile_y0 + J * MXMMQ_Y_STRIDE;
  int* tile_x = tile_y1 + J * MXMMQ_Y_STRIDE;

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
  const int blocks_per_row = hidden / 256;
  const int g0 = blockIdx.x * 64;

  float sum[J * MXMMQ_I / MXMMQ_NTHREADS] = {0.0f};

  // cp.async pipeline (see moe_mxfp4_seg_w1): span parity picks the buffer,
  // the next y group is in flight during each mma, and the IQ2 decode's
  // global loads overlap the pending group.
  seg_load_y_tokens<J>(input, tile_y0, token_routes, ncols, 0,
                       input_token_blocks, top_k);
  seg_cp_commit();
  for (int sb = 0; sb < blocks_per_row; ++sb) {
    const int kb0 = sb * 8;
    seg_load_x_w1_iq2(expert_base, tile_x, blocks_per_row, sb, g0,
                      intermediate);
    seg_load_y_tokens<J>(input, tile_y1, token_routes, ncols, kb0 + 4,
                         input_token_blocks, top_k);
    seg_cp_commit();
    seg_cp_wait<1>();
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y0, sum, 0);
    __syncthreads();
    if (sb + 1 < blocks_per_row) {
      seg_load_y_tokens<J>(input, tile_y0, token_routes, ncols, kb0 + 8,
                           input_token_blocks, top_k);
      seg_cp_commit();
      seg_cp_wait<1>();
    } else {
      seg_cp_wait<0>();
    }
    __syncthreads();
    vec_dot_q8_0_q8_1_mma<J>(tile_x, tile_y1, sum, 32);
    __syncthreads();
  }

  // Fused SwiGLU + route weight + Q8_1 emission -- identical epilogue to
  // the MXFP4 seg W1 (C spilled to the dead weight-staging smem).
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

// ---------------------------------------------------------------- W2 (Q2_K)
// Repacked planes per expert: quant uint32[row*(K/16) + sb16], scale
// uint8[same index], dm half2[row*(K/256) + superblock].
__device__ __forceinline__ void seg_load_x_w2_q2k(
    const char* __restrict__ expert_base, int* __restrict__ x_tile,
    const int rows, const int K, const int superblock, const int row_x0) {
  const uint32_t* quant = reinterpret_cast<const uint32_t*>(expert_base);
  const uint8_t* scale = reinterpret_cast<const uint8_t*>(expert_base) +
                         int64_t(rows) * K / 4;
  const half2* dm = reinterpret_cast<const half2*>(
      reinterpret_cast<const uint8_t*>(expert_base) + int64_t(rows) * K / 4 +
      int64_t(rows) * K / 16);
  const int subblocks_per_row = K / 16;
  const int superblocks_per_row = K / 256;
  const int tid = threadIdx.y * 32 + threadIdx.x;

  half2* x_dm = reinterpret_cast<half2*>(x_tile + 64);
#pragma unroll
  for (int r = 0; r < MXMMQ_I * 16 / MXMMQ_NTHREADS; ++r) {
    const int item = tid + r * MXMMQ_NTHREADS;
    const int i = item / 16;
    const int s = item % 16;
    const int row = row_x0 + i;
    const int idx = row * subblocks_per_row + superblock * 16 + s;
    const uint32_t w = quant[idx];
    const uint8_t sc = scale[idx];
    const float2 d2 = __half22float2(dm[row * superblocks_per_row +
                                        superblock]);
    int* qs = x_tile + i * Q2K_X_STRIDE + s * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qs[j] = q2k_spread_byte((w >> (8 * j)) & 0xFFu);
    }
    // half2 is int-sized, so the row stride in half2 units equals the
    // int stride.
    x_dm[i * Q2K_X_STRIDE + s] =
        __floats2half2_rn(d2.x * float(sc & 0xF), d2.y * float(sc >> 4));
  }
}

// Q2_K x q8_1 mma vec dot over one 32-int span (k00 in {0, 32}): m16n8k16
// fragments so each mma covers exactly one per-16 (d*sc, dmin*m) group; the
// min term uses a ones-matrix mma to synthesize per-16 activation sums.
template <int J>
__device__ __forceinline__ void vec_dot_q2k_q8_1_mma(
    const int* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ sum, const int k00) {
  using namespace vllm_mmq_v2;
  typedef tile<16, 8> tile_C;
  typedef tile<16, 4> tile_A16;
  typedef tile<8, 4> tile_B16;

  constexpr int rows_per_warp = mmq_rows_per_warp(J);
  constexpr int ntx = rows_per_warp / tile_C::I;

  y += (threadIdx.y % ntx) * (tile_C::J * MXMMQ_Y_STRIDE);

  const int* x_qs = x;
  const half2* x_dm = reinterpret_cast<const half2*>(x + 64);
  const int* y_qs = y + 4;
  const float* y_df = reinterpret_cast<const float*>(y);

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

  tile_A16 ones;
  ones.x[0] = 0x01010101;
  ones.x[1] = 0x01010101;

#pragma unroll
  for (int k16 = 0; k16 < 8; ++k16) {
    const int kx = k00 + 4 * k16;  // x int offset of this 16-value group
    const int ky = 4 * k16;        // y span int offset

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
      tile_B16 B;
      load_generic(B, y_qs + j0 * MXMMQ_Y_STRIDE + ky, MXMMQ_Y_STRIDE);

      tile_C Cm;
      mma(Cm, ones, B);

      float dB[tile_C::ne / 2];
#pragma unroll
      for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);
        dB[l] = y_df[j * MXMMQ_Y_STRIDE + ky / 8];
      }

#pragma unroll
      for (int n = 0; n < ntx; ++n) {
        tile_A16 A;
        load_generic(A, x_qs + (i0 + n * tile_A16::I) * Q2K_X_STRIDE + kx,
                     Q2K_X_STRIDE);
        tile_C Cd;
        mma(Cd, A, B);

#pragma unroll
        for (int l = 0; l < tile_C::ne; ++l) {
          const int i = i0 + n * tile_C::I + tile_C::get_i(l);
          const float2 dm = __half22float2(x_dm[i * Q2K_X_STRIDE + kx / 4]);
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
              (float(Cd.x[l]) * dm.x - float(Cm.x[l]) * dm.y) * dB[l % 2];
        }
      }
    }
  }
}

template <typename scalar_t, int J>
__global__ __launch_bounds__(MXMMQ_NTHREADS, 1) void moe_q2k_seg_w2(
    const void* __restrict__ weights, const block_q8_1* __restrict__ mid,
    scalar_t* __restrict__ w2out, const int* __restrict__ perm_ids,
    const int* __restrict__ g_rowseg, const int* __restrict__ g_tseg,
    const int64_t exp_stride_bytes, const int intermediate,
    const int out_row, const int experts) {
  using namespace vllm_mmq_v2;

  extern __shared__ int q2k_smem[];
  int* token_routes = q2k_smem;
  int* tile_y0 = token_routes + J;
  int* tile_y1 = tile_y0 + J * MXMMQ_Y_STRIDE;
  int* tile_x = tile_y1 + J * MXMMQ_Y_STRIDE;

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
  const int row_x0 = blockIdx.x * MXMMQ_I;
  const int mid_blocks_per_route = intermediate / 32;

  float sum[J * MXMMQ_I / MXMMQ_NTHREADS] = {0.0f};

  seg_load_y_mid<J>(mid, tile_y0, token_routes, ncols, 0,
                    mid_blocks_per_route);
  seg_cp_commit();
  for (int sb = 0; sb < intermediate / 256; ++sb) {
    const int kb0 = sb * 8;
    seg_load_x_w2_q2k(expert_base, tile_x, out_row, intermediate, sb,
                      row_x0);
    seg_load_y_mid<J>(mid, tile_y1, token_routes, ncols, kb0 + 4,
                      mid_blocks_per_route);
    seg_cp_commit();
    seg_cp_wait<1>();
    __syncthreads();
    vec_dot_q2k_q8_1_mma<J>(tile_x, tile_y0, sum, 0);
    __syncthreads();
    if (sb + 1 < intermediate / 256) {
      seg_load_y_mid<J>(mid, tile_y0, token_routes, ncols, kb0 + 8,
                        mid_blocks_per_route);
      seg_cp_commit();
      seg_cp_wait<1>();
    } else {
      seg_cp_wait<0>();
    }
    __syncthreads();
    vec_dot_q2k_q8_1_mma<J>(tile_x, tile_y1, sum, 32);
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

// ------------------------------------------------------------------ launch
inline int q2k_w2_smem_bytes(const int J) {
  return seg_smem_bytes(J, Q2K_X_STRIDE);
}

template <typename scalar_t>
inline void launch_moe_iq2_seg(
    const void* quant_x, const void* w1, const void* w2, void* mid,
    scalar_t* w2out, scalar_t* out, const int* topk_ids,
    const float* route_weights, int* meta, const int64_t w1_stride_bytes,
    const int64_t w2_stride_bytes, const int hidden,
    const int input_token_blocks, const int intermediate, const int out_row,
    const int tokens, const int top_k, const int experts,
    const float swiglu_limit, const bool use_j16, cudaStream_t stream) {
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
#define LAUNCH_IQ2_SEG(J)                                                    \
  do {                                                                       \
    const int w1_smem = seg_smem_bytes(J, MXMMQ_X_STRIDE);                   \
    const dim3 g1(intermediate / 64, seg_col_tiles(routes, experts, J));     \
    seg_maybe_opt_in_smem(moe_iq2_seg_w1<J>, w1_smem);                       \
    moe_iq2_seg_w1<J><<<g1, block, w1_smem, stream>>>(                       \
        w1, in, mid_blocks, perm_ids, rowseg, tseg, route_weights,           \
        w1_stride_bytes, hidden, input_token_blocks, intermediate, experts,  \
        top_k, swiglu_limit);                                                \
    const int smem = q2k_w2_smem_bytes(J);                                   \
    seg_maybe_opt_in_smem(moe_q2k_seg_w2<scalar_t, J>, smem);                \
    const dim3 g2((out_row + MXMMQ_I - 1) / MXMMQ_I,                         \
                  seg_col_tiles(routes, experts, J));                        \
    moe_q2k_seg_w2<scalar_t, J><<<g2, block, smem, stream>>>(                \
        w2, mid_blocks, w2out, perm_ids, rowseg, tseg, w2_stride_bytes,      \
        intermediate, out_row, experts);                                     \
  } while (0)
  if (use_j16) LAUNCH_IQ2_SEG(16);
  else LAUNCH_IQ2_SEG(64);
#undef LAUNCH_IQ2_SEG
  seg_reduce<scalar_t><<<tokens, 256, 0, stream>>>(w2out, topk_ids, out,
                                                   out_row, top_k, experts);
}

}  // namespace slimserve::dsv4_ampere

#endif  // USE_ROCM
