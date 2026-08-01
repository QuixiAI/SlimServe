#include <cstdint>
#include <cstdlib>

/* Adapted from ./csrc/quantization/gguf/mmq.cuh
   based on ./vllm/model_executor/layers/fused_moe/experts/triton_moe.py */
// `skip_padding`: moe_align_block_size pads every expert up to a full mmq_x
// block, so at the verify widths this serves most columns of a tile are
// padding -- at 256 routed rows over 164 experts only 256 of 656 slots are
// real. Padding slots hold sorted_token_ids >= ncols_dst, their y-tile is never
// loaded and the write-back already drops them, but the dot product still ran
// for them. The predicate is warp-uniform (the column index is derived from
// threadIdx.y), so skipping costs one branch per column. Callers turn it off
// once the routed rows outnumber the padded slots, where the branch is pure
// overhead.
template <typename scalar_t, int qk, int qr, int qi, bool need_sum,
          typename block_q_t, int mmq_x, int mmq_y, int nwarps,
          allocate_tiles_cuda_t allocate_tiles, load_tiles_cuda_t load_tiles,
          int vdr, vec_dot_q_mul_mat_cuda_t vec_dot, int y_spans_per_stage = 1,
          bool skip_padding = false>
static __device__ __forceinline__ void moe_q(
    const void* __restrict__ vx, const void* __restrict__ vy,
    scalar_t* __restrict__ dst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded, const int exp_stride,
    const int ncols_x, const int nrows_x, const int ncols_y, const int nrows_y,
    const int nrows_dst, const int top_k) {
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi;

  const int ncols_dst = ncols_y * top_k;

  const auto row_dst_0 = blockIdx.x * mmq_y;
  const int& row_x_0 = row_dst_0;

  const auto col_dst_0 = blockIdx.y * mmq_x;

  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
  }

  const int exp_idx = expert_ids[blockIdx.y];
#ifndef USE_ROCM
  // CUDA callers no longer pre-fill dst; zero this tile before bailing out.
  if (exp_idx > 255 || exp_idx < 0) {
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
      const int col_dst = token_offs[j / nwarps];
      if (col_dst >= ncols_dst) continue;
#pragma unroll
      for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
        const auto row_dst = row_dst_0 + threadIdx.x + i;
        if (row_dst >= nrows_dst) continue;
        dst[col_dst * nrows_dst + row_dst] = scalar_t(0);
      }
    }
    return;
  }
#else
  if (exp_idx > 255 || exp_idx < 0) return;
#endif
  if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;

  const block_q_t* x = (const block_q_t*)((char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF * y_spans_per_stage];
  __shared__ half2
      tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1 * y_spans_per_stage];

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(x + row_x_0 * blocks_per_row_x + ib0, tile_x_ql, tile_x_dm,
               tile_x_qh, tile_x_sc, threadIdx.y, nrows_x - row_x_0 - 1,
               threadIdx.x, blocks_per_row_x);

    const int n_per_r = ((qk * blocks_per_warp) / qr);
#pragma unroll
    for (int ir0 = 0; ir0 < qr; ir0 += y_spans_per_stage) {
#pragma unroll
      for (int ir = ir0;
           ir < ir0 + y_spans_per_stage && ib0 * qk + ir * n_per_r < ncols_x;
           ++ir) {
        const auto kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
        const int kbxd = kqs / QI8_1;
        const int span = ir - ir0;

#pragma unroll
        for (int i = 0; i < mmq_x; i += nwarps) {
          const int col_y_eff = token_offs[i / nwarps] / top_k;
          const int block_x = ib0 * (qk / QK8_1) + kbxd;
          if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
            const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
            const int index_y = span * mmq_x * WARP_SIZE_GGUF +
                                (threadIdx.y + i) * WARP_SIZE_GGUF +
                                kqs % WARP_SIZE_GGUF;
            tile_y_qs[index_y] =
                get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
          }
        }

        if (threadIdx.x < n_per_r / QK8_1) {
          const auto kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
          const int col_y_eff = token_offs[threadIdx.y] / top_k;
          const int block_x =
              ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

          if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
            const half2* dsi_src =
                &y[col_y_eff * blocks_per_col_y + block_x].ds;
            half2* dsi_dst =
                &tile_y_ds[span * mmq_x * (WARP_SIZE_GGUF / QI8_1) +
                           threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + kby];

            if (need_sum) {
              *dsi_dst = *dsi_src;
            } else {
              float* dfi_dst = (float*)dsi_dst;
              *dfi_dst = __low2float(*dsi_src);
            }
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (int ir = ir0;
           ir < ir0 + y_spans_per_stage && ib0 * qk + ir * n_per_r < ncols_x;
           ++ir) {
        const int span = ir - ir0;
        const int* span_y_qs = tile_y_qs + span * mmq_x * WARP_SIZE_GGUF;
        const half2* span_y_ds =
            tile_y_ds + span * mmq_x * WARP_SIZE_GGUF / QI8_1;

        // j outermost so a padding column skips its whole k sweep; the k order
        // per (i, j) is unchanged, so the accumulation stays bitwise identical.
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
          if (skip_padding && token_offs[j / nwarps] >= ncols_dst) {
            continue;
          }
          // #pragma unroll // unrolling this loop causes too much register
          // pressure
          for (int k = ir * WARP_SIZE_GGUF / qr;
               k < (ir + 1) * WARP_SIZE_GGUF / qr; k += vdr) {
#pragma unroll
            for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
              sum[i / WARP_SIZE_GGUF][j / nwarps] +=
                  vec_dot(tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, span_y_qs,
                          span_y_ds, threadIdx.x + i, threadIdx.y + j, k);
            }
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const auto row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_0 8
  #define MOE_Y_Q4_0 128
  #define NWARPS_Q4_0 8
#else
  #define MOE_X_Q4_0 4
  #define MOE_Y_Q4_0 32
  #define NWARPS_Q4_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_0, 2)
#endif
    moe_q4_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_0;
  const int mmq_y = MOE_Y_Q4_0;
  const int nwarps = NWARPS_Q4_0;

  moe_q<scalar_t, QK4_0, QR4_0, QI4_0, true, block_q4_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_0<mmq_y>, load_tiles_q4_0<mmq_y, nwarps, need_check>,
        VDR_Q4_0_Q8_1_MMQ, vec_dot_q4_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MOE_X_Q4_0;
  int mmq_y = MOE_Y_Q4_0;
  int nwarps = NWARPS_Q4_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_1 8
  #define MOE_Y_Q4_1 128
  #define NWARPS_Q4_1 8
#else
  #define MOE_X_Q4_1 4
  #define MOE_Y_Q4_1 32
  #define NWARPS_Q4_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_1, 2)
#endif
    moe_q4_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_1;
  const int mmq_y = MOE_Y_Q4_1;
  const int nwarps = NWARPS_Q4_1;

  moe_q<scalar_t, QK4_1, QR4_1, QI4_1, true, block_q4_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_1<mmq_y>, load_tiles_q4_1<mmq_y, nwarps, need_check>,
        VDR_Q4_1_Q8_1_MMQ, vec_dot_q4_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  int mmq_x = MOE_X_Q4_1;
  int mmq_y = MOE_Y_Q4_1;
  int nwarps = NWARPS_Q4_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_0 8
  #define MOE_Y_Q5_0 128
  #define NWARPS_Q5_0 8
#else
  #define MOE_X_Q5_0 4
  #define MOE_Y_Q5_0 32
  #define NWARPS_Q5_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_0, 2)
#endif
    moe_q5_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  moe_q<scalar_t, QK5_0, QR5_0, QI5_0, false, block_q5_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_0<mmq_y>, load_tiles_q5_0<mmq_y, nwarps, need_check>,
        VDR_Q5_0_Q8_1_MMQ, vec_dot_q5_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_0;
  const int mmq_y = MOE_Y_Q5_0;
  const int nwarps = NWARPS_Q5_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_1 8
  #define MOE_Y_Q5_1 128
  #define NWARPS_Q5_1 8
#else
  #define MOE_X_Q5_1 4
  #define MOE_Y_Q5_1 32
  #define NWARPS_Q5_1 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_1, 2)
#endif
    moe_q5_1(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  moe_q<scalar_t, QK5_1, QR5_1, QI5_1, true, block_q5_1, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_1<mmq_y>, load_tiles_q5_1<mmq_y, nwarps, need_check>,
        VDR_Q5_1_Q8_1_MMQ, vec_dot_q5_1_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_1_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_1;
  const int mmq_y = MOE_Y_Q5_1;
  const int nwarps = NWARPS_Q5_1;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_1<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q8_0 8
  #define MOE_Y_Q8_0 128
  #define NWARPS_Q8_0 8
#else
  #define MOE_X_Q8_0 4
  #define MOE_Y_Q8_0 32
  #define NWARPS_Q8_0 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q8_0, 2)
#endif
    moe_q8_0(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  moe_q<scalar_t, QK8_0, QR8_0, QI8_0, false, block_q8_0, mmq_x, mmq_y, nwarps,
        allocate_tiles_q8_0<mmq_y>, load_tiles_q8_0<mmq_y, nwarps, need_check>,
        VDR_Q8_0_Q8_1_MMQ, vec_dot_q8_0_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q8_0_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q8_0;
  const int mmq_y = MOE_Y_Q8_0;
  const int nwarps = NWARPS_Q8_0;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q8_0<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q2_K 4
  #define MOE_Y_Q2_K 32
  #define MOE_NWARPS_Q2_K 4
#else
  #define MOE_X_Q2_K 4
  #define MOE_Y_Q2_K 32
  #define MOE_NWARPS_Q2_K 4
#endif

template <typename scalar_t, bool need_check, int mmq_x, int mmq_y, int nwarps,
          bool skip_padding = false>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* nwarps, 2)
#endif
    moe_q2_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  moe_q<scalar_t, QK_K, QR2_K, QI2_K, false, block_q2_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q2_K<mmq_y>, load_tiles_q2_K<mmq_y, nwarps, need_check>,
        VDR_Q2_K_Q8_1_MMQ, vec_dot_q2_K_q8_1_mul_mat, 2, skip_padding>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

// True when moe_align_block_size left enough padding for skipping it to pay:
// the buffer it sizes is rows + num_experts * (mmq_x - 1), so recovering the
// expert count from the padded length tells us whether experts average fewer
// than mmq_x routed rows -- the regime where most tile columns are padding.
static inline bool moe_skip_padding_pays(int rows, int tokens_post_padded,
                                         int mmq_x) {
  static const bool enabled = [] {
    const char* s = std::getenv("VLLM_GGUF_MOE_SKIP_PAD");
    return !(s && (s[0] == '0' || s[0] == 'f' || s[0] == 'F'));
  }();
  if (!enabled || mmq_x <= 1 || tokens_post_padded <= rows) {
    return false;
  }
  const int num_experts = (tokens_post_padded - rows) / (mmq_x - 1);
  return rows < num_experts * mmq_x;
}

// Taller 4x64 tile once the routed rows can fill the grid twice over: at
// GLM shapes (E=256, w1 [1024,6144] / w2 [6144,512]) mmq_y=64 measured
// 13-16% faster at 1024/2048 routed rows (bs=32/64 verify) and slower below,
// where the wider grid is needed for occupancy. mmq_x stays 4, so
// moe_align_block_size layout is unchanged. VLLM_GGUF_MOE_Q2K_Y64=0 pins the
// legacy 4x32 tile at every width.
static inline bool moe_q2_K_use_y64(int rows) {
#ifdef USE_ROCM
  (void)rows;
  return false;  // ROCm keeps its tuned 8x128 config untouched
#else
  static const bool enabled = [] {
    const char* s = std::getenv("VLLM_GGUF_MOE_Q2K_Y64");
    return !(s && (s[0] == '0' || s[0] == 'f' || s[0] == 'F'));
  }();
  return enabled && rows >= 1024;
#endif
}

template <typename scalar_t>
static void ggml_moe_q2_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q2_K;
  const int nwarps = MOE_NWARPS_Q2_K;
  const int rows = ncols_y * top_k;
  const bool y64 = moe_q2_K_use_y64(rows);
  const int mmq_y = y64 ? 2 * MOE_Y_Q2_K : MOE_Y_Q2_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

#define VLLM_MOE_Q2_K_LAUNCH(NEED_CHECK, SKIP_PADDING, MMQ_Y)                \
  moe_q2_K<scalar_t, NEED_CHECK, MOE_X_Q2_K, MMQ_Y, MOE_NWARPS_Q2_K,         \
           SKIP_PADDING><<<block_nums, block_dims, 0, stream>>>(             \
      w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,     \
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k)

#define VLLM_MOE_Q2_K_LAUNCH_Y(NEED_CHECK, SKIP_PADDING)                     \
  do {                                                                       \
    if (y64) {                                                               \
      VLLM_MOE_Q2_K_LAUNCH(NEED_CHECK, SKIP_PADDING, 2 * MOE_Y_Q2_K);        \
    } else {                                                                 \
      VLLM_MOE_Q2_K_LAUNCH(NEED_CHECK, SKIP_PADDING, MOE_Y_Q2_K);            \
    }                                                                        \
  } while (0)

  const bool need_check = nrows_x % mmq_y != 0;
  const bool skip_padding =
      moe_skip_padding_pays(rows, tokens_post_padded, mmq_x);
  if (need_check) {
    if (skip_padding) {
      VLLM_MOE_Q2_K_LAUNCH_Y(true, true);
    } else {
      VLLM_MOE_Q2_K_LAUNCH_Y(true, false);
    }
  } else {
    if (skip_padding) {
      VLLM_MOE_Q2_K_LAUNCH_Y(false, true);
    } else {
      VLLM_MOE_Q2_K_LAUNCH_Y(false, false);
    }
  }
#undef VLLM_MOE_Q2_K_LAUNCH_Y
#undef VLLM_MOE_Q2_K_LAUNCH
}

#if defined(USE_ROCM)
  #define MOE_X_Q3_K 8
  #define MOE_Y_Q3_K 128
  #define NWARPS_Q3_K 8
#else
  #define MOE_X_Q3_K 4
  #define MOE_Y_Q3_K 32
  #define NWARPS_Q3_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q3_K, 2)
#endif
    moe_q3_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {

  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  moe_q<scalar_t, QK_K, QR3_K, QI3_K, false, block_q3_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q3_K<mmq_y>, load_tiles_q3_K<mmq_y, nwarps, need_check>,
        VDR_Q3_K_Q8_1_MMQ, vec_dot_q3_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}
template <typename scalar_t>
static void ggml_moe_q3_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q3_K;
  const int mmq_y = MOE_Y_Q3_K;
  const int nwarps = NWARPS_Q3_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q3_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q4_K 8
  #define MOE_Y_Q4_K 128
  #define NWARPS_Q4_K 8
#else
  #define MOE_X_Q4_K 4
  #define MOE_Y_Q4_K 32
  #define NWARPS_Q4_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q4_K, 2)
#endif
    moe_q4_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  moe_q<scalar_t, QK_K, QR4_K, QI4_K, true, block_q4_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q4_K<mmq_y>, load_tiles_q4_K<mmq_y, nwarps, need_check>,
        VDR_Q4_K_Q8_1_MMQ, vec_dot_q4_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q4_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q4_K;
  const int mmq_y = MOE_Y_Q4_K;
  const int nwarps = NWARPS_Q4_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q4_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q5_K 8
  #define MOE_Y_Q5_K 128
  #define NWARPS_Q5_K 8
#else
  #define MOE_X_Q5_K 4
  #define MOE_Y_Q5_K 32
  #define NWARPS_Q5_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q5_K, 2)
#endif
    moe_q5_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  moe_q<scalar_t, QK_K, QR5_K, QI5_K, true, block_q5_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q5_K<mmq_y>, load_tiles_q5_K<mmq_y, nwarps, need_check>,
        VDR_Q5_K_Q8_1_MMQ, vec_dot_q5_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q5_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q5_K;
  const int mmq_y = MOE_Y_Q5_K;
  const int nwarps = NWARPS_Q5_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q5_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}

#if defined(USE_ROCM)
  #define MOE_X_Q6_K 8
  #define MOE_Y_Q6_K 128
  #define NWARPS_Q6_K 8
#else
  #define MOE_X_Q6_K 4
  #define MOE_Y_Q6_K 32
  #define NWARPS_Q6_K 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_Q6_K, 2)
#endif
    moe_q6_K(const void* __restrict__ vx, const void* __restrict__ vy,
             scalar_t* __restrict__ dst, const int* sorted_token_ids,
             const int* expert_ids, const int* num_tokens_post_padded,
             const int exp_stride, const int ncols_x, const int nrows_x,
             const int ncols_y, const int nrows_y, const int nrows_dst,
             const int top_k) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  moe_q<scalar_t, QK_K, QR6_K, QI6_K, false, block_q6_K, mmq_x, mmq_y, nwarps,
        allocate_tiles_q6_K<mmq_y>, load_tiles_q6_K<mmq_y, nwarps, need_check>,
        VDR_Q6_K_Q8_1_MMQ, vec_dot_q6_K_q8_1_mul_mat>(
      vx, vy, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
      exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
}

template <typename scalar_t>
static void ggml_moe_q6_K_q8_1_cuda(
    const void* inp, const void* w, scalar_t* dst, const int* sorted_token_ids,
    const int* expert_ids, const int* num_tokens_post_padded,
    const int exp_stride, const int ncols_x, const int nrows_x,
    const int ncols_y, const int nrows_y, const int nrows_dst, const int top_k,
    const int tokens_post_padded, cudaStream_t stream) {
  const int mmq_x = MOE_X_Q6_K;
  const int mmq_y = MOE_Y_Q6_K;
  const int nwarps = NWARPS_Q6_K;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (tokens_post_padded) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  } else {
    constexpr bool need_check = true;
    moe_q6_K<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w, inp, dst, sorted_token_ids, expert_ids, num_tokens_post_padded,
        exp_stride, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst, top_k);
  }
}
