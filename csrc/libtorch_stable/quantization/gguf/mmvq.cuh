// GGUF quantized GEMV.
//
// The kernel geometry follows modern llama.cpp (ggml/src/ggml-cuda/mmvq.cu,
// b10121) rather than the b2899 snapshot this file replaces. Measured on one
// MI300X at llama.cpp's own benchmark shape (q2_K, m=4096, k=14336), the b2899
// geometry was 1.35x-2.49x slower than llama.cpp across batch widths 1..8; the
// three things it lacked are all here:
//
//   nwarps               the whole block cooperates on a row group and strides
//                        by vdr*nwarps*warp_size/qi, instead of one warp owning
//                        a row. Partial sums meet in LDS.
//   rows_per_cuda_block  >1 output row per block at batch>1, which keeps enough
//                        waves resident on 304 CUs for narrow matrices.
//   small_k              when a single block iteration already covers every K
//                        block, widen rows_per_block to nwarps instead of
//                        leaving most of the block idle in the reduction. This
//                        is the MoE w2 shape (k=1024, q2_K): blocks_per_row=4
//                        against nwarps*blocks_per_iter_1warp=8.
//
// Only gfx942/CDNA is targeted here, so llama.cpp's per-architecture table is
// collapsed to its GCN row. The vec_dot callbacks keep the b2899 signature
// (block pointer, not base+index), so vecdotq.cuh is unchanged.

#pragma once

// llama.cpp's MMVQ_PARAMETERS_GCN row, the one gfx942 selects.
static constexpr __host__ __device__ int mmvq_nwarps(int ncols_dst) {
  return ncols_dst <= 4 ? 2 : 1;
}

static constexpr __host__ __device__ int mmvq_rows_per_block(int ncols_dst,
                                                             bool small_k,
                                                             int nwarps) {
  if (ncols_dst == 1) {
    return small_k ? nwarps : 1;
  }
#ifndef USE_ROCM
  // A100 (sm80): at the 4-wide spec-verify batch, 4 rows per block amortizes
  // the per-row weight stream better than 2 across the GLM-5.2 TP4 dense
  // shapes (25.4 -> 20.7 us on the 6144x4096 attn output, -13% summed over
  // the step's dense GEMVs). Measured only for ncols_dst == 4; the GCN
  // geometry below is unchanged for every other width and for ROCm.
  if (ncols_dst == 4) {
    return 4;
  }
#endif
  return ncols_dst <= 8 ? 2 : 1;
}

template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int ncols_dst, bool small_k>
static __global__ void mul_mat_vec_q(const void* __restrict__ vx,
                                     const void* __restrict__ vy,
                                     scalar_t* __restrict__ dst,
                                     const int ncols, const int nrows,
                                     const int nvecs) {
  constexpr int nwarps = mmvq_nwarps(ncols_dst);
  constexpr int rows_per_cuda_block =
      mmvq_rows_per_block(ncols_dst, small_k, nwarps);

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int vec0 = ncols_dst * blockIdx.y;
  if (row0 >= nrows || vec0 >= nvecs) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  // Resolve the tail clamps ONCE, outside the k loop. Doing the min() inside
  // made every x address data-dependent, which stopped the compiler hoisting
  // the shared weight load out of the j loop -- n=8 measured 98.6 us that way
  // against llama.cpp's 34.3 us, while n=1..4 still improved. llama.cpp
  // likewise precomputes a base offset and only adds affine terms in the loop.
  const block_q_t* xrow[rows_per_cuda_block];
#pragma unroll
  for (int i = 0; i < rows_per_cuda_block; ++i) {
    const int row = row0 + i < nrows ? row0 + i : nrows - 1;
    xrow[i] = x + (size_t)row * blocks_per_row;
  }
  const block_q8_1* yvec[ncols_dst];
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    const int vec = vec0 + j < nvecs ? vec0 + j : nvecs - 1;
    yvec[j] = y + (size_t)vec * (nrows_y / QK8_1);
  }

  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  const int kqs = vdr * (tid % (qi / vdr));
  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row; kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1);

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp[j][i] += vec_dot_q_cuda(&xrow[i][kbx], &yvec[j][kby], kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
#pragma unroll
      for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        tmp[j][i] += VLLM_SHFL_XOR_SYNC(tmp[j][i], mask);
      }
      if (threadIdx.x == 0 && row0 + i < nrows && vec0 + j < nvecs) {
        dst[(vec0 + j) * nrows + row0 + i] = tmp[j][i];
      }
    }
  }
}

// blocks_per_row_x < nwarps*blocks_per_iter_1warp means one block iteration
// already covers every K block, so most of the block would sit idle; widen
// rows_per_block instead. Mirrors llama.cpp's should_use_small_k.
template <int qk, int qi, int vdr>
static bool mmvq_use_small_k(int ncols, int ncols_dst) {
  const int blocks_per_row_x = ncols / qk;
  const int blocks_per_iter_1warp = vdr * WARP_SIZE / qi;
  const int nwarps = mmvq_nwarps(ncols_dst);
  return nwarps > 1 && blocks_per_row_x < nwarps * blocks_per_iter_1warp;
}

#define VLLM_MMVQ_LAUNCH(NCOLS, SMALL_K, QK, QI, BLOCK_T, VDR, VECDOT)         \
  do {                                                                         \
    constexpr int nw = mmvq_nwarps(NCOLS);                                     \
    constexpr int rpb = mmvq_rows_per_block(NCOLS, SMALL_K, nw);               \
    const dim3 block_nums((nrows + rpb - 1) / rpb,                             \
                          (nvecs + (NCOLS) - 1) / (NCOLS), 1);                 \
    const dim3 block_dims(WARP_SIZE, nw, 1);                              \
    mul_mat_vec_q<scalar_t, QK, QI, BLOCK_T, VDR, VECDOT, NCOLS, SMALL_K>      \
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows,     \
                                                nvecs);                        \
  } while (0)

#define VLLM_MMVQ_LAUNCH_SK(NCOLS, QK, QI, BLOCK_T, VDR, VECDOT)               \
  do {                                                                         \
    if ((NCOLS) == 1 && mmvq_use_small_k<QK, QI, VDR>(ncols, NCOLS)) {         \
      VLLM_MMVQ_LAUNCH(NCOLS, true, QK, QI, BLOCK_T, VDR, VECDOT);             \
    } else {                                                                   \
      VLLM_MMVQ_LAUNCH(NCOLS, false, QK, QI, BLOCK_T, VDR, VECDOT);            \
    }                                                                          \
  } while (0)

// ncols_dst is templated so the j loop unrolls and the x load is hoisted out
// of it; above 8 the register cost stops paying, so wider batches tile by 8.
#define VLLM_MMVQ_DISPATCH(QK, QI, BLOCK_T, VDR, VECDOT)                       \
  do {                                                                         \
    if (nvecs <= 1) {                                                          \
      VLLM_MMVQ_LAUNCH_SK(1, QK, QI, BLOCK_T, VDR, VECDOT);                    \
    } else if (nvecs <= 2) {                                                   \
      VLLM_MMVQ_LAUNCH(2, false, QK, QI, BLOCK_T, VDR, VECDOT);                \
    } else if (nvecs <= 4) {                                                   \
      VLLM_MMVQ_LAUNCH(4, false, QK, QI, BLOCK_T, VDR, VECDOT);                \
    } else {                                                                   \
      VLLM_MMVQ_LAUNCH(8, false, QK, QI, BLOCK_T, VDR, VECDOT);                \
    }                                                                          \
  } while (0)

template <typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_q4_0_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
                     vec_dot_q4_1_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
                     vec_dot_q5_0_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
                     vec_dot_q5_1_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
                     vec_dot_q8_0_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,
                     vec_dot_q2_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
                     vec_dot_q3_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
                     vec_dot_q4_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
                     vec_dot_q5_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(const void* vx, const void* vy,
                                       scalar_t* dst, const int ncols,
                                       const int nrows, const int nvecs,
                                       cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
                     vec_dot_q6_K_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(const void* vx, const void* vy,
                                          scalar_t* dst, const int ncols,
                                          const int nrows, const int nvecs,
                                          cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(const void* vx, const void* vy,
                                         scalar_t* dst, const int ncols,
                                         const int nrows, const int nvecs,
                                         cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(const void* vx, const void* vy,
                                        scalar_t* dst, const int ncols,
                                        const int nrows, const int nvecs,
                                        cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(const void* vx, const void* vy,
                                          scalar_t* dst, const int ncols,
                                          const int nrows, const int nvecs,
                                          cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(const void* vx, const void* vy,
                                        scalar_t* dst, const int ncols,
                                        const int nrows, const int nvecs,
                                        cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(const void* vx, const void* vy,
                                        scalar_t* dst, const int ncols,
                                        const int nrows, const int nvecs,
                                        cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(const void* vx, const void* vy,
                                         scalar_t* dst, const int ncols,
                                         const int nrows, const int nvecs,
                                         cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ,
                     vec_dot_iq4_nl_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(const void* vx, const void* vy,
                                         scalar_t* dst, const int ncols,
                                         const int nrows, const int nvecs,
                                         cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(const void* vx, const void* vy,
                                        scalar_t* dst, const int ncols,
                                        const int nrows, const int nvecs,
                                        cudaStream_t stream) {
  VLLM_MMVQ_DISPATCH(QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1);
}
