// copied and adapted from
// https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr,
          vec_dot_q_cuda_t vec_dot_q_cuda, int rows_per_block = 1>
static __global__ void moe_vec_q(const void* __restrict__ vx,
                                 const void* __restrict__ vy,
                                 scalar_t* __restrict__ dst,
                                 const int* topk_ids, const int topk,
                                 const int ncols, const int nrows,
                                 const int token_stride) {
  const int row0 = blockIdx.x * rows_per_block;

  const auto token = blockIdx.z / topk;
  const auto expert = (topk_ids)[blockIdx.z];

  if (row0 >= nrows) {
    return;
  }

  if (expert < 0) {
#pragma unroll
    for (int j = 0; j < rows_per_block; ++j) {
      if (threadIdx.x == 0 && row0 + j < nrows) {
        dst[blockIdx.z * nrows + row0 + j] = 0;
      }
    }
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;

  const block_q_t* x = ((const block_q_t*)vx) + expert * nrows * blocks_per_row;
  const block_q8_1* y =
      (const block_q8_1*)(((const int*)vy) + token * token_stride);

  const block_q_t* xrow[rows_per_block];
#pragma unroll
  for (int j = 0; j < rows_per_block; ++j) {
    const int row = row0 + j < nrows ? row0 + j : nrows - 1;
    xrow[j] = x + (size_t)row * blocks_per_row;
  }

  float tmp[rows_per_block] = {0.0f};

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row;
       i += blocks_per_warp) {
    const int iby = i * (qk / QK8_1);  // y block index that aligns with ibx

    const int iqs =
        vdr *
        (threadIdx.x %
         (qi / vdr));  // x block quant index when casting the quants to int

#pragma unroll
    for (int j = 0; j < rows_per_block; ++j) {
      tmp[j] += vec_dot_q_cuda(&xrow[j][i], &y[iby], iqs);
    }
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < rows_per_block; ++j) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      tmp[j] += VLLM_SHFL_XOR_SYNC(tmp[j], mask);
    }

    if (threadIdx.x == 0 && row0 + j < nrows) {
      dst[blockIdx.z * nrows + row0 + j] = tmp[j];
    }
  }
}

template <typename scalar_t>
static __global__ void moe_vec_iq2_xxs_ep(const void* __restrict__ vx,
                                          const void* __restrict__ vy,
                                          scalar_t* __restrict__ dst,
                                          const int* topk_ids, const int topk,
                                          const int ncols, const int nrows,
                                          const int token_stride) {
  const int row = blockIdx.x;
  const int token = blockIdx.z;
  const int blocks_per_row = ncols / QK_K;
  const int blocks_per_warp = WARP_SIZE / QI2_XXS;
  const block_q8_1* y =
      (const block_q8_1*)(((const int*)vy) + token * token_stride);

  for (int route = 0; route < topk; ++route) {
    const int routed_row = token * topk + route;
    const int expert = topk_ids[routed_row];
    if (expert < 0) {
      if (threadIdx.x == 0) {
        dst[routed_row * nrows + row] = 0;
      }
      continue;
    }

    const block_iq2_xxs* x = ((const block_iq2_xxs*)vx) +
                             ((size_t)expert * nrows + row) * blocks_per_row;
    float tmp = 0.0f;
    for (auto i = threadIdx.x / QI2_XXS; i < blocks_per_row;
         i += blocks_per_warp) {
      const int iby = i * (QK_K / QK8_1);
      const int iqs = threadIdx.x % QI2_XXS;
      tmp += vec_dot_iq2_xxs_q8_1(&x[i], &y[iby], iqs);
    }

#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      tmp += VLLM_SHFL_XOR_SYNC(tmp, mask);
    }
    if (threadIdx.x == 0) {
      dst[routed_row * nrows + row] = tmp;
    }
  }
}

template <typename scalar_t>
static void moe_vec_q4_0_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ,
            vec_dot_q4_0_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q4_1_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ,
            vec_dot_q4_1_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q5_0_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ,
            vec_dot_q5_0_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q5_1_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ,
            vec_dot_q5_1_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q8_0_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ,
            vec_dot_q8_0_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_mxfp4_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_MXFP4, QI_MXFP4, block_mxfp4, VDR_MXFP4_Q8_1_MMVQ,
            vec_dot_mxfp4_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q2_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int routed_rows = tokens * top_k;
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  const int rows_per_block = routed_rows >= 64 || ncols <= 1024 ? 4 : 2;
#define VLLM_MOE_VEC_Q2_LAUNCH(ROWS)                                          \
  do {                                                                        \
    const dim3 block_nums((nrows + (ROWS) - 1) / (ROWS), 1, routed_rows);     \
    moe_vec_q<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ,          \
              vec_dot_q2_K_q8_1, (ROWS)>                                      \
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k, \
                                                ncols, nrows, token_stride);  \
  } while (0)
  if (rows_per_block == 4) {
    VLLM_MOE_VEC_Q2_LAUNCH(4);
  } else {
    VLLM_MOE_VEC_Q2_LAUNCH(2);
  }
#undef VLLM_MOE_VEC_Q2_LAUNCH
}

template <typename scalar_t>
static void moe_vec_q3_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ,
            vec_dot_q3_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q4_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ,
            vec_dot_q4_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q5_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ,
            vec_dot_q5_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_q6_K_q8_1_cuda(const void* vx, const void* vy,
                                   scalar_t* dst, const int* topk_ids,
                                   const int top_k, const int tokens,
                                   const int ncols, const int nrows,
                                   const int token_stride,
                                   cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ,
            vec_dot_q6_K_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq2_xxs_q8_1_cuda(
    const void* vx, const void* vy, scalar_t* dst, const int* topk_ids,
    const int top_k, const int tokens, const int ncols, const int nrows,
    const int token_stride, cudaStream_t stream, const bool expert_parallel) {
  if (expert_parallel && top_k > 1) {
    const dim3 block_nums(nrows, 1, tokens);
    const dim3 block_dims(WARP_SIZE, 1, 1);
    moe_vec_iq2_xxs_ep<scalar_t><<<block_nums, block_dims, 0, stream>>>(
        vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
    return;
  }
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq2_xs_q8_1_cuda(const void* vx, const void* vy,
                                     scalar_t* dst, const int* topk_ids,
                                     const int top_k, const int tokens,
                                     const int ncols, const int nrows,
                                     const int token_stride,
                                     cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq2_s_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq3_xxs_q8_1_cuda(const void* vx, const void* vy,
                                      scalar_t* dst, const int* topk_ids,
                                      const int top_k, const int tokens,
                                      const int ncols, const int nrows,
                                      const int token_stride,
                                      cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq1_s_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq1_m_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq4_nl_q8_1_cuda(const void* vx, const void* vy,
                                     scalar_t* dst, const int* topk_ids,
                                     const int top_k, const int tokens,
                                     const int ncols, const int nrows,
                                     const int token_stride,
                                     cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ,
            vec_dot_iq4_nl_q8_1><<<block_nums, block_dims, 0, stream>>>(
      vx, vy, dst, topk_ids, top_k, ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq4_xs_q8_1_cuda(const void* vx, const void* vy,
                                     scalar_t* dst, const int* topk_ids,
                                     const int top_k, const int tokens,
                                     const int ncols, const int nrows,
                                     const int token_stride,
                                     cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}

template <typename scalar_t>
static void moe_vec_iq3_s_q8_1_cuda(const void* vx, const void* vy,
                                    scalar_t* dst, const int* topk_ids,
                                    const int top_k, const int tokens,
                                    const int ncols, const int nrows,
                                    const int token_stride,
                                    cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_nums(block_num_y, 1, tokens * top_k);
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  moe_vec_q<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>
      <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, topk_ids, top_k,
                                              ncols, nrows, token_stride);
}
