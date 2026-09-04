/*
 * Paged variant of fp8_mqa_logits (see fp8_mqa_logits_kernel.cuh and
 * fp8_mqa_logits_design.md): the DeepSeek lightning-indexer decode logits,
 * one CTA per query row, keys fetched straight from the paged indexer cache
 * through the row's block table.
 *
 * Why it exists (2026-09-03): AITER's paged MQA-logits kernel addresses the
 * cache through 32-bit buffer-load offsets, so on the packed cross-layer slab
 * (6 MB block stride, ~25 GB total) any block past ~2 GiB reads garbage and
 * the top-k silently stops seeing those positions - the GLM-5.2/MI300X
 * high-block garbling. Every address here is 64-bit. The MFMA body and the
 * epilogue are the contiguous kernel's, so the result matches it bit for bit
 * on a gathered copy of the same keys.
 *
 * Cache layout per block (block_size rows, HEAD_SIZE fp8 each, "SHUFFLE"):
 *   values: [block_size/block_tile][HEAD_SIZE/head_tile][block_tile][head_tile]
 *   scales: block_size floats after the values (the caller hands a float view
 *           whose row stride is block_stride/4).
 * The grid is static (one CTA per row, rows padded by the caller), lengths
 * and block tables are read on the device, and nothing here allocates - the
 * kernel is safe inside a captured decode CUDA graph.
 */
#pragma once
#ifndef QC_EPI
  #define QC_EPI 1
#endif
#include "mfma_fp8_dot.cuh"
#include <cstdint>

namespace qcrocm {

template <int HEAD_SIZE, int BLOCK_TILE, int HEAD_TILE>
__device__ __forceinline__ long paged_a_frag(const uint8_t* __restrict__ kv,
                                             const int* __restrict__ bt_row,
                                             long block_stride, int block_size,
                                             int n, int k0) {
  // 8 contiguous dims of position n: dims [kk, kk+8) with kk%HEAD_TILE in
  // {0, 8}, so the 8 bytes never straddle a tile chunk.
  const int l = threadIdx.x & 63;
  const int kk = k0 + (l >> 4) * 8;
  const long blk = bt_row[n / block_size];
  const int off = n % block_size;
  const long base = blk * block_stride +
                    (long)(off / BLOCK_TILE) * (BLOCK_TILE * HEAD_SIZE) +
                    (long)(kk / HEAD_TILE) * (HEAD_TILE * BLOCK_TILE) +
                    (long)(off % BLOCK_TILE) * HEAD_TILE + (kk % HEAD_TILE);
  long v;
  __builtin_memcpy(&v, kv + base, 8);
  return v;
}

template <int NUM_HEADS, int HEAD_SIZE, int BLOCK_KV, int NWAVE, int BLOCK_TILE,
          int HEAD_TILE>
__global__ __launch_bounds__(64 * NWAVE) void fp8_paged_mqa_logits(
    const uint8_t* __restrict__ q,        // [M, H, D] e4m3
    const uint8_t* __restrict__ kv,       // paged cache values (bytes)
    const float* __restrict__ kv_scales,  // paged cache scales (float view)
    const float* __restrict__ weights,    // [M, H]
    const int* __restrict__ seq_lens,     // [M] context length per row
    const int* __restrict__ block_table,  // [M, bt_stride]
    float* __restrict__ logits,           // [M, stride_logits_s] fp32
    long stride_q_s, long stride_w_s, long stride_logits_s, long block_stride,
    long scale_stride, int bt_stride, int block_size, int max_len,
    // Context split: CTA (row, blockIdx.y) covers key tiles
    // [blockIdx.y * split_len, +split_len). Logit columns are independent,
    // so splitting a long context across CTAs is a pure occupancy win
    // (one CTA per row left 64 of 304 CUs busy at decode batch sizes);
    // the grid stays static for graph replay, empty splits exit at once.
    int split_len) {
  constexpr int WH = NUM_HEADS / 32;
  constexpr int WN = NWAVE / WH;
  constexpr int N_PER_WAVE = BLOCK_KV / WN;
  constexpr int H_TILES = 2;
  constexpr int N_TILES = N_PER_WAVE / 16;

  const int row = blockIdx.x;
  const int lane = threadIdx.x & 63;
  const int wave = threadIdx.x >> 6;
  const int wh = wave % WH, wn = wave / WH;
  const int lo = lane & 15, hi = lane >> 4;
  const int hrev = bitrev4(lo);

  int end = seq_lens[row];
  end = end > max_len ? max_len : end;
  const int split_begin = (int)blockIdx.y * split_len;
  if (split_len > 0)
    end = end < split_begin + split_len ? end : split_begin + split_len;
  if (end <= split_begin) return;

  const uint8_t* q_row = q + (long)row * stride_q_s;
  const float* w_row = weights + (long)row * stride_w_s;
  const int* bt_row = block_table + (long)row * bt_stride;
  float* out_row = logits + (long)row * stride_logits_s;

  __shared__ float part[WH][BLOCK_KV];

  for (int tile = split_begin; tile < end; tile += BLOCK_KV) {
    const int n_base = tile + wn * 16;
    f32x4 acc[N_TILES][H_TILES];
#pragma unroll
    for (int i = 0; i < N_TILES; ++i)
#pragma unroll
      for (int j = 0; j < H_TILES; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

#pragma unroll
    for (int k = 0; k < HEAD_SIZE; k += 32) {
#pragma unroll
      for (int i = 0; i < N_TILES; ++i) {
        const int n0 = n_base + i * 16 * WN;
        const int n = n0 + lo;
        const long a = (n < end)
                           ? paged_a_frag<HEAD_SIZE, BLOCK_TILE, HEAD_TILE>(
                                 kv, bt_row, block_stride, block_size, n, k)
                           : 0L;
#pragma unroll
        for (int j = 0; j < H_TILES; ++j)
          acc[i][j] = mfma_16x16x32_fp8(
              a,
              load_frag_at_row(q_row, HEAD_SIZE,
                               (NUM_HEADS / 2) * j + 16 * wh + hrev, k),
              acc[i][j]);
      }
    }

#pragma unroll
    for (int i = 0; i < N_TILES; ++i) {
#pragma unroll
      for (int v = 0; v < 4; ++v) {
        const int n = n_base + i * 16 * WN + 4 * hi + v;
        float ks = 0.0f;
        if (n < end) {
          const long blk = bt_row[n / block_size];
          ks = kv_scales[blk * scale_stride + (n % block_size)];
        }
        float sum = 0.0f;
#if QC_EPI == 0
  #pragma unroll
        for (int j = 0; j < H_TILES; ++j) {
          const float t = fmaxf(acc[i][j][v] * ks, 0.0f);
          const float w = w_row[(NUM_HEADS / 2) * j + 16 * wh + hrev];
          sum = (j == 0) ? (w * t) : fmaf(w, t, sum);
        }
#elif QC_EPI == 1
  #pragma unroll
        for (int j = H_TILES - 1; j >= 0; --j) {
          const float t = fmaxf(acc[i][j][v] * ks, 0.0f);
          const float w = w_row[(NUM_HEADS / 2) * j + 16 * wh + hrev];
          sum = (j == H_TILES - 1) ? (w * t) : fmaf(w, t, sum);
        }
#else
        {
          float acc0 = 0.0f;
  #pragma unroll
          for (int j = 0; j < H_TILES; ++j) {
            const float t = fmaxf(acc[i][j][v] * ks, 0.0f);
            acc0 += w_row[j * 16 * WH + wh * 16 + lo] * t;
          }
          sum = acc0;
        }
#endif
        sum += __shfl_xor(sum, 1, 64);
        sum += __shfl_xor(sum, 2, 64);
        sum += __shfl_xor(sum, 4, 64);
        sum += __shfl_xor(sum, 8, 64);
        if (lo == 0) part[wh][n - tile] = sum;
      }
    }
    __syncthreads();

    for (int c = threadIdx.x; c < BLOCK_KV; c += blockDim.x) {
      const int n = tile + c;
      if (n >= end) continue;
      float s = part[0][c];
#pragma unroll
      for (int t = 1; t < WH; ++t) s += part[t][c];
      out_row[n] = s;
    }
    __syncthreads();
  }
}

}  // namespace qcrocm
