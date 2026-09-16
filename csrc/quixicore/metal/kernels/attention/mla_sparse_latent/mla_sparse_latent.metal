#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Sparse NoPE-MLA decode over bf16/f16 latent pages (GLM-5.3-Flash).
//
// Every query row attends its indexer-selected positions
// indices[row, 0:W] (request-local token positions, < 0 = pad) against the
// paged latent cache [num_blocks, block_size, LATENT] (block stride in
// elements, rows contiguous). Scores use the full LATENT-wide latent as key
// and the same latent as value (no rope half). Same online-softmax shape as
// mla_decode_fp8_sparse; the top-k list is split over P partitions (grid
// (H, R, P)) for occupancy at batch 1 (64 heads alone would leave the GPU
// bandwidth-starved), and a reduce merges the partials in fp32.
//
// Numerics: fp32 accumulation; q and latent widened from T; output rounded
// to T once. A row with no valid position (or a partition with none) yields
// zeros (l == 0).
// ---------------------------------------------------------------------------

template <typename T, int LATENT>
kernel void mla_sparse_latent_partition(
    device const T   *q            [[buffer(0)]],   // (R, H, LATENT)
    device const T   *cache        [[buffer(1)]],   // pages, block_stride elems
    device const int *block_table  [[buffer(2)]],   // (R, bt_stride)
    device const int *indices      [[buffer(3)]],   // (R, W)
    device float     *part_acc     [[buffer(4)]],   // (R, H, P, LATENT)
    device float     *part_ml      [[buffer(5)]],   // (R, H, P, 2): m, l
    constant int   &block_size     [[buffer(6)]],
    constant int   &block_stride   [[buffer(7)]],
    constant int   &bt_stride      [[buffer(8)]],
    constant float &scale          [[buffer(9)]],
    constant int   &num_heads      [[buffer(10)]],
    constant int   &width          [[buffer(11)]],  // W
    constant int   &partitions     [[buffer(12)]],  // P
    constant int   &max_block_col  [[buffer(13)]],  // bt columns - 1
    device const int *tlen         [[buffer(14)]],  // (R) valid prefix (opt)
    constant int   &has_tlen       [[buffer(15)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
  constexpr int VPL = LATENT / 32;
  const int head = (int)tgid.x;
  const int row = (int)tgid.y;
  const int part = (int)tgid.z;
  // With the indexer's per-row valid prefix the partitions split the VALID
  // range (every partition busy, chains P times shorter); without it the
  // padded width is split exactly as before (bit-identical partials).
  const int lim = has_tlen ? min(width, max(0, tlen[row])) : width;
  const int per_part = (lim + partitions - 1) / partitions;
  const int j0 = part * per_part;
  const int j1 = min(lim, j0 + per_part);
  const long q_base = ((long)row * num_heads + head) * LATENT;

  float qv[VPL], acc[VPL];
  #pragma clang loop unroll(full)
  for (int i = 0; i < VPL; ++i) {
    qv[i] = float(q[q_base + lane + 32 * i]);
    acc[i] = 0.0f;
  }
  float m = -3.4028234663852886e38f, l = 0.0f;
  device const int *idx_row = indices + (long)row * width;
  device const int *bt_row = block_table + (long)row * bt_stride;
  for (int j = j0; j < j1; ++j) {
    const int t = idx_row[j];
    if (t < 0) { continue; }
    const int block_col = t / block_size;
    if (block_col > max_block_col) { continue; }
    const int block = bt_row[block_col];
    if (block < 0) { continue; }
    const int slot = t - block_col * block_size;
    device const T *lat_row = cache + (long)block * block_stride +
        (long)slot * LATENT;
    float lat[VPL];
    float partial = 0.0f;
    #pragma clang loop unroll(full)
    for (int i = 0; i < VPL; ++i) {
      lat[i] = float(lat_row[lane + 32 * i]);
      partial += qv[i] * lat[i];
    }
    const float score = simd_sum(partial) * scale;
    const float new_m = max(m, score);
    const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
    const float beta = exp(score - new_m);
    #pragma clang loop unroll(full)
    for (int i = 0; i < VPL; ++i) { acc[i] = acc[i] * alpha + beta * lat[i]; }
    l = l * alpha + beta;
    m = new_m;
  }
  const long pbase = (((long)row * num_heads + head) * partitions + part);
  device float *acc_out = part_acc + pbase * LATENT;
  #pragma clang loop unroll(full)
  for (int i = 0; i < VPL; ++i) { acc_out[lane + 32 * i] = acc[i]; }
  if (lane == 0) {
    part_ml[pbase * 2] = m;
    part_ml[pbase * 2 + 1] = l;
  }
}

template <typename T, int LATENT>
kernel void mla_sparse_latent_reduce(
    device const float *part_acc  [[buffer(0)]],   // (R, H, P, LATENT)
    device const float *part_ml   [[buffer(1)]],   // (R, H, P, 2)
    device T           *out       [[buffer(2)]],   // (R, H, LATENT)
    constant int &num_heads       [[buffer(3)]],
    constant int &partitions      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
  constexpr int VPL = LATENT / 32;
  const int head = (int)tgid.x;
  const int row = (int)tgid.y;
  const long pbase = ((long)row * num_heads + head) * partitions;
  // Global max over the partitions that saw a valid position.
  float gm = -3.4028234663852886e38f;
  for (int p = 0; p < partitions; ++p) {
    const float lp = part_ml[(pbase + p) * 2 + 1];
    if (lp > 0.0f) { gm = max(gm, part_ml[(pbase + p) * 2]); }
  }
  float acc[VPL];
  #pragma clang loop unroll(full)
  for (int i = 0; i < VPL; ++i) { acc[i] = 0.0f; }
  float l = 0.0f;
  for (int p = 0; p < partitions; ++p) {
    const float lp = part_ml[(pbase + p) * 2 + 1];
    if (lp <= 0.0f) { continue; }
    const float w = exp(part_ml[(pbase + p) * 2] - gm);
    device const float *a = part_acc + (pbase + p) * LATENT;
    #pragma clang loop unroll(full)
    for (int i = 0; i < VPL; ++i) { acc[i] += w * a[lane + 32 * i]; }
    l += w * lp;
  }
  const long out_base = ((long)row * num_heads + head) * LATENT;
  #pragma clang loop unroll(full)
  for (int i = 0; i < VPL; ++i) {
    out[out_base + lane + 32 * i] = l == 0.0f ? T(0) : T(acc[i] / l);
  }
}

#define instantiate_mla_sparse_latent(type_name, T, LVAL)                        \
  template [[host_name("mla_sparse_latent_partition_" #type_name "_" #LVAL)]]   \
  [[kernel]] void mla_sparse_latent_partition<T, LVAL>(                         \
      device const T *q [[buffer(0)]], device const T *cache [[buffer(1)]],     \
      device const int *block_table [[buffer(2)]],                              \
      device const int *indices [[buffer(3)]],                                  \
      device float *part_acc [[buffer(4)]], device float *part_ml [[buffer(5)]], \
      constant int &block_size [[buffer(6)]],                                   \
      constant int &block_stride [[buffer(7)]],                                 \
      constant int &bt_stride [[buffer(8)]], constant float &scale [[buffer(9)]], \
      constant int &num_heads [[buffer(10)]], constant int &width [[buffer(11)]], \
      constant int &partitions [[buffer(12)]],                                  \
      constant int &max_block_col [[buffer(13)]],                               \
      device const int *tlen [[buffer(14)]],                                    \
      constant int &has_tlen [[buffer(15)]],                                    \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]]);                                 \
  template [[host_name("mla_sparse_latent_reduce_" #type_name "_" #LVAL)]]      \
  [[kernel]] void mla_sparse_latent_reduce<T, LVAL>(                            \
      device const float *part_acc [[buffer(0)]],                               \
      device const float *part_ml [[buffer(1)]], device T *out [[buffer(2)]],   \
      constant int &num_heads [[buffer(3)]],                                    \
      constant int &partitions [[buffer(4)]],                                   \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_mla_sparse_latent(bfloat16, bf16, 512)
instantiate_mla_sparse_latent(float16, half, 512)
