// Absorbed MLA decode for gfx942 (CDNA3), written for Kimi K3's geometry.
//
// The query reaching this kernel has already been absorbed through W_UK, so a
// decode step is a plain MQA against the paged latent cache: one 576-wide row
// per KV token, shared by every query head (the cache has no head axis).
//
//   score       = <q[0:576], kv[row][0:576]>          nope . latent + rope . k_pe
//   out[0:512] += softmax(score) * kv[row][0:512]
//
// The nope/rope split needs no branch: k_pe is rotated when it is inserted, so
// the score simply runs over all 576 lanes of the row and the accumulate stops
// at 512. Each row is read from HBM exactly once and staged in registers to
// serve both.
//
// Why no MFMA. At Kimi K3's TP8 shape (12 heads/rank) a decode token reads
// 1152 B and does ~26 kFLOP: ~23 FLOP/byte against an MI300X balance point of
// ~246. The kernel is memory bound by an order of magnitude, so matrix cores
// would buy nothing -- while their 16-wide tile is exactly what forces the
// head count to be a multiple or divisor of 16 in the hand-assembled kernels.
// Here the head count is a grid dimension, so 12 (or 7, or 13) just works.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <float.h>

namespace qc_rocm_mla {

constexpr int kWave = 64;
constexpr int kLatent = 512;  // kv_lora_rank
constexpr int kRope = 64;     // qk_rope_head_dim
constexpr int kEntry = kLatent + kRope;  // one paged cache row
constexpr int kLatentPerLane = kLatent / kWave;  // 8
constexpr int kEntryPerLane = kEntry / kWave;    // 9

// xor-reduction across a full 64-lane wavefront. Leaves the total in every
// lane, so the score needs no shared memory and all lanes run the identical
// online-softmax update in lockstep.
__device__ __forceinline__ float wave_sum(float v) {
#pragma unroll
  for (int off = 32; off > 0; off >>= 1) v += __shfl_xor(v, off, kWave);
  return v;
}

__device__ __forceinline__ float to_f32(const __hip_bfloat16 x) {
  return __bfloat162float(x);
}
__device__ __forceinline__ float to_f32(const _Float16 x) {
  return (float)x;
}
__device__ __forceinline__ void from_f32(__hip_bfloat16& dst, float v) {
  dst = __float2bfloat16(v);
}
__device__ __forceinline__ void from_f32(_Float16& dst, float v) {
  dst = (_Float16)v;
}

// Stage 1: each wavefront walks one slice of one (request, head)'s context and
// emits a partial numerator plus the running max/sum needed to merge slices.
//
// grid  = (num_heads, num_decodes, num_splits)
// block = 64 (one wavefront)
template <typename scalar_t>
__global__ void mla_decode_partition(
    const scalar_t* __restrict__ q,          // [B, H, 576]
    const scalar_t* __restrict__ kv_cache,   // [num_blocks, block_size, 576]
    const int* __restrict__ block_table,     // [B, max_blocks]
    const int* __restrict__ seq_lens,        // [B]
    float* __restrict__ partial_out,         // [B, H, S, 512]
    float* __restrict__ partial_max,         // [B, H, S]
    float* __restrict__ partial_sum,         // [B, H, S]
    const int num_heads, const int block_size, const int bt_stride,
    const int num_splits, const float scale) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int split = blockIdx.z;
  const int lane = threadIdx.x;

  const int context_len = seq_lens[batch];
  const int per_split = (context_len + num_splits - 1) / num_splits;
  const int start = split * per_split;
  const int end = min(start + per_split, context_len);

  const int64_t stat = (int64_t(batch) * num_heads + head) * num_splits + split;

  if (start >= end) {
    // An empty slice must still be neutral for the reducer.
    if (lane == 0) {
      partial_max[stat] = -FLT_MAX;
      partial_sum[stat] = 0.0f;
    }
    return;
  }

  // The absorbed query is register-resident: 576 / 64 = 9 floats per lane.
  const scalar_t* q_row = q + (int64_t(batch) * num_heads + head) * kEntry;
  float qv[kEntryPerLane];
#pragma unroll
  for (int i = 0; i < kEntryPerLane; i++) qv[i] = to_f32(q_row[lane + kWave * i]);

  float acc[kLatentPerLane];
#pragma unroll
  for (int i = 0; i < kLatentPerLane; i++) acc[i] = 0.0f;
  float running_max = -FLT_MAX;
  float running_sum = 0.0f;

  const int* bt = block_table + int64_t(batch) * bt_stride;
  for (int token = start; token < end; token++) {
    const int block = bt[token / block_size];
    if (block < 0) continue;
    const int64_t base =
        (int64_t(block) * block_size + (token % block_size)) * kEntry;

    // One read of the row serves both the score and the accumulate.
    float kv[kEntryPerLane];
#pragma unroll
    for (int i = 0; i < kEntryPerLane; i++)
      kv[i] = to_f32(kv_cache[base + lane + kWave * i]);

    float partial = 0.0f;
#pragma unroll
    for (int i = 0; i < kEntryPerLane; i++) partial += qv[i] * kv[i];
    const float score = wave_sum(partial) * scale;

    const float new_max = fmaxf(running_max, score);
    const float alpha = __expf(running_max - new_max);
    const float beta = __expf(score - new_max);
#pragma unroll
    for (int i = 0; i < kLatentPerLane; i++) acc[i] = acc[i] * alpha + beta * kv[i];
    running_sum = running_sum * alpha + beta;
    running_max = new_max;
  }

  float* out = partial_out + stat * kLatent;
#pragma unroll
  for (int i = 0; i < kLatentPerLane; i++) out[lane + kWave * i] = acc[i];
  if (lane == 0) {
    partial_max[stat] = running_max;
    partial_sum[stat] = running_sum;
  }
}

// Stage 2: merge the slices of one (request, head) with the usual
// flash-decoding rescale, and write the normalised latent output.
//
// grid = (num_heads, num_decodes), block = 64
template <typename scalar_t>
__global__ void mla_decode_reduce(
    const float* __restrict__ partial_out,  // [B, H, S, 512]
    const float* __restrict__ partial_max,  // [B, H, S]
    const float* __restrict__ partial_sum,  // [B, H, S]
    scalar_t* __restrict__ out,             // [B, H, 512]
    const int num_heads, const int num_splits) {
  const int head = blockIdx.x;
  const int batch = blockIdx.y;
  const int lane = threadIdx.x;
  const int64_t base = (int64_t(batch) * num_heads + head) * num_splits;

  float global_max = -FLT_MAX;
  for (int s = 0; s < num_splits; s++)
    global_max = fmaxf(global_max, partial_max[base + s]);

  if (global_max == -FLT_MAX) {
    // Zero-length context: nothing to attend to.
    scalar_t* dst = out + (int64_t(batch) * num_heads + head) * kLatent;
#pragma unroll
    for (int i = 0; i < kLatentPerLane; i++) from_f32(dst[lane + kWave * i], 0.0f);
    return;
  }

  float acc[kLatentPerLane];
#pragma unroll
  for (int i = 0; i < kLatentPerLane; i++) acc[i] = 0.0f;
  float total = 0.0f;

  for (int s = 0; s < num_splits; s++) {
    const float m = partial_max[base + s];
    if (m == -FLT_MAX) continue;
    const float w = __expf(m - global_max);
    const float* src = partial_out + (base + s) * kLatent;
#pragma unroll
    for (int i = 0; i < kLatentPerLane; i++) acc[i] += w * src[lane + kWave * i];
    total += w * partial_sum[base + s];
  }

  const float inv = total > 0.0f ? 1.0f / total : 0.0f;
  scalar_t* dst = out + (int64_t(batch) * num_heads + head) * kLatent;
#pragma unroll
  for (int i = 0; i < kLatentPerLane; i++)
    from_f32(dst[lane + kWave * i], acc[i] * inv);
}

}  // namespace qc_rocm_mla
