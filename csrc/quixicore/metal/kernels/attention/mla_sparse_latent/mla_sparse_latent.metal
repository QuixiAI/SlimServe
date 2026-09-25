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

// ---------------------------------------------------------------------------
// Head-grouped (MQA) variant for batched decode (2026-09-17, batching
// campaign). The per-head kernel above re-reads every selected latent row
// once per head: at 32 rows x 64 heads x ~2048 positions that is ~4.3 GB
// of L2 traffic per call for ~67 MB of distinct latent, and the call runs
// at the fabric rate (~4.7 ms). Here one threadgroup owns (row, partition,
// group of HG = G * NSG heads): the latent rows are staged ONCE through
// threadgroup memory in chunks of CH positions (register-prefetched one
// chunk ahead, double-buffered), and each simdgroup keeps G heads' q and
// accumulators in registers. Same fp32 partials layout as the per-head
// kernel, so mla_sparse_latent_reduce merges both. Not bit-identical to
// the per-head kernel (partition split differs); same fp32 online softmax.
//
// Element ownership per lane: elements 2*lane + 64*i and +1 (i < VP2), so
// q, latent and partial rows are read/written as 4-byte pairs.
// ---------------------------------------------------------------------------

template <typename T> inline float mla_sl_lo(uint w);
template <typename T> inline float mla_sl_hi(uint w);
template <> inline float mla_sl_lo<bf16>(uint w) { return as_type<float>(w << 16); }
template <> inline float mla_sl_hi<bf16>(uint w) { return as_type<float>(w & 0xFFFF0000u); }
template <> inline float mla_sl_lo<half>(uint w) { return float(as_type<half>(ushort(w & 0xFFFFu))); }
template <> inline float mla_sl_hi<half>(uint w) { return float(as_type<half>(ushort(w >> 16))); }

template <typename T, int LATENT, int G, int NSG>
kernel void mla_sparse_latent_mqa(
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
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
  constexpr int HG = G * NSG;             // heads per threadgroup
  constexpr int VP2 = LATENT / 64;        // uint (2-elem) words per lane
  constexpr int CH = 8;                   // positions per staged chunk
  constexpr int CPR = LATENT * 2 / 16;    // 16-byte chunks per latent row
  constexpr int NT = NSG * 32;
  constexpr int PF = (CH * CPR) / NT;     // prefetch vectors per thread
  static_assert((CH * CPR) % NT == 0, "chunk must tile the threadgroup");

  threadgroup uint4 lat_tg[2][CH * CPR];
  threadgroup int   vflag[2][CH];

  const int hg = (int)tgid.x;
  const int row = (int)tgid.y;
  const int part = (int)tgid.z;
  const int lim = has_tlen ? min(width, max(0, tlen[row])) : width;
  const int per_part = (lim + partitions - 1) / partitions;
  const int j0 = part * per_part;
  const int j1 = min(lim, j0 + per_part);
  const int head0 = hg * HG + (int)sgitg * G;

  // q for this simdgroup's G heads, 2 elements per uint word.
  float qv[G][2 * VP2], acc[G][2 * VP2];
  float m[G], l[G];
  #pragma clang loop unroll(full)
  for (int g = 0; g < G; ++g) {
    device const uint *qw = (device const uint *)(q + ((long)row * num_heads + head0 + g) * LATENT);
    #pragma clang loop unroll(full)
    for (int i = 0; i < VP2; ++i) {
      const uint w = qw[lane + 32 * i];
      qv[g][2 * i] = mla_sl_lo<T>(w);
      qv[g][2 * i + 1] = mla_sl_hi<T>(w);
      acc[g][2 * i] = 0.0f;
      acc[g][2 * i + 1] = 0.0f;
    }
    m[g] = -3.4028234663852886e38f;
    l[g] = 0.0f;
  }

  device const int *idx_row = indices + (long)row * width;
  device const int *bt_row = block_table + (long)row * bt_stride;

  // Stage chunk starting at position index jb into registers (pf) / flags.
  uint4 pf[PF];
  int pflag[PF];
#define MLA_SL_MQA_FETCH(jb_)                                                   \
  {                                                                             \
    _Pragma("clang loop unroll(full)")                                          \
    for (int i = 0; i < PF; ++i) {                                              \
      const int c = (int)tid + i * NT;                                          \
      const int p = c / CPR;                                                    \
      const int cc = c - p * CPR;                                               \
      const int j = (jb_) + p;                                                  \
      int ok = 0;                                                               \
      device const T *lat_row = cache;                                          \
      if (j < j1) {                                                             \
        const int t = idx_row[j];                                               \
        if (t >= 0) {                                                           \
          const int block_col = t / block_size;                                 \
          if (block_col <= max_block_col) {                                     \
            const int block = bt_row[block_col];                                \
            if (block >= 0) {                                                   \
              ok = 1;                                                           \
              lat_row = cache + (long)block * block_stride +                    \
                        (long)(t - block_col * block_size) * LATENT;            \
            }                                                                   \
          }                                                                     \
        }                                                                       \
      }                                                                         \
      pflag[i] = ok;                                                            \
      pf[i] = ok ? ((device const uint4 *)lat_row)[cc] : uint4(0u);             \
    }                                                                           \
  }
#define MLA_SL_MQA_STORE(buf_)                                                  \
  {                                                                             \
    _Pragma("clang loop unroll(full)")                                          \
    for (int i = 0; i < PF; ++i) {                                              \
      const int c = (int)tid + i * NT;                                          \
      lat_tg[(buf_)][c] = pf[i];                                                \
      if ((c % CPR) == 0) vflag[(buf_)][c / CPR] = pflag[i];                    \
    }                                                                           \
  }

  if (j0 < j1) {
    MLA_SL_MQA_FETCH(j0);
    MLA_SL_MQA_STORE(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int cur = 0;
  for (int jb = j0; jb < j1; jb += CH, cur ^= 1) {
    const bool more = (jb + CH) < j1;
    if (more) { MLA_SL_MQA_FETCH(jb + CH); }
    const int pc = min(CH, j1 - jb);
    for (int p = 0; p < pc; ++p) {
      if (!vflag[cur][p]) { continue; }
      threadgroup const uint *lw = (threadgroup const uint *)(lat_tg[cur] + p * CPR);
      float lat[2 * VP2];
      #pragma clang loop unroll(full)
      for (int i = 0; i < VP2; ++i) {
        const uint w = lw[lane + 32 * i];
        lat[2 * i] = mla_sl_lo<T>(w);
        lat[2 * i + 1] = mla_sl_hi<T>(w);
      }
      float partial[G];
      #pragma clang loop unroll(full)
      for (int g = 0; g < G; ++g) {
        float s = 0.0f;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 2 * VP2; ++i) { s += qv[g][i] * lat[i]; }
        partial[g] = s;
      }
      #pragma clang loop unroll(full)
      for (int g = 0; g < G; ++g) {
        const float score = simd_sum(partial[g]) * scale;
        const float new_m = max(m[g], score);
        const float alpha = l[g] == 0.0f ? 0.0f : exp(m[g] - new_m);
        const float beta = exp(score - new_m);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 2 * VP2; ++i) { acc[g][i] = acc[g][i] * alpha + beta * lat[i]; }
        l[g] = l[g] * alpha + beta;
        m[g] = new_m;
      }
    }
    if (more) { MLA_SL_MQA_STORE(cur ^ 1); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef MLA_SL_MQA_FETCH
#undef MLA_SL_MQA_STORE

  #pragma clang loop unroll(full)
  for (int g = 0; g < G; ++g) {
    const long pbase = (((long)row * num_heads + head0 + g) * partitions + part);
    device float2 *acc_out = (device float2 *)(part_acc + pbase * LATENT);
    #pragma clang loop unroll(full)
    for (int i = 0; i < VP2; ++i) {
      acc_out[lane + 32 * i] = float2(acc[g][2 * i], acc[g][2 * i + 1]);
    }
    if (lane == 0) {
      part_ml[pbase * 2] = m[g];
      part_ml[pbase * 2 + 1] = l[g];
    }
  }
}

#define instantiate_mla_sparse_latent_mqa(type_name, T, LVAL, GV, SV)             \
  template [[host_name("mla_sparse_latent_mqa_" #type_name "_" #LVAL "_g" #GV "s" #SV)]] \
  [[kernel]] void mla_sparse_latent_mqa<T, LVAL, GV, SV>(                       \
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
      ushort tid [[thread_index_in_threadgroup]],                               \
      ushort sgitg [[simdgroup_index_in_threadgroup]],                          \
      ushort lane [[thread_index_in_simdgroup]]);

instantiate_mla_sparse_latent_mqa(bfloat16, bf16, 512, 4, 4)
instantiate_mla_sparse_latent_mqa(bfloat16, bf16, 512, 4, 8)
instantiate_mla_sparse_latent_mqa(bfloat16, bf16, 512, 2, 8)
instantiate_mla_sparse_latent_mqa(bfloat16, bf16, 512, 2, 4)
instantiate_mla_sparse_latent_mqa(float16, half, 512, 4, 4)
instantiate_mla_sparse_latent_mqa(float16, half, 512, 4, 8)
instantiate_mla_sparse_latent_mqa(float16, half, 512, 2, 8)
instantiate_mla_sparse_latent_mqa(float16, half, 512, 2, 4)

// ---------------------------------------------------------------------------
// simdgroup-MMA variant (2026-09-17). The scalar kernels above are
// instruction-issue bound (~120 instructions per head-position: 16 strided
// 2-byte loads, converts, a 16-deep FMA chain, simd_sum, two exps, 32 FMAs)
// and every head repeats the loads; a partition sweep at R=32 floors at
// ~2.8-3 ms for any of them. Here a threadgroup owns (row, partition, 16
// heads) and NSG simdgroups each own a LATENT/NSG-wide slice of the latent
// dimension:
//   S^T partial[16 heads x 8 pos]  = Q[16 x KS] . Lat[8 x KS]^T   (half MMA,
//                                    fp32 accumulate, K split over simdgroups,
//                                    summed through threadgroup memory)
//   P = exp(S - m) in half; O[16 x KS] += P[16 x 8] . Lat[8 x KS]  (half MMA)
//   O is rescaled by diag(alpha) with a float MMA only on chunks where some
//   head's running max moved.
// Positions are staged 8 at a time as half rows in threadgroup memory
// (register-prefetched one chunk ahead, double-buffered). Q's 16 rows are
// staged once through the same buffer and held as A fragments in registers.
// Partials layout matches the per-head kernel, so the same reduce merges.
// Numerics: half operands (bf16 -> half is exact for |x| < 65504), fp32
// accumulation; P rounded to half. Not bit-identical to the scalar kernels.
// ---------------------------------------------------------------------------
template <typename T> inline half mla_sl_to_half_lo(uint w);
template <typename T> inline half mla_sl_to_half_hi(uint w);
template <> inline half mla_sl_to_half_lo<bf16>(uint w) { return half(as_type<float>(w << 16)); }
template <> inline half mla_sl_to_half_hi<bf16>(uint w) { return half(as_type<float>(w & 0xFFFF0000u)); }
template <> inline half mla_sl_to_half_lo<half>(uint w) { return as_type<half>(ushort(w & 0xFFFFu)); }
template <> inline half mla_sl_to_half_hi<half>(uint w) { return as_type<half>(ushort(w >> 16)); }

template <typename T>
inline void mla_sl_store_half8(threadgroup half *dst, uint4 w) {
  threadgroup half4 *d4 = (threadgroup half4 *)dst;
  d4[0] = half4(mla_sl_to_half_lo<T>(w.x), mla_sl_to_half_hi<T>(w.x),
                mla_sl_to_half_lo<T>(w.y), mla_sl_to_half_hi<T>(w.y));
  d4[1] = half4(mla_sl_to_half_lo<T>(w.z), mla_sl_to_half_hi<T>(w.z),
                mla_sl_to_half_lo<T>(w.w), mla_sl_to_half_hi<T>(w.w));
}

template <typename T, int LATENT, int NSG>
kernel void mla_sparse_latent_mma(
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
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort tid   [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    ushort lane  [[thread_index_in_simdgroup]]) {
  constexpr int HG = 16;                  // heads per threadgroup (2 m-tiles)
  constexpr int MT = HG / 8;
  constexpr int CH = 8;                   // positions per chunk (one k-tile)
  constexpr int KS = LATENT / NSG;        // latent slice per simdgroup
  constexpr int KF = KS / 8;              // k-tiles per slice
  constexpr int LS = LATENT + 8;          // padded row stride (halves)
  constexpr int CPR = LATENT / 8;         // 16-byte chunks per row
  constexpr int NT = NSG * 32;
  constexpr int PF = (CH * CPR) / NT;     // prefetch vectors per thread
  constexpr int QPF = (HG * CPR) / NT;    // q staging vectors per thread
  static_assert((CH * CPR) % NT == 0, "chunk must tile the threadgroup");
  static_assert((HG * CPR) % NT == 0, "q must tile the threadgroup");
  static_assert(HG * LS <= 2 * CH * LS, "q staging must fit the chunk buffers");

  threadgroup half  sLat[2 * CH * LS];        // [2][CH][LS]; q staged here first
  threadgroup float sS[NSG][HG * CH];         // per-simdgroup S^T partials [h][p]
  threadgroup half  sP[HG * 16];              // P [h][p] (stride 16)
  threadgroup float sD[MT * 64];              // diag(alpha) per m-tile
  threadgroup int   vflag[2][CH];
  threadgroup int   sMoved;

  const int hg = (int)tgid.x;
  const int row = (int)tgid.y;
  const int part = (int)tgid.z;
  const int lim = has_tlen ? min(width, max(0, tlen[row])) : width;
  const int per_part = (lim + partitions - 1) / partitions;
  const int j0 = part * per_part;
  const int j1 = min(lim, j0 + per_part);
  const int head0 = hg * HG;
  const int s = (int)sgitg;

  // ---- stage Q[16 heads][LATENT] once, hold this slice's fragments ----
  {
    device const uint4 *qsrc =
        (device const uint4 *)(q + ((long)row * num_heads + head0) * LATENT);
    #pragma clang loop unroll(full)
    for (int i = 0; i < QPF; ++i) {
      const int c = (int)tid + i * NT;
      const int h = c / CPR, cc = c - h * CPR;
      mla_sl_store_half8<T>(sLat + h * LS + cc * 8, qsrc[h * (CPR) + cc]);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  simdgroup_half8x8 Qf[MT][KF];
  #pragma clang loop unroll(full)
  for (int mt = 0; mt < MT; ++mt) {
    #pragma clang loop unroll(full)
    for (int kf = 0; kf < KF; ++kf) {
      simdgroup_load(Qf[mt][kf], sLat + (mt * 8) * LS + s * KS + kf * 8, LS,
                     ulong2(0, 0), false);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  simdgroup_float8x8 O[MT][KF];
  #pragma clang loop unroll(full)
  for (int mt = 0; mt < MT; ++mt) {
    #pragma clang loop unroll(full)
    for (int nt = 0; nt < KF; ++nt) {
      O[mt][nt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
  }
  // softmax owner threads: tid < 128 -> head h = tid / 8, position p = tid % 8
  const int sh = (int)tid >> 3, sp = (int)tid & 7;
  float m_run = -3.4028234663852886e38f, l_run = 0.0f;

  device const int *idx_row = indices + (long)row * width;
  device const int *bt_row = block_table + (long)row * bt_stride;
  uint4 pf[PF];
  int pflag[PF];
#define MLA_SL_MMA_FETCH(jb_)                                                   \
  {                                                                             \
    _Pragma("clang loop unroll(full)")                                          \
    for (int i = 0; i < PF; ++i) {                                              \
      const int c = (int)tid + i * NT;                                          \
      const int p = c / CPR;                                                    \
      const int cc = c - p * CPR;                                               \
      const int j = (jb_) + p;                                                  \
      int ok = 0;                                                               \
      device const T *lat_row = cache;                                          \
      if (j < j1) {                                                             \
        const int t = idx_row[j];                                               \
        if (t >= 0) {                                                           \
          const int block_col = t / block_size;                                 \
          if (block_col <= max_block_col) {                                     \
            const int block = bt_row[block_col];                                \
            if (block >= 0) {                                                   \
              ok = 1;                                                           \
              lat_row = cache + (long)block * block_stride +                    \
                        (long)(t - block_col * block_size) * LATENT;            \
            }                                                                   \
          }                                                                     \
        }                                                                       \
      }                                                                         \
      pflag[i] = ok;                                                            \
      pf[i] = ok ? ((device const uint4 *)lat_row)[cc] : uint4(0u);             \
    }                                                                           \
  }
#define MLA_SL_MMA_STORE(buf_)                                                  \
  {                                                                             \
    _Pragma("clang loop unroll(full)")                                          \
    for (int i = 0; i < PF; ++i) {                                              \
      const int c = (int)tid + i * NT;                                          \
      const int p = c / CPR;                                                    \
      const int cc = c - p * CPR;                                               \
      mla_sl_store_half8<T>(sLat + ((buf_) * CH + p) * LS + cc * 8, pf[i]);     \
      if (cc == 0) vflag[(buf_)][p] = pflag[i];                                 \
    }                                                                           \
  }

  if (j0 < j1) {
    MLA_SL_MMA_FETCH(j0);
    MLA_SL_MMA_STORE(0);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  int cur = 0;
  for (int jb = j0; jb < j1; jb += CH, cur ^= 1) {
    const bool more = (jb + CH) < j1;
    if (more) { MLA_SL_MMA_FETCH(jb + CH); }
    threadgroup const half *lat = sLat + cur * CH * LS;

    // ---- S^T partial [16 heads x 8 pos] over this simdgroup's K slice ----
    simdgroup_float8x8 Sp[MT];
    #pragma clang loop unroll(full)
    for (int mt = 0; mt < MT; ++mt) Sp[mt] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    #pragma clang loop unroll(full)
    for (int kf = 0; kf < KF; ++kf) {
      simdgroup_half8x8 b;   // Lat^T slice: [k][pos], lat is [pos][k] -> transposed load
      simdgroup_load(b, lat + s * KS + kf * 8, LS, ulong2(0, 0), true);
      #pragma clang loop unroll(full)
      for (int mt = 0; mt < MT; ++mt) simdgroup_multiply_accumulate(Sp[mt], Qf[mt][kf], b, Sp[mt]);
    }
    #pragma clang loop unroll(full)
    for (int mt = 0; mt < MT; ++mt) {
      simdgroup_store(Sp[mt], sS[s] + mt * 8 * CH, CH, ulong2(0, 0), false);
    }
    if (tid == 0) sMoved = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- online softmax over the 8 positions, one thread per (head, pos) ----
    if (tid < HG * CH) {
      const bool valid = (sp < min(CH, j1 - jb)) && vflag[cur][sp] != 0;
      float sc = 0.0f;
      #pragma clang loop unroll(full)
      for (int ss = 0; ss < NSG; ++ss) sc += sS[ss][sh * CH + sp];
      sc = valid ? sc * scale : -3.4028234663852886e38f;
      float cmax = sc;
      cmax = max(cmax, simd_shuffle_xor(cmax, 1));
      cmax = max(cmax, simd_shuffle_xor(cmax, 2));
      cmax = max(cmax, simd_shuffle_xor(cmax, 4));
      const float new_m = max(m_run, cmax);
      const bool any = new_m > -3.0e38f;
      // No valid position in this chunk (and none before): nothing moves
      // (alpha 1, p 0). First valid chunk: alpha 0 (O and l are zero).
      const float alpha = !any ? 1.0f : (l_run == 0.0f ? 0.0f : exp(m_run - new_m));
      const float pv = (valid && any) ? exp(sc - new_m) : 0.0f;
      float psum = pv;
      psum += simd_shuffle_xor(psum, 1);
      psum += simd_shuffle_xor(psum, 2);
      psum += simd_shuffle_xor(psum, 4);
      sP[sh * 16 + sp] = half(pv);
      // All 8 lanes of a head hold identical (m, l) after the shuffles, so
      // every lane advances its own copy (the owner lane alone would leave
      // the other seven with a stale running max).
      if (sp == 0 && alpha != 1.0f) sMoved = 1;   // alpha == 1: max unmoved
      l_run = l_run * alpha + psum;
      m_run = any ? new_m : m_run;
      // diag(alpha) for the rescale: element (sp, sp) of head sh's tile
      const int mt = sh >> 3, r8 = sh & 7;
      sD[mt * 64 + r8 * 8 + sp] = (sp == r8) ? alpha : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- rescale O by diag(alpha) when any head's max moved ----
    if (sMoved) {
      #pragma clang loop unroll(full)
      for (int mt = 0; mt < MT; ++mt) {
        simdgroup_float8x8 dg;
        simdgroup_load(dg, sD + mt * 64, 8, ulong2(0, 0), false);
        #pragma clang loop unroll(full)
        for (int nt = 0; nt < KF; ++nt) {
          simdgroup_float8x8 t;
          simdgroup_multiply(t, dg, O[mt][nt]);
          O[mt][nt] = t;
        }
      }
    }
    // ---- O[16 x KS] += P[16 x 8] . Lat[8 x KS] ----
    simdgroup_half8x8 Pf[MT];
    #pragma clang loop unroll(full)
    for (int mt = 0; mt < MT; ++mt) simdgroup_load(Pf[mt], sP + mt * 8 * 16, 16, ulong2(0, 0), false);
    #pragma clang loop unroll(full)
    for (int nt = 0; nt < KF; ++nt) {
      simdgroup_half8x8 b;   // Lat [pos][cols] as [k][n]
      simdgroup_load(b, lat + s * KS + nt * 8, LS, ulong2(0, 0), false);
      #pragma clang loop unroll(full)
      for (int mt = 0; mt < MT; ++mt) simdgroup_multiply_accumulate(O[mt][nt], Pf[mt], b, O[mt][nt]);
    }
    if (more) { MLA_SL_MMA_STORE(cur ^ 1); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
#undef MLA_SL_MMA_FETCH
#undef MLA_SL_MMA_STORE

  // ---- partials: O rows are heads (stride P*LATENT floats), m/l per head ----
  {
    const long pstride = (long)partitions * LATENT;
    device float *base = part_acc +
        (((long)row * num_heads + head0) * partitions + part) * LATENT + s * KS;
    #pragma clang loop unroll(full)
    for (int mt = 0; mt < MT; ++mt) {
      #pragma clang loop unroll(full)
      for (int nt = 0; nt < KF; ++nt) {
        simdgroup_store(O[mt][nt], base + (long)(mt * 8) * pstride + nt * 8,
                        (ulong)pstride, ulong2(0, 0), false);
      }
    }
    if (tid < HG * CH && sp == 0) {
      const long pbase = (((long)row * num_heads + head0 + sh) * partitions + part);
      part_ml[pbase * 2] = m_run;
      part_ml[pbase * 2 + 1] = l_run;
    }
  }
}

#define instantiate_mla_sparse_latent_mma(type_name, T, LVAL, SV)                 \
  template [[host_name("mla_sparse_latent_mma_" #type_name "_" #LVAL "_s" #SV)]] \
  [[kernel]] void mla_sparse_latent_mma<T, LVAL, SV>(                            \
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
      ushort tid [[thread_index_in_threadgroup]],                               \
      ushort sgitg [[simdgroup_index_in_threadgroup]],                          \
      ushort lane [[thread_index_in_simdgroup]]);

instantiate_mla_sparse_latent_mma(bfloat16, bf16, 512, 4)
instantiate_mla_sparse_latent_mma(bfloat16, bf16, 512, 8)
instantiate_mla_sparse_latent_mma(float16, half, 512, 4)
instantiate_mla_sparse_latent_mma(float16, half, 512, 8)
