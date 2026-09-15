#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// GLM-5.3-Flash pooled indexer (decode + prefill rows) and paged row insert.
//
// pool_logits: one simdgroup per (pool, query row). The pool's KP=4 member
// rows are read from the paged [k(128) | gate(128)] cache through the
// block table; the gate + APE softmax over the members gives per-channel
// probabilities, the pooled key is rounded through T (as the Triton kernel
// feeds its bf16 dot), then scores_h = relu(scale * pk . q_h), logit =
// sum_h w_h * scores_h. Pools past the row's visible count store -inf.
//
// expand_topk: torch/Triton _expand_topk semantics: the selected pools
// (valid first, -1 pad) expand to KP tokens each, the incomplete tail
// pool's tokens follow the last valid pool, the rest of the row is -1.
//
// paged_row_insert: cache[slot // BS, slot % BS, :] = rows[t, :]; PAD slots
// (< 0) land on block 0 row 0 (the KV manager's null block), as the torch
// scatter does.
// ---------------------------------------------------------------------------

constant constexpr int QC_IDX_D = 128;
constant constexpr int QC_IDX_KP = 4;
constant constexpr int QC_IDX_MAX_H = 64;

template <typename T>
kernel void glm5_indexer_pool_logits(
    device const T     *q           [[buffer(0)]],   // [R, H, D]
    device const float *w           [[buffer(1)]],   // [R, H]
    device const float *ape         [[buffer(2)]],   // [KP, D]
    device const T     *cache       [[buffer(3)]],   // pages, page_stride elems
    device const int   *block_table [[buffer(4)]],   // [rows, bt_stride]
    device const int   *row_req     [[buffer(5)]],   // [R]
    device const int   *visible     [[buffer(6)]],   // [R]
    device float       *out         [[buffer(7)]],   // [R, max_pools]
    constant int   &max_pools       [[buffer(8)]],
    constant int   &bt_stride       [[buffer(9)]],
    constant int   &page_stride     [[buffer(10)]],
    constant int   &block_size      [[buffer(11)]],
    constant float &scale           [[buffer(12)]],
    constant int   &row_dim         [[buffer(13)]],
    constant int   &num_heads       [[buffer(14)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
  const int p = (int)tgid.x;
  const int r = (int)tgid.y;
  if (p >= max_pools) return;
  const int vis = visible[r];
  const int n_pools = vis / QC_IDX_KP;
  device float *dst = out + (long)r * max_pools + p;
  if (p >= n_pools) {
    if (lane == 0) *dst = -INFINITY;
    return;
  }
  device const int *bt_row = block_table + (long)row_req[r] * bt_stride;
  const int c0 = (int)lane * 4;
  float k[QC_IDX_KP][4], g[QC_IDX_KP][4];
  for (int m = 0; m < QC_IDX_KP; ++m) {
    const int tok = p * QC_IDX_KP + m;
    const int blk = bt_row[tok / block_size];
    device const T *base = cache + (long)blk * page_stride +
        (long)(tok % block_size) * row_dim;
    for (int j = 0; j < 4; ++j) {
      k[m][j] = float(base[c0 + j]);
      g[m][j] = float(base[QC_IDX_D + c0 + j]) + ape[m * QC_IDX_D + c0 + j];
    }
  }
  float pk[4];
  for (int j = 0; j < 4; ++j) {
    float mx = g[0][j];
    for (int m = 1; m < QC_IDX_KP; ++m) mx = max(mx, g[m][j]);
    float e[QC_IDX_KP], s = 0.0f;
    for (int m = 0; m < QC_IDX_KP; ++m) { e[m] = exp(g[m][j] - mx); s += e[m]; }
    float acc = 0.0f;
    for (int m = 0; m < QC_IDX_KP; ++m) acc += (e[m] / s) * k[m][j];
    pk[j] = float(T(acc));
  }
  float logit = 0.0f;
  device const T *q_r = q + (long)r * num_heads * QC_IDX_D;
  device const float *w_r = w + (long)r * num_heads;
  for (int h = 0; h < num_heads; ++h) {
    device const T *qh = q_r + h * QC_IDX_D + c0;
    float partial = pk[0] * float(qh[0]) + pk[1] * float(qh[1]) +
                    pk[2] * float(qh[2]) + pk[3] * float(qh[3]);
    const float sc = simd_sum(partial);
    logit += max(sc * scale, 0.0f) * w_r[h];
  }
  if (lane == 0) *dst = logit;
}

kernel void glm5_indexer_expand_topk(
    device const int *sel      [[buffer(0)]],   // [R, KSEL] pools, valid first
    device const int *visible  [[buffer(1)]],   // [R]
    device int       *out      [[buffer(2)]],   // [R, OUT_W]
    constant int &ksel         [[buffer(3)]],
    constant int &out_w        [[buffer(4)]],
    constant int &kp           [[buffer(5)]],
    device int       *tlen     [[buffer(6)]],   // [R] valid prefix length (opt)
    constant int &write_tlen   [[buffer(7)]],
    constant int &identity     [[buffer(8)]],   // 1: pool p = p (sel unread)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint3 tpt  [[threads_per_threadgroup]]) {
  const int r = (int)tgid.x;
  const int vis = visible[r];
  const int n_pools = vis / kp;
  const int tail_count = vis - n_pools * kp;
  const int tail_start = n_pools * kp;
  const int n_sel = min(n_pools, ksel);
  const int nt = (ksel * kp + kp - 1 < out_w) ? kp : kp - 1;
  const int tcol0 = n_sel * kp;
  // Every valid entry lies in [0, tcol0 + min(tail_count, nt)): the first
  // n_sel selected pools are the finite-logit ones (valid first) and the
  // tail slots past tail_count are -1. The sparse decode kernel bounds its
  // scan by this length instead of walking the whole padded row.
  if (write_tlen && tid == 0) tlen[r] = tcol0 + min(tail_count, nt);
  device const int *sel_r = sel + (long)r * ksel;
  device int *out_r = out + (long)r * out_w;
  for (int col = (int)tid; col < out_w; col += (int)tpt.x) {
    int val = -1;
    if (col < ksel * kp) {
      const int pool = identity ? (col / kp) : sel_r[col / kp];
      if (pool >= 0 && pool < n_pools) val = pool * kp + (col % kp);
    }
    if (col >= tcol0 && col < tcol0 + nt) {
      const int m = col - tcol0;
      val = (m < tail_count) ? tail_start + m : -1;
    }
    out_r[col] = val;
  }
}

template <typename T, typename I>
kernel void paged_row_insert(
    device const T   *rows        [[buffer(0)]],   // [T, row_dim] (row stride)
    device T         *cache       [[buffer(1)]],
    device const I   *slot_mapping[[buffer(2)]],   // [T] int32 or int64
    constant int &block_size      [[buffer(3)]],
    constant int &page_stride     [[buffer(4)]],
    constant int &row_dim         [[buffer(5)]],
    constant int &row_stride      [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint3 tpt  [[threads_per_threadgroup]]) {
  const int t = (int)tgid.x;
  long slot = (long)slot_mapping[t];
  if (slot < 0) slot = 0;
  const long blk = slot / block_size;
  const long off = slot - blk * block_size;
  device T *dst = cache + blk * page_stride + off * row_dim;
  device const T *src = rows + (long)t * row_stride;
  for (int d = (int)tid; d < row_dim; d += (int)tpt.x) dst[d] = src[d];
}

// glm5_indexer_pack: the indexer's decode-side glue after one fused
// [k | gate | weights] linear over hidden_states. One threadgroup per
// token: k (D values) gets the fp32 LayerNorm (biased variance, eps, affine)
// rounded to T, the gate is copied through, weights are widened to fp32 and
// scaled by n_heads^-0.5. Replaces float/layer_norm/to/cat/float/mul.
template <typename T>
kernel void glm5_indexer_pack(
    device const T     *fused   [[buffer(0)]],   // [T, 2D + H]
    device const float *norm_w  [[buffer(1)]],   // [D]
    device const float *norm_b  [[buffer(2)]],   // [D]
    device T           *packed  [[buffer(3)]],   // [T, 2D]
    device float       *weights [[buffer(4)]],   // [T, H]
    constant int   &D           [[buffer(5)]],
    constant int   &H           [[buffer(6)]],
    constant float &eps         [[buffer(7)]],
    constant float &scale       [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]],
    uint  sg   [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const int t = (int)tgid.x;
  device const T *row = fused + (long)t * (2 * D + H);
  device T *out_row = packed + (long)t * (2 * D);
  // D <= 256 (one value per thread; QC_IDX_D is 128).
  const float x = (int)tid < D ? float(row[tid]) : 0.0f;
  float s = simd_sum(x);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mean = 0.0f;
  for (int i = 0; i < 8; ++i) mean += part[i];
  mean /= (float)D;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float dx = (int)tid < D ? x - mean : 0.0f;
  s = simd_sum(dx * dx);
  if (lane == 0) part[sg] = s;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float var = 0.0f;
  for (int i = 0; i < 8; ++i) var += part[i];
  var /= (float)D;
  const float rstd = rsqrt(var + eps);
  if ((int)tid < D) {
    out_row[tid] = T(dx * rstd * norm_w[tid] + norm_b[tid]);
    out_row[D + tid] = row[D + tid];
  }
  if ((int)tid < H) {
    weights[(long)t * H + tid] = float(row[2 * D + tid]) * scale;
  }
}

#define instantiate_glm5_indexer(tname, T)                                      \
  template [[host_name("glm5_indexer_pool_logits_" #tname)]] [[kernel]] void   \
  glm5_indexer_pool_logits<T>(                                                  \
      device const T *q [[buffer(0)]], device const float *w [[buffer(1)]],     \
      device const float *ape [[buffer(2)]], device const T *cache [[buffer(3)]], \
      device const int *block_table [[buffer(4)]],                              \
      device const int *row_req [[buffer(5)]],                                  \
      device const int *visible [[buffer(6)]], device float *out [[buffer(7)]], \
      constant int &max_pools [[buffer(8)]], constant int &bt_stride [[buffer(9)]], \
      constant int &page_stride [[buffer(10)]],                                 \
      constant int &block_size [[buffer(11)]], constant float &scale [[buffer(12)]], \
      constant int &row_dim [[buffer(13)]], constant int &num_heads [[buffer(14)]], \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]]);                                 \
  template [[host_name("paged_row_insert_" #tname)]] [[kernel]] void           \
  paged_row_insert<T, int>(device const T *rows [[buffer(0)]],                  \
                      device T *cache [[buffer(1)]],                            \
                      device const int *slot_mapping [[buffer(2)]],             \
                      constant int &block_size [[buffer(3)]],                   \
                      constant int &page_stride [[buffer(4)]],                  \
                      constant int &row_dim [[buffer(5)]],                      \
                      constant int &row_stride [[buffer(6)]],                   \
                      uint3 tgid [[threadgroup_position_in_grid]],              \
                      uint tid [[thread_index_in_threadgroup]],                 \
                      uint3 tpt [[threads_per_threadgroup]]);                   \
  template [[host_name("paged_row_insert_" #tname "_i64")]] [[kernel]] void    \
  paged_row_insert<T, long>(device const T *rows [[buffer(0)]],                 \
                      device T *cache [[buffer(1)]],                            \
                      device const long *slot_mapping [[buffer(2)]],            \
                      constant int &block_size [[buffer(3)]],                   \
                      constant int &page_stride [[buffer(4)]],                  \
                      constant int &row_dim [[buffer(5)]],                      \
                      constant int &row_stride [[buffer(6)]],                   \
                      uint3 tgid [[threadgroup_position_in_grid]],              \
                      uint tid [[thread_index_in_threadgroup]],                 \
                      uint3 tpt [[threads_per_threadgroup]]);                   \
  template [[host_name("glm5_indexer_pack_" #tname)]] [[kernel]] void          \
  glm5_indexer_pack<T>(device const T *fused [[buffer(0)]],                     \
                      device const float *norm_w [[buffer(1)]],                 \
                      device const float *norm_b [[buffer(2)]],                 \
                      device T *packed [[buffer(3)]],                           \
                      device float *weights [[buffer(4)]],                      \
                      constant int &D [[buffer(5)]], constant int &H [[buffer(6)]], \
                      constant float &eps [[buffer(7)]],                        \
                      constant float &scale [[buffer(8)]],                      \
                      uint3 tgid [[threadgroup_position_in_grid]],              \
                      uint tid [[thread_index_in_threadgroup]],                 \
                      uint sg [[simdgroup_index_in_threadgroup]],               \
                      uint lane [[thread_index_in_simdgroup]]);

instantiate_glm5_indexer(bfloat16, bfloat)
instantiate_glm5_indexer(float16, half)
