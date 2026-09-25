#include <metal_stdlib>

using namespace metal;

// ---------------------------------------------------------------------------
// Per-step KV metadata for the Metal runner.
//
// kv_meta_prepare: one launch replaces the per-group block-table gather
// (zero + index_select + copy) and the slot-mapping arithmetic (floor_div,
// remainder, gather, casts, mul, add, copy) that the torch path issued for
// every KV-cache group on every step (~10 launches per group; GLM-5.3-Flash
// has six groups). Integer-exact by construction.
//
// Grid: x = group, y in [0, num_reqs_padded) copies batch row y of the
// group's block table (zeros for padded rows); y >= num_reqs_padded handles
// a 256-token chunk of the slot mapping. Slot arithmetic mirrors
// BlockTables.compute_slot_mappings (cp_size == 1): the token's request is
// the CSR segment of query_start_loc that contains it, its block index
// positions[t] / block_size (0 when the group's slot mapping is disabled),
// slot = block_table[req, block_index] * block_size + positions[t] %
// block_size; tokens in [num_tokens, num_tokens_padded) get PAD (-1).
//
// mamba_last_blocks: the "align" mamba cache mode's tail-block gather
// (utils.mamba_get_block_table_tensor): out[r, j] = block_table[r,
// max((seq_lens[r] - 1) / block_size, 0) + j].
// ---------------------------------------------------------------------------

constant constexpr int KV_META_MAX_GROUPS = 8;
constant constexpr int KV_META_PARAMS = 8;   // ints per group
constant constexpr int KV_META_THREADS = 256;
constant constexpr long KV_META_PAD_SLOT = -1;

inline device const int *kv_meta_src(int g, device const int *s0,
    device const int *s1, device const int *s2, device const int *s3,
    device const int *s4, device const int *s5, device const int *s6,
    device const int *s7) {
  switch (g) {
    case 0: return s0; case 1: return s1; case 2: return s2; case 3: return s3;
    case 4: return s4; case 5: return s5; case 6: return s6; default: return s7;
  }
}

inline device int *kv_meta_dst(int g, device int *d0, device int *d1,
    device int *d2, device int *d3, device int *d4, device int *d5,
    device int *d6, device int *d7) {
  switch (g) {
    case 0: return d0; case 1: return d1; case 2: return d2; case 3: return d3;
    case 4: return d4; case 5: return d5; case 6: return d6; default: return d7;
  }
}

template <typename P>
kernel void kv_meta_prepare(
    device const int *idx_mapping [[buffer(0)]],   // [num_reqs]
    device const int *qsl         [[buffer(1)]],   // [num_reqs + 1]
    device const P   *positions   [[buffer(2)]],   // [>= num_tokens]
    device const int *params      [[buffer(3)]],   // [groups, 8]: src_stride, dst_stride, cols, block_size, enabled
    device long      *slots       [[buffer(4)]],   // [groups, slot_stride]
    device const int *s0 [[buffer(5)]],  device const int *s1 [[buffer(6)]],
    device const int *s2 [[buffer(7)]],  device const int *s3 [[buffer(8)]],
    device const int *s4 [[buffer(9)]],  device const int *s5 [[buffer(10)]],
    device const int *s6 [[buffer(11)]], device const int *s7 [[buffer(12)]],
    device int *d0 [[buffer(13)]], device int *d1 [[buffer(14)]],
    device int *d2 [[buffer(15)]], device int *d3 [[buffer(16)]],
    device int *d4 [[buffer(17)]], device int *d5 [[buffer(18)]],
    device int *d6 [[buffer(19)]], device int *d7 [[buffer(20)]],
    constant int &num_reqs          [[buffer(21)]],
    constant int &num_reqs_padded   [[buffer(22)]],
    constant int &num_tokens        [[buffer(23)]],
    constant int &num_tokens_padded [[buffer(24)]],
    constant int &slot_stride       [[buffer(25)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
  const int g = (int)tgid.x;
  device const int *p = params + g * KV_META_PARAMS;
  const int src_stride = p[0];
  const int dst_stride = p[1];
  const int cols = p[2];
  const int block_size = p[3];
  const int enabled = p[4];
  device const int *src = kv_meta_src(g, s0, s1, s2, s3, s4, s5, s6, s7);
  const int y = (int)tgid.y;
  if (y < num_reqs_padded) {
    device int *dst = kv_meta_dst(g, d0, d1, d2, d3, d4, d5, d6, d7) +
                      (long)y * dst_stride;
    const int ridx = y < num_reqs ? idx_mapping[y] : -1;
    if (ridx < 0) {
      for (int c = (int)tid; c < cols; c += KV_META_THREADS) dst[c] = 0;
    } else {
      device const int *row = src + (long)ridx * src_stride;
      for (int c = (int)tid; c < cols; c += KV_META_THREADS) dst[c] = row[c];
    }
    return;
  }
  const int t = (y - num_reqs_padded) * KV_META_THREADS + (int)tid;
  if (t >= num_tokens_padded) return;
  device long *slot_row = slots + (long)g * slot_stride;
  if (t >= num_tokens) {
    slot_row[t] = KV_META_PAD_SLOT;
    return;
  }
  // Largest req with qsl[req] <= t (qsl[0] == 0, ascending, no trailing
  // empty segment among the scheduled requests).
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (qsl[mid] <= t) lo = mid; else hi = mid - 1;
  }
  const int ridx = idx_mapping[lo];
  const long pos = (long)positions[t];
  const long bi = enabled ? pos / block_size : 0;
  const long bo = pos % block_size;
  const long bn = (long)src[(long)ridx * src_stride + bi];
  slot_row[t] = bn * block_size + bo;
}

#define instantiate_kv_meta_prepare(suffix, P)                                   \
  template [[host_name("kv_meta_prepare_" #suffix)]] [[kernel]] void            \
  kv_meta_prepare<P>(                                                            \
      device const int *idx_mapping [[buffer(0)]],                               \
      device const int *qsl [[buffer(1)]], device const P *positions [[buffer(2)]], \
      device const int *params [[buffer(3)]], device long *slots [[buffer(4)]],  \
      device const int *s0 [[buffer(5)]], device const int *s1 [[buffer(6)]],    \
      device const int *s2 [[buffer(7)]], device const int *s3 [[buffer(8)]],    \
      device const int *s4 [[buffer(9)]], device const int *s5 [[buffer(10)]],   \
      device const int *s6 [[buffer(11)]], device const int *s7 [[buffer(12)]],  \
      device int *d0 [[buffer(13)]], device int *d1 [[buffer(14)]],              \
      device int *d2 [[buffer(15)]], device int *d3 [[buffer(16)]],              \
      device int *d4 [[buffer(17)]], device int *d5 [[buffer(18)]],              \
      device int *d6 [[buffer(19)]], device int *d7 [[buffer(20)]],              \
      constant int &num_reqs [[buffer(21)]],                                     \
      constant int &num_reqs_padded [[buffer(22)]],                              \
      constant int &num_tokens [[buffer(23)]],                                   \
      constant int &num_tokens_padded [[buffer(24)]],                            \
      constant int &slot_stride [[buffer(25)]],                                  \
      uint3 tgid [[threadgroup_position_in_grid]],                               \
      uint tid [[thread_index_in_threadgroup]]);

instantiate_kv_meta_prepare(i64, long)
instantiate_kv_meta_prepare(i32, int)

kernel void mamba_last_blocks(
    device const int *block_table [[buffer(0)]],   // [rows, bt_stride]
    device const int *seq_lens    [[buffer(1)]],   // [rows]
    device int       *out         [[buffer(2)]],   // [rows, ncols]
    constant int &bt_stride       [[buffer(3)]],
    constant int &block_size      [[buffer(4)]],
    constant int &ncols           [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
  const int r = (int)tgid.x;
  const int start = max((seq_lens[r] - 1) / block_size, 0);
  device const int *row = block_table + (long)r * bt_stride + start;
  device int *o = out + (long)r * ncols;
  for (int j = (int)tid; j < ncols; j += 32) o[j] = row[j];
}
