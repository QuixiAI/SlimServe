// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

// All request decisions are GPU-resident. Separate decision/copy dispatches
// keep accepted-count updates from racing readers across layers.
kernel void mamba_align_plan(
    device const int* mapping [[buffer(0)]],
    device int* state_idx [[buffer(1)]],
    device const int* computed [[buffer(2)]],
    device const int* query_start [[buffer(3)]],
    device int* accepted [[buffer(4)]],
    device int* src [[buffer(5)]], device int* dst [[buffer(6)]],
    device int* bias [[buffer(7)]], constant int& block_size [[buffer(8)]],
    constant int& n [[buffer(9)]], constant int& post [[buffer(10)]],
    uint row [[thread_position_in_grid]]) {
  if (row >= uint(n)) return;
  int req = mapping[row];
  if (req < 0) return;
  int old = state_idx[req];
  int count = accepted[req];
  src[req] = -1;
  if (post) {
    int running = computed[req] - count + 1;
    int aligned = computed[req] / block_size * block_size;
    int target = aligned / block_size - 1;
    if (aligned < running || target < 0) return;
    int shift = aligned - running;
    if (old == target) accepted[req] = 1;
    if (old == target && shift == 0) return;
    src[req] = old; dst[req] = target; bias[req] = shift;
  } else {
    int after = computed[req] + query_start[row + 1] - query_start[row];
    int target = (after + block_size - 1) / block_size - 1;
    state_idx[req] = target;
    if (old < 0 || old == target) return;
    src[req] = old; dst[req] = target; bias[req] = max(count - 1, 0);
    accepted[req] = 1;
  }
}

// kind: 0 temporal, 1 [block, time, channel], 2 [block, channel, time].
// A lane owns one byte of a channel across all time positions, so left
// shifts within the same conv block have memmove semantics without races.
kernel void mamba_align_copy(
    device uchar* state [[buffer(0)]], device const int* table [[buffer(1)]],
    device const int* mapping [[buffer(2)]], device const int* src [[buffer(3)]],
    device const int* dst [[buffer(4)]], device const int* bias [[buffer(5)]],
    constant ulong& block_stride [[buffer(6)]],
    constant ulong& row_stride [[buffer(7)]],
    constant ulong& copy_bytes [[buffer(8)]],
    constant int& width [[buffer(9)]], constant int& element_size [[buffer(10)]],
    constant int& kind [[buffer(11)]], constant ulong& table_stride [[buffer(12)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint2 position [[thread_position_in_threadgroup]]) {
  uint lane = position.x;
  int req = mapping[group.y];
  if (req < 0) return;
  int source = src[req], target = dst[req], shift = bias[req];
  if (source < 0 || target < 0) return;
  ulong table_row = ulong(group.y) * table_stride;
  long source_block = table[table_row + source + (kind == 0 ? shift : 0)];
  long target_block = table[table_row + target];
  if (source_block < 0 || target_block < 0) return;
  ulong source_base = ulong(source_block) * block_stride;
  ulong target_base = ulong(target_block) * block_stride;
  ulong begin = ulong(group.x) * 4096;
  ulong end = min(begin + 4096, copy_bytes);
  for (ulong i = begin + lane; i < end; i += 256) {
    if (kind == 0) {
      state[target_base + i] = state[source_base + i];
    } else {
      for (int t = 0; t < width - shift; ++t) {
        ulong from = kind == 1 ? ulong(t + shift) * row_stride + i
            : (i / element_size) * row_stride + ulong(t + shift) * element_size + i % element_size;
        ulong to = kind == 1 ? ulong(t) * row_stride + i
            : (i / element_size) * row_stride + ulong(t) * element_size + i % element_size;
        state[target_base + to] = state[source_base + from];
      }
    }
  }
}
