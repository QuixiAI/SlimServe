// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

kernel void v2_logit_bias(
    device float* logits [[buffer(0)]],
    device const int* mapping [[buffer(1)]],
    device const long* positions [[buffer(2)]],
    device const int* allowed_count [[buffer(3)]],
    device const int* allowed_ids [[buffer(4)]],
    device const int* bias_count [[buffer(5)]],
    device const int* bias_ids [[buffer(6)]],
    device const float* biases [[buffer(7)]],
    device const int* min_lens [[buffer(8)]],
    device const int* stop_count [[buffer(9)]],
    device const int* stop_ids [[buffer(10)]],
    constant ulong* strides [[buffer(11)]],
    constant int& vocab [[buffer(12)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
  int req = mapping[row];
  if (req < 0) return;
  device float* values = logits + ulong(row) * strides[0];
  threadgroup float saved[1024];
  int count = allowed_count[req];
  if (count > 0) {
    for (int i = lane; i < count; i += 256) {
      int token = allowed_ids[ulong(req) * strides[1] + i];
      saved[i] = values[token];
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    for (int token = lane; token < vocab; token += 256) values[token] = -INFINITY;
    threadgroup_barrier(mem_flags::mem_device);
    for (int i = lane; i < count; i += 256) {
      int token = allowed_ids[ulong(req) * strides[1] + i];
      values[token] = saved[i];
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (int i = lane; i < bias_count[req]; i += 256) {
    int token = bias_ids[ulong(req) * strides[2] + i];
    values[token] += biases[ulong(req) * strides[3] + i];
  }
  threadgroup_barrier(mem_flags::mem_device);
  if (positions[row] + 1 < min_lens[req]) {
    for (int i = lane; i < stop_count[req]; i += 256) {
      int token = stop_ids[ulong(req) * strides[4] + i];
      values[token] = -INFINITY;
    }
  }
}
