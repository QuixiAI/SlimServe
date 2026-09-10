// SPDX-License-Identifier: Apache-2.0
#include <metal_stdlib>
using namespace metal;

kernel void v2_flatten_sampled(
    device long* output [[buffer(0)]],
    device const long* sampled [[buffer(1)]],
    device const int* counts [[buffer(2)]],
    device const int* offsets [[buffer(3)]],
    constant ulong& stride [[buffer(4)]],
    uint request [[threadgroup_position_in_grid]],
    uint lane [[thread_position_in_threadgroup]]) {
  int start = offsets[request];
  for (int column = lane; column < counts[request]; column += 32) {
    output[start + column] = sampled[ulong(request) * stride + column];
  }
}
