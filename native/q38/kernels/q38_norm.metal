#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// q38_norm_add: residual add + Gemma RMSNorm for one layer seam.
//
//   s       = x + residual            (fp32; x alone when has_res == 0)
//   res_out = bf16(s)
//   o       = bf16(s * rsqrt(mean(s^2) + eps) * (1 + w))
//
// Same math as mittens::gemma_rms_norm_add_dyn (statistic and normed value
// from the unrounded sum). That kernel walks a row with one simdgroup, 160
// iterations per pass, and measures ~60 us per launch on the M1 Ultra; the
// model has two seams per layer, so it was 8 ms of a 39 ms decode step.
// Here a row gets 512 threads, each owning at most Q38_NCH 4-element chunks
// kept in registers, and the square sum is reduced through threadgroup
// memory. dispatch (M, 1, 1) x 512 threads. Requires D % 4 == 0 and
// D <= 512 * 4 * Q38_NCH.
// ---------------------------------------------------------------------------
#define Q38_NCH 4

[[host_name("q38_norm_add")]] kernel void q38_norm_add(
    device const bfloat* x [[buffer(0)]],
    device const bfloat* residual [[buffer(1)]],
    device const bfloat* weight [[buffer(2)]],
    device bfloat* o [[buffer(3)]],
    device bfloat* res_out [[buffer(4)]],
    constant int& D [[buffer(5)]],
    constant float& eps [[buffer(6)]],
    constant int& has_res [[buffer(7)]],
    constant int& o_stride [[buffer(8)]],  // elements between output rows of `o`
    device half* oh [[buffer(9)]],         // optional copy of `o` as half * 2^-4 (XH GEMV input)
    constant int& has_oh [[buffer(10)]],
    device bfloat* tap [[buffer(11)]],        // optional copy of res_out, rows tap_stride elements apart
    constant int& tap_stride [[buffer(12)]],  // 0: none (the DSpine drafter's residual taps)
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[16];
  const long base = (long)tg.x * D;
  const int nchunks = D / 4;
  float4 s[Q38_NCH];
  float ss = 0.f;
#pragma clang loop unroll(full)
  for (int k = 0; k < Q38_NCH; ++k) {
    const int c = int(tid) + k * 512;
    s[k] = float4(0.f);
    if (c < nchunks) {
      float4 v = float4(((device const bfloat4*)(x + base))[c]);
      if (has_res != 0) v += float4(((device const bfloat4*)(residual + base))[c]);
      ((device bfloat4*)(res_out + base))[c] = bfloat4(v);
      if (tap_stride != 0) ((device bfloat4*)(tap + (long)tg.x * tap_stride))[c] = bfloat4(v);
      s[k] = v;
      ss += dot(v, v);
    }
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.f;
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; ++i) tot += part[i];
  const float inv = rsqrt(tot / float(D) + eps);
#pragma clang loop unroll(full)
  for (int k = 0; k < Q38_NCH; ++k) {
    const int c = int(tid) + k * 512;
    if (c < nchunks) {
      const float4 w = float4(((device const bfloat4*)weight)[c]) + 1.0f;
      const float4 nv = s[k] * inv * w;
      ((device bfloat4*)(o + (long)tg.x * o_stride))[c] = bfloat4(nv);
      if (has_oh != 0) ((device half4*)(oh + (long)tg.x * o_stride))[c] = half4(nv * 0.0625f);
    }
  }
}
