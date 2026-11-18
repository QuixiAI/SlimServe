#include <metal_stdlib>
using namespace metal;

// q38 grouped-query attention over the contiguous KV cache.
//
// The model has 24 query heads on 4 KV heads (6 queries per KV head, head
// dim 256). The generic paged-attention kernel walks the cache once per query
// head, so K and V were read 6 times per row; at 6.6k tokens of context that
// was 8 ms of a 34 ms decode step and half of a prefill step. Here one
// simdgroup owns (key partition, KV head, row): each lane holds 8 of the 256
// dims for all 6 queries, K and V are read once, and the softmax is
// accumulated online (running max, rescale on a new max) so no score storage
// or second pass is needed.
//
//   q38_attn_part    grid (n_parts, kv_heads, m) x 32 threads (q38_attn_part_ms: rows of several sequences)
//     -> PA [m][heads][n_parts][256] fp32 unnormalized sum of w * v
//        PM [m][heads][n_parts][2]   fp32 (running max, sum of w)
//   q38_attn_reduce  grid (heads, m) x 32 threads
//     merges the partitions, normalizes, applies the sigmoid output gate and
//     writes BF16 (or half * 2^-4 for an XH output projection).
//
// Layouts: Q, G [m][heads][256] BF16; K, V [slot][kv_heads][256] BF16 with
// slot == position; seq_len[row] = keys visible to the row (causal).

struct q38_attn_params {
  int m;
  int heads;
  int kv_heads;
  int part_size;
  int n_parts;
  int out_half;
  float scale;
};

#define Q38_GQ 6   // queries per KV head
#define Q38_HD 256

static inline void q38_attn_part_body(device const bfloat* Q, device const bfloat* K,
                                      device const bfloat* V, device const int* seq_len,
                                      device float* PA, device float* PM,
                                      constant q38_attn_params& P, uint3 tg, ushort lane) {
  const int part = int(tg.x), kvh = int(tg.y), row = int(tg.z);
  const int l8 = int(lane) * 8;
  const int j0 = part * P.part_size;
  const int j1 = min(j0 + P.part_size, seq_len[row]);

  half qv[Q38_GQ][8];
  float acc[Q38_GQ][8];
  float mx[Q38_GQ], sw[Q38_GQ];
#pragma clang loop unroll(full)
  for (int g = 0; g < Q38_GQ; ++g) {
    device const bfloat4* qp =
        (device const bfloat4*)(Q + ((long)(row * P.heads + kvh * Q38_GQ + g) * Q38_HD + l8));
    const float4 a = float4(qp[0]) * P.scale, b = float4(qp[1]) * P.scale;
#pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
      qv[g][i] = half(a[i]);
      qv[g][4 + i] = half(b[i]);
      acc[g][i] = 0.f;
      acc[g][4 + i] = 0.f;
    }
    mx[g] = -INFINITY;
    sw[g] = 0.f;
  }

  const long kstride = (long)P.kv_heads * Q38_HD;
  const long koff = (long)kvh * Q38_HD + l8;
  for (int j = j0; j < j1; ++j) {
    device const bfloat4* kp = (device const bfloat4*)(K + (long)j * kstride + koff);
    device const bfloat4* vp = (device const bfloat4*)(V + (long)j * kstride + koff);
    const float4 k0 = float4(kp[0]), k1 = float4(kp[1]);
    const float4 v0 = float4(vp[0]), v1 = float4(vp[1]);
#pragma clang loop unroll(full)
    for (int g = 0; g < Q38_GQ; ++g) {
      float d = 0.f;
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) d += float(qv[g][i]) * k0[i] + float(qv[g][4 + i]) * k1[i];
      const float s = simd_sum(d);
      if (s > mx[g]) {  // new running max: rescale what is accumulated (exp(-inf) = 0 the first time)
        const float f = exp(mx[g] - s);
        sw[g] *= f;
#pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) acc[g][i] *= f;
        mx[g] = s;
      }
      const float w = exp(s - mx[g]);
      sw[g] += w;
#pragma clang loop unroll(full)
      for (int i = 0; i < 4; ++i) {
        acc[g][i] += w * v0[i];
        acc[g][4 + i] += w * v1[i];
      }
    }
  }

#pragma clang loop unroll(full)
  for (int g = 0; g < Q38_GQ; ++g) {
    const long u = ((long)(row * P.heads + kvh * Q38_GQ + g) * P.n_parts + part);
    device float4* o = (device float4*)(PA + u * Q38_HD + l8);
    o[0] = float4(acc[g][0], acc[g][1], acc[g][2], acc[g][3]);
    o[1] = float4(acc[g][4], acc[g][5], acc[g][6], acc[g][7]);
    if (lane == 0) {
      PM[u * 2] = mx[g];
      PM[u * 2 + 1] = sw[g];
    }
  }
}


[[host_name("q38_attn_part")]] kernel void q38_attn_part(
    device const bfloat* Q [[buffer(0)]],
    device const bfloat* K [[buffer(1)]],
    device const bfloat* V [[buffer(2)]],
    device const int* seq_len [[buffer(3)]],
    device float* PA [[buffer(4)]],
    device float* PM [[buffer(5)]],
    constant q38_attn_params& P [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  q38_attn_part_body(Q, K, V, seq_len, PA, PM, P, tg, lane);
}

// Rows of several sequences in one dispatch: each row reads the K/V buffers
// of its own sequence (sel[row] in 0..7). One dispatch per sequence measured
// 6 ms slower per step at 8 sequences (16 layers x 14 extra launches). The
// partition size is common, sized for the longest sequence; a shorter one
// leaves its tail partitions empty.
[[host_name("q38_attn_part_ms")]] kernel void q38_attn_part_ms(
    device const bfloat* Q [[buffer(0)]],
    device const bfloat* K0 [[buffer(1)]],
    device const bfloat* K1 [[buffer(2)]],
    device const bfloat* K2 [[buffer(3)]],
    device const bfloat* K3 [[buffer(4)]],
    device const bfloat* K4 [[buffer(5)]],
    device const bfloat* K5 [[buffer(6)]],
    device const bfloat* K6 [[buffer(7)]],
    device const bfloat* K7 [[buffer(8)]],
    device const bfloat* V0 [[buffer(9)]],
    device const bfloat* V1 [[buffer(10)]],
    device const bfloat* V2 [[buffer(11)]],
    device const bfloat* V3 [[buffer(12)]],
    device const bfloat* V4 [[buffer(13)]],
    device const bfloat* V5 [[buffer(14)]],
    device const bfloat* V6 [[buffer(15)]],
    device const bfloat* V7 [[buffer(16)]],
    device const int* seq_len [[buffer(17)]],
    device float* PA [[buffer(18)]],
    device float* PM [[buffer(19)]],
    constant q38_attn_params& P [[buffer(20)]],
    device const int* sel [[buffer(21)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const int g = sel[tg.z];
  device const bfloat* K = g == 0 ? K0 : g == 1 ? K1 : g == 2 ? K2 : g == 3 ? K3 : g == 4 ? K4 : g == 5 ? K5 : g == 6 ? K6 : K7;
  device const bfloat* V = g == 0 ? V0 : g == 1 ? V1 : g == 2 ? V2 : g == 3 ? V3 : g == 4 ? V4 : g == 5 ? V5 : g == 6 ? V6 : V7;
  q38_attn_part_body(Q, K, V, seq_len, PA, PM, P, tg, lane);
}

[[host_name("q38_attn_reduce")]] kernel void q38_attn_reduce(
    device const float* PA [[buffer(0)]],
    device const float* PM [[buffer(1)]],
    device const bfloat* G [[buffer(2)]],
    device bfloat* O [[buffer(3)]],
    constant q38_attn_params& P [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const int head = int(tg.x), row = int(tg.y);
  const int l8 = int(lane) * 8;
  const long u0 = (long)(row * P.heads + head) * P.n_parts;
  float M = -INFINITY;
  for (int p = 0; p < P.n_parts; ++p) M = max(M, PM[(u0 + p) * 2]);
  float4 a0 = float4(0.f), a1 = float4(0.f);
  float L = 0.f;
  for (int p = 0; p < P.n_parts; ++p) {
    const float f = exp(PM[(u0 + p) * 2] - M);  // empty partition: exp(-inf) = 0
    L += PM[(u0 + p) * 2 + 1] * f;
    device const float4* a = (device const float4*)(PA + (u0 + p) * Q38_HD + l8);
    a0 += a[0] * f;
    a1 += a[1] * f;
  }
  const float inv = 1.0f / L;
  const long e = (long)(row * P.heads + head) * Q38_HD + l8;
  device const bfloat4* gp = (device const bfloat4*)(G + e);
  const float4 g0 = float4(gp[0]), g1 = float4(gp[1]);
  const float4 r0 = a0 * inv / (1.0f + exp(-g0));
  const float4 r1 = a1 * inv / (1.0f + exp(-g1));
  if (P.out_half != 0) {
    device half4* o = (device half4*)((device half*)O + e);
    o[0] = half4(r0 * 0.0625f);
    o[1] = half4(r1 * 0.0625f);
  } else {
    device bfloat4* o = (device bfloat4*)(O + e);
    o[0] = bfloat4(r0);
    o[1] = bfloat4(r1);
  }
}
