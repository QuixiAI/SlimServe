#include <metal_stdlib>
using namespace metal;

// Kernels of the DSpine block drafter (tencent/Qwen3.8-27B-DSpine-b16): five
// full-attention layers over [anchor, mask, ...] whose context K/V come from
// fc(five target residuals), with the code bridge after every layer. The
// projections reuse the q38 weight kernels; these are the parts in between.
//
// Norm weights are stored as (w - 1), the convention of q38_norm_add.

// ---------------------------------------------------------------------------
// q38_d_tap: one target residual tap. o[row][0..D) = bf16(t + r) at a row
// stride of o_stride elements (the caller offsets `o` to the tap's column).
// dispatch (m, 1, 1) x 512.
// ---------------------------------------------------------------------------
[[host_name("q38_d_tap")]] kernel void q38_d_tap(
    device const bfloat* t [[buffer(0)]],
    device const bfloat* r [[buffer(1)]],
    device bfloat* o [[buffer(2)]],
    constant int& D [[buffer(3)]],
    constant int& has_res [[buffer(4)]],
    constant int& o_stride [[buffer(5)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]]) {
  const long base = (long)tg.x * D;
  for (int c = int(tid); c < D / 4; c += 512) {
    float4 v = float4(((device const bfloat4*)(t + base))[c]);
    if (has_res != 0) v += float4(((device const bfloat4*)(r + base))[c]);
    ((device bfloat4*)(o + (long)tg.x * o_stride))[c] = bfloat4(v);
  }
}

// ---------------------------------------------------------------------------
// q38_d_qkv: per (head, row) 128-wide vectors out of a projection row
// [q (nq heads) | k (nk) | v (nk)], one dispatch: q heads RMSNorm + rotary
// (pairs (i, i + 64); cos/sin rows fp32 [row][cos 64 | sin 64]) into
// Qd[row][head], k heads normed + rotated into Kd[slot[row]][head], v heads
// copied into Vd[slot[row]][head]. nq == 0 for context rows.
// dispatch (nq + 2 nk, m, 1) x 32.
// ---------------------------------------------------------------------------
struct q38_d_qkv_params {
  int src_stride, nq, nk, cs_row0;
  float eps;
};

[[host_name("q38_d_qkv")]] kernel void q38_d_qkv(
    device const bfloat* src [[buffer(0)]],
    device const bfloat* qw [[buffer(1)]],
    device const bfloat* kw [[buffer(2)]],
    device const float* cs [[buffer(3)]],
    device const int* slot [[buffer(4)]],
    device bfloat* Qd [[buffer(5)]],
    device bfloat* Kd [[buffer(6)]],
    device bfloat* Vd [[buffer(7)]],
    constant q38_d_qkv_params& P [[buffer(8)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const int head = int(tg.x), row = int(tg.y);
  const int l = int(lane) & 15;
  const bool live = lane < 16;
  const bool is_q = head < P.nq, is_v = head >= P.nq + P.nk;
  device const bfloat4* sp = (device const bfloat4*)(src + (long)row * P.src_stride + head * 128);
  float4 a = live ? float4(sp[l]) : float4(0.f);
  float4 b = live ? float4(sp[16 + l]) : float4(0.f);
  if (!is_v) {
    device const bfloat4* w = (device const bfloat4*)(is_q ? qw : kw);
    const float ss = simd_sum(dot(a, a) + dot(b, b));
    const float inv = rsqrt(ss / 128.0f + P.eps);
    a *= inv * (float4(w[l]) + 1.0f);
    b *= inv * (float4(w[16 + l]) + 1.0f);
    device const float4* cp = (device const float4*)(cs + (long)(P.cs_row0 + row) * 128);
    const float4 c = cp[l], s = cp[16 + l];
    const float4 ra = a * c - b * s;
    b = a * s + b * c;
    a = ra;
  }
  if (live) {
    device bfloat4* dp;
    if (is_q)
      dp = (device bfloat4*)(Qd + ((long)row * P.nq + head) * 128);
    else if (!is_v)
      dp = (device bfloat4*)(Kd + ((long)slot[row] * P.nk + (head - P.nq)) * 128);
    else
      dp = (device bfloat4*)(Vd + ((long)slot[row] * P.nk + (head - P.nq - P.nk)) * 128);
    dp[l] = bfloat4(a);
    dp[16 + l] = bfloat4(b);
  }
}

// ---------------------------------------------------------------------------
// Attention: 32 query heads on 8 KV heads, head dim 128, causal over the
// keys [lo, seq_len[row]) of a ring (slot = position & ring_mask). Same
// scheme as q38_attn.metal: one simdgroup per (key partition, KV head, row),
// K/V read once for the 4 queries of the KV head, online softmax.
//   q38_d_attn_part    grid (n_parts, kv_heads, m) x 32
//   q38_d_attn_reduce  grid (heads, m) x 32 -> O [m][heads][128] BF16
// ---------------------------------------------------------------------------
struct q38_d_attn_params {
  int m, heads, kv_heads, part_size, n_parts, lo, ring_mask;
  float scale;
  int out_half;
};

#define Q38D_GQ 4
#define Q38D_HD 128

[[host_name("q38_d_attn_part")]] kernel void q38_d_attn_part(
    device const bfloat* Q [[buffer(0)]],
    device const bfloat* K [[buffer(1)]],
    device const bfloat* V [[buffer(2)]],
    device const int* seq_len [[buffer(3)]],
    device float* PA [[buffer(4)]],
    device float* PM [[buffer(5)]],
    constant q38_d_attn_params& P [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const int part = int(tg.x), kvh = int(tg.y), row = int(tg.z);
  const int l4 = int(lane) * 4;
  const int j0 = P.lo + part * P.part_size;
  const int j1 = min(j0 + P.part_size, seq_len[row]);
  float4 qv[Q38D_GQ], acc[Q38D_GQ];
  float mx[Q38D_GQ], sw[Q38D_GQ];
#pragma clang loop unroll(full)
  for (int g = 0; g < Q38D_GQ; ++g) {
    qv[g] = float4(*(device const bfloat4*)(
                Q + ((long)(row * P.heads + kvh * Q38D_GQ + g) * Q38D_HD + l4))) * P.scale;
    acc[g] = float4(0.f);
    mx[g] = -INFINITY;
    sw[g] = 0.f;
  }
  const long kstride = (long)P.kv_heads * Q38D_HD;
  const long koff = (long)kvh * Q38D_HD + l4;
  for (int j = j0; j < j1; ++j) {
    const long s = (long)(j & P.ring_mask) * kstride + koff;
    const float4 k = float4(*(device const bfloat4*)(K + s));
    const float4 v = float4(*(device const bfloat4*)(V + s));
#pragma clang loop unroll(full)
    for (int g = 0; g < Q38D_GQ; ++g) {
      const float sc = simd_sum(dot(qv[g], k));
      if (sc > mx[g]) {
        const float f = exp(mx[g] - sc);
        sw[g] *= f;
        acc[g] *= f;
        mx[g] = sc;
      }
      const float wt = exp(sc - mx[g]);
      sw[g] += wt;
      acc[g] += wt * v;
    }
  }
#pragma clang loop unroll(full)
  for (int g = 0; g < Q38D_GQ; ++g) {
    const long u = ((long)(row * P.heads + kvh * Q38D_GQ + g) * P.n_parts + part);
    *(device float4*)(PA + u * Q38D_HD + l4) = acc[g];
    if (lane == 0) {
      PM[u * 2] = mx[g];
      PM[u * 2 + 1] = sw[g];
    }
  }
}

[[host_name("q38_d_attn_reduce")]] kernel void q38_d_attn_reduce(
    device const float* PA [[buffer(0)]],
    device const float* PM [[buffer(1)]],
    device bfloat* O [[buffer(2)]],
    constant q38_d_attn_params& P [[buffer(3)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const int head = int(tg.x), row = int(tg.y);
  const int l4 = int(lane) * 4;
  const long u0 = (long)(row * P.heads + head) * P.n_parts;
  float M = -INFINITY;
  for (int p = 0; p < P.n_parts; ++p) M = max(M, PM[(u0 + p) * 2]);
  float4 a = float4(0.f);
  float L = 0.f;
  for (int p = 0; p < P.n_parts; ++p) {
    const float f = exp(PM[(u0 + p) * 2] - M);
    L += PM[(u0 + p) * 2 + 1] * f;
    a += *(device const float4*)(PA + (u0 + p) * Q38D_HD + l4) * f;
  }
  const long eo = (long)(row * P.heads + head) * Q38D_HD + l4;
  if (P.out_half != 0)
    *(device half4*)((device half*)O + eo) = half4(a / L * 0.0625f);
  else
    *(device bfloat4*)(O + eo) = bfloat4(a / L);
}

// ---------------------------------------------------------------------------
// q38_d_attn1: the block's whole attention in one dispatch, straight from the
// q | k | v projection rows: q/k RMSNorm + rotary, causal attention over the
// context ring [lo, ctx_hi) plus the block rows up to its own, merge and
// output. The block's own K/V never go to the ring (they are replaced by
// context rows next cycle anyway). One threadgroup per (KV head, row), 8
// simdgroups: each takes an eighth of the context keys (block key j goes to
// simdgroup j & 7), accumulates the 4 queries of the KV head with an online
// softmax, and the partials meet in threadgroup memory. A lane holds dims
// (2l, 2l+1, 64+2l, 64+2l+1), the two rotary pairs it needs.
// Replaces q38_d_qkv + q38_d_attn_part + q38_d_attn_reduce (three dependent
// dispatches, ~45 us of latency each at these sizes).
// dispatch (kv_heads, m, 1) x 256.
// ---------------------------------------------------------------------------
struct q38_d_attn1_params {
  int heads, kv_heads, lo, ctx_hi, ring_mask, qkv_stride, cs_row0, out_half;
  float scale, eps;
};

static inline float4 q38d_ld(device const bfloat* p, int l) {
  const float2 a = float2(*(device const bfloat2*)(p + 2 * l));
  const float2 b = float2(*(device const bfloat2*)(p + 64 + 2 * l));
  return float4(a, b);
}
// RMSNorm (weight stored as w - 1) + rotary of one 128-vector held across the simdgroup
static inline float4 q38d_nr(float4 x, device const bfloat* w, device const float* cs, int l, float eps) {
  const float inv = rsqrt(simd_sum(dot(x, x)) / 128.0f + eps);
  x *= inv * (q38d_ld(w, l) + 1.0f);
  const float2 c = float2(cs[2 * l], cs[2 * l + 1]), sn = float2(cs[64 + 2 * l], cs[64 + 2 * l + 1]);
  return float4(x.xy * c - x.zw * sn, x.xy * sn + x.zw * c);
}

[[host_name("q38_d_attn1")]] kernel void q38_d_attn1(
    device const bfloat* qkv [[buffer(0)]],
    device const bfloat* qw [[buffer(1)]],
    device const bfloat* kw [[buffer(2)]],
    device const float* cs [[buffer(3)]],
    device const bfloat* K [[buffer(4)]],
    device const bfloat* V [[buffer(5)]],
    device bfloat* O [[buffer(6)]],
    constant q38_d_attn1_params& P [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float pm[8][4][2];
  threadgroup float pa[8][4][128];
  const int kvh = int(tg.x), row = int(tg.y), l = int(lane);
  const int qw_el = P.heads * 128, kvw = P.kv_heads * 128;
  device const bfloat* rowp = qkv + (long)row * P.qkv_stride;
  device const float* csr = cs + (long)(P.cs_row0 + row) * 128;
  float4 qv[4], acc[4];
  float mx[4], sw[4];
#pragma clang loop unroll(full)
  for (int g = 0; g < 4; ++g) {
    qv[g] = q38d_nr(q38d_ld(rowp + (kvh * 4 + g) * 128, l), qw, csr, l, P.eps) * P.scale;
    acc[g] = float4(0.f);
    mx[g] = -INFINITY;
    sw[g] = 0.f;
  }
  const int width = P.ctx_hi - P.lo;
  const int part = (width + 7) / 8;
  const int j0 = P.lo + int(sg) * part, j1 = min(j0 + part, P.ctx_hi);
  const long kstride = (long)kvw;
  for (int j = j0; j < j1; ++j) {
    const long so = (long)(j & P.ring_mask) * kstride + kvh * 128;
    const float4 k = q38d_ld(K + so, l), v = q38d_ld(V + so, l);
#pragma clang loop unroll(full)
    for (int g = 0; g < 4; ++g) {
      const float sc = simd_sum(dot(qv[g], k));
      if (sc > mx[g]) {
        const float f = exp(mx[g] - sc);
        sw[g] *= f;
        acc[g] *= f;
        mx[g] = sc;
      }
      const float wt = exp(sc - mx[g]);
      sw[g] += wt;
      acc[g] += wt * v;
    }
  }
  for (int j = int(sg); j <= row; j += 8) {  // block rows, from their projection rows
    device const bfloat* jp = qkv + (long)j * P.qkv_stride + qw_el + kvh * 128;
    const float4 k = q38d_nr(q38d_ld(jp, l), kw, cs + (long)(P.cs_row0 + j) * 128, l, P.eps);
    const float4 v = q38d_ld(jp + kvw, l);
#pragma clang loop unroll(full)
    for (int g = 0; g < 4; ++g) {
      const float sc = simd_sum(dot(qv[g], k));
      if (sc > mx[g]) {
        const float f = exp(mx[g] - sc);
        sw[g] *= f;
        acc[g] *= f;
        mx[g] = sc;
      }
      const float wt = exp(sc - mx[g]);
      sw[g] += wt;
      acc[g] += wt * v;
    }
  }
#pragma clang loop unroll(full)
  for (int g = 0; g < 4; ++g) {
    pa[sg][g][4 * l] = acc[g].x;
    pa[sg][g][4 * l + 1] = acc[g].y;
    pa[sg][g][4 * l + 2] = acc[g].z;
    pa[sg][g][4 * l + 3] = acc[g].w;
    if (l == 0) {
      pm[sg][g][0] = mx[g];
      pm[sg][g][1] = sw[g];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
#pragma clang loop unroll(full)
  for (int g = 0; g < 4; ++g) {
    float M = -INFINITY;
#pragma clang loop unroll(full)
    for (int s2 = 0; s2 < 8; ++s2) M = max(M, pm[s2][g][0]);
    float4 a = float4(0.f);
    float L = 0.f;
#pragma clang loop unroll(full)
    for (int s2 = 0; s2 < 8; ++s2) {
      const float f = exp(pm[s2][g][0] - M);  // an empty part: exp(-inf) = 0
      L += pm[s2][g][1] * f;
      a += float4(pa[s2][g][4 * l], pa[s2][g][4 * l + 1], pa[s2][g][4 * l + 2], pa[s2][g][4 * l + 3]) * f;
    }
    a /= L;
    const long eo = (long)(row * P.heads + kvh * 4 + g) * 128;
    if (P.out_half != 0) {
      *(device half2*)((device half*)O + eo + 2 * l) = half2(a.xy * 0.0625f);
      *(device half2*)((device half*)O + eo + 64 + 2 * l) = half2(a.zw * 0.0625f);
    } else {
      *(device bfloat2*)(O + eo + 2 * l) = bfloat2(a.xy);
      *(device bfloat2*)(O + eo + 64 + 2 * l) = bfloat2(a.zw);
    }
  }
}

// ---------------------------------------------------------------------------
// q38_d_dense: out[m][N] = bf16(x[m][K] @ w[N][K]^T), all BF16, fp32 sums;
// with has_r the input row is bf16(x + r). dispatch (N / 8, m, 1) x 32. K % 4 == 0.
// ---------------------------------------------------------------------------
[[host_name("q38_d_dense")]] kernel void q38_d_dense(
    device bfloat* out [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    device const bfloat* r [[buffer(5)]],
    constant int& has_r [[buffer(6)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  // 8 outputs per threadgroup (N % 8 == 0): one output each measured 85 us a
  // dispatch at N = 5120, most of it threadgroup turnover
  device const bfloat4* wp = (device const bfloat4*)(w + (long)tg.x * 8 * K);
  device const bfloat4* xp = (device const bfloat4*)(x + (long)tg.y * K);
  device const bfloat4* rp = (device const bfloat4*)(r + (long)tg.y * K);
  const int k4 = K / 4;
  float4 s[8];
#pragma clang loop unroll(full)
  for (int i = 0; i < 8; ++i) s[i] = float4(0.f);
  for (int c = int(lane); c < k4; c += 32) {
    float4 xv = float4(xp[c]);
    if (has_r != 0) xv = float4(bfloat4(xv + float4(rp[c])));
#pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) s[i] += float4(wp[c + i * k4]) * xv;
  }
#pragma clang loop unroll(full)
  for (int i = 0; i < 8; ++i) {
    const float t = simd_sum(s[i].x + s[i].y + s[i].z + s[i].w);
    if (lane == 0) out[(long)tg.y * N + tg.x * 8 + i] = bfloat(t);
  }
}

// q38_d_dense_f: the same product written as fp32 (out[m][N] float).
[[host_name("q38_d_dense_f")]] kernel void q38_d_dense_f(
    device float* out [[buffer(0)]],
    device const bfloat* w [[buffer(1)]],
    device const bfloat* x [[buffer(2)]],
    constant int& N [[buffer(3)]],
    constant int& K [[buffer(4)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  device const bfloat4* wp = (device const bfloat4*)(w + (long)tg.x * K);
  device const bfloat4* xp = (device const bfloat4*)(x + (long)tg.y * K);
  float4 s = float4(0.f);
  for (int c = int(lane); c < K / 4; c += 32) s += float4(wp[c]) * float4(xp[c]);
  const float t = simd_sum(s.x + s.y + s.z + s.w);
  if (lane == 0) out[(long)tg.y * N + tg.x] = t;
}

// ---------------------------------------------------------------------------
// Code bridge (bridge.py DSpineCodeBridge.__call__), one write per layer:
//   raw   = bf16(t + r)                     rms = sqrt(max(mean(raw^2), tiny))
//   proj  = raw / (rms + eps) @ read^T
//   codes = 32 * proj / (|proj| + eps), row 0 = 32 * codebook[anchor]
//   q | s | v = codes @ [G_q; G_p; A]^T
//   msg[t] = sigmoid(q[t] + s[t-1] + b) * v[t-1], msg[0] = 0
//   delta = msg @ write^T
//   out   = bf16(raw + rms * delta) (rows >= 1; row 0 = raw)
//   next  = bf16(float(bf16(out * rsqrt(mean(out^2) + eps))) * w_next)
// as four dispatches: q38_d_dense (read, on t + r), q38_d_bridge_msg,
// q38_d_dense (write), q38_d_bridge_write. All but the last are one small
// threadgroup per output element: a variant with one threadgroup per block
// row doing the dots in long loops measured 2.5 ms (whole bridge in it: 11 ms)
// against 0.3 here, because it leaves most of the GPU idle.
// ---------------------------------------------------------------------------

// q38_d_bridge_msg: codes -> q | s | v -> message, one threadgroup per
// (rank index j, block row t). The codes are the read projections P (taken on
// raw: the reference's 1 / (rms + eps) cancels in the normalization) scaled
// by 32 / |P|, so the scale is applied after the dots:
//   q = s1 * (G_q[j] . P[t]),  s = s0 * (G_p[j] . P[t-1]),  v = s0 * (A[j] . P[t-1])
// Row 1's predecessor is the anchor token: its s | v come from the packed
// table tsv[anchor] = (32 codebook[anchor]) @ [G_p; A]^T. Row 0 gets no message.
// dispatch (512, m, 1) x 32; C == 1024.
[[host_name("q38_d_bridge_msg")]] kernel void q38_d_bridge_msg(
    device const bfloat* proj [[buffer(0)]],      // [m][1024]
    device const bfloat* tsv [[buffer(1)]],       // [V][1024] = [s (512) | v (512)]
    device const bfloat* pw [[buffer(2)]],        // [1536][1024] = [G_q; G_p; A]
    device const bfloat* gate_b [[buffer(3)]],    // [512]
    device bfloat* q_o [[buffer(4)]],             // [m][512]
    device bfloat* msg_o [[buffer(5)]],           // [m][512]
    constant int& anchor [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]]) {
  const int j = int(tg.x), row = int(tg.y);
  if (row == 0) {
    if (lane == 0) {
      q_o[j] = bfloat(0.f);
      msg_o[j] = bfloat(0.f);
    }
    return;
  }
  device const bfloat4* p1 = (device const bfloat4*)(proj + (long)row * 1024);
  device const bfloat4* p0 = (device const bfloat4*)(proj + (long)(row - 1) * 1024);
  device const bfloat4* gq = (device const bfloat4*)(pw + (long)j * 1024);
  device const bfloat4* gp = (device const bfloat4*)(pw + (long)(512 + j) * 1024);
  device const bfloat4* ga = (device const bfloat4*)(pw + (long)(1024 + j) * 1024);
  float4 aq = float4(0.f), as = float4(0.f), av = float4(0.f), n1 = float4(0.f), n0 = float4(0.f);
#pragma clang loop unroll(full)
  for (int k = 0; k < 8; ++k) {
    const int c = int(lane) + k * 32;
    const float4 x1 = float4(p1[c]);
    aq += float4(gq[c]) * x1;
    n1 += x1 * x1;
    if (row > 1) {
      const float4 x0 = float4(p0[c]);
      as += float4(gp[c]) * x0;
      av += float4(ga[c]) * x0;
      n0 += x0 * x0;
    }
  }
  const float s1 = 32.0f / (sqrt(simd_sum(n1.x + n1.y + n1.z + n1.w)) + eps);
  const bfloat qb = bfloat(s1 * simd_sum(aq.x + aq.y + aq.z + aq.w));
  float sv, vv;
  if (row > 1) {
    const float s0 = 32.0f / (sqrt(simd_sum(n0.x + n0.y + n0.z + n0.w)) + eps);
    sv = float(bfloat(s0 * simd_sum(as.x + as.y + as.z + as.w)));
    vv = float(bfloat(s0 * simd_sum(av.x + av.y + av.z + av.w)));
  } else {
    sv = float(tsv[(long)anchor * 1024 + j]);
    vv = float(tsv[(long)anchor * 1024 + 512 + j]);
  }
  if (lane == 0) {
    q_o[(long)row * 512 + j] = qb;
    msg_o[(long)row * 512 + j] = bfloat(vv / (1.0f + exp(-(float(qb) + sv + float(gate_b[j])))));
  }
}

// q38_d_bridge_write: raw = bf16(t + r), its rms, out = bf16(raw + rms * delta)
// (row 0: out = raw), then the next norm. `next` gets a half * 2^-4 copy in
// `next_h` when has_h != 0; raw and rms are kept for the sampling tail.
// dispatch (m, 1, 1) x 512; D <= 8192.
[[host_name("q38_d_bridge_write")]] kernel void q38_d_bridge_write(
    device const bfloat* tin [[buffer(0)]],
    device const bfloat* delta [[buffer(1)]],
    device const bfloat* rin [[buffer(2)]],
    device const bfloat* w_next [[buffer(3)]],
    device bfloat* out [[buffer(4)]],
    device bfloat* next [[buffer(5)]],
    constant int& D [[buffer(6)]],
    constant float& eps [[buffer(7)]],
    device half* next_h [[buffer(8)]],
    constant int& has_h [[buffer(9)]],
    device bfloat* raw_o [[buffer(10)]],
    device float* rms_o [[buffer(11)]],
    uint3 tg [[threadgroup_position_in_grid]],
    ushort tid [[thread_index_in_threadgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    ushort lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[16];
  const long base = (long)tg.x * D;
  const int nchunks = D / 4;
  float4 s[4];
  float ss = 0.f;
#pragma clang loop unroll(full)
  for (int k = 0; k < 4; ++k) {
    const int c = int(tid) + k * 512;
    s[k] = float4(0.f);
    if (c < nchunks) {
      const bfloat4 v = bfloat4(float4(((device const bfloat4*)(tin + base))[c]) +
                                float4(((device const bfloat4*)(rin + base))[c]));
      ((device bfloat4*)(raw_o + base))[c] = v;
      s[k] = float4(v);
      ss += dot(s[k], s[k]);
    }
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tot = 0.f;
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; ++i) tot += part[i];
  const float rms = sqrt(max(tot / float(D), 1.1754943508222875e-38f));
  if (tid == 0) rms_o[tg.x] = rms;
  const float rm = tg.x == 0 ? 0.f : rms;
  threadgroup_barrier(mem_flags::mem_threadgroup);  // part is reused below
  ss = 0.f;
#pragma clang loop unroll(full)
  for (int k = 0; k < 4; ++k) {
    const int c = int(tid) + k * 512;
    if (c < nchunks) {
      const bfloat4 v = bfloat4(s[k] + rm * float4(((device const bfloat4*)(delta + base))[c]));
      ((device bfloat4*)(out + base))[c] = v;
      s[k] = float4(v);
      ss += dot(s[k], s[k]);
    }
  }
  ss = simd_sum(ss);
  if (lane == 0) part[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tot = 0.f;
#pragma clang loop unroll(full)
  for (int i = 0; i < 16; ++i) tot += part[i];
  const float inv = rsqrt(tot / float(D) + eps);
#pragma clang loop unroll(full)
  for (int k = 0; k < 4; ++k) {
    const int c = int(tid) + k * 512;
    if (c < nchunks) {
      const float4 y = float4(bfloat4(s[k] * inv));
      const float4 nv = y * (float4(((device const bfloat4*)w_next)[c]) + 1.0f);
      ((device bfloat4*)(next + base))[c] = bfloat4(nv);
      if (has_h != 0) ((device half4*)(next_h + base))[c] = half4(nv * 0.0625f);
    }
  }
}
