// SPDX-License-Identifier: Apache-2.0
// Fused Qwen3-Next attention prep: per-head gated-q split + gemma QK-RMSNorm
// + partial NeoX RoPE + gate de-interleave, one dispatch per layer.
//
// Replaces the eager Metal chain (2 reshape copies + 2 gemma_rms_norm
// dispatches + the ~20-op torch mrope decomposition = ~25 dispatches/layer
// x 16 layers/step). Text-only serving makes the three mRoPE position rows
// identical, so the host passes the flat T row and plain NeoX rotation over
// the vLLM cos_sin_cache row [cos(R/2) || sin(R/2)] is value-exact.
//
// BIT-EXACT contract vs the current serving stack, piece by piece:
//  - norm mirrors qc_rms_norm's gemma body EXACTLY: 256 threads/row, one
//    element per thread at D=256, simd_sum, sequential 8-slot threadgroup
//    total, rrms = rsqrt(total/D + eps), y = T(float(x)*rrms*(1+float(w)))
//    — same reduction tree, same single final round.
//  - rope mirrors the eager torch chain per op (MPS elementwise = fp32
//    compute, one round per op) ON THE ROUNDED bf16 norm outputs:
//    t1=T(x1*c); t2=T(x2*s); o1=T(t1-t2); u1=T(x2*c); u2=T(x1*s);
//    o2=T(u1+u2). Dims >= ROT pass through the normed value unchanged.
//  - gate is a pure bf16 copy from the per-head [q|gate] interleave to a
//    contiguous [T, Hq*D] buffer (the eager reshape's values, no rounding).
//
// qkv row layout: [Hq x (D q | D gate)] [Hk x D] [Hv x D]. Grid:
// (Hq + Hk, T, 1) threadgroups x 256 threads; q-head groups also copy
// their gate row; V stays a caller-side view of qkv.

#include <metal_stdlib>

using namespace metal;

namespace qc_qkr {
constant constexpr int THREADS = 256;
constant constexpr int SIMDGROUPS = THREADS / 32;
constant constexpr int MAX_HEAD_DIM = 256;  // yv[] threadgroup scratch
}  // namespace qc_qkr

template <typename T, typename IT>
inline void qk_norm_rope_gate_body(
    device const T* qkv, device const T* q_w, device const T* k_w,
    device const T* cos_sin, device const IT* positions, device T* q_out,
    device T* gate_out, device T* k_out, int num_q_heads, int num_k_heads,
    int head_dim, int rot_dim, float eps, int qkv_row, uint head, uint token,
    uint tid, uint sg, uint lane, threadgroup float* shm,
    threadgroup T* yv) {
  const bool is_q = (int)head < num_q_heads;
  const int D = head_dim;
  // yv is MAX_HEAD_DIM wide; the host rejects larger heads, and the whole
  // threadgroup returns here (before any barrier) if one gets through.
  if (D < 1 || D > qc_qkr::MAX_HEAD_DIM) return;
  const ulong row_base = (ulong)token * (ulong)qkv_row;
  device const T* x;
  device const T* w;
  device T* out;
  if (is_q) {
    x = qkv + row_base + (ulong)head * (2 * D);
    w = q_w;
    out = q_out + (ulong)token * (ulong)(num_q_heads * D) + (ulong)head * D;
  } else {
    const int kh = (int)head - num_q_heads;
    x = qkv + row_base + (ulong)num_q_heads * (2 * D) + (ulong)kh * D;
    w = k_w;
    out = k_out + (ulong)token * (ulong)(num_k_heads * D) + (ulong)kh * D;
  }

  // Gemma RMSNorm — the qc_rms_norm reduction tree, verbatim.
  float sq = 0.0f;
  for (int d = (int)tid; d < D; d += qc_qkr::THREADS) {
    const float v = float(x[d]);
    sq += v * v;
  }
  sq = simd_sum(sq);
  if (lane == 0) shm[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    float total = 0.0f;
    for (int i = 0; i < qc_qkr::SIMDGROUPS; ++i) total += shm[i];
    shm[0] = rsqrt(total / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float rrms = shm[0];

  for (int d = (int)tid; d < D; d += qc_qkr::THREADS) {
    yv[d] = T(float(x[d]) * rrms * (1.0f + float(w[d])));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Partial NeoX rotation over the rounded norm outputs; per-op rounding
  // mirrors the eager MPS elementwise chain.
  const int R2 = rot_dim / 2;
  const int pos = (int)positions[token];
  device const T* cs = cos_sin + (ulong)pos * (ulong)rot_dim;
  {
    #pragma clang fp reassociate(off) contract(off)
    for (int d = (int)tid; d < R2; d += qc_qkr::THREADS) {
      const float c = float(cs[d]);
      const float s = float(cs[R2 + d]);
      const float x1 = float(yv[d]);
      const float x2 = float(yv[R2 + d]);
      const T t1 = T(x1 * c);
      const T t2 = T(x2 * s);
      const T u1 = T(x2 * c);
      const T u2 = T(x1 * s);
      out[d] = T(float(t1) - float(t2));
      out[R2 + d] = T(float(u1) + float(u2));
    }
  }
  for (int d = rot_dim + (int)tid; d < D; d += qc_qkr::THREADS) {
    out[d] = yv[d];
  }

  if (is_q) {
    device const T* g = x + D;
    device T* go =
        gate_out + (ulong)token * (ulong)(num_q_heads * D) + (ulong)head * D;
    for (int d = (int)tid; d < D; d += qc_qkr::THREADS) {
      go[d] = g[d];
    }
  }
}

#define instantiate_qk_norm_rope_gate(tname, T, iname, IT)                  \
  [[host_name("qc_qk_norm_rope_gate_" #tname "_" #iname)]] kernel void       \
  qc_qk_norm_rope_gate_##tname##_##iname(                                    \
      device const T* qkv [[buffer(0)]],                                     \
      device const T* q_w [[buffer(1)]],                                     \
      device const T* k_w [[buffer(2)]],                                     \
      device const T* cos_sin [[buffer(3)]],                                 \
      device const IT* positions [[buffer(4)]],                              \
      device T* q_out [[buffer(5)]],                                         \
      device T* gate_out [[buffer(6)]],                                      \
      device T* k_out [[buffer(7)]],                                         \
      constant int& num_q_heads [[buffer(8)]],                               \
      constant int& num_k_heads [[buffer(9)]],                               \
      constant int& head_dim [[buffer(10)]],                                 \
      constant int& rot_dim [[buffer(11)]],                                  \
      constant float& eps [[buffer(12)]],                                    \
      constant int& qkv_row [[buffer(13)]],                                  \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint tid [[thread_index_in_threadgroup]],                              \
      uint sg [[simdgroup_index_in_threadgroup]],                            \
      uint lane [[thread_index_in_simdgroup]]) {                             \
    threadgroup float shm[qc_qkr::SIMDGROUPS];                               \
    threadgroup T yv[qc_qkr::MAX_HEAD_DIM];                                  \
    qk_norm_rope_gate_body<T, IT>(qkv, q_w, k_w, cos_sin, positions, q_out, \
                                  gate_out, k_out, num_q_heads,              \
                                  num_k_heads, head_dim, rot_dim, eps,       \
                                  qkv_row, tgid.x, tgid.y, tid, sg, lane,    \
                                  shm, yv);                                  \
  }

instantiate_qk_norm_rope_gate(bfloat16, bfloat, i64, long)
instantiate_qk_norm_rope_gate(bfloat16, bfloat, i32, int)
instantiate_qk_norm_rope_gate(float16, half, i64, long)
instantiate_qk_norm_rope_gate(float16, half, i32, int)
