#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// MoE router: scoring + optional correction bias + top-k + renormalize +
// scale, one simdgroup per token (GLM-5.3-Flash: sigmoid scores, 288
// experts, top-8, bias-selected / unbiased-weighted, renormalized, x2.5).
//
// Replaces the ~16-op torch `grouped_topk` chain for the single-group case
// (num_expert_group <= 1): sigmoid/softmax, + bias, topk on the biased
// scores, gather the ORIGINAL scores as weights, / sum, * scale.
//
// Each lane owns experts e = lane + 32*i (E <= MAX_E). Selection is K rounds
// of (lane-local max, simd max with lowest-index tie-break), which is
// torch.topk's first-occurrence order for ties. Outputs in selection order
// (descending biased score).
// ---------------------------------------------------------------------------

constant constexpr int QC_ROUTER_MAX_E = 1024;
constant constexpr int QC_ROUTER_PER_LANE = QC_ROUTER_MAX_E / 32;
constant constexpr int QC_ROUTER_MAX_K = 32;

template <typename T>
kernel void moe_router_topk(
    device const T     *logits   [[buffer(0)]],   // [T, E] (row stride)
    device const float *bias     [[buffer(1)]],   // [E] (read iff has_bias)
    device float       *out_w    [[buffer(2)]],   // [T, K] fp32
    device int         *out_ids  [[buffer(3)]],   // [T, K] int32
    constant int   &E            [[buffer(4)]],
    constant int   &K            [[buffer(5)]],
    constant int   &row_stride   [[buffer(6)]],
    constant int   &scoring      [[buffer(7)]],   // 0 sigmoid, 1 softmax
    constant int   &has_bias     [[buffer(8)]],
    constant int   &renormalize  [[buffer(9)]],
    constant float &scale        [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
  const int token = (int)tgid.x;
  device const T *row = logits + (long)token * row_stride;

  float score[QC_ROUTER_PER_LANE];   // original scores (routing weights)
  float sel[QC_ROUTER_PER_LANE];     // biased scores (selection key)
  float x[QC_ROUTER_PER_LANE];
  float m = -INFINITY;
  for (int i = 0; i < QC_ROUTER_PER_LANE; ++i) {
    const int e = (int)lane + 32 * i;
    x[i] = (e < E) ? float(row[e]) : -INFINITY;
    m = max(m, x[i]);
  }
  if (scoring == 1) {
    m = simd_max(m);
    float s = 0.0f;
    for (int i = 0; i < QC_ROUTER_PER_LANE; ++i) {
      const int e = (int)lane + 32 * i;
      x[i] = (e < E) ? exp(x[i] - m) : 0.0f;
      s += x[i];
    }
    s = simd_sum(s);
    for (int i = 0; i < QC_ROUTER_PER_LANE; ++i) score[i] = x[i] / s;
  } else {
    for (int i = 0; i < QC_ROUTER_PER_LANE; ++i) {
      score[i] = 1.0f / (1.0f + exp(-x[i]));
    }
  }
  for (int i = 0; i < QC_ROUTER_PER_LANE; ++i) {
    const int e = (int)lane + 32 * i;
    sel[i] = (e < E) ? (has_bias ? score[i] + bias[e] : score[i]) : -INFINITY;
  }

  float w_k = 0.0f;   // lane k holds the k-th selection
  int id_k = -1;
  for (int k = 0; k < K; ++k) {
    float best = -INFINITY;
    int best_i = -1;
    for (int i = 0; i < QC_ROUTER_PER_LANE; ++i) {
      if (sel[i] > best) { best = sel[i]; best_i = i; }
    }
    const int best_e = (best_i >= 0) ? (int)lane + 32 * best_i : 0x7fffffff;
    const float gmax = simd_max(best);
    // lowest expert index among lanes holding the max
    const int cand = (best == gmax) ? best_e : 0x7fffffff;
    const int win_e = simd_min(cand);
    const float win_w = simd_broadcast(
        (best_i >= 0) ? score[best_i] : 0.0f, (ushort)(win_e & 31));
    if (win_e == best_e) { sel[best_i] = -INFINITY; }
    if ((int)lane == k) { w_k = win_w; id_k = win_e; }
  }
  if (renormalize) {
    const float s = simd_sum(((int)lane < K) ? w_k : 0.0f);
    w_k = w_k / s;
  }
  w_k *= scale;
  if ((int)lane < K) {
    out_w[(long)token * K + lane] = w_k;
    out_ids[(long)token * K + lane] = id_k;
  }
}

#define instantiate_moe_router(tname, T)                                      \
  template [[host_name("moe_router_topk_" #tname)]] [[kernel]] void          \
  moe_router_topk<T>(device const T *logits [[buffer(0)]],                    \
                     device const float *bias [[buffer(1)]],                  \
                     device float *out_w [[buffer(2)]],                       \
                     device int *out_ids [[buffer(3)]],                       \
                     constant int &E [[buffer(4)]], constant int &K [[buffer(5)]], \
                     constant int &row_stride [[buffer(6)]],                  \
                     constant int &scoring [[buffer(7)]],                     \
                     constant int &has_bias [[buffer(8)]],                    \
                     constant int &renormalize [[buffer(9)]],                 \
                     constant float &scale [[buffer(10)]],                    \
                     uint3 tgid [[threadgroup_position_in_grid]],             \
                     uint lane [[thread_index_in_simdgroup]]);

instantiate_moe_router(float32, float)
instantiate_moe_router(bfloat16, bfloat)
instantiate_moe_router(float16, half)
