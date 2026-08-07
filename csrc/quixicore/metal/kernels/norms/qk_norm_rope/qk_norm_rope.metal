#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// ---------------------------------------------------------------------------
// Fused per-head QK-RMSNorm + RoPE over a packed QKV buffer (the Qwen3 / gpt-oss
// attention-prep pattern): qkv is (T, (Hq+Hk+Hv)*D); every Q and K head is RMSNormed
// over its D dims with q_weight/k_weight, rotated by RoPE at positions[token], and
// written to `out`; V heads are copied through unchanged — one dispatch, one pass,
// functional out-of-place (clean MLX purity, no clone prepass).
//
// One warp per (token, head): grid ((Hq+Hk+Hv), T, 1), 32 lanes.
// interleave = 0: NeoX split-half rotation (pair = d, d+D/2) via the rv_fl<D/2>
//   half-vector idiom of rope_kv_insert_norm. interleave = 1: GPT-J interleaved
//   (pair = 2i, 2i+1) via the contiguous-lane-chunk idiom of mla_q_norm_rope (pairs
//   are lane-local, no shuffles). gemma != 0 weights by (1 + w). Full rotary only
//   (rotary_dim == D). cos/sin are separate (max_pos, D/2) bf16 tables (TM RoPE
//   convention), positions int32.
// ---------------------------------------------------------------------------
template <int D>
kernel void qk_norm_rope(device const bf16 *qkv       [[buffer(0)]],
                         device const bf16 *q_weight  [[buffer(1)]],
                         device const bf16 *k_weight  [[buffer(2)]],
                         device const bf16 *cosb      [[buffer(3)]],
                         device const bf16 *sinb      [[buffer(4)]],
                         device const int  *positions [[buffer(5)]],
                         device bf16       *out       [[buffer(6)]],
                         constant int &num_heads_q    [[buffer(7)]],
                         constant int &num_heads_k    [[buffer(8)]],
                         constant int &num_heads_v    [[buffer(9)]],
                         constant float &eps          [[buffer(10)]],
                         constant int &interleave     [[buffer(11)]],
                         constant int &gemma          [[buffer(12)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");
    constexpr int PER_LANE = D / 32;           // contiguous chunk per lane (even for D>=64)
    const int head  = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int HT = num_heads_q + num_heads_k + num_heads_v;
    const int row = token * HT + head;         // (token, head) row of the packed layout

    using row_gl = gl<bf16, 1, 1, -1, D>;
    row_gl gl_in((device bf16 *)qkv, nullptr, nullptr, 1, nullptr);
    row_gl gl_out(out,               nullptr, nullptr, 1, nullptr);

    if (head >= num_heads_q + num_heads_k) {   // V head: copy-through
        using vecD = rv_fl<D>;
        vecD vv;
        load(vv, gl_in, {0, 0, row, 0}, laneId);
        store(gl_out, vv, {0, 0, row, 0}, laneId);
        return;
    }

    const bool is_q = head < num_heads_q;
    device const bf16 *w = is_q ? q_weight : k_weight;
    const int pos = positions[token];

    if (interleave == 0) {
        // NeoX split-half: the rv_fl<D/2> half-vector idiom (rope_kv_insert_norm).
        using cs_gl = gl<bf16, 1, 1, -1, D2>;
        cs_gl gl_c((device bf16 *)cosb, nullptr, nullptr, 1, nullptr);
        cs_gl gl_s((device bf16 *)sinb, nullptr, nullptr, 1, nullptr);
        cs_gl gl_w((device bf16 *)w,    nullptr, nullptr, 1, nullptr);
        using vecH = rv_fl<D2>;
        vecH x1, x2, w1, w2, cv, sv, o1, o2, tmp, sq;
        load(x1, gl_in, {0, 0, row, 0}, laneId);
        load(x2, gl_in, {0, 0, row, 1}, laneId);
        load(w1, gl_w, {0, 0, 0, 0}, laneId);
        load(w2, gl_w, {0, 0, 0, 1}, laneId);
        float ss1 = 0.f, ss2 = 0.f;
        mul(sq, x1, x1); sum(ss1, sq, laneId);
        mul(sq, x2, x2); sum(ss2, sq, laneId);
        const float rms = metal::rsqrt((ss1 + ss2) / (float)D + eps);
        mul(x1, x1, rms); mul(x2, x2, rms);
        if (gemma != 0) { add(w1, w1, 1.0f); add(w2, w2, 1.0f); }
        mul(x1, x1, w1); mul(x2, x2, w2);
        load(cv, gl_c, {0, 0, pos, 0}, laneId);
        load(sv, gl_s, {0, 0, pos, 0}, laneId);
        mul(o1, x1, cv); mul(tmp, x2, sv); sub(o1, o1, tmp);
        mul(o2, x2, cv); mul(tmp, x1, sv); add(o2, o2, tmp);
        store(gl_out, o1, {0, 0, row, 0}, laneId);
        store(gl_out, o2, {0, 0, row, 1}, laneId);
        return;
    }

    // GPT-J interleaved: contiguous lane chunks, pairs (2i, 2i+1) are lane-local
    // (the mla_q_norm_rope idiom with nope_dim == 0).
    const long base = (long)row * D + (long)laneId * PER_LANE;
    float ss = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) {
        const float v = float(qkv[base + k]);
        ss += v * v;
    }
    ss = metal::simd_sum(ss);
    const float rms = metal::rsqrt(ss / (float)D + eps);
    const long wbase = (long)laneId * PER_LANE;
    const long csbase = (long)pos * D2;
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = (int)laneId * PER_LANE + k;
        float w0 = float(w[wbase + k]);
        float w1v = float(w[wbase + k + 1]);
        if (gemma != 0) { w0 += 1.0f; w1v += 1.0f; }
        const float v0 = float(qkv[base + k]) * rms * w0;
        const float v1 = float(qkv[base + k + 1]) * rms * w1v;
        const int p = g0 / 2;
        const float c = float(cosb[csbase + p]);
        const float s = float(sinb[csbase + p]);
        out[base + k]     = bf16(v0 * c - v1 * s);
        out[base + k + 1] = bf16(v0 * s + v1 * c);
    }
}

#define instantiate_qk_norm_rope(DVAL)                                          \
  template [[host_name("qk_norm_rope_" #DVAL)]] [[kernel]] void                 \
  qk_norm_rope<DVAL>(device const bf16 *qkv [[buffer(0)]],                       \
                     device const bf16 *q_weight [[buffer(1)]],                  \
                     device const bf16 *k_weight [[buffer(2)]],                  \
                     device const bf16 *cosb [[buffer(3)]],                      \
                     device const bf16 *sinb [[buffer(4)]],                      \
                     device const int *positions [[buffer(5)]],                  \
                     device bf16 *out [[buffer(6)]],                             \
                     constant int &num_heads_q [[buffer(7)]],                    \
                     constant int &num_heads_k [[buffer(8)]],                    \
                     constant int &num_heads_v [[buffer(9)]],                    \
                     constant float &eps [[buffer(10)]],                         \
                     constant int &interleave [[buffer(11)]],                    \
                     constant int &gemma [[buffer(12)]],                         \
                     uint3 blockIdx [[threadgroup_position_in_grid]],            \
                     uint laneId [[thread_index_in_simdgroup]]);

instantiate_qk_norm_rope(64);
instantiate_qk_norm_rope(128);
instantiate_qk_norm_rope(256);

// ---------------------------------------------------------------------------
// Explicit positioned/partial/M-RoPE variant.  This keeps the established
// packed-QKV contract while replacing model booleans with mathematical
// parameters: rotary_dim, pair layout, norm_weight_offset, and position mode.
// position_mode: 0 = one-dimensional, 1 = sectioned THW, 2 = THW-interleaved.
// M-RoPE always uses split-half pairing.  D=512 is included for heterogeneous
// and multimodal heads; the original full-dimension fast path stays unchanged.
// ---------------------------------------------------------------------------
template <int D>
kernel void qk_norm_rope_positioned(
                         device const bf16 *qkv       [[buffer(0)]],
                         device const bf16 *q_weight  [[buffer(1)]],
                         device const bf16 *k_weight  [[buffer(2)]],
                         device const bf16 *cosb      [[buffer(3)]],
                         device const bf16 *sinb      [[buffer(4)]],
                         device const int  *positions [[buffer(5)]],
                         device bf16       *out       [[buffer(6)]],
                         constant int &num_heads_q    [[buffer(7)]],
                         constant int &num_heads_k    [[buffer(8)]],
                         constant int &num_heads_v    [[buffer(9)]],
                         constant float &eps          [[buffer(10)]],
                         constant int &rotary_dim     [[buffer(11)]],
                         constant int &interleave     [[buffer(12)]],
                         constant float &weight_offset [[buffer(13)]],
                         constant int &position_mode  [[buffer(14)]],
                         constant int &section_t      [[buffer(15)]],
                         constant int &section_h      [[buffer(16)]],
                         constant int &section_w      [[buffer(17)]],
                         constant int &num_tokens     [[buffer(18)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint laneId [[thread_index_in_simdgroup]]) {
    const int head = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int HT = num_heads_q + num_heads_k + num_heads_v;
    const long base = ((long)token * HT + head) * D;

    if (head >= num_heads_q + num_heads_k) {
        for (int d = (int)laneId; d < D; d += 32) out[base + d] = qkv[base + d];
        return;
    }

    const bool is_q = head < num_heads_q;
    device const bf16 *w = is_q ? q_weight : k_weight;
    float ss = 0.0f;
    for (int d = (int)laneId; d < D; d += 32) {
        const float v = float(qkv[base + d]);
        ss += v * v;
    }
    ss = metal::simd_sum(ss);
    const float inv_rms = metal::rsqrt(ss / (float)D + eps);
    const int rp = rotary_dim / 2;

    for (int p = (int)laneId; p < D / 2; p += 32) {
        if (p < rp) {
            const int i0 = interleave != 0 ? 2 * p : p;
            const int i1 = interleave != 0 ? 2 * p + 1 : rp + p;
            const float a = float(qkv[base + i0]) * inv_rms *
                            (float(w[i0]) + weight_offset);
            const float b = float(qkv[base + i1]) * inv_rms *
                            (float(w[i1]) + weight_offset);
            int axis = 0;
            if (position_mode == 2) {
                axis = p % 3;
            } else if (position_mode == 1) {
                axis = p < section_t ? 0 : (p < section_t + section_h ? 1 : 2);
            }
            const int pos = positions[(long)axis * num_tokens + token];
            const long cs = (long)pos * rp + p;
            const float c = float(cosb[cs]);
            const float s = float(sinb[cs]);
            out[base + i0] = bf16(a * c - b * s);
            out[base + i1] = bf16(a * s + b * c);
        } else {
            const int i0 = rotary_dim + 2 * (p - rp);
            out[base + i0] = bf16(float(qkv[base + i0]) * inv_rms *
                                  (float(w[i0]) + weight_offset));
            out[base + i0 + 1] = bf16(float(qkv[base + i0 + 1]) * inv_rms *
                                      (float(w[i0 + 1]) + weight_offset));
        }
    }
    (void)section_w;
}

#define instantiate_qk_norm_rope_positioned(DVAL)                              \
  template [[host_name("qk_norm_rope_positioned_" #DVAL)]] [[kernel]] void     \
  qk_norm_rope_positioned<DVAL>(                                                \
      device const bf16 *qkv [[buffer(0)]],                                     \
      device const bf16 *q_weight [[buffer(1)]],                                \
      device const bf16 *k_weight [[buffer(2)]],                                \
      device const bf16 *cosb [[buffer(3)]],                                    \
      device const bf16 *sinb [[buffer(4)]],                                    \
      device const int *positions [[buffer(5)]],                                \
      device bf16 *out [[buffer(6)]],                                           \
      constant int &num_heads_q [[buffer(7)]],                                  \
      constant int &num_heads_k [[buffer(8)]],                                  \
      constant int &num_heads_v [[buffer(9)]],                                  \
      constant float &eps [[buffer(10)]],                                       \
      constant int &rotary_dim [[buffer(11)]],                                  \
      constant int &interleave [[buffer(12)]],                                  \
      constant float &weight_offset [[buffer(13)]],                             \
      constant int &position_mode [[buffer(14)]],                               \
      constant int &section_t [[buffer(15)]],                                   \
      constant int &section_h [[buffer(16)]],                                   \
      constant int &section_w [[buffer(17)]],                                   \
      constant int &num_tokens [[buffer(18)]],                                  \
      uint3 blockIdx [[threadgroup_position_in_grid]],                          \
      uint laneId [[thread_index_in_simdgroup]]);

instantiate_qk_norm_rope_positioned(64);
instantiate_qk_norm_rope_positioned(128);
instantiate_qk_norm_rope_positioned(256);
instantiate_qk_norm_rope_positioned(512);

}
