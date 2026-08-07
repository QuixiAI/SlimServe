#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// ---------------------------------------------------------------------------
// qk_norm_rope_kv_f16: qk_norm_rope with a fused f16 KV split-store.
//
// Same fused per-head QK-RMSNorm + RoPE as qk_norm_rope over a packed (T, HT*D)
// bf16 QKV buffer, but instead of writing every head back into one packed bf16
// output it SPLITS the result into the three tensors an attention step actually
// consumes, converting the KV halves to f16 in the same pass:
//
//   Q heads (normed + roped)  -> q_out (T, Hq*D) bf16
//   K heads (normed + roped)  -> k_out (T, Hk*D) f16   (contiguous KV-cache layout)
//   V heads (copy-through)    -> v_out (T, Hv*D) f16
//
// This is the f16-KV-store fusion: the K/V heads land directly in the half KV
// cache the decoder reads, so there is no separate un-pack + bf16->f16 cast pass
// after attention-prep. One warp per (token, head); grid ((Hq+Hk+Hv), T, 1), 32
// lanes. interleave/gemma/D semantics match qk_norm_rope exactly. Shape-keyed by D.
// ---------------------------------------------------------------------------
template <int D>
kernel void qk_norm_rope_kv_f16(device const bf16 *qkv       [[buffer(0)]],
                                device const bf16 *q_weight  [[buffer(1)]],
                                device const bf16 *k_weight  [[buffer(2)]],
                                device const bf16 *cosb      [[buffer(3)]],
                                device const bf16 *sinb      [[buffer(4)]],
                                device const int  *positions [[buffer(5)]],
                                device bf16       *q_out     [[buffer(6)]],
                                device half       *k_out     [[buffer(7)]],
                                device half       *v_out     [[buffer(8)]],
                                constant int &num_heads_q    [[buffer(9)]],
                                constant int &num_heads_k    [[buffer(10)]],
                                constant int &num_heads_v    [[buffer(11)]],
                                constant float &eps          [[buffer(12)]],
                                constant int &interleave     [[buffer(13)]],
                                constant int &gemma          [[buffer(14)]],
                                uint3 blockIdx [[threadgroup_position_in_grid]],
                                uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");
    constexpr int PER_LANE = D / 32;
    const int head  = (int)blockIdx.x;
    const int token = (int)blockIdx.y;
    const int HT = num_heads_q + num_heads_k + num_heads_v;
    const int in_row = token * HT + head;      // packed (token, head) input row

    const bool is_q = head < num_heads_q;
    const bool is_k = !is_q && head < num_heads_q + num_heads_k;
    const int kh = head - num_heads_q;
    const int vh = head - num_heads_q - num_heads_k;
    // output row within the destination tensor (its own contiguous head layout)
    const int q_row = token * num_heads_q + head;
    const int k_row = token * num_heads_k + kh;
    const int v_row = token * num_heads_v + vh;

    using in_gl  = gl<bf16, 1, 1, -1, D>;
    using q_gl   = gl<bf16, 1, 1, -1, D>;
    using kv_gl  = gl<half, 1, 1, -1, D>;
    in_gl gl_in((device bf16 *)qkv, nullptr, nullptr, 1, nullptr);
    q_gl  gl_q(q_out, nullptr, nullptr, 1, nullptr);
    kv_gl gl_k(k_out, nullptr, nullptr, 1, nullptr);
    kv_gl gl_v(v_out, nullptr, nullptr, 1, nullptr);

    if (!is_q && !is_k) {                      // V head: copy-through into the f16 V cache
        using vecD = rv_fl<D>;
        vecD vv;
        load(vv, gl_in, {0, 0, in_row, 0}, laneId);
        store(gl_v, vv, {0, 0, v_row, 0}, laneId);
        return;
    }

    device const bf16 *w = is_q ? q_weight : k_weight;
    const int pos = positions[token];

    if (interleave == 0) {
        // NeoX split-half via the rv_fl<D/2> half-vector idiom.
        using cs_gl = gl<bf16, 1, 1, -1, D2>;
        cs_gl gl_c((device bf16 *)cosb, nullptr, nullptr, 1, nullptr);
        cs_gl gl_s((device bf16 *)sinb, nullptr, nullptr, 1, nullptr);
        cs_gl gl_w((device bf16 *)w,    nullptr, nullptr, 1, nullptr);
        using vecH = rv_fl<D2>;
        vecH x1, x2, w1, w2, cv, sv, o1, o2, tmp, sq;
        load(x1, gl_in, {0, 0, in_row, 0}, laneId);
        load(x2, gl_in, {0, 0, in_row, 1}, laneId);
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
        if (is_q) {
            store(gl_q, o1, {0, 0, q_row, 0}, laneId);
            store(gl_q, o2, {0, 0, q_row, 1}, laneId);
        } else {
            store(gl_k, o1, {0, 0, k_row, 0}, laneId);
            store(gl_k, o2, {0, 0, k_row, 1}, laneId);
        }
        return;
    }

    // GPT-J interleaved: contiguous lane chunks, pairs (2i, 2i+1) lane-local.
    const long in_base = (long)in_row * D + (long)laneId * PER_LANE;
    float ss = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) {
        const float v = float(qkv[in_base + k]);
        ss += v * v;
    }
    ss = metal::simd_sum(ss);
    const float rms = metal::rsqrt(ss / (float)D + eps);
    const long wbase = (long)laneId * PER_LANE;
    const long csbase = (long)pos * D2;
    const int out_row = is_q ? q_row : k_row;
    const long out_base = (long)out_row * D + (long)laneId * PER_LANE;
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = (int)laneId * PER_LANE + k;
        float w0 = float(w[wbase + k]);
        float w1v = float(w[wbase + k + 1]);
        if (gemma != 0) { w0 += 1.0f; w1v += 1.0f; }
        const float v0 = float(qkv[in_base + k]) * rms * w0;
        const float v1 = float(qkv[in_base + k + 1]) * rms * w1v;
        const int p = g0 / 2;
        const float c = float(cosb[csbase + p]);
        const float s = float(sinb[csbase + p]);
        const float y0 = v0 * c - v1 * s;
        const float y1 = v0 * s + v1 * c;
        if (is_q) {
            q_out[out_base + k]     = bf16(y0);
            q_out[out_base + k + 1] = bf16(y1);
        } else {
            k_out[out_base + k]     = half(y0);
            k_out[out_base + k + 1] = half(y1);
        }
    }
}

#define instantiate_qk_norm_rope_kv_f16(DVAL)                                    \
  template [[host_name("qk_norm_rope_kv_f16_" #DVAL)]] [[kernel]] void           \
  qk_norm_rope_kv_f16<DVAL>(device const bf16 *qkv [[buffer(0)]],                \
                     device const bf16 *q_weight [[buffer(1)]],                  \
                     device const bf16 *k_weight [[buffer(2)]],                  \
                     device const bf16 *cosb [[buffer(3)]],                      \
                     device const bf16 *sinb [[buffer(4)]],                      \
                     device const int *positions [[buffer(5)]],                  \
                     device bf16 *q_out [[buffer(6)]],                           \
                     device half *k_out [[buffer(7)]],                           \
                     device half *v_out [[buffer(8)]],                           \
                     constant int &num_heads_q [[buffer(9)]],                    \
                     constant int &num_heads_k [[buffer(10)]],                   \
                     constant int &num_heads_v [[buffer(11)]],                   \
                     constant float &eps [[buffer(12)]],                         \
                     constant int &interleave [[buffer(13)]],                    \
                     constant int &gemma [[buffer(14)]],                         \
                     uint3 blockIdx [[threadgroup_position_in_grid]],            \
                     uint laneId [[thread_index_in_simdgroup]]);

instantiate_qk_norm_rope_kv_f16(64);
instantiate_qk_norm_rope_kv_f16(128);
instantiate_qk_norm_rope_kv_f16(256);

}
