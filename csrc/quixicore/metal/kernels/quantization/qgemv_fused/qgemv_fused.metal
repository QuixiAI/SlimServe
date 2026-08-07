#include "tk.metal"
#include <metal_stdlib>

using namespace metal;

namespace mittens {

// ---------------------------------------------------------------------------
// Fused packed-Q4_0 decode GEMVs (batch-1 decode), fp32 activations + output.
//
// These fuse two or three of the projection GEMVs that a transformer block runs
// back-to-back over the SAME activation vector into one launch, so the shared
// (K,1) activation is streamed once and the block-scale reads stay CSE-able. The
// weight is GGUF Q4_0 (block_k=32, 18 bytes/block: half d + 16 packed nibbles;
// value = d * (nibble - 8)), the same packed layout as the standalone qgemv
// q4_0 kernels. One simdgroup (32 lanes) owns one output row: lane owns an
// 8-byte (16-weight) span inside a block, 16 q4_0 blocks are consumed per
// simdgroup iteration, and the per-row dot is reduced with a simd_sum. Shapes
// are runtime (N/K, or Nq/Nkv/K for the QKV fusion) — these are keyed by the
// FUSION SHAPE (up+gate+GELU, up+gate, Q+K+V), not by any model's dims.
// ---------------------------------------------------------------------------

// tanh-approx GELU, matching mittens `gelu` / mx.nn.gelu_approx / F.gelu(tanh).
// tanh is written as 1 - 2/(exp(2z)+1) (the repo's stable form) so a large positive
// argument saturates to +1 instead of nan'ing on Metal's default fast-math exp/exp.
static inline float qgf_gelu_tanh(float x) {
    constexpr float s = 0.7978845608028654f;   // sqrt(2/pi)
    constexpr float a = 0.044715f;
    const float inner = s * (x + a * x * x * x);
    const float th = 1.0f - 2.0f / (metal::exp(inner + inner) + 1.0f);
    return 0.5f * x * (1.0f + th);
}

// One simdgroup's partial dot of dequant(Q4_0 row) . x, pre simd_sum.
static inline float qgf_q4_0_partial(device const uchar *row_base,
                                     device const float *x, int bpr, uint lane) {
    const int block_offset = int(lane >> 1);   // 16 q4_0 blocks per simdgroup step
    const int byte_start = int(lane & 1) * 8;  // 8 packed bytes = 16 weights per lane
    float acc = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 16) {
        device const uchar *block = row_base + (uint)kb * 18;
        const float d = float(((device const half *)block)[0]);
        device const uchar *qs = block + 2 + byte_start;
        const int x0 = (kb << 5) + byte_start;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = qs[i];
            acc += d * float(int(packed & 0x0f) - 8) * x[x0 + i];
            acc += d * float(int(packed >> 4) - 8) * x[x0 + i + 16];
        }
    }
    return acc;
}

// ---- up+gate+GELU: out = gelu_tanh(gate @ x) * (up @ x). One (N,1) output. ----
// The activation reuse is real here: up and gate rows share every x load, so the
// (K,1) vector streams once for two matmuls, and the gated-GELU epilogue writes
// the single vector the down projection consumes (no up/gate round-trip).
[[host_name("qgemv_q4_0_f32_up_gate_gelu")]]
kernel void qgemv_q4_0_f32_up_gate_gelu(
    device float *out        [[buffer(0)]],   // (N, 1) activated output
    device const uchar *up   [[buffer(1)]],   // (N, K/32, 18) Q4_0 up weights
    device const uchar *gate [[buffer(2)]],   // (N, K/32, 18) Q4_0 gate weights
    device const float *x    [[buffer(3)]],   // (K, 1) activation vector
    constant int &N          [[buffer(4)]],
    constant int &K          [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = int(tgid.x);
    if (row >= N) return;
    const int bpr = K / 32;
    device const uchar *up_row = up + (uint)(row * bpr) * 18;
    device const uchar *gate_row = gate + (uint)(row * bpr) * 18;
    const int block_offset = int(lane >> 1);
    const int byte_start = int(lane & 1) * 8;

    float up_sum = 0.0f, gate_sum = 0.0f;
    for (int kb = block_offset; kb < bpr; kb += 16) {
        device const uchar *ub = up_row + (uint)kb * 18;
        device const uchar *gb = gate_row + (uint)kb * 18;
        const float ud = float(((device const half *)ub)[0]);
        const float gd = float(((device const half *)gb)[0]);
        device const uchar *uqs = ub + 2 + byte_start;
        device const uchar *gqs = gb + 2 + byte_start;
        const int x0 = (kb << 5) + byte_start;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const float xl = x[x0 + i];
            const float xh = x[x0 + i + 16];
            const uchar uc = uqs[i];
            const uchar gc = gqs[i];
            up_sum += ud * (float(int(uc & 0x0f) - 8) * xl +
                            float(int(uc >> 4) - 8) * xh);
            gate_sum += gd * (float(int(gc & 0x0f) - 8) * xl +
                              float(int(gc >> 4) - 8) * xh);
        }
    }
    up_sum = metal::simd_sum(up_sum);
    gate_sum = metal::simd_sum(gate_sum);
    if (lane == 0) out[row] = qgf_gelu_tanh(gate_sum) * up_sum;
}

// ---- up+gate: up_out = up @ x, gate_out = gate @ x. Two (N,1) outputs. ----
// One combined row grid of 2N: rows [0,N) are up, [N,2N) are gate. Same launch,
// same activation resident in cache; the caller applies the gate activation.
[[host_name("qgemv_q4_0_f32_up_gate")]]
kernel void qgemv_q4_0_f32_up_gate(
    device float *up_out   [[buffer(0)]],   // (N, 1)
    device float *gate_out [[buffer(1)]],   // (N, 1)
    device const uchar *up   [[buffer(2)]], // (N, K/32, 18)
    device const uchar *gate [[buffer(3)]], // (N, K/32, 18)
    device const float *x    [[buffer(4)]], // (K, 1)
    constant int &N          [[buffer(5)]],
    constant int &K          [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int combined = int(tgid.x);
    if (combined >= 2 * N) return;
    const bool is_gate = combined >= N;
    const int row = is_gate ? combined - N : combined;
    device const uchar *w = is_gate ? gate : up;
    device float *out = is_gate ? gate_out : up_out;
    const int bpr = K / 32;
    device const uchar *row_base = w + (uint)(row * bpr) * 18;
    const float acc = metal::simd_sum(qgf_q4_0_partial(row_base, x, bpr, lane));
    if (lane == 0) out[row] = acc;
}

// ---- QKV: q_out = Wq @ x, k_out = Wk @ x, v_out = Wv @ x. Three outputs. ----
// One combined row grid of Nq + 2*Nkv (grouped-query friendly: Nkv != Nq). The
// three projections read the same activation in one launch instead of three.
[[host_name("qgemv_q4_0_f32_qkv")]]
kernel void qgemv_q4_0_f32_qkv(
    device float *q_out [[buffer(0)]],   // (Nq, 1)
    device float *k_out [[buffer(1)]],   // (Nkv, 1)
    device float *v_out [[buffer(2)]],   // (Nkv, 1)
    device const uchar *qw [[buffer(3)]],   // (Nq,  K/32, 18)
    device const uchar *kw [[buffer(4)]],   // (Nkv, K/32, 18)
    device const uchar *vw [[buffer(5)]],   // (Nkv, K/32, 18)
    device const float *x  [[buffer(6)]],   // (K, 1)
    constant int &Nq  [[buffer(7)]],
    constant int &Nkv [[buffer(8)]],
    constant int &K   [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int combined = int(tgid.x);
    if (combined >= Nq + 2 * Nkv) return;
    device const uchar *w;
    device float *out;
    int row;
    if (combined < Nq) {
        w = qw; out = q_out; row = combined;
    } else if (combined < Nq + Nkv) {
        w = kw; out = k_out; row = combined - Nq;
    } else {
        w = vw; out = v_out; row = combined - Nq - Nkv;
    }
    const int bpr = K / 32;
    device const uchar *row_base = w + (uint)(row * bpr) * 18;
    const float acc = metal::simd_sum(qgf_q4_0_partial(row_base, x, bpr, lane));
    if (lane == 0) out[row] = acc;
}

}  // namespace mittens
