// SlimServe addition (not upstream QuixiCore-Metal): elementwise and
// positional glue for the single-command-buffer decode step encoder in
// tm_metal/qc_metal_serving.mm. Each kernel is deliberately minimal; the
// point of this family is dispatch-count and host-overhead reduction, not
// arithmetic cleverness.

#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Interleaved (GPT-J) RoPE over packed Q and K head vectors, in place.
// Buffer holds rows of head_dim; row r belongs to token positions[r / heads].
// One simdgroup per row; each lane owns head_dim/64 pairs.
kernel void muse_rope_qk(
    device bf16      *qk       [[buffer(0)]],   // (rows, head_dim)
    device const int *positions [[buffer(1)]],  // (rows / heads_per_token)
    constant int   &head_dim   [[buffer(2)]],
    constant int   &heads_per_token [[buffer(3)]],
    constant float &theta      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const float pos = float(positions[row / heads_per_token]);
    device bf16 *base = qk + (long)row * head_dim;
    const int pairs = head_dim / 2;
    for (int j = (int)lane; j < pairs; j += 32) {
        const float inv = metal::pow(theta, -2.0f * float(j) / float(head_dim));
        const float ang = pos * inv;
        const float c = metal::cos(ang);
        const float s = metal::sin(ang);
        const float x1 = float(base[2 * j]);
        const float x2 = float(base[2 * j + 1]);
        base[2 * j]     = bf16(x1 * c - x2 * s);
        base[2 * j + 1] = bf16(x2 * c + x1 * s);
    }
}

// NeoX (half-split) RoPE variant for the DFlash drafter: pairs are
// (j, j + head_dim/2) instead of the interleaved (2j, 2j+1).
kernel void muse_rope_qk_neox(
    device bf16      *qk       [[buffer(0)]],   // (rows, head_dim)
    device const int *positions [[buffer(1)]],  // (rows / heads_per_token)
    constant int   &head_dim   [[buffer(2)]],
    constant int   &heads_per_token [[buffer(3)]],
    constant float &theta      [[buffer(4)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const float pos = float(positions[row / heads_per_token]);
    device bf16 *base = qk + (long)row * head_dim;
    const int half_d = head_dim / 2;
    for (int j = (int)lane; j < half_d; j += 32) {
        const float inv = metal::pow(theta, -2.0f * float(j) / float(head_dim));
        const float ang = pos * inv;
        const float c = metal::cos(ang);
        const float s = metal::sin(ang);
        const float x1 = float(base[j]);
        const float x2 = float(base[j + half_d]);
        base[j]          = bf16(x1 * c - x2 * s);
        base[j + half_d] = bf16(x2 * c + x1 * s);
    }
}

// Row-wise greedy argmax over bf16 logits: one threadgroup per row,
// tree-reduced across 256 threads. Ties resolve to the LOWEST index
// (matches torch.argmax). Softcap/logit-scale are argmax-invariant and
// intentionally skipped.
kernel void muse_argmax(
    device long        *out    [[buffer(0)]],   // (rows)
    device const bf16  *logits [[buffer(1)]],   // (rows, vocab)
    constant int &vocab [[buffer(2)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  tid  [[thread_index_in_threadgroup]]) {
    device const bf16 *row = logits + (long)tgid.x * vocab;
    float best = -INFINITY;
    int best_i = 0;
    for (int i = (int)tid; i < vocab; i += 256) {
        const float v = float(row[i]);
        if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
        }
    }
    threadgroup float sv[256];
    threadgroup int si[256];
    sv[tid] = best;
    si[tid] = best_i;
    threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    for (int s = 128; s > 0; s >>= 1) {
        if ((int)tid < s) {
            const bool take = sv[tid + s] > sv[tid] ||
                              (sv[tid + s] == sv[tid] && si[tid + s] < si[tid]);
            if (take) {
                sv[tid] = sv[tid + s];
                si[tid] = si[tid + s];
            }
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        out[tgid.x] = (long)si[0];
    }
}

// out *= sigmoid(gate), elementwise over n values (n % 4 == 0).
kernel void muse_sigmoid_mul(
    device bf16_4       *out  [[buffer(0)]],
    device const bf16_4 *gate [[buffer(1)]],
    constant int &n4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= n4) return;
    const metal::float4 g = metal::float4(gate[tid]);
    const metal::float4 sig = 1.0f / (1.0f + metal::exp(-g));
    out[tid] = bf16_4(metal::float4(out[tid]) * sig);
}

// SwiGLU epilogue: out = silu(gate) * up, elementwise (n % 4 == 0).
kernel void muse_silu_mul(
    device bf16_4       *out  [[buffer(0)]],
    device const bf16_4 *gate [[buffer(1)]],
    device const bf16_4 *up   [[buffer(2)]],
    constant int &n4 [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= n4) return;
    const metal::float4 g = metal::float4(gate[tid]);
    const metal::float4 sig = 1.0f / (1.0f + metal::exp(-g));
    out[tid] = bf16_4(g * sig * metal::float4(up[tid]));
}

// x += y, elementwise (n % 4 == 0).
kernel void muse_add_inplace(
    device bf16_4       *x [[buffer(0)]],
    device const bf16_4 *y [[buffer(1)]],
    constant int &n4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= n4) return;
    x[tid] = bf16_4(metal::float4(x[tid]) + metal::float4(y[tid]));
}

// Scatter new K/V rows into the paged cache.
// cache layout: (2, num_blocks, block_size, kv_heads, head_dim); k/v are
// (tokens, kv_heads, head_dim); slot_mapping is (tokens).
kernel void muse_kv_store(
    device bf16         *cache [[buffer(0)]],
    device const bf16   *k     [[buffer(1)]],
    device const bf16   *v     [[buffer(2)]],
    device const long   *slots [[buffer(3)]],
    constant int &block_size [[buffer(4)]],
    constant int &kv_heads   [[buffer(5)]],
    constant int &head_dim   [[buffer(6)]],
    constant long &half_elems [[buffer(7)]],  // num_blocks*block_size*kv_heads*head_dim
    constant int &tokens     [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
    const int per_tok = kv_heads * head_dim / 4;
    const int total = per_tok * tokens;
    if ((int)tid >= total) return;
    const int t = (int)tid / per_tok;
    const int e4 = (int)tid % per_tok;
    const long slot = slots[t];
    if (slot < 0) return;
    device bf16_4 *kdst =
        (device bf16_4 *)(cache + slot * kv_heads * head_dim) + e4;
    device bf16_4 *vdst =
        (device bf16_4 *)(cache + half_elems + slot * kv_heads * head_dim) + e4;
    *kdst = ((device const bf16_4 *)(k + (long)t * kv_heads * head_dim))[e4];
    *vdst = ((device const bf16_4 *)(v + (long)t * kv_heads * head_dim))[e4];
}

// Plain vec4 copy: the fused verify step snapshots the residual stream into
// the aux-hidden buffer the DFlash drafter consumes.
kernel void muse_copy(
    device bf16_4       *dst [[buffer(0)]],
    device const bf16_4 *src [[buffer(1)]],
    constant int &n4 [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= n4) return;
    dst[tid] = src[tid];
}

// Transpose+pad+cast a (m, K) bf16 activation block to the (K, 32) half
// layout the contiguous-staging qgemm_sm consumes. Rows >= m pad to zero.
kernel void muse_xpose32(
    device half         *out [[buffer(0)]],   // (K, 32)
    device const ushort *x   [[buffer(1)]],   // (m, K) bf16
    constant int &K [[buffer(2)]],
    constant int &m [[buffer(3)]],
    uint tid [[thread_position_in_grid]]) {
    if ((int)tid >= K * 32) return;
    const int k = (int)tid / 32;
    const int mm = (int)tid % 32;
    half v = half(0.0f);
    if (mm < m) {
        v = half(as_type<float>(uint(x[(size_t)mm * K + k]) << 16));
    }
    out[tid] = v;
}

}  // namespace mittens
