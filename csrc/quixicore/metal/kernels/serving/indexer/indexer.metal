#include "tk.metal"
#include <metal_stdlib>

using namespace metal;

namespace mittens {

// ---------------------------------------------------------------------------
// DeepSeek-V3.2 (DSA/NSA) indexer K quant-and-cache (metal-forge
// cache/gather_kv_cache.metal indexer_k_quant_and_cache; credit AlpinDale). Quantizes the
// indexer K per quant_block_size (canonical 128) into a low-precision e4m3 cache the sparse-
// attention top-k selector reads cheaply. TM-native layout: SEPARATE code cache (uchar,
// num_slots x head_dim) + fp32 scale cache (num_slots x head_dim/qbs), indexed directly by
// slot_mapping (like the TurboQuant KV codec) rather than the reference's interleaved
// single-buffer paged layout. K-only, no RoPE, arbitrary head_dim. use_ue8m0 rounds the fp32
// scale to a power of two (MX). One simdgroup per (token, qblock); the fp8 arithmetic chain
// is faithful so a numpy oracle reproduces codes bit-for-bit. slot < 0 skips (padding).
// ---------------------------------------------------------------------------
template <typename T>
kernel void indexer_k_quant_and_cache(device const T *k          [[buffer(0)]],  // (tokens, head_dim)
                                      device const int *slot_mapping [[buffer(1)]],  // (tokens,)
                                      device uchar *code_cache   [[buffer(2)]],  // (slots, head_dim)
                                      device float *scale_cache  [[buffer(3)]],  // (slots, head_dim/qbs)
                                      constant int &head_dim     [[buffer(4)]],
                                      constant int &quant_block_size [[buffer(5)]],
                                      constant int &use_ue8m0    [[buffer(6)]],
                                      uint2 gid  [[threadgroup_position_in_grid]],
                                      uint  lane [[thread_index_in_simdgroup]]) {
    const int token = (int)gid.x;
    const int qblock = (int)gid.y;
    const int start = qblock * quant_block_size;
    if (start >= head_dim) { return; }
    const int slot = slot_mapping[token];
    if (slot < 0) { return; }
    const int nq = (head_dim + quant_block_size - 1) / quant_block_size;
    const long kbase = (long)token * head_dim;

    float amax = 0.0f;
    for (int i = (int)lane; i < quant_block_size && start + i < head_dim; i += 32) {
        amax = metal::max(amax, metal::fabs(float(k[kbase + start + i])));
    }
    amax = metal::simd_max(amax);
    float scale = metal::max(amax, 1.0e-4f) / 448.0f;
    if (use_ue8m0 != 0) {
        scale = metal::exp2(metal::ceil(metal::log2(scale)));
    }
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    const long cbase = (long)slot * head_dim;
    for (int i = (int)lane; i < quant_block_size && start + i < head_dim; i += 32) {
        const float v = float(k[kbase + start + i]) * inv;
        code_cache[cbase + start + i] = v == 0.0f ? uchar(0) : tk_e4m3_encode(v);
    }
    if (lane == 0) {
        scale_cache[(long)slot * nq + qblock] = scale;
    }
}

// Gather + dequantize the indexer cache back to bf16 K for a slot list: k_out[row] =
// decode(code_cache[slot]) * scale_cache[slot, qblock]. slots (n,) int.
template <typename T>
kernel void indexer_k_gather(device const uchar *code_cache  [[buffer(0)]],
                             device const float *scale_cache [[buffer(1)]],
                             device const int *slots         [[buffer(2)]],  // (n,)
                             device T *k_out                 [[buffer(3)]],  // (n, head_dim)
                             constant int &head_dim          [[buffer(4)]],
                             constant int &quant_block_size  [[buffer(5)]],
                             uint2 gid  [[threadgroup_position_in_grid]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    const int row = (int)gid.x;
    const int qblock = (int)gid.y;
    const int start = qblock * quant_block_size;
    if (start >= head_dim) { return; }
    const int slot = slots[row];
    const int nq = (head_dim + quant_block_size - 1) / quant_block_size;
    const float sc = scale_cache[(long)slot * nq + qblock];
    const long cbase = (long)slot * head_dim;
    const long obase = (long)row * head_dim;
    for (int i = (int)lane; i < quant_block_size && start + i < head_dim; i += 32) {
        k_out[obase + start + i] = T(float(tk_e4m3_decode(code_cache[cbase + start + i])) * sc);
    }
}

// generic byte clone for the functional cache-update prepass (u8 code + f32 scale caches)
kernel void indexer_clone_bytes(device const uchar *src [[buffer(0)]],
                                device uchar *dst [[buffer(1)]],
                                constant uint &n [[buffer(2)]],
                                uint tid [[thread_position_in_grid]]) {
    const uint base = tid * 16;
    if (base + 16 <= n) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j)
            ((device uchar4*)(dst + base))[j] = ((device const uchar4*)(src + base))[j];
    } else {
        for (uint i = base; i < n; ++i) dst[i] = src[i];
    }
}

#define instantiate_indexer(type_name, T)                                        \
  template [[host_name("indexer_k_quant_and_cache_" #type_name)]] [[kernel]] void \
  indexer_k_quant_and_cache<T>(device const T *k [[buffer(0)]],                   \
      device const int *slot_mapping [[buffer(1)]],                              \
      device uchar *code_cache [[buffer(2)]], device float *scale_cache [[buffer(3)]], \
      constant int &head_dim [[buffer(4)]], constant int &quant_block_size [[buffer(5)]], \
      constant int &use_ue8m0 [[buffer(6)]],                                     \
      uint2 gid [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);                                  \
  template [[host_name("indexer_k_gather_" #type_name)]] [[kernel]] void         \
  indexer_k_gather<T>(device const uchar *code_cache [[buffer(0)]],              \
      device const float *scale_cache [[buffer(1)]], device const int *slots [[buffer(2)]], \
      device T *k_out [[buffer(3)]], constant int &head_dim [[buffer(4)]],       \
      constant int &quant_block_size [[buffer(5)]],                              \
      uint2 gid [[threadgroup_position_in_grid]],                                \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_indexer(float32, float)
instantiate_indexer(float16, half)
instantiate_indexer(bfloat16, bf16)

} // namespace mittens

using namespace mittens;

// ---------------------------------------------------------------------------
// DeepSeek-V4 indexer compressed-K insert (Metal sibling of the CUDA
// _fused_kv_compress_norm_rope_insert_indexer_attn tail). The torch side does
// the state gather + softmax + RMSNorm and hands over the normed [tokens, 128]
// bf16 rows; this kernel applies GPT-J interleaved RoPE to the last 64 dims at
// the caller-supplied position (pre-floored to the compressed anchor
// (pos // ratio) * ratio), does the fp32->bf16->fp32 roundtrip the reference
// uses before the UE8M0 absmax, quantizes all 128 dims to e4m3, and stores a
// per-slot record. slot < 0 skips (non-boundary / padded tokens).
//
// Metal-native cache record (kv_cache is (blocks, block_size, 132) uint8):
//   [0, 128):   e4m3 codes
//   [128, 132): one float32 scale (power of two)
// Per-slot records, NOT the CUDA page-segregated layout — the only readers
// are the Metal top-k producer and this writer.
// ---------------------------------------------------------------------------
kernel void dsv4_indexer_kv_insert(
        device const bf16 *kv           [[buffer(0)]],   // (tokens, 128)
        device const bf16 *cosb         [[buffer(1)]],   // (positions, 32)
        device const bf16 *sinb         [[buffer(2)]],   // (positions, 32)
        device const int  *positions    [[buffer(3)]],   // compressed anchors
        device const long *slot_mapping [[buffer(4)]],   // (tokens,), -1 skips
        device uchar      *kv_cache     [[buffer(5)]],
        constant int &block_size        [[buffer(6)]],
        constant int &cache_block_stride [[buffer(7)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint laneId [[thread_index_in_simdgroup]]) {
    constexpr int DIM = 128, NOPE = 64, PER_LANE = 4;
    constexpr int SLOT_BYTES = 132;
    constexpr float FP8_MAX = 448.0f;
    const int token = blockIdx.x;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long base = (slot / block_size) * (long)cache_block_stride +
                      (slot % block_size) * SLOT_BYTES;
    const int pos = positions[token];
    const int d0 = (int)laneId * PER_LANE;
    const long kbase = (long)token * DIM + d0;

    float v[PER_LANE];
    for (int k = 0; k < PER_LANE; ++k) { v[k] = float(kv[kbase + k]); }
    if (d0 >= NOPE) {
        // PER_LANE=4 with an even dim start keeps RoPE pairs lane-local.
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (d0 + j - NOPE) / 2;
            const float c = float(cosb[(long)pos * 32 + p]);
            const float s = float(sinb[(long)pos * 32 + p]);
            const float ev = v[j], ov = v[j + 1];
            v[j] = ev * c - ov * s;
            v[j + 1] = ov * c + ev * s;
        }
    }
    for (int k = 0; k < PER_LANE; ++k) { v[k] = float(bf16(v[k])); }

    float amax = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) {
        amax = metal::max(amax, metal::fabs(v[k]));
    }
    amax = metal::simd_max(amax);
    const float exponent = metal::ceil(
        metal::log2(metal::max(amax, 1e-4f) / FP8_MAX));
    // Fast-math exp2 is approximate even at integer inputs, which both
    // corrupts the stored power-of-two scale and pushes exact e4m3 rounding
    // ties over the midpoint. Build 2^e exactly from the float bit pattern.
    const int ei = metal::clamp((int)exponent, -126, 126);
    const float scale = as_type<float>((uint)((ei + 127) << 23));
    const float inv_scale = as_type<float>((uint)((127 - ei) << 23));

    for (int k = 0; k < PER_LANE; ++k) {
        const float q = metal::clamp(v[k] * inv_scale, -FP8_MAX, FP8_MAX);
        kv_cache[base + d0 + k] = tk_e4m3_encode(q);
    }
    if (laneId == 0) {
        *((device float *)(kv_cache + base + DIM)) = scale;
    }
}

// ---------------------------------------------------------------------------
// Fused DeepSeek-V4 indexer compressor tail (Metal sibling of the Triton
// _fused_kv_compress_norm_rope_insert_indexer_attn): per boundary token,
// gather the 2*ratio-row overlap history from the fp32 state cache
// ([kv 2*128 | score 2*128] rows, h >= ratio reads the +128 head), per-dim
// softmax over history, weighted-sum compress, RMSNorm with the fp32
// weight, then the dsv4_indexer_kv_insert back half verbatim (GPT-J RoPE
// at the compressed anchor, bf16 roundtrip, simd absmax, exact
// power-of-two e4m3 quant, 132-byte record). Mirrors the eager
// Metal-branch tail in fused_compress_quant_cache.py per-op (explicit
// temporaries, precise transcendentals, reassociation/contraction off);
// residual drift is limited to reduction-order ulps in softmax/mean.
// One simdgroup per token; tokens with state slot < 0, a non-boundary
// position, or kv slot < 0 exit early (the eager path masks them the
// same way via output_slots = -1).
// ---------------------------------------------------------------------------
kernel void dsv4_indexer_compress_insert(
        device const float *state_cache  [[buffer(0)]],  // (blocks, bs, 512)
        device const bf16 *cosb          [[buffer(1)]],  // (positions, 32)
        device const bf16 *sinb          [[buffer(2)]],  // (positions, 32)
        device const int  *positions     [[buffer(3)]],  // raw token positions
        device const long *state_slots   [[buffer(4)]],  // compressor slots
        device const int  *token_to_req  [[buffer(5)]],
        device const int  *block_table   [[buffer(6)]],  // (reqs, bt_cols)
        device const float *rms_w        [[buffer(7)]],  // (128,) fp32
        device const long *kv_slots      [[buffer(8)]],  // K-cache slots
        device uchar      *kv_cache      [[buffer(9)]],  // 132-byte records
        constant int &state_block_size   [[buffer(10)]],
        constant int &state_stride0      [[buffer(11)]],
        constant int &state_stride1      [[buffer(12)]],
        constant int &state_width        [[buffer(13)]], // 256
        constant int &compress_ratio     [[buffer(14)]], // 4 (overlap layout)
        constant int &bt_stride          [[buffer(15)]],
        constant int &bt_cols            [[buffer(16)]],
        constant int &kv_block_size      [[buffer(17)]],
        constant int &kv_block_stride    [[buffer(18)]],
        constant float &eps              [[buffer(19)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint laneId [[thread_index_in_simdgroup]]) {
    constexpr int DIM = 128, NOPE = 64, PER_LANE = 4, HISTORY_MAX = 8;
    constexpr int SLOT_BYTES = 132;
    constexpr float FP8_MAX = 448.0f;
    const int token = blockIdx.x;
    const long sslot = state_slots[token];
    const int pos = positions[token];
    if (sslot < 0 || ((pos + 1) % compress_ratio) != 0) { return; }
    const long kvslot = kv_slots[token];
    if (kvslot < 0) { return; }

    const int history = 2 * compress_ratio;
    // The host contract pins compress_ratio so history fits the fixed-size
    // register arrays; a violating runtime value must fail safely instead
    // of writing past vals/scs.
    if (history > HISTORY_MAX) { return; }
    const int req = token_to_req[token];
    const int d0 = (int)laneId * PER_LANE;

    float vals[HISTORY_MAX][PER_LANE];
    float scs[HISTORY_MAX][PER_LANE];
    for (int h = 0; h < history; ++h) {
        const int p = pos - history + 1 + h;
        const bool okh = p >= 0;
        const int sp = metal::max(p, 0);
        const int colc = metal::min(sp / state_block_size, bt_cols - 1);
        const int blk = block_table[req * bt_stride + colc];
        const int off = sp % state_block_size;
        const int head_off = (h >= compress_ratio) ? DIM : 0;
        device const float *row = state_cache + (long)blk * state_stride0 +
                                  (long)off * state_stride1 + head_off;
        for (int k = 0; k < PER_LANE; ++k) {
            vals[h][k] = row[d0 + k];
            scs[h][k] = okh ? row[d0 + k + state_width] : -INFINITY;
        }
    }

    // Torch's rounding points: softmax materialized per element, product
    // rounded, then a sequential sum over the history dim.
    float comp[PER_LANE];
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            float m = scs[0][k];
            for (int h = 1; h < history; ++h) { m = metal::max(m, scs[h][k]); }
            float ex[HISTORY_MAX];
            float s = 0.0f;
            for (int h = 0; h < history; ++h) {
                ex[h] = metal::precise::exp(scs[h][k] - m);
                s += ex[h];
            }
            float acc = 0.0f;
            for (int h = 0; h < history; ++h) {
                const float smh = metal::precise::divide(ex[h], s);
                const float prod = vals[h][k] * smh;
                acc += prod;
            }
            comp[k] = acc;
        }
    }

    float ss = 0.0f;
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            const float c2 = comp[k] * comp[k];
            ss += c2;
        }
    }
    ss = metal::simd_sum(ss);
    const float rrms = metal::precise::rsqrt(
        metal::precise::divide(ss, float(DIM)) + eps);

    float v[PER_LANE];
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            const float t1 = comp[k] * rrms;
            const float t2 = t1 * rms_w[d0 + k];
            v[k] = float(bf16(t2));  // eager hands bf16 to the insert op
        }
    }

    // dsv4_indexer_kv_insert back half at the compressed anchor position.
    const int anchor = (pos / compress_ratio) * compress_ratio;
    const long base = (kvslot / kv_block_size) * (long)kv_block_stride +
                      (kvslot % kv_block_size) * SLOT_BYTES;
    if (d0 >= NOPE) {
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (d0 + j - NOPE) / 2;
            const float c = float(cosb[(long)anchor * 32 + p]);
            const float s = float(sinb[(long)anchor * 32 + p]);
            const float ev = v[j], ov = v[j + 1];
            v[j] = ev * c - ov * s;
            v[j + 1] = ov * c + ev * s;
        }
    }
    for (int k = 0; k < PER_LANE; ++k) { v[k] = float(bf16(v[k])); }

    float amax = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) {
        amax = metal::max(amax, metal::fabs(v[k]));
    }
    amax = metal::simd_max(amax);
    const float exponent = metal::ceil(
        metal::log2(metal::max(amax, 1e-4f) / FP8_MAX));
    const int ei = metal::clamp((int)exponent, -126, 126);
    const float scale = as_type<float>((uint)((ei + 127) << 23));
    const float inv_scale = as_type<float>((uint)((127 - ei) << 23));

    for (int k = 0; k < PER_LANE; ++k) {
        const float q = metal::clamp(v[k] * inv_scale, -FP8_MAX, FP8_MAX);
        kv_cache[base + d0 + k] = tk_e4m3_encode(q);
    }
    if (laneId == 0) {
        *((device float *)(kv_cache + base + DIM)) = scale;
    }
}

// ---------------------------------------------------------------------------
// Indexer Q-side sibling of _fused_indexer_q_rope_quant_metal (torch mirror in
// fused_indexer_q.py): GPT-J interleaved RoPE on the LAST rot dims with the
// fp32->bf16->fp32 roundtrip, per-token-per-head absmax over [raw-fp32 nope |
// bf16-rounded rope], q_scale = 2^ceil(log2(max(amax,1e-4)/448)) built exactly
// from the float bit pattern (see the exp2 note above), Q returned in the
// activation dtype holding value / q_scale, and q_scale folded into the
// returned fp32 weights as ((w * qs) * softmax_scale) * head_scale.
// One simdgroup per head; grid (tokens, H / 8), 256 threads.
// ---------------------------------------------------------------------------
template <typename T, typename CS>
METAL_FUNC void dsv4_indexer_q_body(
        device const T *x,            // (tokens, H, D)
        device const CS *cs,          // (max_pos, 2*half_rot) cos | sin
        device const long *positions, // (tokens,)
        device const float *w,        // (tokens, H)
        device T *q_out,              // (tokens, H, D)
        device float *w_out,          // (tokens, H)
        constant int &H, constant int &D, constant int &nope_dim,
        constant int &half_rot, constant float &softmax_scale,
        constant float &head_scale, uint3 tgid, uint sg, uint lane) {
    const int token = tgid.x;
    const int head = tgid.y * 8 + sg;
    if (head >= H) { return; }
    const long base = ((long)token * H + head) * D;
    const long pos = positions[token];
    device const CS *cs_row = cs + pos * 2 * half_rot;

    float m = 0.0f;
    for (int d = lane; d < nope_dim; d += 32) {
        m = metal::max(m, metal::fabs(float(x[base + d])));
    }
    float re = 0.0f, ro = 0.0f;
    if ((int)lane < half_rot) {
        const float c = float(cs_row[lane]);
        const float s = float(cs_row[half_rot + lane]);
        const float ev = float(x[base + nope_dim + 2 * lane]);
        const float ov = float(x[base + nope_dim + 2 * lane + 1]);
        re = float(bf16(ev * c - ov * s));
        ro = float(bf16(ov * c + ev * s));
        m = metal::max(m, metal::max(metal::fabs(re), metal::fabs(ro)));
    }
    m = metal::simd_max(m);
    const float exponent = metal::ceil(
        metal::log2(metal::max(m, 1e-4f) / 448.0f));
    const int ei = metal::clamp((int)exponent, -126, 126);
    const float qs = as_type<float>((uint)((ei + 127) << 23));
    const float inv = as_type<float>((uint)((127 - ei) << 23));

    for (int d = lane; d < nope_dim; d += 32) {
        q_out[base + d] = T(float(x[base + d]) * inv);
    }
    if ((int)lane < half_rot) {
        q_out[base + nope_dim + 2 * lane] = T(re * inv);
        q_out[base + nope_dim + 2 * lane + 1] = T(ro * inv);
    }
    if (lane == 0) {
        // Torch rounds after each of the three fp32 multiplies; fast-math
        // reassociation of the pure-mul chain lands 1 ulp off. Pin the order.
        #pragma clang fp reassociate(off)
        float wv = w[(long)token * H + head] * qs;
        wv = wv * softmax_scale;
        w_out[(long)token * H + head] = wv * head_scale;
    }
}

#define instantiate_dsv4_indexer_q(suffix, T, CS)                             \
  [[host_name("dsv4_indexer_q_" #suffix)]] kernel void                        \
  dsv4_indexer_q_##suffix(                                                    \
      device const T *x [[buffer(0)]], device const CS *cs [[buffer(1)]],     \
      device const long *positions [[buffer(2)]],                             \
      device const float *w [[buffer(3)]], device T *q_out [[buffer(4)]],     \
      device float *w_out [[buffer(5)]], constant int &H [[buffer(6)]],       \
      constant int &D [[buffer(7)]], constant int &nope_dim [[buffer(8)]],    \
      constant int &half_rot [[buffer(9)]],                                   \
      constant float &softmax_scale [[buffer(10)]],                           \
      constant float &head_scale [[buffer(11)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    dsv4_indexer_q_body<T, CS>(x, cs, positions, w, q_out, w_out, H, D,       \
                               nope_dim, half_rot, softmax_scale, head_scale, \
                               tgid, sg, lane);                               \
  }

instantiate_dsv4_indexer_q(float16, half, half);
instantiate_dsv4_indexer_q(float16_csf32, half, float);
instantiate_dsv4_indexer_q(bfloat16, bf16, bf16);
instantiate_dsv4_indexer_q(bfloat16_csf32, bf16, float);

// ---------------------------------------------------------------------------
// Attention-output inverse RoPE (torch mirror: DeepseekScalingRotaryEmbedding
// .forward_native(..., inverse=True) with is_neox_style=False, called from
// DeepseekV4MetalAttention._o_proj). GPT-J interleaved pairs on the LAST
// rot dims: e' = e*c + o*s, o' = o*c - e*s; nope dims pass through. Torch
// rounds after each elementwise op, in the promoted dtype (fp32 when the
// cos_sin cache is fp32, activation dtype when it matches) — mirrored here
// with explicit temporaries and reassociation/contraction pinned off.
// Input may be a strided head-slice of the padded attention output; output
// is written contiguously as [tokens, H*D] ready for the grouped WO_A GEMVs.
// One simdgroup per head; grid (tokens, H / 8), 256 threads.
// ---------------------------------------------------------------------------
template <typename T, typename CS>
METAL_FUNC void dsv4_o_inv_rope_body(
        device const T *o,            // strided (tokens, H, D)
        device const CS *cs,          // (max_pos, 2*half_rot) cos | sin
        device const long *positions, // (tokens,)
        device T *out,                // contiguous (tokens, H*D)
        constant int &H, constant int &D, constant int &half_rot,
        constant long &tok_stride, constant long &head_stride,
        uint3 tgid, uint sg, uint lane) {
    const int token = tgid.x;
    const int head = tgid.y * 8 + sg;
    if (head >= H) { return; }
    const int rot = 2 * half_rot;
    const int nope = D - rot;
    device const T *src = o + (long)token * tok_stride + (long)head * head_stride;
    device T *dst = out + ((long)token * H + head) * D;
    const long pos = positions[token];
    device const CS *cs_row = cs + pos * 2 * half_rot;

    for (int d = lane; d < nope; d += 32) {
        dst[d] = src[d];
    }
    if ((int)lane < half_rot) {
        #pragma clang fp reassociate(off) contract(off)
        const int e_i = nope + 2 * lane, o_i = nope + 2 * lane + 1;
        const T ev = src[e_i], ov = src[o_i];
        const CS c = cs_row[lane], s = cs_row[half_rot + lane];
        if (sizeof(CS) == 4) {  // fp32 cache: fp32 math, per-op rounding
            const float tc = float(ev) * float(c);
            const float ts = float(ov) * float(s);
            dst[e_i] = T(tc + ts);
            const float uc = float(ov) * float(c);
            const float us = float(ev) * float(s);
            dst[o_i] = T(uc - us);
        } else {
            const T tc = ev * T(c);
            const T ts = ov * T(s);
            dst[e_i] = tc + ts;
            const T uc = ov * T(c);
            const T us = ev * T(s);
            dst[o_i] = uc - us;
        }
    }
}

#define instantiate_dsv4_o_inv_rope(suffix, T, CS)                            \
  [[host_name("dsv4_o_inv_rope_" #suffix)]] kernel void                       \
  dsv4_o_inv_rope_##suffix(                                                   \
      device const T *o [[buffer(0)]], device const CS *cs [[buffer(1)]],     \
      device const long *positions [[buffer(2)]],                             \
      device T *out [[buffer(3)]], constant int &H [[buffer(4)]],             \
      constant int &D [[buffer(5)]], constant int &half_rot [[buffer(6)]],    \
      constant long &tok_stride [[buffer(7)]],                                \
      constant long &head_stride [[buffer(8)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    dsv4_o_inv_rope_body<T, CS>(o, cs, positions, out, H, D, half_rot,        \
                                tok_stride, head_stride, tgid, sg, lane);     \
  }

instantiate_dsv4_o_inv_rope(float16, half, half);
instantiate_dsv4_o_inv_rope(float16_csf32, half, float);
instantiate_dsv4_o_inv_rope(bfloat16, bf16, bf16);
instantiate_dsv4_o_inv_rope(bfloat16_csf32, bf16, float);

// ---------------------------------------------------------------------------
// Decode-path indexer producer (torch mirror: metal_sparse_attn_indexer's
// num_decode branch in metal_indexer.py). Per decode token:
//   logits[j] = k_scale[j] * sum_h w[h] * relu(q[h,:] . e4m3_decode(code[j,:]))
//   for j < cand; -inf otherwise; then top-k_eff by logit with request-local
//   column indices, -1 where fewer candidates exist, rest of the row -1.
// One threadgroup per token: q and w staged in threadgroup memory, one
// simdgroup per candidate (two heads per lane, K broadcast via shuffles),
// then a 1024-wide bitonic sort (logit desc, index asc on ties) and a direct
// topk_indices_buffer row write.  Windows wider than 1024 are streamed in
// 512-candidate tiles: the best 512 survive in the first half of the sort
// scratch and merge with the next tile in the second half.  Scratch therefore
// stays fixed-size through the profile's 65,536-candidate maximum.
// e4m3 NaN codes (0x7f/0xff) decode to 0 like the torch LUT (stale slots).
// ---------------------------------------------------------------------------
constant constexpr int IDXTK_WIDTH = 1024;   // sort width (>= max candidates)
constant constexpr int IDXTK_KEEP = 512;
// The DSV4 indexer geometry is hard-wired (host-checked: q is
// [tokens, 64, 128] at the wrapper).
constant constexpr int IDXTK_H = 64;
constant constexpr int IDXTK_D = 128;

METAL_FUNC float idxtk_decode(uchar v) {
    return ((v & 0x7F) == 0x7F) ? 0.0f : float(tk_e4m3_decode(v));
}

METAL_FUNC void idxtk_sort(threadgroup float *keys, threadgroup int *vals,
                           uint tid) {
    for (int k = 2; k <= IDXTK_WIDTH; k <<= 1) {
        for (int jj = k >> 1; jj > 0; jj >>= 1) {
            for (int i = tid; i < IDXTK_WIDTH; i += 256) {
                const int ixj = i ^ jj;
                if (ixj > i) {
                    const bool desc = ((i & k) == 0);
                    const float ka = keys[i], kb = keys[ixj];
                    const int va = vals[i], vb = vals[ixj];
                    const bool in_order =
                        (ka > kb) || (ka == kb && va < vb);
                    if (in_order != desc) {
                        keys[i] = kb; keys[ixj] = ka;
                        vals[i] = vb; vals[ixj] = va;
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

template <typename T>
METAL_FUNC float idxtk_score_candidate(
        device const uchar *kv_cache, device const int *bt_row, int j,
        int block_size, long kv_block_stride, uint lane,
        threadgroup half *q_s, threadgroup float *w_s) {
    const long base =
        (long)bt_row[j / block_size] * kv_block_stride +
        (long)(j % block_size) * 132;
    device const uchar *code = kv_cache + base;
    const int h0 = lane, h1 = lane + 32;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int dg = 0; dg < IDXTK_D; dg += 32) {
        const float mine = idxtk_decode(code[dg + lane]);
        for (int i = 0; i < 32; ++i) {
            const float kd = metal::simd_broadcast(mine, (ushort)i);
            const int d = dg + i;
            acc0 += float(q_s[h0 * IDXTK_D + d]) * kd;
            acc1 += float(q_s[h1 * IDXTK_D + d]) * kd;
        }
    }
    float part = w_s[h0] * metal::max(acc0, 0.0f) +
                 w_s[h1] * metal::max(acc1, 0.0f);
    part = metal::simd_sum(part);
    const float k_scale = *((device const float *)(code + IDXTK_D));
    return part * k_scale;
}

template <typename T>
METAL_FUNC void dsv4_indexer_topk_decode_body(
        device const T *q,             // (tokens, H, D)
        device const float *w,         // (tokens, H)
        device const uchar *kv_cache,  // (blocks, block_size, 132)
        device const int *block_table, // (rows, bt_stride)
        device const int *cand,        // (tokens,)
        device int *out,               // (buf_rows, out_stride)
        constant int &width, constant int &k_eff, constant int &out_stride,
        constant int &block_size, constant int &bt_stride,
        constant long &kv_block_stride, uint3 tgid, uint tid, uint sg,
        uint lane, threadgroup half *q_s, threadgroup float *w_s,
        threadgroup float *keys, threadgroup int *vals,
        int bt_row_idx) {
    const int token = tgid.x;
    const int n_cand = metal::min(cand[token], width);

    // q stages as half even for bf16 activations: scoring runs in the
    // fp16 domain regardless of T (part of this kernel's numeric
    // contract with the eager mirror, which downcasts identically).
    for (int i = tid; i < IDXTK_H * IDXTK_D; i += 256) {
        q_s[i] = half(q[(long)token * IDXTK_H * IDXTK_D + i]);
    }
    if (tid < IDXTK_H) { w_s[tid] = w[(long)token * IDXTK_H + tid]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // One simdgroup per candidate; lane owns heads (lane, lane+32).  The
    // first pass fills the complete sort scratch.
    device const int *bt_row = block_table + (long)bt_row_idx * bt_stride;
    for (int j = sg; j < IDXTK_WIDTH; j += 8) {
        float logit = -INFINITY;
        if (j < n_cand) {
            logit = idxtk_score_candidate<T>(
                kv_cache, bt_row, j, block_size, kv_block_stride, lane,
                q_s, w_s);
        }
        if (lane == 0) {
            keys[j] = logit;
            vals[j] = j;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    idxtk_sort(keys, vals, tid);

    // Merge each remaining 512-column tile with the retained best 512.
    // The loop bound is the active request width supplied by CPU scheduler
    // metadata, never the profile's configured maximum for a short request.
    for (int base = IDXTK_WIDTH; base < width; base += IDXTK_KEEP) {
        for (int off = sg; off < IDXTK_KEEP; off += 8) {
            const int j = base + off;
            float logit = -INFINITY;
            if (j < n_cand) {
                logit = idxtk_score_candidate<T>(
                    kv_cache, bt_row, j, block_size, kv_block_stride, lane,
                    q_s, w_s);
            }
            if (lane == 0) {
                keys[IDXTK_KEEP + off] = logit;
                vals[IDXTK_KEEP + off] = j;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        idxtk_sort(keys, vals, tid);
    }

    device int *out_row = out + (long)token * out_stride;
    for (int k = tid; k < out_stride; k += 256) {
        out_row[k] =
            (k < k_eff && keys[k] != -INFINITY) ? vals[k] : -1;
    }
}

#define instantiate_dsv4_indexer_topk(suffix, T)                              \
  [[host_name("dsv4_indexer_topk_decode_" #suffix)]] kernel void              \
  dsv4_indexer_topk_decode_##suffix(                                          \
      device const T *q [[buffer(0)]], device const float *w [[buffer(1)]],   \
      device const uchar *kv_cache [[buffer(2)]],                             \
      device const int *block_table [[buffer(3)]],                            \
      device const int *cand [[buffer(4)]], device int *out [[buffer(5)]],    \
      constant int &width [[buffer(6)]], constant int &k_eff [[buffer(7)]],   \
      constant int &out_stride [[buffer(8)]],                                 \
      constant int &block_size [[buffer(9)]],                                 \
      constant int &bt_stride [[buffer(10)]],                                 \
      constant long &kv_block_stride [[buffer(11)]],                          \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    threadgroup half q_s[IDXTK_H * IDXTK_D];                                  \
    threadgroup float w_s[IDXTK_H];                                           \
    threadgroup float keys[IDXTK_WIDTH];                                      \
    threadgroup int vals[IDXTK_WIDTH];                                        \
    dsv4_indexer_topk_decode_body<T>(q, w, kv_cache, block_table, cand, out,  \
                                     width, k_eff, out_stride, block_size,    \
                                     bt_stride, kv_block_stride, tgid, tid,   \
                                     sg, lane, q_s, w_s, keys, vals,          \
                                     (int)tgid.x);                            \
  }

/* Prefill twin: the block-table row comes from the token's REQUEST
   (tok_req) instead of the token itself; per-token cand = ke - ks with
   request-local candidate windows starting at 0 (host-checked), so the
   scored columns, tie order, and output indices mirror the eager
   metal_indexer.py prefill chain (rebased request-local positions). */
#define instantiate_dsv4_indexer_topk_prefill(suffix, T)                       \
  [[host_name("dsv4_indexer_topk_prefill_" #suffix)]] kernel void              \
  dsv4_indexer_topk_prefill_##suffix(                                          \
      device const T *q [[buffer(0)]], device const float *w [[buffer(1)]],   \
      device const uchar *kv_cache [[buffer(2)]],                             \
      device const int *block_table [[buffer(3)]],                            \
      device const int *cand [[buffer(4)]], device int *out [[buffer(5)]],    \
      constant int &width [[buffer(6)]], constant int &k_eff [[buffer(7)]],   \
      constant int &out_stride [[buffer(8)]],                                 \
      constant int &block_size [[buffer(9)]],                                 \
      constant int &bt_stride [[buffer(10)]],                                 \
      constant long &kv_block_stride [[buffer(11)]],                          \
      device const int *tok_req [[buffer(12)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint tid [[thread_index_in_threadgroup]],                               \
      uint sg [[simdgroup_index_in_threadgroup]],                             \
      uint lane [[thread_index_in_simdgroup]]) {                              \
    threadgroup half q_s[IDXTK_H * IDXTK_D];                                  \
    threadgroup float w_s[IDXTK_H];                                           \
    threadgroup float keys[IDXTK_WIDTH];                                      \
    threadgroup int vals[IDXTK_WIDTH];                                        \
    dsv4_indexer_topk_decode_body<T>(q, w, kv_cache, block_table, cand, out,  \
                                     width, k_eff, out_stride, block_size,    \
                                     bt_stride, kv_block_stride, tgid, tid,   \
                                     sg, lane, q_s, w_s, keys, vals,          \
                                     tok_req[tgid.x]);                        \
  }

instantiate_dsv4_indexer_topk_prefill(float16, half);
instantiate_dsv4_indexer_topk_prefill(bfloat16, bf16);

instantiate_dsv4_indexer_topk(float16, half);
instantiate_dsv4_indexer_topk(bfloat16, bf16);

// ---------------------------------------------------------------------------
// Fused DeepSeek-V4 attention-compressor FRONT for the head=512 cr=4
// overlap layers (the 21 layers whose compress legitimately fires every
// spec step): gather the 2*ratio-row overlap history from the fp32 state
// cache ([kv 2*512 | score 2*512] rows, h >= ratio reads the +512 head),
// per-dim softmax over history, weighted-sum compress, RMSNorm with the
// fp32 weight, bf16 rows out. The existing native deepseek_v4_kv_insert
// consumes the output (RoPE at selected_pos + e4m3 + 584-byte record),
// so this kernel owns only the compress front from
// fused_compress_quant_cache.py. Same per-op mirrors as
// dsv4_indexer_compress_insert (explicit temporaries, precise
// transcendentals, reassociation/contraction off, sequential history
// sums); dims are processed in 4-wide chunks of the lane's contiguous
// 16-dim span to keep the history registers bounded.
// One simdgroup per token; non-boundary or state-slot<0 tokens exit
// early (the eager path masks the same rows via output_slots = -1, so
// their output rows are never read).
// ---------------------------------------------------------------------------
kernel void dsv4_compress_front(
        device const float *state_cache  [[buffer(0)]],  // (blocks, bs, 2048)
        device const int  *positions     [[buffer(1)]],
        device const long *state_slots   [[buffer(2)]],
        device const int  *token_to_req  [[buffer(3)]],
        device const int  *block_table   [[buffer(4)]],  // (reqs, bt_cols)
        device const float *rms_w        [[buffer(5)]],  // (512,) fp32
        device bf16       *out           [[buffer(6)]],  // (tokens, 512)
        constant int &state_block_size   [[buffer(7)]],
        constant int &state_stride0      [[buffer(8)]],
        constant int &state_stride1      [[buffer(9)]],
        constant int &state_width        [[buffer(10)]], // 1024
        constant int &compress_ratio     [[buffer(11)]], // 4 (overlap layout)
        constant int &bt_stride          [[buffer(12)]],
        constant int &bt_cols            [[buffer(13)]],
        constant float &eps              [[buffer(14)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint laneId [[thread_index_in_simdgroup]]) {
    constexpr int DIM = 512, PER_LANE = 16, CHUNK = 4, HISTORY_MAX = 8;
    const int token = blockIdx.x;
    const long sslot = state_slots[token];
    const int pos = positions[token];
    if (sslot < 0 || ((pos + 1) % compress_ratio) != 0) { return; }

    const int history = 2 * compress_ratio;
    // Same guard as dsv4_indexer_compress_insert: fail safely on a
    // contract-violating compress_ratio instead of overrunning rowp/okh.
    if (history > HISTORY_MAX) { return; }
    const int req = token_to_req[token];
    const int d0 = (int)laneId * PER_LANE;

    // History row pointers (kv head half applied; score at +state_width).
    device const float *rowp[HISTORY_MAX];
    bool okh[HISTORY_MAX];
    for (int h = 0; h < history; ++h) {
        const int p = pos - history + 1 + h;
        okh[h] = p >= 0;
        const int sp = metal::max(p, 0);
        const int colc = metal::min(sp / state_block_size, bt_cols - 1);
        const int blk = block_table[req * bt_stride + colc];
        const int off = sp % state_block_size;
        const int head_off = (h >= compress_ratio) ? DIM : 0;
        rowp[h] = state_cache + (long)blk * state_stride0 +
                  (long)off * state_stride1 + head_off;
    }

    float comp[PER_LANE];
    for (int c = 0; c < PER_LANE; c += CHUNK) {
        float vals[HISTORY_MAX][CHUNK];
        float scs[HISTORY_MAX][CHUNK];
        for (int h = 0; h < history; ++h) {
            device const float *row = rowp[h] + d0 + c;
            for (int k = 0; k < CHUNK; ++k) {
                vals[h][k] = row[k];
                scs[h][k] = okh[h] ? row[k + state_width] : -INFINITY;
            }
        }
        // Torch's rounding points: softmax materialized per element,
        // product rounded, then a sequential sum over the history dim.
        {
            #pragma clang fp reassociate(off) contract(off)
            for (int k = 0; k < CHUNK; ++k) {
                float m = scs[0][k];
                for (int h = 1; h < history; ++h) {
                    m = metal::max(m, scs[h][k]);
                }
                float ex[HISTORY_MAX];
                float s = 0.0f;
                for (int h = 0; h < history; ++h) {
                    ex[h] = metal::precise::exp(scs[h][k] - m);
                    s += ex[h];
                }
                float acc = 0.0f;
                for (int h = 0; h < history; ++h) {
                    const float smh = metal::precise::divide(ex[h], s);
                    const float prod = vals[h][k] * smh;
                    acc += prod;
                }
                comp[c + k] = acc;
            }
        }
    }

    float ss = 0.0f;
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            const float c2 = comp[k] * comp[k];
            ss += c2;
        }
    }
    ss = metal::simd_sum(ss);
    const float rrms = metal::precise::rsqrt(
        metal::precise::divide(ss, float(DIM)) + eps);

    device bf16 *o = out + (long)token * DIM + d0;
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            const float t1 = comp[k] * rrms;
            const float t2 = t1 * rms_w[d0 + k];
            o[k] = bf16(t2);
        }
    }
}

// ---------------------------------------------------------------------------
// cr=128 twin of dsv4_compress_front (the no-overlap second-level layers,
// state rows [kv 512 | score 512], state_width 512): only the ~tokens/128
// boundary rows need compressing, so non-boundary tokens exit immediately
// instead of running the full (tokens, 128, 512) gather+softmax chain.
// One 128-thread threadgroup per token; the 128 history row offsets are staged
// cooperatively (one per thread), then each thread owns 4 dims and walks
// the history three times (max, exp-sum, weighted acc) — the recomputed
// exp reproduces the materialized-softmax rounding, sums stay sequential
// in ascending history order like the c4 kernel's torch mirror.
// ---------------------------------------------------------------------------
kernel void dsv4_compress_front_c128(
        device const float *state_cache  [[buffer(0)]],  // (blocks, bs, 1024)
        device const int  *positions     [[buffer(1)]],
        device const long *state_slots   [[buffer(2)]],
        device const int  *token_to_req  [[buffer(3)]],
        device const int  *block_table   [[buffer(4)]],  // (reqs, bt_cols)
        device const float *rms_w        [[buffer(5)]],  // (512,) fp32
        device bf16       *out           [[buffer(6)]],  // (tokens, 512)
        constant int &state_block_size   [[buffer(7)]],
        constant int &state_stride0      [[buffer(8)]],
        constant int &state_stride1      [[buffer(9)]],
        constant int &state_width        [[buffer(10)]], // 512
        constant int &compress_ratio     [[buffer(11)]], // 128 (no overlap)
        constant int &bt_stride          [[buffer(12)]],
        constant int &bt_cols            [[buffer(13)]],
        constant float &eps              [[buffer(14)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint laneId [[thread_index_in_simdgroup]]) {
    // Coupled invariants: NTHREADS == HISTORY (each thread stages exactly
    // one history row below) and PER_LANE * NTHREADS == DIM; the launcher
    // dispatches exactly NTHREADS threads.
    constexpr int DIM = 512, PER_LANE = 4, HISTORY = 128, NTHREADS = 128;
    const int token = blockIdx.x;
    const long sslot = state_slots[token];
    const int pos = positions[token];
    if (sslot < 0 || ((pos + 1) % compress_ratio) != 0) { return; }

    const int req = token_to_req[token];
    const int d0 = (int)tid * PER_LANE;

    threadgroup long rowoff[HISTORY];
    threadgroup uchar okh[HISTORY];
    threadgroup float sg_ss[NTHREADS / 32];
    {
        const int h = (int)tid;
        const int p = pos - HISTORY + 1 + h;
        okh[h] = p >= 0;
        const int sp = metal::max(p, 0);
        const int colc = metal::min(sp / state_block_size, bt_cols - 1);
        const int blk = block_table[req * bt_stride + colc];
        const int off = sp % state_block_size;
        rowoff[h] = (long)blk * state_stride0 + (long)off * state_stride1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float comp[PER_LANE];
    for (int k = 0; k < PER_LANE; ++k) {
        const int d = d0 + k;
        float m = -INFINITY;
        for (int h = 0; h < HISTORY; ++h) {
            const float sc = okh[h] ? state_cache[rowoff[h] + state_width + d]
                                    : -INFINITY;
            m = metal::max(m, sc);
        }
        float s = 0.0f;
        {
            #pragma clang fp reassociate(off) contract(off)
            for (int h = 0; h < HISTORY; ++h) {
                const float sc = okh[h]
                    ? state_cache[rowoff[h] + state_width + d] : -INFINITY;
                s += metal::precise::exp(sc - m);
            }
        }
        float acc = 0.0f;
        {
            #pragma clang fp reassociate(off) contract(off)
            for (int h = 0; h < HISTORY; ++h) {
                const float sc = okh[h]
                    ? state_cache[rowoff[h] + state_width + d] : -INFINITY;
                const float smh = metal::precise::divide(
                    metal::precise::exp(sc - m), s);
                const float prod = state_cache[rowoff[h] + d] * smh;
                acc += prod;
            }
        }
        comp[k] = acc;
    }

    float ss = 0.0f;
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            const float c2 = comp[k] * comp[k];
            ss += c2;
        }
    }
    ss = metal::simd_sum(ss);
    if (laneId == 0) { sg_ss[sg] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        // Ascending-sg sequential fold matches the torch mirror's
        // reduction order.
        #pragma clang fp reassociate(off) contract(off)
        float total = sg_ss[0];
        for (int i = 1; i < NTHREADS / 32; ++i) { total += sg_ss[i]; }
        ss = total;
    }
    const float rrms = metal::precise::rsqrt(
        metal::precise::divide(ss, float(DIM)) + eps);

    device bf16 *o = out + (long)token * DIM + d0;
    {
        #pragma clang fp reassociate(off) contract(off)
        for (int k = 0; k < PER_LANE; ++k) {
            const float t1 = comp[k] * rrms;
            const float t2 = t1 * rms_w[d0 + k];
            o[k] = bf16(t2);
        }
    }
}
