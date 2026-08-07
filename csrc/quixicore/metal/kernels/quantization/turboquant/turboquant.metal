#include "tk.metal"
#include <metal_stdlib>

using namespace metal;

namespace mittens {

// ---------------------------------------------------------------------------
// TurboQuant KV-cache codec (arXiv 2502; ported from metal-forge turboquant.metal).
// K: asymmetric uniform quant, per-32-element fp16 scale + zero-point, 2..8 bits.
// V: random-sign FWHT rotation -> per-32 fp16 RMS scale -> Lloyd-Max nearest-centroid
//    (searchsorted against midpoint boundaries) -> packed 2/3/4/8 bits.
// The fp16 arithmetic chain (scale/zp/normalize all routed through half) is kept
// VERBATIM from the reference so a numpy fp16 oracle reproduces codes bit-for-bit.
// TM divergences: random signs + centroids arrive as BUFFERS (host tk.quant.tq_signs /
// lloyd_max_centroids — any head size / seed, no baked tables); k_bits / k_signed /
// v_bits are runtime scalars (not function constants); slot_mapping is int32.
// Attention integration (rotated-domain V accumulate + one deferred inverse FWHT)
// is spec'd in the reference and deferred — the cache format here already supports it.
//
// Grid: (num_tokens, num_kv_heads); threadgroup = HEAD_SIZE threads; one simdgroup
// (32 lanes) == one scale group, so min/max/RMS reductions are simd_* ops.
// Cache layouts (num_blocks, block_size, Hkv, X) with X = packed bytes or HS/32 scales.
// ---------------------------------------------------------------------------

// element i occupies bits [i*bits, i*bits+bits) little-endian in the packed stream
inline uint tq_unpack_bits(const device uchar *bytes, int elem_idx, int bits) {
    const int bit_pos = elem_idx * bits;
    const int byte_idx = bit_pos >> 3;
    const int bit_offset = bit_pos & 7;
    uint raw = uint(bytes[byte_idx]);
    if (bit_offset + bits > 8) {
        raw |= uint(bytes[byte_idx + 1]) << 8;
    }
    return (raw >> bit_offset) & ((1u << bits) - 1u);
}

// assemble one output byte from staged per-element indices (a byte may straddle two
// elements when bits doesn't divide 8, e.g. 3-bit: 8 values -> 3 bytes)
inline uint tq_pack_byte(threadgroup const uchar *idx_buf, int bit_start, int bits) {
    const int first_e = bit_start / bits;
    const int last_e = (bit_start + 7) / bits;
    const uint mask = (1u << bits) - 1u;
    uint byte = 0;
    for (int e = first_e; e <= last_e; e++) {
        const int shift = e * bits - bit_start;
        const uint idx_v = uint(idx_buf[e]) & mask;
        byte |= (shift >= 0) ? (idx_v << shift) : (idx_v >> (-shift));
    }
    return byte & 0xFFu;
}

template <int HEAD_SIZE> inline constexpr int tq_fwht_stages() { return 0; }
template <> inline constexpr int tq_fwht_stages<64>() { return 6; }
template <> inline constexpr int tq_fwht_stages<128>() { return 7; }
template <> inline constexpr int tq_fwht_stages<256>() { return 8; }

template <int HEAD_SIZE> inline constexpr float tq_fwht_inv_sqrt_n() { return 1.0f; }
template <> inline constexpr float tq_fwht_inv_sqrt_n<64>() { return 0.125f; }
template <> inline constexpr float tq_fwht_inv_sqrt_n<128>() { return 0.08838834764831843f; }
template <> inline constexpr float tq_fwht_inv_sqrt_n<256>() { return 0.0625f; }

// forward: x -> H(sign * x) / sqrt(N). Stages 0-4 are intra-simdgroup shuffles; the rest
// go through threadgroup memory. One element per thread (t in [0, HEAD_SIZE)).
template <int HEAD_SIZE>
inline float tq_forward_fwht_scalar(float x, const device float *signs,
                                    threadgroup float *buf, uint t) {
    constexpr int NUM_STAGES = tq_fwht_stages<HEAD_SIZE>();
    x *= signs[t];
    #pragma unroll
    for (int stage = 0; stage < 5; stage++) {
        const uint mask = 1u << stage;
        const float partner = simd_shuffle_xor(x, mask);
        x = (t & mask) ? (partner - x) : (x + partner);
    }
    buf[t] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (int stage = 5; stage < NUM_STAGES; stage++) {
        const uint mask = 1u << stage;
        const float me = buf[t];
        const float partner = buf[t ^ mask];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[t] = (t & mask) ? (partner - me) : (me + partner);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return buf[t] * tq_fwht_inv_sqrt_n<HEAD_SIZE>();
}

// inverse: y -> sign * H(y) / sqrt(N)  (signs applied after the butterflies)
template <int HEAD_SIZE>
inline float tq_inverse_fwht_scalar(float x, const device float *signs,
                                    threadgroup float *buf, uint t) {
    constexpr int NUM_STAGES = tq_fwht_stages<HEAD_SIZE>();
    #pragma unroll
    for (int stage = 0; stage < 5; stage++) {
        const uint mask = 1u << stage;
        const float partner = simd_shuffle_xor(x, mask);
        x = (t & mask) ? (partner - x) : (x + partner);
    }
    buf[t] = x;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (int stage = 5; stage < NUM_STAGES; stage++) {
        const uint mask = 1u << stage;
        const float me = buf[t];
        const float partner = buf[t ^ mask];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        buf[t] = (t & mask) ? (partner - me) : (me + partner);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return buf[t] * tq_fwht_inv_sqrt_n<HEAD_SIZE>() * signs[t];
}

template <typename T, int HEAD_SIZE>
kernel void tq_encode(device const T *key      [[buffer(0)]],  // (tokens, Hkv, HS)
                      device const T *value    [[buffer(1)]],
                      device uchar *key_cache  [[buffer(2)]],  // (blocks, bs, Hkv, k_packed)
                      device uchar *value_cache [[buffer(3)]], // (blocks, bs, Hkv, v_packed)
                      device half *key_scale_cache [[buffer(4)]],   // (..., HS/32)
                      device half *value_scale_cache [[buffer(5)]],
                      device half *key_zero_cache [[buffer(6)]],
                      device const int *slot_mapping [[buffer(7)]], // (tokens,) -1 skip
                      device const float *v_centroids [[buffer(8)]], // (2^v_bits,) ascending
                      device const float *signs [[buffer(9)]],       // (HS,) +-1
                      constant int &num_kv_heads [[buffer(10)]],
                      constant int &block_size [[buffer(11)]],
                      constant int &k_bits [[buffer(12)]],
                      constant int &k_signed [[buffer(13)]],
                      constant int &v_bits [[buffer(14)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint3 tid3 [[thread_position_in_threadgroup]],
                      uint sid  [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr int SG_SIZE = 32;
    const uint t = tid3.x;
    const int token = int(tgid.x);
    const int kvh = int(tgid.y);

    const int slot = slot_mapping[token];
    if (slot < 0) {
        return;
    }
    const int block_idx = slot / block_size;
    const int block_off = slot - block_idx * block_size;

    constexpr int scale_groups = HEAD_SIZE / SG_SIZE;
    const int k_packed = (HEAD_SIZE * k_bits + 7) / 8;
    const int v_packed = (HEAD_SIZE * v_bits + 7) / 8;

    const long src_base = ((long)token * num_kv_heads + kvh) * HEAD_SIZE;
    const float k_val = float(key[src_base + t]);
    const float v_val = float(value[src_base + t]);

    threadgroup float fwht_buf[HEAD_SIZE];
    threadgroup uchar k_idx_buf[HEAD_SIZE];
    threadgroup uchar v_idx_buf[HEAD_SIZE];
    threadgroup float v_boundaries[255];   // midpoints; prefix used for v_bits < 8

    const int num_centroids = 1 << v_bits;
    for (int i = int(t); i < num_centroids - 1; i += HEAD_SIZE) {
        v_boundaries[i] = 0.5f * (v_centroids[i] + v_centroids[i + 1]);
    }

    const long token_base = ((long)block_idx * block_size + block_off) * num_kv_heads;
    const long kc_base = (token_base + kvh) * k_packed;
    const long vc_base = (token_base + kvh) * v_packed;
    const long scale_base = (token_base + kvh) * scale_groups;

    // ---- K: asymmetric uniform, all arithmetic routed through fp16 (reference-exact) ----
    const float k_min_f = simd_min(k_val);
    const float k_max_f = simd_max(k_val);
    half k_scale_h;
    float k_zp_f;
    int k_idx_i;
    if (k_signed != 0) {
        const int max_val = (1 << (k_bits - 1)) - 1;
        k_scale_h = half(half(k_max_f - k_min_f) / half(2.0f * float(max_val)));
        const half k_sum_h = half(k_max_f + k_min_f);
        k_zp_f = rint(float(k_sum_h / (half(2.0f) * k_scale_h)));
        k_idx_i = clamp(int(rint(float(half(k_val) / k_scale_h) - k_zp_f)), -max_val, max_val);
    } else {
        const int max_val = (1 << k_bits) - 1;
        k_scale_h = half(half(k_max_f - k_min_f) / half(float(max_val)));
        k_zp_f = rint(float(half(k_min_f) / k_scale_h));
        k_idx_i = clamp(int(rint(float(half(k_val) / k_scale_h) - k_zp_f)), 0, max_val);
    }
    if (lane == 0) {
        key_scale_cache[scale_base + int(sid)] = k_scale_h;
        key_zero_cache[scale_base + int(sid)] = half(k_zp_f);
    }
    if (k_bits == 8) {
        key_cache[kc_base + t] = uchar(uint(k_idx_i) & 0xFFu);   // two's complement if signed
    } else {
        k_idx_buf[t] = uchar(uint(k_idx_i) & ((1u << k_bits) - 1u));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(t) < k_packed) {
            key_cache[kc_base + t] = uchar(tq_pack_byte(k_idx_buf, int(t) * 8, k_bits));
        }
    }

    // ---- V: FWHT rotation -> fp16 RMS scale -> Lloyd-Max searchsorted -> pack ----
    const float v_rot_f = tq_forward_fwht_scalar<HEAD_SIZE>(v_val, signs, fwht_buf, t);
    const half v_rot_h = half(v_rot_f);
    const float v_sqsum = simd_sum(float(v_rot_h) * float(v_rot_h));
    const half v_scale_h = half(metal::sqrt(v_sqsum * (1.0f / float(SG_SIZE))));
    const float v_norm = float(v_rot_h / v_scale_h);

    threadgroup_barrier(mem_flags::mem_threadgroup);   // publishes v_boundaries
    int v_idx = 0;
    for (int i = 0; i < num_centroids - 1; i++) {
        if (v_norm > v_boundaries[i]) v_idx++;
    }
    v_idx_buf[t] = uchar(v_idx & (num_centroids - 1));
    if (lane == 0) {
        value_scale_cache[scale_base + int(sid)] = v_scale_h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) < v_packed) {
        value_cache[vc_base + t] = uchar(tq_pack_byte(v_idx_buf, int(t) * 8, v_bits));
    }
}

// gather + dequantize a slot list back to float rows (round-trip validation / cache
// export; the attention hot path will fuse this instead of calling it).
template <typename T, int HEAD_SIZE>
kernel void tq_decode(device const uchar *key_cache [[buffer(0)]],
                      device const uchar *value_cache [[buffer(1)]],
                      device const half *key_scale_cache [[buffer(2)]],
                      device const half *value_scale_cache [[buffer(3)]],
                      device const half *key_zero_cache [[buffer(4)]],
                      device const int *slots [[buffer(5)]],       // (n,)
                      device const float *v_centroids [[buffer(6)]],
                      device const float *signs [[buffer(7)]],
                      device T *k_out [[buffer(8)]],               // (n, Hkv, HS)
                      device T *v_out [[buffer(9)]],
                      constant int &num_kv_heads [[buffer(10)]],
                      constant int &block_size [[buffer(11)]],
                      constant int &k_bits [[buffer(12)]],
                      constant int &k_signed [[buffer(13)]],
                      constant int &v_bits [[buffer(14)]],
                      uint3 tgid [[threadgroup_position_in_grid]],
                      uint3 tid3 [[thread_position_in_threadgroup]],
                      uint sid  [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
    constexpr int SG_SIZE = 32;
    const uint t = tid3.x;
    const int row = int(tgid.x);
    const int kvh = int(tgid.y);
    const int slot = slots[row];
    const int block_idx = slot / block_size;
    const int block_off = slot - block_idx * block_size;

    constexpr int scale_groups = HEAD_SIZE / SG_SIZE;
    const int k_packed = (HEAD_SIZE * k_bits + 7) / 8;
    const int v_packed = (HEAD_SIZE * v_bits + 7) / 8;
    const long token_base = ((long)block_idx * block_size + block_off) * num_kv_heads;
    const long kc_base = (token_base + kvh) * k_packed;
    const long vc_base = (token_base + kvh) * v_packed;
    const long scale_base = (token_base + kvh) * scale_groups;
    const long out_base = ((long)row * num_kv_heads + kvh) * HEAD_SIZE;

    threadgroup float fwht_buf[HEAD_SIZE];

    // K
    const float ks = float(key_scale_cache[scale_base + int(sid)]);
    const float kz = float(key_zero_cache[scale_base + int(sid)]);
    float kq;
    if (k_bits == 8 && k_signed != 0) {
        kq = float(char(key_cache[kc_base + t]));
    } else if (k_bits == 8) {
        kq = float(key_cache[kc_base + t]);
    } else {
        kq = float(tq_unpack_bits(key_cache + kc_base, int(t), k_bits));
    }
    k_out[out_base + t] = T((kq + kz) * ks);

    // V: centroid lookup * scale, then inverse FWHT across the head
    const float vs = float(value_scale_cache[scale_base + int(sid)]);
    const uint v_idx = tq_unpack_bits(value_cache + vc_base, int(t), v_bits);
    const float v_rot = v_centroids[v_idx & ((1u << v_bits) - 1u)] * vs;
    v_out[out_base + t] = T(tq_inverse_fwht_scalar<HEAD_SIZE>(v_rot, signs, fwht_buf, t));
}

// generic byte clone for the functional cache-update prepass (u8 / f16 buffers alike)
kernel void tq_clone_bytes(device const uchar *src [[buffer(0)]],
                           device uchar *dst [[buffer(1)]],
                           constant uint &n [[buffer(2)]],
                           uint tid [[thread_position_in_grid]]) {
    const uint base = tid * 16;
    if (base + 16 <= n) {
        ((device uchar4*)(dst + base))[0] = ((device const uchar4*)(src + base))[0];
        ((device uchar4*)(dst + base))[1] = ((device const uchar4*)(src + base))[1];
        ((device uchar4*)(dst + base))[2] = ((device const uchar4*)(src + base))[2];
        ((device uchar4*)(dst + base))[3] = ((device const uchar4*)(src + base))[3];
    } else {
        for (uint i = base; i < n; ++i) dst[i] = src[i];
    }
}

#define instantiate_tq(type_name, T, HS)                                            \
  template [[host_name("tq_encode_" #type_name "_hs" #HS)]] [[kernel]] void         \
  tq_encode<T, HS>(device const T *key [[buffer(0)]],                               \
      device const T *value [[buffer(1)]], device uchar *key_cache [[buffer(2)]],   \
      device uchar *value_cache [[buffer(3)]],                                      \
      device half *key_scale_cache [[buffer(4)]],                                   \
      device half *value_scale_cache [[buffer(5)]],                                 \
      device half *key_zero_cache [[buffer(6)]],                                    \
      device const int *slot_mapping [[buffer(7)]],                                 \
      device const float *v_centroids [[buffer(8)]],                                \
      device const float *signs [[buffer(9)]],                                      \
      constant int &num_kv_heads [[buffer(10)]], constant int &block_size [[buffer(11)]], \
      constant int &k_bits [[buffer(12)]], constant int &k_signed [[buffer(13)]],   \
      constant int &v_bits [[buffer(14)]],                                          \
      uint3 tgid [[threadgroup_position_in_grid]],                                  \
      uint3 tid3 [[thread_position_in_threadgroup]],                                \
      uint sid [[simdgroup_index_in_threadgroup]],                                  \
      uint lane [[thread_index_in_simdgroup]]);                                     \
  template [[host_name("tq_decode_" #type_name "_hs" #HS)]] [[kernel]] void         \
  tq_decode<T, HS>(device const uchar *key_cache [[buffer(0)]],                     \
      device const uchar *value_cache [[buffer(1)]],                                \
      device const half *key_scale_cache [[buffer(2)]],                             \
      device const half *value_scale_cache [[buffer(3)]],                           \
      device const half *key_zero_cache [[buffer(4)]],                              \
      device const int *slots [[buffer(5)]],                                        \
      device const float *v_centroids [[buffer(6)]],                                \
      device const float *signs [[buffer(7)]],                                      \
      device T *k_out [[buffer(8)]], device T *v_out [[buffer(9)]],                 \
      constant int &num_kv_heads [[buffer(10)]], constant int &block_size [[buffer(11)]], \
      constant int &k_bits [[buffer(12)]], constant int &k_signed [[buffer(13)]],   \
      constant int &v_bits [[buffer(14)]],                                          \
      uint3 tgid [[threadgroup_position_in_grid]],                                  \
      uint3 tid3 [[thread_position_in_threadgroup]],                                \
      uint sid [[simdgroup_index_in_threadgroup]],                                  \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_tq_all(type_name, T) \
  instantiate_tq(type_name, T, 64)       \
  instantiate_tq(type_name, T, 128)      \
  instantiate_tq(type_name, T, 256)

instantiate_tq_all(float32, float)
instantiate_tq_all(float16, half)
instantiate_tq_all(bfloat16, bf16)

} // namespace mittens
