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
