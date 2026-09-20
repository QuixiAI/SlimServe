#pragma once

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// TurboQuant KV-cache codec, format version 2 (specs/formats/turboquant.md).
//
// Keys take the rotated Lloyd-Max path: sign vector, unnormalized FWHT,
// 1/sqrt(head_size), per-group FP16 RMS, then a centroid table with midpoint
// decision boundaries. Values take per-group uniform quantization with an FP16
// scale and zero. The rotated path is on keys, not values: attention scores are
// inner products, so key fidelity governs the score distribution.
//
// key/value are [num_tokens, num_kv_heads, head_size] in dt. Caches are indexed
// by slot = slot_mapping[token]; slot < 0 skips the token. head_size is 64, 128
// or 256; key_bits and value_bits are in [2,8]; a scale group is 32 elements.
// centroids is [2^key_bits] ascending fp32, signs is [head_size] +-1 fp32.
//
// The FP16 rounding chain is load-bearing: scales, zeros and the RMS divide are
// computed in fp16 so the codes match the reference oracle bit for bit.
sycl::event turboquant_encode_sycl(
    sycl::queue& q, const void* key, const void* value, std::uint8_t* key_cache,
    std::uint8_t* value_cache, void* key_scale_cache, void* value_scale_cache,
    void* value_zero_cache, const std::int64_t* slot_mapping,
    const float* centroids, const float* signs, std::size_t num_tokens,
    std::size_t num_kv_heads, std::size_t head_size, int key_bits,
    int value_bits, int value_signed, DType dt);

// Inverse of turboquant_encode_sycl for a gathered list of slots. k_out and
// v_out are [num_slots, num_kv_heads, head_size] fp32.
sycl::event turboquant_decode_sycl(
    sycl::queue& q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const void* key_scale_cache,
    const void* value_scale_cache, const void* value_zero_cache,
    const std::int64_t* slots, const float* centroids, const float* signs,
    float* k_out, float* v_out, std::size_t num_slots,
    std::size_t num_kv_heads, std::size_t head_size, int key_bits,
    int value_bits, int value_signed);

}  // namespace quixicore::xpu::kernels
