// TurboQuant KV-cache codec kernels (format version 2). The per-row codec
// lives in turboquant_codec.hpp and is shared verbatim with the host test
// oracle, so bit-for-bit agreement is structural. One work-item owns a whole
// (token, head) row: the LSB-first bit stream lets codes cross byte
// boundaries, so no two work-items may share a byte, and the sequential
// per-group reductions keep the fp32 accumulation order identical to the
// oracle. Correctness-first geometry; a subgroup-cooperative variant is a
// later throughput lever.
//
// Cache layouts (slot-indexed): key_cache [slot, num_kv_heads,
// row_bytes(head_size, key_bits)] u8; value_cache likewise with value_bits;
// key_scale_cache / value_scale_cache / value_zero_cache
// [slot, num_kv_heads, head_size/32] fp16. slot_mapping < 0 skips a token.

#include "quantization/turboquant/turboquant_kernel.hpp"
#include "quantization/turboquant/turboquant_codec.hpp"

namespace quixicore::xpu::kernels {
namespace {

namespace codec = quixicore::xpu::turboquant_codec;

template <typename T>
class TurboquantEncodeKernel;
class TurboquantDecodeKernel;

template <typename T>
sycl::event encode_typed(sycl::queue& q, const T* key, const T* value,
                         std::uint8_t* key_cache, std::uint8_t* value_cache,
                         sycl::half* key_scales, sycl::half* value_scales,
                         sycl::half* value_zeros,
                         const std::int64_t* slot_mapping,
                         const float* centroids, const float* signs,
                         std::size_t num_tokens, std::size_t num_kv_heads,
                         std::size_t head_size, int key_bits, int value_bits,
                         bool value_signed) {
  const std::size_t kbytes = codec::row_bytes(head_size, key_bits);
  const std::size_t vbytes = codec::row_bytes(head_size, value_bits);
  const std::size_t groups = head_size / codec::kGroup;
  return q.parallel_for<TurboquantEncodeKernel<T>>(
      sycl::range<1>(num_tokens * num_kv_heads), [=](sycl::id<1> idx) {
        const std::size_t t = idx[0] / num_kv_heads;
        const std::size_t h = idx[0] - t * num_kv_heads;
        const std::int64_t slot = slot_mapping[t];
        if (slot < 0) return;
        const std::size_t src = (t * num_kv_heads + h) * head_size;
        const std::size_t dst = static_cast<std::size_t>(slot) * num_kv_heads + h;

        float scratch[codec::kMaxHead];
        codec::encode_key_row(key + src, key_cache + dst * kbytes,
                              key_scales + dst * groups, centroids, signs,
                              head_size, key_bits, scratch);
        codec::encode_value_row(value + src, value_cache + dst * vbytes,
                                value_scales + dst * groups,
                                value_zeros + dst * groups, head_size,
                                value_bits, value_signed);
      });
}

}  // namespace

sycl::event turboquant_encode_sycl(
    sycl::queue& q, const void* key, const void* value, std::uint8_t* key_cache,
    std::uint8_t* value_cache, void* key_scale_cache, void* value_scale_cache,
    void* value_zero_cache, const std::int64_t* slot_mapping,
    const float* centroids, const float* signs, std::size_t num_tokens,
    std::size_t num_kv_heads, std::size_t head_size, int key_bits,
    int value_bits, int value_signed, DType dt) {
  if (head_size > codec::kMaxHead || head_size % codec::kGroup != 0 ||
      key_bits < 2 || key_bits > 8 || value_bits < 2 || value_bits > 8) {
    return {};  // outside the format envelope: reject, don't corrupt
  }
  auto* ks = static_cast<sycl::half*>(key_scale_cache);
  auto* vs = static_cast<sycl::half*>(value_scale_cache);
  auto* vz = static_cast<sycl::half*>(value_zero_cache);
  switch (dt) {
    case DType::f32:
      return encode_typed(q, static_cast<const float*>(key),
                          static_cast<const float*>(value), key_cache,
                          value_cache, ks, vs, vz, slot_mapping, centroids,
                          signs, num_tokens, num_kv_heads, head_size, key_bits,
                          value_bits, value_signed != 0);
    case DType::f16:
      return encode_typed(q, static_cast<const half_t*>(key),
                          static_cast<const half_t*>(value), key_cache,
                          value_cache, ks, vs, vz, slot_mapping, centroids,
                          signs, num_tokens, num_kv_heads, head_size, key_bits,
                          value_bits, value_signed != 0);
    case DType::bf16:
      return encode_typed(q, static_cast<const bf16_t*>(key),
                          static_cast<const bf16_t*>(value), key_cache,
                          value_cache, ks, vs, vz, slot_mapping, centroids,
                          signs, num_tokens, num_kv_heads, head_size, key_bits,
                          value_bits, value_signed != 0);
  }
  return {};
}

sycl::event turboquant_decode_sycl(
    sycl::queue& q, const std::uint8_t* key_cache,
    const std::uint8_t* value_cache, const void* key_scale_cache,
    const void* value_scale_cache, const void* value_zero_cache,
    const std::int64_t* slots, const float* centroids, const float* signs,
    float* k_out, float* v_out, std::size_t num_slots,
    std::size_t num_kv_heads, std::size_t head_size, int key_bits,
    int value_bits, int value_signed) {
  if (head_size > codec::kMaxHead || head_size % codec::kGroup != 0 ||
      key_bits < 2 || key_bits > 8 || value_bits < 2 || value_bits > 8) {
    return {};
  }
  const auto* ks = static_cast<const sycl::half*>(key_scale_cache);
  const auto* vs = static_cast<const sycl::half*>(value_scale_cache);
  const auto* vz = static_cast<const sycl::half*>(value_zero_cache);
  const std::size_t kbytes = codec::row_bytes(head_size, key_bits);
  const std::size_t vbytes = codec::row_bytes(head_size, value_bits);
  const std::size_t groups = head_size / codec::kGroup;
  const bool vsigned = value_signed != 0;
  return q.parallel_for<TurboquantDecodeKernel>(
      sycl::range<1>(num_slots * num_kv_heads), [=](sycl::id<1> idx) {
        const std::size_t s = idx[0] / num_kv_heads;
        const std::size_t h = idx[0] - s * num_kv_heads;
        const std::int64_t slot = slots[s];
        const std::size_t out = (s * num_kv_heads + h) * head_size;
        if (slot < 0) {
          for (std::size_t i = 0; i < head_size; ++i) {
            k_out[out + i] = 0.0f;
            v_out[out + i] = 0.0f;
          }
          return;
        }
        const std::size_t src = static_cast<std::size_t>(slot) * num_kv_heads + h;

        float scratch[codec::kMaxHead];
        codec::decode_key_row(key_cache + src * kbytes, ks + src * groups,
                              centroids, signs, head_size, key_bits,
                              k_out + out, scratch);
        codec::decode_value_row(value_cache + src * vbytes, vs + src * groups,
                                vz + src * groups, head_size, value_bits,
                                vsigned, v_out + out);
      });
}

}  // namespace quixicore::xpu::kernels
