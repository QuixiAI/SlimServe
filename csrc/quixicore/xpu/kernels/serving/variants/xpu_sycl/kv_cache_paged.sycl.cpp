// Paged KV-cache write (the write side of paged_attention_*): scatter each
// token's K and V head vectors into the page-table-addressed caches, with
// optional fp8 e4m3 encoding — contract kv_cache_scatter, paged entry
// (vLLM reshape_and_cache_flash semantics; Apache source translated).
//
// key/value are [n_tokens, n_kv_heads, d]; caches are
// [n_pages, page_size, n_kv_heads, d] with page_stride_elems pitch;
// slot_mapping[t] is the FLAT slot (page * page_size + offset), < 0 skips.
// fp8 mode divides by the device-scalar scales before the integer-exact
// e4m3 encode (decode side multiplies — matches paged_attention_*).

#include "serving/serving_kernel.hpp"

#include "quantization/turboquant/turboquant_codec.hpp"

namespace quixicore::xpu::kernels {
namespace {

namespace codec = quixicore::xpu::turboquant_codec;

template <typename T, bool Fp8>
class KvCachePagedWriteKernel;

template <typename T, bool Fp8>
sycl::event paged_write_typed(sycl::queue& q, const T* key, const T* value,
                              void* k_cache, void* v_cache,
                              const std::int64_t* slot_mapping,
                              std::size_t n_tokens, std::size_t n_kv_heads,
                              std::size_t d, std::size_t page_size,
                              std::size_t page_stride_elems,
                              const float* k_scale, const float* v_scale) {
  const std::size_t row = n_kv_heads * d;
  return q.parallel_for<KvCachePagedWriteKernel<T, Fp8>>(
      sycl::range<1>(n_tokens * row), [=](sycl::id<1> idx) {
        const std::size_t t = idx[0] / row;
        const std::size_t e = idx[0] % row;
        const std::int64_t slot = slot_mapping[t];
        if (slot < 0) return;
        const std::size_t page = static_cast<std::size_t>(slot) / page_size;
        const std::size_t off = static_cast<std::size_t>(slot) % page_size;
        const std::size_t dst = page * page_stride_elems + off * row + e;
        if constexpr (Fp8) {
          const float ks = k_scale != nullptr ? *k_scale : 1.0f;
          const float vs = v_scale != nullptr ? *v_scale : 1.0f;
          static_cast<std::uint8_t*>(k_cache)[dst] = codec::e4m3_encode(
              static_cast<float>(key[t * row + e]) / ks);
          static_cast<std::uint8_t*>(v_cache)[dst] = codec::e4m3_encode(
              static_cast<float>(value[t * row + e]) / vs);
        } else {
          static_cast<T*>(k_cache)[dst] = key[t * row + e];
          static_cast<T*>(v_cache)[dst] = value[t * row + e];
        }
      });
}

}  // namespace

sycl::event kv_cache_scatter_paged_sycl(
    sycl::queue& q, const void* key, const void* value, void* k_cache,
    void* v_cache, const std::int64_t* slot_mapping, std::size_t n_tokens,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    int fp8, DType dt) {
#define QX_KVP(T, F)                                                       \
  return paged_write_typed<T, F>(q, static_cast<const T*>(key),            \
                                 static_cast<const T*>(value), k_cache,    \
                                 v_cache, slot_mapping, n_tokens,          \
                                 n_kv_heads, d, page_size,                 \
                                 page_stride_elems, k_scale, v_scale)
  switch (dt) {
    case DType::f32:
      if (fp8) QX_KVP(float, true);
      QX_KVP(float, false);
    case DType::f16:
      if (fp8) QX_KVP(half_t, true);
      QX_KVP(half_t, false);
    case DType::bf16:
      if (fp8) QX_KVP(bf16_t, true);
      QX_KVP(bf16_t, false);
  }
#undef QX_KVP
  return {};
}


namespace {

template <typename T, bool Fp8>
class KvCachePagedGatherKernel;

template <typename T, bool Fp8>
sycl::event paged_gather_typed(sycl::queue& q, const void* k_cache,
                               const void* v_cache, T* k_out, T* v_out,
                               const std::int64_t* slots, std::size_t n,
                               std::size_t n_kv_heads, std::size_t d,
                               std::size_t page_size,
                               std::size_t page_stride_elems,
                               const float* k_scale, const float* v_scale) {
  const std::size_t row = n_kv_heads * d;
  return q.parallel_for<KvCachePagedGatherKernel<T, Fp8>>(
      sycl::range<1>(n * row), [=](sycl::id<1> idx) {
        const std::size_t t = idx[0] / row;
        const std::size_t e = idx[0] % row;
        const std::int64_t slot = slots[t];
        if (slot < 0) {
          k_out[idx[0]] = T(0.0f);
          v_out[idx[0]] = T(0.0f);
          return;
        }
        const std::size_t page = static_cast<std::size_t>(slot) / page_size;
        const std::size_t off = static_cast<std::size_t>(slot) % page_size;
        const std::size_t src = page * page_stride_elems + off * row + e;
        if constexpr (Fp8) {
          const float ks = k_scale != nullptr ? *k_scale : 1.0f;
          const float vs = v_scale != nullptr ? *v_scale : 1.0f;
          k_out[idx[0]] = static_cast<T>(
              codec::e4m3_decode(static_cast<const std::uint8_t*>(k_cache)[src]) * ks);
          v_out[idx[0]] = static_cast<T>(
              codec::e4m3_decode(static_cast<const std::uint8_t*>(v_cache)[src]) * vs);
        } else {
          k_out[idx[0]] = static_cast<const T*>(k_cache)[src];
          v_out[idx[0]] = static_cast<const T*>(v_cache)[src];
        }
      });
}

}  // namespace

sycl::event kv_cache_gather_paged_sycl(
    sycl::queue& q, const void* k_cache, const void* v_cache, void* k_out,
    void* v_out, const std::int64_t* slots, std::size_t n,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    int fp8, DType dt) {
#define QX_KVG(T, F)                                                        \
  return paged_gather_typed<T, F>(q, k_cache, v_cache,                      \
                                  static_cast<T*>(k_out),                   \
                                  static_cast<T*>(v_out), slots, n,         \
                                  n_kv_heads, d, page_size,                 \
                                  page_stride_elems, k_scale, v_scale)
  switch (dt) {
    case DType::f32:
      if (fp8) QX_KVG(float, true);
      QX_KVG(float, false);
    case DType::f16:
      if (fp8) QX_KVG(half_t, true);
      QX_KVG(half_t, false);
    case DType::bf16:
      if (fp8) QX_KVG(bf16_t, true);
      QX_KVG(bf16_t, false);
  }
#undef QX_KVG
  return {};
}

}  // namespace quixicore::xpu::kernels
