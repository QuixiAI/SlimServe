// Multimodal rotary embedding (M-RoPE, Qwen2-VL-style) and its single-
// section degenerate form, positioned RoPE — contract mrope +
// rotary_positioned.
//
// positions is [n_sections, tokens] int64: each rotary pair i in
// [0, rot_dim/2) belongs to the section whose cumulative width (sections[],
// halves) covers i, and takes that section's position axis for its cos/sin
// row. cos_sin_cache is [max_pos, rot_dim] of the activation dtype, cos half
// then sin half (the vLLM layout). NeoX rotates (i, i + rot_dim/2); GPT-J
// rotates (2i, 2i+1). Query [tokens, n_heads, head_size] and key
// [tokens, n_kv_heads, head_size] rotate IN PLACE; dims past rot_dim are
// untouched. One work-item per (token, head, pair) — correctness-first.
//
// Semantics from vllm-xpu-kernels csrc/xpu/sycl/multimodal_rope.cpp
// (Apache; translated).

#include "attention/mrope/mrope_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kMaxSections = 4;

template <typename T>
class MropeKernel;

template <typename T>
sycl::event mrope_typed(sycl::queue& q, T* query, T* key, const T* cache,
                        const std::int64_t* positions,
                        const std::int32_t* sections, std::size_t n_sections,
                        std::size_t tokens, std::size_t n_heads,
                        std::size_t n_kv_heads, std::size_t head_size,
                        std::size_t rot_dim, bool neox) {
  const std::size_t embed = rot_dim / 2;
  const std::size_t heads_total = n_heads + n_kv_heads;
  return q.parallel_for<MropeKernel<T>>(
      sycl::range<1>(tokens * heads_total * embed), [=](sycl::id<1> idx) {
        const std::size_t i = idx[0] % embed;
        const std::size_t th = idx[0] / embed;
        const std::size_t t = th / heads_total;
        const std::size_t h = th % heads_total;

        // Section lookup (cumulative widths over the pair space).
        std::size_t sec = 0;
        if (sections != nullptr) {
          std::size_t hi = 0;
          for (std::size_t s2 = 0; s2 < n_sections; ++s2) {
            hi += static_cast<std::size_t>(sections[s2]);
            if (i < hi) {
              sec = s2;
              break;
            }
          }
        }
        const std::int64_t pos = positions[sec * tokens + t];
        const T* row = cache + static_cast<std::size_t>(pos) * rot_dim;
        const float c = static_cast<float>(row[i]);
        const float s = static_cast<float>(row[embed + i]);

        const bool is_q = h < n_heads;
        T* base = is_q ? query + (t * n_heads + h) * head_size
                       : key + (t * n_kv_heads + (h - n_heads)) * head_size;
        const std::size_t i1 = neox ? i : 2 * i;
        const std::size_t i2 = neox ? i + embed : 2 * i + 1;
        const float x1 = static_cast<float>(base[i1]);
        const float x2 = static_cast<float>(base[i2]);
        base[i1] = static_cast<T>(x1 * c - x2 * s);
        base[i2] = static_cast<T>(x2 * c + x1 * s);
      });
}

}  // namespace

sycl::event mrope_sycl(sycl::queue& q, void* query, void* key,
                       const void* cos_sin_cache,
                       const std::int64_t* positions,
                       const std::int32_t* sections, std::size_t n_sections,
                       std::size_t tokens, std::size_t n_heads,
                       std::size_t n_kv_heads, std::size_t head_size,
                       std::size_t rot_dim, int neox, DType dt) {
  if (n_sections == 0 || n_sections > kMaxSections || rot_dim % 2 != 0 ||
      rot_dim > head_size) {
    return {};  // reject, don't corrupt
  }
  switch (dt) {
    case DType::f32:
      return mrope_typed(q, static_cast<float*>(query),
                         static_cast<float*>(key),
                         static_cast<const float*>(cos_sin_cache), positions,
                         sections, n_sections, tokens, n_heads, n_kv_heads,
                         head_size, rot_dim, neox != 0);
    case DType::f16:
      return mrope_typed(q, static_cast<half_t*>(query),
                         static_cast<half_t*>(key),
                         static_cast<const half_t*>(cos_sin_cache), positions,
                         sections, n_sections, tokens, n_heads, n_kv_heads,
                         head_size, rot_dim, neox != 0);
    case DType::bf16:
      return mrope_typed(q, static_cast<bf16_t*>(query),
                         static_cast<bf16_t*>(key),
                         static_cast<const bf16_t*>(cos_sin_cache), positions,
                         sections, n_sections, tokens, n_heads, n_kv_heads,
                         head_size, rot_dim, neox != 0);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
