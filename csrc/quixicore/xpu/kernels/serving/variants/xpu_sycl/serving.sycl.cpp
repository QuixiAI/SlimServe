// Serving family: embedding lookup + KV-cache scatter/gather. All are indexed
// row copies; the payload is copied by element width (uint16 for f16/bf16,
// uint32 for f32), so one templated kernel covers every dtype. 2D launch
// [rows, row_width] keeps the inner copy coalesced.

#include <stdexcept>

#include "serving/serving_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

// index_src=true: gather, dst[i,j] = src[index[i], j].
// index_src=false: scatter, dst[index[i], j] = src[i, j] (negative index skips).
template <typename W, bool INDEX_SRC>
sycl::event indexed_copy(sycl::queue& q, const W* src, W* dst, const int* index,
                         std::size_t n, std::size_t row) {
  return q.parallel_for(sycl::range<2>(n, row), [=](sycl::id<2> id) {
    const std::size_t i = id[0];
    const std::size_t j = id[1];
    const int idx = index[i];
    if (idx < 0) return;  // only meaningful for scatter; harmless for gather
    if (INDEX_SRC)
      dst[i * row + j] = src[static_cast<std::size_t>(idx) * row + j];
    else
      dst[static_cast<std::size_t>(idx) * row + j] = src[i * row + j];
  });
}

template <bool INDEX_SRC>
sycl::event by_width(sycl::queue& q, const void* src, void* dst, const int* index,
                     std::size_t n, std::size_t row, DType dt) {
  if (dtype_size(dt) == 4)
    return indexed_copy<std::uint32_t, INDEX_SRC>(
        q, static_cast<const std::uint32_t*>(src), static_cast<std::uint32_t*>(dst),
        index, n, row);
  return indexed_copy<std::uint16_t, INDEX_SRC>(
      q, static_cast<const std::uint16_t*>(src), static_cast<std::uint16_t*>(dst),
      index, n, row);
}

}  // namespace

sycl::event embedding_lookup_sycl(sycl::queue& q, const void* table,
                                  const int* ids, void* out, std::size_t n,
                                  std::size_t dim, DType dt) {
  return by_width</*INDEX_SRC=*/true>(q, table, out, ids, n, dim, dt);
}

sycl::event kv_cache_scatter_sycl(sycl::queue& q, void* cache, const void* src,
                                  const int* slots, std::size_t n,
                                  std::size_t row, DType dt) {
  return by_width</*INDEX_SRC=*/false>(q, src, cache, slots, n, row, dt);
}

sycl::event kv_cache_gather_sycl(sycl::queue& q, const void* cache,
                                 const int* idx, void* out, std::size_t n,
                                 std::size_t row, DType dt) {
  return by_width</*INDEX_SRC=*/true>(q, cache, out, idx, n, row, dt);
}

namespace {

// Sentence-embedding pooling head. One subgroup owns each sequence's dim-vector;
// each lane holds DIM/kSub of the pooled accumulators in registers (compile-time
// so DIM is a shape key). Two subgroup reductions: one per token for the RMS
// scale, one at the end for the L2 norm. Reads x once, writes out once.
constexpr int kSub = 16;  // Arc subgroup width used across the backend.

template <typename T, int DIM>
sycl::event pool_dim(sycl::queue& q, const T* x, const T* weight,
                     const int* offsets, T* out, std::size_t batch, float eps) {
  static_assert(DIM % kSub == 0, "DIM must be a multiple of the subgroup width");
  constexpr int kSlots = DIM / kSub;
  return q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(batch * kSub), sycl::range<1>(kSub)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(kSub)]] {
        const sycl::sub_group sg = item.get_sub_group();
        const std::size_t seq = item.get_group(0);
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int start = offsets[seq];
        const int stop = offsets[seq + 1];

        float pooled[kSlots];
#pragma unroll
        for (int s = 0; s < kSlots; ++s) pooled[s] = 0.0f;

        // Per-token RMSNorm (learned weight) accumulated into the pooled mean.
        for (int t = start; t < stop; ++t) {
          const std::size_t base = static_cast<std::size_t>(t) * DIM;
          float ss = 0.0f;
#pragma unroll
          for (int s = 0; s < kSlots; ++s) {
            const float v = static_cast<float>(x[base + lane + s * kSub]);
            ss = sycl::fma(v, v, ss);
          }
          ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
          const float inv = sycl::rsqrt(ss / static_cast<float>(DIM) + eps);
#pragma unroll
          for (int s = 0; s < kSlots; ++s) {
            const int d = lane + s * kSub;
            pooled[s] += static_cast<float>(x[base + d]) *
                         static_cast<float>(weight[d]) * inv;
          }
        }

        // Masked mean over the sequence's tokens, then L2-normalize.
        const int count = stop - start;
        const float inv_tokens =
            count > 0 ? 1.0f / static_cast<float>(count) : 0.0f;
        float ss = 0.0f;
#pragma unroll
        for (int s = 0; s < kSlots; ++s) {
          pooled[s] *= inv_tokens;
          ss = sycl::fma(pooled[s], pooled[s], ss);
        }
        ss = sycl::reduce_over_group(sg, ss, sycl::plus<float>());
        const float inv_l2 = ss == 0.0f ? 1.0f : sycl::rsqrt(ss);
#pragma unroll
        for (int s = 0; s < kSlots; ++s) {
          const int d = lane + s * kSub;
          out[seq * DIM + d] = static_cast<T>(pooled[s] * inv_l2);
        }
      });
}

template <typename T>
sycl::event pool_by_dim(sycl::queue& q, const T* x, const T* weight,
                        const int* offsets, T* out, std::size_t batch,
                        std::size_t dim, float eps) {
  switch (dim) {
    case 256:
      return pool_dim<T, 256>(q, x, weight, offsets, out, batch, eps);
    case 512:
      return pool_dim<T, 512>(q, x, weight, offsets, out, batch, eps);
    case 768:
      return pool_dim<T, 768>(q, x, weight, offsets, out, batch, eps);
    case 1024:
      return pool_dim<T, 1024>(q, x, weight, offsets, out, batch, eps);
    default:
      throw std::invalid_argument(
          "pool_mean_rms_l2: dim must be one of {256,512,768,1024}");
  }
}

}  // namespace

sycl::event pool_mean_rms_l2_sycl(sycl::queue& q, const void* x,
                                  const void* weight, const int* offsets,
                                  void* out, std::size_t batch, std::size_t dim,
                                  float eps, DType dt) {
  switch (dt) {
    case DType::f32:
      return pool_by_dim<float>(q, static_cast<const float*>(x),
                                static_cast<const float*>(weight), offsets,
                                static_cast<float*>(out), batch, dim, eps);
    case DType::f16:
      return pool_by_dim<half_t>(q, static_cast<const half_t*>(x),
                                 static_cast<const half_t*>(weight), offsets,
                                 static_cast<half_t*>(out), batch, dim, eps);
    case DType::bf16:
      return pool_by_dim<bf16_t>(q, static_cast<const bf16_t*>(x),
                                 static_cast<const bf16_t*>(weight), offsets,
                                 static_cast<bf16_t*>(out), batch, dim, eps);
  }
  throw std::invalid_argument("pool_mean_rms_l2: unsupported dtype");
}

}  // namespace quixicore::xpu::kernels
