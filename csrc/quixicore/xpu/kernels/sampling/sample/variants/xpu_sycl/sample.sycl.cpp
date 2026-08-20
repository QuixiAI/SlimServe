// Categorical + top-k sampling, native SYCL. One work-item per row; the
// stateless RNG (rng.hpp) gives one uniform per row. Categorical does a
// max/normalizer/inverse-CDF scan over the temperature-scaled logits; top-k
// selects the k largest logits first, then samples within that set. No vendor
// primitive exists for sampling — native only.

#include "sampling/sample/sample_kernel.hpp"

#include "common/rng.hpp"

#include <limits>

namespace quixicore::xpu::kernels {
namespace {

constexpr int kMaxK = 64;

// One work-group per row (was one work-item per row: catastrophic at real
// shapes -- a handful of rows x 128k vocab under-fills the GPU ~40x). Parallel
// max + normalizer reductions over strided coalesced reads; the inverse-CDF
// selection uses per-thread CONTIGUOUS chunks (order-preserving) + an
// exclusive_scan_over_group so the owning thread does only a short serial scan.
constexpr std::size_t kRowThreads = 256;

template <typename T>
sycl::event categorical_typed(sycl::queue& q, const T* logits, int* out,
                              std::size_t rows, std::size_t vocab, float temp,
                              std::uint32_t seed) {
  const float inv_temp = 1.0f / temp;
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<int, 1> sres(sycl::range<1>(1), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      const auto g = it.get_group();
      const std::size_t row = it.get_group(0);
      const std::size_t lid = it.get_local_id(0);
      const T* rp = logits + row * vocab;

      float lm = -std::numeric_limits<float>::infinity();
      for (std::size_t j = lid; j < vocab; j += kRowThreads)
        lm = sycl::max(lm, static_cast<float>(rp[j]) * inv_temp);
      const float m = sycl::reduce_over_group(g, lm, sycl::maximum<float>());

      float lz = 0.0f;
      for (std::size_t j = lid; j < vocab; j += kRowThreads)
        lz += sycl::exp(static_cast<float>(rp[j]) * inv_temp - m);
      const float Z = sycl::reduce_over_group(g, lz, sycl::plus<float>());

      // contiguous chunk per thread preserves cumulative order for the CDF.
      const std::size_t chunk = (vocab + kRowThreads - 1) / kRowThreads;
      const std::size_t lo = lid * chunk;
      const std::size_t hi = sycl::min(lo + chunk, vocab);
      float mypart = 0.0f;
      for (std::size_t j = lo; j < hi; ++j)
        mypart += sycl::exp(static_cast<float>(rp[j]) * inv_temp - m);
      const float myprefix = sycl::exclusive_scan_over_group(g, mypart, sycl::plus<float>());

      const float target = detail::uniform01(seed, row) * Z;
      if (lid == 0) sres[0] = static_cast<int>(vocab) - 1;
      it.barrier(sycl::access::fence_space::local_space);
      if (mypart > 0.0f && myprefix <= target && target < myprefix + mypart) {
        float c = myprefix;
        int r = static_cast<int>(hi) - 1;
        for (std::size_t j = lo; j < hi; ++j) {
          c += sycl::exp(static_cast<float>(rp[j]) * inv_temp - m);
          if (c >= target) { r = static_cast<int>(j); break; }
        }
        sres[0] = r;
      }
      it.barrier(sycl::access::fence_space::local_space);
      if (lid == 0) out[row] = sres[0];
    });
  });
}

template <typename T>
sycl::event topk_typed(sycl::queue& q, const T* logits, int* out,
                       std::size_t rows, std::size_t vocab, int k, float temp,
                       std::uint32_t seed) {
  const float inv_temp = 1.0f / temp;
  return q.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> idx) {
    const std::size_t t = idx[0];
    const T* row = logits + t * vocab;
    int sel[kMaxK];
    float val[kMaxK];
    for (int i = 0; i < k; ++i) {
      float best = -std::numeric_limits<float>::infinity();
      int be = 0;
      for (std::size_t e = 0; e < vocab; ++e) {
        bool taken = false;
        for (int j = 0; j < i; ++j) taken |= (sel[j] == static_cast<int>(e));
        if (taken) continue;
        const float v = static_cast<float>(row[e]);
        if (v > best) { best = v; be = static_cast<int>(e); }
      }
      sel[i] = be; val[i] = best;
    }
    float m = val[0];
    for (int i = 1; i < k; ++i) m = sycl::max(m, val[i]);
    float Z = 0.0f;
    for (int i = 0; i < k; ++i) Z += sycl::exp(val[i] * inv_temp - m);
    const float target = detail::uniform01(seed, t) * Z;
    float c = 0.0f;
    int chosen = sel[k - 1];
    for (int i = 0; i < k; ++i) {
      c += sycl::exp(val[i] * inv_temp - m);
      if (c >= target) { chosen = sel[i]; break; }
    }
    out[t] = chosen;
  });
}

template <typename F>
sycl::event by_dtype(DType dt, const void* logits, F&& f) {
  switch (dt) {
    case DType::f32: return f(static_cast<const float*>(logits));
    case DType::f16: return f(static_cast<const half_t*>(logits));
    case DType::bf16: return f(static_cast<const bf16_t*>(logits));
  }
  return sycl::event{};
}

}  // namespace

sycl::event sample_categorical_sycl(sycl::queue& q, const void* logits, int* out,
                                    std::size_t rows, std::size_t vocab,
                                    float temperature, std::uint32_t seed, DType dt) {
  return by_dtype(dt, logits, [&](auto* p) {
    return categorical_typed(q, p, out, rows, vocab, temperature, seed);
  });
}

sycl::event top_k_sample_sycl(sycl::queue& q, const void* logits, int* out,
                              std::size_t rows, std::size_t vocab, int k,
                              float temperature, std::uint32_t seed, DType dt) {
  return by_dtype(dt, logits, [&](auto* p) {
    return topk_typed(q, p, out, rows, vocab, k, temperature, seed);
  });
}

}  // namespace quixicore::xpu::kernels
