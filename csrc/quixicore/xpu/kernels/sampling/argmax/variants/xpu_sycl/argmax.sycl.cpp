// Greedy argmax over the last axis, native SYCL. One work-group per row; each
// thread scans a strided slice (16-byte vectorized) for its local (max, idx),
// then an SLM *tree* reduction (log2(threads) steps) picks the row winner
// (lowest index on ties). The tree reduction replaced a serial 256-iteration
// final loop by one thread, which dominated runtime (~5x speedup: 80->~400 GB/s).

#include "sampling/argmax/argmax_kernel.hpp"

#include <limits>

#include "common/vec_map.hpp"

namespace quixicore::xpu::kernels {
namespace {

constexpr std::size_t kRowThreads = 256;

template <typename T>
sycl::event argmax_typed(sycl::queue& q, const T* logits, int* out,
                         std::size_t rows, std::size_t vocab) {
  constexpr int V = detail::vec_width<T>();  // 8 for 2-byte, 4 for 4-byte -> 16B loads
  const sycl::nd_range<1> ndr(sycl::range<1>(rows * kRowThreads),
                              sycl::range<1>(kRowThreads));
  return q.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> svals(sycl::range<1>(kRowThreads), h);
    sycl::local_accessor<int, 1> sidx(sycl::range<1>(kRowThreads), h);
    h.parallel_for(ndr, [=](sycl::nd_item<1> it) {
      const std::size_t row = it.get_group(0);
      const std::size_t lid = it.get_local_id(0);
      const T* lr = logits + row * vocab;

      float best = -std::numeric_limits<float>::infinity();
      int best_i = 0;
      // Vectorized coalesced scan: each thread grabs V contiguous elems per step,
      // adjacent threads read adjacent V-vectors (16-byte coalesced).
      const std::size_t vend = (vocab / V) * V;
      for (std::size_t base = lid * V; base < vend; base += kRowThreads * V) {
        const sycl::vec<T, V> chunk = *reinterpret_cast<const sycl::vec<T, V>*>(lr + base);
#pragma unroll
        for (int k = 0; k < V; ++k) {
          const float val = static_cast<float>(chunk[k]);
          if (val > best) { best = val; best_i = static_cast<int>(base + k); }
        }
      }
      for (std::size_t j = vend + lid; j < vocab; j += kRowThreads) {
        const float val = static_cast<float>(lr[j]);
        if (val > best) { best = val; best_i = static_cast<int>(j); }
      }
      svals[lid] = best;
      sidx[lid] = best_i;
      it.barrier(sycl::access::fence_space::local_space);

      // SLM tree reduction: log2(kRowThreads) steps instead of a serial 256-loop.
      for (std::size_t stride = kRowThreads / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
          const float ov = svals[lid + stride];
          const int oi = sidx[lid + stride];
          if (ov > svals[lid] || (ov == svals[lid] && oi < sidx[lid])) {
            svals[lid] = ov;
            sidx[lid] = oi;
          }
        }
        it.barrier(sycl::access::fence_space::local_space);
      }
      if (lid == 0) out[row] = sidx[0];
    });
  });
}

}  // namespace

sycl::event argmax_sycl(sycl::queue& q, const void* logits, int* out,
                        std::size_t rows, std::size_t vocab, DType dt) {
  switch (dt) {
    case DType::f32:
      return argmax_typed(q, static_cast<const float*>(logits), out, rows, vocab);
    case DType::f16:
      return argmax_typed(q, static_cast<const half_t*>(logits), out, rows, vocab);
    case DType::bf16:
      return argmax_typed(q, static_cast<const bf16_t*>(logits), out, rows, vocab);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
