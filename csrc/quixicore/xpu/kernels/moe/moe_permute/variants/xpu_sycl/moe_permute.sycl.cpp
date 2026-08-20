// Expert-sorted permutation of routed hidden rows and its inverse weighted
// reduce — the prologue/epilogue pair around moe_grouped_qgemm (contract
// moe_route_top_k_prefix_sum_permute / moe_unpermute_weighted_reduce).
//
// Three chained kernels, all allocation-free (cursors is caller scratch [E]):
//   (1) histogram: atomic count of valid routed pairs per expert
//   (2) exclusive prefix into cursors (one work-group scan, E <= 256)
//   (3) scatter: each (token, k) pair claims its destination row with an
//       atomic cursor bump, gathers the hidden row into `permuted`, and
//       records row_map[t*k + j] (-1 for invalid expert ids, which also
//       leaves rows_per_expert untouched — the EP-safe skip)
// Rows within one expert land in a nondeterministic order; the GEMM treats
// them symmetrically and the unpermute reads through row_map, so results
// are order-independent. The reduce: out[t,:] = sum_j w[t,j] *
// permuted[row_map[t*k+j], :] in fp32, skipping -1.
//
// Semantics from vllm-xpu-kernels remap_hidden_states + moe_gather (Apache;
// translated).

#include "moe/moe_permute/moe_permute_kernel.hpp"

namespace quixicore::xpu::kernels {
namespace {

using AtomicI32 =
    sycl::atomic_ref<std::int32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

template <typename T>
class MoePermuteScatterKernel;
template <typename T>
class MoePermuteCountKernel;
template <typename T>
class MoePermuteScanKernel;
template <typename T>
class MoeUnpermuteKernel;

template <typename T>
sycl::event permute_typed(sycl::queue& q, const T* hidden,
                          const int* topk_ids, T* permuted,
                          std::int32_t* rows_per_expert, std::int32_t* row_map,
                          std::int32_t* cursors, std::size_t n_tokens,
                          std::size_t top_k, std::size_t hidden_dim,
                          std::size_t n_experts) {
  const std::size_t pairs = n_tokens * top_k;
  sycl::event e1 = q.submit([&](sycl::handler& h) {
    h.parallel_for<MoePermuteCountKernel<T>>(
        sycl::range<1>(pairs > n_experts ? pairs : n_experts),
        [=](sycl::id<1> idx) {
          if (idx[0] < n_experts) rows_per_expert[idx[0]] = 0;
        });
  });
  sycl::event e2 = q.submit([&](sycl::handler& h) {
    h.depends_on(e1);
    h.parallel_for(sycl::range<1>(pairs), [=](sycl::id<1> idx) {
      const int e = topk_ids[idx[0]];
      if (e >= 0 && static_cast<std::size_t>(e) < n_experts) {
        AtomicI32 c(rows_per_expert[e]);
        c.fetch_add(1);
      }
    });
  });
  sycl::event e3 = q.submit([&](sycl::handler& h) {
    h.depends_on(e2);
    h.parallel_for<MoePermuteScanKernel<T>>(
        sycl::nd_range<1>(sycl::range<1>(1), sycl::range<1>(1)),
        [=](sycl::nd_item<1>) {
          std::int32_t acc = 0;
          for (std::size_t e = 0; e < n_experts; ++e) {
            cursors[e] = acc;
            acc += rows_per_expert[e];
          }
        });
  });
  return q.submit([&](sycl::handler& h) {
    h.depends_on(e3);
    h.parallel_for<MoePermuteScatterKernel<T>>(
        sycl::range<1>(pairs), [=](sycl::id<1> idx) {
          const int e = topk_ids[idx[0]];
          if (e < 0 || static_cast<std::size_t>(e) >= n_experts) {
            row_map[idx[0]] = -1;
            return;
          }
          AtomicI32 c(cursors[e]);
          const std::int32_t dst = c.fetch_add(1);
          row_map[idx[0]] = dst;
          const std::size_t t = idx[0] / top_k;
          const T* src = hidden + t * hidden_dim;
          T* d = permuted + static_cast<std::size_t>(dst) * hidden_dim;
          for (std::size_t i = 0; i < hidden_dim; ++i) d[i] = src[i];
        });
  });
}

template <typename T>
sycl::event unpermute_typed(sycl::queue& q, const T* permuted,
                            const std::int32_t* row_map,
                            const float* topk_weights, T* out,
                            std::size_t n_tokens, std::size_t top_k,
                            std::size_t hidden_dim) {
  return q.parallel_for<MoeUnpermuteKernel<T>>(
      sycl::range<1>(n_tokens * hidden_dim), [=](sycl::id<1> idx) {
        const std::size_t t = idx[0] / hidden_dim;
        const std::size_t i = idx[0] % hidden_dim;
        float acc = 0.0f;
        for (std::size_t j = 0; j < top_k; ++j) {
          const std::int32_t r = row_map[t * top_k + j];
          if (r < 0) continue;
          acc += topk_weights[t * top_k + j] *
                 static_cast<float>(
                     permuted[static_cast<std::size_t>(r) * hidden_dim + i]);
        }
        out[idx[0]] = static_cast<T>(acc);
      });
}

}  // namespace

sycl::event moe_permute_sycl(sycl::queue& q, const void* hidden,
                             const int* topk_ids, void* permuted,
                             std::int32_t* rows_per_expert,
                             std::int32_t* row_map, std::int32_t* cursors,
                             std::size_t n_tokens, std::size_t top_k,
                             std::size_t hidden_dim, std::size_t n_experts,
                             DType dt) {
  switch (dt) {
    case DType::f32:
      return permute_typed(q, static_cast<const float*>(hidden), topk_ids,
                           static_cast<float*>(permuted), rows_per_expert,
                           row_map, cursors, n_tokens, top_k, hidden_dim,
                           n_experts);
    case DType::f16:
      return permute_typed(q, static_cast<const half_t*>(hidden), topk_ids,
                           static_cast<half_t*>(permuted), rows_per_expert,
                           row_map, cursors, n_tokens, top_k, hidden_dim,
                           n_experts);
    case DType::bf16:
      return permute_typed(q, static_cast<const bf16_t*>(hidden), topk_ids,
                           static_cast<bf16_t*>(permuted), rows_per_expert,
                           row_map, cursors, n_tokens, top_k, hidden_dim,
                           n_experts);
  }
  return {};
}

sycl::event moe_unpermute_weighted_reduce_sycl(
    sycl::queue& q, const void* permuted, const std::int32_t* row_map,
    const float* topk_weights, void* out, std::size_t n_tokens,
    std::size_t top_k, std::size_t hidden_dim, DType dt) {
  switch (dt) {
    case DType::f32:
      return unpermute_typed(q, static_cast<const float*>(permuted), row_map,
                             topk_weights, static_cast<float*>(out), n_tokens,
                             top_k, hidden_dim);
    case DType::f16:
      return unpermute_typed(q, static_cast<const half_t*>(permuted), row_map,
                             topk_weights, static_cast<half_t*>(out), n_tokens,
                             top_k, hidden_dim);
    case DType::bf16:
      return unpermute_typed(q, static_cast<const bf16_t*>(permuted), row_map,
                             topk_weights, static_cast<bf16_t*>(out), n_tokens,
                             top_k, hidden_dim);
  }
  return {};
}

}  // namespace quixicore::xpu::kernels
