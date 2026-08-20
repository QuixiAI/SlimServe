// Dispatch layer for the moe family (native only).

#include "quixicore/xpu/ops.hpp"

#include "moe/moe_route/moe_route_kernel.hpp"
#include "moe/grouped_qgemm/grouped_qgemm_kernel.hpp"
#include "moe/moe_permute/moe_permute_kernel.hpp"
#include "moe/nvfp4_moe/nvfp4_moe_kernel.hpp"

namespace quixicore::xpu::ops {

void moe_route_topk(sycl::queue& q, const void* router_logits, int* expert_ids,
                    float* expert_weights, std::size_t n_tokens,
                    std::size_t n_experts, int k, DType dt, MoeGating gating,
                    bool renormalize, float routed_scaling, Variant variant,
                    bool blocking) {
  (void)variant;
  sycl::event ev = kernels::moe_route_topk_sycl(
      q, router_logits, expert_ids, expert_weights, n_tokens, n_experts, k,
      static_cast<int>(gating), renormalize ? 1 : 0, routed_scaling, dt);
  if (blocking) ev.wait();
}

void nvfp4_moe_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *out_f32, std::size_t M, std::size_t E,
                     std::size_t top_k, std::size_t K, std::size_t I, DType act_dt,
                     bool multiply_router_weight, Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_fused_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2, w2_scales,
      w2_global_scales, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt, zeroed);
  if (blocking)
    event.wait();
}

void nvfp4_moe_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *scratch_f32, float *out_f32,
                     std::size_t M, std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                     DType act_dt, bool multiply_router_weight, Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_split_sycl(
      q, hidden, topk_ids, topk_weights, w13, w13_scales, w13_global_scales, w2, w2_scales,
      w2_global_scales, scratch_f32, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt,
      zeroed);
  if (blocking)
    event.wait();
}

void nvfp4_moe_relu2_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w1, const void *w1_scales,
                           const float *w1_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *out_f32, std::size_t M,
                           std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                           DType act_dt, bool multiply_router_weight, Variant variant,
                           bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_relu2_fused_sycl(
      q, hidden, topk_ids, topk_weights, w1, w1_scales, w1_global_scales, w2, w2_scales,
      w2_global_scales, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt, zeroed);
  if (blocking)
    event.wait();
}

void nvfp4_moe_relu2_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w1, const void *w1_scales,
                           const float *w1_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *scratch_f32, float *out_f32,
                           std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
                           std::size_t I, DType act_dt, bool multiply_router_weight,
                           Variant variant, bool blocking) {
  (void)variant;
  const sycl::event zeroed = q.memset(out_f32, 0, M * K * sizeof(float));
  sycl::event event = kernels::nvfp4_moe_relu2_split_sycl(
      q, hidden, topk_ids, topk_weights, w1, w1_scales, w1_global_scales, w2, w2_scales,
      w2_global_scales, scratch_f32, out_f32, M, E, top_k, K, I, multiply_router_weight, act_dt,
      zeroed);
  if (blocking)
    event.wait();
}

void moe_grouped_qgemm(sycl::queue& q, const void* A, const void* W,
                       const void* scales, const float* global_scales, void* C,
                       const std::int32_t* rows_per_expert, std::size_t M_total,
                       std::size_t N, std::size_t K, std::size_t E,
                       std::size_t group, MoeWeightFormat fmt, DType act_dt,
                       Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::moe_grouped_qgemm_sycl(
      q, A, W, scales, global_scales, C, rows_per_expert, M_total, N, K, E,
      group, static_cast<int>(fmt), act_dt);
  if (blocking) ev.wait();
}

void moe_grouped_qswiglu(sycl::queue& q, const void* A, const void* W,
                         const void* scales, const float* global_scales,
                         void* scratch_2i, void* C,
                         const std::int32_t* rows_per_expert,
                         std::size_t M_total, std::size_t I, std::size_t K,
                         std::size_t E, std::size_t group, MoeWeightFormat fmt,
                         DType act_dt, Variant variant, bool blocking) {
  (void)variant;  // native only (composite: grouped qgemm -> glu swiglu)
  sycl::event g = kernels::moe_grouped_qgemm_sycl(
      q, A, W, scales, global_scales, scratch_2i, rows_per_expert, M_total,
      2 * I, K, E, group, static_cast<int>(fmt), act_dt);
  g.wait();  // glu launches on the same in-order-agnostic queue; serialize
  glu(q, scratch_2i, C, M_total, I, act_dt, GluMode::swiglu, Variant::sycl,
      blocking);
}

void moe_permute(sycl::queue& q, const void* hidden, const int* topk_ids,
                 void* permuted, std::int32_t* rows_per_expert,
                 std::int32_t* row_map, std::int32_t* cursors,
                 std::size_t n_tokens, std::size_t top_k,
                 std::size_t hidden_dim, std::size_t n_experts, DType dt,
                 Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::moe_permute_sycl(q, hidden, topk_ids, permuted,
                                             rows_per_expert, row_map, cursors,
                                             n_tokens, top_k, hidden_dim,
                                             n_experts, dt);
  if (blocking) ev.wait();
}

void moe_unpermute_weighted_reduce(sycl::queue& q, const void* permuted,
                                   const std::int32_t* row_map,
                                   const float* topk_weights, void* out,
                                   std::size_t n_tokens, std::size_t top_k,
                                   std::size_t hidden_dim, DType dt,
                                   Variant variant, bool blocking) {
  (void)variant;  // native only
  sycl::event ev = kernels::moe_unpermute_weighted_reduce_sycl(
      q, permuted, row_map, topk_weights, out, n_tokens, top_k, hidden_dim,
      dt);
  if (blocking) ev.wait();
}

}  // namespace quixicore::xpu::ops
