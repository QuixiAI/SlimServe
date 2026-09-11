// SPDX-License-Identifier: Apache-2.0
// Isolated SM120 NVFP4 prefill pipeline-depth experiment. No registered ops.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define MARLIN_NAMESPACE_NAME marlin_prefill_stages
#include "libtorch_stable/moe/marlin_moe_wna16/marlin_template.h"

template <int N, int STAGES>
at::Tensor launch(at::Tensor a, at::Tensor out, at::Tensor weight,
                 at::Tensor scales, at::Tensor global_scale, at::Tensor workspace,
                 at::Tensor sorted_ids, at::Tensor experts, at::Tensor padded,
                 at::Tensor routing_weights, int top_k, bool weighted, int blocks) {
  constexpr int M = 64, K = 64, THREADS = std::min(N, 256);
  constexpr int a_bytes = STAGES * M * K * 2;
  constexpr int b_bytes = STAGES * K * N / 2;
  constexpr int red_bytes = M * (N + 8) * 2;
  constexpr int bias_bytes = N * 2;
  constexpr int min_br = std::min(b_bytes, red_bytes);
  constexpr int max_br = std::max(b_bytes, red_bytes);
  constexpr int smem = std::max(max_br, min_br + bias_bytes) + a_bytes +
                       STAGES * (K / 16) * N + M * 16;
  auto kernel = MARLIN_NAMESPACE_NAME::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), THREADS, M / 16, N / 16, K / 16,
      false, STAGES, 1, false>;
  int sms, max_shared;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, a.get_device()));
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &max_shared, cudaDevAttrMaxSharedMemoryPerMultiprocessor, a.get_device()));
  TORCH_CHECK(smem + 1024 <= max_shared / blocks, "Invalid thread config: smem");
  TORCH_CHECK(workspace.numel() >= sms * 4);
  int m = a.size(0), k = a.size(1), n = out.size(1);
  auto temp = at::empty({std::min(int64_t(n) * sorted_ids.numel(),
                                 int64_t(sms) * 4 * M * 256)},
                        a.options().dtype(at::kFloat));
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kernel<<<sms * blocks, THREADS, smem, at::cuda::getCurrentCUDAStream()>>>(
      (const int4*)a.data_ptr(), (const int4*)weight.data_ptr(),
      (int4*)out.data_ptr(), (int4*)temp.data_ptr(), nullptr, nullptr,
      (const int4*)scales.data_ptr(), global_scale.data_ptr<float>(), nullptr,
      nullptr, sorted_ids.data_ptr<int>(), experts.data_ptr<int>(),
      padded.data_ptr<int>(), routing_weights.data_ptr<float>(), top_k, weighted,
      k / 16, m, n, k, workspace.data_ptr<int>(), false, false, true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor run(at::Tensor a, at::Tensor out, at::Tensor weight,
               at::Tensor scales, at::Tensor global_scale, at::Tensor workspace,
               at::Tensor sorted_ids, at::Tensor experts, at::Tensor padded,
               at::Tensor routing_weights, int top_k, bool weighted,
               int n_tile, int blocks, int stages) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16);
  const c10::cuda::CUDAGuard guard(a.device());
  for (const auto& t : {a, out, weight, scales, global_scale, workspace,
                       sorted_ids, experts, padded, routing_weights})
    TORCH_CHECK(t.device() == a.device() && t.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kInt &&
              scales.scalar_type() == at::kFloat8_e4m3fn &&
              global_scale.scalar_type() == at::kFloat &&
              routing_weights.scalar_type() == at::kFloat);
  TORCH_CHECK(workspace.scalar_type() == at::kInt && sorted_ids.scalar_type() == at::kInt &&
              experts.scalar_type() == at::kInt && padded.scalar_type() == at::kInt);
  TORCH_CHECK(a.dim() == 2 && out.dim() == 2 && out.size(0) == a.size(0) * top_k);
  TORCH_CHECK((top_k == 8 && !weighted && a.size(1) == 4096 && out.size(1) == 1024) ||
              (top_k == 1 && weighted && a.size(1) == 512 && out.size(1) == 4096));
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({288, a.size(1) / 16, out.size(1) * 2}));
  TORCH_CHECK(scales.sizes() == at::IntArrayRef({288, a.size(1) / 16, out.size(1)}));
  TORCH_CHECK(global_scale.numel() == 288 && padded.numel() == 1 &&
              routing_weights.numel() == out.size(0) &&
              experts.numel() * 64 >= sorted_ids.numel());
  TORCH_CHECK(blocks >= 1 && blocks <= 2);
#define CALL(N, S) return launch<N, S>(a, out, weight, scales, global_scale, workspace, \
    sorted_ids, experts, padded, routing_weights, top_k, weighted, blocks)
  if (n_tile == 128) {
    if (stages == 2) { CALL(128, 2); }
    if (stages == 3) { CALL(128, 3); }
    if (stages == 4) { CALL(128, 4); }
  } else if (n_tile == 256) {
    if (stages == 2) { CALL(256, 2); }
    if (stages == 3) { CALL(256, 3); }
    if (stages == 4) { CALL(256, 4); }
  } else if (n_tile == 512) {
    if (stages == 2) { CALL(512, 2); }
    if (stages == 3) { CALL(512, 3); }
  }
#undef CALL
  TORCH_CHECK(false, "unsupported pipeline configuration");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
