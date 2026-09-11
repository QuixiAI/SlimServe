// SPDX-License-Identifier: Apache-2.0
// Isolated Phase 4.1 probe. No registered op or serving dispatch.
// Parked fusion decision and measurements: marlin_glm_swiglu_template.h header;
// perf/optimization_status.md, 2026-09-11 "Phase 4.1 paired Marlin SwiGLU".
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define MARLIN_NAMESPACE_NAME marlin_glm_swiglu
#include "marlin_glm_swiglu_template.h"

template <int K, int N, int BLOCKS>
at::Tensor launch(at::Tensor a, at::Tensor out, at::Tensor weight,
               at::Tensor scales, at::Tensor global_scale, at::Tensor workspace,
               at::Tensor temp, at::Tensor sorted_ids, at::Tensor experts,
               at::Tensor padded, at::Tensor routing_weights) {
  TORCH_CHECK(a.is_cuda() && a.scalar_type() == at::kBFloat16);
  const c10::cuda::CUDAGuard guard(a.device());
  for (const auto& t : {a, out, weight, scales, global_scale, workspace, temp,
                       sorted_ids, experts, padded, routing_weights})
    TORCH_CHECK(t.device() == a.device() && t.is_contiguous());
  TORCH_CHECK(out.scalar_type() == at::kBFloat16 &&
              weight.scalar_type() == at::kInt &&
              scales.scalar_type() == at::kFloat8_e4m3fn &&
              global_scale.scalar_type() == at::kFloat &&
              routing_weights.scalar_type() == at::kFloat &&
              temp.scalar_type() == at::kFloat);
  TORCH_CHECK(workspace.scalar_type() == at::kInt && sorted_ids.scalar_type() == at::kInt &&
              experts.scalar_type() == at::kInt && padded.scalar_type() == at::kInt);
  TORCH_CHECK(a.dim() == 2 && a.size(1) == 4096 && a.size(0) >= 1 && a.size(0) <= 16);
  TORCH_CHECK(out.sizes() == at::IntArrayRef({a.size(0) * 8, 512}));
  TORCH_CHECK(weight.sizes() == at::IntArrayRef({288, 256, 2048}));
  TORCH_CHECK(scales.sizes() == at::IntArrayRef({288, 256, 1024}));
  TORCH_CHECK(global_scale.numel() == 288 && padded.numel() == 1 &&
              routing_weights.numel() == a.size(0) * 8 &&
              experts.numel() * 8 >= sorted_ids.numel());
  // Retained decode schedules from actual-weight counter traces: c1
  // K128/N64/two-CTAs-per-SM; c8/c16 K64/N128/three-CTAs-per-SM.
  // No schedule search. Only column ownership and the epilogue change.
  constexpr int THREADS = 128, SMEM = 16 * 16 + 4 * 16 * K * 2 +
                                     4 * K * N / 2 + 4 * (K / 16) * N;
  auto kernel = MARLIN_NAMESPACE_NAME::Marlin<
      vllm::kBFloat16.id(), vllm::kFE2M1f.id(), vllm::kBFloat16.id(),
      vllm::kFE4M3fn.id(), THREADS, 1, N / 16, K / 16, true, 4, 1, false, true>;
  int sms;
  C10_CUDA_CHECK(cudaDeviceGetAttribute(
      &sms, cudaDevAttrMultiProcessorCount, a.get_device()));
  TORCH_CHECK(workspace.numel() >= sms * 4 && temp.numel() >= sms * BLOCKS * 16 * N);
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
  kernel<<<sms * BLOCKS, THREADS, SMEM, at::cuda::getCurrentCUDAStream()>>>(
      (const int4*)a.data_ptr(), (const int4*)weight.data_ptr(),
      (int4*)out.data_ptr(), (int4*)temp.data_ptr(), nullptr, nullptr,
      (const int4*)scales.data_ptr(), global_scale.data_ptr<float>(), nullptr,
      nullptr, sorted_ids.data_ptr<int>(), experts.data_ptr<int>(),
      padded.data_ptr<int>(), routing_weights.data_ptr<float>(), 8, false,
      256, int(a.size(0)), 1024, 4096, workspace.data_ptr<int>(), false, false, true);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor run(at::Tensor a, at::Tensor out, at::Tensor weight,
               at::Tensor scales, at::Tensor global_scale, at::Tensor workspace,
               at::Tensor temp, at::Tensor sorted_ids, at::Tensor experts,
               at::Tensor padded, at::Tensor routing_weights) {
  if (a.size(0) == 1)
    return launch<128, 64, 2>(a, out, weight, scales, global_scale, workspace,
                             temp, sorted_ids, experts, padded, routing_weights);
  return launch<64, 128, 3>(a, out, weight, scales, global_scale, workspace,
                            temp, sorted_ids, experts, padded, routing_weights);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
