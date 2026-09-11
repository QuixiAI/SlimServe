// SPDX-License-Identifier: Apache-2.0
// Isolated probe: neither registers nor replaces the serving router.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "glm_moe_routing.cuh"

using torch::Tensor;

std::vector<Tensor> run(Tensor logits, Tensor bias, int64_t scoring,
                       bool renormalize, double scaling, int64_t block_size,
                       bool stable) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat32 &&
                logits.is_contiguous() && logits.dim() == 2 &&
                logits.size(0) >= 1 && logits.size(0) <= 16 &&
                logits.size(1) == 288, "expected contiguous CUDA FP32 [1..16,288]");
    TORCH_CHECK(bias.device() == logits.device() && bias.is_contiguous() &&
                bias.scalar_type() == torch::kFloat32 && bias.dim() == 1 &&
                bias.size(0) == 288, "expected same-device contiguous FP32 bias[288]");
    TORCH_CHECK(scoring == 0 || scoring == 1, "expected sigmoid or sqrt-softplus");
    TORCH_CHECK(block_size == 8 || block_size == 16 || block_size == 32 ||
                block_size == 48 || block_size == 64, "unsupported block size");
    const c10::cuda::CUDAGuard guard(logits.device());
    const int M = logits.size(0), numel = M * 8;
    const int capacity = std::min(numel * int(block_size),
                                  numel + 288 * (int(block_size) - 1));
    const int blocks = (capacity + block_size - 1) / block_size;
    auto i32 = logits.options().dtype(torch::kInt32);
    auto weights = torch::empty({M, 8}, logits.options());
    auto ids = torch::empty({M, 8}, i32);
    auto sorted = torch::empty({capacity}, i32);
    auto experts = torch::empty({blocks}, i32);
    auto padded = torch::empty({1}, i32);
    auto kernel = stable ? tms::glm_route::route_align_kernel<288, 8, true>
                         : tms::glm_route::route_align_kernel<288, 8, false>;
    kernel<<<1, tms::glm_route::THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
        logits.data_ptr<float>(), bias.data_ptr<float>(), weights.data_ptr<float>(),
        ids.data_ptr<int32_t>(), sorted.data_ptr<int32_t>(), experts.data_ptr<int32_t>(),
        padded.data_ptr<int32_t>(), M, scoring, scaling, renormalize, block_size,
        capacity, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {weights, ids, sorted, experts, padded};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
