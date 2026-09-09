// SPDX-License-Identifier: Apache-2.0
// Isolated probe; does not register a torch op or replace a serving kernel.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "glm_moe_stable_align.cuh"

using torch::Tensor;

void run_into(Tensor ids, Tensor sorted, Tensor experts, Tensor padded,
              Tensor offsets, int64_t block, bool parallel_count, bool aggregate) {
    TORCH_CHECK(aggregate || parallel_count, "direct count requires 1024 threads");
    TORCH_CHECK(ids.is_cuda() && ids.scalar_type() == torch::kInt32 &&
                ids.is_contiguous() && ids.dim() == 2 && ids.size(0) >= 17 &&
                ids.size(0) <= 8192 && ids.size(1) == 8,
                "expected contiguous CUDA int32 [17..8192,8] IDs");
    TORCH_CHECK(block == 8 || block == 16 || block == 32 || block == 48 || block == 64,
                "unsupported block size");
    const int numel = ids.numel();
    const int capacity = std::min(numel * int(block), numel + 288 * (int(block) - 1));
    const int blocks = (capacity + block - 1) / block;
    const std::vector<std::pair<Tensor, int64_t>> outputs = {
        {sorted, capacity}, {experts, blocks}, {padded, 1}, {offsets, 289}};
    for (const auto& [tensor, size] : outputs) {
        TORCH_CHECK(tensor.device() == ids.device() && tensor.scalar_type() == torch::kInt32 &&
                    tensor.is_contiguous() && tensor.dim() == 1 && tensor.size(0) == size,
                    "incorrect same-device contiguous int32 output geometry");
    }
    const c10::cuda::CUDAGuard guard(ids.device());
    const auto* prop = at::cuda::getCurrentDeviceProperties();
    TORCH_CHECK(prop->major == 12 && prop->minor == 0, "SM120 probe only");
    const auto stream = at::cuda::getCurrentCUDAStream();
    auto counter = !aggregate ? tms::glm_stable_align::count_prefix<1024, false>
        : parallel_count ? tms::glm_stable_align::count_prefix<1024, true>
                         : tms::glm_stable_align::count_prefix<256, true>;
    counter<<<2, parallel_count ? 1024 : 256, 0, stream>>>(
        ids.data_ptr<int>(), sorted.data_ptr<int>(), experts.data_ptr<int>(),
        padded.data_ptr<int>(), offsets.data_ptr<int>(), numel, block, capacity, blocks);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    tms::glm_stable_align::scatter_bitmap<<<288, 256, 0, stream>>>(
        ids.data_ptr<int>(), sorted.data_ptr<int>(), offsets.data_ptr<int>(), numel);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<Tensor> run(Tensor ids, int64_t block, bool parallel_count, bool aggregate) {
    TORCH_CHECK(ids.dim() == 2 && ids.size(0) >= 17 && ids.size(0) <= 8192 &&
                ids.size(1) == 8, "expected [17..8192,8] IDs");
    TORCH_CHECK(block == 8 || block == 16 || block == 32 || block == 48 || block == 64,
                "unsupported block size");
    const int numel = ids.numel();
    const int capacity = std::min(numel * int(block), numel + 288 * (int(block) - 1));
    auto options = ids.options().dtype(torch::kInt32);
    auto sorted = torch::empty({capacity}, options);
    auto experts = torch::empty({(capacity + block - 1) / block}, options);
    auto padded = torch::empty({1}, options);
    auto offsets = torch::empty({289}, options);
    run_into(ids, sorted, experts, padded, offsets, block, parallel_count, aggregate);
    return {sorted, experts, padded};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run, pybind11::arg("ids"), pybind11::arg("block"),
          pybind11::arg("parallel_count") = false, pybind11::arg("aggregate") = true);
    m.def("run_into", &run_into, pybind11::arg("ids"), pybind11::arg("sorted"),
          pybind11::arg("experts"), pybind11::arg("padded"), pybind11::arg("offsets"),
          pybind11::arg("block"), pybind11::arg("parallel_count") = false,
          pybind11::arg("aggregate") = true);
}
