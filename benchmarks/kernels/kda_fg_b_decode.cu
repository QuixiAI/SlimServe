// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "kda_fg_b_decode_sm120.cuh"

void run(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    TORCH_CHECK(x.is_cuda() && w.device() == x.device() && out.device() == x.device());
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && w.scalar_type() == x.scalar_type()
                && out.scalar_type() == x.scalar_type());
    TORCH_CHECK(x.dim() == 3 && x.size(0) == 2 && x.size(2) == 128);
    const int m = x.size(1);
    TORCH_CHECK(m >= 1 && m <= 16 && x.stride(0) == 128 && x.stride(2) == 1);
    TORCH_CHECK(x.stride(1) >= 256 && x.stride(1) % 8 == 0);
    TORCH_CHECK(w.is_contiguous() && w.sizes() == torch::IntArrayRef({2, 2048, 128}));
    TORCH_CHECK(out.is_contiguous() && out.sizes() == torch::IntArrayRef({2, m, 2048}));
    const c10::cuda::CUDAGuard guard(x.device());
    using C = tms::decode_gemm::Cfg<16, 8, 128, 2>;
    tms::kda_fg_b_decode::kda_fg_b_decode_kernel
        <<<dim3(128, 2), 256, C::SMEM_BYTES, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), m, x.stride(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
