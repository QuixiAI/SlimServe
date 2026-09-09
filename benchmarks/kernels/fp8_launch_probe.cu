// SPDX-License-Identifier: Apache-2.0
// Isolated launch-geometry sweep of unchanged serving math. No registered op.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "fp8_decode_gemm.cuh"

namespace fp8 = tms::decode_gemm_fp8;
using torch::Tensor;

// NT, WARPS, STAGES; KCHUNK remains the checkpoint's 128-column scale block.
#define CONFIGS(X) \
    X(8,4,4) X(8,4,8) X(8,8,4) X(8,8,8) \
    X(16,4,4) X(16,4,8) X(16,8,4) X(16,8,8) \
    X(32,4,4) X(32,4,8) X(32,8,4) X(32,8,8)

Tensor run(Tensor x, Tensor weight, Tensor scale, int nt, int warps, int stages) {
    TORCH_CHECK(x.is_cuda(), "CUDA activations required");
    const c10::cuda::CUDAGuard guard(x.device());
    for (const auto& t : {x, weight, scale})
        TORCH_CHECK(t.device() == x.device() && t.is_contiguous(),
                    "contiguous inputs on the same CUDA device required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                weight.scalar_type() == torch::kFloat8_e4m3fn &&
                scale.scalar_type() == torch::kFloat32,
                "BF16 activations, E4M3 weights, FP32 scales required");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 && scale.dim() == 2 &&
                x.size(1) == weight.size(1), "unexpected ranks or K dimension");
    const int m = x.size(0), n = weight.size(0), k = x.size(1);
    TORCH_CHECK(m >= 1 && m <= 16 &&
                ((n == 1024 && k == 4096) || (n == 4096 && k == 512)),
                "only TP4 shared-expert serving shapes are in this probe");
    TORCH_CHECK(scale.sizes() == at::IntArrayRef({n / 128, k / 128}),
                "unexpected block-scale shape");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
                "16-byte-aligned activation and weight allocations required");
    auto out = torch::empty({m, n}, x.options());
    const auto* xp = reinterpret_cast<const __nv_bfloat16*>(x.data_ptr());
    const auto* wp = reinterpret_cast<const uint8_t*>(weight.data_ptr());
    const auto* sp = scale.data_ptr<float>();
    auto* op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (nt == 0 && warps == 0 && stages == 0) {
        fp8::launch_auto(xp, wp, sp, nullptr, op, m, n, k, stream);
    } else {
        bool matched = false;
#define LAUNCH(N, W, S) \
        if (nt == N && warps == W && stages == S) { \
            fp8::launch<N, W, 128, S>(xp, wp, sp, nullptr, op, m, n, k, stream); \
            matched = true; \
        }
        CONFIGS(LAUNCH)
#undef LAUNCH
        TORCH_CHECK(matched, "unsupported launch configuration");
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

pybind11::list resources() {
    pybind11::list rows;
#define RESOURCE(N, W, S) { \
    cudaFuncAttributes attr{}; \
    C10_CUDA_CHECK(cudaFuncGetAttributes( \
        &attr, fp8::fp8_decode_gemm_kernel<N, W, 128, S, __nv_bfloat16>)); \
    pybind11::dict row; \
    row["config"] = pybind11::make_tuple(N, W, S); \
    row["registers"] = attr.numRegs; \
    row["local_bytes"] = attr.localSizeBytes; \
    row["dynamic_shared_bytes"] = fp8::Cfg<N, W, 128, S>::SMEM_BYTES; \
    row["binary_version"] = attr.binaryVersion; \
    rows.append(row); \
}
    CONFIGS(RESOURCE)
#undef RESOURCE
    return rows;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
    m.def("resources", &resources);
}
