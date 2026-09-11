// SPDX-License-Identifier: Apache-2.0
// Isolated Phase 4.1 binding. No installed serving library is modified.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "nvfp4_decode_sm120.cuh"

namespace {
using namespace tms::nvfp4_decode_sm120;

template <bool PIPELINE, int K>
void launch(torch::Tensor x, torch::Tensor w, torch::Tensor sc, torch::Tensor gs,
            torch::Tensor eid, torch::Tensor rows, torch::Tensor counts,
            torch::Tensor offsets, torch::Tensor out, int sms) {
    using C = Cfg<16, 8, 512, 4>;
    constexpr int shared = PIPELINE ? 4 * C::STAGE_BYTES + C::RED_BYTES : C::SMEM_BYTES;
    static_assert(shared <= 48 * 1024);
    const int slots = eid.numel(), n = w.size(1);
    const int blocks = std::min(sms, slots * (n / 16));
    nvfp4_moe_gemm_kernel<16, 8, 512, 4, PIPELINE, K>
        <<<blocks, 256, shared, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
            w.data_ptr<uint8_t>(), reinterpret_cast<const uint8_t*>(sc.data_ptr()),
            gs.data_ptr<float>(), eid.data_ptr<int>(), rows.data_ptr<int>(),
            counts.data_ptr<int>(), offsets.data_ptr<int>(),
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), slots, n, K, 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void run(torch::Tensor x, torch::Tensor w, torch::Tensor sc, torch::Tensor gs,
         torch::Tensor eid, torch::Tensor rows, torch::Tensor counts,
         torch::Tensor offsets, torch::Tensor out, bool pipeline) {
    for (const auto& t : {x, w, sc, gs, eid, rows, counts, offsets, out}) {
        TORCH_CHECK(t.is_cuda() && t.device() == x.device() && t.is_contiguous(),
                    "contiguous tensors on one CUDA device required");
    }
    TORCH_CHECK(x.dim() == 2 && x.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(w.dim() == 3 && w.scalar_type() == torch::kUInt8);
    TORCH_CHECK(sc.dim() == 3 && sc.scalar_type() == torch::kFloat8_e4m3fn);
    const int k = x.size(1), n = w.size(1), slots = eid.numel();
    TORCH_CHECK((k == 4096 && n == 1024) || (k == 512 && n == 4096));
    TORCH_CHECK(slots == 8 && w.size(0) == 8 && w.size(2) == k / 2);
    TORCH_CHECK(sc.size(0) == 8 && sc.size(1) == n && sc.size(2) == k / 16);
    TORCH_CHECK(gs.scalar_type() == torch::kFloat32 && gs.numel() == 8);
    for (const auto& t : {eid, rows, counts, offsets})
        TORCH_CHECK(t.scalar_type() == torch::kInt32);
    TORCH_CHECK(eid.dim() == 1 && rows.sizes() == torch::IntArrayRef({8, 16}));
    TORCH_CHECK(counts.numel() == 8 && offsets.numel() == 8);
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16 && out.sizes() == torch::IntArrayRef({8, n}));
    // The benchmark constructs and checks one valid row per slot, row indices
    // 0 for gate/up or 0..7 for down, with offsets/eids 0..7. No device->host
    // inspection is performed in this graph-captured diagnostic binding.
    TORCH_CHECK(x.size(0) == (k == 4096 ? 1 : 8));
    const c10::cuda::CUDAGuard guard(x.device());
    int sms = 0, major = 0, minor = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, x.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, x.get_device()));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, x.get_device()));
    TORCH_CHECK(major == 12 && minor == 0, "SM120 only");
    if (k == 4096) {
        if (pipeline) launch<true, 4096>(x, w, sc, gs, eid, rows, counts, offsets, out, sms);
        else launch<false, 4096>(x, w, sc, gs, eid, rows, counts, offsets, out, sms);
    } else {
        if (pipeline) launch<true, 512>(x, w, sc, gs, eid, rows, counts, offsets, out, sms);
        else launch<false, 512>(x, w, sc, gs, eid, rows, counts, offsets, out, sms);
    }
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("run", &run); }
