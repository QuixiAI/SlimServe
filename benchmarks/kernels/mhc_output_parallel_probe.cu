// SPDX-License-Identifier: Apache-2.0
// Isolated A/B wrapper. Does not replace or register serving operators.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "mhc_output_parallel.cuh"

using torch::Tensor;
using namespace tms::dsv4_mhc;

template <bool FUSED, bool NORM>
std::vector<Tensor> run_typed(
    Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor fn,
    Tensor scale, Tensor base, Tensor norm, bool candidate) {
    const int T = residual.size(0);
    auto residual_out = FUSED ? torch::empty_like(residual) : residual;
    auto partial = torch::empty({T, 64, 25}, fn.options());
    auto next_post = torch::empty({T, 4}, fn.options());
    auto next_comb = torch::empty({T, 4, 4}, fn.options());
    auto out = torch::empty({T, 4096}, residual.options());
    auto bp = [](Tensor t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
    const __nv_bfloat16 *xp = bp(x), *rp = bp(residual), *np = bp(norm);
    const float *pp = post.data_ptr<float>(), *cp = comb.data_ptr<float>(),
                *fp = fn.data_ptr<float>(), *sp = scale.data_ptr<float>(),
                *basp = base.data_ptr<float>();
    auto rop = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    float *parp = partial.data_ptr<float>(), *npp = next_post.data_ptr<float>(),
          *ncp = next_comb.data_ptr<float>();
    // Pinned GLM-5.3 settings, not the older three-iteration unit-test fixture.
    float rms_eps = 1e-5f, pre_eps = 1e-6f, sinkhorn_eps = 1e-6f,
          post_multiplier = 2.0f, norm_eps = 1e-5f;
    int repeats = 20;
    auto kernel = candidate
        ? fused_pre_transition_output_parallel<FUSED, NORM, 64>
        : fused_pre_transition<FUSED, NORM, 4096, 64, float>;
    int sms = 0, blocks = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &sms, cudaDevAttrMultiProcessorCount, residual.get_device()));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, kernel, THREADS, 0));
    TORCH_CHECK(T * 64 <= sms * blocks, "cooperative grid exceeds residency");
    void* args[] = {&xp, &rp, &pp, &cp, &fp, &rop, &parp, &sp, &basp,
                   &npp, &ncp, &op, &np, &rms_eps, &pre_eps, &sinkhorn_eps,
                   &post_multiplier, &repeats, &norm_eps};
    const auto error = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel), dim3(64, T), dim3(THREADS),
        args, 0, at::cuda::getCurrentCUDAStream());
    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
    return {residual_out, next_post, next_comb, out};
}

std::vector<Tensor> run(
    Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor fn,
    Tensor scale, Tensor base, Tensor norm, bool fused, bool with_norm,
    bool candidate) {
    TORCH_CHECK(residual.is_cuda() && residual.dim() == 3 &&
                residual.size(0) >= 1 && residual.size(0) <= 8 &&
                residual.size(1) == 4 && residual.size(2) == 4096,
                "expected residual [1..8,4,4096]");
    const c10::cuda::CUDAGuard guard(residual.device());
    for (const auto& t : {x, residual, post, comb, fn, scale, base, norm}) {
        TORCH_CHECK(t.device() == residual.device() && t.is_contiguous(),
                    "all inputs must be contiguous and on the residual device");
    }
    for (const auto& t : {x, residual, norm})
        TORCH_CHECK(t.scalar_type() == torch::kBFloat16, "expected BF16 input");
    for (const auto& t : {post, comb, fn, scale, base})
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "expected FP32 parameters");
    const int T = residual.size(0);
    TORCH_CHECK(x.sizes() == at::IntArrayRef({T, 4096}) &&
                post.sizes() == at::IntArrayRef({T, 4}) &&
                comb.sizes() == at::IntArrayRef({T, 4, 4}) &&
                fn.sizes() == at::IntArrayRef({24, 16384}) &&
                scale.numel() == 3 && base.numel() == 24 && norm.numel() == 4096,
                "unexpected probe parameter shape");
#define RUN(F, N) run_typed<F, N>(x, residual, post, comb, fn, scale, base, norm, candidate)
    if (fused) {
        if (with_norm) return RUN(true, true);
        return RUN(true, false);
    }
    if (with_norm) return RUN(false, true);
    return RUN(false, false);
#undef RUN
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
}
