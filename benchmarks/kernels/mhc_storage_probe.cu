// SPDX-License-Identifier: Apache-2.0
// Same serving kernels with FP32 versus losslessly stored BF16 fn weights.
// This isolated extension neither registers nor replaces serving operators.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "mhc_ampere.cuh"

using torch::Tensor;
using namespace tms::dsv4_mhc;

template <bool FUSED, typename FnT, bool PAIRED_FN = false>
std::vector<Tensor> run_typed(
    Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor fn,
    Tensor scale, Tensor base) {
    const int T = residual.size(0);
    auto floats = fn.options().dtype(torch::kFloat32);
    auto residual_out = FUSED ? torch::empty_like(residual) : residual;
    auto partial = torch::empty({T, 64, 25}, floats);
    auto next_post = torch::empty({T, 4}, floats);
    auto next_comb = torch::empty({T, 4, 4}, floats);
    auto out = torch::empty({T, 4096}, residual.options());
    auto bp = [](Tensor t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); };
    const __nv_bfloat16 *xp = bp(x), *rp = bp(residual), *np = nullptr;
    const float *pp = post.data_ptr<float>(), *cp = comb.data_ptr<float>(),
                *sp = scale.data_ptr<float>(), *basp = base.data_ptr<float>();
    const FnT* fp = reinterpret_cast<const FnT*>(fn.data_ptr());
    auto rop = reinterpret_cast<__nv_bfloat16*>(residual_out.data_ptr());
    auto op = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    float *parp = partial.data_ptr<float>(), *npp = next_post.data_ptr<float>(),
          *ncp = next_comb.data_ptr<float>();
    float rms_eps = 1e-5f, pre_eps = 1e-6f, sinkhorn_eps = 1e-6f,
          post_multiplier = 2.0f, norm_eps = 0.0f;
    int repeats = 20;
    const auto stream = at::cuda::getCurrentCUDAStream();
    if (T <= 8) {
        auto kernel = fused_pre_transition<FUSED, false, 4096, 64, FnT>;
        int sms = 0, blocks = 0;
        C10_CUDA_CHECK(cudaDeviceGetAttribute(
            &sms, cudaDevAttrMultiProcessorCount, residual.get_device()));
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, kernel, THREADS, 0));
        TORCH_CHECK(T * 64 <= sms * blocks, "cooperative grid exceeds residency");
        void* args[] = {&xp, &rp, &pp, &cp, &fp, &rop, &parp, &sp, &basp,
                       &npp, &ncp, &op, &np, &rms_eps, &pre_eps, &sinkhorn_eps,
                       &post_multiplier, &repeats, &norm_eps};
        C10_CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<const void*>(kernel), dim3(64, T), dim3(THREADS),
            args, 0, stream));
    } else {
        if (T >= 64) {
            auto kernel = partials_prefill<
                FUSED, 4096, FnT, PAIRED_FN && std::is_same_v<FnT, __nv_bfloat16>>;
            static const auto configured = cudaFuncSetAttribute(
                kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, PREFILL_SMEM);
            C10_CUDA_CHECK(configured);
            kernel<<<dim3(SPLITS, (T + PREFILL_TILE - 1) / PREFILL_TILE),
                     THREADS, PREFILL_SMEM, stream>>>(
                xp, rp, pp, cp, fp, rop, parp, T);
        } else {
            partials<MIXES, FUSED, FnT><<<dim3(SPLITS, T), THREADS, 0, stream>>>(
                xp, rp, pp, cp, fp, rop, parp, 4096);
        }
        finalize_pre_mix<<<T, 32, 0, stream>>>(
            parp, sp, basp, npp, ncp, 4096, rms_eps, pre_eps, sinkhorn_eps,
            post_multiplier, repeats);
        apply_pre_mix<MIXES + 1><<<dim3(4096 / THREADS, T), THREADS, 0, stream>>>(
            parp, FUSED ? rop : rp, op, 4096);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {residual_out, next_post, next_comb, out};
}

template <bool PAIRED_FN>
std::vector<Tensor> run(
    Tensor x, Tensor residual, Tensor post, Tensor comb, Tensor fn,
    Tensor scale, Tensor base, bool fused) {
    TORCH_CHECK(residual.is_cuda() && residual.dim() == 3 &&
                residual.size(0) >= 1 && residual.size(0) <= 7616 &&
                residual.size(1) == 4 && residual.size(2) == 4096,
                "expected residual [1..7616,4,4096]");
    const c10::cuda::CUDAGuard guard(residual.device());
    for (const auto& t : {x, residual, post, comb, fn, scale, base})
        TORCH_CHECK(t.device() == residual.device() && t.is_contiguous(),
                    "all inputs must be contiguous and on the residual device");
    for (const auto& t : {x, residual})
        TORCH_CHECK(t.scalar_type() == torch::kBFloat16, "expected BF16 activations");
    for (const auto& t : {post, comb, scale, base})
        TORCH_CHECK(t.scalar_type() == torch::kFloat32, "expected FP32 parameters");
    TORCH_CHECK(fn.scalar_type() == torch::kFloat32 ||
                fn.scalar_type() == torch::kBFloat16, "expected FP32 or BF16 fn");
    if constexpr (PAIRED_FN) {
        TORCH_CHECK(reinterpret_cast<uintptr_t>(fn.data_ptr()) % 4 == 0,
                    "paired fn staging requires a four-byte-aligned allocation");
    }
    const int T = residual.size(0);
    TORCH_CHECK(x.sizes() == at::IntArrayRef({T, 4096}) &&
                post.sizes() == at::IntArrayRef({T, 4}) &&
                comb.sizes() == at::IntArrayRef({T, 4, 4}) &&
                fn.sizes() == at::IntArrayRef({24, 16384}) &&
                scale.numel() == 3 && base.numel() == 24,
                "unexpected probe parameter shape");
#define RUN(F, TYPE) run_typed<F, TYPE, PAIRED_FN>(x, residual, post, comb, fn, scale, base)
    if (fn.scalar_type() == torch::kBFloat16) {
        if (fused) return RUN(true, __nv_bfloat16);
        return RUN(false, __nv_bfloat16);
    }
    if (fused) return RUN(true, float);
    return RUN(false, float);
#undef RUN
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run<false>);
    m.def("run_paired", &run<true>);
}
