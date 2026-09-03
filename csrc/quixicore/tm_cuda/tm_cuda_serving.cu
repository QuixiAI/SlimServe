// tk_cuda serving/decode bindings: torch wrappers over the validated W4/W5
// kernels (kernels/serving/*_kernels.cuh). Registered into the _C module by
// init_serving(m), called from tm_cuda_ext.cu's PYBIND11_MODULE.
#include "kv_cache_kernels.cuh"
#include "paged_attn_v2_kernels.cuh"
#include "rope_kv_kernels.cuh"
#include "attn_q_kernels.cuh"
#include "mla_kernels.cuh"
#include "beam_xcache_kernels.cuh"
#include "attn_varlen_kernels.cuh"
#include "slot_mapping_kernels.cuh"
#include "v2_batch_kernels.cuh"
#include "indexer_logits_mma.cuh"
#include "indexer_paged_logits.cuh"
#include "sampling_kernels.cuh"
#include "spec_beam_kernels.cuh"
#include "turboquant_kernels.cuh"
#include "mhc_ampere.cuh"
#include "dsv4_router_ampere.cuh"
#include "dsv4_projection_ampere.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>

using namespace tms;
namespace py = pybind11;

#define CK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
static const half* hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }
static half* hpm(torch::Tensor& t) { return reinterpret_cast<half*>(t.data_ptr()); }
static const __nv_bfloat16* bp(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
static __nv_bfloat16* bpm(torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }
static const float* fp(const torch::Tensor& t) { return t.data_ptr<float>(); }
static float* fpm(torch::Tensor& t) { return t.data_ptr<float>(); }

static torch::Tensor py_dsv4_router_gemm(torch::Tensor x,
                                         torch::Tensor weight) {
    CK(x); CK(weight);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                    weight.scalar_type() == torch::kBFloat16,
                "DSV4 router requires bf16 input and weight");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2,
                "DSV4 router requires 2D input and weight");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8 &&
                    x.size(1) == dsv4_router::HIDDEN,
                "DSV4 router input must have shape [1..8, 4096]");
    TORCH_CHECK(weight.size(0) == dsv4_router::EXPERTS &&
                    weight.size(1) == dsv4_router::HIDDEN,
                "DSV4 router weight must have shape [256, 4096]");
    auto output = torch::empty(
        {x.size(0), dsv4_router::EXPERTS},
        x.options().dtype(torch::kFloat));
    dsv4_router::launch(bp(x), bp(weight), fpm(output), int(x.size(0)),
                        stream());
    return output;
}

// Diagnostic slot for bf16_hash_router out-of-range token IDs:
// {flag, token_index, raw_id_lo32, raw_id_hi32}. Allocated lazily; the kernel
// records the first offender and treats the token as padding instead of
// dereferencing garbage.
static int32_t* dsv4_hash_router_debug_slot() {
    static int32_t* slot = nullptr;
    if (slot == nullptr) {
        cudaMalloc(&slot, 4 * sizeof(int32_t));
        cudaMemset(slot, 0, 4 * sizeof(int32_t));
    }
    return slot;
}

// Returns {flag, token_index, raw_id_lo32, raw_id_hi32} as a CPU int32 tensor
// and clears the slot. Synchronous D2H; intended for post-step diagnostics.
static torch::Tensor py_dsv4_hash_router_debug() {
    auto out = torch::zeros({4}, torch::dtype(torch::kInt));
    int32_t* slot = dsv4_hash_router_debug_slot();
    cudaMemcpy(out.data_ptr<int32_t>(), slot, 4 * sizeof(int32_t),
               cudaMemcpyDeviceToHost);
    cudaMemset(slot, 0, 4 * sizeof(int32_t));
    return out;
}

static std::tuple<torch::Tensor, torch::Tensor> py_dsv4_hash_router(
        torch::Tensor x, torch::Tensor weight, torch::Tensor input_ids,
        torch::Tensor tid2eid, double routed_scaling_factor,
        c10::optional<torch::Tensor> is_padding) {
    CK(x); CK(weight); CK(input_ids); CK(tid2eid);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                    weight.scalar_type() == torch::kBFloat16,
                "DSV4 hash router requires bf16 input and weight");
    TORCH_CHECK((input_ids.scalar_type() == torch::kInt ||
                 input_ids.scalar_type() == torch::kLong) &&
                    tid2eid.scalar_type() == torch::kInt,
                "DSV4 hash router requires int32/int64 input IDs and int32 table");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2 &&
                    input_ids.dim() == 1 && tid2eid.dim() == 2,
                "invalid DSV4 hash router tensor rank");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8 &&
                    x.size(1) == dsv4_router::HIDDEN,
                "DSV4 hash router input must have shape [1..8, 4096]");
    TORCH_CHECK(weight.size(0) == dsv4_router::EXPERTS &&
                    weight.size(1) == dsv4_router::HIDDEN,
                "DSV4 hash router weight must have shape [256, 4096]");
    TORCH_CHECK(input_ids.size(0) == x.size(0) &&
                    tid2eid.size(1) == dsv4_router::HASH_TOPK,
                "DSV4 hash router IDs must match tokens and top-k 6");
    const bool* padding_ptr = nullptr;
    if (is_padding.has_value()) {
        TORCH_CHECK(is_padding->is_cuda() && is_padding->is_contiguous(),
                    "is_padding must be contiguous CUDA");
        TORCH_CHECK(is_padding->scalar_type() == torch::kBool &&
                        is_padding->numel() == x.size(0),
                    "DSV4 hash router padding mask must be bool [tokens]");
        padding_ptr = is_padding->data_ptr<bool>();
    }
    auto topk_weights = torch::empty(
        {x.size(0), dsv4_router::HASH_TOPK},
        x.options().dtype(torch::kFloat));
    auto topk_ids = torch::empty(
        {x.size(0), dsv4_router::HASH_TOPK},
        x.options().dtype(torch::kInt));
    if (input_ids.scalar_type() == torch::kInt) {
        dsv4_router::launch_hash(
            bp(x), bp(weight), input_ids.data_ptr<int32_t>(),
            tid2eid.data_ptr<int32_t>(), padding_ptr,
            float(routed_scaling_factor), fpm(topk_weights),
            topk_ids.data_ptr<int32_t>(), int(x.size(0)),
            int(tid2eid.size(0)), dsv4_hash_router_debug_slot(), stream());
    } else {
        dsv4_router::launch_hash(
            bp(x), bp(weight), input_ids.data_ptr<int64_t>(),
            tid2eid.data_ptr<int32_t>(), padding_ptr,
            float(routed_scaling_factor), fpm(topk_weights),
            topk_ids.data_ptr<int32_t>(), int(x.size(0)),
            int(tid2eid.size(0)), dsv4_hash_router_debug_slot(), stream());
    }
    return {topk_weights, topk_ids};
}

static torch::Tensor py_dsv4_projection_gemv(torch::Tensor x,
                                             torch::Tensor weight,
                                             bool bf16_output) {
    CK(x); CK(weight);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                    weight.scalar_type() == torch::kBFloat16,
                "DSV4 projection requires bf16 input and weight");
    TORCH_CHECK(x.dim() == 2 && weight.dim() == 2,
                "DSV4 projection requires 2D input and weight");
    TORCH_CHECK(x.size(0) >= 1 && x.size(0) <= 8 &&
                    x.size(1) == dsv4_projection::HIDDEN,
                "DSV4 projection input must have shape [1..8, 4096]");
    TORCH_CHECK(weight.size(1) == dsv4_projection::HIDDEN,
                "DSV4 projection weight must have 4096 columns");
    auto output = torch::empty({x.size(0), weight.size(0)}, x.options().dtype(
        bf16_output ? torch::kBFloat16 : torch::kFloat));
    if (bf16_output) {
        dsv4_projection::launch(bp(x), bp(weight), bpm(output), int(x.size(0)),
                                int(weight.size(0)), stream());
    } else {
        dsv4_projection::launch(bp(x), bp(weight), fpm(output), int(x.size(0)),
                                int(weight.size(0)), stream());
    }
    return output;
}

__global__ void fill_short_context_topk_indices_kernel(
        int* __restrict__ output, const int64_t* __restrict__ positions,
        int rows, int topk, int compress_ratio) {
    const int row = int(blockIdx.x);
    if (row >= rows) return;
    const int num_compressed = int((positions[row] + 1) / compress_ratio);
    for (int col = int(threadIdx.x); col < topk; col += int(blockDim.x)) {
        output[(size_t)row * topk + col] = col < num_compressed ? col : -1;
    }
}

static void py_fill_short_context_topk_indices(
        torch::Tensor output, torch::Tensor positions, int64_t topk,
        int64_t compress_ratio) {
    CK(output); CK(positions);
    TORCH_CHECK(output.scalar_type() == torch::kInt,
                "output must be int32");
    TORCH_CHECK(positions.scalar_type() == torch::kLong,
                "positions must be int64");
    TORCH_CHECK(output.dim() == 2 && positions.dim() == 1,
                "output must be 2D and positions must be 1D");
    TORCH_CHECK(topk > 0 && topk <= output.size(1), "invalid topk");
    TORCH_CHECK(compress_ratio > 0, "compress_ratio must be positive");
    const int rows = int(positions.size(0));
    TORCH_CHECK(rows <= output.size(0), "output has too few rows");
    if (rows > 0) {
        fill_short_context_topk_indices_kernel<<<rows, 256, 0, stream()>>>(
            output.data_ptr<int>(), positions.data_ptr<int64_t>(), rows,
            int(topk), int(compress_ratio));
    }
}

// ---- DeepSeek-V4 mHC, decode-specialized for Ampere ----
static bool dsv4_mhc_cooperative_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("VLLM_DSV4_MHC_COOPERATIVE");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

static int dsv4_mhc_splits() {
    static const int splits = [] {
        const char* value = std::getenv("VLLM_DSV4_MHC_SPLITS");
        if (value == nullptr) return 64;
        const int parsed = std::atoi(value);
        return parsed == 32 || parsed == 64 ? parsed : 64;
    }();
    return splits;
}

template <typename FnT, bool FUSED_POST, bool RMS_NORM, int NSPLITS>
static void launch_dsv4_mhc_pre_transition(
        const torch::Tensor* x, torch::Tensor residual,
        const torch::Tensor* post_mix, const torch::Tensor* comb_mix,
        torch::Tensor fn, torch::Tensor* residual_out, torch::Tensor partial,
        torch::Tensor scale, torch::Tensor base, torch::Tensor next_post,
        torch::Tensor next_comb, torch::Tensor layer_input,
        const torch::Tensor* norm_weight, float rms_eps, float pre_eps,
        float sinkhorn_eps, float post_multiplier, int sinkhorn_repeat,
        float norm_eps) {
    const __nv_bfloat16* x_ptr = FUSED_POST ? bp(*x) : nullptr;
    const __nv_bfloat16* residual_ptr = bp(residual);
    const float* post_ptr = FUSED_POST ? fp(*post_mix) : nullptr;
    const float* comb_ptr = FUSED_POST ? fp(*comb_mix) : nullptr;
    const FnT* fn_ptr = reinterpret_cast<const FnT*>(fn.data_ptr());
    __nv_bfloat16* residual_out_ptr =
        FUSED_POST ? bpm(*residual_out) : nullptr;
    float* partial_ptr = fpm(partial);
    const float* scale_ptr = fp(scale);
    const float* base_ptr = fp(base);
    float* next_post_ptr = fpm(next_post);
    float* next_comb_ptr = fpm(next_comb);
    __nv_bfloat16* layer_input_ptr = bpm(layer_input);
    const __nv_bfloat16* norm_ptr = RMS_NORM ? bp(*norm_weight) : nullptr;
    auto kernel =
        dsv4_mhc::fused_pre_transition<FUSED_POST, RMS_NORM, 4096, NSPLITS,
                                       FnT>;
    void* args[] = {
        &x_ptr, &residual_ptr, &post_ptr, &comb_ptr, &fn_ptr,
        &residual_out_ptr, &partial_ptr, &scale_ptr, &base_ptr,
        &next_post_ptr, &next_comb_ptr, &layer_input_ptr, &norm_ptr,
        &rms_eps, &pre_eps, &sinkhorn_eps, &post_multiplier,
        &sinkhorn_repeat, &norm_eps,
    };
    const cudaError_t error = cudaLaunchCooperativeKernel(
        reinterpret_cast<const void*>(kernel), dim3(NSPLITS, 1),
        dim3(dsv4_mhc::THREADS), args, 0, stream());
    TORCH_CHECK(error == cudaSuccess,
                "DSV4 cooperative mHC launch failed: ",
                cudaGetErrorString(error));
}

template <bool FUSED_POST, bool RMS_NORM>
static void launch_dsv4_mhc_pre_transition_selected(
        const torch::Tensor* x, torch::Tensor residual,
        const torch::Tensor* post_mix, const torch::Tensor* comb_mix,
        torch::Tensor fn, torch::Tensor* residual_out, torch::Tensor partial,
        torch::Tensor scale, torch::Tensor base, torch::Tensor next_post,
        torch::Tensor next_comb, torch::Tensor layer_input,
        const torch::Tensor* norm_weight, float rms_eps, float pre_eps,
        float sinkhorn_eps, float post_multiplier, int sinkhorn_repeat,
        float norm_eps) {
#define LAUNCH_MHC_TYPED(FN_T, NSPLITS)                                      \
    launch_dsv4_mhc_pre_transition<FN_T, FUSED_POST, RMS_NORM, NSPLITS>(     \
        x, residual, post_mix, comb_mix, fn, residual_out, partial, scale,   \
        base, next_post, next_comb, layer_input, norm_weight, rms_eps,       \
        pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_repeat, norm_eps)
    if (fn.scalar_type() == torch::kHalf) {
        if (dsv4_mhc_splits() == 32) {
            LAUNCH_MHC_TYPED(half, 32);
        } else {
            LAUNCH_MHC_TYPED(half, 64);
        }
    } else if (dsv4_mhc_splits() == 32) {
        LAUNCH_MHC_TYPED(float, 32);
    } else {
        LAUNCH_MHC_TYPED(float, 64);
    }
#undef LAUNCH_MHC_TYPED
}

template <int NOUT, bool FUSED_POST>
static void launch_dsv4_mhc_partials(
        const __nv_bfloat16* x, const __nv_bfloat16* residual,
        const float* post, const float* comb, torch::Tensor fn,
        __nv_bfloat16* residual_out, float* partial, int hidden_size,
        dim3 grid) {
    if (fn.scalar_type() == torch::kHalf) {
        dsv4_mhc::partials<NOUT, FUSED_POST, half>
            <<<grid, dsv4_mhc::THREADS, 0, stream()>>>(
                x, residual, post, comb,
                reinterpret_cast<const half*>(fn.data_ptr()), residual_out,
                partial, hidden_size);
    } else {
        dsv4_mhc::partials<NOUT, FUSED_POST, float>
            <<<grid, dsv4_mhc::THREADS, 0, stream()>>>(
                x, residual, post, comb, fp(fn), residual_out, partial,
                hidden_size);
    }
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_dsv4_mhc_pre(
        torch::Tensor residual, torch::Tensor fn, torch::Tensor hc_scale,
        torch::Tensor hc_base, double rms_eps, double pre_eps,
        double sinkhorn_eps, double post_multiplier, int64_t sinkhorn_repeat,
        c10::optional<torch::Tensor> norm_weight, double norm_eps) {
    CK(residual); CK(fn); CK(hc_scale); CK(hc_base);
    TORCH_CHECK(residual.scalar_type() == torch::kBFloat16, "residual must be bf16");
    TORCH_CHECK(fn.scalar_type() == torch::kFloat32 ||
                fn.scalar_type() == torch::kFloat16,
                "fn must be float16 or float32");
    const int T = residual.size(0), H = residual.size(2);
    TORCH_CHECK(residual.size(1) == dsv4_mhc::HC && fn.size(0) == dsv4_mhc::MIXES,
                "DSV4 mHC expects hc_mult=4 and 24 mix rows");
    auto float_options = fn.options().dtype(torch::kFloat32);
    auto partial = torch::empty({T, 64, dsv4_mhc::MIXES + 1}, float_options);
    auto post = torch::empty({T, dsv4_mhc::HC}, float_options);
    auto comb = torch::empty({T, dsv4_mhc::HC, dsv4_mhc::HC}, float_options);
    auto layer_input = torch::empty({T, H}, residual.options());
    if (dsv4_mhc_cooperative_enabled() && T == 1 && H == 4096) {
        if (norm_weight) {
            launch_dsv4_mhc_pre_transition_selected<false, true>(
                nullptr, residual, nullptr, nullptr, fn, nullptr, partial,
                hc_scale, hc_base, post, comb, layer_input, &*norm_weight,
                float(rms_eps), float(pre_eps), float(sinkhorn_eps),
                float(post_multiplier), int(sinkhorn_repeat), float(norm_eps));
        } else {
            launch_dsv4_mhc_pre_transition_selected<false, false>(
                nullptr, residual, nullptr, nullptr, fn, nullptr, partial,
                hc_scale, hc_base, post, comb, layer_input, nullptr,
                float(rms_eps), float(pre_eps), float(sinkhorn_eps),
                float(post_multiplier), int(sinkhorn_repeat), float(norm_eps));
        }
        return {post, comb, layer_input};
    }
    launch_dsv4_mhc_partials<dsv4_mhc::MIXES, false>(
        nullptr, bp(residual), nullptr, nullptr, fn, nullptr, fpm(partial), H,
        dim3(dsv4_mhc::SPLITS, T));
    dsv4_mhc::finalize_pre_mix<<<T, 32, 0, stream()>>>(
        fpm(partial), fp(hc_scale), fp(hc_base), fpm(post),
        fpm(comb), H, float(rms_eps), float(pre_eps),
        float(sinkhorn_eps), float(post_multiplier), int(sinkhorn_repeat));
    if (norm_weight) {
        TORCH_CHECK(norm_weight->is_cuda() && norm_weight->is_contiguous(),
                    "norm_weight must be contiguous CUDA");
        TORCH_CHECK(H == 4096 && norm_weight->scalar_type() == torch::kBFloat16 &&
                    norm_weight->numel() == H,
                    "fused DSV4 mHC RMSNorm expects a 4096-element bf16 weight");
        dsv4_mhc::apply_pre_mix_rms_norm<dsv4_mhc::MIXES + 1>
            <<<T, dsv4_mhc::THREADS, 0, stream()>>>(
                fp(partial), bp(residual), bp(*norm_weight), bpm(layer_input),
                float(norm_eps));
    } else {
        dsv4_mhc::apply_pre_mix<dsv4_mhc::MIXES + 1>
            <<<dim3((H + dsv4_mhc::THREADS - 1) / dsv4_mhc::THREADS, T),
                dsv4_mhc::THREADS, 0, stream()>>>(
                fp(partial), bp(residual), bpm(layer_input), H);
    }
    return {post, comb, layer_input};
}

static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
py_dsv4_mhc_fused_post_pre(
        torch::Tensor x, torch::Tensor residual, torch::Tensor post_mix,
        torch::Tensor comb_mix, torch::Tensor fn, torch::Tensor hc_scale,
        torch::Tensor hc_base, double rms_eps, double pre_eps,
        double sinkhorn_eps, double post_multiplier, int64_t sinkhorn_repeat,
        c10::optional<torch::Tensor> norm_weight, double norm_eps) {
    CK(x); CK(residual); CK(post_mix); CK(comb_mix); CK(fn); CK(hc_scale); CK(hc_base);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 &&
                residual.scalar_type() == torch::kBFloat16, "x/residual must be bf16");
    TORCH_CHECK((fn.scalar_type() == torch::kFloat32 ||
                 fn.scalar_type() == torch::kFloat16) &&
                post_mix.scalar_type() == torch::kFloat32 &&
                comb_mix.scalar_type() == torch::kFloat32,
                "mHC fn must be float16/float32 and mixes must be float32");
    const int T = residual.size(0), H = residual.size(2);
    TORCH_CHECK(residual.size(1) == dsv4_mhc::HC && fn.size(0) == dsv4_mhc::MIXES,
                "DSV4 mHC expects hc_mult=4 and 24 mix rows");
    auto residual_out = torch::empty_like(residual);
    auto float_options = fn.options().dtype(torch::kFloat32);
    auto partial = torch::empty({T, 64, dsv4_mhc::MIXES + 1}, float_options);
    auto next_post = torch::empty({T, dsv4_mhc::HC}, float_options);
    auto next_comb = torch::empty(
        {T, dsv4_mhc::HC, dsv4_mhc::HC}, float_options);
    auto layer_input = torch::empty({T, H}, residual.options());
    if (dsv4_mhc_cooperative_enabled() && T == 1 && H == 4096) {
        if (norm_weight) {
            launch_dsv4_mhc_pre_transition_selected<true, true>(
                &x, residual, &post_mix, &comb_mix, fn, &residual_out,
                partial, hc_scale, hc_base, next_post, next_comb, layer_input,
                &*norm_weight, float(rms_eps), float(pre_eps),
                float(sinkhorn_eps), float(post_multiplier),
                int(sinkhorn_repeat), float(norm_eps));
        } else {
            launch_dsv4_mhc_pre_transition_selected<true, false>(
                &x, residual, &post_mix, &comb_mix, fn, &residual_out,
                partial, hc_scale, hc_base, next_post, next_comb, layer_input,
                nullptr, float(rms_eps), float(pre_eps), float(sinkhorn_eps),
                float(post_multiplier), int(sinkhorn_repeat), float(norm_eps));
        }
        return {residual_out, next_post, next_comb, layer_input};
    }
    launch_dsv4_mhc_partials<dsv4_mhc::MIXES, true>(
        bp(x), bp(residual), fp(post_mix), fp(comb_mix), fn,
        bpm(residual_out), fpm(partial), H, dim3(dsv4_mhc::SPLITS, T));
    dsv4_mhc::finalize_pre_mix<<<T, 32, 0, stream()>>>(
        fpm(partial), fp(hc_scale), fp(hc_base), fpm(next_post),
        fpm(next_comb), H, float(rms_eps), float(pre_eps),
        float(sinkhorn_eps), float(post_multiplier), int(sinkhorn_repeat));
    if (norm_weight) {
        TORCH_CHECK(norm_weight->is_cuda() && norm_weight->is_contiguous(),
                    "norm_weight must be contiguous CUDA");
        TORCH_CHECK(H == 4096 && norm_weight->scalar_type() == torch::kBFloat16 &&
                    norm_weight->numel() == H,
                    "fused DSV4 mHC RMSNorm expects a 4096-element bf16 weight");
        dsv4_mhc::apply_pre_mix_rms_norm<dsv4_mhc::MIXES + 1>
            <<<T, dsv4_mhc::THREADS, 0, stream()>>>(
                fp(partial), bp(residual_out), bp(*norm_weight),
                bpm(layer_input), float(norm_eps));
    } else {
        dsv4_mhc::apply_pre_mix<dsv4_mhc::MIXES + 1>
            <<<dim3((H + dsv4_mhc::THREADS - 1) / dsv4_mhc::THREADS, T),
                dsv4_mhc::THREADS, 0, stream()>>>(
                fp(partial), bp(residual_out), bpm(layer_input), H);
    }
    return {residual_out, next_post, next_comb, layer_input};
}

static torch::Tensor py_dsv4_mhc_post(
        torch::Tensor x, torch::Tensor residual, torch::Tensor post_mix,
        torch::Tensor comb_mix) {
    CK(x); CK(residual); CK(post_mix); CK(comb_mix);
    const int T = residual.size(0), H = residual.size(2);
    auto output = torch::empty_like(residual);
    dsv4_mhc::post<<<dim3((H + dsv4_mhc::THREADS - 1) / dsv4_mhc::THREADS, T),
                          dsv4_mhc::THREADS, 0, stream()>>>(
        bp(x), bp(residual), fp(post_mix), fp(comb_mix), bpm(output), H);
    return output;
}

static torch::Tensor py_dsv4_hc_head(
        torch::Tensor residual, torch::Tensor fn, torch::Tensor hc_scale,
        torch::Tensor hc_base, double rms_eps, double hc_eps) {
    CK(residual); CK(fn); CK(hc_scale); CK(hc_base);
    TORCH_CHECK(fn.scalar_type() == torch::kFloat32 ||
                fn.scalar_type() == torch::kFloat16,
                "HC head fn must be float16 or float32");
    const int T = residual.size(0), H = residual.size(2);
    TORCH_CHECK(residual.size(1) == dsv4_mhc::HC && fn.size(0) == dsv4_mhc::HC,
                "DSV4 HC head expects hc_mult=4");
    auto partial = torch::empty(
        {T, dsv4_mhc::SPLITS, dsv4_mhc::HC + 1},
        fn.options().dtype(torch::kFloat32));
    auto output = torch::empty({T, H}, residual.options());
    launch_dsv4_mhc_partials<dsv4_mhc::HC, false>(
        nullptr, bp(residual), nullptr, nullptr, fn, nullptr, fpm(partial), H,
        dim3(dsv4_mhc::SPLITS, T));
    dsv4_mhc::finalize_head_mix<<<T, 32, 0, stream()>>>(
        fpm(partial), fp(hc_scale), fp(hc_base), H, float(rms_eps), float(hc_eps));
    dsv4_mhc::apply_pre_mix<dsv4_mhc::HC + 1>
        <<<dim3((H + dsv4_mhc::THREADS - 1) / dsv4_mhc::THREADS, T),
            dsv4_mhc::THREADS, 0, stream()>>>(
            fp(partial), bp(residual), bpm(output), H);
    return output;
}

// ---- kv cache management ----
static void py_kv_scatter(torch::Tensor key, torch::Tensor value, torch::Tensor slot_mapping,
                          torch::Tensor key_cache, torch::Tensor value_cache, int64_t block_size) {
    CK(key); CK(value); CK(slot_mapping); CK(key_cache); CK(value_cache);
    const int T = key.size(0), H = key.size(1), D = key.size(2);
    kv_cache_scatter<half><<<T, 128, 0, stream()>>>(hp(key), hp(value),
        slot_mapping.data_ptr<int64_t>(), hpm(key_cache), hpm(value_cache), H, D, int(block_size));
}
static std::tuple<torch::Tensor, torch::Tensor> py_kv_gather(
        torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_table,
        torch::Tensor cu_seq_lens, int64_t num_tokens, int64_t H, int64_t D, int64_t block_size) {
    CK(key_cache); CK(value_cache); CK(block_table); CK(cu_seq_lens);
    auto k = torch::empty({num_tokens, H, D}, key_cache.options());
    auto v = torch::empty({num_tokens, H, D}, key_cache.options());
    const int B = cu_seq_lens.numel() - 1;
    kv_cache_gather<half><<<int(num_tokens), 128, 0, stream()>>>(hp(key_cache), hp(value_cache),
        hpm(k), hpm(v), block_table.data_ptr<int>(), cu_seq_lens.data_ptr<int>(),
        int(num_tokens), B, int(block_size), int(block_table.size(1)), int(H), int(D));
    return {k, v};
}
static void py_copy_blocks(torch::Tensor ks, torch::Tensor vs, torch::Tensor kd, torch::Tensor vd,
                           torch::Tensor pairs, int64_t block_elems) {
    CK(ks); CK(vs); CK(kd); CK(vd); CK(pairs);
    const int n = pairs.size(0);
    kv_cache_copy_blocks<half><<<n, 128, 0, stream()>>>(hp(ks), hp(vs), hpm(kd), hpm(vd),
        pairs.data_ptr<int64_t>(), int(block_elems));
}

// ---- decode attention ----
static torch::Tensor py_paged_attention(torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
        torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale, int64_t num_kv_heads,
        c10::optional<torch::Tensor> alibi, c10::optional<torch::Tensor> block_mask, int64_t window) {
    CK(q); CK(kc); CK(vc); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    auto out = torch::empty_like(q);
    const float* al = alibi ? alibi->data_ptr<float>() : nullptr;
    const int* bm = block_mask ? block_mask->data_ptr<int>() : nullptr;
    dim3 grid(H, B);
    #define LAUNCH(DD) paged_attention<half, DD><<<grid, 32, 0, stream()>>>(hp(q), hp(kc), hp(vc), \
        bt.data_ptr<int>(), ctx.data_ptr<int>(), hpm(out), int(block_size), int(bt.size(1)), \
        float(scale), H, int(num_kv_heads), al, al ? 1 : 0, bm, bm ? 1 : 0, int(window))
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}
static torch::Tensor py_paged_attention_v2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
        torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale, int64_t num_kv_heads,
        int64_t partition_size, int64_t max_context, int64_t window) {
    CK(q); CK(kc); CK(vc); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    const int P = int((max_context + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, D}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    auto out = torch::empty_like(q);
    dim3 pg(H, B, P), rg(H, B);
    #define LAUNCH(DD) do { \
        paged_attention_partition<half, DD><<<pg, 32, 0, stream()>>>(hp(q), hp(kc), hp(vc), \
            bt.data_ptr<int>(), ctx.data_ptr<int>(), tmp.data_ptr<float>(), ml.data_ptr<float>(), \
            es.data_ptr<float>(), int(block_size), int(bt.size(1)), float(scale), H, \
            int(num_kv_heads), P, int(partition_size), int(window)); \
        paged_attention_reduce<half, DD><<<rg, 32, 0, stream()>>>(tmp.data_ptr<float>(), \
            ml.data_ptr<float>(), es.data_ptr<float>(), hpm(out), H, P); } while (0)
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}

// ---- rope + insert ----
static void py_rope_kv_insert(torch::Tensor k, torch::Tensor v, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, torch::Tensor slots, torch::Tensor kc, torch::Tensor vc,
        int64_t num_kv_heads, int64_t block_size, c10::optional<torch::Tensor> norm_weight,
        bool gemma, double eps) {
    CK(k); CK(v); CK(cosb); CK(sinb); CK(positions); CK(slots); CK(kc); CK(vc);
    const int M = k.size(0), D = k.size(1);
    const half* w = norm_weight ? hp(*norm_weight) : nullptr;
    #define LAUNCH(DD, NORM) rope_kv_insert<half, DD, NORM><<<M, 32, 0, stream()>>>(hp(k), hp(v), \
        hp(cosb), hp(sinb), positions.data_ptr<int>(), slots.data_ptr<int64_t>(), hpm(kc), hpm(vc), \
        w, int(num_kv_heads), int(block_size), gemma ? 1 : 0, float(eps))
    if (D == 64)  { if (w) LAUNCH(64, true);  else LAUNCH(64, false); }
    else if (D == 128) { if (w) LAUNCH(128, true); else LAUNCH(128, false); }
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
}
static torch::Tensor py_rope_q(torch::Tensor q, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, int64_t num_heads, c10::optional<torch::Tensor> norm_weight,
        bool gemma, double eps) {
    CK(q); CK(cosb); CK(sinb); CK(positions);
    const int M = q.size(0), D = q.size(1);
    auto out = torch::empty_like(q);
    const half* w = norm_weight ? hp(*norm_weight) : nullptr;
    #define LAUNCH(DD, NORM) rope_q<half, DD, NORM><<<M, 32, 0, stream()>>>(hp(q), hp(cosb), \
        hp(sinb), positions.data_ptr<int>(), hpm(out), w, int(num_heads), gemma ? 1 : 0, float(eps))
    if (D == 64)  { if (w) LAUNCH(64, true);  else LAUNCH(64, false); }
    else if (D == 128) { if (w) LAUNCH(128, true); else LAUNCH(128, false); }
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}

// ---- quantized-KV attention ----
static torch::Tensor py_attn_q(torch::Tensor q, torch::Tensor Kq, torch::Tensor Vq,
                               const std::string& fmt, bool causal, double scale,
                               int64_t B, int64_t H, int64_t N, int64_t D) {
    CK(q); CK(Kq); CK(Vq);
    auto out = torch::empty_like(q);
    dim3 grid{unsigned(N), unsigned(H), unsigned(B)};
    #define LAUNCH(FMT, DD, C) attn_q<tmq::FMT, DD, C><<<grid, 32, 0, stream()>>>(hp(q), \
        Kq.data_ptr<uint8_t>(), Vq.data_ptr<uint8_t>(), hpm(out), int(N), int(H), float(scale))
    #define DISPATCH(FMT) do { \
        if (D == 64)  { if (causal) LAUNCH(FMT, 64, true);  else LAUNCH(FMT, 64, false);  } \
        else if (D == 128) { if (causal) LAUNCH(FMT, 128, true); else LAUNCH(FMT, 128, false); } \
        else TORCH_CHECK(false, "D must be 64/128"); return out; } while (0)
    if (fmt == "q8_0") DISPATCH(q8_0);
    if (fmt == "q4_0") DISPATCH(q4_0);
    if (fmt == "fp8_e4m3") DISPATCH(fp8_e4m3);
    TORCH_CHECK(false, "attn_q format must be q8_0/q4_0/fp8_e4m3");
    #undef DISPATCH
    #undef LAUNCH
}

// ---- MLA ----
static torch::Tensor py_mla_decode(torch::Tensor q, torch::Tensor kv_cache, torch::Tensor bt,
                                   torch::Tensor ctx, int64_t block_size, double scale) {
    CK(q); CK(kv_cache); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "mla_decode expects QK=576");
    auto out = torch::empty({B, H, 512}, q.options());
    mla_decode<512, 64><<<dim3(H, B), 32, 0, stream()>>>(bp(q), bp(kv_cache), bt.data_ptr<int>(),
        ctx.data_ptr<int>(), bpm(out), int(block_size), int(bt.size(1)), float(scale), H);
    return out;
}
static torch::Tensor py_mla_decode_fp8(torch::Tensor q, torch::Tensor data, torch::Tensor scl,
                                       torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale) {
    CK(q); CK(data); CK(scl); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    auto out = torch::empty_like(q);
    mla_decode_fp8<<<dim3(H, B), 32, 0, stream()>>>(bp(q), data.data_ptr<uint8_t>(),
        scl.data_ptr<uint8_t>(), bt.data_ptr<int>(), ctx.data_ptr<int>(), bpm(out),
        int(block_size), int(bt.size(1)), float(scale), H);
    return out;
}
static void py_mla_kv_insert_fp8(torch::Tensor kv, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, torch::Tensor slots, torch::Tensor data, torch::Tensor scl,
        int64_t block_size) {
    CK(kv); CK(cosb); CK(sinb); CK(positions); CK(slots); CK(data); CK(scl);
    mla_kv_insert_fp8<<<int(kv.size(0)), 32, 0, stream()>>>(bp(kv), bp(cosb), bp(sinb),
        positions.data_ptr<int>(), slots.data_ptr<int64_t>(), data.data_ptr<uint8_t>(),
        scl.data_ptr<uint8_t>(), int(block_size));
}
static torch::Tensor py_mla_q_norm_rope(torch::Tensor q, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, int64_t num_heads, int64_t nope_dim, int64_t rope_dim,
        int64_t norm_mode, double eps, c10::optional<torch::Tensor> norm_weight) {
    CK(q); CK(cosb); CK(sinb); CK(positions);
    const int M = q.size(0), D = q.size(1);
    auto out = torch::empty_like(q);
    const __nv_bfloat16* w = norm_weight ? bp(*norm_weight) : nullptr;
    #define LAUNCH(DD) mla_q_norm_rope<DD><<<M, 32, 0, stream()>>>(bp(q), bp(cosb), bp(sinb), \
        positions.data_ptr<int>(), bpm(out), int(num_heads), int(nope_dim), int(rope_dim), \
        int(norm_mode), float(eps), w)
    if (D == 128) LAUNCH(128); else if (D == 192) LAUNCH(192);
    else if (D == 256) LAUNCH(256); else if (D == 512) LAUNCH(512);
    else TORCH_CHECK(false, "D must be 128/192/256/512");
    #undef LAUNCH
    return out;
}

// ---- samplers ----
static torch::Tensor py_sample(torch::Tensor logits, const std::string& mode, int64_t seed,
                               double temperature, double param, int64_t K) {
    CK(logits);
    TORCH_CHECK(logits.scalar_type() == torch::kFloat, "logits must be fp32");
    const int T = logits.size(0), V = logits.size(1);
    auto out = torch::empty({T}, logits.options().dtype(torch::kInt));
    const float invtemp = temperature > 0 ? float(1.0 / temperature) : 1.0f;
    const float* lp = logits.data_ptr<float>();
    int* op = out.data_ptr<int>();
    if (mode == "argmax")      argmax_k<float><<<T, 32, 0, stream()>>>(lp, op, V);
    else if (mode == "categorical") sample_categorical<float><<<T, 32, 0, stream()>>>(lp, op, V, unsigned(seed), invtemp);
    else if (mode == "top_k")  top_k_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, int(K), unsigned(seed), invtemp);
    else if (mode == "top_p")  top_p_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, float(param), unsigned(seed), invtemp);
    else if (mode == "min_p")  min_p_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, float(param), unsigned(seed), invtemp);
    else if (mode == "typical_p") typical_p_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, float(param), unsigned(seed), invtemp);
    else TORCH_CHECK(false, "unknown sample mode ", mode);
    return out;
}
static torch::Tensor py_apply_penalties(torch::Tensor logits, torch::Tensor prev_tokens,
        torch::Tensor parent_ids, double temperature, double rep, double presence, double freq,
        torch::Tensor bias, int64_t eos_id, int64_t min_length, int64_t gen_len) {
    CK(logits); CK(prev_tokens); CK(parent_ids); CK(bias);
    const int T = logits.size(0), V = logits.size(1), L = prev_tokens.size(1);
    auto counts = torch::zeros({T, V}, logits.options().dtype(torch::kInt));
    auto out = torch::empty_like(logits);
    penalty_histogram<<<(T * L + 255) / 256, 256, 0, stream()>>>(prev_tokens.data_ptr<int>(),
        counts.data_ptr<int>(), V, L, T * L, parent_ids.data_ptr<int>());
    apply_penalty<float><<<T, 32, 0, stream()>>>(logits.data_ptr<float>(), counts.data_ptr<int>(),
        out.data_ptr<float>(), V, temperature > 0 ? float(1.0 / temperature) : 1.0f,
        float(rep), float(presence), float(freq), bias.data_ptr<float>(),
        int(eos_id), int(min_length), int(gen_len));
    return out;
}
static torch::Tensor py_apply_token_bitmask(torch::Tensor logits, torch::Tensor bitmask) {
    CK(logits); CK(bitmask);
    const int T = logits.size(0), V = logits.size(1);
    auto out = torch::empty_like(logits);
    apply_token_bitmask<float><<<T, 32, 0, stream()>>>(logits.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(bitmask.data_ptr<int>()), out.data_ptr<float>(),
        V, int(bitmask.size(1)));
    return out;
}
static torch::Tensor py_apply_bad_words(torch::Tensor logits, torch::Tensor bad_ids, torch::Tensor bad_lens) {
    CK(logits); CK(bad_ids); CK(bad_lens);
    const int T = logits.size(0), V = logits.size(1);
    auto out = torch::empty_like(logits);
    apply_bad_words<float><<<T, 32, 0, stream()>>>(logits.data_ptr<float>(), out.data_ptr<float>(),
        bad_ids.data_ptr<int>(), bad_lens.data_ptr<int>(), V, int(bad_ids.size(1)));
    return out;
}

// ---- beam + spec ----
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_beam_advance(
        torch::Tensor logits, torch::Tensor cum_log_probs, int64_t BM) {
    CK(logits); CK(cum_log_probs);
    const int rows = logits.size(0), V = logits.size(1), two_bm = 2 * int(BM);
    const int B = rows / int(BM);
    auto sc = torch::empty({rows, two_bm}, logits.options());
    auto tok = torch::empty({rows, two_bm}, logits.options().dtype(torch::kInt));
    auto nt = torch::empty({B, int(BM)}, tok.options());
    auto par = torch::empty({B, int(BM)}, tok.options());
    auto nc = torch::empty({B, int(BM)}, logits.options());
    beam_topk_partials<float><<<rows, 32, 0, stream()>>>(logits.data_ptr<float>(),
        cum_log_probs.data_ptr<float>(), sc.data_ptr<float>(), tok.data_ptr<int>(), V, two_bm);
    beam_select<<<B, 32, 0, stream()>>>(sc.data_ptr<float>(), tok.data_ptr<int>(),
        nt.data_ptr<int>(), par.data_ptr<int>(), nc.data_ptr<float>(), int(BM), two_bm);
    return {nt, par, nc};
}
static std::tuple<torch::Tensor, torch::Tensor> py_spec_verify_linear(
        torch::Tensor draft, torch::Tensor draft_probs, torch::Tensor target_probs,
        torch::Tensor bonus, torch::Tensor accept_u, int64_t seed) {
    CK(draft); CK(draft_probs); CK(target_probs); CK(bonus); CK(accept_u);
    const int B = draft.size(0), S = draft.size(1), V = draft_probs.size(2);
    auto out = torch::empty({B, S + 1}, draft.options());
    auto cnt = torch::empty({B}, draft.options());
    spec_verify_linear<<<B, 32, 0, stream()>>>(draft.data_ptr<int>(), draft_probs.data_ptr<float>(),
        target_probs.data_ptr<float>(), bonus.data_ptr<int>(), accept_u.data_ptr<float>(),
        out.data_ptr<int>(), cnt.data_ptr<int>(), S, V, unsigned(seed));
    return {out, cnt};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_build_dynamic_tree(torch::Tensor parents) {
    CK(parents);
    const int B = parents.size(0), N = parents.size(1);
    auto nt = torch::empty_like(parents);
    auto ns = torch::empty_like(parents);
    auto pos = torch::empty_like(parents);
    build_dynamic_tree<<<B, 32, 0, stream()>>>(parents.data_ptr<int>(), nt.data_ptr<int>(),
        ns.data_ptr<int>(), pos.data_ptr<int>(), N);
    return {nt, ns, pos};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_spec_verify_tree(
        torch::Tensor draft, torch::Tensor target_probs, torch::Tensor nt, torch::Tensor ns,
        torch::Tensor tree_valid, int64_t seed) {
    CK(draft); CK(target_probs); CK(nt); CK(ns); CK(tree_valid);
    const int B = nt.size(0), N = nt.size(1), V = target_probs.size(2);
    auto ai = torch::empty({B, N}, nt.options());
    auto at = torch::empty({B, N}, nt.options());
    auto an = torch::empty({B}, nt.options());
    spec_verify_tree<<<B, 32, 0, stream()>>>(draft.data_ptr<int>(), target_probs.data_ptr<float>(),
        nt.data_ptr<int>(), ns.data_ptr<int>(), ai.data_ptr<int>(), at.data_ptr<int>(),
        an.data_ptr<int>(), N, V, unsigned(seed), tree_valid.data_ptr<int>());
    return {ai, at, an};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_spec_compact(
        torch::Tensor out_tokens, torch::Tensor accepted_cnt, torch::Tensor seq_lens) {
    CK(out_tokens); CK(accepted_cnt); CK(seq_lens);
    const int B = out_tokens.size(0), Sp1 = out_tokens.size(1);
    auto pt = torch::empty({B * Sp1}, out_tokens.options());
    auto pp = torch::empty({B * Sp1}, out_tokens.options());
    auto cu = torch::empty({B + 1}, out_tokens.options());
    spec_compact<<<1, 256, 0, stream()>>>(out_tokens.data_ptr<int>(), accepted_cnt.data_ptr<int>(),
        seq_lens.data_ptr<int>(), pt.data_ptr<int>(), pp.data_ptr<int>(), cu.data_ptr<int>(), B, Sp1);
    return {pt, pp, cu};
}
static torch::Tensor py_spec_update_kv_meta(torch::Tensor seq_lens, torch::Tensor accepted_cnt) {
    CK(seq_lens); CK(accepted_cnt);
    const int B = seq_lens.numel();
    auto out = torch::empty_like(seq_lens);
    spec_update_kv_meta<<<(B + 255) / 256, 256, 0, stream()>>>(seq_lens.data_ptr<int>(),
        accepted_cnt.data_ptr<int>(), out.data_ptr<int>(), B);
    return out;
}

// ---- Step-4 additions: bf16 MLA insert, partitioned/sparse MLA decodes,
// gqa_staged decode, dense sliding-window attention ----
static void py_mla_kv_insert(torch::Tensor kv_c, torch::Tensor k_pe, torch::Tensor cosb,
        torch::Tensor sinb, torch::Tensor positions, torch::Tensor slots,
        torch::Tensor kv_cache, int64_t block_size, int64_t norm_mode, double eps,
        c10::optional<torch::Tensor> norm_weight) {
    CK(kv_c); CK(k_pe); CK(cosb); CK(sinb); CK(positions); CK(slots); CK(kv_cache);
    const int T = kv_c.size(0), LATENT = kv_c.size(1), rope_dim = k_pe.size(1);
    const __nv_bfloat16* w = norm_weight ? bp(*norm_weight) : nullptr;
    #define LAUNCH(L) mla_kv_insert<L><<<T, 32, 0, stream()>>>(bp(kv_c), bp(k_pe), bp(cosb), \
        bp(sinb), positions.data_ptr<int>(), slots.data_ptr<int64_t>(), bpm(kv_cache), \
        int(block_size), rope_dim, int(norm_mode), float(eps), w)
    if (LATENT == 128) LAUNCH(128); else if (LATENT == 256) LAUNCH(256);
    else if (LATENT == 512) LAUNCH(512);
    else TORCH_CHECK(false, "LATENT must be 128/256/512");
    #undef LAUNCH
}
static torch::Tensor py_mla_decode_partition(torch::Tensor q, torch::Tensor kv_cache,
        torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale,
        int64_t partition_size, int64_t max_context) {
    CK(q); CK(kv_cache); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "mla_decode_partition expects QK=576");
    const int P = int((max_context + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    auto out = torch::empty({B, H, 512}, q.options());
    mla_decode_partition<512, 64><<<dim3(H, B, P), 32, 0, stream()>>>(bp(q), bp(kv_cache),
        bt.data_ptr<int>(), ctx.data_ptr<int>(), tmp.data_ptr<float>(), ml.data_ptr<float>(),
        es.data_ptr<float>(), int(block_size), int(bt.size(1)), float(scale), H, P,
        int(partition_size));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}
static torch::Tensor py_mla_decode_fp8_partition(torch::Tensor q, torch::Tensor data,
        torch::Tensor scl, torch::Tensor bt, torch::Tensor ctx, int64_t block_size,
        double scale, int64_t partition_size, int64_t max_context) {
    CK(q); CK(data); CK(scl); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    const int P = int((max_context + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    auto out = torch::empty_like(q);
    mla_decode_fp8_v<false, true><<<dim3(H, B, P), 32, 0, stream()>>>(bp(q),
        data.data_ptr<uint8_t>(), scl.data_ptr<uint8_t>(), bt.data_ptr<int>(),
        ctx.data_ptr<int>(), nullptr, nullptr, 0, nullptr, tmp.data_ptr<float>(),
        ml.data_ptr<float>(), es.data_ptr<float>(), int(block_size), int(bt.size(1)),
        float(scale), H, P, int(partition_size));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}
// Native replacement for the DSA indexer's Triton metadata kernel.
static void py_indexer_metadata(torch::Tensor query_start_loc,
        torch::Tensor uncompressed_seq_lens, torch::Tensor cu_compressed_seq_lens,
        torch::Tensor row_start_cu, torch::Tensor token_to_seq,
        torch::Tensor cu_ks, torch::Tensor cu_ke, int64_t query_slice_start,
        int64_t query_slice_stop, int64_t dcp_rank, int64_t dcp_world,
        int64_t dcp_interleave, int64_t compress_ratio) {
    CK(query_start_loc); CK(uncompressed_seq_lens); CK(cu_compressed_seq_lens);
    CK(row_start_cu); CK(token_to_seq); CK(cu_ks); CK(cu_ke);
    const int num_reqs = (int)query_start_loc.size(0) - 1;
    if (num_reqs <= 0) return;
    indexer_metadata<<<num_reqs, 256, 0, stream()>>>(
        query_start_loc.data_ptr<int>(), uncompressed_seq_lens.data_ptr<int>(),
        cu_compressed_seq_lens.data_ptr<int>(), row_start_cu.data_ptr<int>(),
        token_to_seq.data_ptr<int>(), cu_ks.data_ptr<int>(), cu_ke.data_ptr<int>(),
        (int)query_slice_start, (int)query_slice_stop, (int)dcp_rank,
        (int)dcp_world, (int)dcp_interleave, (int)compress_ratio);
}

// Native replacement for vLLM's Triton _compute_slot_mapping_kernel.
static void py_compute_slot_mapping(torch::Tensor query_start_loc,
        torch::Tensor positions, torch::Tensor block_table,
        torch::Tensor slot_mapping, int64_t num_tokens, int64_t max_num_tokens,
        int64_t block_size, int64_t kv_cache_block_size,
        int64_t blocks_per_kv_block, int64_t cp_world, int64_t cp_rank,
        int64_t cp_interleave, int64_t pad_id) {
    CK(query_start_loc); CK(positions); CK(block_table); CK(slot_mapping);
    TORCH_CHECK(positions.scalar_type() == torch::kLong, "positions must be int64");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping int64");
    const int num_reqs = (int)query_start_loc.size(0) - 1;
    compute_slot_mapping<<<num_reqs + 1, 256, 0, stream()>>>(
        (long)num_tokens, (long)max_num_tokens, query_start_loc.data_ptr<int>(),
        reinterpret_cast<const long*>(positions.data_ptr()),
        block_table.data_ptr<int>(), (int)block_table.size(1), (int)block_size,
        reinterpret_cast<long*>(slot_mapping.data_ptr()),
        (int)kv_cache_block_size, (int)blocks_per_kv_block, (int)cp_world,
        (int)cp_rank, (int)cp_interleave, (long)pad_id);
}

// ---- V2 model-runner batch-prep (v2_batch_kernels.cuh) ----
#define CKD(x, d) do { CK(x); TORCH_CHECK((x).scalar_type() == (d), #x " must be " #d); } while (0)

static void py_apply_write(torch::Tensor out, int64_t row_stride,
        torch::Tensor indices, torch::Tensor starts, torch::Tensor contents,
        torch::Tensor cu_lens) {
    CK(out); CKD(indices, torch::kInt); CKD(starts, torch::kInt);
    CK(contents); CKD(cu_lens, torch::kInt);
    TORCH_CHECK(out.scalar_type() == contents.scalar_type(),
                "out/contents dtype mismatch");
    const int n = indices.numel();
    if (n == 0) return;
    const auto esize = out.element_size();
    if (esize == 4)
        apply_write<int><<<n, 256, 0, stream()>>>(
            reinterpret_cast<int*>(out.data_ptr()), (long)row_stride,
            indices.data_ptr<int>(), starts.data_ptr<int>(),
            reinterpret_cast<const int*>(contents.data_ptr()),
            cu_lens.data_ptr<int>());
    else if (esize == 8)
        apply_write<long><<<n, 256, 0, stream()>>>(
            reinterpret_cast<long*>(out.data_ptr()), (long)row_stride,
            indices.data_ptr<int>(), starts.data_ptr<int>(),
            reinterpret_cast<const long*>(contents.data_ptr()),
            cu_lens.data_ptr<int>());
    else
        TORCH_CHECK(false, "apply_write supports 4/8-byte elements");
}

static void py_apply_write_multi(torch::Tensor out_ptrs, torch::Tensor out_strides,
        torch::Tensor indices, torch::Tensor starts, torch::Tensor contents,
        torch::Tensor cu_lens, torch::Tensor group_ids) {
    CKD(out_ptrs, torch::kUInt64); CKD(out_strides, torch::kLong);
    CKD(indices, torch::kInt); CKD(starts, torch::kInt);
    CKD(contents, torch::kInt); CKD(cu_lens, torch::kInt);
    CKD(group_ids, torch::kInt);
    const int n = indices.numel();
    if (n == 0) return;
    apply_write_multi<<<n, 256, 0, stream()>>>(
        reinterpret_cast<const unsigned long long*>(out_ptrs.data_ptr()),
        out_strides.data_ptr<int64_t>(), indices.data_ptr<int>(),
        starts.data_ptr<int>(), contents.data_ptr<int>(),
        cu_lens.data_ptr<int>(), group_ids.data_ptr<int>());
}

static void py_prepare_pos_seq_lens(torch::Tensor pos, torch::Tensor seq_lens,
        torch::Tensor idx_mapping, torch::Tensor query_start_loc,
        torch::Tensor num_computed_tokens, int64_t max_num_reqs) {
    CKD(pos, torch::kLong); CKD(seq_lens, torch::kInt);
    CKD(idx_mapping, torch::kInt); CKD(query_start_loc, torch::kInt);
    CKD(num_computed_tokens, torch::kInt);
    const int num_reqs = idx_mapping.numel();
    prepare_pos_seq_lens<<<num_reqs + 1, 256, 0, stream()>>>(
        reinterpret_cast<long*>(pos.data_ptr()), seq_lens.data_ptr<int>(),
        idx_mapping.data_ptr<int>(), query_start_loc.data_ptr<int>(),
        num_computed_tokens.data_ptr<int>(), (int)max_num_reqs);
}

static void py_prepare_prefill_inputs(torch::Tensor input_ids,
        torch::Tensor next_prefill_tokens, torch::Tensor idx_mapping,
        torch::Tensor query_start_loc, torch::Tensor all_token_ids,
        int64_t all_token_ids_stride, torch::Tensor prefill_lens,
        torch::Tensor num_computed_tokens) {
    CKD(input_ids, torch::kInt); CKD(next_prefill_tokens, torch::kInt);
    CKD(idx_mapping, torch::kInt); CKD(query_start_loc, torch::kInt);
    CKD(all_token_ids, torch::kInt); CKD(prefill_lens, torch::kInt);
    CKD(num_computed_tokens, torch::kInt);
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    prepare_prefill_inputs<<<num_reqs, 256, 0, stream()>>>(
        input_ids.data_ptr<int>(), next_prefill_tokens.data_ptr<int>(),
        idx_mapping.data_ptr<int>(), query_start_loc.data_ptr<int>(),
        all_token_ids.data_ptr<int>(), (long)all_token_ids_stride,
        prefill_lens.data_ptr<int>(), num_computed_tokens.data_ptr<int>());
}

static void py_combine_sampled_and_draft_tokens(torch::Tensor input_ids,
        torch::Tensor idx_mapping, torch::Tensor last_sampled_tokens,
        torch::Tensor query_start_loc, torch::Tensor seq_lens,
        torch::Tensor prefill_len, torch::Tensor draft_tokens,
        int64_t draft_tokens_stride, torch::Tensor cu_num_logits,
        torch::Tensor logits_indices, int64_t num_new_sampled_tokens) {
    CKD(input_ids, torch::kInt); CKD(idx_mapping, torch::kInt);
    CKD(last_sampled_tokens, torch::kLong); CKD(query_start_loc, torch::kInt);
    CKD(seq_lens, torch::kInt); CKD(prefill_len, torch::kInt);
    CKD(draft_tokens, torch::kLong); CKD(cu_num_logits, torch::kInt);
    CKD(logits_indices, torch::kLong);
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    combine_sampled_and_draft_tokens<<<num_reqs, 256, 0, stream()>>>(
        input_ids.data_ptr<int>(), idx_mapping.data_ptr<int>(),
        last_sampled_tokens.data_ptr<int64_t>(), query_start_loc.data_ptr<int>(),
        seq_lens.data_ptr<int>(), prefill_len.data_ptr<int>(),
        draft_tokens.data_ptr<int64_t>(), (long)draft_tokens_stride,
        cu_num_logits.data_ptr<int>(),
        reinterpret_cast<long*>(logits_indices.data_ptr()),
        (int)num_new_sampled_tokens);
}

static void py_get_num_sampled_and_rejected(torch::Tensor num_sampled,
        torch::Tensor num_rejected, torch::Tensor seq_lens,
        torch::Tensor cu_num_logits, torch::Tensor idx_mapping,
        torch::Tensor prefill_len) {
    CKD(num_sampled, torch::kInt); CKD(num_rejected, torch::kInt);
    CKD(seq_lens, torch::kInt); CKD(cu_num_logits, torch::kInt);
    CKD(idx_mapping, torch::kInt); CKD(prefill_len, torch::kInt);
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    get_num_sampled_and_rejected<<<(num_reqs + 255) / 256, 256, 0, stream()>>>(
        num_sampled.data_ptr<int>(), num_rejected.data_ptr<int>(),
        seq_lens.data_ptr<int>(), cu_num_logits.data_ptr<int>(),
        idx_mapping.data_ptr<int>(), prefill_len.data_ptr<int>(), num_reqs);
}

static void py_post_update(torch::Tensor idx_mapping,
        torch::Tensor num_computed_tokens, torch::Tensor last_sampled_tokens,
        c10::optional<torch::Tensor> output_bin_counts,
        int64_t output_bin_counts_stride, torch::Tensor sampled_tokens,
        int64_t sampled_tokens_stride, torch::Tensor num_sampled,
        torch::Tensor num_rejected, c10::optional<torch::Tensor> query_start_loc,
        torch::Tensor all_token_ids, int64_t all_token_ids_stride,
        torch::Tensor total_len) {
    CKD(idx_mapping, torch::kInt); CKD(num_computed_tokens, torch::kInt);
    CKD(last_sampled_tokens, torch::kLong); CKD(sampled_tokens, torch::kLong);
    CKD(num_sampled, torch::kInt); CKD(num_rejected, torch::kInt);
    CKD(all_token_ids, torch::kInt); CKD(total_len, torch::kInt);
    int* bin_counts = nullptr;
    if (output_bin_counts) {
        torch::Tensor& bc = *output_bin_counts;
        CKD(bc, torch::kInt);
        bin_counts = bc.data_ptr<int>();
    }
    const int* qsl = nullptr;
    if (query_start_loc) {
        torch::Tensor& q = *query_start_loc;
        CKD(q, torch::kInt);
        qsl = q.data_ptr<int>();
    }
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    post_update<<<(num_reqs + 255) / 256, 256, 0, stream()>>>(
        idx_mapping.data_ptr<int>(), num_computed_tokens.data_ptr<int>(),
        last_sampled_tokens.data_ptr<int64_t>(), bin_counts,
        (long)output_bin_counts_stride, sampled_tokens.data_ptr<int64_t>(),
        (long)sampled_tokens_stride, num_sampled.data_ptr<int>(),
        num_rejected.data_ptr<int>(), qsl, all_token_ids.data_ptr<int>(),
        (long)all_token_ids_stride, total_len.data_ptr<int>(), num_reqs);
}

static void py_post_update_num_computed_tokens(torch::Tensor idx_mapping,
        torch::Tensor num_computed_tokens, torch::Tensor query_start_loc) {
    CKD(idx_mapping, torch::kInt); CKD(num_computed_tokens, torch::kInt);
    CKD(query_start_loc, torch::kInt);
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    post_update_num_computed_tokens<<<(num_reqs + 255) / 256, 256, 0, stream()>>>(
        idx_mapping.data_ptr<int>(), num_computed_tokens.data_ptr<int>(),
        query_start_loc.data_ptr<int>(), num_reqs);
}

static void py_expand_idx_mapping(torch::Tensor idx_mapping,
        torch::Tensor expanded_idx_mapping, torch::Tensor expanded_local_pos,
        torch::Tensor cu_num_logits) {
    CKD(idx_mapping, torch::kInt); CKD(expanded_idx_mapping, torch::kInt);
    CKD(expanded_local_pos, torch::kInt); CKD(cu_num_logits, torch::kInt);
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    expand_idx_mapping<<<num_reqs, 256, 0, stream()>>>(
        idx_mapping.data_ptr<int>(), expanded_idx_mapping.data_ptr<int>(),
        expanded_local_pos.data_ptr<int>(), cu_num_logits.data_ptr<int>());
}

static void py_gather_block_tables(torch::Tensor idx_mapping,
        torch::Tensor src_ptrs, torch::Tensor dst_ptrs, torch::Tensor strides,
        torch::Tensor num_blocks, int64_t num_blocks_stride, int64_t num_reqs,
        int64_t num_reqs_padded) {
    CKD(idx_mapping, torch::kInt); CKD(src_ptrs, torch::kUInt64);
    CKD(dst_ptrs, torch::kUInt64); CKD(strides, torch::kLong);
    CKD(num_blocks, torch::kInt);
    const int num_groups = src_ptrs.numel();
    if (num_groups == 0 || num_reqs_padded == 0) return;
    dim3 grid((unsigned)num_groups, (unsigned)num_reqs_padded);
    gather_block_tables<<<grid, 256, 0, stream()>>>(
        idx_mapping.data_ptr<int>(),
        reinterpret_cast<const unsigned long long*>(src_ptrs.data_ptr()),
        reinterpret_cast<const unsigned long long*>(dst_ptrs.data_ptr()),
        strides.data_ptr<int64_t>(), num_blocks.data_ptr<int>(),
        (long)num_blocks_stride, (int)num_reqs);
}

static void py_compute_slot_mappings(torch::Tensor idx_mapping,
        torch::Tensor query_start_loc, torch::Tensor pos,
        torch::Tensor block_table_ptrs, torch::Tensor block_table_strides,
        torch::Tensor block_sizes, torch::Tensor slot_mappings,
        int64_t slot_mappings_stride, int64_t max_num_tokens, int64_t cp_rank,
        int64_t cp_size, int64_t cp_interleave, int64_t pad_id) {
    CKD(idx_mapping, torch::kInt); CKD(query_start_loc, torch::kInt);
    CKD(pos, torch::kLong); CKD(block_table_ptrs, torch::kUInt64);
    CKD(block_table_strides, torch::kLong); CKD(block_sizes, torch::kInt);
    CKD(slot_mappings, torch::kLong);
    const int num_groups = block_table_ptrs.numel();
    const int num_reqs = idx_mapping.numel();
    if (num_groups == 0) return;
    dim3 grid((unsigned)num_groups, (unsigned)(num_reqs + 1));
    compute_slot_mappings<<<grid, 256, 0, stream()>>>(
        (long)max_num_tokens, idx_mapping.data_ptr<int>(),
        query_start_loc.data_ptr<int>(),
        reinterpret_cast<const long*>(pos.data_ptr()),
        reinterpret_cast<const unsigned long long*>(block_table_ptrs.data_ptr()),
        block_table_strides.data_ptr<int64_t>(), block_sizes.data_ptr<int>(),
        reinterpret_cast<long*>(slot_mappings.data_ptr()),
        (long)slot_mappings_stride, (int)cp_rank, (int)cp_size,
        (int)cp_interleave, (long)pad_id);
}

static void py_prepare_uniform_decode(torch::Tensor seq_lens,
        torch::Tensor decode_seq_lens, torch::Tensor block_table,
        int64_t block_table_stride, torch::Tensor expanded_block_table,
        int64_t expanded_bt_stride, torch::Tensor decode_lens,
        int64_t max_decode_len, int64_t num_decode_tokens) {
    CKD(seq_lens, torch::kInt); CKD(decode_seq_lens, torch::kInt);
    CKD(block_table, torch::kInt); CKD(expanded_block_table, torch::kInt);
    CKD(decode_lens, torch::kInt);
    if (num_decode_tokens == 0) return;
    prepare_uniform_decode<<<(unsigned)num_decode_tokens, 256, 0, stream()>>>(
        seq_lens.data_ptr<int>(), decode_seq_lens.data_ptr<int>(),
        block_table.data_ptr<int>(), (long)block_table_stride,
        expanded_block_table.data_ptr<int>(), (long)expanded_bt_stride,
        decode_lens.data_ptr<int>(), (int)max_decode_len);
}

static void py_prepare_dflash_inputs(torch::Tensor out_input_ids,
        torch::Tensor out_query_positions, torch::Tensor out_query_start_loc,
        torch::Tensor out_seq_lens, torch::Tensor out_query_slot_mapping,
        torch::Tensor out_context_positions,
        torch::Tensor out_context_slot_mapping, torch::Tensor out_sample_indices,
        torch::Tensor out_sample_pos, torch::Tensor out_sample_idx_mapping,
        torch::Tensor target_positions, torch::Tensor target_query_start_loc,
        torch::Tensor idx_mapping, torch::Tensor last_sampled,
        torch::Tensor next_prefill_tokens, torch::Tensor num_sampled,
        torch::Tensor num_rejected, torch::Tensor block_table,
        int64_t block_table_stride, int64_t parallel_drafting_token_id,
        int64_t block_size, int64_t num_query_per_req,
        int64_t num_speculative_steps, int64_t max_num_reqs,
        int64_t max_num_tokens, int64_t max_model_len, bool sample_from_anchor,
        int64_t pad_slot_id, int64_t num_reqs, int64_t max_tokens_per_req) {
    CKD(out_input_ids, torch::kInt); CKD(out_query_positions, torch::kLong);
    CKD(out_query_start_loc, torch::kInt); CKD(out_seq_lens, torch::kInt);
    CKD(out_query_slot_mapping, torch::kLong);
    CKD(out_context_positions, torch::kLong);
    CKD(out_context_slot_mapping, torch::kLong);
    CKD(out_sample_indices, torch::kLong); CKD(out_sample_pos, torch::kLong);
    CKD(out_sample_idx_mapping, torch::kInt);
    CKD(target_positions, torch::kLong);
    CKD(target_query_start_loc, torch::kInt); CKD(idx_mapping, torch::kInt);
    CKD(last_sampled, torch::kLong); CKD(next_prefill_tokens, torch::kInt);
    CKD(num_sampled, torch::kInt); CKD(num_rejected, torch::kInt);
    CKD(block_table, torch::kInt);
    if (num_reqs == 0) return;
    prepare_dflash_inputs<<<(unsigned)num_reqs, 256, 0, stream()>>>(
        out_input_ids.data_ptr<int>(),
        reinterpret_cast<long*>(out_query_positions.data_ptr()),
        out_query_start_loc.data_ptr<int>(), out_seq_lens.data_ptr<int>(),
        reinterpret_cast<long*>(out_query_slot_mapping.data_ptr()),
        reinterpret_cast<long*>(out_context_positions.data_ptr()),
        reinterpret_cast<long*>(out_context_slot_mapping.data_ptr()),
        reinterpret_cast<long*>(out_sample_indices.data_ptr()),
        reinterpret_cast<long*>(out_sample_pos.data_ptr()),
        out_sample_idx_mapping.data_ptr<int>(),
        reinterpret_cast<const long*>(target_positions.data_ptr()),
        target_query_start_loc.data_ptr<int>(), idx_mapping.data_ptr<int>(),
        last_sampled.data_ptr<int64_t>(), next_prefill_tokens.data_ptr<int>(),
        num_sampled.data_ptr<int>(), num_rejected.data_ptr<int>(),
        block_table.data_ptr<int>(), (long)block_table_stride,
        (int)parallel_drafting_token_id, (int)block_size,
        (int)num_query_per_req, (int)num_speculative_steps, (int)max_num_reqs,
        (long)max_num_tokens, (long)max_model_len, sample_from_anchor ? 1 : 0,
        (long)pad_slot_id, (int)max_tokens_per_req);
}

// DSA indexer decode logits over the paged fp8 K cache. Replaces
// fp8_paged_mqa_logits_torch, which loops the batch in Python and calls
// .item() -- a host sync that CUDA graph capture rejects outright.
static torch::Tensor py_fp8_paged_mqa_logits(torch::Tensor q,
        torch::Tensor kv_cache, torch::Tensor weights,
        torch::Tensor context_lens, torch::Tensor block_tables,
        int64_t max_model_len, int64_t token_shard_rank,
        int64_t token_shard_world_size, int64_t trivial_topk) {
    CK(q); CK(weights); CK(context_lens); CK(block_tables);
    TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8, "kv_cache must be uint8");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens int32");
    TORCH_CHECK(block_tables.scalar_type() == torch::kInt, "block_tables int32");
    constexpr int D = 128;
    const int B = (int)q.size(0), H = (int)q.size(2);
    TORCH_CHECK(q.scalar_type() == torch::kFloat8_e4m3fn,
                "indexer Q must be FP8 E4M3FN");
    TORCH_CHECK(weights.scalar_type() == torch::kFloat,
                "quantized indexer weights must be FP32");
    TORCH_CHECK(q.size(3) == D, "indexer head_dim must be ", D);
    TORCH_CHECK(q.size(1) == 1, "paged logits path is next_n==1 (decode)");
    TORCH_CHECK(H <= 64, "H exceeds heads per block");
    TORCH_CHECK(token_shard_world_size > 0 && token_shard_rank >= 0 &&
                    token_shard_rank < token_shard_world_size,
                "invalid indexer token shard");
    TORCH_CHECK(kv_cache.dim() == 3 || kv_cache.dim() == 4,
                "kv_cache must be [num_blocks, block_size, bytes] or "
                "[num_blocks, block_size, 1, bytes]");
    const int block_size = (int)kv_cache.size(1);
    const long kv_block_stride = (long)kv_cache.stride(0);
    TORCH_CHECK(kv_block_stride >= (long)block_size * (D + 4),
                "kv_cache block stride is too small for fp8 indexer cache");

    // TP ranks score disjoint interleaved token ranges. Interleaving balances
    // every prefix rather than leaving short contexts entirely on rank zero.
    // Local top-k candidates are sufficient for an exact global top-k merge,
    // while both this kernel and persistent_topk process only 1/TP of the
    // 1M-token ceiling.
    const int local_max_len = int(
        (max_model_len + token_shard_world_size - 1) / token_shard_world_size);
    auto out = torch::empty({B, (long)local_max_len},
                            q.options().dtype(torch::kFloat));
    static const int persistent_ctas = [] {
        const char* value = std::getenv("VLLM_DSV4_INDEXER_CTAS");
        return value == nullptr ? 216 : std::max(1, std::atoi(value));
    }();
    static const int short_context_threshold = [] {
        const char* value =
            std::getenv("VLLM_DSV4_INDEXER_SHORT_CONTEXT_THRESHOLD");
        return value == nullptr ? 4096 : std::max(0, std::atoi(value));
    }();
    static const int head_shard_schedule = [] {
        const char* value = std::getenv("VLLM_DSV4_INDEXER_HEAD_SHARD_SCHEDULE");
        return value == nullptr ? 3 : std::max(0, std::atoi(value));
    }();
    const int logical_nt = H > 32
        ? 32
        : short_context_threshold > 0 ? 8 : 16;
    const int max_tiles = int((local_max_len + logical_nt - 1) / logical_nt);
    const dim3 grid((unsigned)std::min(max_tiles, persistent_ctas),
                    (unsigned)B);
    auto launch = [&]<int NT, int HEAD_WARPS, int TOKEN_GROUPS>() {
        const size_t smem =
            tms::indexer_paged_mqa_logits_smem<D, NT, HEAD_WARPS, TOKEN_GROUPS>();
        auto kern =
            tms::indexer_paged_mqa_logits<D, NT, HEAD_WARPS, TOKEN_GROUPS>;
        constexpr int threads =
            32 * HEAD_WARPS * TOKEN_GROUPS > 128
                ? 32 * HEAD_WARPS * TOKEN_GROUPS
                : 128;
        if (smem > 48 * 1024)
            cudaFuncSetAttribute(
                kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        kern<<<grid, threads, smem, stream()>>>(
            reinterpret_cast<const uint8_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
            weights.data_ptr<float>(), context_lens.data_ptr<int>(),
            block_tables.data_ptr<int>(), out.data_ptr<float>(),
            H, block_size, kv_block_stride,
            (int)block_tables.size(1), local_max_len,
            int(token_shard_rank), int(token_shard_world_size),
            short_context_threshold, int(trivial_topk));
    };
    if (H <= 16) {
        if (head_shard_schedule == 0)
            launch.template operator()<16, 1, 1>();
        else if (head_shard_schedule == 2)
            launch.template operator()<32, 1, 4>();
        else
            launch.template operator()<16, 1, 2>();
    } else if (H <= 32) {
        if (head_shard_schedule == 0)
            launch.template operator()<16, 2, 1>();
        else if (head_shard_schedule == 1)
            launch.template operator()<16, 2, 2>();
        else if (head_shard_schedule == 3)
            launch.template operator()<32, 2, 4>();
        else
            launch.template operator()<32, 2, 2>();
    } else {
        launch.template operator()<32, 4, 4>();
    }
    return out;
}

__global__ void indexer_pack_tp_candidates_k(
        const float* __restrict__ logits, const int* __restrict__ indices,
        float* __restrict__ packed, int rows, int k, int logit_stride,
        int token_shard_rank, int token_shard_world) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * k;
    if (i >= total) return;
    const int row = i / k;
    const int local_id = indices[i];
    const bool valid = local_id >= 0 && local_id < logit_stride;
    packed[2 * i] = valid ? logits[(size_t)row * logit_stride + local_id]
                          : -3.4028234663852886e38f;
    packed[2 * i + 1] = valid
        ? float(local_id * token_shard_world + token_shard_rank)
        : -1.0f;
}

static torch::Tensor py_indexer_pack_tp_candidates(
        torch::Tensor logits, torch::Tensor indices, int64_t token_shard_rank,
        int64_t token_shard_world_size) {
    CKD(logits, torch::kFloat); CKD(indices, torch::kInt);
    TORCH_CHECK(logits.dim() == 2 && indices.dim() == 2 &&
                    logits.size(0) == indices.size(0),
                "indexer TP candidate shape mismatch");
    const int rows = int(indices.size(0)), k = int(indices.size(1));
    auto packed = torch::empty({rows, k, 2}, logits.options());
    const int total = rows * k;
    if (total > 0)
        indexer_pack_tp_candidates_k<<<(total + 255) / 256, 256, 0, stream()>>>(
            logits.data_ptr<float>(), indices.data_ptr<int>(),
            packed.data_ptr<float>(), rows, k, int(logits.size(1)),
            int(token_shard_rank), int(token_shard_world_size));
    return packed;
}

__global__ void indexer_unpack_tp_scores_k(
        const float* __restrict__ packed, float* __restrict__ scores,
        int* __restrict__ lengths, int rows, int candidates) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * candidates;
    if (i < total) scores[i] = packed[2 * i];
    if (i < rows) lengths[i] = candidates;
}

static std::tuple<torch::Tensor, torch::Tensor> py_indexer_unpack_tp_scores(
        torch::Tensor packed) {
    CKD(packed, torch::kFloat);
    TORCH_CHECK(packed.dim() == 3 && packed.size(2) == 2,
                "indexer gathered candidates must be [rows, candidates, 2]");
    const int rows = int(packed.size(0)), candidates = int(packed.size(1));
    auto scores = torch::empty({rows, candidates}, packed.options());
    auto lengths = torch::empty(
        {rows}, packed.options().dtype(torch::kInt));
    const int work = std::max(rows * candidates, rows);
    if (work > 0)
        indexer_unpack_tp_scores_k<<<(work + 255) / 256, 256, 0, stream()>>>(
            packed.data_ptr<float>(), scores.data_ptr<float>(),
            lengths.data_ptr<int>(), rows, candidates);
    return std::make_tuple(scores, lengths);
}

__global__ void indexer_resolve_tp_candidates_k(
        const float* __restrict__ packed, const int* __restrict__ positions,
        int* __restrict__ output, int rows, int k, int candidates) {
    const int i = int(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * k;
    if (i >= total) return;
    const int row = i / k;
    const int pos = positions[i];
    output[i] = pos >= 0 && pos < candidates
        ? __float2int_rn(packed[((size_t)row * candidates + pos) * 2 + 1])
        : -1;
}

static void py_indexer_resolve_tp_candidates(
        torch::Tensor packed, torch::Tensor positions, torch::Tensor output) {
    CKD(packed, torch::kFloat); CKD(positions, torch::kInt);
    CKD(output, torch::kInt);
    TORCH_CHECK(packed.dim() == 3 && packed.size(2) == 2 &&
                    positions.dim() == 2 && output.sizes() == positions.sizes() &&
                    packed.size(0) == positions.size(0),
                "indexer TP resolve shape mismatch");
    const int rows = int(positions.size(0)), k = int(positions.size(1));
    const int total = rows * k;
    if (total > 0)
        indexer_resolve_tp_candidates_k<<<(total + 255) / 256, 256, 0, stream()>>>(
            packed.data_ptr<float>(), positions.data_ptr<int>(),
            output.data_ptr<int>(), rows, k, int(packed.size(1)));
}

// Non-paged (prefill) indexer logits: the tensor-core kernel from
// indexer_logits_mma.cuh, which was validated standalone but never bound.
static torch::Tensor py_fp8_mqa_logits(torch::Tensor q, torch::Tensor k,
        torch::Tensor kscale, torch::Tensor weights, torch::Tensor ks,
        torch::Tensor ke) {
    CK(q); CK(k); CK(kscale); CK(weights); CK(ks); CK(ke);
    constexpr int D = 128, NT = 64, NWARP = 4;
    const int M = (int)q.size(0), H = (int)q.size(1), N = (int)k.size(0);
    TORCH_CHECK(q.size(2) == D, "indexer head_dim must be ", D);
    TORCH_CHECK(H <= 16 * NWARP, "H exceeds heads per block");
    auto out = torch::empty({M, N}, q.options().dtype(torch::kFloat));
    const size_t smem = tms::indexer_mqa_logits_mma_smem<D, NT, NWARP>();
    auto kern = tms::indexer_mqa_logits_mma<D, NT, NWARP>;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    dim3 grid((unsigned)((N + NT - 1) / NT), (unsigned)M);
    kern<<<grid, 32 * NWARP, smem, stream()>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        kscale.data_ptr<float>(), weights.data_ptr<float>(),
        ks.data_ptr<int>(), ke.data_ptr<int>(), out.data_ptr<float>(), M, N, H);
    return out;
}

// GLM-5.2-Vision, bf16 cache. NFP8=0 means every element is read as bf16, so a
// slot is 576 bf16 = 1152 B. Ampere (sm80) has no native fp8e4nv, so vLLM cannot
// store an fp8 KV cache there -- this is the geometry that actually runs on A100.
static torch::Tensor py_mla_decode_bf16_sparse_glm(torch::Tensor q, torch::Tensor kv,
        torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale, int64_t partition_size) {
    CK(q); CK(kv); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "GLM MLA expects q width 576, got ", q.size(2));
    const int max_topk = indices.size(1);
    auto out = torch::empty({B, H, 512}, q.options());
    const uint8_t* kvp = reinterpret_cast<const uint8_t*>(kv.data_ptr());
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false, 576, 512, 0, 0><<<dim3(H, B), 32, 0, stream()>>>(
            bp(q), kvp, nullptr, bt.data_ptr<int>(), nullptr, indices.data_ptr<int>(),
            topk_length.data_ptr<int>(), max_topk, bpm(out), nullptr, nullptr, nullptr,
            int(block_size), int(bt.size(1)), float(scale), H, 1, 0, 1.0f);
        return out;
    }
    // Partitioned like the fp8 GLM path and the NoPE bf16 path: the
    // unpartitioned launch measured glm52-q2k-8 (bf16 KV, TP8, 8 local
    // heads) at 9.6 tok/s c1 on 2026-09-03.
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true, 576, 512, 0, 0><<<dim3(H, B, P), 32, 0, stream()>>>(
        bp(q), kvp, nullptr, bt.data_ptr<int>(), nullptr, indices.data_ptr<int>(),
        topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size), 1.0f);
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}

// GLM-5.3-Flash (glm5_next) NoPE MLA, bf16 cache. There is no rope segment:
// q and each slot are 512 bf16 latents (1024 B/slot) and the whole width both
// scores and accumulates, i.e. the same template at QW = VW = 512.
static torch::Tensor py_mla_decode_bf16_sparse_nope(torch::Tensor q, torch::Tensor kv,
        torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale, int64_t partition_size) {
    CK(q); CK(kv); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 512, "NoPE MLA expects q width 512, got ", q.size(2));
    const int max_topk = indices.size(1);
    auto out = torch::empty({B, H, 512}, q.options());
    const uint8_t* kvp = reinterpret_cast<const uint8_t*>(kv.data_ptr());
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false, 512, 512, 0, 0><<<dim3(H, B), 32, 0, stream()>>>(
            bp(q), kvp, nullptr, bt.data_ptr<int>(), nullptr, indices.data_ptr<int>(),
            topk_length.data_ptr<int>(), max_topk, bpm(out), nullptr, nullptr, nullptr,
            int(block_size), int(bt.size(1)), float(scale), H, 1, 0, 1.0f);
        return out;
    }
    // Partitioned, as the GLM fp8 path above: the unpartitioned launch is one
    // warp per (head, token) walking the selected list serially, a dependent
    // index -> block table -> 1 KB row chain per token. At TP8 (8 local heads,
    // B=1) that is 8 warps on 108 SMs: measured ~75 us per selected token
    // per decode step across the 11 DSA layers, 155 ms/token at the 2048
    // top-k (2026-09-03). P partitions over blockIdx.z, exact online-softmax
    // merge in the reduce.
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true, 512, 512, 0, 0><<<dim3(H, B, P), 32, 0, stream()>>>(
        bp(q), kvp, nullptr, bt.data_ptr<int>(), nullptr, indices.data_ptr<int>(),
        topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size), 1.0f);
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}

// GLM-5.2-Vision geometry: q/cache slot are 576 fp8 elements, the value is the
// leading 512, and the cache carries one per-tensor kv_scale (vLLM's fp8 MLA
// layout) rather than per-64 e8 exponents. `indices` are request-local logical
// token positions resolved through `bt`, which is exactly what vLLM's
// topk_indices_buffer already holds -- no global-index conversion needed.
static torch::Tensor py_mla_decode_fp8_sparse_glm(torch::Tensor q, torch::Tensor data,
        torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale, double kv_scale,
        int64_t partition_size) {
    CK(q); CK(data); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "GLM MLA expects q width 576, got ", q.size(2));
    const int max_topk = indices.size(1);
    auto out = torch::empty({B, H, 512}, q.options());
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false, 576, 512, 576, 1><<<dim3(H, B), 32, 0, stream()>>>(
            bp(q), data.data_ptr<uint8_t>(), nullptr, bt.data_ptr<int>(), nullptr,
            indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, bpm(out),
            nullptr, nullptr, nullptr, int(block_size), int(bt.size(1)), float(scale), H, 1, 0,
            float(kv_scale));
        return out;
    }
    // Partitioned: the unpartitioned launch is one 32-thread warp per
    // (head, token) walking the whole index list serially -- 64 warps across
    // 108 SMs at B=1, which profiled at 48% of ALL decode GPU time. Splitting
    // the list over blockIdx.z multiplies the exposed parallelism by P and the
    // reduce merges the online-softmax partials exactly.
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true, 576, 512, 576, 1><<<dim3(H, B, P), 32, 0, stream()>>>(
        bp(q), data.data_ptr<uint8_t>(), nullptr, bt.data_ptr<int>(), nullptr,
        indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size),
        float(kv_scale));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}

// Split-q GLM sparse decode: q_nope head-major [H, B, 512] (the natural bmm
// output, pre-transpose) + q_pe [B, H, 64], replacing the per-layer
// torch.cat([ql_nope, q_pe]) with direct reads. Same values, same order.
static torch::Tensor py_mla_decode_fp8_sparse_glm_splitq(torch::Tensor q_nope,
        torch::Tensor q_pe, torch::Tensor data, torch::Tensor bt,
        torch::Tensor indices, torch::Tensor topk_length, int64_t block_size,
        double scale, double kv_scale, int64_t partition_size) {
    CK(q_nope); CK(data); CK(bt); CK(indices); CK(topk_length);
    const int H = q_nope.size(0), B = q_nope.size(1);
    TORCH_CHECK(q_nope.size(2) == 512 && q_pe.size(2) == 64,
                "splitq expects nope width 512 and pe width 64");
    TORCH_CHECK(q_pe.size(0) == B && q_pe.size(1) == H,
                "q_pe must be [B, H, 64] matching q_nope [H, B, 512]");
    // q_pe may be a strided split view: unit inner stride and a batch stride
    // that is H x the head stride let the kernel read it in place.
    TORCH_CHECK(q_pe.is_cuda() && q_pe.stride(2) == 1 &&
                q_pe.stride(0) == H * q_pe.stride(1),
                "q_pe must have unit inner stride and batch stride == H * head stride");
    const int pe_stride = int(q_pe.stride(1));
    const int max_topk = indices.size(1);
    auto out = torch::empty({B, H, 512}, q_nope.options());
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false, 576, 512, 576, 1><<<dim3(H, B), 32, 0, stream()>>>(
            bp(q_nope), data.data_ptr<uint8_t>(), nullptr, bt.data_ptr<int>(), nullptr,
            indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, bpm(out),
            nullptr, nullptr, nullptr, int(block_size), int(bt.size(1)), float(scale), H,
            1, 0, float(kv_scale), bp(q_pe), pe_stride);
        return out;
    }
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q_nope.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true, 576, 512, 576, 1><<<dim3(H, B, P), 32, 0, stream()>>>(
        bp(q_nope), data.data_ptr<uint8_t>(), nullptr, bt.data_ptr<int>(), nullptr,
        indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size),
        float(kv_scale), bp(q_pe), pe_stride);
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}

// out[t, h] = sum_k w[t, k] * x[t, k, h], float accumulate, one rounding to
// bf16. Replaces the out.mul_(topk_weights) + moe_sum pair (one launch fewer
// and one fewer bf16 rounding; tolerance vs the pair is <= K ulp).
__global__ void moe_weighted_sum_k(const __nv_bfloat16* __restrict__ x,
                                   const float* __restrict__ w,
                                   __nv_bfloat16* __restrict__ out,
                                   int K, int Hdim) {
    const int t = blockIdx.x;
    for (int h = blockIdx.y * blockDim.x + threadIdx.x; h < Hdim;
         h += gridDim.y * blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k)
            acc += w[t * K + k] * float(x[((long)t * K + k) * Hdim + h]);
        out[(long)t * Hdim + h] = __nv_bfloat16(acc);
    }
}

static void py_moe_weighted_sum(torch::Tensor x, torch::Tensor w, torch::Tensor out) {
    CK(x); CK(w); CK(out);
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16 && out.scalar_type() == torch::kBFloat16
                && w.scalar_type() == torch::kFloat, "moe_weighted_sum: bf16 x/out, f32 w");
    const int T = x.size(0), K = x.size(1), Hdim = x.size(2);
    TORCH_CHECK(w.size(0) == T && w.size(1) == K && out.size(0) == T && out.size(1) == Hdim,
                "moe_weighted_sum shape mismatch");
    if (T == 0) return;
    const int hb = std::min(24, (Hdim + 255) / 256);
    moe_weighted_sum_k<<<dim3(T, hb), 256, 0, stream()>>>(
        bp(x), w.data_ptr<float>(), bpm(out), K, Hdim);
}

// Effective top-k length per row (last valid index + 1) in one launch.
static torch::Tensor py_sparse_topk_tlen(torch::Tensor indices) {
    CKD(indices, torch::kInt);
    const int rows = indices.size(0), topk = indices.size(1);
    auto tlen = torch::empty({rows}, indices.options());
    if (rows > 0)
        sparse_topk_tlen<<<rows, 256, 0, stream()>>>(
            indices.data_ptr<int>(), tlen.data_ptr<int>(), topk);
    return tlen;
}

static int dsv4_mla_persistent_partitions();

static int dsv4_partitions(int max_topk, int partition_size) {
    if (max_topk <= 0) return 0;
    if (partition_size <= 0) return 1;
    // The persistent kernel balances the request's actual sparse length over
    // its logical partitions. More logical partitions than resident warps only
    // enlarge the fp32 partial workspace and make each resident warp loop over
    // additional empty partitions. This is especially costly for C128A, whose
    // metadata width is padded for the 1M-token ceiling: batch-32 warmup used
    // to request 17.5 GiB of partials for a handful of valid indices.
    return std::min(
        int((max_topk + partition_size - 1) / partition_size),
        dsv4_mla_persistent_partitions());
}

static int dsv4_mla_persistent_partitions() {
    static const int partitions = [] {
        const char* value = std::getenv("VLLM_DSV4_MLA_PERSISTENT_PARTITIONS");
        return value == nullptr ? 64 : std::max(1, std::atoi(value));
    }();
    return partitions;
}

static bool dsv4_mla_adaptive_partitions_enabled() {
    static const bool enabled = [] {
        const char* value = std::getenv("VLLM_DSV4_MLA_ADAPTIVE_PARTITIONS");
        return value == nullptr || value[0] != '0';
    }();
    return enabled;
}

static void launch_mla_decode_fp8_sparse_dsv4_merged(
        torch::Tensor q,
        torch::Tensor main_cache, torch::Tensor main_bt,
        torch::Tensor main_indices, torch::Tensor main_topk_length,
        bool main_indices_are_slots, int main_block_size, int main_partitions,
        torch::Tensor extra_cache, torch::Tensor extra_bt,
        torch::Tensor extra_indices, torch::Tensor extra_topk_length,
        bool extra_indices_are_slots, int extra_block_size,
        int extra_partitions, double scale, int partition_size,
        float* tmp, float* ml, float* es) {
    const int B = q.size(0), H = q.size(1);
    const int main_launched = std::min(
        main_partitions, dsv4_mla_persistent_partitions());
    const int extra_launched = std::min(
        extra_partitions, dsv4_mla_persistent_partitions());
    const int launched = main_launched + extra_launched;
    if (launched <= 0) return;

    // Page format follows the cache dtype: uint8 = fp8_ds_mla packed pages
    // (576B data + 8B UE8M0 scales); bf16 = plain 512-element bf16 rows
    // (1024B, no scale plane -- the NFP8=0 instantiation never reads it).
    const bool bf16_pages = main_cache.scalar_type() == torch::kBFloat16;
    constexpr int SCALE_SLOT_STRIDE_BYTES = 8;
    const int main_page_stride_bytes =
        int(main_cache.stride(0) * main_cache.element_size());
    const int extra_page_stride_bytes =
        int(extra_cache.stride(0) * extra_cache.element_size());
    const int main_scale_block_offset_bytes =
        bf16_pages ? 0 : main_block_size * 576;
    const int extra_scale_block_offset_bytes =
        bf16_pages ? 0 : extra_block_size * 576;
    const int total_partitions = main_partitions + extra_partitions;
    const uint8_t* main_data =
        reinterpret_cast<const uint8_t*>(main_cache.data_ptr());
    const uint8_t* extra_data =
        reinterpret_cast<const uint8_t*>(extra_cache.data_ptr());
    #define QC_DSV4_MERGED_ARGS \
        bp(q), main_data, main_data, \
        main_bt.numel() > 0 ? main_bt.data_ptr<int>() : nullptr, nullptr, \
        main_indices.data_ptr<int>(), main_topk_length.data_ptr<int>(), \
        int(main_indices.size(1)), nullptr, tmp, ml, es, main_block_size, \
        main_bt.numel() > 0 ? int(main_bt.size(1)) : 0, float(scale), H, \
        main_partitions, partition_size, 1.0f, nullptr, 0, \
        main_page_stride_bytes, main_scale_block_offset_bytes, \
        SCALE_SLOT_STRIDE_BYTES, total_partitions, 0, \
        main_indices_are_slots, true, \
        extra_data, extra_data, \
        extra_bt.numel() > 0 ? extra_bt.data_ptr<int>() : nullptr, nullptr, \
        extra_indices.data_ptr<int>(), extra_topk_length.data_ptr<int>(), \
        int(extra_indices.size(1)), extra_block_size, \
        extra_bt.numel() > 0 ? int(extra_bt.size(1)) : 0, \
        extra_partitions, partition_size, 1.0f, extra_page_stride_bytes, \
        extra_scale_block_offset_bytes, SCALE_SLOT_STRIDE_BYTES, \
        main_partitions, extra_indices_are_slots, main_launched, \
        extra_launched, int(main_cache.size(0)), int(extra_cache.size(0))
    if (bf16_pages) {
        mla_decode_fp8_v<true, true, 512, 512, 0, 0>
            <<<dim3(H, B, launched), 32, 0, stream()>>>(QC_DSV4_MERGED_ARGS);
    } else {
        mla_decode_fp8_v<true, true>
            <<<dim3(H, B, launched), 32, 0, stream()>>>(QC_DSV4_MERGED_ARGS);
    }
    #undef QC_DSV4_MERGED_ARGS
}

static torch::Tensor py_mla_decode_fp8_sparse_dsv4(
        torch::Tensor q,
        torch::Tensor main_cache, torch::Tensor main_bt,
        torch::Tensor main_indices, torch::Tensor main_topk_length,
        bool main_indices_are_slots,
        torch::Tensor extra_cache, torch::Tensor extra_bt,
        torch::Tensor extra_indices, torch::Tensor extra_topk_length,
        bool extra_indices_are_slots,
        c10::optional<torch::Tensor> sink,
        int64_t main_block_size, int64_t extra_block_size,
        double scale, int64_t partition_size) {
    CK(q); CK(main_indices); CK(main_topk_length);
    CK(extra_indices); CK(extra_topk_length);
    TORCH_CHECK(main_cache.is_cuda(), "main_cache must be CUDA");
    TORCH_CHECK(extra_cache.is_cuda(), "extra_cache must be CUDA");
    if (main_bt.numel() > 0) CK(main_bt);
    if (extra_bt.numel() > 0) CK(extra_bt);
    if (sink) {
        TORCH_CHECK(sink->is_cuda() && sink->is_contiguous(),
                    "sink must be contiguous CUDA");
    }
    TORCH_CHECK(q.scalar_type() == torch::kBFloat16, "q must be bf16");
    TORCH_CHECK(main_cache.scalar_type() == torch::kUInt8 ||
                    main_cache.scalar_type() == torch::kBFloat16,
                "main_cache must be uint8 (fp8_ds_mla pages) or bf16 "
                "(plain 512-element rows)");
    TORCH_CHECK(extra_cache.scalar_type() == main_cache.scalar_type(),
                "extra_cache dtype must match main_cache");
    if (main_cache.scalar_type() == torch::kBFloat16) {
        TORCH_CHECK(main_cache.stride(-1) == 1 && extra_cache.stride(-1) == 1,
                    "bf16 pages must have unit inner stride");
    }
    TORCH_CHECK(main_indices.scalar_type() == torch::kInt, "main_indices must be int32");
    TORCH_CHECK(extra_indices.scalar_type() == torch::kInt, "extra_indices must be int32");
    TORCH_CHECK(q.size(2) == 512, "DSV4 MLA expects q width 512, got ", q.size(2));
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(main_indices.size(0) == B && main_topk_length.size(0) == B,
                "main sparse metadata batch mismatch");
    TORCH_CHECK(extra_indices.size(0) == B && extra_topk_length.size(0) == B,
                "extra sparse metadata batch mismatch");
    if (sink) TORCH_CHECK(sink->size(0) >= H, "sink must have at least H entries");

    int ps = int(partition_size);
    if (dsv4_mla_adaptive_partitions_enabled() &&
        extra_indices.size(1) >= 1024 && ps > 2)
        ps = 2;
    int main_p = dsv4_partitions(int(main_indices.size(1)), ps);
    int extra_p = dsv4_partitions(int(extra_indices.size(1)), ps);
    // Split-K partials are fp32 [B, H, partitions, 512]. Keep full persistent
    // occupancy for latency-critical batch-one decode, but bound graph warmup
    // and high-concurrency decode to a fixed workspace. The kernel balances the
    // actual sparse length over however many partitions remain.
    constexpr int64_t MAX_PARTIAL_BYTES = int64_t(256) << 20;
    const int active_sources = (main_p > 0 ? 1 : 0) + (extra_p > 0 ? 1 : 0);
    if (active_sources > 0) {
        const int64_t bytes_per_partition =
            int64_t(B) * H * 512 * int64_t(sizeof(float));
        const int max_total_partitions = std::max(
            active_sources,
            int(MAX_PARTIAL_BYTES / std::max<int64_t>(bytes_per_partition, 1)));
        const int per_source_cap =
            std::max(1, max_total_partitions / active_sources);
        main_p = std::min(main_p, per_source_cap);
        extra_p = std::min(extra_p, per_source_cap);
    }
    const int total_p = main_p + extra_p;
    auto out = torch::empty_like(q);
    if (total_p == 0) {
        out.zero_();
        return out;
    }
    if (ps <= 0) {
        const int max_width = std::max(int(main_indices.size(1)), int(extra_indices.size(1)));
        ps = std::max(max_width, 1);
    }

    auto opts = q.options().dtype(torch::kFloat);
    // Sentinel diagnostic (VLLM_DSV4_MLA_TMP_SENTINEL=1): fill the value
    // partials with a finite marker to map exactly which dims the writer
    // leaves unwritten while still writing finite ml/es (the reducer then
    // consumes whatever is here - the NaN mechanism under investigation).
    static const bool tmp_sentinel =
        std::getenv("VLLM_DSV4_MLA_TMP_SENTINEL") != nullptr &&
        std::getenv("VLLM_DSV4_MLA_TMP_SENTINEL")[0] == '1';
    auto tmp = tmp_sentinel
                   ? torch::full({B, H, total_p, 512}, 12345.0f, opts)
                   : torch::empty({B, H, total_p, 512}, opts);
    // ml/es MUST be initialized, not torch::empty: the reducer's per-row
    // active count is min(partitions, length-in-TOKENS), so it reads every
    // partial slot for real rows, while the persistent writer can skip
    // partials it balanced away. An unwritten torch::empty slot then feeds
    // the merge with recycled allocator bytes: small stale floats are
    // harmless (exp -> 0), large ones silently corrupt the softmax merge,
    // and inf/NaN patterns surface as NaN attention output. -inf/0 makes an
    // unwritten slot an authentic empty partial, which the reducer's
    // !(mp > NEG_INF) guard already skips. (Root cause of the 2026-08
    // DSV4 NaN-seed/degeneration incident.)
    auto ml = torch::full({B, H, total_p},
                          -std::numeric_limits<float>::infinity(), opts);
    auto es = torch::zeros({B, H, total_p}, opts);
    launch_mla_decode_fp8_sparse_dsv4_merged(
        q, main_cache, main_bt, main_indices, main_topk_length,
        main_indices_are_slots, int(main_block_size), main_p,
        extra_cache, extra_bt, extra_indices, extra_topk_length,
        extra_indices_are_slots, int(extra_block_size), extra_p, scale, ps,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>());
    // Diagnostic (VLLM_DSV4_MLA_DEBUG_PARTIALS=1): census the split
    // partials between the writer and the reducer. Syncs; offline replay
    // and debug boots only.
    static const bool debug_partials =
        std::getenv("VLLM_DSV4_MLA_DEBUG_PARTIALS") != nullptr &&
        std::getenv("VLLM_DSV4_MLA_DEBUG_PARTIALS")[0] == '1';
    if (debug_partials) {
        auto ml_bad = (torch::isnan(ml) | torch::isinf(ml)).sum().item<int64_t>();
        auto es_bad = (torch::isnan(es) | torch::isinf(es)).sum().item<int64_t>();
        auto tmp_bad =
            (torch::isnan(tmp) | torch::isinf(tmp)).sum().item<int64_t>();
        int64_t sentinel_left = 0;
        if (tmp_sentinel) {
            auto written_parts = torch::isfinite(ml) &
                                 (ml > -std::numeric_limits<float>::infinity());
            auto sent = (tmp == 12345.0f);
            // sentinel retained inside partitions whose ml says "written"
            sentinel_left =
                (sent & written_parts.unsqueeze(-1)).sum().item<int64_t>();
        }
        if (ml_bad || es_bad || tmp_bad || sentinel_left) {
            auto ml_rows = (torch::isnan(ml) | torch::isinf(ml))
                               .any(-1).any(-1).nonzero().flatten();
            auto per_part = (torch::isnan(tmp) | torch::isinf(tmp))
                                .any(-1).sum({0, 1});
            printf("MLA_DEBUG_PARTIALS: B=%d H=%d main_p=%d extra_p=%d "
                   "ml_bad=%ld es_bad=%ld tmp_bad=%ld sentinel_in_written=%ld "
                   "first_bad_batch=%ld\n",
                   B, H, main_p, extra_p, long(ml_bad), long(es_bad),
                   long(tmp_bad), long(sentinel_left),
                   ml_rows.numel() ? long(ml_rows[0].item<int64_t>()) : -1);
            auto pp = per_part.cpu();
            printf("MLA_DEBUG_PARTIALS per-partition tmp bad-head counts:");
            for (int p = 0; p < total_p; p++)
                if (pp[p].item<int64_t>())
                    printf(" p%d=%ld", p, long(pp[p].item<int64_t>()));
            printf(" (main 0..%d, extra %d..%d)\n", main_p - 1, main_p,
                   total_p - 1);
            fflush(stdout);
        }
    }
    dsv4_attention_reduce_active_channels<__nv_bfloat16, 512>
        <<<dim3(H, B), 512, total_p * sizeof(float), stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        main_topk_length.data_ptr<int>(), extra_topk_length.data_ptr<int>(),
        sink ? sink->data_ptr<float>() : nullptr, bpm(out), H, main_p, extra_p,
        total_p);
    return out;
}

static torch::Tensor py_mla_decode_fp8_sparse(torch::Tensor q, torch::Tensor data,
        torch::Tensor scl, torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale, int64_t partition_size) {
    CK(q); CK(data); CK(scl); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    const int max_topk = indices.size(1);
    auto out = torch::empty_like(q);
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false><<<dim3(H, B), 32, 0, stream()>>>(bp(q),
            data.data_ptr<uint8_t>(), scl.data_ptr<uint8_t>(), bt.data_ptr<int>(), nullptr,
            indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, bpm(out),
            nullptr, nullptr, nullptr, int(block_size), int(bt.size(1)), float(scale), H, 1, 0);
        return out;
    }
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true><<<dim3(H, B, P), 32, 0, stream()>>>(bp(q),
        data.data_ptr<uint8_t>(), scl.data_ptr<uint8_t>(), bt.data_ptr<int>(), nullptr,
        indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}
static torch::Tensor py_paged_attention_gqa_staged(torch::Tensor q, torch::Tensor kc,
        torch::Tensor vc, torch::Tensor bt, torch::Tensor ctx, int64_t block_size,
        double scale, int64_t num_kv_heads) {
    CK(q); CK(kc); CK(vc); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    const int gs = H / int(num_kv_heads);
    auto out = torch::empty_like(q);
    dim3 grid{unsigned(num_kv_heads), unsigned(B)};
    #define LAUNCH(DD) paged_attention_gqa_staged<half, DD><<<grid, 32 * gs, 0, stream()>>>( \
        hp(q), hp(kc), hp(vc), bt.data_ptr<int>(), ctx.data_ptr<int>(), hpm(out), \
        int(block_size), int(bt.size(1)), float(scale), H, int(num_kv_heads))
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}
static torch::Tensor py_attn_window(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    double scale, int64_t window) {
    CK(q); CK(k); CK(v);
    const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    auto out = torch::empty_like(q);
    dim3 grid{unsigned(N), unsigned(B * H)};
    #define LAUNCH(DD) attn_window<half, DD><<<grid, 32, 0, stream()>>>( \
        hp(q), hp(k), hp(v), hpm(out), N, float(scale), int(window))
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}

// ---- TurboQuant KV cache (turboquant_kernels.cuh) ----
// kv_cache arrives as the backend's (num_blocks, block_size, Hk, slot) view of
// the (num_blocks, Hk, block_size, slot) allocation -- strided, so no CK.
#define CKTQ(x) TORCH_CHECK(x.is_cuda(), #x " must be CUDA")

static void py_turboquant_store_fp8(torch::Tensor key, torch::Tensor value,
        torch::Tensor kv_cache, torch::Tensor slot_mapping,
        int64_t num_kv_heads, int64_t kps, int64_t vqb) {
    CK(key); CK(value); CK(slot_mapping); CKTQ(kv_cache);
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.stride(3) == 1,
                "kv_cache must be uint8 with contiguous slots");
    const int NH = key.size(0), D = key.size(1), H = int(num_kv_heads);
    const int block_size = kv_cache.size(1);
    const int vdb = (D * int(vqb) + 7) / 8;
    TORCH_CHECK(D <= 512, "turboquant store supports D <= 512");
    if (NH == 0) return;
    #define LAUNCH(T, RD, ST) tq_store_fp8<T, ST><<<NH, 32, 0, stream()>>>( \
        RD(key), RD(value), kv_cache.data_ptr<uint8_t>(), \
        slot_mapping.data_ptr<ST>(), kv_cache.stride(0), kv_cache.stride(1), \
        kv_cache.stride(2), D, H, block_size, int(kps), int(vqb), vdb)
    #define DISPATCH_SLOT(T, RD) do { \
        if (slot_mapping.scalar_type() == torch::kLong) LAUNCH(T, RD, int64_t); \
        else LAUNCH(T, RD, int); } while (0)
    if (key.scalar_type() == torch::kHalf) DISPATCH_SLOT(half, hp);
    else if (key.scalar_type() == torch::kBFloat16) DISPATCH_SLOT(__nv_bfloat16, bp);
    else TORCH_CHECK(false, "turboquant_store_fp8: key must be fp16/bf16");
    #undef DISPATCH_SLOT
    #undef LAUNCH
}

static void py_turboquant_store_mse(torch::Tensor y, torch::Tensor norms,
        torch::Tensor value, torch::Tensor midpoints, torch::Tensor kv_cache,
        torch::Tensor slot_mapping, int64_t num_kv_heads, int64_t mse_bits,
        int64_t kps, int64_t vqb) {
    CK(y); CK(norms); CK(value); CK(midpoints); CK(slot_mapping); CKTQ(kv_cache);
    TORCH_CHECK(kv_cache.scalar_type() == torch::kUInt8 && kv_cache.stride(3) == 1,
                "kv_cache must be uint8 with contiguous slots");
    TORCH_CHECK(y.scalar_type() == torch::kFloat && value.scalar_type() == torch::kFloat,
                "turboquant_store_mse expects fp32 y/value");
    const int NH = y.size(0), D = y.size(1), H = int(num_kv_heads);
    const int block_size = kv_cache.size(1);
    const int mse_bytes = (D * int(mse_bits) + 7) / 8;
    const int vdb = (D * int(vqb) + 7) / 8;
    TORCH_CHECK(D <= 512, "turboquant store supports D <= 512");
    if (NH == 0) return;
    #define LAUNCH(ST) tq_store_mse<ST><<<NH, 32, 0, stream()>>>( \
        y.data_ptr<float>(), norms.data_ptr<float>(), value.data_ptr<float>(), \
        midpoints.data_ptr<float>(), kv_cache.data_ptr<uint8_t>(), \
        slot_mapping.data_ptr<ST>(), kv_cache.stride(0), kv_cache.stride(1), \
        kv_cache.stride(2), D, H, block_size, int(mse_bits), mse_bytes, \
        1 << int(mse_bits), int(kps), int(vqb), vdb)
    if (slot_mapping.scalar_type() == torch::kLong) LAUNCH(int64_t); else LAUNCH(int);
    #undef LAUNCH
}

static void py_turboquant_decode_stage1(torch::Tensor q_rot, torch::Tensor kv_cache,
        torch::Tensor block_table, torch::Tensor seq_lens, torch::Tensor centroids,
        torch::Tensor mid_o, int64_t num_splits, int64_t mse_bits, int64_t kps,
        int64_t vqb, double scale, bool key_fp8, bool norm_correction,
        int64_t window) {
    CK(centroids); CKTQ(q_rot); CKTQ(kv_cache); CKTQ(block_table);
    CKTQ(seq_lens); CKTQ(mid_o);
    TORCH_CHECK(kv_cache.stride(3) == 1 && q_rot.stride(2) == 1 && mid_o.stride(3) == 1,
                "innermost dims must be contiguous");
    const int B = q_rot.size(0), Hq = q_rot.size(1), D = q_rot.size(2);
    const int Hk = kv_cache.size(2), block_size = kv_cache.size(1);
    TORCH_CHECK(block_table.scalar_type() == torch::kInt &&
                    seq_lens.scalar_type() == torch::kInt,
                "turboquant decode expects int32 block_table and seq_lens");
    TORCH_CHECK(centroids.scalar_type() == torch::kFloat &&
                    mid_o.scalar_type() == torch::kFloat,
                "turboquant decode expects fp32 centroids and mid_o");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) >= B &&
                    seq_lens.numel() >= B,
                "turboquant decode metadata batch is smaller than query batch");
    TORCH_CHECK(Hk > 0 && Hq % Hk == 0,
                "turboquant decode requires query heads divisible by KV heads");
    TORCH_CHECK(mid_o.dim() == 4 && mid_o.size(0) >= B &&
                    mid_o.size(1) >= Hq && mid_o.size(2) >= num_splits &&
                    mid_o.size(3) >= D + 1,
                "turboquant decode mid_o workspace is too small");
    const int vdb = (D * int(vqb) + 7) / 8;
    TORCH_CHECK(kps + vdb + 4 <= kv_cache.size(3),
                "turboquant decode cache slot is smaller than packed K/V layout");
    TORCH_CHECK(num_splits > 0, "turboquant decode requires num_splits > 0");
    TORCH_CHECK(D % 32 == 0 && D <= 512,
                "turboquant decode supports 32-aligned D <= 512");
    const int mse_bytes = (D * int(mse_bits) + 7) / 8;
    if (B == 0) return;
    dim3 grid(B, Hq, unsigned(num_splits));
    #define LAUNCH(T, RD) tq_decode_stage1<T><<<grid, 32, 0, stream()>>>( \
        RD(q_rot), kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int>(), \
        seq_lens.data_ptr<int>(), centroids.data_ptr<float>(), \
        mid_o.data_ptr<float>(), q_rot.stride(0), q_rot.stride(1), \
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2), \
        block_table.stride(0), mid_o.stride(0), mid_o.stride(1), mid_o.stride(2), \
        D, block_size, int(num_splits), Hq / Hk, int(mse_bits), mse_bytes, \
        int(kps), int(vqb), vdb, float(scale), key_fp8 ? 1 : 0, \
        norm_correction ? 1 : 0, int(window))
    if (q_rot.scalar_type() == torch::kFloat) LAUNCH(float, fp);
    else if (q_rot.scalar_type() == torch::kHalf) LAUNCH(half, hp);
    else if (q_rot.scalar_type() == torch::kBFloat16) LAUNCH(__nv_bfloat16, bp);
    else TORCH_CHECK(false, "turboquant_decode_stage1: q must be fp32/fp16/bf16");
    #undef LAUNCH
}

static void py_turboquant_decode_stage2(torch::Tensor mid_o, torch::Tensor output,
        torch::Tensor lse, torch::Tensor seq_lens, int64_t num_splits) {
    CKTQ(mid_o); CKTQ(output); CKTQ(lse); CKTQ(seq_lens);
    TORCH_CHECK(mid_o.stride(3) == 1 && output.stride(2) == 1,
                "innermost dims must be contiguous");
    const int B = output.size(0), Hq = output.size(1), D = output.size(2);
    TORCH_CHECK(D % 32 == 0 && D <= 512,
                "turboquant decode supports 32-aligned D <= 512");
    if (B == 0) return;
    dim3 grid(B, Hq);
    #define LAUNCH(T, WP) tq_decode_stage2<T><<<grid, 32, 0, stream()>>>( \
        mid_o.data_ptr<float>(), WP(output), lse.data_ptr<float>(), \
        seq_lens.data_ptr<int>(), mid_o.stride(0), mid_o.stride(1), \
        mid_o.stride(2), output.stride(0), output.stride(1), lse.stride(0), \
        int(num_splits), D)
    if (output.scalar_type() == torch::kHalf) LAUNCH(half, hpm);
    else if (output.scalar_type() == torch::kBFloat16) LAUNCH(__nv_bfloat16, bpm);
    else if (output.scalar_type() == torch::kFloat) LAUNCH(float, fpm);
    else TORCH_CHECK(false, "turboquant_decode_stage2: output must be fp32/fp16/bf16");
    #undef LAUNCH
}

static void py_turboquant_dequant_kv(torch::Tensor kv_cache, torch::Tensor block_table,
        torch::Tensor centroids, torch::Tensor k_out, torch::Tensor v_out,
        int64_t num_positions, int64_t mse_bits, int64_t kps, int64_t vqb,
        bool key_fp8, bool norm_correction) {
    CK(centroids); CKTQ(kv_cache); CKTQ(block_table); CKTQ(k_out); CKTQ(v_out);
    TORCH_CHECK(kv_cache.stride(3) == 1 && k_out.stride(3) == 1 && v_out.stride(3) == 1,
                "innermost dims must be contiguous");
    TORCH_CHECK(k_out.scalar_type() == torch::kHalf, "dequant writes fp16");
    const int B = k_out.size(0), Hk = k_out.size(1), D = k_out.size(3);
    const int block_size = kv_cache.size(1);
    TORCH_CHECK(D % 32 == 0 && D <= 512,
                "turboquant dequant supports 32-aligned D <= 512");
    const int mse_bytes = (D * int(mse_bits) + 7) / 8;
    const int vdb = (D * int(vqb) + 7) / 8;
    if (num_positions == 0) return;
    dim3 grid(unsigned(num_positions), B * Hk);
    tq_full_dequant_kv<<<grid, 32, 0, stream()>>>(
        kv_cache.data_ptr<uint8_t>(), block_table.data_ptr<int>(),
        centroids.data_ptr<float>(), hpm(k_out), hpm(v_out),
        k_out.stride(0), k_out.stride(1), k_out.stride(2),
        v_out.stride(0), v_out.stride(1), v_out.stride(2),
        kv_cache.stride(0), kv_cache.stride(1), kv_cache.stride(2),
        block_table.stride(0), D, block_size, Hk, mse_bytes, int(kps), int(vqb),
        vdb, int(mse_bits), key_fp8 ? 1 : 0, norm_correction ? 1 : 0);
}

void init_serving(py::module_& m) {
    m.def("dsv4_router_gemm", &py_dsv4_router_gemm, py::arg("x"),
          py::arg("weight"));
    m.def("dsv4_hash_router", &py_dsv4_hash_router, py::arg("x"),
          py::arg("weight"), py::arg("input_ids"), py::arg("tid2eid"),
          py::arg("routed_scaling_factor"), py::arg("is_padding") = c10::nullopt);
    m.def("dsv4_hash_router_debug", &py_dsv4_hash_router_debug);
    m.def("dsv4_projection_gemv", &py_dsv4_projection_gemv, py::arg("x"),
          py::arg("weight"), py::arg("bf16_output") = false);
    m.def("fill_short_context_topk_indices",
          &py_fill_short_context_topk_indices, py::arg("output"),
          py::arg("positions"), py::arg("topk"),
          py::arg("compress_ratio"));
    m.def("dsv4_mhc_pre", &py_dsv4_mhc_pre,
          py::arg("residual"), py::arg("fn"), py::arg("hc_scale"),
          py::arg("hc_base"), py::arg("rms_eps"), py::arg("pre_eps"),
          py::arg("sinkhorn_eps"), py::arg("post_multiplier"),
          py::arg("sinkhorn_repeat"), py::arg("norm_weight") = c10::nullopt,
          py::arg("norm_eps") = 0.0);
    m.def("dsv4_mhc_fused_post_pre", &py_dsv4_mhc_fused_post_pre,
          py::arg("x"), py::arg("residual"), py::arg("post_mix"),
          py::arg("comb_mix"), py::arg("fn"), py::arg("hc_scale"),
          py::arg("hc_base"), py::arg("rms_eps"), py::arg("pre_eps"),
          py::arg("sinkhorn_eps"), py::arg("post_multiplier"),
          py::arg("sinkhorn_repeat"), py::arg("norm_weight") = c10::nullopt,
          py::arg("norm_eps") = 0.0);
    m.def("dsv4_mhc_post", &py_dsv4_mhc_post,
          py::arg("x"), py::arg("residual"), py::arg("post_mix"),
          py::arg("comb_mix"));
    m.def("dsv4_hc_head", &py_dsv4_hc_head,
          py::arg("residual"), py::arg("fn"), py::arg("hc_scale"),
          py::arg("hc_base"), py::arg("rms_eps"), py::arg("hc_eps"));
    m.def("mla_kv_insert", &py_mla_kv_insert, py::arg("kv_c"), py::arg("k_pe"), py::arg("cos"),
          py::arg("sin"), py::arg("positions"), py::arg("slot_mapping"), py::arg("kv_cache"),
          py::arg("block_size"), py::arg("norm_mode") = 0, py::arg("eps") = 1e-6,
          py::arg("norm_weight") = py::none());
    m.def("mla_decode_partition", &py_mla_decode_partition);
    m.def("mla_decode_fp8_partition", &py_mla_decode_fp8_partition);
    m.def("indexer_metadata", &py_indexer_metadata,
          py::arg("query_start_loc"), py::arg("uncompressed_seq_lens"),
          py::arg("cu_compressed_seq_lens"), py::arg("row_start_cu"),
          py::arg("token_to_seq"), py::arg("cu_ks"), py::arg("cu_ke"),
          py::arg("query_slice_start"), py::arg("query_slice_stop"),
          py::arg("dcp_rank"), py::arg("dcp_world"),
          py::arg("dcp_interleave"), py::arg("compress_ratio"));
    m.def("fp8_paged_mqa_logits", &py_fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_tables"),
          py::arg("max_model_len"), py::arg("token_shard_rank") = 0,
          py::arg("token_shard_world_size") = 1,
          py::arg("trivial_topk") = 0);
    m.def("indexer_pack_tp_candidates", &py_indexer_pack_tp_candidates,
          py::arg("logits"), py::arg("indices"), py::arg("token_shard_rank"),
          py::arg("token_shard_world_size"));
    m.def("indexer_unpack_tp_scores", &py_indexer_unpack_tp_scores,
          py::arg("packed"));
    m.def("indexer_resolve_tp_candidates", &py_indexer_resolve_tp_candidates,
          py::arg("packed"), py::arg("positions"), py::arg("output"));
    m.def("fp8_mqa_logits", &py_fp8_mqa_logits,
          py::arg("q"), py::arg("k"), py::arg("kscale"), py::arg("weights"),
          py::arg("ks"), py::arg("ke"));
    m.def("compute_slot_mapping", &py_compute_slot_mapping,
          py::arg("query_start_loc"), py::arg("positions"),
          py::arg("block_table"), py::arg("slot_mapping"),
          py::arg("num_tokens"), py::arg("max_num_tokens"),
          py::arg("block_size"), py::arg("kv_cache_block_size"),
          py::arg("blocks_per_kv_block"), py::arg("cp_world"),
          py::arg("cp_rank"), py::arg("cp_interleave"), py::arg("pad_id"));
    m.def("apply_write", &py_apply_write,
          py::arg("out"), py::arg("row_stride"), py::arg("indices"),
          py::arg("starts"), py::arg("contents"), py::arg("cu_lens"));
    m.def("apply_write_multi", &py_apply_write_multi,
          py::arg("out_ptrs"), py::arg("out_strides"), py::arg("indices"),
          py::arg("starts"), py::arg("contents"), py::arg("cu_lens"),
          py::arg("group_ids"));
    m.def("prepare_pos_seq_lens", &py_prepare_pos_seq_lens,
          py::arg("pos"), py::arg("seq_lens"), py::arg("idx_mapping"),
          py::arg("query_start_loc"), py::arg("num_computed_tokens"),
          py::arg("max_num_reqs"));
    m.def("prepare_prefill_inputs", &py_prepare_prefill_inputs,
          py::arg("input_ids"), py::arg("next_prefill_tokens"),
          py::arg("idx_mapping"), py::arg("query_start_loc"),
          py::arg("all_token_ids"), py::arg("all_token_ids_stride"),
          py::arg("prefill_lens"), py::arg("num_computed_tokens"));
    m.def("combine_sampled_and_draft_tokens", &py_combine_sampled_and_draft_tokens,
          py::arg("input_ids"), py::arg("idx_mapping"),
          py::arg("last_sampled_tokens"), py::arg("query_start_loc"),
          py::arg("seq_lens"), py::arg("prefill_len"), py::arg("draft_tokens"),
          py::arg("draft_tokens_stride"), py::arg("cu_num_logits"),
          py::arg("logits_indices"), py::arg("num_new_sampled_tokens"));
    m.def("get_num_sampled_and_rejected", &py_get_num_sampled_and_rejected,
          py::arg("num_sampled"), py::arg("num_rejected"), py::arg("seq_lens"),
          py::arg("cu_num_logits"), py::arg("idx_mapping"),
          py::arg("prefill_len"));
    m.def("post_update", &py_post_update,
          py::arg("idx_mapping"), py::arg("num_computed_tokens"),
          py::arg("last_sampled_tokens"), py::arg("output_bin_counts"),
          py::arg("output_bin_counts_stride"), py::arg("sampled_tokens"),
          py::arg("sampled_tokens_stride"), py::arg("num_sampled"),
          py::arg("num_rejected"), py::arg("query_start_loc"),
          py::arg("all_token_ids"), py::arg("all_token_ids_stride"),
          py::arg("total_len"));
    m.def("post_update_num_computed_tokens", &py_post_update_num_computed_tokens,
          py::arg("idx_mapping"), py::arg("num_computed_tokens"),
          py::arg("query_start_loc"));
    m.def("expand_idx_mapping", &py_expand_idx_mapping,
          py::arg("idx_mapping"), py::arg("expanded_idx_mapping"),
          py::arg("expanded_local_pos"), py::arg("cu_num_logits"));
    m.def("gather_block_tables", &py_gather_block_tables,
          py::arg("idx_mapping"), py::arg("src_ptrs"), py::arg("dst_ptrs"),
          py::arg("strides"), py::arg("num_blocks"),
          py::arg("num_blocks_stride"), py::arg("num_reqs"),
          py::arg("num_reqs_padded"));
    m.def("compute_slot_mappings", &py_compute_slot_mappings,
          py::arg("idx_mapping"), py::arg("query_start_loc"), py::arg("pos"),
          py::arg("block_table_ptrs"), py::arg("block_table_strides"),
          py::arg("block_sizes"), py::arg("slot_mappings"),
          py::arg("slot_mappings_stride"), py::arg("max_num_tokens"),
          py::arg("cp_rank"), py::arg("cp_size"), py::arg("cp_interleave"),
          py::arg("pad_id"));
    m.def("prepare_uniform_decode", &py_prepare_uniform_decode,
          py::arg("seq_lens"), py::arg("decode_seq_lens"),
          py::arg("block_table"), py::arg("block_table_stride"),
          py::arg("expanded_block_table"), py::arg("expanded_bt_stride"),
          py::arg("decode_lens"), py::arg("max_decode_len"),
          py::arg("num_decode_tokens"));
    m.def("prepare_dflash_inputs", &py_prepare_dflash_inputs,
          py::arg("out_input_ids"), py::arg("out_query_positions"),
          py::arg("out_query_start_loc"), py::arg("out_seq_lens"),
          py::arg("out_query_slot_mapping"), py::arg("out_context_positions"),
          py::arg("out_context_slot_mapping"), py::arg("out_sample_indices"),
          py::arg("out_sample_pos"), py::arg("out_sample_idx_mapping"),
          py::arg("target_positions"), py::arg("target_query_start_loc"),
          py::arg("idx_mapping"), py::arg("last_sampled"),
          py::arg("next_prefill_tokens"), py::arg("num_sampled"),
          py::arg("num_rejected"), py::arg("block_table"),
          py::arg("block_table_stride"), py::arg("parallel_drafting_token_id"),
          py::arg("block_size"), py::arg("num_query_per_req"),
          py::arg("num_speculative_steps"), py::arg("max_num_reqs"),
          py::arg("max_num_tokens"), py::arg("max_model_len"),
          py::arg("sample_from_anchor"), py::arg("pad_slot_id"),
          py::arg("num_reqs"), py::arg("max_tokens_per_req"));
    m.def("mla_decode_bf16_sparse_glm", &py_mla_decode_bf16_sparse_glm, py::arg("q"),
          py::arg("kv"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"),
          py::arg("partition_size") = 0);
    m.def("mla_decode_bf16_sparse_nope", &py_mla_decode_bf16_sparse_nope, py::arg("q"),
          py::arg("kv"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"),
          py::arg("partition_size") = 0);
    m.def("mla_decode_fp8_sparse_glm", &py_mla_decode_fp8_sparse_glm, py::arg("q"),
          py::arg("data"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"),
          py::arg("kv_scale"), py::arg("partition_size") = 0);
    m.def("sparse_topk_tlen", &py_sparse_topk_tlen, py::arg("indices"));
    m.def("mla_decode_fp8_sparse_glm_splitq", &py_mla_decode_fp8_sparse_glm_splitq,
          py::arg("q_nope"), py::arg("q_pe"), py::arg("data"), py::arg("block_table"),
          py::arg("indices"), py::arg("topk_length"), py::arg("block_size"),
          py::arg("scale"), py::arg("kv_scale"), py::arg("partition_size") = 0);
    m.def("moe_weighted_sum", &py_moe_weighted_sum, py::arg("x"), py::arg("w"),
          py::arg("out"));
    m.def("mla_decode_fp8_sparse", &py_mla_decode_fp8_sparse, py::arg("q"), py::arg("data"),
          py::arg("scale_cache"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"),
          py::arg("partition_size") = 0);
    m.def("mla_decode_fp8_sparse_dsv4", &py_mla_decode_fp8_sparse_dsv4,
          py::arg("q"),
          py::arg("main_cache"), py::arg("main_block_table"),
          py::arg("main_indices"), py::arg("main_topk_length"),
          py::arg("main_indices_are_slots"),
          py::arg("extra_cache"), py::arg("extra_block_table"),
          py::arg("extra_indices"), py::arg("extra_topk_length"),
          py::arg("extra_indices_are_slots"),
          py::arg("sink"),
          py::arg("main_block_size"), py::arg("extra_block_size"),
          py::arg("scale"), py::arg("partition_size") = 0);
    m.def("paged_attention_gqa_staged", &py_paged_attention_gqa_staged);
    m.def("attn_window", &py_attn_window, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale"), py::arg("window") = 0);
    m.def("kv_scatter", &py_kv_scatter);
    m.def("kv_gather", &py_kv_gather);
    m.def("copy_blocks", &py_copy_blocks);
    m.def("paged_attention", &py_paged_attention,
          py::arg("q"), py::arg("key_cache"), py::arg("value_cache"), py::arg("block_table"),
          py::arg("context_lens"), py::arg("block_size"), py::arg("scale"), py::arg("num_kv_heads"),
          py::arg("alibi") = py::none(), py::arg("block_mask") = py::none(), py::arg("window") = 0);
    m.def("paged_attention_v2", &py_paged_attention_v2,
          py::arg("q"), py::arg("key_cache"), py::arg("value_cache"), py::arg("block_table"),
          py::arg("context_lens"), py::arg("block_size"), py::arg("scale"), py::arg("num_kv_heads"),
          py::arg("partition_size"), py::arg("max_context"), py::arg("window") = 0);
    m.def("rope_kv_insert", &py_rope_kv_insert,
          py::arg("k"), py::arg("v"), py::arg("cos"), py::arg("sin"), py::arg("positions"),
          py::arg("slot_mapping"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("num_kv_heads"), py::arg("block_size"), py::arg("norm_weight") = py::none(),
          py::arg("gemma") = false, py::arg("eps") = 1e-6);
    m.def("rope_q", &py_rope_q,
          py::arg("q"), py::arg("cos"), py::arg("sin"), py::arg("positions"), py::arg("num_heads"),
          py::arg("norm_weight") = py::none(), py::arg("gemma") = false, py::arg("eps") = 1e-6);
    m.def("attn_q", &py_attn_q);
    m.def("mla_decode", &py_mla_decode);
    m.def("mla_decode_fp8", &py_mla_decode_fp8);
    m.def("mla_kv_insert_fp8", &py_mla_kv_insert_fp8);
    m.def("mla_q_norm_rope", &py_mla_q_norm_rope,
          py::arg("q"), py::arg("cos"), py::arg("sin"), py::arg("positions"), py::arg("num_heads"),
          py::arg("nope_dim"), py::arg("rope_dim"), py::arg("norm_mode") = 0, py::arg("eps") = 1e-6,
          py::arg("norm_weight") = py::none());
    m.def("sample", &py_sample, py::arg("logits"), py::arg("mode"), py::arg("seed") = 0,
          py::arg("temperature") = 1.0, py::arg("param") = 0.9, py::arg("k") = 40);
    m.def("apply_penalties", &py_apply_penalties);
    m.def("apply_token_bitmask", &py_apply_token_bitmask);
    m.def("apply_bad_words", &py_apply_bad_words);
    m.def("beam_advance", &py_beam_advance);
    m.def("spec_verify_linear", &py_spec_verify_linear);
    m.def("build_dynamic_tree", &py_build_dynamic_tree);
    m.def("spec_verify_tree", &py_spec_verify_tree);
    m.def("spec_compact", &py_spec_compact);
    m.def("spec_update_kv_meta", &py_spec_update_kv_meta);
    m.def("turboquant_store_fp8", &py_turboquant_store_fp8,
          py::arg("key"), py::arg("value"), py::arg("kv_cache"),
          py::arg("slot_mapping"), py::arg("num_kv_heads"),
          py::arg("key_packed_size"), py::arg("value_quant_bits"));
    m.def("turboquant_store_mse", &py_turboquant_store_mse,
          py::arg("y"), py::arg("norms"), py::arg("value"), py::arg("midpoints"),
          py::arg("kv_cache"), py::arg("slot_mapping"), py::arg("num_kv_heads"),
          py::arg("mse_bits"), py::arg("key_packed_size"),
          py::arg("value_quant_bits"));
    m.def("turboquant_decode_stage1", &py_turboquant_decode_stage1,
          py::arg("q_rot"), py::arg("kv_cache"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("centroids"), py::arg("mid_o"),
          py::arg("num_kv_splits"), py::arg("mse_bits"),
          py::arg("key_packed_size"), py::arg("value_quant_bits"),
          py::arg("scale"), py::arg("key_fp8"), py::arg("norm_correction"),
          py::arg("sliding_window") = 0);
    m.def("turboquant_decode_stage2", &py_turboquant_decode_stage2,
          py::arg("mid_o"), py::arg("output"), py::arg("lse"),
          py::arg("seq_lens"), py::arg("num_kv_splits"));
    m.def("turboquant_dequant_kv", &py_turboquant_dequant_kv,
          py::arg("kv_cache"), py::arg("block_table"), py::arg("centroids"),
          py::arg("k_out"), py::arg("v_out"), py::arg("num_positions"),
          py::arg("mse_bits"), py::arg("key_packed_size"),
          py::arg("value_quant_bits"), py::arg("key_fp8"),
          py::arg("norm_correction"));
}
