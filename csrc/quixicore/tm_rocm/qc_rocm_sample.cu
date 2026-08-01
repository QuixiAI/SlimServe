// QuixiCore-HIP V2 sampler / spec-decode bindings for gfx942.
//
// The ROCm counterpart of the sampler half of tm_cuda/tm_cuda_m6.cu. The
// kernels are not duplicated: this unit includes the same
// serving/v2_sample_kernels.cuh the CUDA build does, whose cross-lane
// primitives and tt_exp/tt_div select the CDNA lowering under
// __HIP_PLATFORM_AMD__ while keeping Triton's 32-lane reduction order, so the
// bitwise contract holds on both targets.
//
// The turboquant trio in the CUDA unit (fwht_rotate, permute_cols,
// moe_lora_align) is deliberately absent -- it pulls quant/turboquant.cuh,
// which carries its own wave32 shuffle assumptions and is not on this serving
// path. Registered into the module by qc_rocm_serving.cu.
#include "v2_sample_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define M6CK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t m6st() { return at::cuda::getCurrentCUDAStream(); }

#define DISPATCH_M6(t, ...) do {                                                  \
    if (t.scalar_type() == torch::kFloat)        { using scalar_t = float;          __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kHalf)    { using scalar_t = __half;         __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kBFloat16){ using scalar_t = __nv_bfloat16;  __VA_ARGS__ } \
    else TORCH_CHECK(false, "want fp32/fp16/bf16"); } while (0)
template <typename S> static const S* c6(const torch::Tensor& t) { return reinterpret_cast<const S*>(t.data_ptr()); }
template <typename S> static S* m6(torch::Tensor& t) { return reinterpret_cast<S*>(t.data_ptr()); }

static void py_v2_apply_temperature(torch::Tensor logits,
        torch::Tensor expanded_idx_mapping, torch::Tensor temperature) {
    TORCH_CHECK(logits.is_cuda() && logits.scalar_type() == torch::kFloat);
    const int num_tokens = logits.size(0), V = logits.size(1);
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, (V + 8191) / 8192);
    tmv2s::v2_temperature_k<<<grid, 256, 0, m6st()>>>(
        logits.data_ptr<float>(), logits.stride(0),
        expanded_idx_mapping.data_ptr<int>(), temperature.data_ptr<float>(), V);
}

template <typename S, typename VT>
static void launch_v2_gumbel(torch::Tensor& local_argmax,
        torch::Tensor& local_max, float* pl_ptr, int64_t pl_stride,
        const int64_t* col_ptr, bool per_token_col, const torch::Tensor& logits,
        const torch::Tensor& expanded_idx_mapping, const torch::Tensor& seeds,
        const torch::Tensor& pos, const torch::Tensor& temperature,
        bool apply_temperature, int V, dim3 grid, cudaStream_t s) {
    int64_t* la = local_argmax.data_ptr<int64_t>();
    const int64_t la_s = local_argmax.stride(0);
    VT* lm = (VT*)local_max.data_ptr();
    const int64_t lm_s = local_max.stride(0);
    const S* lg = c6<S>(logits);
    const int64_t lg_s = logits.stride(0);
    const int* eim = expanded_idx_mapping.data_ptr<int>();
    const int64_t* sd = seeds.data_ptr<int64_t>();
    const int64_t* ps = pos.data_ptr<int64_t>();
    const float* tp = temperature.data_ptr<float>();
    const bool has_pl = pl_ptr != nullptr;
    if (apply_temperature) {
        if (has_pl)
            tmv2s::v2_gumbel_sample_k<S, VT, true, true><<<grid, 128, 0, s>>>(
                la, la_s, lm, lm_s, pl_ptr, pl_stride, col_ptr, per_token_col,
                lg, lg_s, eim, sd, ps, tp, V);
        else
            tmv2s::v2_gumbel_sample_k<S, VT, true, false><<<grid, 128, 0, s>>>(
                la, la_s, lm, lm_s, nullptr, 0, nullptr, 0,
                lg, lg_s, eim, sd, ps, tp, V);
    } else {
        if (has_pl)
            tmv2s::v2_gumbel_sample_k<S, VT, false, true><<<grid, 128, 0, s>>>(
                la, la_s, lm, lm_s, pl_ptr, pl_stride, col_ptr, per_token_col,
                lg, lg_s, eim, sd, ps, tp, V);
        else
            tmv2s::v2_gumbel_sample_k<S, VT, false, false><<<grid, 128, 0, s>>>(
                la, la_s, lm, lm_s, nullptr, 0, nullptr, 0,
                lg, lg_s, eim, sd, ps, tp, V);
    }
}

static void py_v2_gumbel_sample(torch::Tensor local_argmax,
        torch::Tensor local_max, c10::optional<torch::Tensor> processed_logits,
        c10::optional<torch::Tensor> processed_logits_col, torch::Tensor logits,
        torch::Tensor expanded_idx_mapping, torch::Tensor seeds,
        torch::Tensor pos, torch::Tensor temperature, bool apply_temperature,
        bool per_token_col) {
    const int num_tokens = logits.size(0), V = logits.size(1);
    const int num_blocks = local_argmax.size(1);
    if (num_tokens == 0) return;
    const bool has_pl = processed_logits.has_value();
    float* pl_ptr = has_pl ? processed_logits->data_ptr<float>() : nullptr;
    const int64_t pl_stride = has_pl ? processed_logits->stride(0) : 0;
    torch::Tensor col;
    const int64_t* col_ptr = nullptr;
    if (processed_logits_col.has_value()) {
        col = processed_logits_col->to(torch::kLong);
        col_ptr = col.data_ptr<int64_t>();
    }
    dim3 grid(num_tokens, num_blocks);
    auto s = m6st();
    DISPATCH_M6(logits, {
        if (local_max.scalar_type() == torch::kDouble)
            launch_v2_gumbel<scalar_t, double>(local_argmax, local_max, pl_ptr,
                pl_stride, col_ptr, per_token_col, logits, expanded_idx_mapping,
                seeds, pos, temperature, apply_temperature, V, grid, s);
        else
            launch_v2_gumbel<scalar_t, float>(local_argmax, local_max, pl_ptr,
                pl_stride, col_ptr, per_token_col, logits, expanded_idx_mapping,
                seeds, pos, temperature, apply_temperature, V, grid, s);
    });
}

static void py_v2_topk_log_softmax(torch::Tensor out, torch::Tensor logits,
        torch::Tensor topk_ids) {
    const int batch = logits.size(0), V = logits.size(1);
    const int topk = topk_ids.size(1);
    if (batch == 0) return;
    DISPATCH_M6(logits, tmv2s::v2_topk_log_softmax_k<scalar_t>
        <<<batch, 128, 0, m6st()>>>(out.data_ptr<float>(), c6<scalar_t>(logits),
        logits.stride(0), topk_ids.data_ptr<int64_t>(), topk, V););
}

static void py_v2_ranks(torch::Tensor out, torch::Tensor logits,
        torch::Tensor token_ids) {
    const int batch = logits.size(0), V = logits.size(1);
    if (batch == 0) return;
    DISPATCH_M6(logits, tmv2s::v2_ranks_k<scalar_t><<<batch, 256, 0, m6st()>>>(
        out.data_ptr<int64_t>(), c6<scalar_t>(logits), logits.stride(0),
        token_ids.data_ptr<int64_t>(), V););
}

static void py_v2_fill_logprob_token_ids(torch::Tensor out_token_ids,
        torch::Tensor out_valid_mask, torch::Tensor sampled_token_ids,
        torch::Tensor topk_indices, torch::Tensor expanded_idx_mapping,
        torch::Tensor num_per_req_token_ids, torch::Tensor per_req_token_ids,
        int64_t num_topk) {
    const int batch = out_token_ids.size(0);
    if (batch == 0) return;
    tmv2s::v2_fill_logprob_token_ids_k<<<batch, 128, 0, m6st()>>>(
        out_token_ids.data_ptr<int64_t>(), out_token_ids.stride(0),
        out_valid_mask.data_ptr<bool>(), out_valid_mask.stride(0),
        sampled_token_ids.data_ptr<int64_t>(), topk_indices.data_ptr<int>(),
        topk_indices.stride(0), expanded_idx_mapping.data_ptr<int>(),
        num_per_req_token_ids.data_ptr<int>(), per_req_token_ids.data_ptr<int>(),
        per_req_token_ids.stride(0), int(num_topk));
}

static void py_v2_penalties(torch::Tensor logits,
        torch::Tensor expanded_idx_mapping, torch::Tensor token_ids,
        torch::Tensor expanded_local_pos, torch::Tensor repetition_penalty,
        torch::Tensor frequency_penalty, torch::Tensor presence_penalty,
        torch::Tensor prompt_bin_mask, torch::Tensor output_bin_counts) {
    TORCH_CHECK(logits.scalar_type() == torch::kFloat);
    const int num_tokens = logits.size(0), V = logits.size(1);
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, (V + 8191) / 8192);
    tmv2s::v2_penalties_k<<<grid, 256, 0, m6st()>>>(
        logits.data_ptr<float>(), logits.stride(0),
        expanded_idx_mapping.data_ptr<int>(), token_ids.data_ptr<int>(),
        expanded_local_pos.data_ptr<int>(), repetition_penalty.data_ptr<float>(),
        frequency_penalty.data_ptr<float>(), presence_penalty.data_ptr<float>(),
        prompt_bin_mask.data_ptr<int>(), prompt_bin_mask.stride(0),
        output_bin_counts.data_ptr<int>(), output_bin_counts.stride(0), V);
}

static void py_v2_bincount(torch::Tensor expanded_idx_mapping,
        torch::Tensor all_token_ids, torch::Tensor prompt_len,
        torch::Tensor prefill_len, torch::Tensor prompt_bin_mask,
        torch::Tensor output_bin_counts, int64_t max_prefill_len) {
    const int num_tokens = expanded_idx_mapping.numel();
    if (num_tokens == 0) return;
    dim3 grid(num_tokens, (max_prefill_len + 1023) / 1024);
    tmv2s::v2_bincount_k<<<grid, 256, 0, m6st()>>>(
        expanded_idx_mapping.data_ptr<int>(), all_token_ids.data_ptr<int>(),
        all_token_ids.stride(0), prompt_len.data_ptr<int>(),
        prefill_len.data_ptr<int>(), prompt_bin_mask.data_ptr<int>(),
        prompt_bin_mask.stride(0), output_bin_counts.data_ptr<int>(),
        output_bin_counts.stride(0));
}

static void py_v2_prompt_logprobs_token_ids(torch::Tensor out,
        torch::Tensor query_start_loc, torch::Tensor idx_mapping,
        torch::Tensor num_computed_tokens, torch::Tensor all_token_ids) {
    const int num_reqs = idx_mapping.numel();
    if (num_reqs == 0) return;
    tmv2s::v2_prompt_logprobs_token_ids_k<<<num_reqs, 256, 0, m6st()>>>(
        out.data_ptr<int64_t>(), query_start_loc.data_ptr<int>(),
        idx_mapping.data_ptr<int>(), num_computed_tokens.data_ptr<int>(),
        all_token_ids.data_ptr<int>(), all_token_ids.stride(0));
}

static void py_v2_rejection_sample(torch::Tensor sampled,
        torch::Tensor num_sampled, torch::Tensor target_rejected_lse,
        torch::Tensor draft_rejected_lse, torch::Tensor target_logits,
        torch::Tensor t_local_argmax, torch::Tensor t_local_max,
        torch::Tensor t_local_sumexp, torch::Tensor draft_sampled,
        c10::optional<torch::Tensor> draft_logits, torch::Tensor d_local_max,
        torch::Tensor d_local_sumexp, torch::Tensor cu_num_logits,
        torch::Tensor idx_mapping, torch::Tensor temperature,
        torch::Tensor seed, torch::Tensor pos, int64_t vocab_num_blocks) {
    const int num_reqs = cu_num_logits.numel() - 1;
    TORCH_CHECK(vocab_num_blocks <= 32, "padded stats width must fit one warp");
    if (num_reqs == 0) return;
    const bool has_draft = draft_logits.has_value();
    const float* d_ptr = has_draft ? draft_logits->data_ptr<float>() : nullptr;
    const int64_t d_s0 = has_draft ? draft_logits->stride(0) : 0;
    const int64_t d_s1 = has_draft ? draft_logits->stride(1) : 0;
    auto s = m6st();
#define V2_REJ_LAUNCH(HD, DS)                                                 \
    tmv2s::v2_rejection_k<HD, DS><<<num_reqs, 32, 0, s>>>(                    \
        sampled.data_ptr<int64_t>(), sampled.stride(0),                       \
        num_sampled.data_ptr<int>(), target_rejected_lse.data_ptr<float>(),   \
        draft_rejected_lse.data_ptr<float>(),                                 \
        target_logits.data_ptr<float>(), target_logits.stride(0),             \
        t_local_argmax.data_ptr<int64_t>(), t_local_argmax.stride(0),         \
        t_local_max.data_ptr<float>(), t_local_max.stride(0),                 \
        t_local_sumexp.data_ptr<float>(), t_local_sumexp.stride(0),           \
        draft_sampled.data_ptr<DS>(), d_ptr, d_s0, d_s1,                      \
        d_local_max.data_ptr<float>(), d_local_max.stride(0),                 \
        d_local_sumexp.data_ptr<float>(), d_local_sumexp.stride(0),           \
        cu_num_logits.data_ptr<int>(), idx_mapping.data_ptr<int>(),           \
        temperature.data_ptr<float>(), seed.data_ptr<int64_t>(),              \
        pos.data_ptr<int64_t>(), int(vocab_num_blocks))
    const bool ds64 = draft_sampled.scalar_type() == torch::kLong;
    if (has_draft) { if (ds64) V2_REJ_LAUNCH(true, int64_t); else V2_REJ_LAUNCH(true, int32_t); }
    else { if (ds64) V2_REJ_LAUNCH(false, int64_t); else V2_REJ_LAUNCH(false, int32_t); }
#undef V2_REJ_LAUNCH
}

static void py_v2_resample(torch::Tensor rl_argmax, torch::Tensor rl_max,
        torch::Tensor target_logits, torch::Tensor target_rejected_lse,
        c10::optional<torch::Tensor> draft_logits,
        torch::Tensor draft_rejected_lse, torch::Tensor rejected_step,
        torch::Tensor cu_num_logits, torch::Tensor expanded_idx_mapping,
        torch::Tensor draft_sampled, torch::Tensor temperature,
        torch::Tensor seed, torch::Tensor pos, int64_t vocab_size) {
    const int num_reqs = cu_num_logits.numel() - 1;
    const int num_blocks = rl_argmax.size(1);
    const int V = int(vocab_size);
    if (num_reqs == 0) return;
    const bool has_draft = draft_logits.has_value();
    const float* d_ptr = has_draft ? draft_logits->data_ptr<float>() : nullptr;
    const int64_t d_s0 = has_draft ? draft_logits->stride(0) : 0;
    const int64_t d_s1 = has_draft ? draft_logits->stride(1) : 0;
    dim3 grid(num_reqs, num_blocks);
    auto s = m6st();
#define V2_RES_LAUNCH(HD, VT, DS)                                             \
    tmv2s::v2_resample_k<HD, VT, DS><<<grid, 32, 0, s>>>(                     \
        rl_argmax.data_ptr<int64_t>(), rl_argmax.stride(0),                   \
        (VT*)rl_max.data_ptr(), rl_max.stride(0),                             \
        target_logits.data_ptr<float>(), target_logits.stride(0),             \
        target_rejected_lse.data_ptr<float>(), d_ptr, d_s0, d_s1,             \
        draft_rejected_lse.data_ptr<float>(), rejected_step.data_ptr<int>(),  \
        cu_num_logits.data_ptr<int>(), expanded_idx_mapping.data_ptr<int>(),  \
        draft_sampled.data_ptr<DS>(), temperature.data_ptr<float>(),          \
        seed.data_ptr<int64_t>(), pos.data_ptr<int64_t>(), V)
    const bool fp64 = rl_max.scalar_type() == torch::kDouble;
    const bool ds64 = draft_sampled.scalar_type() == torch::kLong;
    if (has_draft) {
        if (fp64) { if (ds64) V2_RES_LAUNCH(true, double, int64_t); else V2_RES_LAUNCH(true, double, int32_t); }
        else { if (ds64) V2_RES_LAUNCH(true, float, int64_t); else V2_RES_LAUNCH(true, float, int32_t); }
    } else {
        if (fp64) { if (ds64) V2_RES_LAUNCH(false, double, int64_t); else V2_RES_LAUNCH(false, double, int32_t); }
        else { if (ds64) V2_RES_LAUNCH(false, float, int64_t); else V2_RES_LAUNCH(false, float, int32_t); }
    }
#undef V2_RES_LAUNCH
}

static void py_v2_grammar_bitmask(torch::Tensor logits,
        torch::Tensor logits_indices, torch::Tensor bitmask, int64_t num_masks) {
    const int V = logits.size(1);
    if (num_masks == 0) return;
    dim3 grid(num_masks, (V + 8191) / 8192);
    #define GB_LAUNCH(T, PTR) tmv2s::v2_grammar_bitmask_k<T><<<grid, 256, 0, m6st()>>>( \
        reinterpret_cast<T*>(PTR), logits.stride(0), \
        logits_indices.data_ptr<int>(), bitmask.data_ptr<int>(), \
        bitmask.stride(0), V)
    if (logits.scalar_type() == torch::kFloat)
        GB_LAUNCH(float, logits.data_ptr<float>());
    else if (logits.scalar_type() == torch::kBFloat16)
        GB_LAUNCH(__nv_bfloat16, logits.data_ptr());
    else if (logits.scalar_type() == torch::kHalf)
        GB_LAUNCH(__half, logits.data_ptr());
    else
        TORCH_CHECK(false, "grammar bitmask: unsupported logits dtype");
    #undef GB_LAUNCH
}

static void py_v2_min_p(torch::Tensor logits, torch::Tensor expanded_idx_mapping,
        torch::Tensor min_p) {
    TORCH_CHECK(logits.scalar_type() == torch::kFloat);
    const int num_tokens = logits.size(0), V = logits.size(1);
    if (num_tokens == 0) return;
    tmv2s::v2_min_p_k<<<num_tokens, 256, 0, m6st()>>>(
        logits.data_ptr<float>(), logits.stride(0),
        expanded_idx_mapping.data_ptr<int>(), min_p.data_ptr<float>(), V);
}

static void py_v2_logit_bias(torch::Tensor logits,
        torch::Tensor expanded_idx_mapping, torch::Tensor pos,
        torch::Tensor num_allowed_token_ids, torch::Tensor allowed_token_ids,
        torch::Tensor num_logit_bias, torch::Tensor logit_bias_token_ids,
        torch::Tensor logit_bias, torch::Tensor min_lens,
        torch::Tensor num_stop_token_ids, torch::Tensor stop_token_ids) {
    TORCH_CHECK(logits.scalar_type() == torch::kFloat);
    TORCH_CHECK(allowed_token_ids.size(1) <= 1024 &&
                logit_bias_token_ids.size(1) <= 1024 &&
                stop_token_ids.size(1) <= 1024);
    const int num_tokens = logits.size(0), V = logits.size(1);
    if (num_tokens == 0) return;
    tmv2s::v2_logit_bias_k<<<num_tokens, 1024, 0, m6st()>>>(
        logits.data_ptr<float>(), logits.stride(0), V,
        expanded_idx_mapping.data_ptr<int>(),
        num_allowed_token_ids.data_ptr<int>(),
        allowed_token_ids.data_ptr<int>(), allowed_token_ids.stride(0),
        num_logit_bias.data_ptr<int>(), logit_bias_token_ids.data_ptr<int>(),
        logit_bias_token_ids.stride(0), logit_bias.data_ptr<float>(),
        logit_bias.stride(0), pos.data_ptr<int64_t>(), min_lens.data_ptr<int>(),
        num_stop_token_ids.data_ptr<int>(), stop_token_ids.data_ptr<int>(),
        stop_token_ids.stride(0));
}

static void py_v2_bad_words(torch::Tensor logits,
        torch::Tensor expanded_idx_mapping, torch::Tensor bad_word_token_ids,
        torch::Tensor bad_word_offsets, torch::Tensor num_bad_words,
        torch::Tensor all_token_ids, torch::Tensor prompt_len,
        torch::Tensor total_len, torch::Tensor input_ids,
        torch::Tensor expanded_local_pos) {
    TORCH_CHECK(logits.scalar_type() == torch::kFloat);
    const int num_tokens = logits.size(0);
    if (num_tokens == 0) return;
    tmv2s::v2_bad_words_k<<<num_tokens, 128, 0, m6st()>>>(
        logits.data_ptr<float>(), logits.stride(0),
        expanded_idx_mapping.data_ptr<int>(),
        bad_word_token_ids.data_ptr<int>(), bad_word_token_ids.stride(0),
        bad_word_offsets.data_ptr<int>(), bad_word_offsets.stride(0),
        num_bad_words.data_ptr<int>(), all_token_ids.data_ptr<int>(),
        all_token_ids.stride(0), prompt_len.data_ptr<int>(),
        total_len.data_ptr<int>(), input_ids.data_ptr<int>(),
        expanded_local_pos.data_ptr<int>());
}

static void py_v2_local_logits_stats(torch::Tensor t_local_argmax,
        torch::Tensor t_local_max, torch::Tensor t_local_sumexp,
        torch::Tensor d_local_max, torch::Tensor d_local_sumexp,
        torch::Tensor target_logits, c10::optional<torch::Tensor> draft_logits,
        torch::Tensor expanded_idx_mapping, torch::Tensor expanded_local_pos,
        torch::Tensor temperature, int64_t vocab_size,
        int64_t num_speculative_steps) {
    TORCH_CHECK(target_logits.scalar_type() == torch::kFloat);
    const int num_logits = target_logits.size(0);
    const int nvb = t_local_argmax.size(1);
    if (num_logits == 0) return;
    const bool has_draft = draft_logits.has_value();
    const float* d_ptr = has_draft ? draft_logits->data_ptr<float>() : nullptr;
    const int64_t d_s0 = has_draft ? draft_logits->stride(0) : 0;
    const int64_t d_s1 = has_draft ? draft_logits->stride(1) : 0;
    dim3 grid(num_logits, nvb);
    auto s = m6st();
#define V2_STATS_LAUNCH(HD)                                                   \
    tmv2s::v2_local_logits_stats_k<HD><<<grid, 128, 0, s>>>(                  \
        t_local_argmax.data_ptr<int64_t>(), t_local_argmax.stride(0),         \
        t_local_max.data_ptr<float>(), t_local_max.stride(0),                 \
        t_local_sumexp.data_ptr<float>(), t_local_sumexp.stride(0),           \
        d_local_max.data_ptr<float>(), d_local_max.stride(0),                 \
        d_local_sumexp.data_ptr<float>(), d_local_sumexp.stride(0),           \
        target_logits.data_ptr<float>(), target_logits.stride(0),             \
        d_ptr, d_s0, d_s1, expanded_idx_mapping.data_ptr<int>(),              \
        expanded_local_pos.data_ptr<int>(), temperature.data_ptr<float>(),    \
        int(vocab_size), int(num_speculative_steps))
    if (has_draft) V2_STATS_LAUNCH(true); else V2_STATS_LAUNCH(false);
#undef V2_STATS_LAUNCH
}

static void py_v2_insert_resampled(torch::Tensor sampled,
        torch::Tensor num_sampled, torch::Tensor rl_argmax,
        torch::Tensor rl_max, int64_t resample_num_blocks,
        torch::Tensor cu_num_logits, torch::Tensor expanded_idx_mapping,
        torch::Tensor temperature) {
    const int num_reqs = cu_num_logits.numel() - 1;
    if (num_reqs == 0) return;
    auto s = m6st();
#define V2_INS_LAUNCH(VT)                                                     \
    tmv2s::v2_insert_resampled_k<VT><<<num_reqs, 32, 0, s>>>(                 \
        sampled.data_ptr<int64_t>(), sampled.stride(0),                       \
        num_sampled.data_ptr<int>(), rl_argmax.data_ptr<int64_t>(),           \
        rl_argmax.stride(0), (VT*)rl_max.data_ptr(), rl_max.stride(0),        \
        int(resample_num_blocks), cu_num_logits.data_ptr<int>(),              \
        expanded_idx_mapping.data_ptr<int>(), temperature.data_ptr<float>())
    if (rl_max.scalar_type() == torch::kDouble) V2_INS_LAUNCH(double);
    else V2_INS_LAUNCH(float);
#undef V2_INS_LAUNCH
}

static void py_v2_flatten_sampled(torch::Tensor flat_sampled,
        torch::Tensor sampled, torch::Tensor num_sampled,
        torch::Tensor cu_num_logits) {
    const int num_reqs = cu_num_logits.numel() - 1;
    if (num_reqs == 0) return;
    tmv2s::v2_flatten_sampled_k<<<num_reqs, 32, 0, m6st()>>>(
        flat_sampled.data_ptr<int64_t>(), sampled.data_ptr<int64_t>(),
        sampled.stride(0), num_sampled.data_ptr<int>(),
        cu_num_logits.data_ptr<int>());
}


void init_sample(py::module_& m) {
    m.def("v2_grammar_bitmask", &py_v2_grammar_bitmask);
    m.def("v2_min_p", &py_v2_min_p);
    m.def("v2_logit_bias", &py_v2_logit_bias);
    m.def("v2_bad_words", &py_v2_bad_words);
    m.def("v2_local_logits_stats", &py_v2_local_logits_stats,
          py::arg("t_local_argmax"), py::arg("t_local_max"),
          py::arg("t_local_sumexp"), py::arg("d_local_max"),
          py::arg("d_local_sumexp"), py::arg("target_logits"),
          py::arg("draft_logits"), py::arg("expanded_idx_mapping"),
          py::arg("expanded_local_pos"), py::arg("temperature"),
          py::arg("vocab_size"), py::arg("num_speculative_steps"));
    m.def("v2_insert_resampled", &py_v2_insert_resampled);
    m.def("v2_flatten_sampled", &py_v2_flatten_sampled);
    m.def("v2_apply_temperature", &py_v2_apply_temperature);
    m.def("v2_gumbel_sample", &py_v2_gumbel_sample, py::arg("local_argmax"),
          py::arg("local_max"), py::arg("processed_logits"),
          py::arg("processed_logits_col"), py::arg("logits"),
          py::arg("expanded_idx_mapping"), py::arg("seeds"), py::arg("pos"),
          py::arg("temperature"), py::arg("apply_temperature"),
          py::arg("per_token_col"));
    m.def("v2_topk_log_softmax", &py_v2_topk_log_softmax);
    m.def("v2_ranks", &py_v2_ranks);
    m.def("v2_fill_logprob_token_ids", &py_v2_fill_logprob_token_ids);
    m.def("v2_penalties", &py_v2_penalties);
    m.def("v2_bincount", &py_v2_bincount);
    m.def("v2_prompt_logprobs_token_ids", &py_v2_prompt_logprobs_token_ids);
    m.def("v2_rejection_sample", &py_v2_rejection_sample,
          py::arg("sampled"), py::arg("num_sampled"),
          py::arg("target_rejected_lse"), py::arg("draft_rejected_lse"),
          py::arg("target_logits"), py::arg("t_local_argmax"),
          py::arg("t_local_max"), py::arg("t_local_sumexp"),
          py::arg("draft_sampled"), py::arg("draft_logits"),
          py::arg("d_local_max"), py::arg("d_local_sumexp"),
          py::arg("cu_num_logits"), py::arg("idx_mapping"),
          py::arg("temperature"), py::arg("seed"), py::arg("pos"),
          py::arg("vocab_num_blocks"));
    m.def("v2_resample", &py_v2_resample, py::arg("rl_argmax"),
          py::arg("rl_max"), py::arg("target_logits"),
          py::arg("target_rejected_lse"), py::arg("draft_logits"),
          py::arg("draft_rejected_lse"), py::arg("rejected_step"),
          py::arg("cu_num_logits"), py::arg("expanded_idx_mapping"),
          py::arg("draft_sampled"), py::arg("temperature"), py::arg("seed"),
          py::arg("pos"), py::arg("vocab_size"));
}
