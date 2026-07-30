// tk_cuda MF-M4 bindings: logits-processor zoo (logits_proc_kernels.cuh) +
// EAGLE spec-decode core (eagle_kernels.cuh). Registered by init_m4(m).
#include "../serving/logits_proc_kernels.cuh"
#include "../serving/eagle_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define M4CK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t m4s() { return at::cuda::getCurrentCUDAStream(); }

// ---- logits processors: (logits fp32 (R,V), per-row params) -> masked fp32 ----
static torch::Tensor lp_alloc(const torch::Tensor& logits) { return torch::empty_like(logits); }

static torch::Tensor py_top_nsigma(torch::Tensor logits, torch::Tensor nsigma) {
    M4CK(logits); M4CK(nsigma); auto o = lp_alloc(logits);
    tmlp::top_nsigma_mask<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(),
        o.data_ptr<float>(), logits.size(1), nsigma.data_ptr<float>());
    return o;
}
static torch::Tensor py_top_a(torch::Tensor logits, torch::Tensor top_a) {
    M4CK(logits); M4CK(top_a); auto o = lp_alloc(logits);
    tmlp::top_a_mask<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(),
        o.data_ptr<float>(), logits.size(1), top_a.data_ptr<float>());
    return o;
}
static torch::Tensor py_epsilon_cutoff(torch::Tensor logits, torch::Tensor eps) {
    M4CK(logits); M4CK(eps); auto o = lp_alloc(logits);
    tmlp::epsilon_cutoff_mask<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(),
        o.data_ptr<float>(), logits.size(1), eps.data_ptr<float>());
    return o;
}
static torch::Tensor py_eta_cutoff(torch::Tensor logits, torch::Tensor eta) {
    M4CK(logits); M4CK(eta); auto o = lp_alloc(logits);
    tmlp::eta_cutoff_mask<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(),
        o.data_ptr<float>(), logits.size(1), eta.data_ptr<float>());
    return o;
}
static torch::Tensor py_xtc(torch::Tensor logits, torch::Tensor thresholds, torch::Tensor apply_xtc) {
    M4CK(logits); M4CK(thresholds); M4CK(apply_xtc); auto o = lp_alloc(logits);
    tmlp::xtc_mask<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(), o.data_ptr<float>(),
        logits.size(1), thresholds.data_ptr<float>(), apply_xtc.data_ptr<int>());
    return o;
}
static torch::Tensor py_quadratic(torch::Tensor logits, torch::Tensor factors, torch::Tensor curves) {
    M4CK(logits); M4CK(factors); M4CK(curves); auto o = lp_alloc(logits);
    tmlp::quadratic_transform<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(),
        o.data_ptr<float>(), logits.size(1), factors.data_ptr<float>(), curves.data_ptr<float>());
    return o;
}
static torch::Tensor py_skew(torch::Tensor probs, torch::Tensor skews) {
    M4CK(probs); M4CK(skews); auto o = lp_alloc(probs);
    tmlp::skew_transform<<<probs.size(0), 256, (256 / 32) * sizeof(float), m4s()>>>(
        probs.data_ptr<float>(), o.data_ptr<float>(), probs.size(1), skews.data_ptr<float>());
    return o;
}
static torch::Tensor py_no_repeat_ngram(torch::Tensor logits, torch::Tensor history,
                                        torch::Tensor hist_len, int64_t ngram) {
    M4CK(logits); M4CK(history); M4CK(hist_len); auto o = lp_alloc(logits);
    tmlp::no_repeat_ngram_mask<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(),
        o.data_ptr<float>(), logits.size(1), history.data_ptr<int>(), hist_len.data_ptr<int>(),
        history.size(1), int(ngram));
    return o;
}
static torch::Tensor py_dry(torch::Tensor logits, torch::Tensor history, torch::Tensor hist_len,
        c10::optional<torch::Tensor> is_breaker, torch::Tensor multiplier, torch::Tensor base_,
        torch::Tensor allowed_length, int64_t max_ngram, int64_t max_occ) {
    M4CK(logits); M4CK(history); M4CK(hist_len); M4CK(multiplier); M4CK(base_); M4CK(allowed_length);
    auto o = lp_alloc(logits);
    const uint8_t* br = is_breaker ? is_breaker->data_ptr<uint8_t>() : nullptr;
    tmlp::dry_penalty<<<logits.size(0), 32, 0, m4s()>>>(logits.data_ptr<float>(), o.data_ptr<float>(),
        logits.size(1), history.data_ptr<int>(), hist_len.data_ptr<int>(), history.size(1), br,
        multiplier.data_ptr<float>(), base_.data_ptr<float>(), allowed_length.data_ptr<int>(),
        int(max_ngram), int(max_occ));
    return o;
}

// ---- EAGLE ----
static torch::Tensor py_rejection_greedy(torch::Tensor target_argmax, torch::Tensor draft,
        torch::Tensor bonus, torch::Tensor cu_num_draft, int64_t batch, int64_t out_stride) {
    M4CK(target_argmax); M4CK(draft); M4CK(bonus); M4CK(cu_num_draft);
    auto out = torch::full({batch, out_stride}, -1, target_argmax.options());
    tmeg::rejection_greedy_sample<<<(batch + 31) / 32, 32, 0, m4s()>>>(out.data_ptr<int64_t>(),
        target_argmax.data_ptr<int64_t>(), draft.data_ptr<int64_t>(), bonus.data_ptr<int64_t>(),
        cu_num_draft.data_ptr<int>(), nullptr, 0, int(batch), int(out_stride), 1);
    return out;
}
static torch::Tensor py_rejection_random(torch::Tensor target_probs, torch::Tensor draft_probs,
        torch::Tensor draft, torch::Tensor bonus, torch::Tensor recovered, torch::Tensor uniform,
        torch::Tensor cu_num_draft, int64_t batch, int64_t out_stride) {
    M4CK(target_probs); M4CK(draft); M4CK(bonus); M4CK(recovered); M4CK(uniform); M4CK(cu_num_draft);
    const int V = target_probs.size(1);
    auto out = torch::full({batch, out_stride}, -1, draft.options());
    tmeg::rejection_random_sample<<<(batch + 31) / 32, 32, 0, m4s()>>>(out.data_ptr<int64_t>(),
        target_probs.data_ptr<float>(), draft_probs.data_ptr<float>(), draft.data_ptr<int64_t>(),
        bonus.data_ptr<int64_t>(), recovered.data_ptr<int64_t>(), uniform.data_ptr<float>(),
        cu_num_draft.data_ptr<int>(), nullptr, 0, 0, int(batch), int(out_stride), 1, V, V);
    return out;
}
static torch::Tensor py_sample_recovered(torch::Tensor target_probs, torch::Tensor draft_probs,
        torch::Tensor draft, torch::Tensor inv_q, torch::Tensor cu_num_draft, int64_t batch) {
    M4CK(target_probs); M4CK(draft_probs); M4CK(draft); M4CK(inv_q); M4CK(cu_num_draft);
    const int TD = draft.size(0), V = target_probs.size(1);
    auto out = torch::empty({TD}, draft.options());
    tmeg::sample_recovered_tokens<<<TD, 32, 0, m4s()>>>(out.data_ptr<int64_t>(),
        target_probs.data_ptr<float>(), draft_probs.data_ptr<float>(), draft.data_ptr<int64_t>(),
        inv_q.data_ptr<float>(), cu_num_draft.data_ptr<int>(), 0, int(batch), TD, V, V, V, V);
    return out;
}
static std::tuple<torch::Tensor, torch::Tensor> py_eagle_prepare(torch::Tensor cu_num_draft,
        torch::Tensor valid_count, torch::Tensor query_start_loc) {
    M4CK(cu_num_draft); M4CK(valid_count); M4CK(query_start_loc);
    const int R = cu_num_draft.numel();
    auto tis = torch::empty({R}, cu_num_draft.options());
    auto nrt = torch::empty({R}, cu_num_draft.options());
    tmeg::eagle_prepare_inputs_padded<<<(R + 31) / 32, 32, 0, m4s()>>>(tis.data_ptr<int>(),
        nrt.data_ptr<int>(), cu_num_draft.data_ptr<int>(), valid_count.data_ptr<int>(),
        query_start_loc.data_ptr<int>(), R);
    return {tis, nrt};
}

void init_m4(py::module_& m) {
    m.def("top_nsigma_mask", &py_top_nsigma);
    m.def("top_a_mask", &py_top_a);
    m.def("epsilon_cutoff_mask", &py_epsilon_cutoff);
    m.def("eta_cutoff_mask", &py_eta_cutoff);
    m.def("xtc_mask", &py_xtc);
    m.def("quadratic_transform", &py_quadratic);
    m.def("skew_transform", &py_skew);
    m.def("no_repeat_ngram_mask", &py_no_repeat_ngram);
    m.def("dry_penalty", &py_dry, py::arg("logits"), py::arg("history"), py::arg("hist_len"),
          py::arg("is_breaker") = py::none(), py::arg("multiplier"), py::arg("base"),
          py::arg("allowed_length"), py::arg("max_ngram") = 8, py::arg("max_occ") = 8);
    m.def("rejection_greedy_sample", &py_rejection_greedy);
    m.def("rejection_random_sample", &py_rejection_random);
    m.def("sample_recovered_tokens", &py_sample_recovered);
    m.def("eagle_prepare_inputs", &py_eagle_prepare);
}
