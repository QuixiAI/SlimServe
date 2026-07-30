// tk_cuda W3 elementwise/norm/training bindings: torch wrappers over the
// validated kernels in kernels/elementwise/tm_elementwise_kernels.cuh.
// Registered into the _C module by init_elementwise(m) from tm_cuda_ext.cu.
#include "../elementwise/tm_elementwise_kernels.cuh"
#include "../elementwise/tm_norm_quant_kernels.cuh"   // MF-M2 norm/AZP/group quant
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;
using namespace tme;

#define ECK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t estream() { return at::cuda::getCurrentCUDAStream(); }

// Dispatch over the three supported element types; binds `scalar_t` and `PTR`.
#define DISPATCH_T(t, ...) do {                                                   \
    if (t.scalar_type() == torch::kFloat)        { using scalar_t = float;          __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kHalf)    { using scalar_t = __half;         __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kBFloat16){ using scalar_t = __nv_bfloat16;  __VA_ARGS__ } \
    else TORCH_CHECK(false, "unsupported dtype (want fp32/fp16/bf16)");           \
} while (0)
template <typename S> static const S* cp(const torch::Tensor& t) { return reinterpret_cast<const S*>(t.data_ptr()); }
template <typename S> static S* mp(torch::Tensor& t) { return reinterpret_cast<S*>(t.data_ptr()); }

static inline int rows_of(const torch::Tensor& t) {
    return int(t.numel() / t.size(-1));
}

// ---- norms ----
static torch::Tensor py_rms_norm(torch::Tensor x, torch::Tensor w, double eps) {
    ECK(x); ECK(w);
    const int D = x.size(-1), M = rows_of(x);
    auto o = torch::empty_like(x);
    DISPATCH_T(x, rms_norm_fwd<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(w), mp<scalar_t>(o), D, float(eps)););
    return o;
}
static torch::Tensor py_rms_norm_bwd_dx(torch::Tensor x, torch::Tensor w, torch::Tensor dy,
                                        torch::Tensor rstd) {
    ECK(x); ECK(w); ECK(dy); ECK(rstd);
    const int D = x.size(-1), M = rows_of(x);
    auto dx = torch::empty_like(x);
    DISPATCH_T(x, rms_norm_bwd_dx<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(w), cp<scalar_t>(dy), rstd.data_ptr<float>(),
        mp<scalar_t>(dx), D););
    return dx;
}
static std::tuple<torch::Tensor, torch::Tensor> py_rms_norm_bwd_fused(
        torch::Tensor x, torch::Tensor w, torch::Tensor dy, double eps) {
    ECK(x); ECK(w); ECK(dy);
    const int D = x.size(-1), M = rows_of(x);
    auto dx = torch::empty_like(x);
    auto dw = torch::zeros({D}, x.options().dtype(torch::kFloat));
    DISPATCH_T(x, rms_norm_bwd_fused<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(w), cp<scalar_t>(dy), mp<scalar_t>(dx),
        dw.data_ptr<float>(), D, float(eps)););
    return {dx, dw};
}
static torch::Tensor py_layernorm(torch::Tensor x, torch::Tensor w, torch::Tensor b, double eps) {
    ECK(x); ECK(w); ECK(b);
    const int D = x.size(-1), M = rows_of(x);
    auto o = torch::empty_like(x);
    DISPATCH_T(x, layernorm_fwd<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(w), cp<scalar_t>(b), mp<scalar_t>(o), D, float(eps)););
    return o;
}
static torch::Tensor py_layernorm_bwd_dx(torch::Tensor x, torch::Tensor w, torch::Tensor dy,
                                         torch::Tensor mean, torch::Tensor rstd) {
    ECK(x); ECK(w); ECK(dy); ECK(mean); ECK(rstd);
    const int D = x.size(-1), M = rows_of(x);
    auto dx = torch::empty_like(x);
    DISPATCH_T(x, layernorm_bwd_dx<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(w), cp<scalar_t>(dy), mean.data_ptr<float>(),
        rstd.data_ptr<float>(), mp<scalar_t>(dx), D););
    return dx;
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_layernorm_bwd_fused(
        torch::Tensor x, torch::Tensor w, torch::Tensor dy, double eps) {
    ECK(x); ECK(w); ECK(dy);
    const int D = x.size(-1), M = rows_of(x);
    auto dx = torch::empty_like(x);
    auto dw = torch::zeros({D}, x.options().dtype(torch::kFloat));
    auto db = torch::zeros({D}, x.options().dtype(torch::kFloat));
    DISPATCH_T(x, layernorm_bwd_fused<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(w), cp<scalar_t>(dy), mp<scalar_t>(dx),
        dw.data_ptr<float>(), db.data_ptr<float>(), D, float(eps)););
    return {dx, dw, db};
}

// ---- add_norm family ----
static std::tuple<torch::Tensor, torch::Tensor> py_rms_norm_add(
        torch::Tensor x, torch::Tensor res, torch::Tensor w, double eps) {
    ECK(x); ECK(res); ECK(w);
    const int D = x.size(-1), M = rows_of(x);
    auto o = torch::empty_like(x), ro = torch::empty_like(x);
    DISPATCH_T(x, (rms_norm_add_k<scalar_t, false, false><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(res), cp<scalar_t>(w), mp<scalar_t>(o),
        nullptr, mp<scalar_t>(ro), nullptr, D, float(eps), 0.0f)););
    return {o, ro};
}
static std::tuple<torch::Tensor, torch::Tensor> py_layernorm_add(
        torch::Tensor x, torch::Tensor res, torch::Tensor w, torch::Tensor b, double eps) {
    ECK(x); ECK(res); ECK(w); ECK(b);
    const int D = x.size(-1), M = rows_of(x);
    auto o = torch::empty_like(x), ro = torch::empty_like(x);
    DISPATCH_T(x, (layernorm_add_k<scalar_t, false, false><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(res), cp<scalar_t>(w), cp<scalar_t>(b),
        mp<scalar_t>(o), nullptr, mp<scalar_t>(ro), nullptr, D, float(eps), 0.0f)););
    return {o, ro};
}
static std::tuple<torch::Tensor, torch::Tensor> py_rms_norm_add_fp8(
        torch::Tensor x, torch::Tensor res, torch::Tensor w, double eps, double inv_scale) {
    ECK(x); ECK(res); ECK(w);
    const int D = x.size(-1), M = rows_of(x);
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kUInt8));
    auto ro = torch::empty_like(x);
    DISPATCH_T(x, (rms_norm_add_k<scalar_t, true, false><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(res), cp<scalar_t>(w), nullptr,
        codes.data_ptr<uint8_t>(), mp<scalar_t>(ro), nullptr, D, float(eps), float(inv_scale))););
    return {codes, ro};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_rms_norm_add_fp8_dyn(
        torch::Tensor x, torch::Tensor res, torch::Tensor w, double eps) {
    ECK(x); ECK(res); ECK(w);
    const int D = x.size(-1), M = rows_of(x);
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kUInt8));
    auto ro = torch::empty_like(x);
    auto scale = torch::empty({M}, x.options().dtype(torch::kFloat));
    DISPATCH_T(x, (rms_norm_add_k<scalar_t, true, true><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(res), cp<scalar_t>(w), nullptr,
        codes.data_ptr<uint8_t>(), mp<scalar_t>(ro), scale.data_ptr<float>(),
        D, float(eps), 0.0f)););
    return {codes, ro, scale};
}
static std::tuple<torch::Tensor, torch::Tensor> py_layernorm_add_fp8(
        torch::Tensor x, torch::Tensor res, torch::Tensor w, torch::Tensor b,
        double eps, double inv_scale) {
    ECK(x); ECK(res); ECK(w); ECK(b);
    const int D = x.size(-1), M = rows_of(x);
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kUInt8));
    auto ro = torch::empty_like(x);
    DISPATCH_T(x, (layernorm_add_k<scalar_t, true, false><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(res), cp<scalar_t>(w), cp<scalar_t>(b), nullptr,
        codes.data_ptr<uint8_t>(), mp<scalar_t>(ro), nullptr, D, float(eps), float(inv_scale))););
    return {codes, ro};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_layernorm_add_fp8_dyn(
        torch::Tensor x, torch::Tensor res, torch::Tensor w, torch::Tensor b, double eps) {
    ECK(x); ECK(res); ECK(w); ECK(b);
    const int D = x.size(-1), M = rows_of(x);
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kUInt8));
    auto ro = torch::empty_like(x);
    auto scale = torch::empty({M}, x.options().dtype(torch::kFloat));
    DISPATCH_T(x, (layernorm_add_k<scalar_t, true, true><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(res), cp<scalar_t>(w), cp<scalar_t>(b), nullptr,
        codes.data_ptr<uint8_t>(), mp<scalar_t>(ro), scale.data_ptr<float>(),
        D, float(eps), 0.0f)););
    return {codes, ro, scale};
}

// ---- softmax / activations ----
static torch::Tensor py_softmax(torch::Tensor x) {
    ECK(x);
    const int D = x.size(-1), M = rows_of(x);
    auto o = torch::empty_like(x);
    DISPATCH_T(x, softmax_fwd<scalar_t><<<M, 32, 0, estream()>>>(
        cp<scalar_t>(x), mp<scalar_t>(o), D););
    return o;
}
static torch::Tensor py_gelu(torch::Tensor x) {
    ECK(x);
    const long n = x.numel();
    auto o = torch::empty_like(x);
    DISPATCH_T(x, gelu_fwd<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(x), mp<scalar_t>(o), n););
    return o;
}
static torch::Tensor py_gelu_bwd(torch::Tensor x, torch::Tensor dy) {
    ECK(x); ECK(dy);
    const long n = x.numel();
    auto dx = torch::empty_like(x);
    DISPATCH_T(x, gelu_bwd<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(dy), mp<scalar_t>(dx), n););
    return dx;
}
static torch::Tensor py_glu(torch::Tensor x, torch::Tensor gate, int64_t mode,
                            double alpha, double limit) {
    ECK(x); ECK(gate);
    const long n = x.numel();
    auto o = torch::empty_like(x);
    DISPATCH_T(x, glu_fwd<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(gate), mp<scalar_t>(o), n, int(mode),
        float(alpha), float(limit)););
    return o;
}
static std::tuple<torch::Tensor, torch::Tensor> py_glu_bwd(
        torch::Tensor x, torch::Tensor gate, torch::Tensor dc, int64_t mode,
        double alpha, double limit) {
    ECK(x); ECK(gate); ECK(dc);
    const long n = x.numel();
    auto da = torch::empty_like(x), db = torch::empty_like(x);
    DISPATCH_T(x, glu_bwd<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(gate), cp<scalar_t>(dc), mp<scalar_t>(da),
        mp<scalar_t>(db), n, int(mode), float(alpha), float(limit)););
    return {da, db};
}
static torch::Tensor py_dropout(torch::Tensor x, int64_t seed, double p) {
    ECK(x);
    const long n = x.numel();
    auto o = torch::empty_like(x);
    const float inv_keep = 1.0f / (1.0f - float(p));
    DISPATCH_T(x, dropout_fwd<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(x), mp<scalar_t>(o), uint32_t(seed), float(p), inv_keep, n););
    return o;
}
static torch::Tensor py_dropout_bwd(torch::Tensor dy, int64_t seed, double p) {
    ECK(dy);
    const long n = dy.numel();
    auto dx = torch::empty_like(dy);
    const float inv_keep = 1.0f / (1.0f - float(p));
    DISPATCH_T(dy, dropout_bwd<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(dy), mp<scalar_t>(dx), uint32_t(seed), float(p), inv_keep, n););
    return dx;
}

// ---- cross entropy (mw routed when few rows relative to vocab) ----
static bool ce_use_mw(int rows, int V, int64_t use_mw) {
    if (use_mw >= 0) return use_mw > 0;
    return rows < 512 && V >= 4096;
}
static std::tuple<torch::Tensor, torch::Tensor> py_cross_entropy(
        torch::Tensor logits, torch::Tensor targets, int64_t ignore_index,
        double label_smoothing, double z_loss, double softcap, int64_t use_mw) {
    ECK(logits); ECK(targets);
    const int V = logits.size(-1), Tn = rows_of(logits);
    auto loss = torch::empty({Tn}, logits.options().dtype(torch::kFloat));
    auto lse = torch::empty({Tn}, logits.options().dtype(torch::kFloat));
    const bool mw = ce_use_mw(Tn, V, use_mw);
    DISPATCH_T(logits, {
        if (mw) cross_entropy_fwd_mw<scalar_t><<<Tn, 128, 0, estream()>>>(
            cp<scalar_t>(logits), targets.data_ptr<int>(), loss.data_ptr<float>(),
            lse.data_ptr<float>(), V, int(ignore_index), float(label_smoothing),
            float(z_loss), float(softcap));
        else cross_entropy_fwd<scalar_t><<<Tn, 32, 0, estream()>>>(
            cp<scalar_t>(logits), targets.data_ptr<int>(), loss.data_ptr<float>(),
            lse.data_ptr<float>(), V, int(ignore_index), float(label_smoothing),
            float(z_loss), float(softcap));
    });
    return {loss, lse};
}
static torch::Tensor py_cross_entropy_bwd(
        torch::Tensor logits, torch::Tensor targets, torch::Tensor lse, torch::Tensor grad_out,
        int64_t ignore_index, double label_smoothing, double z_loss, double softcap,
        int64_t use_mw) {
    ECK(logits); ECK(targets); ECK(lse); ECK(grad_out);
    const int V = logits.size(-1), Tn = rows_of(logits);
    auto grad = torch::empty_like(logits);
    const bool mw = ce_use_mw(Tn, V, use_mw);
    DISPATCH_T(logits, {
        if (mw) cross_entropy_bwd_mw<scalar_t><<<Tn, 128, 0, estream()>>>(
            cp<scalar_t>(logits), targets.data_ptr<int>(), lse.data_ptr<float>(),
            grad_out.data_ptr<float>(), mp<scalar_t>(grad), V, int(ignore_index),
            float(label_smoothing), float(z_loss), float(softcap));
        else cross_entropy_bwd<scalar_t><<<Tn, 32, 0, estream()>>>(
            cp<scalar_t>(logits), targets.data_ptr<int>(), lse.data_ptr<float>(),
            grad_out.data_ptr<float>(), mp<scalar_t>(grad), V, int(ignore_index),
            float(label_smoothing), float(z_loss), float(softcap));
    });
    return grad;
}

// ---- embedding ----
static torch::Tensor py_embedding(torch::Tensor token_ids, torch::Tensor table,
                                  c10::optional<torch::Tensor> pos_table, double scale) {
    ECK(token_ids); ECK(table);
    const int n_tok = token_ids.numel(), D = table.size(-1), vocab = table.size(0);
    auto out = torch::empty({n_tok, D}, table.options());
    const bool up = pos_table.has_value();
    if (up) { ECK((*pos_table)); }
    DISPATCH_T(table, embedding_lookup<scalar_t><<<n_tok, 128, 0, estream()>>>(
        token_ids.data_ptr<int>(), cp<scalar_t>(table),
        up ? cp<scalar_t>(*pos_table) : cp<scalar_t>(table), mp<scalar_t>(out),
        D, vocab, float(scale), up ? 1 : 0););
    return out;
}
static torch::Tensor py_embedding_bwd(torch::Tensor token_ids, torch::Tensor dY,
                                      int64_t vocab, double scale) {
    ECK(token_ids); ECK(dY);
    const int n_tok = token_ids.numel(), D = dY.size(-1);
    auto dtable = torch::zeros({vocab, D}, dY.options().dtype(torch::kFloat));
    DISPATCH_T(dY, embedding_backward<scalar_t><<<n_tok, 128, 0, estream()>>>(
        token_ids.data_ptr<int>(), cp<scalar_t>(dY), dtable.data_ptr<float>(),
        D, int(vocab), float(scale)););
    return dtable;
}
static torch::Tensor py_embedding_bwd_sorted(torch::Tensor sorted_ids, torch::Tensor perm,
                                             torch::Tensor dY, int64_t vocab, double scale) {
    ECK(sorted_ids); ECK(perm); ECK(dY);
    const int n_tok = sorted_ids.numel(), D = dY.size(-1);
    auto dtable = torch::zeros({vocab, D}, dY.options().dtype(torch::kFloat));
    DISPATCH_T(dY, embedding_backward_sorted<scalar_t><<<n_tok, 128, 0, estream()>>>(
        sorted_ids.data_ptr<int>(), perm.data_ptr<int>(), cp<scalar_t>(dY),
        dtable.data_ptr<float>(), D, int(vocab), n_tok, float(scale)););
    return dtable;
}
static torch::Tensor py_merge_multimodal(torch::Tensor text, torch::Tensor modal,
                                         torch::Tensor src) {
    ECK(text); ECK(modal); ECK(src);
    const int n_tok = text.size(0), D = text.size(-1), n_modal = modal.size(0);
    auto out = torch::empty_like(text);
    DISPATCH_T(text, merge_multimodal_spans<scalar_t><<<n_tok, 128, 0, estream()>>>(
        cp<scalar_t>(text), cp<scalar_t>(modal), src.data_ptr<int>(), mp<scalar_t>(out),
        D, n_modal););
    return out;
}
static torch::Tensor py_build_multimodal_src(torch::Tensor span_offsets, torch::Tensor span_lengths,
                                             torch::Tensor modal_starts, int64_t num_tok) {
    ECK(span_offsets); ECK(span_lengths); ECK(modal_starts);
    auto src = torch::empty({num_tok}, span_offsets.options());
    build_multimodal_src<<<unsigned((num_tok + 127) / 128), 128, 0, estream()>>>(
        span_offsets.data_ptr<int>(), span_lengths.data_ptr<int>(),
        modal_starts.data_ptr<int>(), src.data_ptr<int>(),
        int(span_offsets.numel()), int(num_tok));
    return src;
}

// ---- hadamard / adamw / add ----
template <typename S, int D, int LPR>
static void launch_hadamard(const torch::Tensor& x, torch::Tensor& out, float scale, int nrows) {
    constexpr int rpb = 4 * (32 / LPR);   // 128-thread blocks, 4 warps
    hadamard_k<S, D, LPR><<<unsigned((nrows + rpb - 1) / rpb), 128, 0, estream()>>>(
        cp<S>(x), mp<S>(out), scale, nrows);
}
static torch::Tensor py_hadamard(torch::Tensor x, double scale) {
    ECK(x);
    const int D = x.size(-1), nrows = rows_of(x);
    auto out = torch::empty_like(x);
    DISPATCH_T(x, {
        if (D == 64) launch_hadamard<scalar_t, 64, 8>(x, out, float(scale), nrows);
        else if (D == 128) launch_hadamard<scalar_t, 128, 16>(x, out, float(scale), nrows);
        else if (D == 256) launch_hadamard<scalar_t, 256, 32>(x, out, float(scale), nrows);
        else if (D == 512) launch_hadamard<scalar_t, 512, 32>(x, out, float(scale), nrows);
        else TORCH_CHECK(false, "hadamard: D must be 64/128/256/512");
    });
    return out;
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_adamw(
        torch::Tensor param, torch::Tensor grad, torch::Tensor m, torch::Tensor v,
        double lr, double beta1, double beta2, double eps, double wd, int64_t step) {
    ECK(param); ECK(grad); ECK(m); ECK(v);
    const long n = param.numel();
    auto po = torch::empty_like(param);
    auto mo = torch::empty_like(m), vo = torch::empty_like(v);
    const float bc1 = 1.0f - powf(float(beta1), float(step));
    const float bc2 = 1.0f - powf(float(beta2), float(step));
    DISPATCH_T(param, adamw_step<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(param), cp<scalar_t>(grad), m.data_ptr<float>(), v.data_ptr<float>(),
        mp<scalar_t>(po), mo.data_ptr<float>(), vo.data_ptr<float>(),
        float(lr), float(beta1), float(beta2), float(eps), float(wd), bc1, bc2, n););
    return {po, mo, vo};
}
static torch::Tensor py_add(torch::Tensor x, torch::Tensor y) {
    ECK(x); ECK(y);
    const long n = x.numel();
    auto o = torch::empty_like(x);
    DISPATCH_T(x, add_ew<scalar_t><<<unsigned((n + 255) / 256), 256, 0, estream()>>>(
        cp<scalar_t>(x), cp<scalar_t>(y), mp<scalar_t>(o), n););
    return o;
}

// ---- MF-M2: norm+quant epilogues, AZP int8, group int8 (half input) ----
static const half* nqhp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }

// rms_norm_quant -> (codes u8, scale f32 [dyn], res_out [resid]).
static std::vector<torch::Tensor> py_rms_norm_quant(torch::Tensor x, torch::Tensor weight,
        bool fp8, bool dyn, double eps, double inv_static, c10::optional<torch::Tensor> residual) {
    ECK(x); ECK(weight);
    const int D = x.size(-1), M = rows_of(x);
    const bool resid = residual.has_value();
    if (resid) ECK((*residual));
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kUInt8));
    auto scale = torch::empty({dyn ? M : 1}, x.options().dtype(torch::kFloat));
    auto res_out = resid ? torch::empty_like(x) : torch::empty({0}, x.options());
    const half* rp = resid ? nqhp(*residual) : nullptr;
    half* rop = resid ? reinterpret_cast<half*>(res_out.data_ptr()) : nullptr;
    auto s = estream();
    #define RNQ(FP8, DYN, RES) tmnq::rms_norm_quant<half, FP8, DYN, RES><<<M, 256, 0, s>>>( \
        codes.data_ptr<uint8_t>(), scale.data_ptr<float>(), rop, nqhp(x), rp, nqhp(weight), \
        D, float(eps), float(inv_static))
    if (resid) { if (fp8) { dyn ? RNQ(true,true,true) : RNQ(true,false,true); }
                 else      { dyn ? RNQ(false,true,true) : RNQ(false,false,true); } }
    else       { if (fp8) { dyn ? RNQ(true,true,false) : RNQ(true,false,false); }
                 else      { dyn ? RNQ(false,true,false) : RNQ(false,false,false); } }
    #undef RNQ
    std::vector<torch::Tensor> out = {codes};
    if (dyn) out.push_back(scale);
    if (resid) out.push_back(res_out);
    return out;
}
static std::vector<torch::Tensor> py_azp_int8_quant(torch::Tensor x, bool dyn,
        double scale_static, int64_t azp_static) {
    ECK(x);
    const int D = x.size(-1), M = rows_of(x);
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kChar));
    auto scale = torch::empty({dyn ? M : 1}, x.options().dtype(torch::kFloat));
    auto azp = torch::empty({dyn ? M : 1}, x.options().dtype(torch::kInt));
    auto s = estream();
    if (dyn) tmnq::azp_int8_quant<half, true><<<M, 32, 0, s>>>(codes.data_ptr<int8_t>(),
        scale.data_ptr<float>(), azp.data_ptr<int>(), nqhp(x), D, 0, 0);
    else tmnq::azp_int8_quant<half, false><<<M, 32, 0, s>>>(codes.data_ptr<int8_t>(),
        scale.data_ptr<float>(), azp.data_ptr<int>(), nqhp(x), D, float(scale_static), int(azp_static));
    return {codes, scale, azp};
}
static std::tuple<torch::Tensor, torch::Tensor> py_per_token_group_int8(torch::Tensor x,
        int64_t group_size, double eps) {
    ECK(x);
    const int H = x.size(-1), T = rows_of(x), NG = H / int(group_size);
    auto codes = torch::empty(x.sizes(), x.options().dtype(torch::kChar));
    auto scales = torch::empty({T, NG}, x.options().dtype(torch::kFloat));
    dim3 g(NG, T);
    tmnq::per_token_group_int8_quant<half><<<g, 32, 0, estream()>>>(codes.data_ptr<int8_t>(),
        scales.data_ptr<float>(), nqhp(x), H, int(group_size), NG, float(eps));
    return {codes, scales};
}

void init_elementwise(py::module_& m) {
    m.def("rms_norm", &py_rms_norm, py::arg("x"), py::arg("w"), py::arg("eps") = 1e-5);
    m.def("rms_norm_bwd_dx", &py_rms_norm_bwd_dx);
    m.def("rms_norm_bwd_fused", &py_rms_norm_bwd_fused, py::arg("x"), py::arg("w"),
          py::arg("dy"), py::arg("eps") = 1e-5);
    m.def("layernorm", &py_layernorm, py::arg("x"), py::arg("w"), py::arg("b"),
          py::arg("eps") = 1e-5);
    m.def("layernorm_bwd_dx", &py_layernorm_bwd_dx);
    m.def("layernorm_bwd_fused", &py_layernorm_bwd_fused, py::arg("x"), py::arg("w"),
          py::arg("dy"), py::arg("eps") = 1e-5);
    m.def("rms_norm_add", &py_rms_norm_add, py::arg("x"), py::arg("residual"), py::arg("w"),
          py::arg("eps") = 1e-5);
    m.def("layernorm_add", &py_layernorm_add, py::arg("x"), py::arg("residual"), py::arg("w"),
          py::arg("b"), py::arg("eps") = 1e-5);
    m.def("rms_norm_add_fp8", &py_rms_norm_add_fp8, py::arg("x"), py::arg("residual"),
          py::arg("w"), py::arg("eps"), py::arg("inv_scale"));
    m.def("rms_norm_add_fp8_dyn", &py_rms_norm_add_fp8_dyn, py::arg("x"), py::arg("residual"),
          py::arg("w"), py::arg("eps") = 1e-5);
    m.def("layernorm_add_fp8", &py_layernorm_add_fp8, py::arg("x"), py::arg("residual"),
          py::arg("w"), py::arg("b"), py::arg("eps"), py::arg("inv_scale"));
    m.def("layernorm_add_fp8_dyn", &py_layernorm_add_fp8_dyn, py::arg("x"), py::arg("residual"),
          py::arg("w"), py::arg("b"), py::arg("eps") = 1e-5);
    m.def("softmax", &py_softmax);
    m.def("gelu", &py_gelu);
    m.def("gelu_bwd", &py_gelu_bwd);
    m.def("glu", &py_glu, py::arg("x"), py::arg("gate"), py::arg("mode"),
          py::arg("alpha") = 1.702, py::arg("limit") = 7.0);
    m.def("glu_bwd", &py_glu_bwd, py::arg("x"), py::arg("gate"), py::arg("dc"), py::arg("mode"),
          py::arg("alpha") = 1.702, py::arg("limit") = 7.0);
    m.def("dropout", &py_dropout, py::arg("x"), py::arg("seed"), py::arg("p"));
    m.def("dropout_bwd", &py_dropout_bwd, py::arg("dy"), py::arg("seed"), py::arg("p"));
    m.def("cross_entropy", &py_cross_entropy, py::arg("logits"), py::arg("targets"),
          py::arg("ignore_index") = -100, py::arg("label_smoothing") = 0.0,
          py::arg("z_loss") = 0.0, py::arg("softcap") = 0.0, py::arg("use_mw") = -1);
    m.def("cross_entropy_bwd", &py_cross_entropy_bwd, py::arg("logits"), py::arg("targets"),
          py::arg("lse"), py::arg("grad_out"), py::arg("ignore_index") = -100,
          py::arg("label_smoothing") = 0.0, py::arg("z_loss") = 0.0, py::arg("softcap") = 0.0,
          py::arg("use_mw") = -1);
    m.def("embedding", &py_embedding, py::arg("token_ids"), py::arg("table"),
          py::arg("pos_table") = py::none(), py::arg("scale") = 1.0);
    m.def("embedding_bwd", &py_embedding_bwd, py::arg("token_ids"), py::arg("dY"),
          py::arg("vocab"), py::arg("scale") = 1.0);
    m.def("embedding_bwd_sorted", &py_embedding_bwd_sorted, py::arg("sorted_ids"),
          py::arg("perm"), py::arg("dY"), py::arg("vocab"), py::arg("scale") = 1.0);
    m.def("merge_multimodal", &py_merge_multimodal);
    m.def("build_multimodal_src", &py_build_multimodal_src);
    m.def("hadamard", &py_hadamard, py::arg("x"), py::arg("scale") = 1.0);
    m.def("adamw", &py_adamw, py::arg("param"), py::arg("grad"), py::arg("m"), py::arg("v"),
          py::arg("lr"), py::arg("beta1") = 0.9, py::arg("beta2") = 0.999,
          py::arg("eps") = 1e-8, py::arg("wd") = 0.0, py::arg("step") = 1);
    m.def("add", &py_add);
    // MF-M2 norm/AZP/group quant
    m.def("rms_norm_quant", &py_rms_norm_quant, py::arg("x"), py::arg("weight"),
          py::arg("fp8") = true, py::arg("dyn") = true, py::arg("eps") = 1e-5,
          py::arg("inv_static") = 1.0, py::arg("residual") = py::none());
    m.def("azp_int8_quant", &py_azp_int8_quant, py::arg("x"), py::arg("dyn") = true,
          py::arg("scale_static") = 1.0, py::arg("azp_static") = 0);
    m.def("per_token_group_int8_quant", &py_per_token_group_int8, py::arg("x"),
          py::arg("group_size"), py::arg("eps") = 1e-6);
}
