// tk_cuda torch extension: python-facing wrappers over the validated W1/W2
// quant kernels (kernels/quant/tm_kernels.cuh). API mirrors ThunderMittens'
// tk package: qgemv/qgemm/qflux_gelu take the SAME packed uint8 tensors that
// quant.py produces (byte-identical across Metal and CUDA).
//
// Format dispatch: runtime string -> template instantiation via the X-macro
// list below (all 29 weight formats).
#include "tm_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using namespace tmq;

#define CHECK_CUDA_T(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")

#define TMQ_FORMATS(X) \
    X(q8_0) X(q4_0) X(q4_1) X(q5_0) X(q5_1) X(kU4B8) X(kU4) X(hqq) \
    X(fp8_e4m3) X(e5m2) X(fp8_block) X(fp4_e2m1) X(mxfp8) X(mxfp4) X(nvfp4) \
    X(mxfp6_e3m2) X(mxfp6_e2m3) X(bitnet) \
    X(q2_K) X(q3_K) X(q4_K) X(q5_K) X(q6_K) \
    X(iq4_nl) X(iq4_xs) X(iq2_xxs) X(iq2_xs) X(iq3_xxs) X(iq1_s)

static int format_block_k(const std::string& f) {
#define X(F) if (f == #F) return F::block_k;
    TMQ_FORMATS(X)
#undef X
    TORCH_CHECK(false, "unknown quant format: ", f);
}

// ---- qgemv: D(N) = dequant(Wq(N, K/bk, bytes)) @ x(K) ----
torch::Tensor py_qgemv(torch::Tensor Wq, torch::Tensor x, const std::string& fmt) {
    CHECK_CUDA_T(Wq); CHECK_CUDA_T(x);
    TORCH_CHECK(x.scalar_type() == torch::kHalf, "x must be fp16");
    const int N = Wq.size(0);
    const int K = x.numel();
    auto D = torch::empty({N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
#define X(F) if (fmt == #F) { qgemv<F><<<N, 32, 0, stream>>>( \
        reinterpret_cast<half*>(D.data_ptr()), Wq.data_ptr<uint8_t>(), \
        reinterpret_cast<const half*>(x.data_ptr()), N, K); return D; }
    TMQ_FORMATS(X)
#undef X
    TORCH_CHECK(false, "unknown quant format: ", fmt);
}

// ---- qgemm / qflux_gelu: Y(M,N) = X(M,K) @ dequant(Wq)^T [+bias, gelu] ----
// Superblock formats (block_k > 64) route through full dequant + fp16_raw.
template<bool FLUX>
static torch::Tensor qmm_impl(torch::Tensor X, torch::Tensor Wq, const std::string& fmt,
                              const c10::optional<torch::Tensor>& bias) {
    CHECK_CUDA_T(X); CHECK_CUDA_T(Wq);
    TORCH_CHECK(X.scalar_type() == torch::kHalf, "X must be fp16");
    const int M = X.size(0), K = X.size(1), N = Wq.size(0);
    TORCH_CHECK(K % 16 == 0 && N % 16 == 0, "K, N must be multiples of 16");
    auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(N / 16, (M + 15) / 16);
    const float* bp = nullptr;
    if (FLUX) {
        TORCH_CHECK(bias.has_value() && bias->scalar_type() == torch::kFloat, "flux needs fp32 bias");
        bp = bias->data_ptr<float>();
    }
    // Decode shapes (small M) leave most SMs idle in the warp-per-tile grid;
    // route the plain GEMM through qgemm_ksplit (K sliced across blockIdx.z,
    // fp32 atomic combine into a zeroed Y). FLUX keeps the single-pass grid —
    // its bias+gelu epilogue can't ride the partial-sum accumulation.
    const long tiles = long(N / 16) * ((M + 15) / 16);
    const bool use_ksplit = !FLUX && tiles < 832;   // < ~half the 82-SM chip
    torch::Tensor Wf;   // keep alive
    auto launch = [&](const uint8_t* w, auto fmt_tag) {
        using FMT = decltype(fmt_tag);
        if constexpr (FLUX) {
            qflux_gelu<FMT><<<grid, 32, 0, stream>>>(Y.data_ptr<float>(),
                reinterpret_cast<const half*>(X.data_ptr()), w, bp, M, N, K);
        } else if (use_ksplit) {
            const int chunk = qgemm_pick_kchunk(M, N, K, FMT::block_k);
            dim3 gz(N / 16, (M + 15) / 16, (K + chunk - 1) / chunk);
            cudaMemsetAsync(Y.data_ptr<float>(), 0, sizeof(float) * size_t(M) * N, stream);
            qgemm_ksplit<FMT><<<gz, 32, 0, stream>>>(Y.data_ptr<float>(),
                reinterpret_cast<const half*>(X.data_ptr()), w, M, N, K, chunk);
        } else {
            qgemm<FMT><<<grid, 32, 0, stream>>>(Y.data_ptr<float>(),
                reinterpret_cast<const half*>(X.data_ptr()), w, M, N, K);
        }
    };
#define X(F) if (fmt == #F) { \
        if constexpr (F::block_k > 64) { \
            Wf = torch::empty({N, K}, X.options()); \
            dequant_to_fp16<F><<<(int)((size_t(N)*K + 255)/256), 256, 0, stream>>>( \
                reinterpret_cast<half*>(Wf.data_ptr()), Wq.data_ptr<uint8_t>(), N, K); \
            launch(reinterpret_cast<const uint8_t*>(Wf.data_ptr()), fp16_raw{}); \
        } else launch(Wq.data_ptr<uint8_t>(), F{}); \
        return Y; }
    TMQ_FORMATS(X)
#undef X
    TORCH_CHECK(false, "unknown quant format: ", fmt);
}
// Dequant W (any format) to a fresh fp16 (N,K) buffer.
static torch::Tensor dequant_weights_fp16(torch::Tensor Wq, const std::string& fmt,
                                          int N, int K, cudaStream_t stream,
                                          const torch::TensorOptions& opts) {
    auto Wf = torch::empty({N, K}, opts.dtype(torch::kHalf));
    const int blocks = int((size_t(N) * K + 255) / 256);
#define X(F) if (fmt == #F) { dequant_to_fp16<F><<<blocks, 256, 0, stream>>>( \
        reinterpret_cast<half*>(Wf.data_ptr()), Wq.data_ptr<uint8_t>(), N, K); return Wf; }
    TMQ_FORMATS(X)
#undef X
    TORCH_CHECK(false, "unknown quant format: ", fmt);
}

torch::Tensor py_qgemm(torch::Tensor X, torch::Tensor Wq, const std::string& fmt) {
    CHECK_CUDA_T(X); CHECK_CUDA_T(Wq);
    TORCH_CHECK(X.scalar_type() == torch::kHalf, "X must be fp16");
    const int M = X.size(0), K = X.size(1), N = Wq.size(0);
    // Prefill (M >= 64): the naive per-tile kernel is ~7 TFLOP/s; dequant W to
    // fp16 once and hand the GEMM to cuBLAS (torch::matmul) — ~40-62 TFLOP/s,
    // a 6-8x win. The full-W materialization is amortized across the M rows;
    // decode (M<64) never takes this path (it uses GEMV / ksplit, which move
    // ~4x less weight memory and don't want an fp16 blow-up of W).
    if (M >= 64) {
        auto stream = at::cuda::getCurrentCUDAStream();
        auto Wf = dequant_weights_fp16(Wq, fmt, N, K, stream, X.options());
        return torch::matmul(X, Wf.t()).to(torch::kFloat);
    }
    return qmm_impl<false>(X, Wq, fmt, c10::nullopt);
}
torch::Tensor py_qflux_gelu(torch::Tensor X, torch::Tensor Wq, torch::Tensor bias, const std::string& fmt) {
    return qmm_impl<true>(X, Wq, fmt, bias);
}

// ---- integer GEMV paths ----
torch::Tensor py_qgemv_w8a8(torch::Tensor Wq, torch::Tensor Xq, torch::Tensor w_scale, torch::Tensor a_scale) {
    CHECK_CUDA_T(Wq); CHECK_CUDA_T(Xq); CHECK_CUDA_T(w_scale); CHECK_CUDA_T(a_scale);
    const int N = Wq.size(0), K = Wq.size(1);
    auto D = torch::empty({N}, w_scale.options().dtype(torch::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    qgemv_w8a8<<<(N + 1) / 2, 32, 0, stream>>>(reinterpret_cast<half*>(D.data_ptr()),
        Wq.data_ptr<int8_t>(), Xq.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(w_scale.data_ptr()),
        reinterpret_cast<const half*>(a_scale.data_ptr()), N, K);
    return D;
}
torch::Tensor py_qgemv_w2a8(torch::Tensor Wq, torch::Tensor Xq, torch::Tensor a_scale, int64_t K) {
    CHECK_CUDA_T(Wq); CHECK_CUDA_T(Xq); CHECK_CUDA_T(a_scale);
    const int N = Wq.size(0);
    auto D = torch::empty({N}, a_scale.options().dtype(torch::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    qgemv_w2a8<<<N, 32, 0, stream>>>(reinterpret_cast<half*>(D.data_ptr()),
        Wq.data_ptr<uint8_t>(), Xq.data_ptr<int8_t>(),
        reinterpret_cast<const half*>(a_scale.data_ptr()), N, (int)K);
    return D;
}

// ---- runtime activation quantizers ----
std::tuple<torch::Tensor, torch::Tensor> py_quantize_per_token(torch::Tensor A, const std::string& kind) {
    CHECK_CUDA_T(A);
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "A must be fp32");
    const int T = A.size(0), D = A.size(1);
    auto codes = torch::empty({T, D}, A.options().dtype(torch::kUInt8));
    auto scale = torch::empty({T}, A.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (kind == "fp8") quantize_per_token<true><<<T, 32, 0, stream>>>(codes.data_ptr<uint8_t>(), scale.data_ptr<float>(), A.data_ptr<float>(), T, D);
    else               quantize_per_token<false><<<T, 32, 0, stream>>>(codes.data_ptr<uint8_t>(), scale.data_ptr<float>(), A.data_ptr<float>(), T, D);
    return {codes, scale};
}
std::tuple<torch::Tensor, torch::Tensor> py_quantize_per_tensor(torch::Tensor A, const std::string& kind) {
    CHECK_CUDA_T(A);
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "A must be fp32");
    const size_t n = A.numel();
    auto codes = torch::empty_like(A, A.options().dtype(torch::kUInt8));
    auto scale = torch::empty({1}, A.options());
    auto omax = torch::zeros({1}, A.options().dtype(torch::kInt));
    auto stream = at::cuda::getCurrentCUDAStream();
    int blocks = int((n + 16 * 256 - 1) / (16 * 256));
    tensor_absmax<<<blocks, 256, 0, stream>>>(reinterpret_cast<unsigned*>(omax.data_ptr<int>()), A.data_ptr<float>(), n);
    if (kind == "fp8") tensor_encode<true><<<blocks, 256, 0, stream>>>(codes.data_ptr<uint8_t>(), scale.data_ptr<float>(), reinterpret_cast<const unsigned*>(omax.data_ptr<int>()), A.data_ptr<float>(), n);
    else               tensor_encode<false><<<blocks, 256, 0, stream>>>(codes.data_ptr<uint8_t>(), scale.data_ptr<float>(), reinterpret_cast<const unsigned*>(omax.data_ptr<int>()), A.data_ptr<float>(), n);
    return {codes, scale};
}

// ---- fused LM head sampling (argmax / gumbel categorical) ----
torch::Tensor py_lm_head_sample(torch::Tensor h, torch::Tensor W, const std::string& fmt,
                                double temperature, int64_t seed, bool gumbel) {
    CHECK_CUDA_T(h); CHECK_CUDA_T(W);
    TORCH_CHECK(h.scalar_type() == torch::kHalf, "h must be fp16");
    const int T = h.size(0), K = h.size(1);
    const int TILE_V = 1024;
    const float invtemp = temperature > 0 ? float(1.0 / temperature) : 1.0f;
    auto stream = at::cuda::getCurrentCUDAStream();
    int V;
    if (fmt == "fp16") V = W.size(0);
    else V = W.size(0);
    const int nvt = (V + TILE_V - 1) / TILE_V;
    auto pv = torch::empty({T, nvt}, h.options().dtype(torch::kFloat));
    auto pi = torch::empty({T, nvt}, h.options().dtype(torch::kInt));
    auto out = torch::empty({T}, h.options().dtype(torch::kInt));
    dim3 grid(nvt, T);
    if (fmt == "fp16") {
        TORCH_CHECK(W.scalar_type() == torch::kHalf);
        lm_head_argcat_partials<<<grid, 32, 0, stream>>>(
            reinterpret_cast<const half*>(h.data_ptr()), reinterpret_cast<const half*>(W.data_ptr()),
            pv.data_ptr<float>(), pi.data_ptr<int>(), nullptr, V, K, TILE_V, nvt, invtemp,
            unsigned(seed), gumbel ? 1 : 0, 0);
    } else {
#define X(F) if (fmt == #F) lm_head_argcat_partials_q<F><<<grid, 32, 0, stream>>>( \
            reinterpret_cast<const half*>(h.data_ptr()), W.data_ptr<uint8_t>(), \
            pv.data_ptr<float>(), pi.data_ptr<int>(), nullptr, V, K, TILE_V, nvt, invtemp, \
            unsigned(seed), gumbel ? 1 : 0, 0); else
        TMQ_FORMATS(X)
#undef X
        TORCH_CHECK(false, "unknown quant format: ", fmt);
    }
    lm_head_argcat_reduce<<<T, 32, 0, stream>>>(pv.data_ptr<float>(), pi.data_ptr<int>(), out.data_ptr<int>(), nvt);
    return out;
}

// ---- fused LM head top-k / top-p sampling (Step 4) ----
// mode: "top_k" (Gumbel-max among the global top-k) or "top_p" (nucleus with
// the TRUE full-vocab normalizer from per-tile lses). pool_k = per-tile
// over-selection size for top-p (>= expected nucleus tail in any tile).
torch::Tensor py_lm_head_sample_topk(torch::Tensor h, torch::Tensor W, const std::string& fmt,
                                     int64_t k, double temperature, int64_t seed) {
    CHECK_CUDA_T(h); CHECK_CUDA_T(W);
    TORCH_CHECK(h.scalar_type() == torch::kHalf, "h must be fp16");
    TORCH_CHECK(k >= 1 && k <= LMH_MAX_K, "k must be in [1,64]");
    const int T = h.size(0), K = h.size(1), V = W.size(0);
    const int TILE_V = 1024;
    const int nvt = (V + TILE_V - 1) / TILE_V;
    const float invtemp = temperature > 0 ? float(1.0 / temperature) : 1.0f;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto pv = torch::empty({T, nvt, k}, h.options().dtype(torch::kFloat));
    auto pi = torch::empty({T, nvt, k}, h.options().dtype(torch::kInt));
    auto out = torch::empty({T}, h.options().dtype(torch::kInt));
    dim3 grid(nvt, T);
    if (fmt == "fp16") {
        TORCH_CHECK(W.scalar_type() == torch::kHalf);
        lm_head_topk_partials<0><<<grid, 32, 0, stream>>>(
            reinterpret_cast<const half*>(h.data_ptr()), reinterpret_cast<const half*>(W.data_ptr()),
            pv.data_ptr<float>(), pi.data_ptr<int>(), nullptr, V, K, TILE_V, nvt, int(k), 0,
            invtemp, nullptr);
    } else {
#define X(F) if (fmt == #F) lm_head_topk_partials_q<F, 0><<<grid, 32, 0, stream>>>( \
            reinterpret_cast<const half*>(h.data_ptr()), W.data_ptr<uint8_t>(), \
            pv.data_ptr<float>(), pi.data_ptr<int>(), nullptr, V, K, TILE_V, nvt, int(k), 0, \
            invtemp, nullptr); else
        TMQ_FORMATS(X)
#undef X
        TORCH_CHECK(false, "unknown quant format: ", fmt);
    }
    lm_head_topk_reduce<<<T, 32, 0, stream>>>(pv.data_ptr<float>(), pi.data_ptr<int>(),
        out.data_ptr<int>(), nvt, int(k), unsigned(seed), invtemp);
    return out;
}
torch::Tensor py_lm_head_sample_topp(torch::Tensor h, torch::Tensor W, const std::string& fmt,
                                     double p, int64_t pool_k, double temperature, int64_t seed) {
    CHECK_CUDA_T(h); CHECK_CUDA_T(W);
    TORCH_CHECK(h.scalar_type() == torch::kHalf, "h must be fp16");
    TORCH_CHECK(pool_k >= 1 && pool_k <= LMH_MAX_K, "pool_k must be in [1,64]");
    const int T = h.size(0), K = h.size(1), V = W.size(0);
    const int TILE_V = 1024;
    const int nvt = (V + TILE_V - 1) / TILE_V;
    const float invtemp = temperature > 0 ? float(1.0 / temperature) : 1.0f;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto pv = torch::empty({T, nvt, pool_k}, h.options().dtype(torch::kFloat));
    auto pi = torch::empty({T, nvt, pool_k}, h.options().dtype(torch::kInt));
    auto lse = torch::empty({T, nvt}, h.options().dtype(torch::kFloat));
    auto out = torch::empty({T}, h.options().dtype(torch::kInt));
    dim3 grid(nvt, T);
    if (fmt == "fp16") {
        TORCH_CHECK(W.scalar_type() == torch::kHalf);
        lm_head_topk_partials<1><<<grid, 32, 0, stream>>>(
            reinterpret_cast<const half*>(h.data_ptr()), reinterpret_cast<const half*>(W.data_ptr()),
            pv.data_ptr<float>(), pi.data_ptr<int>(), nullptr, V, K, TILE_V, nvt, int(pool_k), 0,
            invtemp, lse.data_ptr<float>());
    } else {
#define X(F) if (fmt == #F) lm_head_topk_partials_q<F, 1><<<grid, 32, 0, stream>>>( \
            reinterpret_cast<const half*>(h.data_ptr()), W.data_ptr<uint8_t>(), \
            pv.data_ptr<float>(), pi.data_ptr<int>(), nullptr, V, K, TILE_V, nvt, int(pool_k), 0, \
            invtemp, lse.data_ptr<float>()); else
        TMQ_FORMATS(X)
#undef X
        TORCH_CHECK(false, "unknown quant format: ", fmt);
    }
    lm_head_topp_reduce<<<T, 32, 0, stream>>>(pv.data_ptr<float>(), pi.data_ptr<int>(),
        out.data_ptr<int>(), nvt, int(pool_k), float(p), unsigned(seed), invtemp,
        lse.data_ptr<float>());
    return out;
}

// ---- Step-4 qgemm variants ----
torch::Tensor py_qgemm_actorder(torch::Tensor X, torch::Tensor Wq, torch::Tensor perm,
                                const std::string& fmt) {
    CHECK_CUDA_T(X); CHECK_CUDA_T(Wq); CHECK_CUDA_T(perm);
    TORCH_CHECK(X.scalar_type() == torch::kHalf, "X must be fp16");
    const int M = X.size(0), K = X.size(1), N = Wq.size(0);
    auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(N / 16, (M + 15) / 16);
#define X(F) if (fmt == #F) { qgemm_actorder<F><<<grid, 32, 0, stream>>>(Y.data_ptr<float>(), \
        reinterpret_cast<const half*>(X.data_ptr()), Wq.data_ptr<uint8_t>(), \
        perm.data_ptr<int>(), M, N, K); return Y; }
    X(q4_0) X(kU4) X(kU4B8) X(q8_0)
#undef X
    TORCH_CHECK(false, "qgemm_actorder: unsupported format ", fmt);
}
torch::Tensor py_qgemm_blockscale(torch::Tensor X, torch::Tensor Wq, torch::Tensor scale2d) {
    CHECK_CUDA_T(X); CHECK_CUDA_T(Wq); CHECK_CUDA_T(scale2d);
    TORCH_CHECK(X.scalar_type() == torch::kHalf && scale2d.scalar_type() == torch::kHalf);
    const int M = X.size(0), K = X.size(1), N = Wq.size(0);
    auto Y = torch::empty({M, N}, X.options().dtype(torch::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 grid(N / 16, (M + 15) / 16);
    qgemm_blockscale<fp8_raw><<<grid, 32, 0, stream>>>(Y.data_ptr<float>(),
        reinterpret_cast<const half*>(X.data_ptr()), Wq.data_ptr<uint8_t>(),
        reinterpret_cast<const half*>(scale2d.data_ptr()), M, N, K);
    return Y;
}

void init_serving(pybind11::module_& m);       // tm_cuda_serving.cu
void init_elementwise(pybind11::module_& m);   // tm_cuda_elementwise.cu
void init_w6(pybind11::module_& m);            // tm_cuda_w6.cu
void init_moe_quant(pybind11::module_& m);     // tm_cuda_moe_quant.cu
void init_m4(pybind11::module_& m);            // tm_cuda_m4.cu
void init_m5(pybind11::module_& m);            // tm_cuda_m5.cu
void init_m6(pybind11::module_& m);            // tm_cuda_m6.cu
void init_mf_followups(pybind11::module_& m);  // tm_cuda_mf_followups.cu

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    init_serving(m);
    init_elementwise(m);
    init_w6(m);
    init_moe_quant(m);
    init_m4(m);
    init_m5(m);
    init_m6(m);
    init_mf_followups(m);
    m.def("qgemv", &py_qgemv, "D(N) = dequant(Wq) @ x, fp16 x");
    m.def("qgemm", &py_qgemm, "Y(M,N) = X(M,K) @ dequant(Wq)^T");
    m.def("qflux_gelu", &py_qflux_gelu, "gelu_tanh(X @ dequant(Wq)^T + bias)");
    m.def("qgemv_w8a8", &py_qgemv_w8a8, "int8 W x int8 X GEMV (dp4a)");
    m.def("qgemv_w2a8", &py_qgemv_w2a8, "BitNet ternary W x int8 X GEMV");
    m.def("quantize_per_token", &py_quantize_per_token, "(codes, scale) per row; kind='fp8'|'int8'");
    m.def("quantize_per_tensor", &py_quantize_per_tensor, "(codes, scale); kind='fp8'|'int8'");
    m.def("lm_head_sample", &py_lm_head_sample, "fused head + argmax/categorical; fmt='fp16'|<quant>");
    m.def("lm_head_sample_topk", &py_lm_head_sample_topk, pybind11::arg("h"), pybind11::arg("W"),
          pybind11::arg("format") = "fp16", pybind11::arg("k") = 40,
          pybind11::arg("temperature") = 1.0, pybind11::arg("seed") = 0);
    m.def("lm_head_sample_topp", &py_lm_head_sample_topp, pybind11::arg("h"), pybind11::arg("W"),
          pybind11::arg("format") = "fp16", pybind11::arg("p") = 0.9,
          pybind11::arg("pool_k") = 32, pybind11::arg("temperature") = 1.0,
          pybind11::arg("seed") = 0);
    m.def("qgemm_actorder", &py_qgemm_actorder, "GPTQ act-order GEMM; perm (K,) int32");
    m.def("qgemm_blockscale", &py_qgemm_blockscale, "fp8_raw codes + (N/128,K/128) fp16 scales");
    m.def("format_block_k", &format_block_k);
}
