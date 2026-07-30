// tk_cuda MF-M1 bindings: quantized MoE grouped GEMMs (fp8/nvfp4/wna16),
// fused activation-quant, per-token-group / nvfp4 experts quantizers, and
// scored routing (kernels/moe_quant/tm_moe_quant_kernels.cuh).
// Registered by init_moe_quant(m) from tm_cuda_ext.cu.
#include "../moe_quant/tm_moe_quant_kernels.cuh"
#include "../quant/q2k_ampere.cuh"
#include "../quant/q2k_moe_ampere.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;
using namespace tmoeq;

#define QK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")

// ---------------------------------------------------------------- Q2_K (sm80)
// Repack GGUF Q2_K blocks into the GEMV's three planes. Done once per expert
// tensor at load time; `src` is the raw [E, N, K/256*84] GGUF payload.
static std::vector<torch::Tensor> py_q2k_repack(torch::Tensor src, int64_t E,
                                                int64_t N, int64_t K) {
    QK(src);
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
    const int nj = (int)K / 16, nsb = (int)K / 256;
    auto i32 = src.options().dtype(torch::kInt);
    auto qp = torch::empty({E, N, nj}, i32);
    auto sp = torch::empty({E, N, nj}, src.options().dtype(torch::kByte));
    auto dp = torch::empty({E, N, nsb, 2}, src.options().dtype(torch::kHalf));
    const size_t src_e = (size_t)N * nsb * 84;
    for (int64_t e = 0; e < E; ++e) {
        const int blocks = (int)((N * nsb + 255) / 256);
        tmq_a100::q2k_repack<<<blocks, 256, 0,
                          at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const uint8_t*>(src.data_ptr()) + e * src_e,
            reinterpret_cast<uint32_t*>(qp.data_ptr()) + (size_t)e * N * nj,
            reinterpret_cast<uint8_t*>(sp.data_ptr()) + (size_t)e * N * nj,
            reinterpret_cast<half2*>(dp.data_ptr()) + (size_t)e * N * nsb,
            (int)N, (int)K);
    }
    return {qp, sp, dp};
}

// int8 activations with a per-256 scale and per-16 sums (the amortized SUM x
// term of the Q2_K factorization d*sc*(q.x) - dmin*m*(SUM x)).
static std::vector<torch::Tensor> py_q2k_quant_a8(torch::Tensor X) {
    QK(X);
    TORCH_CHECK(X.scalar_type() == torch::kHalf, "X must be fp16");
    const int M = (int)X.size(0), K = (int)X.size(1);
    TORCH_CHECK(K % 256 == 0, "K must be a multiple of 256");
    auto xq = torch::empty({M, K}, X.options().dtype(torch::kChar));
    auto xs = torch::empty({M, K / 256}, X.options());
    auto xsum = torch::empty({M, K / 16}, X.options().dtype(torch::kInt));
    dim3 g((unsigned)((K / 256 + 127) / 128), (unsigned)M);
    tmq_a100::q2k_quant_a8<<<g, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const __half*>(X.data_ptr()),
        reinterpret_cast<int8_t*>(xq.data_ptr()),
        reinterpret_cast<__half*>(xs.data_ptr()), xsum.data_ptr<int>(), M, K);
    return {xq, xs, xsum};
}

static torch::Tensor py_q2k_moe_gemv_a8(torch::Tensor qp, torch::Tensor sp,
        torch::Tensor dp, torch::Tensor xq, torch::Tensor xs,
        torch::Tensor xsum, torch::Tensor topk_ids, int64_t top_k, int64_t N,
        int64_t K) {
    QK(qp); QK(sp); QK(dp); QK(xq); QK(xs); QK(xsum); QK(topk_ids);
    TORCH_CHECK(topk_ids.scalar_type() == torch::kInt, "topk_ids must be int32");
    constexpr int NR = 4, KC = 4096, QB = 2;
    const int rows = (int)topk_ids.numel();
    auto Y = torch::empty({rows, N}, xs.options().dtype(torch::kFloat));
    const size_t smem = tmq_a100::q2k_moe_gemv_smem<NR, KC>();
    auto kern = tmq_a100::q2k_moe_gemv_a8<NR, KC, QB>;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    const int nwarp = 256 / 32;
    dim3 grid((unsigned)((N + nwarp * NR - 1) / (nwarp * NR)), (unsigned)rows);
    kern<<<grid, 256, smem, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const uint32_t*>(qp.data_ptr()),
        reinterpret_cast<const uint8_t*>(sp.data_ptr()),
        reinterpret_cast<const half2*>(dp.data_ptr()),
        reinterpret_cast<const int8_t*>(xq.data_ptr()),
        reinterpret_cast<const __half*>(xs.data_ptr()), xsum.data_ptr<int>(),
        topk_ids.data_ptr<int>(), Y.data_ptr<float>(), rows, (int)top_k,
        (int)N, (int)K);
    return Y;
}
static cudaStream_t qstream() { return at::cuda::getCurrentCUDAStream(); }
static const half* hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }

// ---- quantized MoE grouped GEMMs (A = permuted fp16, expert_of_tile per 32-tile) ----
static torch::Tensor py_moe_gemm_fp8(torch::Tensor A, torch::Tensor B, torch::Tensor Bscale,
                                     torch::Tensor eot, int64_t N, int64_t K) {
    QK(A); QK(B); QK(Bscale); QK(eot);
    const int rows = A.size(0);
    auto Y = torch::zeros({rows, N}, A.options().dtype(torch::kFloat));
    dim3 grid(N / 16, rows / 32);      // 32-row M-blocked tiles (one block per expert tile)
    moe_gemm_fp8<<<grid, 32, 0, qstream()>>>(Y.data_ptr<float>(), hp(A),
        B.data_ptr<uint8_t>(), Bscale.data_ptr<float>(), eot.data_ptr<int>(), rows, int(N), int(K));
    return Y;
}
static torch::Tensor py_moe_gemm_wna16(torch::Tensor A, torch::Tensor qweight, torch::Tensor scales,
        c10::optional<torch::Tensor> qzeros, torch::Tensor eot, int64_t N, int64_t K,
        int64_t group_size, int64_t bit) {
    QK(A); QK(qweight); QK(scales); QK(eot);
    const int rows = A.size(0);
    auto Y = torch::zeros({rows, N}, A.options().dtype(torch::kFloat));
    const uint8_t* qz = qzeros ? qzeros->data_ptr<uint8_t>() : nullptr;
    const int has_zp = qzeros ? 1 : 0;
    dim3 grid(N / 16, rows / 32);      // 32-row M-blocked tiles (one block per expert tile)
    if (bit == 4)
        moe_gemm_wna16<4><<<grid, 32, 0, qstream()>>>(Y.data_ptr<float>(), hp(A),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr()), hp(scales), qz,
            eot.data_ptr<int>(), rows, int(N), int(K), int(group_size), has_zp);
    else
        moe_gemm_wna16<8><<<grid, 32, 0, qstream()>>>(Y.data_ptr<float>(), hp(A),
            reinterpret_cast<const uint32_t*>(qweight.data_ptr()), hp(scales), qz,
            eot.data_ptr<int>(), rows, int(N), int(K), int(group_size), has_zp);
    return Y;
}
static torch::Tensor py_moe_gemm_nvfp4(torch::Tensor A, torch::Tensor B, torch::Tensor Asc,
        torch::Tensor Bsc, torch::Tensor alphas, torch::Tensor eot, torch::Tensor erow0,
        torch::Tensor sfo, int64_t N, int64_t K) {
    QK(A); QK(B); QK(Asc); QK(Bsc); QK(alphas); QK(eot); QK(erow0); QK(sfo);
    const int rows = eot.size(0) * 32;
    auto Y = torch::zeros({rows, N}, alphas.options().dtype(torch::kFloat));
    dim3 grid(N / 16, rows / 32);      // 32-row M-blocked tiles (one block per expert tile)
    moe_gemm_nvfp4<<<grid, 32, 0, qstream()>>>(Y.data_ptr<float>(), A.data_ptr<uint8_t>(),
        B.data_ptr<uint8_t>(), Asc.data_ptr<uint8_t>(), Bsc.data_ptr<uint8_t>(),
        alphas.data_ptr<float>(), eot.data_ptr<int>(), erow0.data_ptr<int>(),
        sfo.data_ptr<int>(), rows, int(N), int(K));
    return Y;
}

// ---- fused activation-quant + group/experts quantizers ----
static std::tuple<torch::Tensor, torch::Tensor> py_silu_and_mul_quant(
        torch::Tensor input, bool fp8, int64_t group_size, double static_scale) {
    QK(input);
    const int T = input.size(0), H = input.size(1) / 2;
    auto out = torch::empty({T, H}, input.options().dtype(torch::kUInt8));
    if (group_size <= 0) {                                  // static per-tensor
        auto sc = torch::empty({1}, input.options().dtype(torch::kFloat));
        const long n = (long)T * H;
        if (fp8) silu_and_mul_quant_static<half, true><<<(n+255)/256, 256, 0, qstream()>>>(
            out.data_ptr<uint8_t>(), hp(input), 1.0f/float(static_scale), H, T);
        else silu_and_mul_quant_static<half, false><<<(n+255)/256, 256, 0, qstream()>>>(
            out.data_ptr<uint8_t>(), hp(input), 1.0f/float(static_scale), H, T);
        sc.fill_(static_scale);
        return {out, sc};
    }
    const int NG = H / group_size;
    auto sc = torch::empty({T, NG}, input.options().dtype(torch::kFloat));
    dim3 g(NG, T);
    if (fp8) silu_and_mul_quant_perblock<half, true><<<g, 32, 0, qstream()>>>(
        out.data_ptr<uint8_t>(), sc.data_ptr<float>(), hp(input), H, int(group_size), NG);
    else silu_and_mul_quant_perblock<half, false><<<g, 32, 0, qstream()>>>(
        out.data_ptr<uint8_t>(), sc.data_ptr<float>(), hp(input), H, int(group_size), NG);
    return {out, sc};
}
static std::tuple<torch::Tensor, torch::Tensor> py_per_token_group_quant_fp8(
        torch::Tensor input, int64_t group_size, bool ue8m0, double eps) {
    QK(input);
    const int T = input.size(0), H = input.size(1), NG = H / group_size;
    auto out = torch::empty({T, H}, input.options().dtype(torch::kUInt8));
    auto sc = torch::empty({T, NG}, input.options().dtype(torch::kFloat));
    dim3 g(NG, T);
    if (ue8m0) per_token_group_quant_fp8<half, true><<<g, 32, 0, qstream()>>>(
        out.data_ptr<uint8_t>(), sc.data_ptr<float>(), hp(input), H, int(group_size), NG, float(eps));
    else per_token_group_quant_fp8<half, false><<<g, 32, 0, qstream()>>>(
        out.data_ptr<uint8_t>(), sc.data_ptr<float>(), hp(input), H, int(group_size), NG, float(eps));
    return {out, sc};
}

// ---- scored routing ----
static std::tuple<torch::Tensor, torch::Tensor> py_moe_route_scored(
        torch::Tensor logits, int64_t K, int64_t mode, bool renormalize, double scaling) {
    QK(logits);
    const int T = logits.size(0), E = logits.size(1);
    auto ids = torch::empty({T, K}, logits.options().dtype(torch::kInt));
    auto w = torch::empty({T, K}, logits.options().dtype(torch::kFloat));
    moe_route_scored<half><<<T, 32, 0, qstream()>>>(hp(logits), ids.data_ptr<int>(),
        w.data_ptr<float>(), E, int(K), int(mode), renormalize ? 1 : 0, float(scaling));
    return {ids, w};
}

void init_moe_quant(py::module_& m) {
    m.def("q2k_repack", &py_q2k_repack, py::arg("src"), py::arg("E"),
          py::arg("N"), py::arg("K"));
    m.def("q2k_quant_a8", &py_q2k_quant_a8, py::arg("X"));
    m.def("q2k_moe_gemv_a8", &py_q2k_moe_gemv_a8, py::arg("qp"), py::arg("sp"),
          py::arg("dp"), py::arg("xq"), py::arg("xs"), py::arg("xsum"),
          py::arg("topk_ids"), py::arg("top_k"), py::arg("N"), py::arg("K"));
    m.def("moe_gemm_fp8", &py_moe_gemm_fp8);
    m.def("moe_gemm_wna16", &py_moe_gemm_wna16, py::arg("A"), py::arg("qweight"), py::arg("scales"),
          py::arg("qzeros") = py::none(), py::arg("eot"), py::arg("N"), py::arg("K"),
          py::arg("group_size"), py::arg("bit"));
    m.def("moe_gemm_nvfp4", &py_moe_gemm_nvfp4);
    m.def("silu_and_mul_quant", &py_silu_and_mul_quant, py::arg("input"), py::arg("fp8") = true,
          py::arg("group_size") = 0, py::arg("static_scale") = 1.0);
    m.def("per_token_group_quant_fp8", &py_per_token_group_quant_fp8, py::arg("input"),
          py::arg("group_size"), py::arg("ue8m0") = false, py::arg("eps") = 1e-6);
    m.def("moe_route_scored", &py_moe_route_scored, py::arg("logits"), py::arg("K"),
          py::arg("mode") = 0, py::arg("renormalize") = true, py::arg("scaling") = 1.0);
}
