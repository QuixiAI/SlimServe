// tk_cuda W6 bindings: MoE pipeline, linear-attention family, cmplx_matmul
// (kernels/moe/tm_moe_kernels.cuh, kernels/lin_attn_tm/tm_linattn_kernels.cuh).
// Registered by init_w6(m) from tm_cuda_ext.cu.
#include "../moe/tm_moe_kernels.cuh"
#include "../lin_attn_tm/tm_linattn_kernels.cuh"
#include "../lin_attn_tm/gdn_kernels.cuh"          // MF-M3 gated DeltaNet
#include "../mamba2/selective_scan_kernels.cuh"    // MF-M3 varlen selective scan
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define WCK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t wstream() { return at::cuda::getCurrentCUDAStream(); }

#define DISPATCH_W6(t, ...) do {                                                  \
    if (t.scalar_type() == torch::kFloat)        { using scalar_t = float;          __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kHalf)    { using scalar_t = __half;         __VA_ARGS__ } \
    else if (t.scalar_type() == torch::kBFloat16){ using scalar_t = __nv_bfloat16;  __VA_ARGS__ } \
    else TORCH_CHECK(false, "unsupported dtype (want fp32/fp16/bf16)");           \
} while (0)
template <typename S> static const S* wcp(const torch::Tensor& t) { return reinterpret_cast<const S*>(t.data_ptr()); }
template <typename S> static S* wmp(torch::Tensor& t) { return reinterpret_cast<S*>(t.data_ptr()); }

// ---- MoE ----
static std::tuple<torch::Tensor, torch::Tensor> py_moe_route_topk(
        torch::Tensor logits, int64_t K) {
    WCK(logits);
    const int T = logits.size(0), E = logits.size(1);
    auto ids = torch::empty({T, (long)K}, logits.options().dtype(torch::kInt));
    auto wts = torch::empty({T, (long)K}, logits.options().dtype(torch::kFloat));
    DISPATCH_W6(logits, tmoe::moe_route_topk<scalar_t><<<T, 32, 0, wstream()>>>(
        wcp<scalar_t>(logits), ids.data_ptr<int>(), wts.data_ptr<float>(), E, int(K)););
    return {ids, wts};
}

// Full permute + padded schedule (histogram -> scan -> scatter -> pad).
// Returns (offsets(E+1), sorted_row_idx(TK), inv_idx(TK), off_pad(E+1),
//          expert_of_tile(max_tiles), gather_idx(total_pad_max), inv_pad(TK)).
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                  torch::Tensor, torch::Tensor, torch::Tensor>
py_moe_align(torch::Tensor topk_ids, int64_t E) {
    WCK(topk_ids);
    const int TK = topk_ids.numel();
    const int K = topk_ids.size(-1);
    const int total_pad_max = ((TK + 31 * int(E)) + 31) / 32 * 32;
    const int max_tiles = total_pad_max / 32;
    auto opts = topk_ids.options();
    auto counts = torch::zeros({E}, opts);
    auto offsets = torch::empty({E + 1}, opts);
    auto cursor = torch::empty({E}, opts);
    auto sri = torch::empty({TK}, opts);
    auto inv = torch::empty({TK}, opts);
    auto off_pad = torch::empty({E + 1}, opts);
    auto eot = torch::empty({max_tiles}, opts);
    auto gix = torch::empty({total_pad_max}, opts);
    auto inv_pad = torch::empty({TK}, opts);
    auto s = wstream();
    tmoe::moe_histogram<<<(TK + 255) / 256, 256, 0, s>>>(topk_ids.data_ptr<int>(),
        counts.data_ptr<int>(), TK);
    tmoe::moe_scan_offsets<<<1, MOE_SCAN_NT, 0, s>>>(counts.data_ptr<int>(),
        offsets.data_ptr<int>(), cursor.data_ptr<int>(), int(E));
    tmoe::moe_scatter<<<(TK + 255) / 256, 256, 0, s>>>(topk_ids.data_ptr<int>(),
        cursor.data_ptr<int>(), sri.data_ptr<int>(), inv.data_ptr<int>(), TK);
    tmoe::moe_pad_offsets<<<1, MOE_SCAN_NT, 0, s>>>(offsets.data_ptr<int>(),
        off_pad.data_ptr<int>(), eot.data_ptr<int>(), gix.data_ptr<int>(),
        int(E), max_tiles, total_pad_max);
    tmoe::moe_pad_scatter<<<(TK + 255) / 256, 256, 0, s>>>(sri.data_ptr<int>(),
        offsets.data_ptr<int>(), off_pad.data_ptr<int>(), gix.data_ptr<int>(),
        inv_pad.data_ptr<int>(), TK, int(E), K);
    return {offsets, sri, inv, off_pad, eot, gix, inv_pad};
}

static torch::Tensor py_moe_gather(torch::Tensor x, torch::Tensor gather_idx) {
    WCK(x); WCK(gather_idx);
    const int H = x.size(-1), P = gather_idx.numel();
    auto out = torch::empty({P, H}, x.options());
    DISPATCH_W6(x, tmoe::moe_gather<scalar_t><<<P, 128, 0, wstream()>>>(
        wcp<scalar_t>(x), gather_idx.data_ptr<int>(), wmp<scalar_t>(out), H););
    return out;
}

static torch::Tensor py_moe_gemm(torch::Tensor A, torch::Tensor W,
                                 torch::Tensor expert_of_tile) {
    WCK(A); WCK(W); WCK(expert_of_tile);
    const int rows = A.size(0), K_dim = A.size(1), N_out = W.size(-1);
    TORCH_CHECK(W.size(1) == K_dim, "W (E,K,N) K mismatch");
    auto out = torch::zeros({rows, N_out}, A.options());
    dim3 grid{unsigned((N_out + 31) / 32), unsigned(expert_of_tile.numel())};
    DISPATCH_W6(A, tmoe::moe_grouped_gemm_rect<scalar_t><<<grid, 256, 0, wstream()>>>(
        wmp<scalar_t>(out), wcp<scalar_t>(A), wcp<scalar_t>(W),
        expert_of_tile.data_ptr<int>(), rows, K_dim, N_out););
    return out;
}

static torch::Tensor py_moe_gemm_swiglu(torch::Tensor A, torch::Tensor W1,
                                        torch::Tensor expert_of_tile) {
    WCK(A); WCK(W1); WCK(expert_of_tile);
    const int rows = A.size(0), H = A.size(1);
    const int inter = W1.size(-1) / 2;
    TORCH_CHECK(W1.size(1) == H, "W1 (E,H,2*inter) H mismatch");
    auto out = torch::zeros({rows, inter}, A.options());
    dim3 grid{unsigned((inter + 31) / 32), unsigned(expert_of_tile.numel())};
    DISPATCH_W6(A, tmoe::moe_grouped_gemm_swiglu<scalar_t><<<grid, 256, 0, wstream()>>>(
        wmp<scalar_t>(out), wcp<scalar_t>(A), wcp<scalar_t>(W1),
        expert_of_tile.data_ptr<int>(), rows, H, inter););
    return out;
}

static torch::Tensor py_moe_finalize(torch::Tensor expert_out, torch::Tensor inv,
                                     torch::Tensor topk_weights) {
    WCK(expert_out); WCK(inv); WCK(topk_weights);
    const int T = topk_weights.size(0), K = topk_weights.size(1);
    const int Hdim = expert_out.size(-1);
    auto out = torch::empty({T, Hdim}, expert_out.options());
    DISPATCH_W6(expert_out, tmoe::moe_finalize<scalar_t><<<T, 32, 0, wstream()>>>(
        wcp<scalar_t>(expert_out), inv.data_ptr<int>(), topk_weights.data_ptr<float>(),
        wmp<scalar_t>(out), K, Hdim););
    return out;
}

// ---- linear attention (D=64), (B,H,N,D) ----
static torch::Tensor py_linear_attn(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    bool causal, bool chunked) {
    WCK(q); WCK(k); WCK(v);
    const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    TORCH_CHECK(D == 64, "linear_attn: D must be 64");
    auto o = torch::empty_like(q);
    dim3 gbh{unsigned(H), unsigned(B)};
    auto s = wstream();
    DISPATCH_W6(q, {
        if (!causal) {
            tmla::linear_attn<scalar_t, 64><<<gbh, 64, 0, s>>>(
                wcp<scalar_t>(q), wcp<scalar_t>(k), wcp<scalar_t>(v), wmp<scalar_t>(o), N);
        } else if (!chunked) {
            tmla::lin_attn_causal<scalar_t, 64><<<gbh, 64, 0, s>>>(
                wcp<scalar_t>(q), wcp<scalar_t>(k), wcp<scalar_t>(v), wmp<scalar_t>(o), N);
        } else {
            TORCH_CHECK(N % LIN_CHUNK_L == 0, "chunked: N must be a multiple of 64");
            const int C = N / LIN_CHUNK_L;
            auto opts = q.options().dtype(torch::kFloat);
            auto S = torch::empty({B, H, C, 64, 64}, opts);
            auto Sex = torch::empty({B, H, C, 64, 64}, opts);
            dim3 gc{unsigned(C), unsigned(H), unsigned(B)};
            tmla::lin_chunk_kv<scalar_t, 64><<<gc, 64, 0, s>>>(
                wcp<scalar_t>(k), wcp<scalar_t>(v), S.data_ptr<float>(), N);
            tmla::lin_chunk_scan<64><<<B * H, 256, 0, s>>>(
                S.data_ptr<float>(), Sex.data_ptr<float>(), C);
            tmla::lin_chunk_out<scalar_t, 64><<<gc, 64, 0, s>>>(
                wcp<scalar_t>(q), wcp<scalar_t>(k), wcp<scalar_t>(v),
                Sex.data_ptr<float>(), wmp<scalar_t>(o), N);
        }
    });
    return o;
}

static torch::Tensor py_cmplx_matmul(torch::Tensor A, torch::Tensor B) {
    WCK(A); WCK(B);
    const int N = A.size(1), K = A.size(2), M = B.size(2);
    TORCH_CHECK(A.size(0) == 2 && B.size(0) == 2 && B.size(1) == K,
                "cmplx_matmul: A (2,N,K), B (2,K,M)");
    auto D = torch::empty({2, N, M}, A.options());
    dim3 blk{16, 16};
    dim3 grd{unsigned((M + 15) / 16), unsigned((N + 15) / 16)};
    DISPATCH_W6(A, tmla::cmplx_matmul<scalar_t><<<grd, blk, 0, wstream()>>>(
        wcp<scalar_t>(A), wcp<scalar_t>(B), wmp<scalar_t>(D), N, K, M););
    return D;
}

// ---- MF-M3 gated DeltaNet (varlen, paged state) ----
static torch::Tensor py_gdn_linear_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
        torch::Tensor g, torch::Tensor beta, torch::Tensor state_pool, torch::Tensor cu_seqlens,
        torch::Tensor slot_mapping, int64_t Hk, int64_t Hv, int64_t Dk, int64_t Dv) {
    WCK(q); WCK(k); WCK(v); WCK(g); WCK(beta); WCK(state_pool); WCK(cu_seqlens); WCK(slot_mapping);
    const int T = q.size(0);
    const int R = cu_seqlens.numel() - 1;
    auto y = torch::empty({T, int(Hv), int(Dv)}, v.options());
    dim3 grid(unsigned(Dv), 1, unsigned(R * Hv));
    DISPATCH_W6(q, tmgdn::gdn_linear_attention<scalar_t><<<grid, 32, 0, wstream()>>>(
        wcp<scalar_t>(q), wcp<scalar_t>(k), wcp<scalar_t>(v), wcp<scalar_t>(g), wcp<scalar_t>(beta),
        wmp<scalar_t>(state_pool), cu_seqlens.data_ptr<int>(), slot_mapping.data_ptr<int>(),
        wmp<scalar_t>(y), R, int(Hk), int(Hv), int(Dk), int(Dv)););
    return y;
}

// ---- MF-M3 varlen Mamba-1 selective scan (paged state) ----
static torch::Tensor py_selective_scan_varlen(torch::Tensor u, torch::Tensor delta,
        torch::Tensor A, torch::Tensor B, torch::Tensor C, c10::optional<torch::Tensor> D,
        c10::optional<torch::Tensor> delta_bias, c10::optional<torch::Tensor> z,
        torch::Tensor query_start_loc, c10::optional<torch::Tensor> cache_indices,
        c10::optional<torch::Tensor> has_initial_state, torch::Tensor state,
        int64_t dstate, int64_t n_groups, bool delta_softplus, int64_t null_block_id) {
    WCK(u); WCK(delta); WCK(A); WCK(B); WCK(C); WCK(query_start_loc); WCK(state);
    const int dim = u.size(0), total_tokens = u.size(1);
    const int batch = query_start_loc.numel() - 1;
    auto out = torch::empty_like(u);
    const int P = ((int(dstate) + 31) / 32) * 32;
    const int shmem = ((P + 31) / 32) * sizeof(float);
    const float* Dp = D ? D->data_ptr<float>() : nullptr;
    const float* dbp = delta_bias ? delta_bias->data_ptr<float>() : nullptr;
    const int* cip = cache_indices ? cache_indices->data_ptr<int>() : nullptr;
    const uint8_t* hip = has_initial_state ? has_initial_state->data_ptr<uint8_t>() : nullptr;
    dim3 grid(dim, batch);
    DISPATCH_W6(u, {
        const scalar_t* zp = z ? wcp<scalar_t>(*z) : nullptr;
        tmss::selective_scan_fwd_varlen<scalar_t><<<grid, P, shmem, wstream()>>>(
            wcp<scalar_t>(u), wcp<scalar_t>(delta), A.data_ptr<float>(), wcp<scalar_t>(B),
            wcp<scalar_t>(C), Dp, dbp, zp, query_start_loc.data_ptr<int>(), cip, hip,
            wmp<scalar_t>(out), state.data_ptr<float>(), batch, dim, total_tokens, int(dstate),
            int(n_groups), D ? 1 : 0, delta_bias ? 1 : 0, z ? 1 : 0, delta_softplus ? 1 : 0,
            cache_indices ? 1 : 0, has_initial_state ? 1 : 0, int(null_block_id));
    });
    return out;
}

void init_w6(py::module_& m) {
    m.def("selective_scan_varlen", &py_selective_scan_varlen, py::arg("u"), py::arg("delta"),
          py::arg("A"), py::arg("B"), py::arg("C"), py::arg("D") = py::none(),
          py::arg("delta_bias") = py::none(), py::arg("z") = py::none(),
          py::arg("query_start_loc"), py::arg("cache_indices") = py::none(),
          py::arg("has_initial_state") = py::none(), py::arg("state"), py::arg("dstate"),
          py::arg("n_groups"), py::arg("delta_softplus") = true, py::arg("null_block_id") = -1);
    m.def("gdn_linear_attention", &py_gdn_linear_attention, py::arg("q"), py::arg("k"),
          py::arg("v"), py::arg("g"), py::arg("beta"), py::arg("state_pool"),
          py::arg("cu_seqlens"), py::arg("slot_mapping"), py::arg("Hk"), py::arg("Hv"),
          py::arg("Dk"), py::arg("Dv"));
    m.def("moe_route_topk", &py_moe_route_topk, py::arg("logits"), py::arg("K"));
    m.def("moe_align", &py_moe_align, py::arg("topk_ids"), py::arg("E"),
          "-> (offsets, sorted_row_idx, inv_idx, off_pad, expert_of_tile, gather_idx, inv_pad)");
    m.def("moe_gather", &py_moe_gather);
    m.def("moe_gemm", &py_moe_gemm, "grouped rect GEMM: A(rows,K) @ W(E,K,N) by expert_of_tile");
    m.def("moe_gemm_swiglu", &py_moe_gemm_swiglu, "silu(A@W1_gate)*(A@W1_up), W1 (E,H,2*inter)");
    m.def("moe_finalize", &py_moe_finalize);
    m.def("linear_attn", &py_linear_attn, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("causal") = false, py::arg("chunked") = false);
    m.def("cmplx_matmul", &py_cmplx_matmul);
}
