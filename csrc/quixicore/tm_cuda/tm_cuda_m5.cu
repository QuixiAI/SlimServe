// tk_cuda MF-M5 bindings: sparse-attention + serving extras
// (sparse_serving_kernels.cuh). Registered by init_m5(m).
#include "../serving/sparse_serving_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;
using namespace tmsp;

#define M5CK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t m5s() { return at::cuda::getCurrentCUDAStream(); }
static const half* m5hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }
static half* m5hpm(torch::Tensor& t) { return reinterpret_cast<half*>(t.data_ptr()); }

// merge_attn_states (fp16). prefix/suffix (tokens, heads, head_size); lse (tokens, heads).
static std::tuple<torch::Tensor, torch::Tensor> py_merge_attn_states(
        torch::Tensor prefix_out, torch::Tensor prefix_lse, torch::Tensor suffix_out,
        torch::Tensor suffix_lse, int64_t prefix_num_tokens) {
    M5CK(prefix_out); M5CK(prefix_lse); M5CK(suffix_out); M5CK(suffix_lse);
    const int T = prefix_out.size(0), H = prefix_out.size(1), D = prefix_out.size(2);
    auto out = torch::empty_like(prefix_out);
    auto olse = torch::empty({T, H}, prefix_lse.options());
    const long n = (long)T * H * D;
    merge_attn_states<half><<<(n + 255) / 256, 256, 0, m5s()>>>(m5hpm(out), olse.data_ptr<float>(),
        m5hp(prefix_out), prefix_lse.data_ptr<float>(), m5hp(suffix_out),
        suffix_lse.data_ptr<float>(), H, D, T, int(prefix_num_tokens));
    return {out, olse};
}

// tau_tail (fp16, in-place on qkv). qkv (tokens, 3*q_dim); tok_qv_lin (tokens, 2*heads).
static void py_tau_tail(torch::Tensor qkv, torch::Tensor tok_qv_lin, torch::Tensor tau_pos_table,
        torch::Tensor positions, int64_t n_heads, int64_t head_dim) {
    M5CK(qkv); M5CK(tok_qv_lin); M5CK(tau_pos_table); M5CK(positions);
    const int T = qkv.size(0), q_dim = int(n_heads) * int(head_dim);
    const int elements = T * int(n_heads) * int(head_dim);
    tau_tail<half><<<(elements + 255) / 256, 256, 0, m5s()>>>(m5hpm(qkv), m5hp(tok_qv_lin),
        m5hp(tau_pos_table), positions.data_ptr<int64_t>(), elements, int(n_heads),
        int(head_dim), q_dim);
}

// lightning indexer fp8 K-quant-and-cache. k (tokens, head_dim); returns the
// packed uint8 cache (num_blocks, cache_block_size, cache_stride).
static torch::Tensor py_indexer_k_quant(torch::Tensor k, torch::Tensor slot_mapping,
        int64_t num_blocks, int64_t head_dim, int64_t quant_block_size,
        int64_t cache_block_size, bool ue8m0) {
    M5CK(k); M5CK(slot_mapping);
    const int NT = k.size(0);
    const int nqb = int(head_dim) / int(quant_block_size);
    const int cache_stride = int(head_dim) + nqb * 4;
    auto cache = torch::zeros({num_blocks, cache_block_size, cache_stride},
                              k.options().dtype(torch::kUInt8));
    dim3 grid(NT, nqb);
    indexer_k_quant_and_cache<half><<<grid, 32, 0, m5s()>>>(m5hp(k), cache.data_ptr<uint8_t>(),
        slot_mapping.data_ptr<int64_t>(), NT, int(head_dim), int(quant_block_size),
        int(cache_block_size), cache_stride, ue8m0 ? 1 : 0);
    return cache;
}

// MInference vertical/slash builder -> (block_count, block_offset, column_count, column_index).
static std::vector<torch::Tensor> py_convert_vertical_slash(torch::Tensor q_seqlens,
        torch::Tensor kv_seqlens, torch::Tensor vertical, torch::Tensor slash,
        int64_t num_heads, int64_t num_rows, int64_t block_size_m, int64_t block_size_n,
        bool causal) {
    M5CK(q_seqlens); M5CK(kv_seqlens); M5CK(vertical); M5CK(slash);
    const int B = q_seqlens.numel();
    const int nnz_v = vertical.size(-1), nnz_s = slash.size(-1);
    const long RT = (long)B * num_heads * num_rows;
    auto opt = q_seqlens.options();
    auto bc = torch::zeros({RT}, opt);
    auto bo = torch::zeros({RT * nnz_s}, opt);
    auto cc = torch::zeros({RT}, opt);
    auto ci = torch::zeros({RT * nnz_v}, opt);
    dim3 grid(int(num_heads), B, (int(num_rows) + 63) / 64);
    convert_vertical_slash_indexes<<<grid, 64, 0, m5s()>>>(q_seqlens.data_ptr<int>(),
        kv_seqlens.data_ptr<int>(), vertical.data_ptr<int>(), slash.data_ptr<int>(),
        bc.data_ptr<int>(), bo.data_ptr<int>(), cc.data_ptr<int>(), ci.data_ptr<int>(),
        int(num_heads), int(num_rows), nnz_v, nnz_s, int(block_size_m), int(block_size_n),
        causal ? 1 : 0);
    return {bc, bo, cc, ci};
}

void init_m5(py::module_& m) {
    m.def("merge_attn_states", &py_merge_attn_states, py::arg("prefix_out"), py::arg("prefix_lse"),
          py::arg("suffix_out"), py::arg("suffix_lse"), py::arg("prefix_num_tokens"));
    m.def("tau_tail", &py_tau_tail);
    m.def("indexer_k_quant", &py_indexer_k_quant, py::arg("k"), py::arg("slot_mapping"),
          py::arg("num_blocks"), py::arg("head_dim"), py::arg("quant_block_size"),
          py::arg("cache_block_size"), py::arg("ue8m0") = false);
    m.def("convert_vertical_slash_indexes", &py_convert_vertical_slash, py::arg("q_seqlens"),
          py::arg("kv_seqlens"), py::arg("vertical"), py::arg("slash"), py::arg("num_heads"),
          py::arg("num_rows"), py::arg("block_size_m") = 64, py::arg("block_size_n") = 64,
          py::arg("causal") = true);
}
