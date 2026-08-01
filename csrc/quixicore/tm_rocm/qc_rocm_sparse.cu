// QuixiCore-HIP sparse-MLA indexer bindings for gfx942.
//
// The ROCm sparse backend's own index kernels, which have no CUDA counterpart
// (see rocm/sparse_indexer_kernels.cuh). Registered into the module by
// qc_rocm_serving.cu, which owns the single PYBIND11_MODULE.
#include "sparse_indexer_kernels.cuh"

#include <c10/util/Float8_e4m3fnuz.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace py = pybind11;

#define SPCK(x) \
    TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous GPU")
static cudaStream_t spst() { return at::cuda::getCurrentCUDAStream(); }

static void py_convert_req_index_to_global_index(
    torch::Tensor req_id, torch::Tensor block_table, torch::Tensor token_indices,
    torch::Tensor cu_seqlens, torch::Tensor out, int64_t block_size,
    int64_t topk) {
    SPCK(req_id);
    SPCK(block_table);
    SPCK(token_indices);
    SPCK(cu_seqlens);
    SPCK(out);
    TORCH_CHECK(req_id.scalar_type() == torch::kInt, "req_id int32");
    TORCH_CHECK(block_table.scalar_type() == torch::kInt, "block_table int32");
    TORCH_CHECK(token_indices.scalar_type() == torch::kInt, "indices int32");
    TORCH_CHECK(cu_seqlens.scalar_type() == torch::kInt, "cu_seqlens int32");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out int32");

    const int num_tokens = (int)req_id.size(0);
    if (num_tokens == 0) return;
    const int max_num_blocks_per_req = (int)block_table.size(1);
    qcrocm::convert_req_index_to_global_index<<<num_tokens, 256, 0, spst()>>>(
        req_id.data_ptr<int>(), block_table.data_ptr<int>(),
        token_indices.data_ptr<int>(), cu_seqlens.data_ptr<int>(),
        out.data_ptr<int>(), max_num_blocks_per_req, (int)block_size,
        (int)topk, block_table.stride(0), block_table.stride(1),
        token_indices.stride(0), token_indices.stride(1));
}

static void py_generate_sparse_seqlen(torch::Tensor seq_lens,
                                      torch::Tensor cu_query_lens,
                                      torch::Tensor out, int64_t topk_token) {
    SPCK(seq_lens);
    SPCK(cu_query_lens);
    SPCK(out);
    TORCH_CHECK(seq_lens.scalar_type() == torch::kInt, "seq_lens int32");
    TORCH_CHECK(cu_query_lens.scalar_type() == torch::kInt, "cu_query int32");
    TORCH_CHECK(out.scalar_type() == torch::kInt, "out int32");

    const int num_seqs = (int)seq_lens.size(0);
    if (num_seqs == 0) return;
    qcrocm::generate_sparse_seqlen<<<num_seqs, 256, 0, spst()>>>(
        seq_lens.data_ptr<int>(), cu_query_lens.data_ptr<int>(),
        out.data_ptr<int>(), (int)topk_token);
}


// The fp8 store goes through c10's scalar type rather than a raw HIP cast:
// torch's conversion is the one measured bitwise-equal to ROCm Triton's over
// the kernel's whole output range, so using it removes rounding as a variable.
static void py_indexer_k_quant_and_cache(torch::Tensor k, torch::Tensor kv_cache,
                                         torch::Tensor kv_cache_scale,
                                         torch::Tensor slot_mapping,
                                         int64_t block_size,
                                         int64_t block_tile_size,
                                         int64_t head_tile_size,
                                         double fp8_max, int64_t shuffle) {
    SPCK(k);
    SPCK(slot_mapping);
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping i64");
    TORCH_CHECK(kv_cache_scale.scalar_type() == torch::kFloat, "scale fp32");
    const int num_tokens = (int)slot_mapping.size(0);
    if (num_tokens == 0) return;
    const int head_dim = (int)k.size(-1);
    const int threads = head_dim < 256 ? head_dim : 256;

    using FP8 = c10::Float8_e4m3fnuz;
    auto* cache = reinterpret_cast<FP8*>(kv_cache.data_ptr());
    auto launch = [&](auto dummy) {
        using T = decltype(dummy);
        qcrocm::indexer_k_quant_and_cache<T, FP8><<<num_tokens, threads, 0, spst()>>>(
            reinterpret_cast<const T*>(k.data_ptr()), cache,
            kv_cache_scale.data_ptr<float>(),
            reinterpret_cast<const long*>(slot_mapping.data_ptr()),
            kv_cache_scale.stride(0), kv_cache.stride(0), (int)block_size,
            head_dim, (int)block_tile_size, (int)head_tile_size,
            (float)fp8_max, (int)shuffle);
    };
    if (k.scalar_type() == torch::kBFloat16)
        launch(c10::BFloat16{});
    else if (k.scalar_type() == torch::kHalf)
        launch(c10::Half{});
    else if (k.scalar_type() == torch::kFloat)
        launch(float{});
    else
        TORCH_CHECK(false, "indexer_k_quant: want bf16/fp16/fp32 k");
}

void init_sparse(py::module_& m) {
    m.def("convert_req_index_to_global_index",
          &py_convert_req_index_to_global_index, py::arg("req_id"),
          py::arg("block_table"), py::arg("token_indices"),
          py::arg("cu_seqlens"), py::arg("out"), py::arg("block_size"),
          py::arg("topk"));
    m.def("indexer_k_quant_and_cache", &py_indexer_k_quant_and_cache,
          py::arg("k"), py::arg("kv_cache"), py::arg("kv_cache_scale"),
          py::arg("slot_mapping"), py::arg("block_size"),
          py::arg("block_tile_size"), py::arg("head_tile_size"),
          py::arg("fp8_max"), py::arg("shuffle"));
    m.def("generate_sparse_seqlen", &py_generate_sparse_seqlen,
          py::arg("seq_lens"), py::arg("cu_query_lens"), py::arg("out"),
          py::arg("topk_token"));
}
