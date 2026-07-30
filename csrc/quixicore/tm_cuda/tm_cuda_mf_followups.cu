// tk_cuda MetalForge follow-up bindings: EAGLE bookkeeping (eagle_kernels.cuh),
// lightning-indexer cp_gather + fp8-output merge_attn_states
// (sparse_serving_kernels.cuh), TurboQuant tq_encode codec (turboquant.cuh),
// and the Mamba selective_scan APC variant (selective_scan_kernels.cuh).
// Registered by init_mf_followups(m).
#include "../serving/eagle_kernels.cuh"
#include "../serving/sparse_serving_kernels.cuh"
#include "../quant/turboquant.cuh"
#include "../mamba2/selective_scan_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;

#define FCK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t fst() { return at::cuda::getCurrentCUDAStream(); }
static const half* fhp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }

// ---------------- EAGLE bookkeeping ----------------
static std::tuple<torch::Tensor, torch::Tensor> py_eagle_prepare_next_token(
        torch::Tensor sampled_token_ids, torch::Tensor discard_request_mask,
        torch::Tensor backup_next_token_ids, int64_t vocab_size) {
    FCK(sampled_token_ids); FCK(discard_request_mask); FCK(backup_next_token_ids);
    const int R = sampled_token_ids.size(0), nsp = sampled_token_ids.size(1);
    auto nt = torch::empty({R}, backup_next_token_ids.options());
    auto vc = torch::empty({R}, backup_next_token_ids.options());
    tmeg::eagle_prepare_next_token_padded<<<(R + 127) / 128, 128, 0, fst()>>>(
        nt.data_ptr<int64_t>(), vc.data_ptr<int64_t>(), sampled_token_ids.data_ptr<int64_t>(),
        discard_request_mask.data_ptr<uint8_t>(), backup_next_token_ids.data_ptr<int64_t>(),
        vocab_size, nsp, R, sampled_token_ids.stride(0));
    return {nt, vc};
}

static std::tuple<torch::Tensor, torch::Tensor> py_eagle_step_slot_mapping(
        torch::Tensor seq_lens, torch::Tensor positions, torch::Tensor block_table,
        int64_t block_size, int64_t max_model_len, int64_t pad_id, int64_t batch_size,
        int64_t input_batch_size, int64_t n_blocks_per_req) {
    FCK(seq_lens); FCK(positions); FCK(block_table);
    auto cp = torch::empty({input_batch_size}, positions.options());
    auto sm = torch::empty({input_batch_size}, positions.options());
    tmeg::eagle_step_slot_mapping_metadata<<<(input_batch_size + 127) / 128, 128, 0, fst()>>>(
        seq_lens.data_ptr<int>(), cp.data_ptr<int64_t>(), sm.data_ptr<int64_t>(),
        positions.data_ptr<int64_t>(), block_table.data_ptr<int>(), block_size, max_model_len,
        pad_id, int(batch_size), int(input_batch_size), block_table.stride(0), n_blocks_per_req);
    return {cp, sm};
}

static torch::Tensor py_eagle_expand_int64(torch::Tensor input, torch::Tensor cu_num_tokens,
        int64_t total, int64_t replace_from, int64_t replace_to) {
    FCK(input); FCK(cu_num_tokens);
    const int B = input.numel();
    auto out = torch::empty({total}, input.options());
    tmeg::eagle_expand_int64<<<(B + 127) / 128, 128, 0, fst()>>>(out.data_ptr<int64_t>(),
        input.data_ptr<int64_t>(), cu_num_tokens.data_ptr<int64_t>(), replace_from, replace_to, B);
    return out;
}

static std::vector<torch::Tensor> py_copy_and_expand_eagle(torch::Tensor target_token_ids,
        torch::Tensor target_positions, torch::Tensor next_token_ids, torch::Tensor query_start_loc,
        torch::Tensor query_end_loc, int64_t padding_token_id, int64_t parallel_drafting_token_id,
        int64_t num_padding_slots_per_request, int64_t shift_input_ids, int64_t out_len) {
    FCK(target_token_ids); FCK(target_positions); FCK(next_token_ids);
    FCK(query_start_loc); FCK(query_end_loc);
    const int R = next_token_ids.numel();
    const long total_in = target_token_ids.numel();
    auto lopt = target_token_ids.options();
    auto iopt = query_start_loc.options();
    auto bopt = lopt.dtype(torch::kUInt8);
    auto oi = torch::zeros({out_len}, lopt);
    auto op = torch::zeros({out_len}, lopt);
    auto rej = torch::zeros({out_len}, bopt);
    auto msk = torch::zeros({out_len}, bopt);
    auto ni = torch::zeros({(long)R * num_padding_slots_per_request}, iopt);
    auto hm = torch::zeros({total_in}, iopt);
    tmeg::copy_and_expand_eagle_inputs<<<(R + 127) / 128, 128, 0, fst()>>>(
        target_token_ids.data_ptr<int64_t>(), target_positions.data_ptr<int64_t>(),
        next_token_ids.data_ptr<int64_t>(), oi.data_ptr<int64_t>(), op.data_ptr<int64_t>(),
        rej.data_ptr<uint8_t>(), msk.data_ptr<uint8_t>(), ni.data_ptr<int>(), hm.data_ptr<int>(),
        query_start_loc.data_ptr<int>(), query_end_loc.data_ptr<int>(), padding_token_id,
        parallel_drafting_token_id, total_in, num_padding_slots_per_request, int(shift_input_ids), R);
    return {oi, op, rej, msk, ni, hm};
}

// ---------------- lightning-indexer cp_gather ----------------
static std::tuple<torch::Tensor, torch::Tensor> py_cp_gather_indexer(torch::Tensor kv_cache,
        torch::Tensor block_table, torch::Tensor cu_seq_lens, int64_t head_dim,
        int64_t cache_block_size, int64_t num_tokens, int64_t quant_block_size) {
    FCK(kv_cache); FCK(block_table); FCK(cu_seq_lens);
    const int batch = block_table.size(0), num_blocks = block_table.size(1);
    const long block_stride = kv_cache.stride(0);
    const long token_stride = head_dim;
    const int nqb = int(head_dim) / int(quant_block_size);
    auto dst_k = torch::zeros({num_tokens, head_dim}, kv_cache.options());
    auto dst_scale = torch::zeros({num_tokens, nqb}, kv_cache.options().dtype(torch::kFloat32));
    dim3 grid(int(num_tokens), nqb);
    tmsp::cp_gather_indexer_k_quant_cache<<<grid, 32, 0, fst()>>>(kv_cache.data_ptr<uint8_t>(),
        dst_k.data_ptr<uint8_t>(), reinterpret_cast<uint8_t*>(dst_scale.data_ptr<float>()),
        block_table.data_ptr<int>(), cu_seq_lens.data_ptr<int>(), batch, token_stride,
        int(head_dim), block_stride, int(cache_block_size), num_blocks, int(num_tokens),
        int(quant_block_size));
    return {dst_k, dst_scale};
}

// ---------------- fp8-output merge_attn_states (fp16 in) ----------------
static std::tuple<torch::Tensor, torch::Tensor> py_merge_attn_states_fp8(
        torch::Tensor prefix_out, torch::Tensor prefix_lse, torch::Tensor suffix_out,
        torch::Tensor suffix_lse, int64_t prefix_num_tokens, double output_scale) {
    FCK(prefix_out); FCK(prefix_lse); FCK(suffix_out); FCK(suffix_lse);
    const int T = prefix_out.size(0), H = prefix_out.size(1), D = prefix_out.size(2);
    auto out = torch::empty({T, H, D}, prefix_out.options().dtype(torch::kUInt8));
    auto olse = torch::empty({T, H}, prefix_lse.options());
    const long n = (long)T * H * D;
    tmsp::merge_attn_states_fp8<half><<<(n + 255) / 256, 256, 0, fst()>>>(out.data_ptr<uint8_t>(),
        olse.data_ptr<float>(), fhp(prefix_out), prefix_lse.data_ptr<float>(), fhp(suffix_out),
        suffix_lse.data_ptr<float>(), H, D, T, int(prefix_num_tokens), float(output_scale));
    return {out, olse};
}

// ---------------- TurboQuant tq_encode (fp32) ----------------
static std::vector<torch::Tensor> py_tq_encode(torch::Tensor key, torch::Tensor value,
        torch::Tensor slot_mapping, torch::Tensor v_centroids, torch::Tensor sign,
        int64_t num_slots, int64_t block_size, int64_t k_bits, int64_t k_signed, int64_t v_bits) {
    FCK(key); FCK(value); FCK(slot_mapping); FCK(v_centroids); FCK(sign);
    const int NT = key.size(0), NKV = key.size(1), HS = key.size(2);
    const int scale_groups = HS / 32;
    const int k_packed = (HS * int(k_bits) + 7) / 8, v_packed = (HS * int(v_bits) + 7) / 8;
    auto u8 = key.options().dtype(torch::kUInt8);
    auto h16 = key.options().dtype(torch::kFloat16);
    auto kc = torch::zeros({num_slots * NKV * k_packed}, u8);
    auto vc = torch::zeros({num_slots * NKV * v_packed}, u8);
    auto ks = torch::zeros({num_slots * NKV * scale_groups}, h16);
    auto vs = torch::zeros({num_slots * NKV * scale_groups}, h16);
    auto kz = torch::zeros({num_slots * NKV * scale_groups}, h16);
    dim3 grid(NT, NKV);
    auto* kcp = kc.data_ptr<uint8_t>(); auto* vcp = vc.data_ptr<uint8_t>();
    auto* ksp = reinterpret_cast<half*>(ks.data_ptr()); auto* vsp = reinterpret_cast<half*>(vs.data_ptr());
    auto* kzp = reinterpret_cast<half*>(kz.data_ptr());
    #define TQ(HH) tm6::tq_encode<float, HH><<<grid, HH, 0, fst()>>>(key.data_ptr<float>(), \
        value.data_ptr<float>(), kcp, vcp, ksp, vsp, kzp, slot_mapping.data_ptr<int64_t>(), \
        v_centroids.data_ptr<float>(), sign.data_ptr<float>(), NKV, int(block_size), \
        int(k_bits), int(k_signed), int(v_bits))
    if (HS == 64) TQ(64); else if (HS == 128) TQ(128);
    else if (HS == 256) TQ(256); else if (HS == 512) TQ(512);
    else TORCH_CHECK(false, "head_dim must be 64/128/256/512");
    #undef TQ
    return {kc, vc, ks, vs, kz};
}

// ---------------- Mamba selective_scan APC (fp32) ----------------
static torch::Tensor py_selective_scan_apc(torch::Tensor u, torch::Tensor delta, torch::Tensor A,
        torch::Tensor B, torch::Tensor C, torch::Tensor D, torch::Tensor delta_bias, torch::Tensor z,
        torch::Tensor query_start_loc, torch::Tensor cache_indices, torch::Tensor has_initial_state,
        torch::Tensor state, torch::Tensor block_idx_first, torch::Tensor block_idx_last,
        torch::Tensor initial_state_idx, torch::Tensor cu_chunk_seqlen, torch::Tensor last_chunk_indices,
        int64_t dstate, int64_t n_groups, int64_t has_d, int64_t has_delta_bias, int64_t has_z,
        int64_t delta_softplus, int64_t null_block_id, int64_t block_size, int64_t use_chunk_metadata) {
    FCK(u); FCK(delta); FCK(A); FCK(B); FCK(C); FCK(state);
    const int dim = u.size(0), total = u.size(1), batch = query_start_loc.numel() - 1;
    auto out = torch::empty({dim, total}, u.options());
    const int cache_indices_stride = cache_indices.size(1);
    const int block = (int(dstate) + 31) / 32 * 32;
    const size_t shbytes = ((block + 31) / 32) * sizeof(float);
    dim3 grid(dim, batch);
    tmss::selective_scan_fwd_varlen_apc<float><<<grid, block, shbytes, fst()>>>(
        u.data_ptr<float>(), delta.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), D.data_ptr<float>(), delta_bias.data_ptr<float>(), z.data_ptr<float>(),
        query_start_loc.data_ptr<int>(), cache_indices.data_ptr<int>(),
        has_initial_state.data_ptr<uint8_t>(), out.data_ptr<float>(), state.data_ptr<float>(),
        block_idx_first.data_ptr<int>(), block_idx_last.data_ptr<int>(),
        initial_state_idx.data_ptr<int>(), cu_chunk_seqlen.data_ptr<int>(),
        last_chunk_indices.data_ptr<int>(), batch, dim, total, int(dstate), int(n_groups),
        int(has_d), int(has_delta_bias), int(has_z), int(delta_softplus), int(null_block_id),
        int(block_size), cache_indices_stride, int(use_chunk_metadata));
    return out;
}

void init_mf_followups(py::module_& m) {
    m.def("eagle_prepare_next_token", &py_eagle_prepare_next_token, py::arg("sampled_token_ids"),
          py::arg("discard_request_mask"), py::arg("backup_next_token_ids"), py::arg("vocab_size"));
    m.def("eagle_step_slot_mapping", &py_eagle_step_slot_mapping, py::arg("seq_lens"),
          py::arg("positions"), py::arg("block_table"), py::arg("block_size"), py::arg("max_model_len"),
          py::arg("pad_id"), py::arg("batch_size"), py::arg("input_batch_size"), py::arg("n_blocks_per_req"));
    m.def("eagle_expand_int64", &py_eagle_expand_int64, py::arg("input"), py::arg("cu_num_tokens"),
          py::arg("total"), py::arg("replace_from"), py::arg("replace_to"));
    m.def("copy_and_expand_eagle", &py_copy_and_expand_eagle, py::arg("target_token_ids"),
          py::arg("target_positions"), py::arg("next_token_ids"), py::arg("query_start_loc"),
          py::arg("query_end_loc"), py::arg("padding_token_id"), py::arg("parallel_drafting_token_id"),
          py::arg("num_padding_slots_per_request"), py::arg("shift_input_ids"), py::arg("out_len"));
    m.def("cp_gather_indexer", &py_cp_gather_indexer, py::arg("kv_cache"), py::arg("block_table"),
          py::arg("cu_seq_lens"), py::arg("head_dim"), py::arg("cache_block_size"),
          py::arg("num_tokens"), py::arg("quant_block_size"));
    m.def("merge_attn_states_fp8", &py_merge_attn_states_fp8, py::arg("prefix_out"),
          py::arg("prefix_lse"), py::arg("suffix_out"), py::arg("suffix_lse"),
          py::arg("prefix_num_tokens"), py::arg("output_scale"));
    m.def("tq_encode", &py_tq_encode, py::arg("key"), py::arg("value"), py::arg("slot_mapping"),
          py::arg("v_centroids"), py::arg("sign"), py::arg("num_slots"), py::arg("block_size"),
          py::arg("k_bits"), py::arg("k_signed"), py::arg("v_bits"));
    m.def("selective_scan_apc", &py_selective_scan_apc, py::arg("u"), py::arg("delta"), py::arg("A"),
          py::arg("B"), py::arg("C"), py::arg("D"), py::arg("delta_bias"), py::arg("z"),
          py::arg("query_start_loc"), py::arg("cache_indices"), py::arg("has_initial_state"),
          py::arg("state"), py::arg("block_idx_first"), py::arg("block_idx_last"),
          py::arg("initial_state_idx"), py::arg("cu_chunk_seqlen"), py::arg("last_chunk_indices"),
          py::arg("dstate"), py::arg("n_groups"), py::arg("has_d"), py::arg("has_delta_bias"),
          py::arg("has_z"), py::arg("delta_softplus"), py::arg("null_block_id"), py::arg("block_size"),
          py::arg("use_chunk_metadata"));
}
