// QuixiCore-HIP serving bindings for gfx942.
//
// The ROCm counterpart of tm_cuda/tm_cuda_serving.cu, carrying only the batch
// prep / slot mapping / indexer metadata ops -- the ones that replace Triton on
// the MI300X serving path. The kernels themselves are NOT duplicated: this unit
// includes the same two headers the CUDA build does, and hipify converts them
// in the build tree. Everything else in the CUDA unit (Ampere MMA indexer
// logits, paged attention, the sampling and MLA kernels) is deliberately absent
// -- on MI300X those roles belong to AITER and _rocm_C.
#include "slot_mapping_kernels.cuh"
#include "v2_batch_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using namespace tms;
namespace py = pybind11;

#define CK(x) \
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

// Native replacement for the DSA indexer's Triton metadata kernel.
static void py_indexer_metadata(
    torch::Tensor query_start_loc, torch::Tensor uncompressed_seq_lens,
    torch::Tensor cu_compressed_seq_lens, torch::Tensor row_start_cu,
    torch::Tensor token_to_seq, torch::Tensor cu_ks, torch::Tensor cu_ke,
    int64_t query_slice_start, int64_t query_slice_stop, int64_t dcp_rank,
    int64_t dcp_world, int64_t dcp_interleave, int64_t compress_ratio) {
  CK(query_start_loc);
  CK(uncompressed_seq_lens);
  CK(cu_compressed_seq_lens);
  CK(row_start_cu);
  CK(token_to_seq);
  CK(cu_ks);
  CK(cu_ke);
  const int num_reqs = (int)query_start_loc.size(0) - 1;
  if (num_reqs <= 0) return;
  indexer_metadata<<<num_reqs, 256, 0, stream()>>>(
      query_start_loc.data_ptr<int>(), uncompressed_seq_lens.data_ptr<int>(),
      cu_compressed_seq_lens.data_ptr<int>(), row_start_cu.data_ptr<int>(),
      token_to_seq.data_ptr<int>(), cu_ks.data_ptr<int>(),
      cu_ke.data_ptr<int>(), (int)query_slice_start, (int)query_slice_stop,
      (int)dcp_rank, (int)dcp_world, (int)dcp_interleave, (int)compress_ratio);
}

// Native replacement for vLLM's Triton _compute_slot_mapping_kernel.
static void py_compute_slot_mapping(
    torch::Tensor query_start_loc, torch::Tensor positions,
    torch::Tensor block_table, torch::Tensor slot_mapping, int64_t num_tokens,
    int64_t max_num_tokens, int64_t block_size, int64_t kv_cache_block_size,
    int64_t blocks_per_kv_block, int64_t cp_world, int64_t cp_rank,
    int64_t cp_interleave, int64_t pad_id) {
  CK(query_start_loc);
  CK(positions);
  CK(block_table);
  CK(slot_mapping);
  TORCH_CHECK(positions.scalar_type() == torch::kLong,
              "positions must be int64");
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
#define CKD(x, d)                                             \
  do {                                                        \
    CK(x);                                                    \
    TORCH_CHECK((x).scalar_type() == (d), #x " must be " #d); \
  } while (0)

static void py_apply_write(torch::Tensor out, int64_t row_stride,
                           torch::Tensor indices, torch::Tensor starts,
                           torch::Tensor contents, torch::Tensor cu_lens) {
  CK(out);
  CKD(indices, torch::kInt);
  CKD(starts, torch::kInt);
  CK(contents);
  CKD(cu_lens, torch::kInt);
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

static void py_apply_write_multi(torch::Tensor out_ptrs,
                                 torch::Tensor out_strides,
                                 torch::Tensor indices, torch::Tensor starts,
                                 torch::Tensor contents, torch::Tensor cu_lens,
                                 torch::Tensor group_ids) {
  CKD(out_ptrs, torch::kUInt64);
  CKD(out_strides, torch::kLong);
  CKD(indices, torch::kInt);
  CKD(starts, torch::kInt);
  CKD(contents, torch::kInt);
  CKD(cu_lens, torch::kInt);
  CKD(group_ids, torch::kInt);
  const int n = indices.numel();
  if (n == 0) return;
  apply_write_multi<<<n, 256, 0, stream()>>>(
      reinterpret_cast<const unsigned long long*>(out_ptrs.data_ptr()),
      out_strides.data_ptr<int64_t>(), indices.data_ptr<int>(),
      starts.data_ptr<int>(), contents.data_ptr<int>(), cu_lens.data_ptr<int>(),
      group_ids.data_ptr<int>());
}

static void py_prepare_pos_seq_lens(torch::Tensor pos, torch::Tensor seq_lens,
                                    torch::Tensor idx_mapping,
                                    torch::Tensor query_start_loc,
                                    torch::Tensor num_computed_tokens,
                                    int64_t max_num_reqs) {
  CKD(pos, torch::kLong);
  CKD(seq_lens, torch::kInt);
  CKD(idx_mapping, torch::kInt);
  CKD(query_start_loc, torch::kInt);
  CKD(num_computed_tokens, torch::kInt);
  const int num_reqs = idx_mapping.numel();
  prepare_pos_seq_lens<<<num_reqs + 1, 256, 0, stream()>>>(
      reinterpret_cast<long*>(pos.data_ptr()), seq_lens.data_ptr<int>(),
      idx_mapping.data_ptr<int>(), query_start_loc.data_ptr<int>(),
      num_computed_tokens.data_ptr<int>(), (int)max_num_reqs);
}

static void py_prepare_prefill_inputs(
    torch::Tensor input_ids, torch::Tensor next_prefill_tokens,
    torch::Tensor idx_mapping, torch::Tensor query_start_loc,
    torch::Tensor all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor prefill_lens, torch::Tensor num_computed_tokens) {
  CKD(input_ids, torch::kInt);
  CKD(next_prefill_tokens, torch::kInt);
  CKD(idx_mapping, torch::kInt);
  CKD(query_start_loc, torch::kInt);
  CKD(all_token_ids, torch::kInt);
  CKD(prefill_lens, torch::kInt);
  CKD(num_computed_tokens, torch::kInt);
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) return;
  prepare_prefill_inputs<<<num_reqs, 256, 0, stream()>>>(
      input_ids.data_ptr<int>(), next_prefill_tokens.data_ptr<int>(),
      idx_mapping.data_ptr<int>(), query_start_loc.data_ptr<int>(),
      all_token_ids.data_ptr<int>(), (long)all_token_ids_stride,
      prefill_lens.data_ptr<int>(), num_computed_tokens.data_ptr<int>());
}

static void py_combine_sampled_and_draft_tokens(
    torch::Tensor input_ids, torch::Tensor idx_mapping,
    torch::Tensor last_sampled_tokens, torch::Tensor query_start_loc,
    torch::Tensor seq_lens, torch::Tensor prefill_len,
    torch::Tensor draft_tokens, int64_t draft_tokens_stride,
    torch::Tensor cu_num_logits, torch::Tensor logits_indices,
    int64_t num_new_sampled_tokens) {
  CKD(input_ids, torch::kInt);
  CKD(idx_mapping, torch::kInt);
  CKD(last_sampled_tokens, torch::kLong);
  CKD(query_start_loc, torch::kInt);
  CKD(seq_lens, torch::kInt);
  CKD(prefill_len, torch::kInt);
  CKD(draft_tokens, torch::kLong);
  CKD(cu_num_logits, torch::kInt);
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
                                            torch::Tensor num_rejected,
                                            torch::Tensor seq_lens,
                                            torch::Tensor cu_num_logits,
                                            torch::Tensor idx_mapping,
                                            torch::Tensor prefill_len) {
  CKD(num_sampled, torch::kInt);
  CKD(num_rejected, torch::kInt);
  CKD(seq_lens, torch::kInt);
  CKD(cu_num_logits, torch::kInt);
  CKD(idx_mapping, torch::kInt);
  CKD(prefill_len, torch::kInt);
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) return;
  get_num_sampled_and_rejected<<<(num_reqs + 255) / 256, 256, 0, stream()>>>(
      num_sampled.data_ptr<int>(), num_rejected.data_ptr<int>(),
      seq_lens.data_ptr<int>(), cu_num_logits.data_ptr<int>(),
      idx_mapping.data_ptr<int>(), prefill_len.data_ptr<int>(), num_reqs);
}

static void py_post_update(
    torch::Tensor idx_mapping, torch::Tensor num_computed_tokens,
    torch::Tensor last_sampled_tokens,
    c10::optional<torch::Tensor> output_bin_counts,
    int64_t output_bin_counts_stride, torch::Tensor sampled_tokens,
    int64_t sampled_tokens_stride, torch::Tensor num_sampled,
    torch::Tensor num_rejected, c10::optional<torch::Tensor> query_start_loc,
    torch::Tensor all_token_ids, int64_t all_token_ids_stride,
    torch::Tensor total_len) {
  CKD(idx_mapping, torch::kInt);
  CKD(num_computed_tokens, torch::kInt);
  CKD(last_sampled_tokens, torch::kLong);
  CKD(sampled_tokens, torch::kLong);
  CKD(num_sampled, torch::kInt);
  CKD(num_rejected, torch::kInt);
  CKD(all_token_ids, torch::kInt);
  CKD(total_len, torch::kInt);
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

static void py_post_update_num_computed_tokens(
    torch::Tensor idx_mapping, torch::Tensor num_computed_tokens,
    torch::Tensor query_start_loc) {
  CKD(idx_mapping, torch::kInt);
  CKD(num_computed_tokens, torch::kInt);
  CKD(query_start_loc, torch::kInt);
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) return;
  post_update_num_computed_tokens<<<(num_reqs + 255) / 256, 256, 0, stream()>>>(
      idx_mapping.data_ptr<int>(), num_computed_tokens.data_ptr<int>(),
      query_start_loc.data_ptr<int>(), num_reqs);
}

static void py_expand_idx_mapping(torch::Tensor idx_mapping,
                                  torch::Tensor expanded_idx_mapping,
                                  torch::Tensor expanded_local_pos,
                                  torch::Tensor cu_num_logits) {
  CKD(idx_mapping, torch::kInt);
  CKD(expanded_idx_mapping, torch::kInt);
  CKD(expanded_local_pos, torch::kInt);
  CKD(cu_num_logits, torch::kInt);
  const int num_reqs = idx_mapping.numel();
  if (num_reqs == 0) return;
  expand_idx_mapping<<<num_reqs, 256, 0, stream()>>>(
      idx_mapping.data_ptr<int>(), expanded_idx_mapping.data_ptr<int>(),
      expanded_local_pos.data_ptr<int>(), cu_num_logits.data_ptr<int>());
}

static void py_gather_block_tables(
    torch::Tensor idx_mapping, torch::Tensor src_ptrs, torch::Tensor dst_ptrs,
    torch::Tensor strides, torch::Tensor num_blocks, int64_t num_blocks_stride,
    int64_t num_reqs, int64_t num_reqs_padded) {
  CKD(idx_mapping, torch::kInt);
  CKD(src_ptrs, torch::kUInt64);
  CKD(dst_ptrs, torch::kUInt64);
  CKD(strides, torch::kLong);
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

static void py_compute_slot_mappings(
    torch::Tensor idx_mapping, torch::Tensor query_start_loc, torch::Tensor pos,
    torch::Tensor block_table_ptrs, torch::Tensor block_table_strides,
    torch::Tensor block_sizes, torch::Tensor slot_mappings,
    int64_t slot_mappings_stride, int64_t max_num_tokens, int64_t cp_rank,
    int64_t cp_size, int64_t cp_interleave, int64_t pad_id) {
  CKD(idx_mapping, torch::kInt);
  CKD(query_start_loc, torch::kInt);
  CKD(pos, torch::kLong);
  CKD(block_table_ptrs, torch::kUInt64);
  CKD(block_table_strides, torch::kLong);
  CKD(block_sizes, torch::kInt);
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

static void py_prepare_uniform_decode(
    torch::Tensor seq_lens, torch::Tensor decode_seq_lens,
    torch::Tensor block_table, int64_t block_table_stride,
    torch::Tensor expanded_block_table, int64_t expanded_bt_stride,
    torch::Tensor decode_lens, int64_t max_decode_len,
    int64_t num_decode_tokens) {
  CKD(seq_lens, torch::kInt);
  CKD(decode_seq_lens, torch::kInt);
  CKD(block_table, torch::kInt);
  CKD(expanded_block_table, torch::kInt);
  CKD(decode_lens, torch::kInt);
  if (num_decode_tokens == 0) return;
  prepare_uniform_decode<<<(unsigned)num_decode_tokens, 256, 0, stream()>>>(
      seq_lens.data_ptr<int>(), decode_seq_lens.data_ptr<int>(),
      block_table.data_ptr<int>(), (long)block_table_stride,
      expanded_block_table.data_ptr<int>(), (long)expanded_bt_stride,
      decode_lens.data_ptr<int>(), (int)max_decode_len);
}

static void py_prepare_dflash_inputs(
    torch::Tensor out_input_ids, torch::Tensor out_query_positions,
    torch::Tensor out_query_start_loc, torch::Tensor out_seq_lens,
    torch::Tensor out_query_slot_mapping, torch::Tensor out_context_positions,
    torch::Tensor out_context_slot_mapping, torch::Tensor out_sample_indices,
    torch::Tensor out_sample_pos, torch::Tensor out_sample_idx_mapping,
    torch::Tensor target_positions, torch::Tensor target_query_start_loc,
    torch::Tensor idx_mapping, torch::Tensor last_sampled,
    torch::Tensor next_prefill_tokens, torch::Tensor num_sampled,
    torch::Tensor num_rejected, torch::Tensor block_table,
    int64_t block_table_stride, int64_t parallel_drafting_token_id,
    int64_t block_size, int64_t num_query_per_req,
    int64_t num_speculative_steps, int64_t max_num_reqs, int64_t max_num_tokens,
    int64_t max_model_len, bool sample_from_anchor, int64_t pad_slot_id,
    int64_t num_reqs, int64_t max_tokens_per_req) {
  CKD(out_input_ids, torch::kInt);
  CKD(out_query_positions, torch::kLong);
  CKD(out_query_start_loc, torch::kInt);
  CKD(out_seq_lens, torch::kInt);
  CKD(out_query_slot_mapping, torch::kLong);
  CKD(out_context_positions, torch::kLong);
  CKD(out_context_slot_mapping, torch::kLong);
  CKD(out_sample_indices, torch::kLong);
  CKD(out_sample_pos, torch::kLong);
  CKD(out_sample_idx_mapping, torch::kInt);
  CKD(target_positions, torch::kLong);
  CKD(target_query_start_loc, torch::kInt);
  CKD(idx_mapping, torch::kInt);
  CKD(last_sampled, torch::kLong);
  CKD(next_prefill_tokens, torch::kInt);
  CKD(num_sampled, torch::kInt);
  CKD(num_rejected, torch::kInt);
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
      (int)parallel_drafting_token_id, (int)block_size, (int)num_query_per_req,
      (int)num_speculative_steps, (int)max_num_reqs, (long)max_num_tokens,
      (long)max_model_len, sample_from_anchor ? 1 : 0, (long)pad_slot_id,
      (int)max_tokens_per_req);
}

// qc_rocm_sample.cu: the V2 sampler / spec-decode ops. Split across translation
// units the way the CUDA build is, but only one may define the module.
void init_sample(py::module_& m);
// qc_rocm_sparse.cu: the ROCm sparse-MLA indexer index kernels.
void init_sparse(py::module_& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "QuixiCore-HIP serving kernels (gfx942)";
  init_sample(m);
  init_sparse(m);
  m.def("indexer_metadata", &py_indexer_metadata, py::arg("query_start_loc"),
        py::arg("uncompressed_seq_lens"), py::arg("cu_compressed_seq_lens"),
        py::arg("row_start_cu"), py::arg("token_to_seq"), py::arg("cu_ks"),
        py::arg("cu_ke"), py::arg("query_slice_start"),
        py::arg("query_slice_stop"), py::arg("dcp_rank"), py::arg("dcp_world"),
        py::arg("dcp_interleave"), py::arg("compress_ratio"));
  m.def("compute_slot_mapping", &py_compute_slot_mapping,
        py::arg("query_start_loc"), py::arg("positions"),
        py::arg("block_table"), py::arg("slot_mapping"), py::arg("num_tokens"),
        py::arg("max_num_tokens"), py::arg("block_size"),
        py::arg("kv_cache_block_size"), py::arg("blocks_per_kv_block"),
        py::arg("cp_world"), py::arg("cp_rank"), py::arg("cp_interleave"),
        py::arg("pad_id"));
  m.def("apply_write", &py_apply_write, py::arg("out"), py::arg("row_stride"),
        py::arg("indices"), py::arg("starts"), py::arg("contents"),
        py::arg("cu_lens"));
  m.def("apply_write_multi", &py_apply_write_multi, py::arg("out_ptrs"),
        py::arg("out_strides"), py::arg("indices"), py::arg("starts"),
        py::arg("contents"), py::arg("cu_lens"), py::arg("group_ids"));
  m.def("prepare_pos_seq_lens", &py_prepare_pos_seq_lens, py::arg("pos"),
        py::arg("seq_lens"), py::arg("idx_mapping"), py::arg("query_start_loc"),
        py::arg("num_computed_tokens"), py::arg("max_num_reqs"));
  m.def("prepare_prefill_inputs", &py_prepare_prefill_inputs,
        py::arg("input_ids"), py::arg("next_prefill_tokens"),
        py::arg("idx_mapping"), py::arg("query_start_loc"),
        py::arg("all_token_ids"), py::arg("all_token_ids_stride"),
        py::arg("prefill_lens"), py::arg("num_computed_tokens"));
  m.def("combine_sampled_and_draft_tokens",
        &py_combine_sampled_and_draft_tokens, py::arg("input_ids"),
        py::arg("idx_mapping"), py::arg("last_sampled_tokens"),
        py::arg("query_start_loc"), py::arg("seq_lens"), py::arg("prefill_len"),
        py::arg("draft_tokens"), py::arg("draft_tokens_stride"),
        py::arg("cu_num_logits"), py::arg("logits_indices"),
        py::arg("num_new_sampled_tokens"));
  m.def("get_num_sampled_and_rejected", &py_get_num_sampled_and_rejected,
        py::arg("num_sampled"), py::arg("num_rejected"), py::arg("seq_lens"),
        py::arg("cu_num_logits"), py::arg("idx_mapping"),
        py::arg("prefill_len"));
  m.def("post_update", &py_post_update, py::arg("idx_mapping"),
        py::arg("num_computed_tokens"), py::arg("last_sampled_tokens"),
        py::arg("output_bin_counts"), py::arg("output_bin_counts_stride"),
        py::arg("sampled_tokens"), py::arg("sampled_tokens_stride"),
        py::arg("num_sampled"), py::arg("num_rejected"),
        py::arg("query_start_loc"), py::arg("all_token_ids"),
        py::arg("all_token_ids_stride"), py::arg("total_len"));
  m.def("post_update_num_computed_tokens", &py_post_update_num_computed_tokens,
        py::arg("idx_mapping"), py::arg("num_computed_tokens"),
        py::arg("query_start_loc"));
  m.def("expand_idx_mapping", &py_expand_idx_mapping, py::arg("idx_mapping"),
        py::arg("expanded_idx_mapping"), py::arg("expanded_local_pos"),
        py::arg("cu_num_logits"));
  m.def("gather_block_tables", &py_gather_block_tables, py::arg("idx_mapping"),
        py::arg("src_ptrs"), py::arg("dst_ptrs"), py::arg("strides"),
        py::arg("num_blocks"), py::arg("num_blocks_stride"),
        py::arg("num_reqs"), py::arg("num_reqs_padded"));
  m.def("compute_slot_mappings", &py_compute_slot_mappings,
        py::arg("idx_mapping"), py::arg("query_start_loc"), py::arg("pos"),
        py::arg("block_table_ptrs"), py::arg("block_table_strides"),
        py::arg("block_sizes"), py::arg("slot_mappings"),
        py::arg("slot_mappings_stride"), py::arg("max_num_tokens"),
        py::arg("cp_rank"), py::arg("cp_size"), py::arg("cp_interleave"),
        py::arg("pad_id"));
  m.def("prepare_uniform_decode", &py_prepare_uniform_decode,
        py::arg("seq_lens"), py::arg("decode_seq_lens"), py::arg("block_table"),
        py::arg("block_table_stride"), py::arg("expanded_block_table"),
        py::arg("expanded_bt_stride"), py::arg("decode_lens"),
        py::arg("max_decode_len"), py::arg("num_decode_tokens"));
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
}
