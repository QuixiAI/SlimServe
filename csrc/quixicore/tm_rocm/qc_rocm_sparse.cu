// QuixiCore-HIP sparse-MLA indexer bindings for gfx942.
//
// The ROCm sparse backend's own index kernels, which have no CUDA counterpart
// (see rocm/sparse_indexer_kernels.cuh). Registered into the module by
// qc_rocm_serving.cu, which owns the single PYBIND11_MODULE.
#include "fp8_mqa_logits_kernel.cuh"
#include "fp8_paged_mqa_logits_kernel.cuh"
#include "mfma_fp8_dot.cuh"
#include "sparse_indexer_kernels.cuh"

#include <c10/util/Float8_e4m3fnuz.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <algorithm>
#include <type_traits>

namespace py = pybind11;

#define SPCK(x) \
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous GPU")
static cudaStream_t spst() { return at::cuda::getCurrentCUDAStream(); }

static void py_convert_req_index_to_global_index(
    torch::Tensor req_id, torch::Tensor block_table,
    torch::Tensor token_indices, torch::Tensor cu_seqlens, torch::Tensor out,
    int64_t block_size, int64_t topk) {
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
      out.data_ptr<int>(), max_num_blocks_per_req, (int)block_size, (int)topk,
      block_table.stride(0), block_table.stride(1), token_indices.stride(0),
      token_indices.stride(1));
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
static void py_indexer_k_quant_and_cache(
    torch::Tensor k, torch::Tensor kv_cache, torch::Tensor kv_cache_scale,
    torch::Tensor slot_mapping, int64_t block_size, int64_t block_tile_size,
    int64_t head_tile_size, double fp8_max, int64_t shuffle) {
  SPCK(k);
  SPCK(slot_mapping);
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping i64");
  TORCH_CHECK(kv_cache_scale.scalar_type() == torch::kFloat, "scale fp32");
  const int num_tokens = (int)slot_mapping.size(0);
  if (num_tokens == 0) return;
  const int head_dim = (int)k.size(-1);
  // One wave per token. Measured on MI300X: 64 threads beat 128 at every
  // token count and by 1.6x at 8192 (perf/optimization_status.md in
  // QuixiCore-ROCm, 2026-08-01). amax is a max, so the reduction is
  // order-independent and the block size cannot change the result.
  const int threads = 64;

  using FP8 = c10::Float8_e4m3fnuz;
  auto* cache = reinterpret_cast<FP8*>(kv_cache.data_ptr());
  auto launch = [&](auto dummy) {
    using T = decltype(dummy);
    qcrocm::indexer_k_quant_and_cache<T, FP8>
        <<<num_tokens, threads, 0, spst()>>>(
            reinterpret_cast<const T*>(k.data_ptr()), cache,
            kv_cache_scale.data_ptr<float>(),
            reinterpret_cast<const long*>(slot_mapping.data_ptr()),
            kv_cache_scale.stride(0), kv_cache.stride(0), (int)block_size,
            head_dim, (int)block_tile_size, (int)head_tile_size, (float)fp8_max,
            (int)shuffle);
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

static void py_cp_gather_indexer_quant_cache(
    torch::Tensor kv_cache, torch::Tensor kv_cache_scale, torch::Tensor k_fp8,
    torch::Tensor k_scale, torch::Tensor block_table, torch::Tensor cu_seqlen,
    torch::Tensor token_to_seq, int64_t block_size, int64_t block_tile_size,
    int64_t head_tile_size, int64_t num_batches, int64_t num_blocks,
    int64_t shuffle) {
  SPCK(block_table);
  SPCK(cu_seqlen);
  SPCK(token_to_seq);
  TORCH_CHECK(k_scale.scalar_type() == torch::kFloat, "k_scale fp32");
  TORCH_CHECK(kv_cache_scale.scalar_type() == torch::kFloat,
              "cache scale fp32");
  const int num_tokens = (int)k_fp8.size(0);
  if (num_tokens == 0) return;
  const int head_dim = (int)k_fp8.size(-1);

  using FP8 = c10::Float8_e4m3fnuz;
  qcrocm::cp_gather_indexer_quant_cache<FP8><<<num_tokens, 64, 0, spst()>>>(
      reinterpret_cast<const FP8*>(kv_cache.data_ptr()),
      kv_cache_scale.data_ptr<float>(),
      reinterpret_cast<FP8*>(k_fp8.data_ptr()), k_scale.data_ptr<float>(),
      block_table.data_ptr<int>(), cu_seqlen.data_ptr<int>(),
      token_to_seq.data_ptr<int>(), (int)block_size, block_table.stride(0),
      kv_cache.stride(0), kv_cache_scale.stride(0), head_dim,
      (int)block_tile_size, (int)head_tile_size, num_tokens, (int)num_batches,
      (int)block_table.size(1), (int)num_blocks, (int)shuffle);
}

// Layout probe: one 16x16 MFMA tile per wave over the full K, so the fragment
// mapping can be checked against tl.dot in isolation before the full logits
// kernel depends on it. D[m = 4*(l/16)+v][n = l%16].
__global__ void mfma_dot_probe_k(const uint8_t* __restrict__ a,
                                 const uint8_t* __restrict__ b,
                                 float* __restrict__ out, int M, int N, int K) {
  const int l = threadIdx.x & 63;
  const int m0 = blockIdx.y * 16, n0 = blockIdx.x * 16;
  qcrocm::f32x4 acc = {0.f, 0.f, 0.f, 0.f};
  for (int k = 0; k < K; k += 32)
    acc = qcrocm::mfma_16x16x32_fp8(qcrocm::load_a_frag(a, K, m0, k),
                                    qcrocm::load_b_frag(b, K, n0, k), acc);
  const int n = n0 + (l & 15);
#pragma unroll
  for (int v = 0; v < 4; ++v)
    out[(long)(m0 + 4 * (l >> 4) + v) * N + n] = acc[v];
}

static void py_mfma_dot_probe(torch::Tensor a, torch::Tensor b,
                              torch::Tensor out) {
  const int M = (int)a.size(0), K = (int)a.size(1), N = (int)b.size(0);
  dim3 grid(N / 16, M / 16);
  mfma_dot_probe_k<<<grid, 64, 0, spst()>>>(
      reinterpret_cast<const uint8_t*>(a.data_ptr()),
      reinterpret_cast<const uint8_t*>(b.data_ptr()), out.data_ptr<float>(), M,
      N, K);
}

// Not wired into dispatch: the dot is verified bitwise but the epilogue
// ordering is still being pinned against Triton, so this stays a bound-but-
// unused op until the differential test is clean.
static void py_fp8_mqa_logits(torch::Tensor q, torch::Tensor kv,
                              torch::Tensor kv_scales, torch::Tensor weights,
                              torch::Tensor cu_start, torch::Tensor cu_end,
                              torch::Tensor logits) {
  const int M = (int)q.size(0), H = (int)q.size(1), D = (int)q.size(2);
  const int N = (int)kv.size(0);
  TORCH_CHECK(D == 128, "fp8_mqa_logits: D must be 128");
  TORCH_CHECK(H == 32 || H == 64, "fp8_mqa_logits: H must be 32 or 64");
  auto launch = [&](auto tag) {
    constexpr int HH = decltype(tag)::value;
    qcrocm::fp8_mqa_logits<HH, 128, 64, 4><<<M, 256, 0, spst()>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(kv.data_ptr()),
        kv_scales.data_ptr<float>(), weights.data_ptr<float>(),
        cu_start.data_ptr<int>(), cu_end.data_ptr<int>(),
        logits.data_ptr<float>(), q.stride(0), weights.stride(0),
        logits.stride(0), N);
  };
  if (H == 32)
    launch(std::integral_constant<int, 32>{});
  else
    launch(std::integral_constant<int, 64>{});
}

// Paged decode logits on the indexer cache (64-bit addressing; see
// fp8_paged_mqa_logits_kernel.cuh). kv_values: uint8 [NB, block_size*D]
// (block-strided view of the packed slab), kv_scales: float32 [NB, block_size]
// (same stride in floats), seq_lens: int32 [M], block_table: int32 [M, W],
// logits: fp32 [M, L] (only [0, seq_len) of each row is written).
static void py_fp8_paged_mqa_logits(torch::Tensor q, torch::Tensor kv_values,
                                    torch::Tensor kv_scales,
                                    torch::Tensor weights,
                                    torch::Tensor seq_lens,
                                    torch::Tensor block_table,
                                    torch::Tensor logits, int64_t block_size,
                                    int64_t block_tile, int64_t head_tile) {
  const int M = (int)q.size(0), H = (int)q.size(1), D = (int)q.size(2);
  TORCH_CHECK(D == 128, "fp8_paged_mqa_logits: D must be 128");
  TORCH_CHECK(H == 32 || H == 64, "fp8_paged_mqa_logits: H must be 32 or 64");
  TORCH_CHECK(block_tile == 16 && head_tile == 16,
              "fp8_paged_mqa_logits: 16x16 tiles only");
  TORCH_CHECK(block_size % 16 == 0, "fp8_paged_mqa_logits: block_size % 16");
  TORCH_CHECK(seq_lens.scalar_type() == torch::kInt &&
                  block_table.scalar_type() == torch::kInt,
              "fp8_paged_mqa_logits: int32 seq_lens/block_table");
  TORCH_CHECK(kv_values.stride(1) == 1 && kv_scales.stride(1) == 1,
              "fp8_paged_mqa_logits: cache views must be row-contiguous");
  TORCH_CHECK(
      logits.size(0) >= M && seq_lens.numel() >= M && block_table.size(0) >= M,
      "fp8_paged_mqa_logits: row counts");
  if (M == 0) return;
  const int max_len = (int)logits.size(1);
  // Split each row's context into 1024-key chunks across CTAs: the grid
  // (M rows x splits) is static per padded batch size (graph-safe) and
  // fills the GPU where one CTA per row could not.
  constexpr int kSplitLen = 1024;
  const int splits = std::max(1, (max_len + kSplitLen - 1) / kSplitLen);
  auto launch = [&](auto tag) {
    constexpr int HH = decltype(tag)::value;
    qcrocm::fp8_paged_mqa_logits<HH, 128, 64, 4, 16, 16>
        <<<dim3(M, splits), 256, 0, spst()>>>(
            reinterpret_cast<const uint8_t*>(q.data_ptr()),
            reinterpret_cast<const uint8_t*>(kv_values.data_ptr()),
            kv_scales.data_ptr<float>(), weights.data_ptr<float>(),
            seq_lens.data_ptr<int>(), block_table.data_ptr<int>(),
            logits.data_ptr<float>(), q.stride(0), weights.stride(0),
            logits.stride(0), kv_values.stride(0), kv_scales.stride(0),
            (int)block_table.stride(0), (int)block_size, max_len, kSplitLen);
  };
  if (H == 32)
    launch(std::integral_constant<int, 32>{});
  else
    launch(std::integral_constant<int, 64>{});
}

void init_sparse(py::module_& m) {
  m.def("mqa_logits_paged_gfx942", &py_fp8_paged_mqa_logits, py::arg("q"),
        py::arg("kv_values"), py::arg("kv_scales"), py::arg("weights"),
        py::arg("seq_lens"), py::arg("block_table"), py::arg("logits"),
        py::arg("block_size"), py::arg("block_tile"), py::arg("head_tile"));
  m.def("convert_req_index_to_global_index",
        &py_convert_req_index_to_global_index, py::arg("req_id"),
        py::arg("block_table"), py::arg("token_indices"), py::arg("cu_seqlens"),
        py::arg("out"), py::arg("block_size"), py::arg("topk"));
  m.def("indexer_k_quant_and_cache", &py_indexer_k_quant_and_cache,
        py::arg("k"), py::arg("kv_cache"), py::arg("kv_cache_scale"),
        py::arg("slot_mapping"), py::arg("block_size"),
        py::arg("block_tile_size"), py::arg("head_tile_size"),
        py::arg("fp8_max"), py::arg("shuffle"));
  m.def("cp_gather_indexer_quant_cache", &py_cp_gather_indexer_quant_cache,
        py::arg("kv_cache"), py::arg("kv_cache_scale"), py::arg("k_fp8"),
        py::arg("k_scale"), py::arg("block_table"), py::arg("cu_seqlen"),
        py::arg("token_to_seq"), py::arg("block_size"),
        py::arg("block_tile_size"), py::arg("head_tile_size"),
        py::arg("num_batches"), py::arg("num_blocks"), py::arg("shuffle"));
  m.def("mqa_logits_gfx942", &py_fp8_mqa_logits, py::arg("q"), py::arg("kv"),
        py::arg("kv_scales"), py::arg("weights"), py::arg("cu_start"),
        py::arg("cu_end"), py::arg("logits"));
  m.def("mfma_dot_probe", &py_mfma_dot_probe, py::arg("a"), py::arg("b"),
        py::arg("out"));
  m.def("generate_sparse_seqlen", &py_generate_sparse_seqlen,
        py::arg("seq_lens"), py::arg("cu_query_lens"), py::arg("out"),
        py::arg("topk_token"));
}
