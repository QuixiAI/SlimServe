// HIP MLA decode binding for gfx942.
//
// Replaces both the AITER path (hand-assembled, query-head count baked into
// the code object, so only multiples and divisors of 16 run) and the Triton
// fallback for absorbed MLA decode. The kernel treats the head count as a grid
// dimension, so Kimi K3 runs at any tensor-parallel size that divides its 96
// heads -- including TP8's 12 per rank.
//
// Bound into the single _quixicore_C module by qc_rocm_serving.cu.
#include "mla_decode_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace py = pybind11;
using namespace qc_rocm_mla;

#define CK(x) \
  TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")

static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }

// Split count, swept on MI300X (gfx942) at seq=3072, H=12, over B in
// {1,2,4,8,16,32}: the optimum is 48-64 slices at *every* batch size, i.e. it
// tracks the work in one slice, not the total wave count. Scaling slices down
// as batch grows -- the obvious "we already have enough parallelism" rule --
// costs 29% at B=16 and 50% at B=32, so it is not done here.
//
// ~48 KV tokens per slice is the knee. Below it the per-slice fixed cost (the
// query load, and a reduce pass that grows with the slice count) dominates;
// above it there is too little parallelism to hide the load latency.
static int choose_num_splits(int batch, int num_heads, int max_seq_len) {
  constexpr int kMinTokensPerSplit = 48;
  constexpr int kMaxSplits = 64;
  const int by_work = max_seq_len / kMinTokensPerSplit;
  return std::max(1, std::min(by_work, kMaxSplits));
}

// max_seq_len comes from the metadata builder rather than seq_lens.max():
// reading it off the device would sync the decode hot path and cannot be
// captured into a CUDA graph.
void mla_decode_fwd(torch::Tensor q, torch::Tensor kv_cache,
                    torch::Tensor block_table, torch::Tensor seq_lens,
                    torch::Tensor out, double scale, int64_t max_seq_len,
                    int64_t num_splits_override) {
  CK(q);
  CK(kv_cache);
  CK(block_table);
  CK(seq_lens);
  CK(out);
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kEntry, "q must be [B, H, ", kEntry,
              "]");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == kLatent, "out must be [B, H, ",
              kLatent, "]");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.size(2) == kEntry,
              "kv_cache must be [num_blocks, block_size, ", kEntry, "]");
  TORCH_CHECK(q.scalar_type() == out.scalar_type() &&
                  q.scalar_type() == kv_cache.scalar_type(),
              "q, kv_cache and out must share a dtype");
  TORCH_CHECK(block_table.scalar_type() == torch::kInt &&
                  seq_lens.scalar_type() == torch::kInt,
              "block_table and seq_lens must be int32");

  const int batch = (int)q.size(0);
  const int num_heads = (int)q.size(1);
  const int block_size = (int)kv_cache.size(1);
  const int bt_stride = (int)block_table.stride(0);
  if (batch == 0 || num_heads == 0) return;

  const int num_splits =
      num_splits_override > 0
          ? (int)num_splits_override
          : choose_num_splits(batch, num_heads, (int)max_seq_len);

  auto f32 = q.options().dtype(torch::kFloat);
  auto partial_out = torch::empty({batch, num_heads, num_splits, kLatent}, f32);
  auto partial_max = torch::empty({batch, num_heads, num_splits}, f32);
  auto partial_sum = torch::empty({batch, num_heads, num_splits}, f32);

  const dim3 grid1(num_heads, batch, num_splits);
  const dim3 grid2(num_heads, batch);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(q.scalar_type(), "mla_decode_fwd", [&] {
    using hip_t = std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
                                     __hip_bfloat16, _Float16>;
    mla_decode_partition<hip_t><<<grid1, kWave, 0, stream()>>>(
        reinterpret_cast<const hip_t*>(q.data_ptr<scalar_t>()),
        reinterpret_cast<const hip_t*>(kv_cache.data_ptr<scalar_t>()),
        block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
        partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(), num_heads, block_size, bt_stride,
        num_splits, (float)scale);
    mla_decode_reduce<hip_t><<<grid2, kWave, 0, stream()>>>(
        partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
        partial_sum.data_ptr<float>(),
        reinterpret_cast<hip_t*>(out.data_ptr<scalar_t>()), num_heads,
        num_splits);
  });
}

// Sparse (DSA top-k) decode over a possibly block-strided latent cache.
//
// `kv_cache` may be a layer's view into the packed cross-layer KV slab:
// rows are contiguous within a block, blocks are kv_cache.stride(0)
// elements apart. `kv_indices` are the global page-size-1 token ids vLLM's
// convert pass builds for the aiter path (-1 entries are skipped), spanned
// per query token by `kv_indptr`. One query row per (token, head), which
// under sparse_mla_force_mqa covers decode and prefill alike.
void mla_sparse_decode_fwd(torch::Tensor q, torch::Tensor kv_cache,
                           torch::Tensor kv_indptr, torch::Tensor kv_indices,
                           torch::Tensor out, double scale, int64_t max_topk,
                           int64_t num_splits_override) {
  CK(q);
  CK(kv_indptr);
  CK(kv_indices);
  CK(out);
  TORCH_CHECK(kv_cache.is_cuda(), "kv_cache must be CUDA");
  TORCH_CHECK(q.dim() == 3 && q.size(2) == kEntry, "q must be [T, H, ", kEntry,
              "]");
  TORCH_CHECK(out.dim() == 3 && out.size(2) == kLatent, "out must be [T, H, ",
              kLatent, "]");
  TORCH_CHECK(kv_cache.dim() == 3 && kv_cache.size(2) == kEntry,
              "kv_cache must be [num_blocks, block_size, ", kEntry, "]");
  // The packed slab hands out block-strided views: only the inner two dims
  // must be dense.
  TORCH_CHECK(kv_cache.stride(2) == 1 && kv_cache.stride(1) == kEntry,
              "kv_cache rows must be dense within a block");
  TORCH_CHECK(q.scalar_type() == out.scalar_type() &&
                  q.scalar_type() == kv_cache.scalar_type(),
              "q, kv_cache and out must share a dtype");
  TORCH_CHECK(kv_indptr.scalar_type() == torch::kInt &&
                  kv_indices.scalar_type() == torch::kInt,
              "kv_indptr and kv_indices must be int32");
  TORCH_CHECK(kv_indptr.size(0) == q.size(0) + 1, "kv_indptr must be [T + 1]");

  const int tokens = (int)q.size(0);
  const int num_heads = (int)q.size(1);
  const int block_size = (int)kv_cache.size(1);
  const int64_t cache_block_stride = kv_cache.stride(0);
  if (tokens == 0 || num_heads == 0) return;

  const int num_splits =
      num_splits_override > 0
          ? (int)num_splits_override
          : choose_num_splits(tokens, num_heads, (int)max_topk);

  auto f32 = q.options().dtype(torch::kFloat);
  auto partial_out =
      torch::empty({tokens, num_heads, num_splits, kLatent}, f32);
  auto partial_max = torch::empty({tokens, num_heads, num_splits}, f32);
  auto partial_sum = torch::empty({tokens, num_heads, num_splits}, f32);

  const dim3 grid1(num_heads, tokens, num_splits);
  const dim3 grid2(num_heads, tokens);

  AT_DISPATCH_REDUCED_FLOATING_TYPES(
      q.scalar_type(), "mla_sparse_decode_fwd", [&] {
        using hip_t = std::conditional_t<std::is_same_v<scalar_t, at::BFloat16>,
                                         __hip_bfloat16, _Float16>;
        mla_sparse_decode_partition<hip_t><<<grid1, kWave, 0, stream()>>>(
            reinterpret_cast<const hip_t*>(q.data_ptr<scalar_t>()),
            reinterpret_cast<const hip_t*>(kv_cache.data_ptr<scalar_t>()),
            kv_indptr.data_ptr<int>(), kv_indices.data_ptr<int>(),
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(), num_heads, block_size,
            cache_block_stride, num_splits, (float)scale);
        mla_decode_reduce<hip_t><<<grid2, kWave, 0, stream()>>>(
            partial_out.data_ptr<float>(), partial_max.data_ptr<float>(),
            partial_sum.data_ptr<float>(),
            reinterpret_cast<hip_t*>(out.data_ptr<scalar_t>()), num_heads,
            num_splits);
      });
}

void init_mla(py::module_& m) {
  m.def("mla_sparse_decode_fwd", &mla_sparse_decode_fwd, py::arg("q"),
        py::arg("kv_cache"), py::arg("kv_indptr"), py::arg("kv_indices"),
        py::arg("out"), py::arg("scale"), py::arg("max_topk"),
        py::arg("num_splits") = -1);
  m.def("mla_decode_fwd", &mla_decode_fwd, py::arg("q"), py::arg("kv_cache"),
        py::arg("block_table"), py::arg("seq_lens"), py::arg("out"),
        py::arg("scale"), py::arg("max_seq_len"), py::arg("num_splits") = -1);
}
