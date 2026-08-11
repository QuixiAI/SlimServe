// Persistent TopK kernel for DeepSeek V3 sparse attention indexer.
// See persistent_topk.cuh for kernel implementation.

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdlib>

#include "torch_utils.h"

#ifndef USE_ROCM
  #include "persistent_topk.cuh"
#endif

namespace {

#ifndef USE_ROCM
template <int TopK, int NGPU = 1>
void launch_persistent_topk(const torch::stable::Tensor& logits,
                            const torch::stable::Tensor& lengths,
                            torch::stable::Tensor& output,
                            torch::stable::Tensor& workspace,
                            int64_t max_seq_len,
                            vllm::persistent::PeerRankData* peer_data = nullptr,
                            vllm::persistent::PeerRankSignals peer_signals = {},
                            vllm::persistent::PeerSignal* self_signal = nullptr,
                            int rank = 0) {
  namespace P = vllm::persistent;

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const int64_t num_rows = logits.size(0);
  const int64_t stride = logits.stride(0);
  const cudaStream_t stream = get_current_cuda_stream();

  if constexpr (NGPU == 1 && TopK == 512) {
    static const bool short_radix_enabled = [] {
      const char* value = std::getenv("VLLM_DSV4_TOPK_CUB1024");
      return value != nullptr && value[0] == '1';
    }();
    if (short_radix_enabled && max_seq_len > TopK && max_seq_len <= 1024) {
      P::short_row_radix_topk_kernel<TopK>
          <<<num_rows, 256, 0, stream>>>(
              logits.const_data_ptr<float>(),
              lengths.const_data_ptr<int32_t>(),
              output.mutable_data_ptr<int32_t>(),
              static_cast<uint32_t>(stride));
      return;
    }
  }

  static int num_sms = 0;
  static int max_smem_per_block = 0;
  if (num_sms == 0) {
    const cudaDeviceProp* device_prop = get_device_prop();
    num_sms = device_prop->multiProcessorCount;
    max_smem_per_block = device_prop->sharedMemPerBlockOptin;
  }

  // Diagnostic dispatch override for the NaN-row hunt: rows above the
  // threshold take FilteredTopK, at or below it the persistent cooperative
  // kernel. Default 32 is the production heuristic; 0 forces FilteredTopK
  // for every batch, a huge value forces the persistent kernel for every
  // batch.
  static const uint32_t filtered_min_rows = [] {
    const char* value = std::getenv("VLLM_DSV4_FILTERED_TOPK_MIN_ROWS");
    return value == nullptr
               ? 32u
               : static_cast<uint32_t>(std::max(0L, std::atol(value)));
  }();

  if constexpr (NGPU == 1) {
    if (num_rows > filtered_min_rows && max_smem_per_block >= 128 * 1024) {
    cudaError_t status =
        vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
            logits.const_data_ptr<float>(), output.mutable_data_ptr<int32_t>(),
            lengths.const_data_ptr<int32_t>(), static_cast<uint32_t>(num_rows),
            static_cast<uint32_t>(TopK), static_cast<uint32_t>(stride), stream);
    STD_TORCH_CHECK(status == cudaSuccess,
                    "FilteredTopK failed: ", cudaGetErrorString(status));
      return;
    }
  }
  {
    STD_TORCH_CHECK(num_rows <= 32 || NGPU == 1,
                    "peer persistent_topk supports at most 32 rows");
    STD_TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA tensor");
    STD_TORCH_CHECK(
        workspace.scalar_type() == torch::headeronly::ScalarType::Byte,
        "workspace must be uint8");

    int effective_max_smem;
    if (num_rows <= 4) {
      effective_max_smem =
          std::min(max_smem_per_block, static_cast<int>(P::kSmemMedium));
    } else if (num_rows <= 8) {
      constexpr int kSmemCapMedium = 48 * 1024;
      effective_max_smem = std::min(max_smem_per_block, kSmemCapMedium);
    } else {
      effective_max_smem = max_smem_per_block;
    }

    size_t available_for_ordered =
        static_cast<size_t>(effective_max_smem) - P::kFixedSmemLarge;
    uint32_t max_chunk_elements =
        static_cast<uint32_t>(available_for_ordered / sizeof(uint32_t));

    uint32_t vec_size = 1;
    if (stride % 4 == 0)
      vec_size = 4;
    else if (stride % 2 == 0)
      vec_size = 2;

    max_chunk_elements = (max_chunk_elements / vec_size) * vec_size;
    uint32_t min_chunk = vec_size * P::kThreadsPerBlock;
    if (max_chunk_elements < min_chunk) max_chunk_elements = min_chunk;

    uint32_t ctas_per_group =
        (static_cast<uint32_t>(stride) + max_chunk_elements - 1) /
        max_chunk_elements;
    uint32_t chunk_size =
        (static_cast<uint32_t>(stride) + ctas_per_group - 1) / ctas_per_group;
    chunk_size = ((chunk_size + vec_size - 1) / vec_size) * vec_size;
    if (chunk_size > max_chunk_elements) chunk_size = max_chunk_elements;

    size_t smem_size = P::kFixedSmemLarge + chunk_size * sizeof(uint32_t);
    if (smem_size < P::kSmemMedium) smem_size = P::kSmemMedium;

    // Query occupancy for the instantiation that will actually launch;
    // overestimating it deadlocks the cooperative barrier.
    int occupancy = 1;
    cudaError_t occ_err = cudaSuccess;
    if (vec_size == 4) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 4, NGPU>,
          P::kThreadsPerBlock,
          smem_size);
    } else if (vec_size == 2) {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 2, NGPU>,
          P::kThreadsPerBlock,
          smem_size);
    } else {
      occ_err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &occupancy, P::persistent_topk_kernel<TopK, 1, NGPU>,
          P::kThreadsPerBlock,
          smem_size);
    }
    STD_TORCH_CHECK(occ_err == cudaSuccess,
                    "persistent_topk occupancy query failed: ",
                    cudaGetErrorString(occ_err));
    if (occupancy < 1) occupancy = 1;

    // The cooperative spin-wait barrier only runs when at least one row hits
    // the radix path (seq_len > RADIX_THRESHOLD). Below that, non-CTA-0 CTAs
    // early-exit, so oversubscription can't deadlock and headroom is wasted.
    const bool needs_cooperative =
        static_cast<uint32_t>(max_seq_len) > P::RADIX_THRESHOLD;

    const uint32_t hw_resident_cap =
        static_cast<uint32_t>(num_sms) * static_cast<uint32_t>(occupancy);
    uint32_t max_resident_ctas = hw_resident_cap;
    if (needs_cooperative) {
      // Reserve one CTA per SM when occupancy allows; fall back to a single
      // CTA when occupancy == 1 (the most deadlock-prone case — any straggler
      // kernel that takes the only slot on one SM hangs the barrier). Never
      // drop below one full group's worth.
      uint32_t headroom = (occupancy > 1) ? static_cast<uint32_t>(num_sms) : 1u;
      if (max_resident_ctas >= headroom + ctas_per_group) {
        max_resident_ctas -= headroom;
      }
    }
    uint32_t num_groups = std::min(max_resident_ctas / ctas_per_group,
                                   static_cast<uint32_t>(num_rows));
    if (num_groups == 0) num_groups = 1;
    uint32_t total_ctas = num_groups * ctas_per_group;

    // If the cooperative launch wouldn't fit, fall back to FilteredTopK
    // instead of deadlocking. Only relevant when needs_cooperative.
    if (needs_cooperative && total_ctas > hw_resident_cap) {
      STD_TORCH_CHECK(
          max_smem_per_block >= 128 * 1024,
          "persistent_topk would oversubscribe and the FilteredTopK "
          "fallback requires >=128KB smem per block (have ",
          max_smem_per_block, "). total_ctas=", total_ctas,
          " > num_sms*occupancy=", hw_resident_cap, " (TopK=", TopK,
          ", vec_size=", vec_size, ", ctas_per_group=", ctas_per_group,
          ", smem=", smem_size, ").");
      cudaError_t status =
          vllm::FilteredTopKRaggedTransform<float, int32_t, TopK>(
              logits.const_data_ptr<float>(),
              output.mutable_data_ptr<int32_t>(),
              lengths.const_data_ptr<int32_t>(),
              static_cast<uint32_t>(num_rows), static_cast<uint32_t>(TopK),
              static_cast<uint32_t>(stride), stream);
      STD_TORCH_CHECK(status == cudaSuccess, "FilteredTopK fallback failed: ",
                      cudaGetErrorString(status));
      return;
    }

    size_t state_bytes = num_groups * sizeof(P::RadixRowState);
    STD_TORCH_CHECK(workspace.size(0) >= static_cast<int64_t>(state_bytes),
                    "workspace too small, need ", state_bytes, " bytes");

    // Zero the per-group RadixRowState region before launch.
    //
    // Issued UNCONDITIONALLY so the memset is captured as its own node in
    // the cudagraph (a separate cudaMemsetAsync node, sequenced before the
    // persistent_topk_kernel launch on the same stream). The previous
    // host-side guard `if (needs_cooperative)` was evaluated at capture time;
    // when capture-time max_seq_len <= RADIX_THRESHOLD (always true under
    // FULL_DECODE_ONLY with max_model_len < 32 K) the memset would NOT be
    // captured, leaving the workspace state to accumulate across replays.
    // That's a latent correctness bug if the runtime data ever takes the
    // radix path, and removes one variable while debugging hangs in the
    // decode/medium paths.
    //
    // Cost is sub-microsecond: state_bytes = num_groups * sizeof(RadixRowState)
    // is ~3 KB per group, ~100 KB for the largest grids on this hardware.
    //
    // Why the memset is required (regardless of which path the kernel takes):
    //   1. arrival_counter accumulates within a launch and is never reset,
    //      so a prior call leaves it at a large positive value. Without this
    //      reset, the very first wait_ge in the next call sees counter >>
    //      target and returns instantly, breaking the barrier.
    //   2. The previous in-kernel init only ran in CTA-0 with intra-CTA
    //      __syncthreads(), so it had no happens-before edge to CTA-1+'s
    //      first red_release. cudaMemsetAsync is stream-ordered: the zero
    //      is globally visible before any CTA runs.
    {
      cudaError_t mz_err = cudaMemsetAsync(
          workspace.mutable_data_ptr<uint8_t>(), 0, state_bytes, stream);
      STD_TORCH_CHECK(mz_err == cudaSuccess,
                      "row_states memset failed: ", cudaGetErrorString(mz_err));
    }

    P::PersistentTopKParams params;
    params.input = logits.const_data_ptr<float>();
    params.output = output.mutable_data_ptr<int32_t>();
    params.lengths = lengths.const_data_ptr<int32_t>();
    params.num_rows = static_cast<uint32_t>(num_rows);
    params.stride = static_cast<uint32_t>(stride);
    params.top_k = static_cast<uint32_t>(TopK);
    params.chunk_size = chunk_size;
    params.row_states = reinterpret_cast<P::RadixRowState*>(
        workspace.mutable_data_ptr<uint8_t>());
    params.ctas_per_group = ctas_per_group;
    params.max_seq_len = static_cast<uint32_t>(max_seq_len);
    params.peer_data = peer_data;
    params.peer_signals = peer_signals;
    params.self_signal = self_signal;
    params.rank = rank;

  #define LAUNCH_PERSISTENT(TOPK_VAL, VS)                                     \
    do {                                                                      \
      auto kernel = &P::persistent_topk_kernel<TOPK_VAL, VS, NGPU>;           \
      cudaError_t err = cudaFuncSetAttribute(                                 \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);    \
      STD_TORCH_CHECK(err == cudaSuccess,                                     \
                      "Failed to set smem: ", cudaGetErrorString(err));       \
      if constexpr (NGPU == 1) {                                              \
        kernel<<<total_ctas, P::kThreadsPerBlock, smem_size, stream>>>(       \
            params);                                                          \
      } else {                                                                \
        void* args[] = {&params};                                             \
        err = cudaLaunchCooperativeKernel(                                    \
            reinterpret_cast<const void*>(kernel), dim3(total_ctas),          \
            dim3(P::kThreadsPerBlock), args, smem_size, stream);               \
        STD_TORCH_CHECK(err == cudaSuccess,                                   \
                        "peer persistent_topk launch failed: ",               \
                        cudaGetErrorString(err));                              \
      }                                                                       \
    } while (0)

    if (vec_size == 4) {
      LAUNCH_PERSISTENT(TopK, 4);
    } else if (vec_size == 2) {
      LAUNCH_PERSISTENT(TopK, 2);
    } else {
      LAUNCH_PERSISTENT(TopK, 1);
    }
  #undef LAUNCH_PERSISTENT
  }

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "persistent_topk failed: ", cudaGetErrorString(err));
}
#endif

}  // anonymous namespace

void persistent_topk(const torch::stable::Tensor& logits,
                     const torch::stable::Tensor& lengths,
                     torch::stable::Tensor& output,
                     torch::stable::Tensor& workspace, int64_t k,
                     int64_t max_seq_len) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda(), "logits must be CUDA tensor");
  STD_TORCH_CHECK(lengths.is_cuda(), "lengths must be CUDA tensor");
  STD_TORCH_CHECK(output.is_cuda(), "output must be CUDA tensor");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "Only float32 supported");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int,
                  "lengths must be int32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "output must be int32");
  STD_TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  STD_TORCH_CHECK(lengths.dim() == 1 || lengths.dim() == 2,
                  "lengths must be 1D or 2D");
  STD_TORCH_CHECK(lengths.is_contiguous(), "lengths must be contiguous");
  STD_TORCH_CHECK(output.dim() == 2, "output must be 2D");

  const int64_t num_rows = logits.size(0);

  STD_TORCH_CHECK(lengths.numel() == num_rows, "lengths size mismatch");
  STD_TORCH_CHECK(output.size(0) == num_rows && output.size(1) == k,
                  "output size mismatch");
  STD_TORCH_CHECK(
      k == 512 || k == 1024 || k == 2048,
      "persistent_topk supports k=512, k=1024, or k=2048, got k=", k);

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());

  if (k == 512) {
    launch_persistent_topk<512>(logits, lengths, output, workspace,
                                max_seq_len);
  } else if (k == 1024) {
    launch_persistent_topk<1024>(logits, lengths, output, workspace,
                                 max_seq_len);
  } else {
    launch_persistent_topk<2048>(logits, lengths, output, workspace,
                                 max_seq_len);
  }
#else
  STD_TORCH_CHECK(false, "persistent_topk is not supported on ROCm");
#endif
}

void launch_dsv4_indexer_peer_topk_impl(
    torch::stable::Tensor& logits,
    const torch::stable::Tensor& lengths, torch::stable::Tensor& output,
    torch::stable::Tensor& workspace, int64_t k, int64_t max_seq_len,
    int64_t peer_data_ptr, const std::array<int64_t, 8>& signal_ptrs,
    int64_t self_signal_ptr, int rank, int world_size,
    int64_t output_peer_data_ptr) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda() && lengths.is_cuda() && output.is_cuda(),
                  "DSV4 peer top-k requires CUDA tensors");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "DSV4 peer top-k logits must be float32");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int,
                  "DSV4 peer top-k lengths must be int32");
  STD_TORCH_CHECK(output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "DSV4 peer top-k output must be int32");
  STD_TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous(),
                  "DSV4 peer top-k logits must be contiguous 2D");
  STD_TORCH_CHECK(lengths.is_contiguous() && lengths.numel() == logits.size(0),
                  "DSV4 peer top-k lengths shape mismatch");
  STD_TORCH_CHECK(output.dim() == 2 && output.size(0) == logits.size(0) &&
                      output.size(1) == k,
                  "DSV4 peer top-k output shape mismatch");
  STD_TORCH_CHECK(logits.size(0) <= 32,
                  "DSV4 peer top-k currently supports at most 32 rows");
  STD_TORCH_CHECK(k == 512 || k == 1024 || k == 2048,
                  "DSV4 peer top-k supports k=512, 1024, or 2048");

  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  namespace P = vllm::persistent;
  auto* peer_data = reinterpret_cast<P::PeerRankData*>(peer_data_ptr);
  P::PeerRankSignals peer_signals{};
  for (int index = 0; index < world_size; ++index) {
    peer_signals.signals[index] =
        reinterpret_cast<P::PeerSignal*>(signal_ptrs[index]);
  }
  auto* self_signal = reinterpret_cast<P::PeerSignal*>(self_signal_ptr);

  if (output_peer_data_ptr != 0 && max_seq_len <= P::HIST2048_THRESHOLD) {
    P::OwnerPeerTopKParams params{};
    params.input = logits.const_data_ptr<float>();
    params.lengths = lengths.const_data_ptr<int32_t>();
    params.output = output.mutable_data_ptr<int32_t>();
    params.num_rows = static_cast<uint32_t>(logits.size(0));
    params.stride = static_cast<uint32_t>(logits.stride(0));
    params.input_peers = peer_data;
    params.output_peers =
        reinterpret_cast<P::PeerRankData*>(output_peer_data_ptr);
    params.peer_signals = peer_signals;
    params.self_signal = self_signal;
    params.rank = rank;
    const cudaStream_t stream = get_current_cuda_stream();

#define LAUNCH_OWNER_TOPK(TOPK, NGPU)                                        \
    do {                                                                      \
      const bool batch_one = params.num_rows == 1;                            \
      auto batch_one_kernel =                                                 \
          P::owner_peer_decode_topk_batch_one_kernel<TOPK, NGPU>;             \
      auto multirow_kernel = P::owner_peer_decode_topk_kernel<TOPK, NGPU>;    \
      const void* kernel = batch_one                                          \
                               ? reinterpret_cast<const void*>(                \
                                     batch_one_kernel)                         \
                               : reinterpret_cast<const void*>(multirow_kernel); \
      cudaError_t attr = cudaFuncSetAttribute(                                \
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, P::kSmemMedium); \
      STD_TORCH_CHECK(attr == cudaSuccess,                                    \
                      "owner peer top-k shared-memory setup failed: ",       \
                      cudaGetErrorString(attr));                              \
      void* args[] = {&params};                                               \
      cudaError_t launch = batch_one                                          \
                               ? cudaLaunchKernel(                             \
                                     kernel, dim3(1),                          \
                                     dim3(P::kThreadsPerBlock), args,          \
                                     P::kSmemMedium, stream)                   \
                               : cudaLaunchCooperativeKernel(                  \
                                     kernel, dim3(params.num_rows),            \
                                     dim3(P::kThreadsPerBlock), args,          \
                                     P::kSmemMedium, stream);                  \
      STD_TORCH_CHECK(launch == cudaSuccess,                                  \
                      "owner peer top-k launch failed: ",                    \
                      cudaGetErrorString(launch));                            \
    } while (0)
#define DISPATCH_OWNER_TOPK(NGPU)         \
    do {                                  \
      if (k == 512) {                     \
        LAUNCH_OWNER_TOPK(512, NGPU);     \
      } else if (k == 1024) {             \
        LAUNCH_OWNER_TOPK(1024, NGPU);    \
      } else {                            \
        LAUNCH_OWNER_TOPK(2048, NGPU);    \
      }                                   \
    } while (0)
    switch (world_size) {
      case 2:
        DISPATCH_OWNER_TOPK(2);
        break;
      case 4:
        DISPATCH_OWNER_TOPK(4);
        break;
      case 8:
        DISPATCH_OWNER_TOPK(8);
        break;
      default:
        STD_TORCH_CHECK(false,
                        "owner peer top-k requires TP world size 2, 4, or 8");
    }
#undef DISPATCH_OWNER_TOPK
#undef LAUNCH_OWNER_TOPK
    return;
  }

#define LAUNCH_PEER_TOPK(TOPK, NGPU)                                   \
  launch_persistent_topk<TOPK, NGPU>(                                  \
      logits, lengths, output, workspace, max_seq_len, peer_data,       \
      peer_signals, self_signal, rank)
#define DISPATCH_TOPK(NGPU)                \
  do {                                     \
    if (k == 512) {                        \
      LAUNCH_PEER_TOPK(512, NGPU);         \
    } else if (k == 1024) {                \
      LAUNCH_PEER_TOPK(1024, NGPU);        \
    } else {                               \
      LAUNCH_PEER_TOPK(2048, NGPU);        \
    }                                      \
  } while (0)

  switch (world_size) {
    case 2:
      DISPATCH_TOPK(2);
      break;
    case 4:
      DISPATCH_TOPK(4);
      break;
    case 8:
      DISPATCH_TOPK(8);
      break;
    default:
      STD_TORCH_CHECK(false,
                      "DSV4 peer top-k requires TP world size 2, 4, or 8");
  }
#undef DISPATCH_TOPK
#undef LAUNCH_PEER_TOPK
#else
  STD_TORCH_CHECK(false, "DSV4 peer top-k is CUDA-only");
#endif
}

void launch_dsv4_indexer_token_merge_impl(
    torch::stable::Tensor& logits, const torch::stable::Tensor& lengths,
    const torch::stable::Tensor& local_indices,
    torch::stable::Tensor& output, int64_t k, int64_t logits_peer_data_ptr,
    int64_t indices_peer_data_ptr, int64_t lengths_peer_data_ptr,
    int64_t logits_byte_offset, int64_t indices_byte_offset,
    int64_t lengths_byte_offset,
    const std::array<int64_t, 8>& signal_ptrs, int64_t self_signal_ptr,
    int rank, int world_size) {
#ifndef USE_ROCM
  STD_TORCH_CHECK(logits.is_cuda() && lengths.is_cuda() &&
                      local_indices.is_cuda() && output.is_cuda(),
                  "DSV4 token-shard merge requires CUDA tensors");
  STD_TORCH_CHECK(logits.scalar_type() == torch::headeronly::ScalarType::Float,
                  "DSV4 token-shard logits must be float32");
  STD_TORCH_CHECK(lengths.scalar_type() == torch::headeronly::ScalarType::Int &&
                      local_indices.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      output.scalar_type() == torch::headeronly::ScalarType::Int,
                  "DSV4 token-shard lengths and indices must be int32");
  STD_TORCH_CHECK(logits.dim() == 2 && logits.is_contiguous(),
                  "DSV4 token-shard logits must be contiguous 2D");
  STD_TORCH_CHECK(lengths.is_contiguous() &&
                      lengths.numel() == logits.size(0),
                  "DSV4 token-shard lengths shape mismatch");
  STD_TORCH_CHECK(local_indices.is_contiguous() &&
                      local_indices.dim() == 2 &&
                      local_indices.size(0) == logits.size(0) &&
                      local_indices.size(1) == k,
                  "DSV4 token-shard local indices shape mismatch");
  STD_TORCH_CHECK(output.is_contiguous() && output.sizes() == local_indices.sizes(),
                  "DSV4 token-shard output shape mismatch");
  STD_TORCH_CHECK(logits.size(0) <= 32,
                  "DSV4 token-shard merge supports at most 32 rows");
  STD_TORCH_CHECK(k == 512,
                  "DSV4 token-shard merge currently supports top-k=512");

  namespace P = vllm::persistent;
  P::TokenShardMergeParams params{};
  params.logits = reinterpret_cast<P::PeerRankData*>(logits_peer_data_ptr);
  params.indices = reinterpret_cast<P::PeerRankData*>(indices_peer_data_ptr);
  params.lengths = reinterpret_cast<P::PeerRankData*>(lengths_peer_data_ptr);
  for (int index = 0; index < world_size; ++index) {
    params.peer_signals.signals[index] =
        reinterpret_cast<P::PeerSignal*>(signal_ptrs[index]);
  }
  params.self_signal = reinterpret_cast<P::PeerSignal*>(self_signal_ptr);
  params.output = output.mutable_data_ptr<int32_t>();
  params.rows = static_cast<uint32_t>(logits.size(0));
  params.logit_stride = static_cast<uint32_t>(logits.stride(0));
  params.top_k = static_cast<uint32_t>(k);
  params.logits_byte_offset = static_cast<uint32_t>(logits_byte_offset);
  params.indices_byte_offset = static_cast<uint32_t>(indices_byte_offset);
  params.lengths_byte_offset = static_cast<uint32_t>(lengths_byte_offset);
  params.rank = rank;

  const cudaStream_t stream = get_current_cuda_stream();
#define LAUNCH_TOKEN_MERGE(NGPU)                                             \
  do {                                                                       \
    auto kernel = &P::token_shard_candidate_merge_kernel<NGPU, 512>;         \
    void* args[] = {&params};                                                \
    const cudaError_t err = cudaLaunchCooperativeKernel(                     \
        reinterpret_cast<const void*>(kernel), dim3(params.rows), dim3(256), \
        args, 0, stream);                                                     \
    STD_TORCH_CHECK(err == cudaSuccess,                                      \
                    "DSV4 token-shard merge launch failed: ",               \
                    cudaGetErrorString(err));                                \
  } while (0)
  if (world_size == 2) {
    LAUNCH_TOKEN_MERGE(2);
  } else if (world_size == 4) {
    LAUNCH_TOKEN_MERGE(4);
  } else {
    STD_TORCH_CHECK(false,
                    "DSV4 token-shard merge requires TP world size 2 or 4");
  }
#undef LAUNCH_TOKEN_MERGE
#else
  STD_TORCH_CHECK(false, "DSV4 token-shard merge is CUDA-only");
#endif
}
