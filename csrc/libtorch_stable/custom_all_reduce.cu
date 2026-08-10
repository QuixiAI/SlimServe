#include "torch_utils.h"

#include <torch/csrc/stable/macros.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/ops.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/csrc/stable/device.h>

#include "custom_all_reduce.cuh"
#include "cuda_compat.h"
#include "libtorch_stable/quantization/gguf/vecdotq.cuh"
#include "quixicore/quant/dsv4_moe_ampere.cuh"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

void launch_dsv4_indexer_peer_topk_impl(
    torch::stable::Tensor& logits, const torch::stable::Tensor& lengths,
    torch::stable::Tensor& output, torch::stable::Tensor& workspace, int64_t k,
    int64_t max_seq_len, int64_t peer_data_ptr,
    const std::array<int64_t, 8>& signal_ptrs, int64_t self_signal_ptr,
    int rank, int world_size, int64_t output_peer_data_ptr);

void launch_dsv4_indexer_token_merge_impl(
    torch::stable::Tensor& logits, const torch::stable::Tensor& lengths,
    const torch::stable::Tensor& local_indices,
    torch::stable::Tensor& output, int64_t k, int64_t logits_peer_data_ptr,
    int64_t indices_peer_data_ptr, int64_t lengths_peer_data_ptr,
    int64_t logits_byte_offset, int64_t indices_byte_offset,
    int64_t lengths_byte_offset,
    const std::array<int64_t, 8>& signal_ptrs, int64_t self_signal_ptr,
    int rank, int world_size);

fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs,
                      torch::stable::Tensor& rank_data, int64_t rank,
                      bool fully_connected) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8)
    throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0)
    throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size)
    throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(
      ipc_ptrs, rank_data.mutable_data_ptr(), rank_data.numel(), rank,
      world_size, fully_connected);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(torch::stable::Tensor& t) {
  if (t.is_contiguous()) {
    return true;
  }
  int64_t storage_nbytes = 0;
  TORCH_ERROR_CODE_CHECK(aoti_torch_get_storage_size(t.get(), &storage_nbytes));
  return storage_nbytes - t.storage_offset() * t.element_size() ==
         static_cast<int64_t>(t.numel() * t.element_size());
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, torch::stable::Tensor& inp,
                torch::stable::Tensor& out, fptr_t _reg_buffer,
                int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.numel()) == (out.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(out));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    STD_TORCH_CHECK((input_size) <= (reg_buffer_sz_bytes));
    STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.mutable_data_ptr();
  }
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Float: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.mutable_data_ptr()),
                           out.numel());
      break;
    }
    case torch::headeronly::ScalarType::Half: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.mutable_data_ptr()),
                          out.numel());
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->allreduce<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()), out.numel());
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "custom allreduce only supports float32, float16 and bfloat16");
  }
}

/**
 * Fused one-shot allreduce + residual add + RMSNorm:
 *   residual += allreduce(inp); out = rmsnorm(residual) * weight
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce_add_rms_norm(fptr_t _fa, torch::stable::Tensor& inp,
                             torch::stable::Tensor& residual,
                             torch::stable::Tensor& weight,
                             torch::stable::Tensor& out, double epsilon,
                             fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK((inp.scalar_type()) == (out.scalar_type()));
  STD_TORCH_CHECK((inp.scalar_type()) == (residual.scalar_type()));
  STD_TORCH_CHECK((inp.scalar_type()) == (weight.scalar_type()));
  STD_TORCH_CHECK((inp.numel()) == (out.numel()));
  STD_TORCH_CHECK((inp.numel()) == (residual.numel()));
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(residual.is_contiguous());
  STD_TORCH_CHECK(weight.is_contiguous());

  int64_t hidden_size = inp.size(inp.dim() - 1);
  STD_TORCH_CHECK((weight.numel()) == (hidden_size));
  int64_t num_tokens = inp.numel() / hidden_size;

  auto input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    STD_TORCH_CHECK((input_size) <= (reg_buffer_sz_bytes));
    STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.mutable_data_ptr();
  }
  switch (out.scalar_type()) {
    case torch::headeronly::ScalarType::Half: {
      fa->allreduce_norm<half>(
          stream, reinterpret_cast<half*>(reg_buffer),
          reinterpret_cast<half*>(residual.mutable_data_ptr()),
          reinterpret_cast<const half*>(weight.const_data_ptr()),
          reinterpret_cast<half*>(out.mutable_data_ptr()),
          static_cast<int>(num_tokens), static_cast<int>(hidden_size),
          static_cast<float>(epsilon));
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case torch::headeronly::ScalarType::BFloat16: {
      fa->allreduce_norm<nv_bfloat16>(
          stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<nv_bfloat16*>(residual.mutable_data_ptr()),
          reinterpret_cast<const nv_bfloat16*>(weight.const_data_ptr()),
          reinterpret_cast<nv_bfloat16*>(out.mutable_data_ptr()),
          static_cast<int>(num_tokens), static_cast<int>(hidden_size),
          static_cast<float>(epsilon));
      break;
    }
#endif
    default:
      throw std::runtime_error(
          "fused allreduce rms norm only supports float16 and bfloat16");
  }
}

__global__ void dsv4_add_local_partials(
    nv_bfloat16* output, const nv_bfloat16* lhs,
    const nv_bfloat16* rhs, int elements) {
  const int index = int(blockIdx.x) * int(blockDim.x) + int(threadIdx.x);
  if (index < elements) output[index] = lhs[index] + rhs[index];
}

void all_reduce_dsv4_mhc(
    fptr_t _fa, torch::stable::Tensor& inp,
    const std::optional<torch::stable::Tensor>& addend,
    const torch::stable::Tensor& residual,
    const torch::stable::Tensor& post_mix,
    const torch::stable::Tensor& comb_mix, const torch::stable::Tensor& fn,
    torch::stable::Tensor& residual_out, torch::stable::Tensor& partial,
    const torch::stable::Tensor& scale, const torch::stable::Tensor& base,
    torch::stable::Tensor& next_post, torch::stable::Tensor& next_comb,
    torch::stable::Tensor& layer_input,
    const std::optional<torch::stable::Tensor>& norm_weight,
    const std::optional<torch::stable::Tensor>& quant_input,
    double rms_eps, double pre_eps, double sinkhorn_eps,
    double post_multiplier, int64_t sinkhorn_repeat, double norm_eps,
    bool input_prepared, bool own_projections, bool publish_prepared,
    fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK(inp.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  if (addend) {
    STD_TORCH_CHECK(addend->scalar_type() == inp.scalar_type());
    STD_TORCH_CHECK(addend->sizes() == inp.sizes());
    STD_TORCH_CHECK(_is_weak_contiguous(const_cast<torch::stable::Tensor&>(*addend)));
  }
  STD_TORCH_CHECK(residual.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(residual_out.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(layer_input.scalar_type() ==
                  torch::headeronly::ScalarType::BFloat16);
  STD_TORCH_CHECK(post_mix.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(comb_mix.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(
      fn.scalar_type() == torch::headeronly::ScalarType::Float ||
          fn.scalar_type() == torch::headeronly::ScalarType::Half,
      "DSV4 mHC fn weights must be float16 or float32");
  STD_TORCH_CHECK(partial.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(scale.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(base.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(next_post.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(next_comb.scalar_type() ==
                  torch::headeronly::ScalarType::Float);
  const int64_t local_hidden = 4096 / fa->world_size_;
  const bool channel_owned_x =
      inp.dim() == 2 && inp.size(0) == 1 && inp.size(1) == local_hidden;
  STD_TORCH_CHECK(
      inp.dim() == 2 && inp.size(0) == 1 &&
          (inp.size(1) == 4096 || channel_owned_x),
      "DSV4 mHC input must be BF16[1,4096] TP partial or channel-owned "
      "BF16[1,4096/TP]");
  STD_TORCH_CHECK(!channel_owned_x || !addend.has_value(),
                  "DSV4 channel-owned mHC input cannot have an addend");
  STD_TORCH_CHECK(residual.dim() == 3 && residual.size(0) == 1 &&
                  residual.size(1) == 4 && residual.size(2) == 4096);
  const bool local_input_owned =
      layer_input.dim() == 2 && layer_input.size(0) == 1 &&
      layer_input.size(1) == local_hidden;
  STD_TORCH_CHECK(
      layer_input.dim() == 2 && layer_input.size(0) == 1 &&
          (layer_input.size(1) == 4096 || local_input_owned),
      "DSV4 mHC layer input must be BF16[1,4096] or rank-owned "
      "BF16[1,4096/TP]");
  STD_TORCH_CHECK(fn.numel() == 24 * 4 * 4096);
  STD_TORCH_CHECK(partial.numel() == 32 * 25);
  STD_TORCH_CHECK(_is_weak_contiguous(inp));
  STD_TORCH_CHECK(residual.is_contiguous());
  STD_TORCH_CHECK(post_mix.is_contiguous());
  STD_TORCH_CHECK(comb_mix.is_contiguous());
  STD_TORCH_CHECK(fn.is_contiguous());
  STD_TORCH_CHECK(residual_out.is_contiguous());
  STD_TORCH_CHECK(partial.is_contiguous());
  STD_TORCH_CHECK(scale.is_contiguous());
  STD_TORCH_CHECK(base.is_contiguous());
  STD_TORCH_CHECK(next_post.is_contiguous());
  STD_TORCH_CHECK(next_comb.is_contiguous());
  STD_TORCH_CHECK(layer_input.is_contiguous());
  STD_TORCH_CHECK(norm_weight.has_value() == quant_input.has_value(),
                  "DSV4 fused norm requires a Q8_1 output");
  if (norm_weight) {
    STD_TORCH_CHECK(norm_weight->scalar_type() == inp.scalar_type() &&
                        norm_weight->dim() == 1 &&
                        norm_weight->numel() == 4096 &&
                        norm_weight->is_contiguous(),
                    "DSV4 fused norm weight must be contiguous BF16[4096]");
    const int64_t quant_hidden = local_input_owned ? local_hidden : 4096;
    STD_TORCH_CHECK(
        quant_input->scalar_type() == torch::headeronly::ScalarType::Int &&
            quant_input->dim() == 2 && quant_input->size(0) == 1 &&
            quant_input->size(1) == quant_hidden / 32 * 9 &&
            quant_input->is_contiguous(),
        "DSV4 fused norm Q8_1 output has the wrong ownership width");
  }
  STD_TORCH_CHECK(!local_input_owned ||
                      (norm_weight.has_value() && own_projections),
                  "DSV4 local input ownership requires fused norm and owned "
                  "mHC projections");
  STD_TORCH_CHECK(!channel_owned_x || local_input_owned,
                  "DSV4 channel-owned producer requires channel-owned output");

  const int64_t input_size = inp.numel() * inp.element_size();
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  const bool inputs_are_registered = reg_buffer == nullptr;
  if (reg_buffer) {
    STD_TORCH_CHECK(input_size <= reg_buffer_sz_bytes);
    if (addend) {
      const int elements = int(inp.numel());
      dsv4_add_local_partials<<<(elements + 255) / 256, 256, 0, stream>>>(
          reinterpret_cast<nv_bfloat16*>(reg_buffer),
          reinterpret_cast<const nv_bfloat16*>(inp.const_data_ptr()),
          reinterpret_cast<const nv_bfloat16*>(addend->const_data_ptr()),
          elements);
    } else {
      STD_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.const_data_ptr(), input_size,
                                     cudaMemcpyDeviceToDevice, stream));
    }
  } else {
    reg_buffer = inp.mutable_data_ptr();
  }

  auto* input_ptr = reinterpret_cast<nv_bfloat16*>(reg_buffer);
  const auto* residual_ptr =
      reinterpret_cast<const nv_bfloat16*>(residual.const_data_ptr());
  const auto* post_ptr = reinterpret_cast<const float*>(post_mix.const_data_ptr());
  const auto* comb_ptr = reinterpret_cast<const float*>(comb_mix.const_data_ptr());
  auto* residual_out_ptr =
      reinterpret_cast<nv_bfloat16*>(residual_out.mutable_data_ptr());
  auto* partial_ptr = reinterpret_cast<float*>(partial.mutable_data_ptr());
  const auto* scale_ptr = reinterpret_cast<const float*>(scale.const_data_ptr());
  const auto* base_ptr = reinterpret_cast<const float*>(base.const_data_ptr());
  auto* next_post_ptr = reinterpret_cast<float*>(next_post.mutable_data_ptr());
  auto* next_comb_ptr = reinterpret_cast<float*>(next_comb.mutable_data_ptr());
  auto* layer_input_ptr =
      reinterpret_cast<nv_bfloat16*>(layer_input.mutable_data_ptr());
  const auto* norm_weight_ptr = norm_weight
      ? reinterpret_cast<const nv_bfloat16*>(norm_weight->const_data_ptr())
      : nullptr;
  auto* quant_input_ptr = quant_input
      ? reinterpret_cast<tms::dsv4_mhc::block_q8_1*>(
            const_cast<void*>(quant_input->const_data_ptr()))
      : nullptr;

  auto launch = [&](auto* fn_ptr) {
    if (addend && inputs_are_registered) {
      auto* addend_ptr = reinterpret_cast<nv_bfloat16*>(
          const_cast<void*>(addend->const_data_ptr()));
      fa->allreduce_dsv4_mhc_add(
          stream, input_ptr, addend_ptr, residual_ptr, post_ptr, comb_ptr,
          fn_ptr, residual_out_ptr, partial_ptr, scale_ptr, base_ptr,
          next_post_ptr, next_comb_ptr, layer_input_ptr,
          static_cast<float>(rms_eps), static_cast<float>(pre_eps),
          static_cast<float>(sinkhorn_eps), static_cast<float>(post_multiplier),
          static_cast<int>(sinkhorn_repeat), norm_weight_ptr, quant_input_ptr,
          static_cast<float>(norm_eps), input_prepared, own_projections,
          publish_prepared, local_input_owned);
    } else {
      fa->allreduce_dsv4_mhc(
          stream, input_ptr, residual_ptr, post_ptr, comb_ptr, fn_ptr,
          residual_out_ptr, partial_ptr, scale_ptr, base_ptr, next_post_ptr,
          next_comb_ptr, layer_input_ptr, static_cast<float>(rms_eps),
          static_cast<float>(pre_eps), static_cast<float>(sinkhorn_eps),
          static_cast<float>(post_multiplier),
          static_cast<int>(sinkhorn_repeat), norm_weight_ptr, quant_input_ptr,
          static_cast<float>(norm_eps), input_prepared, own_projections,
          publish_prepared, local_input_owned, channel_owned_x);
    }
  };
  if (fn.scalar_type() == torch::headeronly::ScalarType::Half) {
    launch(reinterpret_cast<const half*>(fn.const_data_ptr()));
  } else {
    launch(reinterpret_cast<const float*>(fn.const_data_ptr()));
  }
}

void wait_dsv4_mhc(fptr_t _fa, const torch::stable::Tensor& anchor) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  const torch::stable::accelerator::DeviceGuard device_guard(
      anchor.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(anchor.get_device_index());
  fa->wait_dsv4_mhc(stream);
}

void dsv4_channel_owned_mhc(
    fptr_t _fa, const torch::stable::Tensor& inp,
    const std::optional<torch::stable::Tensor>& addend,
    const torch::stable::Tensor& residual,
    const torch::stable::Tensor& post_mix,
    const torch::stable::Tensor& comb_mix, const torch::stable::Tensor& fn,
    torch::stable::Tensor& residual_out, torch::stable::Tensor& partial,
    const torch::stable::Tensor& scale, const torch::stable::Tensor& base,
    torch::stable::Tensor& next_post, torch::stable::Tensor& next_comb,
    torch::stable::Tensor& layer_input,
    const torch::stable::Tensor& norm_weight,
    torch::stable::Tensor& quant_input, double rms_eps, double pre_eps,
    double sinkhorn_eps, double post_multiplier, int64_t sinkhorn_repeat,
    double norm_eps, fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr, "DSV4 channel-owned communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4 ||
                      fa->world_size_ == 8,
                  "DSV4 channel ownership requires TP2, TP4, or TP8");
  const int64_t local_hidden = 4096 / fa->world_size_;
  const bool reduce_input = inp.dim() == 2 && inp.size(0) == 1 &&
                            inp.size(1) == 4096;
  const torch::stable::accelerator::DeviceGuard device_guard(
      inp.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(inp.get_device_index());

  STD_TORCH_CHECK(
      inp.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          inp.dim() == 2 && inp.size(0) == 1 &&
          (inp.size(1) == local_hidden || inp.size(1) == 4096) &&
          inp.is_contiguous(),
      "DSV4 channel-owned input must be contiguous BF16[1,4096/TP] or "
      "a graph-registered BF16[1,4096] TP partial");
  STD_TORCH_CHECK(!addend || reduce_input,
                  "DSV4 channel-owned addend requires a full-H TP partial");
  if (addend) {
    STD_TORCH_CHECK(
        addend->scalar_type() == inp.scalar_type() &&
            addend->sizes() == inp.sizes() && addend->is_contiguous(),
        "DSV4 channel-owned addend must match the full-H input");
  }
  STD_TORCH_CHECK(
      residual.scalar_type() == inp.scalar_type() && residual.dim() == 3 &&
          residual.size(0) == 1 && residual.size(1) == 4 &&
          residual.size(2) == local_hidden && residual.is_contiguous(),
      "DSV4 channel-owned residual must be contiguous BF16[1,4,4096/TP]");
  STD_TORCH_CHECK(residual_out.scalar_type() == inp.scalar_type() &&
                      residual_out.sizes() == residual.sizes() &&
                      residual_out.is_contiguous(),
                  "DSV4 channel-owned residual output shape mismatch");
  STD_TORCH_CHECK(layer_input.scalar_type() == inp.scalar_type() &&
                      layer_input.dim() == 2 && layer_input.size(0) == 1 &&
                      layer_input.size(1) == local_hidden &&
                      layer_input.is_contiguous(),
                  "DSV4 channel-owned layer input shape mismatch");
  STD_TORCH_CHECK(post_mix.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      post_mix.numel() == 4 && post_mix.is_contiguous());
  STD_TORCH_CHECK(comb_mix.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      comb_mix.numel() == 16 && comb_mix.is_contiguous());
  STD_TORCH_CHECK(
      (fn.scalar_type() == torch::headeronly::ScalarType::Half ||
       fn.scalar_type() == torch::headeronly::ScalarType::Float) &&
          fn.numel() == 24 * 4 * 4096 && fn.is_contiguous(),
      "DSV4 channel-owned fn must be contiguous FP16/FP32[24,16384]");
  STD_TORCH_CHECK(partial.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      partial.numel() >=
                          vllm::dsv4_mhc_channel_owned::kPartialValues &&
                      partial.is_contiguous(),
                  "DSV4 channel-owned partial workspace is too small");
  STD_TORCH_CHECK(scale.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      scale.numel() == 3 && scale.is_contiguous());
  STD_TORCH_CHECK(base.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      base.numel() == 24 && base.is_contiguous());
  STD_TORCH_CHECK(next_post.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      next_post.numel() == 4 && next_post.is_contiguous());
  STD_TORCH_CHECK(next_comb.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      next_comb.numel() == 16 && next_comb.is_contiguous());
  STD_TORCH_CHECK(norm_weight.scalar_type() == inp.scalar_type() &&
                      norm_weight.numel() == 4096 &&
                      norm_weight.is_contiguous(),
                  "DSV4 channel-owned norm weight must be BF16[4096]");
  STD_TORCH_CHECK(
      quant_input.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_input.numel() == local_hidden / 32 * 9 &&
          quant_input.is_contiguous(),
      "DSV4 channel-owned Q8_1 output must have 9 int32 per local block");

  void* shared_partial = reinterpret_cast<void*>(reg_buffer_ptr);
  if (shared_partial != nullptr) {
    STD_TORCH_CHECK(
        vllm::dsv4_mhc_channel_owned::kPartialValues * sizeof(float) <=
        reg_buffer_sz_bytes);
  } else {
    shared_partial = partial.mutable_data_ptr();
  }
  auto* partial_peers = fa->resolve_rank_data(stream, shared_partial);
  vllm::RankData* input_peers = nullptr;
  vllm::RankData* addend_peers = nullptr;
  if (reduce_input) {
    STD_TORCH_CHECK(
        reg_buffer_ptr == 0,
        "DSV4 full-H ownership transition requires CUDA graph capture");
    input_peers = fa->resolve_rank_data(
        stream, const_cast<void*>(inp.const_data_ptr()));
    if (addend) {
      addend_peers = fa->resolve_rank_data(
          stream, const_cast<void*>(addend->const_data_ptr()));
    }
  }
  auto* partial_ptr = reinterpret_cast<float*>(shared_partial);
  auto* quant_ptr = reinterpret_cast<tms::dsv4_mhc::block_q8_1*>(
      quant_input.mutable_data_ptr());
  STD_TORCH_CHECK(
      fa->dsv4_mhc_schedule_ ==
          vllm::CustomAllreduce::Dsv4MhcSchedule::kAsyncSplit,
      "DSV4 channel ownership requires VLLM_DSV4_MHC_SCHEDULE=async");
  const nv_bfloat16* owned_input =
      reinterpret_cast<const nv_bfloat16*>(inp.const_data_ptr());

#define LAUNCH_OWNERSHIP_REDUCE(NGPU)                                        \
  vllm::dsv4_tp_reduce_scatter::launch<NGPU>(                                \
      reinterpret_cast<const nv_bfloat16*>(inp.const_data_ptr()),            \
      addend ? reinterpret_cast<const nv_bfloat16*>(addend->const_data_ptr()) \
             : nullptr,                                                       \
      input_peers, addend_peers, fa->sg_, fa->self_sg_,                      \
      reinterpret_cast<nv_bfloat16*>(layer_input.mutable_data_ptr()), 1,      \
      fa->rank_, true, stream)
  if (reduce_input) {
    switch (fa->world_size_) {
      case 2:
        LAUNCH_OWNERSHIP_REDUCE(2);
        break;
      case 4:
        LAUNCH_OWNERSHIP_REDUCE(4);
        break;
      case 8:
        LAUNCH_OWNERSHIP_REDUCE(8);
        break;
    }
    owned_input = reinterpret_cast<const nv_bfloat16*>(
        layer_input.const_data_ptr());
  }
#undef LAUNCH_OWNERSHIP_REDUCE
  fa->wait_dsv4_mhc(stream);

#if 0  // Rejected diagnostic: fixed mHC work did not scale with TP.
  const char* channel_mhc_schedule =
      std::getenv("VLLM_DSV4_CHANNEL_MHC_SCHEDULE");
  const bool use_monolithic =
      channel_mhc_schedule != nullptr &&
      std::strcmp(channel_mhc_schedule, "monolithic") == 0;
  if (use_monolithic) {
#define LAUNCH_CHANNEL_OWNED_MONOLITHIC(FN_T, NGPU)                          \
    do {                                                                       \
      auto kernel = vllm::dsv4_mhc_channel_owned::                            \
          post_pre_monolithic<FN_T, NGPU>;                                    \
      const nv_bfloat16* kernel_input = owned_input;                           \
      const nv_bfloat16* kernel_residual =                                    \
          reinterpret_cast<const nv_bfloat16*>(residual.const_data_ptr());    \
      const float* kernel_post =                                              \
          reinterpret_cast<const float*>(post_mix.const_data_ptr());          \
      const float* kernel_comb =                                              \
          reinterpret_cast<const float*>(comb_mix.const_data_ptr());          \
      const FN_T* kernel_fn =                                                 \
          reinterpret_cast<const FN_T*>(fn.const_data_ptr());                 \
      nv_bfloat16* kernel_residual_out = reinterpret_cast<nv_bfloat16*>(      \
          residual_out.mutable_data_ptr());                                   \
      float* kernel_debug =                                                   \
          reinterpret_cast<float*>(partial.mutable_data_ptr());               \
      const float* kernel_scale =                                             \
          reinterpret_cast<const float*>(scale.const_data_ptr());             \
      const float* kernel_base =                                              \
          reinterpret_cast<const float*>(base.const_data_ptr());              \
      float* kernel_next_post =                                               \
          reinterpret_cast<float*>(next_post.mutable_data_ptr());             \
      float* kernel_next_comb =                                               \
          reinterpret_cast<float*>(next_comb.mutable_data_ptr());             \
      nv_bfloat16* kernel_layer_input = reinterpret_cast<nv_bfloat16*>(       \
          layer_input.mutable_data_ptr());                                    \
      const nv_bfloat16* kernel_norm_weight =                                 \
          reinterpret_cast<const nv_bfloat16*>(norm_weight.const_data_ptr()); \
      float kernel_rms_eps = static_cast<float>(rms_eps);                     \
      float kernel_pre_eps = static_cast<float>(pre_eps);                     \
      float kernel_sinkhorn_eps = static_cast<float>(sinkhorn_eps);           \
      float kernel_post_multiplier = static_cast<float>(post_multiplier);     \
      int kernel_sinkhorn_repeat = static_cast<int>(sinkhorn_repeat);         \
      float kernel_norm_eps = static_cast<float>(norm_eps);                   \
      void* kernel_args[] = {                                                 \
          &partial_peers, &fa->sg_, &fa->self_sg_, &kernel_input,             \
          &kernel_residual, &kernel_post, &kernel_comb, &kernel_fn,            \
          &kernel_residual_out, &partial_ptr, &kernel_debug, &kernel_scale,    \
          &kernel_base, &kernel_next_post, &kernel_next_comb,                  \
          &kernel_layer_input, &kernel_norm_weight, &quant_ptr,                \
          &kernel_rms_eps, &kernel_pre_eps, &kernel_sinkhorn_eps,              \
          &kernel_post_multiplier, &kernel_sinkhorn_repeat, &kernel_norm_eps, \
          &fa->rank_};                                                        \
      CUDACHECK(cudaLaunchCooperativeKernel(                                  \
          reinterpret_cast<const void*>(kernel),                              \
          dim3(vllm::dsv4_mhc_channel_owned::kUrgentSplits),                  \
          dim3(vllm::dsv4_mhc_channel_owned::kThreads), kernel_args, 0,       \
          stream));                                                           \
    } while (0)

#define DISPATCH_CHANNEL_OWNED_MONOLITHIC(NGPU)                              \
    do {                                                                       \
      if (fn.scalar_type() == torch::headeronly::ScalarType::Half) {          \
        LAUNCH_CHANNEL_OWNED_MONOLITHIC(half, NGPU);                           \
      } else {                                                                 \
        LAUNCH_CHANNEL_OWNED_MONOLITHIC(float, NGPU);                          \
      }                                                                        \
    } while (0)
    switch (fa->world_size_) {
      case 2:
        DISPATCH_CHANNEL_OWNED_MONOLITHIC(2);
        break;
      case 4:
        DISPATCH_CHANNEL_OWNED_MONOLITHIC(4);
        break;
      case 8:
        DISPATCH_CHANNEL_OWNED_MONOLITHIC(8);
        break;
    }
#undef DISPATCH_CHANNEL_OWNED_MONOLITHIC
#undef LAUNCH_CHANNEL_OWNED_MONOLITHIC
    STD_CUDA_CHECK(cudaGetLastError());
    return;
  }

#endif

#define LAUNCH_CHANNEL_OWNED(FN_T, NGPU)                                      \
  do {                                                                         \
  auto urgent_kernel =                                                        \
      vllm::dsv4_mhc_channel_owned::post_pre_urgent<FN_T, NGPU>;               \
  const nv_bfloat16* urgent_input = owned_input;                               \
  const nv_bfloat16* urgent_residual = reinterpret_cast<const nv_bfloat16*>(   \
      residual.const_data_ptr());                                             \
  const float* urgent_post =                                                   \
      reinterpret_cast<const float*>(post_mix.const_data_ptr());               \
  const float* urgent_comb =                                                   \
      reinterpret_cast<const float*>(comb_mix.const_data_ptr());               \
  const FN_T* urgent_fn = reinterpret_cast<const FN_T*>(fn.const_data_ptr());  \
  nv_bfloat16* urgent_residual_out = reinterpret_cast<nv_bfloat16*>(           \
      residual_out.mutable_data_ptr());                                        \
  float* urgent_debug = reinterpret_cast<float*>(partial.mutable_data_ptr());  \
  const float* urgent_scale =                                                  \
      reinterpret_cast<const float*>(scale.const_data_ptr());                  \
  const float* urgent_base =                                                   \
      reinterpret_cast<const float*>(base.const_data_ptr());                   \
  nv_bfloat16* urgent_layer_input = reinterpret_cast<nv_bfloat16*>(            \
      layer_input.mutable_data_ptr());                                         \
  const nv_bfloat16* urgent_norm_weight =                                      \
      reinterpret_cast<const nv_bfloat16*>(norm_weight.const_data_ptr());      \
  float urgent_rms_eps = static_cast<float>(rms_eps);                          \
  float urgent_pre_eps = static_cast<float>(pre_eps);                          \
  float urgent_norm_eps = static_cast<float>(norm_eps);                        \
  void* urgent_args[] = {                                                      \
      &partial_peers, &fa->sg_, &fa->self_sg_, &urgent_input,                  \
      &urgent_residual, &urgent_post, &urgent_comb, &urgent_fn,                 \
      &urgent_residual_out, &partial_ptr, &urgent_debug, &urgent_scale,         \
      &urgent_base, &urgent_layer_input, &urgent_norm_weight, &quant_ptr,       \
      &urgent_rms_eps, &urgent_pre_eps, &urgent_norm_eps, &fa->rank_};          \
  CUDACHECK(cudaLaunchCooperativeKernel(                                       \
      reinterpret_cast<const void*>(urgent_kernel),                            \
      dim3(vllm::dsv4_mhc_channel_owned::kUrgentSplits),                       \
      dim3(vllm::dsv4_mhc_channel_owned::kThreads), urgent_args, 0, stream));   \
  CUDACHECK(cudaEventRecord(fa->dsv4_mhc_urgent_done_, stream));               \
  CUDACHECK(cudaStreamWaitEvent(fa->dsv4_mhc_deferred_stream_,                \
                                fa->dsv4_mhc_urgent_done_, 0));                \
  auto deferred_kernel =                                                       \
      vllm::dsv4_mhc_channel_owned::post_pre_deferred<FN_T, NGPU>;            \
  const nv_bfloat16* deferred_residual = reinterpret_cast<const nv_bfloat16*>( \
      residual_out.const_data_ptr());                                          \
  const FN_T* deferred_fn =                                                    \
      reinterpret_cast<const FN_T*>(fn.const_data_ptr());                     \
  const float* deferred_scale =                                                \
      reinterpret_cast<const float*>(scale.const_data_ptr());                 \
  const float* deferred_base =                                                 \
      reinterpret_cast<const float*>(base.const_data_ptr());                  \
  float* deferred_debug =                                                     \
      reinterpret_cast<float*>(partial.mutable_data_ptr());                   \
  float* deferred_post = reinterpret_cast<float*>(next_post.mutable_data_ptr()); \
  float* deferred_comb = reinterpret_cast<float*>(next_comb.mutable_data_ptr()); \
  float deferred_sinkhorn_eps = static_cast<float>(sinkhorn_eps);              \
  float deferred_post_multiplier = static_cast<float>(post_multiplier);       \
  int deferred_sinkhorn_repeat = static_cast<int>(sinkhorn_repeat);            \
  void* deferred_args[] = {                                                    \
      &partial_peers, &fa->sg_, &fa->self_sg_, &deferred_residual,            \
      &deferred_fn, &partial_ptr, &deferred_debug, &deferred_scale,            \
      &deferred_base,                                                          \
      &deferred_post, &deferred_comb, &deferred_sinkhorn_eps,                  \
      &deferred_post_multiplier, &deferred_sinkhorn_repeat, &fa->rank_};       \
  CUDACHECK(cudaLaunchCooperativeKernel(                                       \
      reinterpret_cast<const void*>(deferred_kernel),                         \
      dim3(vllm::dsv4_mhc_channel_owned::kDeferredSplits *                    \
           vllm::dsv4_mhc_channel_owned::kDeferredPartitions),                \
      dim3(vllm::dsv4_mhc_channel_owned::kThreads), deferred_args, 0,         \
      fa->dsv4_mhc_deferred_stream_));                                         \
  CUDACHECK(cudaEventRecord(fa->dsv4_mhc_deferred_done_,                      \
                            fa->dsv4_mhc_deferred_stream_));                   \
  fa->dsv4_mhc_deferred_pending_ = true;                                       \
  } while (0)

#define DISPATCH_CHANNEL_OWNED(NGPU)                                          \
  do {                                                                         \
    if (fn.scalar_type() == torch::headeronly::ScalarType::Half) {             \
      LAUNCH_CHANNEL_OWNED(half, NGPU);                                        \
    } else {                                                                   \
      LAUNCH_CHANNEL_OWNED(float, NGPU);                                       \
    }                                                                          \
  } while (0)
  switch (fa->world_size_) {
    case 2:
      DISPATCH_CHANNEL_OWNED(2);
      break;
    case 4:
      DISPATCH_CHANNEL_OWNED(4);
      break;
    case 8:
      DISPATCH_CHANNEL_OWNED(8);
      break;
  }
#undef DISPATCH_CHANNEL_OWNED
#undef LAUNCH_CHANNEL_OWNED
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_owned_attention_projections(
    fptr_t _fa, const torch::stable::Tensor& local_input,
    const torch::stable::Tensor& local_quant,
    const torch::stable::Tensor& aligned_q8_weight,
    const torch::stable::Tensor& bf16_weight0,
    const torch::stable::Tensor& bf16_weight1,
    const torch::stable::Tensor& bf16_weight2,
    torch::stable::Tensor& q8_output, torch::stable::Tensor& bf16_output0,
    torch::stable::Tensor& bf16_output1, torch::stable::Tensor& bf16_output2,
    torch::stable::Tensor& partial, torch::stable::Tensor& reduced,
    fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr, "DSV4 input-owned communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4 ||
                      fa->world_size_ == 8,
                  "DSV4 input ownership requires TP2, TP4, or TP8");
  const int64_t local_hidden = 4096 / fa->world_size_;
  const auto bf16 = torch::headeronly::ScalarType::BFloat16;
  const auto fp32 = torch::headeronly::ScalarType::Float;
  STD_TORCH_CHECK(
      local_input.scalar_type() == bf16 && local_input.dim() == 2 &&
          local_input.size(0) == 1 && local_input.size(1) == local_hidden &&
          local_input.is_contiguous(),
      "DSV4 attention input must be contiguous owned BF16[1,4096/TP]");
  STD_TORCH_CHECK(
      local_quant.scalar_type() == torch::headeronly::ScalarType::Int &&
          local_quant.dim() == 2 && local_quant.size(0) == 1 &&
          local_quant.numel() == local_hidden / 32 * 9 &&
          local_quant.is_contiguous(),
      "DSV4 attention quant input must be owned Q8_1 blocks");
  STD_TORCH_CHECK(
      aligned_q8_weight.scalar_type() == torch::headeronly::ScalarType::Byte &&
          aligned_q8_weight.dim() == 2 && aligned_q8_weight.size(0) > 0 &&
          aligned_q8_weight.size(1) == 4096 / 32 * 34 &&
          aligned_q8_weight.is_contiguous(),
      "DSV4 attention Q8 weight must be aligned byte-neutral Q8_0");
  auto check_weight = [&](const torch::stable::Tensor& weight) {
    STD_TORCH_CHECK(weight.scalar_type() == bf16 && weight.dim() == 2 &&
                        weight.size(1) == 4096 && weight.is_contiguous(),
                    "DSV4 attention BF16 weight must be [rows,4096]");
  };
  check_weight(bf16_weight0);
  check_weight(bf16_weight1);
  check_weight(bf16_weight2);
  const int q8_rows = int(aligned_q8_weight.size(0));
  const int rows0 = int(bf16_weight0.size(0));
  const int rows1 = int(bf16_weight1.size(0));
  const int rows2 = int(bf16_weight2.size(0));
  const int total = q8_rows + rows0 + rows1 + rows2;
  STD_TORCH_CHECK(total % vllm::packed_t<float>::P::size == 0,
                  "DSV4 attention projection rows must be pack aligned");
  STD_TORCH_CHECK(q8_output.scalar_type() == bf16 &&
                      q8_output.numel() == q8_rows &&
                      q8_output.is_contiguous());
  STD_TORCH_CHECK(bf16_output0.scalar_type() == fp32 &&
                      bf16_output0.numel() == rows0 &&
                      bf16_output0.is_contiguous());
  STD_TORCH_CHECK(bf16_output1.scalar_type() == bf16 &&
                      bf16_output1.numel() == rows1 &&
                      bf16_output1.is_contiguous());
  STD_TORCH_CHECK(bf16_output2.scalar_type() == fp32 &&
                      bf16_output2.numel() == rows2 &&
                      bf16_output2.is_contiguous());
  STD_TORCH_CHECK(partial.scalar_type() == fp32 && partial.numel() >= total &&
                      partial.is_contiguous());
  STD_TORCH_CHECK(reduced.scalar_type() == fp32 && reduced.numel() >= total &&
                      reduced.is_contiguous());

  const torch::stable::accelerator::DeviceGuard device_guard(
      local_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(local_input.get_device_index());
  float* partial_ptr = reinterpret_cast<float*>(reg_buffer_ptr);
  if (partial_ptr != nullptr) {
    STD_TORCH_CHECK(int64_t(total) * sizeof(float) <= reg_buffer_sz_bytes);
  } else {
    partial_ptr = reinterpret_cast<float*>(partial.mutable_data_ptr());
  }
  auto launch = [&](auto ngpu_tag) {
    constexpr int ngpu = decltype(ngpu_tag)::value;
    vllm::dsv4_tp_input_owned::launch_attention<ngpu>(
        reinterpret_cast<const nv_bfloat16*>(local_input.const_data_ptr()),
        reinterpret_cast<const block_q8_1*>(local_quant.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(aligned_q8_weight.const_data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(bf16_weight0.const_data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(bf16_weight1.const_data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(bf16_weight2.const_data_ptr()),
        partial_ptr, q8_rows, rows0, rows1, rows2, fa->rank_, stream);
  };
  switch (fa->world_size_) {
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
  }
  auto* reduced_ptr = reinterpret_cast<float*>(reduced.mutable_data_ptr());
  fa->allreduce<float>(stream, partial_ptr, reduced_ptr, total);
  vllm::dsv4_tp_input_owned::launch_attention_finalize(
      reduced_ptr,
      reinterpret_cast<nv_bfloat16*>(q8_output.mutable_data_ptr()),
      reinterpret_cast<float*>(bf16_output0.mutable_data_ptr()),
      reinterpret_cast<nv_bfloat16*>(bf16_output1.mutable_data_ptr()),
      reinterpret_cast<float*>(bf16_output2.mutable_data_ptr()), q8_rows,
      rows0, rows1, rows2, stream);
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_gather_owned_q8(
    fptr_t _fa, const torch::stable::Tensor& local_quant,
    torch::stable::Tensor& full_quant, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr && (fa->world_size_ == 2 ||
                                   fa->world_size_ == 4 ||
                                   fa->world_size_ == 8),
                  "DSV4 Q8 gather requires TP2, TP4, or TP8");
  const int local_blocks = 4096 / fa->world_size_ / 32;
  STD_TORCH_CHECK(
      local_quant.scalar_type() == torch::headeronly::ScalarType::Int &&
          local_quant.dim() == 2 && local_quant.size(0) == 1 &&
          local_quant.numel() == local_blocks * 9 &&
          local_quant.is_contiguous(),
      "DSV4 owned Q8 input shape mismatch");
  STD_TORCH_CHECK(
      full_quant.scalar_type() == torch::headeronly::ScalarType::Int &&
          full_quant.dim() == 2 && full_quant.size(0) == 1 &&
          full_quant.numel() == 4096 / 32 * 9 && full_quant.is_contiguous(),
      "DSV4 full Q8 output shape mismatch");
  const torch::stable::accelerator::DeviceGuard device_guard(
      local_quant.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(local_quant.get_device_index());
  void* published = reinterpret_cast<void*>(reg_buffer_ptr);
  if (published != nullptr) {
    const int64_t bytes = local_quant.numel() * local_quant.element_size();
    STD_TORCH_CHECK(bytes <= reg_buffer_sz_bytes);
    STD_CUDA_CHECK(cudaMemcpyAsync(published, local_quant.const_data_ptr(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    published = const_cast<void*>(local_quant.const_data_ptr());
  }
  auto* input_peers = fa->resolve_rank_data(stream, published);
  auto launch = [&](auto ngpu_tag) {
    constexpr int ngpu = decltype(ngpu_tag)::value;
    vllm::dsv4_tp_input_owned::launch_gather_q8<ngpu>(
        input_peers, fa->sg_, fa->self_sg_,
        reinterpret_cast<block_q8_1*>(full_quant.mutable_data_ptr()),
        local_blocks, fa->rank_, stream);
  };
  switch (fa->world_size_) {
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
  }
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_gather_owned_bf16(
    fptr_t _fa, const torch::stable::Tensor& local_input,
    torch::stable::Tensor& full_output, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr && (fa->world_size_ == 2 ||
                                   fa->world_size_ == 4 ||
                                   fa->world_size_ == 8),
                  "DSV4 BF16 gather requires TP2, TP4, or TP8");
  const int64_t local_hidden = 4096 / fa->world_size_;
  STD_TORCH_CHECK(
      local_input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          local_input.dim() >= 2 &&
          local_input.size(local_input.dim() - 1) == local_hidden &&
          local_input.is_contiguous(),
      "DSV4 owned BF16 input shape mismatch");
  const int64_t rows = local_input.numel() / local_hidden;
  STD_TORCH_CHECK(rows >= 1 && rows <= 8,
                  "DSV4 BF16 gather supports 1..8 rows");
  STD_TORCH_CHECK(
      full_output.scalar_type() == local_input.scalar_type() &&
          full_output.numel() == rows * 4096 && full_output.is_contiguous(),
      "DSV4 full BF16 output shape mismatch");
  const torch::stable::accelerator::DeviceGuard device_guard(
      local_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(local_input.get_device_index());
  void* published = reinterpret_cast<void*>(reg_buffer_ptr);
  if (published != nullptr) {
    const int64_t bytes = local_input.numel() * local_input.element_size();
    STD_TORCH_CHECK(bytes <= reg_buffer_sz_bytes);
    STD_CUDA_CHECK(cudaMemcpyAsync(published, local_input.const_data_ptr(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    published = const_cast<void*>(local_input.const_data_ptr());
  }
  auto* input_peers = fa->resolve_rank_data(stream, published);
  auto launch = [&](auto ngpu_tag) {
    constexpr int ngpu = decltype(ngpu_tag)::value;
    vllm::dsv4_tp_input_owned::launch_gather_bf16<ngpu>(
        input_peers, fa->sg_, fa->self_sg_,
        reinterpret_cast<nv_bfloat16*>(full_output.mutable_data_ptr()),
        int(rows), int(local_hidden), fa->rank_, stream);
  };
  switch (fa->world_size_) {
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
  }
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_owned_router(
    fptr_t _fa, const torch::stable::Tensor& local_input,
    const torch::stable::Tensor& weight, torch::stable::Tensor& output,
    torch::stable::Tensor& partial, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr && (fa->world_size_ == 2 ||
                                   fa->world_size_ == 4 ||
                                   fa->world_size_ == 8),
                  "DSV4 owned router requires TP2, TP4, or TP8");
  const int64_t local_hidden = 4096 / fa->world_size_;
  const int rows = int(weight.size(0));
  STD_TORCH_CHECK(local_input.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16 &&
                      local_input.dim() == 2 && local_input.size(0) == 1 &&
                      local_input.size(1) == local_hidden &&
                      local_input.is_contiguous());
  STD_TORCH_CHECK(weight.scalar_type() == local_input.scalar_type() &&
                      weight.dim() == 2 && weight.size(1) == 4096 &&
                      weight.is_contiguous());
  STD_TORCH_CHECK(rows % vllm::packed_t<float>::P::size == 0);
  STD_TORCH_CHECK(output.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      output.numel() == rows && output.is_contiguous());
  STD_TORCH_CHECK(partial.scalar_type() == output.scalar_type() &&
                      partial.numel() >= rows && partial.is_contiguous());
  const torch::stable::accelerator::DeviceGuard device_guard(
      local_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(local_input.get_device_index());
  float* partial_ptr = reinterpret_cast<float*>(reg_buffer_ptr);
  if (partial_ptr != nullptr) {
    STD_TORCH_CHECK(int64_t(rows) * sizeof(float) <= reg_buffer_sz_bytes);
  } else {
    partial_ptr = reinterpret_cast<float*>(partial.mutable_data_ptr());
  }
  auto launch = [&](auto ngpu_tag) {
    constexpr int ngpu = decltype(ngpu_tag)::value;
    vllm::dsv4_tp_input_owned::launch_attention<ngpu>(
        reinterpret_cast<const nv_bfloat16*>(local_input.const_data_ptr()),
        nullptr, nullptr,
        reinterpret_cast<const nv_bfloat16*>(weight.const_data_ptr()), nullptr,
        nullptr, partial_ptr, 0, rows, 0, 0, fa->rank_, stream);
  };
  switch (fa->world_size_) {
    case 2:
      launch(std::integral_constant<int, 2>{});
      break;
    case 4:
      launch(std::integral_constant<int, 4>{});
      break;
    case 8:
      launch(std::integral_constant<int, 8>{});
      break;
  }
  fa->allreduce<float>(
      stream, partial_ptr, reinterpret_cast<float*>(output.mutable_data_ptr()),
      rows);
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_channel_owned_q2_down(
    fptr_t _fa, const torch::stable::Tensor& quant_mid,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& topk_ids, torch::stable::Tensor& output,
    fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 channel-owned Q2_K communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4,
                  "DSV4 channel-owned Q2_K requires TP2 or TP4");
  const int local_k = 2048 / fa->world_size_;
  const int local_rows = 4096 / fa->world_size_;
  const int top_k = static_cast<int>(topk_ids.numel());
  const torch::stable::accelerator::DeviceGuard device_guard(
      quant_mid.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(quant_mid.get_device_index());

  STD_TORCH_CHECK(
      quant_mid.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_mid.dim() == 2 && quant_mid.size(0) == top_k &&
          quant_mid.size(1) == local_k / 32 * 9 && quant_mid.is_contiguous(),
      "DSV4 channel-owned Q8_1 intermediate shape mismatch");
  STD_TORCH_CHECK(
      weights.scalar_type() == torch::headeronly::ScalarType::Byte &&
          weights.dim() == 3 && weights.size(1) == local_rows &&
          weights.size(2) == 2048 / QK_K * sizeof(block_q2_K) &&
          weights.is_contiguous(),
      "DSV4 channel-owned W2 must be repacked uint8[E,4096/TP,672]");
  STD_TORCH_CHECK(topk_ids.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      topk_ids.dim() == 1 && (top_k == 6 || top_k == 8) &&
                      topk_ids.is_contiguous(),
                  "DSV4 channel-owned top-k ids must be int32[6|8]");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.size(0) == 1 &&
          output.size(1) == local_rows && output.is_contiguous(),
      "DSV4 channel-owned Q2_K output must be BF16[1,4096/TP]");

  void* shared_quant = reinterpret_cast<void*>(reg_buffer_ptr);
  const int64_t quant_bytes =
      quant_mid.numel() * quant_mid.element_size();
  if (shared_quant != nullptr) {
    STD_TORCH_CHECK(quant_bytes <= reg_buffer_sz_bytes,
                    "DSV4 channel-owned Q8_1 registration buffer is too small");
    STD_CUDA_CHECK(cudaMemcpyAsync(shared_quant, quant_mid.const_data_ptr(),
                                   quant_bytes, cudaMemcpyDeviceToDevice,
                                   stream));
  } else {
    shared_quant = const_cast<void*>(quant_mid.const_data_ptr());
  }
  auto* quant_peers = fa->resolve_rank_data(stream, shared_quant);

#define LAUNCH_CHANNEL_Q2(NGPU, LOCAL_K, TOP_K)                               \
  do {                                                                         \
    vllm::dsv4_mhc_channel_owned::channel_q2_handshake<NGPU, false>            \
        <<<1, 32, 0, stream>>>(fa->sg_, fa->self_sg_, fa->rank_);              \
    vllm::dsv4_mhc_channel_owned::channel_q2_down<NGPU, TOP_K, LOCAL_K>         \
        <<<(local_rows + 7) / 8, 256, 0, stream>>>(                            \
            reinterpret_cast<const uint8_t*>(weights.const_data_ptr()),        \
            quant_peers,                                                       \
            reinterpret_cast<const int*>(topk_ids.const_data_ptr()),           \
            reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),         \
            weights.stride(0), local_rows, static_cast<int>(weights.size(0))); \
    vllm::dsv4_mhc_channel_owned::channel_q2_handshake<NGPU, true>             \
        <<<1, 32, 0, stream>>>(fa->sg_, fa->self_sg_, fa->rank_);              \
  } while (0)

  if (fa->world_size_ == 2) {
    if (top_k == 6) {
      LAUNCH_CHANNEL_Q2(2, 1024, 6);
    } else {
      LAUNCH_CHANNEL_Q2(2, 1024, 8);
    }
  } else if (top_k == 6) {
    LAUNCH_CHANNEL_Q2(4, 512, 6);
  } else {
    LAUNCH_CHANNEL_Q2(4, 512, 8);
  }
#undef LAUNCH_CHANNEL_Q2
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_channel_owned_q2_down_pending(
    fptr_t _fa, const torch::stable::Tensor& pending,
    const torch::stable::Tensor& addend, torch::stable::Tensor& scratch,
    torch::stable::Tensor& output,
    fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 channel-owned pending communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4,
                  "DSV4 channel-owned pending down requires TP2 or TP4");
  const int local_rows = 4096 / fa->world_size_;
  const torch::stable::accelerator::DeviceGuard device_guard(
      pending.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(pending.get_device_index());

  STD_TORCH_CHECK(
      pending.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          pending.dim() == 2 && pending.size(0) == 1 &&
          pending.size(1) == 4096 && pending.is_contiguous(),
      "DSV4 channel-owned pending payload must be BF16[1,4096]");
  STD_TORCH_CHECK(
      addend.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          addend.sizes() == pending.sizes() && addend.is_contiguous(),
      "DSV4 channel-owned shared addend must be BF16[1,4096]");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.size(0) == 1 &&
          output.size(1) == local_rows && output.is_contiguous(),
      "DSV4 channel-owned pending output must be BF16[1,4096/TP]");
  constexpr int64_t gathered_quant_bytes =
      6 * (2048 / QK8_1) * sizeof(block_q8_1);
  const int64_t required_scratch_bytes =
      gathered_quant_bytes + int64_t(local_rows) * sizeof(float);
  STD_TORCH_CHECK(
      scratch.scalar_type() == torch::headeronly::ScalarType::Byte &&
          scratch.is_contiguous() &&
          scratch.numel() >= required_scratch_bytes,
      "DSV4 channel-owned pending scratch is too small");

  const int64_t pending_bytes = pending.numel() * pending.element_size();
  const int64_t local_quant_bytes =
      6 * (2048 / fa->world_size_ / QK8_1) * sizeof(block_q8_1);
  const int64_t full_quant_bytes =
      6 * (2048 / QK8_1) * sizeof(block_q8_1);
  const int64_t publication_slot_bytes =
      (std::max(local_quant_bytes, full_quant_bytes) + pending_bytes + 255) &
      ~int64_t(255);
  void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
  STD_TORCH_CHECK(
      publication != nullptr &&
          2 * publication_slot_bytes <= reg_buffer_sz_bytes,
      "DSV4 parity publication buffer is missing or too small");
  const auto publication_it = fa->buffers_.find(publication);
  STD_TORCH_CHECK(publication_it != fa->buffers_.end(),
                  "DSV4 parity publication buffer is not registered");
  auto* publication_peers = publication_it->second;
  auto* header = reinterpret_cast<
      const vllm::dsv4_q2_mhc_ar::PendingQ2Header*>(
      static_cast<const uint8_t*>(pending.const_data_ptr()) + pending_bytes -
      sizeof(vllm::dsv4_q2_mhc_ar::PendingQ2Header));
  auto* gathered_quant = reinterpret_cast<block_q8_1*>(
      scratch.mutable_data_ptr());
  auto* gathered_addend = reinterpret_cast<float*>(
      static_cast<uint8_t*>(scratch.mutable_data_ptr()) +
      gathered_quant_bytes);
  cudaStreamCaptureStatus capture_status;
  STD_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  const bool push_addend = capture_status == cudaStreamCaptureStatusActive;

#define LAUNCH_PENDING_CHANNEL_Q2(                                            \
    NGPU, LOCAL_K, GATHER_ADDEND, PUSH_ADDEND, PUSH_QUANT)                    \
  do {                                                                         \
    vllm::dsv4_mhc_channel_owned::                                             \
        channel_q2_gather<NGPU, 6, LOCAL_K, GATHER_ADDEND, PUSH_ADDEND,       \
                          PUSH_QUANT>                                          \
        <<<1, 512, 0, stream>>>(                                                \
            reinterpret_cast<const block_q8_1*>(pending.const_data_ptr()),      \
            reinterpret_cast<const nv_bfloat16*>(addend.const_data_ptr()),      \
            publication_peers, gathered_quant, gathered_addend, fa->sg_,        \
            fa->self_sg_, local_rows, fa->rank_);                               \
    vllm::dsv4_mhc_channel_owned::                                             \
        channel_q2_down<NGPU, 6, LOCAL_K, true, PUSH_QUANT>                   \
        <<<(local_rows + 7) / 8, 256, 0, stream>>>(                            \
            nullptr, publication_peers, nullptr,                               \
            reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),         \
            0, local_rows, 0, header, nullptr, 0, fa->rank_,                   \
            PUSH_QUANT ? nullptr : gathered_quant,                            \
            (GATHER_ADDEND || PUSH_ADDEND) ? gathered_addend : nullptr,        \
            PUSH_QUANT ? fa->self_sg_ : nullptr);                             \
  } while (0)

  if (fa->world_size_ == 2) {
    if (push_addend) {
      LAUNCH_PENDING_CHANNEL_Q2(2, 1024, false, true, true);
    } else {
      LAUNCH_PENDING_CHANNEL_Q2(2, 1024, true, false, false);
    }
  } else {
    if (push_addend) {
      LAUNCH_PENDING_CHANNEL_Q2(4, 512, false, true, true);
    } else {
      LAUNCH_PENDING_CHANNEL_Q2(4, 512, true, false, false);
    }
  }
#undef LAUNCH_PENDING_CHANNEL_Q2
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_channel_owned_moe(
    fptr_t _fa, const torch::stable::Tensor& quant_input,
    const torch::stable::Tensor& w1, const torch::stable::Tensor& w2,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& addend, torch::stable::Tensor& scratch,
    torch::stable::Tensor& output, double swiglu_limit,
    fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 channel-owned MoE communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4,
                  "DSV4 channel-owned MoE requires TP2 or TP4");
  const int local_intermediate = 2048 / fa->world_size_;
  const int local_rows = 4096 / fa->world_size_;
  const torch::stable::accelerator::DeviceGuard device_guard(
      quant_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(quant_input.get_device_index());

  STD_TORCH_CHECK(
      quant_input.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_input.dim() == 2 && quant_input.size(0) == 1 &&
          quant_input.size(1) == 4096 / QK8_1 * 9 &&
          quant_input.is_contiguous(),
      "DSV4 channel-owned MoE input must be Q8_1 int32[1,1152]");
  STD_TORCH_CHECK(
      w1.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w1.dim() == 3 && w1.size(1) == 2 * local_intermediate &&
          w1.is_contiguous(),
      "DSV4 channel-owned W1 must be repacked IQ2_XXS uint8 experts");
  STD_TORCH_CHECK(
      w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w2.dim() == 3 && w2.size(0) == w1.size(0) &&
          w2.size(1) == local_rows &&
          w2.size(2) == 2048 / QK_K * sizeof(block_q2_K) &&
          w2.is_contiguous(),
      "DSV4 channel-owned W2 must be repacked uint8[E,4096/TP,672]");
  STD_TORCH_CHECK(
      topk_weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          topk_weights.numel() == 6 && topk_weights.is_contiguous(),
      "DSV4 channel-owned route weights must be contiguous float32[6]");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_ids.numel() == 6 && topk_ids.is_contiguous(),
      "DSV4 channel-owned route ids must be contiguous int32[6]");
  STD_TORCH_CHECK(
      addend.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          addend.dim() == 2 && addend.size(0) == 1 &&
          addend.size(1) == 4096 && addend.is_contiguous(),
      "DSV4 channel-owned shared addend must be BF16[1,4096]");
  STD_TORCH_CHECK(
      scratch.scalar_type() == torch::headeronly::ScalarType::Byte &&
          scratch.is_contiguous() &&
          scratch.numel() >= int64_t(local_rows) * sizeof(float),
      "DSV4 channel-owned MoE scratch is too small");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.size(0) == 1 &&
          output.size(1) == local_rows && output.is_contiguous(),
      "DSV4 channel-owned MoE output must be BF16[1,4096/TP]");

  constexpr int64_t full_quant_bytes =
      6 * (2048 / QK8_1) * sizeof(block_q8_1);
  const int64_t publication_slot_bytes =
      (full_quant_bytes + int64_t(4096) * sizeof(nv_bfloat16) + 255) &
      ~int64_t(255);
  void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
  STD_TORCH_CHECK(
      publication != nullptr &&
          2 * publication_slot_bytes <= reg_buffer_sz_bytes,
      "DSV4 producer-owned publication buffer is missing or too small");
  const auto publication_it = fa->buffers_.find(publication);
  STD_TORCH_CHECK(publication_it != fa->buffers_.end(),
                  "DSV4 producer-owned publication buffer is not registered");
  auto* publication_peers = publication_it->second;
  cudaStreamCaptureStatus capture_status;
  STD_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  const bool direct_addend = capture_status == cudaStreamCaptureStatusActive;
  auto* addend_peers = direct_addend
      ? fa->resolve_rank_data(
            stream, const_cast<void*>(addend.const_data_ptr()))
      : nullptr;

  slimserve::dsv4_ampere::
      launch_iq2_xxs_gate_up_swiglu_q8_1_decode_publish(
          w1.const_data_ptr(), quant_input.const_data_ptr(),
          reinterpret_cast<slimserve::dsv4_ampere::PublicationRankData*>(
              publication_peers),
          &fa->self_sg_->dsv4_channel_q2_epoch, fa->rank_, fa->world_size_,
          reinterpret_cast<const int*>(topk_ids.const_data_ptr()),
          reinterpret_cast<const float*>(topk_weights.const_data_ptr()),
          w1.stride(0), 4096, local_intermediate,
          static_cast<int>(w1.size(0)), static_cast<float>(swiglu_limit), true,
          stream);

#define LAUNCH_PRODUCER_OWNED_MOE(                                            \
    NGPU, LOCAL_K, OWNERSHIP_THREADS, DIRECT_ADDEND)                          \
  do {                                                                         \
    vllm::dsv4_mhc_channel_owned::                                             \
        channel_q2_gather<NGPU, 6, LOCAL_K, false, !DIRECT_ADDEND, false,     \
                          true, DIRECT_ADDEND>                                 \
        <<<1, OWNERSHIP_THREADS, 0, stream>>>(                                 \
            nullptr,                                                          \
            reinterpret_cast<const nv_bfloat16*>(addend.const_data_ptr()),     \
            publication_peers, nullptr, nullptr, fa->sg_,                     \
            fa->self_sg_, local_rows, fa->rank_);                              \
    vllm::dsv4_mhc_channel_owned::                                             \
        channel_q2_down<NGPU, 6, LOCAL_K, false, true, true>                  \
        <<<(local_rows + 7) / 8, 256, 0, stream>>>(                            \
            reinterpret_cast<const uint8_t*>(w2.const_data_ptr()),             \
            publication_peers,                                                \
            reinterpret_cast<const int*>(topk_ids.const_data_ptr()),           \
            reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),         \
            w2.stride(0), local_rows, static_cast<int>(w2.size(0)), nullptr,   \
            DIRECT_ADDEND ? addend_peers : nullptr, 0, fa->rank_, nullptr,    \
            nullptr, fa->self_sg_);                                           \
  } while (0)

  if (fa->world_size_ == 2) {
    if (direct_addend) {
      LAUNCH_PRODUCER_OWNED_MOE(2, 1024, 512, true);
    } else {
      LAUNCH_PRODUCER_OWNED_MOE(2, 1024, 512, false);
    }
  } else {
    if (direct_addend) {
      LAUNCH_PRODUCER_OWNED_MOE(4, 512, 512, true);
    } else {
      LAUNCH_PRODUCER_OWNED_MOE(4, 512, 512, false);
    }
  }
#undef LAUNCH_PRODUCER_OWNED_MOE
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_output_owned_moe(
    fptr_t _fa, const torch::stable::Tensor& quant_input,
    const torch::stable::Tensor& w1, const torch::stable::Tensor& w2,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& shared_quant,
    const torch::stable::Tensor& shared_w2, torch::stable::Tensor& output,
    double swiglu_limit, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 output-owned MoE communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4,
                  "DSV4 output-owned MoE requires TP2 or TP4");
  const int local_intermediate = 2048 / fa->world_size_;
  const int local_rows = 4096 / fa->world_size_;
  const torch::stable::accelerator::DeviceGuard device_guard(
      quant_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(quant_input.get_device_index());

  STD_TORCH_CHECK(
      quant_input.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_input.dim() == 2 && quant_input.size(0) == 1 &&
          quant_input.size(1) == 4096 / QK8_1 * 9 &&
          quant_input.is_contiguous(),
      "DSV4 output-owned MoE input must be Q8_1 int32[1,1152]");
  STD_TORCH_CHECK(
      w1.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w1.dim() == 3 && w1.size(1) == 2 * local_intermediate &&
          w1.is_contiguous(),
      "DSV4 output-owned W1 must be repacked IQ2_XXS uint8 experts");
  STD_TORCH_CHECK(
      w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w2.dim() == 3 && w2.size(0) == w1.size(0) &&
          w2.size(1) == local_rows &&
          w2.size(2) == 2048 / QK_K * sizeof(block_q2_K) &&
          w2.is_contiguous(),
      "DSV4 output-owned routed W2 must be uint8[E,4096/TP,672]");
  STD_TORCH_CHECK(
      topk_weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          topk_weights.numel() == 6 && topk_weights.is_contiguous(),
      "DSV4 output-owned route weights must be contiguous float32[6]");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_ids.numel() == 6 && topk_ids.is_contiguous(),
      "DSV4 output-owned route ids must be contiguous int32[6]");
  STD_TORCH_CHECK(
      shared_quant.scalar_type() == torch::headeronly::ScalarType::Int &&
          shared_quant.dim() == 2 && shared_quant.size(0) == 1 &&
          shared_quant.size(1) == local_intermediate / QK8_1 * 9 &&
          shared_quant.is_contiguous(),
      "DSV4 output-owned shared activation must be local Q8_1");
  STD_TORCH_CHECK(
      shared_w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          shared_w2.dim() == 2 && shared_w2.size(0) == local_rows &&
          shared_w2.size(1) ==
              2048 / QK8_0 * static_cast<int64_t>(sizeof(block_q8_0)) &&
          shared_w2.is_contiguous(),
      "DSV4 output-owned shared W2 must be aligned uint8[4096/TP,2176]");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.size(0) == 1 &&
          output.size(1) == local_rows && output.is_contiguous(),
      "DSV4 output-owned MoE output must be BF16[1,4096/TP]");

  constexpr int64_t full_quant_bytes =
      6 * (2048 / QK8_1) * sizeof(block_q8_1);
  const int64_t publication_slot_bytes =
      (full_quant_bytes + int64_t(4096) * sizeof(nv_bfloat16) + 255) &
      ~int64_t(255);
  void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
  STD_TORCH_CHECK(
      publication != nullptr &&
          2 * publication_slot_bytes <= reg_buffer_sz_bytes,
      "DSV4 output-owned publication buffer is missing or too small");
  const auto publication_it = fa->buffers_.find(publication);
  STD_TORCH_CHECK(publication_it != fa->buffers_.end(),
                  "DSV4 output-owned publication buffer is not registered");
  auto* publication_peers = publication_it->second;
  slimserve::dsv4_ampere::
      launch_iq2_xxs_gate_up_swiglu_q8_1_decode_publish(
          w1.const_data_ptr(), quant_input.const_data_ptr(),
          reinterpret_cast<slimserve::dsv4_ampere::PublicationRankData*>(
              publication_peers),
          &fa->self_sg_->dsv4_channel_q2_epoch, fa->rank_, fa->world_size_,
          reinterpret_cast<const int*>(topk_ids.const_data_ptr()),
          reinterpret_cast<const float*>(topk_weights.const_data_ptr()),
          w1.stride(0), 4096, local_intermediate,
          static_cast<int>(w1.size(0)), static_cast<float>(swiglu_limit), true,
          stream);

#define LAUNCH_OUTPUT_OWNED_MOE(NGPU, LOCAL_K)                               \
  do {                                                                        \
    vllm::dsv4_mhc_channel_owned::                                            \
        channel_shared_q8_publish<NGPU, 6, LOCAL_K>                          \
        <<<1, 64, 0, stream>>>(                                               \
            reinterpret_cast<const block_q8_1*>(                             \
                shared_quant.const_data_ptr()),                              \
            publication_peers, fa->sg_, fa->self_sg_, fa->rank_);            \
    vllm::dsv4_mhc_channel_owned::                                            \
        channel_q2_down<NGPU, 6, LOCAL_K, false, true, false, true>          \
        <<<(local_rows + 7) / 8, 256, 0, stream>>>(                           \
            reinterpret_cast<const uint8_t*>(w2.const_data_ptr()),            \
            publication_peers,                                                \
            reinterpret_cast<const int*>(topk_ids.const_data_ptr()),          \
            reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),         \
            w2.stride(0), local_rows, static_cast<int>(w2.size(0)), nullptr,  \
            nullptr, 0, fa->rank_, nullptr, nullptr, fa->self_sg_,            \
            reinterpret_cast<const uint8_t*>(shared_w2.const_data_ptr()));     \
  } while (0)

  if (fa->world_size_ == 2) {
    LAUNCH_OUTPUT_OWNED_MOE(2, 1024);
  } else {
    LAUNCH_OUTPUT_OWNED_MOE(4, 512);
  }
#undef LAUNCH_OUTPUT_OWNED_MOE
  STD_CUDA_CHECK(cudaGetLastError());
}

#if 0  // Rejected diagnostics: slower than the retained publication path.
void dsv4_output_owned_moe_fused_shared(
    fptr_t _fa, const torch::stable::Tensor& quant_input,
    const torch::stable::Tensor& w1, const torch::stable::Tensor& w2,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& shared_w1,
    const torch::stable::Tensor& shared_w2,
    torch::stable::Tensor& shared_scratch, torch::stable::Tensor& output,
    double swiglu_limit, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 fused-shared MoE communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4,
                  "DSV4 fused-shared MoE requires TP2 or TP4");
  const int local_intermediate = 2048 / fa->world_size_;
  const int local_rows = 4096 / fa->world_size_;
  const torch::stable::accelerator::DeviceGuard device_guard(
      quant_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(quant_input.get_device_index());

  STD_TORCH_CHECK(
      quant_input.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_input.dim() == 2 && quant_input.size(0) == 1 &&
          quant_input.size(1) == 4096 / QK8_1 * 9 &&
          quant_input.is_contiguous(),
      "DSV4 fused-shared input must be Q8_1 int32[1,1152]");
  STD_TORCH_CHECK(
      w1.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w1.dim() == 3 && w1.size(1) == 2 * local_intermediate &&
          w1.is_contiguous(),
      "DSV4 fused-shared W1 must be repacked IQ2_XXS uint8 experts");
  STD_TORCH_CHECK(
      w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w2.dim() == 3 && w2.size(0) == w1.size(0) &&
          w2.size(1) == local_rows &&
          w2.size(2) == 2048 / QK_K * sizeof(block_q2_K) &&
          w2.is_contiguous(),
      "DSV4 fused-shared routed W2 must be uint8[E,4096/TP,672]");
  STD_TORCH_CHECK(
      topk_weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          topk_weights.numel() == 6 && topk_weights.is_contiguous(),
      "DSV4 fused-shared route weights must be contiguous float32[6]");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_ids.numel() == 6 && topk_ids.is_contiguous(),
      "DSV4 fused-shared route ids must be contiguous int32[6]");
  STD_TORCH_CHECK(
      shared_w1.scalar_type() == torch::headeronly::ScalarType::Byte &&
          shared_w1.dim() == 2 &&
          shared_w1.size(0) == 2 * local_intermediate &&
          shared_w1.size(1) ==
              4096 / QK8_0 * static_cast<int64_t>(sizeof(block_q8_0)) &&
          shared_w1.is_contiguous(),
      "DSV4 fused-shared W1 must be Q8_0 uint8[2*2048/TP,4352]");
  STD_TORCH_CHECK(
      shared_w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          shared_w2.dim() == 2 && shared_w2.size(0) == local_rows &&
          shared_w2.size(1) ==
              2048 / QK8_0 * static_cast<int64_t>(sizeof(block_q8_0)) &&
          shared_w2.is_contiguous(),
      "DSV4 fused-shared W2 must be aligned uint8[4096/TP,2176]");
  STD_TORCH_CHECK(
      shared_scratch.scalar_type() ==
              torch::headeronly::ScalarType::BFloat16 &&
          shared_scratch.dim() == 2 && shared_scratch.size(0) == 1 &&
          shared_scratch.size(1) == local_intermediate &&
          shared_scratch.is_contiguous(),
      "DSV4 fused-shared scratch must be BF16[1,2048/TP]");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.size(0) == 1 &&
          output.size(1) == local_rows && output.is_contiguous(),
      "DSV4 fused-shared output must be BF16[1,4096/TP]");

  constexpr int64_t full_quant_bytes =
      6 * (2048 / QK8_1) * sizeof(block_q8_1);
  const int64_t publication_slot_bytes =
      (full_quant_bytes + int64_t(4096) * sizeof(nv_bfloat16) + 255) &
      ~int64_t(255);
  void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
  STD_TORCH_CHECK(
      publication != nullptr &&
          2 * publication_slot_bytes <= reg_buffer_sz_bytes,
      "DSV4 fused-shared publication buffer is missing or too small");
  const auto publication_it = fa->buffers_.find(publication);
  STD_TORCH_CHECK(publication_it != fa->buffers_.end(),
                  "DSV4 fused-shared publication buffer is not registered");
  auto* publication_peers = publication_it->second;

  slimserve::dsv4_ampere::
      launch_iq2_xxs_gate_up_swiglu_q8_1_decode_publish(
          w1.const_data_ptr(), quant_input.const_data_ptr(),
          reinterpret_cast<slimserve::dsv4_ampere::PublicationRankData*>(
              publication_peers),
          &fa->self_sg_->dsv4_channel_q2_epoch, fa->rank_, fa->world_size_,
          reinterpret_cast<const int*>(topk_ids.const_data_ptr()),
          reinterpret_cast<const float*>(topk_weights.const_data_ptr()),
          w1.stride(0), 4096, local_intermediate,
          static_cast<int>(w1.size(0)), static_cast<float>(swiglu_limit), true,
          stream);

#define LAUNCH_OUTPUT_OWNED_FUSED_SHARED(NGPU, LOCAL_K)                       \
  do {                                                                         \
    CUDACHECK((vllm::dsv4_mhc_channel_owned::                                \
                  launch_channel_shared_gate_up_publish<NGPU, 6, LOCAL_K>(    \
                      shared_w1.const_data_ptr(),                              \
                      quant_input.const_data_ptr(),                            \
                      reinterpret_cast<nv_bfloat16*>(                          \
                          shared_scratch.mutable_data_ptr()),                  \
                      publication_peers, fa->sg_, fa->self_sg_,               \
                      4096 / QK8_0, static_cast<float>(swiglu_limit),          \
                      fa->rank_, stream)));                                   \
    vllm::dsv4_mhc_channel_owned::                                            \
        channel_q2_down<NGPU, 6, LOCAL_K, false, true, false, true>           \
        <<<(local_rows + 7) / 8, 256, 0, stream>>>(                            \
            reinterpret_cast<const uint8_t*>(w2.const_data_ptr()),            \
            publication_peers,                                                \
            reinterpret_cast<const int*>(topk_ids.const_data_ptr()),          \
            reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),        \
            w2.stride(0), local_rows, static_cast<int>(w2.size(0)), nullptr,  \
            nullptr, 0, fa->rank_, nullptr, nullptr, fa->self_sg_,            \
            reinterpret_cast<const uint8_t*>(shared_w2.const_data_ptr()));     \
  } while (0)

  if (fa->world_size_ == 2) {
    LAUNCH_OUTPUT_OWNED_FUSED_SHARED(2, 1024);
  } else {
    LAUNCH_OUTPUT_OWNED_FUSED_SHARED(4, 512);
  }
#undef LAUNCH_OUTPUT_OWNED_FUSED_SHARED
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_output_owned_moe_direct_shared(
    fptr_t _fa, const torch::stable::Tensor& quant_input,
    const torch::stable::Tensor& w1, const torch::stable::Tensor& w2,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& shared_quant,
    const torch::stable::Tensor& shared_w2, torch::stable::Tensor& output,
    double swiglu_limit, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 direct-shared MoE communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4,
                  "DSV4 direct-shared MoE requires TP2 or TP4");
  const int local_intermediate = 2048 / fa->world_size_;
  const int local_rows = 4096 / fa->world_size_;
  const torch::stable::accelerator::DeviceGuard device_guard(
      quant_input.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(quant_input.get_device_index());

  STD_TORCH_CHECK(
      quant_input.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_input.dim() == 2 && quant_input.size(0) == 1 &&
          quant_input.size(1) == 4096 / QK8_1 * 9 &&
          quant_input.is_contiguous(),
      "DSV4 direct-shared input must be Q8_1 int32[1,1152]");
  STD_TORCH_CHECK(
      w1.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w1.dim() == 3 && w1.size(1) == 2 * local_intermediate &&
          w1.is_contiguous(),
      "DSV4 direct-shared W1 must be repacked IQ2_XXS uint8 experts");
  STD_TORCH_CHECK(
      w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          w2.dim() == 3 && w2.size(0) == w1.size(0) &&
          w2.size(1) == local_rows &&
          w2.size(2) == 2048 / QK_K * sizeof(block_q2_K) &&
          w2.is_contiguous(),
      "DSV4 direct-shared routed W2 must be uint8[E,4096/TP,672]");
  STD_TORCH_CHECK(
      topk_weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          topk_weights.numel() == 6 && topk_weights.is_contiguous(),
      "DSV4 direct-shared route weights must be contiguous float32[6]");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_ids.numel() == 6 && topk_ids.is_contiguous(),
      "DSV4 direct-shared route ids must be contiguous int32[6]");
  STD_TORCH_CHECK(
      shared_quant.scalar_type() == torch::headeronly::ScalarType::Int &&
          shared_quant.dim() == 2 && shared_quant.size(0) == 1 &&
          shared_quant.size(1) == local_intermediate / QK8_1 * 9 &&
          shared_quant.is_contiguous(),
      "DSV4 direct-shared activation must be local Q8_1");
  STD_TORCH_CHECK(
      shared_w2.scalar_type() == torch::headeronly::ScalarType::Byte &&
          shared_w2.dim() == 2 && shared_w2.size(0) == local_rows &&
          shared_w2.size(1) ==
              2048 / QK8_0 * static_cast<int64_t>(sizeof(block_q8_0)) &&
          shared_w2.is_contiguous(),
      "DSV4 direct-shared W2 must be aligned uint8[4096/TP,2176]");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.size(0) == 1 &&
          output.size(1) == local_rows && output.is_contiguous(),
      "DSV4 direct-shared output must be BF16[1,4096/TP]");

  constexpr int64_t full_quant_bytes =
      6 * (2048 / QK8_1) * sizeof(block_q8_1);
  const int64_t publication_slot_bytes =
      (full_quant_bytes + int64_t(4096) * sizeof(nv_bfloat16) + 255) &
      ~int64_t(255);
  void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
  STD_TORCH_CHECK(
      publication != nullptr &&
          2 * publication_slot_bytes <= reg_buffer_sz_bytes,
      "DSV4 direct-shared publication buffer is missing or too small");
  const auto publication_it = fa->buffers_.find(publication);
  STD_TORCH_CHECK(publication_it != fa->buffers_.end(),
                  "DSV4 direct-shared publication buffer is not registered");
  auto* publication_peers = publication_it->second;
  cudaStreamCaptureStatus capture_status;
  STD_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  STD_TORCH_CHECK(
      capture_status == cudaStreamCaptureStatusActive,
      "DSV4 direct-shared peer activation requires CUDA graph capture");
  auto* shared_quant_peers = fa->resolve_rank_data(
      stream, const_cast<void*>(shared_quant.const_data_ptr()));

  slimserve::dsv4_ampere::
      launch_iq2_xxs_gate_up_swiglu_q8_1_decode_publish(
          w1.const_data_ptr(), quant_input.const_data_ptr(),
          reinterpret_cast<slimserve::dsv4_ampere::PublicationRankData*>(
              publication_peers),
          &fa->self_sg_->dsv4_channel_q2_epoch, fa->rank_, fa->world_size_,
          reinterpret_cast<const int*>(topk_ids.const_data_ptr()),
          reinterpret_cast<const float*>(topk_weights.const_data_ptr()),
          w1.stride(0), 4096, local_intermediate,
          static_cast<int>(w1.size(0)), static_cast<float>(swiglu_limit), true,
          stream);

#define LAUNCH_OUTPUT_OWNED_DIRECT_SHARED(NGPU, LOCAL_K)                     \
  do {                                                                         \
    vllm::dsv4_mhc_channel_owned::channel_q2_ready_advance<NGPU>              \
        <<<1, 32, 0, stream>>>(fa->sg_, fa->self_sg_, fa->rank_);             \
    vllm::dsv4_mhc_channel_owned::                                            \
        channel_q2_down<NGPU, 6, LOCAL_K, false, true, false, true, true>     \
        <<<(local_rows + 7) / 8, 256, 0, stream>>>(                            \
            reinterpret_cast<const uint8_t*>(w2.const_data_ptr()),            \
            publication_peers,                                                \
            reinterpret_cast<const int*>(topk_ids.const_data_ptr()),          \
            reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),        \
            w2.stride(0), local_rows, static_cast<int>(w2.size(0)), nullptr,  \
            nullptr, 0, fa->rank_, nullptr, nullptr, fa->self_sg_,            \
            reinterpret_cast<const uint8_t*>(shared_w2.const_data_ptr()),     \
            shared_quant_peers, 0);                                           \
  } while (0)
  if (fa->world_size_ == 2) {
    LAUNCH_OUTPUT_OWNED_DIRECT_SHARED(2, 1024);
  } else {
    LAUNCH_OUTPUT_OWNED_DIRECT_SHARED(4, 512);
  }
#undef LAUNCH_OUTPUT_OWNED_DIRECT_SHARED
  STD_CUDA_CHECK(cudaGetLastError());
}

#endif

void dsv4_output_owned_q8(
    fptr_t _fa, const torch::stable::Tensor& local_quant,
    const torch::stable::Tensor& aligned_weight,
    torch::stable::Tensor& output, int64_t rows_per_cta,
    fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 output-owned Q8 communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4 ||
                      fa->world_size_ == 8,
                  "DSV4 output-owned Q8 requires TP2, TP4, or TP8");
  STD_TORCH_CHECK(
      local_quant.scalar_type() == torch::headeronly::ScalarType::Int &&
          local_quant.dim() == 2 && local_quant.is_contiguous() &&
          local_quant.size(0) >= 1 &&
          local_quant.size(0) <=
              vllm::dsv4_tp_output_owned::kMaxTokens &&
          local_quant.size(1) %
                  vllm::dsv4_tp_output_owned::kQ8Words ==
              0,
      "DSV4 output-owned Q8 activation must be packed int32 [1..8, blocks*9]");
  const int tokens = static_cast<int>(local_quant.size(0));
  const int local_blocks = static_cast<int>(
      local_quant.size(1) / vllm::dsv4_tp_output_owned::kQ8Words);
  const int full_blocks = local_blocks * fa->world_size_;
  STD_TORCH_CHECK(
      full_blocks > 0 &&
          full_blocks <=
              vllm::dsv4_tp_output_owned::kMaxFullColumns /
                  vllm::dsv4_tp_output_owned::kQ8Block,
      "DSV4 output-owned Q8 supports at most 8192 full input columns");
  const int local_rows = 4096 / fa->world_size_;
  STD_TORCH_CHECK(
      aligned_weight.scalar_type() ==
              torch::headeronly::ScalarType::Byte &&
          aligned_weight.dim() == 2 && aligned_weight.is_contiguous() &&
          aligned_weight.size(0) == local_rows &&
          aligned_weight.size(1) == int64_t(full_blocks) * 34,
      "DSV4 output-owned Q8 weight must be aligned uint8 [4096/TP, full_K/32*34]");
  STD_TORCH_CHECK(
      output.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          output.dim() == 2 && output.is_contiguous() &&
          output.size(0) == tokens && output.size(1) == local_rows,
      "DSV4 output-owned Q8 output must be BF16 [tokens,4096/TP]");
  STD_TORCH_CHECK(rows_per_cta == 1 || rows_per_cta == 2 ||
                      rows_per_cta == 4,
                  "DSV4 output-owned Q8 rows_per_cta must be 1, 2, or 4");
  STD_TORCH_CHECK(
      reg_buffer_ptr != 0 &&
          2 * vllm::dsv4_tp_output_owned::kSlotBytes <=
              reg_buffer_sz_bytes,
      "DSV4 output-owned Q8 publication buffer is missing or too small");
  void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
  const auto publication_it = fa->buffers_.find(publication);
  STD_TORCH_CHECK(publication_it != fa->buffers_.end(),
                  "DSV4 output-owned Q8 publication buffer is not registered");
  auto* publication_peers = publication_it->second;

  const torch::stable::accelerator::DeviceGuard device_guard(
      local_quant.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(local_quant.get_device_index());
#define LAUNCH_OUTPUT_OWNED_Q8(NGPU)                                         \
  vllm::dsv4_tp_output_owned::launch<NGPU>(                                  \
      reinterpret_cast<const block_q8_1*>(local_quant.const_data_ptr()),      \
      reinterpret_cast<const uint8_t*>(aligned_weight.const_data_ptr()),      \
      publication_peers, fa->sg_, fa->self_sg_,                              \
      reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()), tokens,      \
      local_rows, local_blocks, fa->rank_, static_cast<int>(rows_per_cta),    \
      stream)
  if (fa->world_size_ == 2) {
    LAUNCH_OUTPUT_OWNED_Q8(2);
  } else if (fa->world_size_ == 4) {
    LAUNCH_OUTPUT_OWNED_Q8(4);
  } else {
    LAUNCH_OUTPUT_OWNED_Q8(8);
  }
#undef LAUNCH_OUTPUT_OWNED_Q8
  STD_CUDA_CHECK(cudaGetLastError());
}

void dsv4_owned_reduce_scatter(
    fptr_t _fa, const torch::stable::Tensor& input,
    const std::optional<torch::stable::Tensor>& addend,
    torch::stable::Tensor& output, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr,
                  "DSV4 owned reduce-scatter communicator is null");
  STD_TORCH_CHECK(fa->world_size_ == 2 || fa->world_size_ == 4 ||
                      fa->world_size_ == 8,
                  "DSV4 owned reduce-scatter requires TP2, TP4, or TP8");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          input.dim() == 2 && input.is_contiguous() &&
          input.size(0) >= 1 &&
          input.size(0) <= vllm::dsv4_tp_reduce_scatter::kMaxTokens &&
          input.size(1) == vllm::dsv4_tp_reduce_scatter::kHidden,
      "DSV4 owned reduce-scatter input must be BF16 [1..8,4096]");
  if (addend) {
    STD_TORCH_CHECK(addend->scalar_type() == input.scalar_type() &&
                        addend->sizes() == input.sizes() &&
                        addend->is_contiguous(),
                    "DSV4 owned reduce-scatter addend must match input");
  }
  const int local_hidden =
      vllm::dsv4_tp_reduce_scatter::kHidden / fa->world_size_;
  STD_TORCH_CHECK(
      output.scalar_type() == input.scalar_type() && output.dim() == 2 &&
          output.is_contiguous() && output.size(0) == input.size(0) &&
          output.size(1) == local_hidden,
      "DSV4 owned reduce-scatter output must be BF16 [tokens,4096/TP]");
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  const bool direct = reg_buffer_ptr == 0;
  vllm::RankData* input_peers = nullptr;
  vllm::RankData* addend_peers = nullptr;
  if (direct) {
    input_peers = fa->resolve_rank_data(
        stream, const_cast<void*>(input.const_data_ptr()));
    if (addend) {
      addend_peers = fa->resolve_rank_data(
          stream, const_cast<void*>(addend->const_data_ptr()));
    }
  } else {
    STD_TORCH_CHECK(
        2 * vllm::dsv4_tp_reduce_scatter::kSlotBytes <=
            reg_buffer_sz_bytes,
        "DSV4 owned reduce-scatter publication buffer is too small");
    void* publication = reinterpret_cast<void*>(reg_buffer_ptr);
    const auto publication_it = fa->buffers_.find(publication);
    STD_TORCH_CHECK(
        publication_it != fa->buffers_.end(),
        "DSV4 owned reduce-scatter publication buffer is not registered");
    input_peers = publication_it->second;
  }
  const auto* addend_ptr = addend
      ? reinterpret_cast<const nv_bfloat16*>(addend->const_data_ptr())
      : nullptr;
#define LAUNCH_OWNED_RS(NGPU)                                                 \
  vllm::dsv4_tp_reduce_scatter::launch<NGPU>(                                 \
      reinterpret_cast<const nv_bfloat16*>(input.const_data_ptr()),           \
      addend_ptr, input_peers, addend_peers, fa->sg_, fa->self_sg_,           \
      reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()),              \
      static_cast<int>(input.size(0)), fa->rank_, direct, stream)
  if (fa->world_size_ == 2) {
    LAUNCH_OWNED_RS(2);
  } else if (fa->world_size_ == 4) {
    LAUNCH_OWNED_RS(4);
  } else {
    LAUNCH_OWNED_RS(8);
  }
#undef LAUNCH_OWNED_RS
  STD_CUDA_CHECK(cudaGetLastError());
}

void all_reduce_dsv4_q2_mhc(
    fptr_t _fa, const torch::stable::Tensor& pending,
    const torch::stable::Tensor& addend,
    torch::stable::Tensor& producer_output,
    const torch::stable::Tensor& residual,
    const torch::stable::Tensor& post_mix,
    const torch::stable::Tensor& comb_mix, const torch::stable::Tensor& fn,
    torch::stable::Tensor& residual_out, torch::stable::Tensor& partial,
    const torch::stable::Tensor& scale, const torch::stable::Tensor& base,
    torch::stable::Tensor& next_post, torch::stable::Tensor& next_comb,
    torch::stable::Tensor& layer_input,
    const torch::stable::Tensor& norm_weight,
    torch::stable::Tensor& quant_input, double rms_eps, double pre_eps,
    double sinkhorn_eps, double post_multiplier, int64_t sinkhorn_repeat,
    double norm_eps, fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr, "DSV4 Q2_K mHC communicator is null");
  const torch::stable::accelerator::DeviceGuard device_guard(
      pending.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(pending.get_device_index());

  STD_TORCH_CHECK(
      pending.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
          pending.dim() == 2 && pending.size(0) == 1 &&
          pending.size(1) == 4096 && pending.is_contiguous(),
      "DSV4 pending Q2_K input must be contiguous BF16[1,4096]");
  STD_TORCH_CHECK(
      addend.scalar_type() == pending.scalar_type() &&
          addend.sizes() == pending.sizes() && addend.is_contiguous(),
      "DSV4 pending Q2_K addend must match pending input");
  STD_TORCH_CHECK(
      producer_output.scalar_type() == pending.scalar_type() &&
          producer_output.sizes() == pending.sizes() &&
          producer_output.is_contiguous(),
      "DSV4 Q2_K producer output must be contiguous BF16[1,4096]");
  STD_TORCH_CHECK(
      residual.scalar_type() == pending.scalar_type() &&
          residual.dim() == 3 && residual.size(0) == 1 &&
          residual.size(1) == 4 && residual.size(2) == 4096 &&
          residual.is_contiguous(),
      "DSV4 Q2_K mHC residual must be contiguous BF16[1,4,4096]");
  STD_TORCH_CHECK(post_mix.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      post_mix.numel() == 4 && post_mix.is_contiguous());
  STD_TORCH_CHECK(comb_mix.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      comb_mix.numel() == 16 && comb_mix.is_contiguous());
  STD_TORCH_CHECK(fn.scalar_type() == torch::headeronly::ScalarType::Half ||
                  fn.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(fn.numel() == 24 * 4 * 4096 && fn.is_contiguous());
  STD_TORCH_CHECK(partial.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      partial.numel() >=
                          vllm::dsv4_q2_mhc_ar::kRequiredPartialValues &&
                      partial.is_contiguous(),
                  "DSV4 Q2_K mHC partial workspace is too small");
  STD_TORCH_CHECK(norm_weight.scalar_type() == pending.scalar_type() &&
                  norm_weight.numel() == 4096 && norm_weight.is_contiguous());
  STD_TORCH_CHECK(
      quant_input.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_input.dim() == 2 && quant_input.size(0) == 1 &&
          quant_input.size(1) == 4096 / 32 * 9 && quant_input.is_contiguous());

  void* local_output = reinterpret_cast<void*>(reg_buffer_ptr);
  if (local_output != nullptr) {
    STD_TORCH_CHECK(producer_output.numel() * producer_output.element_size() <=
                    reg_buffer_sz_bytes);
  } else {
    local_output = producer_output.mutable_data_ptr();
  }
  const auto* pending_bytes =
      reinterpret_cast<const uint8_t*>(pending.const_data_ptr());
  const auto* header =
      reinterpret_cast<const vllm::dsv4_q2_mhc_ar::PendingQ2Header*>(
          pending_bytes + pending.numel() * pending.element_size() -
          sizeof(vllm::dsv4_q2_mhc_ar::PendingQ2Header));

  auto launch = [&](auto* fn_ptr) {
    fa->allreduce_dsv4_q2_mhc(
        stream, reinterpret_cast<const block_q8_1*>(pending.const_data_ptr()),
        header, reinterpret_cast<nv_bfloat16*>(local_output),
        reinterpret_cast<const nv_bfloat16*>(addend.const_data_ptr()),
        reinterpret_cast<const nv_bfloat16*>(residual.const_data_ptr()),
        reinterpret_cast<const float*>(post_mix.const_data_ptr()),
        reinterpret_cast<const float*>(comb_mix.const_data_ptr()), fn_ptr,
        reinterpret_cast<nv_bfloat16*>(residual_out.mutable_data_ptr()),
        reinterpret_cast<float*>(partial.mutable_data_ptr()),
        reinterpret_cast<const float*>(scale.const_data_ptr()),
        reinterpret_cast<const float*>(base.const_data_ptr()),
        reinterpret_cast<float*>(next_post.mutable_data_ptr()),
        reinterpret_cast<float*>(next_comb.mutable_data_ptr()),
        reinterpret_cast<nv_bfloat16*>(layer_input.mutable_data_ptr()),
        static_cast<float>(rms_eps), static_cast<float>(pre_eps),
        static_cast<float>(sinkhorn_eps), static_cast<float>(post_multiplier),
        static_cast<int>(sinkhorn_repeat),
        reinterpret_cast<const nv_bfloat16*>(norm_weight.const_data_ptr()),
        reinterpret_cast<tms::dsv4_mhc::block_q8_1*>(
            quant_input.mutable_data_ptr()),
        static_cast<float>(norm_eps));
  };
  if (fn.scalar_type() == torch::headeronly::ScalarType::Half) {
    launch(reinterpret_cast<const half*>(fn.const_data_ptr()));
  } else {
    launch(reinterpret_cast<const float*>(fn.const_data_ptr()));
  }
}

void dsv4_indexer_peer_topk(
    fptr_t _fa, torch::stable::Tensor& logits,
    const torch::stable::Tensor& lengths, torch::stable::Tensor& output,
    torch::stable::Tensor& workspace, int64_t k, int64_t max_seq_len,
    fptr_t reg_buffer_ptr, int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr, "DSV4 peer top-k communicator is null");
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(logits.get_device_index());

  void* shared_input = reinterpret_cast<void*>(reg_buffer_ptr);
  if (shared_input != nullptr) {
    const int64_t bytes = logits.numel() * logits.element_size();
    STD_TORCH_CHECK(bytes <= reg_buffer_sz_bytes,
                    "DSV4 peer top-k registration buffer is too small");
    STD_CUDA_CHECK(cudaMemcpyAsync(shared_input, logits.const_data_ptr(), bytes,
                                   cudaMemcpyDeviceToDevice, stream));
  } else {
    shared_input = logits.mutable_data_ptr();
  }
  auto* peer_data = fa->resolve_rank_data(stream, shared_input);
  vllm::RankData* output_peer_data = nullptr;
  if (reg_buffer_ptr == 0) {
    output_peer_data =
        fa->resolve_rank_data(stream, output.mutable_data_ptr());
  }
  std::array<int64_t, 8> signal_ptrs{};
  for (int index = 0; index < fa->world_size_; ++index) {
    signal_ptrs[index] = reinterpret_cast<int64_t>(fa->sg_.signals[index]);
  }
  launch_dsv4_indexer_peer_topk_impl(
      logits, lengths, output, workspace, k, max_seq_len,
      reinterpret_cast<int64_t>(peer_data), signal_ptrs,
      reinterpret_cast<int64_t>(fa->self_sg_), fa->rank_, fa->world_size_,
      reinterpret_cast<int64_t>(output_peer_data));
}

void dsv4_indexer_token_merge(
    fptr_t _fa, torch::stable::Tensor& logits,
    const torch::stable::Tensor& lengths,
    const torch::stable::Tensor& local_indices,
    torch::stable::Tensor& output, int64_t k, fptr_t reg_buffer_ptr,
    int64_t reg_buffer_sz_bytes) {
  auto* fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fa != nullptr, "DSV4 token-shard communicator is null");
  const torch::stable::accelerator::DeviceGuard device_guard(
      logits.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(logits.get_device_index());

  const int64_t logits_bytes = logits.numel() * logits.element_size();
  const int64_t indices_bytes =
      local_indices.numel() * local_indices.element_size();
  const int64_t lengths_bytes = lengths.numel() * lengths.element_size();
  constexpr int64_t alignment = 256;
  const auto align_up = [=](int64_t value) {
    return (value + alignment - 1) & ~(alignment - 1);
  };

  vllm::RankData* logits_peers;
  vllm::RankData* indices_peers;
  vllm::RankData* lengths_peers;
  int64_t logits_offset = 0;
  int64_t indices_offset = 0;
  int64_t lengths_offset = 0;
  void* shared_input = reinterpret_cast<void*>(reg_buffer_ptr);
  if (shared_input != nullptr) {
    const int64_t aligned_indices_bytes = align_up(indices_bytes);
    const int64_t aligned_lengths_bytes = align_up(lengths_bytes);
    lengths_offset = reg_buffer_sz_bytes - aligned_lengths_bytes;
    indices_offset = lengths_offset - aligned_indices_bytes;
    STD_TORCH_CHECK(logits_bytes <= indices_offset,
                    "DSV4 token-shard registration buffer is too small");
    auto* base = static_cast<char*>(shared_input);
    STD_CUDA_CHECK(cudaMemcpyAsync(base, logits.const_data_ptr(), logits_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    STD_CUDA_CHECK(cudaMemcpyAsync(base + indices_offset,
                                   local_indices.const_data_ptr(), indices_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    STD_CUDA_CHECK(cudaMemcpyAsync(base + lengths_offset,
                                   lengths.const_data_ptr(), lengths_bytes,
                                   cudaMemcpyDeviceToDevice, stream));
    logits_peers = fa->resolve_rank_data(stream, base);
    indices_peers = logits_peers;
    lengths_peers = logits_peers;
  } else {
    logits_peers = fa->resolve_rank_data(stream, logits.mutable_data_ptr());
    indices_peers = fa->resolve_rank_data(
        stream, const_cast<void*>(local_indices.const_data_ptr()));
    lengths_peers = fa->resolve_rank_data(
        stream, const_cast<void*>(lengths.const_data_ptr()));
  }

  std::array<int64_t, 8> signal_ptrs{};
  for (int index = 0; index < fa->world_size_; ++index) {
    signal_ptrs[index] = reinterpret_cast<int64_t>(fa->sg_.signals[index]);
  }
  launch_dsv4_indexer_token_merge_impl(
      logits, lengths, local_indices, output, k,
      reinterpret_cast<int64_t>(logits_peers),
      reinterpret_cast<int64_t>(indices_peers),
      reinterpret_cast<int64_t>(lengths_peers), logits_offset, indices_offset,
      lengths_offset, signal_ptrs, reinterpret_cast<int64_t>(fa->self_sg_),
      fa->rank_, fa->world_size_);
}

void dispose(fptr_t _fa) {
  delete reinterpret_cast<vllm::CustomAllreduce*>(_fa);
}

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  STD_TORCH_CHECK(fake_ipc_ptrs.size() == fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
std::tuple<std::vector<int64_t>, std::vector<int64_t>>
get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa,
                            const std::vector<std::vector<int64_t>>& handles,
                            const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  fa->register_graph_buffers(bytes, offsets);
}

std::tuple<fptr_t, torch::stable::Tensor> allocate_shared_buffer_and_handle(
    int64_t size) {
  int device_index;
  STD_CUDA_CHECK(cudaGetDevice(&device_index));
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  void* buffer;
  cudaStreamCaptureMode mode = cudaStreamCaptureModeRelaxed;
  const cudaStream_t stream = get_current_cuda_stream(device_index);
  STD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Allocate buffer
#if defined(USE_ROCM)
  // data buffers need to be "uncached" for signal on MI200
  STD_CUDA_CHECK(
      hipExtMallocWithFlags((void**)&buffer, size, hipDeviceMallocUncached));
#else
  STD_CUDA_CHECK(cudaMalloc((void**)&buffer, size));
#endif
  STD_CUDA_CHECK(cudaMemsetAsync(buffer, 0, size, stream));
  STD_CUDA_CHECK(cudaStreamSynchronize(stream));
  STD_CUDA_CHECK(cudaThreadExchangeStreamCaptureMode(&mode));

  // Create IPC memhandle for the allocated buffer.
  // Will use it in open_mem_handle.
  auto handle = torch::stable::empty(
      {static_cast<int64_t>(sizeof(cudaIpcMemHandle_t))},
      torch::headeronly::ScalarType::Byte, std::nullopt,
      torch::stable::Device(torch::stable::DeviceType::CPU));
  STD_CUDA_CHECK(cudaIpcGetMemHandle(
      (cudaIpcMemHandle_t*)handle.mutable_data_ptr(), buffer));

  return std::make_tuple(reinterpret_cast<fptr_t>(buffer), handle);
}

fptr_t open_mem_handle(torch::stable::Tensor& mem_handle) {
  void* ipc_ptr;
  STD_CUDA_CHECK(cudaIpcOpenMemHandle(
      (void**)&ipc_ptr,
      *((const cudaIpcMemHandle_t*)mem_handle.const_data_ptr()),
      cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<fptr_t>(ipc_ptr);
}

void free_shared_buffer(fptr_t buffer) {
  STD_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(buffer)));
}
