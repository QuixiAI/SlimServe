#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>

#include "../../../cuda_compat.h"
#include "../../cub_helpers.h"
#include "../../dispatch_utils.h"
#include "../../torch_utils.h"

#include <torch/csrc/stable/ops.h>

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"
#include "moe.cuh"
#include "moe_vec.cuh"

#ifndef USE_ROCM
  #include "mmq_v2/mmq_v2.cuh"
  #include "../../../quixicore/quant/dsv4_moe_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_mxfp4_moe_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_mxfp4_mmq_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_mxfp4_seg_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_hybrid_seg_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_shared_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_o_proj_ampere.cuh"
  #include "../../../quixicore/quant/dsv4_q8_ampere.cuh"
#endif

// Q8 gemv
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[off * kx], (int32_t*)vy + off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

template <typename scalar_t, typename weight_t>
static __global__ void dsv4_slot_sum_kernel(const float* __restrict__ slots,
                                            const weight_t* __restrict__ weights,
                                            scalar_t* __restrict__ out,
                                            const int64_t values,
                                            const int out_row,
                                            const int top_k,
                                            const int64_t stride_w_token,
                                            const int64_t stride_w_topk) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= values) {
    return;
  }
  const int64_t token = idx / out_row;
  const int row = idx - token * (int64_t)out_row;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    if (k >= top_k) {
      break;
    }
    const float v = slots[(token * (int64_t)top_k + k) * out_row + row];
    const float ww =
        static_cast<float>(weights[token * stride_w_token + k * stride_w_topk]);
    sum += (isfinite(v) && isfinite(ww)) ? v * ww : 0.0f;
  }
  out[idx] = static_cast<scalar_t>(sum);
}

torch::stable::Tensor ggml_dequantize(
    torch::stable::Tensor W,  // quant weight
    int64_t type, int64_t m, int64_t n,
    std::optional<torch::headeronly::ScalarType> const& dtype) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      W.get_device_index());
  auto dtype_ = dtype.value_or(torch::headeronly::ScalarType::Half);
  auto DW = torch::stable::empty({m, n}, dtype_, std::nullopt, W.device());
  torch::stable::fill_(DW, 0.0);
  cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(DW.scalar_type(), "ggml_dequantize", [&] {
    auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
    // An unhandled type used to return nullptr and be called anyway, which
    // segfaults instead of naming the type that is missing.
    STD_TORCH_CHECK(to_cuda != nullptr,
                    "ggml_dequantize: no dequant kernel for GGUF type ", type);
    to_cuda((void*)W.data_ptr(), (scalar_t*)DW.data_ptr(), m * n, stream);
  });

  return DW;
}

#ifndef USE_ROCM
// Dequantize into a caller-provided buffer. Skips the alloc and zero-fill of
// ggml_dequantize so a persistent scratch can be reused per call, which keeps
// the wide-batch cuBLAS route CUDA-graph safe (fixed pointer, no capture-time
// allocs beyond the first touch).
void ggml_dequantize_into(torch::stable::Tensor W, int64_t type, int64_t m,
                          int64_t n, torch::stable::Tensor Y) {
  const torch::stable::accelerator::DeviceGuard device_guard(
      W.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      Y.scalar_type(), "ggml_dequantize_into", [&] {
        auto to_cuda = ggml_get_to_cuda<scalar_t>(type);
        STD_TORCH_CHECK(
            to_cuda != nullptr,
            "ggml_dequantize_into: no dequant kernel for GGUF type ", type);
        to_cuda((void*)W.data_ptr(), (scalar_t*)Y.data_ptr(), m * n, stream);
      });
}
#endif  // USE_ROCM

torch::stable::Tensor ggml_mul_mat_vec_a8(
    torch::stable::Tensor W,  // quant weight
    torch::stable::Tensor X,  // input
    int64_t type, int64_t row) {
  int64_t col = X.sizes()[1];
  int64_t vecs = X.sizes()[0];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({vecs, row}, X.scalar_type(), std::nullopt,
                                W.device());
  // No output pre-fill: the vector kernels write every in-range dst element.
  // Audited on gfx942 by poisoning Y with NaN and checking for survivors over
  // Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0 and IQ2_XXS, at row counts that do and
  // do not divide the tile -- clean for every type. The fill it replaces cost
  // ~1.8us against 7-9us of kernel, i.e. a fifth of every bs=1 decode matmul,
  // and there are hundreds of them per step.
  //
  // The mmq/moe entry points below keep theirs: the same audit shows the
  // IQ2_XXS *tile* kernel leaving elements unwritten, so there the fill is
  // load-bearing.
  cudaStream_t stream = get_current_cuda_stream();
  auto quant_X = torch::stable::empty({vecs, padded / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, W.device());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_mul_mat_vec_a8", [&] {
        quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                         (void*)quant_X.data_ptr(), col, vecs,
                                         stream);
        switch (type) {
          case 2:
            mul_mat_vec_q4_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 3:
            mul_mat_vec_q4_1_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 6:
            mul_mat_vec_q5_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 7:
            mul_mat_vec_q5_1_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 8:
            mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 39:
            mul_mat_vec_mxfp4_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 10:
            mul_mat_vec_q2_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 11:
            mul_mat_vec_q3_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 12:
            mul_mat_vec_q4_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 13:
            mul_mat_vec_q5_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 14:
            mul_mat_vec_q6_K_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 16:
            mul_mat_vec_iq2_xxs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 17:
            mul_mat_vec_iq2_xs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 18:
            mul_mat_vec_iq3_xxs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 19:
            mul_mat_vec_iq1_s_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 20:
            mul_mat_vec_iq4_nl_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 21:
            mul_mat_vec_iq3_s_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 22:
            mul_mat_vec_iq2_s_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 23:
            mul_mat_vec_iq4_xs_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
          case 29:
            mul_mat_vec_iq1_m_q8_1_cuda<scalar_t>(
                (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                (scalar_t*)Y.data_ptr(), col, row, vecs, stream);
            break;
        }
      });
  return Y;
}

torch::stable::Tensor ggml_quantize_q8_1(torch::stable::Tensor X) {
  STD_TORCH_CHECK(X.dim() == 2, "ggml_quantize_q8_1: X must be 2D");
  STD_TORCH_CHECK(X.is_contiguous(),
                  "ggml_quantize_q8_1: X must be contiguous");
  const int64_t tokens = X.size(0);
  const int64_t cols = X.size(1);
  const int64_t padded = (cols + 511) / 512 * 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto quant = torch::stable::empty(
      {tokens, padded / 32 * 9}, torch::headeronly::ScalarType::Int,
      std::nullopt, X.device());
  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_quantize_q8_1", [&] {
        quantize_row_q8_1_cuda<scalar_t>(
            static_cast<const scalar_t*>(X.data_ptr()), quant.data_ptr(),
            int(cols), int(tokens), stream);
      });
  return quant;
}

#ifndef USE_ROCM
std::tuple<torch::stable::Tensor, torch::stable::Tensor>
ggml_dsv4_rms_norm_q8_1(torch::stable::Tensor X,
                         torch::stable::Tensor weight, double epsilon) {
  STD_TORCH_CHECK(X.dim() == 2 && X.size(1) == 4096,
                  "DSV4 fused RMSNorm/Q8_1 requires [tokens, 4096]");
  STD_TORCH_CHECK(weight.dim() == 1 && weight.numel() == 4096,
                  "DSV4 fused RMSNorm/Q8_1 weight must have 4096 values");
  STD_TORCH_CHECK(X.scalar_type() == weight.scalar_type(),
                  "DSV4 fused RMSNorm/Q8_1 dtype mismatch");
  STD_TORCH_CHECK(X.is_contiguous() && weight.is_contiguous(),
                  "DSV4 fused RMSNorm/Q8_1 inputs must be contiguous");
  const int64_t tokens = X.size(0);
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto output = torch::stable::empty(X.sizes(), X.scalar_type(), std::nullopt,
                                     X.device());
  auto quant = torch::stable::empty(
      {tokens, 4096 / 32 * 9}, torch::headeronly::ScalarType::Int,
      std::nullopt, X.device());
  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_rms_norm_q8_1", [&] {
        slimserve::dsv4_ampere::launch_rms_norm_q8_1<scalar_t>(
            static_cast<const scalar_t*>(X.data_ptr()),
            static_cast<const scalar_t*>(weight.data_ptr()),
            static_cast<scalar_t*>(output.data_ptr()), quant.data_ptr(),
            int(tokens), float(epsilon), stream);
      });
  return {output, quant};
}

torch::stable::Tensor ggml_mul_mat_vec_prequant_a8(
    torch::stable::Tensor W, torch::stable::Tensor X,
    torch::stable::Tensor quant_X, int64_t type, int64_t row) {
  STD_TORCH_CHECK(type == 8,
                  "prequantized DSV4 GEMV currently supports Q8_0 weights");
  STD_TORCH_CHECK(X.dim() == 2 && quant_X.dim() == 2,
                  "prequantized DSV4 GEMV inputs must be 2D");
  STD_TORCH_CHECK(quant_X.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "prequantized DSV4 GEMV input must use packed int storage");
  const int64_t col = X.size(1);
  const int64_t vecs = X.size(0);
  const int64_t padded = (col + 511) / 512 * 512;
  STD_TORCH_CHECK(quant_X.size(0) == vecs &&
                      quant_X.size(1) == padded / 32 * 9,
                  "prequantized DSV4 GEMV packed shape mismatch");
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({vecs, row}, X.scalar_type(), std::nullopt,
                                X.device());
  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_mul_mat_vec_prequant_a8", [&] {
        mul_mat_vec_q8_0_q8_1_cuda<scalar_t>(
            W.data_ptr(), quant_X.data_ptr(),
            static_cast<scalar_t*>(Y.data_ptr()), int(col), int(row),
            int(vecs), stream);
      });
  return Y;
}

torch::stable::Tensor ggml_dsv4_repack_q8_0_aligned(
    torch::stable::Tensor W) {
  STD_TORCH_CHECK(W.dim() == 2 && W.is_contiguous(),
                  "DSV4 Q8_0 repack requires a contiguous 2D tensor");
  STD_TORCH_CHECK(W.scalar_type() == torch::headeronly::ScalarType::Byte,
                  "DSV4 Q8_0 repack requires byte storage");
  STD_TORCH_CHECK(W.size(1) % int64_t(sizeof(block_q8_0)) == 0,
                  "DSV4 Q8_0 repack received an invalid row stride");
  const int64_t total_blocks = W.numel() / sizeof(block_q8_0);
  const torch::stable::accelerator::DeviceGuard device_guard(
      W.get_device_index());
  auto aligned = torch::stable::empty(W.sizes(), W.scalar_type(), std::nullopt,
                                      W.device());
  slimserve::dsv4_ampere::launch_repack_q8_0_aligned(
      W.data_ptr(), aligned.data_ptr(), total_blocks,
      get_current_cuda_stream());
  return aligned;
}

torch::stable::Tensor ggml_dsv4_mul_mat_vec_aligned_q8_0(
    torch::stable::Tensor W, torch::stable::Tensor X,
    const std::optional<torch::stable::Tensor>& quant_input, int64_t row,
    int64_t rows_per_cta) {
  STD_TORCH_CHECK(W.dim() == 2 && W.is_contiguous() && X.dim() == 2 &&
                      X.is_contiguous(),
                  "DSV4 aligned Q8_0 GEMV requires contiguous 2D inputs");
  STD_TORCH_CHECK(W.scalar_type() == torch::headeronly::ScalarType::Byte,
                  "DSV4 aligned Q8_0 GEMV requires byte weight storage");
  STD_TORCH_CHECK(row == W.size(0) &&
                      W.size(1) % int64_t(sizeof(block_q8_0)) == 0,
                  "DSV4 aligned Q8_0 GEMV weight shape mismatch");
  const int64_t blocks_per_row = W.size(1) / sizeof(block_q8_0);
  const int64_t cols = blocks_per_row * QK8_0;
  const int64_t tokens = X.size(0);
  const int64_t padded = (cols + 511) / 512 * 512;
  STD_TORCH_CHECK(X.size(1) == cols,
                  "DSV4 aligned Q8_0 GEMV activation shape mismatch");

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({tokens, row}, X.scalar_type(), std::nullopt,
                                X.device());
  torch::stable::Tensor quant_X = quant_input
      ? *quant_input
      : torch::stable::empty(
            {tokens, padded / QK8_1 * int64_t(sizeof(block_q8_1)) /
                         int64_t(sizeof(int32_t))},
            torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  STD_TORCH_CHECK(quant_X.dim() == 2 &&
                      quant_X.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      quant_X.size(0) == tokens &&
                      quant_X.size(1) == padded / 32 * 9,
                  "DSV4 aligned Q8_0 GEMV packed activation shape mismatch");

  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_mul_mat_vec_aligned_q8_0", [&] {
        if (!quant_input) {
          quantize_row_q8_1_cuda<scalar_t>(
              static_cast<const scalar_t*>(X.data_ptr()), quant_X.data_ptr(),
              int(cols), int(tokens), stream);
        }
        slimserve::dsv4_ampere::launch_aligned_q8_0_q8_1_gemv<scalar_t>(
            W.data_ptr(), quant_X.data_ptr(),
            static_cast<scalar_t*>(Y.data_ptr()), int(tokens), int(row),
            int(blocks_per_row), int(padded / QK8_1), int(rows_per_cta),
            stream);
      });
  return Y;
}

torch::stable::Tensor ggml_dsv4_shared_gate_up_swiglu(
    torch::stable::Tensor W, torch::stable::Tensor X, double swiglu_limit,
    const std::optional<torch::stable::Tensor>& quant_input) {
  STD_TORCH_CHECK(X.sizes().size() == 2,
                  "ggml_dsv4_shared_gate_up_swiglu: X must be 2D");
  STD_TORCH_CHECK(W.sizes().size() == 2 && W.sizes()[0] % 2 == 0,
                  "ggml_dsv4_shared_gate_up_swiglu: W must have 2*N rows");
  const int64_t tokens = X.sizes()[0];
  const int64_t cols = X.sizes()[1];
  const int64_t intermediate = W.sizes()[0] / 2;
  STD_TORCH_CHECK(cols > 0 && cols % QK8_0 == 0,
                  "ggml_dsv4_shared_gate_up_swiglu: K must be a multiple of 32");
  const int64_t blocks_per_row = cols / QK8_0;
  STD_TORCH_CHECK(W.sizes()[1] == blocks_per_row * int64_t(sizeof(block_q8_0)),
                  "ggml_dsv4_shared_gate_up_swiglu: invalid Q8_0 row stride");

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({tokens, intermediate}, X.scalar_type(),
                                std::nullopt, X.device());
  const int64_t padded = (cols + 511) / 512 * 512;
  torch::stable::Tensor quant_X = quant_input
      ? *quant_input
      : torch::stable::empty(
            {tokens, padded / 32 * 9},
            torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  STD_TORCH_CHECK(quant_X.scalar_type() ==
                      torch::headeronly::ScalarType::Int &&
                      quant_X.size(0) == tokens &&
                      quant_X.size(1) == padded / 32 * 9,
                  "ggml_dsv4_shared_gate_up_swiglu: invalid packed Q8_1 input");
  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_shared_gate_up_swiglu", [&] {
        if (!quant_input) {
          quantize_row_q8_1_cuda<scalar_t>(
              static_cast<const scalar_t*>(X.data_ptr()), quant_X.data_ptr(),
              cols, tokens, stream);
        }
        slimserve::dsv4_ampere::launch_q8_0_gate_up_swiglu<scalar_t>(
            W.data_ptr(), quant_X.data_ptr(),
            static_cast<scalar_t*>(Y.data_ptr()), tokens, blocks_per_row,
            intermediate, padded / QK8_1, float(swiglu_limit),
            stream);
      });
  return Y;
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
ggml_dsv4_shared_gate_up_swiglu_q8_1(
    torch::stable::Tensor W, torch::stable::Tensor X, double swiglu_limit,
    const std::optional<torch::stable::Tensor>& quant_input) {
  STD_TORCH_CHECK(X.dim() == 2 && X.size(0) == 1,
                  "fused shared-expert output packing is decode-only");
  STD_TORCH_CHECK(W.dim() == 2 && W.size(0) % 2 == 0,
                  "shared gate/up weight must have 2*N rows");
  const int64_t cols = X.size(1);
  const int64_t intermediate = W.size(0) / 2;
  STD_TORCH_CHECK(cols > 0 && cols % QK8_0 == 0,
                  "shared gate/up K must be a multiple of 32");
  STD_TORCH_CHECK(intermediate > 0 && intermediate % QK8_1 == 0,
                  "shared intermediate must be a multiple of 32");
  const int64_t blocks_per_row = cols / QK8_0;
  STD_TORCH_CHECK(W.size(1) == blocks_per_row * int64_t(sizeof(block_q8_0)),
                  "shared gate/up Q8_0 row stride is invalid");

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto output = torch::stable::empty({1, intermediate}, X.scalar_type(),
                                     std::nullopt, X.device());
  const int64_t padded_x = (cols + 511) / 512 * 512;
  torch::stable::Tensor quant_x = quant_input
      ? *quant_input
      : torch::stable::empty(
            {1, padded_x / 32 * 9},
            torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  auto quant_output = torch::stable::empty(
      {1, intermediate / 32 * 9}, torch::headeronly::ScalarType::Int,
      std::nullopt, X.device());
  STD_TORCH_CHECK(quant_x.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      quant_x.size(0) == 1 &&
                      quant_x.size(1) == padded_x / 32 * 9,
                  "shared gate/up packed Q8_1 input is invalid");

  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_shared_gate_up_swiglu_q8_1", [&] {
        if (!quant_input) {
          quantize_row_q8_1_cuda<scalar_t>(
              static_cast<const scalar_t*>(X.data_ptr()), quant_x.data_ptr(),
              cols, 1, stream);
        }
        slimserve::dsv4_ampere::launch_q8_0_gate_up_swiglu_q8_1<scalar_t>(
            W.data_ptr(), quant_x.data_ptr(),
            static_cast<scalar_t*>(output.data_ptr()),
            quant_output.data_ptr(), blocks_per_row, intermediate,
            padded_x / QK8_1, float(swiglu_limit), stream);
      });
  return {output, quant_output};
}

std::tuple<torch::stable::Tensor, torch::stable::Tensor>
ggml_dsv4_o_proj_q8_0(
    torch::stable::Tensor W, torch::stable::Tensor O,
    torch::stable::Tensor positions, torch::stable::Tensor cos_sin_cache,
    int64_t local_groups, int64_t rope_dim) {
  STD_TORCH_CHECK(O.dim() == 3 && O.is_contiguous(),
                  "DSV4 Q8 o_proj expects contiguous [tokens, heads, dim]");
  STD_TORCH_CHECK(W.dim() == 2 && W.is_contiguous(),
                  "DSV4 Q8 o_proj expects a contiguous packed matrix");
  STD_TORCH_CHECK(positions.dim() == 1 && positions.is_contiguous(),
                  "DSV4 Q8 o_proj positions must be contiguous");
  STD_TORCH_CHECK(cos_sin_cache.dim() == 2 &&
                      cos_sin_cache.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      cos_sin_cache.is_contiguous(),
                  "DSV4 Q8 o_proj requires contiguous FP32 cos/sin cache");
  STD_TORCH_CHECK(local_groups > 0 && O.size(1) % local_groups == 0,
                  "DSV4 Q8 o_proj head/group shape mismatch");
  const int64_t tokens = O.size(0);
  const int64_t local_heads = O.size(1);
  const int64_t head_dim = O.size(2);
  const int64_t heads_per_group = local_heads / local_groups;
  const int64_t group_dim = heads_per_group * head_dim;
  STD_TORCH_CHECK(group_dim > 0 && group_dim % QK8_1 == 0,
                  "DSV4 Q8 o_proj group width must be Q8 aligned");
  STD_TORCH_CHECK(rope_dim > 0 && rope_dim <= head_dim && rope_dim % 2 == 0 &&
                      cos_sin_cache.size(1) == rope_dim,
                  "DSV4 Q8 o_proj RoPE shape mismatch");
  STD_TORCH_CHECK(W.size(0) % local_groups == 0,
                  "DSV4 Q8 o_proj output rows must divide local groups");
  const int64_t rows_per_group = W.size(0) / local_groups;
  const int64_t blocks_per_row = group_dim / QK8_0;
  STD_TORCH_CHECK(
      W.size(1) == blocks_per_row * int64_t(sizeof(block_q8_0)),
      "DSV4 Q8 o_proj packed row stride mismatch");
  STD_TORCH_CHECK(positions.numel() >= tokens,
                  "DSV4 Q8 o_proj positions are shorter than tokens");
  STD_TORCH_CHECK(positions.scalar_type() ==
                          torch::headeronly::ScalarType::Long ||
                      positions.scalar_type() ==
                          torch::headeronly::ScalarType::Int,
                  "DSV4 Q8 o_proj positions must be int32 or int64");

  const torch::stable::accelerator::DeviceGuard device_guard(
      O.get_device_index());
  auto quant_o = torch::stable::empty(
      {tokens * local_groups, group_dim / 32 * 9},
      torch::headeronly::ScalarType::Int, std::nullopt, O.device());
  auto z = torch::stable::empty(
      {tokens, local_groups * rows_per_group}, O.scalar_type(), std::nullopt,
      O.device());
  auto quant_z = torch::stable::empty(
      {tokens, local_groups * rows_per_group / 32 * 9},
      torch::headeronly::ScalarType::Int, std::nullopt, O.device());
  cudaStream_t stream = get_current_cuda_stream();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      O.scalar_type(), "ggml_dsv4_o_proj_q8_0", [&] {
        if (positions.scalar_type() == torch::headeronly::ScalarType::Long) {
          slimserve::dsv4_ampere::launch_inverse_rope_quant_q8_1<
              scalar_t, int64_t>(
              static_cast<const scalar_t*>(O.data_ptr()),
              static_cast<const int64_t*>(positions.data_ptr()),
              static_cast<const float*>(cos_sin_cache.data_ptr()),
              quant_o.data_ptr(), int(tokens), int(local_groups),
              int(heads_per_group), int(head_dim), int(rope_dim),
              cos_sin_cache.stride(0), stream);
        } else {
          slimserve::dsv4_ampere::launch_inverse_rope_quant_q8_1<
              scalar_t, int32_t>(
              static_cast<const scalar_t*>(O.data_ptr()),
              static_cast<const int32_t*>(positions.data_ptr()),
              static_cast<const float*>(cos_sin_cache.data_ptr()),
              quant_o.data_ptr(), int(tokens), int(local_groups),
              int(heads_per_group), int(head_dim), int(rope_dim),
              cos_sin_cache.stride(0), stream);
        }
        slimserve::dsv4_ampere::launch_grouped_q8_0_q8_1_gemv<scalar_t>(
            W.data_ptr(), quant_o.data_ptr(),
            static_cast<scalar_t*>(z.data_ptr()), int(tokens),
            int(local_groups), int(rows_per_group), int(blocks_per_row),
            stream);
        quantize_row_q8_1_cuda<scalar_t>(
            static_cast<const scalar_t*>(z.data_ptr()), quant_z.data_ptr(),
            int(local_groups * rows_per_group), int(tokens), stream);
      });
  return {z, quant_z};
}

#endif

#ifndef USE_ROCM
// Ampere int8-MMA q8_0 path (mmq_v2). Additive: only q8_0 on sm80+ with a K
// that is a whole number of 256-element iterations routes here, and
// VLLM_GGUF_MMQ_V2=0 forces the legacy kernels.
static bool mmq_v2_env_enabled() {
  static const bool enabled = [] {
    const char* s = std::getenv("VLLM_GGUF_MMQ_V2");
    return s == nullptr || (s[0] != '0' && s[0] != 'f' && s[0] != 'F');
  }();
  return enabled;
}

struct mmq_v2_device_info {
  bool sm80_plus;
  int nsm;
};

static mmq_v2_device_info mmq_v2_get_device_info() {
  int device = 0;
  cudaGetDevice(&device);
  static mmq_v2_device_info info[16] = {};
  static bool init[16] = {};
  if (device < 16 && !init[device]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    info[device].sm80_plus = prop.major >= 8;
    info[device].nsm = prop.multiProcessorCount;
    init[device] = true;
  }
  return device < 16 ? info[device] : mmq_v2_device_info{false, 0};
}

static bool ggml_mul_mat_a8_use_v2(int64_t type, int64_t col, int64_t row) {
  if (type != 8 || !mmq_v2_env_enabled()) {
    return false;
  }
  const mmq_v2_device_info info = mmq_v2_get_device_info();
  return info.sm80_plus && vllm_mmq_v2::mmq_v2_supported(col, row);
}

static torch::stable::Tensor ggml_mul_mat_a8_v2(torch::stable::Tensor W,
                                                torch::stable::Tensor X,
                                                int64_t row, int64_t col,
                                                int64_t batch) {
  const int nsm = mmq_v2_get_device_info().nsm;
  auto Y = torch::stable::empty({batch, row}, X.scalar_type(), std::nullopt,
                                W.device());
  cudaStream_t stream = get_current_cuda_stream();

  auto quant_X = torch::stable::empty({vllm_mmq_v2::mmq_v2_y_ints(batch, col)},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, W.device());

  const bool split = vllm_mmq_v2::mmq_v2_needs_scratch(col, row, batch, nsm);
  auto scratch = torch::stable::empty({split ? batch * row : int64_t(1)},
                                      torch::headeronly::ScalarType::Float,
                                      std::nullopt, W.device());
  if (split) {
    torch::stable::fill_(scratch, 0.0);
  }

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_mul_mat_a8_v2", [&] {
        vllm_mmq_v2::quantize_mmq_q8_1_d4_cuda<scalar_t>(
            (scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(), batch, col,
            stream);
        vllm_mmq_v2::ggml_mul_mat_q8_0_q8_1_v2_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (float*)scratch.data_ptr(), col, row,
            batch, row, nsm, stream);
        if (split) {
          vllm_mmq_v2::mmq_v2_finalize<scalar_t>((float*)scratch.data_ptr(),
                                                 (scalar_t*)Y.data_ptr(),
                                                 batch * row, stream);
        }
      });
  return Y;
}
#endif  // USE_ROCM

torch::stable::Tensor ggml_mul_mat_a8(torch::stable::Tensor W,  // quant weight
                                      torch::stable::Tensor X,  // input
                                      int64_t type, int64_t row) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  int64_t batch = X.sizes()[0];
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
#ifndef USE_ROCM
  if (ggml_mul_mat_a8_use_v2(type, col, row)) {
    return ggml_mul_mat_a8_v2(W, X, row, col, batch);
  }
#endif
  auto Y = torch::stable::empty({batch, row}, X.scalar_type(), std::nullopt,
                                W.device());
#ifdef USE_ROCM
  torch::stable::fill_(Y, 0.0);
#endif
  cudaStream_t stream = get_current_cuda_stream();
  auto quant_X = torch::stable::empty({batch, padded / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, W.device());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_mul_mat_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, batch, stream);

    switch (type) {
      case 2:
        ggml_mul_mat_q4_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 3:
        ggml_mul_mat_q4_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 39:
        ggml_mul_mat_mxfp4_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 6:
        ggml_mul_mat_q5_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 7:
        ggml_mul_mat_q5_1_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 8:
        ggml_mul_mat_q8_0_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 10:
        ggml_mul_mat_q2_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 11:
        ggml_mul_mat_q3_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 12:
        ggml_mul_mat_q4_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 13:
        ggml_mul_mat_q5_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
      case 14:
        ggml_mul_mat_q6_K_q8_1_cuda(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), col, row, batch, padded, row, stream);
        break;
    }
  });
  return Y;
}

#ifndef USE_ROCM
// Tensor-core grouped MXFP4 MoE tiles (dsv4_mxfp4_mmq_ampere.cuh). The gate
// also widens the alignment metadata to the 64-column tile via
// ggml_moe_get_block_size, so both readers must use this one cached value.
static bool mxfp4_mmq_v2_enabled() {
  static const bool enabled = [] {
    const char* s = std::getenv("VLLM_GGUF_MXFP4_MMQ_V2");
    return s == nullptr || s[0] != '0';
  }();
  return enabled;
}
#endif

torch::stable::Tensor ggml_moe_a8(torch::stable::Tensor X,  // input
                                  torch::stable::Tensor W,  // expert weights
                                  torch::stable::Tensor sorted_token_ids,
                                  torch::stable::Tensor expert_ids,
                                  torch::stable::Tensor num_tokens_post_padded,
                                  int64_t type, int64_t row, int64_t top_k,
                                  int64_t tokens) {
  int64_t col = X.sizes()[1];
  int64_t padded = (col + 512 - 1) / 512 * 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({tokens * top_k, row}, X.scalar_type(),
                                std::nullopt, W.device());
#ifdef USE_ROCM
  torch::stable::fill_(Y, 0.0);
#endif
  cudaStream_t stream = get_current_cuda_stream();
  auto quant_X = torch::stable::empty({tokens, padded / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, W.device());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_a8", [&] {
    quantize_row_q8_1_cuda((scalar_t*)X.data_ptr(), (void*)quant_X.data_ptr(),
                           col, tokens, stream);
    switch (type) {
      case 2:
        ggml_moe_q4_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 16:
        ggml_moe_iq2_xxs_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 39:
#ifndef USE_ROCM
        if (mxfp4_mmq_v2_enabled()) {
          // Tile metadata is 64-wide in this mode (ggml_moe_get_block_size).
          if (slimserve::dsv4_ampere::moe_mxfp4_mmq_v2_supported(col)) {
            slimserve::dsv4_ampere::launch_moe_mxfp4_mmq_v2(
                (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
                (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
                (int*)expert_ids.data_ptr(),
                (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
                tokens, padded, row, top_k, sorted_token_ids.sizes()[0],
                /*repacked=*/false, stream);
          } else {
            ggml_moe_mxfp4_w64_q8_1_cuda(
                (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
                (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
                (int*)expert_ids.data_ptr(),
                (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
                tokens, padded, row, top_k, sorted_token_ids.sizes()[0],
                stream);
          }
          break;
        }
#endif
        ggml_moe_mxfp4_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 3:
        ggml_moe_q4_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 6:
        ggml_moe_q5_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 7:
        ggml_moe_q5_1_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 8:
        ggml_moe_q8_0_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 10:
        ggml_moe_q2_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 11:
        ggml_moe_q3_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 12:
        ggml_moe_q4_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 13:
        ggml_moe_q5_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
      case 14:
        ggml_moe_q6_K_q8_1_cuda(
            (void*)quant_X.data_ptr(), (void*)W.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)sorted_token_ids.data_ptr(),
            (int*)expert_ids.data_ptr(),
            (int*)num_tokens_post_padded.data_ptr(), W.stride(0), col, row,
            tokens, padded, row, top_k, sorted_token_ids.sizes()[0], stream);
        break;
    }
  });
  return Y;
}

torch::stable::Tensor ggml_moe_a8_vec(
    torch::stable::Tensor X,  // input
    torch::stable::Tensor W,  // expert weights
    torch::stable::Tensor topk_ids, int64_t top_k, int64_t type, int64_t row,
    int64_t tokens, bool expert_parallel) {
  int64_t col = X.sizes()[1];
  const int64_t padded = (col + 512 - 1) / 512 * 512;
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  auto Y = torch::stable::empty({tokens * top_k, row}, X.scalar_type(),
                                std::nullopt, W.device());
#ifdef USE_ROCM
  torch::stable::fill_(Y, 0.0);
#endif
  cudaStream_t stream = get_current_cuda_stream();
  auto quant_X = torch::stable::empty({tokens, padded / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, W.device());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_moe_vec_a8", [&] {
    quantize_row_q8_1_cuda<scalar_t>((scalar_t*)X.data_ptr(),
                                     (void*)quant_X.data_ptr(), col, tokens,
                                     stream);
    switch (type) {
      case 2:
        moe_vec_q4_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 3:
        moe_vec_q4_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 6:
        moe_vec_q5_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 7:
        moe_vec_q5_1_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 8:
        moe_vec_q8_0_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 39:
        moe_vec_mxfp4_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 10:
        moe_vec_q2_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 11:
        moe_vec_q3_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 12:
        moe_vec_q4_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 13:
        moe_vec_q5_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 14:
        moe_vec_q6_K_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 16:
        moe_vec_iq2_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream, expert_parallel);
        break;
      case 17:
        moe_vec_iq2_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 18:
        moe_vec_iq3_xxs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 19:
        moe_vec_iq1_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 20:
        moe_vec_iq4_nl_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 21:
        moe_vec_iq3_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 22:
        moe_vec_iq2_s_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 23:
        moe_vec_iq4_xs_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
      case 29:
        moe_vec_iq1_m_q8_1_cuda<scalar_t>(
            (void*)W.data_ptr(), (void*)quant_X.data_ptr(),
            (scalar_t*)Y.data_ptr(), (int*)topk_ids.data_ptr(), top_k, tokens,
            col, row, quant_X.stride(0), stream);
        break;
    }
  });
  return Y;
}

torch::stable::Tensor ggml_dsv4_moe_w1_a8(
    torch::stable::Tensor X, torch::stable::Tensor W1,
    torch::stable::Tensor topk_weights, torch::stable::Tensor topk_ids,
    torch::stable::Tensor sorted_token_ids,
    torch::stable::Tensor w1_expert_ids,
    torch::stable::Tensor num_tokens_post_padded, int64_t intermediate,
    int64_t top_k, int64_t tokens, double swiglu_limit, bool w1_repacked,
    const std::optional<torch::stable::Tensor>& quant_input) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false,
                  "ggml_dsv4_moe_w1_a8 is a CUDA-only DSV4 fast path");
#else
  STD_TORCH_CHECK(X.dim() == 2 && X.size(0) == tokens,
                  "DSV4 output-owned W1 input shape mismatch");
  STD_TORCH_CHECK(W1.dim() == 3 && W1.size(1) == 2 * intermediate,
                  "DSV4 output-owned W1 weight shape mismatch");
  STD_TORCH_CHECK(
      topk_weights.scalar_type() == torch::headeronly::ScalarType::Float &&
          topk_weights.sizes() == topk_ids.sizes() &&
          topk_weights.size(0) == tokens && topk_weights.size(1) == top_k &&
          topk_weights.is_contiguous(),
      "DSV4 output-owned W1 route weight shape mismatch");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_ids.is_contiguous(),
      "DSV4 output-owned W1 route ids must be contiguous int32");
  STD_TORCH_CHECK(
      sorted_token_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          w1_expert_ids.scalar_type() ==
              torch::headeronly::ScalarType::Int &&
          num_tokens_post_padded.scalar_type() ==
              torch::headeronly::ScalarType::Int,
      "DSV4 output-owned W1 alignment metadata must be int32");

  const int64_t hidden = X.size(1);
  const int64_t padded_x = (hidden + 511) / 512 * 512;
  const int64_t padded_mid = (intermediate + 511) / 512 * 512;
  const int64_t routed_rows = tokens * top_k;
  const int experts = static_cast<int>(W1.size(0));
  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  auto quant_mid = torch::stable::empty(
      {routed_rows, padded_mid / QK8_1 * 9},
      torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  torch::stable::Tensor quant_x = quant_input
      ? *quant_input
      : torch::stable::empty(
            {tokens, padded_x / QK8_1 * 9},
            torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  STD_TORCH_CHECK(
      quant_x.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_x.dim() == 2 && quant_x.size(0) == tokens &&
          quant_x.size(1) == padded_x / QK8_1 * 9 &&
          quant_x.is_contiguous(),
      "DSV4 output-owned W1 packed input shape mismatch");
  if (!quant_input) {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "ggml_dsv4_moe_w1_quant", [&] {
          quantize_row_q8_1_cuda<scalar_t>(
              static_cast<const scalar_t*>(X.const_data_ptr()),
              quant_x.mutable_data_ptr(), static_cast<int>(hidden),
              static_cast<int>(tokens), stream);
        });
  }

  const bool direct = tokens <= 8 && (top_k == 6 || top_k == 8) &&
                      intermediate % QK8_1 == 0 && hidden % QK_K == 0;
  if (direct) {
    slimserve::dsv4_ampere::launch_iq2_xxs_gate_up_swiglu_q8_1_decode(
        W1.const_data_ptr(), quant_x.const_data_ptr(),
        quant_mid.mutable_data_ptr(),
        reinterpret_cast<int*>(topk_ids.mutable_data_ptr()),
        reinterpret_cast<const float*>(topk_weights.const_data_ptr()),
        W1.stride(0), static_cast<int>(hidden),
        static_cast<int>(intermediate), static_cast<int>(tokens),
        static_cast<int>(top_k), experts, static_cast<float>(swiglu_limit),
        w1_repacked, nullptr, nullptr, 0, stream);
  } else {
    slimserve::dsv4_ampere::launch_iq2_xxs_gate_up_swiglu_q8_1(
        W1.const_data_ptr(), quant_x.const_data_ptr(),
        quant_mid.mutable_data_ptr(),
        reinterpret_cast<int*>(sorted_token_ids.mutable_data_ptr()),
        reinterpret_cast<int*>(w1_expert_ids.mutable_data_ptr()),
        reinterpret_cast<const float*>(topk_weights.const_data_ptr()),
        reinterpret_cast<int*>(num_tokens_post_padded.mutable_data_ptr()),
        W1.stride(0), static_cast<int>(hidden), static_cast<int>(padded_x),
        static_cast<int>(intermediate), static_cast<int>(padded_mid),
        static_cast<int>(tokens), static_cast<int>(top_k),
        static_cast<int>(sorted_token_ids.size(0)),
        static_cast<float>(swiglu_limit), true, w1_repacked, stream);
  }
  return quant_mid;
#endif
}

torch::stable::Tensor ggml_dsv4_moe_down_output_owned(
    torch::stable::Tensor W2, torch::stable::Tensor quant_mid,
    torch::stable::Tensor topk_ids, int64_t tokens, int64_t top_k) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(
      false,
      "ggml_dsv4_moe_down_output_owned is a CUDA-only DSV4 fast path");
#else
  STD_TORCH_CHECK(W2.dim() == 3 && W2.is_contiguous(),
                  "DSV4 output-owned W2 must be a contiguous expert tensor");
  STD_TORCH_CHECK(
      quant_mid.scalar_type() == torch::headeronly::ScalarType::Int &&
          quant_mid.dim() == 2 && quant_mid.size(0) == tokens * top_k &&
          quant_mid.is_contiguous(),
      "DSV4 output-owned W2 packed activation shape mismatch");
  STD_TORCH_CHECK(
      topk_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          topk_ids.dim() == 2 && topk_ids.size(0) == tokens &&
          topk_ids.size(1) == top_k && topk_ids.is_contiguous(),
      "DSV4 output-owned W2 route ids must be contiguous int32");
  STD_TORCH_CHECK(top_k == 6 || top_k == 8,
                  "DSV4 output-owned W2 supports top-k 6 or 8");
  const int64_t intermediate = quant_mid.size(1) / 9 * QK8_1;
  STD_TORCH_CHECK(
      intermediate == 2048 &&
          W2.size(2) == intermediate / QK_K * sizeof(block_q2_K),
      "DSV4 output-owned W2 requires full-K Q2_K weights and Q8_1 input");
  const int64_t local_rows = W2.size(1);
  const torch::stable::accelerator::DeviceGuard device_guard(
      W2.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();
  auto output = torch::stable::empty(
      {tokens, local_rows}, torch::headeronly::ScalarType::BFloat16,
      std::nullopt, W2.device());
  slimserve::dsv4_ampere::launch_q2_k_down_sum_repacked<nv_bfloat16>(
      W2.const_data_ptr(), quant_mid.const_data_ptr(),
      reinterpret_cast<int*>(topk_ids.mutable_data_ptr()),
      reinterpret_cast<nv_bfloat16*>(output.mutable_data_ptr()), W2.stride(0),
      static_cast<int>(intermediate), static_cast<int>(local_rows),
      static_cast<int>(tokens), static_cast<int>(top_k),
      static_cast<int>(W2.size(0)), stream);
  return output;
#endif
}

torch::stable::Tensor ggml_dsv4_moe_a8(
    torch::stable::Tensor X,   // [tokens, hidden]
    torch::stable::Tensor W1,  // [experts, 2 * intermediate, packed IQ2_XXS]
    torch::stable::Tensor W2,  // [experts, out_row, packed Q2_K]
    torch::stable::Tensor topk_weights,
    torch::stable::Tensor topk_ids,
    torch::stable::Tensor sorted_token_ids,
    torch::stable::Tensor w1_expert_ids,
    torch::stable::Tensor w2_expert_ids,
    torch::stable::Tensor num_tokens_post_padded, int64_t intermediate,
    int64_t out_row, int64_t top_k, int64_t tokens, double swiglu_limit,
    bool w1_repacked,
    bool w2_repacked,
    const std::optional<torch::stable::Tensor>& quant_input,
    bool defer_down) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_moe_a8 is a CUDA-only DSV4 fast path");
#else
  STD_TORCH_CHECK(X.dim() == 2, "ggml_dsv4_moe_a8: X must be 2D");
  STD_TORCH_CHECK(W1.dim() == 3 && W2.dim() == 3,
                  "ggml_dsv4_moe_a8: W1/W2 must be 3D expert tensors");
  STD_TORCH_CHECK(topk_weights.dim() == 2,
                  "ggml_dsv4_moe_a8: topk_weights must be 2D");
  STD_TORCH_CHECK(topk_weights.scalar_type() ==
                      torch::headeronly::ScalarType::Float,
                  "ggml_dsv4_moe_a8: topk_weights must be float32");
  STD_TORCH_CHECK(topk_weights.is_contiguous(),
                  "ggml_dsv4_moe_a8: topk_weights must be contiguous");
  STD_TORCH_CHECK(topk_ids.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "ggml_dsv4_moe_a8: topk_ids must be int32");
  STD_TORCH_CHECK(topk_ids.is_contiguous(),
                  "ggml_dsv4_moe_a8: topk_ids must be contiguous");
  STD_TORCH_CHECK(topk_weights.size(0) == tokens &&
                      topk_weights.size(1) == top_k,
                  "ggml_dsv4_moe_a8: topk_weights shape mismatch");
  STD_TORCH_CHECK(W1.size(1) == 2 * intermediate,
                  "ggml_dsv4_moe_a8: W1 must contain combined gate/up rows");
  STD_TORCH_CHECK(W2.size(1) == out_row,
                  "ggml_dsv4_moe_a8: W2 output rows mismatch");
  STD_TORCH_CHECK(W1.size(0) == W2.size(0),
                  "ggml_dsv4_moe_a8: W1/W2 expert count mismatch");
  STD_TORCH_CHECK(sorted_token_ids.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "ggml_dsv4_moe_a8: sorted_token_ids must be int32");
  STD_TORCH_CHECK(w1_expert_ids.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "ggml_dsv4_moe_a8: w1_expert_ids must be int32");
  STD_TORCH_CHECK(w2_expert_ids.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "ggml_dsv4_moe_a8: w2_expert_ids must be int32");
  STD_TORCH_CHECK(num_tokens_post_padded.scalar_type() ==
                      torch::headeronly::ScalarType::Int,
                  "ggml_dsv4_moe_a8: num_tokens_post_padded must be int32");

  const int64_t col = X.sizes()[1];
  const int64_t padded_x = (col + 512 - 1) / 512 * 512;
  const int64_t routed_rows = tokens * top_k;
  const int64_t padded_mid = (intermediate + 512 - 1) / 512 * 512;
  const int experts = (int)W1.size(0);
  const int tokens_post_padded = (int)sorted_token_ids.size(0);

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();

  static const bool dsv4_raw_down_sum_enabled = [] {
    const char* value = std::getenv("VLLM_GGUF_DSV4_DOWN_SUM");
    return value != nullptr &&
           (value[0] == 'r' || value[0] == 'R');
  }();
  static const bool dsv4_w1_decode_enabled = [] {
    const char* value = std::getenv("VLLM_GGUF_DSV4_W1_DECODE");
    return value == nullptr || value[0] != '0';
  }();
  static const bool dsv4_w1_cooperative_enabled = [] {
    const char* value = std::getenv("VLLM_DSV4_W1_COOPERATIVE");
    return value != nullptr && value[0] == '1';
  }();
  constexpr int64_t q2_k_block_bytes = sizeof(block_q2_K);
  const int64_t w2_intermediate =
      W2.size(2) / q2_k_block_bytes * int64_t(QK_K);
  const bool channel_owned_down =
      w2_repacked && w2_intermediate == 2048 &&
      ((intermediate == 512 && out_row == 1024) ||
       (intermediate == 1024 && out_row == 2048));
  STD_TORCH_CHECK(
      !w2_repacked || w2_intermediate == intermediate || channel_owned_down,
      "repacked DSV4 Q2_K layout must be a local-K partial or a full-K "
      "output-row shard");
  const bool raw_fused_down = dsv4_raw_down_sum_enabled && tokens <= 8 &&
                              (top_k == 6 || top_k == 8);
  const bool fused_down = w2_repacked || raw_fused_down;
  const int64_t pending_payload_bytes =
      routed_rows * (padded_mid / QK8_1) * int64_t(sizeof(block_q8_1));
  const int64_t output_storage_rows = channel_owned_down ? 4096 : out_row;
  auto out = torch::stable::empty({tokens, output_storage_rows}, X.scalar_type(),
                                  std::nullopt, X.device());
  const int64_t output_bytes = out.numel() * out.element_size();
  const bool pending_down =
      defer_down && dsv4_w1_decode_enabled && w2_repacked && tokens == 1 &&
      top_k == 6 && intermediate % 32 == 0 && col % QK_K == 0 &&
      // The custom-allreduce-owned Q2_K producer is built and validated for
      // the 512/1024 shards; the 256 (TP8) shard uses the plain fused down.
      (intermediate == 512 || intermediate == 1024) &&
      ((out_row == 4096 && w2_intermediate == intermediate) ||
       channel_owned_down) &&
      pending_payload_bytes +
              int64_t(sizeof(slimserve::dsv4_ampere::PendingQ2Header)) <=
          output_bytes;
  STD_TORCH_CHECK(
      !channel_owned_down || pending_down,
      "DSV4 output-row-sharded W2 is decode-only and requires deferred down");
  auto down = torch::stable::empty(
      {fused_down ? 1 : routed_rows * out_row},
      torch::headeronly::ScalarType::Float, std::nullopt, X.device());
  torch::stable::Tensor quant_x = quant_input
      ? *quant_input
      : torch::stable::empty(
            {tokens, padded_x / 32 * 9},
            torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  STD_TORCH_CHECK(quant_x.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      quant_x.dim() == 2 && quant_x.size(0) == tokens &&
                      quant_x.size(1) == padded_x / 32 * 9 &&
                      quant_x.is_contiguous(),
                  "ggml_dsv4_moe_a8: invalid packed Q8_1 input");
  auto quant_mid = torch::stable::empty(
      {pending_down ? 0 : routed_rows, padded_mid / 32 * 9},
      torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  void* quant_mid_ptr =
      pending_down ? out.mutable_data_ptr() : quant_mid.mutable_data_ptr();
  auto* pending_header = pending_down
      ? reinterpret_cast<slimserve::dsv4_ampere::PendingQ2Header*>(
            static_cast<uint8_t*>(out.mutable_data_ptr()) + output_bytes -
            sizeof(slimserve::dsv4_ampere::PendingQ2Header))
      : nullptr;

  if (!quant_input) {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "ggml_dsv4_moe_x", [&] {
          quantize_row_q8_1_cuda<scalar_t>(
              (const scalar_t*)X.data_ptr(), (void*)quant_x.data_ptr(),
              (int)col, (int)tokens, stream);
        });
  }
  const bool direct_w1 = dsv4_w1_decode_enabled && fused_down && tokens <= 8 &&
                         (top_k == 6 || top_k == 8) &&
                         intermediate % 32 == 0 && col % QK_K == 0;
  if (direct_w1) {
    const bool cooperative_w1 = dsv4_w1_cooperative_enabled && tokens == 1 &&
                                top_k == 6 && intermediate == 512;
    if (cooperative_w1) {
      auto w1_scratch = torch::stable::empty(
          {routed_rows, intermediate},
          torch::headeronly::ScalarType::Float, std::nullopt, X.device());
      slimserve::dsv4_ampere::
          launch_iq2_xxs_gate_up_swiglu_q8_1_decode_cooperative(
              (void*)W1.data_ptr(), (void*)quant_x.data_ptr(), quant_mid_ptr,
              (float*)w1_scratch.mutable_data_ptr(),
              (int*)topk_ids.data_ptr(),
              (const float*)topk_weights.data_ptr(), W1.stride(0), (int)col,
              (int)intermediate, (int)tokens, (int)top_k, experts,
              (float)swiglu_limit, w1_repacked, pending_header,
              W2.const_data_ptr(), W2.stride(0), stream);
    } else {
      slimserve::dsv4_ampere::launch_iq2_xxs_gate_up_swiglu_q8_1_decode(
          (void*)W1.data_ptr(), (void*)quant_x.data_ptr(), quant_mid_ptr,
          (int*)topk_ids.data_ptr(),
          (const float*)topk_weights.data_ptr(), W1.stride(0), (int)col,
          (int)intermediate, (int)tokens, (int)top_k, experts,
          (float)swiglu_limit, w1_repacked, pending_header,
          W2.const_data_ptr(), W2.stride(0), stream);
    }
  } else {
    slimserve::dsv4_ampere::launch_iq2_xxs_gate_up_swiglu_q8_1(
        (void*)W1.data_ptr(), (void*)quant_x.data_ptr(),
        quant_mid_ptr, (int*)sorted_token_ids.data_ptr(),
        (int*)w1_expert_ids.data_ptr(),
        (const float*)topk_weights.data_ptr(),
        (int*)num_tokens_post_padded.data_ptr(), W1.stride(0), (int)col,
        (int)padded_x, (int)intermediate, (int)padded_mid, (int)tokens,
        (int)top_k, tokens_post_padded, (float)swiglu_limit, fused_down,
        w1_repacked, stream);
  }
  if (pending_down) return out;
  if (w2_repacked) {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "ggml_dsv4_moe_down_repacked", [&] {
          slimserve::dsv4_ampere::launch_q2_k_down_sum_repacked<scalar_t>(
              (void*)W2.data_ptr(), quant_mid_ptr,
              (int*)topk_ids.data_ptr(), (scalar_t*)out.data_ptr(),
              W2.stride(0), (int)intermediate, (int)out_row, (int)tokens,
              (int)top_k, experts, stream);
        });
  } else if (raw_fused_down) {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ggml_dsv4_moe_down", [&] {
      slimserve::dsv4_ampere::launch_q2_k_down_weighted_sum<scalar_t>(
          (void*)W2.data_ptr(), quant_mid_ptr,
          (int*)topk_ids.data_ptr(), (scalar_t*)out.data_ptr(), W2.stride(0),
          (int)intermediate, (int)padded_mid, (int)out_row, (int)tokens,
          (int)top_k, experts, stream);
    });
  } else {
    ggml_moe_q2_K_q8_1_cuda<float>(
        (void*)quant_mid.data_ptr(), (void*)W2.data_ptr(),
        (float*)down.data_ptr(), (int*)sorted_token_ids.data_ptr(),
        (int*)w2_expert_ids.data_ptr(),
        (int*)num_tokens_post_padded.data_ptr(), W2.stride(0),
        (int)intermediate, (int)out_row, (int)routed_rows, (int)padded_mid,
        (int)out_row, /*top_k=*/1, tokens_post_padded, stream);
    const int64_t out_values = tokens * out_row;
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "ggml_dsv4_moe_sum", [&] {
          dsv4_slot_sum_kernel<scalar_t, float>
              <<<(out_values + 255) / 256, 256, 0, stream>>>(
                  (const float*)down.data_ptr(),
                  (const float*)topk_weights.data_ptr(),
                  (scalar_t*)out.data_ptr(), out_values, (int)out_row,
                  (int)top_k, topk_weights.stride(0), topk_weights.stride(1));
        });
  }
  return out;
#endif
}

torch::stable::Tensor ggml_dsv4_repack_q2_k(
    torch::stable::Tensor W2, int64_t intermediate) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_repack_q2_k is CUDA-only");
#else
  STD_TORCH_CHECK(W2.dim() == 3, "DSV4 W2 must be a 3D expert tensor");
  STD_TORCH_CHECK(intermediate > 0 && intermediate % QK_K == 0,
                  "DSV4 Q2_K intermediate must be a multiple of 256");
  const int experts = (int)W2.size(0);
  const int rows = (int)W2.size(1);
  STD_TORCH_CHECK(W2.size(2) == intermediate / QK_K * sizeof(block_q2_K),
                  "DSV4 W2 packed width does not match Q2_K intermediate");
  auto output = torch::stable::empty(W2.sizes(), W2.scalar_type(),
                                     std::nullopt, W2.device());
  const torch::stable::accelerator::DeviceGuard device_guard(
      W2.get_device_index());
  slimserve::dsv4_ampere::launch_repack_q2_k_experts(
      W2.data_ptr(), output.data_ptr(), experts, rows, (int)intermediate,
      W2.stride(0), get_current_cuda_stream());
  return output;
#endif
}

torch::stable::Tensor ggml_dsv4_repack_iq2_xxs(
    torch::stable::Tensor W1, int64_t hidden) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_repack_iq2_xxs is CUDA-only");
#else
  STD_TORCH_CHECK(W1.dim() == 3, "DSV4 W1 must be a 3D expert tensor");
  STD_TORCH_CHECK(hidden > 0 && hidden % QK_K == 0,
                  "DSV4 IQ2_XXS hidden size must be a multiple of 256");
  const int experts = (int)W1.size(0);
  const int rows = (int)W1.size(1);
  STD_TORCH_CHECK(
      W1.size(2) == hidden / QK_K * sizeof(block_iq2_xxs),
      "DSV4 W1 packed width does not match IQ2_XXS hidden size");
  const int64_t blocks_per_expert = int64_t(rows) * hidden / QK_K;
  STD_TORCH_CHECK((blocks_per_expert * sizeof(half)) % alignof(uint2) == 0,
                  "DSV4 IQ2_XXS scale plane must preserve code alignment");
  auto output = torch::stable::empty(W1.sizes(), W1.scalar_type(),
                                     std::nullopt, W1.device());
  const torch::stable::accelerator::DeviceGuard device_guard(
      W1.get_device_index());
  slimserve::dsv4_ampere::launch_repack_iq2_xxs_experts(
      W1.data_ptr(), output.data_ptr(), experts, rows, (int)hidden,
      W1.stride(0), get_current_cuda_stream());
  return output;
#endif
}

torch::stable::Tensor ggml_dsv4_repack_mxfp4(torch::stable::Tensor W,
                                             int64_t values_per_row) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_repack_mxfp4 is CUDA-only");
#else
  STD_TORCH_CHECK(W.dim() == 3, "DSV4 MXFP4 weight must be a 3D expert tensor");
  STD_TORCH_CHECK(values_per_row > 0 && values_per_row % 32 == 0,
                  "DSV4 MXFP4 row length must be a multiple of 32");
  const int experts = (int)W.size(0);
  const int rows = (int)W.size(1);
  STD_TORCH_CHECK(W.size(2) == values_per_row / 32 * 17,
                  "DSV4 MXFP4 packed width does not match row length");
  const int64_t nblocks = int64_t(rows) * values_per_row / 32;
  STD_TORCH_CHECK(nblocks % 16 == 0,
                  "DSV4 MXFP4 scale plane must preserve code alignment");
  auto output = torch::stable::empty(W.sizes(), W.scalar_type(), std::nullopt,
                                     W.device());
  const torch::stable::accelerator::DeviceGuard device_guard(
      W.get_device_index());
  slimserve::dsv4_ampere::launch_repack_mxfp4_experts(
      W.data_ptr(), output.mutable_data_ptr(), experts, nblocks, W.stride(0),
      get_current_cuda_stream());
  return output;
#endif
}

torch::stable::Tensor ggml_dsv4_moe_a8_mxfp4(
    torch::stable::Tensor X,   // [tokens, hidden]
    torch::stable::Tensor W1,  // [experts, 2 * intermediate, packed MXFP4]
    torch::stable::Tensor W2,  // [experts, out_row, packed MXFP4]
    torch::stable::Tensor topk_weights, torch::stable::Tensor topk_ids,
    int64_t intermediate, int64_t out_row, int64_t top_k, int64_t tokens,
    double swiglu_limit, bool w1_repacked, bool w2_repacked,
    const std::optional<torch::stable::Tensor>& quant_input) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_moe_a8_mxfp4 is a CUDA-only fast path");
#else
  STD_TORCH_CHECK(X.dim() == 2, "ggml_dsv4_moe_a8_mxfp4: X must be 2D");
  STD_TORCH_CHECK(W1.dim() == 3 && W2.dim() == 3,
                  "ggml_dsv4_moe_a8_mxfp4: W1/W2 must be 3D expert tensors");
  STD_TORCH_CHECK(topk_weights.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      topk_weights.is_contiguous(),
                  "ggml_dsv4_moe_a8_mxfp4: topk_weights must be contiguous "
                  "float32");
  STD_TORCH_CHECK(topk_ids.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      topk_ids.is_contiguous(),
                  "ggml_dsv4_moe_a8_mxfp4: topk_ids must be contiguous int32");
  STD_TORCH_CHECK(top_k == 6 || top_k == 8,
                  "ggml_dsv4_moe_a8_mxfp4: top_k must be 6 or 8");
  // The per-route warp-GEMV shape stays profitable well past decode widths
  // because DSV4 routing is near-uncorrelated (little tile reuse for MMQ);
  // the Python dispatch picks the crossover. 256 routes x 128 output CTAs is
  // a sane upper bound for the fp32 mid buffers this op allocates.
  STD_TORCH_CHECK(tokens >= 1 && tokens <= 256,
                  "ggml_dsv4_moe_a8_mxfp4: tokens out of range");
  STD_TORCH_CHECK(W1.size(1) == 2 * intermediate,
                  "ggml_dsv4_moe_a8_mxfp4: W1 must combine gate/up rows");
  STD_TORCH_CHECK(W2.size(1) == out_row,
                  "ggml_dsv4_moe_a8_mxfp4: W2 output rows mismatch");
  STD_TORCH_CHECK(W1.size(0) == W2.size(0),
                  "ggml_dsv4_moe_a8_mxfp4: expert count mismatch");
  STD_TORCH_CHECK(intermediate % 32 == 0,
                  "ggml_dsv4_moe_a8_mxfp4: intermediate must be 32-aligned");

  const int64_t col = X.sizes()[1];
  STD_TORCH_CHECK(col % QK8_1 == 0,
                  "ggml_dsv4_moe_a8_mxfp4: hidden must be 32-aligned");
  const int64_t padded_x = (col + 512 - 1) / 512 * 512;
  const int64_t routed_rows = tokens * top_k;
  const int experts = (int)W1.size(0);

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();

  auto out = torch::stable::empty({tokens, out_row}, X.scalar_type(),
                                  std::nullopt, X.device());
  torch::stable::Tensor quant_x = quant_input
      ? *quant_input
      : torch::stable::empty(
            {tokens, padded_x / 32 * 9},
            torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  STD_TORCH_CHECK(quant_x.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      quant_x.dim() == 2 && quant_x.size(0) == tokens &&
                      quant_x.size(1) == padded_x / 32 * 9 &&
                      quant_x.is_contiguous(),
                  "ggml_dsv4_moe_a8_mxfp4: invalid packed Q8_1 input");
  auto quant_mid = torch::stable::empty(
      {routed_rows, intermediate / 32 * 9},
      torch::headeronly::ScalarType::Int, std::nullopt, X.device());

  if (!quant_input) {
    VLLM_STABLE_DISPATCH_FLOATING_TYPES(
        X.scalar_type(), "ggml_dsv4_moe_a8_mxfp4_x", [&] {
          quantize_row_q8_1_cuda<scalar_t>(
              (const scalar_t*)X.data_ptr(), (void*)quant_x.data_ptr(),
              (int)col, (int)tokens, stream);
        });
  }
  slimserve::dsv4_ampere::launch_mxfp4_gate_up_swiglu_q8_1_decode(
      (const void*)W1.data_ptr(), (const void*)quant_x.data_ptr(),
      (void*)quant_mid.mutable_data_ptr(), (const int*)topk_ids.data_ptr(),
      (const float*)topk_weights.data_ptr(), W1.stride(0), (int)col,
      (int)(padded_x / QK8_1), (int)intermediate, (int)tokens, (int)top_k,
      experts, (float)swiglu_limit, w1_repacked, stream);
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_moe_a8_mxfp4_down", [&] {
        slimserve::dsv4_ampere::launch_mxfp4_down_sum<scalar_t>(
            (const void*)W2.data_ptr(), (const void*)quant_mid.data_ptr(),
            (const int*)topk_ids.data_ptr(),
            (scalar_t*)out.mutable_data_ptr(), W2.stride(0), (int)intermediate,
            (int)out_row, (int)tokens, (int)top_k, experts, w2_repacked,
            stream);
      });
  return out;
#endif
}

// Segmented (permutation-based) wide MXFP4 MoE: device-side route grouping,
// tensor-core W1 with fused SwiGLU+route-weight+Q8_1 epilogue, tensor-core
// W2, deterministic per-token reduce. No moe_align metadata, static grids --
// see dsv4_mxfp4_seg_ampere.cuh for the design provenance.
torch::stable::Tensor ggml_dsv4_moe_a8_mxfp4_seg(
    torch::stable::Tensor X,   // [tokens, hidden]
    torch::stable::Tensor W1,  // [experts, 2 * intermediate, packed MXFP4]
    torch::stable::Tensor W2,  // [experts, out_row, packed MXFP4]
    torch::stable::Tensor topk_weights, torch::stable::Tensor topk_ids,
    int64_t intermediate, int64_t out_row, int64_t top_k, int64_t tokens,
    double swiglu_limit) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_moe_a8_mxfp4_seg is a CUDA-only path");
#else
  STD_TORCH_CHECK(X.dim() == 2, "mxfp4_seg: X must be 2D");
  STD_TORCH_CHECK(W1.dim() == 3 && W2.dim() == 3,
                  "mxfp4_seg: W1/W2 must be 3D expert tensors");
  STD_TORCH_CHECK(topk_weights.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      topk_weights.is_contiguous(),
                  "mxfp4_seg: topk_weights must be contiguous float32");
  STD_TORCH_CHECK(topk_ids.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      topk_ids.is_contiguous(),
                  "mxfp4_seg: topk_ids must be contiguous int32");
  STD_TORCH_CHECK(W1.size(1) == 2 * intermediate,
                  "mxfp4_seg: W1 must combine gate/up rows");
  STD_TORCH_CHECK(W2.size(1) == out_row, "mxfp4_seg: W2 rows mismatch");
  STD_TORCH_CHECK(W1.size(0) == W2.size(0) &&
                      W1.size(0) <= slimserve::dsv4_ampere::SEG_MAX_EXPERTS,
                  "mxfp4_seg: expert count invalid");
  const int64_t col = X.sizes()[1];
  STD_TORCH_CHECK(col % 256 == 0 && intermediate % 256 == 0 &&
                      out_row % 128 == 0 && intermediate % 64 == 0,
                  "mxfp4_seg: unsupported shape");

  const int64_t padded_x = (col + 512 - 1) / 512 * 512;
  const int64_t routes = tokens * top_k;
  const int experts = (int)W1.size(0);

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();

  auto out = torch::stable::empty({tokens, out_row}, X.scalar_type(),
                                  std::nullopt, X.device());
  auto quant_x = torch::stable::empty({tokens, padded_x / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, X.device());
  auto quant_mid = torch::stable::empty(
      {routes, intermediate / 32 * 9}, torch::headeronly::ScalarType::Int,
      std::nullopt, X.device());
  auto meta = torch::stable::empty(
      {slimserve::dsv4_ampere::seg_meta_ints(experts, (int)routes)},
      torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  auto w2out = torch::stable::empty({routes, out_row}, X.scalar_type(),
                                    std::nullopt, X.device());

  static const int j16_rows = [] {
    const char* s = std::getenv("VLLM_GGUF_DSV4_SEG_J16_ROWS");
    return s ? std::atoi(s) : 1536;
  }();
  const bool use_j16 = routes < j16_rows;

  int* meta_ptr = (int*)meta.mutable_data_ptr();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_moe_a8_mxfp4_seg", [&] {
        quantize_row_q8_1_cuda<scalar_t>(
            (const scalar_t*)X.data_ptr(), (void*)quant_x.mutable_data_ptr(),
            (int)col, (int)tokens, stream);
        slimserve::dsv4_ampere::launch_moe_mxfp4_seg<scalar_t>(
            (const void*)quant_x.data_ptr(), (const void*)W1.data_ptr(),
            (const void*)W2.data_ptr(), (void*)quant_mid.mutable_data_ptr(),
            (scalar_t*)w2out.mutable_data_ptr(),
            (scalar_t*)out.mutable_data_ptr(), (const int*)topk_ids.data_ptr(),
            (const float*)topk_weights.data_ptr(), meta_ptr, W1.stride(0),
            W2.stride(0), (int)col, (int)(padded_x / 32), (int)intermediate,
            (int)out_row, (int)tokens, (int)top_k, experts,
            (float)swiglu_limit, use_j16, stream);
      });
  return out;
#endif
}

// Segmented wide pipeline for the hybrid (IQ2_XXS gate/up, Q2_K down)
// expert pair. Both weights must be in their load-time repacked layouts
// (paired IQ2 planes; three-plane Q2_K) -- the production A100 state.
torch::stable::Tensor ggml_dsv4_moe_a8_iq2_seg(
    torch::stable::Tensor X,   // [tokens, hidden]
    torch::stable::Tensor W1,  // [experts, 2 * intermediate, iq2 packed]
    torch::stable::Tensor W2,  // [experts, out_row, q2k packed]
    torch::stable::Tensor topk_weights, torch::stable::Tensor topk_ids,
    int64_t intermediate, int64_t out_row, int64_t top_k, int64_t tokens,
    double swiglu_limit) {
#ifdef USE_ROCM
  STD_TORCH_CHECK(false, "ggml_dsv4_moe_a8_iq2_seg is a CUDA-only path");
#else
  STD_TORCH_CHECK(X.dim() == 2, "iq2_seg: X must be 2D");
  STD_TORCH_CHECK(W1.dim() == 3 && W2.dim() == 3,
                  "iq2_seg: W1/W2 must be 3D expert tensors");
  STD_TORCH_CHECK(topk_weights.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      topk_weights.is_contiguous(),
                  "iq2_seg: topk_weights must be contiguous float32");
  STD_TORCH_CHECK(topk_ids.scalar_type() ==
                          torch::headeronly::ScalarType::Int &&
                      topk_ids.is_contiguous(),
                  "iq2_seg: topk_ids must be contiguous int32");
  STD_TORCH_CHECK(W1.size(1) == 2 * intermediate,
                  "iq2_seg: W1 must combine gate/up rows");
  STD_TORCH_CHECK(W2.size(1) == out_row, "iq2_seg: W2 rows mismatch");
  STD_TORCH_CHECK(W1.size(0) == W2.size(0) &&
                      W1.size(0) <= slimserve::dsv4_ampere::SEG_MAX_EXPERTS,
                  "iq2_seg: expert count invalid");
  const int64_t col = X.sizes()[1];
  STD_TORCH_CHECK(col % 256 == 0 && intermediate % 256 == 0 &&
                      out_row % 128 == 0,
                  "iq2_seg: unsupported shape");

  const int64_t padded_x = (col + 512 - 1) / 512 * 512;
  const int64_t routes = tokens * top_k;
  const int experts = (int)W1.size(0);

  const torch::stable::accelerator::DeviceGuard device_guard(
      X.get_device_index());
  cudaStream_t stream = get_current_cuda_stream();

  auto out = torch::stable::empty({tokens, out_row}, X.scalar_type(),
                                  std::nullopt, X.device());
  auto quant_x = torch::stable::empty({tokens, padded_x / 32 * 9},
                                      torch::headeronly::ScalarType::Int,
                                      std::nullopt, X.device());
  auto quant_mid = torch::stable::empty(
      {routes, intermediate / 32 * 9}, torch::headeronly::ScalarType::Int,
      std::nullopt, X.device());
  auto meta = torch::stable::empty(
      {slimserve::dsv4_ampere::seg_meta_ints(experts, (int)routes)},
      torch::headeronly::ScalarType::Int, std::nullopt, X.device());
  auto w2out = torch::stable::empty({routes, out_row}, X.scalar_type(),
                                    std::nullopt, X.device());

  static const int j16_rows = [] {
    const char* s = std::getenv("VLLM_GGUF_DSV4_SEG_J16_ROWS");
    return s ? std::atoi(s) : 1536;
  }();
  const bool use_j16 = routes < j16_rows;

  int* meta_ptr = (int*)meta.mutable_data_ptr();
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      X.scalar_type(), "ggml_dsv4_moe_a8_iq2_seg", [&] {
        quantize_row_q8_1_cuda<scalar_t>(
            (const scalar_t*)X.data_ptr(), (void*)quant_x.mutable_data_ptr(),
            (int)col, (int)tokens, stream);
        slimserve::dsv4_ampere::launch_moe_iq2_seg<scalar_t>(
            (const void*)quant_x.data_ptr(), (const void*)W1.data_ptr(),
            (const void*)W2.data_ptr(), (void*)quant_mid.mutable_data_ptr(),
            (scalar_t*)w2out.mutable_data_ptr(),
            (scalar_t*)out.mutable_data_ptr(), (const int*)topk_ids.data_ptr(),
            (const float*)topk_weights.data_ptr(), meta_ptr, W1.stride(0),
            W2.stride(0), (int)col, (int)(padded_x / 32), (int)intermediate,
            (int)out_row, (int)tokens, (int)top_k, experts,
            (float)swiglu_limit, use_j16, stream);
      });
  return out;
#endif
}

int64_t ggml_moe_get_block_size(int64_t type) {
  switch (type) {
    case 2:
      return MOE_X_Q4_0;
    case 3:
      return MOE_X_Q4_1;
    case 6:
      return MOE_X_Q5_0;
    case 7:
      return MOE_X_Q5_1;
    case 8:
      return MOE_X_Q8_0;
    case 16:
      return MOE_X_IQ2_XXS;
    case 39:
#ifndef USE_ROCM
      // The tensor-core grouped tile is 64 routed columns wide; the dp4a
      // fallback for unsupported K is instantiated at the same width so the
      // expert_ids metadata stays consistent (see ggml_moe_a8 case 39).
      if (mxfp4_mmq_v2_enabled()) {
        return slimserve::dsv4_ampere::MXMMQ_J;
      }
#endif
      return MOE_X_MXFP4;
    case 10:
      return MOE_X_Q2_K;
    case 11:
      return MOE_X_Q3_K;
    case 12:
      return MOE_X_Q4_K;
    case 13:
      return MOE_X_Q5_K;
    case 14:
      return MOE_X_Q6_K;
  }
  return 0;
}
