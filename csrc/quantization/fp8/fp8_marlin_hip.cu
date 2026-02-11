// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include "quantization/fp8/fp8_marlin_hip_kernel.hip"
#include "quantization/fp8/fp8_marlin_mfma_hip_kernel.hip"

namespace {

bool is_cdna3_arch(const std::string& arch) {
  return arch.rfind("gfx94", 0) == 0 || arch.rfind("gfx95", 0) == 0;
}

bool is_cdna_arch(const std::string& arch) {
  return arch.rfind("gfx90", 0) == 0 || is_cdna3_arch(arch);
}

}  // namespace

torch::Tensor fp8_marlin_gemm(torch::Tensor& a, torch::Tensor& b_q_weight,
                              torch::Tensor& b_scales, torch::Tensor& workspace,
                              int64_t num_bits, bool fp8_is_fnuz,
                              int64_t size_m, int64_t size_n, int64_t size_k) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string arch = dprops->gcnArchName;
  TORCH_CHECK(is_cdna_arch(arch),
              "fp8_marlin_gemm (ROCm) requires CDNA-class GPU, got ",
              arch);

  TORCH_CHECK(num_bits == 8, "num_bits must be 8. Got = ", num_bits);
  int pack_factor = 32 / num_bits;

  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);

  TORCH_CHECK(size_k % kTileSize == 0, "size_k = ", size_k,
              " is not divisible by tile_size = ", kTileSize);
  TORCH_CHECK((size_k / kTileSize) == b_q_weight.size(0),
              "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0),
              ", size_k = ", size_k, ", tile_size = ", kTileSize);
  TORCH_CHECK(b_q_weight.size(1) % kTileSize == 0,
              "b_q_weight.size(1) = ", b_q_weight.size(1),
              " is not divisible by tile_size = ", kTileSize);
  int actual_size_n = (b_q_weight.size(1) / kTileSize) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  TORCH_CHECK(b_scales.scalar_type() == a.scalar_type(),
              "b_scales dtype = ", b_scales.scalar_type(),
              " does not match a dtype = ", a.scalar_type());

  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  torch::Tensor c = torch::empty({size_m, size_n}, options);

  int thread_k = -1;
  int thread_n = -1;
  int sms = -1;

  int num_groups = -1;
  int group_size = -1;
  int group_blocks = -1;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 2, "b_scales rank = ", b_rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ",
              b_scales.size(1), " is not size_n = ", size_n);
  num_groups = b_scales.size(0);
  TORCH_CHECK(num_groups >= 1, "b_scales dim 0 = ", num_groups,
              " is invalid");
  if (num_groups == 1) {
    group_size = -1;
    group_blocks = -1;
  } else {
    TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                " is not divisible by num_groups = ", num_groups);
    group_size = size_k / num_groups;
    TORCH_CHECK(group_size % kTileSize == 0, "group_size = ", group_size,
                " is not divisible by tile_size = ", kTileSize);
    group_blocks = group_size / kTileSize;
    TORCH_CHECK(group_blocks == 2 || group_blocks == 4 || group_blocks == 8 ||
                    group_blocks == 16,
                "fp8_marlin_gemm only supports group_size {32, 64, 128, 256}; got ",
                group_size);
  }

  TORCH_CHECK(size_n % kMinThreadN == 0, "size_n = ", size_n,
              ", is not divisible by min_thread_n = ", kMinThreadN);
  int min_workspace_size = (size_n / kMinThreadN) * kMaxPar;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(dev);
  hipStream_t hip_stream = stream;

  // Zero workspace (barrier locks) to ensure clean state for K-reduction.
  // hipMemsetAsync is CUDA-graph-capturable.
  (void)hipMemsetAsync(workspace.data_ptr(), 0,
                       min_workspace_size * sizeof(int), hip_stream);

  int err = 0;
  if (a.scalar_type() == at::ScalarType::Half) {
    err = fp8_marlin_hip_gemm<__half>(
        a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
        b_scales.data_ptr(), static_cast<int>(size_m),
        static_cast<int>(size_n), static_cast<int>(size_k),
        workspace.data_ptr(), dev, hip_stream, fp8_is_fnuz, thread_k, thread_n,
        sms,
        group_blocks, kMaxPar);
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    err = fp8_marlin_hip_gemm<__hip_bfloat16>(
        a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
        b_scales.data_ptr(), static_cast<int>(size_m),
        static_cast<int>(size_n), static_cast<int>(size_k),
        workspace.data_ptr(), dev, hip_stream, fp8_is_fnuz, thread_k, thread_n,
        sms,
        group_blocks, kMaxPar);
  } else {
    TORCH_CHECK(false, "fp8_marlin_gemm only supports bfloat16 and float16");
  }

  if (err == ERR_PROB_SHAPE) {
    TORCH_CHECK(false, "Problem (m=", size_m, ", n=", size_n, ", k=", size_k,
                ") not compatible with thread_k=", thread_k, ", thread_n=",
                thread_n, ".");
  } else if (err == ERR_KERN_SHAPE) {
    TORCH_CHECK(false, "No kernel implementation for thread_k=", thread_k,
                ", thread_n=", thread_n, ", groupsize=", group_size, ".");
  }

  return c;
}

torch::Tensor fp8_mfma_marlin_gemm(
    torch::Tensor& a,           // FP8 [M, K] (W8A8) or BF16/FP16 [M, K] (W8A16)
    torch::Tensor& b_q_weight,  // MFMA-tiled FP8 weights [K/16, N*4]
    torch::Tensor& b_scales,    // Weight scales [num_groups, N]
    std::optional<torch::Tensor> a_scales,  // Per-token FP32 scales [M] (W8A8) or None (W8A16)
    bool fp8_is_fnuz,
    int64_t size_m, int64_t size_n, int64_t size_k) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string arch = dprops->gcnArchName;
  TORCH_CHECK(is_cdna3_arch(arch),
              "fp8_mfma_marlin_gemm requires CDNA3+ GPU (gfx94x/gfx95x), got ",
              arch);

  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);
  TORCH_CHECK(size_k % 32 == 0, "size_k = ", size_k,
              " must be divisible by 32 for MFMA kernel");

  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  // Compute group_blocks from b_scales shape.
  int num_groups = b_scales.size(0);
  int group_blocks = -1;
  if (num_groups > 1) {
    TORCH_CHECK(size_k % num_groups == 0);
    int group_size = size_k / num_groups;
    TORCH_CHECK(group_size % kTileSize == 0);
    group_blocks = group_size / kTileSize;
    TORCH_CHECK(group_blocks == 2 || group_blocks == 4 ||
                group_blocks == 8 || group_blocks == 16,
                "Unsupported group_size ", group_size);
  }

  int dev = a.get_device();
  int sms = dprops->multiProcessorCount;
  auto stream = at::cuda::getCurrentCUDAStream(dev);
  hipStream_t hip_stream = stream;

  int err = 0;

  if (a_scales.has_value()) {
    // W8A8: FP8 activations with per-token scales.
    auto& as = a_scales.value();
    TORCH_CHECK(as.device().is_cuda(), "a_scales is not on GPU");
    TORCH_CHECK(as.scalar_type() == at::ScalarType::Float,
                "a_scales must be float32");
    TORCH_CHECK(as.numel() >= size_m, "a_scales.numel = ", as.numel(),
                " is less than size_m = ", size_m);

    auto out_dtype = b_scales.dtype();
    auto options = torch::TensorOptions().dtype(out_dtype).device(a.device());
    torch::Tensor c = torch::empty({size_m, size_n}, options);

    if (out_dtype == at::ScalarType::Half) {
      err = dense_fp8_mfma::dense_fp8_mfma_gemm<__half>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), as.data_ptr(),
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      err = dense_fp8_mfma::dense_fp8_mfma_gemm<__hip_bfloat16>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), as.data_ptr(),
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else {
      TORCH_CHECK(false,
                  "fp8_mfma_marlin_gemm W8A8 only supports float16/bfloat16");
    }

    if (err == ERR_PROB_SHAPE) {
      TORCH_CHECK(false, "W8A8 MFMA: problem (m=", size_m, ", n=", size_n,
                  ", k=", size_k, ") incompatible.");
    } else if (err == ERR_KERN_SHAPE) {
      TORCH_CHECK(false, "W8A8 MFMA: no kernel for this configuration.");
    }
    return c;

  } else {
    // W8A16: BF16/FP16 activations, FP8 weights decoded via FP16 MFMA.
    auto out_dtype = b_scales.dtype();
    TORCH_CHECK(a.scalar_type() == out_dtype,
                "W8A16 MFMA: a dtype (", a.scalar_type(),
                ") must match b_scales dtype (", out_dtype, ")");

    auto options = torch::TensorOptions().dtype(out_dtype).device(a.device());
    torch::Tensor c = torch::empty({size_m, size_n}, options);

    if (out_dtype == at::ScalarType::Half) {
      err = dense_fp8_mfma::dense_w8a16_mfma_gemm<__half>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), fp8_is_fnuz,
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      err = dense_fp8_mfma::dense_w8a16_mfma_gemm<__hip_bfloat16>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), fp8_is_fnuz,
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else {
      TORCH_CHECK(false,
                  "fp8_mfma_marlin_gemm W8A16 only supports float16/bfloat16");
    }

    if (err == ERR_PROB_SHAPE) {
      TORCH_CHECK(false, "W8A16 MFMA: problem (m=", size_m, ", n=", size_n,
                  ", k=", size_k, ") incompatible.");
    } else if (err == ERR_KERN_SHAPE) {
      TORCH_CHECK(false, "W8A16 MFMA: no kernel for this configuration.");
    }
    return c;
  }
}

torch::Tensor int4_mfma_marlin_gemm(
    torch::Tensor& a,           // FP8 [M, K] (W4A8) or BF16/FP16 [M, K] (W4A16)
    torch::Tensor& b_q_weight,  // MFMA-tiled INT4
    torch::Tensor& b_scales,    // Weight scales [num_groups, N]
    std::optional<torch::Tensor> a_scales,  // Per-token FP32 [M] (W4A8) or None (W4A16)
    int64_t size_m, int64_t size_n, int64_t size_k) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string arch = dprops->gcnArchName;
  TORCH_CHECK(is_cdna3_arch(arch),
              "int4_mfma_marlin_gemm requires CDNA3+ GPU (gfx94x/gfx95x), got ",
              arch);

  TORCH_CHECK(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0),
              ", size_m = ", size_m);
  TORCH_CHECK(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1),
              ", size_k = ", size_k);
  TORCH_CHECK(size_k % 32 == 0, "size_k = ", size_k,
              " must be divisible by 32 for MFMA kernel");
  TORCH_CHECK(size_n % 16 == 0, "size_n = ", size_n,
              " must be divisible by 16 for MFMA kernel");

  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.is_contiguous(), "A is not contiguous");
  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");

  // Compute group_blocks from b_scales shape.
  int num_groups = b_scales.size(0);
  int group_blocks = -1;
  if (num_groups > 1) {
    TORCH_CHECK(size_k % num_groups == 0);
    int group_size = size_k / num_groups;
    TORCH_CHECK(group_size % kTileSize == 0);
    group_blocks = group_size / kTileSize;
    TORCH_CHECK(group_blocks == 2 || group_blocks == 4 ||
                group_blocks == 8,
                "Unsupported group_size ", group_size,
                " (group_blocks=", group_blocks, ")");
  }

  int dev = a.get_device();
  int sms = dprops->multiProcessorCount;
  auto stream = at::cuda::getCurrentCUDAStream(dev);
  hipStream_t hip_stream = stream;

  int err = 0;

  if (a_scales.has_value()) {
    // W4A8: FP8 activations with per-token scales, INT4 weights decoded to FP8.
    auto& as = a_scales.value();
    TORCH_CHECK(as.device().is_cuda(), "a_scales is not on GPU");
    TORCH_CHECK(as.scalar_type() == at::ScalarType::Float,
                "a_scales must be float32");
    TORCH_CHECK(as.numel() >= size_m, "a_scales.numel = ", as.numel(),
                " is less than size_m = ", size_m);

    auto out_dtype = b_scales.dtype();
    auto options = torch::TensorOptions().dtype(out_dtype).device(a.device());
    torch::Tensor c = torch::empty({size_m, size_n}, options);

    if (out_dtype == at::ScalarType::Half) {
      err = dense_fp8_mfma::dense_w4a8_mfma_gemm<__half>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), as.data_ptr(),
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else if (out_dtype == at::ScalarType::BFloat16) {
      err = dense_fp8_mfma::dense_w4a8_mfma_gemm<__hip_bfloat16>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), as.data_ptr(),
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else {
      TORCH_CHECK(false,
                  "int4_mfma_marlin_gemm W4A8 only supports float16/bfloat16");
    }

    if (err == ERR_PROB_SHAPE) {
      TORCH_CHECK(false, "W4A8 MFMA: problem (m=", size_m, ", n=", size_n,
                  ", k=", size_k, ") incompatible.");
    } else if (err == ERR_KERN_SHAPE) {
      TORCH_CHECK(false, "W4A8 MFMA: no kernel for this configuration.");
    }
    return c;

  } else {
    // W4A16: BF16/FP16 activations, INT4 weights via FP16 MFMA.
    auto out_dtype = a.dtype();
    TORCH_CHECK(out_dtype == at::ScalarType::Half ||
                out_dtype == at::ScalarType::BFloat16,
                "int4_mfma_marlin_gemm W4A16 only supports float16/bfloat16 activations");

    auto options = torch::TensorOptions().dtype(out_dtype).device(a.device());
    torch::Tensor c = torch::empty({size_m, size_n}, options);

    if (out_dtype == at::ScalarType::Half) {
      err = dense_fp8_mfma::dense_w4a16_mfma_gemm<__half>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(),
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    } else {
      err = dense_fp8_mfma::dense_w4a16_mfma_gemm<__hip_bfloat16>(
          a.data_ptr(), b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(),
          static_cast<int>(size_m), static_cast<int>(size_n),
          static_cast<int>(size_k), sms, hip_stream, group_blocks);
    }

    if (err == ERR_PROB_SHAPE) {
      TORCH_CHECK(false, "W4A16 MFMA: problem (m=", size_m, ", n=", size_n,
                  ", k=", size_k, ") incompatible.");
    } else if (err == ERR_KERN_SHAPE) {
      TORCH_CHECK(false, "W4A16 MFMA: no kernel for this configuration.");
    }
    return c;
  }
}
