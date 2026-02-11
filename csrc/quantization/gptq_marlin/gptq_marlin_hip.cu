// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <optional>
#include <cstdint>

#include "core/scalar_type.hpp"
#include "quantization/fp8/fp8_marlin_hip_kernel.hip"
#include "quantization/gptq_marlin/gptq_marlin_hip_kernel.hip"
#include "quantization/gptq_marlin/marlin_constants.h"

namespace {

bool is_cdna3_arch(const std::string& arch) {
  return arch.rfind("gfx94", 0) == 0 || arch.rfind("gfx95", 0) == 0;
}

bool is_cdna_arch(const std::string& arch) {
  return arch.rfind("gfx90", 0) == 0 || is_cdna3_arch(arch);
}

template <typename T>
__global__ void permute_cols_kernel(const T* __restrict__ a_ptr,
                                    const int* __restrict__ perm_ptr,
                                    T* __restrict__ out_ptr, int size_m,
                                    int size_k, int lda, int block_rows) {
  int start_row = block_rows * blockIdx.x;
  int finish_row = start_row + block_rows;
  if (finish_row > size_m) {
    finish_row = size_m;
  }
  int cur_rows = finish_row - start_row;
  for (int i = 0; i < cur_rows; ++i) {
    int row = start_row + i;
    const T* a_row = a_ptr + row * lda;
    T* out_row = out_ptr + row * size_k;
    int iters = size_k / blockDim.x;
    int rest = size_k % blockDim.x;
    int base_k = 0;
    for (int iter = 0; iter < iters; ++iter) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_ptr[cur_k];
      out_row[cur_k] = a_row[src_pos];
      base_k += blockDim.x;
    }
    if (rest && threadIdx.x < rest) {
      int cur_k = base_k + threadIdx.x;
      int src_pos = perm_ptr[cur_k];
      out_row[cur_k] = a_row[src_pos];
    }
  }
}

}  // namespace

torch::Tensor gptq_marlin_gemm(
    torch::Tensor& a, std::optional<torch::Tensor> c_or_none,
    torch::Tensor& b_q_weight,
    std::optional<torch::Tensor> const& b_bias_or_none,
    torch::Tensor& b_scales,
    std::optional<torch::Tensor> const& a_scales,
    std::optional<torch::Tensor> const& global_scale,
    std::optional<torch::Tensor> const& b_zeros_or_none,
    std::optional<torch::Tensor> const& g_idx_or_none,
    std::optional<torch::Tensor> const& perm_or_none,
    torch::Tensor& workspace,
    vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool is_k_full,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto dprops = at::cuda::getCurrentDeviceProperties();
  std::string arch = dprops->gcnArchName;
  TORCH_CHECK(is_cdna_arch(arch),
              "gptq_marlin_gemm (ROCm) requires CDNA-class GPU, got ",
              arch);

  bool is_fp8 = b_type_id == vllm::kFE4M3fn.id();
  bool is_fp4 = b_type_id == vllm::kFE2M1f.id();
  int num_bits = 0;
  if (b_type_id == vllm::kU4B8.id() || b_type_id == vllm::kS4.id() ||
      b_type_id == vllm::kU4.id()) {
    num_bits = 4;
  } else if (b_type_id == vllm::kU8B128.id() || b_type_id == vllm::kU8.id()) {
    num_bits = 8;
  } else if (is_fp8) {
    num_bits = 8;
  } else if (is_fp4) {
    num_bits = 4;
  } else {
    TORCH_CHECK(false,
                "gptq_marlin_gemm (ROCm) currently supports int4/int8/FP8/FP4 "
                "weights.");
  }

  auto has_nonempty = [](const std::optional<torch::Tensor>& t) {
    return t.has_value() && t.value().numel() > 0;
  };

  torch::Tensor b_zeros;
  if (b_zeros_or_none.has_value()) {
    b_zeros = b_zeros_or_none.value();
    TORCH_CHECK(b_zeros.device().is_cuda(), "b_zeros is not on GPU");
    TORCH_CHECK(b_zeros.is_contiguous(), "b_zeros is not contiguous");
  } else {
    b_zeros = torch::empty({0}, a.options());
  }

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
  int pack_factor = 32 / num_bits;
  int actual_size_n = (b_q_weight.size(1) / kTileSize) * pack_factor;
  TORCH_CHECK(size_n == actual_size_n, "size_n = ", size_n,
              ", actual_size_n = ", actual_size_n);

  TORCH_CHECK(a.device().is_cuda(), "A is not on GPU");
  TORCH_CHECK(a.stride(1) == 1, "A.stride(1) is not 1");
  TORCH_CHECK(a.stride(0) % 8 == 0, "A.stride(0) must divisible by 8");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0,
              "A must aligned to 16 bytes");

  TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
  TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  TORCH_CHECK(b_scales.device().is_cuda(), "b_scales is not on GPU");
  TORCH_CHECK(b_scales.is_contiguous(), "b_scales is not contiguous");
  vllm::ScalarTypeId a_type_id;
  vllm::ScalarTypeId c_type_id;
  vllm::ScalarTypeId s_type_id;
  auto c_dtype = a.dtype();
  if (a.scalar_type() == at::ScalarType::Half) {
    a_type_id = vllm::kFloat16.id();
    c_type_id = vllm::kFloat16.id();
  } else if (a.scalar_type() == at::ScalarType::BFloat16) {
    a_type_id = vllm::kBFloat16.id();
    c_type_id = vllm::kBFloat16.id();
  } else {
    c_dtype = b_scales.dtype();
    if (b_scales.scalar_type() == at::ScalarType::Half) {
      c_type_id = vllm::kFloat16.id();
    } else if (b_scales.scalar_type() == at::ScalarType::BFloat16) {
      c_type_id = vllm::kBFloat16.id();
    } else {
      c_type_id = vllm::kBFloat16.id();
      TORCH_CHECK(c_or_none.has_value(),
                  "c must be passed for W4A8-FP4");
      torch::Tensor c = c_or_none.value();
      c_dtype = c.dtype();
      if (c.scalar_type() == at::ScalarType::Half) {
        c_type_id = vllm::kFloat16.id();
      } else if (c.scalar_type() == at::ScalarType::BFloat16) {
        c_type_id = vllm::kBFloat16.id();
      } else {
        TORCH_CHECK(false, "unsupported c dtype");
      }
    }

    if (a.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      a_type_id = vllm::kFE4M3fn.id();
    } else if (a.scalar_type() == at::ScalarType::Char) {
      a_type_id = vllm::kS8.id();
    } else {
      TORCH_CHECK(false, "unsupported `a` scalar_type");
    }
  }

  s_type_id = c_type_id;
  if (b_type_id == vllm::kFE2M1f.id()) {
    if (b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn) {
      s_type_id = vllm::kFE4M3fn.id();
    } else if (b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu) {
      s_type_id = vllm::kFE8M0fnu.id();
    } else {
      TORCH_CHECK(false,
                  "When b_type = float4_e2m1f, b_scale scalar type must be "
                  "float8_e4m3fn (for NVFP4) or float8_e8m0fnu (for MXFP4).");
    }
  }

  vllm::ScalarType a_type = vllm::ScalarType::from_id(a_type_id);
  vllm::ScalarType b_type = vllm::ScalarType::from_id(b_type_id);
  vllm::ScalarType c_type = vllm::ScalarType::from_id(c_type_id);
  vllm::ScalarType s_type = vllm::ScalarType::from_id(s_type_id);
  (void)c_type;
  (void)s_type;

  if (!is_fp4) {
    TORCH_CHECK(b_scales.scalar_type() == c_dtype,
                "b_scales dtype = ", b_scales.scalar_type(),
                " does not match c dtype = ", c_dtype);
  }

  auto options = torch::TensorOptions().dtype(c_dtype).device(a.device());
  torch::Tensor c = c_or_none.has_value() ? c_or_none.value()
                                          : torch::empty({size_m, size_n}, options);

  int thread_k = -1;
  int thread_n = -1;
  int sms = dprops->multiProcessorCount;

  int group_size = -1;
  int group_blocks = -1;

  int b_rank = b_scales.sizes().size();
  TORCH_CHECK(b_rank == 2, "b_scales rank = ", b_rank, " is not 2");
  TORCH_CHECK(b_scales.size(1) == size_n, "b_scales dim 1 = ",
              b_scales.size(1), " is not size_n = ", size_n);
  int num_groups = b_scales.size(0);
  TORCH_CHECK(num_groups >= 1, "b_scales dim 0 = ", num_groups,
              " is invalid");

  torch::Tensor g_idx, perm, a_tmp;
  if (g_idx_or_none.has_value() && perm_or_none.has_value()) {
    g_idx = g_idx_or_none.value();
    perm = perm_or_none.value();
    TORCH_CHECK(g_idx.device().is_cuda(), "g_idx is not on GPU");
    TORCH_CHECK(g_idx.is_contiguous(), "g_idx is not contiguous");
    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");
    TORCH_CHECK((g_idx.size(-1) == 0 && perm.size(-1) == 0) ||
                    (g_idx.size(-1) == size_k && perm.size(-1) == size_k),
                "Unexpected g_idx.size(-1) = ", g_idx.size(-1),
                " and perm.size(-1) = ", perm.size(-1),
                ", where size_k = ", size_k);
  } else {
    g_idx = torch::empty({0}, options);
    perm = torch::empty({0}, options);
    a_tmp = torch::empty({0}, options);
  }
  bool has_act_order = g_idx.size(-1) > 0 && perm.size(-1) > 0;

  if (has_act_order) {
    TORCH_CHECK(is_k_full,
                "gptq_marlin_gemm (ROCm) act_order requires is_k_full=true.");
    TORCH_CHECK(a.element_size() == 2,
                "gptq_marlin_gemm (ROCm) act_order only supports 16-bit activations.");
    a_tmp = torch::empty({size_m, size_k}, options);
    if (is_k_full) {
      TORCH_CHECK(num_groups > 1, "For act_order, num_groups must be > 1");
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  ", is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = 0;
    }
  } else {
    if (num_groups > 1) {
      TORCH_CHECK(size_k % num_groups == 0, "size_k = ", size_k,
                  " is not divisible by num_groups = ", num_groups);
      group_size = size_k / num_groups;
    } else {
      group_size = -1;
    }
  }

  if (group_size == -1) {
    group_blocks = -1;
  } else if (group_size > 0) {
    TORCH_CHECK(group_size % kTileSize == 0, "group_size = ", group_size,
                " is not divisible by tile_size = ", kTileSize);
    group_blocks = group_size / kTileSize;
  }

  bool has_zp = b_zeros.size(-1) > 0;
  if (has_zp) {
    TORCH_CHECK(b_type == vllm::kU4 || b_type == vllm::kU8,
                "b_type must be u4 or u8 when has_zp = True. Got = ",
                b_type.str());
  } else {
    TORCH_CHECK(b_type == vllm::kU4B8 || b_type == vllm::kU8B128 ||
                    b_type == vllm::kS4 || b_type == vllm::kS8 ||
                    b_type == vllm::kFE4M3fn || b_type == vllm::kFE2M1f,
                "b_type must be uint4b8, uint8b128, int4, int8, "
                "float8_e4m3fn or float4_e2m1f when has_zp = False. Got = ",
                b_type.str());
  }
  if (has_zp && is_zp_float) {
    TORCH_CHECK(a.scalar_type() == at::ScalarType::Half,
                "Computation type must be float16 (half) when using float zero "
                "points.");
  }

  if (has_zp) {
    int rank = b_zeros.sizes().size();
    TORCH_CHECK(rank == 2, "b_zeros rank = ", rank, " is not 2");
    if (is_zp_float) {
      TORCH_CHECK(b_zeros.size(1) == size_n,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not size_n = ", size_n);
      TORCH_CHECK(num_groups == b_zeros.size(0),
                  "b_zeros dim 0 = ", b_zeros.size(0),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(num_groups != -1, "num_groups must be != -1");
    } else {
      TORCH_CHECK(b_zeros.size(0) == num_groups,
                  "b_zeros dim 0 = ", b_zeros.size(0),
                  " is not num_groups = ", num_groups);
      TORCH_CHECK(b_zeros.size(1) == size_n / pack_factor,
                  "b_zeros dim 1 = ", b_zeros.size(1),
                  " is not size_n / pack_factor = ", size_n / pack_factor);
    }
  }

  bool has_global_scale = has_nonempty(global_scale);
  bool scale_is_e4m3 = false;
  bool scale_is_e8m0 = false;
  if (is_fp4) {
    scale_is_e4m3 =
        b_scales.scalar_type() == at::ScalarType::Float8_e4m3fn;
    scale_is_e8m0 =
        b_scales.scalar_type() == at::ScalarType::Float8_e8m0fnu;
    TORCH_CHECK(scale_is_e4m3 || scale_is_e8m0,
                "gptq_marlin_gemm (ROCm) fp4 scales must be float8_e4m3fn "
                "(nvfp4) or float8_e8m0fnu (mxfp4). Got ",
                b_scales.scalar_type());
    TORCH_CHECK(group_blocks == (scale_is_e4m3 ? 1 : 2),
                "gptq_marlin_gemm (ROCm) fp4 expects group_size ",
                (scale_is_e4m3 ? 16 : 32), ", got group_size ", group_size);
    if (scale_is_e4m3) {
      TORCH_CHECK(has_global_scale,
                  "gptq_marlin_gemm (ROCm) requires global_scale for nvfp4.");
      TORCH_CHECK(global_scale.value().numel() > 0,
                  "gptq_marlin_gemm (ROCm) global_scale is empty.");
      TORCH_CHECK(global_scale.value().scalar_type() == c.scalar_type(),
                  "gptq_marlin_gemm (ROCm) global_scale dtype = ",
                  global_scale.value().scalar_type(),
                  " does not match c dtype = ", c.scalar_type());
    } else {
      TORCH_CHECK(!has_global_scale,
                  "gptq_marlin_gemm (ROCm) global_scale is only valid for nvfp4.");
    }
  } else {
    TORCH_CHECK(!has_global_scale,
                "gptq_marlin_gemm (ROCm) does not support global_scale for this dtype.");
    if (group_blocks != -1) {
      TORCH_CHECK(group_blocks == 2 || group_blocks == 4 || group_blocks == 8 ||
                      group_blocks == 16,
                  "gptq_marlin_gemm (ROCm) only supports group_size {32, 64, 128, 256} "
                  "for int/fp8, got ", group_size);
    }
  }

  torch::Tensor a_scales_tensor;
  auto options_fp32 = torch::TensorOptions().dtype(at::kFloat).device(a.device());
  if (a_scales.has_value()) {
    a_scales_tensor = a_scales.value();
    TORCH_CHECK(a_scales_tensor.device().is_cuda(),
                "a_scales is not on GPU");
    TORCH_CHECK(a_scales_tensor.is_contiguous(),
                "a_scales is not contiguous");
    TORCH_CHECK(a_type.size_bits() == 8,
                "a_scales can only be used for 8bit activation.");
  } else {
    a_scales_tensor = torch::empty({0}, options_fp32);
    TORCH_CHECK(a_type.size_bits() != 8,
                "the a_scales parameter must be passed for 8bit activation.");
  }

  TORCH_CHECK(a_scales_tensor.scalar_type() == at::ScalarType::Float,
              "scalar type of a_scales must be float");
  if (a_type.size_bits() == 16) {
    TORCH_CHECK(a.scalar_type() == c.scalar_type(),
                "scalar type of a must be the same with c for 16 bit activation");
  }

  TORCH_CHECK(size_n % kMinThreadN == 0, "size_n = ", size_n,
              ", is not divisible by min_thread_n = ", kMinThreadN);
  int min_workspace_size = sms;
  TORCH_CHECK(workspace.numel() >= min_workspace_size,
              "workspace.numel = ", workspace.numel(),
              " is below min_workspace_size = ", min_workspace_size);

  int dev = a.get_device();
  auto stream = at::cuda::getCurrentCUDAStream(dev);
  hipStream_t hip_stream = stream;

  const void* A_ptr = a.data_ptr();
  int lda = a.stride(0);
  if (has_act_order) {
    int block_rows = (static_cast<int>(size_m) + sms - 1) / sms;
    if (a.element_size() == 2) {
      permute_cols_kernel<uint16_t><<<sms, kDefaultThreads, 0, hip_stream>>>(
          reinterpret_cast<const uint16_t*>(A_ptr),
          perm.data_ptr<int>(),
          reinterpret_cast<uint16_t*>(a_tmp.data_ptr()),
          static_cast<int>(size_m),
          static_cast<int>(size_k),
          lda,
          block_rows);
    } else {
      permute_cols_kernel<uint8_t><<<sms, kDefaultThreads, 0, hip_stream>>>(
          reinterpret_cast<const uint8_t*>(A_ptr),
          perm.data_ptr<int>(),
          reinterpret_cast<uint8_t*>(a_tmp.data_ptr()),
          static_cast<int>(size_m),
          static_cast<int>(size_k),
          lda,
          block_rows);
    }
    A_ptr = a_tmp.data_ptr();
    lda = static_cast<int>(size_k);
    if (is_k_full) {
      has_act_order = false;
    }
  }

  const void* global_scale_ptr = nullptr;
  if (has_global_scale) {
    global_scale_ptr = global_scale.value().data_ptr();
  }

  bool a_is_fp8 = a.scalar_type() == at::ScalarType::Float8_e4m3fn;
  bool a_fp8_is_fnuz = is_cdna3_arch(arch);
  const float* a_scales_ptr =
      a_scales_tensor.numel() > 0 ? a_scales_tensor.data_ptr<float>() : nullptr;
  bool has_bias = b_bias_or_none.has_value();
  const void* b_bias_ptr =
      has_bias ? b_bias_or_none.value().data_ptr() : nullptr;
  const int* g_idx_ptr = has_act_order ? g_idx.data_ptr<int>() : nullptr;
  const void* b_zeros_ptr = has_zp ? b_zeros.data_ptr() : nullptr;
  if (has_bias) {
    TORCH_CHECK(b_bias_or_none.value().device().is_cuda(),
                "b_bias is not on GPU");
    TORCH_CHECK(b_bias_or_none.value().is_contiguous(),
                "b_bias is not contiguous");
    TORCH_CHECK(b_bias_or_none.value().size(0) == size_n,
                "b_bias.size(0) != size_n");
    TORCH_CHECK(b_bias_or_none.value().stride(0) == 1,
                "b_bias.stride(0) != 1");
  }

  int err = 0;
  if (c_type_id == vllm::kFloat16.id()) {
    if (is_fp8) {
      bool fp8_is_fnuz = arch.rfind("gfx94", 0) == 0;
      err = fp8_marlin_hip_gemm<__half>(
          A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), static_cast<int>(size_m),
          static_cast<int>(size_n), static_cast<int>(size_k),
          workspace.data_ptr(), dev, hip_stream, fp8_is_fnuz, thread_k,
          thread_n, sms,
          group_blocks, kMaxPar);
    } else if (is_fp4) {
      if (scale_is_e4m3) {
        err = fp4_marlin_hip_gemm<__half, true>(
            A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
            b_bias_ptr, a_scales_ptr, b_scales.data_ptr(), global_scale_ptr,
            b_zeros_ptr,
            static_cast<int>(size_m),
            static_cast<int>(size_n), static_cast<int>(size_k),
            workspace.data_ptr(), dev, hip_stream, thread_k, thread_n, sms,
            group_blocks, kMaxPar, has_bias, has_act_order, has_zp,
            is_zp_float, a_is_fp8, a_fp8_is_fnuz, g_idx_ptr);
      } else {
        err = fp4_marlin_hip_gemm<__half, false>(
            A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
            b_bias_ptr, a_scales_ptr, b_scales.data_ptr(), nullptr,
            b_zeros_ptr,
            static_cast<int>(size_m),
            static_cast<int>(size_n), static_cast<int>(size_k),
            workspace.data_ptr(), dev, hip_stream, thread_k, thread_n, sms,
            group_blocks, kMaxPar, has_bias, has_act_order, has_zp,
            is_zp_float, a_is_fp8, a_fp8_is_fnuz, g_idx_ptr);
      }
    } else {
      err = gptq_marlin_hip_gemm<__half>(
          A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
          b_bias_ptr, a_scales_ptr, b_scales.data_ptr(), b_zeros_ptr,
          static_cast<int>(size_m),
          static_cast<int>(size_n), static_cast<int>(size_k),
          workspace.data_ptr(), dev, hip_stream, thread_k, thread_n, sms,
          num_bits, group_blocks, kMaxPar, has_bias, has_act_order, has_zp,
          is_zp_float, a_is_fp8, a_fp8_is_fnuz, g_idx_ptr);
    }
  } else if (c_type_id == vllm::kBFloat16.id()) {
    if (is_fp8) {
      bool fp8_is_fnuz = arch.rfind("gfx94", 0) == 0;
      err = fp8_marlin_hip_gemm<__hip_bfloat16>(
          A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
          b_scales.data_ptr(), static_cast<int>(size_m),
          static_cast<int>(size_n), static_cast<int>(size_k),
          workspace.data_ptr(), dev, hip_stream, fp8_is_fnuz, thread_k,
          thread_n, sms,
          group_blocks, kMaxPar);
    } else if (is_fp4) {
      if (scale_is_e4m3) {
        err = fp4_marlin_hip_gemm<__hip_bfloat16, true>(
            A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
            b_bias_ptr, a_scales_ptr, b_scales.data_ptr(), global_scale_ptr,
            b_zeros_ptr,
            static_cast<int>(size_m),
            static_cast<int>(size_n), static_cast<int>(size_k),
            workspace.data_ptr(), dev, hip_stream, thread_k, thread_n, sms,
            group_blocks, kMaxPar, has_bias, has_act_order, has_zp,
            is_zp_float, a_is_fp8, a_fp8_is_fnuz, g_idx_ptr);
      } else {
        err = fp4_marlin_hip_gemm<__hip_bfloat16, false>(
            A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
            b_bias_ptr, a_scales_ptr, b_scales.data_ptr(), nullptr,
            b_zeros_ptr,
            static_cast<int>(size_m),
            static_cast<int>(size_n), static_cast<int>(size_k),
            workspace.data_ptr(), dev, hip_stream, thread_k, thread_n, sms,
            group_blocks, kMaxPar, has_bias, has_act_order, has_zp,
            is_zp_float, a_is_fp8, a_fp8_is_fnuz, g_idx_ptr);
      }
    } else {
      err = gptq_marlin_hip_gemm<__hip_bfloat16>(
          A_ptr, b_q_weight.data_ptr(), c.data_ptr(),
          b_bias_ptr, a_scales_ptr, b_scales.data_ptr(), b_zeros_ptr,
          static_cast<int>(size_m),
          static_cast<int>(size_n), static_cast<int>(size_k),
          workspace.data_ptr(), dev, hip_stream, thread_k, thread_n, sms,
          num_bits, group_blocks, kMaxPar, has_bias, has_act_order, has_zp,
          is_zp_float, a_is_fp8, a_fp8_is_fnuz, g_idx_ptr);
    }
  } else {
    TORCH_CHECK(false,
                "gptq_marlin_gemm (ROCm) only supports bfloat16 and float16 outputs");
  }

  if (err == ERR_PROB_SHAPE) {
    TORCH_CHECK(size_n % thread_n == 0, "prob_n = ", size_n,
                " is not divisible by thread_n = ", thread_n);
    TORCH_CHECK(size_k % thread_k == 0, "prob_k = ", size_k,
                " is not divisible by thread_k = ", thread_k);
    TORCH_CHECK(false, "Invalid thread config: thread_k = ", thread_k,
                ", thread_n = ", thread_n, " for MKN = [", size_m, ", ",
                size_k, ", ", size_n, "]");
  } else if (err == ERR_KERN_SHAPE) {
    TORCH_CHECK(false, "Unsupported shapes: MNK = [", size_m, ", ", size_n,
                ", ", size_k, "]", ", has_act_order = ", has_act_order,
                ", num_groups = ", num_groups, ", group_size = ", group_size,
                ", thread_k = ", thread_k, ", thread_n = ", thread_n,
                ", num_bits = ", num_bits, ", has_zp = ", has_zp,
                ", is_zp_float = ", is_zp_float);
  }

  return c;
}
