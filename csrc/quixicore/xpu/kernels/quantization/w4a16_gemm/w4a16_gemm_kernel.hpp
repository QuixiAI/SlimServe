#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// W4A16 GEMM on the Xe tensor engine (DPAS via SYCL joint_matrix):
// C[M,N] = A[M,K] . dequant(W)^T, where W is [N,K] int4 group-quantized (the
// qgemv_int4 / quantize_int4_group encoding) and A/C are 16-bit float (act_dt in
// {f16, bf16}). The weight tile is dequantized on the fly into SLM and fed to
// joint_matrix as the B operand; fp32 accumulation. act_dt f32 is not supported
// (a16 means 16-bit activation); returns an empty event.
sycl::event w4a16_gemm_sycl(sycl::queue& q, const void* A, const void* w_packed,
                            const void* scales, void* C, std::size_t M,
                            std::size_t N, std::size_t K, std::size_t group,
                            DType act_dt);

}  // namespace quixicore::xpu::kernels
