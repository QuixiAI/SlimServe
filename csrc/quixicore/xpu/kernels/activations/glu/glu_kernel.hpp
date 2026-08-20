#pragma once

#include <cstddef>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"

namespace quixicore::xpu::kernels {

// mode: 0 swiglu (silu gate), 1 geglu (gelu gate), 2 reglu (relu gate),
//       3 glu (sigmoid gate). Input [rows, 2*d], output [rows, d].
sycl::event glu_sycl(sycl::queue& q, const void* x, void* out, std::size_t rows,
                     std::size_t d, DType dt, int mode);


// GEGLU with a fused f16 output (the f16-context variant of glu, mirroring
// attention_f16ctx). Input `x` is [rows, 2*d] row-major (gate half then value
// half, same layout as glu); `out` is [rows, d] and always f16:
//   out[r,i] = f16( gelu_tanh(x[r,i]) * x[r,d+i] )
// The gate uses the tanh GELU approximation (matching the embeddinggemma.c
// source), NOT the erf GELU that glu mode=geglu uses. Folds the f32/dt->f16
// convert a downstream f16 GEMM needs into the activation. Shape: GEGLU -> f16.
sycl::event glu_gelu_f16_sycl(sycl::queue& q, const void* x, void* out,
                              std::size_t rows, std::size_t d, DType dt);

}  // namespace quixicore::xpu::kernels
