/**
 * @file
 * @brief CDNA3 fp8 MFMA primitives for the DSA indexer logits kernel.
 *
 * `v_mfma_f32_16x16x32_fp8_fp8` is what Triton emits for
 * `tl.dot(..., input_precision="ieee")` on fp8 operands (verified against the
 * compiled AMDGCN), so calling the same instruction with the same fragment
 * layout and K order is what makes a HIP port bitwise-equal rather than merely
 * close. See fp8_mqa_logits_design.md.
 *
 * Fragment layout for one 64-lane wave, cross-checked against QuixiCore-ROCm's
 * tm_qmm_mfma.cuh (16x16x16 f16 form) and llama.cpp's mma.cuh (16x16x32 i8
 * form):
 *
 *   A[M=16,K=32] : lane l, byte v in 0..7 -> A[m = l%16      ][k = 8*(l/16)+v]
 *   B[K=32,N=16] : lane l, byte v in 0..7 -> B[k = 8*(l/16)+v][n = l%16      ]
 *   D[M=16,N=16] : lane l, reg  v in 0..3 -> D[m = 4*(l/16)+v][n = l%16      ]
 */
#pragma once
#include <cstdint>

namespace qcrocm {

typedef __attribute__((__vector_size__(4 * sizeof(float)))) float f32x4;

// 8 fp8 values per lane per operand, passed as one 64-bit register pair --
// the `kWidth = 8` the TTGIR reports for both dot operands.
__device__ __forceinline__ f32x4 mfma_16x16x32_fp8(long a, long b, f32x4 acc) {
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc, 0, 0, 0);
}

// Load this lane's A fragment: 8 contiguous k of row (m0 + l%16).
// `lda` is the row stride of A in elements.
__device__ __forceinline__ long load_a_frag(const uint8_t* A, long lda, int m0,
                                            int k0) {
    const int l = threadIdx.x & 63;
    const long off = (long)(m0 + (l & 15)) * lda + k0 + (l >> 4) * 8;
    long v;
    __builtin_memcpy(&v, A + off, 8);
    return v;
}

// Load this lane's B fragment for B[k][n] held as Bt[n][k] (row-major over k,
// which is how the KV tile arrives): 8 contiguous k of row (n0 + l%16).
__device__ __forceinline__ long load_b_frag(const uint8_t* Bt, long ldb, int n0,
                                            int k0) {
    const int l = threadIdx.x & 63;
    const long off = (long)(n0 + (l & 15)) * ldb + k0 + (l >> 4) * 8;
    long v;
    __builtin_memcpy(&v, Bt + off, 8);
    return v;
}

}  // namespace qcrocm
