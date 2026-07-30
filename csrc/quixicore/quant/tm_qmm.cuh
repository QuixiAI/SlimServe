/**
 * @file
 * @brief Shared pieces of the quantized-matmul family: the generic Marlin
 * zero-shuffle fragment loader (any FMT straight into m16n8k16 fragments),
 * the fp16 X fragment loader, the raw mma wrapper, and the fp16_raw
 * passthrough format used by the full-dequant route.
 */
#pragma once
#include "quant_formats.cuh"
#include "quant_formats_tables.cuh"
#include <cuda_fp16.h>

namespace tmq {

// fp16 passthrough "format": lets the full-dequant route reuse the same kernel.
struct fp16_raw {
    static constexpr int block_k = 16, block_bytes = 32;   // 16 halfs
    __device__ static float dequant(const uint8_t* base, int col) {
        return __half2float(reinterpret_cast<const __half*>(base)[col]);
    }
};

// ---- dequant one 16x16 W chunk (rows n0.., cols k0..) into an fp16 mma B fragment.
// Fragment convention (TK 2.0 row layout): data[k] = rows +8*(k%2), cols +8*(k/2);
// lane owns the pair (lane/4, (lane%4)*2). Each lane's rows hit different packed
// rows (base recomputed); the 16-col span stays inside one block (block_k % 16 == 0).
template<typename FMT>
__device__ __forceinline__ void load_wfrag(half2 frag[4], const uint8_t* Wq, int bpr, int n0, int k0) {
    const int kb = k0 / FMT::block_k, cin = k0 % FMT::block_k;
    const int lane = threadIdx.x & 31;
    const int r = n0 + lane / 4;
    const int c = cin + (lane % 4) * 2;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const int rr = r + (k % 2) * 8;
        const int cc = c + (k / 2) * 8;
        const uint8_t* base = Wq + (size_t(rr) * bpr + kb) * FMT::block_bytes;
        frag[k] = __floats2half2_rn(FMT::dequant(base, cc), FMT::dequant(base, cc + 1));
    }
}

// X fragment (A operand, fp16 row-major (M,K)): same layout, plain loads.
__device__ __forceinline__ void load_xfrag(half2 frag[4], const half* X, int K, int m0, int k0) {
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4;
    const int c = k0 + (lane % 4) * 2;
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        const half* p = X + size_t(r + (k % 2) * 8) * K + c + (k / 2) * 8;
        frag[k] = *reinterpret_cast<const half2*>(p);
    }
}

// m16n8k16 fp16 mma with fp32 accum; D 16x16 as two 16x8 halves.
__device__ __forceinline__ void mma16816(float d[4], const half2 a[4], const half2 b[2]) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

}  // namespace tmq
