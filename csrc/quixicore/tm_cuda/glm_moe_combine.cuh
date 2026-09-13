#pragma once
// Fused MoE combine for the Marlin path: out[t] = shared[t] + sum_k x[t, k] in one
// launch, fp32 accumulation, a single bf16 rounding. Replaces the moe_sum kernel,
// the finalize copy into the modular kernel's output buffer and the inductor add
// of the shared-expert output: three launches per MoE layer at decode.
// x is the per-assignment Marlin w2 output [T, topk, D] (D contiguous), shared
// the shared-expert MLP output [T, D], out [T, D]; D is a multiple of 8 so every
// thread moves 16-byte packs. TOPK > 0 unrolls the top-k loop, TOPK == 0 reads
// the runtime top-k. The kernel is total: it indexes nothing beyond [T, topk, D].
// All three base pointers must also be 16-byte aligned (checked by the binding).
#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tms::glm_moe_combine {

constexpr int THREADS = 256;
constexpr int VEC = 8;  // bf16 per 16-byte pack

struct __align__(16) pack_t {
    __nv_bfloat162 v[VEC / 2];
};

__device__ __forceinline__ void accumulate(float (&acc)[VEC], const __nv_bfloat16* p) {
    const pack_t packed = *reinterpret_cast<const pack_t*>(p);
#pragma unroll
    for (int j = 0; j < VEC / 2; ++j) {
        const float2 f = __bfloat1622float2(packed.v[j]);
        acc[2 * j] += f.x;
        acc[2 * j + 1] += f.y;
    }
}

template <int TOPK>
__global__ __launch_bounds__(THREADS) void moe_sum_add_kernel(
        __nv_bfloat16* __restrict__ out,           // [T, D]
        const __nv_bfloat16* __restrict__ x,       // [T, topk, D]
        const __nv_bfloat16* __restrict__ shared,  // [T, D]
        int64_t num_tokens, int d, int topk) {
    const int64_t n_vec = d / VEC;
    const int64_t total = num_tokens * n_vec;
    const int k_count = TOPK > 0 ? TOPK : topk;
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < total;
         i += int64_t(gridDim.x) * blockDim.x) {
        const int64_t token = i / n_vec;
        const int64_t col = (i - token * n_vec) * VEC;
        float acc[VEC];
#pragma unroll
        for (int j = 0; j < VEC; ++j) acc[j] = 0.0f;
        accumulate(acc, shared + token * d + col);
        const __nv_bfloat16* xt = x + token * int64_t(k_count) * d + col;
        if constexpr (TOPK > 0) {
#pragma unroll
            for (int k = 0; k < TOPK; ++k) accumulate(acc, xt + int64_t(k) * d);
        } else {
            for (int k = 0; k < k_count; ++k) accumulate(acc, xt + int64_t(k) * d);
        }
        pack_t o;
#pragma unroll
        for (int j = 0; j < VEC / 2; ++j)
            o.v[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        *reinterpret_cast<pack_t*>(out + token * d + col) = o;
    }
}

// Host-side launch behind the native binding (tm_cuda_serving.cu).
inline void launch_moe_sum_add(__nv_bfloat16* out, const __nv_bfloat16* x,
                               const __nv_bfloat16* shared, int64_t num_tokens,
                               int d, int topk, cudaStream_t stream) {
    if (num_tokens == 0) return;
    const int64_t total = num_tokens * (d / VEC);
    const int blocks = int(std::min<int64_t>((total + THREADS - 1) / THREADS, 16384));
    if (topk == 8)
        moe_sum_add_kernel<8><<<blocks, THREADS, 0, stream>>>(out, x, shared, num_tokens, d, topk);
    else
        moe_sum_add_kernel<0><<<blocks, THREADS, 0, stream>>>(out, x, shared, num_tokens, d, topk);
}

}  // namespace tms::glm_moe_combine
