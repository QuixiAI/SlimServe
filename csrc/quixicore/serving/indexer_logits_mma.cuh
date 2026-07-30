/**
 * @file
 * @brief DSA indexer MQA logits on Ampere tensor cores (m16n8k16), fp8 K.
 *
 * The scalar version in `indexer_logits_kernels.cuh` is correct but ~9% of fp32
 * peak, and measured 5.3x SLOWER than the torch fallback -- because torch sends
 * the D-contraction to cuBLAS tensor cores while the scalar kernel does fp32 FMA
 * out of shared memory. Writing it in CUDA instead of Triton does not make it
 * fast; putting the reduction dim on tensor cores does.
 *
 *   score[h,n] = (SUM_d q[m,h,d] * k[n,d]) * kscale[n]
 *   logits[m,n] = SUM_h relu(score[h,n]) * weights[m,h]
 *
 * The h-reduction is fused into the mma epilogue, so the [H,M,N] score tensor
 * the torch path materializes (16 GiB at N=131072) is never formed.
 *
 * fp8 is decoded to half in shared memory once per tile: e4m3 is bits at rest,
 * decode is software, and the multiply rides the fp16 tensor cores.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "quant_formats.cuh"   // tmq::e4m3_decode

namespace tms {

// Bulk e4m3 -> half. Decoding inside the GEMM block re-does the same work once
// per grid row/column (the key tile is decoded once per query row); hoisting it
// costs one pass over the data and a bf16 copy, which is what the torch path
// does too (`k_fp8.to(torch.bfloat16)`).
__global__ void indexer_decode_e4m3(const uint8_t* __restrict__ src,
                                    half* __restrict__ dst, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __float2half(tmq::e4m3_decode(src[i]));
}


__device__ __forceinline__ void mma_m16n8k16(float d[4], const half2 a[4],
                                             const half2 b[2]) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// One block per (query row m, tile of NT keys). blockDim.x = 32 * NWARP.
// Warp w owns heads [16w, 16w+16); it walks the NT keys in 16-wide steps.
template <int D, int NT, int NWARP>
__global__ __launch_bounds__(32 * NWARP) void indexer_mqa_logits_mma(
    const uint8_t* __restrict__ q,        // [M, H, D] e4m3
    const uint8_t* __restrict__ k,        // [N, D]    e4m3
    const float* __restrict__ kscale,     // [N]
    const float* __restrict__ weights,    // [M, H]
    const int* __restrict__ ks,           // [M]
    const int* __restrict__ ke,           // [M]
    float* __restrict__ logits,           // [M, N]
    int M, int N, int H) {
    extern __shared__ char smem_raw[];
    half* q_s = reinterpret_cast<half*>(smem_raw);          // [H][D]
    half* k_s = q_s + (size_t)16 * NWARP * D;               // [NT][D]
    float* part = reinterpret_cast<float*>(k_s + (size_t)NT * D);  // [NT]

    const int m = blockIdx.y;
    const int n0 = blockIdx.x * NT;
    if (m >= M || n0 >= N) return;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HB = 16 * NWARP;                              // heads per block

    // decode q for this row (heads this block covers) and the key tile
    for (int i = tid; i < HB * D; i += blockDim.x) {
        const int hh = i / D, dd = i % D;
        q_s[i] = (hh < H)
                     ? __float2half(tmq::e4m3_decode(q[((size_t)m * H + hh) * D + dd]))
                     : __float2half(0.f);
    }
    for (int i = tid; i < NT * D; i += blockDim.x) {
        const int nn = i / D, dd = i % D;
        const int n = n0 + nn;
        k_s[i] = (n < N)
                     ? __float2half(tmq::e4m3_decode(k[(size_t)n * D + dd]))
                     : __float2half(0.f);
    }
    for (int i = tid; i < NT; i += blockDim.x) part[i] = 0.f;
    __syncthreads();

    const int h_base = 16 * warp;
    // fragment row/col this lane owns (standard m16n8k16 layout)
    const int fr = lane >> 2, fc = (lane & 3) * 2;

    for (int nb = 0; nb < NT; nb += 16) {
        float acc0[4] = {0, 0, 0, 0}, acc1[4] = {0, 0, 0, 0};
        for (int d0 = 0; d0 < D; d0 += 16) {
            half2 a[4], b[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int r = h_base + fr + 8 * (t & 1);
                const int c = d0 + fc + 8 * (t >> 1);
                a[t] = *reinterpret_cast<const half2*>(&q_s[(size_t)r * D + c]);
                const int rn = nb + fr + 8 * (t & 1);
                b[t] = *reinterpret_cast<const half2*>(&k_s[(size_t)rn * D + c]);
            }
            const half2 b0[2] = {b[0], b[2]};   // keys nb .. nb+7
            const half2 b1[2] = {b[1], b[3]};   // keys nb+8 .. nb+15
            mma_m16n8k16(acc0, a, b0);
            mma_m16n8k16(acc1, a, b1);
        }
        // Epilogue. acc[0,1] hold h = h_base+fr, acc[2,3] hold h = h_base+fr+8,
        // and the low bit of e selects the key column. For a fixed key the
        // contributing lanes share (lane & 3), i.e. differ by 4/8/16 -- so a
        // shuffle reduction over fr collapses 8 lanes before touching shared
        // memory. The naive form issued 4 atomicAdds per lane into part[].
#pragma unroll
        for (int half_i = 0; half_i < 2; ++half_i) {
            const float* acc = half_i ? acc1 : acc0;
            const int nsub = nb + 8 * half_i;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int nn = nsub + fc + pair;
                const int n = n0 + nn;
                const int h_lo = h_base + fr, h_hi = h_base + fr + 8;
                float v = 0.f;
                if (nn < NT && n < N) {
                    const float sc = kscale[n];
                    if (h_lo < H)
                        v += fmaxf(acc[pair] * sc, 0.f) *
                             weights[(size_t)m * H + h_lo];
                    if (h_hi < H)
                        v += fmaxf(acc[2 + pair] * sc, 0.f) *
                             weights[(size_t)m * H + h_hi];
                }
#pragma unroll
                for (int off = 4; off <= 16; off <<= 1)
                    v += __shfl_xor_sync(0xffffffffu, v, off);
                if (fr == 0 && nn < NT && n < N) atomicAdd(&part[nn], v);
            }
        }
    }
    __syncthreads();

    const int lo = ks[m], hi = ke[m];
    for (int i = tid; i < NT; i += blockDim.x) {
        const int n = n0 + i;
        if (n < N)
            logits[(size_t)m * N + n] = (n >= lo && n < hi) ? part[i] : -INFINITY;
    }
}

// Same kernel with q/k already decoded to half (see indexer_decode_e4m3).
template <int D, int NT, int NWARP>
__global__ __launch_bounds__(32 * NWARP) void indexer_mqa_logits_mma_pre(
    const half* __restrict__ q, const half* __restrict__ k,
    const float* __restrict__ kscale, const float* __restrict__ weights,
    const int* __restrict__ ks, const int* __restrict__ ke,
    float* __restrict__ logits, int M, int N, int H) {
    extern __shared__ char smem_raw[];
    half* q_s = reinterpret_cast<half*>(smem_raw);
    half* k_s = q_s + (size_t)16 * NWARP * D;
    float* part = reinterpret_cast<float*>(k_s + (size_t)NT * D);

    const int m = blockIdx.y;
    const int n0 = blockIdx.x * NT;
    if (m >= M || n0 >= N) return;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HB = 16 * NWARP;

    for (int i = tid; i < HB * D; i += blockDim.x) {
        const int hh = i / D, dd = i % D;
        q_s[i] = (hh < H) ? q[((size_t)m * H + hh) * D + dd] : __float2half(0.f);
    }
    for (int i = tid; i < NT * D; i += blockDim.x) {
        const int nn = i / D, dd = i % D, n = n0 + nn;
        k_s[i] = (n < N) ? k[(size_t)n * D + dd] : __float2half(0.f);
    }
    for (int i = tid; i < NT; i += blockDim.x) part[i] = 0.f;
    __syncthreads();

    const int h_base = 16 * warp, fr = lane >> 2, fc = (lane & 3) * 2;
    for (int nb = 0; nb < NT; nb += 16) {
        float acc0[4] = {0,0,0,0}, acc1[4] = {0,0,0,0};
        for (int d0 = 0; d0 < D; d0 += 16) {
            half2 a[4], b[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int r = h_base + fr + 8 * (t & 1);
                const int c = d0 + fc + 8 * (t >> 1);
                a[t] = *reinterpret_cast<const half2*>(&q_s[(size_t)r * D + c]);
                const int rn = nb + fr + 8 * (t & 1);
                b[t] = *reinterpret_cast<const half2*>(&k_s[(size_t)rn * D + c]);
            }
            const half2 b0[2] = {b[0], b[2]};
            const half2 b1[2] = {b[1], b[3]};
            mma_m16n8k16(acc0, a, b0);
            mma_m16n8k16(acc1, a, b1);
        }
        // Epilogue. acc[0,1] hold h = h_base+fr, acc[2,3] hold h = h_base+fr+8,
        // and the low bit of e selects the key column. For a fixed key the
        // contributing lanes share (lane & 3), i.e. differ by 4/8/16 -- so a
        // shuffle reduction over fr collapses 8 lanes before touching shared
        // memory. The naive form issued 4 atomicAdds per lane into part[].
#pragma unroll
        for (int hi_ = 0; hi_ < 2; ++hi_) {
            const float* acc = hi_ ? acc1 : acc0;
            const int nsub = nb + 8 * hi_;
#pragma unroll
            for (int pair = 0; pair < 2; ++pair) {
                const int nn = nsub + fc + pair;
                const int n = n0 + nn;
                const int h_lo = h_base + fr, h_hi = h_base + fr + 8;
                float v = 0.f;
                if (nn < NT && n < N) {
                    const float sc = kscale[n];
                    if (h_lo < H)
                        v += fmaxf(acc[pair] * sc, 0.f) *
                             weights[(size_t)m * H + h_lo];
                    if (h_hi < H)
                        v += fmaxf(acc[2 + pair] * sc, 0.f) *
                             weights[(size_t)m * H + h_hi];
                }
#pragma unroll
                for (int off = 4; off <= 16; off <<= 1)
                    v += __shfl_xor_sync(0xffffffffu, v, off);
                if (fr == 0 && nn < NT && n < N) atomicAdd(&part[nn], v);
            }
        }
    }
    __syncthreads();
    const int lo = ks[m], hi2 = ke[m];
    for (int i = tid; i < NT; i += blockDim.x) {
        const int n = n0 + i;
        if (n < N)
            logits[(size_t)m * N + n] = (n >= lo && n < hi2) ? part[i] : -INFINITY;
    }
}

template <int D, int NT, int NWARP>
__host__ __device__ constexpr size_t indexer_mqa_logits_mma_smem() {
    return (size_t)16 * NWARP * D * sizeof(half) + (size_t)NT * D * sizeof(half) +
           (size_t)NT * sizeof(float);
}

}  // namespace tms
