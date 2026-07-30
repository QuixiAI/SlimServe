/**
 * @file
 * @brief DSA indexer decode logits over the *paged* fp8 K cache, Ampere mma.
 *
 * This is the decode-path twin of `indexer_logits_mma.cuh`. It replaces
 * `fp8_paged_mqa_logits_torch`, which is not merely slow but *uncapturable*:
 * it loops over the batch in Python and calls `context_lens[i].item()`, a
 * device->host sync. A host round-trip has no CUDA-graph representation, so
 * that fallback fails capture with `operation not permitted when stream is
 * capturing`. Reading the context length on device is what removes it.
 *
 *   logits[i,n] = (SUM_h relu(SUM_d k[n,d] * q[i,h,d]) * w[i,h]) * kscale[n]
 *   for n < context_lens[i], else -inf.
 *
 * Scale placement matches the torch reference exactly: relu first, then the
 * head-weighted sum, and only then the per-token scale. Folding kscale in
 * early would agree only while the quant scale stays positive; the test
 * asserts that rather than the kernel assuming it.
 *
 * Paged layout, per block of `block_size` tokens (bytes):
 *   [0, block_size*D)              fp8 e4m3 keys, token-major
 *   [block_size*D, block_size*(D+4))  fp32 per-token scales
 * i.e. keys and scales are split at block granularity, not interleaved.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>
#include "quant_formats.cuh"   // tmq::e4m3_decode

namespace tms {

__device__ __forceinline__ void mma_m16n8k16_paged(float d[4], const half2 a[4],
                                                   const half2 b[2]) {
    const uint32_t* A = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* B = reinterpret_cast<const uint32_t*>(b);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]), "r"(B[1]));
}

// grid = (cdiv(max_model_len, NT), batch), blockDim.x = 32 * NWARP.
// `logits` must be pre-filled with -inf; blocks past a request's context
// length return without writing, which keeps the launch shape independent of
// any device-side length and therefore graph-capturable.
template <int D, int NT, int NWARP>
__global__ __launch_bounds__(32 * NWARP) void indexer_paged_mqa_logits(
    const uint8_t* __restrict__ q,            // [B, 1, H, D] e4m3
    const uint8_t* __restrict__ kv_cache,     // [num_blocks, block_size*(D+4)]
    const float* __restrict__ weights,        // [B, H]
    const int* __restrict__ context_lens,     // [B]
    const int* __restrict__ block_tables,     // [B, bt_stride]
    float* __restrict__ logits,               // [B, max_model_len]
    int H, int block_size, int bt_stride, int max_model_len) {
    extern __shared__ char smem_raw[];
    half* q_s = reinterpret_cast<half*>(smem_raw);           // [16*NWARP][D]
    half* k_s = q_s + (size_t)16 * NWARP * D;                // [NT][D]
    float* part = reinterpret_cast<float*>(k_s + (size_t)NT * D);  // [NT]
    float* ksc = part + NT;                                  // [NT]

    const int b = blockIdx.y;
    const int n0 = blockIdx.x * NT;
    const int seq_len = context_lens[b];
    if (n0 >= seq_len) return;

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int HB = 16 * NWARP;
    const size_t blk_bytes = (size_t)block_size * (D + 4);
    const size_t sc_off = (size_t)block_size * D;

    for (int i = tid; i < HB * D; i += blockDim.x) {
        const int hh = i / D, dd = i % D;
        q_s[i] = (hh < H) ? __float2half(tmq::e4m3_decode(
                                q[((size_t)b * H + hh) * D + dd]))
                          : __float2half(0.f);
    }
    for (int i = tid; i < NT * D; i += blockDim.x) {
        const int nn = i / D, dd = i % D, n = n0 + nn;
        half v = __float2half(0.f);
        if (n < seq_len) {
            const int page = block_tables[(size_t)b * bt_stride + n / block_size];
            const int within = n % block_size;
            v = __float2half(tmq::e4m3_decode(
                kv_cache[(size_t)page * blk_bytes + (size_t)within * D + dd]));
        }
        k_s[i] = v;
    }
    for (int i = tid; i < NT; i += blockDim.x) {
        part[i] = 0.f;
        const int n = n0 + i;
        float s = 0.f;
        if (n < seq_len) {
            const int page = block_tables[(size_t)b * bt_stride + n / block_size];
            const int within = n % block_size;
            s = *reinterpret_cast<const float*>(
                kv_cache + (size_t)page * blk_bytes + sc_off + (size_t)within * 4);
        }
        ksc[i] = s;
    }
    __syncthreads();

    const int h_base = 16 * warp, fr = lane >> 2, fc = (lane & 3) * 2;
    for (int nb = 0; nb < NT; nb += 16) {
        float acc0[4] = {0, 0, 0, 0}, acc1[4] = {0, 0, 0, 0};
        for (int d0 = 0; d0 < D; d0 += 16) {
            half2 a[4], bb[4];
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int r = h_base + fr + 8 * (t & 1);
                const int c = d0 + fc + 8 * (t >> 1);
                a[t] = *reinterpret_cast<const half2*>(&q_s[(size_t)r * D + c]);
                const int rn = nb + fr + 8 * (t & 1);
                bb[t] = *reinterpret_cast<const half2*>(&k_s[(size_t)rn * D + c]);
            }
            const half2 b0[2] = {bb[0], bb[2]};
            const half2 b1[2] = {bb[1], bb[3]};
            mma_m16n8k16_paged(acc0, a, b0);
            mma_m16n8k16_paged(acc1, a, b1);
        }
        // relu and the head weighting happen here; kscale is applied at the
        // epilogue below to match the reference's operation order.
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
                if (nn < NT && n < seq_len) {
                    if (h_lo < H)
                        v += fmaxf(acc[pair], 0.f) * weights[(size_t)b * H + h_lo];
                    if (h_hi < H)
                        v += fmaxf(acc[2 + pair], 0.f) *
                             weights[(size_t)b * H + h_hi];
                }
#pragma unroll
                for (int off = 4; off <= 16; off <<= 1)
                    v += __shfl_xor_sync(0xffffffffu, v, off);
                if (fr == 0 && nn < NT && n < seq_len) atomicAdd(&part[nn], v);
            }
        }
    }
    __syncthreads();

    for (int i = tid; i < NT; i += blockDim.x) {
        const int n = n0 + i;
        if (n < seq_len && n < max_model_len)
            logits[(size_t)b * max_model_len + n] = part[i] * ksc[i];
    }
}

template <int D, int NT, int NWARP>
__host__ __device__ constexpr size_t indexer_paged_mqa_logits_smem() {
    return (size_t)16 * NWARP * D * sizeof(half) + (size_t)NT * D * sizeof(half) +
           (size_t)NT * sizeof(float) * 2;
}

}  // namespace tms
