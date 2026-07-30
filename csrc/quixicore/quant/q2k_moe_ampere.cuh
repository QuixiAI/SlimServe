/**
 * @file
 * @brief Grouped (MoE) Q2_K GEMV on Ampere -- expert-indexed twin of
 *        `q2k_gemv_a8` in q2k_ampere.cuh.
 *
 * Semantics match vLLM's `ggml_moe_a8_vec`:
 *   out[i, :] = W[expert(i)] @ x[src(i), :]
 *   expert(i) = topk_ids[i],  src(i) = i / top_k
 * so w1 sees each token top_k times and w2 is called with top_k == 1.
 *
 * Every structural decision is inherited from the dense kernel, where they were
 * measured: UNROLL-independent weight loads for memory-level parallelism, one
 * 16-element sub-block per lane so 32 lanes cover 128 B contiguous, and the
 * QB-batched int32 epilogue (sc_j*A_j and m_j*S_j accumulate as IMAD because d,
 * dmin and the activation scale g are constant inside a superblock).
 *
 * M collapses to 1 here: a routed row is a single token's activation. The
 * amortization that the dense kernel got from staging M activations now comes
 * from NR output rows per warp reusing one staged activation tile.
 *
 * PARKED -- bound but deliberately not wired into the serving path. Measured
 * 1.06x over ggml_moe_a8_vec in the DRAM-bound regime (A100, 52 distinct
 * experts, 72 MB working set: 489 vs 460 GB/s), which does not pay for
 * repacking every expert tensor at load time or the extra memory that costs at
 * TP4. The dense kernel's win does not carry over: MoE reads a different
 * expert per routed row, so there is no weight reuse to amortize.
 */
#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace tmq_a100 {

template <int NR, int KC>
__host__ __device__ constexpr size_t q2k_moe_gemv_smem() {
    return (size_t)4 * (KC / 16) * sizeof(uint32_t) +
           (size_t)(KC / 16) * sizeof(int32_t) +
           (size_t)(KC / 256 + 1) * sizeof(__half);
}

// grid = (ceil(N / (nwarp*NR)), num_rows), blockDim.x = 256.
template <int NR = 4, int KC = 4096, int QB = 2>
__global__ __launch_bounds__(256) void q2k_moe_gemv_a8(
    const uint32_t* __restrict__ qp,     // [E][N][K/16]
    const uint8_t* __restrict__ sp,      // [E][N][K/16]
    const half2* __restrict__ dp,        // [E][N][K/256]
    const int8_t* __restrict__ xq,       // [T][K]
    const __half* __restrict__ xs,       // [T][K/256]
    const int32_t* __restrict__ xsum,    // [T][K/16]
    const int* __restrict__ topk_ids,    // [num_rows]
    float* __restrict__ Y,               // [num_rows][N]
    int num_rows, int top_k, int N, int K) {
    extern __shared__ unsigned char smem[];
    const int NJC = KC / 16;
    uint32_t* sxp = reinterpret_cast<uint32_t*>(smem);          // [4][NJC]
    int32_t* ssum = reinterpret_cast<int32_t*>(sxp + (size_t)4 * NJC);
    __half* sxs = reinterpret_cast<__half*>(ssum + (size_t)NJC);

    const int row = blockIdx.y;
    if (row >= num_rows) return;
    const int e = topk_ids[row];
    if (e < 0) return;                    // unrouted row (expert_map miss)
    const int s = row / top_k;

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int nwarp = blockDim.x >> 5;
    const int nbase = (blockIdx.x * nwarp + warp) * NR;
    const int nj = K / 16, nsb = K / 256;

    // Expert-selected weight planes and this row's source activation.
    const uint32_t* qpe = qp + (size_t)e * N * nj;
    const uint8_t* spe = sp + (size_t)e * N * nj;
    const half2* dpe = dp + (size_t)e * N * nsb;
    const int8_t* xqs = xq + (size_t)s * K;
    const __half* xss = xs + (size_t)s * nsb;
    const int32_t* xsums = xsum + (size_t)s * nj;

    float acc[NR];
#pragma unroll
    for (int r = 0; r < NR; ++r) acc[r] = 0.f;

    for (int kc = 0; kc < K; kc += KC) {
        const int klen = min(KC, K - kc);
        __syncthreads();
        for (int t = threadIdx.x; t < klen / 16; t += blockDim.x) {
            const uint32_t* src = reinterpret_cast<const uint32_t*>(
                xqs + (size_t)(kc / 16 + t) * 16);
#pragma unroll
            for (int p = 0; p < 4; ++p) sxp[(size_t)p * NJC + t] = src[p];
            ssum[t] = xsums[kc / 16 + t];
        }
        for (int t = threadIdx.x; t < klen / 256; t += blockDim.x)
            sxs[t] = xss[kc / 256 + t];
        __syncthreads();

        const int jlo = kc / 16, jhi = jlo + klen / 16;
#pragma unroll
        for (int r = 0; r < NR; ++r) {
            const int n = nbase + r;
            if (n >= N) break;
            for (int jb = jlo + lane * QB; jb < jhi; jb += 32 * QB) {
                uint32_t wv[QB], scv[QB];
#pragma unroll
                for (int i = 0; i < QB; ++i) {
                    wv[i] = qpe[(size_t)n * nj + jb + i];
                    scv[i] = spe[(size_t)n * nj + jb + i];
                }
                const float2 dm = __half22float2(dpe[(size_t)n * nsb + (jb >> 4)]);
                const int jl = jb - jlo;
                int Ai = 0, Bi = 0;
#pragma unroll
                for (int i = 0; i < QB; ++i) {
                    const uint32_t w = wv[i];
                    const uint32_t sc = scv[i];
                    const size_t xb = (size_t)jl + i;
                    int A = 0;
                    A = __dp4a((int)((w >> 0) & 0x03030303u),
                               (int)sxp[0 * NJC + xb], A);
                    A = __dp4a((int)((w >> 2) & 0x03030303u),
                               (int)sxp[1 * NJC + xb], A);
                    A = __dp4a((int)((w >> 4) & 0x03030303u),
                               (int)sxp[2 * NJC + xb], A);
                    A = __dp4a((int)((w >> 6) & 0x03030303u),
                               (int)sxp[3 * NJC + xb], A);
                    Ai += (int)(sc & 0xFu) * A;
                    Bi += (int)(sc >> 4) * ssum[xb];
                }
                const float g = __half2float(sxs[jl >> 4]);
                acc[r] += g * (dm.x * (float)Ai - dm.y * (float)Bi);
            }
        }
    }

    // Reduce each output row across the warp's lanes.
#pragma unroll
    for (int r = 0; r < NR; ++r) {
        const int n = nbase + r;
        if (n >= N) break;
        float v = acc[r];
#pragma unroll
        for (int off = 16; off; off >>= 1)
            v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) Y[(size_t)row * N + n] = v;
    }
}

}  // namespace tmq_a100
