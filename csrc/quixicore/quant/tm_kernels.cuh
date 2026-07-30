/**
 * @file
 * @brief All validated W1/W2 quant kernels in one header, shared by the
 * standalone golden harnesses and the tm_cuda torch extension. Bodies are
 * the exact code validated against the quant.py goldens - edit here, retest
 * via the harnesses.
 */
#pragma once
#include "tm_qmm.cuh"
#include "tm_rng.cuh"
#include "../serving/tm_warp.cuh"
#include <cuda_fp16.h>

#define LMH_NEG_INF_DEF
#ifndef LMH_NEG_INF
#define LMH_NEG_INF (-3.4028234663852886e38f)
#endif

namespace tmq {

// ---- exactness kernel: dequantize every weight on the GPU ----
template<typename FMT>
__global__ void dequant_all(float* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    int bpr = K / FMT::block_k;
    const uint8_t* base = Wq + (size_t(row) * bpr + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = FMT::dequant(base, col % FMT::block_k);
}

// ---- qgemv: D = dequant(Wq) @ X, warp per row ----
template<typename FMT>
__global__ void qgemv(half* D, const uint8_t* Wq, const half* X, int N, int K) {
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;   // 32 threads
    const int bpr  = K / FMT::block_k;
    const uint8_t* row_base = Wq + size_t(row) * bpr * FMT::block_bytes;

    constexpr int CPL = 8;                        // contiguous cols per lane
    constexpr int LPB = FMT::block_k / CPL;       // lanes per block (2..32 for block_k 16..256)
    constexpr int BPI = (32 / LPB) > 0 ? (32 / LPB) : 1;  // blocks per warp iteration
    const int b_off = lane / LPB;
    const int col0  = (lane % LPB) * CPL;

    float acc = 0.0f;
    if constexpr (FMT::block_k <= 256) {
        for (int kb = b_off; kb < bpr; kb += BPI) {
            const uint8_t* base = Wq + (size_t(row) * bpr + kb) * FMT::block_bytes;
            const half* xp = X + kb * FMT::block_k + col0;
            float w[8];
            dequant8<FMT>(base, col0, w);
            #pragma unroll
            for (int i = 0; i < 8; i++) acc += w[i] * __half2float(xp[i]);
        }
    }
    (void)row_base;
    // warp reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) D[row] = __float2half(acc);
}



// ---- qgemm: one warp per 16x16 output tile ----
// Y(M,N) = X(M,K) @ W(N,K)^T. B fragment for mma.ABt: W rows are the mma "columns";
// the W fragment layout above IS the col-major B operand of m16n8k16 when split
// into its two 8-col (= 8 W-row) halves: b0 = {data[0], data[1]}, b1 = {data[2], data[3]}.
template<typename FMT>
__global__ void qgemm(float* Y, const half* X, const uint8_t* Wq, int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float acc0[4] = {0, 0, 0, 0};   // rows m0..+16, W-rows n0..n0+8
    float acc1[4] = {0, 0, 0, 0};   // rows m0..+16, W-rows n0+8..n0+16
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 a[4], b[4];
        load_xfrag(a, X, K, m0, k0);
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        // mma B operand wants (k-major pairs per W-row group); our frag has
        // data[0,1] = W rows n0..(+8 in data[1]? no: rows +8*(k%2)) —
        // data[0]=rows n0+0..7? lane r = n0+lane/4 (+8 for k%2=1).
        // b half 0 (W rows n0..n0+7 x k0..k0+15): needs {cols 0-7, cols 8-15} = {data[0], data[2]}
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    // store: acc layout of m16n8: d[0,1] = (row lane/4, col (lane%4)*2 +0/1), d[2,3] = row+8
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    if (r < M) {
        Y[size_t(r) * N + c0]     = acc0[0];
        Y[size_t(r) * N + c0 + 1] = acc0[1];
        Y[size_t(r) * N + c0 + 8]     = acc1[0];
        Y[size_t(r) * N + c0 + 8 + 1] = acc1[1];
    }
    if (r + 8 < M) {
        Y[size_t(r + 8) * N + c0]     = acc0[2];
        Y[size_t(r + 8) * N + c0 + 1] = acc0[3];
        Y[size_t(r + 8) * N + c0 + 8]     = acc1[2];
        Y[size_t(r + 8) * N + c0 + 8 + 1] = acc1[3];
    }
}

// ---- qgemm_ksplit (perf pass): the warp-per-16x16-tile qgemm starves the
// GPU at decode shapes — M=64, N=512 is only 128 warps across 82 SMs (half
// the chip idle, no latency hiding). Split the K axis across blockIdx.z:
// each warp reduces its [k_beg, k_end) slice and atomicAdd's the fp32
// partial into Y (zeroed by the host). Same fragment math as qgemm. ----
template<typename FMT>
__global__ void qgemm_ksplit(float* Y, const half* X, const uint8_t* Wq,
                             int M, int N, int K, int k_chunk) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int k_beg = blockIdx.z * k_chunk;
    const int k_end = min(K, k_beg + k_chunk);
    const int bpr = K / FMT::block_k;

    float acc0[4] = {0, 0, 0, 0};
    float acc1[4] = {0, 0, 0, 0};
    for (int k0 = k_beg; k0 < k_end; k0 += 16) {
        half2 a[4], b[4];
        load_xfrag(a, X, K, m0, k0);
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    if (r < M) {
        atomicAdd(&Y[size_t(r) * N + c0],         acc0[0]);
        atomicAdd(&Y[size_t(r) * N + c0 + 1],     acc0[1]);
        atomicAdd(&Y[size_t(r) * N + c0 + 8],     acc1[0]);
        atomicAdd(&Y[size_t(r) * N + c0 + 8 + 1], acc1[1]);
    }
    if (r + 8 < M) {
        atomicAdd(&Y[size_t(r + 8) * N + c0],         acc0[2]);
        atomicAdd(&Y[size_t(r + 8) * N + c0 + 1],     acc0[3]);
        atomicAdd(&Y[size_t(r + 8) * N + c0 + 8],     acc1[2]);
        atomicAdd(&Y[size_t(r + 8) * N + c0 + 8 + 1], acc1[3]);
    }
}

// Host-side split heuristic: enough K-slices to keep ~2 resident warps per
// SM slot busy, aligned to the format block so a slice never splits a block.
static inline int qgemm_pick_kchunk(int M, int N, int K, int block_k) {
    const long tiles = long((N + 15) / 16) * ((M + 15) / 16);
    const int target = 1664;                      // ~82 SMs x 20 warps
    int splits = int((target + tiles - 1) / tiles);
    const int align = (block_k > 16) ? block_k : 16;
    int chunk = ((K / (splits > 0 ? splits : 1)) + align - 1) / align * align;
    if (chunk < align) chunk = align;
    return chunk;
}

// full-dequant kernel (route for 256-superblock formats)
template<typename FMT>
__global__ void dequant_to_fp16(half* out, const uint8_t* Wq, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * K) return;
    int row = idx / K, col = idx % K;
    const uint8_t* base = Wq + (size_t(row) * (K / FMT::block_k) + col / FMT::block_k) * FMT::block_bytes;
    out[idx] = __float2half(FMT::dequant(base, col % FMT::block_k));
}

// ---- qgemm_actorder: GPTQ act-order with an IN-KERNEL g_idx gather (TM
// qgemm.metal 160-212). The weight is quantized in PERMUTED K order (groups
// contiguous); the X columns are gathered by perm during the fragment fill —
// no materialized permuted-X copy: Y[m,n] = sum_i deq(Wq)[n,i] * X[m, perm[i]].
// Mirrors qgemm otherwise (warp per 16x16 tile, mma_ABt fragment halves). ----
template<typename FMT>
__global__ void qgemm_actorder(float* Y, const half* X, const uint8_t* Wq, const int* perm,
                               int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;
    const int lane = threadIdx.x & 31;

    float acc0[4] = {0, 0, 0, 0};
    float acc1[4] = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 a[4], b[4];
        // X fragment with the K-axis gather (element loads; perm cols non-contiguous)
        const int r = m0 + lane / 4;
        const int c = k0 + (lane % 4) * 2;
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            const int rr = r + (k % 2) * 8;
            const int cc = c + (k / 2) * 8;
            a[k] = __halves2half2(X[size_t(rr) * K + perm[cc]],
                                  X[size_t(rr) * K + perm[cc + 1]]);
        }
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    if (r < M) {
        Y[size_t(r) * N + c0]     = acc0[0];
        Y[size_t(r) * N + c0 + 1] = acc0[1];
        Y[size_t(r) * N + c0 + 8]     = acc1[0];
        Y[size_t(r) * N + c0 + 8 + 1] = acc1[1];
    }
    if (r + 8 < M) {
        Y[size_t(r + 8) * N + c0]     = acc0[2];
        Y[size_t(r + 8) * N + c0 + 1] = acc0[3];
        Y[size_t(r + 8) * N + c0 + 8]     = acc1[2];
        Y[size_t(r + 8) * N + c0 + 8 + 1] = acc1[3];
    }
}

// ---- qgemm_blockscale: fp8_raw codes-only weights scaled by a SEPARATE
// (N/128, K/128) fp16 per-tile scale buffer (compressed-tensors fp8_block
// without per-row replication; TM qgemm.metal 214-252). A 16x16 fragment
// never straddles a 128-tile, so one scalar scale per k-step. ----
template<typename FMT>
__global__ void qgemm_blockscale(float* Y, const half* X, const uint8_t* Wq,
                                 const half* scale2d, int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;
    const int KT = K / 128;
    const int lane = threadIdx.x & 31;

    float acc0[4] = {0, 0, 0, 0};
    float acc1[4] = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 a[4], b[4];
        load_xfrag(a, X, K, m0, k0);
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        const half2 sc = __half2half2(scale2d[(n0 / 128) * KT + (k0 / 128)]);
        #pragma unroll
        for (int k = 0; k < 4; k++) b[k] = __hmul2(b[k], sc);
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    if (r < M) {
        Y[size_t(r) * N + c0]     = acc0[0];
        Y[size_t(r) * N + c0 + 1] = acc0[1];
        Y[size_t(r) * N + c0 + 8]     = acc1[0];
        Y[size_t(r) * N + c0 + 8 + 1] = acc1[1];
    }
    if (r + 8 < M) {
        Y[size_t(r + 8) * N + c0]     = acc0[2];
        Y[size_t(r + 8) * N + c0 + 1] = acc0[3];
        Y[size_t(r + 8) * N + c0 + 8]     = acc1[2];
        Y[size_t(r + 8) * N + c0 + 8 + 1] = acc1[3];
    }
}



// tanh-approx GELU, matching TM's substrate gelu() / F.gelu(approximate='tanh')
__device__ __forceinline__ float gelu_tanh(float x) {
    const float c = 0.7978845608028654f;   // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

template<typename FMT>
__global__ void qflux_gelu(float* Y, const half* X, const uint8_t* Wq, const float* bias,
                           int M, int N, int K) {
    const int n0 = blockIdx.x * 16;
    const int m0 = blockIdx.y * 16;
    const int bpr = K / FMT::block_k;

    float acc0[4] = {0, 0, 0, 0};
    float acc1[4] = {0, 0, 0, 0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 a[4], b[4];
        load_xfrag(a, X, K, m0, k0);
        load_wfrag<FMT>(b, Wq, bpr, n0, k0);
        half2 b0[2] = {b[0], b[2]};
        half2 b1[2] = {b[1], b[3]};
        mma16816(acc0, a, b0);
        mma16816(acc1, a, b1);
    }
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    const float bA = bias[c0], bB = bias[c0 + 1], bC = bias[c0 + 8], bD = bias[c0 + 9];
    if (r < M) {
        Y[size_t(r) * N + c0]     = gelu_tanh(acc0[0] + bA);
        Y[size_t(r) * N + c0 + 1] = gelu_tanh(acc0[1] + bB);
        Y[size_t(r) * N + c0 + 8] = gelu_tanh(acc1[0] + bC);
        Y[size_t(r) * N + c0 + 9] = gelu_tanh(acc1[1] + bD);
    }
    if (r + 8 < M) {
        Y[size_t(r + 8) * N + c0]     = gelu_tanh(acc0[2] + bA);
        Y[size_t(r + 8) * N + c0 + 1] = gelu_tanh(acc0[3] + bB);
        Y[size_t(r + 8) * N + c0 + 8] = gelu_tanh(acc1[2] + bC);
        Y[size_t(r + 8) * N + c0 + 9] = gelu_tanh(acc1[3] + bD);
    }
}



__device__ __forceinline__ int warp_sum_i32(int v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}
__device__ __forceinline__ float warp_sum_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

// ---- W8A8: two rows per warp, X loaded once (uint4 = 16 int8), dp4a x4 per row ----
__global__ void qgemv_w8a8(half* D, const int8_t* Wq, const int8_t* Xq,
                           const half* w_scale, const half* a_scale, int N, int K) {
    const int row0 = blockIdx.x * 2;
    const bool two = row0 + 1 < N;
    const int lane = threadIdx.x;
    const uint4* w0 = reinterpret_cast<const uint4*>(Wq + size_t(row0) * K);
    const uint4* w1 = reinterpret_cast<const uint4*>(Wq + size_t(row0 + (two ? 1 : 0)) * K);
    const uint4* xv = reinterpret_cast<const uint4*>(Xq);
    int acc0 = 0, acc1 = 0;
    for (int u = lane; u < K / 16; u += 32) {
        const uint4 x = xv[u];
        const uint4 a = w0[u];
        acc0 = idot4(a.x, x.x, acc0); acc0 = idot4(a.y, x.y, acc0);
        acc0 = idot4(a.z, x.z, acc0); acc0 = idot4(a.w, x.w, acc0);
        const uint4 b = w1[u];
        acc1 = idot4(b.x, x.x, acc1); acc1 = idot4(b.y, x.y, acc1);
        acc1 = idot4(b.z, x.z, acc1); acc1 = idot4(b.w, x.w, acc1);
    }
    for (int k = (K & ~15) + lane * 4; k + 4 <= K; k += 128) {   // K%16 tail (K%4==0)
        const unsigned x = reinterpret_cast<const unsigned*>(Xq)[k / 4];
        acc0 = idot4(reinterpret_cast<const unsigned*>(Wq)[(size_t(row0) * K + k) / 4], x, acc0);
        acc1 = idot4(reinterpret_cast<const unsigned*>(Wq)[(size_t(row0 + (two ? 1 : 0)) * K + k) / 4], x, acc1);
    }
    acc0 = warp_sum_i32(acc0);
    acc1 = warp_sum_i32(acc1);
    if (lane == 0) {
        D[row0] = __float2half(float(acc0) * __half2float(w_scale[row0]) * __half2float(a_scale[0]));
        if (two) D[row0 + 1] = __float2half(float(acc1) * __half2float(w_scale[row0 + 1]) * __half2float(a_scale[0]));
    }
}

// ---- W2A8 BitNet: block-major walk, per-group scale applied once per 8-code span ----
__global__ void qgemv_w2a8(half* D, const uint8_t* Wq, const int8_t* Xq,
                           const half* a_scale, int N, int K) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    const int bpr = K / bitnet::block_k;
    const uint8_t* row_base = Wq + size_t(row) * bpr * bitnet::block_bytes;
    constexpr int CPL = 8;                          // codes per lane
    constexpr int LPB = bitnet::block_k / CPL;      // 4 lanes per block
    constexpr int BPI = 32 / LPB;                   // 8 blocks per iteration
    const int b_off = lane / LPB;
    const int col0  = (lane % LPB) * CPL;
    float lane_acc = 0.0f;
    for (int g = b_off; g < bpr; g += BPI) {
        const uint8_t* base = row_base + size_t(g) * bitnet::block_bytes;
        const uint16_t codes = *reinterpret_cast<const uint16_t*>(base + 2 + (col0 >> 2));
        const int8_t* x = Xq + g * bitnet::block_k + col0;
        int isum = 0, ixsum = 0;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            isum += int((codes >> (2 * i)) & 3) * int(x[i]);   // code in {0,1,2}
            ixsum += int(x[i]);                                // subtract the -1 bias
        }
        lane_acc += float(isum - ixsum) * bitnet::gscale(base);
    }
    const float facc = warp_sum_f32(lane_acc);
    if (lane == 0) D[row] = __float2half(facc * __half2float(a_scale[0]));
}



// e4m3_encode is now the consolidated __host__ __device__ one in quant_formats.cuh.
// float -> symmetric int8 [-127,127], RNE (matches np.rint).
__device__ __forceinline__ int8_t int8_encode(float x) {
    float r = rintf(x);
    r = fminf(fmaxf(r, -127.0f), 127.0f);
    return int8_t(int(r));
}

__device__ __forceinline__ float warp_max_f32(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// ---- per-token (per row of A (T,D)): one warp per row ----
template<bool FP8>
__global__ void quantize_per_token(uint8_t* codes, float* scale, const float* A, int T, int D) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const float* row = A + size_t(t) * D;
    float amax = 0.0f;
    for (int d = lane; d < D; d += 32) amax = fmaxf(amax, fabsf(row[d]));
    amax = warp_max_f32(amax);
    const float QMAX = FP8 ? 448.0f : 127.0f;
    float s = amax / QMAX;
    float inv = (s == 0.0f) ? 1.0f : 1.0f / s;
    if (lane == 0) scale[t] = s;
    for (int d = lane; d < D; d += 32) {
        float v = row[d] * inv;
        codes[size_t(t) * D + d] = FP8 ? e4m3_encode(v) : uint8_t(int8_encode(v));
    }
}

// ---- per-tensor: pass 1 absmax via atomicMax on order-preserving uint mapping ----
__device__ __forceinline__ unsigned float_to_ordered(float f) {
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float ordered_to_float(unsigned u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}
__global__ void tensor_absmax(unsigned* omax, const float* A, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    float m = 0.0f;
    for (size_t j = i * 16; j < min(n, (i + 1) * 16); j++) m = fmaxf(m, fabsf(A[j]));
    m = warp_max_f32(m);
    if ((threadIdx.x & 31) == 0) atomicMax(omax, float_to_ordered(m));
}
template<bool FP8>
__global__ void tensor_encode(uint8_t* codes, float* scale, const unsigned* omax, const float* A, size_t n) {
    const float QMAX = FP8 ? 448.0f : 127.0f;
    float s = ordered_to_float(*omax) / QMAX;
    float inv = (s == 0.0f) ? 1.0f : 1.0f / s;
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = s;
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (size_t j = i * 16; j < min(n, (i + 1) * 16); j++) {
        float v = A[j] * inv;
        codes[j] = FP8 ? e4m3_encode(v) : uint8_t(int8_encode(v));
    }
}



__device__ __forceinline__ void warp_argmax(float& best, int& bi) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, off);
        int   oi = __shfl_xor_sync(0xffffffffu, bi, off);
        if (ov > best || (ov == best && oi < bi)) { best = ov; bi = oi; }
    }
}

// ---- fp16 weights ----
__global__ void lm_head_argcat_partials(const half* h, const half* W, float* part_val,
                                        int* part_id, const float* bias, int V, int K,
                                        int TILE_V, int num_vtiles, float invtemp,
                                        unsigned seed, int use_gumbel, int use_bias) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const half* hrow = h + size_t(t) * K;
    float best = LMH_NEG_INF;
    int bi = (v0 + lane < v1) ? v0 + lane : v0;
    for (int v = v0 + lane; v < v1; v += 32) {
        const half* wrow = W + size_t(v) * K;
        float acc = 0.0f;
        for (int j = 0; j < K / 2; j++) {
            half2 w2 = reinterpret_cast<const half2*>(wrow)[j];
            half2 h2 = reinterpret_cast<const half2*>(hrow)[j];
            acc += __half2float(w2.x) * __half2float(h2.x) + __half2float(w2.y) * __half2float(h2.y);
        }
        float ls = acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, unsigned(t), unsigned(v));
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    warp_argmax(best, bi);
    if (lane == 0) {
        part_val[size_t(t) * num_vtiles + vtile] = best;
        part_id[size_t(t) * num_vtiles + vtile] = bi;
    }
}

// ---- quantized weights (any FMT) ----
template<typename FMT>
__global__ void lm_head_argcat_partials_q(const half* h, const uint8_t* Wq, float* part_val,
                                          int* part_id, const float* bias, int V, int K,
                                          int TILE_V, int num_vtiles, float invtemp,
                                          unsigned seed, int use_gumbel, int use_bias) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    const half* hrow = h + size_t(t) * K;
    float best = LMH_NEG_INF;
    int bi = (v0 + lane < v1) ? v0 + lane : v0;
    for (int v = v0 + lane; v < v1; v += 32) {
        const uint8_t* row_base = Wq + size_t(v) * bpr * FMT::block_bytes;
        float acc = 0.0f;
        for (int kb = 0; kb < bpr; kb++) {
            const uint8_t* base = row_base + size_t(kb) * FMT::block_bytes;
            const half* xp = hrow + kb * FMT::block_k;
            for (int c0 = 0; c0 < FMT::block_k; c0 += 8) {
                float w[8];
                dequant8<FMT>(base, c0, w);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc += w[i] * __half2float(xp[c0 + i]);
            }
        }
        float ls = acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, unsigned(t), unsigned(v));
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    warp_argmax(best, bi);
    if (lane == 0) {
        part_val[size_t(t) * num_vtiles + vtile] = best;
        part_id[size_t(t) * num_vtiles + vtile] = bi;
    }
}

__global__ void lm_head_argcat_reduce(const float* part_val, const int* part_id,
                                      int* out_idx, int num_vtiles) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const size_t base = size_t(t) * num_vtiles;
    float best = LMH_NEG_INF;
    int bi = 0x7fffffff;
    for (int j = lane; j < num_vtiles; j += 32) {
        float v = part_val[base + j];
        int id = part_id[base + j];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[t] = bi;
}

// ============================================================================
// Fused LM-head top-k / top-p (nucleus) sampling — TM lm_head.metal 180-462.
// partials: warp per (vocab tile, token); each lane collects its owned logits
// into per-lane local slots, Family-B masked_topk_local merges the tile's
// top-k winners into (part_val, part_id). Top-p partials ALSO emit a per-tile
// tempered logsumexp so the reduce can build the TRUE full-vocab normalizer
// (exact nucleus cutoff, not the pool approximation).
// reduce: warp per token — Family-A masked_topk over the merged pool, then
// Gumbel-max (top-k) or logit-threshold bisection + Gumbel-max (top-p), noise
// indexed by the GLOBAL vocab id so fused draws equal the unfused sampler's.
// TILE_V <= 2048 (64 local slots per lane).
// ============================================================================
#define LMH_MAX_K 64
#define LMH_MAX_PER_LANE 64

// USE_LSE=1 also emits the per-tile tempered logsumexp (top-p path).
template<int USE_LSE>
__global__ void lm_head_topk_partials(const half* h, const half* W, float* part_val,
                                      int* part_id, const float* bias, int V, int K,
                                      int TILE_V, int num_vtiles, int topk, int use_bias,
                                      float invtemp, float* part_lse) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const half* hrow = h + size_t(t) * K;
    float mine_val[LMH_MAX_PER_LANE];
    int   mine_id[LMH_MAX_PER_LANE];
    bool  used[LMH_MAX_PER_LANE];
    int   nmine = 0;
    float lmax = LMH_NEG_INF, lsum = 0.0f;
    for (int v = v0 + lane; v < v1; v += 32) {
        const half* wrow = W + size_t(v) * K;
        float acc = 0.0f;
        for (int j = 0; j < K / 2; j++) {
            half2 w2 = reinterpret_cast<const half2*>(wrow)[j];
            half2 h2 = reinterpret_cast<const half2*>(hrow)[j];
            acc += __half2float(w2.x) * __half2float(h2.x) + __half2float(w2.y) * __half2float(h2.y);
        }
        float ls = acc;
        if (use_bias) ls += bias[v];
        if (USE_LSE) {
            const float tl = ls * invtemp;
            if (tl > lmax) { lsum = lsum * expf(lmax - tl) + 1.0f; lmax = tl; }
            else           { lsum += expf(tl - lmax); }
        }
        mine_val[nmine] = ls;
        mine_id[nmine] = v;
        ++nmine;
    }
    if (USE_LSE) {
        const float gmax = tms::warp_max_f(lmax);
        const float gsum = tms::warp_sum_f((lmax == LMH_NEG_INF) ? 0.0f : lsum * expf(lmax - gmax));
        if (lane == 0)
            part_lse[size_t(t) * num_vtiles + vtile] = (gsum > 0.0f) ? (gmax + logf(gsum)) : LMH_NEG_INF;
    }
    const size_t pbase = (size_t(t) * num_vtiles + vtile) * topk;
    tms::masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF,
        [&](int kk, float gbest, int gid) {
            if (lane == 0) {
                part_val[pbase + kk] = gbest;
                part_id[pbase + kk] = (gbest == LMH_NEG_INF) ? -1 : gid;
            }
        });
}

template<typename FMT, int USE_LSE>
__global__ void lm_head_topk_partials_q(const half* h, const uint8_t* Wq, float* part_val,
                                        int* part_id, const float* bias, int V, int K,
                                        int TILE_V, int num_vtiles, int topk, int use_bias,
                                        float invtemp, float* part_lse) {
    const int vtile = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const int v0 = vtile * TILE_V, v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    const half* hrow = h + size_t(t) * K;
    float mine_val[LMH_MAX_PER_LANE];
    int   mine_id[LMH_MAX_PER_LANE];
    bool  used[LMH_MAX_PER_LANE];
    int   nmine = 0;
    float lmax = LMH_NEG_INF, lsum = 0.0f;
    for (int v = v0 + lane; v < v1; v += 32) {
        const uint8_t* row_base = Wq + size_t(v) * bpr * FMT::block_bytes;
        float acc = 0.0f;
        for (int kb = 0; kb < bpr; kb++) {
            const uint8_t* base = row_base + size_t(kb) * FMT::block_bytes;
            const half* xp = hrow + kb * FMT::block_k;
            for (int c0 = 0; c0 < FMT::block_k; c0 += 8) {
                float w[8];
                dequant8<FMT>(base, c0, w);
                #pragma unroll
                for (int i = 0; i < 8; i++) acc += w[i] * __half2float(xp[c0 + i]);
            }
        }
        float ls = acc;
        if (use_bias) ls += bias[v];
        if (USE_LSE) {
            const float tl = ls * invtemp;
            if (tl > lmax) { lsum = lsum * expf(lmax - tl) + 1.0f; lmax = tl; }
            else           { lsum += expf(tl - lmax); }
        }
        mine_val[nmine] = ls;
        mine_id[nmine] = v;
        ++nmine;
    }
    if (USE_LSE) {
        const float gmax = tms::warp_max_f(lmax);
        const float gsum = tms::warp_sum_f((lmax == LMH_NEG_INF) ? 0.0f : lsum * expf(lmax - gmax));
        if (lane == 0)
            part_lse[size_t(t) * num_vtiles + vtile] = (gsum > 0.0f) ? (gmax + logf(gsum)) : LMH_NEG_INF;
    }
    const size_t pbase = (size_t(t) * num_vtiles + vtile) * topk;
    tms::masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF,
        [&](int kk, float gbest, int gid) {
            if (lane == 0) {
                part_val[pbase + kk] = gbest;
                part_id[pbase + kk] = (gbest == LMH_NEG_INF) ? -1 : gid;
            }
        });
}

// top-k reduce: global merge of the per-tile partial winners, then Gumbel-max
// among the k winners (tempered; noise by global vocab id).
__global__ void lm_head_topk_reduce(const float* part_val, const int* part_id,
                                    int* out_idx, int num_vtiles, int topk,
                                    unsigned seed, float invtemp) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const int ncand = num_vtiles * topk;
    const size_t base = size_t(t) * ncand;
    int   chosen_id[LMH_MAX_K];
    float chosen_val[LMH_MAX_K];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = part_id[base + idx];
        v = part_val[base + idx];
        valid = id >= 0;
    };
    tms::masked_topk(cand, ncand, topk, lane, LMH_NEG_INF, chosen_id, chosen_val);
    float best = LMH_NEG_INF;
    int bi = chosen_id[0];
    for (int kk = 0; kk < topk; ++kk) {
        if (chosen_id[kk] < 0) continue;
        const float g = rng_gumbel(seed, unsigned(t), unsigned(chosen_id[kk]));
        const float p = chosen_val[kk] * invtemp + g;
        if (p > best || (p == best && chosen_id[kk] < bi)) { best = p; bi = chosen_id[kk]; }
    }
    if (lane == 0) out_idx[t] = bi;
}

// top-p reduce: nucleus over the merged over-selected pool with the TRUE
// full-vocab normalizer from the per-tile lses; 32-step bisection of the
// tempered-logit threshold, then Gumbel-max over {ls >= L}.
__global__ void lm_head_topp_reduce(const float* part_val, const int* part_id,
                                    int* out_idx, int num_vtiles, int topk, float p,
                                    unsigned seed, float invtemp, const float* part_lse) {
    const int t = blockIdx.x, lane = threadIdx.x;
    const int ncand = num_vtiles * topk;
    const size_t base = size_t(t) * ncand;
    float mx = LMH_NEG_INF;
    for (int j = lane; j < ncand; j += 32)
        if (part_id[base + j] >= 0) mx = fmaxf(mx, part_val[base + j] * invtemp);
    mx = tms::warp_max_f(mx);
    const size_t lbase = size_t(t) * num_vtiles;
    float Z = 0.0f;
    for (int vt = lane; vt < num_vtiles; vt += 32) {
        const float pl = part_lse[lbase + vt];
        if (pl > LMH_NEG_INF) Z += expf(pl - mx);
    }
    Z = tms::warp_sum_f(Z);
    float lo = mx - 40.0f, hi = mx;             // largest L with mass{ls>=L} >= p
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int j = lane; j < ncand; j += 32) {
            const float ls = part_val[base + j] * invtemp;
            if (part_id[base + j] >= 0 && ls >= mid) sm += expf(ls - mx);
        }
        sm = tms::warp_sum_f(sm) / Z;
        if (sm >= p) lo = mid; else hi = mid;
    }
    const float L = lo;
    float best = LMH_NEG_INF;
    int bi = -1;
    for (int j = lane; j < ncand; j += 32) {
        const int id = part_id[base + j];
        const float ls = part_val[base + j] * invtemp;
        if (id < 0 || ls < L) continue;
        const float pert = ls + rng_gumbel(seed, unsigned(t), unsigned(id));
        if (pert > best || (pert == best && id < bi)) { best = pert; bi = id; }
    }
    float gbest = best;
    int gid = (bi < 0) ? 0x7fffffff : bi;
    warp_argmax(gbest, gid);
    if (lane == 0) out_idx[t] = (gbest == LMH_NEG_INF) ? -1 : gid;
}

}  // namespace tmq
