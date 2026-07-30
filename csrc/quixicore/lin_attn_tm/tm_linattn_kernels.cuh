/**
 * @file
 * @brief CUDA/SM86 port of ThunderMittens' linear-attention family:
 * linear_attn (non-causal two-phase), lin_attn_causal (serial chunked scan),
 * and the 3-kernel chunked-PARALLEL causal decomposition lin_chunk_kv ->
 * lin_chunk_scan -> lin_chunk_out (grid (C,H,B) instead of one warp per
 * (b,h) — the long-N fix).
 *
 * Identity feature map, unnormalized: out_i = sum_{j<=i} (q_i . k_j) v_j
 * (inclusive; non-causal drops the j<=i). Layout (B, H, N, D) contiguous.
 *
 * Scheme: one block of D threads per (b,h)[,chunk]; thread j OWNS column j of
 * the (D,D) KV state in shared memory (16KB fp32 at D=64) — rank-1 updates
 * and q@KV reads touch only that column, so the serial-scan body needs no
 * barriers. All accumulation fp32 (TM round-trips the state through bf16 for
 * its mma; the scalar form keeps fp32 — strictly tighter). The register-tile
 * mma route is a perf-pass item.
 *
 * NOTE: this is the TM-API twin of kernels/based's TK linear_attention_ampere
 * (which fuses the based feature maps); these are the plain identity-map
 * kernels + the chunk-parallel decomposition that TK port doesn't have.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tmla {

template <typename T> __device__ __forceinline__ float ltf(T v);
template <> __device__ __forceinline__ float ltf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float ltf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float ltf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T lft(float v);
template <> __device__ __forceinline__ float          lft<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         lft<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  lft<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

#define LIN_CHUNK_L 64

// Non-causal: KV = sum_j k_j^T v_j, then out_i = q_i @ KV.
// grid (H, B), block D threads; thread j owns KV column j.
template <typename T, int D>
__global__ void linear_attn(const T* __restrict__ q, const T* __restrict__ k,
                            const T* __restrict__ v, T* __restrict__ o, int N) {
    __shared__ float kv[D * D];
    const int j = threadIdx.x;
    const long base = ((long)blockIdx.y * gridDim.x + blockIdx.x) * N * D;
    for (int i = 0; i < D; ++i) kv[i * D + j] = 0.0f;
    for (int r = 0; r < N; ++r) {
        const float vj = ltf(v[base + (long)r * D + j]);
        for (int i = 0; i < D; ++i)
            kv[i * D + j] += ltf(k[base + (long)r * D + i]) * vj;
    }
    for (int r = 0; r < N; ++r) {
        float acc = 0.0f;
        for (int i = 0; i < D; ++i)
            acc += ltf(q[base + (long)r * D + i]) * kv[i * D + j];
        o[base + (long)r * D + j] = lft<T>(acc);
    }
}

// Serial causal scan (inclusive): per row, state += k_r^T v_r THEN
// out_r = q_r @ state. grid (H, B), block D.
template <typename T, int D>
__global__ void lin_attn_causal(const T* __restrict__ q, const T* __restrict__ k,
                                const T* __restrict__ v, T* __restrict__ o, int N) {
    __shared__ float kv[D * D];
    const int j = threadIdx.x;
    const long base = ((long)blockIdx.y * gridDim.x + blockIdx.x) * N * D;
    for (int i = 0; i < D; ++i) kv[i * D + j] = 0.0f;
    for (int r = 0; r < N; ++r) {
        const float vj = ltf(v[base + (long)r * D + j]);
        float acc = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float kij = kv[i * D + j] + ltf(k[base + (long)r * D + i]) * vj;
            kv[i * D + j] = kij;
            acc += ltf(q[base + (long)r * D + i]) * kij;
        }
        o[base + (long)r * D + j] = lft<T>(acc);
    }
}

// K1: per-chunk KV block sums. S (B,H,C,D,D) fp32; grid (C, H, B), block D.
template <typename T, int D>
__global__ void lin_chunk_kv(const T* __restrict__ k, const T* __restrict__ v,
                             float* __restrict__ S, int N) {
    const int j = threadIdx.x;
    const int c = blockIdx.x, C = N / LIN_CHUNK_L;
    const long base = ((long)blockIdx.z * gridDim.y + blockIdx.y) * N * D;
    const long sbase = (((long)blockIdx.z * gridDim.y + blockIdx.y) * C + c) * D * D;
    float col[D];
    #pragma unroll
    for (int i = 0; i < D; ++i) col[i] = 0.0f;
    for (int t = 0; t < LIN_CHUNK_L; ++t) {
        const long r = (long)c * LIN_CHUNK_L + t;
        const float vj = ltf(v[base + r * D + j]);
        #pragma unroll
        for (int i = 0; i < D; ++i) col[i] += ltf(k[base + r * D + i]) * vj;
    }
    for (int i = 0; i < D; ++i) S[sbase + (long)i * D + j] = col[i];
}

// K2: exclusive prefix over the chunk axis. grid (B*H), block 256; each thread
// owns D*D/256 state elements and marches C serially.
template <int D>
__global__ void lin_chunk_scan(const float* __restrict__ Sin, float* __restrict__ Sex, int C) {
    const long base = (long)blockIdx.x * C * D * D;
    for (int e = threadIdx.x; e < D * D; e += blockDim.x) {
        float run = 0.0f;
        long idx = base + e;
        for (int c = 0; c < C; ++c, idx += D * D) {
            const float t = Sin[idx];
            Sex[idx] = run;
            run += t;
        }
    }
}

// K3: per-chunk serial scan body seeded with the scanned inter-chunk state.
// grid (C, H, B), block D.
template <typename T, int D>
__global__ void lin_chunk_out(const T* __restrict__ q, const T* __restrict__ k,
                              const T* __restrict__ v, const float* __restrict__ Sex,
                              T* __restrict__ o, int N) {
    __shared__ float kv[D * D];
    const int j = threadIdx.x;
    const int c = blockIdx.x, C = N / LIN_CHUNK_L;
    const long base = ((long)blockIdx.z * gridDim.y + blockIdx.y) * N * D;
    const long sbase = (((long)blockIdx.z * gridDim.y + blockIdx.y) * C + c) * D * D;
    for (int i = 0; i < D; ++i) kv[i * D + j] = Sex[sbase + (long)i * D + j];
    for (int t = 0; t < LIN_CHUNK_L; ++t) {
        const long r = (long)c * LIN_CHUNK_L + t;
        const float vj = ltf(v[base + r * D + j]);
        float acc = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float kij = kv[i * D + j] + ltf(k[base + r * D + i]) * vj;
            kv[i * D + j] = kij;
            acc += ltf(q[base + r * D + i]) * kij;
        }
        o[base + r * D + j] = lft<T>(acc);
    }
}

// Complex GEMM: D = A @ B with a leading (real, imag) plane axis — A (2,N,K),
// B (2,K,M), D (2,N,M). One thread per output element, fp32 accumulate of the
// 4-product complex multiply. (TM builds this from 4 real mmas; the tensor-core
// route is a perf-pass item — fftconv already has the TK complex mma on CUDA.)
template <typename T>
__global__ void cmplx_matmul(const T* __restrict__ A, const T* __restrict__ B,
                             T* __restrict__ Dm, int N, int K, int M) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (n >= N || m >= M) return;
    const T* Ar = A;               const T* Ai = A + (long)N * K;
    const T* Br = B;               const T* Bi = B + (long)K * M;
    float dr = 0.0f, di = 0.0f;
    for (int kk = 0; kk < K; ++kk) {
        const float ar = ltf(Ar[(long)n * K + kk]), ai = ltf(Ai[(long)n * K + kk]);
        const float br = ltf(Br[(long)kk * M + m]), bi = ltf(Bi[(long)kk * M + m]);
        dr += ar * br - ai * bi;
        di += ar * bi + ai * br;
    }
    Dm[(long)n * M + m] = lft<T>(dr);
    Dm[(long)N * M + (long)n * M + m] = lft<T>(di);
}

} // namespace tmla
