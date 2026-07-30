/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's norm+quant epilogues and asymmetric
 * int8 quantizers (layer_norm_quant.metal, quantization.metal AZP path).
 * Complements kernels/elementwise/tm_elementwise_kernels.cuh (rms_norm_add_k
 * had only the fp8 dynamic add path); this adds int8 output, static scale, and
 * the residual/non-residual permutations, plus AZP int8 and group int8.
 *
 *   rms_norm_quant<T,FP8,DYN,RESID> : rms(+residual)*w -> fp8/int8 codes, per-row
 *       dynamic scale (amax/QMAX) or a static inv_scale. RESID writes res_out.
 *   azp_int8_quant<T,DYN>          : asymmetric int8. dynamic = per-row min/max,
 *       scale=(max-min)/255, zp=rint(-128-min/scale); static uses passed scale/azp.
 *   per_token_group_int8_quant     : per group_size block, symmetric int8.
 *
 * int8 saturation is [-128,127] (MetalForge float_to_int8_rn_sat / int32_sat),
 * NOT the symmetric [-127,127] of tm's int8_encode. fp8 uses tmq::e4m3_encode.
 */
#pragma once
#include "../quant/quant_formats.cuh"    // e4m3_encode
#include "../serving/tm_warp.cuh"        // warp_sum_f/max_f/min_f
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tmnq {

template <typename T> __device__ __forceinline__ float nf(T v);
template <> __device__ __forceinline__ float nf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float nf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float nf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T fn(float v);
template <> __device__ __forceinline__ float          fn<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         fn<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  fn<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

__device__ __forceinline__ int8_t i8sat(float x) {             // MetalForge float_to_int8_rn_sat
    return int8_t(int(fmaxf(-128.0f, fminf(127.0f, rintf(x)))));
}
__device__ __forceinline__ int8_t i8sat_i(int x) {
    return int8_t(max(-128, min(127, x)));
}

// Block-wide reductions that broadcast the result to every thread (for the
// multi-warp-per-row norm kernels). `partials` needs blockDim.x/32 floats;
// `bcast` is one float. Both must be __syncthreads-safe across the two calls.
__device__ __forceinline__ float block_sum_bcast(float v, float* partials, float* bcast) {
    v = tms::warp_sum_f(v);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) partials[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = (threadIdx.x < nw) ? partials[threadIdx.x] : 0.0f;
        t = tms::warp_sum_f(t);
        if (threadIdx.x == 0) *bcast = t;
    }
    __syncthreads();
    return *bcast;
}
__device__ __forceinline__ float block_max_bcast(float v, float* partials, float* bcast) {
    v = tms::warp_max_f(v);
    const int w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if ((threadIdx.x & 31) == 0) partials[w] = v;
    __syncthreads();
    if (w == 0) {
        float t = (threadIdx.x < nw) ? partials[threadIdx.x] : 0.0f;
        t = tms::warp_max_f(t);
        if (threadIdx.x == 0) *bcast = t;
    }
    __syncthreads();
    return *bcast;
}

// RMSNorm (+optional residual) with quantized output. One *block* (multiple
// warps) per row — D is split across all threads and reduced block-wide, which
// lifts the one-warp-per-row occupancy/latency wall. FP8 -> e4m3 codes; else int8
// codes. DYN -> per-row amax/QMAX scale in scale_out; else static inv_scale.
// RESID -> v=x+residual, write res_out. Launch with blockDim a multiple of 32.
template <typename T, bool FP8, bool DYN, bool RESID>
__global__ void rms_norm_quant(uint8_t* __restrict__ codes, float* __restrict__ scale_out,
                               T* __restrict__ res_out, const T* __restrict__ x,
                               const T* __restrict__ residual, const T* __restrict__ weight,
                               int D, float eps, float inv_static) {
    const long base = (long)blockIdx.x * D;
    const int tid = threadIdx.x, BLK = blockDim.x;
    __shared__ float part[32];
    __shared__ float bcast;
    // Pass 1: sum-of-squares and (DYN) the un-normalized max|v*w| in one sweep.
    // Since inv_rms is a positive per-row constant, amax = max|v*inv_rms*w| =
    // inv_rms * max|v*w| — so the dynamic-scale max folds into this pass (2 sweeps
    // total instead of 3).
    float ss = 0.0f, maxvw = 0.0f;
    for (int j = tid; j < D; j += BLK) {
        float v = nf(x[base + j]);
        if (RESID) { v += nf(residual[base + j]); res_out[base + j] = fn<T>(v); }
        ss += v * v;
        if (DYN) maxvw = fmaxf(maxvw, fabsf(v * nf(weight[j])));
    }
    ss = block_sum_bcast(ss, part, &bcast);
    const float inv_rms = rsqrtf(ss / float(D) + eps);
    const float qmax = FP8 ? 448.0f : 127.0f;

    float inv_scale = inv_static;
    if (DYN) {
        const float amax = block_max_bcast(maxvw, part, &bcast) * inv_rms;
        const float scale = amax / qmax;
        inv_scale = scale > 0.0f ? 1.0f / scale : 0.0f;
        if (tid == 0) scale_out[blockIdx.x] = scale;
    }
    for (int j = tid; j < D; j += BLK) {
        const float v = RESID ? nf(res_out[base + j]) : nf(x[base + j]);
        const float y = v * inv_rms * nf(weight[j]) * inv_scale;
        codes[base + j] = FP8 ? tmq::e4m3_encode(y) : uint8_t(i8sat(y));
    }
}

// Asymmetric int8 (AZP). One warp per row. DYN: per-row min/max ->
// scale=(max-min)/255, zp=rint(-128-min/scale), write scale_out/azp_out; else
// use passed static scale_static/azp_static. q = rint(x/scale)+zp, sat[-128,127].
template <typename T, bool DYN>
__global__ void azp_int8_quant(int8_t* __restrict__ codes, float* __restrict__ scale_out,
                               int* __restrict__ azp_out, const T* __restrict__ x, int D,
                               float scale_static, int azp_static) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float scale = scale_static;
    int zp = azp_static;
    if (DYN) {
        float mn = 3.4028234663852886e38f, mx = -3.4028234663852886e38f;
        for (int j = lane; j < D; j += 32) {
            const float v = nf(x[base + j]);
            mn = fminf(mn, v); mx = fmaxf(mx, v);
        }
        mn = tms::warp_min_f(mn); mx = tms::warp_max_f(mx);
        scale = (mx - mn) / 255.0f;
        zp = int(rintf(-128.0f - mn / scale));
        if (lane == 0) { scale_out[blockIdx.x] = scale; azp_out[blockIdx.x] = zp; }
    }
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    for (int j = lane; j < D; j += 32)
        codes[base + j] = i8sat_i(int(rintf(nf(x[base + j]) * inv)) + zp);
}

// Symmetric per-token-group int8 (feeds block-scaled int8 GEMM). One warp per
// (token, group). scale = amax/127 -> scales[token,group]. q = rint(x/scale).
template <typename T>
__global__ void per_token_group_int8_quant(int8_t* __restrict__ codes,
                                           float* __restrict__ scales, const T* __restrict__ x,
                                           int hidden, int group_size, int n_groups, float eps) {
    const int token = blockIdx.y, g = blockIdx.x, lane = threadIdx.x;
    const int c0 = g * group_size;
    const T* row = x + (long)token * hidden;
    float amax = 0.0f;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        amax = fmaxf(amax, fabsf(nf(row[c])));
    amax = tms::warp_max_f(fmaxf(amax, eps));
    const float scale = amax / 127.0f;
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (lane == 0) scales[(long)token * n_groups + g] = scale;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        codes[(long)token * hidden + c] = i8sat(nf(row[c]) * inv);
}

} // namespace tmnq
