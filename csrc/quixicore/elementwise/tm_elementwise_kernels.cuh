/**
 * @file
 * @brief CUDA/SM86 port of ThunderMittens' W3 elementwise / norm / training
 * family: rms_norm, layernorm, add_norm (+fp8 e4m3 epilogues), softmax, gelu,
 * glu (6 modes), dropout, cross_entropy (+_mw), embedding (+2 backwards +
 * multimodal spans), hadamard, adamw, add.
 *
 * Faithful math ports of the .metal sources (formula-for-formula, same
 * clamps/approximations, same RNG contract). One warp per row replaces one
 * simdgroup per row; simd_sum/simd_max -> warp_sum_f/warp_max_f (tm_warp.cuh);
 * metal::atomic_float -> native atomicAdd(float*). Row kernels take runtime D
 * (the Metal compile-time-D register residency is a perf detail, revisited in
 * the perf pass). All compute fp32 (fp64 nowhere), I/O templated on
 * T in {float, __half, __nv_bfloat16}.
 *
 * Layout contract kept from TM: the fp8 add_norm epilogues write
 * codes[row*D + j] with j strided by lane (identical to rv_fl's w*32+lane map),
 * so packed outputs are byte-identical to the Metal kernels.
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include "../quant/tm_rng.cuh"
#include "../serving/tm_warp.cuh"

namespace tme {

using tms::warp_sum_f;
using tms::warp_max_f;
using tmq::rng_uniform;

// ---- T <-> float ----------------------------------------------------------
template <typename T> __device__ __forceinline__ float tf(T v);
template <> __device__ __forceinline__ float tf<float>(float v)                   { return v; }
template <> __device__ __forceinline__ float tf<__half>(__half v)                 { return __half2float(v); }
template <> __device__ __forceinline__ float tf<__nv_bfloat16>(__nv_bfloat16 v)   { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T ft(float v);
template <> __device__ __forceinline__ float          ft<float>(float v)          { return v; }
template <> __device__ __forceinline__ __half         ft<__half>(float v)         { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  ft<__nv_bfloat16>(float v)  { return __float2bfloat16(v); }

// float -> e4m3 code, RNE, clamp +-448 (same as quant_rt.cu / TM tk_e4m3_encode).
// __host__ __device__ so the harness oracle replays it exactly.
__host__ __device__ __forceinline__ uint8_t tme_e4m3_encode(float x) {
    const unsigned sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (!(a < 448.0f)) return uint8_t(sign | 0x7Eu);
    if (a < 0.0009765625f) {
        int mant = int(rintf(a * 512.0f));
        return uint8_t(sign | unsigned(mant));
    }
    int e;
    frexpf(a, &e);
    int E = e - 1;
    if (E < -6) {
        int mant = int(rintf(a * 512.0f));
        if (mant >= 8) return uint8_t(sign | (1u << 3));
        return uint8_t(sign | unsigned(mant));
    }
    float two_m = a / exp2f(float(E));
    int mant = int(rintf((two_m - 1.0f) * 8.0f));
    int exp = E + 7;
    if (mant >= 8) { mant = 0; exp += 1; }
    if (exp > 15 || (exp == 15 && mant > 6)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (unsigned(exp) << 3) | unsigned(mant));
}

// ===========================================================================
// RMSNorm: y = x * rsqrt(mean(x^2)+eps) * w. One warp per row, lane-strided D.
// ===========================================================================
template <typename T>
__global__ void rms_norm_fwd(const T* __restrict__ x, const T* __restrict__ w,
                             T* __restrict__ o, int D, float eps) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float ss = 0.0f;
    for (int j = lane; j < D; j += 32) { const float v = tf(x[base + j]); ss += v * v; }
    ss = warp_sum_f(ss);
    const float inv = rsqrtf(ss / float(D) + eps);
    for (int j = lane; j < D; j += 32)
        o[base + j] = ft<T>(tf(x[base + j]) * inv * tf(w[j]));
}

// RMSNorm backward, dX only (Liger factorization; rstd precomputed host-side).
// m = dY*W, s = sum m*x:  dX_i = rstd*m_i - (rstd^3 * s / D) * x_i.
template <typename T>
__global__ void rms_norm_bwd_dx(const T* __restrict__ x, const T* __restrict__ w,
                                const T* __restrict__ dy, const float* __restrict__ rstd,
                                T* __restrict__ dx, int D) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    const float r = rstd[blockIdx.x];
    float s = 0.0f;
    for (int j = lane; j < D; j += 32)
        s += (tf(dy[base + j]) * tf(w[j])) * tf(x[base + j]);
    s = warp_sum_f(s);
    const float c = r * r * r * s / float(D);
    for (int j = lane; j < D; j += 32) {
        const float m = tf(dy[base + j]) * tf(w[j]);
        dx[base + j] = ft<T>(r * m - c * tf(x[base + j]));
    }
}

// Fully-fused RMSNorm backward: rstd in-kernel, dX written, dweight[j] += dY*x*rstd
// via float atomics (dweight (D,) fp32, zeroed first).
template <typename T>
__global__ void rms_norm_bwd_fused(const T* __restrict__ x, const T* __restrict__ w,
                                   const T* __restrict__ dy, T* __restrict__ dx,
                                   float* __restrict__ dweight, int D, float eps) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float ssq = 0.0f, s = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float xv = tf(x[base + j]);
        ssq += xv * xv;
        s   += (tf(dy[base + j]) * tf(w[j])) * xv;
    }
    ssq = warp_sum_f(ssq);
    s   = warp_sum_f(s);
    const float r = rsqrtf(ssq / float(D) + eps);
    const float c = r * r * r * s / float(D);
    for (int j = lane; j < D; j += 32) {
        const float xv = tf(x[base + j]);
        const float m  = tf(dy[base + j]) * tf(w[j]);
        dx[base + j] = ft<T>(r * m - c * xv);
        atomicAdd(dweight + j, tf(dy[base + j]) * xv * r);
    }
}

// ===========================================================================
// LayerNorm: y = (x-mean)*rsqrt(var+eps)*w + b.
// ===========================================================================
template <typename T>
__global__ void layernorm_fwd(const T* __restrict__ x, const T* __restrict__ w,
                              const T* __restrict__ b, T* __restrict__ o,
                              int D, float eps) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float sx = 0.0f;
    for (int j = lane; j < D; j += 32) sx += tf(x[base + j]);
    const float mu = warp_sum_f(sx) / float(D);
    float var = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float d = tf(x[base + j]) - mu;
        var += d * d;
    }
    var = warp_sum_f(var) / float(D);
    const float inv = rsqrtf(var + eps);
    for (int j = lane; j < D; j += 32)
        o[base + j] = ft<T>((tf(x[base + j]) - mu) * inv * tf(w[j]) + tf(b[j]));
}

// LayerNorm backward, dX only. g = dY*W, xhat = (x-mu)*rstd:
//   dX_i = rstd*(g_i - mean(g) - xhat_i*mean(g*xhat)). mean/rstd precomputed.
template <typename T>
__global__ void layernorm_bwd_dx(const T* __restrict__ x, const T* __restrict__ w,
                                 const T* __restrict__ dy, const float* __restrict__ mean,
                                 const float* __restrict__ rstd, T* __restrict__ dx, int D) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    const float mu = mean[blockIdx.x];
    const float r = rstd[blockIdx.x];
    float s1 = 0.0f, s2 = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float g = tf(dy[base + j]) * tf(w[j]);
        const float xhat = (tf(x[base + j]) - mu) * r;
        s1 += g; s2 += g * xhat;
    }
    s1 = warp_sum_f(s1) / float(D);
    s2 = warp_sum_f(s2) / float(D);
    for (int j = lane; j < D; j += 32) {
        const float g = tf(dy[base + j]) * tf(w[j]);
        const float xhat = (tf(x[base + j]) - mu) * r;
        dx[base + j] = ft<T>(r * (g - s1 - xhat * s2));
    }
}

// Fully-fused LayerNorm backward: mean+rstd in-kernel, dX written, dweight[j] +=
// dY*xhat and dbias[j] += dY via float atomics (both (D,) fp32, zeroed first).
template <typename T>
__global__ void layernorm_bwd_fused(const T* __restrict__ x, const T* __restrict__ w,
                                    const T* __restrict__ dy, T* __restrict__ dx,
                                    float* __restrict__ dweight, float* __restrict__ dbias,
                                    int D, float eps) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float sx = 0.0f, sxx = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float xv = tf(x[base + j]);
        sx += xv; sxx += xv * xv;
    }
    sx = warp_sum_f(sx); sxx = warp_sum_f(sxx);
    const float mu = sx / float(D);
    const float r  = rsqrtf(sxx / float(D) - mu * mu + eps);
    float s1 = 0.0f, s2 = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float g = tf(dy[base + j]) * tf(w[j]);
        const float xhat = (tf(x[base + j]) - mu) * r;
        s1 += g; s2 += g * xhat;
    }
    s1 = warp_sum_f(s1) / float(D);
    s2 = warp_sum_f(s2) / float(D);
    for (int j = lane; j < D; j += 32) {
        const float dyv = tf(dy[base + j]);
        const float g = dyv * tf(w[j]);
        const float xhat = (tf(x[base + j]) - mu) * r;
        dx[base + j] = ft<T>(r * (g - s1 - xhat * s2));
        atomicAdd(dweight + j, dyv * xhat);
        atomicAdd(dbias + j, dyv);
    }
}

// ===========================================================================
// add_norm: fused residual add + norm; o = norm(x+res)*w(+b), res_out = x+res.
// fp8 variants emit e4m3 codes (static inv_scale, or dynamic per-row absmax/448
// with the row scale written out). Ref: vLLM fused_add_rms_norm.
// ===========================================================================
template <typename T, bool FP8, bool DYN>
__global__ void rms_norm_add_k(const T* __restrict__ x, const T* __restrict__ residual,
                               const T* __restrict__ w, T* __restrict__ o,
                               uint8_t* __restrict__ codes, T* __restrict__ res_out,
                               float* __restrict__ scale_out, int D, float eps,
                               float inv_scale_static) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float ms = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float v = tf(x[base + j]) + tf(residual[base + j]);
        res_out[base + j] = ft<T>(v);
        ms += v * v;
    }
    ms = warp_sum_f(ms) / float(D);
    const float inv = rsqrtf(ms + eps);
    if (!FP8) {
        for (int j = lane; j < D; j += 32)
            o[base + j] = ft<T>(tf(res_out[base + j]) * inv * tf(w[j]));
        return;
    }
    float inv_scale = inv_scale_static;
    if (DYN) {
        float amax = 0.0f;
        for (int j = lane; j < D; j += 32)
            amax = fmaxf(amax, fabsf(tf(res_out[base + j]) * inv * tf(w[j])));
        amax = warp_max_f(amax);
        const float s = amax / 448.0f;
        inv_scale = s > 0.0f ? 1.0f / s : 0.0f;
        if (lane == 0) scale_out[blockIdx.x] = s;
    }
    for (int j = lane; j < D; j += 32)
        codes[base + j] = tme_e4m3_encode(tf(res_out[base + j]) * inv * tf(w[j]) * inv_scale);
}

template <typename T, bool FP8, bool DYN>
__global__ void layernorm_add_k(const T* __restrict__ x, const T* __restrict__ residual,
                                const T* __restrict__ w, const T* __restrict__ b,
                                T* __restrict__ o, uint8_t* __restrict__ codes,
                                T* __restrict__ res_out, float* __restrict__ scale_out,
                                int D, float eps, float inv_scale_static) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float sx = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float v = tf(x[base + j]) + tf(residual[base + j]);
        res_out[base + j] = ft<T>(v);
        sx += v;
    }
    const float mu = warp_sum_f(sx) / float(D);
    float var = 0.0f;
    for (int j = lane; j < D; j += 32) {
        const float d = tf(res_out[base + j]) - mu;
        var += d * d;
    }
    var = warp_sum_f(var) / float(D);
    const float inv = rsqrtf(var + eps);
    if (!FP8) {
        for (int j = lane; j < D; j += 32)
            o[base + j] = ft<T>((tf(res_out[base + j]) - mu) * inv * tf(w[j]) + tf(b[j]));
        return;
    }
    float inv_scale = inv_scale_static;
    if (DYN) {
        float amax = 0.0f;
        for (int j = lane; j < D; j += 32)
            amax = fmaxf(amax, fabsf((tf(res_out[base + j]) - mu) * inv * tf(w[j]) + tf(b[j])));
        amax = warp_max_f(amax);
        const float s = amax / 448.0f;
        inv_scale = s > 0.0f ? 1.0f / s : 0.0f;
        if (lane == 0) scale_out[blockIdx.x] = s;
    }
    for (int j = lane; j < D; j += 32)
        codes[base + j] = tme_e4m3_encode(((tf(res_out[base + j]) - mu) * inv * tf(w[j]) + tf(b[j])) * inv_scale);
}

// ===========================================================================
// Softmax over the last axis. One warp per row.
// ===========================================================================
template <typename T>
__global__ void softmax_fwd(const T* __restrict__ x, T* __restrict__ o, int D) {
    const long base = (long)blockIdx.x * D;
    const int lane = threadIdx.x;
    float m = -3.4028234663852886e38f;
    for (int j = lane; j < D; j += 32) m = fmaxf(m, tf(x[base + j]));
    m = warp_max_f(m);
    float s = 0.0f;
    for (int j = lane; j < D; j += 32) s += expf(tf(x[base + j]) - m);
    s = warp_sum_f(s);
    for (int j = lane; j < D; j += 32)
        o[base + j] = ft<T>(expf(tf(x[base + j]) - m) / s);
}

// ===========================================================================
// GELU (tanh approximation) fwd/bwd, flat elementwise.
//   y = 0.5*x*(1 + tanh(k*(x + a*x^3))), k = sqrt(2/pi), a = 0.044715
// ===========================================================================
__device__ __forceinline__ float gelu_tanh_f(float x) {
    const float k = 0.7978845608028654f, a = 0.044715f;
    return 0.5f * x * (1.0f + tanhf(k * (x + a * x * x * x)));
}
__device__ __forceinline__ float gelu_bwd_grad(float xv) {
    const float k = 0.7978845608028654f, a = 0.044715f;
    const float inner = k * (xv + a * xv * xv * xv);
    const float t = tanhf(inner);
    const float dinner = k * (1.0f + 3.0f * a * xv * xv);
    return 0.5f * (1.0f + t) + 0.5f * xv * (1.0f - t * t) * dinner;
}
template <typename T>
__global__ void gelu_fwd(const T* __restrict__ x, T* __restrict__ o, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) o[i] = ft<T>(gelu_tanh_f(tf(x[i])));
}
template <typename T>
__global__ void gelu_bwd(const T* __restrict__ x, const T* __restrict__ dy,
                         T* __restrict__ dx, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dx[i] = ft<T>(tf(dy[i]) * gelu_bwd_grad(tf(x[i])));
}

// ===========================================================================
// GLU family, 6 modes: 0 reglu, 1 geglu(tanh), 2 swiglu, 3 swiglu_oai(alpha,
// limit), 4 geglu_erf (A&S erf approx; bwd differentiates the APPROXIMATION,
// bit-consistent with the fwd), 5 geglu_quick. out = act(x)*gate.
// ===========================================================================
__device__ __forceinline__ float glu_tanh(float x) {
    return 1.0f - 2.0f / (expf(x + x) + 1.0f);
}
#define GLU_ERF_P  0.3275911f
#define GLU_ERF_A1 0.254829592f
#define GLU_ERF_A2 (-0.284496736f)
#define GLU_ERF_A3 1.421413741f
#define GLU_ERF_A4 (-1.453152027f)
#define GLU_ERF_A5 1.061405429f
__device__ __forceinline__ float glu_erf_approx(float x) {
    const float sx = x < 0.0f ? -1.0f : 1.0f;
    x = fabsf(x);
    const float t = 1.0f / (1.0f + GLU_ERF_P * x);
    const float poly = (((((GLU_ERF_A5 * t + GLU_ERF_A4) * t) + GLU_ERF_A3) * t + GLU_ERF_A2) * t + GLU_ERF_A1) * t;
    return sx * (1.0f - poly * expf(-x * x));
}
// Exact analytic derivative of glu_erf_approx (see TM glu.metal): with t = 1/(1+P|x|),
// d/dx [1 - poly*exp(-x^2)] = exp(-x^2) * (2|x|*poly + P*t^2*dpoly/dt).
__device__ __forceinline__ float glu_erf_approx_deriv(float x) {
    const float ax = fabsf(x);
    const float t = 1.0f / (1.0f + GLU_ERF_P * ax);
    const float poly = (((((GLU_ERF_A5 * t + GLU_ERF_A4) * t) + GLU_ERF_A3) * t + GLU_ERF_A2) * t + GLU_ERF_A1) * t;
    const float dpoly_dt = (((5.0f * GLU_ERF_A5 * t + 4.0f * GLU_ERF_A4) * t + 3.0f * GLU_ERF_A3) * t
                            + 2.0f * GLU_ERF_A2) * t + GLU_ERF_A1;
    return expf(-ax * ax) * (2.0f * ax * poly + GLU_ERF_P * t * t * dpoly_dt);
}
#define GLU_GELU_COEF_A 0.044715f
#define GLU_GELU_QUICK_COEF (-1.702f)
#define GLU_SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GLU_SQRT_2_INV 0.70710678118654752440084436210484f

__device__ __forceinline__ float glu_eval(int mode, float x0, float x1, float alpha, float limit) {
    if (mode == 0) return x0 * x1 * (x0 > 0.0f ? 1.0f : 0.0f);
    if (mode == 1) {
        const float inner = GLU_SQRT_2_OVER_PI * x0 * (1.0f + GLU_GELU_COEF_A * x0 * x0);
        return 0.5f * x0 * (1.0f + glu_tanh(inner)) * x1;
    }
    if (mode == 2) return (x0 / (1.0f + expf(-x0))) * x1;
    if (mode == 3) {
        x0 = fminf(x0, limit);
        x1 = fmaxf(fminf(x1, limit), -limit);
        return (x0 / (1.0f + expf(-x0 * alpha))) * (1.0f + x1);
    }
    if (mode == 4) return 0.5f * x0 * (1.0f + glu_erf_approx(x0 * GLU_SQRT_2_INV)) * x1;
    return (x0 * (1.0f / (1.0f + expf(GLU_GELU_QUICK_COEF * x0)))) * x1;
}

__device__ __forceinline__ void glu_grad(int mode, float a, float b, float dc,
                                         float alpha, float limit, float& da, float& db) {
    if (mode == 0) {                       // reglu
        const float m = a > 0.0f ? 1.0f : 0.0f;
        db = dc * a * m;
        da = dc * b * m;
        return;
    }
    if (mode == 1) {                       // geglu (tanh)
        const float inner = GLU_SQRT_2_OVER_PI * a * (1.0f + GLU_GELU_COEF_A * a * a);
        const float t = glu_tanh(inner);
        const float dz = GLU_SQRT_2_OVER_PI * (1.0f + 3.0f * GLU_GELU_COEF_A * a * a);
        db = dc * (0.5f * a * (1.0f + t));
        da = dc * b * (0.5f * (1.0f + t) + 0.5f * a * (1.0f - t * t) * dz);
        return;
    }
    if (mode == 2) {                       // swiglu
        const float s = 1.0f / (1.0f + expf(-a));
        db = dc * (a * s);
        da = dc * b * (s * (1.0f + a * (1.0f - s)));
        return;
    }
    if (mode == 3) {                       // swiglu_oai (clamped swish * (1+clamp(b)))
        const float x0 = fminf(a, limit);
        const float x1 = fmaxf(fminf(b, limit), -limit);
        const float s0 = 1.0f / (1.0f + expf(-x0 * alpha));
        const float f = x0 * s0;
        const float ind_a = a < limit ? 1.0f : 0.0f;
        const float ind_b = (b < limit && b > -limit) ? 1.0f : 0.0f;
        db = dc * f * ind_b;
        da = dc * (1.0f + x1) * (s0 + x0 * alpha * s0 * (1.0f - s0)) * ind_a;
        return;
    }
    if (mode == 4) {                       // geglu_erf
        const float u = a * GLU_SQRT_2_INV;
        const float e = glu_erf_approx(u);
        const float de = glu_erf_approx_deriv(u) * GLU_SQRT_2_INV;
        db = dc * (0.5f * a * (1.0f + e));
        da = dc * b * (0.5f * (1.0f + e) + 0.5f * a * de);
        return;
    }
    // mode 5: geglu_quick
    const float s = 1.0f / (1.0f + expf(GLU_GELU_QUICK_COEF * a));
    db = dc * (a * s);
    da = dc * b * (s + a * (-GLU_GELU_QUICK_COEF) * s * (1.0f - s));
}

template <typename T>
__global__ void glu_fwd(const T* __restrict__ x, const T* __restrict__ gate,
                        T* __restrict__ out, long n, int mode, float alpha, float limit) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ft<T>(glu_eval(mode, tf(x[i]), tf(gate[i]), alpha, limit));
}
template <typename T>
__global__ void glu_bwd(const T* __restrict__ x, const T* __restrict__ gate,
                        const T* __restrict__ dc, T* __restrict__ da, T* __restrict__ db,
                        long n, int mode, float alpha, float limit) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float dai, dbi;
        glu_grad(mode, tf(x[i]), tf(gate[i]), tf(dc[i]), alpha, limit, dai, dbi);
        da[i] = ft<T>(dai);
        db[i] = ft<T>(dbi);
    }
}

// ===========================================================================
// Inverted dropout, mask-free: keep = rng_uniform(seed, i, 0) >= p; the bwd
// recomputes the identical mask from the seed. inv_keep = 1/(1-p) from host.
// ===========================================================================
template <typename T>
__global__ void dropout_fwd(const T* __restrict__ x, T* __restrict__ out,
                            uint32_t seed, float p, float inv_keep, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = rng_uniform(seed, uint32_t(i), 0u);
    out[i] = (u >= p) ? ft<T>(tf(x[i]) * inv_keep) : ft<T>(0.0f);
}
template <typename T>
__global__ void dropout_bwd(const T* __restrict__ dy, T* __restrict__ dx,
                            uint32_t seed, float p, float inv_keep, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = rng_uniform(seed, uint32_t(i), 0u);
    dx[i] = (u >= p) ? ft<T>(tf(dy[i]) * inv_keep) : ft<T>(0.0f);
}

// ===========================================================================
// Fused cross-entropy over the vocab axis: online logsumexp, probs never
// materialized. ignore_index, label smoothing, z_loss (z*lse^2), Gemma-2
// softcap (cap*tanh(z/cap), grad through the tanh). lse in natural log.
// 1-warp/row kernels + _mw 4-warp variants (128 threads, shared-mem merge of
// the partial (max, sumexp, sum) states) for small-T / large-V shapes.
// ===========================================================================
#define CE_NEG_INF (-3.4028234663852886e38f)
__device__ __forceinline__ float ce_softcap(float z, float softcap) {
    return (softcap > 0.0f) ? softcap * tanhf(z / softcap) : z;
}

template <typename T>
__global__ void cross_entropy_fwd(const T* __restrict__ logits, const int* __restrict__ targets,
                                  float* __restrict__ loss, float* __restrict__ lse_out,
                                  int V, int ignore_index, float label_smoothing,
                                  float z_loss, float softcap) {
    const long base = (long)blockIdx.x * V;
    const int lane = threadIdx.x;
    const int y = targets[blockIdx.x];
    if (y == ignore_index) {
        if (lane == 0) { loss[blockIdx.x] = 0.0f; lse_out[blockIdx.x] = 0.0f; }
        return;
    }
    float m = CE_NEG_INF, l = 0.0f, sx = 0.0f;
    for (int i = lane; i < V; i += 32) {
        const float x = ce_softcap(tf(logits[base + i]), softcap);
        sx += x;
        const float nm = fmaxf(m, x);
        l = l * expf(m - nm) + expf(x - nm);
        m = nm;
    }
    const float M = warp_max_f(m);
    l = warp_sum_f(l * expf(m - M));
    sx = warp_sum_f(sx);
    const float lse = M + logf(l);
    const float x_y = ce_softcap(tf(logits[base + y]), softcap);
    const float eps = label_smoothing;
    float ls = (1.0f - eps) * (lse - x_y);
    if (eps > 0.0f) ls += eps * (lse - sx / float(V));
    if (z_loss > 0.0f) ls += z_loss * lse * lse;
    if (lane == 0) { loss[blockIdx.x] = ls; lse_out[blockIdx.x] = lse; }
}

template <typename T>
__global__ void cross_entropy_bwd(const T* __restrict__ logits, const int* __restrict__ targets,
                                  const float* __restrict__ lse_in, const float* __restrict__ grad_out,
                                  T* __restrict__ grad_logits, int V, int ignore_index,
                                  float label_smoothing, float z_loss, float softcap) {
    const long base = (long)blockIdx.x * V;
    const int lane = threadIdx.x;
    const int y = targets[blockIdx.x];
    if (y == ignore_index) {
        for (int i = lane; i < V; i += 32) grad_logits[base + i] = ft<T>(0.0f);
        return;
    }
    const float lse = lse_in[blockIdx.x];
    const float go = grad_out[blockIdx.x];
    const float eps = label_smoothing;
    const float zc = 1.0f + 2.0f * z_loss * lse;
    const float smooth = eps / float(V);
    for (int i = lane; i < V; i += 32) {
        const float capped = ce_softcap(tf(logits[base + i]), softcap);
        const float p = expf(capped - lse);
        float g = zc * p - smooth - (1.0f - eps) * ((i == y) ? 1.0f : 0.0f);
        if (softcap > 0.0f) { const float t = capped / softcap; g *= (1.0f - t * t); }
        grad_logits[base + i] = ft<T>(g * go);
    }
}

#define CE_NW 4
template <typename T>
__global__ void cross_entropy_fwd_mw(const T* __restrict__ logits, const int* __restrict__ targets,
                                     float* __restrict__ loss, float* __restrict__ lse_out,
                                     int V, int ignore_index, float label_smoothing,
                                     float z_loss, float softcap) {
    __shared__ float sh_m[CE_NW], sh_l[CE_NW], sh_sx[CE_NW];
    const long base = (long)blockIdx.x * V;
    const int tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32;
    const int y = targets[blockIdx.x];
    if (y == ignore_index) {
        if (tid == 0) { loss[blockIdx.x] = 0.0f; lse_out[blockIdx.x] = 0.0f; }
        return;
    }
    float m = CE_NEG_INF, l = 0.0f, sx = 0.0f;
    for (int i = tid; i < V; i += 32 * CE_NW) {
        const float x = ce_softcap(tf(logits[base + i]), softcap);
        sx += x;
        const float nm = fmaxf(m, x);
        l = l * expf(m - nm) + expf(x - nm);
        m = nm;
    }
    const float Mw = warp_max_f(m);
    l = warp_sum_f(l * expf(m - Mw));
    sx = warp_sum_f(sx);
    if (lane == 0) { sh_m[warp] = Mw; sh_l[warp] = l; sh_sx[warp] = sx; }
    __syncthreads();
    if (tid == 0) {
        float M = sh_m[0], L = sh_l[0], SX = sh_sx[0];
        for (int w = 1; w < CE_NW; ++w) {
            const float nM = fmaxf(M, sh_m[w]);
            L = L * expf(M - nM) + sh_l[w] * expf(sh_m[w] - nM);
            M = nM;
            SX += sh_sx[w];
        }
        const float lse = M + logf(L);
        const float x_y = ce_softcap(tf(logits[base + y]), softcap);
        const float eps = label_smoothing;
        float ls = (1.0f - eps) * (lse - x_y);
        if (eps > 0.0f) ls += eps * (lse - SX / float(V));
        if (z_loss > 0.0f) ls += z_loss * lse * lse;
        loss[blockIdx.x] = ls;
        lse_out[blockIdx.x] = lse;
    }
}

template <typename T>
__global__ void cross_entropy_bwd_mw(const T* __restrict__ logits, const int* __restrict__ targets,
                                     const float* __restrict__ lse_in, const float* __restrict__ grad_out,
                                     T* __restrict__ grad_logits, int V, int ignore_index,
                                     float label_smoothing, float z_loss, float softcap) {
    const long base = (long)blockIdx.x * V;
    const int tid = threadIdx.x;
    const int y = targets[blockIdx.x];
    if (y == ignore_index) {
        for (int i = tid; i < V; i += 32 * CE_NW) grad_logits[base + i] = ft<T>(0.0f);
        return;
    }
    const float lse = lse_in[blockIdx.x];
    const float go = grad_out[blockIdx.x];
    const float eps = label_smoothing;
    const float zc = 1.0f + 2.0f * z_loss * lse;
    const float smooth = eps / float(V);
    for (int i = tid; i < V; i += 32 * CE_NW) {
        const float capped = ce_softcap(tf(logits[base + i]), softcap);
        const float p = expf(capped - lse);
        float g = zc * p - smooth - (1.0f - eps) * ((i == y) ? 1.0f : 0.0f);
        if (softcap > 0.0f) { const float t = capped / softcap; g *= (1.0f - t * t); }
        grad_logits[base + i] = ft<T>(g * go);
    }
}

// ===========================================================================
// Embedding: lookup gather (+optional pos table, neg/OOB -> zeros), multimodal
// span select + on-device src map build, and TWO backwards (atomic scatter-add
// into zeroed fp32 dtable; sorted-segment sole-writer, no atomics).
// One block per token, threads striding D.
// ===========================================================================
template <typename T>
__global__ void embedding_lookup(const int* __restrict__ token_ids, const T* __restrict__ table,
                                 const T* __restrict__ pos_table, T* __restrict__ out,
                                 int D, int vocab, float scale, int use_pos) {
    const int t = blockIdx.x;
    const int tok = token_ids[t];
    const bool valid = tok >= 0 && tok < vocab;
    const T* trow = table + (long)(valid ? tok : 0) * D;
    const T* prow = pos_table + (long)t * D;
    T* orow = out + (long)t * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v = valid ? tf(trow[d]) * scale : 0.0f;
        if (use_pos) v += tf(prow[d]);
        orow[d] = ft<T>(v);
    }
}

// out[t] = (src[t] >= 0) ? modal[src[t]] : text[t].
template <typename T>
__global__ void merge_multimodal_spans(const T* __restrict__ text, const T* __restrict__ modal,
                                       const int* __restrict__ src, T* __restrict__ out,
                                       int D, int n_modal) {
    const int t = blockIdx.x;
    const int sm = src[t];
    const bool use_modal = sm >= 0 && sm < n_modal;
    const T* srow = use_modal ? (modal + (long)sm * D) : (text + (long)t * D);
    T* orow = out + (long)t * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) orow[d] = srow[d];
}

// src[t] = modal_starts[k] + (t - span_offsets[k]) if t is inside span k, else -1.
__global__ void build_multimodal_src(const int* __restrict__ span_offsets,
                                     const int* __restrict__ span_lengths,
                                     const int* __restrict__ modal_starts,
                                     int* __restrict__ src, int num_spans, int num_tok) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= num_tok) return;
    int s = -1;
    for (int k = 0; k < num_spans; ++k) {
        const int o = t - span_offsets[k];
        if (o >= 0 && o < span_lengths[k]) { s = modal_starts[k] + o; break; }
    }
    src[t] = s;
}

// dtable[token_ids[t]*D + d] += scale * dY[t*D + d]; dtable (vocab,D) fp32 zeroed.
template <typename T>
__global__ void embedding_backward(const int* __restrict__ token_ids, const T* __restrict__ dY,
                                   float* __restrict__ dtable, int D, int vocab, float scale) {
    const int t = blockIdx.x;
    const int tok = token_ids[t];
    if (tok < 0 || tok >= vocab) return;
    const long trow = (long)tok * D;
    const long drow = (long)t * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        atomicAdd(dtable + trow + d, tf(dY[drow + d]) * scale);
}

// Sorted-segment backward: host argsorts ids; only each id's segment start
// accumulates its run and writes the row once (sole writer, no atomics).
template <typename T>
__global__ void embedding_backward_sorted(const int* __restrict__ sorted_ids,
                                          const int* __restrict__ perm, const T* __restrict__ dY,
                                          float* __restrict__ dtable, int D, int vocab,
                                          int n_tok, float scale) {
    const int i = blockIdx.x;
    const int id = sorted_ids[i];
    if (id < 0 || id >= vocab) return;
    if (i > 0 && sorted_ids[i - 1] == id) return;
    const long orow = (long)id * D;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int j = i; j < n_tok && sorted_ids[j] == id; ++j)
            acc += tf(dY[(long)perm[j] * D + d]);
        dtable[orow + d] = acc * scale;
    }
}

// ===========================================================================
// Walsh-Hadamard transform over the final axis. LPR lanes per row, each lane
// owns E = D/LPR consecutive elements; low log2(E) index bits butterflied in
// registers, lane bits via __shfl_xor_sync. Self-inverse (fwd only); scale on
// store. D=64 LPR=8, D=128 LPR=16, D=256/512 LPR=32. 128-thread blocks (4
// warps); shuffles stay within each warp's row group.
// ===========================================================================
template <typename T, int D, int LPR>
__global__ void hadamard_k(const T* __restrict__ x, T* __restrict__ out,
                           float scale, int nrows) {
    constexpr int E = D / LPR;
    constexpr int RSG = 32 / LPR;       // rows per warp
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int warps = blockDim.x / 32;
    const int rsg = lane / LPR;
    const int lin = lane % LPR;
    const long row = ((long)blockIdx.x * warps + warp) * RSG + rsg;
    const bool live = row < (long)nrows;

    float v[E];
    if (live) {
        const long base = row * D + (long)lin * E;
        #pragma unroll
        for (int i = 0; i < E; ++i) v[i] = tf(x[base + i]);
    } else {
        #pragma unroll
        for (int i = 0; i < E; ++i) v[i] = 0.0f;   // dead rows still ride the shuffles
    }
    #pragma unroll
    for (int h = 1; h < E; h <<= 1) {
        #pragma unroll
        for (int i = 0; i < E; ++i) {
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
    }
    #pragma unroll
    for (int m = 1; m < LPR; m <<= 1) {
        const bool upper = (lin & m) != 0;
        #pragma unroll
        for (int i = 0; i < E; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = upper ? (p - v[i]) : (v[i] + p);
        }
    }
    if (live) {
        const long base = row * D + (long)lin * E;
        #pragma unroll
        for (int i = 0; i < E; ++i) out[base + i] = ft<T>(v[i] * scale);
    }
}

// ===========================================================================
// AdamW step (decoupled weight decay), pure functional: new param/m/v out.
// param/grad in T; moments fp32; bc1 = 1-beta1^t, bc2 = 1-beta2^t from host.
// ===========================================================================
template <typename T>
__global__ void adamw_step(const T* __restrict__ param, const T* __restrict__ grad,
                           const float* __restrict__ m_in, const float* __restrict__ v_in,
                           T* __restrict__ param_out, float* __restrict__ m_out,
                           float* __restrict__ v_out, float lr, float beta1, float beta2,
                           float eps, float wd, float bc1, float bc2, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = tf(grad[i]);
    const float m = beta1 * m_in[i] + (1.0f - beta1) * g;
    const float v = beta2 * v_in[i] + (1.0f - beta2) * g * g;
    const float mhat = m / bc1;
    const float vhat = v / bc2;
    const float p = tf(param[i]);
    param_out[i] = ft<T>(p - lr * (mhat / (sqrtf(vhat) + eps) + wd * p));
    m_out[i] = m;
    v_out[i] = v;
}

// ===========================================================================
// Elementwise add (the add_rt smoke kernel).
// ===========================================================================
template <typename T>
__global__ void add_ew(const T* __restrict__ x, const T* __restrict__ y,
                       T* __restrict__ out, long n) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = ft<T>(tf(x[i]) + tf(y[i]));
}

} // namespace tme
