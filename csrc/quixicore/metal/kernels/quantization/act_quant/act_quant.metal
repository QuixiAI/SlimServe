#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Fused gated-activation -> dynamic quantization epilogues: act = glu_eval(mode, x, gate)
// computed twice (amax pass + encode pass — memory-bound, the activation is cheap) with the
// intermediate (rows, D) bf16 round-trip eliminated. Buffer order matches tk.glu(x, gate):
// x is the ACTIVATED operand, gate the multiplier. mode: 0 = swiglu (silu(x)*gate),
// 1 = swiglu_oai (gpt-oss clamped, alpha-scaled sigmoid, (1+gate)) — mapped onto the shared
// glu_eval modes 2 / 3 so the math is definitionally identical to kernels/glu.
//   per-token:  scale (rows,)      = amax/QMAX             -> qgemm_fp8_scaled / qgemm_w8a8
//   per-group:  scale (rows, D/G)  (G % 4 == 0, canonical 128; ue8m0 rounds up to 2^k)
// ---------------------------------------------------------------------------

METAL_FUNC float actq_eval(int mode, float x0, float x1, float alpha, float limit) {
    return glu_eval(mode == 1 ? 3 : 2, x0, x1, alpha, limit);
}

template <int MODE>
METAL_FUNC float actq_eval_mode(float x0, float x1, float alpha, float limit) {
    return glu_eval(MODE == 1 ? 3 : 2, x0, x1, alpha, limit);
}

template <typename T>
kernel void silu_mul_quant_fp8(device const T *x     [[buffer(0)]],
                               device const T *gate  [[buffer(1)]],
                               device uchar   *codes [[buffer(2)]],
                               device float   *scale [[buffer(3)]],
                               constant int   &D     [[buffer(4)]],
                               constant int   &mode  [[buffer(5)]],
                               constant float &alpha [[buffer(6)]],
                               constant float &limit [[buffer(7)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = (D % 4 == 0) ? D / 4 : 0;     // vec4 path when D is 4-aligned
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 xv = float4(((device const T4*)(x + base))[c]);
        const float4 gv = float4(((device const T4*)(gate + base))[c]);
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) amax = max(amax, fabs(actq_eval(mode, xv[j], gv[j], alpha, limit)));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(actq_eval(mode, float(x[base + i]), float(gate[base + i]), alpha, limit)));
    }
    amax = simd_max(amax);
    const float s = amax / 448.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 xv = float4(((device const T4*)(x + base))[c]);
        const float4 gv = float4(((device const T4*)(gate + base))[c]);
        uchar4 out;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) out[j] = tk_e4m3_encode(actq_eval(mode, xv[j], gv[j], alpha, limit) * inv);
        ((device uchar4*)(codes + base))[c] = out;
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        codes[base + i] = tk_e4m3_encode(actq_eval(mode, float(x[base + i]), float(gate[base + i]), alpha, limit) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

template <typename T>
kernel void silu_mul_quant_int8(device const T *x     [[buffer(0)]],
                                device const T *gate  [[buffer(1)]],
                                device char    *codes [[buffer(2)]],
                                device float   *scale [[buffer(3)]],
                                constant int   &D     [[buffer(4)]],
                                constant int   &mode  [[buffer(5)]],
                                constant float &alpha [[buffer(6)]],
                                constant float &limit [[buffer(7)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = (D % 4 == 0) ? D / 4 : 0;
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 xv = float4(((device const T4*)(x + base))[c]);
        const float4 gv = float4(((device const T4*)(gate + base))[c]);
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) amax = max(amax, fabs(actq_eval(mode, xv[j], gv[j], alpha, limit)));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(actq_eval(mode, float(x[base + i]), float(gate[base + i]), alpha, limit)));
    }
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 xv = float4(((device const T4*)(x + base))[c]);
        const float4 gv = float4(((device const T4*)(gate + base))[c]);
        char4 out;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) out[j] = tk_int8_encode(actq_eval(mode, xv[j], gv[j], alpha, limit) * inv);
        ((device char4*)(codes + base))[c] = out;
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        codes[base + i] = tk_int8_encode(actq_eval(mode, float(x[base + i]), float(gate[base + i]), alpha, limit) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

template <typename T, int MODE>
kernel void silu_mul_quant_fp8_group(device const T *x     [[buffer(0)]],
                                     device const T *gate  [[buffer(1)]],
                                     device uchar   *codes [[buffer(2)]],
                                     device float   *scale [[buffer(3)]],
                                     constant int   &D     [[buffer(4)]],
                                     constant int   &G     [[buffer(5)]],
                                     constant int   &ue8m0 [[buffer(6)]],
                                     constant int   &mode  [[buffer(7)]],
                                     constant float &alpha [[buffer(8)]],
                                     constant float &limit [[buffer(9)]],
                                     uint row  [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int ngroups = D / G;
    const int gchunks = G / 4;                       // G % 4 == 0 (host-enforced)
    for (int g = 0; g < ngroups; ++g) {
        const long gbase = base + (long)g * G;
        float amax = 0.0f;
        for (int c = (int)lane; c < gchunks; c += 32) {
            const float4 xv = float4(((device const T4*)(x + gbase))[c]);
            const float4 gv = float4(((device const T4*)(gate + gbase))[c]);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) amax = max(amax, fabs(actq_eval_mode<MODE>(xv[j], gv[j], alpha, limit)));
        }
        amax = simd_max(amax);
        float s = amax / 448.0f;
        if (ue8m0 != 0 && amax > 0.0f) {
            s = exp2(ceil(log2(max(amax, 1e-10f) / 448.0f)));
        }
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        for (int c = (int)lane; c < gchunks; c += 32) {
            const float4 xv = float4(((device const T4*)(x + gbase))[c]);
            const float4 gv = float4(((device const T4*)(gate + gbase))[c]);
            uchar4 out;
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) out[j] = tk_e4m3_encode(actq_eval_mode<MODE>(xv[j], gv[j], alpha, limit) * inv);
            ((device uchar4*)(codes + gbase))[c] = out;
        }
        if (lane == 0) {
            scale[(long)row * ngroups + g] = s;
        }
    }
}

#define instantiate_act_quant_group(MODE_NAME, MODE_ID, type_name, T)                \
  template [[host_name("silu_mul_quant_fp8_group_" #MODE_NAME "_" #type_name)]] [[kernel]] void\
  silu_mul_quant_fp8_group<T, MODE_ID>(device const T *x [[buffer(0)]],              \
                              device const T *gate [[buffer(1)]],                    \
                              device uchar *codes [[buffer(2)]],                     \
                              device float *scale [[buffer(3)]],                     \
                              constant int &D [[buffer(4)]],                         \
                              constant int &G [[buffer(5)]],                         \
                              constant int &ue8m0 [[buffer(6)]],                     \
                              constant int &mode [[buffer(7)]],                      \
                              constant float &alpha [[buffer(8)]],                   \
                              constant float &limit [[buffer(9)]],                   \
                              uint row [[threadgroup_position_in_grid]],             \
                              uint lane [[thread_index_in_simdgroup]]);

#define instantiate_act_quant_group_mode(MODE_NAME, MODE_ID)             \
  instantiate_act_quant_group(MODE_NAME, MODE_ID, float32, float)        \
  instantiate_act_quant_group(MODE_NAME, MODE_ID, float16, half)         \
  instantiate_act_quant_group(MODE_NAME, MODE_ID, bfloat16, bf16)

instantiate_act_quant_group_mode(swiglu, 0)
instantiate_act_quant_group_mode(swiglu_oai, 1)

#define instantiate_act_quant_token(type_name, T)                                \
  template [[host_name("silu_mul_quant_fp8_" #type_name)]] [[kernel]] void       \
  silu_mul_quant_fp8<T>(device const T *x [[buffer(0)]],                        \
                        device const T *gate [[buffer(1)]],                      \
                        device uchar *codes [[buffer(2)]],                       \
                        device float *scale [[buffer(3)]],                       \
                        constant int &D [[buffer(4)]],                           \
                        constant int &mode [[buffer(5)]],                        \
                        constant float &alpha [[buffer(6)]],                     \
                        constant float &limit [[buffer(7)]],                     \
                        uint row [[threadgroup_position_in_grid]],               \
                        uint lane [[thread_index_in_simdgroup]]);                \
  template [[host_name("silu_mul_quant_int8_" #type_name)]] [[kernel]] void      \
  silu_mul_quant_int8<T>(device const T *x [[buffer(0)]],                       \
                         device const T *gate [[buffer(1)]],                     \
                         device char *codes [[buffer(2)]],                       \
                         device float *scale [[buffer(3)]],                      \
                         constant int &D [[buffer(4)]],                          \
                         constant int &mode [[buffer(5)]],                       \
                         constant float &alpha [[buffer(6)]],                    \
                         constant float &limit [[buffer(7)]],                    \
                         uint row [[threadgroup_position_in_grid]],              \
                         uint lane [[thread_index_in_simdgroup]]);

instantiate_act_quant_token(float32, float)
instantiate_act_quant_token(float16, half)
instantiate_act_quant_token(bfloat16, bf16)
