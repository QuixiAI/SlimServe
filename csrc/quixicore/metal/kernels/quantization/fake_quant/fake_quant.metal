#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

template <typename T>
kernel void fake_quant_int8(device const T *x     [[buffer(0)]],
                            device bf16    *x_q   [[buffer(1)]],
                            device char    *codes [[buffer(2)]],
                            device float   *scale [[buffer(3)]],
                            constant int   &D     [[buffer(4)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = D / 4;
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4 *)(x + base))[c]);
        amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    }
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    const float sh = float(half(s));
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4 *)(x + base))[c]) * inv;
        const char4 q = char4(tk_int8_encode(v.x), tk_int8_encode(v.y),
                              tk_int8_encode(v.z), tk_int8_encode(v.w));
        ((device char4 *)(codes + base))[c] = q;
        ((device bf16_4 *)(x_q + base))[c] =
            bf16_4(float4(float(q.x), float(q.y), float(q.z), float(q.w)) * sh);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

template <typename T>
kernel void silu_mul_fake_quant_int8(device const T *x     [[buffer(0)]],
                                     device const T *gate  [[buffer(1)]],
                                     device bf16    *x_q   [[buffer(2)]],
                                     device char    *codes [[buffer(3)]],
                                     device float   *scale [[buffer(4)]],
                                     constant int   &D     [[buffer(5)]],
                                     constant int   &mode  [[buffer(6)]],
                                     constant float &alpha [[buffer(7)]],
                                     constant float &limit [[buffer(8)]],
                                     uint row [[threadgroup_position_in_grid]],
                                     uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = D / 4;
    const int gmode = (mode == 1) ? 3 : 2;
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 xv = float4(((device const T4 *)(x + base))[c]);
        const float4 gv = float4(((device const T4 *)(gate + base))[c]);
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            amax = max(amax, fabs(glu_eval(gmode, xv[j], gv[j], alpha, limit)));
        }
    }
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    const float sh = float(half(s));
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 xv = float4(((device const T4 *)(x + base))[c]);
        const float4 gv = float4(((device const T4 *)(gate + base))[c]);
        char4 q;
        float4 dq;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            q[j] = tk_int8_encode(glu_eval(gmode, xv[j], gv[j], alpha, limit) * inv);
            dq[j] = float(q[j]) * sh;
        }
        ((device char4 *)(codes + base))[c] = q;
        ((device bf16_4 *)(x_q + base))[c] = bf16_4(dq);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

#define instantiate_fake_quant(type_name, T)                                      \
  template [[host_name("fake_quant_int8_" #type_name)]] [[kernel]] void           \
  fake_quant_int8<T>(device const T *x [[buffer(0)]],                             \
                     device bf16 *x_q [[buffer(1)]],                              \
                     device char *codes [[buffer(2)]],                             \
                     device float *scale [[buffer(3)]],                            \
                     constant int &D [[buffer(4)]],                                \
                     uint row [[threadgroup_position_in_grid]],                    \
                     uint lane [[thread_index_in_simdgroup]]);                     \
  template [[host_name("silu_mul_fake_quant_int8_" #type_name)]] [[kernel]] void  \
  silu_mul_fake_quant_int8<T>(device const T *x [[buffer(0)]],                    \
                              device const T *gate [[buffer(1)]],                 \
                              device bf16 *x_q [[buffer(2)]],                     \
                              device char *codes [[buffer(3)]],                   \
                              device float *scale [[buffer(4)]],                  \
                              constant int &D [[buffer(5)]],                      \
                              constant int &mode [[buffer(6)]],                   \
                              constant float &alpha [[buffer(7)]],                \
                              constant float &limit [[buffer(8)]],                \
                              uint row [[threadgroup_position_in_grid]],          \
                              uint lane [[thread_index_in_simdgroup]]);

instantiate_fake_quant(float32, float)
instantiate_fake_quant(float16, half)
instantiate_fake_quant(bfloat16, bf16)
