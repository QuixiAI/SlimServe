#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Runtime per-token (per-row) activation quantization. These are the first
// GPU-side quantizers in ThunderMittens (everything in tk/quant.py is host numpy).
//
// One simdgroup (32 lanes) processes one row of length D (any D): cross-lane
// simd_max gives the per-row absmax, scale = absmax / QMAX, then each element is
// encoded (round-to-nearest) to fp8 e4m3 (QMAX=448) or symmetric int8 (QMAX=127).
//
//   codes[row, i] = encode(x[row, i] / scale[row]) ;  scale[row] = absmax(row) / QMAX
//
// Reconstruct as scale[row] * decode(codes[row, i]). Ref: vLLM
// dynamic_per_token_scaled_fp8_quant (quantization/w8a8/fp8/common.cu).
// ---------------------------------------------------------------------------

template <typename T>
kernel void quantize_per_token_fp8(device const T *x     [[buffer(0)]],
                                   device uchar   *codes [[buffer(1)]],
                                   device float   *scale [[buffer(2)]],
                                   constant int   &D     [[buffer(3)]],
                                   uint row  [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = (D % 4 == 0) ? D / 4 : 0;   // vec4 path only for aligned D
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]);
        amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(float(x[base + i])));
    }
    amax = simd_max(amax);
    const float s = amax / 448.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]) * inv;
        ((device uchar4*)(codes + base))[c] =
            uchar4(tk_e4m3_encode(v.x), tk_e4m3_encode(v.y), tk_e4m3_encode(v.z), tk_e4m3_encode(v.w));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        codes[base + i] = tk_e4m3_encode(float(x[base + i]) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

template <typename T>
kernel void quantize_per_token_int8(device const T *x     [[buffer(0)]],
                                    device char    *codes [[buffer(1)]],
                                    device float   *scale [[buffer(2)]],
                                    constant int   &D     [[buffer(3)]],
                                    uint row  [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int nchunks = (D % 4 == 0) ? D / 4 : 0;
    float amax = 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]);
        amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        amax = max(amax, fabs(float(x[base + i])));
    }
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int c = (int)lane; c < nchunks; c += 32) {
        const float4 v = float4(((device const T4*)(x + base))[c]) * inv;
        ((device char4*)(codes + base))[c] =
            char4(tk_int8_encode(v.x), tk_int8_encode(v.y), tk_int8_encode(v.z), tk_int8_encode(v.w));
    }
    for (int i = nchunks * 4 + (int)lane; i < D; i += 32) {
        codes[base + i] = tk_int8_encode(float(x[base + i]) * inv);
    }
    if (lane == 0) {
        scale[row] = s;
    }
}

// ---------------------------------------------------------------------------
// Per-tensor (global) dynamic quantization. Two passes: (1) reduce the global
// absmax into an atomic_uint via the P3 order-preserving float mapping; (2) read
// it back, form scale = absmax/QMAX, and encode every element. Complements the
// per-row quantizers; exercises mittens::atomic_max_float.
// ---------------------------------------------------------------------------
template <typename T>
kernel void quant_tensor_absmax(device const T *x         [[buffer(0)]],
                                device atomic_uint *scale_u [[buffer(1)]],
                                constant int  &n          [[buffer(2)]],
                                uint tid  [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    // 16 elements per thread (vec4 x4) -> 16x fewer contended atomics than the old
    // one-element-per-thread version, and vectorized loads.
    using T4 = vec<T, 4>;
    const long base = (long)tid * 16;
    float amax = 0.0f;
    if (base + 16 <= (long)n) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const float4 v = float4(((device const T4*)(x + base))[j]);
            amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
        }
    } else {
        for (long i = base; i < (long)n; ++i) amax = max(amax, fabs(float(x[i])));
    }
    amax = simd_max(amax);
    if (lane == 0 && amax > 0.0f) { atomic_max_float(scale_u, amax); }   // P3
}

template <typename T>
kernel void quant_tensor_encode_fp8(device const T   *x         [[buffer(0)]],
                                    device const uint *scale_u  [[buffer(1)]],
                                    device uchar     *codes     [[buffer(2)]],
                                    device float     *scale_out [[buffer(3)]],
                                    constant int     &n         [[buffer(4)]],
                                    uint tid [[thread_position_in_grid]]) {
    using T4 = vec<T, 4>;
    const float s = orderable_uint_to_float(scale_u[0]) / 448.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (tid == 0) { scale_out[0] = s; }
    const long base = (long)tid * 4;
    if (base + 4 <= (long)n) {
        const float4 v = float4(((device const T4*)(x + base))[0]) * inv;
        ((device uchar4*)(codes + base))[0] =
            uchar4(tk_e4m3_encode(v.x), tk_e4m3_encode(v.y), tk_e4m3_encode(v.z), tk_e4m3_encode(v.w));
    } else {
        for (long i = base; i < (long)n; ++i) codes[i] = tk_e4m3_encode(float(x[i]) * inv);
    }
}

template <typename T>
kernel void quant_tensor_encode_int8(device const T   *x         [[buffer(0)]],
                                     device const uint *scale_u  [[buffer(1)]],
                                     device char      *codes     [[buffer(2)]],
                                     device float     *scale_out [[buffer(3)]],
                                     constant int     &n         [[buffer(4)]],
                                     uint tid [[thread_position_in_grid]]) {
    using T4 = vec<T, 4>;
    const float s = orderable_uint_to_float(scale_u[0]) / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    if (tid == 0) { scale_out[0] = s; }
    const long base = (long)tid * 4;
    if (base + 4 <= (long)n) {
        const float4 v = float4(((device const T4*)(x + base))[0]) * inv;
        ((device char4*)(codes + base))[0] =
            char4(tk_int8_encode(v.x), tk_int8_encode(v.y), tk_int8_encode(v.z), tk_int8_encode(v.w));
    } else {
        for (long i = base; i < (long)n; ++i) codes[i] = tk_int8_encode(float(x[i]) * inv);
    }
}

#define instantiate_quant_tensor(type_name, T)                                 \
  template [[host_name("quant_tensor_absmax_" #type_name)]] [[kernel]] void     \
  quant_tensor_absmax<T>(device const T *x [[buffer(0)]],                       \
                         device atomic_uint *scale_u [[buffer(1)]],             \
                         constant int &n [[buffer(2)]],                         \
                         uint tid [[thread_position_in_grid]],                  \
                         uint lane [[thread_index_in_simdgroup]]);              \
  template [[host_name("quant_tensor_encode_fp8_" #type_name)]] [[kernel]] void \
  quant_tensor_encode_fp8<T>(device const T *x [[buffer(0)]],                   \
                             device const uint *scale_u [[buffer(1)]],          \
                             device uchar *codes [[buffer(2)]],                 \
                             device float *scale_out [[buffer(3)]],             \
                             constant int &n [[buffer(4)]],                     \
                             uint tid [[thread_position_in_grid]]);             \
  template [[host_name("quant_tensor_encode_int8_" #type_name)]] [[kernel]] void\
  quant_tensor_encode_int8<T>(device const T *x [[buffer(0)]],                  \
                              device const uint *scale_u [[buffer(1)]],         \
                              device char *codes [[buffer(2)]],                 \
                              device float *scale_out [[buffer(3)]],            \
                              constant int &n [[buffer(4)]],                    \
                              uint tid [[thread_position_in_grid]]);

instantiate_quant_tensor(float32, float)
instantiate_quant_tensor(float16, half)
instantiate_quant_tensor(bfloat16, bf16)

// Per-input-channel AWQ-style calibration. One simdgroup covers 32 channels,
// so every token read is contiguous across lanes. Chunked accumulation is the
// same operation with the previous fp32 result supplied as running.
template <typename T>
kernel void calibration_absmax(
    device const T *x [[buffer(0)]],
    device const float *running [[buffer(1)]],
    device float *out [[buffer(2)]],
    constant int &tokens [[buffer(3)]],
    constant int &channels [[buffer(4)]],
    constant int &has_running [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
  const int channel = int(group) * 32 + int(lane);
  threadgroup float partial_max[256];
  threadgroup uint partial_nan[256];
  float amax = channel < channels && has_running != 0 ? running[channel] : 0.0f;
  bool saw_nan = metal::isnan(amax);
  amax = metal::max(amax, 0.0f);
  for (int token = int(simd); channel < channels && token < tokens; token += 8) {
    const float value = float(x[(long)token * channels + channel]);
    saw_nan = saw_nan || metal::isnan(value);
    if (!metal::isnan(value)) {
      amax = metal::max(amax, metal::abs(value));
    }
  }
  partial_max[tid] = amax;
  partial_nan[tid] = saw_nan ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd == 0 && channel < channels) {
    float total_max = partial_max[lane];
    uint total_nan = partial_nan[lane];
    #pragma clang loop unroll(full)
    for (int warp = 1; warp < 8; ++warp) {
      total_max = metal::max(total_max, partial_max[warp * 32 + lane]);
      total_nan |= partial_nan[warp * 32 + lane];
    }
    out[channel] = total_nan != 0 ? NAN : total_max;
  }
}

#define instantiate_calibration_absmax(type_name, T)                           \
  template [[host_name("calibration_absmax_" #type_name)]] [[kernel]] void    \
  calibration_absmax<T>(device const T *x [[buffer(0)]],                       \
      device const float *running [[buffer(1)]], device float *out [[buffer(2)]],\
      constant int &tokens [[buffer(3)]], constant int &channels [[buffer(4)]],\
      constant int &has_running [[buffer(5)]],                                 \
      uint group [[threadgroup_position_in_grid]],                             \
      uint lane [[thread_index_in_simdgroup]],                                 \
      uint simd [[simdgroup_index_in_threadgroup]],                            \
      uint tid [[thread_index_in_threadgroup]]);

instantiate_calibration_absmax(float32, float)
instantiate_calibration_absmax(float16, half)
instantiate_calibration_absmax(bfloat16, bf16)

#define instantiate_quant_rt(type_name, T)                                     \
  template [[host_name("quantize_per_token_fp8_" #type_name)]] [[kernel]] void \
  quantize_per_token_fp8<T>(device const T *x [[buffer(0)]],                   \
                            device uchar *codes [[buffer(1)]],                 \
                            device float *scale [[buffer(2)]],                 \
                            constant int &D [[buffer(3)]],                     \
                            uint row [[threadgroup_position_in_grid]],         \
                            uint lane [[thread_index_in_simdgroup]]);          \
  template [[host_name("quantize_per_token_int8_" #type_name)]] [[kernel]] void\
  quantize_per_token_int8<T>(device const T *x [[buffer(0)]],                  \
                             device char *codes [[buffer(1)]],                 \
                             device float *scale [[buffer(2)]],                \
                             constant int &D [[buffer(3)]],                    \
                             uint row [[threadgroup_position_in_grid]],        \
                             uint lane [[thread_index_in_simdgroup]]);

instantiate_quant_rt(float32, float)
instantiate_quant_rt(float16, half)
instantiate_quant_rt(bfloat16, bf16)

// ---------------------------------------------------------------------------
// Per-GROUP dynamic quantization (group size G along the row, canonical G=128): the
// activation-side layout for block-quantized GEMMs (DeepSeek fp8_block etc.).
// scale layout (rows, D/G) f32 row-major. ue8m0 != 0 rounds the fp8 scale UP to a
// power of two (exp = ceil(log2(amax/448)), the mla_kv_insert_fp8 idiom) — the MX
// convention block-scaled consumers expect. G % 4 == 0, D % G == 0.
// ---------------------------------------------------------------------------
template <typename T>
kernel void quantize_per_group_fp8(device const T *x     [[buffer(0)]],
                                   device uchar   *codes [[buffer(1)]],
                                   device float   *scale [[buffer(2)]],
                                   constant int   &D     [[buffer(3)]],
                                   constant int   &G     [[buffer(4)]],
                                   constant int   &ue8m0 [[buffer(5)]],
                                   uint row  [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int ngroups = D / G;
    for (int g = 0; g < ngroups; ++g) {
        const long gbase = base + (long)g * G;
        float amax = 0.0f;
        for (int c = (int)lane; c < G / 4; c += 32) {
            const float4 v = float4(((device const T4*)(x + gbase))[c]);
            amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
        }
        amax = simd_max(amax);
        float s = amax / 448.0f;
        if (ue8m0 != 0 && amax > 0.0f) {
            s = exp2(ceil(log2(max(amax, 1e-10f) / 448.0f)));
        }
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        for (int c = (int)lane; c < G / 4; c += 32) {
            const float4 v = float4(((device const T4*)(x + gbase))[c]) * inv;
            ((device uchar4*)(codes + gbase))[c] =
                uchar4(tk_e4m3_encode(v.x), tk_e4m3_encode(v.y),
                       tk_e4m3_encode(v.z), tk_e4m3_encode(v.w));
        }
        if (lane == 0) {
            scale[(long)row * ngroups + g] = s;
        }
    }
}

template <typename T>
kernel void quantize_per_group_int8(device const T *x     [[buffer(0)]],
                                    device char    *codes [[buffer(1)]],
                                    device float   *scale [[buffer(2)]],
                                    constant int   &D     [[buffer(3)]],
                                    constant int   &G     [[buffer(4)]],
                                    uint row  [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    using T4 = vec<T, 4>;
    const long base = (long)row * D;
    const int ngroups = D / G;
    for (int g = 0; g < ngroups; ++g) {
        const long gbase = base + (long)g * G;
        float amax = 0.0f;
        for (int c = (int)lane; c < G / 4; c += 32) {
            const float4 v = float4(((device const T4*)(x + gbase))[c]);
            amax = max(amax, max(max(fabs(v.x), fabs(v.y)), max(fabs(v.z), fabs(v.w))));
        }
        amax = simd_max(amax);
        const float s = amax / 127.0f;
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        for (int c = (int)lane; c < G / 4; c += 32) {
            const float4 v = float4(((device const T4*)(x + gbase))[c]) * inv;
            ((device char4*)(codes + gbase))[c] =
                char4(tk_int8_encode(v.x), tk_int8_encode(v.y),
                      tk_int8_encode(v.z), tk_int8_encode(v.w));
        }
        if (lane == 0) {
            scale[(long)row * ngroups + g] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// ASYMMETRIC per-token int8 (zero-point) quantization — vLLM dynamic_scaled_int8_azp:
//   scale = (max - min) / 255 ;  azp = rint(-128 - min/scale) ;
//   q = clamp(rint(x/scale) + azp, -128, 127) ; reconstruct scale * (q - azp).
// Constant rows (max == min) fall back to a symmetric-style scale so the value
// round-trips (range 0 would otherwise divide by zero).
// ---------------------------------------------------------------------------
template <typename T>
kernel void quantize_per_token_int8_azp(device const T *x     [[buffer(0)]],
                                        device char    *codes [[buffer(1)]],
                                        device float   *scale [[buffer(2)]],
                                        device int     *azp_out [[buffer(3)]],
                                        constant int   &D     [[buffer(4)]],
                                        uint row  [[threadgroup_position_in_grid]],
                                        uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    float mn = 3.4028234663852886e38f, mx = -3.4028234663852886e38f;
    for (int i = (int)lane; i < D; i += 32) {
        const float v = float(x[base + i]);
        mn = min(mn, v);
        mx = max(mx, v);
    }
    mn = simd_min(mn);
    mx = simd_max(mx);
    const float range = mx - mn;
    const float s = range > 0.0f ? range / 255.0f
                                 : max(fabs(mn) / 127.0f, 1e-7f);
    const float inv = 1.0f / s;
    const int azp = int(rint(-128.0f - mn * inv));
    for (int i = (int)lane; i < D; i += 32) {
        const int q = int(rint(float(x[base + i]) * inv)) + azp;
        codes[base + i] = char(clamp(q, -128, 127));
    }
    if (lane == 0) {
        scale[row] = s;
        azp_out[row] = azp;
    }
}

#define instantiate_quant_group(type_name, T)                                       \
  template [[host_name("quantize_per_group_fp8_" #type_name)]] [[kernel]] void      \
  quantize_per_group_fp8<T>(device const T *x [[buffer(0)]],                        \
                            device uchar *codes [[buffer(1)]],                      \
                            device float *scale [[buffer(2)]],                      \
                            constant int &D [[buffer(3)]],                          \
                            constant int &G [[buffer(4)]],                          \
                            constant int &ue8m0 [[buffer(5)]],                      \
                            uint row [[threadgroup_position_in_grid]],              \
                            uint lane [[thread_index_in_simdgroup]]);               \
  template [[host_name("quantize_per_group_int8_" #type_name)]] [[kernel]] void     \
  quantize_per_group_int8<T>(device const T *x [[buffer(0)]],                       \
                             device char *codes [[buffer(1)]],                      \
                             device float *scale [[buffer(2)]],                     \
                             constant int &D [[buffer(3)]],                         \
                             constant int &G [[buffer(4)]],                         \
                             uint row [[threadgroup_position_in_grid]],             \
                             uint lane [[thread_index_in_simdgroup]]);              \
  template [[host_name("quantize_per_token_int8_azp_" #type_name)]] [[kernel]] void \
  quantize_per_token_int8_azp<T>(device const T *x [[buffer(0)]],                   \
                                 device char *codes [[buffer(1)]],                  \
                                 device float *scale [[buffer(2)]],                 \
                                 device int *azp_out [[buffer(3)]],                 \
                                 constant int &D [[buffer(4)]],                     \
                                 uint row [[threadgroup_position_in_grid]],         \
                                 uint lane [[thread_index_in_simdgroup]]);

instantiate_quant_group(float32, float)
instantiate_quant_group(float16, half)
instantiate_quant_group(bfloat16, bf16)
