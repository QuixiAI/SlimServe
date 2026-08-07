#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Latency-oriented linear layer for autoregressive decode. One simdgroup owns
// one (batch, output-channel) dot product. The optional erf-GELU epilogue
// removes a second launch for decoder FFN expansion.
template <typename T>
kernel void decode_linear(
    device const T *input [[buffer(0)]],
    device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant int &batch [[buffer(4)]],
    constant int &in_dim [[buffer(5)]],
    constant int &out_dim [[buffer(6)]],
    constant int &gelu [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const long input_base = (long)batch_index * in_dim;
    const long weight_base = (long)output_channel * in_dim;
    float accumulator = 0.0f;
    for (int k = int(lane); k < in_dim; k += 32) {
        accumulator += float(input[input_base + k]) * float(weight[weight_base + k]);
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) {
        accumulator += float(bias[output_channel]);
        if (gelu != 0) accumulator = glu_gelu_erf(accumulator);
        output[(long)batch_index * out_dim + output_channel] = T(accumulator);
    }
}

#define instantiate_decode_linear(type_name, T)                                \
  template [[host_name("decode_linear_" #type_name)]] [[kernel]] void         \
  decode_linear<T>(device const T *input [[buffer(0)]],                        \
                   device const T *weight [[buffer(1)]],                       \
                   device const T *bias [[buffer(2)]],                         \
                   device T *output [[buffer(3)]],                             \
                   constant int &batch [[buffer(4)]],                          \
                   constant int &in_dim [[buffer(5)]],                         \
                   constant int &out_dim [[buffer(6)]],                        \
                   constant int &gelu [[buffer(7)]],                           \
                   uint2 group [[threadgroup_position_in_grid]],               \
                   uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_linear(float32, float)
instantiate_decode_linear(bfloat16, bf16)

template <typename T>
kernel void decode_linear_residual(
    device const T *input [[buffer(0)]],
    device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device const T *residual [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &in_dim [[buffer(6)]],
    constant int &out_dim [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const long input_base = (long)batch_index * in_dim;
    const long weight_base = (long)output_channel * in_dim;
    const long output_index = (long)batch_index * out_dim + output_channel;
    float accumulator = 0.0f;
    for (int k = int(lane); k < in_dim; k += 32) {
        accumulator += float(input[input_base + k]) * float(weight[weight_base + k]);
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) {
        // Match a materialized framework linear: round the linear result to T
        // before evaluating the residual addition.
        const T linear = T(accumulator + float(bias[output_channel]));
        output[output_index] = T(float(linear) + float(residual[output_index]));
    }
}

#define instantiate_decode_linear_residual(type_name, T)                       \
  template [[host_name("decode_linear_residual_" #type_name)]] [[kernel]]     \
  void decode_linear_residual<T>(device const T *input [[buffer(0)]],          \
    device const T *weight [[buffer(1)]], device const T *bias [[buffer(2)]],  \
    device const T *residual [[buffer(3)]], device T *output [[buffer(4)]],    \
    constant int &batch [[buffer(5)]], constant int &in_dim [[buffer(6)]],     \
    constant int &out_dim [[buffer(7)]],                                       \
    uint2 group [[threadgroup_position_in_grid]],                              \
    uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_linear_residual(float32, float)
instantiate_decode_linear_residual(bfloat16, bf16)

// Persistent q8_0 weight path. Packed blocks are {fp16 scale; int8 codes[32]}.
template <typename T>
kernel void decode_linear_q8(
    device const T *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device const T *residual [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &in_dim [[buffer(6)]],
    constant int &out_dim [[buffer(7)]],
    constant int &gelu [[buffer(8)]],
    constant int &use_residual [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const int blocks = in_dim / 32;
    const long input_base = (long)batch_index * in_dim;
    const long weight_base = (long)output_channel * blocks * 34;
    float accumulator = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *packed = weight + weight_base + (long)block * 34;
        const float scale = float(((device const half *)packed)[0]);
        const int code = int(((device const char *)(packed + 2))[lane]);
        accumulator += scale * float(code) * float(input[input_base + block * 32 + lane]);
    }
    accumulator = metal::simd_sum(accumulator);
    if (lane == 0) {
        const long output_index = (long)batch_index * out_dim + output_channel;
        T rounded = T(accumulator + float(bias[output_channel]));
        if (gelu != 0) rounded = T(glu_gelu_erf(float(rounded)));
        if (use_residual != 0) {
            rounded = T(float(rounded) + float(residual[output_index]));
        }
        output[output_index] = rounded;
    }
}

#define instantiate_decode_linear_q8(type_name, T)                             \
  template [[host_name("decode_linear_q8_" #type_name)]] [[kernel]]          \
  void decode_linear_q8<T>(device const T *input [[buffer(0)]],               \
    device const uchar *weight [[buffer(1)]], device const T *bias [[buffer(2)]], \
    device const T *residual [[buffer(3)]], device T *output [[buffer(4)]],   \
    constant int &batch [[buffer(5)]], constant int &in_dim [[buffer(6)]],    \
    constant int &out_dim [[buffer(7)]], constant int &gelu [[buffer(8)]],    \
    constant int &use_residual [[buffer(9)]],                                 \
    uint2 group [[threadgroup_position_in_grid]],                             \
    uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_linear_q8(float32, float)
instantiate_decode_linear_q8(bfloat16, bf16)

METAL_FUNC float decode_epilogue(float value, int activation) {
    if (activation == 1) return glu_gelu_erf(value);
    if (activation == 2) return value / (1.0f + metal::exp(-value));
    return value;
}

template <typename T>
METAL_FUNC float decode_dense_dot(
    device const T *input, device const T *weight, long input_base,
    long weight_base, int in_dim, uint lane) {
    float accumulator = 0.0f;
    for (int k = int(lane); k < in_dim; k += 32) {
        accumulator += float(input[input_base + k]) * float(weight[weight_base + k]);
    }
    return metal::simd_sum(accumulator);
}

template <typename T, typename FMT>
METAL_FUNC float decode_packed_dot(
    device const T *input, device const uchar *weight, long input_base,
    int output_channel, int in_dim, uint lane) {
    const int blocks_per_row = in_dim / FMT::block_k;
    const long row_base = (long)output_channel * blocks_per_row * FMT::block_bytes;
    const int spans = in_dim / 8;
    float accumulator = 0.0f;
    for (int span = int(lane); span < spans; span += 32) {
        const int col0 = span * 8;
        const int block = col0 / FMT::block_k;
        const int column_in_block = col0 % FMT::block_k;
        device const uchar *base = weight + row_base + (long)block * FMT::block_bytes;
        float values[8];
        tk_dequant8_f32<FMT>(base, column_in_block, values);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            accumulator += float(input[input_base + col0 + i]) * values[i];
        }
    }
    return metal::simd_sum(accumulator);
}

// q4_0 stores the low and high 16-value halves in the same 16 packed bytes.
// One lane consumes a whole block, so each scale and block address is decoded
// once instead of being duplicated across a pair of lanes.
template <typename T>
METAL_FUNC float decode_packed_dot_q4(
    device const T *input, device const uchar *weight, long input_base,
    int output_channel, int in_dim, uint lane) {
    const int blocks_per_row = in_dim / 32;
    const long row_base = (long)output_channel * blocks_per_row * 18;
    float accumulator = 0.0f;
    for (int block = int(lane); block < blocks_per_row; block += 32) {
        device const uchar *base = weight + row_base + (long)block * 18;
        const float scale = float(((device const half *)base)[0]);
        device const uchar *codes = base + 2;
        const long x0 = input_base + block * 32;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            accumulator += float(input[x0 + i]) *
                scale * float(int(packed & 0x0f) - 8);
            accumulator += float(input[x0 + i + 16]) *
                scale * float(int(packed >> 4) - 8);
        }
    }
    return metal::simd_sum(accumulator);
}

#define specialize_decode_packed_dot_q4(T)                                  \
  template <> METAL_FUNC float decode_packed_dot<T, q4_0>(                  \
    device const T *input, device const uchar *weight, long input_base,     \
    int output_channel, int in_dim, uint lane) {                             \
    return decode_packed_dot_q4<T>(                                          \
        input, weight, input_base, output_channel, in_dim, lane);            \
  }

specialize_decode_packed_dot_q4(float)
specialize_decode_packed_dot_q4(half)
specialize_decode_packed_dot_q4(bf16)

// NVFP4 stores both 8-value halves of a 16-value block in the same eight
// bytes.  Consuming a complete block per lane halves the scale/address work
// relative to the generic 8-value span decoder while retaining its FP32
// dequantization contract.
template <typename T>
METAL_FUNC float decode_packed_dot_nvfp4(
    device const T *input, device const uchar *weight, long input_base,
    int output_channel, int in_dim, uint lane) {
    const int blocks_per_row = in_dim / 16;
    const long row_base = (long)output_channel * blocks_per_row * 9;
    float accumulator = 0.0f;
    for (int block = int(lane); block < blocks_per_row; block += 32) {
        device const uchar *base = weight + row_base + (long)block * 9;
        const float scale = float(tk_e4m3_decode(base[0]));
        device const uchar *codes = base + 1;
        const long x0 = input_base + block * 16;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = codes[i];
            const float low = scale * float(tk_e2m1_decode(uint(packed & 0x0f)));
            const float high = scale * float(tk_e2m1_decode(uint(packed >> 4)));
            accumulator += float(input[x0 + i]) * low;
            accumulator += float(input[x0 + i + 8]) * high;
        }
    }
    return metal::simd_sum(accumulator);
}

#define specialize_decode_packed_dot_nvfp4(T)                              \
  template <> METAL_FUNC float decode_packed_dot<T, nvfp4>(                \
    device const T *input, device const uchar *weight, long input_base,     \
    int output_channel, int in_dim, uint lane) {                            \
    return decode_packed_dot_nvfp4<T>(                                      \
        input, weight, input_base, output_channel, in_dim, lane);           \
  }

specialize_decode_packed_dot_nvfp4(float)
specialize_decode_packed_dot_nvfp4(half)
specialize_decode_packed_dot_nvfp4(bf16)

// MXFP4 stores one E8M0 scale and sixteen packed bytes per 32-value block.
// Consume the complete block per lane so the power-of-two scale expansion and
// each packed byte are paid once while preserving FP32 dequantization.
template <typename T>
METAL_FUNC float decode_packed_dot_mxfp4(
    device const T *input, device const uchar *weight, long input_base,
    int output_channel, int in_dim, uint lane) {
    const int blocks_per_row = in_dim / mxfp4::block_k;
    const long row_base =
        (long)output_channel * blocks_per_row * mxfp4::block_bytes;
    float accumulator = 0.0f;
    for (int block = int(lane); block < blocks_per_row; block += 32) {
        device const uchar *base =
            weight + row_base + (long)block * mxfp4::block_bytes;
        const float scale = tk_e8m0_decode_f32(base[0]);
        device const uchar *codes = base + 1;
        const long x0 = input_base + block * mxfp4::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            const float low =
                scale * float(tk_e2m1_decode(uint(packed & 0x0f)));
            const float high =
                scale * float(tk_e2m1_decode(uint(packed >> 4)));
            accumulator += float(input[x0 + i]) * low;
            accumulator += float(input[x0 + i + 16]) * high;
        }
    }
    return metal::simd_sum(accumulator);
}

#define specialize_decode_packed_dot_mxfp4(T)                              \
  template <> METAL_FUNC float decode_packed_dot<T, mxfp4>(                 \
    device const T *input, device const uchar *weight, long input_base,     \
    int output_channel, int in_dim, uint lane) {                            \
    return decode_packed_dot_mxfp4<T>(                                      \
        input, weight, input_base, output_channel, in_dim, lane);           \
  }

specialize_decode_packed_dot_mxfp4(float)
specialize_decode_packed_dot_mxfp4(half)
specialize_decode_packed_dot_mxfp4(bf16)

// Unified dense decode-linear epilogue.  The operation is
// residual + activation(x @ weight.T + bias), with each optional component
// controlled by a scalar flag and a single output-dtype rounding at the store.
template <typename T>
kernel void decode_linear_epilogue_dense(
    device const T *input [[buffer(0)]],
    device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device const T *residual [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &in_dim [[buffer(6)]],
    constant int &out_dim [[buffer(7)]],
    constant int &activation [[buffer(8)]],
    constant int &use_bias [[buffer(9)]],
    constant int &use_residual [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const long input_base = (long)batch_index * in_dim;
    const long output_index = (long)batch_index * out_dim + output_channel;
    float value = decode_dense_dot(
        input, weight, input_base, (long)output_channel * in_dim, in_dim, lane);
    if (lane == 0) {
        if (use_bias != 0) value += float(bias[output_channel]);
        value = decode_epilogue(value, activation);
        if (use_residual != 0) value += float(residual[output_index]);
        output[output_index] = T(value);
    }
}

template <typename T, typename FMT>
kernel void decode_linear_epilogue_packed(
    device const T *input [[buffer(0)]],
    device const uchar *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]],
    device const T *residual [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &batch [[buffer(5)]],
    constant int &in_dim [[buffer(6)]],
    constant int &out_dim [[buffer(7)]],
    constant int &activation [[buffer(8)]],
    constant int &use_bias [[buffer(9)]],
    constant int &use_residual [[buffer(10)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int output_channel = int(group.x);
    const int batch_index = int(group.y);
    if (output_channel >= out_dim || batch_index >= batch) return;
    const long input_base = (long)batch_index * in_dim;
    const long output_index = (long)batch_index * out_dim + output_channel;
    float value = decode_packed_dot<T, FMT>(
        input, weight, input_base, output_channel, in_dim, lane);
    if (lane == 0) {
        if (use_bias != 0) value += float(bias[output_channel]);
        value = decode_epilogue(value, activation);
        if (use_residual != 0) value += float(residual[output_index]);
        output[output_index] = T(value);
    }
}

// Two projections share input reads and reduce together before the SwiGLU
// epilogue: silu(gate) * up.
template <typename T>
kernel void decode_swiglu_dense(
    device const T *input [[buffer(0)]],
    device const T *gate_weight [[buffer(1)]],
    device const T *up_weight [[buffer(2)]],
    device const T *gate_bias [[buffer(3)]],
    device const T *up_bias [[buffer(4)]],
    device T *output [[buffer(5)]],
    constant int &batch [[buffer(6)]],
    constant int &in_dim [[buffer(7)]],
    constant int &out_dim [[buffer(8)]],
    constant int &use_bias [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int n = int(group.x), b = int(group.y);
    if (n >= out_dim || b >= batch) return;
    const long input_base = (long)b * in_dim;
    const long weight_base = (long)n * in_dim;
    float gate = 0.0f, up = 0.0f;
    for (int k = int(lane); k < in_dim; k += 32) {
        const float x = float(input[input_base + k]);
        gate += x * float(gate_weight[weight_base + k]);
        up += x * float(up_weight[weight_base + k]);
    }
    gate = metal::simd_sum(gate);
    up = metal::simd_sum(up);
    if (lane == 0) {
        if (use_bias != 0) {
            gate += float(gate_bias[n]);
            up += float(up_bias[n]);
        }
        const float activated = gate / (1.0f + metal::exp(-gate));
        output[(long)b * out_dim + n] = T(activated * up);
    }
}

template <typename T, typename FMT>
kernel void decode_swiglu_packed(
    device const T *input [[buffer(0)]],
    device const uchar *gate_weight [[buffer(1)]],
    device const uchar *up_weight [[buffer(2)]],
    device const T *gate_bias [[buffer(3)]],
    device const T *up_bias [[buffer(4)]],
    device T *output [[buffer(5)]],
    constant int &batch [[buffer(6)]],
    constant int &in_dim [[buffer(7)]],
    constant int &out_dim [[buffer(8)]],
    constant int &use_bias [[buffer(9)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int n = int(group.x), b = int(group.y);
    if (n >= out_dim || b >= batch) return;
    const long input_base = (long)b * in_dim;
    float gate = decode_packed_dot<T, FMT>(
        input, gate_weight, input_base, n, in_dim, lane);
    float up = decode_packed_dot<T, FMT>(
        input, up_weight, input_base, n, in_dim, lane);
    if (lane == 0) {
        if (use_bias != 0) {
            gate += float(gate_bias[n]);
            up += float(up_bias[n]);
        }
        const float activated = gate / (1.0f + metal::exp(-gate));
        output[(long)b * out_dim + n] = T(activated * up);
    }
}

#define instantiate_decode_epilogue_dense(type_name, T)                      \
  template [[host_name("decode_linear_epilogue_dense_" #type_name)]]        \
  [[kernel]] void decode_linear_epilogue_dense<T>(                           \
    device const T *input [[buffer(0)]], device const T *weight [[buffer(1)]], \
    device const T *bias [[buffer(2)]], device const T *residual [[buffer(3)]], \
    device T *output [[buffer(4)]], constant int &batch [[buffer(5)]],        \
    constant int &in_dim [[buffer(6)]], constant int &out_dim [[buffer(7)]], \
    constant int &activation [[buffer(8)]], constant int &use_bias [[buffer(9)]], \
    constant int &use_residual [[buffer(10)]],                               \
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("decode_swiglu_dense_" #type_name)]] [[kernel]]      \
  void decode_swiglu_dense<T>(                                               \
    device const T *input [[buffer(0)]], device const T *gate_weight [[buffer(1)]], \
    device const T *up_weight [[buffer(2)]], device const T *gate_bias [[buffer(3)]], \
    device const T *up_bias [[buffer(4)]], device T *output [[buffer(5)]],   \
    constant int &batch [[buffer(6)]], constant int &in_dim [[buffer(7)]],   \
    constant int &out_dim [[buffer(8)]], constant int &use_bias [[buffer(9)]], \
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

#define instantiate_decode_epilogue_packed(fmt_name, FMT, type_name, T)       \
  template [[host_name("decode_linear_epilogue_" fmt_name "_" #type_name)]] \
  [[kernel]] void decode_linear_epilogue_packed<T, FMT>(                     \
    device const T *input [[buffer(0)]], device const uchar *weight [[buffer(1)]], \
    device const T *bias [[buffer(2)]], device const T *residual [[buffer(3)]], \
    device T *output [[buffer(4)]], constant int &batch [[buffer(5)]],        \
    constant int &in_dim [[buffer(6)]], constant int &out_dim [[buffer(7)]], \
    constant int &activation [[buffer(8)]], constant int &use_bias [[buffer(9)]], \
    constant int &use_residual [[buffer(10)]],                               \
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("decode_swiglu_" fmt_name "_" #type_name)]] [[kernel]] \
  void decode_swiglu_packed<T, FMT>(                                         \
    device const T *input [[buffer(0)]], device const uchar *gate_weight [[buffer(1)]], \
    device const uchar *up_weight [[buffer(2)]], device const T *gate_bias [[buffer(3)]], \
    device const T *up_bias [[buffer(4)]], device T *output [[buffer(5)]],   \
    constant int &batch [[buffer(6)]], constant int &in_dim [[buffer(7)]],   \
    constant int &out_dim [[buffer(8)]], constant int &use_bias [[buffer(9)]], \
    uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

#define instantiate_decode_epilogue_type(type_name, T)                       \
  instantiate_decode_epilogue_dense(type_name, T)                            \
  instantiate_decode_epilogue_packed("q4_0", q4_0, type_name, T)            \
  instantiate_decode_epilogue_packed("q8_0", q8_0, type_name, T)            \
  instantiate_decode_epilogue_packed("q6_K", q6_K, type_name, T)            \
  instantiate_decode_epilogue_packed("mxfp8", mxfp8, type_name, T)          \
  instantiate_decode_epilogue_packed("nvfp4", nvfp4, type_name, T)          \
  instantiate_decode_epilogue_packed("mxfp4", mxfp4, type_name, T)

instantiate_decode_epilogue_type(float32, float)
instantiate_decode_epilogue_type(float16, half)
instantiate_decode_epilogue_type(bfloat16, bf16)

} // namespace mittens
