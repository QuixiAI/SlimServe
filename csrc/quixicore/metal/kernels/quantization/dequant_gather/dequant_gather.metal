#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Gather packed GGUF rows while dequantizing directly into fp16. One thread
// owns an aligned 8-column span and emits two vector stores. Invalid ids
// produce an all-zero row.
template <typename FMT>
kernel void dequant_gather(
    device const uchar *table [[buffer(0)]],
    device const int *ids [[buffer(1)]],
    device half *output [[buffer(2)]],
    constant int &rows [[buffer(3)]],
    constant int &columns [[buffer(4)]],
    constant int &tokens [[buffer(5)]],
    constant float &scale [[buffer(6)]],
    uint tid [[thread_position_in_grid]]) {
    // The standalone MPS metallib uses Metal's fast math mode.  Preserve the
    // source multiplication order so it honors the bit-exact FP32-decode/scale
    // followed by one FP16-rounding contract just like the
    // MLX build, which compiles with -fno-fast-math.
    #pragma clang fp reassociate(off)
    const int spans_per_row = columns / 8;
    const uint total = (uint)tokens * (uint)spans_per_row;
    if (tid >= total) return;
    const int token = int(tid / spans_per_row);
    const int col0 = int(tid % spans_per_row) * 8;
    const int row = ids[token];
    device half4 *dst = (device half4 *)(output + (long)token * columns + col0);
    if (row < 0 || row >= rows) {
        dst[0] = half4(0.0h);
        dst[1] = half4(0.0h);
        return;
    }

    const int blocks_per_row = columns / FMT::block_k;
    const int block = col0 / FMT::block_k;
    const int column_in_block = col0 % FMT::block_k;
    device const uchar *base =
        table + ((long)row * blocks_per_row + block) * FMT::block_bytes;
    float values[8];
    tk_dequant8_f32<FMT>(base, column_in_block, values);
    dst[0] = half4(float4(values[0], values[1], values[2], values[3]) * scale);
    dst[1] = half4(float4(values[4], values[5], values[6], values[7]) * scale);
}

#define instantiate_dequant_gather(name, FMT)                                  \
  template [[host_name("dequant_gather_" name)]] [[kernel]]                  \
  void dequant_gather<FMT>(device const uchar *table [[buffer(0)]],           \
    device const int *ids [[buffer(1)]], device half *output [[buffer(2)]],   \
    constant int &rows [[buffer(3)]], constant int &columns [[buffer(4)]],    \
    constant int &tokens [[buffer(5)]], constant float &scale [[buffer(6)]],  \
    uint tid [[thread_position_in_grid]]);

instantiate_dequant_gather("q4_0", q4_0)
instantiate_dequant_gather("q8_0", q8_0)
instantiate_dequant_gather("q6_K", q6_K)

// General packed embedding lookup.  Format-specialized fp32 dequantization and the optional additive
// epilogue remain in fp32; T is rounded exactly once at the store.  `add` may
// represent a positional embedding, residual, or any other elementwise term.
template <typename T, typename FMT>
kernel void quantized_embedding_lookup(
    device const uchar *table [[buffer(0)]],
    device const int *ids [[buffer(1)]],
    device const T *add [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant int &rows [[buffer(4)]],
    constant int &columns [[buffer(5)]],
    constant int &tokens [[buffer(6)]],
    constant float &scale [[buffer(7)]],
    constant int &use_add [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    const int spans_per_row = columns / 8;
    const uint total = uint(tokens) * uint(spans_per_row);
    if (tid >= total) return;
    const int token = int(tid / spans_per_row);
    const int col0 = int(tid % spans_per_row) * 8;
    const int row = ids[token];
    const long output_base = (long)token * columns + col0;
    if (row < 0 || row >= rows) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) output[output_base + i] = T(0.0f);
        return;
    }

    const int blocks_per_row = columns / FMT::block_k;
    const int block = col0 / FMT::block_k;
    const int column_in_block = col0 % FMT::block_k;
    device const uchar *base =
        table + ((long)row * blocks_per_row + block) * FMT::block_bytes;
    float values[8];
    tk_dequant8_f32<FMT>(base, column_in_block, values);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        float value = values[i] * scale;
        if (use_add != 0) value += float(add[output_base + i]);
        output[output_base + i] = T(value);
    }
}

// CSR-style embedding bag.  A thread owns eight output columns and accumulates
// every valid row in fp32.  Repeated ids are intentional and naturally add
// multiple times.  Invalid ids are skipped and do not contribute to mean's
// denominator.
template <typename T, typename FMT>
kernel void quantized_embedding_bag(
    device const uchar *table [[buffer(0)]],
    device const int *ids [[buffer(1)]],
    device const int *offsets [[buffer(2)]],
    device const float *sample_weights [[buffer(3)]],
    device T *output [[buffer(4)]],
    constant int &rows [[buffer(5)]],
    constant int &columns [[buffer(6)]],
    constant int &id_count [[buffer(7)]],
    constant int &bags [[buffer(8)]],
    constant float &scale [[buffer(9)]],
    constant int &use_weights [[buffer(10)]],
    constant int &mean_mode [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
    #pragma clang fp reassociate(off)
    const int spans_per_row = columns / 8;
    const uint total = uint(bags) * uint(spans_per_row);
    if (tid >= total) return;
    const int bag = int(tid / spans_per_row);
    const int col0 = int(tid % spans_per_row) * 8;
    const int begin = offsets[bag];
    const int end = offsets[bag + 1];
    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 0.0f};
    int valid = 0;
    if (begin >= 0 && end >= begin && end <= id_count) {
        const int blocks_per_row = columns / FMT::block_k;
        const int block = col0 / FMT::block_k;
        const int column_in_block = col0 % FMT::block_k;
        for (int index = begin; index < end; ++index) {
            const int row = ids[index];
            if (row < 0 || row >= rows) continue;
            device const uchar *base =
                table + ((long)row * blocks_per_row + block) * FMT::block_bytes;
            float values[8];
            tk_dequant8_f32<FMT>(base, column_in_block, values);
            const float row_scale = scale *
                (use_weights != 0 ? sample_weights[index] : 1.0f);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) accum[i] += values[i] * row_scale;
            ++valid;
        }
    }
    const float denominator = mean_mode != 0 && valid > 0 ? float(valid) : 1.0f;
    const long output_base = (long)bag * columns + col0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) output[output_base + i] = T(accum[i] / denominator);
}

#define instantiate_quantized_embedding_type(fmt_name, FMT, type_name, T)     \
  template [[host_name("quantized_embedding_" fmt_name "_" type_name)]]     \
  [[kernel]] void quantized_embedding_lookup<T, FMT>(                        \
    device const uchar *table [[buffer(0)]], device const int *ids [[buffer(1)]], \
    device const T *add [[buffer(2)]], device T *output [[buffer(3)]],        \
    constant int &rows [[buffer(4)]], constant int &columns [[buffer(5)]],    \
    constant int &tokens [[buffer(6)]], constant float &scale [[buffer(7)]],  \
    constant int &use_add [[buffer(8)]], uint tid [[thread_position_in_grid]]); \
  template [[host_name("quantized_embedding_bag_" fmt_name "_" type_name)]] \
  [[kernel]] void quantized_embedding_bag<T, FMT>(                           \
    device const uchar *table [[buffer(0)]], device const int *ids [[buffer(1)]], \
    device const int *offsets [[buffer(2)]],                                  \
    device const float *sample_weights [[buffer(3)]],                         \
    device T *output [[buffer(4)]], constant int &rows [[buffer(5)]],         \
    constant int &columns [[buffer(6)]], constant int &id_count [[buffer(7)]], \
    constant int &bags [[buffer(8)]], constant float &scale [[buffer(9)]],    \
    constant int &use_weights [[buffer(10)]],                                \
    constant int &mean_mode [[buffer(11)]], uint tid [[thread_position_in_grid]]);

#define instantiate_quantized_embedding(fmt_name, FMT)                       \
  instantiate_quantized_embedding_type(fmt_name, FMT, "float16", half)      \
  instantiate_quantized_embedding_type(fmt_name, FMT, "bfloat16", bf16)    \
  instantiate_quantized_embedding_type(fmt_name, FMT, "float32", float)

instantiate_quantized_embedding("q4_0", q4_0)
instantiate_quantized_embedding("q8_0", q8_0)
instantiate_quantized_embedding("q4_K", q4_K)
instantiate_quantized_embedding("q5_K", q5_K)
instantiate_quantized_embedding("q6_K", q6_K)
instantiate_quantized_embedding("q2_K", q2_K)
instantiate_quantized_embedding("q3_K", q3_K)
instantiate_quantized_embedding("iq4_nl", iq4_nl)
instantiate_quantized_embedding("iq4_xs", iq4_xs)
instantiate_quantized_embedding("kU4B8", kU4B8)
instantiate_quantized_embedding("kU4", kU4)
instantiate_quantized_embedding("hqq", hqq)
instantiate_quantized_embedding("fp8_e4m3", fp8_e4m3)
instantiate_quantized_embedding("mxfp8", mxfp8)
instantiate_quantized_embedding("nvfp4", nvfp4)
instantiate_quantized_embedding("mxfp4", mxfp4)

} // namespace mittens
