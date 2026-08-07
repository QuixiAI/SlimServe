#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

// Window attention specialized for head dimension 32. The qkv input keeps the
// natural contiguous projection layout (BW,N,3,H,32). One simdgroup owns one
// (window,head,query) row and performs online fp32 softmax, including
// pre-gathered relative bias and an optional shifted-window mask.
constant float SWIN_NEG_INF = -3.4028234663852886e38f;

template <typename T>
kernel void swin_attn_d32(
    device const T *qkv [[buffer(0)]],
    device const T *relative_bias [[buffer(1)]],
    device const float *mask [[buffer(2)]],
    device T *output [[buffer(3)]],
    constant int &BW [[buffer(4)]],
    constant int &N [[buffer(5)]],
    constant int &H [[buffer(6)]],
    constant int &windows_per_image [[buffer(7)]],
    constant int &has_mask [[buffer(8)]],
    uint3 pos [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int query_index = int(pos.x);
    const int head = int(pos.y);
    const int window = int(pos.z);
    const int D = 32;
    const float scale = metal::rsqrt(32.0f);
    const long qbase = ((((long)window * N + query_index) * 3) * H + head) * D;
    const float query = float(qkv[qbase + lane]);

    float maximum = SWIN_NEG_INF;
    float denominator = 0.0f;
    float accumulator = 0.0f;
    const int mask_window = windows_per_image > 0 ? window % windows_per_image : 0;
    for (int key_index = 0; key_index < N; ++key_index) {
        const long keybase = ((((long)window * N + key_index) * 3 + 1) * H + head) * D;
        float score = metal::simd_sum(query * float(qkv[keybase + lane])) * scale;
        score += float(relative_bias[((long)head * N + query_index) * N + key_index]);
        if (has_mask) {
            score += mask[((long)mask_window * N + query_index) * N + key_index];
        }

        const float next_maximum = metal::max(maximum, score);
        const float correction = metal::exp(maximum - next_maximum);
        const float probability = metal::exp(score - next_maximum);
        const long valuebase = ((((long)window * N + key_index) * 3 + 2) * H + head) * D;
        accumulator = accumulator * correction + probability * float(qkv[valuebase + lane]);
        denominator = denominator * correction + probability;
        maximum = next_maximum;
    }
    const long outbase = (((long)window * N + query_index) * H + head) * D;
    output[outbase + lane] = T(accumulator / denominator);
}

#define instantiate_swin_attn(type_name, T)                                     \
  template [[host_name("swin_attn_d32_" #type_name)]] [[kernel]] void          \
  swin_attn_d32<T>(device const T *qkv [[buffer(0)]],                           \
                   device const T *relative_bias [[buffer(1)]],                 \
                   device const float *mask [[buffer(2)]],                      \
                   device T *output [[buffer(3)]],                              \
                   constant int &BW [[buffer(4)]],                              \
                   constant int &N [[buffer(5)]],                               \
                   constant int &H [[buffer(6)]],                               \
                   constant int &windows_per_image [[buffer(7)]],               \
                   constant int &has_mask [[buffer(8)]],                        \
                   uint3 pos [[threadgroup_position_in_grid]],                  \
                   uint lane [[thread_index_in_simdgroup]]);

instantiate_swin_attn(float32, float)
instantiate_swin_attn(float16, half)
instantiate_swin_attn(bfloat16, bf16)

} // namespace mittens
