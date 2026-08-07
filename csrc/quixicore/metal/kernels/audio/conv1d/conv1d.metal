#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

template <typename T>
kernel void audio_conv1d(
    device const T *x [[buffer(0)]], device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]], device T *out [[buffer(3)]],
    constant int &length [[buffer(4)]], constant int &in_channels [[buffer(5)]],
    constant int &out_length [[buffer(6)]], constant int &out_channels [[buffer(7)]],
    constant int &kernel_size [[buffer(8)]], constant int &stride [[buffer(9)]],
    constant int &padding [[buffer(10)]], constant int &dilation [[buffer(11)]],
    constant int &has_bias [[buffer(12)]], constant int &batch_size [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  const long total_per_batch = (long)out_length * out_channels;
  if ((long)gid >= (long)batch_size * total_per_batch) return;
  const int batch = int((long)gid / total_per_batch);
  const int rem = int((long)gid - (long)batch * total_per_batch);
  const int t = rem / out_channels, oc = rem - t * out_channels;
  float sum = has_bias != 0 ? float(bias[oc]) : 0.0f;
  const long weight_base = (long)oc * kernel_size * in_channels;
  for (int k = 0; k < kernel_size; ++k) {
    const int source_t = t * stride + k * dilation - padding;
    if (source_t < 0 || source_t >= length) continue;
    const long x_base = ((long)batch * length + source_t) * in_channels;
    const long w_base = weight_base + (long)k * in_channels;
    for (int c = 0; c < in_channels; ++c) {
      sum += float(x[x_base + c]) * float(weight[w_base + c]);
    }
  }
  out[gid] = T(sum);
}

template <typename T>
kernel void audio_depthwise_conv1d(
    device const T *x [[buffer(0)]], device const T *weight [[buffer(1)]],
    device const T *bias [[buffer(2)]], device T *out [[buffer(3)]],
    constant int &length [[buffer(4)]], constant int &channels [[buffer(5)]],
    constant int &out_length [[buffer(6)]], constant int &kernel_size [[buffer(7)]],
    constant int &stride [[buffer(8)]], constant int &padding [[buffer(9)]],
    constant int &dilation [[buffer(10)]], constant int &has_bias [[buffer(11)]],
    constant int &activation [[buffer(12)]], constant int &batch_size [[buffer(13)]],
    uint gid [[thread_position_in_grid]]) {
  const long total_per_batch = (long)out_length * channels;
  if ((long)gid >= (long)batch_size * total_per_batch) return;
  const int batch = int((long)gid / total_per_batch);
  const int rem = int((long)gid - (long)batch * total_per_batch);
  const int t = rem / channels, c = rem - t * channels;
  float sum = has_bias != 0 ? float(bias[c]) : 0.0f;
  for (int k = 0; k < kernel_size; ++k) {
    const int source_t = t * stride + k * dilation - padding;
    if (source_t >= 0 && source_t < length) {
      sum += float(x[((long)batch * length + source_t) * channels + c]) *
             float(weight[(long)c * kernel_size + k]);
    }
  }
  if (activation == 1) sum *= 1.0f / (1.0f + metal::exp(-sum));
  out[gid] = T(sum);
}

#define instantiate_audio_conv(type_name, T)                                      \
  template [[host_name("audio_conv1d_" #type_name)]] [[kernel]] void             \
  audio_conv1d<T>(device const T *x [[buffer(0)]], device const T *weight [[buffer(1)]],\
    device const T *bias [[buffer(2)]], device T *out [[buffer(3)]],               \
    constant int &length [[buffer(4)]], constant int &in_channels [[buffer(5)]],   \
    constant int &out_length [[buffer(6)]], constant int &out_channels [[buffer(7)]],\
    constant int &kernel_size [[buffer(8)]], constant int &stride [[buffer(9)]],   \
    constant int &padding [[buffer(10)]], constant int &dilation [[buffer(11)]],   \
    constant int &has_bias [[buffer(12)]], constant int &batch_size [[buffer(13)]],\
    uint gid [[thread_position_in_grid]]);                                        \
  template [[host_name("audio_depthwise_conv1d_" #type_name)]] [[kernel]] void   \
  audio_depthwise_conv1d<T>(device const T *x [[buffer(0)]],                      \
    device const T *weight [[buffer(1)]], device const T *bias [[buffer(2)]],      \
    device T *out [[buffer(3)]], constant int &length [[buffer(4)]],              \
    constant int &channels [[buffer(5)]], constant int &out_length [[buffer(6)]], \
    constant int &kernel_size [[buffer(7)]], constant int &stride [[buffer(8)]],   \
    constant int &padding [[buffer(9)]], constant int &dilation [[buffer(10)]],    \
    constant int &has_bias [[buffer(11)]], constant int &activation [[buffer(12)]],\
    constant int &batch_size [[buffer(13)]],                                      \
    uint gid [[thread_position_in_grid]]);

instantiate_audio_conv(float32, float)
instantiate_audio_conv(float16, half)
instantiate_audio_conv(bfloat16, bf16)

}  // namespace mittens
