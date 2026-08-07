#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

template <typename T>
kernel void extract_patches_2d(
    device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
    constant int &height [[buffer(2)]], constant int &width [[buffer(3)]],
    constant int &channels [[buffer(4)]], constant int &out_height [[buffer(5)]],
    constant int &out_width [[buffer(6)]], constant int &kernel_h [[buffer(7)]],
    constant int &kernel_w [[buffer(8)]], constant int &stride_h [[buffer(9)]],
    constant int &stride_w [[buffer(10)]], constant int &pad_h [[buffer(11)]],
    constant int &pad_w [[buffer(12)]],
    uint patch [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const int patches_per_batch = out_height * out_width;
  const int batch = int(patch) / patches_per_batch;
  const int spatial = int(patch) - batch * patches_per_batch;
  const int oy = spatial / out_width;
  const int ox = spatial - oy * out_width;
  const int patch_dim = kernel_h * kernel_w * channels;
  const long out_base = (long)patch * patch_dim;
  for (int index = int(tid); index < patch_dim; index += int(threads)) {
    const int c = index % channels;
    const int kernel_index = index / channels;
    const int ky = kernel_index / kernel_w;
    const int kx = kernel_index - ky * kernel_w;
    const int iy = oy * stride_h + ky - pad_h;
    const int ix = ox * stride_w + kx - pad_w;
    T value = T(0.0f);
    if (iy >= 0 && iy < height && ix >= 0 && ix < width) {
      value = x[((long)batch * height * width + iy * width + ix) * channels + c];
    }
    out[out_base + index] = value;
  }
}

template <typename T>
kernel void extract_patches_3d(
    device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
    constant int &frames [[buffer(2)]], constant int &height [[buffer(3)]],
    constant int &width [[buffer(4)]], constant int &channels [[buffer(5)]],
    constant int &out_frames [[buffer(6)]], constant int &out_height [[buffer(7)]],
    constant int &out_width [[buffer(8)]], constant int &kernel_t [[buffer(9)]],
    constant int &kernel_h [[buffer(10)]], constant int &kernel_w [[buffer(11)]],
    constant int &stride_t [[buffer(12)]], constant int &stride_h [[buffer(13)]],
    constant int &stride_w [[buffer(14)]], constant int &pad_t [[buffer(15)]],
    constant int &pad_h [[buffer(16)]], constant int &pad_w [[buffer(17)]],
    uint patch [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const int patches_per_batch = out_frames * out_height * out_width;
  const int batch = int(patch) / patches_per_batch;
  int spatial = int(patch) - batch * patches_per_batch;
  const int ot = spatial / (out_height * out_width);
  spatial -= ot * out_height * out_width;
  const int oy = spatial / out_width, ox = spatial - oy * out_width;
  const int patch_dim = kernel_t * kernel_h * kernel_w * channels;
  const long out_base = (long)patch * patch_dim;
  for (int index = int(tid); index < patch_dim; index += int(threads)) {
    const int c = index % channels;
    int kernel_index = index / channels;
    const int kx = kernel_index % kernel_w; kernel_index /= kernel_w;
    const int ky = kernel_index % kernel_h;
    const int kt = kernel_index / kernel_h;
    const int it = ot * stride_t + kt * 1 - pad_t;
    const int iy = oy * stride_h + ky - pad_h;
    const int ix = ox * stride_w + kx - pad_w;
    T value = T(0.0f);
    if (it >= 0 && it < frames && iy >= 0 && iy < height && ix >= 0 && ix < width) {
      value = x[((((long)batch * frames + it) * height + iy) * width + ix) * channels + c];
    }
    out[out_base + index] = value;
  }
}

template <typename T>
kernel void interpolate_position_2d(
    device const T *table [[buffer(0)]], device T *out [[buffer(1)]],
    constant int &in_h [[buffer(2)]], constant int &in_w [[buffer(3)]],
    constant int &out_h [[buffer(4)]], constant int &out_w [[buffer(5)]],
    constant int &channels [[buffer(6)]], constant int &align_corners [[buffer(7)]],
    uint spatial [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const int oy = int(spatial) / out_w;
  const int ox = int(spatial) - oy * out_w;
  const float fy = align_corners != 0 && out_h > 1
      ? float(oy) * float(in_h - 1) / float(out_h - 1)
      : (float(oy) + 0.5f) * float(in_h) / float(out_h) - 0.5f;
  const float fx = align_corners != 0 && out_w > 1
      ? float(ox) * float(in_w - 1) / float(out_w - 1)
      : (float(ox) + 0.5f) * float(in_w) / float(out_w) - 0.5f;
  const float cy = metal::clamp(fy, 0.0f, float(in_h - 1));
  const float cx = metal::clamp(fx, 0.0f, float(in_w - 1));
  const int y0 = int(metal::floor(cy)), x0 = int(metal::floor(cx));
  const int y1 = metal::min(y0 + 1, in_h - 1), x1 = metal::min(x0 + 1, in_w - 1);
  const float wy = cy - float(y0), wx = cx - float(x0);
  for (int c = int(tid); c < channels; c += int(threads)) {
    const float a = metal::mix(float(table[((long)y0 * in_w + x0) * channels + c]),
                               float(table[((long)y0 * in_w + x1) * channels + c]), wx);
    const float b = metal::mix(float(table[((long)y1 * in_w + x0) * channels + c]),
                               float(table[((long)y1 * in_w + x1) * channels + c]), wx);
    out[((long)spatial * channels) + c] = T(metal::mix(a, b, wy));
  }
}

template <typename T>
kernel void avg_pool2d_tokens(
    device const T *x [[buffer(0)]], device T *out [[buffer(1)]],
    constant int &height [[buffer(2)]], constant int &width [[buffer(3)]],
    constant int &channels [[buffer(4)]], constant int &out_height [[buffer(5)]],
    constant int &out_width [[buffer(6)]], constant int &kernel_h [[buffer(7)]],
    constant int &kernel_w [[buffer(8)]], constant int &stride_h [[buffer(9)]],
    constant int &stride_w [[buffer(10)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
  const int outputs_per_batch = out_height * out_width;
  const int batch = int(output_index) / outputs_per_batch;
  const int spatial = int(output_index) - batch * outputs_per_batch;
  const int oy = spatial / out_width, ox = spatial - oy * out_width;
  const int y0 = oy * stride_h, x0 = ox * stride_w;
  const int y1 = metal::min(y0 + kernel_h, height);
  const int x1 = metal::min(x0 + kernel_w, width);
  const float inv_count = 1.0f / float(metal::max((y1 - y0) * (x1 - x0), 1));
  for (int c = int(tid); c < channels; c += int(threads)) {
    float sum = 0.0f;
    for (int y = y0; y < y1; ++y) for (int xx = x0; xx < x1; ++xx) {
      sum += float(x[((long)batch * height * width + y * width + xx) * channels + c]);
    }
    out[(long)output_index * channels + c] = T(sum * inv_count);
  }
}

template <typename T>
kernel void factorized_position_2d(
    device const int *position_ids [[buffer(0)]], device const T *table [[buffer(1)]],
    device const int *valid_mask [[buffer(2)]], device T *out [[buffer(3)]],
    constant int &tokens [[buffer(4)]], constant int &max_position [[buffer(5)]],
    constant int &channels [[buffer(6)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)tokens * channels;
  if ((long)gid >= total) return;
  const int token = int((long)gid / channels);
  const int channel = int((long)gid - (long)token * channels);
  if (valid_mask[token] == 0) {
    out[gid] = T(0.0f);
    return;
  }
  const int x = position_ids[(long)token * 2];
  const int y = position_ids[(long)token * 2 + 1];
  if (x < 0 || x >= max_position || y < 0 || y >= max_position) {
    out[gid] = T(0.0f);
    return;
  }
  out[gid] = T(float(table[(long)x * channels + channel]) +
               float(table[((long)max_position + y) * channels + channel]));
}

[[host_name("pool_tokens_by_position_zero")]]
kernel void pool_tokens_by_position_zero(
    device metal::atomic_float *out [[buffer(0)]],
    device metal::atomic_int *out_mask [[buffer(1)]],
    constant long &output_values [[buffer(2)]],
    constant long &output_tokens [[buffer(3)]],
    uint gid [[thread_position_in_grid]]) {
  if ((long)gid < output_values)
    metal::atomic_store_explicit(&out[gid], 0.0f, metal::memory_order_relaxed);
  if ((long)gid < output_tokens)
    metal::atomic_store_explicit(&out_mask[gid], 0, metal::memory_order_relaxed);
}

template <typename T>
kernel void pool_tokens_by_position_scatter(
    device const T *x [[buffer(0)]], device const int *position_ids [[buffer(1)]],
    device const int *valid_mask [[buffer(2)]],
    device metal::atomic_float *out [[buffer(3)]],
    device metal::atomic_int *out_mask [[buffer(4)]],
    constant int &batch_size [[buffer(5)]], constant int &tokens [[buffer(6)]],
    constant int &channels [[buffer(7)]], constant int &output_length [[buffer(8)]],
    constant int &kernel_size [[buffer(9)]], constant int &source_width [[buffer(10)]],
    constant float &scale [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const long total = (long)batch_size * tokens * channels;
  if ((long)gid >= total) return;
  const int token = int((long)gid / channels);
  const int channel = int((long)gid - (long)token * channels);
  if (valid_mask[token] == 0) return;
  const int px = position_ids[(long)token * 2];
  const int py = position_ids[(long)token * 2 + 1];
  if (px < 0 || py < 0) return;
  const int pooled_width = source_width / kernel_size;
  const int bucket = px / kernel_size + pooled_width * (py / kernel_size);
  if (bucket < 0 || bucket >= output_length) return;
  const int batch = token / tokens;
  const int output_token = batch * output_length + bucket;
  metal::atomic_fetch_add_explicit(
      &out[(long)output_token * channels + channel], float(x[gid]) * scale,
      metal::memory_order_relaxed);
  if (channel == 0)
    metal::atomic_store_explicit(&out_mask[output_token], 1, metal::memory_order_relaxed);
}

#define instantiate_patch_ops(type_name, T)                                      \
  template [[host_name("extract_patches_2d_" #type_name)]] [[kernel]] void       \
  extract_patches_2d<T>(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],\
    constant int &height [[buffer(2)]], constant int &width [[buffer(3)]],         \
    constant int &channels [[buffer(4)]], constant int &out_height [[buffer(5)]],  \
    constant int &out_width [[buffer(6)]], constant int &kernel_h [[buffer(7)]],   \
    constant int &kernel_w [[buffer(8)]], constant int &stride_h [[buffer(9)]],    \
    constant int &stride_w [[buffer(10)]], constant int &pad_h [[buffer(11)]],     \
    constant int &pad_w [[buffer(12)]], uint patch [[threadgroup_position_in_grid]],\
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]);\
  template [[host_name("extract_patches_3d_" #type_name)]] [[kernel]] void      \
  extract_patches_3d<T>(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],\
    constant int &frames [[buffer(2)]], constant int &height [[buffer(3)]],       \
    constant int &width [[buffer(4)]], constant int &channels [[buffer(5)]],     \
    constant int &out_frames [[buffer(6)]], constant int &out_height [[buffer(7)]],\
    constant int &out_width [[buffer(8)]], constant int &kernel_t [[buffer(9)]], \
    constant int &kernel_h [[buffer(10)]], constant int &kernel_w [[buffer(11)]],\
    constant int &stride_t [[buffer(12)]], constant int &stride_h [[buffer(13)]],\
    constant int &stride_w [[buffer(14)]], constant int &pad_t [[buffer(15)]],   \
    constant int &pad_h [[buffer(16)]], constant int &pad_w [[buffer(17)]],      \
    uint patch [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\
    uint threads [[threads_per_threadgroup]]);                                   \
  template [[host_name("interpolate_position_2d_" #type_name)]] [[kernel]] void  \
  interpolate_position_2d<T>(device const T *table [[buffer(0)]], device T *out [[buffer(1)]],\
    constant int &in_h [[buffer(2)]], constant int &in_w [[buffer(3)]],            \
    constant int &out_h [[buffer(4)]], constant int &out_w [[buffer(5)]],          \
    constant int &channels [[buffer(6)]], constant int &align_corners [[buffer(7)]],\
    uint spatial [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],\
    uint threads [[threads_per_threadgroup]]);                                    \
  template [[host_name("avg_pool2d_tokens_" #type_name)]] [[kernel]] void        \
  avg_pool2d_tokens<T>(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],\
    constant int &height [[buffer(2)]], constant int &width [[buffer(3)]],         \
    constant int &channels [[buffer(4)]], constant int &out_height [[buffer(5)]],  \
    constant int &out_width [[buffer(6)]], constant int &kernel_h [[buffer(7)]],   \
    constant int &kernel_w [[buffer(8)]], constant int &stride_h [[buffer(9)]],    \
    constant int &stride_w [[buffer(10)]], uint output_index [[threadgroup_position_in_grid]],\
    uint tid [[thread_position_in_threadgroup]], uint threads [[threads_per_threadgroup]]);     \
  template [[host_name("factorized_position_2d_" #type_name)]] [[kernel]] void \
  factorized_position_2d<T>(device const int *position_ids [[buffer(0)]],        \
    device const T *table [[buffer(1)]], device const int *valid_mask [[buffer(2)]],\
    device T *out [[buffer(3)]], constant int &tokens [[buffer(4)]],              \
    constant int &max_position [[buffer(5)]], constant int &channels [[buffer(6)]],\
    uint gid [[thread_position_in_grid]]);                                        \
  template [[host_name("pool_tokens_by_position_scatter_" #type_name)]] [[kernel]] void\
  pool_tokens_by_position_scatter<T>(device const T *x [[buffer(0)]],            \
    device const int *position_ids [[buffer(1)]], device const int *valid_mask [[buffer(2)]],\
    device metal::atomic_float *out [[buffer(3)]], device metal::atomic_int *out_mask [[buffer(4)]],\
    constant int &batch_size [[buffer(5)]], constant int &tokens [[buffer(6)]],   \
    constant int &channels [[buffer(7)]], constant int &output_length [[buffer(8)]],\
    constant int &kernel_size [[buffer(9)]], constant int &source_width [[buffer(10)]],\
    constant float &scale [[buffer(11)]],                                        \
    uint gid [[thread_position_in_grid]]);

instantiate_patch_ops(float32, float)
instantiate_patch_ops(float16, half)
instantiate_patch_ops(bfloat16, bf16)

}  // namespace mittens
