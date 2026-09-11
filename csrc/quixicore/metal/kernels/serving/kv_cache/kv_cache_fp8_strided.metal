// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include "tk.metal"
#include <metal_stdlib>
using namespace metal;
using namespace mittens;

// FP8 E4M3 pages use byte storage and explicit 64-bit physical strides.
template <typename T>
kernel void kv_cache_scatter_fp8_strided(
    device const T* key [[buffer(0)]], device const T* value [[buffer(1)]],
    device const long* slot_mapping [[buffer(2)]],
    device uchar* key_cache [[buffer(3)]],
    device uchar* value_cache [[buffer(4)]],
    constant int& num_heads [[buffer(5)]],
    constant int& head_size [[buffer(6)]],
    constant int& block_size [[buffer(7)]],
    constant ulong& cache_block_stride [[buffer(8)]],
    device const float* k_scale [[buffer(9)]],
    device const float* v_scale [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  const long slot = slot_mapping[token];
  if (slot < 0) {
    return;
  }

  const long block = slot / block_size;
  const long block_offset = slot % block_size;
  const int row_elems = num_heads * head_size;
  const long src_base = (long)token * row_elems;
  const long dst_base = block * cache_block_stride + block_offset * row_elems;

  for (int i = (int)tid; i < row_elems; i += (int)tptg) {
    key_cache[dst_base + i] =
        tk_e4m3_encode(float(key[src_base + i]) / k_scale[i / head_size]);
    value_cache[dst_base + i] =
        tk_e4m3_encode(float(value[src_base + i]) / v_scale[i / head_size]);
  }
}

template <typename T>
kernel void kv_cache_gather_range_fp8(
    device const uchar* key_cache [[buffer(0)]],
    device const uchar* value_cache [[buffer(1)]],
    device T* key_out [[buffer(2)]], device T* value_out [[buffer(3)]],
    device const int* block_table [[buffer(4)]],
    constant int& token_start [[buffer(5)]],
    constant int& num_tokens [[buffer(6)]],
    constant int& num_blocks [[buffer(7)]],
    constant int& block_size [[buffer(8)]],
    constant int& num_heads [[buffer(9)]],
    constant int& head_size [[buffer(10)]],
    constant long& cache_block_stride [[buffer(11)]],
    device const float* k_scale [[buffer(12)]],
    device const float* v_scale [[buffer(13)]],
    uint token [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
  if ((int)token >= num_tokens) {
    return;
  }

  const int logical_token = token_start + (int)token;
  const int table_col = logical_token / block_size;
  const int block_offset = logical_token - table_col * block_size;
  const int block = block_table[table_col];
  const int row_elems = num_heads * head_size;
  const long out_base = (long)token * row_elems;

  if (block < 0 || block >= num_blocks) {
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
      key_out[out_base + i] = T(0);
      value_out[out_base + i] = T(0);
    }
    return;
  }

  const long cache_base =
      (long)block * cache_block_stride + (long)block_offset * row_elems;
  for (int i = (int)tid; i < row_elems; i += (int)tptg) {
    key_out[out_base + i] = T(float(tk_e4m3_decode(key_cache[cache_base + i])) *
                              k_scale[i / head_size]);
    value_out[out_base + i] =
        T(float(tk_e4m3_decode(value_cache[cache_base + i])) *
          v_scale[i / head_size]);
  }
}

template [[host_name("kv_cache_scatter_fp8_strided_float32")]] [[kernel]] void
kv_cache_scatter_fp8_strided<float>(device const float* key [[buffer(0)]],
                                    device const float* value [[buffer(1)]],
                                    device const long* slot_mapping
                                    [[buffer(2)]],
                                    device uchar* key_cache [[buffer(3)]],
                                    device uchar* value_cache [[buffer(4)]],
                                    constant int& num_heads [[buffer(5)]],
                                    constant int& head_size [[buffer(6)]],
                                    constant int& block_size [[buffer(7)]],
                                    constant ulong& cache_block_stride
                                    [[buffer(8)]],
                                    device const float* k_scale [[buffer(9)]],
                                    device const float* v_scale [[buffer(10)]],
                                    uint token [[threadgroup_position_in_grid]],
                                    uint tid [[thread_position_in_threadgroup]],
                                    uint tptg [[threads_per_threadgroup]]);

template [[host_name("kv_cache_gather_range_fp8_float32")]] [[kernel]] void
kv_cache_gather_range_fp8<float>(device const uchar* key_cache [[buffer(0)]],
                                 device const uchar* value_cache [[buffer(1)]],
                                 device float* key_out [[buffer(2)]],
                                 device float* value_out [[buffer(3)]],
                                 device const int* block_table [[buffer(4)]],
                                 constant int& token_start [[buffer(5)]],
                                 constant int& num_tokens [[buffer(6)]],
                                 constant int& num_blocks [[buffer(7)]],
                                 constant int& block_size [[buffer(8)]],
                                 constant int& num_heads [[buffer(9)]],
                                 constant int& head_size [[buffer(10)]],
                                 constant long& cache_block_stride
                                 [[buffer(11)]],
                                 device const float* k_scale [[buffer(12)]],
                                 device const float* v_scale [[buffer(13)]],
                                 uint token [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tptg [[threads_per_threadgroup]]);

template [[host_name("kv_cache_scatter_fp8_strided_float16")]] [[kernel]] void
kv_cache_scatter_fp8_strided<half>(device const half* key [[buffer(0)]],
                                   device const half* value [[buffer(1)]],
                                   device const long* slot_mapping
                                   [[buffer(2)]],
                                   device uchar* key_cache [[buffer(3)]],
                                   device uchar* value_cache [[buffer(4)]],
                                   constant int& num_heads [[buffer(5)]],
                                   constant int& head_size [[buffer(6)]],
                                   constant int& block_size [[buffer(7)]],
                                   constant ulong& cache_block_stride
                                   [[buffer(8)]],
                                   device const float* k_scale [[buffer(9)]],
                                   device const float* v_scale [[buffer(10)]],
                                   uint token [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint tptg [[threads_per_threadgroup]]);

template [[host_name("kv_cache_gather_range_fp8_float16")]] [[kernel]] void
kv_cache_gather_range_fp8<half>(device const uchar* key_cache [[buffer(0)]],
                                device const uchar* value_cache [[buffer(1)]],
                                device half* key_out [[buffer(2)]],
                                device half* value_out [[buffer(3)]],
                                device const int* block_table [[buffer(4)]],
                                constant int& token_start [[buffer(5)]],
                                constant int& num_tokens [[buffer(6)]],
                                constant int& num_blocks [[buffer(7)]],
                                constant int& block_size [[buffer(8)]],
                                constant int& num_heads [[buffer(9)]],
                                constant int& head_size [[buffer(10)]],
                                constant long& cache_block_stride
                                [[buffer(11)]],
                                device const float* k_scale [[buffer(12)]],
                                device const float* v_scale [[buffer(13)]],
                                uint token [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tptg [[threads_per_threadgroup]]);

template [[host_name("kv_cache_scatter_fp8_strided_bfloat16")]] [[kernel]] void
kv_cache_scatter_fp8_strided<bf16>(device const bf16* key [[buffer(0)]],
                                   device const bf16* value [[buffer(1)]],
                                   device const long* slot_mapping
                                   [[buffer(2)]],
                                   device uchar* key_cache [[buffer(3)]],
                                   device uchar* value_cache [[buffer(4)]],
                                   constant int& num_heads [[buffer(5)]],
                                   constant int& head_size [[buffer(6)]],
                                   constant int& block_size [[buffer(7)]],
                                   constant ulong& cache_block_stride
                                   [[buffer(8)]],
                                   device const float* k_scale [[buffer(9)]],
                                   device const float* v_scale [[buffer(10)]],
                                   uint token [[threadgroup_position_in_grid]],
                                   uint tid [[thread_position_in_threadgroup]],
                                   uint tptg [[threads_per_threadgroup]]);

template [[host_name("kv_cache_gather_range_fp8_bfloat16")]] [[kernel]] void
kv_cache_gather_range_fp8<bf16>(device const uchar* key_cache [[buffer(0)]],
                                device const uchar* value_cache [[buffer(1)]],
                                device bf16* key_out [[buffer(2)]],
                                device bf16* value_out [[buffer(3)]],
                                device const int* block_table [[buffer(4)]],
                                constant int& token_start [[buffer(5)]],
                                constant int& num_tokens [[buffer(6)]],
                                constant int& num_blocks [[buffer(7)]],
                                constant int& block_size [[buffer(8)]],
                                constant int& num_heads [[buffer(9)]],
                                constant int& head_size [[buffer(10)]],
                                constant long& cache_block_stride
                                [[buffer(11)]],
                                device const float* k_scale [[buffer(12)]],
                                device const float* v_scale [[buffer(13)]],
                                uint token [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tptg [[threads_per_threadgroup]]);
