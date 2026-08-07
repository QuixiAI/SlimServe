#include <metal_stdlib>
#include "tk.metal"

using namespace metal;

namespace mittens {

METAL_FUNC uint base_q_load_code(
    device const uchar* codes, int row, int column, int columns, int bits) {
  const long row_bytes = (long(columns) * bits) >> 3;
  const int bit_index = column * bits;
  const int byte_index = bit_index >> 3;
  const int shift = bit_index & 7;
  device const uchar* row_codes = codes + long(row) * row_bytes;
  uint value = uint(row_codes[byte_index]);
  if (shift + bits > 8) value |= uint(row_codes[byte_index + 1]) << 8;
  return (value >> shift) & ((1u << bits) - 1u);
}

METAL_FUNC float base_q_load_scale_value(
    device const uchar* values, long index, int scale_type) {
  if (scale_type == 0) return float(((device const bf16*)values)[index]);
  if (scale_type == 1) return float(((device const half*)values)[index]);
  if (scale_type == 2) return tk_e8m0_decode_f32(uint(values[index]));
  return float(tk_e4m3_decode(values[index]));
}

METAL_FUNC float base_q_load_weight(
    device const uchar* codes, device const uchar* scales,
    device const uchar* biases, int row, int column, int columns,
    int group_size, int bits, int scale_type, int symmetric) {
  const int groups_per_row = columns / group_size;
  const long group = long(row) * groups_per_row + column / group_size;
  const float scale = base_q_load_scale_value(scales, group, scale_type);
  const uint code = base_q_load_code(codes, row, column, columns, bits);
  if (symmetric != 0) {
    const int signed_code = int(code) - (1 << (bits - 1));
    return float(signed_code) * scale;
  }
  return float(code) * scale + base_q_load_scale_value(biases, group, scale_type);
}

template <typename T>
METAL_FUNC float base_q_gemv_partial(
    device const uchar* codes, device const uchar* scales,
    device const uchar* biases, device const T* input, int row, int inner,
    int group_size, int bits, int scale_type, int symmetric, uint lane) {
  float accumulator = 0.0f;
  constexpr int values_per_lane = 8;
  const int groups_per_row = inner / group_size;
  for (int first = int(lane) * values_per_lane; first < inner;
       first += 32 * values_per_lane) {
    const long group = long(row) * groups_per_row + first / group_size;
    const float scale = base_q_load_scale_value(scales, group, scale_type);
    const float bias = symmetric != 0
        ? 0.0f
        : base_q_load_scale_value(biases, group, scale_type);
    for (int offset = 0; offset < values_per_lane; ++offset) {
      const int k = first + offset;
      if (k >= inner) break;
      const uint code = base_q_load_code(codes, row, k, inner, bits);
      const float weight = symmetric != 0
          ? float(int(code) - (1 << (bits - 1))) * scale
          : float(code) * scale + bias;
      accumulator += weight * float(input[k]);
    }
  }
  return accumulator;
}

template <typename T>
kernel void base_qdequant_kernel(
    device T* output [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    constant int& rows [[buffer(4)]],
    constant int& columns [[buffer(5)]],
    constant int& group_size [[buffer(6)]],
    constant int& bits [[buffer(7)]],
    constant int& scale_type [[buffer(8)]],
    constant int& symmetric [[buffer(9)]],
    uint tid [[thread_position_in_grid]]) {
  const long total = long(rows) * columns;
  if (long(tid) >= total) return;
  const int row = int(long(tid) / columns);
  const int column = int(long(tid) - long(row) * columns);
  output[tid] = T(base_q_load_weight(
      codes, scales, biases, row, column, columns, group_size, bits,
      scale_type, symmetric));
}

template <typename T>
kernel void base_qgemv_kernel(
    device T* output [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    device const T* input [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    constant int& inner [[buffer(6)]],
    constant int& columns [[buffer(7)]],
    constant int& group_size [[buffer(8)]],
    constant int& bits [[buffer(9)]],
    constant int& scale_type [[buffer(10)]],
    constant int& symmetric [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgid.x);
  float accumulator = base_q_gemv_partial(
      codes, scales, biases, input, row, inner, group_size, bits, scale_type,
      symmetric, lane);
  accumulator = simd_sum(accumulator);
  if (lane == 0) output[row * columns] = T(accumulator);
}

template <typename T>
kernel void base_qgemv_qkv_kernel(
    device T* q_output [[buffer(0)]],
    device T* k_output [[buffer(1)]],
    device T* v_output [[buffer(2)]],
    device const uchar* q_codes [[buffer(3)]],
    device const uchar* q_scales [[buffer(4)]],
    device const uchar* q_biases [[buffer(5)]],
    device const uchar* k_codes [[buffer(6)]],
    device const uchar* k_scales [[buffer(7)]],
    device const uchar* k_biases [[buffer(8)]],
    device const uchar* v_codes [[buffer(9)]],
    device const uchar* v_scales [[buffer(10)]],
    device const uchar* v_biases [[buffer(11)]],
    device const T* input [[buffer(12)]],
    constant int& q_rows [[buffer(13)]],
    constant int& k_rows [[buffer(14)]],
    constant int& v_rows [[buffer(15)]],
    constant int& inner [[buffer(16)]],
    constant int& group_size [[buffer(17)]],
    constant int& bits [[buffer(18)]],
    constant int& scale_type [[buffer(19)]],
    constant int& symmetric [[buffer(20)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int combined = int(tgid.x);
  device const uchar* codes;
  device const uchar* scales;
  device const uchar* biases;
  device T* output;
  int row;
  if (combined < q_rows) {
    codes = q_codes; scales = q_scales; biases = q_biases;
    output = q_output; row = combined;
  } else if (combined < q_rows + k_rows) {
    codes = k_codes; scales = k_scales; biases = k_biases;
    output = k_output; row = combined - q_rows;
  } else {
    codes = v_codes; scales = v_scales; biases = v_biases;
    output = v_output; row = combined - q_rows - k_rows;
  }
  float accumulator = base_q_gemv_partial(
      codes, scales, biases, input, row, inner, group_size, bits, scale_type,
      symmetric, lane);
  accumulator = simd_sum(accumulator);
  if (lane == 0) output[row] = T(accumulator);
}

template <typename T>
kernel void base_qgemv_swiglu_kernel(
    device T* output [[buffer(0)]],
    device const uchar* gate_codes [[buffer(1)]],
    device const uchar* gate_scales [[buffer(2)]],
    device const uchar* gate_biases [[buffer(3)]],
    device const uchar* up_codes [[buffer(4)]],
    device const uchar* up_scales [[buffer(5)]],
    device const uchar* up_biases [[buffer(6)]],
    device const T* input [[buffer(7)]],
    constant int& rows [[buffer(8)]],
    constant int& inner [[buffer(9)]],
    constant int& group_size [[buffer(10)]],
    constant int& bits [[buffer(11)]],
    constant int& scale_type [[buffer(12)]],
    constant int& symmetric [[buffer(13)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgid.x);
  constexpr int values_per_lane = 8;
  const int groups_per_row = inner / group_size;
  float gate_accumulator = 0.0f;
  float up_accumulator = 0.0f;
  for (int first = int(lane) * values_per_lane; first < inner;
       first += 32 * values_per_lane) {
    const long group = long(row) * groups_per_row + first / group_size;
    const float gate_scale = base_q_load_scale_value(gate_scales, group, scale_type);
    const float up_scale = base_q_load_scale_value(up_scales, group, scale_type);
    const float gate_bias = symmetric != 0
        ? 0.0f : base_q_load_scale_value(gate_biases, group, scale_type);
    const float up_bias = symmetric != 0
        ? 0.0f : base_q_load_scale_value(up_biases, group, scale_type);
    for (int offset = 0; offset < values_per_lane; ++offset) {
      const int k = first + offset;
      if (k >= inner) break;
      const float activation = float(input[k]);
      const uint gate_code = base_q_load_code(gate_codes, row, k, inner, bits);
      const uint up_code = base_q_load_code(up_codes, row, k, inner, bits);
      const float gate_weight = symmetric != 0
          ? float(int(gate_code) - (1 << (bits - 1))) * gate_scale
          : float(gate_code) * gate_scale + gate_bias;
      const float up_weight = symmetric != 0
          ? float(int(up_code) - (1 << (bits - 1))) * up_scale
          : float(up_code) * up_scale + up_bias;
      gate_accumulator += gate_weight * activation;
      up_accumulator += up_weight * activation;
    }
  }
  gate_accumulator = simd_sum(gate_accumulator);
  up_accumulator = simd_sum(up_accumulator);
  if (lane == 0) {
    const float silu = gate_accumulator /
        (1.0f + metal::exp(-gate_accumulator));
    output[row] = T(silu * up_accumulator);
  }
}

template <typename T>
kernel void base_qgemm_kernel(
    device T* output [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    device const T* input [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    constant int& inner [[buffer(6)]],
    constant int& columns [[buffer(7)]],
    constant int& group_size [[buffer(8)]],
    constant int& bits [[buffer(9)]],
    constant int& scale_type [[buffer(10)]],
    constant int& symmetric [[buffer(11)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int row = int(tgid.y);
  const int column = int(tgid.x) * 32 + int(lane);
  if (column >= columns) return;
  float accumulator = 0.0f;
  for (int k = 0; k < inner; ++k) {
    const float weight = base_q_load_weight(
        codes, scales, biases, row, k, inner, group_size, bits,
        scale_type, symmetric);
    accumulator += weight * float(input[long(k) * columns + column]);
  }
  output[long(row) * columns + column] = T(accumulator);
}

template <typename T>
kernel void base_qembedding_kernel(
    device T* output [[buffer(0)]],
    device const uchar* codes [[buffer(1)]],
    device const uchar* scales [[buffer(2)]],
    device const uchar* biases [[buffer(3)]],
    device const int* ids [[buffer(4)]],
    constant int& rows [[buffer(5)]],
    constant int& columns [[buffer(6)]],
    constant int& tokens [[buffer(7)]],
    constant int& group_size [[buffer(8)]],
    constant int& bits [[buffer(9)]],
    constant int& scale_type [[buffer(10)]],
    constant int& symmetric [[buffer(11)]],
    uint tid [[thread_position_in_grid]]) {
  const long total = long(tokens) * columns;
  if (long(tid) >= total) return;
  const int token = int(long(tid) / columns);
  const int column = int(long(tid) - long(token) * columns);
  const int row = ids[token];
  output[tid] = row < 0 || row >= rows
      ? T(0.0f)
      : T(base_q_load_weight(
          codes, scales, biases, row, column, columns, group_size, bits,
          scale_type, symmetric));
}

// Decode a 32x32 BaseQN weight tile laid out as logical (output, K) rows
// directly into the column-layout register fragment consumed by mma_ABt.
// Expert stacks are ordinary leading dimensions on the same separate planes;
// row_base selects the active expert without introducing another packed ABI.
template <typename T, typename RT>
METAL_FUNC void base_q_dequant_into_register_col(
    thread RT& destination, device const uchar* codes,
    device const uchar* scales, device const uchar* biases, int row_base,
    int columns, int output_tile, int inner_tile, int group_size, int bits,
    int scale_type, int symmetric, uint lane) {
  const int quad = int(lane) / 4;
  const int simd_x = (quad & 4) + (int(lane) / 2) % 4;
  const int simd_y = (quad & 2) * 2 + (int(lane) % 2) * 2;
  #pragma clang loop unroll(full)
  for (int i = 0; i < RT::height; ++i) {
    #pragma clang loop unroll(full)
    for (int j = 0; j < RT::width; ++j) {
      const int row = output_tile * RT::rows + i * TILE_DIM + simd_y;
      const int column = inner_tile * RT::cols + j * TILE_DIM + simd_x;
      destination.tiles[i][j].data.thread_elements()[0] =
          typename RT::dtype(base_q_load_weight(
              codes, scales, biases, row_base + row, column, columns,
              group_size, bits, scale_type, symmetric));
      destination.tiles[i][j].data.thread_elements()[1] =
          typename RT::dtype(base_q_load_weight(
              codes, scales, biases, row_base + row + 1, column, columns,
              group_size, bits, scale_type, symmetric));
    }
  }
}

// Padded grouped expert projection. A row tile belongs to exactly one expert,
// as established by QuixiCore's existing MoE align/gather schedule. Codes are
// (E,N,K*bits/8), scales/biases are (E,N,K/group_size), and weights contract as
// A @ dequant(W[e])^T without materializing an expert stack.
template <typename T>
kernel void base_qmoe_gemm_kernel(
    device T* output [[buffer(0)]],
    device T* input [[buffer(1)]],
    device const uchar* codes [[buffer(2)]],
    device const uchar* scales [[buffer(3)]],
    device const uchar* biases [[buffer(4)]],
    device const int* expert_of_tile [[buffer(5)]],
    constant int& total_rows [[buffer(6)]],
    constant int& inner [[buffer(7)]],
    constant int& output_rows [[buffer(8)]],
    constant int& group_size [[buffer(9)]],
    constant int& bits [[buffer(10)]],
    constant int& scale_type [[buffer(11)]],
    constant int& symmetric [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int output_tile = int(tgid.x);
  const int row_tile = int(tgid.y);
  const int expert = expert_of_tile[row_tile];
  if (expert < 0) return;

  using global_layout = gl<T, 1, 1, -1, -1>;
  global_layout input_layout(input, nullptr, nullptr, total_rows, inner);
  global_layout output_layout(output, nullptr, nullptr, total_rows, output_rows);
  constexpr int TILE = 32;
  rt<T, TILE, TILE> input_fragment;
  rt<T, TILE, TILE, ducks::rt_layout::col> weight_fragment;
  rt<float, TILE, TILE> accumulator;
  zero(accumulator);
  const int row_base = expert * output_rows;
  for (int inner_tile = 0; inner_tile < inner / TILE; ++inner_tile) {
    load(input_fragment, input_layout, {0, 0, row_tile, inner_tile}, lane);
    base_q_dequant_into_register_col<T>(
        weight_fragment, codes, scales, biases, row_base, inner,
        output_tile, inner_tile, group_size, bits, scale_type, symmetric, lane);
    mma_ABt(accumulator, input_fragment, weight_fragment, accumulator);
  }
  store(output_layout, accumulator, {0, 0, row_tile, output_tile}, lane);
}

// Fused expert GEMM1. The BaseQN output-row axis is [gate(inter), up(inter)],
// matching the established quantized-MoE contract.
template <typename T>
kernel void base_qmoe_swiglu_kernel(
    device T* output [[buffer(0)]],
    device T* input [[buffer(1)]],
    device const uchar* codes [[buffer(2)]],
    device const uchar* scales [[buffer(3)]],
    device const uchar* biases [[buffer(4)]],
    device const int* expert_of_tile [[buffer(5)]],
    constant int& total_rows [[buffer(6)]],
    constant int& inner [[buffer(7)]],
    constant int& intermediate [[buffer(8)]],
    constant int& group_size [[buffer(9)]],
    constant int& bits [[buffer(10)]],
    constant int& scale_type [[buffer(11)]],
    constant int& symmetric [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint warp [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
  const int output_tile = int(tgid.x);
  const int row_tile = int(tgid.y);
  const int expert = expert_of_tile[row_tile];
  if (expert < 0) return;

  using global_layout = gl<T, 1, 1, -1, -1>;
  global_layout input_layout(input, nullptr, nullptr, total_rows, inner);
  global_layout output_layout(output, nullptr, nullptr, total_rows, intermediate);
  constexpr int TILE = 32;
  const int packed_rows = 2 * intermediate;
  const int row_base = expert * packed_rows;
  const int up_tile = intermediate / TILE + output_tile;
  rt<T, TILE, TILE> input_fragment;
  rt<T, TILE, TILE, ducks::rt_layout::col> gate_weight, up_weight;
  rt<float, TILE, TILE> gate, up;
  threadgroup st<float, TILE, TILE> partials[3];
  zero(gate);
  zero(up);
  for (int inner_tile = int(warp); inner_tile < inner / TILE;
       inner_tile += 4) {
    load(input_fragment, input_layout, {0, 0, row_tile, inner_tile}, lane);
    base_q_dequant_into_register_col<T>(
        gate_weight, codes, scales, biases, row_base, inner,
        output_tile, inner_tile, group_size, bits, scale_type, symmetric, lane);
    base_q_dequant_into_register_col<T>(
        up_weight, codes, scales, biases, row_base, inner,
        up_tile, inner_tile, group_size, bits, scale_type, symmetric, lane);
    mma_ABt(gate, input_fragment, gate_weight, gate);
    mma_ABt(up, input_fragment, up_weight, up);
  }
  if (warp > 0) store(partials[warp - 1], gate, lane);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (warp == 0) {
    rt<float, TILE, TILE> partial;
    #pragma clang loop unroll(full)
    for (int other = 0; other < 3; ++other) {
      load(partial, partials[other], lane);
      add(gate, gate, partial);
    }
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (warp > 0) store(partials[warp - 1], up, lane);
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (warp != 0) return;
  {
    rt<float, TILE, TILE> partial;
    #pragma clang loop unroll(full)
    for (int other = 0; other < 3; ++other) {
      load(partial, partials[other], lane);
      add(up, up, partial);
    }
  }
  silu(gate, gate);
  mul(gate, gate, up);
  store(output_layout, gate, {0, 0, row_tile, output_tile}, lane);
}

#define INSTANTIATE_BASE_Q(T, name)                                          \
  template [[host_name("base_qdequant_" name)]] [[kernel]]                 \
  void base_qdequant_kernel<T>(                                              \
      device T* output [[buffer(0)]],                                        \
      device const uchar* codes [[buffer(1)]],                               \
      device const uchar* scales [[buffer(2)]],                              \
      device const uchar* biases [[buffer(3)]],                              \
      constant int& rows [[buffer(4)]], constant int& columns [[buffer(5)]], \
      constant int& group_size [[buffer(6)]], constant int& bits [[buffer(7)]], \
      constant int& scale_type [[buffer(8)]], constant int& symmetric [[buffer(9)]], \
      uint tid [[thread_position_in_grid]]);                                 \
  template [[host_name("base_qgemv_" name)]] [[kernel]]                    \
  void base_qgemv_kernel<T>(                                                 \
      device T* output [[buffer(0)]],                                        \
      device const uchar* codes [[buffer(1)]],                               \
      device const uchar* scales [[buffer(2)]],                              \
      device const uchar* biases [[buffer(3)]], device const T* input [[buffer(4)]], \
      constant int& rows [[buffer(5)]], constant int& inner [[buffer(6)]],   \
      constant int& columns [[buffer(7)]], constant int& group_size [[buffer(8)]], \
      constant int& bits [[buffer(9)]], constant int& scale_type [[buffer(10)]], \
      constant int& symmetric [[buffer(11)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);                              \
  template [[host_name("base_qgemm_" name)]] [[kernel]]                    \
  void base_qgemm_kernel<T>(                                                 \
      device T* output [[buffer(0)]],                                        \
      device const uchar* codes [[buffer(1)]],                               \
      device const uchar* scales [[buffer(2)]],                              \
      device const uchar* biases [[buffer(3)]], device const T* input [[buffer(4)]], \
      constant int& rows [[buffer(5)]], constant int& inner [[buffer(6)]],   \
      constant int& columns [[buffer(7)]], constant int& group_size [[buffer(8)]], \
      constant int& bits [[buffer(9)]], constant int& scale_type [[buffer(10)]], \
      constant int& symmetric [[buffer(11)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);                              \
  template [[host_name("base_qgemv_qkv_" name)]] [[kernel]]               \
  void base_qgemv_qkv_kernel<T>(                                             \
      device T* q_output [[buffer(0)]], device T* k_output [[buffer(1)]],    \
      device T* v_output [[buffer(2)]],                                      \
      device const uchar* q_codes [[buffer(3)]],                             \
      device const uchar* q_scales [[buffer(4)]],                            \
      device const uchar* q_biases [[buffer(5)]],                            \
      device const uchar* k_codes [[buffer(6)]],                             \
      device const uchar* k_scales [[buffer(7)]],                            \
      device const uchar* k_biases [[buffer(8)]],                            \
      device const uchar* v_codes [[buffer(9)]],                             \
      device const uchar* v_scales [[buffer(10)]],                           \
      device const uchar* v_biases [[buffer(11)]], device const T* input [[buffer(12)]], \
      constant int& q_rows [[buffer(13)]], constant int& k_rows [[buffer(14)]], \
      constant int& v_rows [[buffer(15)]], constant int& inner [[buffer(16)]], \
      constant int& group_size [[buffer(17)]], constant int& bits [[buffer(18)]], \
      constant int& scale_type [[buffer(19)]], constant int& symmetric [[buffer(20)]], \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);                              \
  template [[host_name("base_qgemv_swiglu_" name)]] [[kernel]]            \
  void base_qgemv_swiglu_kernel<T>(                                          \
      device T* output [[buffer(0)]],                                        \
      device const uchar* gate_codes [[buffer(1)]],                          \
      device const uchar* gate_scales [[buffer(2)]],                         \
      device const uchar* gate_biases [[buffer(3)]],                         \
      device const uchar* up_codes [[buffer(4)]],                            \
      device const uchar* up_scales [[buffer(5)]],                           \
      device const uchar* up_biases [[buffer(6)]], device const T* input [[buffer(7)]], \
      constant int& rows [[buffer(8)]], constant int& inner [[buffer(9)]],   \
      constant int& group_size [[buffer(10)]], constant int& bits [[buffer(11)]], \
      constant int& scale_type [[buffer(12)]], constant int& symmetric [[buffer(13)]], \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);                              \
  template [[host_name("base_qembedding_" name)]] [[kernel]]               \
  void base_qembedding_kernel<T>(                                            \
      device T* output [[buffer(0)]],                                        \
      device const uchar* codes [[buffer(1)]],                               \
      device const uchar* scales [[buffer(2)]],                              \
      device const uchar* biases [[buffer(3)]], device const int* ids [[buffer(4)]], \
      constant int& rows [[buffer(5)]], constant int& columns [[buffer(6)]], \
      constant int& tokens [[buffer(7)]], constant int& group_size [[buffer(8)]], \
      constant int& bits [[buffer(9)]], constant int& scale_type [[buffer(10)]], \
      constant int& symmetric [[buffer(11)]],                               \
      uint tid [[thread_position_in_grid]]);                                \
  template [[host_name("base_qmoe_gemm_" name)]] [[kernel]]                \
  void base_qmoe_gemm_kernel<T>(                                            \
      device T* output [[buffer(0)]], device T* input [[buffer(1)]],        \
      device const uchar* codes [[buffer(2)]],                              \
      device const uchar* scales [[buffer(3)]],                             \
      device const uchar* biases [[buffer(4)]],                             \
      device const int* expert_of_tile [[buffer(5)]],                       \
      constant int& total_rows [[buffer(6)]], constant int& inner [[buffer(7)]], \
      constant int& output_rows [[buffer(8)]],                              \
      constant int& group_size [[buffer(9)]], constant int& bits [[buffer(10)]], \
      constant int& scale_type [[buffer(11)]], constant int& symmetric [[buffer(12)]], \
      uint3 tgid [[threadgroup_position_in_grid]],                          \
      uint lane [[thread_index_in_simdgroup]]);                             \
  template [[host_name("base_qmoe_swiglu_" name)]] [[kernel]]             \
  void base_qmoe_swiglu_kernel<T>(                                          \
      device T* output [[buffer(0)]], device T* input [[buffer(1)]],        \
      device const uchar* codes [[buffer(2)]],                              \
      device const uchar* scales [[buffer(3)]],                             \
      device const uchar* biases [[buffer(4)]],                             \
      device const int* expert_of_tile [[buffer(5)]],                       \
      constant int& total_rows [[buffer(6)]], constant int& inner [[buffer(7)]], \
      constant int& intermediate [[buffer(8)]],                             \
      constant int& group_size [[buffer(9)]], constant int& bits [[buffer(10)]], \
      constant int& scale_type [[buffer(11)]], constant int& symmetric [[buffer(12)]], \
      uint3 tgid [[threadgroup_position_in_grid]],                          \
      uint warp [[simdgroup_index_in_threadgroup]],                        \
      uint lane [[thread_index_in_simdgroup]]);

INSTANTIATE_BASE_Q(half, "float16")
INSTANTIATE_BASE_Q(bf16, "bfloat16")
INSTANTIATE_BASE_Q(float, "float32")

} // namespace mittens
