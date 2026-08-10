#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm.
namespace dsv4_tp_input_owned {

constexpr int kHidden = 4096;
constexpr int kThreads = 256;

// Compute all replicated attention-input projections from one channel-owned
// activation. Q8_0 and BF16 rows share one output vector so the caller can
// reduce every partial with one collective.
template <int NGPU>
__global__ void __launch_bounds__(kThreads, 2) attention_partials(
    const nv_bfloat16* local_input, const block_q8_1* local_quant,
    const uint8_t* aligned_q8_weight, const nv_bfloat16* bf16_weight0,
    const nv_bfloat16* bf16_weight1, const nv_bfloat16* bf16_weight2,
    float* partial, int q8_rows, int bf16_rows0, int bf16_rows1,
    int bf16_rows2, int rank) {
  constexpr int kLocalHidden = kHidden / NGPU;
  constexpr int kFullQ8Blocks = kHidden / QK8_1;
  constexpr int kLocalQ8Blocks = kLocalHidden / QK8_1;
  constexpr int kWarps = kThreads / 32;
  const int output_row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  float value = 0.0f;

  if (output_row < q8_rows) {
    if (tid < 64) {
      const int64_t total_blocks = int64_t(q8_rows) * kFullQ8Blocks;
      const auto* scales = reinterpret_cast<const half*>(aligned_q8_weight);
      const auto* codes = reinterpret_cast<const int8_t*>(
          aligned_q8_weight + total_blocks * sizeof(half));
      const int code_index = 2 * (tid & 3);
      for (int local_block = tid / 4; local_block < kLocalQ8Blocks;
           local_block += 16) {
        const int weight_block = rank * kLocalQ8Blocks + local_block;
        const int64_t flat_weight_block =
            int64_t(output_row) * kFullQ8Blocks + weight_block;
        const int* weight_codes = reinterpret_cast<const int*>(
            codes + flat_weight_block * QK8_1);
        const int* input_codes =
            reinterpret_cast<const int*>(local_quant[local_block].qs);
        int dot = __dp4a(weight_codes[code_index], input_codes[code_index], 0);
        dot = __dp4a(weight_codes[code_index + 1],
                     input_codes[code_index + 1], dot);
        value += __half2float(scales[flat_weight_block]) *
                 __low2float(local_quant[local_block].ds) * dot;
      }
    }
  } else {
    int row = output_row - q8_rows;
    const nv_bfloat16* weight = bf16_weight0;
    if (row >= bf16_rows0) {
      row -= bf16_rows0;
      weight = bf16_weight1;
    }
    if (row >= bf16_rows1 && output_row >= q8_rows + bf16_rows0 + bf16_rows1) {
      row -= bf16_rows1;
      weight = bf16_weight2;
    }
    const auto* weight_row =
        weight + int64_t(row) * kHidden + rank * kLocalHidden;
    for (int column = tid * 8; column < kLocalHidden;
         column += kThreads * 8) {
      const uint4 input_vector =
          *reinterpret_cast<const uint4*>(local_input + column);
      const uint4 weight_vector =
          *reinterpret_cast<const uint4*>(weight_row + column);
      const auto* input_values =
          reinterpret_cast<const nv_bfloat16*>(&input_vector);
      const auto* weight_values =
          reinterpret_cast<const nv_bfloat16*>(&weight_vector);
#pragma unroll
      for (int element = 0; element < 8; ++element) {
        value = fmaf(float(input_values[element]), float(weight_values[element]),
                     value);
      }
    }
  }

  value = tms::dsv4_mhc::warp_sum(value);
  __shared__ float warp_partials[kWarps];
  if (lane == 0) warp_partials[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarps ? warp_partials[lane] : 0.0f;
    value = tms::dsv4_mhc::warp_sum(value);
    if (lane == 0) partial[output_row] = value;
  }
}

__global__ void finalize_attention(
    const float* reduced, nv_bfloat16* q8_output, float* bf16_output0,
    nv_bfloat16* bf16_output1, float* bf16_output2, int q8_rows,
    int bf16_rows0, int bf16_rows1, int bf16_rows2) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = q8_rows + bf16_rows0 + bf16_rows1 + bf16_rows2;
  if (item >= total) return;
  if (item < q8_rows) {
    q8_output[item] = __float2bfloat16_rn(reduced[item]);
    return;
  }
  int row = item - q8_rows;
  if (row < bf16_rows0) {
    bf16_output0[row] = reduced[item];
    return;
  }
  row -= bf16_rows0;
  if (row < bf16_rows1) {
    bf16_output1[row] = __float2bfloat16_rn(reduced[item]);
    return;
  }
  row -= bf16_rows1;
  bf16_output2[row] = reduced[item];
}

template <int NGPU>
inline void launch_attention(
    const nv_bfloat16* local_input, const block_q8_1* local_quant,
    const uint8_t* aligned_q8_weight, const nv_bfloat16* bf16_weight0,
    const nv_bfloat16* bf16_weight1, const nv_bfloat16* bf16_weight2,
    float* partial, int q8_rows, int bf16_rows0, int bf16_rows1,
    int bf16_rows2, int rank, cudaStream_t stream) {
  const int total = q8_rows + bf16_rows0 + bf16_rows1 + bf16_rows2;
  attention_partials<NGPU><<<total, kThreads, 0, stream>>>(
      local_input, local_quant, aligned_q8_weight, bf16_weight0, bf16_weight1,
      bf16_weight2, partial, q8_rows, bf16_rows0, bf16_rows1, bf16_rows2,
      rank);
}

inline void launch_attention_finalize(
    const float* reduced, nv_bfloat16* q8_output, float* bf16_output0,
    nv_bfloat16* bf16_output1, float* bf16_output2, int q8_rows,
    int bf16_rows0, int bf16_rows1, int bf16_rows2, cudaStream_t stream) {
  const int total = q8_rows + bf16_rows0 + bf16_rows1 + bf16_rows2;
  finalize_attention<<<(total + 255) / 256, 256, 0, stream>>>(
      reduced, q8_output, bf16_output0, bf16_output1, bf16_output2, q8_rows,
      bf16_rows0, bf16_rows1, bf16_rows2);
}

template <int NGPU>
__global__ void __launch_bounds__(256, 1) gather_q8_registered(
    RankData* input_rank_data, RankSignals signals, Signal* self_signal,
    block_q8_1* output, int local_blocks, int rank) {
  barrier_at_start<NGPU>(signals, self_signal, rank);
  const RankData inputs = *input_rank_data;
  for (int block = threadIdx.x; block < NGPU * local_blocks;
       block += blockDim.x) {
    const int source_rank = block / local_blocks;
    const int source_block = block - source_rank * local_blocks;
    const auto* source = reinterpret_cast<const uint32_t*>(
        reinterpret_cast<const block_q8_1*>(inputs.ptrs[source_rank]) +
        source_block);
    auto* destination = reinterpret_cast<uint32_t*>(output + block);
#pragma unroll
    for (int word = 0; word < 9; ++word) destination[word] = source[word];
  }
  barrier_at_end<NGPU, true>(signals, self_signal, rank);
}

template <int NGPU>
inline void launch_gather_q8(RankData* input_rank_data, RankSignals signals,
                             Signal* self_signal, block_q8_1* output,
                             int local_blocks, int rank,
                             cudaStream_t stream) {
  gather_q8_registered<NGPU><<<1, 256, 0, stream>>>(
      input_rank_data, signals, self_signal, output, local_blocks, rank);
}

template <int NGPU>
__global__ void __launch_bounds__(256, 1) gather_bf16_registered(
    RankData* input_rank_data, RankSignals signals, Signal* self_signal,
    nv_bfloat16* output, int rows, int local_hidden, int rank) {
  barrier_at_start<NGPU>(signals, self_signal, rank);
  const RankData inputs = *input_rank_data;
  constexpr int kValuesPerPack = sizeof(uint4) / sizeof(nv_bfloat16);
  const int local_packs = local_hidden / kValuesPerPack;
  const int full_packs = NGPU * local_packs;
  for (int item = blockIdx.x * blockDim.x + threadIdx.x;
       item < rows * full_packs; item += gridDim.x * blockDim.x) {
    const int row = item / full_packs;
    const int hidden_pack = item - row * full_packs;
    const int source_rank = hidden_pack / local_packs;
    const int source_pack = hidden_pack - source_rank * local_packs;
    const auto* source =
        reinterpret_cast<const uint4*>(inputs.ptrs[source_rank]);
    reinterpret_cast<uint4*>(output)[item] =
        source[row * local_packs + source_pack];
  }
  barrier_at_end<NGPU, true>(signals, self_signal, rank);
}

template <int NGPU>
inline void launch_gather_bf16(RankData* input_rank_data, RankSignals signals,
                               Signal* self_signal, nv_bfloat16* output,
                               int rows, int local_hidden, int rank,
                               cudaStream_t stream) {
  gather_bf16_registered<NGPU><<<8, 256, 0, stream>>>(
      input_rank_data, signals, self_signal, output, rows, local_hidden, rank);
}

}  // namespace dsv4_tp_input_owned
