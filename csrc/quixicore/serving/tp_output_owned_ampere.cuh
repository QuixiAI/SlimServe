#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm after RankData,
// RankSignals, Signal, and the acquire/release helpers are defined.
namespace dsv4_tp_output_owned {

constexpr int kQ8Block = 32;
constexpr int kQ8Words = 9;
constexpr int kMaxFullColumns = 8192;
constexpr int kMaxTokens = 8;
constexpr int kSlotBytes =
    kMaxTokens * (kMaxFullColumns / kQ8Block) * sizeof(block_q8_1);

// Publish local K shards to every output owner. The payload stays in Q8_1 and
// is parity-buffered; no full-hidden BF16 partial is ever materialized.
template <int NGPU>
__global__ void __launch_bounds__(64, 1) publish_q8_shards(
    const block_q8_1* local_quant, RankData* publication_rank_data,
    RankSignals signals, Signal* self_signal, int tokens, int local_blocks,
    int rank) {
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0) epoch = self_signal->dsv4_owned_epoch + 1;
  __syncthreads();

  const int slot = epoch & 1;
  const int full_blocks = NGPU * local_blocks;
  const RankData peers = *publication_rank_data;
  for (int item = threadIdx.x; item < tokens * local_blocks;
       item += blockDim.x) {
    const int token = item / local_blocks;
    const int local_block = item - token * local_blocks;
    const int destination_block =
        token * full_blocks + rank * local_blocks + local_block;
    const auto* source = reinterpret_cast<const uint32_t*>(local_quant + item);
#pragma unroll
    for (int destination = 0; destination < NGPU; ++destination) {
      auto* gathered = reinterpret_cast<block_q8_1*>(
          reinterpret_cast<uint8_t*>(
              const_cast<void*>(peers.ptrs[destination])) +
          slot * kSlotBytes);
      auto* target = reinterpret_cast<uint32_t*>(gathered + destination_block);
#pragma unroll
      for (int word = 0; word < kQ8Words; ++word) target[word] = source[word];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(&signals.signals[peer]->dsv4_owned_ready[rank], epoch);
    while (ld_flag_acquire(&self_signal->dsv4_owned_ready[peer]) < epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_owned_epoch = epoch;
}

template <int NGPU, int ROWS_PER_CTA, int TOKENS_PER_CTA>
__global__ void __launch_bounds__(64, 2) aligned_q8_output_rows(
    const uint8_t* aligned_weight, RankData* publication_rank_data,
    const Signal* self_signal, nv_bfloat16* output, int tokens, int rows,
    int full_blocks, int rank) {
  const int row0 = ROWS_PER_CTA * blockIdx.x;
  const int token0 = TOKENS_PER_CTA * blockIdx.y;
  if (row0 >= rows) return;

  const RankData peers = *publication_rank_data;
  const int slot = self_signal->dsv4_owned_epoch & 1;
  const auto* activation = reinterpret_cast<const block_q8_1*>(
      reinterpret_cast<const uint8_t*>(peers.ptrs[rank]) +
      slot * kSlotBytes);
  const int64_t total_weight_blocks = int64_t(rows) * full_blocks;
  const auto* scales = reinterpret_cast<const half*>(aligned_weight);
  const auto* codes = reinterpret_cast<const int8_t*>(
      aligned_weight + total_weight_blocks * sizeof(half));

  const int tid = 32 * threadIdx.y + threadIdx.x;
  const int code_index = 2 * (tid & 3);
  float partial[TOKENS_PER_CTA][ROWS_PER_CTA] = {};
  for (int block = tid / 4; block < full_blocks; block += 16) {
#pragma unroll
    for (int row_delta = 0; row_delta < ROWS_PER_CTA; ++row_delta) {
      const int row = min(row0 + row_delta, rows - 1);
      const int64_t weight_block = int64_t(row) * full_blocks + block;
      const auto* weight_codes = reinterpret_cast<const int32_t*>(
          codes + weight_block * kQ8Block);
      const float weight_scale = __half2float(scales[weight_block]);
#pragma unroll
      for (int token_delta = 0; token_delta < TOKENS_PER_CTA; ++token_delta) {
        const int token = min(token0 + token_delta, tokens - 1);
        const block_q8_1& input = activation[token * full_blocks + block];
        const auto* input_codes = reinterpret_cast<const int32_t*>(input.qs);
        int dot = 0;
        dot = __dp4a(weight_codes[code_index], input_codes[code_index], dot);
        dot = __dp4a(weight_codes[code_index + 1],
                     input_codes[code_index + 1], dot);
        partial[token_delta][row_delta] =
            fmaf(weight_scale * __low2float(input.ds), float(dot),
                 partial[token_delta][row_delta]);
      }
    }
  }

  __shared__ float peer_warp[TOKENS_PER_CTA][ROWS_PER_CTA][32];
  if (threadIdx.y == 1) {
#pragma unroll
    for (int token_delta = 0; token_delta < TOKENS_PER_CTA; ++token_delta) {
#pragma unroll
      for (int row_delta = 0; row_delta < ROWS_PER_CTA; ++row_delta) {
        peer_warp[token_delta][row_delta][threadIdx.x] =
            partial[token_delta][row_delta];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y != 0) return;

#pragma unroll
  for (int token_delta = 0; token_delta < TOKENS_PER_CTA; ++token_delta) {
#pragma unroll
    for (int row_delta = 0; row_delta < ROWS_PER_CTA; ++row_delta) {
      float value = partial[token_delta][row_delta] +
                    peer_warp[token_delta][row_delta][threadIdx.x];
#pragma unroll
      for (int mask = 16; mask > 0; mask >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, mask);
      }
      if (threadIdx.x == 0 && token0 + token_delta < tokens &&
          row0 + row_delta < rows) {
        output[int64_t(token0 + token_delta) * rows + row0 + row_delta] =
            __float2bfloat16_rn(value);
      }
    }
  }
}

template <int NGPU, int ROWS_PER_CTA, int TOKENS_PER_CTA>
inline void launch_rows(const uint8_t* weight, RankData* publication_rank_data,
                        const Signal* self_signal, nv_bfloat16* output,
                        int tokens, int rows, int full_blocks, int rank,
                        cudaStream_t stream) {
  const dim3 grid((rows + ROWS_PER_CTA - 1) / ROWS_PER_CTA,
                  (tokens + TOKENS_PER_CTA - 1) / TOKENS_PER_CTA);
  aligned_q8_output_rows<NGPU, ROWS_PER_CTA, TOKENS_PER_CTA>
      <<<grid, dim3(32, 2), 0, stream>>>(weight, publication_rank_data,
                                        self_signal, output, tokens, rows,
                                        full_blocks, rank);
}

template <int NGPU, int ROWS_PER_CTA>
inline void launch_tokens(const uint8_t* weight,
                          RankData* publication_rank_data,
                          const Signal* self_signal, nv_bfloat16* output,
                          int tokens, int rows, int full_blocks, int rank,
                          cudaStream_t stream) {
  if (tokens <= 1) {
    launch_rows<NGPU, ROWS_PER_CTA, 1>(weight, publication_rank_data,
                                      self_signal, output, tokens, rows,
                                      full_blocks, rank, stream);
  } else if (tokens <= 2) {
    launch_rows<NGPU, ROWS_PER_CTA, 2>(weight, publication_rank_data,
                                      self_signal, output, tokens, rows,
                                      full_blocks, rank, stream);
  } else if (tokens <= 4) {
    launch_rows<NGPU, ROWS_PER_CTA, 4>(weight, publication_rank_data,
                                      self_signal, output, tokens, rows,
                                      full_blocks, rank, stream);
  } else {
    launch_rows<NGPU, ROWS_PER_CTA, 8>(weight, publication_rank_data,
                                      self_signal, output, tokens, rows,
                                      full_blocks, rank, stream);
  }
}

template <int NGPU>
inline void launch(const block_q8_1* local_quant, const uint8_t* weight,
                   RankData* publication_rank_data, RankSignals signals,
                   Signal* self_signal, nv_bfloat16* output, int tokens,
                   int rows, int local_blocks, int rank, int rows_per_cta,
                   cudaStream_t stream) {
  publish_q8_shards<NGPU><<<1, 64, 0, stream>>>(
      local_quant, publication_rank_data, signals, self_signal, tokens,
      local_blocks, rank);
  const int full_blocks = NGPU * local_blocks;
  if (rows_per_cta >= 4) {
    launch_tokens<NGPU, 4>(weight, publication_rank_data, self_signal, output,
                           tokens, rows, full_blocks, rank, stream);
  } else if (rows_per_cta >= 2) {
    launch_tokens<NGPU, 2>(weight, publication_rank_data, self_signal, output,
                           tokens, rows, full_blocks, rank, stream);
  } else {
    launch_tokens<NGPU, 1>(weight, publication_rank_data, self_signal, output,
                           tokens, rows, full_blocks, rank, stream);
  }
}

}  // namespace dsv4_tp_output_owned
