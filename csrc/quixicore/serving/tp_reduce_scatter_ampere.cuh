#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm.
namespace dsv4_tp_reduce_scatter {

constexpr int kHidden = 4096;
constexpr int kMaxTokens = 8;
constexpr int kSlotBytes = kMaxTokens * kHidden * sizeof(nv_bfloat16);

template <int NGPU, bool ADDEND>
__global__ void __launch_bounds__(512, 1) reduce_scatter_registered(
    RankData* input_rank_data, RankData* addend_rank_data,
    RankSignals signals, Signal* self_signal, nv_bfloat16* output,
    int local_size, int rank) {
  using P = typename packed_t<nv_bfloat16>::P;
  using A = typename packed_t<nv_bfloat16>::A;
  const RankData inputs = *input_rank_data;
  const RankData addends =
      addend_rank_data == nullptr ? inputs : *addend_rank_data;
  barrier_at_start<NGPU>(signals, self_signal, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < local_size;
       idx += gridDim.x * blockDim.x) {
    const int source_index = rank * local_size + idx;
    if constexpr (!ADDEND) {
      reinterpret_cast<P*>(output)[idx] =
          packed_reduce<P, NGPU, A>(
              (const P**)&inputs.ptrs[0], source_index);
    } else {
      A reduced = {};
#pragma unroll
      for (int peer = 0; peer < NGPU; ++peer) {
        A combined = upcast(
            reinterpret_cast<const P*>(inputs.ptrs[peer])[source_index]);
        packed_assign_add(
            combined,
            upcast(reinterpret_cast<const P*>(
                addends.ptrs[peer])[source_index]));
        packed_assign_add(reduced, upcast(downcast<P>(combined)));
      }
      reinterpret_cast<P*>(output)[idx] = downcast<P>(reduced);
    }
  }
  barrier_at_end<NGPU, true>(signals, self_signal, rank);
}

// One launch establishes channel ownership. Each rank publishes one weighted
// hidden partial, then reduces only its contiguous H/TP destination rows.
// Optional local addend fusion preserves the existing shared+routed BF16
// boundary while avoiding a second publication and collective.
template <int NGPU, bool ADDEND, bool DIRECT>
__global__ void __launch_bounds__(256, 2) reduce_scatter(
    const nv_bfloat16* input, const nv_bfloat16* addend,
    RankData* input_rank_data, RankData* addend_rank_data,
    RankSignals signals, Signal* self_signal, nv_bfloat16* output, int tokens,
    int rank) {
  constexpr int kLocalHidden = kHidden / NGPU;
  __shared__ uint32_t epoch;
  if (threadIdx.x == 0) epoch = self_signal->dsv4_rs_epoch + 1;
  __syncthreads();

  const RankData peers = *input_rank_data;
  const RankData addend_peers =
      addend_rank_data == nullptr ? peers : *addend_rank_data;
  const int slot = epoch & 1;
  if constexpr (!DIRECT) {
    auto* local_publication = reinterpret_cast<nv_bfloat16*>(
        reinterpret_cast<uint8_t*>(const_cast<void*>(peers.ptrs[rank])) +
        slot * kSlotBytes);
    for (int item = threadIdx.x; item < tokens * kHidden;
         item += blockDim.x) {
      if constexpr (ADDEND) {
        local_publication[item] = __float2bfloat16_rn(
            float(input[item]) + float(addend[item]));
      } else {
        local_publication[item] = input[item];
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(&signals.signals[peer]->dsv4_rs_ready[rank], epoch);
    while (ld_flag_acquire(&self_signal->dsv4_rs_ready[peer]) < epoch) {
    }
  }
  __syncthreads();

  for (int item = threadIdx.x; item < tokens * kLocalHidden;
       item += blockDim.x) {
    const int token = item / kLocalHidden;
    const int local_row = item - token * kLocalHidden;
    const int source_index = token * kHidden + rank * kLocalHidden + local_row;
    float value = 0.0f;
#pragma unroll
    for (int peer = 0; peer < NGPU; ++peer) {
      if constexpr (DIRECT) {
        const auto* peer_input =
            reinterpret_cast<const nv_bfloat16*>(peers.ptrs[peer]);
        if constexpr (ADDEND) {
          const auto* peer_addend = reinterpret_cast<const nv_bfloat16*>(
              addend_peers.ptrs[peer]);
          value += float(__float2bfloat16_rn(
              float(peer_input[source_index]) +
              float(peer_addend[source_index])));
        } else {
          value += float(peer_input[source_index]);
        }
      } else {
        const auto* peer_publication = reinterpret_cast<const nv_bfloat16*>(
            reinterpret_cast<const uint8_t*>(peers.ptrs[peer]) +
            slot * kSlotBytes);
        value += float(peer_publication[source_index]);
      }
    }
    output[item] = __float2bfloat16_rn(value);
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_rs_epoch = epoch;
}

template <int NGPU>
inline void launch(const nv_bfloat16* input, const nv_bfloat16* addend,
                   RankData* input_rank_data, RankData* addend_rank_data,
                   RankSignals signals, Signal* self_signal,
                   nv_bfloat16* output, int tokens, int rank, bool direct,
                   cudaStream_t stream) {
  if (direct) {
    constexpr int kPacked = packed_t<nv_bfloat16>::P::size;
    const int local_size = tokens * (kHidden / NGPU) / kPacked;
    const int blocks = min(defaultBlockLimit, (local_size + 511) / 512);
    if (addend == nullptr) {
      reduce_scatter_registered<NGPU, false><<<blocks, 512, 0, stream>>>(
          input_rank_data, nullptr, signals, self_signal, output, local_size,
          rank);
    } else {
      reduce_scatter_registered<NGPU, true><<<blocks, 512, 0, stream>>>(
          input_rank_data, addend_rank_data, signals, self_signal, output,
          local_size, rank);
    }
  } else if (addend == nullptr) {
    reduce_scatter<NGPU, false, false><<<1, 256, 0, stream>>>(
        input, nullptr, input_rank_data, nullptr, signals, self_signal, output,
        tokens, rank);
  } else {
    reduce_scatter<NGPU, true, false><<<1, 256, 0, stream>>>(
        input, addend, input_rank_data, nullptr, signals, self_signal, output,
        tokens, rank);
  }
}

}  // namespace dsv4_tp_reduce_scatter
