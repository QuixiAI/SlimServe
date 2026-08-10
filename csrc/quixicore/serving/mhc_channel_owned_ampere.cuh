#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm, after the peer
// pointer and synchronization primitives have been defined.
namespace dsv4_mhc_channel_owned {

constexpr int kHiddenSize = 4096;
constexpr int kHC = 4;
constexpr int kMixes = 24;
constexpr int kDeferred = kMixes - kHC;
constexpr int kThreads = 256;
constexpr int kUrgentSplits = 8;
constexpr int kDeferredSplits = 8;
constexpr int kDeferredPartitions = 2;
constexpr int kDeferredPerPartition = kDeferred / kDeferredPartitions;
// Peer-visible messages are parity buffered. Projection and norm scratch are
// rank-local portions of the same graph-registered allocation.
constexpr int kUrgentOffset = 0;
constexpr int kUrgentValues = kHC + 1;
constexpr int kNormOffset = kUrgentOffset + 2 * kUrgentValues;
constexpr int kDeferredOffset = kNormOffset + 2;
constexpr int kUrgentScratchOffset = kDeferredOffset + 2 * kDeferred;
constexpr int kNormScratchOffset =
    kUrgentScratchOffset + kUrgentSplits * kUrgentValues;
constexpr int kDeferredScratchOffset = kNormScratchOffset + kUrgentSplits;
constexpr int kStateOffset =
    kDeferredScratchOffset +
    kDeferredPartitions * kDeferredSplits * kDeferredPerPartition;
constexpr int kPartialValues = kStateOffset + kUrgentValues + 1;

#if 0  // Rejected monolithic mHC diagnostic; see perf/optimization_status.md.
// The monolithic schedule uses the same 266-float peer allocation as the
// split schedule. It publishes all 24 projection partials and the residual
// norm in one exchange, then publishes the BF16-rounded input norm.
constexpr int kMonolithicValues = kMixes + 1;
constexpr int kMonolithicScratchOffset = 0;
constexpr int kMonolithicMessageOffset =
    kMonolithicScratchOffset + kUrgentSplits * kMonolithicValues;
constexpr int kMonolithicNormScratchOffset =
    kMonolithicMessageOffset + 2 * kMonolithicValues;
constexpr int kMonolithicNormMessageOffset =
    kMonolithicNormScratchOffset + kUrgentSplits;
constexpr int kMonolithicStateOffset = kMonolithicNormMessageOffset + 2;
static_assert(kMonolithicStateOffset + kHC + 2 <= kPartialValues);
#endif

template <int Stage>
DINLINE int prepare_exchange(Signal* self_signal, uint32_t* epoch) {
  if (threadIdx.x == 0) {
    *epoch = self_signal->dsv4_channel_epoch[Stage] + 1;
  }
  __syncthreads();
  return *epoch & 1;
}

// Thread 0 owns the complete compact payload. After the block barrier, one lane
// per peer publishes readiness in parallel. Parity buffering prevents payload
// overwrite while a peer consumes the current message.
template <int Stage, int NGPU>
DINLINE void publish_and_wait(const RankSignals& signals, Signal* self_signal,
                              int rank, uint32_t epoch) {
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(
        &signals.signals[peer]->dsv4_channel_ready[Stage][rank], epoch);
    while (ld_flag_acquire(
               &self_signal->dsv4_channel_ready[Stage][peer]) < epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_channel_epoch[Stage] = epoch;
}

template <typename FnT, int NGPU>
__global__ void __launch_bounds__(kThreads, 2) post_pre_urgent(
    RankData* partial_rank_data, RankSignals signals, Signal* self_signal,
    const nv_bfloat16* x, const nv_bfloat16* residual,
    const float* post_mix, const float* comb_mix, const FnT* fn,
    nv_bfloat16* residual_out, float* partial, float* debug,
    const float* scale,
    const float* base, nv_bfloat16* layer_input,
    const nv_bfloat16* norm_weight,
    tms::dsv4_mhc::block_q8_1* quant_input, float rms_eps, float pre_eps,
    float norm_eps, int rank) {
  static_assert(kHiddenSize % NGPU == 0);
  constexpr int kLocalHidden = kHiddenSize / NGPU;
  constexpr int kLocalTotal = kHC * kLocalHidden;
  constexpr int kWarps = kThreads / 32;
  const int split = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ float current_mix[kHC + kHC * kHC];
  __shared__ float warp_partials[kWarps][kHC + 1];
  __shared__ float reduced[kUrgentValues];
  __shared__ uint32_t exchange_epoch;
  if (tid < kHC) current_mix[tid] = post_mix[tid];
  if (tid < kHC * kHC) current_mix[kHC + tid] = comb_mix[tid];
  __syncthreads();

  float projection[kHC] = {};
  float residual_square_sum = 0.0f;
  for (int flat = split * kThreads + tid; flat < kLocalTotal;
       flat += kUrgentSplits * kThreads) {
    const int stream = flat / kLocalHidden;
    const int dim = flat - stream * kLocalHidden;
    const float input_value = float(x[dim]);
    float value = current_mix[stream] * input_value;
#pragma unroll
    for (int input_stream = 0; input_stream < kHC; ++input_stream) {
      value += current_mix[kHC + input_stream * kHC + stream] *
               float(residual[input_stream * kLocalHidden + dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    residual_out[flat] = rounded;
    value = float(rounded);
    residual_square_sum += value * value;
#pragma unroll
    for (int output = 0; output < kHC; ++output) {
      const size_t fn_index =
          size_t(output) * kHC * kHiddenSize +
          size_t(stream) * kHiddenSize + rank * kLocalHidden + dim;
      projection[output] += value * float(fn[fn_index]);
    }
  }

#pragma unroll
  for (int output = 0; output < kHC; ++output) {
    const float sum = tms::dsv4_mhc::warp_sum(projection[output]);
    if (lane == 0) warp_partials[warp][output] = sum;
  }
  const float square_sum = tms::dsv4_mhc::warp_sum(residual_square_sum);
  if (lane == 0) warp_partials[warp][kHC] = square_sum;
  __syncthreads();

  if (warp == 0) {
    for (int output = lane; output <= kHC; output += 32) {
      float value = warp_partials[0][output];
#pragma unroll
      for (int source_warp = 1; source_warp < kWarps; ++source_warp) {
        value += warp_partials[source_warp][output];
      }
      partial[kUrgentScratchOffset + split * kUrgentValues + output] =
          value;
    }
  }
  cooperative_groups::this_grid().sync();

  if (split == 0 && warp == 0) {
    for (int output = lane; output < kUrgentValues; output += 32) {
      float value = partial[kUrgentScratchOffset + output];
#pragma unroll
      for (int source_split = 1; source_split < kUrgentSplits; ++source_split) {
        value += partial[kUrgentScratchOffset +
                         source_split * kUrgentValues + output];
      }
      reduced[output] = value;
    }
  }
  if (split == 0) {
    __syncthreads();
    const int urgent_slot = prepare_exchange<0>(self_signal, &exchange_epoch);
    const int urgent_base = kUrgentOffset + urgent_slot * kUrgentValues;
    if (tid == 0) {
#pragma unroll
      for (int output = 0; output < kUrgentValues; ++output) {
        partial[urgent_base + output] = reduced[output];
      }
    }

    publish_and_wait<0, NGPU>(signals, self_signal, rank, exchange_epoch);
    const RankData peers = *partial_rank_data;
    if (warp == 0) {
      for (int output = lane; output < kUrgentValues; output += 32) {
        float value = reinterpret_cast<const float*>(peers.ptrs[0])[
            urgent_base + output];
#pragma unroll
        for (int peer = 1; peer < NGPU; ++peer) {
          value += reinterpret_cast<const float*>(peers.ptrs[peer])[
              urgent_base + output];
        }
        reduced[output] = value;
      }
    }
    __syncthreads();
    const float residual_inverse_rms =
        rsqrtf(reduced[kHC] / float(kHC * kHiddenSize) + rms_eps);
    if (tid < kHC) {
      partial[kStateOffset + tid] = tms::dsv4_mhc::sigmoid(
          reduced[tid] * residual_inverse_rms * scale[0] + base[tid]) +
          pre_eps;
    }
    if (tid == 0) partial[kStateOffset + kHC] = residual_inverse_rms;
  }
  cooperative_groups::this_grid().sync();

  float input_square_sum = 0.0f;
  for (int dim = split * kThreads + tid; dim < kLocalHidden;
       dim += kUrgentSplits * kThreads) {
    float value = 0.0f;
#pragma unroll
    for (int stream = 0; stream < kHC; ++stream) {
      value += partial[kStateOffset + stream] *
               float(residual_out[stream * kLocalHidden + dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    layer_input[dim] = rounded;
    const float as_float = float(rounded);
    input_square_sum += as_float * as_float;
  }
  input_square_sum = tms::dsv4_mhc::warp_sum(input_square_sum);
  if (lane == 0) warp_partials[warp][0] = input_square_sum;
  __syncthreads();
  if (tid == 0) {
    float value = warp_partials[0][0];
#pragma unroll
    for (int source_warp = 1; source_warp < kWarps; ++source_warp) {
      value += warp_partials[source_warp][0];
    }
    partial[kNormScratchOffset + split] = value;
  }
  cooperative_groups::this_grid().sync();

  if (split == 0) {
    if (tid == 0) {
      float value = partial[kNormScratchOffset];
#pragma unroll
      for (int source_split = 1; source_split < kUrgentSplits; ++source_split) {
        value += partial[kNormScratchOffset + source_split];
      }
      reduced[0] = value;
    }
    __syncthreads();
    const int norm_slot = prepare_exchange<1>(self_signal, &exchange_epoch);
    if (tid == 0) partial[kNormOffset + norm_slot] = reduced[0];

    // BF16 rounding occurs before RMSNorm, so preserve it with a second scalar
    // publication rather than deriving the input norm approximately.
    publish_and_wait<1, NGPU>(signals, self_signal, rank, exchange_epoch);
    const RankData peers = *partial_rank_data;
    if (tid == 0) {
      float value =
          reinterpret_cast<const float*>(peers.ptrs[0])[kNormOffset + norm_slot];
#pragma unroll
      for (int peer = 1; peer < NGPU; ++peer) {
        value += reinterpret_cast<const float*>(peers.ptrs[peer])[
            kNormOffset + norm_slot];
      }
      partial[kStateOffset + kUrgentValues] =
          rsqrtf(value / float(kHiddenSize) + norm_eps);
    }
  }
  cooperative_groups::this_grid().sync();

  const int hidden_offset = rank * kLocalHidden;
  for (int dim = split * kThreads + tid; dim < kLocalHidden;
       dim += kUrgentSplits * kThreads) {
    layer_input[dim] = __float2bfloat16_rn(
        float(layer_input[dim]) * partial[kStateOffset + kUrgentValues] *
        float(norm_weight[hidden_offset + dim]));
  }

  constexpr int kLocalBlocks = kLocalHidden / 32;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int global_warp = split * kWarpsPerBlock + warp;
  for (int block = global_warp; block < kLocalBlocks;
       block += kUrgentSplits * kWarpsPerBlock) {
    const float value = float(layer_input[block * 32 + lane]);
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float qscale = amax / 127.0f;
    quant_input[block].qs[lane] =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / qscale));
    if (lane == 0) quant_input[block].ds = __floats2half2_rn(qscale, sum);
  }
}

#if 0  // Rejected monolithic mHC diagnostic; not part of the serving path.
template <typename FnT, int NGPU>
__global__ void __launch_bounds__(kThreads, 1) post_pre_monolithic(
    RankData* partial_rank_data, RankSignals signals, Signal* self_signal,
    const nv_bfloat16* x, const nv_bfloat16* residual,
    const float* post_mix, const float* comb_mix, const FnT* fn,
    nv_bfloat16* residual_out, float* partial, float* debug,
    const float* scale, const float* base, float* next_post, float* next_comb,
    nv_bfloat16* layer_input, const nv_bfloat16* norm_weight,
    tms::dsv4_mhc::block_q8_1* quant_input, float rms_eps, float pre_eps,
    float sinkhorn_eps, float post_multiplier, int sinkhorn_repeat,
    float norm_eps, int rank) {
  static_assert(kHiddenSize % NGPU == 0);
  constexpr int kLocalHidden = kHiddenSize / NGPU;
  constexpr int kLocalTotal = kHC * kLocalHidden;
  constexpr int kWarps = kThreads / 32;
  const int split = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  __shared__ float current_mix[kHC + kHC * kHC];
  __shared__ float warp_partials[kWarps][kMonolithicValues];
  __shared__ float reduced[kMonolithicValues];
  __shared__ float prepared[kDeferred];
  __shared__ uint32_t exchange_epoch;
  if (tid < kHC) current_mix[tid] = post_mix[tid];
  if (tid < kHC * kHC) current_mix[kHC + tid] = comb_mix[tid];
  __syncthreads();

  float projection[kMixes] = {};
  float residual_square_sum = 0.0f;
  for (int flat = split * kThreads + tid; flat < kLocalTotal;
       flat += kUrgentSplits * kThreads) {
    const int stream = flat / kLocalHidden;
    const int dim = flat - stream * kLocalHidden;
    float value = current_mix[stream] * float(x[dim]);
#pragma unroll
    for (int input_stream = 0; input_stream < kHC; ++input_stream) {
      value += current_mix[kHC + input_stream * kHC + stream] *
               float(residual[input_stream * kLocalHidden + dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    residual_out[flat] = rounded;
    value = float(rounded);
    residual_square_sum += value * value;
#pragma unroll
    for (int output = 0; output < kMixes; ++output) {
      const size_t fn_index =
          size_t(output) * kHC * kHiddenSize +
          size_t(stream) * kHiddenSize + rank * kLocalHidden + dim;
      projection[output] += value * float(fn[fn_index]);
    }
  }

#pragma unroll
  for (int output = 0; output < kMixes; ++output) {
    const float sum = tms::dsv4_mhc::warp_sum(projection[output]);
    if (lane == 0) warp_partials[warp][output] = sum;
  }
  const float square_sum = tms::dsv4_mhc::warp_sum(residual_square_sum);
  if (lane == 0) warp_partials[warp][kMixes] = square_sum;
  __syncthreads();

  if (warp == 0) {
    for (int output = lane; output < kMonolithicValues; output += 32) {
      float value = warp_partials[0][output];
#pragma unroll
      for (int source_warp = 1; source_warp < kWarps; ++source_warp) {
        value += warp_partials[source_warp][output];
      }
      partial[kMonolithicScratchOffset + split * kMonolithicValues + output] =
          value;
    }
  }
  cooperative_groups::this_grid().sync();

  if (split == 0) {
    if (warp == 0) {
      for (int output = lane; output < kMonolithicValues; output += 32) {
        float value = partial[kMonolithicScratchOffset + output];
#pragma unroll
        for (int source_split = 1; source_split < kUrgentSplits;
             ++source_split) {
          value += partial[kMonolithicScratchOffset +
                           source_split * kMonolithicValues + output];
        }
        reduced[output] = value;
      }
    }
    __syncthreads();
    const int projection_slot =
        prepare_exchange<0>(self_signal, &exchange_epoch);
    const int projection_base =
        kMonolithicMessageOffset + projection_slot * kMonolithicValues;
    if (tid < kMonolithicValues) {
      partial[projection_base + tid] = reduced[tid];
    }
    publish_and_wait<0, NGPU>(signals, self_signal, rank, exchange_epoch);

    const RankData peers = *partial_rank_data;
    if (warp == 0) {
      for (int output = lane; output < kMonolithicValues; output += 32) {
        float value = reinterpret_cast<const float*>(peers.ptrs[0])[
            projection_base + output];
#pragma unroll
        for (int peer = 1; peer < NGPU; ++peer) {
          value += reinterpret_cast<const float*>(peers.ptrs[peer])[
              projection_base + output];
        }
        reduced[output] = value;
      }
    }
    __syncthreads();

    const float residual_inverse_rms =
        rsqrtf(reduced[kMixes] / float(kHC * kHiddenSize) + rms_eps);
    if (tid < kHC) {
      partial[kMonolithicStateOffset + tid] = tms::dsv4_mhc::sigmoid(
          reduced[tid] * residual_inverse_rms * scale[0] + base[tid]) +
          pre_eps;
      prepared[tid] = tms::dsv4_mhc::sigmoid(
                          reduced[kHC + tid] * residual_inverse_rms * scale[1] +
                          base[kHC + tid]) *
                      post_multiplier;
    }
    if (tid == 0) {
      partial[kMonolithicStateOffset + kHC] = residual_inverse_rms;
    }

    if (warp == 0) {
      float matrix = 0.0f;
      if (lane < kHC * kHC) {
        matrix = reduced[2 * kHC + lane] * residual_inverse_rms * scale[2] +
                 base[2 * kHC + lane];
      }
      float row_max = matrix;
      row_max = fmaxf(row_max,
                      __shfl_xor_sync(0xffffffffu, row_max, 1, kHC));
      row_max = fmaxf(row_max,
                      __shfl_xor_sync(0xffffffffu, row_max, 2, kHC));
      if (lane < kHC * kHC) matrix = expf(matrix - row_max);
      float row_sum = matrix;
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, kHC);
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, kHC);
      if (lane < kHC * kHC) matrix = matrix / row_sum + sinkhorn_eps;
      for (int iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
        if (iteration > 0) {
          row_sum = matrix;
          row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, kHC);
          row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, kHC);
          if (lane < kHC * kHC) matrix /= row_sum + sinkhorn_eps;
        }
        float column_sum = matrix;
        column_sum += __shfl_xor_sync(0xffffffffu, column_sum, kHC);
        column_sum += __shfl_xor_sync(0xffffffffu, column_sum, 2 * kHC);
        if (lane < kHC * kHC) matrix /= column_sum + sinkhorn_eps;
      }
      if (lane < kHC * kHC) prepared[kHC + lane] = matrix;
    }
    __syncthreads();
    if (tid < kHC) next_post[tid] = prepared[tid];
    if (tid < kHC * kHC) next_comb[tid] = prepared[kHC + tid];
  }
  cooperative_groups::this_grid().sync();

  float input_square_sum = 0.0f;
  for (int dim = split * kThreads + tid; dim < kLocalHidden;
       dim += kUrgentSplits * kThreads) {
    float value = 0.0f;
#pragma unroll
    for (int stream = 0; stream < kHC; ++stream) {
      value += partial[kMonolithicStateOffset + stream] *
               float(residual_out[stream * kLocalHidden + dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    layer_input[dim] = rounded;
    const float as_float = float(rounded);
    input_square_sum += as_float * as_float;
  }
  input_square_sum = tms::dsv4_mhc::warp_sum(input_square_sum);
  if (lane == 0) warp_partials[warp][0] = input_square_sum;
  __syncthreads();
  if (tid == 0) {
    float value = warp_partials[0][0];
#pragma unroll
    for (int source_warp = 1; source_warp < kWarps; ++source_warp) {
      value += warp_partials[source_warp][0];
    }
    partial[kMonolithicNormScratchOffset + split] = value;
  }
  cooperative_groups::this_grid().sync();

  if (split == 0) {
    if (tid == 0) {
      float value = partial[kMonolithicNormScratchOffset];
#pragma unroll
      for (int source_split = 1; source_split < kUrgentSplits;
           ++source_split) {
        value += partial[kMonolithicNormScratchOffset + source_split];
      }
      reduced[0] = value;
    }
    __syncthreads();
    const int norm_slot = prepare_exchange<1>(self_signal, &exchange_epoch);
    if (tid == 0) {
      partial[kMonolithicNormMessageOffset + norm_slot] = reduced[0];
    }
    publish_and_wait<1, NGPU>(signals, self_signal, rank, exchange_epoch);

    const RankData peers = *partial_rank_data;
    if (tid == 0) {
      float value = reinterpret_cast<const float*>(peers.ptrs[0])[
          kMonolithicNormMessageOffset + norm_slot];
#pragma unroll
      for (int peer = 1; peer < NGPU; ++peer) {
        value += reinterpret_cast<const float*>(peers.ptrs[peer])[
            kMonolithicNormMessageOffset + norm_slot];
      }
      partial[kMonolithicStateOffset + kHC + 1] =
          rsqrtf(value / float(kHiddenSize) + norm_eps);
    }
  }
  cooperative_groups::this_grid().sync();

  const int hidden_offset = rank * kLocalHidden;
  for (int dim = split * kThreads + tid; dim < kLocalHidden;
       dim += kUrgentSplits * kThreads) {
    layer_input[dim] = __float2bfloat16_rn(
        float(layer_input[dim]) *
        partial[kMonolithicStateOffset + kHC + 1] *
        float(norm_weight[hidden_offset + dim]));
  }

  constexpr int kLocalBlocks = kLocalHidden / 32;
  constexpr int kWarpsPerBlock = kThreads / 32;
  const int global_warp = split * kWarpsPerBlock + warp;
  for (int block = global_warp; block < kLocalBlocks;
       block += kUrgentSplits * kWarpsPerBlock) {
    const float value = float(layer_input[block * 32 + lane]);
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float qscale = amax / 127.0f;
    quant_input[block].qs[lane] =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / qscale));
    if (lane == 0) quant_input[block].ds = __floats2half2_rn(qscale, sum);
  }
  (void)debug;
}
#endif

template <int NGPU, bool Complete>
__global__ void channel_q2_handshake(RankSignals signals, Signal* self_signal,
                                     int rank) {
  const uint32_t epoch = self_signal->dsv4_channel_q2_epoch + 1;
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    FlagType* remote = Complete
        ? &signals.signals[peer]->dsv4_channel_q2_done[rank]
        : &signals.signals[peer]->dsv4_channel_q2_ready[rank];
    FlagType* local = Complete
        ? &self_signal->dsv4_channel_q2_done[peer]
        : &self_signal->dsv4_channel_q2_ready[peer];
    st_flag_release(remote, epoch);
    while (ld_flag_acquire(local) < epoch) {
    }
  }
  __syncthreads();
  if constexpr (Complete) {
    if (threadIdx.x == 0) self_signal->dsv4_channel_q2_epoch = epoch;
  }
}

#if 0  // Rejected direct-publication diagnostic.
template <int NGPU>
__global__ void channel_q2_ready_advance(RankSignals signals,
                                         Signal* self_signal, int rank) {
  const uint32_t epoch = self_signal->dsv4_channel_q2_epoch + 1;
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(&signals.signals[peer]->dsv4_channel_q2_ready[rank],
                    epoch);
    while (ld_flag_acquire(&self_signal->dsv4_channel_q2_ready[peer]) <
           epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_channel_q2_epoch = epoch;
}
#endif

// Publish the producer boundary once, gather each tiny route shard exactly
// once, and acknowledge consumption before the peer can reuse its buffer.
// Q2 output CTAs then consume rank-local data instead of issuing the same
// remote reads independently for every eight output rows.
template <int NGPU, int TOP_K, int LOCAL_K, bool GATHER_ADDEND = true,
          bool PUSH_ADDEND = false, bool PUSH_QUANT = false,
          bool QUANT_PREPUBLISHED = false, bool RESERVE_ADDEND = false>
DINLINE void channel_q2_gather_body(
    const block_q8_1* local_quant, const nv_bfloat16* local_addend,
    RankData* publication_rank_data, block_q8_1* gathered_quant,
    float* gathered_addend, RankSignals signals, Signal* self_signal,
    int local_rows, int rank) {
  constexpr int kLocalQ8Blocks = LOCAL_K / QK8_1;
  constexpr int kFullQ8Blocks = NGPU * kLocalQ8Blocks;
  constexpr int kLocalQuantBytes =
      TOP_K * kLocalQ8Blocks * sizeof(block_q8_1);
  constexpr int kFullQuantBytes =
      TOP_K * kFullQ8Blocks * sizeof(block_q8_1);
  constexpr int kQuantStorageBytes =
      (PUSH_QUANT || QUANT_PREPUBLISHED) ? kFullQuantBytes
                                        : kLocalQuantBytes;
  constexpr int kAddendBytes =
      (GATHER_ADDEND || PUSH_ADDEND || RESERVE_ADDEND)
      ? kHiddenSize * sizeof(nv_bfloat16)
      : 0;
  constexpr int kSlotBytes =
      (kQuantStorageBytes + kAddendBytes + 255) & ~255;
  const uint32_t epoch = self_signal->dsv4_channel_q2_epoch + 1;
  const int publication_slot = epoch & 1;
  const RankData publication_peers = *publication_rank_data;
  auto* local_slot = reinterpret_cast<uint8_t*>(
      const_cast<void*>(publication_peers.ptrs[rank])) +
      publication_slot * kSlotBytes;
  if constexpr (PUSH_QUANT) {
    constexpr int kLocalItems = TOP_K * kLocalQ8Blocks;
    for (int item = threadIdx.x; item < NGPU * kLocalItems;
         item += blockDim.x) {
      const int destination = item / kLocalItems;
      const int source_item = item - destination * kLocalItems;
      const int route = source_item / kLocalQ8Blocks;
      const int local_block = source_item - route * kLocalQ8Blocks;
      auto* destination_quant = reinterpret_cast<block_q8_1*>(
          reinterpret_cast<uint8_t*>(
              const_cast<void*>(publication_peers.ptrs[destination])) +
          publication_slot * kSlotBytes);
      destination_quant[route * kFullQ8Blocks +
                        rank * kLocalQ8Blocks + local_block] =
          local_quant[source_item];
    }
  } else if constexpr (!QUANT_PREPUBLISHED) {
    for (int offset = threadIdx.x * 16; offset < kLocalQuantBytes;
         offset += blockDim.x * 16) {
      *reinterpret_cast<uint4*>(local_slot + offset) =
          *reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(local_quant) + offset);
    }
  }
  if constexpr (GATHER_ADDEND) {
    for (int offset = threadIdx.x * 16; offset < kAddendBytes;
         offset += blockDim.x * 16) {
      *reinterpret_cast<uint4*>(local_slot + kQuantStorageBytes + offset) =
          *reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(local_addend) + offset);
    }
  } else if constexpr (PUSH_ADDEND) {
    constexpr int kLocalRows = kHiddenSize / NGPU;
    constexpr int kLocalRowBytes = kLocalRows * sizeof(nv_bfloat16);
    for (int offset = threadIdx.x * 16; offset < kAddendBytes;
         offset += blockDim.x * 16) {
      const int destination = offset / kLocalRowBytes;
      const int destination_offset = offset - destination * kLocalRowBytes;
      auto* destination_slot = reinterpret_cast<uint8_t*>(
          const_cast<void*>(publication_peers.ptrs[destination])) +
          publication_slot * kSlotBytes + kQuantStorageBytes;
      *reinterpret_cast<uint4*>(destination_slot + rank * kLocalRowBytes +
                                destination_offset) =
          *reinterpret_cast<const uint4*>(
              reinterpret_cast<const uint8_t*>(local_addend) + offset);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(&signals.signals[peer]->dsv4_channel_q2_ready[rank], epoch);
    while (ld_flag_acquire(&self_signal->dsv4_channel_q2_ready[peer]) < epoch) {
    }
  }
  __syncthreads();

  if constexpr (!PUSH_QUANT && !QUANT_PREPUBLISHED) {
    for (int item = threadIdx.x; item < TOP_K * kFullQ8Blocks;
         item += blockDim.x) {
      const int route = item / kFullQ8Blocks;
      const int block = item - route * kFullQ8Blocks;
      const int source_rank = block / kLocalQ8Blocks;
      const int local_block = block - source_rank * kLocalQ8Blocks;
      const auto* source = reinterpret_cast<const block_q8_1*>(
          reinterpret_cast<const uint8_t*>(publication_peers.ptrs[source_rank]) +
          publication_slot * kSlotBytes);
      gathered_quant[item] = source[route * kLocalQ8Blocks + local_block];
    }
  }

  if constexpr (PUSH_ADDEND) {
    if (gathered_addend != nullptr) {
    for (int pair = threadIdx.x; pair < local_rows / 2;
         pair += blockDim.x) {
      float2 value = {0.0f, 0.0f};
#pragma unroll
      for (int peer = 0; peer < NGPU; ++peer) {
        const auto* addend = reinterpret_cast<const nv_bfloat162*>(
            reinterpret_cast<const uint8_t*>(publication_peers.ptrs[rank]) +
            publication_slot * kSlotBytes + kQuantStorageBytes +
            peer * local_rows * sizeof(nv_bfloat16));
        const float2 peer_value = __bfloat1622float2(addend[pair]);
        value.x += peer_value.x;
        value.y += peer_value.y;
      }
      reinterpret_cast<float2*>(gathered_addend)[pair] = value;
    }
    }
  } else if constexpr (GATHER_ADDEND) {
    for (int row = threadIdx.x; row < local_rows; row += blockDim.x) {
      const int global_row = rank * local_rows + row;
      float value = 0.0f;
#pragma unroll
      for (int peer = 0; peer < NGPU; ++peer) {
        const auto* addend = reinterpret_cast<const nv_bfloat16*>(
            reinterpret_cast<const uint8_t*>(publication_peers.ptrs[peer]) +
            publication_slot * kSlotBytes + kQuantStorageBytes);
        value += float(addend[global_row]);
      }
      gathered_addend[row] = value;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_channel_q2_epoch = epoch;
}

template <int NGPU, int TOP_K, int LOCAL_K, bool GATHER_ADDEND = true,
          bool PUSH_ADDEND = false, bool PUSH_QUANT = false,
          bool QUANT_PREPUBLISHED = false, bool RESERVE_ADDEND = false>
__global__ void channel_q2_gather(
    const block_q8_1* local_quant, const nv_bfloat16* local_addend,
    RankData* publication_rank_data, block_q8_1* gathered_quant,
    float* gathered_addend, RankSignals signals, Signal* self_signal,
    int local_rows, int rank) {
  channel_q2_gather_body<NGPU, TOP_K, LOCAL_K, GATHER_ADDEND, PUSH_ADDEND,
                         PUSH_QUANT, QUANT_PREPUBLISHED, RESERVE_ADDEND>(
      local_quant, local_addend, publication_rank_data, gathered_quant,
      gathered_addend, signals, self_signal, local_rows, rank);
}

#if 0  // Rejected fused shared-expert publication diagnostic.
DINLINE int32_t channel_load_i8x4_unaligned(const int8_t* ptr) {
  const auto* bytes = reinterpret_cast<const uint8_t*>(ptr);
  return int32_t(uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
                 (uint32_t(bytes[2]) << 16) |
                 (uint32_t(bytes[3]) << 24));
}

DINLINE int32_t channel_dot_q8(const block_q8_0& weight,
                              const block_q8_1& input) {
  int32_t dot = 0;
#pragma unroll
  for (int element = 0; element < QK8_0; element += 4) {
    dot = __dp4a(channel_load_i8x4_unaligned(weight.qs + element),
                 *reinterpret_cast<const int32_t*>(input.qs + element), dot);
  }
  return dot;
}

template <int NGPU, int TOP_K, int LOCAL_K>
__device__ __noinline__ void channel_shared_publish_leader(
    const nv_bfloat16* swiglu_scratch,
    RankData* publication_rank_data, RankSignals signals,
    Signal* self_signal, int rank) {
  constexpr int kBlockWarps = 8;
  constexpr int kLocalBlocks = LOCAL_K / QK8_1;
  constexpr int kFullBlocks = NGPU * kLocalBlocks;
  constexpr int kRoutedQuantBytes =
      TOP_K * kFullBlocks * sizeof(block_q8_1);
  constexpr int kPayloadBytes = kHiddenSize * sizeof(nv_bfloat16);
  constexpr int kSlotBytes =
      (kRoutedQuantBytes + kPayloadBytes + 255) & ~255;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const uint32_t epoch = self_signal->dsv4_channel_q2_epoch + 1;
  const int publication_slot = epoch & 1;
  const RankData peers = *publication_rank_data;

  for (int quant_block = warp; quant_block < kLocalBlocks;
       quant_block += kBlockWarps) {
    const int dim = quant_block * QK8_1 + lane;
    const float value = float(swiglu_scratch[dim]);
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float qscale = amax / 127.0f;
    const int8_t code =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / qscale));
    for (int destination = 0; destination < NGPU; ++destination) {
      auto* destination_shared = reinterpret_cast<block_q8_1*>(
          reinterpret_cast<uint8_t*>(
              const_cast<void*>(peers.ptrs[destination])) +
          publication_slot * kSlotBytes + kRoutedQuantBytes);
      block_q8_1& quant =
          destination_shared[rank * kLocalBlocks + quant_block];
      quant.qs[lane] = code;
      if (lane == 0) quant.ds = __floats2half2_rn(qscale, sum);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(&signals.signals[peer]->dsv4_channel_q2_ready[rank],
                    epoch);
    while (ld_flag_acquire(&self_signal->dsv4_channel_q2_ready[peer]) <
           epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_channel_q2_epoch = epoch;
}

// Shared gate/up is intermediate-owned. Compute SwiGLU, pack the local Q8_1
// shard directly into every output owner's routed-MoE publication slot, and
// establish the one readiness boundary consumed by the output-stationary down
// kernel. The preceding routed W1 publication is ordered on the same stream,
// so this replaces the standalone shared-copy launch and its extra fence.
template <int NGPU, int TOP_K, int LOCAL_K, int WARPS_PER_PAIR = 4>
__global__ void __launch_bounds__(256, 4) channel_shared_gate_up_publish(
    const block_q8_0* __restrict__ weights,
    const block_q8_1* __restrict__ input,
    nv_bfloat16* __restrict__ swiglu_scratch,
    RankData* publication_rank_data, RankSignals signals,
    Signal* self_signal, int blocks_per_row, float swiglu_limit, int rank) {
  constexpr int kBlockWarps = 8;
  constexpr int kPairsPerBlock = kBlockWarps / WARPS_PER_PAIR;
  static_assert(NGPU * LOCAL_K == 2048);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = warp / WARPS_PER_PAIR;
  const int warp_in_pair = warp % WARPS_PER_PAIR;
  __shared__ float gate_partials[kBlockWarps];
  __shared__ float up_partials[kBlockWarps];

  for (int row_base = blockIdx.x * kPairsPerBlock; row_base < LOCAL_K;
       row_base += gridDim.x * kPairsPerBlock) {
    const int row = row_base + pair;
    float gate_accum = 0.0f;
    float up_accum = 0.0f;
    if (row < LOCAL_K) {
      const block_q8_0* gate = weights + int64_t(row) * blocks_per_row;
      const block_q8_0* up =
          weights + int64_t(LOCAL_K + row) * blocks_per_row;
      for (int block = warp_in_pair * 32 + lane; block < blocks_per_row;
           block += 32 * WARPS_PER_PAIR) {
        const block_q8_1 input_block = input[block];
        const float input_scale = __low2float(input_block.ds);
        const block_q8_0 gate_block = gate[block];
        const block_q8_0 up_block = up[block];
        gate_accum = fmaf(float(channel_dot_q8(gate_block, input_block)),
                          __half2float(gate_block.d) * input_scale,
                          gate_accum);
        up_accum = fmaf(float(channel_dot_q8(up_block, input_block)),
                        __half2float(up_block.d) * input_scale, up_accum);
      }
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      gate_accum += __shfl_xor_sync(0xffffffffu, gate_accum, offset);
      up_accum += __shfl_xor_sync(0xffffffffu, up_accum, offset);
    }
    if (lane == 0) {
      gate_partials[warp] = gate_accum;
      up_partials[warp] = up_accum;
    }
    __syncthreads();

    if (row < LOCAL_K && lane == 0 && warp_in_pair == 0) {
#pragma unroll
      for (int source_warp = 1; source_warp < WARPS_PER_PAIR;
           ++source_warp) {
        gate_accum += gate_partials[warp + source_warp];
        up_accum += up_partials[warp + source_warp];
      }
      if (swiglu_limit > 0.0f) {
        gate_accum = fminf(gate_accum, swiglu_limit);
        up_accum =
            fminf(fmaxf(up_accum, -swiglu_limit), swiglu_limit);
      }
      const float silu = gate_accum / (1.0f + expf(-gate_accum));
      swiglu_scratch[row] = __float2bfloat16_rn(silu * up_accum);
    }
    __syncthreads();
  }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x != 0) return;
  channel_shared_publish_leader<NGPU, TOP_K, LOCAL_K>(
      swiglu_scratch, publication_rank_data, signals, self_signal, rank);
}

template <int NGPU, int TOP_K, int LOCAL_K>
inline cudaError_t launch_channel_shared_gate_up_publish(
    const void* weights, const void* input, nv_bfloat16* swiglu_scratch,
    RankData* publication_rank_data, RankSignals signals,
    Signal* self_signal, int blocks_per_row, float swiglu_limit, int rank,
    cudaStream_t stream) {
  auto kernel = channel_shared_gate_up_publish<NGPU, TOP_K, LOCAL_K, 4>;
  int blocks_per_sm = 0;
  int sm_count = 0;
  int device = 0;
  cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, 256, 0);
  if (error != cudaSuccess) return error;
  if ((error = cudaGetDevice(&device)) != cudaSuccess) return error;
  if ((error = cudaDeviceGetAttribute(
           &sm_count, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess) {
    return error;
  }
  constexpr int kPairsPerBlock = 2;
  const int row_blocks = (LOCAL_K + kPairsPerBlock - 1) / kPairsPerBlock;
  const int grid_blocks =
      row_blocks < blocks_per_sm * sm_count ? row_blocks
                                            : blocks_per_sm * sm_count;
  const auto* weight_ptr = static_cast<const block_q8_0*>(weights);
  const auto* input_ptr = static_cast<const block_q8_1*>(input);
  void* args[] = {
      &weight_ptr, &input_ptr, &swiglu_scratch, &publication_rank_data,
      &signals, &self_signal, &blocks_per_row, &swiglu_limit, &rank,
  };
  return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                     dim3(grid_blocks), dim3(256), args, 0,
                                     stream);
}

#endif

// Shared W1 is intermediate-sharded. Publish its compact Q8_1 activation into
// every output owner's parity slot so each rank can apply an output-stationary
// shared W2 without materializing or reducing a full-hidden partial.
template <int NGPU, int TOP_K, int LOCAL_K>
__global__ void channel_shared_q8_publish(
    const block_q8_1* local_shared_quant,
    RankData* publication_rank_data, RankSignals signals,
    Signal* self_signal, int rank) {
  constexpr int kLocalBlocks = LOCAL_K / QK8_1;
  constexpr int kFullBlocks = NGPU * kLocalBlocks;
  constexpr int kRoutedQuantBytes =
      TOP_K * kFullBlocks * sizeof(block_q8_1);
  constexpr int kPayloadBytes = kHiddenSize * sizeof(nv_bfloat16);
  constexpr int kSlotBytes =
      (kRoutedQuantBytes + kPayloadBytes + 255) & ~255;
  static_assert(kFullBlocks == 2048 / QK8_1);
  static_assert(kFullBlocks * sizeof(block_q8_1) <= kPayloadBytes);

  const uint32_t epoch = self_signal->dsv4_channel_q2_epoch + 1;
  const int publication_slot = epoch & 1;
  const RankData peers = *publication_rank_data;
  for (int item = threadIdx.x; item < NGPU * kLocalBlocks;
       item += blockDim.x) {
    const int destination = item / kLocalBlocks;
    const int local_block = item - destination * kLocalBlocks;
    auto* destination_shared = reinterpret_cast<block_q8_1*>(
        reinterpret_cast<uint8_t*>(
            const_cast<void*>(peers.ptrs[destination])) +
        publication_slot * kSlotBytes + kRoutedQuantBytes);
    destination_shared[rank * kLocalBlocks + local_block] =
        local_shared_quant[local_block];
  }
  __syncthreads();
  if (threadIdx.x == 0) __threadfence_system();
  __syncthreads();
  if (threadIdx.x < NGPU) {
    const int peer = threadIdx.x;
    st_flag_release(&signals.signals[peer]->dsv4_channel_q2_ready[rank], epoch);
    while (ld_flag_acquire(&self_signal->dsv4_channel_q2_ready[peer]) < epoch) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_signal->dsv4_channel_q2_epoch = epoch;
}

template <int NGPU, int TOP_K, int LOCAL_K>
struct ChannelQ2DownShared {
  static constexpr int kSubblocks = NGPU * LOCAL_K / 16;
  uint32_t activation[TOP_K][4][kSubblocks];
  int activation_sum[TOP_K][kSubblocks];
  float activation_scale[TOP_K][kSubblocks / 2];
};

// Output-stationary Q2_K down projection. Each rank owns contiguous hidden
// output rows and the corresponding full-K packed weights. Q8_1 route
// activations remain intermediate-sharded; this kernel reads those shards
// directly over NVLink and never materializes a full-hidden TP partial.
template <int NGPU, int TOP_K, int LOCAL_K, bool PENDING = false,
          bool ASSEMBLED_QUANT = false, bool FOLD_ADDEND = false,
          bool FOLD_SHARED_Q8 = false>
DINLINE void channel_q2_down_body(
    const uint8_t* __restrict__ weights, RankData* quant_mid_rank_data,
    const int* __restrict__ topk_ids, nv_bfloat16* __restrict__ output,
    int64_t expert_stride, int local_rows, int experts,
    const dsv4_q2_mhc_ar::PendingQ2Header* pending_header = nullptr,
    RankData* addend_rank_data = nullptr, int addend_offset_bytes = 0,
    int rank = 0, const block_q8_1* gathered_quant = nullptr,
    const float* gathered_addend = nullptr,
    Signal* publication_signal = nullptr,
    ChannelQ2DownShared<NGPU, TOP_K, LOCAL_K>* shared = nullptr,
    int output_block = 0,
    const uint8_t* __restrict__ shared_weights = nullptr) {
  constexpr int kFullK = NGPU * LOCAL_K;
  constexpr int kLocalSubblocks = LOCAL_K / 16;
  constexpr int kSubblocks = kFullK / 16;
  constexpr int kQ8BlocksPerRank = LOCAL_K / QK8_1;
  constexpr int kSuperblocks = kFullK / QK_K;
  constexpr int kQb = kFullK / 512;
  static_assert(kFullK == 2048);
  static_assert(kSubblocks == 32 * kQb);

  auto& activation = shared->activation;
  auto& activation_sum = shared->activation_sum;
  auto& activation_scale = shared->activation_scale;
  const block_q8_1* local_quant = gathered_quant;
  const nv_bfloat16* local_addend = nullptr;
  const block_q8_1* local_shared_quant = nullptr;
  if constexpr (ASSEMBLED_QUANT) {
    constexpr int kFullQuantBytes =
        TOP_K * (kFullK / QK8_1) * sizeof(block_q8_1);
    constexpr int kAddendBytes = kHiddenSize * sizeof(nv_bfloat16);
    constexpr int kSlotBytes =
        (kFullQuantBytes + kAddendBytes + 255) & ~255;
    const int publication_slot =
        publication_signal->dsv4_channel_q2_epoch & 1;
    local_quant = reinterpret_cast<const block_q8_1*>(
        reinterpret_cast<const uint8_t*>(quant_mid_rank_data->ptrs[rank]) +
        publication_slot * kSlotBytes);
    local_addend = reinterpret_cast<const nv_bfloat16*>(
        reinterpret_cast<const uint8_t*>(local_quant) + kFullQuantBytes);
    if constexpr (FOLD_SHARED_Q8) {
      local_shared_quant = reinterpret_cast<const block_q8_1*>(local_addend);
    }
  }
  if constexpr (PENDING) {
    if (pending_header->magic != dsv4_q2_mhc_ar::kPendingQ2Magic ||
        pending_header->intermediate != LOCAL_K) {
      return;
    }
    weights = reinterpret_cast<const uint8_t*>(pending_header->down_weights);
    topk_ids = pending_header->topk_ids;
    expert_stride = pending_header->down_expert_stride;
    experts = pending_header->experts;
  }

  for (int item = threadIdx.x; item < TOP_K * kSubblocks;
       item += blockDim.x) {
    const int slot = item / kSubblocks;
    const int subblock = item - slot * kSubblocks;
    const int source_rank = subblock / kLocalSubblocks;
    const int local_subblock = subblock - source_rank * kLocalSubblocks;
    const int expert = topk_ids[slot];
    uint32_t plane[4] = {0, 0, 0, 0};
    int sum = 0;
    float qscale = 0.0f;
    if (expert >= 0 && expert < experts) {
      const auto* peer_q8 = reinterpret_cast<const block_q8_1*>(
          quant_mid_rank_data->ptrs[source_rank]);
      const block_q8_1& q8 = local_quant == nullptr
          ? peer_q8[slot * kQ8BlocksPerRank + local_subblock / 2]
          : local_quant[slot * (kFullK / QK8_1) + subblock / 2];
      const int8_t* q = q8.qs + (local_subblock & 1) * 16;
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        const uint32_t byte = static_cast<uint8_t>(q[element]);
        plane[element & 3] |= byte << (8 * (element >> 2));
        sum += int(q[element]);
      }
      qscale = __low2float(q8.ds);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      activation[slot][p][subblock] = plane[p];
    }
    activation_sum[slot][subblock] = sum;
    if ((subblock & 1) == 0) {
      activation_scale[slot][subblock / 2] = qscale;
    }
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row = output_block * 8 + warp;
  if (row >= local_rows) return;
  float accum = 0.0f;
#pragma unroll
  for (int slot = 0; slot < TOP_K; ++slot) {
    const int expert = topk_ids[slot];
    if (expert < 0 || expert >= experts) continue;
    const uint8_t* expert_base = weights + int64_t(expert) * expert_stride;
    const uint32_t* quant = reinterpret_cast<const uint32_t*>(expert_base);
    const uint8_t* scales = expert_base + int64_t(local_rows) * kFullK / 4;
    const half2* dm = reinterpret_cast<const half2*>(
        scales + int64_t(local_rows) * kFullK / 16);
    const int quant_row = row * kSubblocks;
    const int dm_row = row * kSuperblocks;
    const int jb = lane * kQb;
    const float2 scale_min = __half22float2(dm[dm_row + jb / 16]);
#pragma unroll
    for (int q8_group = 0; q8_group < kQb / 2; ++q8_group) {
      int dot_scaled = 0;
      int min_scaled = 0;
#pragma unroll
      for (int q = 0; q < 2; ++q) {
        const int j = jb + q8_group * 2 + q;
        const uint32_t packed = quant[quant_row + j];
        const int scale_min_nibbles = scales[quant_row + j];
        int dot = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          dot = __dp4a(int((packed >> (2 * p)) & 0x03030303u),
                       int(activation[slot][p][j]), dot);
        }
        dot_scaled += (scale_min_nibbles & 0x0f) * dot;
        min_scaled +=
            (scale_min_nibbles >> 4) * activation_sum[slot][j];
      }
      const float input_scale =
          activation_scale[slot][jb / 2 + q8_group];
      accum += input_scale *
               (scale_min.x * float(dot_scaled) -
                scale_min.y * float(min_scaled));
    }
  }
  float shared_accum = 0.0f;
  if constexpr (FOLD_SHARED_Q8) {
    constexpr int kSharedBlocks = kFullK / QK8_1;
    const int64_t total_weight_blocks =
        int64_t(local_rows) * kSharedBlocks;
    const auto* shared_scales =
        reinterpret_cast<const half*>(shared_weights);
    const auto* shared_codes = reinterpret_cast<const int8_t*>(
        shared_scales + total_weight_blocks);
    for (int block = lane; block < kSharedBlocks; block += 32) {
      const int64_t weight_block = int64_t(row) * kSharedBlocks + block;
      const auto* weight_codes = reinterpret_cast<const int32_t*>(
          shared_codes + weight_block * QK8_0);
      const auto* input_codes =
          reinterpret_cast<const int32_t*>(local_shared_quant[block].qs);
      int dot = 0;
#pragma unroll
      for (int word = 0; word < QK8_0 / 4; ++word) {
        dot = __dp4a(weight_codes[word], input_codes[word], dot);
      }
      shared_accum = fmaf(
          float(dot), __half2float(shared_scales[weight_block]) *
                          __low2float(local_shared_quant[block].ds),
          shared_accum);
    }
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    accum += __shfl_down_sync(0xffffffffu, accum, offset);
    if constexpr (FOLD_SHARED_Q8) {
      shared_accum +=
          __shfl_down_sync(0xffffffffu, shared_accum, offset);
    }
  }
  if (lane == 0) {
    if constexpr (FOLD_SHARED_Q8) {
      const float routed = float(__float2bfloat16_rn(accum));
      const float shared_value =
          float(__float2bfloat16_rn(shared_accum));
      accum = routed + shared_value;
    } else if constexpr (PENDING || FOLD_ADDEND) {
      // Preserve the native routed-MoE boundary before combining independently
      // produced shared-expert terms. This also makes accumulation order fixed
      // and directly comparable across the old and channel-owned layouts.
      accum = float(__float2bfloat16_rn(accum));
      if (gathered_addend != nullptr) {
        accum += gathered_addend[row];
      } else if (addend_rank_data != nullptr) {
        const int global_row = rank * local_rows + row;
#pragma unroll
        for (int peer = 0; peer < NGPU; ++peer) {
          const auto* addend = reinterpret_cast<const nv_bfloat16*>(
              reinterpret_cast<const uint8_t*>(
                  addend_rank_data->ptrs[peer]) +
              addend_offset_bytes);
          accum += float(addend[global_row]);
        }
      } else if constexpr (ASSEMBLED_QUANT) {
#pragma unroll
        for (int peer = 0; peer < NGPU; ++peer) {
          accum += float(local_addend[peer * local_rows + row]);
        }
      } else {
        const int global_row = rank * local_rows + row;
        const RankData* addend_peers = addend_rank_data == nullptr
            ? quant_mid_rank_data
            : addend_rank_data;
#pragma unroll
        for (int peer = 0; peer < NGPU; ++peer) {
          const auto* addend = reinterpret_cast<const nv_bfloat16*>(
              reinterpret_cast<const uint8_t*>(addend_peers->ptrs[peer]) +
              addend_offset_bytes);
          accum += float(addend[global_row]);
        }
      }
    }
    output[row] = __float2bfloat16_rn(accum);
  }
}

template <int NGPU, int TOP_K, int LOCAL_K, bool PENDING = false,
          bool ASSEMBLED_QUANT = false, bool FOLD_ADDEND = false,
          bool FOLD_SHARED_Q8 = false>
__global__ void __launch_bounds__(256, 2) channel_q2_down(
    const uint8_t* __restrict__ weights, RankData* quant_mid_rank_data,
    const int* __restrict__ topk_ids, nv_bfloat16* __restrict__ output,
    int64_t expert_stride, int local_rows, int experts,
    const dsv4_q2_mhc_ar::PendingQ2Header* pending_header = nullptr,
    RankData* addend_rank_data = nullptr, int addend_offset_bytes = 0,
    int rank = 0, const block_q8_1* gathered_quant = nullptr,
    const float* gathered_addend = nullptr,
    Signal* publication_signal = nullptr,
    const uint8_t* __restrict__ shared_weights = nullptr) {
  __shared__ ChannelQ2DownShared<NGPU, TOP_K, LOCAL_K> shared;
  channel_q2_down_body<NGPU, TOP_K, LOCAL_K, PENDING, ASSEMBLED_QUANT,
                       FOLD_ADDEND, FOLD_SHARED_Q8>(
      weights, quant_mid_rank_data, topk_ids, output, expert_stride, local_rows,
      experts, pending_header, addend_rank_data, addend_offset_bytes, rank,
      gathered_quant, gathered_addend, publication_signal, &shared, blockIdx.x,
      shared_weights);
}

#if 0  // Rejected cooperative direct-read shared-expert diagnostic.
template <int NGPU, int TOP_K, int LOCAL_K>
__global__ void __launch_bounds__(256, 2) channel_q2_down_direct_shared(
    const uint8_t* __restrict__ weights, RankData* publication_rank_data,
    RankData* shared_quant_rank_data, RankSignals signals,
    Signal* self_signal, const int* __restrict__ topk_ids,
    nv_bfloat16* __restrict__ output, int64_t expert_stride, int local_rows,
    int experts, int rank, const uint8_t* __restrict__ shared_weights,
    int shared_quant_offset_bytes) {
  const uint32_t epoch = self_signal->dsv4_channel_q2_epoch + 1;
  if (blockIdx.x == 0) {
    if (threadIdx.x == 0) __threadfence_system();
    __syncthreads();
    if (threadIdx.x < NGPU) {
      const int peer = threadIdx.x;
      st_flag_release(&signals.signals[peer]->dsv4_channel_q2_ready[rank],
                      epoch);
      while (ld_flag_acquire(&self_signal->dsv4_channel_q2_ready[peer]) <
             epoch) {
      }
    }
  }
  cooperative_groups::this_grid().sync();

  __shared__ ChannelQ2DownShared<NGPU, TOP_K, LOCAL_K> shared;
  for (int output_block = blockIdx.x; output_block < (local_rows + 7) / 8;
       output_block += gridDim.x) {
    channel_q2_down_body<NGPU, TOP_K, LOCAL_K, false, true, false, true, true,
                         true>(
        weights, publication_rank_data, topk_ids, output, expert_stride,
        local_rows, experts, nullptr, nullptr, 0, rank, nullptr, nullptr,
        self_signal, &shared, output_block, shared_weights,
        shared_quant_rank_data, shared_quant_offset_bytes);
    __syncthreads();
  }
  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    self_signal->dsv4_channel_q2_epoch = epoch;
  }
}

template <int NGPU, int TOP_K, int LOCAL_K>
inline cudaError_t launch_channel_q2_down_direct_shared(
    const uint8_t* weights, RankData* publication_rank_data,
    RankData* shared_quant_rank_data, RankSignals signals,
    Signal* self_signal, const int* topk_ids, nv_bfloat16* output,
    int64_t expert_stride, int local_rows, int experts, int rank,
    const uint8_t* shared_weights, int shared_quant_offset_bytes,
    cudaStream_t stream) {
  auto kernel = channel_q2_down_direct_shared<NGPU, TOP_K, LOCAL_K>;
  int blocks_per_sm = 0;
  int sm_count = 0;
  int device = 0;
  cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, 256, 0);
  if (error != cudaSuccess) return error;
  if ((error = cudaGetDevice(&device)) != cudaSuccess) return error;
  if ((error = cudaDeviceGetAttribute(
           &sm_count, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess) {
    return error;
  }
  const int output_blocks = (local_rows + 7) / 8;
  const int resident_blocks = blocks_per_sm * sm_count;
  const int blocks =
      output_blocks < resident_blocks ? output_blocks : resident_blocks;
  void* args[] = {
      &weights, &publication_rank_data, &shared_quant_rank_data, &signals,
      &self_signal, &topk_ids, &output, &expert_stride, &local_rows, &experts,
      &rank, &shared_weights, &shared_quant_offset_bytes,
  };
  return cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                     dim3(blocks), dim3(256), args, 0, stream);
}
#endif

template <typename FnT, int NGPU>
__global__ void __launch_bounds__(kThreads, 2) post_pre_deferred(
    RankData* partial_rank_data, RankSignals signals, Signal* self_signal,
    const nv_bfloat16* residual_out, const FnT* fn, float* partial, float* debug,
    const float* scale, const float* base, float* next_post,
    float* next_comb, float sinkhorn_eps, float post_multiplier,
    int sinkhorn_repeat, int rank) {
  static_assert(kHiddenSize % NGPU == 0);
  constexpr int kLocalHidden = kHiddenSize / NGPU;
  constexpr int kLocalTotal = kHC * kLocalHidden;
  constexpr int kWarps = kThreads / 32;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int split = blockIdx.x % kDeferredSplits;
  const int partition = blockIdx.x / kDeferredSplits;
  __shared__ float warp_partials[kWarps][kDeferredPerPartition];
  __shared__ float reduced[kDeferred];
  __shared__ float prepared[kDeferred];
  __shared__ uint32_t exchange_epoch;

  float projection[kDeferredPerPartition] = {};
  for (int flat = split * kThreads + tid; flat < kLocalTotal;
       flat += kDeferredSplits * kThreads) {
    const int stream = flat / kLocalHidden;
    const int dim = flat - stream * kLocalHidden;
    const float value = float(residual_out[flat]);
#pragma unroll
    for (int local_output = 0; local_output < kDeferredPerPartition;
         ++local_output) {
      const int output =
          kHC + partition * kDeferredPerPartition + local_output;
      const size_t fn_index =
          size_t(output) * kHC * kHiddenSize +
          size_t(stream) * kHiddenSize + rank * kLocalHidden + dim;
      projection[local_output] += value * float(fn[fn_index]);
    }
  }
#pragma unroll
  for (int output = 0; output < kDeferredPerPartition; ++output) {
    const float sum = tms::dsv4_mhc::warp_sum(projection[output]);
    if (lane == 0) warp_partials[warp][output] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    for (int output = lane; output < kDeferredPerPartition; output += 32) {
      float value = warp_partials[0][output];
#pragma unroll
      for (int source_warp = 1; source_warp < kWarps; ++source_warp) {
        value += warp_partials[source_warp][output];
      }
      partial[kDeferredScratchOffset +
              (partition * kDeferredSplits + split) *
                  kDeferredPerPartition +
              output] = value;
    }
  }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x != 0) return;
  if (warp == 0) {
    for (int output = lane; output < kDeferred; output += 32) {
      const int source_partition = output / kDeferredPerPartition;
      const int source_output = output % kDeferredPerPartition;
      float value = partial[kDeferredScratchOffset +
                            source_partition * kDeferredSplits *
                                kDeferredPerPartition +
                            source_output];
#pragma unroll
      for (int source_split = 1; source_split < kDeferredSplits;
           ++source_split) {
        value += partial[kDeferredScratchOffset +
                         (source_partition * kDeferredSplits + source_split) *
                             kDeferredPerPartition +
                         source_output];
      }
      reduced[output] = value;
    }
  }
  __syncthreads();
  const int deferred_slot = prepare_exchange<2>(self_signal, &exchange_epoch);
  const int deferred_base = kDeferredOffset + deferred_slot * kDeferred;
  if (tid == 0) {
#pragma unroll
    for (int output = 0; output < kDeferred; ++output) {
      partial[deferred_base + output] = reduced[output];
    }
  }
  publish_and_wait<2, NGPU>(signals, self_signal, rank, exchange_epoch);
  const RankData peers = *partial_rank_data;
  if (warp == 0) {
    for (int output = lane; output < kDeferred; output += 32) {
      float value = reinterpret_cast<const float*>(peers.ptrs[0])[
          deferred_base + output];
#pragma unroll
      for (int peer = 1; peer < NGPU; ++peer) {
        value += reinterpret_cast<const float*>(peers.ptrs[peer])[
            deferred_base + output];
      }
      reduced[output] = value;
    }
  }
  __syncthreads();
  const float inverse_residual_rms = partial[kStateOffset + kHC];
  if (tid < kHC) {
    prepared[tid] = tms::dsv4_mhc::sigmoid(
                        reduced[tid] * inverse_residual_rms * scale[1] +
                        base[kHC + tid]) *
                    post_multiplier;
  }
  float matrix = 0.0f;
  if (lane < kHC * kHC) {
    matrix = reduced[kHC + lane] * inverse_residual_rms * scale[2] +
             base[2 * kHC + lane];
  }
  float row_max = matrix;
  row_max = fmaxf(row_max,
                  __shfl_xor_sync(0xffffffffu, row_max, 1, kHC));
  row_max = fmaxf(row_max,
                  __shfl_xor_sync(0xffffffffu, row_max, 2, kHC));
  if (lane < kHC * kHC) matrix = expf(matrix - row_max);
  float row_sum = matrix;
  row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, kHC);
  row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, kHC);
  if (lane < kHC * kHC) matrix = matrix / row_sum + sinkhorn_eps;
  for (int iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
    if (iteration > 0) {
      row_sum = matrix;
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, kHC);
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, kHC);
      if (lane < kHC * kHC) matrix /= row_sum + sinkhorn_eps;
    }
    float column_sum = matrix;
    column_sum += __shfl_xor_sync(0xffffffffu, column_sum, kHC);
    column_sum += __shfl_xor_sync(0xffffffffu, column_sum, 2 * kHC);
    if (lane < kHC * kHC) matrix /= column_sum + sinkhorn_eps;
  }
  if (lane < kHC * kHC) prepared[kHC + lane] = matrix;
  __syncthreads();
  if (tid < kHC) {
    next_post[tid] = prepared[tid];
  }
  if (tid < kHC * kHC) {
    next_comb[tid] = prepared[kHC + tid];
  }
}

}  // namespace dsv4_mhc_channel_owned
