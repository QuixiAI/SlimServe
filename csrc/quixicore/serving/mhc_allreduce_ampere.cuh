#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm, after the peer
// pointer and synchronization primitives have been defined.
namespace dsv4_mhc_ar {

template <int NGPU, int NSPLITS = 32, bool ADD_LOCAL_PARTIAL = false,
          bool RMS_NORM_Q8 = false, typename FnT = float>
__global__ void __launch_bounds__(tms::dsv4_mhc::THREADS, 1)
    fused_allreduce_post_pre(
        RankData* rank_data, RankData* addend_rank_data, RankSignals signals,
        Signal* self_signal,
        const nv_bfloat16* x, const nv_bfloat16* residual,
        const float* post_mix, const float* comb_mix, const FnT* fn,
        nv_bfloat16* residual_out, float* partial, const float* scale,
        const float* base, float* next_post, float* next_comb,
        nv_bfloat16* layer_input, const nv_bfloat16* norm_weight,
        tms::dsv4_mhc::block_q8_1* quant_input, float rms_eps, float pre_eps,
        float sinkhorn_eps, float post_multiplier, int sinkhorn_repeat,
        float norm_eps, int rank) {
  static_assert(NSPLITS <= kMaxBlocks);
  constexpr int HIDDEN_SIZE = 4096;
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int NOUT = tms::dsv4_mhc::MIXES;
  constexpr int THREADS = tms::dsv4_mhc::THREADS;
  constexpr int TOTAL = HC * HIDDEN_SIZE;
  constexpr int VALUES = HIDDEN_SIZE / THREADS;
  constexpr int PARTIALS = NOUT + 1;
  constexpr int FN_PARTITIONS = 2;
  constexpr int ACCUMS = NOUT / FN_PARTITIONS;
  static_assert(NOUT % FN_PARTITIONS == 0);

  const int split = blockIdx.x;
  const int logical_split = split % NSPLITS;
  const int fn_partition = split / NSPLITS;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const RankData peers = *rank_data;
  RankData addend_peers;
  if constexpr (ADD_LOCAL_PARTIAL) addend_peers = *addend_rank_data;

  __shared__ float mix_coeffs[HC + HC * HC];
  if (tid < HC) mix_coeffs[tid] = post_mix[token * HC + tid];
  if (tid < HC * HC)
    mix_coeffs[HC + tid] = comb_mix[token * HC * HC + tid];
  __syncthreads();

  barrier_at_start<NGPU>(signals, self_signal, rank);

  float accum[ACCUMS];
#pragma unroll
  for (int output = 0; output < ACCUMS; ++output) accum[output] = 0.0f;
  float square_sum = 0.0f;

  const int first_flat = logical_split * THREADS + tid;
  const int dim = first_flat & (HIDDEN_SIZE - 1);
  // Both flat values owned by a thread are mHC streams at the same hidden
  // dimension. Preserve the rank-ordered reduction while reusing it across
  // those streams.
  auto load_local = [&](int peer) {
    const float lhs = float(
        reinterpret_cast<const nv_bfloat16*>(peers.ptrs[peer])[
            token * HIDDEN_SIZE + dim]);
    if constexpr (ADD_LOCAL_PARTIAL) {
      const float rhs = float(
          reinterpret_cast<const nv_bfloat16*>(addend_peers.ptrs[peer])[
              token * HIDDEN_SIZE + dim]);
      return float(__float2bfloat16_rn(lhs + rhs));
    }
    return lhs;
  };
  float reduced_x = load_local(0);
#pragma unroll
  for (int peer = 1; peer < NGPU; ++peer) reduced_x += load_local(peer);
  const nv_bfloat16 rounded_x = __float2bfloat16_rn(reduced_x);

  for (int flat = first_flat;
       tid < THREADS && flat < TOTAL;
       flat += NSPLITS * THREADS) {
    const int stream = flat / HIDDEN_SIZE;

    float value = mix_coeffs[stream] * float(rounded_x);
#pragma unroll
    for (int input_stream = 0; input_stream < HC; ++input_stream) {
      value += mix_coeffs[HC + input_stream * HC + stream] *
               float(residual[(token * HC + input_stream) * HIDDEN_SIZE + dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    if (fn_partition == 0) {
      residual_out[token * TOTAL + flat] = rounded;
    }
    value = float(rounded);
    if (fn_partition == 0) {
      square_sum += value * value;
    }
#pragma unroll
    for (int local_output = 0; local_output < ACCUMS; ++local_output) {
      const int output = fn_partition * ACCUMS + local_output;
      accum[local_output] += value * float(fn[output * TOTAL + flat]);
    }
  }

  __shared__ float warp_partials[THREADS / 32][ACCUMS + 1];
  if (tid < THREADS) {
#pragma unroll
    for (int output = 0; output < ACCUMS; ++output) {
      const float sum = tms::dsv4_mhc::warp_sum(accum[output]);
      if (lane == 0) warp_partials[warp][output] = sum;
    }
    const float sum = tms::dsv4_mhc::warp_sum(square_sum);
    if (lane == 0) warp_partials[warp][ACCUMS] = sum;
  }
  __syncthreads();

  if (warp == 0) {
    for (int local_output = lane; local_output < ACCUMS; local_output += 32) {
      float block_sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp)
        block_sum += warp_partials[source_warp][local_output];
      const int output = fn_partition * ACCUMS + local_output;
      partial[(token * NSPLITS + logical_split) * PARTIALS + output] =
          block_sum;
    }
    if (lane == 0 && fn_partition == 0) {
      float block_sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp)
        block_sum += warp_partials[source_warp][ACCUMS];
      partial[(token * NSPLITS + logical_split) * PARTIALS + NOUT] =
          block_sum;
    }
  }

  cooperative_groups::this_grid().sync();
  // No rank may reuse its local partial until every peer has finished reading
  // it. All physical blocks participate because the signal is block-indexed.
  barrier_at_end<NGPU, true>(signals, self_signal, rank);
  if constexpr (RMS_NORM_Q8) {
    if (split == 0) {
      tms::dsv4_mhc::finalize_pre_mix_block<NSPLITS>(
          partial, scale, base, next_post, next_comb, token, HIDDEN_SIZE,
          rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_repeat);
    }
    cooperative_groups::this_grid().sync();

    const float* pre = partial + token * NSPLITS * (NOUT + 1);
    float* norm_scratch =
        partial + (token + 1) * NSPLITS * (NOUT + 1) - 17;
    if (split < 2) {
      constexpr int VIRTUAL_VALUES = HIDDEN_SIZE / (2 * THREADS);
      const int virtual_tid = split * THREADS + tid;
      float square_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < VIRTUAL_VALUES; ++i) {
        const int dim = virtual_tid * VIRTUAL_VALUES + i;
        float value = 0.0f;
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
          value +=
              pre[stream] *
              float(residual_out[(token * HC + stream) * HIDDEN_SIZE + dim]);
        }
        const nv_bfloat16 rounded = __float2bfloat16_rn(value);
        layer_input[token * HIDDEN_SIZE + dim] = rounded;
        const float as_float = float(rounded);
        square_sum += as_float * as_float;
      }
      // Match cub::BlockReduce<float, 1024>::Reduce as used by vLLM's
      // 512-thread RMSNorm kernel. CUB's WarpReduce visits shuffle offsets
      // from low to high; reversing this tree changes the FP32 rounding and
      // can alter generated tokens after many decoder layers.
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const float other =
            __shfl_down_sync(0xffffffffu, square_sum, offset);
        if (lane + offset < 32) square_sum += other;
      }
      if (lane == 0) {
        norm_scratch[split * (THREADS / 32) + warp] = square_sum;
      }
    }
    cooperative_groups::this_grid().sync();

    if (split == 0 && tid == 0) {
      float square_sum = norm_scratch[0];
#pragma unroll
      for (int source_warp = 1; source_warp < 16; ++source_warp) {
        square_sum += norm_scratch[source_warp];
      }
      norm_scratch[16] =
          rsqrtf(square_sum / float(HIDDEN_SIZE) + norm_eps);
    }
    cooperative_groups::this_grid().sync();

    if (split >= HIDDEN_SIZE / THREADS) return;
    const int dim = split * THREADS + tid;
    const nv_bfloat16 normalized = __float2bfloat16_rn(
        float(layer_input[token * HIDDEN_SIZE + dim]) * norm_scratch[16] *
        float(norm_weight[dim]));
    layer_input[token * HIDDEN_SIZE + dim] = normalized;
    const float value = float(normalized);
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float qscale = amax / 127.0f;
    const int block = split * (THREADS / 32) + warp;
    quant_input[token * (HIDDEN_SIZE / 32) + block].qs[lane] =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / qscale));
    if (lane == 0) {
      quant_input[token * (HIDDEN_SIZE / 32) + block].ds =
          __floats2half2_rn(qscale, sum);
    }
  } else {
    if (split != 0) return;
    tms::dsv4_mhc::finalize_pre_mix_block<NSPLITS>(
        partial, scale, base, next_post, next_comb, token, HIDDEN_SIZE,
        rms_eps, pre_eps, sinkhorn_eps, post_multiplier, sinkhorn_repeat);
    __syncthreads();

    const float* pre = partial + token * NSPLITS * (NOUT + 1);
    nv_bfloat16 values[VALUES];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
      const int dim = tid + i * THREADS;
      float value = 0.0f;
#pragma unroll
      for (int stream = 0; stream < HC; ++stream) {
        value += pre[stream] *
                 float(residual_out[(token * HC + stream) * HIDDEN_SIZE + dim]);
      }
      values[i] = __float2bfloat16_rn(value);
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
      const int dim = tid + i * THREADS;
      layer_input[token * HIDDEN_SIZE + dim] = values[i];
    }
  }
}

template <int NSPLITS>
__device__ __forceinline__ void finalize_urgent_pre_mix(
    float* partial, const float* scale, const float* base, int token,
    int hidden_size, float rms_eps, float pre_eps) {
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int PARTIALS = tms::dsv4_mhc::MIXES + 1;
  const int lane = threadIdx.x;
  __shared__ float mixes[HC];
  __shared__ float inverse_rms;

  if (lane < HC) {
    float value = 0.0f;
#pragma unroll
    for (int split = 0; split < NSPLITS; ++split) {
      value += partial[(token * NSPLITS + split) * PARTIALS + lane];
    }
    mixes[lane] = value;
  } else if (lane == HC) {
    float square_sum = 0.0f;
#pragma unroll
    for (int split = 0; split < NSPLITS; ++split) {
      square_sum +=
          partial[(token * NSPLITS + split) * PARTIALS + PARTIALS - 1];
    }
    inverse_rms = rsqrtf(square_sum / float(HC * hidden_size) + rms_eps);
  }
  __syncwarp();

  if (lane < HC) {
    partial[token * NSPLITS * PARTIALS + lane] =
        tms::dsv4_mhc::sigmoid(
            mixes[lane] * inverse_rms * scale[0] + base[lane]) +
        pre_eps;
  }
  // The square-sum slot in split zero is dead after this reduction. Preserve
  // inverse RMS there for the deferred post/comb projection.
  if (lane == HC) {
    partial[token * NSPLITS * PARTIALS + PARTIALS - 1] = inverse_rms;
  }
}

template <int NSPLITS>
__device__ __forceinline__ void apply_deferred_post_comb(
    const float* mixes, float inverse_rms, const float* scale,
    const float* base, float* post, float* comb, int token,
    float sinkhorn_eps, float post_multiplier, int sinkhorn_repeat) {
  constexpr int HC = tms::dsv4_mhc::HC;
  const int lane = threadIdx.x;

  if (lane < HC) {
    post[token * HC + lane] =
        tms::dsv4_mhc::sigmoid(
            mixes[lane] * inverse_rms * scale[1] + base[HC + lane]) *
        post_multiplier;
  }

  // Match finalize_pre_mix_block exactly, with mixes[4..19] corresponding to
  // the original fn outputs 8..23.
  float matrix = 0.0f;
  if (lane < HC * HC) {
    matrix = mixes[HC + lane] * inverse_rms * scale[2] +
             base[2 * HC + lane];
  }
  float row_max = matrix;
  row_max = fmaxf(row_max,
                  __shfl_xor_sync(0xffffffffu, row_max, 1, HC));
  row_max = fmaxf(row_max,
                  __shfl_xor_sync(0xffffffffu, row_max, 2, HC));
  if (lane < HC * HC) matrix = expf(matrix - row_max);
  float row_sum = matrix;
  row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, HC);
  row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, HC);
  if (lane < HC * HC) matrix = matrix / row_sum + sinkhorn_eps;

  for (int iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
    if (iteration > 0) {
      row_sum = matrix;
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, HC);
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, HC);
      if (lane < HC * HC) matrix /= row_sum + sinkhorn_eps;
    }
    float column_sum = matrix;
    column_sum += __shfl_xor_sync(0xffffffffu, column_sum, HC);
    column_sum += __shfl_xor_sync(0xffffffffu, column_sum, 2 * HC);
    if (lane < HC * HC) matrix /= column_sum + sinkhorn_eps;
  }
  if (lane < HC * HC) comb[token * HC * HC + lane] = matrix;
}

template <int NSPLITS>
__device__ __forceinline__ void finalize_deferred_post_comb(
    float* partial, const float* scale, const float* base, float* post,
    float* comb, int token, float sinkhorn_eps, float post_multiplier,
    int sinkhorn_repeat) {
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int NOUT = tms::dsv4_mhc::MIXES;
  constexpr int PARTIALS = NOUT + 1;
  constexpr int DEFERRED = NOUT - HC;
  const int lane = threadIdx.x;
  __shared__ float mixes[DEFERRED];
  __shared__ float inverse_rms;

  if (lane < DEFERRED) {
    float value = 0.0f;
#pragma unroll
    for (int split = 0; split < NSPLITS; ++split) {
      value += partial[(token * NSPLITS + split) * PARTIALS + HC + lane];
    }
    mixes[lane] = value;
  }
  if (lane == DEFERRED) {
    inverse_rms = partial[token * NSPLITS * PARTIALS + PARTIALS - 1];
  }
  __syncwarp();
  apply_deferred_post_comb<NSPLITS>(
      mixes, inverse_rms, scale, base, post, comb, token, sinkhorn_eps,
      post_multiplier, sinkhorn_repeat);
}

template <int NGPU, int NSPLITS>
__device__ __forceinline__ void publish_owned_urgent_pre_mix(
    float* partial, const float* scale, const float* base,
    RankSignals signals, int rank, int token, int hidden_size,
    float rms_eps, float pre_eps) {
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int NOUT = tms::dsv4_mhc::MIXES;
  constexpr int PARTIALS = NOUT + 1;
  constexpr int OWNED = (HC + NGPU - 1) / NGPU;
  const int lane = threadIdx.x;

  if (lane < OWNED) {
    const int output = rank + lane * NGPU;
    if (output < HC) {
      float value = 0.0f;
      float square_sum = 0.0f;
#pragma unroll
      for (int split = 0; split < NSPLITS; ++split) {
        value += partial[(token * NSPLITS + split) * PARTIALS + output];
        square_sum +=
            partial[(token * NSPLITS + split) * PARTIALS + NOUT];
      }
      const float inverse_rms =
          rsqrtf(square_sum / float(HC * hidden_size) + rms_eps);
      const float prepared =
          tms::dsv4_mhc::sigmoid(value * inverse_rms * scale[0] +
                                 base[output]) +
          pre_eps;
#pragma unroll
      for (int peer = 0; peer < NGPU; ++peer) {
        signals.signals[peer]->dsv4_mhc_pre[output] = prepared;
      }
    }
  }
  if (lane == HC) {
    float square_sum = 0.0f;
#pragma unroll
    for (int split = 0; split < NSPLITS; ++split) {
      square_sum +=
          partial[(token * NSPLITS + split) * PARTIALS + NOUT];
    }
    partial[token * NSPLITS * PARTIALS + NOUT] =
        rsqrtf(square_sum / float(HC * hidden_size) + rms_eps);
  }
  __syncwarp();
}

__device__ __forceinline__ void load_published_post_comb(
    const Signal* self_signal, float* mix_coeffs, float sinkhorn_eps,
    int sinkhorn_repeat) {
  constexpr int HC = tms::dsv4_mhc::HC;
  const int lane = threadIdx.x;
  if (lane < HC) {
    mix_coeffs[lane] = self_signal->dsv4_mhc_deferred[lane];
  }

  float matrix = 0.0f;
  if (lane < HC * HC) {
    matrix = self_signal->dsv4_mhc_deferred[HC + lane];
  }
  float row_max = matrix;
  row_max = fmaxf(row_max,
                  __shfl_xor_sync(0xffffffffu, row_max, 1, HC));
  row_max = fmaxf(row_max,
                  __shfl_xor_sync(0xffffffffu, row_max, 2, HC));
  if (lane < HC * HC) matrix = expf(matrix - row_max);
  float row_sum = matrix;
  row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, HC);
  row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, HC);
  if (lane < HC * HC) matrix = matrix / row_sum + sinkhorn_eps;

  for (int iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
    if (iteration > 0) {
      row_sum = matrix;
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1, HC);
      row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2, HC);
      if (lane < HC * HC) matrix /= row_sum + sinkhorn_eps;
    }
    float column_sum = matrix;
    column_sum += __shfl_xor_sync(0xffffffffu, column_sum, HC);
    column_sum += __shfl_xor_sync(0xffffffffu, column_sum, 2 * HC);
    if (lane < HC * HC) matrix /= column_sum + sinkhorn_eps;
  }
  if (lane < HC * HC) mix_coeffs[HC + lane] = matrix;
}

// Critical half of the decoder mHC transition. Only the four coefficients
// required to build the next sublayer input remain on the latency path.
template <int NGPU, int NSPLITS = 32, bool ADD_LOCAL_PARTIAL = false,
          bool RMS_NORM_Q8 = false, bool INPUT_PREPARED = false,
          bool OWN_PROJECTIONS = false, bool LOCAL_INPUT_OWNED = false,
          bool CHANNEL_OWNED_X = false, typename FnT = float>
__global__ void __launch_bounds__(tms::dsv4_mhc::THREADS, 1)
    fused_allreduce_post_pre_urgent(
        RankData* rank_data, RankData* addend_rank_data, RankSignals signals,
        Signal* self_signal, const nv_bfloat16* x,
        const nv_bfloat16* residual, const float* post_mix,
        const float* comb_mix, const FnT* fn, nv_bfloat16* residual_out,
        float* partial, const float* scale, const float* base,
        nv_bfloat16* layer_input, const nv_bfloat16* norm_weight,
        tms::dsv4_mhc::block_q8_1* quant_input, float rms_eps, float pre_eps,
        float sinkhorn_eps, int sinkhorn_repeat, float norm_eps, int rank) {
  static_assert(NSPLITS <= kMaxBlocks);
  constexpr int HIDDEN_SIZE = 4096;
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int NOUT = tms::dsv4_mhc::MIXES;
  constexpr int THREADS = tms::dsv4_mhc::THREADS;
  constexpr int TOTAL = HC * HIDDEN_SIZE;
  constexpr int PARTIALS = NOUT + 1;
  constexpr int LOCAL_HIDDEN =
      LOCAL_INPUT_OWNED ? HIDDEN_SIZE / NGPU : HIDDEN_SIZE;
  static_assert(!LOCAL_INPUT_OWNED || RMS_NORM_Q8,
                "local mHC input ownership requires fused RMSNorm/Q8");
  static_assert(!CHANNEL_OWNED_X || !ADD_LOCAL_PARTIAL,
                "channel-owned mHC input cannot carry a TP addend");

  const int split = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const RankData peers = *rank_data;
  RankData addend_peers;
  if constexpr (ADD_LOCAL_PARTIAL) addend_peers = *addend_rank_data;

  __shared__ float mix_coeffs[HC + HC * HC];
  if constexpr (!INPUT_PREPARED) {
    if (tid < HC) mix_coeffs[tid] = post_mix[token * HC + tid];
    if (tid < HC * HC)
      mix_coeffs[HC + tid] = comb_mix[token * HC * HC + tid];
  }
  __syncthreads();
  barrier_at_start<NGPU, INPUT_PREPARED>(signals, self_signal, rank);
  if constexpr (INPUT_PREPARED) {
    load_published_post_comb(self_signal, mix_coeffs, sinkhorn_eps,
                             sinkhorn_repeat);
    __syncthreads();
  }

  constexpr int OWNED_PRE = (HC + NGPU - 1) / NGPU;
  constexpr int PRE_ACCUMS = OWN_PROJECTIONS ? OWNED_PRE : HC;
  float accum[PRE_ACCUMS];
#pragma unroll
  for (int output = 0; output < PRE_ACCUMS; ++output) accum[output] = 0.0f;
  float square_sum = 0.0f;
  const int first_flat = split * THREADS + tid;
  const int dim = first_flat & (HIDDEN_SIZE - 1);
  auto load_partial = [&](int peer) {
    const float lhs = float(
        reinterpret_cast<const nv_bfloat16*>(peers.ptrs[peer])[
            token * HIDDEN_SIZE + dim]);
    if constexpr (ADD_LOCAL_PARTIAL) {
      const float rhs = float(
          reinterpret_cast<const nv_bfloat16*>(addend_peers.ptrs[peer])[
              token * HIDDEN_SIZE + dim]);
      return float(__float2bfloat16_rn(lhs + rhs));
    }
    return lhs;
  };
  float reduced_x;
  if constexpr (CHANNEL_OWNED_X) {
    constexpr int INPUT_SHARD = HIDDEN_SIZE / NGPU;
    const int owner = dim / INPUT_SHARD;
    const int local_dim = dim - owner * INPUT_SHARD;
    reduced_x = float(
        reinterpret_cast<const nv_bfloat16*>(peers.ptrs[owner])[
            token * INPUT_SHARD + local_dim]);
  } else {
    reduced_x = load_partial(0);
#pragma unroll
    for (int peer = 1; peer < NGPU; ++peer) reduced_x += load_partial(peer);
  }
  const nv_bfloat16 rounded_x = __float2bfloat16_rn(reduced_x);

  for (int flat = first_flat; flat < TOTAL;
       flat += NSPLITS * THREADS) {
    const int stream = flat / HIDDEN_SIZE;
    float value = mix_coeffs[stream] * float(rounded_x);
#pragma unroll
    for (int input_stream = 0; input_stream < HC; ++input_stream) {
      value += mix_coeffs[HC + input_stream * HC + stream] *
               float(residual[(token * HC + input_stream) * HIDDEN_SIZE + dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    residual_out[token * TOTAL + flat] = rounded;
    value = float(rounded);
    square_sum += value * value;
#pragma unroll
    for (int local_output = 0; local_output < PRE_ACCUMS; ++local_output) {
      const int output = OWN_PROJECTIONS
                             ? rank + local_output * NGPU
                             : local_output;
      if (output < HC) {
        accum[local_output] += value * float(fn[output * TOTAL + flat]);
      }
    }
  }

  __shared__ float warp_partials[THREADS / 32][PRE_ACCUMS + 1];
#pragma unroll
  for (int output = 0; output < PRE_ACCUMS; ++output) {
    const float sum = tms::dsv4_mhc::warp_sum(accum[output]);
    if (lane == 0) warp_partials[warp][output] = sum;
  }
  const float square_block = tms::dsv4_mhc::warp_sum(square_sum);
  if (lane == 0) warp_partials[warp][PRE_ACCUMS] = square_block;
  __syncthreads();

  if (warp == 0) {
    for (int local_output = lane; local_output < PRE_ACCUMS;
         local_output += 32) {
      float block_sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp)
        block_sum += warp_partials[source_warp][local_output];
      const int output = OWN_PROJECTIONS
                             ? rank + local_output * NGPU
                             : local_output;
      partial[(token * NSPLITS + split) * PARTIALS + output] = block_sum;
    }
    if (lane == 0) {
      float block_sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp)
        block_sum += warp_partials[source_warp][PRE_ACCUMS];
      partial[(token * NSPLITS + split) * PARTIALS + NOUT] = block_sum;
    }
  }

  cooperative_groups::this_grid().sync();
  if (split == 0) {
    // The grid barrier proves that every local CTA has completed its peer
    // reads. One cross-rank handshake is therefore sufficient to protect the
    // registered input buffers from reuse. Keep finalization in the leader CTA
    // and use the existing grid sync below to release the other CTAs.
    if constexpr (OWN_PROJECTIONS) {
      publish_owned_urgent_pre_mix<NGPU, NSPLITS>(
          partial, scale, base, signals, rank, token, HIDDEN_SIZE, rms_eps,
          pre_eps);
      barrier_at_end<NGPU, false>(signals, self_signal, rank);
    } else {
      barrier_at_end<NGPU, true>(signals, self_signal, rank);
      finalize_urgent_pre_mix<NSPLITS>(partial, scale, base, token, HIDDEN_SIZE,
                                       rms_eps, pre_eps);
    }
  }
  cooperative_groups::this_grid().sync();

  const float* pre = OWN_PROJECTIONS
                         ? self_signal->dsv4_mhc_pre
                         : partial + token * NSPLITS * PARTIALS;
  if constexpr (RMS_NORM_Q8) {
    float* norm_scratch =
        partial + (token + 1) * NSPLITS * PARTIALS - 17;
    const int local_start = LOCAL_INPUT_OWNED ? rank * LOCAL_HIDDEN : 0;
    if (split < 2) {
      constexpr int VIRTUAL_VALUES = HIDDEN_SIZE / (2 * THREADS);
      const int virtual_tid = split * THREADS + tid;
      float input_square_sum = 0.0f;
#pragma unroll
      for (int i = 0; i < VIRTUAL_VALUES; ++i) {
        const int input_dim = virtual_tid * VIRTUAL_VALUES + i;
        float value = 0.0f;
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
          value += pre[stream] *
                   float(residual_out[(token * HC + stream) * HIDDEN_SIZE +
                                      input_dim]);
        }
        const nv_bfloat16 rounded = __float2bfloat16_rn(value);
        if constexpr (LOCAL_INPUT_OWNED) {
          if (input_dim >= local_start &&
              input_dim < local_start + LOCAL_HIDDEN) {
            layer_input[token * LOCAL_HIDDEN + input_dim - local_start] =
                rounded;
          }
        } else {
          layer_input[token * HIDDEN_SIZE + input_dim] = rounded;
        }
        const float as_float = float(rounded);
        input_square_sum += as_float * as_float;
      }
#pragma unroll
      for (int offset = 1; offset < 32; offset <<= 1) {
        const float other =
            __shfl_down_sync(0xffffffffu, input_square_sum, offset);
        if (lane + offset < 32) input_square_sum += other;
      }
      if (lane == 0)
        norm_scratch[split * (THREADS / 32) + warp] = input_square_sum;
    }
    cooperative_groups::this_grid().sync();

    if (split == 0 && tid == 0) {
      float input_square_sum = norm_scratch[0];
#pragma unroll
      for (int source_warp = 1; source_warp < 16; ++source_warp)
        input_square_sum += norm_scratch[source_warp];
      norm_scratch[16] =
          rsqrtf(input_square_sum / float(HIDDEN_SIZE) + norm_eps);
    }
    cooperative_groups::this_grid().sync();

    if (split >= LOCAL_HIDDEN / THREADS) return;
    const int local_dim = split * THREADS + tid;
    const int input_dim = local_start + local_dim;
    const nv_bfloat16 normalized = __float2bfloat16_rn(
        float(layer_input[token * LOCAL_HIDDEN + local_dim]) * norm_scratch[16] *
        float(norm_weight[input_dim]));
    layer_input[token * LOCAL_HIDDEN + local_dim] = normalized;
    const float value = float(normalized);
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float qscale = amax / 127.0f;
    const int block = split * (THREADS / 32) + warp;
    quant_input[token * (LOCAL_HIDDEN / 32) + block].qs[lane] =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / qscale));
    if (lane == 0) {
      quant_input[token * (LOCAL_HIDDEN / 32) + block].ds =
          __floats2half2_rn(qscale, sum);
    }
  } else {
    if (split != 0) return;
    constexpr int VALUES = HIDDEN_SIZE / THREADS;
    nv_bfloat16 values[VALUES];
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
      const int input_dim = tid + i * THREADS;
      float value = 0.0f;
#pragma unroll
      for (int stream = 0; stream < HC; ++stream) {
        value += pre[stream] *
                 float(residual_out[(token * HC + stream) * HIDDEN_SIZE +
                                    input_dim]);
      }
      values[i] = __float2bfloat16_rn(value);
    }
#pragma unroll
    for (int i = 0; i < VALUES; ++i) {
      const int input_dim = tid + i * THREADS;
      layer_input[token * HIDDEN_SIZE + input_dim] = values[i];
    }
  }
}

template <int NSPLITS = 32, typename FnT = float>
__global__ void __launch_bounds__(tms::dsv4_mhc::THREADS, 1)
    fused_post_pre_deferred(const nv_bfloat16* residual_out, const FnT* fn,
                            float* partial, const float* scale,
                            const float* base, float* next_post,
                            float* next_comb, float sinkhorn_eps,
                            float post_multiplier, int sinkhorn_repeat) {
  constexpr int HIDDEN_SIZE = 4096;
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int NOUT = tms::dsv4_mhc::MIXES;
  constexpr int THREADS = tms::dsv4_mhc::THREADS;
  constexpr int TOTAL = HC * HIDDEN_SIZE;
  constexpr int PARTIALS = NOUT + 1;
  constexpr int PARTITIONS = 2;
  constexpr int ACCUMS = (NOUT - HC) / PARTITIONS;
  static_assert((NOUT - HC) % PARTITIONS == 0);

  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int logical_split = blockIdx.x % NSPLITS;
  const int partition = blockIdx.x / NSPLITS;
  __shared__ float warp_partials[THREADS / 32][ACCUMS];
  float accum[ACCUMS];
#pragma unroll
  for (int output = 0; output < ACCUMS; ++output) accum[output] = 0.0f;

  for (int flat = logical_split * THREADS + tid; flat < TOTAL;
       flat += NSPLITS * THREADS) {
    const float value = float(residual_out[token * TOTAL + flat]);
#pragma unroll
    for (int local_output = 0; local_output < ACCUMS; ++local_output) {
      const int output = HC + partition * ACCUMS + local_output;
      accum[local_output] += value * float(fn[output * TOTAL + flat]);
    }
  }

#pragma unroll
  for (int output = 0; output < ACCUMS; ++output) {
    const float sum = tms::dsv4_mhc::warp_sum(accum[output]);
    if (lane == 0) warp_partials[warp][output] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    for (int local_output = lane; local_output < ACCUMS;
         local_output += 32) {
      float block_sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp)
        block_sum += warp_partials[source_warp][local_output];
      const int output = HC + partition * ACCUMS + local_output;
      partial[(token * NSPLITS + logical_split) * PARTIALS + output] =
          block_sum;
    }
  }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0) {
    finalize_deferred_post_comb<NSPLITS>(
        partial, scale, base, next_post, next_comb, token, sinkhorn_eps,
        post_multiplier, sinkhorn_repeat);
  }
}

// Rank-owned deferred projection. Each rank computes a disjoint subset of the
// 20 non-urgent rows and remote-publishes prepared post coefficients or
// pre-Sinkhorn matrix logits. The next urgent kernel's existing start epoch is
// the visibility boundary; this kernel deliberately adds no peer fence.
template <int NGPU, int NSPLITS = 16, typename FnT = float>
__global__ void __launch_bounds__(tms::dsv4_mhc::THREADS, 1)
    fused_post_pre_deferred_owned(
        const nv_bfloat16* residual_out, const FnT* fn, float* partial,
        const float* scale, const float* base, RankSignals signals, int rank,
        float post_multiplier) {
  constexpr int HIDDEN_SIZE = 4096;
  constexpr int HC = tms::dsv4_mhc::HC;
  constexpr int NOUT = tms::dsv4_mhc::MIXES;
  constexpr int THREADS = tms::dsv4_mhc::THREADS;
  constexpr int TOTAL = HC * HIDDEN_SIZE;
  constexpr int PARTIALS = NOUT + 1;
  constexpr int DEFERRED = NOUT - HC;
  constexpr int OWNED = (DEFERRED + NGPU - 1) / NGPU;

  const int split = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  __shared__ float warp_partials[THREADS / 32][OWNED];
  float accum[OWNED];
#pragma unroll
  for (int local_output = 0; local_output < OWNED; ++local_output) {
    accum[local_output] = 0.0f;
  }

  for (int flat = split * THREADS + tid; flat < TOTAL;
       flat += NSPLITS * THREADS) {
    const float value = float(residual_out[token * TOTAL + flat]);
#pragma unroll
    for (int local_output = 0; local_output < OWNED; ++local_output) {
      const int deferred = rank + local_output * NGPU;
      if (deferred < DEFERRED) {
        const int output = HC + deferred;
        accum[local_output] += value * float(fn[output * TOTAL + flat]);
      }
    }
  }

#pragma unroll
  for (int local_output = 0; local_output < OWNED; ++local_output) {
    const float sum = tms::dsv4_mhc::warp_sum(accum[local_output]);
    if (lane == 0) warp_partials[warp][local_output] = sum;
  }
  __syncthreads();
  if (warp == 0) {
    for (int local_output = lane; local_output < OWNED;
         local_output += 32) {
      const int deferred = rank + local_output * NGPU;
      if (deferred < DEFERRED) {
        float block_sum = 0.0f;
#pragma unroll
        for (int source_warp = 0; source_warp < THREADS / 32; ++source_warp) {
          block_sum += warp_partials[source_warp][local_output];
        }
        const int output = HC + deferred;
        partial[(token * NSPLITS + split) * PARTIALS + output] = block_sum;
      }
    }
  }

  cooperative_groups::this_grid().sync();
  if (blockIdx.x == 0 && warp == 0) {
    for (int local_output = lane; local_output < OWNED;
         local_output += 32) {
      const int deferred = rank + local_output * NGPU;
      if (deferred < DEFERRED) {
        const int output = HC + deferred;
        float value = 0.0f;
#pragma unroll
        for (int source_split = 0; source_split < NSPLITS; ++source_split) {
          value += partial[(token * NSPLITS + source_split) * PARTIALS +
                           output];
        }
        const float inverse_rms =
            partial[token * NSPLITS * PARTIALS + NOUT];
        const float prepared =
            deferred < HC
                ? tms::dsv4_mhc::sigmoid(value * inverse_rms * scale[1] +
                                         base[output]) *
                      post_multiplier
                : value * inverse_rms * scale[2] + base[output];
#pragma unroll
        for (int peer = 0; peer < NGPU; ++peer) {
          signals.signals[peer]->dsv4_mhc_deferred[deferred] = prepared;
        }
      }
    }
  }
}

}  // namespace dsv4_mhc_ar
