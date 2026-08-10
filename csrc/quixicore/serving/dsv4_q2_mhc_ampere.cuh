#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm. GGUF block
// layouts and the custom-allreduce peer/signal primitives are already visible.
namespace dsv4_q2_mhc_ar {

constexpr int kTopK = 6;
constexpr int kHidden = 4096;
constexpr int kProducerCtas = 256;
constexpr int kPhysicalSplits = 64;
constexpr int kRowsPerPhysicalSplit = kHidden / kPhysicalSplits;
constexpr int kLogicalSplits = 16;
constexpr int kConsumerCtas = kLogicalSplits;
constexpr int kUrgentOutputs = 4;
constexpr int kPartials = 25;
constexpr int kRequiredPartialValues = kLogicalSplits * kPartials;
constexpr int kCompletionCounters = kPhysicalSplits + 4;
constexpr uint32_t kPendingQ2Magic = 0x44535132u;

struct __align__(16) PendingQ2Header {
  uint64_t down_weights;
  int64_t down_expert_stride;
  int topk_ids[8];
  int intermediate;
  int experts;
  uint32_t magic;
  int reserved;
};
static_assert(sizeof(PendingQ2Header) == 64);

// Retain the measured standalone Q2_K geometry. The only added work is the
// shared-expert add, one system fence, and one progress counter per 16 rows.
template <int K>
__global__ void __launch_bounds__(256, 2) q2_progress_producer(
    const block_q8_1* pending_q8, const PendingQ2Header* header,
    nv_bfloat16* local_output, const nv_bfloat16* local_addend) {
  constexpr int kSubblocks = K / 16;
  constexpr int kQ8Blocks = K / QK8_1;
  constexpr int kSuperblocks = K / QK_K;
  constexpr int kQb = K / 512;
  static_assert(K == 512 || K == 1024);
  static_assert(kSubblocks == 32 * kQb);

  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int task = int(blockIdx.x);
  const int row_base = (task * 8 + warp) * 2;
  auto* completion = reinterpret_cast<unsigned int*>(
      const_cast<PendingQ2Header*>(header)) - kCompletionCounters;

  __shared__ uint32_t activation[kTopK][4][kSubblocks];
  __shared__ int activation_sum[kTopK][kSubblocks];
  __shared__ float activation_scale[kTopK][kQ8Blocks];
  for (int item = tid; item < kTopK * kSubblocks; item += blockDim.x) {
    const int slot = item / kSubblocks;
    const int subblock = item - slot * kSubblocks;
    const int expert = header->topk_ids[slot];
    uint32_t plane[4] = {0, 0, 0, 0};
    int sum = 0;
    float qscale = 0.0f;
    if (expert >= 0 && expert < header->experts) {
      const block_q8_1& q8 =
          pending_q8[slot * kQ8Blocks + subblock / 2];
      const int8_t* q = q8.qs + (subblock & 1) * 16;
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        const uint32_t byte = static_cast<uint8_t>(q[element]);
        plane[element & 3] |= byte << (8 * (element >> 2));
        sum += int(q[element]);
      }
      qscale = __low2float(q8.ds);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p)
      activation[slot][p][subblock] = plane[p];
    activation_sum[slot][subblock] = sum;
    if ((subblock & 1) == 0)
      activation_scale[slot][subblock / 2] = qscale;
  }
  __syncthreads();

  const auto* weights =
      reinterpret_cast<const uint8_t*>(header->down_weights);
  float q2_accum[2] = {0.0f, 0.0f};
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    const int expert = header->topk_ids[slot];
    if (expert < 0 || expert >= header->experts) continue;
    const uint8_t* expert_base =
        weights + int64_t(expert) * header->down_expert_stride;
    const uint32_t* quant = reinterpret_cast<const uint32_t*>(expert_base);
    const uint8_t* scales = expert_base + int64_t(kHidden) * K / 4;
    const half2* dm = reinterpret_cast<const half2*>(
        scales + int64_t(kHidden) * K / 16);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
      const int row = row_base + r;
      const int quant_row = row * kSubblocks;
      const int dm_row = row * kSuperblocks;
      const int jb = lane * kQb;
      int dot_scaled = 0;
      int min_scaled = 0;
#pragma unroll
      for (int q = 0; q < kQb; ++q) {
        const int j = jb + q;
        const uint32_t packed = quant[quant_row + j];
        const int scale_min = scales[quant_row + j];
        int dot = 0;
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          dot = __dp4a(int((packed >> (2 * p)) & 0x03030303u),
                       int(activation[slot][p][j]), dot);
        }
        dot_scaled += (scale_min & 0x0f) * dot;
        min_scaled += (scale_min >> 4) * activation_sum[slot][j];
      }
      const float2 scale_min = __half22float2(dm[dm_row + jb / 16]);
      const float input_scale = activation_scale[slot][jb / 2];
      q2_accum[r] += input_scale *
          (scale_min.x * float(dot_scaled) -
           scale_min.y * float(min_scaled));
    }
  }

#pragma unroll
  for (int r = 0; r < 2; ++r) {
    float value = q2_accum[r];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
      value += __shfl_down_sync(0xffffffffu, value, offset);
    if (lane == 0) {
      const int row = row_base + r;
      const nv_bfloat16 routed = __float2bfloat16_rn(value);
      local_output[row] =
          local_addend == nullptr
              ? routed
              : __float2bfloat16_rn(float(routed) +
                                    float(local_addend[row]));
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence_system();
    atomicAdd(completion + task / 16, 1u);
  }
}

// A high-priority fixed consumer set follows Q2 row-group readiness while the
// producer is still running. Each 256-row publication group maps exactly onto
// one retained logical mHC split and preserves its warp/block reduction order.
template <int NGPU, typename FnT>
__global__ void __launch_bounds__(256, 1) q2_mhc_consumer(
    const PendingQ2Header* header, RankData* output_rank_data,
    RankSignals signals, Signal* self_signal, const nv_bfloat16* residual,
    const float* post_mix, const float* comb_mix, const FnT* fn,
    nv_bfloat16* residual_out, float* partial, const float* scale,
    const float* base, nv_bfloat16* layer_input,
    const nv_bfloat16* norm_weight,
    tms::dsv4_mhc::block_q8_1* quant_input, float rms_eps, float pre_eps,
    float norm_eps, int rank) {
  const int logical_split = int(blockIdx.x);
  const int tid = int(threadIdx.x);
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const RankData peers = *output_rank_data;
  auto* completion = reinterpret_cast<unsigned int*>(
      const_cast<PendingQ2Header*>(header)) - kCompletionCounters;
  auto* phase = completion + kPhysicalSplits;

  __shared__ float warp_partials[8][kUrgentOutputs + 1];
  __shared__ float mix_coeffs[4 + 16];
  if (tid < 4) mix_coeffs[tid] = post_mix[tid];
  if (tid < 16) mix_coeffs[4 + tid] = comb_mix[tid];
  __syncthreads();

  if (tid == 0) {
    while (atomicAdd(completion + logical_split, 0u) != 16u) {}
  }
  __syncthreads();

  const uint32_t ready = self_signal->_flag[logical_split] + 1;
  if (tid < NGPU) {
    st_flag_release(&signals.signals[tid]->start[logical_split][rank], ready);
    while (ld_flag_acquire(
               &self_signal->start[logical_split][tid]) != ready) {}
  }
  __syncthreads();
  if (tid == 0) self_signal->_flag[logical_split] = ready;

  float urgent_accum[kUrgentOutputs] = {0.0f, 0.0f, 0.0f, 0.0f};
  float square_sum = 0.0f;
  const int reduced_dim = logical_split * 256 + tid;
  float reduced_x = float(
      reinterpret_cast<const nv_bfloat16*>(peers.ptrs[0])[reduced_dim]);
#pragma unroll
  for (int peer = 1; peer < NGPU; ++peer) {
    reduced_x += float(
        reinterpret_cast<const nv_bfloat16*>(peers.ptrs[peer])[reduced_dim]);
  }
  const nv_bfloat16 rounded_x = __float2bfloat16_rn(reduced_x);
#pragma unroll
  for (int stream = 0; stream < 4; ++stream) {
    float value = mix_coeffs[stream] * float(rounded_x);
#pragma unroll
    for (int input_stream = 0; input_stream < 4; ++input_stream) {
      value += mix_coeffs[4 + input_stream * 4 + stream] *
               float(residual[input_stream * kHidden + reduced_dim]);
    }
    const nv_bfloat16 rounded = __float2bfloat16_rn(value);
    residual_out[stream * kHidden + reduced_dim] = rounded;
    value = float(rounded);
    square_sum += value * value;
#pragma unroll
    for (int output = 0; output < kUrgentOutputs; ++output) {
      urgent_accum[output] +=
          value *
          float(fn[(output * 4 + stream) * kHidden + reduced_dim]);
    }
  }

#pragma unroll
  for (int output = 0; output < kUrgentOutputs; ++output) {
    const float sum = tms::dsv4_mhc::warp_sum(urgent_accum[output]);
    if (lane == 0) warp_partials[warp][output] = sum;
  }
  const float square_block = tms::dsv4_mhc::warp_sum(square_sum);
  if (lane == 0) warp_partials[warp][kUrgentOutputs] = square_block;
  __syncthreads();

  if (warp == 0) {
    if (lane < kUrgentOutputs) {
      float value = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < 8; ++source_warp)
        value += warp_partials[source_warp][lane];
      partial[logical_split * kPartials + lane] = value;
    }
    if (lane == kUrgentOutputs) {
      float value = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < 8; ++source_warp)
        value += warp_partials[source_warp][kUrgentOutputs];
      partial[logical_split * kPartials + 24] = value;
    }
  }

  __syncthreads();
  if (tid == 0) {
    __threadfence();
    atomicAdd(phase, 1u);
  }
  __syncthreads();
  if (tid == 0) while (atomicAdd(phase, 0u) != kConsumerCtas) {}
  __syncthreads();

  if (logical_split == 0) {
    barrier_at_end<NGPU, true>(signals, self_signal, rank);
    dsv4_mhc_ar::finalize_urgent_pre_mix<kLogicalSplits>(
        partial, scale, base, 0, kHidden, rms_eps, pre_eps);
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      atomicExch(phase + 1, 1u);
    }
  } else {
    if (tid == 0) while (atomicAdd(phase + 1, 0u) != 1u) {}
    __syncthreads();
  }

  const float* pre = partial;
  float* norm_scratch = partial + kLogicalSplits * kPartials - 17;
  if (logical_split < 2) {
    constexpr int kVirtualValues = kHidden / (2 * 256);
    const int virtual_tid = logical_split * 256 + tid;
    float input_square_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kVirtualValues; ++i) {
      const int dim = virtual_tid * kVirtualValues + i;
      float value = 0.0f;
#pragma unroll
      for (int stream = 0; stream < 4; ++stream)
        value += pre[stream] *
                 float(residual_out[stream * kHidden + dim]);
      const nv_bfloat16 rounded = __float2bfloat16_rn(value);
      layer_input[dim] = rounded;
      value = float(rounded);
      input_square_sum += value * value;
    }
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
      const float other =
          __shfl_down_sync(0xffffffffu, input_square_sum, offset);
      if (lane + offset < 32) input_square_sum += other;
    }
    if (lane == 0)
      norm_scratch[logical_split * 8 + warp] = input_square_sum;
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    atomicAdd(phase + 2, 1u);
  }
  __syncthreads();
  if (tid == 0) while (atomicAdd(phase + 2, 0u) != kConsumerCtas) {}
  __syncthreads();

  if (logical_split == 0 && tid == 0) {
    float input_square_sum = norm_scratch[0];
#pragma unroll
    for (int source_warp = 1; source_warp < 16; ++source_warp)
      input_square_sum += norm_scratch[source_warp];
    norm_scratch[16] =
        rsqrtf(input_square_sum / float(kHidden) + norm_eps);
    __threadfence();
    atomicExch(phase + 3, 1u);
  }
  if (logical_split != 0 && tid == 0)
    while (atomicAdd(phase + 3, 0u) != 1u) {}
  __syncthreads();

  const int dim = logical_split * 256 + tid;
  const nv_bfloat16 normalized = __float2bfloat16_rn(
      float(layer_input[dim]) * norm_scratch[16] *
      float(norm_weight[dim]));
  layer_input[dim] = normalized;
  const float value = float(normalized);
  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
    sum += __shfl_xor_sync(0xffffffffu, sum, mask);
  }
  const float qscale = amax / 127.0f;
  const int qblock = logical_split * 8 + warp;
  quant_input[qblock].qs[lane] =
      amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / qscale));
  if (lane == 0)
    quant_input[qblock].ds = __floats2half2_rn(qscale, sum);
}

}  // namespace dsv4_q2_mhc_ar
