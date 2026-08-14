#pragma once

#ifndef USE_ROCM

// DeepSeek-V4 hybrid "Q4K tail" experts (layers 37-42 of the q4ktail
// artifact) ship Q4_K gate/up and Q4_K down.  Before this path they ran the
// generic GGUF MoE route: a float intermediate for gate|up, a separate SwiGLU
// launch, a Q8_1 requantize, an MMQ down that needs four moe_align metadata
// launches per layer, and a weighted moe_sum.  These two kernels mirror the
// fused IQ2_XXS/Q2_K decode pair in dsv4_moe_ampere.cuh: one launch computes
// gate+up+SwiGLU+route-weight and emits Q8_1 directly, one launch computes the
// weighted down sum with no materialized [token, route, hidden] tensor.
//
// Raw GGUF block_q4_K rows are consumed directly (no repack): the layout is
// 4-byte aligned at every vec_dot access and the tail is only six layers, so
// a byte-neutral repack has no measured case yet.
namespace slimserve::dsv4_ampere {

// One warp owns one intermediate row (gate row `row`, up row
// `intermediate + row`).  Half-warps split the superblocks; within a
// superblock the 16 lanes cover the 16 vec_dot positions (iqs = 0,2..30).
__device__ __forceinline__ void q4_k_gate_up_row_dot(
    const block_q4_K* __restrict__ expert_weights,
    const block_q8_1* __restrict__ input_row, const int blocks_per_row,
    const int intermediate, const int row, float& gate, float& up) {
  const int lane = threadIdx.x & 31;
  const int half = lane >> 4;
  const int half_lane = lane & 15;
  const block_q4_K* gate_row = expert_weights + row * blocks_per_row;
  const block_q4_K* up_row =
      expert_weights + (intermediate + row) * blocks_per_row;

  gate = 0.0f;
  up = 0.0f;
  for (int block = half; block < blocks_per_row; block += 2) {
    const block_q8_1* q8 = input_row + block * (QK_K / QK8_1);
    gate += vec_dot_q4_K_q8_1(gate_row + block, q8, 2 * half_lane);
    up += vec_dot_q4_K_q8_1(up_row + block, q8, 2 * half_lane);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, offset);
    up += __shfl_down_sync(0xffffffffu, up, offset);
  }
}

// A 32-warp CTA computes one complete Q8_1 output block; warp 0 performs the
// SwiGLU/quant epilogue with the route weight folded in, matching the Q8_1
// handoff the down kernel expects (same contract as the IQ2 decode kernel).
template <int TOP_K>
__global__ __launch_bounds__(1024, 1) void q4_k_gate_up_swiglu_q8_1_decode(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output, const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int experts,
    const float swiglu_limit) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int row = blockIdx.x * 32 + warp;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / QK8_1;
  const int blocks_per_row = hidden / QK_K;
  const int q8_blocks_per_row = hidden / QK8_1;

  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < experts && row < intermediate) {
    const block_q4_K* expert_weights = reinterpret_cast<const block_q4_K*>(
        reinterpret_cast<const char*>(weights) +
        int64_t(expert) * expert_stride_bytes);
    q4_k_gate_up_row_dot(expert_weights,
                         input + int64_t(token) * q8_blocks_per_row,
                         blocks_per_row, intermediate, row, gate, up);
  }

  __shared__ float values[32];
  if (lane == 0) {
    float value = 0.0f;
    if (expert >= 0 && expert < experts && row < intermediate) {
      if (swiglu_limit > 0.0f) {
        gate = fminf(gate, swiglu_limit);
        up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
      }
      value = (gate / (1.0f + expf(-gate))) * up * route_weights[route];
      if (!isfinite(value)) {
        value = 0.0f;
      }
    }
    values[warp] = value;
  }
  __syncthreads();

  if (warp == 0) {
    const float value = values[lane];
    float amax = fabsf(value);
    float sum = value;
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask));
      sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    }
    const float scale = amax / 127.0f;
    const int8_t quant =
        amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
    block_q8_1* out = output + int64_t(route) * blocks_per_mid + blockIdx.x;
    out->qs[lane] = quant;
    if (lane == 0) {
      out->ds = __floats2half2_rn(scale, sum);
    }
  }
}

// Down leg, mirroring q2_k_down_weighted_sum: route weights are already
// folded into quant_mid, so each half warp computes one output row across all
// routes and writes the final token result directly.
template <typename out_t, int TOP_K>
__global__ __launch_bounds__(256, 1) void q4_k_down_weighted_sum(
    const void* __restrict__ vw, const block_q8_1* __restrict__ quant_mid,
    const int* __restrict__ topk_ids, out_t* __restrict__ output,
    const int64_t exp_stride, const int intermediate, const int out_rows,
    const int tokens, const int experts) {
  const int token = blockIdx.y;
  const int half_lane = threadIdx.x & 15;
  const int row_lane = threadIdx.x >> 4;
  const int blocks_per_weight_row = intermediate / QK_K;
  const int blocks_per_mid = intermediate / QK8_1;

#pragma unroll
  for (int row_step = 0; row_step < 4; ++row_step) {
    const int row = blockIdx.x * 64 + row_lane + row_step * 16;
    if (token >= tokens || row >= out_rows) {
      continue;
    }
    float total = 0.0f;
#pragma unroll
    for (int slot = 0; slot < TOP_K; ++slot) {
      const int route = token * TOP_K + slot;
      const int expert = topk_ids[route];
      float value = 0.0f;
      if (expert >= 0 && expert < experts) {
        const block_q4_K* weight = reinterpret_cast<const block_q4_K*>(
            static_cast<const char*>(vw) + int64_t(expert) * exp_stride);
        const block_q8_1* input = quant_mid + route * blocks_per_mid;
        for (int block = 0; block < blocks_per_weight_row; ++block) {
          value += vec_dot_q4_K_q8_1(
              weight + row * blocks_per_weight_row + block,
              input + block * (QK_K / QK8_1), 2 * half_lane);
        }
      }
      value = half_warp_sum(value);
      if (half_lane == 0) {
        total += isfinite(value) ? value : 0.0f;
      }
    }
    if (half_lane == 0) {
      output[int64_t(token) * out_rows + row] = out_t(total);
    }
  }
}

inline void launch_q4_k_gate_up_swiglu_q8_1_decode(
    const void* weights, const void* input, void* output,
    const int* topk_ids, const float* route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int top_k,
    const int experts, const float swiglu_limit, cudaStream_t stream) {
  const dim3 grid((intermediate + 31) / 32, tokens * top_k, 1);
  if (top_k == 6) {
    q4_k_gate_up_swiglu_q8_1_decode<6><<<grid, 1024, 0, stream>>>(
        weights, static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1*>(output), topk_ids, route_weights,
        expert_stride_bytes, hidden, intermediate, tokens, experts,
        swiglu_limit);
  } else {
    q4_k_gate_up_swiglu_q8_1_decode<8><<<grid, 1024, 0, stream>>>(
        weights, static_cast<const block_q8_1*>(input),
        static_cast<block_q8_1*>(output), topk_ids, route_weights,
        expert_stride_bytes, hidden, intermediate, tokens, experts,
        swiglu_limit);
  }
}

template <typename out_t>
inline void launch_q4_k_down_weighted_sum(
    const void* w, const void* quant_mid, const int* topk_ids, out_t* output,
    const int64_t exp_stride, const int intermediate, const int out_rows,
    const int tokens, const int top_k, const int experts,
    cudaStream_t stream) {
  const dim3 grid((out_rows + 63) / 64, tokens, 1);
  if (top_k == 6) {
    q4_k_down_weighted_sum<out_t, 6><<<grid, 256, 0, stream>>>(
        w, static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        exp_stride, intermediate, out_rows, tokens, experts);
  } else {
    q4_k_down_weighted_sum<out_t, 8><<<grid, 256, 0, stream>>>(
        w, static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        exp_stride, intermediate, out_rows, tokens, experts);
  }
}

}  // namespace slimserve::dsv4_ampere

#endif  // !USE_ROCM
