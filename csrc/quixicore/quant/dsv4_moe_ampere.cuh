#pragma once

#ifndef USE_ROCM

#include <cooperative_groups.h>
#include <cstdlib>

// DeepSeek-V4's routed experts use a combined GGUF W1 tensor:
// [expert, gate rows | up rows, packed IQ2_XXS].  A generic MoE path writes
// both projections to fp32, launches SwiGLU, writes another fp32 tensor, then
// quantizes it for Q2_K down.  On Ampere that creates two large intermediates
// and decodes the same activation tile independently for gate and up.
//
// The loading path converts each expert's 66-byte IQ2_XXS blocks to an
// equal-size scale/code SoA. It decodes IQ2_XXS once at tile load (the
// optimized ROCm strategy), computes paired gate/up tiles against one staged
// Q8_1 activation, applies SwiGLU, and emits Q8_1 directly.
// One warp owns one routed row and one 32-value intermediate tile, which makes
// the output quantization a warp reduction with no cross-block coordination.
namespace slimserve::dsv4_ampere {

// IQ2_XXS needs eight routed rows to amortize its codebook decode.  The
// specialized Python route aligns W1 to eight and expands W2's expert map back
// to its native four-wide Q2_K tiles, so these widths are no longer coupled.
constexpr int kMmqY = 32;
constexpr int kWideRouteThreshold = 256;
constexpr uint32_t kPendingQ2Magic = 0x44535132u;  // "DSQ2"

// Decode-only handoff from IQ2_XXS/SwiGLU to the custom-allreduce-owned Q2_K
// producer. The Q8_1 payload occupies the beginning of the normal BF16 output
// allocation and this descriptor occupies its final cache line. The down
// weight pointer is consumed on the same rank that produced it; peer ranks
// only read the published BF16 output tiles.
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

// Deliberately matches custom-allreduce's device RankData without depending on
// that transport header. The fused producer consumes only the peer pointers;
// synchronization and ownership epochs remain the communicator's concern.
struct __align__(16) PublicationRankData {
  const void* ptrs[8];
};
static_assert(sizeof(PublicationRankData) == 8 * sizeof(void*));

static_assert(kMmqY == WARP_SIZE_GGUF);

__device__ __forceinline__ uint32_t iq2_xxs_unpack_signs(uint8_t value) {
  // IQ2_XXS stores seven signs; parity determines the eighth. Keep the
  // expanded bits in registers instead of fetching ksigns_iq2xs from
  // constant memory in the decode-critical inner loop.
  const uint32_t parity = __popc(value) & 1u;
  return (uint32_t(value) ^ (parity << 7)) * 0x01010101u;
}

template <bool REPACKED>
__device__ __forceinline__ void iq2_xxs_gate_up_row_decode(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ input,
    const uint2* __restrict__ grid,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int expert, const int row, const int token,
    float& gate, float& up) {
  const int lane = threadIdx.x & 31;
  const int blocks_per_row = hidden / QK_K;
  const int q8_blocks_per_row = hidden / QK8_1;
  const char* expert_weights = reinterpret_cast<const char*>(weights) +
      int64_t(expert) * expert_stride_bytes;
  const block_iq2_xxs* raw_weights =
      reinterpret_cast<const block_iq2_xxs*>(expert_weights);
  const int paired_blocks_per_expert = intermediate * blocks_per_row;
  const half2* aligned_d = reinterpret_cast<const half2*>(expert_weights);
  const uint4* aligned_q = reinterpret_cast<const uint4*>(
      expert_weights + int64_t(paired_blocks_per_expert) * sizeof(half2));
  const block_q8_1* input_row =
      input + int64_t(token) * q8_blocks_per_row;

  gate = 0.0f;
  up = 0.0f;
  // Eight lanes cover the eight 32-value groups in an IQ2_XXS superblock;
  // four lane groups therefore consume four superblocks per iteration.
  for (int block0 = 0; block0 < blocks_per_row; block0 += 4) {
    const int block = block0 + (lane >> 3);
    const int group = lane & 7;
    if (block >= blocks_per_row) {
      continue;
    }
    const int gate_block_index = row * blocks_per_row + block;
    const int up_block_index =
        (intermediate + row) * blocks_per_row + block;
    const block_q8_1& input_block = input_row[block * 8 + group];
    const int* activation = reinterpret_cast<const int*>(input_block.qs);

    uint32_t gate_code_x;
    uint32_t gate_code_y;
    uint32_t up_code_x;
    uint32_t up_code_y;
    half gate_d;
    half up_d;
    if constexpr (REPACKED) {
      const int paired_block_index = row * blocks_per_row + block;
      const uint4 code = aligned_q[paired_block_index * 8 + group];
      gate_code_x = code.x;
      gate_code_y = code.y;
      up_code_x = code.z;
      up_code_y = code.w;
      const half2 paired_d = aligned_d[paired_block_index];
      gate_d = __low2half(paired_d);
      up_d = __high2half(paired_d);
    } else {
      const block_iq2_xxs& gate_block = raw_weights[gate_block_index];
      const block_iq2_xxs& up_block = raw_weights[up_block_index];
      const uint16_t* gate_q = gate_block.qs + 4 * group;
      const uint16_t* up_q = up_block.qs + 4 * group;
      gate_code_x = uint32_t(gate_q[0]) | (uint32_t(gate_q[1]) << 16);
      gate_code_y = uint32_t(gate_q[2]) | (uint32_t(gate_q[3]) << 16);
      up_code_x = uint32_t(up_q[0]) | (uint32_t(up_q[1]) << 16);
      up_code_y = uint32_t(up_q[2]) | (uint32_t(up_q[3]) << 16);
      gate_d = gate_block.d;
      up_d = up_block.d;
    }

    int gate_dot = 0;
    int up_dot = 0;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      const uint8_t gate_grid_index = uint8_t(gate_code_x >> (8 * part));
      const uint8_t up_grid_index = uint8_t(up_code_x >> (8 * part));
      const uint2 gate_grid = grid[gate_grid_index];
      const uint2 up_grid = grid[up_grid_index];
      const uint32_t gate_signs =
          iq2_xxs_unpack_signs(uint8_t(gate_code_y));
      const uint32_t up_signs =
          iq2_xxs_unpack_signs(uint8_t(up_code_y));
      const uint32_t gate_signs0 =
          __vcmpne4(gate_signs & 0x08040201u, 0);
      const uint32_t gate_signs1 =
          __vcmpne4(gate_signs & 0x80402010u, 0);
      const uint32_t up_signs0 =
          __vcmpne4(up_signs & 0x08040201u, 0);
      const uint32_t up_signs1 =
          __vcmpne4(up_signs & 0x80402010u, 0);
      gate_dot = __dp4a(
          int(__vsub4(gate_grid.x ^ gate_signs0, gate_signs0)),
          activation[2 * part], gate_dot);
      gate_dot = __dp4a(
          int(__vsub4(gate_grid.y ^ gate_signs1, gate_signs1)),
          activation[2 * part + 1], gate_dot);
      up_dot = __dp4a(
          int(__vsub4(up_grid.x ^ up_signs0, up_signs0)),
          activation[2 * part], up_dot);
      up_dot = __dp4a(
          int(__vsub4(up_grid.y ^ up_signs1, up_signs1)),
          activation[2 * part + 1], up_dot);
      gate_code_y >>= 7;
      up_code_y >>= 7;
    }

    const float input_scale = __low2float(input_block.ds);
    const float gate_scale = __half2float(gate_d) *
                             float(2u * gate_code_y + 1u) * 0.125f;
    const float up_scale = __half2float(up_d) *
                           float(2u * up_code_y + 1u) * 0.125f;
    gate = fmaf(float(gate_dot), gate_scale * input_scale, gate);
    up = fmaf(float(up_dot), up_scale * input_scale, up);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, offset);
    up += __shfl_down_sync(0xffffffffu, up, offset);
  }
}

// Decode-batch geometry from the optimized DS4 CUDA path: one eight-lane
// group owns a row, and each lane consumes complete IQ2_XXS superblocks. Four
// independent rows share a warp. This keeps the byte-neutral paired layout and
// arithmetic, but removes the full-warp-per-row launch footprint.
template <bool REPACKED>
__device__ __forceinline__ void iq2_xxs_gate_up_row_decode_qwarp8(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ input,
    const uint2* __restrict__ grid,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int expert, const int row, const int token,
    float& gate, float& up) {
  const int lane = threadIdx.x & 7;
  const int blocks_per_row = hidden / QK_K;
  const int q8_blocks_per_row = hidden / QK8_1;
  const char* expert_weights = reinterpret_cast<const char*>(weights) +
      int64_t(expert) * expert_stride_bytes;
  const block_iq2_xxs* raw_weights =
      reinterpret_cast<const block_iq2_xxs*>(expert_weights);
  const int paired_blocks_per_expert = intermediate * blocks_per_row;
  const half2* aligned_d = reinterpret_cast<const half2*>(expert_weights);
  const uint4* aligned_q = reinterpret_cast<const uint4*>(
      expert_weights + int64_t(paired_blocks_per_expert) * sizeof(half2));
  const block_q8_1* input_row =
      input + int64_t(token) * q8_blocks_per_row;

  gate = 0.0f;
  up = 0.0f;
  for (int block = lane; block < blocks_per_row; block += 8) {
    const int gate_block_index = row * blocks_per_row + block;
    const int up_block_index =
        (intermediate + row) * blocks_per_row + block;
    half gate_d;
    half up_d;
    if constexpr (REPACKED) {
      const half2 paired_d = aligned_d[gate_block_index];
      gate_d = __low2half(paired_d);
      up_d = __high2half(paired_d);
    } else {
      gate_d = raw_weights[gate_block_index].d;
      up_d = raw_weights[up_block_index].d;
    }

#pragma unroll
    for (int group = 0; group < 8; ++group) {
      const block_q8_1& input_block = input_row[block * 8 + group];
      const int* activation = reinterpret_cast<const int*>(input_block.qs);
      uint32_t gate_code_x;
      uint32_t gate_code_y;
      uint32_t up_code_x;
      uint32_t up_code_y;
      if constexpr (REPACKED) {
        const uint4 code = aligned_q[gate_block_index * 8 + group];
        gate_code_x = code.x;
        gate_code_y = code.y;
        up_code_x = code.z;
        up_code_y = code.w;
      } else {
        const uint16_t* gate_q = raw_weights[gate_block_index].qs + 4 * group;
        const uint16_t* up_q = raw_weights[up_block_index].qs + 4 * group;
        gate_code_x = uint32_t(gate_q[0]) | (uint32_t(gate_q[1]) << 16);
        gate_code_y = uint32_t(gate_q[2]) | (uint32_t(gate_q[3]) << 16);
        up_code_x = uint32_t(up_q[0]) | (uint32_t(up_q[1]) << 16);
        up_code_y = uint32_t(up_q[2]) | (uint32_t(up_q[3]) << 16);
      }

      int gate_dot = 0;
      int up_dot = 0;
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        const uint2 gate_grid = grid[uint8_t(gate_code_x >> (8 * part))];
        const uint2 up_grid = grid[uint8_t(up_code_x >> (8 * part))];
        const uint32_t gate_signs =
            iq2_xxs_unpack_signs(uint8_t(gate_code_y));
        const uint32_t up_signs =
            iq2_xxs_unpack_signs(uint8_t(up_code_y));
        const uint32_t gate_signs0 =
            __vcmpne4(gate_signs & 0x08040201u, 0);
        const uint32_t gate_signs1 =
            __vcmpne4(gate_signs & 0x80402010u, 0);
        const uint32_t up_signs0 =
            __vcmpne4(up_signs & 0x08040201u, 0);
        const uint32_t up_signs1 =
            __vcmpne4(up_signs & 0x80402010u, 0);
        gate_dot = __dp4a(
            int(__vsub4(gate_grid.x ^ gate_signs0, gate_signs0)),
            activation[2 * part], gate_dot);
        gate_dot = __dp4a(
            int(__vsub4(gate_grid.y ^ gate_signs1, gate_signs1)),
            activation[2 * part + 1], gate_dot);
        up_dot = __dp4a(
            int(__vsub4(up_grid.x ^ up_signs0, up_signs0)),
            activation[2 * part], up_dot);
        up_dot = __dp4a(
            int(__vsub4(up_grid.y ^ up_signs1, up_signs1)),
            activation[2 * part + 1], up_dot);
        gate_code_y >>= 7;
        up_code_y >>= 7;
      }

      const float input_scale = __low2float(input_block.ds);
      const float gate_scale = __half2float(gate_d) *
                               float(2u * gate_code_y + 1u) * 0.125f;
      const float up_scale = __half2float(up_d) *
                             float(2u * up_code_y + 1u) * 0.125f;
      gate = fmaf(float(gate_dot), gate_scale * input_scale, gate);
      up = fmaf(float(up_dot), up_scale * input_scale, up);
    }
  }

#pragma unroll
  for (int offset = 4; offset > 0; offset >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, offset, 8);
    up += __shfl_down_sync(0xffffffffu, up, offset, 8);
  }
}

// A 32-warp CTA computes one complete Q8_1 block, then its first warp performs
// the SwiGLU/quant epilogue. This avoids a float activation intermediate, but
// TP4 has only 96 such CTAs and therefore cannot occupy all 108 A100 SMs.
template <int TOP_K, bool REPACKED, bool PUBLISH_QUANT = false,
          int PUBLISH_NGPU = 0>
__global__ __launch_bounds__(1024, 1) void
iq2_xxs_gate_up_swiglu_q8_1_decode(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output,
    const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int experts,
    const float swiglu_limit, PendingQ2Header* pending,
    const void* down_weights, const int64_t down_expert_stride,
    PublicationRankData* publication_rank_data,
    const uint32_t* publication_epoch, const int publication_rank,
    const int publication_world_size) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int row = blockIdx.x * 32 + warp;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / QK8_1;

  if (pending != nullptr && blockIdx.x == 0 && route == 0 &&
      threadIdx.x < 68) {
    reinterpret_cast<uint32_t*>(pending)[-68 + int(threadIdx.x)] = 0;
  }
  if (pending != nullptr && blockIdx.x == 0 && lane == 0 && token == 0) {
    if (route < TOP_K) pending->topk_ids[route] = expert;
    if (route == 0) {
      pending->down_weights = reinterpret_cast<uint64_t>(down_weights);
      pending->down_expert_stride = down_expert_stride;
      pending->intermediate = intermediate;
      pending->experts = experts;
      pending->magic = kPendingQ2Magic;
      pending->reserved = 0;
    }
  }

  // IQ2 indices are divergent within every warp. Constant-memory lookup
  // serialization is especially costly here because 32 warps decode gate and
  // up together. Stage the complete 2 KiB codebook once per CTA, matching the
  // optimized DS4 CUDA and ROCm kernels.
  __shared__ __align__(16) uint2 shared_grid[256];
  if (threadIdx.x < 256) {
    shared_grid[threadIdx.x] =
        reinterpret_cast<const uint2*>(iq2xxs_grid)[threadIdx.x];
  }
  __syncthreads();

  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < experts && row < intermediate) {
    iq2_xxs_gate_up_row_decode<REPACKED>(
        weights, input, shared_grid, expert_stride_bytes, hidden, intermediate,
        expert, row, token, gate, up);
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
    if constexpr (PUBLISH_QUANT) {
      constexpr int kFullIntermediate = 2048;
      constexpr int kFullQ8Blocks = kFullIntermediate / QK8_1;
      constexpr int kFullQuantBytes =
          TOP_K * kFullQ8Blocks * sizeof(block_q8_1);
      constexpr int kAddendBytes = 4096 * sizeof(nv_bfloat16);
      constexpr int kSlotBytes =
          (kFullQuantBytes + kAddendBytes + 255) & ~255;
      const int publication_slot = (*publication_epoch + 1) & 1;
      const PublicationRankData peers = *publication_rank_data;
      const int destination_block =
          route * kFullQ8Blocks + publication_rank * blocks_per_mid +
          blockIdx.x;
#pragma unroll
      for (int destination = 0; destination < PUBLISH_NGPU; ++destination) {
          auto* destination_quant = reinterpret_cast<block_q8_1*>(
              reinterpret_cast<uint8_t*>(
                  const_cast<void*>(peers.ptrs[destination])) +
              publication_slot * kSlotBytes);
          destination_quant[destination_block].qs[lane] = quant;
          if (lane == 0) {
            destination_quant[destination_block].ds =
                __floats2half2_rn(scale, sum);
          }
      }
    } else {
      block_q8_1* out =
          output + int64_t(route) * blocks_per_mid + blockIdx.x;
      out->qs[lane] = quant;
      if (lane == 0) {
        out->ds = __floats2half2_rn(scale, sum);
      }
    }
  }

}

template <int TOP_K, bool REPACKED>
__global__ __launch_bounds__(256, 2) void
iq2_xxs_gate_up_swiglu_q8_1_decode_qwarp8(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output,
    const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int experts,
    const float swiglu_limit, PendingQ2Header* pending,
    const void* down_weights, const int64_t down_expert_stride) {
  const int lane = threadIdx.x & 7;
  const int row_lane = threadIdx.x >> 3;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int row = blockIdx.x * 32 + row_lane;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / QK8_1;

  if (pending != nullptr && blockIdx.x == 0 && route == 0 &&
      threadIdx.x < 68) {
    reinterpret_cast<uint32_t*>(pending)[-68 + int(threadIdx.x)] = 0;
  }
  if (pending != nullptr && blockIdx.x == 0 && threadIdx.x == 0 &&
      token == 0) {
    pending->topk_ids[route] = expert;
    if (route == 0) {
      pending->down_weights = reinterpret_cast<uint64_t>(down_weights);
      pending->down_expert_stride = down_expert_stride;
      pending->intermediate = intermediate;
      pending->experts = experts;
      pending->magic = kPendingQ2Magic;
      pending->reserved = 0;
    }
  }

  __shared__ __align__(16) uint2 shared_grid[256];
  shared_grid[threadIdx.x] =
      reinterpret_cast<const uint2*>(iq2xxs_grid)[threadIdx.x];
  __syncthreads();

  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < experts && row < intermediate) {
    iq2_xxs_gate_up_row_decode_qwarp8<REPACKED>(
        weights, input, shared_grid, expert_stride_bytes, hidden, intermediate,
        expert, row, token, gate, up);
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
      if (!isfinite(value)) value = 0.0f;
    }
    values[row_lane] = value;
  }
  __syncthreads();

  if (threadIdx.x < 32) {
    const float value = values[threadIdx.x];
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
    out->qs[threadIdx.x] = quant;
    if (threadIdx.x == 0) out->ds = __floats2half2_rn(scale, sum);
  }
}

// TP4 decode has only 16 Q8 output blocks x 6 routes. The 1024-thread kernel
// above therefore launches 96 one-residency CTAs on 108 SMs. Split each output
// block across two 256-thread producer CTAs, retain exact FP32 SwiGLU values in
// scratch, then quantize after one in-kernel grid handoff. The 192-CTA producer
// wave gives Ampere enough independent weight streams without adding a graph
// node or changing the Q8_1 handoff consumed by Q2_K down.
template <int TOP_K, bool REPACKED>
__global__ __launch_bounds__(256, 2) void
iq2_xxs_gate_up_swiglu_q8_1_decode_cooperative(
    const void* __restrict__ weights,
    const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output,
    float* __restrict__ scratch,
    const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int experts,
    const float swiglu_limit, PendingQ2Header* pending,
    const void* down_weights, const int64_t down_expert_stride) {
  constexpr int kRowsPerHalf = 16;
  constexpr int kWarps = 8;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int half = blockIdx.x & 1;
  const int output_block = blockIdx.x >> 1;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / QK8_1;

  if (pending != nullptr && output_block == 0 && half == 0 && route == 0 &&
      threadIdx.x < 68) {
    reinterpret_cast<uint32_t*>(pending)[-68 + int(threadIdx.x)] = 0;
  }
  if (pending != nullptr && output_block == 0 && half == 0 && lane == 0 &&
      token == 0) {
    if (route < TOP_K) pending->topk_ids[route] = expert;
    if (route == 0) {
      pending->down_weights = reinterpret_cast<uint64_t>(down_weights);
      pending->down_expert_stride = down_expert_stride;
      pending->intermediate = intermediate;
      pending->experts = experts;
      pending->magic = kPendingQ2Magic;
      pending->reserved = 0;
    }
  }

  __shared__ __align__(16) uint2 shared_grid[256];
  shared_grid[threadIdx.x] =
      reinterpret_cast<const uint2*>(iq2xxs_grid)[threadIdx.x];
  __syncthreads();

#pragma unroll
  for (int row_iteration = 0; row_iteration < 2; ++row_iteration) {
    const int row = output_block * 32 + half * kRowsPerHalf + warp +
                    row_iteration * kWarps;
    float gate = 0.0f;
    float up = 0.0f;
    if (expert >= 0 && expert < experts && row < intermediate) {
      iq2_xxs_gate_up_row_decode<REPACKED>(
          weights, input, shared_grid, expert_stride_bytes, hidden,
          intermediate, expert, row, token, gate, up);
    }
    if (lane == 0) {
      float value = 0.0f;
      if (expert >= 0 && expert < experts && row < intermediate) {
        if (swiglu_limit > 0.0f) {
          gate = fminf(gate, swiglu_limit);
          up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
        }
        value = (gate / (1.0f + expf(-gate))) * up * route_weights[route];
        if (!isfinite(value)) value = 0.0f;
      }
      scratch[int64_t(route) * intermediate + row] = value;
    }
  }

  cooperative_groups::this_grid().sync();

  if (half == 0 && warp == 0) {
    const int row = output_block * 32 + lane;
    const float value = row < intermediate
                            ? scratch[int64_t(route) * intermediate + row]
                            : 0.0f;
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
    block_q8_1* out = output + int64_t(route) * blocks_per_mid + output_block;
    out->qs[lane] = quant;
    if (lane == 0) out->ds = __floats2half2_rn(scale, sum);
  }
}

__device__ __forceinline__ void store_zero_q8_1(
    block_q8_1* __restrict__ dst, const int route, const int routed_rows,
    const int blocks_per_mid) {
  if (route >= routed_rows) {
    return;
  }
  block_q8_1* block = dst + route * blocks_per_mid + blockIdx.x;
  block->qs[threadIdx.x] = 0;
  if (threadIdx.x == 0) {
    block->ds = __floats2half2_rn(0.0f, 0.0f);
  }
}

template <int mmq_y, int nwarps, bool need_check>
__device__ __forceinline__ void load_tiles_iq2_xxs_repacked_pair(
    const half2* __restrict__ dq, const uint4* __restrict__ qs,
    int* __restrict__ gate_ql, half2* __restrict__ gate_dm,
    int* __restrict__ up_ql, half2* __restrict__ up_dm,
    const int i_offset, const int i_max, const int k,
    const int blocks_per_row) {
  float* gate_dmf = reinterpret_cast<float*>(gate_dm);
  float* up_dmf = reinterpret_cast<float*>(up_dm);
  const int g = k / 2;
  const int hlf = k % 2;
  const int sb = min(g / 8, blocks_per_row - 1);
  const int ib32 = g % 8;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if constexpr (need_check) {
      i = min(i, i_max);
    }
    const int block = i * blocks_per_row + sb;
    const uint4 code = qs[block * 8 + ib32];
    const uint8_t* gate_grid_index =
        reinterpret_cast<const uint8_t*>(&code.x);
    const uint8_t* up_grid_index =
        reinterpret_cast<const uint8_t*>(&code.z);
    int* gate_dst =
        &gate_ql[i * (IQ2_XXS_TILE_INTS + 1) + g * 8 + hlf * 4];
    int* up_dst = &up_ql[i * (IQ2_XXS_TILE_INTS + 1) + g * 8 + hlf * 4];
#pragma unroll
    for (int l = 0; l < 2; ++l) {
      const int e = 2 * hlf + l;
      const uint2 gate_grid = reinterpret_cast<const uint2*>(
          iq2xxs_grid)[gate_grid_index[e]];
      const uint32_t gate_signs =
          iq2_xxs_unpack_signs(uint8_t(code.y >> (7 * e)));
      const uint32_t gate_signs0 =
          __vcmpne4(gate_signs & 0x08040201u, 0);
      const uint32_t gate_signs1 =
          __vcmpne4(gate_signs & 0x80402010u, 0);
      gate_dst[2 * l] =
          __vsub4(gate_grid.x ^ gate_signs0, gate_signs0);
      gate_dst[2 * l + 1] =
          __vsub4(gate_grid.y ^ gate_signs1, gate_signs1);

      const uint2 up_grid =
          reinterpret_cast<const uint2*>(iq2xxs_grid)[up_grid_index[e]];
      const uint32_t up_signs =
          iq2_xxs_unpack_signs(uint8_t(code.w >> (7 * e)));
      const uint32_t up_signs0 = __vcmpne4(up_signs & 0x08040201u, 0);
      const uint32_t up_signs1 = __vcmpne4(up_signs & 0x80402010u, 0);
      up_dst[2 * l] = __vsub4(up_grid.x ^ up_signs0, up_signs0);
      up_dst[2 * l + 1] = __vsub4(up_grid.y ^ up_signs1, up_signs1);
    }
    if (hlf == 0) {
      const float2 paired_d = __half22float2(dq[block]);
      gate_dmf[i * IQ2_XXS_TILE_SCALES + g] =
          paired_d.x * (0.5f + (code.y >> 28)) * 0.25f;
      up_dmf[i * IQ2_XXS_TILE_SCALES + g] =
          paired_d.y * (0.5f + (code.w >> 28)) * 0.25f;
    }
  }
}

template <int mmq_x, bool need_check, bool fold_route_weight, bool REPACKED>
__global__ __launch_bounds__(WARP_SIZE_GGUF * mmq_x, 2)
void iq2_xxs_gate_up_swiglu_q8_1(
    const void* __restrict__ vw, const void* __restrict__ vy,
    void* __restrict__ vdst, const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const float* __restrict__ route_weights,
    const int* __restrict__ num_tokens_post_padded, const int64_t exp_stride,
    const int hidden, const int hidden_padded, const int intermediate,
    const int intermediate_padded, const int tokens, const int top_k,
    const int tokens_post_padded, const float swiglu_limit) {
  const int row0 = blockIdx.x * kMmqY;
  const int sorted_col = blockIdx.y * mmq_x + threadIdx.y;
  const int routed_rows = tokens * top_k;
  const int blocks_per_mid = intermediate_padded / QK8_1;
  const int route = sorted_token_ids[sorted_col];
  block_q8_1* dst = static_cast<block_q8_1*>(vdst);

  // The grid is graph-static while the live padded count is dynamic.  Every
  // branch here is block-uniform except route validity, which is warp-uniform.
  const bool inactive_block =
      blockIdx.y * mmq_x > num_tokens_post_padded[0];
  const int expert = expert_ids[blockIdx.y];
  if (inactive_block || expert < 0 || row0 >= intermediate) {
    store_zero_q8_1(dst, route, routed_rows, blocks_per_mid);
    return;
  }

  const bool route_live = route < routed_rows;
  const int token = route_live ? route / top_k : 0;
  const int blocks_per_weight_row = hidden / QK_K;
  const int blocks_per_input_row = hidden_padded / QK8_1;

  const char* expert_w = static_cast<const char*>(vw) +
      static_cast<int64_t>(expert) * exp_stride;
  const int paired_blocks_per_expert =
      intermediate * blocks_per_weight_row;
  const half2* aligned_d = reinterpret_cast<const half2*>(expert_w);
  const uint4* aligned_q = reinterpret_cast<const uint4*>(
      expert_w + int64_t(paired_blocks_per_expert) * sizeof(half2));
  const block_q8_1* input = static_cast<const block_q8_1*>(vy);

  __shared__ int gate_qs[kMmqY * (IQ2_XXS_TILE_INTS + 1)];
  __shared__ float gate_d[kMmqY * IQ2_XXS_TILE_SCALES];
  __shared__ int up_qs[kMmqY * (IQ2_XXS_TILE_INTS + 1)];
  __shared__ float up_d[kMmqY * IQ2_XXS_TILE_SCALES];
  __shared__ int input_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ float input_d[mmq_x * (WARP_SIZE_GGUF / QI8_1)];

  float gate = 0.0f;
  float up = 0.0f;
  constexpr int blocks_per_warp = WARP_SIZE_GGUF / QI2_XXS_MMQ;

  for (int ib0 = 0; ib0 < blocks_per_weight_row; ib0 += blocks_per_warp) {
    const int gate_block = row0 * blocks_per_weight_row + ib0;
    const int up_block =
        (intermediate + row0) * blocks_per_weight_row + ib0;
    if constexpr (REPACKED) {
      load_tiles_iq2_xxs_repacked_pair<kMmqY, mmq_x, need_check>(
          aligned_d + gate_block, aligned_q + gate_block * 8, gate_qs,
          reinterpret_cast<half2*>(gate_d), up_qs,
          reinterpret_cast<half2*>(up_d), threadIdx.y,
          intermediate - row0 - 1, threadIdx.x, blocks_per_weight_row);
    } else {
      const block_iq2_xxs* raw =
          reinterpret_cast<const block_iq2_xxs*>(expert_w);
      load_tiles_iq2_xxs<kMmqY, mmq_x, need_check>(
          raw + gate_block, gate_qs, reinterpret_cast<half2*>(gate_d),
          nullptr, nullptr, threadIdx.y, intermediate - row0 - 1,
          threadIdx.x, blocks_per_weight_row);
      load_tiles_iq2_xxs<kMmqY, mmq_x, need_check>(
          raw + up_block, up_qs, reinterpret_cast<half2*>(up_d),
          nullptr, nullptr, threadIdx.y, intermediate - row0 - 1,
          threadIdx.x, blocks_per_weight_row);
    }
    __syncthreads();

#pragma unroll
    for (int ir = 0; ir < QR2_XXS_MMQ; ++ir) {
      const int kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int input_block =
          ib0 * (QK_K / QK8_1) + kqs / QI8_1;
      if (route_live && input_block < blocks_per_input_row) {
        const block_q8_1* src =
            input + token * blocks_per_input_row + input_block;
        input_qs[threadIdx.y * WARP_SIZE_GGUF + threadIdx.x] =
            get_int_from_int8_aligned(src->qs, threadIdx.x % QI8_1);
      } else {
        input_qs[threadIdx.y * WARP_SIZE_GGUF + threadIdx.x] = 0;
      }

      if (threadIdx.x < WARP_SIZE_GGUF / QI8_1) {
        const int scale_block = ib0 * (QK_K / QK8_1) +
                                ir * (WARP_SIZE_GGUF / QI8_1) + threadIdx.x;
        input_d[threadIdx.y * (WARP_SIZE_GGUF / QI8_1) + threadIdx.x] =
            route_live && scale_block < blocks_per_input_row
                ? __low2float(
                      input[token * blocks_per_input_row + scale_block].ds)
                : 0.0f;
      }
      __syncthreads();

      if (route_live) {
#pragma unroll
        for (int k = ir * WARP_SIZE_GGUF / QR2_XXS_MMQ;
             k < (ir + 1) * WARP_SIZE_GGUF / QR2_XXS_MMQ;
             k += VDR_IQ2_XXS_Q8_1_MMQ) {
          gate += vec_dot_iq2_xxs_q8_1_mul_mat(
              gate_qs, reinterpret_cast<const half2*>(gate_d), nullptr,
              nullptr, input_qs, reinterpret_cast<const half2*>(input_d),
              threadIdx.x, threadIdx.y, k);
          up += vec_dot_iq2_xxs_q8_1_mul_mat(
              up_qs, reinterpret_cast<const half2*>(up_d), nullptr, nullptr,
              input_qs, reinterpret_cast<const half2*>(input_d), threadIdx.x,
              threadIdx.y, k);
        }
      }
      __syncthreads();
    }
  }

  const int row = row0 + threadIdx.x;
  float value = 0.0f;
  if (route_live && row < intermediate) {
    if (swiglu_limit > 0.0f) {
      gate = fminf(gate, swiglu_limit);
      up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
    }
    value = (gate / (1.0f + expf(-gate))) * up;
    if constexpr (fold_route_weight) {
      value *= route_weights[route];
    }
    if (!isfinite(value)) {
      value = 0.0f;
    }
  }

  float amax = fabsf(value);
  float sum = value;
#pragma unroll
  for (int mask = WARP_SIZE_GGUF / 2; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, WARP_SIZE_GGUF));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, WARP_SIZE_GGUF);
  }

  if (!route_live) {
    return;
  }
  const float scale = amax / 127.0f;
  const int8_t q = amax == 0.0f ? 0 : static_cast<int8_t>(roundf(value / scale));
  block_q8_1* out = dst + route * blocks_per_mid + blockIdx.x;
  out->qs[threadIdx.x] = q;
  if (threadIdx.x == 0) {
    out->ds = __floats2half2_rn(scale, sum);
  }
}

__device__ __forceinline__ float half_warp_sum(float value) {
  const int lane = threadIdx.x & 31;
  const unsigned mask = lane < 16 ? 0x0000ffffu : 0xffff0000u;
#pragma unroll
  for (int offset = 8; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  return value;
}

// Repack the combined [gate rows | up rows] tensor as paired rows. One half2
// carries both block scales and one uint4 carries both code pairs, so the
// decode kernel consumes gate/up with one aligned instruction while preserving
// the original tensor byte count and expert stride.
static __global__ void repack_iq2_xxs_experts(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    const int experts, const int blocks_per_expert,
    const int64_t expert_stride) {
  const int blocks_per_projection = blocks_per_expert / 2;
  const int64_t pairs = int64_t(experts) * blocks_per_projection * 8;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < pairs; index += int64_t(gridDim.x) * blockDim.x) {
    const int pair = index & 7;
    const int64_t global_pair_block = index >> 3;
    const int expert = global_pair_block / blocks_per_projection;
    const int pair_block = global_pair_block -
        int64_t(expert) * blocks_per_projection;
    const uint8_t* expert_input = src + int64_t(expert) * expert_stride;
    const uint8_t* gate_input = expert_input +
        int64_t(pair_block) * sizeof(block_iq2_xxs);
    const uint8_t* up_input = expert_input +
        int64_t(blocks_per_projection + pair_block) * sizeof(block_iq2_xxs);
    uint8_t* output = dst + int64_t(expert) * expert_stride;

    if (pair == 0) {
      half2 scale;
      memcpy(&scale.x, gate_input, sizeof(half));
      memcpy(&scale.y, up_input, sizeof(half));
      reinterpret_cast<half2*>(output)[pair_block] = scale;
    }
    uint4 code;
    memcpy(&code.x, gate_input + sizeof(half) + pair * sizeof(uint2),
           sizeof(uint2));
    memcpy(&code.z, up_input + sizeof(half) + pair * sizeof(uint2),
           sizeof(uint2));
    uint4* codes = reinterpret_cast<uint4*>(
        output + int64_t(blocks_per_projection) * sizeof(half2));
    codes[pair_block * 8 + pair] = code;
  }
}

// Byte-neutral per-expert Q2_K layout. Each expert keeps the same byte count
// and stride as raw GGUF, but splits the 2-bit words, scale/min nibbles, and
// superblock d/dmin pairs into contiguous planes. This is the measured GLM-5.2
// Ampere layout specialized for DSV4's short down projection.
static __global__ void repack_q2_k_experts(
    const uint8_t* __restrict__ src, uint8_t* __restrict__ dst,
    const int experts, const int rows, const int cols,
    const int64_t expert_stride) {
  const int blocks_per_row = cols / QK_K;
  const int64_t total = int64_t(experts) * rows * blocks_per_row;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total; index += int64_t(gridDim.x) * blockDim.x) {
    const int expert = index / (int64_t(rows) * blocks_per_row);
    const int within_expert = index -
        int64_t(expert) * rows * blocks_per_row;
    const int row = within_expert / blocks_per_row;
    const int block = within_expert - row * blocks_per_row;
    const block_q2_K* input = reinterpret_cast<const block_q2_K*>(
        src + int64_t(expert) * expert_stride);
    const block_q2_K& value = input[row * blocks_per_row + block];

    uint8_t* expert_out = dst + int64_t(expert) * expert_stride;
    uint32_t* quant = reinterpret_cast<uint32_t*>(expert_out);
    uint8_t* scale = expert_out + int64_t(rows) * cols / 4;
    half2* dm = reinterpret_cast<half2*>(
        scale + int64_t(rows) * cols / 16);
    const int subblocks_per_row = cols / 16;
    const int superblocks_per_row = cols / QK_K;

#pragma unroll
    for (int subblock = 0; subblock < 16; ++subblock) {
      const int chunk = subblock >> 3;
      const int remainder = subblock & 7;
      const int shift_group = remainder >> 1;
      const int half = remainder & 1;
      uint32_t packed = 0;
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        const uint32_t q =
            (value.qs[chunk * 32 + half * 16 + element] >>
             (2 * shift_group)) & 3u;
        packed |= q << (8 * (element >> 2) + 2 * (element & 3));
      }
      const int output_subblock =
          row * subblocks_per_row + block * 16 + subblock;
      quant[output_subblock] = packed;
      scale[output_subblock] = value.scales[subblock];
    }
    dm[row * superblocks_per_row + block] = value.dm;
  }
}

template <int TOP_K, int K, int NR, int QB, typename out_t>
__global__ __launch_bounds__(256, 2) void q2_k_down_sum_repacked(
    const uint8_t* __restrict__ weights,
    const block_q8_1* __restrict__ quant_mid,
    const int* __restrict__ topk_ids, out_t* __restrict__ output,
    const int64_t expert_stride, const int out_rows, const int tokens,
    const int experts) {
  constexpr int kSubblocks = K / 16;
  constexpr int kQ8Blocks = K / QK8_1;
  constexpr int kSuperblocks = K / QK_K;
  // K=512/1024/2048 fill the warp exactly (one/two/four subblocks per lane).
  // K=256 (the TP8 shard) has 16 subblocks, so half the warp idles in the
  // subblock loop -- see the jb bound below.
  static_assert(kSubblocks <= 32 * QB);

  __shared__ uint32_t activation[TOP_K][4][kSubblocks];
  __shared__ int activation_sum[TOP_K][kSubblocks];
  __shared__ float activation_scale[TOP_K][kQ8Blocks];

  const int token = blockIdx.y;
  if (token >= tokens) {
    return;
  }

  // Convert each standard Q8_1 route once into four conflict-free dp4a
  // operand planes. Every output warp in this CTA reuses the staged tile.
  for (int item = threadIdx.x; item < TOP_K * kSubblocks;
       item += blockDim.x) {
    const int slot = item / kSubblocks;
    const int subblock = item - slot * kSubblocks;
    const int expert = topk_ids[token * TOP_K + slot];
    uint32_t plane[4] = {0, 0, 0, 0};
    int sum = 0;
    float scale = 0.0f;
    if (expert >= 0 && expert < experts) {
      const block_q8_1& q8 =
          quant_mid[(token * TOP_K + slot) * kQ8Blocks + subblock / 2];
      const int8_t* q = q8.qs + (subblock & 1) * 16;
#pragma unroll
      for (int element = 0; element < 16; ++element) {
        const uint32_t byte = static_cast<uint8_t>(q[element]);
        plane[element & 3] |= byte << (8 * (element >> 2));
        sum += int(q[element]);
      }
      scale = __low2float(q8.ds);
    }
#pragma unroll
    for (int p = 0; p < 4; ++p) {
      activation[slot][p][subblock] = plane[p];
    }
    activation_sum[slot][subblock] = sum;
    if ((subblock & 1) == 0) {
      activation_scale[slot][subblock / 2] = scale;
    }
  }
  __syncthreads();

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int row_base = (blockIdx.x * 8 + warp) * NR;
  float accum[NR] = {0.0f};

#pragma unroll
  for (int slot = 0; slot < TOP_K; ++slot) {
    const int expert = topk_ids[token * TOP_K + slot];
    if (expert < 0 || expert >= experts) {
      continue;
    }
    const uint8_t* expert_base =
        weights + int64_t(expert) * expert_stride;
    const uint32_t* quant = reinterpret_cast<const uint32_t*>(expert_base);
    const uint8_t* scales =
        expert_base + int64_t(out_rows) * K / 4;
    const half2* dm = reinterpret_cast<const half2*>(
        scales + int64_t(out_rows) * K / 16);

#pragma unroll
    for (int r = 0; r < NR; ++r) {
      const int row = row_base + r;
      if (row >= out_rows) {
        continue;
      }
      const int quant_row = row * kSubblocks;
      const int dm_row = row * kSuperblocks;
      const int jb = lane * QB;
      if (jb < kSubblocks) {  // K=256: lanes 16..31 have no subblock
        int dot_scaled = 0;
        int min_scaled = 0;
#pragma unroll
        for (int q = 0; q < QB; ++q) {
          const int j = jb + q;
          const uint32_t packed = quant[quant_row + j];
          const int scale_min = scales[quant_row + j];
          int dot = 0;
#pragma unroll
          for (int p = 0; p < 4; ++p) {
            dot = __dp4a(
                int((packed >> (2 * p)) & 0x03030303u),
                int(activation[slot][p][j]), dot);
          }
          dot_scaled += (scale_min & 0x0f) * dot;
          min_scaled += (scale_min >> 4) * activation_sum[slot][j];
        }
        const float2 scale_min = __half22float2(dm[dm_row + jb / 16]);
        const float input_scale = activation_scale[slot][jb / 2];
        accum[r] += input_scale *
            (scale_min.x * float(dot_scaled) -
             scale_min.y * float(min_scaled));
      }
    }
  }

#pragma unroll
  for (int r = 0; r < NR; ++r) {
    float value = accum[r];
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    const int row = row_base + r;
    if (lane == 0 && row < out_rows) {
      output[int64_t(token) * out_rows + row] = out_t(value);
    }
  }
}

// DS4-style decode down leg. Route weights are already folded into quant_mid,
// so each half warp computes one output row across all routes and writes the
// final token result. No [token, route, hidden] down tensor is materialized.
template <typename out_t, int TOP_K>
__global__ __launch_bounds__(256, 1) void q2_k_down_weighted_sum(
    const void* __restrict__ vw, const block_q8_1* __restrict__ quant_mid,
    const int* __restrict__ topk_ids, out_t* __restrict__ output,
    const int64_t exp_stride, const int intermediate,
    const int intermediate_padded, const int out_rows, const int tokens,
    const int experts) {
  const int token = blockIdx.y;
  const int half_lane = threadIdx.x & 15;
  const int row_lane = threadIdx.x >> 4;
  const int blocks_per_weight_row = intermediate / QK_K;
  const int blocks_per_mid = intermediate_padded / QK8_1;

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
        const block_q2_K* weight = reinterpret_cast<const block_q2_K*>(
            static_cast<const char*>(vw) + int64_t(expert) * exp_stride);
        const block_q8_1* input = quant_mid + route * blocks_per_mid;
        for (int block = 0; block < blocks_per_weight_row; ++block) {
          value += vec_dot_q2_K_q8_1(
              weight + row * blocks_per_weight_row + block,
              input + block * (QK_K / QK8_1), half_lane);
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

template <int mmq_x, bool fold_route_weight, bool REPACKED>
inline void launch_iq2_xxs_gate_up_swiglu_q8_1_width(
    const void* w, const void* input, void* output,
    const int* sorted_token_ids, const int* expert_ids,
    const float* route_weights,
    const int* num_tokens_post_padded, const int64_t exp_stride,
    const int hidden, const int hidden_padded, const int intermediate,
    const int intermediate_padded, const int tokens, const int top_k,
    const int tokens_post_padded, const float swiglu_limit,
    cudaStream_t stream) {
  const dim3 grid((intermediate_padded + kMmqY - 1) / kMmqY,
                  tokens_post_padded / mmq_x, 1);
  const dim3 block(WARP_SIZE_GGUF, mmq_x, 1);
  if (intermediate % kMmqY == 0) {
    iq2_xxs_gate_up_swiglu_q8_1<mmq_x, false, fold_route_weight, REPACKED>
        <<<grid, block, 0, stream>>>(
        w, input, output, sorted_token_ids, expert_ids,
        route_weights,
        num_tokens_post_padded, exp_stride, hidden, hidden_padded,
        intermediate, intermediate_padded, tokens, top_k,
        tokens_post_padded, swiglu_limit);
  } else {
    iq2_xxs_gate_up_swiglu_q8_1<mmq_x, true, fold_route_weight, REPACKED>
        <<<grid, block, 0, stream>>>(
        w, input, output, sorted_token_ids, expert_ids,
        route_weights,
        num_tokens_post_padded, exp_stride, hidden, hidden_padded,
        intermediate, intermediate_padded, tokens, top_k,
        tokens_post_padded, swiglu_limit);
  }
}

inline void launch_iq2_xxs_gate_up_swiglu_q8_1_decode(
    const void* weights, const void* input, void* output,
    const int* topk_ids, const float* route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int top_k,
    const int experts, const float swiglu_limit, const bool w1_repacked,
    PendingQ2Header* pending, const void* down_weights,
    const int64_t down_expert_stride,
    cudaStream_t stream) {
  const dim3 grid((intermediate + 31) / 32, tokens * top_k, 1);
  static const bool qwarp8 = [] {
    const char* value = std::getenv("VLLM_DSV4_W1_QWARP8");
    return value != nullptr && value[0] == '1';
  }();
  if (top_k == 6) {
    if (w1_repacked) {
      if (qwarp8) {
        iq2_xxs_gate_up_swiglu_q8_1_decode_qwarp8<6, true>
            <<<grid, 256, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input),
            static_cast<block_q8_1*>(output), topk_ids, route_weights,
            expert_stride_bytes, hidden, intermediate, tokens, experts,
            swiglu_limit, pending, down_weights, down_expert_stride);
      } else {
        iq2_xxs_gate_up_swiglu_q8_1_decode<6, true>
            <<<grid, 1024, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input),
            static_cast<block_q8_1*>(output), topk_ids, route_weights,
            expert_stride_bytes, hidden, intermediate, tokens, experts,
            swiglu_limit, pending, down_weights, down_expert_stride, nullptr,
            nullptr, 0, 0);
      }
    } else {
      if (qwarp8) {
        iq2_xxs_gate_up_swiglu_q8_1_decode_qwarp8<6, false>
            <<<grid, 256, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input),
            static_cast<block_q8_1*>(output), topk_ids, route_weights,
            expert_stride_bytes, hidden, intermediate, tokens, experts,
            swiglu_limit, pending, down_weights, down_expert_stride);
      } else {
        iq2_xxs_gate_up_swiglu_q8_1_decode<6, false>
            <<<grid, 1024, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input),
            static_cast<block_q8_1*>(output), topk_ids, route_weights,
            expert_stride_bytes, hidden, intermediate, tokens, experts,
            swiglu_limit, pending, down_weights, down_expert_stride, nullptr,
            nullptr, 0, 0);
      }
    }
  } else if (w1_repacked) {
    if (qwarp8) {
      iq2_xxs_gate_up_swiglu_q8_1_decode_qwarp8<8, true>
          <<<grid, 256, 0, stream>>>(
          weights, static_cast<const block_q8_1*>(input),
          static_cast<block_q8_1*>(output), topk_ids, route_weights,
          expert_stride_bytes, hidden, intermediate, tokens, experts,
          swiglu_limit, pending, down_weights, down_expert_stride);
    } else {
      iq2_xxs_gate_up_swiglu_q8_1_decode<8, true><<<grid, 1024, 0, stream>>>(
          weights,
          static_cast<const block_q8_1*>(input),
          static_cast<block_q8_1*>(output), topk_ids, route_weights,
          expert_stride_bytes, hidden, intermediate, tokens, experts,
          swiglu_limit, pending, down_weights, down_expert_stride, nullptr,
          nullptr, 0, 0);
    }
  } else {
    if (qwarp8) {
      iq2_xxs_gate_up_swiglu_q8_1_decode_qwarp8<8, false>
          <<<grid, 256, 0, stream>>>(
          weights, static_cast<const block_q8_1*>(input),
          static_cast<block_q8_1*>(output), topk_ids, route_weights,
          expert_stride_bytes, hidden, intermediate, tokens, experts,
          swiglu_limit, pending, down_weights, down_expert_stride);
    } else {
      iq2_xxs_gate_up_swiglu_q8_1_decode<8, false><<<grid, 1024, 0, stream>>>(
          weights,
          static_cast<const block_q8_1*>(input),
          static_cast<block_q8_1*>(output), topk_ids, route_weights,
          expert_stride_bytes, hidden, intermediate, tokens, experts,
          swiglu_limit, pending, down_weights, down_expert_stride, nullptr,
          nullptr, 0, 0);
    }
  }
}

inline void launch_iq2_xxs_gate_up_swiglu_q8_1_decode_publish(
    const void* weights, const void* input,
    PublicationRankData* publication_rank_data,
    const uint32_t* publication_epoch, const int publication_rank,
    const int publication_world_size, const int* topk_ids,
    const float* route_weights, const int64_t expert_stride_bytes,
    const int hidden, const int intermediate, const int experts,
    const float swiglu_limit, const bool w1_repacked, cudaStream_t stream) {
  const dim3 grid((intermediate + 31) / 32, 6, 1);
  if (w1_repacked && publication_world_size == 2) {
    iq2_xxs_gate_up_swiglu_q8_1_decode<6, true, true, 2>
        <<<grid, 1024, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input), nullptr, topk_ids,
            route_weights, expert_stride_bytes, hidden, intermediate, 1,
            experts, swiglu_limit, nullptr, nullptr, 0, publication_rank_data,
            publication_epoch, publication_rank, publication_world_size);
  } else if (w1_repacked) {
    iq2_xxs_gate_up_swiglu_q8_1_decode<6, true, true, 4>
        <<<grid, 1024, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input), nullptr, topk_ids,
            route_weights, expert_stride_bytes, hidden, intermediate, 1,
            experts, swiglu_limit, nullptr, nullptr, 0, publication_rank_data,
            publication_epoch, publication_rank, publication_world_size);
  } else if (publication_world_size == 2) {
    iq2_xxs_gate_up_swiglu_q8_1_decode<6, false, true, 2>
        <<<grid, 1024, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input), nullptr, topk_ids,
            route_weights, expert_stride_bytes, hidden, intermediate, 1,
            experts, swiglu_limit, nullptr, nullptr, 0, publication_rank_data,
            publication_epoch, publication_rank, publication_world_size);
  } else {
    iq2_xxs_gate_up_swiglu_q8_1_decode<6, false, true, 4>
        <<<grid, 1024, 0, stream>>>(
            weights, static_cast<const block_q8_1*>(input), nullptr, topk_ids,
            route_weights, expert_stride_bytes, hidden, intermediate, 1,
            experts, swiglu_limit, nullptr, nullptr, 0, publication_rank_data,
            publication_epoch, publication_rank, publication_world_size);
  }
}

inline void launch_iq2_xxs_gate_up_swiglu_q8_1_decode_cooperative(
    const void* weights, const void* input, void* output, float* scratch,
    const int* topk_ids, const float* route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int intermediate, const int tokens, const int top_k,
    const int experts, const float swiglu_limit, const bool w1_repacked,
    PendingQ2Header* pending, const void* down_weights,
    const int64_t down_expert_stride, cudaStream_t stream) {
  const dim3 grid(2 * ((intermediate + 31) / 32), tokens * top_k, 1);
  const void* weights_arg = weights;
  const void* input_arg = input;
  void* output_arg = output;
  const int* topk_ids_arg = topk_ids;
  const float* route_weights_arg = route_weights;
  int64_t expert_stride_arg = expert_stride_bytes;
  int hidden_arg = hidden;
  int intermediate_arg = intermediate;
  int tokens_arg = tokens;
  int experts_arg = experts;
  float swiglu_limit_arg = swiglu_limit;
  const void* down_weights_arg = down_weights;
  int64_t down_expert_stride_arg = down_expert_stride;
  void* args[] = {
      &weights_arg, &input_arg, &output_arg, &scratch, &topk_ids_arg,
      &route_weights_arg, &expert_stride_arg, &hidden_arg, &intermediate_arg,
      &tokens_arg, &experts_arg, &swiglu_limit_arg, &pending,
      &down_weights_arg, &down_expert_stride_arg,
  };
  cudaError_t error;
  if (w1_repacked) {
    auto kernel = iq2_xxs_gate_up_swiglu_q8_1_decode_cooperative<6, true>;
    error = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                        grid, dim3(256), args, 0, stream);
  } else {
    auto kernel = iq2_xxs_gate_up_swiglu_q8_1_decode_cooperative<6, false>;
    error = cudaLaunchCooperativeKernel(reinterpret_cast<const void*>(kernel),
                                        grid, dim3(256), args, 0, stream);
  }
  if (error != cudaSuccess) {
    throw std::runtime_error(
        std::string("DSV4 cooperative W1 launch failed: ") +
        cudaGetErrorString(error));
  }
}

// Routed-row count at which the fused IQ2 W1 route switches from the 4-wide
// to the 8-wide tile layout. Runtime-tunable for A/B isolation of the wide
// layout's cost (the Python dispatch reads the same variable so the
// expert-map expansion stays consistent with the launched width).
inline int iq2_wide_route_threshold() {
  static const int rows = [] {
    const char* value = std::getenv("VLLM_GGUF_DSV4_W1_WIDE_ROWS");
    if (value == nullptr) {
      return kWideRouteThreshold;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : kWideRouteThreshold;
  }();
  return rows;
}

template <bool REPACKED>
inline void launch_iq2_xxs_gate_up_swiglu_q8_1_layout(
    const void* w, const void* input, void* output,
    const int* sorted_token_ids, const int* expert_ids,
    const float* route_weights,
    const int* num_tokens_post_padded, const int64_t exp_stride,
    const int hidden, const int hidden_padded, const int intermediate,
    const int intermediate_padded, const int tokens, const int top_k,
    const int tokens_post_padded, const float swiglu_limit,
    const bool fold_route_weight, cudaStream_t stream) {
  if (tokens * top_k >= iq2_wide_route_threshold()) {
    if (fold_route_weight) {
      launch_iq2_xxs_gate_up_swiglu_q8_1_width<8, true, REPACKED>(
          w, input, output, sorted_token_ids, expert_ids, route_weights,
          num_tokens_post_padded, exp_stride, hidden, hidden_padded,
          intermediate, intermediate_padded, tokens, top_k,
          tokens_post_padded, swiglu_limit, stream);
    } else {
      launch_iq2_xxs_gate_up_swiglu_q8_1_width<8, false, REPACKED>(
          w, input, output, sorted_token_ids, expert_ids, route_weights,
          num_tokens_post_padded, exp_stride, hidden, hidden_padded,
          intermediate, intermediate_padded, tokens, top_k,
          tokens_post_padded, swiglu_limit, stream);
    }
  } else if (fold_route_weight) {
    launch_iq2_xxs_gate_up_swiglu_q8_1_width<4, true, REPACKED>(
        w, input, output, sorted_token_ids, expert_ids, route_weights,
        num_tokens_post_padded, exp_stride, hidden, hidden_padded,
        intermediate, intermediate_padded, tokens, top_k,
        tokens_post_padded, swiglu_limit, stream);
  } else {
    launch_iq2_xxs_gate_up_swiglu_q8_1_width<4, false, REPACKED>(
        w, input, output, sorted_token_ids, expert_ids, route_weights,
        num_tokens_post_padded, exp_stride, hidden, hidden_padded,
        intermediate, intermediate_padded, tokens, top_k,
        tokens_post_padded, swiglu_limit, stream);
  }
}

inline void launch_iq2_xxs_gate_up_swiglu_q8_1(
    const void* w, const void* input, void* output,
    const int* sorted_token_ids, const int* expert_ids,
    const float* route_weights,
    const int* num_tokens_post_padded, const int64_t exp_stride,
    const int hidden, const int hidden_padded, const int intermediate,
    const int intermediate_padded, const int tokens, const int top_k,
    const int tokens_post_padded, const float swiglu_limit,
    const bool fold_route_weight, const bool w1_repacked,
    cudaStream_t stream) {
  if (w1_repacked) {
    launch_iq2_xxs_gate_up_swiglu_q8_1_layout<true>(
        w, input, output, sorted_token_ids, expert_ids, route_weights,
        num_tokens_post_padded, exp_stride, hidden, hidden_padded,
        intermediate, intermediate_padded, tokens, top_k,
        tokens_post_padded, swiglu_limit, fold_route_weight, stream);
  } else {
    launch_iq2_xxs_gate_up_swiglu_q8_1_layout<false>(
        w, input, output, sorted_token_ids, expert_ids, route_weights,
        num_tokens_post_padded, exp_stride, hidden, hidden_padded,
        intermediate, intermediate_padded, tokens, top_k,
        tokens_post_padded, swiglu_limit, fold_route_weight, stream);
  }
}

template <typename out_t>
inline void launch_q2_k_down_weighted_sum(
    const void* w, const void* quant_mid, const int* topk_ids, out_t* output,
    const int64_t exp_stride, const int intermediate,
    const int intermediate_padded, const int out_rows, const int tokens,
    const int top_k, const int experts, cudaStream_t stream) {
  const dim3 grid((out_rows + 63) / 64, tokens, 1);
  if (top_k == 6) {
    q2_k_down_weighted_sum<out_t, 6><<<grid, 256, 0, stream>>>(
        w, static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        exp_stride, intermediate, intermediate_padded, out_rows, tokens,
        experts);
  } else {
    q2_k_down_weighted_sum<out_t, 8><<<grid, 256, 0, stream>>>(
        w, static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        exp_stride, intermediate, intermediate_padded, out_rows, tokens,
        experts);
  }
}

inline void launch_repack_q2_k_experts(
    const void* input, void* output, const int experts, const int rows,
    const int cols, const int64_t expert_stride, cudaStream_t stream) {
  const int64_t blocks = int64_t(experts) * rows * (cols / QK_K);
  const int grid = int((blocks + 255) / 256);
  repack_q2_k_experts<<<grid, 256, 0, stream>>>(
      static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output),
      experts, rows, cols, expert_stride);
}

inline void launch_repack_iq2_xxs_experts(
    const void* input, void* output, const int experts, const int rows,
    const int cols, const int64_t expert_stride, cudaStream_t stream) {
  const int blocks_per_expert = rows * (cols / QK_K);
  const int64_t pairs = int64_t(experts) * blocks_per_expert * 8;
  const int grid = int((pairs + 255) / 256);
  repack_iq2_xxs_experts<<<grid, 256, 0, stream>>>(
      static_cast<const uint8_t*>(input), static_cast<uint8_t*>(output),
      experts, blocks_per_expert, expert_stride);
}

inline int q2_k_down_rows_per_warp() {
  static const int rows = [] {
    const char* value = std::getenv("VLLM_DSV4_Q2_DOWN_ROWS");
    if (value == nullptr) {
      return 2;
    }
    const int parsed = std::atoi(value);
    return parsed == 4 || parsed == 8 ? parsed : 2;
  }();
  return rows;
}

template <typename out_t, int TOP_K, int NR>
inline void launch_q2_k_down_sum_repacked_topk(
    const void* weights, const void* quant_mid, const int* topk_ids,
    out_t* output, const int64_t expert_stride, const int intermediate,
    const int out_rows, const int tokens, const int experts,
    cudaStream_t stream) {
  const dim3 grid((out_rows + 8 * NR - 1) / (8 * NR), tokens, 1);
  if (intermediate == 256) {
    // TP8 shard. QB=1 leaves half the warp idle in the subblock loop; the
    // kernel guards for it. Was previously (and silently) launched as the
    // 512 instantiation, which read past every row.
    q2_k_down_sum_repacked<TOP_K, 256, NR, 1, out_t>
        <<<grid, 256, 0, stream>>>(
        static_cast<const uint8_t*>(weights),
        static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        expert_stride, out_rows, tokens, experts);
  } else if (intermediate == 1024) {
    q2_k_down_sum_repacked<TOP_K, 1024, NR, 2, out_t>
        <<<grid, 256, 0, stream>>>(
        static_cast<const uint8_t*>(weights),
        static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        expert_stride, out_rows, tokens, experts);
  } else if (intermediate == 2048) {
    q2_k_down_sum_repacked<TOP_K, 2048, NR, 4, out_t>
        <<<grid, 256, 0, stream>>>(
        static_cast<const uint8_t*>(weights),
        static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        expert_stride, out_rows, tokens, experts);
  } else {
    q2_k_down_sum_repacked<TOP_K, 512, NR, 1, out_t>
        <<<grid, 256, 0, stream>>>(
        static_cast<const uint8_t*>(weights),
        static_cast<const block_q8_1*>(quant_mid), topk_ids, output,
        expert_stride, out_rows, tokens, experts);
  }
}

template <typename out_t>
inline void launch_q2_k_down_sum_repacked(
    const void* weights, const void* quant_mid, const int* topk_ids,
    out_t* output, const int64_t expert_stride, const int intermediate,
    const int out_rows, const int tokens, const int top_k, const int experts,
    cudaStream_t stream) {
  const int rows_per_warp = q2_k_down_rows_per_warp();
  if (top_k == 6) {
    if (rows_per_warp == 2) {
      launch_q2_k_down_sum_repacked_topk<out_t, 6, 2>(
          weights, quant_mid, topk_ids, output, expert_stride, intermediate,
          out_rows, tokens, experts, stream);
    } else if (rows_per_warp == 8) {
      launch_q2_k_down_sum_repacked_topk<out_t, 6, 8>(
          weights, quant_mid, topk_ids, output, expert_stride, intermediate,
          out_rows, tokens, experts, stream);
    } else {
      launch_q2_k_down_sum_repacked_topk<out_t, 6, 4>(
          weights, quant_mid, topk_ids, output, expert_stride, intermediate,
          out_rows, tokens, experts, stream);
    }
  } else {
    if (rows_per_warp == 2) {
      launch_q2_k_down_sum_repacked_topk<out_t, 8, 2>(
          weights, quant_mid, topk_ids, output, expert_stride, intermediate,
          out_rows, tokens, experts, stream);
    } else if (rows_per_warp == 8) {
      launch_q2_k_down_sum_repacked_topk<out_t, 8, 8>(
          weights, quant_mid, topk_ids, output, expert_stride, intermediate,
          out_rows, tokens, experts, stream);
    } else {
      launch_q2_k_down_sum_repacked_topk<out_t, 8, 4>(
          weights, quant_mid, topk_ids, output, expert_stride, intermediate,
          out_rows, tokens, experts, stream);
    }
  }
}

}  // namespace slimserve::dsv4_ampere

#endif  // !USE_ROCM
