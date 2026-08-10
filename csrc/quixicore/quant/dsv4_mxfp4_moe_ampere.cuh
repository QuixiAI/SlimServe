#pragma once

#ifndef USE_ROCM

// DeepSeek-V4 MXFP4 routed experts, Ampere fused decode path.
//
// The MXFP4 build stores both W1 ([expert, gate|up, packed]) and W2
// ([expert, out_row, packed]) as OCP MXFP4: 32 values in 17 bytes (one e8m0
// scale byte + 16 bytes of e2m1 nibbles). The generic MoE route dequantizes
// through the MMVQ vector kernels one projection at a time with fp32
// intermediates; on A100 that measured ~32 tok/s at TP4 decode.
//
// This path mirrors the tuned IQ2_XXS/Q2_K pipeline in dsv4_moe_ampere.cuh
// (itself ported from the optimized ROCm strategy): quantize the activation
// to Q8_1 once, compute paired gate/up rows against the staged activation,
// apply SwiGLU and the route weight, emit Q8_1 directly, and consume it with
// a weighted MXFP4 down kernel that accumulates all routes into the output.
// One warp owns one intermediate row, so the Q8_1 emission is a warp
// reduction with no cross-block coordination.
//
// The raw GGUF layout interleaves the scale byte with the codes (17-byte
// blocks), which breaks aligned vectorized loads. A byte-neutral repack
// splits each expert into a scale array followed by a 16-byte-aligned code
// array (same total bytes, same expert stride), so the inner loop issues one
// uint4 per block. e2m1 decode goes through the same integer table as the
// MMVQ kernel: table values are 2x the true magnitudes and the 0.5 factor is
// folded into the block scale, keeping the inner loop on __dp4a.

namespace slimserve::dsv4_ampere {

__device__ __forceinline__ float mxfp4_scale_to_fp32(uint8_t x) {
  // e8m0: the byte is the fp32 exponent field; 0 is the smallest normal.
  const uint32_t bits = (x == 0) ? 0x00400000u : ((uint32_t)x << 23);
  float r;
  memcpy(&r, &bits, 4);
  return r;
}

// Expand 8 packed e2m1 codes (one int) into two ints of 2x-value int8 lanes.
// NOTE: CUDA's __byte_perm takes 4-bit nibble selectors (16-bit total), not
// AMD-style per-byte selectors, so the ROCm/XPU byte-permute table trick
// needs a selector repack that erodes its advantage; the scalar table loop
// compiles to predicated selects and measured fine at decode widths.
__device__ __forceinline__ void mxfp4_expand8(int q4, int& lo, int& hi) {
  static constexpr int8_t kValues[16] = {0, 1,  2,  3,  4,  6,  8,  12,
                                         0, -1, -2, -3, -4, -6, -8, -12};
  const uint32_t l = (uint32_t)q4, h = ((uint32_t)q4 >> 4);
  int8_t bl[4], bh[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    bl[i] = kValues[(l >> (8 * i)) & 0xF];
    bh[i] = kValues[(h >> (8 * i)) & 0xF];
  }
  memcpy(&lo, bl, 4);
  memcpy(&hi, bh, 4);
}

// One token's staged Q8_1 block held in registers so gate and up rows can
// both dot against a single set of activation loads (the shared-activation
// pattern from the tuned XPU nvfp4_row_dot_pair).
struct Q8Block {
  int v[8];
  float d;
};

__device__ __forceinline__ void mxfp4_load_q8(const block_q8_1* q8,
                                              Q8Block& r) {
  const int* q8i = reinterpret_cast<const int*>(q8->qs);
#pragma unroll
  for (int i = 0; i < 8; ++i) r.v[i] = q8i[i];
  r.d = __low2float(q8->ds);
}

// One MXFP4 block (32 values) against a register-staged Q8_1 block. `codes`
// holds the 16 nibble bytes; `scale` is the raw e8m0 byte.
__device__ __forceinline__ float mxfp4_block_dot_regs(const uint4 codes,
                                                      const uint8_t scale,
                                                      const Q8Block& q8) {
  int sumi = 0;
  int lo, hi;
  mxfp4_expand8((int)codes.x, lo, hi);
  sumi = __dp4a(lo, q8.v[0], sumi);
  sumi = __dp4a(hi, q8.v[4], sumi);
  mxfp4_expand8((int)codes.y, lo, hi);
  sumi = __dp4a(lo, q8.v[1], sumi);
  sumi = __dp4a(hi, q8.v[5], sumi);
  mxfp4_expand8((int)codes.z, lo, hi);
  sumi = __dp4a(lo, q8.v[2], sumi);
  sumi = __dp4a(hi, q8.v[6], sumi);
  mxfp4_expand8((int)codes.w, lo, hi);
  sumi = __dp4a(lo, q8.v[3], sumi);
  sumi = __dp4a(hi, q8.v[7], sumi);
  return mxfp4_scale_to_fp32(scale) * 0.5f * q8.d * float(sumi);
}

__device__ __forceinline__ float mxfp4_block_dot(const uint4 codes,
                                                 const uint8_t scale,
                                                 const block_q8_1* q8) {
  Q8Block r;
  mxfp4_load_q8(q8, r);
  return mxfp4_block_dot_regs(codes, scale, r);
}

// Fetch one block's codes+scale from either layout. `nblocks` is the number
// of MXFP4 blocks per expert (repacked scale-region size in bytes).
template <bool REPACKED>
__device__ __forceinline__ void mxfp4_load_block(
    const char* __restrict__ expert_base, const int64_t nblocks,
    const int64_t block_index, uint4& codes, uint8_t& scale) {
  if constexpr (REPACKED) {
    scale = reinterpret_cast<const uint8_t*>(expert_base)[block_index];
    codes = reinterpret_cast<const uint4*>(expert_base + nblocks)[block_index];
  } else {
    const char* block = expert_base + block_index * 17;
    scale = *reinterpret_cast<const uint8_t*>(block);
    memcpy(&codes, block + 1, 16);
  }
}

// Row dot for one (gate,up) pair. Eight lanes cooperate on a row; the caller
// reduces across the 8-lane group.
template <bool REPACKED>
__device__ __forceinline__ void mxfp4_gate_up_row_dot(
    const char* __restrict__ expert_base, const block_q8_1* __restrict__ input,
    const int64_t nblocks, const int hidden, const int input_token_blocks,
    const int intermediate, const int row, const int token, const int lane8,
    float& gate, float& up) {
  const int blocks_per_row = hidden / 32;
  const block_q8_1* token_input =
      input + int64_t(token) * input_token_blocks;
  const int64_t gate_base = int64_t(row) * blocks_per_row;
  const int64_t up_base = int64_t(row + intermediate) * blocks_per_row;
  for (int k = lane8; k < blocks_per_row; k += 8) {
    Q8Block q8;
    mxfp4_load_q8(token_input + k, q8);
    uint4 codes;
    uint8_t scale;
    mxfp4_load_block<REPACKED>(expert_base, nblocks, gate_base + k, codes,
                               scale);
    gate += mxfp4_block_dot_regs(codes, scale, q8);
    mxfp4_load_block<REPACKED>(expert_base, nblocks, up_base + k, codes,
                               scale);
    up += mxfp4_block_dot_regs(codes, scale, q8);
  }
}

// Decode W1: grid (intermediate/32, tokens*top_k), 256 threads = 32 rows x 8
// lanes. Each 8-lane group owns one intermediate row (its gate and up rows),
// applies SwiGLU and the route weight, and lane groups then cooperate on the
// 32-wide Q8_1 output block. Mirrors iq2_xxs_gate_up_swiglu_q8_1_decode.
template <int TOP_K, bool REPACKED>
__global__ __launch_bounds__(256, 2) void mxfp4_gate_up_swiglu_q8_1_decode(
    const void* __restrict__ weights, const block_q8_1* __restrict__ input,
    block_q8_1* __restrict__ output, const int* __restrict__ topk_ids,
    const float* __restrict__ route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int input_token_blocks, const int intermediate, const int tokens,
    const int experts, const float swiglu_limit) {
  const int lane8 = threadIdx.x & 7;
  const int row_lane = threadIdx.x >> 3;
  const int route = blockIdx.y;
  const int token = route / TOP_K;
  const int row = blockIdx.x * 32 + row_lane;
  const int expert = route < tokens * TOP_K ? topk_ids[route] : -1;
  const int blocks_per_mid = intermediate / QK8_1;
  const int64_t nblocks = expert_stride_bytes / 17;

  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < experts && row < intermediate) {
    const char* expert_base = reinterpret_cast<const char*>(weights) +
                              int64_t(expert) * expert_stride_bytes;
    mxfp4_gate_up_row_dot<REPACKED>(expert_base, input, nblocks, hidden,
                                    input_token_blocks, intermediate, row,
                                    token, lane8, gate, up);
  }
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    gate += __shfl_down_sync(0xffffffffu, gate, mask);
    up += __shfl_down_sync(0xffffffffu, up, mask);
  }

  __shared__ float values[32];
  if (lane8 == 0) {
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

// Down: out[token, j] = sum_r dot(W2[e_r, j, :], mid_q8[route_r, :]).  The
// route weight was already folded into the Q8_1 mid activation by the W1
// kernel, so the down pass is an unweighted accumulation over the token's
// routes. Grid (out_row/32, tokens), 256 threads = 32 rows x 8 lanes.
template <typename scalar_t, int TOP_K, bool REPACKED>
__global__ __launch_bounds__(256, 2) void mxfp4_down_sum(
    const void* __restrict__ weights, const block_q8_1* __restrict__ mid,
    const int* __restrict__ topk_ids, scalar_t* __restrict__ out,
    const int64_t expert_stride_bytes, const int intermediate,
    const int out_row, const int tokens, const int experts) {
  const int lane8 = threadIdx.x & 7;
  const int row_lane = threadIdx.x >> 3;
  const int token = blockIdx.y;
  const int row = blockIdx.x * 32 + row_lane;
  const int blocks_per_row = intermediate / 32;
  const int mid_blocks_per_route = intermediate / QK8_1;
  const int64_t nblocks = expert_stride_bytes / 17;
  if (row >= out_row) return;

  float acc = 0.0f;
#pragma unroll 1
  for (int r = 0; r < TOP_K; ++r) {
    const int route = token * TOP_K + r;
    const int expert = topk_ids[route];
    if (expert < 0 || expert >= experts) continue;
    const char* expert_base = reinterpret_cast<const char*>(weights) +
                              int64_t(expert) * expert_stride_bytes;
    const block_q8_1* route_mid =
        mid + int64_t(route) * mid_blocks_per_route;
    const int64_t row_base = int64_t(row) * blocks_per_row;
    for (int k = lane8; k < blocks_per_row; k += 8) {
      uint4 codes;
      uint8_t scale;
      mxfp4_load_block<REPACKED>(expert_base, nblocks, row_base + k, codes,
                                 scale);
      acc += mxfp4_block_dot(codes, scale, route_mid + k);
    }
  }
#pragma unroll
  for (int mask = 4; mask > 0; mask >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, mask);
  }
  if (lane8 == 0) {
    out[int64_t(token) * out_row + row] = scalar_t(acc);
  }
}

// Byte-neutral AoS(17) -> SoA(scales | 16B codes) repack. One thread per
// block; reads raw, writes split. Output has the same shape/stride as input.
static __global__ void repack_mxfp4_experts(
    const uint8_t* __restrict__ raw, uint8_t* __restrict__ packed,
    const int64_t nblocks, const int64_t expert_stride_bytes) {
  const int expert = blockIdx.y;
  const int64_t block =
      int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (block >= nblocks) return;
  const uint8_t* src =
      raw + int64_t(expert) * expert_stride_bytes + block * 17;
  uint8_t* dst = packed + int64_t(expert) * expert_stride_bytes;
  dst[block] = src[0];
  uint4 codes;
  memcpy(&codes, src + 1, 16);
  reinterpret_cast<uint4*>(dst + nblocks)[block] = codes;
}

inline void launch_repack_mxfp4_experts(const void* raw, void* packed,
                                        const int experts,
                                        const int64_t nblocks,
                                        const int64_t expert_stride_bytes,
                                        cudaStream_t stream) {
  const dim3 grid((unsigned)((nblocks + 255) / 256), experts);
  repack_mxfp4_experts<<<grid, 256, 0, stream>>>(
      static_cast<const uint8_t*>(raw), static_cast<uint8_t*>(packed),
      nblocks, expert_stride_bytes);
}

inline void launch_mxfp4_gate_up_swiglu_q8_1_decode(
    const void* weights, const void* input, void* output,
    const int* topk_ids, const float* route_weights,
    const int64_t expert_stride_bytes, const int hidden,
    const int input_token_blocks, const int intermediate, const int tokens,
    const int top_k, const int experts, const float swiglu_limit,
    const bool repacked, cudaStream_t stream) {
  const dim3 grid(intermediate / 32, tokens * top_k);
  const auto in = static_cast<const block_q8_1*>(input);
  const auto out = static_cast<block_q8_1*>(output);
#define LAUNCH_MXFP4_W1(TOPK, REPACKED)                                     \
  mxfp4_gate_up_swiglu_q8_1_decode<TOPK, REPACKED>                          \
      <<<grid, 256, 0, stream>>>(weights, in, out, topk_ids, route_weights, \
                                 expert_stride_bytes, hidden,               \
                                 input_token_blocks, intermediate, tokens,  \
                                 experts, swiglu_limit)
  if (top_k == 6) {
    if (repacked) LAUNCH_MXFP4_W1(6, true);
    else LAUNCH_MXFP4_W1(6, false);
  } else {
    if (repacked) LAUNCH_MXFP4_W1(8, true);
    else LAUNCH_MXFP4_W1(8, false);
  }
#undef LAUNCH_MXFP4_W1
}

template <typename scalar_t>
inline void launch_mxfp4_down_sum(
    const void* weights, const void* mid, const int* topk_ids, scalar_t* out,
    const int64_t expert_stride_bytes, const int intermediate,
    const int out_row, const int tokens, const int top_k, const int experts,
    const bool repacked, cudaStream_t stream) {
  const dim3 grid((out_row + 31) / 32, tokens);
  const auto mid_blocks = static_cast<const block_q8_1*>(mid);
#define LAUNCH_MXFP4_W2(TOPK, REPACKED)                                    \
  mxfp4_down_sum<scalar_t, TOPK, REPACKED>                                 \
      <<<grid, 256, 0, stream>>>(weights, mid_blocks, topk_ids, out,       \
                                 expert_stride_bytes, intermediate,        \
                                 out_row, tokens, experts)
  if (top_k == 6) {
    if (repacked) LAUNCH_MXFP4_W2(6, true);
    else LAUNCH_MXFP4_W2(6, false);
  } else {
    if (repacked) LAUNCH_MXFP4_W2(8, true);
    else LAUNCH_MXFP4_W2(8, false);
  }
#undef LAUNCH_MXFP4_W2
}

}  // namespace slimserve::dsv4_ampere

#endif  // USE_ROCM
