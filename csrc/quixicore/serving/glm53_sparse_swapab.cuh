// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// SM120 GLM53 BF16 sparse MLA candidate. Mechanism precedent: FlashInfer
// #4751/#4802 candidates-on-M / register-resident Q, adapted to our BF16 KV
// and FP16 probability/value product. No FP8 conversions or cache changes.
namespace slimserve::glm53_swapab {
using BF = __nv_bfloat16;
constexpr int HEADS = 32, DIM = 512, KEYS = 32, THREADS = 160;
// A 512-element row aliases every QK row onto the same four shared banks.
// Rotate each row by four banks, preserving the bulk copy's 16-byte alignment.
constexpr int SHARED_DIM = DIM + 8;

struct alignas(128) Shared {
  BF kv[2][KEYS][SHARED_DIM];
  int valid[2][KEYS];
  half prob[4][KEYS][8];
  alignas(8) unsigned long long ready[2], released[2];
};

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return unsigned(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void init_bar(unsigned long long* p, unsigned count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::
               "r"(smem_addr(p)), "r"(count) : "memory");
}
__device__ __forceinline__ void wait_bar(unsigned long long* p, unsigned phase) {
  asm volatile("{ .reg .pred done; wait_loop: "
               "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1; "
               "@!done bra wait_loop; }" ::
               "r"(smem_addr(p)), "r"(phase) : "memory");
}
__device__ __forceinline__ void release_bar(unsigned long long* p) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::
               "r"(smem_addr(p)) : "memory");
}
__device__ __forceinline__ void expect_bytes(unsigned long long* p) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::
               "r"(smem_addr(p)), "r"(KEYS * DIM * 2) : "memory");
}
__device__ __forceinline__ void copy_row(BF* dst, const BF* src,
                                          unsigned long long* bar) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
               "[%0], [%1], 1024, [%2];" ::
               "r"(smem_addr(dst)), "l"(src), "r"(smem_addr(bar)) : "memory");
}

template <bool BF16>
__device__ __forceinline__ void mma(unsigned a0, unsigned a1, unsigned a2,
                                    unsigned a3, unsigned b0, unsigned b1, float* c) {
#define MMA(SCALAR) asm volatile( \
    "mma.sync.aligned.m16n8k16.row.col.f32." SCALAR "." SCALAR ".f32 " \
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" \
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) \
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1))
  if constexpr (BF16) { MMA("bf16"); } else { MMA("f16"); }
#undef MMA
}

__device__ __forceinline__ unsigned pack_half(float lo, float hi) {
  half2 value = __floats2half2_rn(lo, hi);
  return reinterpret_cast<unsigned&>(value);
}
__device__ __forceinline__ unsigned load_pair(const BF* p) {
  return *reinterpret_cast<const unsigned*>(p);
}
__device__ __forceinline__ unsigned bf_pair_to_half(unsigned value) {
  const auto floats = __bfloat1622float2(
      *reinterpret_cast<const __nv_bfloat162*>(&value));
  return pack_half(floats.x, floats.y);
}
__device__ __forceinline__ void load_value_matrix(const BF* address, unsigned* a) {
  // The four 8x8 matrices form V^T's 16x16 MMA A fragment. Convert after
  // the collective transpose, retaining the existing FP16 value boundary.
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
               "{%0,%1,%2,%3}, [%4];" :
               "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) :
               "r"(smem_addr(address)));
#pragma unroll
  for (int i = 0; i < 4; i++) a[i] = bf_pair_to_half(a[i]);
}
__device__ __forceinline__ float warp8_sum(float value) {
  value += __shfl_xor_sync(0xffffffff, value, 4);
  value += __shfl_xor_sync(0xffffffff, value, 8);
  value += __shfl_xor_sync(0xffffffff, value, 16);
  return value;
}
__device__ __forceinline__ float warp8_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 4));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 8));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 16));
  return value;
}

__global__ __launch_bounds__(THREADS, 1) void sparse_nope(
    const BF* __restrict__ query, const BF* __restrict__ cache,
    const int* __restrict__ tables, const int* __restrict__ indices,
    const int* __restrict__ lengths, BF* __restrict__ output,
    int topk, int table_stride, int block_size, int64_t page_stride, float scale) {
  extern __shared__ __align__(128) unsigned char storage[];
  auto& sm = *reinterpret_cast<Shared*>(storage);
  const int token = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const int length = min(max(lengths[token], 0), topk);
  const int tiles = (length + KEYS - 1) / KEYS;
  if (threadIdx.x == 0) {
    init_bar(&sm.ready[0], 1); init_bar(&sm.ready[1], 1);
    init_bar(&sm.released[0], 128); init_bar(&sm.released[1], 128);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  if (warp == 4) {
    // One IO lane owns each selected latent. Real request-local pages and
    // strides are preserved; invalid slots copy page0 but remain masked.
    for (int tile = 0; tile < tiles; tile++) {
      const int buf = tile & 1;
      wait_bar(&sm.released[buf], 1 ^ ((tile / 2) & 1));
      const int pos = tile * KEYS + lane;
      const int selected = pos < length ? indices[int64_t(token) * topk + pos] : -1;
      const int col = selected >= 0 ? selected / block_size : -1;
      const int block = col >= 0 && col < table_stride
                            ? tables[int64_t(token) * table_stride + col] : -1;
      sm.valid[buf][lane] = block >= 0;
      const int64_t address = block >= 0
          ? int64_t(block) * page_stride + int64_t(selected % block_size) * DIM : 0;
      __syncwarp();
      if (lane == 0) expect_bytes(&sm.ready[buf]);
      __syncwarp();
      copy_row(sm.kv[buf][lane], cache + address, &sm.ready[buf]);
    }
    return;
  }

  const int gid = lane / 4, tid = lane % 4;
  const BF* q = query + (int64_t(token) * HEADS + warp * 8 + gid) * DIM;
  unsigned q_frag[DIM / 16][2];
#pragma unroll
  for (int k = 0; k < DIM / 16; k++) {
    q_frag[k][0] = load_pair(q + k * 16 + tid * 2);
    q_frag[k][1] = load_pair(q + k * 16 + tid * 2 + 8);
  }
  float acc[DIM / 16][4] = {};
  float maxima[2] = {-1e30f, -1e30f}, sums[2] = {};
  for (int tile = 0; tile < tiles; tile++) {
    const int buf = tile & 1;
    wait_bar(&sm.ready[buf], (tile / 2) & 1);
    float score[2][4] = {};
#pragma unroll
    for (int k = 0; k < DIM / 16; k++) {
#pragma unroll
      for (int m = 0; m < 2; m++) {
        const BF* r0 = sm.kv[buf][m * 16 + gid] + k * 16 + tid * 2;
        const BF* r1 = r0 + 8 * SHARED_DIM;
        mma<true>(load_pair(r0), load_pair(r1), load_pair(r0 + 8),
                  load_pair(r1 + 8), q_frag[k][0], q_frag[k][1], score[m]);
      }
    }
    float alpha[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
      float maximum = -1e30f;
#pragma unroll
      for (int m = 0; m < 2; m++) {
        score[m][h] = sm.valid[buf][m * 16 + gid] ? score[m][h] * scale : -INFINITY;
        score[m][2 + h] = sm.valid[buf][m * 16 + gid + 8]
                             ? score[m][2 + h] * scale : -INFINITY;
        maximum = fmaxf(maximum, fmaxf(score[m][h], score[m][2 + h]));
      }
      maximum = fmaxf(maxima[h], warp8_max(maximum));
      alpha[h] = __expf(maxima[h] - maximum);
      maxima[h] = maximum;
      float subtotal = 0;
#pragma unroll
      for (int m = 0; m < 2; m++) {
        score[m][h] = __expf(score[m][h] - maximum);
        score[m][2 + h] = __expf(score[m][2 + h] - maximum);
        subtotal += score[m][h] + score[m][2 + h];
        sm.prob[warp][m * 16 + gid][tid * 2 + h] = __float2half_rn(score[m][h]);
        sm.prob[warp][m * 16 + gid + 8][tid * 2 + h] = __float2half_rn(score[m][2 + h]);
      }
      sums[h] = sums[h] * alpha[h] + warp8_sum(subtotal);
    }
    __syncwarp();
    unsigned p_frag[2][2];
#pragma unroll
    for (int k = 0; k < 2; k++) {
      p_frag[k][0] = pack_half(__half2float(sm.prob[warp][k * 16 + tid * 2][gid]),
                              __half2float(sm.prob[warp][k * 16 + tid * 2 + 1][gid]));
      p_frag[k][1] = pack_half(__half2float(sm.prob[warp][k * 16 + tid * 2 + 8][gid]),
                              __half2float(sm.prob[warp][k * 16 + tid * 2 + 9][gid]));
    }
#pragma unroll
    for (int m = 0; m < DIM / 16; m++) {
      float pv[4] = {};
#pragma unroll
      for (int k = 0; k < 2; k++) {
        const int r = k * 16 + lane % 8 + (lane / 16) * 8;
        const int d = m * 16 + ((lane / 8) % 2) * 8;
        unsigned values[4];
        load_value_matrix(&sm.kv[buf][r][d], values);
        mma<false>(values[0], values[1], values[2], values[3],
                   p_frag[k][0], p_frag[k][1], pv);
      }
#pragma unroll
      for (int i = 0; i < 4; i++) acc[m][i] = acc[m][i] * alpha[i % 2] + pv[i];
    }
    // Every shared-memory reader participates in release. Do not delegate
    // arrival to lane0: buffer reuse must wait for all 128 math threads.
    release_bar(&sm.released[buf]);
  }
#pragma unroll
  for (int m = 0; m < DIM / 16; m++) {
#pragma unroll
    for (int h = 0; h < 2; h++) {
      const int64_t row = (int64_t(token) * HEADS + warp * 8 + tid * 2 + h) * DIM;
      output[row + m * 16 + gid] = __float2bfloat16_rn(sums[h] > 0 ? acc[m][h] / sums[h] : 0);
      output[row + m * 16 + gid + 8] = __float2bfloat16_rn(sums[h] > 0 ? acc[m][2 + h] / sums[h] : 0);
    }
  }
}
}  // namespace slimserve::glm53_swapab
