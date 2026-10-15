#pragma once

// Included from custom_all_reduce.cuh inside namespace vllm, after the peer
// pointer and synchronization primitives have been defined.
//
// GLM-5.3 mHC transition fused with the tensor-parallel all-reduce of the
// producing projection (o_proj or the MoE output), for decode-sized batches.
// One plain launch replaces the one-shot all-reduce kernel and the split
// Triton pair (per-split partials, then sinkhorn + pre-mix + RMSNorm):
//
//   x            = sum over ranks of the TP partial (rank order, bf16-rounded
//                  like cross_device_reduce_1stage)
//   residual_out = post * x + comb^T residual              (bf16, [T, 4, D])
//   partials     = fn . residual_out, |residual_out|^2     (per block)
//   pre/post/comb = sinkhorn(fn projections, scale, base)  (per token)
//   layer_input  = rmsnorm(pre . residual_out) * weight    (bf16, [T, D])
//
// The grid is NBLOCKS blocks of THREADS threads; block b owns hidden
// dimensions [b * THREADS, (b + 1) * THREADS). Phase one walks the tokens in
// groups (eight tokens, four on eight ranks): the peers' slices, the residual slices and the
// per-token mixes of a group are copied into shared memory with 16-byte
// cp.async requests (the same request size as the one-shot all-reduce; PCIe
// peer reads are priced per request, not per byte), double-buffered so the
// next group's copies fly while this group is reduced. Every thread reduces
// its dimension in rank order, mixes the residual streams and accumulates
// the fn projections with its fn column held in registers; the 25 warp
// sums use a transposed shuffle reduction (31 shuffles, one output per
// lane). A block publishes the group's partial rows and counts one arrival
// per token with a release-add. Phase two spreads the per-token finalize
// over the blocks: block b waits for token b's arrivals (b + NBLOCKS, ...
// for larger batches), loads the residual streams and the partial rows
// together and writes the pre-mix and RMSNorm, so a batch of T tokens costs
// one finalize, not T serialized ones. The next site's comb coefficients
// leave the kernel as raw logits: their sinkhorn (twenty serial
// normalizations, the longest chain of the transition and needed only by
// the next site) runs in `sinkhorn_deferred` on a side stream that the
// launcher forks after this kernel and joins before the next consumer, the
// schedule the DSV4 fused all-reduce already uses. Peer input buffers are
// read after the block-indexed start barrier and released by the end
// barrier, exactly like the one-shot all-reduce.
namespace glm5_mhc_ar {

constexpr int HIDDEN = 4096;
constexpr int HC = 4;
constexpr int NOUT = 24;            // fn rows: 4 pre, 4 post, 16 comb
constexpr int PARTIALS = NOUT + 1;  // + the square sum
constexpr int PARTIAL_STRIDE = 32;  // floats per partial row
// PARTIAL_STRIDE, NBLOCKS and the lanes of a warp are all 32 for this
// geometry (HIDDEN 4096 over 128 threads) by coincidence, not by
// construction: the finalize loop's BLOCKS_PER_WARP and the static_asserts
// below are what tie them together.
constexpr int THREADS = 128;
constexpr int NBLOCKS = HIDDEN / THREADS;
constexpr int VEC = 8;                          // bf16 per 16-byte vector
constexpr int VECS_PER_BLOCK = THREADS / VEC;   // vectors per token slice
constexpr int FIN_VECS = HIDDEN / THREADS / VEC;  // finalize vectors per thread
constexpr int MAX_TOKENS = 64;
constexpr int MIX_FLOATS = HC + HC * HC;  // post then comb, per token
constexpr int MIX_VECS = MIX_FLOATS / 4;  // 16-byte vectors per token
constexpr int DEFERRED_THREADS = MAX_TOKENS * HC;  // one lane per comb row
// The transition's serial cost is its per-block token loop, and NBLOCKS
// dimension blocks occupy 32 of this card's 188 SMs. Above one token the grid
// therefore carries a second, flat axis: TOKEN_CHUNKS copies of the NBLOCKS
// dimension blocks, chunk c owning a contiguous token range. Every token is
// still covered by exactly NBLOCKS blocks, so `arrivals` and the `partial`
// row layout are unchanged. The flat block index addresses one signal slot
// per block, so the product must stay inside kMaxBlocks.
constexpr int TOKEN_CHUNKS = 8;
static_assert(NBLOCKS * TOKEN_CHUNKS <= kMhcMaxBlocks, "one signal slot per block");

// Chunks for a launch. Splitting the tokens doubles the `fn` column read
// (each chunk's blocks load their own dimension's column), which is a fixed
// 1.5 MB per extra chunk; below ~8 tokens the latency it removes is worth
// more than those bytes, and above it the trade inverts - chunking every
// width measured on ws4 as -13 % of the transition arithmetic at T=4 and
// -17 % at T=8, but +13 % at T=16 and +10 % at T=64. Gated as below, the
// shipped numbers are -9 % at T=4 and -13 % at T=8 with T=16 and T=64 back
// on the unchunked baseline. The fuse policy never sends more than
// GLM5_MHC_FUSE_TOKENS here anyway; this keeps the kernel monotone if it
// ever does. Never more chunks than tokens, so no launched block owns an
// empty range and T = 1 keeps today's 32-block grid.
constexpr int TOKEN_CHUNK_MAX_TOKENS = 8;
inline int token_chunks(int num_tokens) {
  // Up to TOKEN_CHUNKS copies of the dimension grid, so a block walks at
  // most MAX_TOKENS / TOKEN_CHUNKS tokens serially (2026-09-17: at 16 tokens
  // the one-chunk grid spent ~10 us in that walk).
  if (num_tokens < 2) return 1;
  return num_tokens < TOKEN_CHUNKS ? num_tokens : TOKEN_CHUNKS;
}
static_assert(THREADS % 32 == 0 && PARTIALS <= 32 && NBLOCKS % (THREADS / 32) == 0);
static_assert(HIDDEN % (THREADS * VEC) == 0, "whole vectors per thread");
static_assert(MIX_FLOATS % 4 == 0 && 32 % HC == 0 && DEFERRED_THREADS <= 1024);

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

// Transposed warp reduction: 32 values per lane in, lane l leaves with the
// warp's sum of value l (31 shuffles instead of 5 per value).
template <int OFFSET>
__device__ __forceinline__ void transpose_step(float (&values)[32], int lane) {
  const bool upper = (lane & OFFSET) != 0;
#pragma unroll
  for (int i = 0; i < OFFSET; ++i) {
    const float send = upper ? values[i] : values[i + OFFSET];
    const float keep = upper ? values[i + OFFSET] : values[i];
    values[i] = keep + __shfl_xor_sync(0xffffffffu, send, OFFSET);
  }
}

__device__ __forceinline__ float transpose_reduce(float (&values)[32],
                                                  int lane) {
  transpose_step<16>(values, lane);
  transpose_step<8>(values, lane);
  transpose_step<4>(values, lane);
  transpose_step<2>(values, lane);
  transpose_step<1>(values, lane);
  return values[0];
}

__device__ __forceinline__ float sigmoid(float value) {
  return 1.0f / (1.0f + expf(-value));
}

__device__ __forceinline__ unsigned int ld_acquire_gpu(
    const unsigned int* ptr) {
  unsigned int value;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(ptr)
               : "memory");
  return value;
}

// One release-add per block and token: the writes every thread of the block
// made before the preceding __syncthreads are published with it (release
// cumulativity through the barrier, the pattern the all-reduce barriers use).
__device__ __forceinline__ void red_release_gpu_add(unsigned int* ptr,
                                                    unsigned int value) {
  asm volatile("red.release.gpu.global.add.u32 [%0], %1;" ::"l"(ptr),
               "r"(value)
               : "memory");
}

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
  const unsigned int dst =
      static_cast<unsigned int>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
               "l"(gmem)
               : "memory");
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;" ::: "memory");
}

template <int PENDING>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(PENDING) : "memory");
}

// Sinkhorn on four lanes (`mask`), lane r of the group holding row r in
// registers: row softmax + eps, then alternating column / row normalization
// with eps in every denominator, the same sequence as the split Triton
// finalize. Row normalizations are lane-local; the column sums cross the
// group with two xor shuffles per column. Exponentials and reciprocals are
// the approximate instructions Triton lowers to.
__device__ __forceinline__ void sinkhorn_rows(float (&row)[HC], float hc_eps,
                                              int sinkhorn_repeat,
                                              unsigned int mask) {
  const float row_max = fmaxf(fmaxf(row[0], row[1]), fmaxf(row[2], row[3]));
  float row_sum = 0.0f;
#pragma unroll
  for (int col = 0; col < HC; ++col) {
    row[col] = __expf(row[col] - row_max);
    row_sum += row[col];
  }
  const float inverse = __fdividef(1.0f, row_sum);
#pragma unroll
  for (int col = 0; col < HC; ++col) row[col] = row[col] * inverse + hc_eps;
  for (int iteration = 0; iteration < sinkhorn_repeat; ++iteration) {
    if (iteration > 0) {
      float sum = hc_eps;
#pragma unroll
      for (int col = 0; col < HC; ++col) sum += row[col];
      const float row_inverse = __fdividef(1.0f, sum);
#pragma unroll
      for (int col = 0; col < HC; ++col) row[col] *= row_inverse;
    }
    float column_sum[HC];
#pragma unroll
    for (int col = 0; col < HC; ++col) {
      float sum = row[col];
      sum += __shfl_xor_sync(mask, sum, 1);
      sum += __shfl_xor_sync(mask, sum, 2);
      column_sum[col] = sum;
    }
#pragma unroll
    for (int col = 0; col < HC; ++col) {
      row[col] *= __fdividef(1.0f, hc_eps + column_sum[col]);
    }
  }
}

// The deferred half of the transition: the raw comb logits the fused kernel
// left in `comb` ([num_tokens, 4, 4]) become the next site's coefficients in
// place. One block, four lanes per token.
__global__ void __launch_bounds__(DEFERRED_THREADS) sinkhorn_deferred(
    float* comb, int num_tokens, float hc_eps, int sinkhorn_repeat) {
  const int token = threadIdx.x / HC;
  const int row_index = threadIdx.x - token * HC;
  if (token >= num_tokens) return;
  const unsigned int mask = 0xfu << ((threadIdx.x & 31) & ~(HC - 1));
  float4* rows = reinterpret_cast<float4*>(comb + token * HC * HC);
  const float4 loaded = rows[row_index];
  float row[HC] = {loaded.x, loaded.y, loaded.z, loaded.w};
  sinkhorn_rows(row, hc_eps, sinkhorn_repeat, mask);
  rows[row_index] = make_float4(row[0], row[1], row[2], row[3]);
}

// Eight bf16 of a 16-byte vector as floats.
__device__ __forceinline__ void unpack_bf16x8(const uint4 packed,
                                              float* values) {
  const nv_bfloat162* pairs = reinterpret_cast<const nv_bfloat162*>(&packed);
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i) {
    const float2 f = __bfloat1622float2(pairs[i]);
    values[2 * i] = f.x;
    values[2 * i + 1] = f.y;
  }
}

__device__ __forceinline__ uint4 pack_bf16x8(const float* values) {
  uint4 packed;
  nv_bfloat162* pairs = reinterpret_cast<nv_bfloat162*>(&packed);
#pragma unroll
  for (int i = 0; i < VEC / 2; ++i) {
    pairs[i] = __floats2bfloat162_rn(values[2 * i], values[2 * i + 1]);
  }
  return packed;
}

// Middle barrier of the reduce-scatter variant: releases this rank's stage-1
// writes (its reduced slice in the IPC scratch) and acquires the peers'. It
// uses the `end` slots like the two-stage all-reduce's inter-stage barrier,
// and its wait is monotonic (>=): a fast peer may already have advanced the
// slot with the site's final barrier before this rank reads it, and the
// final barrier's own wait (!=) is safe because nothing writes the slot
// again until every rank has passed it.
// The transition's barriers live on the Signal's mhc_* slots (kMhcMaxBlocks
// of them) so its grid may exceed the plain all-reduce's block cap. Same
// protocol as barrier_at_start / barrier_at_end: a per-block counter, each
// rank writes the expected value into every peer's slot for this block and
// waits for its own.
template <int NGPU>
__device__ __forceinline__ void mhc_barrier_start(const RankSignals& sg,
                                                  Signal* self_sg, int rank) {
  const uint32_t flag = self_sg->mhc_flag[blockIdx.x] + 1;
  if (threadIdx.x < NGPU) {
    st_flag_volatile(&sg.signals[threadIdx.x]->mhc_start[blockIdx.x][rank],
                     flag);
    while (ld_flag_volatile(&self_sg->mhc_start[blockIdx.x][threadIdx.x]) !=
           flag) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->mhc_flag[blockIdx.x] = flag;
}
template <int NGPU>
__device__ __forceinline__ void barrier_mid(const RankSignals& sg,
                                            Signal* self_sg, int rank) {
  __syncthreads();
  const uint32_t flag = self_sg->mhc_flag[blockIdx.x] + 1;
  if (threadIdx.x < NGPU) {
    st_flag_release(&sg.signals[threadIdx.x]->mhc_end[blockIdx.x][rank], flag);
    while (ld_flag_acquire(&self_sg->mhc_end[blockIdx.x][threadIdx.x]) <
           flag) {
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) self_sg->mhc_flag[blockIdx.x] = flag;
}
template <int NGPU>
__device__ __forceinline__ void mhc_barrier_end(const RankSignals& sg,
                                                Signal* self_sg, int rank) {
  __syncthreads();
  const uint32_t flag = self_sg->mhc_flag[blockIdx.x] + 1;
  if (threadIdx.x < NGPU) {
    st_flag_volatile(&sg.signals[threadIdx.x]->mhc_end[blockIdx.x][rank],
                     flag);
    while (ld_flag_volatile(&self_sg->mhc_end[blockIdx.x][threadIdx.x]) !=
           flag) {
    }
  }
  if (threadIdx.x == 0) self_sg->mhc_flag[blockIdx.x] = flag;
}

// RS: exchange by reduce-scatter + all-gather instead of one-shot peer reads.
// Stage 1 reduces this rank's HIDDEN / NGPU dimensions for every token from
// the NGPU inputs (rank order, fp32, rounded to bf16 like the one-shot path)
// into the rank's IPC scratch; after the middle barrier the transition reads
// each dimension block's reduced slice from its owner. PCIe bytes per rank
// fall from (NGPU - 1) x T x HIDDEN x 2 to 2 x (NGPU - 1) / NGPU of that,
// which is what makes the fused site cheaper than the split pair above
// eight tokens (2026-09-17, Phase 8 / P3).
template <int NGPU, bool FUSED_NORM, bool RS = false>
__global__ void __launch_bounds__(THREADS) allreduce_transition(
    RankData* rank_data, RankSignals signals, Signal* self_signal,
    const nv_bfloat16* residual, const float* post_mix, const float* comb_mix,
    const float* fn, nv_bfloat16* residual_out, float* partial,
    unsigned int* arrivals, const float* scale, const float* base,
    float* next_post, float* next_comb, nv_bfloat16* layer_input,
    const nv_bfloat16* norm_weight, float rms_eps, float hc_eps,
    float post_multiplier, int sinkhorn_repeat, float norm_eps, int rank,
    int num_tokens) {
  constexpr int WARPS = THREADS / 32;
  // Tokens staged per round; the double-buffered peer stage must stay under
  // the static shared-memory limit on eight ranks.
  constexpr int TOKEN_GROUP = NGPU > 4 ? 4 : 8;
  constexpr int GROUP_VECS = TOKEN_GROUP * NGPU * VECS_PER_BLOCK;
  constexpr int RES_VECS = TOKEN_GROUP * HC * VECS_PER_BLOCK;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  // Flat grid: `chunks` copies of the NBLOCKS dimension blocks.
  const int chunks = gridDim.x / NBLOCKS;
  const int chunk = blockIdx.x / NBLOCKS;
  const int dim_block = blockIdx.x - chunk * NBLOCKS;
  const int dim0 = dim_block * THREADS;
  const int dim = dim0 + tid;
  const RankData peers = *rank_data;

  // Double-buffered group staging: peers' slices [peer][token][vector],
  // residual slices [token][stream][vector], mixes [token][post | comb].
  __shared__ __align__(16) uint4 stage[2][GROUP_VECS];
  __shared__ __align__(16) uint4 res_stage[2][RES_VECS];
  __shared__ __align__(16) float mix_stage[2][TOKEN_GROUP * MIX_FLOATS];
  __shared__ float warp_partials[TOKEN_GROUP][WARPS][PARTIALS];
  __shared__ float mixes[PARTIALS];
  __shared__ float pre[HC];
  __shared__ float norm_partials[WARPS];

  // This dimension's fn column, [stream][output], reused for every token.
  float fn_col[HC][NOUT];
#pragma unroll
  for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
    for (int output = 0; output < NOUT; ++output) {
      fn_col[stream][output] = fn[(output * HC + stream) * HIDDEN + dim];
    }
  }

  // Issue one group's copies into buffer `buf` (one commit group).
  auto issue_group = [&](int token0, int group, int buf) {
    if constexpr (RS) {
      // All-gather: this dimension block's reduced slice from its owner.
      const int owner = dim0 / (HIDDEN / NGPU);
      const uint4* source =
          reinterpret_cast<const uint4*>(
              reinterpret_cast<Signal*>(signals.signals[owner]) + 1) +
          (size_t(token0) * (HIDDEN / NGPU) + (dim0 - owner * (HIDDEN / NGPU))) /
              VEC;
      for (int i = tid; i < group * VECS_PER_BLOCK; i += THREADS) {
        const int token = i / VECS_PER_BLOCK;
        const int vec = i - token * VECS_PER_BLOCK;
        cp_async_16(&stage[buf][token * VECS_PER_BLOCK + vec],
                    source + token * (HIDDEN / NGPU / VEC) + vec);
      }
    } else {
#pragma unroll
    for (int peer = 0; peer < NGPU; ++peer) {
      const uint4* source = reinterpret_cast<const uint4*>(peers.ptrs[peer]) +
                            (size_t(token0) * HIDDEN + dim0) / VEC;
      for (int i = tid; i < group * VECS_PER_BLOCK; i += THREADS) {
        const int token = i / VECS_PER_BLOCK;
        const int vec = i - token * VECS_PER_BLOCK;
        cp_async_16(
            &stage[buf][(peer * TOKEN_GROUP + token) * VECS_PER_BLOCK + vec],
            source + token * (HIDDEN / VEC) + vec);
      }
    }
    }
    for (int i = tid; i < group * HC * VECS_PER_BLOCK; i += THREADS) {
      const int token = i / (HC * VECS_PER_BLOCK);
      const int rest = i - token * (HC * VECS_PER_BLOCK);
      const int stream = rest / VECS_PER_BLOCK;
      const int vec = rest - stream * VECS_PER_BLOCK;
      cp_async_16(&res_stage[buf][i],
                  reinterpret_cast<const uint4*>(
                      residual + ((token0 + token) * HC + stream) * HIDDEN +
                      dim0) +
                      vec);
    }
    if (tid < group * MIX_VECS) {
      const int token = tid / MIX_VECS;
      const int vec = tid - token * MIX_VECS;
      const float* source =
          vec == 0 ? post_mix + (token0 + token) * HC
                   : comb_mix + (token0 + token) * HC * HC + (vec - 1) * 4;
      cp_async_16(&mix_stage[buf][token * MIX_FLOATS + vec * 4], source);
    }
    cp_async_commit();
  };

  // The GEMM behind this site may start pulling its weights into L2 now.
  pdl_launch_dependents();
  mhc_barrier_start<NGPU>(signals, self_signal, rank);

  constexpr int OWN = HIDDEN / NGPU;   // dimensions a rank reduces
  constexpr int OWN_VECS = OWN / VEC;  // 16-byte vectors per token per rank
  uint4* tmps[NGPU];
#pragma unroll
  for (int p = 0; p < NGPU; ++p) {
    tmps[p] = reinterpret_cast<uint4*>(
        reinterpret_cast<Signal*>(signals.signals[p]) + 1);
  }
  if constexpr (RS) {
    // Stage 1: reduce-scatter over the whole grid.
    const int total = num_tokens * OWN_VECS;
    for (int i = blockIdx.x * THREADS + tid; i < total;
         i += gridDim.x * THREADS) {
      const int token = i / OWN_VECS;
      const int vec = i - token * OWN_VECS;
      const size_t off = (size_t(token) * HIDDEN + rank * OWN) / VEC + vec;
      float acc[VEC];
      {
        const uint4 v = reinterpret_cast<const uint4*>(peers.ptrs[0])[off];
        unpack_bf16x8(v, acc);
      }
#pragma unroll
      for (int p = 1; p < NGPU; ++p) {
        const uint4 v = reinterpret_cast<const uint4*>(peers.ptrs[p])[off];
        float e[VEC];
        unpack_bf16x8(v, e);
#pragma unroll
        for (int k = 0; k < VEC; ++k) acc[k] += e[k];
      }
      tmps[rank][size_t(token) * OWN_VECS + vec] = pack_bf16x8(acc);
    }
    barrier_mid<NGPU>(signals, self_signal, rank);
  }

  // This chunk's contiguous token range.
  const int per_chunk = (num_tokens + chunks - 1) / chunks;
  const int tok_begin = min(chunk * per_chunk, num_tokens);
  const int tok_end = min(tok_begin + per_chunk, num_tokens);
  const int my_tokens = tok_end - tok_begin;

  const int groups = (my_tokens + TOKEN_GROUP - 1) / TOKEN_GROUP;
  issue_group(tok_begin, min(TOKEN_GROUP, my_tokens), 0);
  for (int g = 0; g < groups; ++g) {
    const int token0 = tok_begin + g * TOKEN_GROUP;
    const int group = min(TOKEN_GROUP, tok_end - token0);
    const int buf = g & 1;
    if (g + 1 < groups) {
      // The other buffer was last read before the previous group's barriers.
      issue_group(token0 + TOKEN_GROUP,
                  min(TOKEN_GROUP, tok_end - token0 - TOKEN_GROUP),
                  buf ^ 1);
      cp_async_wait<1>();
    } else {
      cp_async_wait<0>();
    }
    __syncthreads();
    const nv_bfloat16* staged =
        reinterpret_cast<const nv_bfloat16*>(stage[buf]);
    const nv_bfloat16* res_staged =
        reinterpret_cast<const nv_bfloat16*>(res_stage[buf]);
    const float* mix_staged = mix_stage[buf];

    for (int t = 0; t < group; ++t) {
      const int token = token0 + t;
      // Rank-ordered reduction (identical on every rank), rounded to bf16
      // like the one-shot all-reduce output the split path consumed.
      float xr;
      if constexpr (RS) {
        xr = __bfloat162float(staged[t * THREADS + tid]);
      } else {
        float x = 0.0f;
#pragma unroll
        for (int peer = 0; peer < NGPU; ++peer) {
          x += __bfloat162float(
              staged[(peer * TOKEN_GROUP + t) * THREADS + tid]);
        }
        xr = __bfloat162float(__float2bfloat16_rn(x));
      }

      float res[HC];
#pragma unroll
      for (int stream = 0; stream < HC; ++stream) {
        res[stream] =
            __bfloat162float(res_staged[(t * HC + stream) * THREADS + tid]);
      }
      const float* mix = mix_staged + t * MIX_FLOATS;

      float values[32];
#pragma unroll
      for (int i = 0; i < 32; ++i) values[i] = 0.0f;
#pragma unroll
      for (int stream = 0; stream < HC; ++stream) {
        // Same multiply / FMA order as the split kernels: source stream 0 is
        // the plain product, the rest accumulate with FMAs.
        float value = fmaf(mix[stream], xr, mix[HC + stream] * res[0]);
#pragma unroll
        for (int source = 1; source < HC; ++source) {
          value = fmaf(mix[HC + source * HC + stream], res[source], value);
        }
        const nv_bfloat16 rounded = __float2bfloat16_rn(value);
        residual_out[(token * HC + stream) * HIDDEN + dim] = rounded;
        const float a = __bfloat162float(rounded);
        values[NOUT] = fmaf(a, a, values[NOUT]);
#pragma unroll
        for (int output = 0; output < NOUT; ++output) {
          values[output] = fmaf(a, fn_col[stream][output], values[output]);
        }
      }
      const float reduced = transpose_reduce(values, lane);
      if (lane < PARTIALS) warp_partials[t][warp][lane] = reduced;
    }
    __syncthreads();
    for (int i = tid; i < group * PARTIALS; i += THREADS) {
      const int t = i / PARTIALS;
      const int output = i - t * PARTIALS;
      float block_sum = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < WARPS; ++source_warp) {
        block_sum += warp_partials[t][source_warp][output];
      }
      partial[((token0 + t) * NBLOCKS + dim_block) * PARTIAL_STRIDE +
              output] = block_sum;
    }
    // Publish this block's residual_out slices and partial rows of the
    // group, then count one arrival per token for the finalizing blocks.
    __syncthreads();
    if (tid < group) red_release_gpu_add(arrivals + token0 + tid, 1u);
  }

  for (int token = blockIdx.x; token < num_tokens; token += gridDim.x) {
    // The norm weight does not depend on the arrivals: load it while
    // thread 0 waits.
    uint4 weight_packed[FIN_VECS];
    if constexpr (FUSED_NORM) {
      const uint4* weight = reinterpret_cast<const uint4*>(norm_weight);
#pragma unroll
      for (int v = 0; v < FIN_VECS; ++v) {
        weight_packed[v] = weight[v * THREADS + tid];
      }
    }
    if (tid == 0) {
      while (ld_acquire_gpu(arrivals + token) < unsigned(NBLOCKS)) {
      }
      // Every increment has landed; reset for the next launch on the stream.
      arrivals[token] = 0u;
    }
    // Thread 0's acquire orders the block's reads below behind every
    // publisher's release (through the barrier); the reads bypass L1.
    __syncthreads();

    // The token's four residual streams (16-byte L2 reads of the slices the
    // other blocks published) go in flight together with the partial rows.
    uint4 streams_packed[HC][FIN_VECS];
    {
      const uint4* rows =
          reinterpret_cast<const uint4*>(residual_out + token * HC * HIDDEN);
#pragma unroll
      for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
        for (int v = 0; v < FIN_VECS; ++v) {
          streams_packed[stream][v] =
              __ldcg(rows + stream * (HIDDEN / VEC) + v * THREADS + tid);
        }
      }
    }

    // Partial rows summed in block order: warp w adds its eight blocks'
    // rows (independent loads, one L2 round trip), then one thread per
    // value adds the four warp sums.
    {
      constexpr int BLOCKS_PER_WARP = NBLOCKS / WARPS;
      float total = 0.0f;
      if (lane < PARTIALS) {
        const float* rows =
            partial +
            (token * NBLOCKS + warp * BLOCKS_PER_WARP) * PARTIAL_STRIDE;
        float loaded[BLOCKS_PER_WARP];
#pragma unroll
        for (int i = 0; i < BLOCKS_PER_WARP; ++i) {
          loaded[i] = __ldcg(rows + i * PARTIAL_STRIDE + lane);
        }
#pragma unroll
        for (int i = 0; i < BLOCKS_PER_WARP; ++i) total += loaded[i];
        warp_partials[0][warp][lane] = total;
      }
    }
    __syncthreads();
    if (tid < PARTIALS) {
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < WARPS; ++source_warp) {
        total += warp_partials[0][source_warp][tid];
      }
      mixes[tid] = total;
    }
    __syncthreads();
    const float inverse_rms =
        rsqrtf(mixes[NOUT] / float(HC * HIDDEN) + rms_eps);
    if (tid < HC) {
      pre[tid] = sigmoid(mixes[tid] * inverse_rms * scale[0] + base[tid]) +
                 hc_eps;
    } else if (tid < 2 * HC) {
      next_post[token * HC + tid - HC] =
          post_multiplier *
          sigmoid(mixes[tid] * inverse_rms * scale[1] + base[tid]);
    } else if (tid < 2 * HC + HC * HC) {
      // Raw comb logits; `sinkhorn_deferred` normalizes them off this path.
      next_comb[token * HC * HC + tid - 2 * HC] =
          mixes[tid] * inverse_rms * scale[2] + base[tid];
    }
    __syncthreads();

    // Pre-mix over the four streams, then the optional RMSNorm. Both bf16
    // rounding boundaries of the split kernels are preserved.
    uint4 mixed_packed[FIN_VECS];
    float norm_sum = 0.0f;
#pragma unroll
    for (int v = 0; v < FIN_VECS; ++v) {
      float mixed[VEC];
      {
        float s0[VEC];
        unpack_bf16x8(streams_packed[0][v], s0);
#pragma unroll
        for (int e = 0; e < VEC; ++e) mixed[e] = pre[0] * s0[e];
      }
#pragma unroll
      for (int stream = 1; stream < HC; ++stream) {
        float s[VEC];
        unpack_bf16x8(streams_packed[stream][v], s);
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          mixed[e] = fmaf(pre[stream], s[e], mixed[e]);
        }
      }
      mixed_packed[v] = pack_bf16x8(mixed);
      if constexpr (FUSED_NORM) {
        float rounded[VEC];
        unpack_bf16x8(mixed_packed[v], rounded);
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
          norm_sum = fmaf(rounded[e], rounded[e], norm_sum);
        }
      }
    }
    uint4* out = reinterpret_cast<uint4*>(layer_input + token * HIDDEN);
    if constexpr (FUSED_NORM) {
      const float sum = warp_sum(norm_sum);
      if (lane == 0) norm_partials[warp] = sum;
      __syncthreads();
      float total = 0.0f;
#pragma unroll
      for (int source_warp = 0; source_warp < WARPS; ++source_warp) {
        total += norm_partials[source_warp];
      }
      const float inverse = rsqrtf(total / float(HIDDEN) + norm_eps);
#pragma unroll
      for (int v = 0; v < FIN_VECS; ++v) {
        const int vec = v * THREADS + tid;
        float rounded[VEC];
        float w[VEC];
        unpack_bf16x8(mixed_packed[v], rounded);
        unpack_bf16x8(weight_packed[v], w);
        float normed[VEC];
#pragma unroll
        for (int e = 0; e < VEC; ++e) normed[e] = rounded[e] * inverse * w[e];
        out[vec] = pack_bf16x8(normed);
      }
    } else {
#pragma unroll
      for (int v = 0; v < FIN_VECS; ++v) {
        out[v * THREADS + tid] = mixed_packed[v];
      }
    }
    // The next token's mixes / pre / norm_partials writes follow at least
    // one barrier after every read above.
    __syncthreads();
  }

  mhc_barrier_end<NGPU>(signals, self_signal, rank);
}

}  // namespace glm5_mhc_ar
