// SPDX-License-Identifier: Apache-2.0
// NVFP4 (e2m1 weights, e4m3 group-16 scales, fp32 global scale) routed-expert
// MoE for DECODE-sized batches, reading the Marlin-packed expert weights in
// place: two kernels per layer instead of Marlin gemm1 + act_and_mul + Marlin
// gemm2 + moe_sum_add, one assignment row per CTA, no alignment metadata.
//
// Why a second decode path next to Marlin (notebook "Sizing the next levers",
// 2026-09-17): at one token x top-8 the Marlin MoE pair streams 28 MB of
// expert bytes at 1.0-1.2 TB/s on a card whose cold read ceiling is 1.62 TB/s
// (dense Marlin at the same bytes reads at the same 1.19 TB/s; our fp8 decode
// GEMM at 1.45-1.50). Marlin's persistent split-K slices, the locked fp32
// reduction and the m_block_size_8 tile are built for many rows per expert;
// at decode an expert block holds one or two.
//
// Structure (both kernels): a CTA owns one activation row and 16 NJ output
// columns (NJ words of a 64-column Marlin tile); the activation row(s) are
// staged into dynamic shared memory once. Each of the sixteen warps owns one
// 16-k Marlin tile of every 16-tile chunk (16 warps beat 8 by 10-20 % at one
// token: a CTA's chunks run serially, so warps are the only source of bytes
// in flight per SM) and runs its own cp.async ring of STAGES chunks (each lane copies exactly the NJ words it will dequantize,
// lanes 0..7 the tile's scale bytes) - warp-private, so the loop needs a
// warp sync, not a block barrier. Global pointers advance by constant
// strides and the ring index is unrolled: the first cut of this loop spent
// half its ~200 instructions per chunk on address arithmetic and a block
// barrier and was issue-bound at 0.4 us per chunk whatever the ring depth
// (perf/results/2026-09-17/p9-moe-decode). Dequantization is Marlin's
// (nvfp4_moe_prefill_ampere.cuh: e2m1 -> bf16 bit trick, e4m3 scale decode,
// bf16 fragment multiply) and the product runs on mma.m16n8k16 against the
// row (A rows 8..15 zero). The warps' fp32 partials are summed in warp
// order. Deterministic: every sum has a fixed order; no atomics, no
// workspace.
//   gemv1: CTA = (slot s = token * top_k + k, column group) of the gate/up
//     projection; epilogue = global scale, the optional clamp (gate from above,
//     up to +/- clamp: silu_and_mul_with_clamp's act-first form, which GLM-5.3
//     uses with limit 10), SiLU(gate) * up in fp32, bf16 into the intermediate
//     [M * top_k, N].
//   gemv2: CTA = (token, column group) of the down projection; the chunk
//     sequence runs over the token's top_k experts in slot order, each
//     expert's partial is scaled by topk_weight * global scale and added in
//     fp32; the shared-expert output is added when given (the moe_sum_add
//     fold); bf16 into [M, D].
//
// Layout contract (gptq_marlin_repack.cu, tests/glm5_next/test_nvfp4_marlin_layout.py):
// per expert the packed weight is int32 [K/16][N*2]; the 16k x 64n tile
// (kt, nt) is the 128 words at ((kt * N/64) + nt) * 128 and lane L owns words
// 4L..4L+3 (word j = n16 block j: columns 64nt + 16j + {0..15}). Scales are
// uint8 [K/16][N] with the Marlin scale permutation and the [0,2,1,3] byte
// pre-swizzle: the 8 bytes at [kt][64nt + 8(L/4)] decode into the (frag_b0,
// frag_b1) scale pairs of words 0..3 (bytes 0-3 -> words 0, 1; 4-7 -> 2, 3).
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <cooperative_groups.h>
#include "../tm_cuda/bf16_decode_gemm.cuh"

namespace tms::nvfp4dec {

constexpr int WARPS = 16;
constexpr int THREADS = WARPS * 32;
constexpr int KC = WARPS;             // k-tiles per chunk: one per warp
constexpr int CHUNK_K = 16 * KC;      // 256 k per chunk

using tms::decode_gemm::pdl_trigger;
using tms::decode_gemm::pdl_wait;
using tms::decode_gemm::smem_u32;
using tms::decode_gemm::cp_async_commit;
using tms::decode_gemm::cp_async_wait;

template <int BYTES>
__device__ __forceinline__ void cp_async(uint32_t dst, const void* src) {
    static_assert(BYTES == 4 || BYTES == 8 || BYTES == 16, "cp.async sizes");
    if constexpr (BYTES == 16)
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
    else
        asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(BYTES));
}

__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], unsigned a0, unsigned a2, unsigned b0, unsigned b1) {
    // A rows 8..15 (a1, a3) are zero: one activation row per CTA.
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(0u), "r"(a2), "r"(0u), "r"(b0), "r"(b1));
}
// Marlin's e2m1 -> bf16 dequant (value * 2^-126 lands in the bf16 bits).
__device__ __forceinline__ void dequant_e2m1_bf16(unsigned q, unsigned& lo, unsigned& hi) {
    const unsigned out1 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    q <<= 4;
    const unsigned out2 = (q & 0x80008000u) | ((q & 0x70007000u) >> 6);
    lo = out2;
    hi = out1;
}
// Marlin's e4m3 scale bytes -> bf16x2 (bytes 0/2 -> first, 1/3 -> second).
__device__ __forceinline__ void dequant_e4m3_scales_bf16(unsigned q, unsigned& first, unsigned& second) {
    const unsigned out1 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    q <<= 8;
    const unsigned out2 = ((q & 0x80008000u) >> 1) | ((q & 0x7F007F00u) >> 4);
    first = out2;
    second = out1;
}
__device__ __forceinline__ unsigned bf16x2_mul(unsigned a, unsigned b) {
    __nv_bfloat162 r = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&a), *reinterpret_cast<__nv_bfloat162*>(&b));
    return *reinterpret_cast<unsigned*>(&r);
}
__device__ __forceinline__ unsigned bf16_bcast(unsigned packed, int half) {
    const unsigned v = half ? (packed >> 16) : (packed & 0xFFFFu);
    return v | (v << 16);
}

// Shared-memory geometry of one staged k-tile for NJ words per lane.
template <int NJ>
struct TileBytes {
    static constexpr int W = NJ * 4;                    // the lane's words
    static constexpr int WORDS = 32 * W;                // 128 NJ B: lane L at L * W
    static constexpr int SW = NJ == 4 ? 8 : 4;          // scale bytes per lane group (L / 4)
    static constexpr int SCALES = 8 * SW;               // 8 lane groups
    static constexpr int TILE = WORDS + SCALES;
};
// Dynamic shared memory of the kernels: the staged activation row(s) then the ring
// (the ring lives in dynamic memory so the wide configs can pass the 48 KB static cap).
template <int NJ, int STAGES>
__host__ __device__ constexpr int gemv1_ring_bytes() { return STAGES * WARPS * 2 * TileBytes<NJ>::TILE; }
template <int NJ, int STAGES>
__host__ __device__ constexpr int gemv2_ring_bytes() { return STAGES * WARPS * TileBytes<NJ>::TILE; }

// Per-warp copy state for one operand: the lane's word pointer and (lanes 0..7)
// scale pointer at the warp's k-tile of the current chunk; `advance` steps one chunk.
template <int NJ>
struct Cursor {
    const int32_t* w;
    const uint8_t* s;
    size_t wstep, sstep;
    // Expert base (Be, Se) of a packed [K/16][Ncols*2] / [K/16][Ncols] operand; tile column nt, blocks j0.
    __device__ __forceinline__ void reset(const int32_t* Be, const uint8_t* Se, int Ncols, int nt, int j0, int warp, int lane) {
        w = Be + (size_t(warp) * (Ncols / 64) + nt) * 128 + 4 * lane + j0;
        s = Se + size_t(warp) * Ncols + 64 * nt + 8 * lane + 4 * (j0 / 2);
        wstep = size_t(KC) * (Ncols / 64) * 128;
        sstep = size_t(KC) * Ncols;
    }
    __device__ __forceinline__ void issue(uint32_t dst, int lane) {
        using T = TileBytes<NJ>;
        cp_async<T::W>(dst + lane * T::W, w);
        if (lane < 8) cp_async<T::SW>(dst + T::WORDS + lane * T::SW, s);
    }
    __device__ __forceinline__ void advance() { w += wstep; s += sstep; }
};

// acc[j][h] += A(k-tile) . B(word j, n8 half h) from the staged tile.
template <int NJ>
__device__ __forceinline__ void mma_staged(float (&acc)[NJ][2][4], const unsigned char* tile, unsigned a0, unsigned a2,
                                           int j0, int lane) {
    using T = TileBytes<NJ>;
    unsigned w[NJ];
    if constexpr (NJ == 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(tile + lane * T::W);
        w[0] = v.x; w[1] = v.y; w[2] = v.z; w[3] = v.w;
    } else if constexpr (NJ == 2) {
        const uint2 v = *reinterpret_cast<const uint2*>(tile + lane * T::W);
        w[0] = v.x; w[1] = v.y;
    } else {
        w[0] = *reinterpret_cast<const unsigned*>(tile + lane * T::W);
    }
    unsigned sw[(NJ + 1) / 2];
    if constexpr (NJ == 4) {
        const uint2 v = *reinterpret_cast<const uint2*>(tile + T::WORDS + (lane >> 2) * 8);
        sw[0] = v.x; sw[1] = v.y;
    } else {
        sw[0] = *reinterpret_cast<const unsigned*>(tile + T::WORDS + (lane >> 2) * 4);
    }
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
        unsigned fs0, fs1;
        dequant_e4m3_scales_bf16(sw[j / 2], fs0, fs1);
        // Word parity picks the scale pair; for NJ >= 2 it is the compile-time parity of j.
        const unsigned fs = (NJ == 1 ? ((j0 & 1) != 0) : ((j & 1) != 0)) ? fs1 : fs0;
        unsigned b00, b01, b10, b11;
        dequant_e2m1_bf16(w[j] << 8, b00, b01);   // columns 16j + L/4
        dequant_e2m1_bf16(w[j], b10, b11);        // columns 16j + 8 + L/4
        const unsigned s0 = bf16_bcast(fs, 0), s1 = bf16_bcast(fs, 1);
        mma_bf16_16816(acc[j][0], a0, a2, bf16x2_mul(b00, s0), bf16x2_mul(b01, s0));
        mma_bf16_16816(acc[j][1], a0, a2, bf16x2_mul(b10, s1), bf16x2_mul(b11, s1));
    }
}

template <int NJ>
__device__ __forceinline__ void zero_acc(float (&acc)[NJ][2][4]) {
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int h = 0; h < 2; ++h)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[j][h][v] = 0.0f;
}

// Warp-order sum of the lanes-0..3 row-0 partials: red [WARPS][NJ*16] floats
// per operand. Lane L < 4 holds columns 16j + 8h + 2L + {0,1}.
template <int NJ>
__device__ __forceinline__ void store_partial(float* red, const float (&acc)[NJ][2][4], int warp, int lane) {
    if (lane < 4) {
#pragma unroll
        for (int j = 0; j < NJ; ++j)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = 16 * j + 8 * h + 2 * lane;
                red[warp * (NJ * 16) + col] = acc[j][h][0];
                red[warp * (NJ * 16) + col + 1] = acc[j][h][1];
            }
    }
}
template <int NJ>
__device__ __forceinline__ float warp_sum(const float* red, int col) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) v += red[w * (NJ * 16) + col];
    return v;
}

// The router folded into gemv1 (`logits` != nullptr): every CTA of slot s = token * top_k + k
// recomputes the token's routing from the fp32 router logits - glm_moe_routing.cuh's
// selection exactly: sigmoid / sqrt-softplus scores, bias-only top-k with ties to the lowest
// index and NaN below every finite score, weights = unbiased score (renormalized over the
// top-k) * scaling - and takes expert sel[k]; the CTAs of column group 0 write topk_ids and
// topk_w for gemv2 and the runner. One warp's eight argmax rounds cost ~0.5 us inside the
// prologue against the 4.8 us the separate route_align launch held on the critical path.
constexpr int ROUTE_MAX_E = 512;
enum RouteScoring : int { ROUTE_SIGMOID = 0, ROUTE_SQRT_SOFTPLUS = 1 };
__device__ __forceinline__ float route_score(float x, int scoring) {
    if (scoring == ROUTE_SIGMOID) return 1.0f / (1.0f + expf(-x));
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    return sqrtf(sp);
}
// Returns the slot's expert (always in [0, E)) and its weight. `choice`/`score` are E floats
// each in shared memory; every thread of the block takes part.
__device__ __forceinline__ int route_slot(const float* __restrict__ logits, const float* __restrict__ bias, int E,
                                          int scoring, float scaling, int renorm, int top_k, int k,
                                          float* choice, float* score, float& weight_out, int tid, int lane, int warp) {
    for (int e = tid; e < E; e += THREADS) {
        const float sc = route_score(logits[e], scoring);
        score[e] = sc;
        const float c = sc + bias[e];
        choice[e] = (c == c) ? c : -FLT_MAX;
    }
    __shared__ int sel_e;
    __shared__ float sel_w;
    __syncthreads();
    if (warp == 0) {
        // Each lane keeps its candidates in registers (expert lane + 32 j); a round is a
        // register argmax, a warp argmax by shuffles and one register invalidation.
        constexpr int NPL = ROUTE_MAX_E / 32;
        float v[NPL];
#pragma unroll
        for (int j = 0; j < NPL; ++j) {
            const int e = lane + 32 * j;
            v[j] = e < E ? choice[e] : -INFINITY;
        }
        float wsum = 0.0f;
        int mine = 0;
        float mine_s = 0.0f;
        for (int r = 0; r < top_k; ++r) {
            float best = -INFINITY; int bj = 0;
#pragma unroll
            for (int j = 0; j < NPL; ++j) {
                if (v[j] > best) { best = v[j]; bj = j; }   // lower j = lower expert index within the lane
            }
            int best_e = lane + 32 * bj;
            if (best == -INFINITY) best_e = E;
            // Warp argmax in two redux ops: the max orderable key, then the lowest expert
            // index among the lanes holding it (ties to the lowest index, as route_align).
            int key = __float_as_int(best);
            key = key >= 0 ? key : key ^ 0x7FFFFFFF;
            const int kmax = __reduce_max_sync(0xffffffffu, key);
            best_e = __reduce_min_sync(0xffffffffu, key == kmax ? best_e : E);
            if ((best_e & 31) == lane) {
#pragma unroll
                for (int j = 0; j < NPL; ++j)
                    if (j == (best_e >> 5)) v[j] = -INFINITY;
            }
            const float sc = score[best_e];
            wsum += sc;
            if (r == k) { mine = best_e; mine_s = sc; }
        }
        if (lane == 0) {
            const float inv = renorm ? 1.0f / fmaxf(wsum, 1e-20f) : 1.0f;
            sel_e = mine;
            sel_w = mine_s * inv * scaling;
        }
    }
    __syncthreads();
    weight_out = sel_w;
    return sel_e;
}

// gemv1: act[s][col] = silu(gs * gate) * (gs * up) for slot s and its expert.
//   x [M][K] bf16 (row stride ldx), B int32 [E][K/16][(2N)*2], S uint8 [E][K/16][2N],
//   G fp32 [E] (gstride 1) or [1] (gstride 0), topk_ids int32 [M*top_k], act [M*top_k][N] bf16,
//   clamp < 0 for none. grid (N / (16 NJ), M * top_k), dynamic smem K * 2 (the row) + gemv1_ring_bytes
//   (+ 2 * E * 4 with the folded router). K % (CHUNK_K * STAGES) == 0.
//   Folded router: logits fp32 [M][E], bias fp32 [E] (nullptr = topk_ids is an input);
//   topk_ids int32 [M*top_k] and topk_w fp32 [M*top_k] are then written by column group 0.
template <int NJ, int STAGES>
__global__ void __launch_bounds__(THREADS) nvfp4_moe_gemv1_kernel(
        const __nv_bfloat16* __restrict__ x, int ldx,
        const int32_t* __restrict__ B, const uint8_t* __restrict__ S, const float* __restrict__ G,
        int32_t* __restrict__ topk_ids, __nv_bfloat16* __restrict__ act,
        int top_k, int N, int K, int E, int gstride, float clamp,
        const float* __restrict__ logits, const float* __restrict__ bias, int scoring, float scaling, int renorm,
        float* __restrict__ topk_w, int fixed, int pdl) {
    using T = TileBytes<NJ>;
    constexpr int WTILE = 2 * T::TILE;            // the warp's gate + up tiles of one chunk
    constexpr int STAGE = WARPS * WTILE;
    __shared__ float red[2][WARPS][NJ * 16];
    extern __shared__ __align__(16) unsigned char dyn[];   // the token's row: K bf16, then the ring
    const __nv_bfloat16* xs = reinterpret_cast<const __nv_bfloat16*>(dyn);
    unsigned char* ring = dyn + size_t(K) * 2;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int s = blockIdx.y, token = s / top_k;
    const int cg = blockIdx.x;
    const int nt = cg / (4 / NJ), j0 = (cg % (4 / NJ)) * NJ;
    const int col0 = 64 * nt + 16 * j0;
    // Dependent launch: wait first (topk_ids and x come from the predecessor); the
    // trigger comes after the stream. An early trigger let the next kernel's CTAs
    // (gemv2: 512 threads, ~27 KB smem each) sit on the SMs spinning in
    // griddepcontrol.wait while this kernel streamed, which halved gemv1's
    // residency on 128 SMs (45 KB per CTA against the 99 KB budget): in situ the
    // pair's first serving round saved 2 us per layer of the 7 us measured cold.
    if (pdl) pdl_wait();
    {
        const __nv_bfloat16* xr = x + size_t(token) * ldx;   // the row: independent of the expert, ahead of the routing
        for (int i = tid; i < K / 8; i += THREADS) cp_async<16>(smem_u32(dyn + 16 * i), xr + 8 * i);
    }
    int e;
    if (logits != nullptr) {
        // The folded router: the routed slots come from the logits (El wide: the
        // fixed expert, if any, is expert E - 1 and has no logit); the fixed slot
        // is the last of a token's slots and carries weight 1.0.
        const int routed_k = fixed >= 0 ? top_k - 1 : top_k, El = fixed >= 0 ? E - 1 : E;
        const int k = s - token * top_k;
        float w;
        if (k >= routed_k) {
            e = fixed; w = 1.0f;
        } else {
            float* choice = reinterpret_cast<float*>(dyn + size_t(K) * 2 + gemv1_ring_bytes<NJ, STAGES>());
            e = route_slot(logits + size_t(token) * El, bias, El, scoring, scaling, renorm, routed_k, k,
                           choice, choice + El, w, tid, lane, warp);
        }
        if (cg == 0 && tid == 0) { topk_ids[s] = e; topk_w[s] = w; }
    } else {
        e = topk_ids[s];
    }
    if (e < 0 || e >= E) {
        for (int c = tid; c < 16 * NJ; c += THREADS) act[size_t(s) * N + col0 + c] = __float2bfloat16_rn(0.0f);
        cp_async_commit();
        cp_async_wait<0>();                          // the row copies were issued above
        return;
    }
    const int N2 = 2 * N;                            // the packed row: gate columns then up columns
    Cursor<NJ> cg_, cu_;
    cg_.reset(B + size_t(e) * (K / 16) * (size_t(N2) * 2), S + size_t(e) * (K / 16) * size_t(N2), N2, nt, j0, warp, lane);
    cu_.reset(B + size_t(e) * (K / 16) * (size_t(N2) * 2), S + size_t(e) * (K / 16) * size_t(N2), N2, nt + N / 64, j0, warp, lane);
    const uint32_t rbase = smem_u32(ring) + warp * WTILE;
    const unsigned char* rb = ring + warp * WTILE;
    const int nchunks = K / CHUNK_K;
    auto issue = [&](int stage) {
        cg_.issue(rbase + stage * STAGE, lane);
        cu_.issue(rbase + stage * STAGE + T::TILE, lane);
        cg_.advance();
        cu_.advance();
    };

    float accg[NJ][2][4], accu[NJ][2][4];
    zero_acc<NJ>(accg);
    zero_acc<NJ>(accu);
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        issue(st);
        cp_async_commit();
    }
    // Group 0 holds every thread's slice of the staged row: the row is read by all
    // warps, so its completion needs the block barrier once (the ring is warp-private).
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const __nv_bfloat16* xw = xs + 16 * warp + (lane & 3) * 2;   // + CHUNK_K per chunk
    for (int c = 0; c < nchunks; c += STAGES) {
#pragma unroll
        for (int st = 0; st < STAGES; ++st) {
            cp_async_wait<STAGES - 2>();
            __syncwarp();                            // the warp's own copies (lanes 0..7 wrote the scales)
            if (c + st + STAGES - 1 < nchunks) issue((st + STAGES - 1) % STAGES);
            cp_async_commit();
            const unsigned char* tile = rb + st * STAGE;
            const __nv_bfloat16* xa = xw + (c + st) * CHUNK_K;
            const unsigned a0 = *reinterpret_cast<const unsigned*>(xa);
            const unsigned a2 = *reinterpret_cast<const unsigned*>(xa + 8);
            mma_staged<NJ>(accg, tile, a0, a2, j0, lane);
            mma_staged<NJ>(accu, tile + T::TILE, a0, a2, j0, lane);
        }
    }
    cp_async_wait<0>();
    if (pdl) pdl_trigger();
    store_partial<NJ>(&red[0][0][0], accg, warp, lane);
    store_partial<NJ>(&red[1][0][0], accu, warp, lane);
    __syncthreads();
    const float gs = G[e * gstride];
    for (int c = tid; c < 16 * NJ; c += THREADS) {
        float g = gs * warp_sum<NJ>(&red[0][0][0], c);
        float u = gs * warp_sum<NJ>(&red[1][0][0], c);
        if (clamp >= 0.0f) {
            g = fminf(g, clamp);
            u = fmaxf(fminf(u, clamp), -clamp);
        }
        const float silu = g / (1.0f + __expf(-g));
        act[size_t(s) * N + col0 + c] = __float2bfloat16_rn(silu * u);
    }
}

// gemv2: out[m][col] = shared[m][col] + sum_k topk_w[m][k] * gs_e * (act[m*top_k+k] . W2_e^T)[col].
//   act [M*top_k][Ki] bf16, B int32 [E][Ki/16][D*2], S uint8 [E][Ki/16][D], G fp32 [E] or [1],
//   topk_ids int32 [M*top_k], topk_w fp32 [M*top_k] or nullptr (weights already on the input),
//   shared [M][D] bf16 or nullptr, out [M][D] bf16. grid (D / (16 NJ), M), dynamic smem
//   ceil(top_k / split) * Ki * 2 (the CTA's rows) + gemv2_ring_bytes. Ki % CHUNK_K == 0, top_k <= 64. The chunk sequence is
//   (slot, chunk-of-Ki) flattened, so the ring streams across expert boundaries.
template <int NJ, int STAGES>
__global__ void __launch_bounds__(THREADS) nvfp4_moe_gemv2_kernel(
        const __nv_bfloat16* __restrict__ act,
        const int32_t* __restrict__ B, const uint8_t* __restrict__ S, const float* __restrict__ G,
        const int32_t* __restrict__ topk_ids, const float* __restrict__ topk_w,
        const __nv_bfloat16* __restrict__ shared, __nv_bfloat16* __restrict__ out,
        int top_k, int D, int Ki, int E, int gstride, int pdl) {
    using T = TileBytes<NJ>;
    constexpr int STAGE = WARPS * T::TILE;
    __shared__ float red[WARPS][NJ * 16];
    __shared__ float red_out[NJ * 16];               // this CTA's column sums, read by cluster rank 0
    __shared__ int s_e[64];
    __shared__ float s_w[64];
    extern __shared__ __align__(16) unsigned char dyn[];   // the CTA's slots' rows: (top_k / split) * Ki bf16, then the ring
    const __nv_bfloat16* as = reinterpret_cast<const __nv_bfloat16*>(dyn);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int m = blockIdx.y;
    // Launched as a cluster of `split` CTAs along z, each takes ceil(top_k / split)
    // of the token's slots (the last rank the remainder: nine slots split in two
    // are five and four) and rank 0 sums the ranks' column sums in rank order (DSMEM).
    cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int split = int(cluster.num_blocks()), rank = int(cluster.block_rank());
    const int kmax = (top_k + split - 1) / split, k0 = rank * kmax;
    const int kper = max(min(kmax, top_k - k0), 0);
    unsigned char* ring = dyn + size_t(kmax) * Ki * 2;   // the rows area is sized for the fullest rank
    const int cg = blockIdx.x;
    const int nt = cg / (4 / NJ), j0 = (cg % (4 / NJ)) * NJ;
    const int col0 = 64 * nt + 16 * j0;
    if (pdl) pdl_wait();                             // act, topk_ids, weights, shared: all from predecessors
    // The token's experts and weights (top_k <= 64).
    if (tid < kper) {
        const int s = m * top_k + k0 + tid;
        const int e = topk_ids[s];
        const bool ok = e >= 0 && e < E;
        s_e[tid] = ok ? e : -1;
        s_w[tid] = ok ? (topk_w != nullptr ? topk_w[s] : 1.0f) * G[(ok ? e : 0) * gstride] : 0.0f;
    }
    {
        const __nv_bfloat16* ar = act + size_t(m * top_k + k0) * Ki;
        for (int i = tid; i < kper * Ki / 8; i += THREADS) cp_async<16>(smem_u32(dyn + 16 * i), ar + 8 * i);
    }
    __syncthreads();
    const int cpe = Ki / CHUNK_K;                    // chunks per expert (2 at Ki = 512)
    const int nchunks = kper * cpe;
    const size_t ewords = size_t(Ki / 16) * (size_t(D) * 2), escales = size_t(Ki / 16) * size_t(D);
    const uint32_t rbase = smem_u32(ring) + warp * T::TILE;
    const unsigned char* rb = ring + warp * T::TILE;

    // Producer: slot kp, chunk cp within it; the cursor is reset at each slot boundary.
    Cursor<NJ> cur;
    int kp = 0, cp = 0;
    auto set_slot = [&]() {
        const int e = s_e[kp];
        if (e >= 0) cur.reset(B + size_t(e) * ewords, S + size_t(e) * escales, D, nt, j0, warp, lane);
    };
    set_slot();
    auto issue = [&](int stage) {
        if (s_e[kp] >= 0) cur.issue(rbase + stage * STAGE, lane);
        cur.advance();
        if (++cp == cpe) {
            cp = 0;
            if (++kp < kper) set_slot();
        }
    };

    float acc[NJ][2][4], acce[NJ][2][4];
    zero_acc<NJ>(acc);
    zero_acc<NJ>(acce);
#pragma unroll
    for (int st = 0; st < STAGES - 1; ++st) {
        if (st < nchunks) issue(st);
        cp_async_commit();
    }
    cp_async_wait<STAGES - 2>();   // group 0: every thread's slice of the staged rows
    __syncthreads();
    // Consumer: chunk q = kc * cpe + cc; the activation offset is uniform (128 q) since Ki = 128 cpe.
    const __nv_bfloat16* aw = as + 16 * warp + (lane & 3) * 2;
    int kc = 0, cc = 0;
    for (int q0 = 0; q0 < nchunks; q0 += STAGES) {
#pragma unroll
        for (int st = 0; st < STAGES; ++st) {
            const int q = q0 + st;
            if (q < nchunks) {
                cp_async_wait<STAGES - 2>();
                __syncwarp();
                if (q + STAGES - 1 < nchunks) issue((st + STAGES - 1) % STAGES);
                cp_async_commit();
                if (s_e[kc] >= 0) {
                    const __nv_bfloat16* xa = aw + q * CHUNK_K;
                    const unsigned a0 = *reinterpret_cast<const unsigned*>(xa);
                    const unsigned a2 = *reinterpret_cast<const unsigned*>(xa + 8);
                    mma_staged<NJ>(acce, rb + st * STAGE, a0, a2, j0, lane);
                }
                if (++cc == cpe) {
                    // The expert's partial is complete for this warp: scale and add in slot order.
                    cc = 0;
                    const float w = s_w[kc++];
#pragma unroll
                    for (int j = 0; j < NJ; ++j)
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            acc[j][h][0] += w * acce[j][h][0];
                            acc[j][h][1] += w * acce[j][h][1];
                        }
                    zero_acc<NJ>(acce);
                }
            }
        }
    }
    cp_async_wait<0>();
    if (pdl) pdl_trigger();
    store_partial<NJ>(&red[0][0], acc, warp, lane);
    __syncthreads();
    if (split == 1) {
        for (int c = tid; c < 16 * NJ; c += THREADS) {
            float v = warp_sum<NJ>(&red[0][0], c);
            if (shared != nullptr) v += __bfloat162float(shared[size_t(m) * D + col0 + c]);
            out[size_t(m) * D + col0 + c] = __float2bfloat16_rn(v);
        }
        return;
    }
    for (int c = tid; c < 16 * NJ; c += THREADS) red_out[c] = warp_sum<NJ>(&red[0][0], c);
    cluster.sync();                                  // every rank's red_out is complete and visible
    if (rank == 0) {
        for (int c = tid; c < 16 * NJ; c += THREADS) {
            float v = red_out[c];
            for (int r = 1; r < split; ++r) v += cluster.map_shared_rank(red_out, r)[c];
            if (shared != nullptr) v += __bfloat162float(shared[size_t(m) * D + col0 + c]);
            out[size_t(m) * D + col0 + c] = __float2bfloat16_rn(v);
        }
    }
    cluster.sync();                                  // peers keep their shared memory until rank 0 has read it
}

}  // namespace tms::nvfp4dec
