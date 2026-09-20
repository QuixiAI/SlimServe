#pragma once
// FP8 block-scaled weights (e4m3 with 128x128 fp32 scales), bf16 activations, M <= 16 tokens, on
// tensor cores: out[M, N] = x[M, K] . dequant(W)[N, K]^T with
//   dequant(W)[n][k] = bf16(scale[n/128][k/128] * W[n][k]),
// i.e. the weights the mma multiplies are bit-for-bit the BF16 tensors a dequantizing checkpoint
// conversion would store, at half the bytes. Same pipeline as bf16_decode_gemm.cuh (cp.async staged
// K chunks, warps splitting the k16 steps of a chunk, one shared-memory reduction). The B fragments
// come from two 16-bit shared loads per n8 tile (the two e4m3 pairs a lane needs: k = 2q, 2q+1 and
// 2q+8, 2q+9), converted through f16x2 (exact for e4m3) and scaled in fp32. One K chunk is one scale
// block and a block's NT rows sit inside one 128-row scale group, so a chunk has a single scale.
// Requirements: K % 128 == 0 (16-byte aligned rows follow), rows beyond N clamped for the load and
// skipped for the store.
#include <cstdlib>
#include <cuda_fp16.h>
#include "bf16_decode_gemm.cuh"

namespace tms::decode_gemm_fp8 {
using tms::decode_gemm::MT;
using tms::decode_gemm::PAD;
using tms::decode_gemm::smem_u32;
using tms::decode_gemm::cp_async16;
using tms::decode_gemm::cp_async_commit;
using tms::decode_gemm::cp_async_wait;
using tms::decode_gemm::ldmatrix_x4;
using tms::decode_gemm::mma_bf16_16816;
using tms::decode_gemm::to_out;
using tms::decode_gemm::check_cuda_status;
using tms::decode_gemm::pdl_trigger;
using tms::decode_gemm::pdl_wait;
using tms::decode_gemm::prefetch_l2;
using tms::decode_gemm::prefetch_bulk_l2;
using tms::decode_gemm::touch_l2;
using tms::decode_gemm::pdl_mode;
using tms::decode_gemm::use_pdl;

constexpr int SB = 128;     // scale block: 128 rows x 128 k
constexpr int WPAD = 16;    // bytes of padding per fp8 weight row in smem (conflict-free 16-bit loads)

// Two e4m3 values (packed low/high) -> bf16x2 of scale * value. e4m3 -> f16 is exact; the only
// rounding is the final bf16 of the fp32 product, the same rounding a checkpoint dequant applies.
__device__ __forceinline__ uint32_t e4m3x2_scaled_to_bf16x2(uint16_t packed, float s) {
    uint32_t h2;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(h2) : "h"(packed));
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
    const __nv_bfloat162 b = __floats2bfloat162_rn(f.x * s, f.y * s);
    return *reinterpret_cast<const uint32_t*>(&b);
}

template <int NT, int WARPS, int KCHUNK, int STAGES>
struct Cfg {
    static constexpr int THREADS = WARPS * 32;
    static constexpr int XROW = KCHUNK + PAD;          // bf16 elements per x row in smem
    static constexpr int X_BYTES = MT * XROW * 2;
    static constexpr int WROW = KCHUNK + WPAD;         // bytes per weight row in smem
    static constexpr int W_BYTES = NT * WROW;
    static constexpr int STAGE_BYTES = X_BYTES + W_BYTES;
    static constexpr int RED_BYTES = WARPS * MT * NT * 4;
    static constexpr int SMEM_BYTES = STAGES * STAGE_BYTES > RED_BYTES ? STAGES * STAGE_BYTES : RED_BYTES;
    static constexpr int KSTEPS = KCHUNK / 16;
    static constexpr int NTILES = NT / 8;
    static constexpr int XVEC = KCHUNK / 8;            // 16-byte vectors per x row per chunk
    static constexpr int WVEC = KCHUNK / 16;           // 16-byte vectors per w row per chunk

    static_assert(SB % NT == 0, "a block's rows lie inside one scale row group");
    static_assert(KCHUNK % SB == 0, "a chunk covers whole scale blocks");
    static constexpr int SCALES_PER_CHUNK = KCHUNK / SB;
    static_assert(KSTEPS % WARPS == 0, "warps split the k16 steps of a chunk evenly");
    static_assert(NT % 8 == 0, "NT is a multiple of the n8 tile");
    static_assert(STAGES >= 2, "double buffering at least");
    static_assert(STAGE_BYTES % 16 == 0 && WROW % 16 == 0, "16-byte aligned stages and rows");
};

// CHANNEL: one fp32 scale per weight row ([N]) instead of the 128x128 block
// scales; each lane keeps the scales of the rows its B fragments cover.
// GATED: the block's NT rows are NT/2 gate rows (n0h + r) and their NT/2 up rows
// (N/2 + n0h + r) of a merged [gate; up] weight, and the epilogue writes
// silu(clamp(gate)) * clamp(up) (silu_and_mul_with_clamp's act-first form,
// clamp < 0 for none) into out[M][N/2]: the shared experts' activation launch
// folded into their gate/up GEMM.
template <int NT, int WARPS, int KCHUNK, int STAGES, typename OutT, bool CHANNEL = false, bool GATED = false>
__global__ void __launch_bounds__(WARPS * 32) fp8_decode_gemm_kernel(
        const __nv_bfloat16* __restrict__ x,     // [M, K]
        const uint8_t* __restrict__ w,           // [N, K] e4m3
        const float* __restrict__ scale,         // [ceil(N/128), K/128], or [N] when CHANNEL
        const float* __restrict__ bias,          // [N] or nullptr
        OutT* __restrict__ out,                  // [M, N] (GATED: [M, N/2])
        int M, int N, int K, float clamp, int pdl) {
    using C = Cfg<NT, WARPS, KCHUNK, STAGES>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n0 = blockIdx.x * NT;
    const int n0h = blockIdx.x * (NT / 2);                       // GATED: the block's first gate row
    auto row_of = [&](int r) { return GATED ? (r < NT / 2 ? n0h + r : N / 2 + n0h + (r - NT / 2)) : n0 + r; };
    const int nchunks = K / KCHUNK;
    // this block's scale row(s): one value per scale block (GATED: the gate rows' and the up rows')
    const float* srow = scale + size_t((GATED ? n0h : n0) / SB) * (K / SB);
    const float* srow_u = GATED ? scale + size_t((N / 2 + n0h) / SB) * (K / SB) : srow;
    if (pdl) {
        // GATED (the shared experts gate/up + activation) triggers after its stream
        // instead: its dependent is the shared down GEMM, whose early-launched CTAs
        // would otherwise sit on the SMs while this kernel and the routed experts
        // stream (c1 plain 227.5 -> 217.4 with the trigger here, 2026-09-18).
        if constexpr (!GATED) pdl_trigger();
        // The weight slice does not depend on the predecessor: request every
        // 128-byte line of the block's NT rows (NT * K bytes) into L2 now, so
        // the staged cp.async loads below hit L2 once the activation is ready.
        const size_t row_bytes = size_t(K);
        if (pdl == 4) {
            // launch overlap only: no prefetch
        } else if (pdl == 2) {
            // one bulk prefetch per row (K bytes, contiguous)
            if (tid < NT) prefetch_bulk_l2(w + size_t(min(row_of(tid), N - 1)) * K, uint32_t(row_bytes));
        } else {
            const size_t lines_per_row = row_bytes / 128;
            for (size_t i = tid; i < size_t(NT) * lines_per_row; i += C::THREADS) {
                const int r = int(i / lines_per_row);
                const size_t off = (i - size_t(r) * lines_per_row) * 128;
                const int n = min(row_of(r), N - 1);
                if (pdl == 3) touch_l2(w + size_t(n) * K + off);
                else prefetch_l2(w + size_t(n) * K + off);
            }
        }
        if constexpr (!CHANNEL) { if (tid == 0) { prefetch_l2(srow); if (GATED) prefetch_l2(srow_u); } }   // one line holds the row's scales
        pdl_wait();
    }

    auto stage_x = [&](int s) { return smem_raw + s * C::STAGE_BYTES; };
    auto stage_w = [&](int s) { return smem_raw + s * C::STAGE_BYTES + C::X_BYTES; };

    auto load_chunk = [&](int c, int s) {
        const int k0 = c * KCHUNK;
        unsigned char* xs = stage_x(s);
        unsigned char* ws = stage_w(s);
        for (int i = tid; i < MT * C::XVEC; i += C::THREADS) {
            const int m = i / C::XVEC, v = i - m * C::XVEC;
            const bool valid = m < M;
            const __nv_bfloat16* src = x + size_t(valid ? m : 0) * K + k0 + v * 8;
            cp_async16(smem_u32(xs + (m * C::XROW + v * 8) * 2), src, valid);
        }
        for (int i = tid; i < NT * C::WVEC; i += C::THREADS) {
            const int r = i / C::WVEC, v = i - r * C::WVEC;
            const int n = min(row_of(r), N - 1);
            cp_async16(smem_u32(ws + r * C::WROW + v * 16), w + size_t(n) * K + k0 + v * 16, true);
        }
    };

    float acc[C::NTILES][4];
#pragma unroll
    for (int j = 0; j < C::NTILES; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    const int mat = lane >> 3, mrow = lane & 7;
    const int a_row = mrow + 8 * (mat & 1), a_col = 8 * (mat >> 1);   // ldmatrix.x4 A addressing
    const int bn = lane >> 2, bk = (lane & 3) * 2;                     // B fragment: row bn of each n8 tile, k pairs bk, bk+8
    float row_scale[C::NTILES];
    if constexpr (CHANNEL) {
#pragma unroll
        for (int j = 0; j < C::NTILES; ++j) row_scale[j] = __ldg(scale + min(row_of(8 * j + bn), N - 1));
    }

#pragma unroll
    for (int c = 0; c < STAGES - 1; ++c) {
        if (c < nchunks) load_chunk(c, c);
        cp_async_commit();
    }

    for (int c = 0; c < nchunks; ++c) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int cn = c + STAGES - 1;
            if (cn < nchunks) load_chunk(cn, cn % STAGES);
            cp_async_commit();
        }
        const int s = c % STAGES;
        const uint32_t xs = smem_u32(stage_x(s));
        const unsigned char* ws = stage_w(s);
        float chunk_scale[C::SCALES_PER_CHUNK], chunk_scale_u[C::SCALES_PER_CHUNK];
        if constexpr (!CHANNEL) {
#pragma unroll
            for (int q = 0; q < C::SCALES_PER_CHUNK; ++q) {
                chunk_scale[q] = __ldg(srow + c * C::SCALES_PER_CHUNK + q);
                if constexpr (GATED) chunk_scale_u[q] = __ldg(srow_u + c * C::SCALES_PER_CHUNK + q);
            }
        }
#pragma unroll
        for (int step = warp; step < C::KSTEPS; step += WARPS) {
            const int k0 = step * 16;
            uint32_t a[4];
            ldmatrix_x4(a, xs + (a_row * C::XROW + k0 + a_col) * 2);
#pragma unroll
            for (int j = 0; j < C::NTILES; ++j) {
                const unsigned char* p = ws + (8 * j + bn) * C::WROW + k0 + bk;
                const uint16_t lo = *reinterpret_cast<const uint16_t*>(p);
                const uint16_t hi = *reinterpret_cast<const uint16_t*>(p + 8);
                const float s = CHANNEL ? row_scale[j]
                                        : ((GATED && 8 * j + bn >= NT / 2) ? chunk_scale_u[k0 / SB] : chunk_scale[k0 / SB]);
                mma_bf16_16816(acc[j], a, e4m3x2_scaled_to_bf16x2(lo, s), e4m3x2_scaled_to_bf16x2(hi, s));
            }
        }
    }

    cp_async_wait<0>();
    if constexpr (GATED) { if (pdl) pdl_trigger(); }
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem_raw);   // [WARPS][MT][NT]
    {
        float* mine = red + warp * (MT * NT);
        const int r0 = lane >> 2, cc = (lane & 3) * 2;
#pragma unroll
        for (int j = 0; j < C::NTILES; ++j) {
            const int n = 8 * j + cc;
            mine[r0 * NT + n] = acc[j][0];
            mine[r0 * NT + n + 1] = acc[j][1];
            mine[(r0 + 8) * NT + n] = acc[j][2];
            mine[(r0 + 8) * NT + n + 1] = acc[j][3];
        }
    }
    __syncthreads();
    if constexpr (GATED) {
        constexpr int NH = NT / 2;
        const int Nh = N / 2;
        for (int i = tid; i < MT * NH; i += C::THREADS) {
            const int m = i / NH, n = i - m * NH;
            if (m >= M || n0h + n >= Nh) continue;
            float g = 0.0f, u = 0.0f;
#pragma unroll
            for (int wv = 0; wv < WARPS; ++wv) {
                g += red[wv * (MT * NT) + m * NT + n];
                u += red[wv * (MT * NT) + m * NT + NH + n];
            }
            if (bias != nullptr) { g += bias[n0h + n]; u += bias[Nh + n0h + n]; }
            if (clamp >= 0.0f) {
                g = fminf(g, clamp);
                u = fmaxf(fminf(u, clamp), -clamp);
            }
            const float v = g / (1.0f + __expf(-g)) * u;
            out[size_t(m) * Nh + n0h + n] = to_out<OutT>(v);
        }
        return;
    }
    for (int i = tid; i < MT * NT; i += C::THREADS) {
        const int m = i / NT, n = i - m * NT;
        if (m >= M || n0 + n >= N) continue;
        float v = 0.0f;
#pragma unroll
        for (int wv = 0; wv < WARPS; ++wv) v += red[wv * (MT * NT) + i];
        if (bias != nullptr) v += bias[n0 + n];
        out[size_t(m) * N + n0 + n] = to_out<OutT>(v);
    }
}

// QC_PDL selects the decode GEMM launch mode (read once per process, see
// bf16_decode_gemm.cuh: 0 off; 1 line-hint prefetch; 2 bulk L2 prefetch;
// 3 real loads; 4 attribute, trigger and wait only). Default 4.

template <int NT, int WARPS, int KCHUNK, int STAGES, typename OutT, bool CHANNEL = false, bool GATED = false>
static inline void launch(const __nv_bfloat16* x, const uint8_t* w, const float* scale, const float* bias, OutT* out,
                   int M, int N, int K, cudaStream_t stream, float clamp = -1.0f) {
    using C = Cfg<NT, WARPS, KCHUNK, STAGES>;
    auto kern = fp8_decode_gemm_kernel<NT, WARPS, KCHUNK, STAGES, OutT, CHANNEL, GATED>;
    // The cache belongs to this module's kernel and the current device.
    // An external inline function's GNU_UNIQUE flag can be coalesced across
    // DSOs even though their CUDA kernel handles require separate setup.
    static thread_local int configured_device = -1;
    int device = -1;
    check_cuda_status(cudaGetDevice(&device));
    if (configured_device != device) {
        check_cuda_status(cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM_BYTES));
        configured_device = device;
    }
    const int blocks = (N + NT - 1) / NT;
    if (use_pdl()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = dim3(blocks);
        cfg.blockDim = dim3(C::THREADS);
        cfg.dynamicSmemBytes = C::SMEM_BYTES;
        cfg.stream = stream;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        check_cuda_status(cudaLaunchKernelEx(&cfg, kern, x, w, scale, bias, out, M, N, K, clamp, pdl_mode()));
    } else {
        kern<<<blocks, C::THREADS, C::SMEM_BYTES, stream>>>(x, w, scale, bias, out, M, N, K, clamp, 0);
    }
}

// Retained configs (fp8_gemm16_bench.py, 2026-09-07, RTX PRO 6000): 32 rows /
// 4 stages for N >= 2048 (dense gate_up 17 us = 1.45 TB/s, dense down 10, DSA o_proj 12, DSA q_b 5, shared
// down 2.1-2.9 us in that historical sweep). Its ROT24 was only 96/48 MiB for shared gate/up/down,
// below this GPU's 128 MiB L2. The 2026-09-08 actual-weight cache control measures higher latency
// with >3 L2 capacities; the old sweep does NOT establish cold-memory configuration optimality.
// At N = 1024 (the shared-expert gate_up, which runs
// on the aux stream underneath the Marlin routed-expert kernels) the grid is the limit, so 8 rows (128 blocks)
// up to M = 8 and 16 rows above, where the 8-row tile re-streams the 16-row x tile too often. The stage count
// there is chosen under contention, not in isolation: in isolation 4 stages is best (4.5-5.7 us vs cuBLAS
// bf16 6.8-8.5), but in the serving trace the 4-stage kernel takes 20-23 us next to Marlin where the 8-stage
// one takes 7-8 us (notebook 2026-09-07, swap-set part 2 follow-up); deeper prefetch keeps more bytes in
// flight while the SMs are shared. The 16-row branch follows by analogy pending a profiled c16 round.
template <typename OutT, bool CHANNEL = false, bool GATED = false>
inline void launch_auto(const __nv_bfloat16* x, const uint8_t* w, const float* scale, const float* bias, OutT* out,
                        int M, int N, int K, cudaStream_t stream, float clamp = -1.0f) {
    if (N >= 2048) launch<32, 8, 128, 4, OutT, CHANNEL, GATED>(x, w, scale, bias, out, M, N, K, stream, clamp);
    // Under programmatic dependent launch the weight slice is L2-resident by
    // the time the wait returns, and the pipeline's per-chunk round trip is
    // the limit: eight stages keep seven chunks in flight instead of three.
    else if (M <= 8) launch<8, 8, 128, 8, OutT, CHANNEL, GATED>(x, w, scale, bias, out, M, N, K, stream, clamp);
    else launch<16, 8, 128, 8, OutT, CHANNEL, GATED>(x, w, scale, bias, out, M, N, K, stream, clamp);
}

// N up to the vocabulary shard of a channel-scaled LM head (GLM-5.3-Flash at
// TP4: 38720 rows, 1210 blocks of the 32-row config).
inline bool supports(int M, int N, int K) {
    return M >= 1 && M <= MT && K % 128 == 0 && K >= 512 && N % 8 == 0 && N >= 1024 && N <= 65536;
}

}  // namespace tms::decode_gemm_fp8
