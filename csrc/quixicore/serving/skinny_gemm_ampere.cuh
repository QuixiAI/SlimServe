#pragma once
// Skinny bf16 GEMM for decode / speculative-verify batches on Ampere:
//   Y[M, N] = X[M, K] . W[N, K]^T,   M <= 128, bf16 in, fp32 accumulate.
//
// Motive (GLM-5.3 TP4 x DP2 record, 2026-09-12): the per-rank dense
// projections at M = 128 (32 requests x 4 draft rows) run through cuBLAS at
// 0.3-0.9 TB/s of weight traffic against a 1.7 TB/s roofline - the N per
// rank is small under TP4 (512-6144), so the 128xN tile grids are 8-96 CTAs
// and cuBLAS's own split-K only partly fills the GPU. This kernel streams
// each weight tile once with a 3-stage cp.async pipeline and splits K across
// CTAs so every shape puts >= 2 CTAs per SM in flight; the fp32 partials are
// summed by a small reduce kernel that also applies the bias and rounds.
//
// STATUS (2026-09-12): correct (fp32 split-K partials; more accurate than
// cuBLAS, whose bf16 split-K reduction leaves ~0.4 abs error on 300-scale
// outputs at K = 4096) but NOT faster: 77 us vs cuBLAS 52 us at M=128, K=4096,
// N=6144 (0.65 TB/s). Ablations: no-mma 77 us (loads are the limit), weights
// only 48 us (1.04 TB/s), activations only 53 us (every N tile re-reads the
// 1 MB activation slice: 96 MB of L2 traffic), compute only 39 us. Next
// design if resumed: 256-wide N tiles (4x less L2 re-read) with the activation
// tile staged once per CTA. TRIED AND REJECTED (2026-09-12): activations in
// registers from a fragment-permuted copy with an 8-stage weights-only smem
// pipeline - 122 us at 128x4096x6144 (0.41 TB/s), worse at every shape; the
// per-lane 16 B L2 fragment loads inside the main loop stall harder than the
// shared-memory staging. Standalone op only; not on the serving path.
// Tile: BM x BN over a K slice, BK = 64 per stage, 4 warps. Warps tile the
// rows (BM = 64/128: 16/32 rows per warp) or, for BM <= 32, the columns.
// Fragments come from shared memory through ldmatrix; the smem rows carry
// 16 B of padding so the ldmatrix reads are bank-conflict free.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tms::skinny {

constexpr int BK = 64;
constexpr int LDS = BK + 8;  // smem row stride in bf16 (144 B)

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem, bool pred) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    const int bytes = pred ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}
__device__ __forceinline__ void mma_bf16_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int BM, int BN, int NW, int STAGES>
struct Layout {
    static constexpr int THREADS = NW * 32;
    // Warp grid: rows are split across warps first (m16 granularity), then columns.
    static constexpr int WM = (BM / 16) < NW ? (BM / 16) : NW;
    static constexpr int WN = NW / WM;
    static_assert(WM * WN == NW, "warp grid");
    static constexpr int ROWS = BM / WM;   // rows per warp
    static constexpr int COLS = BN / WN;   // cols per warp
    static constexpr int MT = ROWS / 16;   // m16 tiles per warp
    static constexpr int NT = COLS / 8;    // n8 tiles per warp
    static_assert(ROWS % 16 == 0 && COLS % 16 == 0, "warp tile must be m16 x n16 multiples");
    static constexpr int A_CHUNKS = BM * BK / 8;        // 16 B chunks per stage
    static constexpr int B_CHUNKS = BN * BK / 8;
    static_assert(A_CHUNKS % THREADS == 0 && B_CHUNKS % THREADS == 0, "chunks per thread");
    static constexpr int SMEM_STAGE = (BM + BN) * LDS * 2;
    static constexpr int SMEM = STAGES * SMEM_STAGE;
};

// partial[split][M][N] (fp32) = X[:, slice] . W[:, slice]^T for this CTA's K slice.
template <int BM, int BN, int NW, int STAGES>
__global__ void __launch_bounds__(NW * 32) skinny_gemm_bf16_kernel(
        const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
        float* __restrict__ partial, int M, int N, int K, int k_slice) {
    using L = Layout<BM, BN, NW, STAGES>;
    constexpr int THREADS = L::THREADS;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    const int n0 = blockIdx.x * BN;
    const int split = blockIdx.y;
    const int k0 = split * k_slice;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp / L::WN, wn = warp % L::WN;
    const int row_base = wm * L::ROWS, col_base = wn * L::COLS;

    auto a_stage = [&](int s) { return smem + s * (BM + BN) * LDS; };
    auto b_stage = [&](int s) { return smem + s * (BM + BN) * LDS + BM * LDS; };

    // Issue the loads of one BK stage (predicated on rows/cols in range).
    auto load_stage = [&](int s, int kk) {
        __nv_bfloat16* as = a_stage(s);
        __nv_bfloat16* bs = b_stage(s);
#pragma unroll
        for (int i = 0; i < L::A_CHUNKS / THREADS; ++i) {
            const int c = tid + i * THREADS;
            const int r = c >> 3, c8 = (c & 7) * 8;
            const bool ok = r < M;
            const __nv_bfloat16* src = x + (ok ? size_t(r) * K : 0) + kk + c8;
            cp_async_16(as + r * LDS + c8, src, ok);
        }
#pragma unroll
        for (int i = 0; i < L::B_CHUNKS / THREADS; ++i) {
            const int c = tid + i * THREADS;
            const int r = c >> 3, c8 = (c & 7) * 8;
            const int n = n0 + r;
            const bool ok = n < N;
            const __nv_bfloat16* src = w + (ok ? size_t(n) * K : 0) + kk + c8;
            cp_async_16(bs + r * LDS + c8, src, ok);
        }
    };

    float acc[L::MT][L::NT][4];
#pragma unroll
    for (int i = 0; i < L::MT; ++i)
#pragma unroll
        for (int j = 0; j < L::NT; ++j)
#pragma unroll
            for (int v = 0; v < 4; ++v) acc[i][j][v] = 0.0f;

    const int steps = k_slice / BK;
    // Prologue: STAGES - 1 stages in flight.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < steps) load_stage(s, k0 + s * BK);
        cp_async_commit();
    }
    for (int step = 0; step < steps; ++step) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        // Prefetch the stage STAGES-1 ahead (its buffer was consumed last iteration).
        {
            const int nxt = step + STAGES - 1;
            if (nxt < steps) load_stage(nxt % STAGES, k0 + nxt * BK);
            cp_async_commit();
        }
        const __nv_bfloat16* as = a_stage(step % STAGES);
        const __nv_bfloat16* bs = b_stage(step % STAGES);
        // Fragments double-buffered across the k16 steps: the ldmatrix for
        // step kk+1 is issued before the mma chain of step kk (ncu 2026-09-12:
        // 76% of cycles had no eligible warp, a third of the stall on the
        // fixed-latency ldmatrix -> mma dependency).
        unsigned a[2][L::MT][4];
        unsigned b[2][L::NT / 2][4];
        auto load_frags = [&](int buf, int kk) {
#pragma unroll
            for (int i = 0; i < L::MT; ++i) {
                const int r = row_base + i * 16 + (lane & 15);
                const int c = kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a[buf][i], as + r * LDS + c);
            }
#pragma unroll
            for (int j = 0; j < L::NT / 2; ++j) {
                const int q = lane >> 3;
                const int r = col_base + j * 16 + (q >> 1) * 8 + (lane & 7);
                const int c = kk * 16 + (q & 1) * 8;
                ldmatrix_x4(b[buf][j], bs + r * LDS + c);
            }
        };
        load_frags(0, 0);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            const int cur = kk & 1;
            if (kk + 1 < BK / 16) load_frags(cur ^ 1, kk + 1);
#pragma unroll
            for (int j = 0; j < L::NT / 2; ++j) {
#pragma unroll
                for (int i = 0; i < L::MT; ++i) {
                    mma_bf16_16816(acc[i][2 * j], a[cur][i], b[cur][j][0], b[cur][j][1]);
                    mma_bf16_16816(acc[i][2 * j + 1], a[cur][i], b[cur][j][2], b[cur][j][3]);
                }
            }
        }
    }
    cp_async_wait<0>();
    // Epilogue: fp32 partials for this split.
    float* out = partial + size_t(split) * M * N;
    const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < L::MT; ++i) {
#pragma unroll
        for (int j = 0; j < L::NT; ++j) {
            const int n = n0 + col_base + j * 8 + t2;
            if (n >= N) continue;
            const int r0 = row_base + i * 16 + g, r1 = r0 + 8;
            if (r0 < M) *reinterpret_cast<float2*>(out + size_t(r0) * N + n) = make_float2(acc[i][j][0], acc[i][j][1]);
            if (r1 < M) *reinterpret_cast<float2*>(out + size_t(r1) * N + n) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// out[M][N] (bf16) = sum over splits of partial + bias.
__global__ void skinny_reduce_kernel(const float* __restrict__ partial, const __nv_bfloat16* __restrict__ bias,
                                     __nv_bfloat16* __restrict__ out, int MN, int N, int splits) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= MN) return;
    float2 s = *reinterpret_cast<const float2*>(partial + i);
    for (int k = 1; k < splits; ++k) {
        const float2 p = *reinterpret_cast<const float2*>(partial + size_t(k) * MN + i);
        s.x += p.x; s.y += p.y;
    }
    if (bias != nullptr) {
        const int n = i % N;
        s.x += __bfloat162float(bias[n]); s.y += __bfloat162float(bias[n + 1]);
    }
    *reinterpret_cast<__nv_bfloat162*>(out + i) = __floats2bfloat162_rn(s.x, s.y);
}

}  // namespace tms::skinny
