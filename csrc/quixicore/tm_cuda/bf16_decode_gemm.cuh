#pragma once
// bf16 decode GEMM for M <= 16 tokens on tensor cores: out[M, N] = x[M, K] . W[N, K]^T,
// fp32 accumulation, bf16 or fp32 output, optional bias. Written for the GLM-5.3-Flash
// backbone projections on sm_120 (4x RTX PRO 6000), where cuBLAS runs the decode
// shapes at 59-89% of the ~1.6 TB/s streaming ceiling and the row-per-block fp32
// GEMV (dsv4_projection_ampere.cuh) falls back to cuBLAS speed at M >= 8 because it
// re-reads x per token.
//
// Block = NT weight rows x the whole K. Each K chunk (KCHUNK values of every row of
// the tile plus the 16 padded token rows of x) is staged through shared memory with
// cp.async, STAGES deep; ldmatrix hands the m16n8k16 fragments to mma.sync. The
// WARPS warps split the k16 steps of a chunk (warp w takes steps w, w+WARPS, ...),
// so the per-warp partial accumulators are reduced through shared memory once at
// the end. x rows >= M are zero-filled by cp.async (src-size 0); the tile's rows
// beyond N are clamped for the load and skipped for the store.
// Requirements: K % KCHUNK == 0, KCHUNK % (16 * WARPS) == 0, 16-byte aligned rows.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace tms::decode_gemm {

constexpr int MT = 16;        // token rows per mma; M is padded to this
constexpr int PAD = 8;        // elements of padding per smem row (16 B): conflict-free ldmatrix

inline void check_cuda_status(cudaError_t status) {
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return uint32_t(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
    const int bytes = valid ? 16 : 0;   // 0 -> zero-fill, nothing is read
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <typename OutT> __device__ __forceinline__ OutT to_out(float v);
template <> __device__ __forceinline__ float to_out<float>(float v) { return v; }
template <> __device__ __forceinline__ __nv_bfloat16 to_out<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

template <int NT, int WARPS, int KCHUNK, int STAGES>
struct Cfg {
    static constexpr int THREADS = WARPS * 32;
    static constexpr int ROW = KCHUNK + PAD;              // smem row stride, elements
    static constexpr int X_TILE = MT * ROW;               // elements
    static constexpr int W_TILE = NT * ROW;
    static constexpr int STAGE = X_TILE + W_TILE;
    static constexpr int SMEM_BYTES = STAGES * STAGE * 2;
    static constexpr int KSTEPS = KCHUNK / 16;            // k16 steps per chunk
    static constexpr int NTILES = NT / 8;                 // n8 tiles per block
    static constexpr int KVEC = KCHUNK / 8;               // 16-byte vectors per row per chunk
    static constexpr int RED_BYTES = WARPS * MT * NT * 4; // cross-warp reduction scratch
    static_assert(KSTEPS % WARPS == 0, "warps split the k16 steps of a chunk evenly");
    static_assert(NT % 8 == 0 && NT >= 8, "NT is a multiple of the n8 tile");
    static_assert(STAGES >= 2, "double buffering at least");
    static_assert(RED_BYTES <= SMEM_BYTES, "reduction scratch reuses the stage buffers");
};

template <int NT, int WARPS, int KCHUNK, int STAGES, typename OutT>
__global__ void __launch_bounds__(WARPS * 32) bf16_decode_gemm_kernel(
        const __nv_bfloat16* __restrict__ x,     // [M, K]
        const __nv_bfloat16* __restrict__ w,     // [N, K]
        const float* __restrict__ bias,          // [N] or nullptr
        OutT* __restrict__ out,                  // [M, N]
        int M, int N, int K) {
    using C = Cfg<NT, WARPS, KCHUNK, STAGES>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int n0 = blockIdx.x * NT;
    const int nchunks = K / KCHUNK;

    auto stage_x = [&](int s) { return smem + s * C::STAGE; };
    auto stage_w = [&](int s) { return smem + s * C::STAGE + C::X_TILE; };

    // Issue the cp.async loads of chunk c into stage s.
    auto load_chunk = [&](int c, int s) {
        const int k0 = c * KCHUNK;
        __nv_bfloat16* xs = stage_x(s);
        __nv_bfloat16* ws = stage_w(s);
        for (int i = tid; i < MT * C::KVEC; i += C::THREADS) {
            const int m = i / C::KVEC, v = i - m * C::KVEC;
            const bool valid = m < M;
            const __nv_bfloat16* src = x + size_t(valid ? m : 0) * K + k0 + v * 8;
            cp_async16(smem_u32(xs + m * C::ROW + v * 8), src, valid);
        }
        for (int i = tid; i < NT * C::KVEC; i += C::THREADS) {
            const int r = i / C::KVEC, v = i - r * C::KVEC;
            const int n = min(n0 + r, N - 1);
            cp_async16(smem_u32(ws + r * C::ROW + v * 8), w + size_t(n) * K + k0 + v * 8, true);
        }
    };

    float acc[C::NTILES][4];
#pragma unroll
    for (int j = 0; j < C::NTILES; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc[j][e] = 0.0f;

    // ldmatrix addressing: lane supplies the row address of matrix (lane / 8).
    const int mat = lane >> 3, mrow = lane & 7;
    // A (x) tile: a0 = rows 0-7 / k 0-7, a1 = rows 8-15 / k 0-7, a2 = rows 0-7 / k 8-15, a3 = rows 8-15 / k 8-15.
    const int a_row = mrow + 8 * (mat & 1), a_col = 8 * (mat >> 1);
    // B (w) tile pair: r0 = tile 2p k 0-7, r1 = tile 2p k 8-15, r2 = tile 2p+1 k 0-7, r3 = tile 2p+1 k 8-15.
    const int b_row = mrow + 8 * (mat >> 1), b_col = 8 * (mat & 1);

    // Prologue: STAGES-1 chunks in flight.
#pragma unroll
    for (int c = 0; c < STAGES - 1; ++c) {
        if (c < nchunks) load_chunk(c, c);
        cp_async_commit();
    }

    for (int c = 0; c < nchunks; ++c) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();   // chunk c landed for everyone; stage (c-1)%STAGES is free
        {
            const int cn = c + STAGES - 1;
            if (cn < nchunks) load_chunk(cn, cn % STAGES);
            cp_async_commit();
        }
        const int s = c % STAGES;
        const uint32_t xs = smem_u32(stage_x(s));
        const uint32_t ws = smem_u32(stage_w(s));
#pragma unroll
        for (int step = warp; step < C::KSTEPS; step += WARPS) {
            const int k0 = step * 16;
            uint32_t a[4];
            ldmatrix_x4(a, xs + (a_row * C::ROW + k0 + a_col) * 2);
#pragma unroll
            for (int p = 0; p < C::NTILES / 2; ++p) {
                uint32_t b[4];
                ldmatrix_x4(b, ws + ((16 * p + b_row) * C::ROW + k0 + b_col) * 2);
                mma_bf16_16816(acc[2 * p], a, b[0], b[1]);
                mma_bf16_16816(acc[2 * p + 1], a, b[2], b[3]);
            }
            if constexpr (C::NTILES % 2 == 1) {
                // Last single n8 tile: x2 with lanes 0-15 addressing (k 0-7, k 8-15).
                uint32_t b[2];
                const int r = mrow, col = 8 * (mat & 1);
                ldmatrix_x2(b, ws + ((16 * (C::NTILES / 2) + r) * C::ROW + k0 + col) * 2);
                mma_bf16_16816(acc[C::NTILES - 1], a, b[0], b[1]);
            }
        }
    }

    // Cross-warp reduction of the K partials through shared memory.
    cp_async_wait<0>();
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

template <int NT, int WARPS, int KCHUNK, int STAGES, typename OutT>
static inline void launch(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias, OutT* out,
                   int M, int N, int K, cudaStream_t stream) {
    using C = Cfg<NT, WARPS, KCHUNK, STAGES>;
    auto kern = bf16_decode_gemm_kernel<NT, WARPS, KCHUNK, STAGES, OutT>;
    // Internal linkage keeps this state local to its CUDA module, not an ELF
    // GNU_UNIQUE flag shared by separately loaded probe/serving libraries.
    // Attributes are device-specific; thread-local state also avoids host races.
    static thread_local int configured_device = -1;
    int device = -1;
    check_cuda_status(cudaGetDevice(&device));
    if (configured_device != device) {
        check_cuda_status(cudaFuncSetAttribute(
            kern, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM_BYTES));
        configured_device = device;
    }
    const int blocks = (N + NT - 1) / NT;
    kern<<<blocks, C::THREADS, C::SMEM_BYTES, stream>>>(x, w, bias, out, M, N, K);
}

// Production configs (microbench 2026-09-07, gemm16_bench.py on RTX PRO 6000): 32 rows /
// 3 stages for N >= 4096 (o_proj, q_b, in_proj, dense, shared down at 1.38-1.53 TB/s),
// 16 rows / 6 stages below that (fused_qkv_a 2048 x 4096 at 1.39-1.41 TB/s). Deeper
// pipelines and 256-wide chunks changed nothing: the 10-25 us kernels sit ~1.5 us above
// the streaming ceiling from launch ramp and tail, which no staging depth recovers.
template <typename OutT>
inline void launch_auto(const __nv_bfloat16* x, const __nv_bfloat16* w, const float* bias, OutT* out,
                        int M, int N, int K, cudaStream_t stream) {
    if (N >= 4096) launch<32, 8, 128, 3>(x, w, bias, out, M, N, K, stream);
    else launch<16, 8, 128, 6>(x, w, bias, out, M, N, K, stream);
}

// Shapes the kernel serves: what the microbench showed it beating cuBLAS at every M.
inline bool supports(int M, int N, int K) {
    return M >= 1 && M <= MT && K % 128 == 0 && K >= 512 && N >= 2048 && N <= 16384;
}

}  // namespace tms::decode_gemm
