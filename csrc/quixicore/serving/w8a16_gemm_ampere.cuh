#pragma once
// W8A16 skinny GEMM for Ampere: Y[M, N] = X[M, K] . W[N, K]^T with fp8 e4m3
// weights (per-output-channel scale) and bf16 activations, M <= 128.
//
// Motive (GLM-5.3 TP4 x DP2 record, 2026-09-12): the ~2.5 GB of bf16
// attention / KDA / indexer weights per rank are streamed by cuBLAS at
// 0.3-0.9 TB/s at M = 128, and fp8 marlin (built for M <= 64, 4-bit) is
// slower still. Halving the weight bytes only pays with a kernel shaped for
// M = 32-128: this one.
//
// Weight layout (packed once at load by w8a16_pack): fragment order
//   Wp[k_tile = K/16][n_quad = N/32][lane 0..31][4 n8 tiles][4 bytes]
// where lane t holds W[n = 8*n_tile + t/4][k = 16*k_tile + {2(t%4), 2(t%4)+1,
// 2(t%4)+8, 2(t%4)+9}] - exactly the mma.m16n8k16 B fragment (b0 = first two,
// b1 = last two). A BN x BK tile is therefore (BN/8)*(BK/16)*128 contiguous
// bytes per k_tile row, copied with plain 16 B cp.async and read back as one
// 32-bit shared load per (n8 tile, k16 step) per thread.
//
// fp8 -> fp16: e4m3 bits (s eeee mmm) placed in the high byte of a 16-bit
// word and shifted right by one give fp16 (0 eeee mmm0000000) = value * 2^-8
// (fp16 bias 15 vs e4m3 bias 7), sign restored separately; the 2^8 is folded
// into the per-channel scale applied in the reduce. Activations are converted
// bf16 -> fp16 while staging the A tile (fp16 keeps 3 more mantissa bits than
// bf16; hidden states here are post-norm and well inside the fp16 range - the
// canaries and recall gates check that assumption in serving).
//
// CTA: BM = 128 rows x BN cols over a K slice (split-K across CTAs, fp32
// partials), 4 warps each owning 32 rows, ldmatrix A fragments, 2-stage
// cp.async pipeline for the weights, the A tile staged through registers.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tms::w8a16 {

constexpr int BK = 64;
constexpr int LDA = BK + 8;   // fp16 elements per A smem row (144 B)

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
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma_f16_16816(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// Four packed e4m3 bytes -> two fp16x2 words (b0: bytes 0,1; b1: bytes 2,3), each value * 2^-8.
__device__ __forceinline__ void fp8x4_to_f16x2x2(unsigned p, unsigned& b0, unsigned& b1) {
    // byte i -> half i: place byte in the high byte of its 16-bit lane, shift right 1 (keeps sign at bit 15? no:
    // sign lands at bit 14). Do sign separately: mask 0x7F into exp/mant and OR the sign at bit 15.
    unsigned w0 = ((p & 0x000000FFu) << 8) | ((p & 0x0000FF00u) << 16);   // byte0 -> bits 8..15, byte1 -> bits 24..31
    unsigned w1 = ((p & 0x00FF0000u) >> 8) | ((p & 0xFF000000u));         // byte2 -> bits 8..15, byte3 -> bits 24..31
    const unsigned sign0 = w0 & 0x80008000u, sign1 = w1 & 0x80008000u;
    b0 = ((w0 & 0x7F007F00u) >> 1) | sign0;
    b1 = ((w1 & 0x7F007F00u) >> 1) | sign1;
}

template <int BN, int STAGES, int NW = 4>
struct Layout {
    static constexpr int BM = 128;
    static constexpr int THREADS = NW * 32;
    static constexpr int WN = NW / 4;         // warps along N (4 along M, 32 rows each)
    static constexpr int MT = 2;              // m16 tiles per warp (32 rows)
    static constexpr int COLS = BN / WN;      // columns per warp
    static constexpr int NT = COLS / 8;       // n8 tiles per warp
    static_assert(COLS % 32 == 0, "warp columns in quads");
    static constexpr int B_TILE_BYTES = (BK / 16) * (BN / 32) * 512;   // per stage
    static constexpr int A_STAGE_BYTES = BM * LDA * 2;
    static constexpr int SMEM = STAGES * (A_STAGE_BYTES + B_TILE_BYTES);
    static_assert(B_TILE_BYTES % (THREADS * 16) == 0, "B chunks per thread");
    static_assert((BM * BK / 8) % THREADS == 0, "A chunks per thread");
};

template <int BN, int STAGES, int NW = 4>
__global__ void __launch_bounds__(NW * 32) w8a16_gemm_kernel(
        const __half* __restrict__ x,            // [M, K] fp16 (pre-converted)
        const uint8_t* __restrict__ wp,          // packed fp8, see layout
        float* __restrict__ partial,             // [splits, M, N] fp32 (unscaled)
        int M, int N, int K, int k_slice) {
    using L = Layout<BN, STAGES, NW>;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __half* a_smem = reinterpret_cast<__half*>(smem_raw);                                  // STAGES x BM x LDA
    unsigned char* b_smem = smem_raw + STAGES * L::A_STAGE_BYTES;                          // STAGES x B_TILE_BYTES
    const int n0 = blockIdx.x * BN;
    const int split = blockIdx.y;
    const int k0 = split * k_slice;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 3, wn = warp >> 2;
    const int row_base = wm * 32;
    const int quad_base = wn * (L::COLS / 32);   // this warp's first quad within the tile
    const int KT = K / 16;

    // A stage (fp16, converted once per call by w8a16_convert_a): cp.async.
    auto stage_a = [&](int s, int kk) {
        __half* as = a_smem + s * L::BM * LDA;
#pragma unroll
        for (int i = 0; i < (L::BM * BK / 8) / L::THREADS; ++i) {
            const int c = tid + i * L::THREADS;
            const int r = c >> 3, c8 = (c & 7) * 8;
            const bool ok = r < M;
            const __half* src = x + (ok ? size_t(r) * K : 0) + kk + c8;
            cp_async_16(as + r * LDA + c8, src, ok);
        }
    };
    // B stage: tile rows n_tile0 .. n_tile0 + BN/8, k tiles kk/16 .. +BK/16, 128 B each.
    // smem B stage layout: [kt in stage][quad in tile][lane][16 B]; global: [k_tile][n_quad][lane][16 B].
    const int NQ_ALL = N / 32, nq0 = n0 / 32;
    auto load_b = [&](int s, int kk) {
        unsigned char* bs = b_smem + s * L::B_TILE_BYTES;
        const int kt0 = kk / 16;
#pragma unroll
        for (int i = 0; i < L::B_TILE_BYTES / (L::THREADS * 16); ++i) {
            const int c = tid + i * L::THREADS;            // 16 B chunk index within the stage
            const int kt = c / ((BN / 32) * 32);
            const int rem = c - kt * ((BN / 32) * 32);
            const int q = rem / 32, ln = rem - q * 32;
            const int gq = nq0 + q;
            const bool ok = gq < NQ_ALL;
            const uint8_t* src = wp + ((size_t(kt0 + kt) * NQ_ALL + (ok ? gq : 0)) * 32 + ln) * 16;
            cp_async_16(bs + c * 16, src, ok);
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
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < steps) { load_b(s, k0 + s * BK); stage_a(s, k0 + s * BK); }
        cp_async_commit();
    }
    for (int step = 0; step < steps; ++step) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nxt = step + STAGES - 1;
            if (nxt < steps) { load_b(nxt % STAGES, k0 + nxt * BK); stage_a(nxt % STAGES, k0 + nxt * BK); }
            cp_async_commit();
        }
        const __half* as = a_smem + (step % STAGES) * L::BM * LDA;
        const unsigned char* bs = b_smem + (step % STAGES) * L::B_TILE_BYTES;
        // Per k16 step: 2 ldmatrix (A) + BN/32 LDS.128 (B, 4 n8 tiles each) issued
        // a step ahead, converted in bulk, then the MT x NT mma chain.
        unsigned a[2][L::MT][4];
        uint4 braw[2][L::NT / 4];
        auto load_step = [&](int buf, int kk) {
#pragma unroll
            for (int i = 0; i < L::MT; ++i) {
                const int r = row_base + i * 16 + (lane & 15);
                const int c = kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a[buf][i], as + r * LDA + c);
            }
#pragma unroll
            for (int q = 0; q < L::NT / 4; ++q) {
                braw[buf][q] = *reinterpret_cast<const uint4*>(bs + ((kk * (BN / 32) + quad_base + q) * 32 + lane) * 16);
            }
        };
        load_step(0, 0);
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk) {
            const int cur = kk & 1;
            if (kk + 1 < BK / 16) load_step(cur ^ 1, kk + 1);
            unsigned b0[L::NT], b1[L::NT];
#pragma unroll
            for (int q = 0; q < L::NT / 4; ++q) {
                fp8x4_to_f16x2x2(braw[cur][q].x, b0[4 * q], b1[4 * q]);
                fp8x4_to_f16x2x2(braw[cur][q].y, b0[4 * q + 1], b1[4 * q + 1]);
                fp8x4_to_f16x2x2(braw[cur][q].z, b0[4 * q + 2], b1[4 * q + 2]);
                fp8x4_to_f16x2x2(braw[cur][q].w, b0[4 * q + 3], b1[4 * q + 3]);
            }
#pragma unroll
            for (int j = 0; j < L::NT; ++j)
#pragma unroll
                for (int i = 0; i < L::MT; ++i) mma_f16_16816(acc[i][j], a[cur][i], b0[j], b1[j]);
        }
    }
    cp_async_wait<0>();
    float* out = partial + size_t(split) * M * N;
    const int g = lane >> 2, t2 = (lane & 3) * 2;
#pragma unroll
    for (int i = 0; i < L::MT; ++i) {
#pragma unroll
        for (int j = 0; j < L::NT; ++j) {
            const int n = n0 + quad_base * 32 + j * 8 + t2;
            if (n >= N) continue;
            const int r0 = row_base + i * 16 + g, r1 = r0 + 8;
            if (r0 < M) *reinterpret_cast<float2*>(out + size_t(r0) * N + n) = make_float2(acc[i][j][0], acc[i][j][1]);
            if (r1 < M) *reinterpret_cast<float2*>(out + size_t(r1) * N + n) = make_float2(acc[i][j][2], acc[i][j][3]);
        }
    }
}

// out[M][N] bf16 = (sum over splits of partial) * scale[n] (+ bias[n]); scale carries the 2^8.
__global__ void w8a16_reduce_kernel(const float* __restrict__ partial, const float* __restrict__ scale,
                                    const __nv_bfloat16* __restrict__ bias, __nv_bfloat16* __restrict__ out,
                                    int MN, int N, int splits) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= MN) return;
    float2 s = *reinterpret_cast<const float2*>(partial + i);
    for (int k = 1; k < splits; ++k) {
        const float2 p = *reinterpret_cast<const float2*>(partial + size_t(k) * MN + i);
        s.x += p.x; s.y += p.y;
    }
    const int n = i % N;
    s.x *= scale[n]; s.y *= scale[n + 1];
    if (bias != nullptr) { s.x += __bfloat162float(bias[n]); s.y += __bfloat162float(bias[n + 1]); }
    *reinterpret_cast<__nv_bfloat162*>(out + i) = __floats2bfloat162_rn(s.x, s.y);
}

// bf16 -> fp16 activation pre-pass (8 values per thread).
__global__ void w8a16_convert_a_kernel(const __nv_bfloat16* __restrict__ x, __half* __restrict__ y, int n8) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 v = reinterpret_cast<const uint4*>(x)[i];
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint4 o; __half2* h2 = reinterpret_cast<__half2*>(&o);
#pragma unroll
    for (int j = 0; j < 4; ++j) h2[j] = __float22half2_rn(__bfloat1622float2(b2[j]));
    reinterpret_cast<uint4*>(y)[i] = o;
}

// Pack W[N, K] fp8 (row-major, k contiguous) into fragment order (see header).
__global__ void w8a16_pack_kernel(const uint8_t* __restrict__ w, uint8_t* __restrict__ wp, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per (k_tile, n_quad, lane, q)
    const int KT = K / 16, NQ = N / 32;
    if (idx >= KT * NQ * 32 * 4) return;
    const int q = idx & 3, lane = (idx >> 2) & 31, nq = (idx >> 7) % NQ, kt = (idx >> 7) / NQ;
    const int n = nq * 32 + q * 8 + (lane >> 2), kb = kt * 16 + (lane & 3) * 2;
    const uint8_t* row = w + size_t(n) * K;
    uint8_t* dst = wp + size_t(idx) * 4;
    dst[0] = row[kb]; dst[1] = row[kb + 1]; dst[2] = row[kb + 8]; dst[3] = row[kb + 9];
}

// Inverse of the pack for the prefill path: W[N, K] bf16 = fp8 * scale (scale carries 2^8: divide it out).
__global__ void w8a16_dequant_kernel(const uint8_t* __restrict__ wp, const float* __restrict__ scale,
                                     __nv_bfloat16* __restrict__ w, int N, int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // one thread per packed 4-byte word
    const int KT = K / 16, NQ = N / 32;
    if (idx >= KT * NQ * 32 * 4) return;
    const int q = idx & 3, lane = (idx >> 2) & 31, nq = (idx >> 7) % NQ, kt = (idx >> 7) / NQ;
    const int n = nq * 32 + q * 8 + (lane >> 2), kb = kt * 16 + (lane & 3) * 2;
    const unsigned p = *reinterpret_cast<const unsigned*>(wp + size_t(idx) * 4);
    unsigned b0, b1;
    fp8x4_to_f16x2x2(p, b0, b1);
    const float sc = scale[n];   // includes 2^8 -> fp16 value * 2^-8 * sc = true value
    const __half2 h0 = *reinterpret_cast<const __half2*>(&b0), h1 = *reinterpret_cast<const __half2*>(&b1);
    const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
    __nv_bfloat16* row = w + size_t(n) * K;
    row[kb] = __float2bfloat16_rn(f0.x * sc); row[kb + 1] = __float2bfloat16_rn(f0.y * sc);
    row[kb + 8] = __float2bfloat16_rn(f1.x * sc); row[kb + 9] = __float2bfloat16_rn(f1.y * sc);
}

}  // namespace tms::w8a16
