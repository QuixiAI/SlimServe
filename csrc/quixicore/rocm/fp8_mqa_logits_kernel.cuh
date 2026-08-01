/**
 * @file
 * @brief DSA indexer MQA logits on CDNA3, fp8 MFMA. See
 * fp8_mqa_logits_design.md for how the layout and epilogue order were
 * established.
 *
 *   score[h,n] = relu( (SUM_d q[m,h,d] * k[n,d]) * kscale[n] )
 *   logits[m,n] = SUM_h score[h,n] * weights[m,h]
 *
 * One block per query row m, walking that row's [start, end) in BLOCK_KV tiles.
 * The d-contraction rides v_mfma_f32_16x16x32_fp8_fp8, the instruction Triton
 * emits, which is what makes the dot bitwise-equal (mfma_dot_probe checks that
 * in isolation).
 *
 * The tile is computed **transposed** -- D[n][h] rather than D[h][n] -- by
 * feeding KV as the A operand and Q as B. That is not a stylistic choice: it is
 * `isTransposed = true` in Triton's mma layout, and it decides the epilogue's
 * arithmetic. With D[n][h], one lane holds h = l%16 (+16 per h-tile) and
 * n = 4*(l/16)+v, so per output column a lane owns exactly two heads. That is
 * the `v_mul_f32` + `v_fmac_f32` pair in Triton's ISA: the per-head weight
 * multiply is FUSED into the accumulate, one rounding rather than the two a
 * separate multiply and add would take. The remaining 16 heads of a wave's
 * range are then folded by an xor butterfly over lane%16 (offsets 8,4,2,1 --
 * the DPP stages), and the two head-halves finally meet through LDS.
 *
 * The butterfly runs 1,2,4,8 ASCENDING. The full DPP inventory in the generated
 * ISA is quad_perm[1,0,3,2] (xor 1), quad_perm[2,3,0,1] (xor 2), row_shr/shl:4
 * and row_shr/shl:8 -- 16 of each, i.e. one per stage per column per dot site.
 * Reading only the first few instructions of the listing suggests 8 comes
 * first, but that is scheduler interleaving across independent columns, not the
 * dependency order; descending measured strictly worse.
 *
 * NOT YET BITWISE. Measured against Triton at M=64,H=64,D=128,N=256, all of the
 * residue being accumulation order since mfma_dot_probe shows the dot itself is
 * exact:
 *   QC_EPI=0 (h-tile 0 plain, 1 fused) 5768/16384, descending butterfly 6601
 *   QC_EPI=1 (h-tile 1 plain, 0 fused) 5484/16384   <- current
 *   QC_EPI=2 (separate multiplies)     5484/16384   -- identical to 1 because
 *       -ffp-contract=fast re-forms the FMA, so this does not actually test an
 *       unfused epilogue; use -ffp-contract=off to make that variant meaningful.
 * max|diff| 3.1e-5, mean 1.8e-6 throughout.
 */
#pragma once
#ifndef QC_EPI
  #define QC_EPI 1
#endif
#include "mfma_fp8_dot.cuh"

#include <cstdint>

namespace qcrocm {

// NWAVE waves arranged [2 heads, 2 kv], matching warpsPerCTA = [2, 2].
template <int NUM_HEADS, int HEAD_SIZE, int BLOCK_KV, int NWAVE>
__global__ __launch_bounds__(64 * NWAVE) void fp8_mqa_logits(
    const uint8_t* __restrict__ q,        // [M, H, D] e4m3
    const uint8_t* __restrict__ kv,       // [N, D]    e4m3
    const float* __restrict__ kv_scales,  // [N]
    const float* __restrict__ weights,    // [M, H]
    const int* __restrict__ cu_start, const int* __restrict__ cu_end,
    float* __restrict__ logits,  // [M, N] fp32, pre-filled -inf
    long stride_q_s, long stride_w_s, long stride_logits_s, int seq_len_kv) {
    constexpr int WH = 2;                        // waves over heads
    constexpr int WN = NWAVE / WH;               // waves over kv
    constexpr int H_PER_WAVE = NUM_HEADS / WH;   // 32
    constexpr int N_PER_WAVE = BLOCK_KV / WN;    // 32
    constexpr int H_TILES = H_PER_WAVE / 16;     // 2
    constexpr int N_TILES = N_PER_WAVE / 16;     // 2

    const int row = blockIdx.x;
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int wh = wave % WH, wn = wave / WH;
    const int lo = lane & 15, hi = lane >> 4;

    int start = cu_start[row];
    int end = cu_end[row];
    start = start < 0 ? 0 : start;
    end = end > seq_len_kv ? seq_len_kv : end;
    if (start >= end) return;

    const uint8_t* q_row = q + (long)row * stride_q_s;
    const float* w_row = weights + (long)row * stride_w_s;
    float* out_row = logits + (long)row * stride_logits_s;

    __shared__ float part[WH][BLOCK_KV];

    for (int tile = start; tile < end; tile += BLOCK_KV) {
        // Warps repeat cyclically over each dim (tile j at j*16*W + w*16),
        // which is how Triton's mma layout covers a 64-wide dim with 2 warps.
        const int n_base = tile + wn * 16;

        // acc[i][j] = D[n from tile i][h from tile j], transposed on purpose.
        f32x4 acc[N_TILES][H_TILES];
#pragma unroll
        for (int i = 0; i < N_TILES; ++i)
#pragma unroll
            for (int j = 0; j < H_TILES; ++j) acc[i][j] = f32x4{0.f, 0.f, 0.f, 0.f};

#pragma unroll
        for (int k = 0; k < HEAD_SIZE; k += 32) {
#pragma unroll
            for (int i = 0; i < N_TILES; ++i) {
                const int n0 = n_base + i * 16 * WN;
                // A = KV rows; out-of-range rows contribute zero.
                const long a =
                    (n0 + lo < end) ? load_a_frag(kv, HEAD_SIZE, n0, k) : 0L;
#pragma unroll
                for (int j = 0; j < H_TILES; ++j)
                    acc[i][j] = mfma_16x16x32_fp8(
                        a, load_b_frag(q_row, HEAD_SIZE, j * 16 * WH + wh * 16, k),
                        acc[i][j]);
            }
        }

#pragma unroll
        for (int i = 0; i < N_TILES; ++i) {
#pragma unroll
            for (int v = 0; v < 4; ++v) {
                const int n = n_base + i * 16 * WN + 4 * hi + v;
                const float ks = (n < end) ? kv_scales[n] : 0.0f;

                // Two heads per lane per column: plain multiply then a fused
                // multiply-add, exactly as Triton's v_mul/v_fmac pair.
                float sum = 0.0f;
#if QC_EPI == 0   // h-tile 0 plain, h-tile 1 fused
#pragma unroll
                for (int j = 0; j < H_TILES; ++j) {
                    const float t = fmaxf(acc[i][j][v] * ks, 0.0f);
                    const float w = w_row[j * 16 * WH + wh * 16 + lo];
                    sum = (j == 0) ? (w * t) : fmaf(w, t, sum);
                }
#elif QC_EPI == 1  // h-tile 1 plain, h-tile 0 fused
#pragma unroll
                for (int j = H_TILES - 1; j >= 0; --j) {
                    const float t = fmaxf(acc[i][j][v] * ks, 0.0f);
                    const float w = w_row[j * 16 * WH + wh * 16 + lo];
                    sum = (j == H_TILES - 1) ? (w * t) : fmaf(w, t, sum);
                }
#else              // no fusion: separate multiplies then an add
                {
                    float acc0 = 0.0f;
#pragma unroll
                    for (int j = 0; j < H_TILES; ++j) {
                        const float t = fmaxf(acc[i][j][v] * ks, 0.0f);
                        acc0 += w_row[j * 16 * WH + wh * 16 + lo] * t;
                    }
                    sum = acc0;
                }
#endif
                // Fold the wave's remaining heads across lane%16.
                sum += __shfl_xor(sum, 1, 64);
                sum += __shfl_xor(sum, 2, 64);
                sum += __shfl_xor(sum, 4, 64);
                sum += __shfl_xor(sum, 8, 64);
                if (lo == 0) part[wh][n - tile] = sum;
            }
        }
        __syncthreads();

        for (int c = threadIdx.x; c < BLOCK_KV; c += blockDim.x) {
            const int n = tile + c;
            if (n >= end) continue;
            float s = part[0][c];
#pragma unroll
            for (int t = 1; t < WH; ++t) s += part[t][c];
            out_row[n] = s;
        }
        __syncthreads();
    }
}

}  // namespace qcrocm
