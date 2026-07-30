/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's quantized MoE grouped GEMMs (moe.metal)
 * on the mma-fragment path (m16n8k16 tensor-core via tm_qmm.cuh), consuming the
 * existing tmq/tmoe permuted-expert schedule (expert_of_tile per 16/32-row tile,
 * A already gathered into expert-contiguous padded rows by tmoe::moe_gather).
 *
 *   moe_gemm_fp8   : B = e4m3 (E,N,K), rowwise scale B_scale[e*N+n], A = fp16.
 *   moe_gemm_nvfp4 : both operands fp4 e2m1, per-16-block e4m3 scales
 *                    (A swizzled per (row,group), B plain per (e,n,group)),
 *                    per-expert alpha.
 *   moe_gemm_wna16 : uint32-packed int4/int8 (GPTQ/AWQ de-interleave order),
 *                    per-(e,n,group) scale + optional packed zero-point, A = fp16.
 *
 * SwiGLU is NOT fused here (matches MetalForge) — it lives in the fused
 * activation-quant stage between GEMM1 and GEMM2 (see silu_and_mul_* below).
 *
 * Build (standalone harness): see moe_quant_test.cu header.
 */
#pragma once
#include "../quant/tm_qmm.cuh"           // load_wfrag/load_xfrag/mma16816, fp8_raw
#include "../quant/quant_formats.cuh"    // e4m3/e2m1 decode+encode, nvfp4_sf_offset
#include "../serving/tm_warp.cuh"        // warp_max_f (activation/experts quant)
#include <cuda_fp16.h>

namespace tmoeq {

using namespace tmq;

// Store the m16n8 accumulator halves with a per-output-column (n) scale.
// Column map (mma_ABt, from tm_qmm/qgemm): lane owns (row=m0+lane/4,
// col c0=n0+(lane%4)*2); acc0={c0,c0+1}, acc1={c0+8,c0+9}; rows +8 in acc[2,3].
__device__ __forceinline__ void store_scaled(float* Y, int N, int total_rows, int m0, int n0,
                                             const float acc0[4], const float acc1[4],
                                             float s0, float s1, float s8, float s9) {
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, c0 = n0 + (lane % 4) * 2;
    if (r < total_rows) {
        Y[size_t(r) * N + c0]         = acc0[0] * s0;
        Y[size_t(r) * N + c0 + 1]     = acc0[1] * s1;
        Y[size_t(r) * N + c0 + 8]     = acc1[0] * s8;
        Y[size_t(r) * N + c0 + 8 + 1] = acc1[1] * s9;
    }
    if (r + 8 < total_rows) {
        Y[size_t(r + 8) * N + c0]         = acc0[2] * s0;
        Y[size_t(r + 8) * N + c0 + 1]     = acc0[3] * s1;
        Y[size_t(r + 8) * N + c0 + 8]     = acc1[2] * s8;
        Y[size_t(r + 8) * N + c0 + 8 + 1] = acc1[3] * s9;
    }
}

// ---- fp8 e4m3 weight-only grouped GEMM. B (E,N,K) e4m3 bytes, rowwise fp32
// scale B_scale[e*N+n]. A = permuted fp16 (total_rows, K). Y (total_rows, N)
// fp32. grid (N/16, total_rows/32) — each block owns the full 32-row expert tile
// and reuses each dequantized B fragment across both 16-row M-subtiles (halves
// the weight load+dequant traffic). Padding tiles (expert_of_tile<0) exit. ----
__global__ void moe_gemm_fp8(float* __restrict__ Y, const half* __restrict__ A,
                             const uint8_t* __restrict__ B, const float* __restrict__ B_scale,
                             const int* __restrict__ expert_of_tile,
                             int total_rows, int N, int K) {
    const int n0 = blockIdx.x * 16, m0 = blockIdx.y * 32;
    const int e = expert_of_tile[blockIdx.y];
    if (e < 0) return;
    const uint8_t* Be = B + (size_t)e * N * K;      // fp8_raw: 1 byte/elem, (N,K)
    const float* Se = B_scale + (size_t)e * N;
    const int bpr = K / fp8_raw::block_k;           // K/128

    float accT0[4] = {0}, accT1[4] = {0}, accB0[4] = {0}, accB1[4] = {0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 aT[4], aB[4], b[4];
        load_xfrag(aT, A, K, m0, k0);
        load_xfrag(aB, A, K, m0 + 16, k0);
        load_wfrag<fp8_raw>(b, Be, bpr, n0, k0);    // dequant once, reused by aT and aB
        half2 b0[2] = {b[0], b[2]}, b1[2] = {b[1], b[3]};
        mma16816(accT0, aT, b0); mma16816(accT1, aT, b1);
        mma16816(accB0, aB, b0); mma16816(accB1, aB, b1);
    }
    const int c0 = n0 + (threadIdx.x % 4) * 2;
    store_scaled(Y, N, total_rows, m0,      n0, accT0, accT1, Se[c0], Se[c0 + 1], Se[c0 + 8], Se[c0 + 9]);
    store_scaled(Y, N, total_rows, m0 + 16, n0, accB0, accB1, Se[c0], Se[c0 + 1], Se[c0 + 8], Se[c0 + 9]);
}

// ---- nvfp4 dual-operand grouped GEMM. A & B both fp4 e2m1 packed 2/byte
// (row, K/2); per-16-block e4m3 scales: A-scale swizzled per (row_in_expert,
// group) with per-expert sf_offsets base, B-scale plain per (e,n,group);
// per-expert fp32 alpha. Both fp4 fragments dequant to fp16, mma16816, then
// ×(A_scale·B_scale) per (row,col,group) folded via the 16-wide k-block scale,
// ×alpha in the epilogue. Since block_k(16) == the mma k-step is 16, one
// A/B scale pair applies to the whole 16-wide fragment. ----
// Load a 16x16 fp4 A chunk into an mma A fragment, dequant to fp16, pre-scaled
// by the per-(row_in_expert, group) swizzled A-scale. `m0` = global row of the
// tile (for the packed bytes); `loc0` = row-in-expert of that global row (for
// the swizzled scale); the two differ by a per-expert constant.
__device__ __forceinline__ void load_afrag_nvfp4(half2 frag[4], const uint8_t* A,
        const uint8_t* Asc_e, int K, int num_k_tiles, int m0, int loc0, int k0) {
    const int lane = threadIdx.x & 31;
    const int r = m0 + lane / 4, lr = loc0 + lane / 4, c = k0 + (lane % 4) * 2;
    const int Kh = K / 2;
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int rr = r + (kk % 2) * 8, lrr = lr + (kk % 2) * 8, cc = c + (kk / 2) * 8;
        const int group = cc / 16;
        const float as = e4m3_decode(Asc_e[nvfp4_sf_offset(lrr, group, num_k_tiles)]);
        const uint8_t byte = A[(size_t)rr * Kh + cc / 2];   // cc even -> low nib=cc, high=cc+1
        frag[kk] = __floats2half2_rn(e2m1_decode(byte & 0xF) * as,
                                     e2m1_decode(byte >> 4) * as);
    }
}
// Load a 16x16 fp4 B chunk (W rows n0.., cols k0..), dequant to fp16 pre-scaled
// by the per-(n,group) B-scale (plain layout). Fragment convention as load_wfrag.
__device__ __forceinline__ void load_wfrag_nvfp4(half2 frag[4], const uint8_t* B,
        const uint8_t* Bsc_e, int K, int groups, int n0, int k0) {
    const int lane = threadIdx.x & 31;
    const int r = n0 + lane / 4, c = k0 + (lane % 4) * 2;
    const int Kh = K / 2;
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int rr = r + (kk % 2) * 8, cc = c + (kk / 2) * 8;
        const int group = cc / 16;
        const float bs = e4m3_decode(Bsc_e[(size_t)rr * groups + group]);
        const uint8_t byte = B[(size_t)rr * Kh + cc / 2];
        frag[kk] = __floats2half2_rn(e2m1_decode(byte & 0xF) * bs,
                                     e2m1_decode(byte >> 4) * bs);
    }
}

// grid (N/16, total_rows/32) — 32-row block; each dequantized B fragment is
// reused across both 16-row M-subtiles (halves the weight load+dequant). A_scale
// is a single swizzled buffer indexed by a per-expert sf_offsets[e] base (in
// units of padded_groups); the two subtiles differ only by loc0 (+16). Padding
// tiles exit.
__global__ void moe_gemm_nvfp4(float* __restrict__ Y, const uint8_t* __restrict__ A,
        const uint8_t* __restrict__ B, const uint8_t* __restrict__ A_scale,
        const uint8_t* __restrict__ B_scale, const float* __restrict__ alphas,
        const int* __restrict__ expert_of_tile, const int* __restrict__ expert_row0,
        const int* __restrict__ sf_offsets, int total_rows, int N, int K) {
    const int n0 = blockIdx.x * 16, m0 = blockIdx.y * 32;
    const int e = expert_of_tile[blockIdx.y];
    if (e < 0) return;
    const int groups = K / 16, num_k_tiles = (K + 63) / 64, padded_groups = num_k_tiles * 4;
    const uint8_t* Asc_e = A_scale + (size_t)sf_offsets[e] * padded_groups;
    const uint8_t* Bsc_e = B_scale + ((size_t)e * N) * groups;
    const uint8_t* Be = B + ((size_t)e * N) * (K / 2);
    const int loc0 = m0 - expert_row0[e];                   // row-in-expert of global row m0

    float accT0[4] = {0}, accT1[4] = {0}, accB0[4] = {0}, accB1[4] = {0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 aT[4], aB[4], b[4];
        load_afrag_nvfp4(aT, A, Asc_e, K, num_k_tiles, m0,      loc0,      k0);
        load_afrag_nvfp4(aB, A, Asc_e, K, num_k_tiles, m0 + 16, loc0 + 16, k0);
        load_wfrag_nvfp4(b, Be, Bsc_e, K, groups, n0, k0);    // dequant once, reused
        half2 b0[2] = {b[0], b[2]}, b1[2] = {b[1], b[3]};
        mma16816(accT0, aT, b0); mma16816(accT1, aT, b1);
        mma16816(accB0, aB, b0); mma16816(accB1, aB, b1);
    }
    const float al = alphas[e];
    store_scaled(Y, N, total_rows, m0,      n0, accT0, accT1, al, al, al, al);
    store_scaled(Y, N, total_rows, m0 + 16, n0, accB0, accB1, al, al, al, al);
}

// ---- WNA16 (GPTQ/AWQ weight-only int4/int8) grouped GEMM. Per expert, weight
// qweight is uint32-packed (E,N,K/pack) with the vLLM de-interleave order
// {0,2,4,6,1,3,5,7} (int4) / {0,2,1,3} (int8); scale scales[e,n,k/group_size]
// (T dtype); optional packed zero-point qzeros (default zp 8/int4, 128/int8).
// value = (q - zp) * scale. A = fp16. Both bit widths via BIT template. ----
template <int BIT>
__device__ __forceinline__ float wna16_deq(const uint32_t* qw, size_t n_base, int k,
                                            const uint8_t* qz, size_t z_base, int local_out,
                                            int group, int groups, bool has_zp,
                                            float scale) {
    constexpr int PACK = 32 / BIT;
    const int inner = k & (PACK - 1);
    int local;
    if (BIT == 4) { const int ord[8] = {0,2,4,6,1,3,5,7}; local = ord[inner & 7]; }
    else          { const int ord[4] = {0,2,1,3};         local = ord[inner & 3]; }
    const uint32_t packed = qw[n_base + k / PACK];
    const float q = (BIT == 4) ? float((packed >> (local * 4)) & 0xFu)
                               : float((packed >> (local * 8)) & 0xFFu);
    float zp;
    if (!has_zp) zp = (BIT == 4) ? 8.0f : 128.0f;
    else if (BIT == 4) { const uint8_t p = qz[z_base + (local_out / 2) * groups + group];
                         zp = float((p >> ((local_out & 1) * 4)) & 0xFu); }
    else zp = float(qz[z_base + local_out * groups + group]);
    return (q - zp) * scale;
}

// Load a 16x16 W chunk (rows n0.., cols k0..) into an mma B fragment for WNA16.
// Same fragment convention as tm_qmm::load_wfrag: data[k]=rows+8*(k%2),cols+8*(k/2),
// lane owns (lane/4,(lane%4)*2). Weight is dequantized to fp16 per element.
template <int BIT>
__device__ __forceinline__ void load_wfrag_wna16(half2 frag[4], const uint32_t* qw,
        const uint8_t* qz, const half* scales, int N, int K, int groups, int group_size,
        bool has_zp, int n0, int k0) {
    constexpr int PACK = 32 / BIT;
    const int packed_k = K / PACK;
    const int lane = threadIdx.x & 31;
    const int r = n0 + lane / 4, c = k0 + (lane % 4) * 2;
    #pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        const int rr = r + (kk % 2) * 8, cc = c + (kk / 2) * 8;
        const size_t nb = (size_t)rr * packed_k;
        const int g0 = cc / group_size, g1 = (cc + 1) / group_size;
        const float s0 = __half2float(scales[(size_t)rr * groups + g0]);
        const float s1 = __half2float(scales[(size_t)rr * groups + g1]);
        const float w0 = wna16_deq<BIT>(qw, nb, cc,   qz, 0, rr, g0, groups, has_zp, s0);
        const float w1 = wna16_deq<BIT>(qw, nb, cc+1, qz, 0, rr, g1, groups, has_zp, s1);
        frag[kk] = __floats2half2_rn(w0, w1);
    }
}

// grid (N/16, total_rows/32) — 32-row block; the (relatively expensive) WNA16
// de-interleave+dequant of each B fragment is reused across both 16-row
// M-subtiles. Padding tiles exit.
template <int BIT>
__global__ void moe_gemm_wna16(float* __restrict__ Y, const half* __restrict__ A,
        const uint32_t* __restrict__ qweight, const half* __restrict__ scales,
        const uint8_t* __restrict__ qzeros, const int* __restrict__ expert_of_tile,
        int total_rows, int N, int K, int group_size, int has_zp) {
    constexpr int PACK = 32 / BIT;
    const int n0 = blockIdx.x * 16, m0 = blockIdx.y * 32;
    const int e = expert_of_tile[blockIdx.y];
    if (e < 0) return;
    const int groups = K / group_size;
    const uint32_t* qw = qweight + (size_t)e * N * (K / PACK);
    const half* sc = scales + (size_t)e * N * groups;
    const uint8_t* qz = qzeros
        ? qzeros + (BIT == 4 ? (size_t)e * ((N + 1) / 2) * groups : (size_t)e * N * groups)
        : nullptr;

    float accT0[4] = {0}, accT1[4] = {0}, accB0[4] = {0}, accB1[4] = {0};
    for (int k0 = 0; k0 < K; k0 += 16) {
        half2 aT[4], aB[4], b[4];
        load_xfrag(aT, A, K, m0, k0);
        load_xfrag(aB, A, K, m0 + 16, k0);
        load_wfrag_wna16<BIT>(b, qw, qz, sc, N, K, groups, group_size, has_zp != 0, n0, k0);
        half2 b0[2] = {b[0], b[2]}, b1[2] = {b[1], b[3]};
        mma16816(accT0, aT, b0); mma16816(accT1, aT, b1);
        mma16816(accB0, aB, b0); mma16816(accB1, aB, b1);
    }
    store_scaled(Y, N, total_rows, m0,      n0, accT0, accT1, 1.0f, 1.0f, 1.0f, 1.0f);
    store_scaled(Y, N, total_rows, m0 + 16, n0, accB0, accB1, 1.0f, 1.0f, 1.0f, 1.0f);
}

// ===========================================================================
// Fused SiLU-and-mul activation with quantized output (the GEMM1->GEMM2 stage).
// Input is [gate | up] = (num_tokens, 2*hidden); out = quant(silu(gate)*up).
// ===========================================================================
__device__ __forceinline__ float mq_silu_mul(float g, float u) {
    return (g / (1.0f + expf(-g))) * u;
}

// Static per-tensor scale (MetalForge silu_and_mul_fp8_quant). 1 thread/elem.
template <typename T, bool FP8>
__global__ void silu_and_mul_quant_static(uint8_t* __restrict__ out, const T* __restrict__ input,
                                          float inv_scale, int hidden, long n_tok) {
    const long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_tok * hidden) return;
    const long token = i / hidden, col = i - token * hidden;
    const T* row = input + token * hidden * 2;
    const float v = mq_silu_mul(float(row[col]), float(row[hidden + col])) * inv_scale;
    out[i] = FP8 ? tmq::e4m3_encode(v) : uint8_t(int8_t(fmaxf(-127.f, fminf(127.f, rintf(v)))));
}

// Dynamic per-block scale (MetalForge silu_and_mul_per_block_{fp8,int8}_quant):
// block = `group_size` contiguous output cols; scale = amax/QMAX (fp8 448 w/ a
// MIN_SCALE floor, int8 127). One warp per (token, group). scales -> [token,group].
// Reuses the rms_norm_add_k DYN idiom. (Transposed [group,token] scale layout is
// a follow-up when the block-scaled consumer GEMM is ported.)
template <typename T, bool FP8>
__global__ void silu_and_mul_quant_perblock(uint8_t* __restrict__ out, float* __restrict__ scales,
                                            const T* __restrict__ input, int hidden, int group_size,
                                            int n_groups) {
    const int token = blockIdx.y, g = blockIdx.x, lane = threadIdx.x;
    const int c0 = g * group_size;
    const T* row = input + (long)token * hidden * 2;
    float amax = 0.0f;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        amax = fmaxf(amax, fabsf(mq_silu_mul(float(row[c]), float(row[hidden + c]))));
    amax = tms::warp_max_f(amax);
    const float qmax = FP8 ? 448.0f : 127.0f;
    const float minsc = FP8 ? (1.0f / (448.0f * 512.0f)) : 1e-12f;
    const float scale = fmaxf(amax / qmax, minsc);
    const float inv = 1.0f / scale;
    if (lane == 0) scales[(long)token * n_groups + g] = scale;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32) {
        const float v = mq_silu_mul(float(row[c]), float(row[hidden + c])) * inv;
        out[(long)token * hidden + c] = FP8 ? tmq::e4m3_encode(v)
                                            : uint8_t(int8_t(fmaxf(-127.f, fminf(127.f, rintf(v)))));
    }
}

// ===========================================================================
// per_token_group_quant_fp8 (activation quantizer feeding the block-scaled GEMM):
// per contiguous group_size block: scale = amax/448 (optional ue8m0 rounding),
// encode e4m3. One warp per (token, group). scales -> [token, group] fp32.
// ===========================================================================
template <typename T, bool UE8M0>
__global__ void per_token_group_quant_fp8(uint8_t* __restrict__ out, float* __restrict__ scales,
                                          const T* __restrict__ input, int hidden,
                                          int group_size, int n_groups, float eps) {
    const int token = blockIdx.y, g = blockIdx.x, lane = threadIdx.x;
    const int c0 = g * group_size;
    const T* row = input + (long)token * hidden;
    float amax = 0.0f;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        amax = fmaxf(amax, fabsf(float(row[c])));
    amax = tms::warp_max_f(fmaxf(amax, eps));
    float scale = amax / 448.0f;
    if (UE8M0) scale = e8m0_decode(e8m0_encode(scale));    // round to power of two
    const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (lane == 0) scales[(long)token * n_groups + g] = scale;
    for (int c = c0 + lane; c < c0 + group_size && c < hidden; c += 32)
        out[(long)token * hidden + c] = tmq::e4m3_encode(float(row[c]) * inv);
}

// ===========================================================================
// nvfp4 experts-quant: quantize activations to fp4 + swizzled e4m3 block scales
// in the expert-segmented layout consumed by moe_gemm_nvfp4. Per 16-block:
// amax -> sf = e4m3(global_scale[e]*amax/6); codes = e2m1(x / (sf/global_scale)).
// Optional FUSE_SILU reads [gate|up]. One warp per (row, group); packs 2 nibbles/byte.
// ===========================================================================
template <typename T, bool FUSE_SILU>
__global__ void nvfp4_experts_quant(uint8_t* __restrict__ out_codes, uint8_t* __restrict__ out_sc,
        const T* __restrict__ input, const float* __restrict__ global_scale,
        const int* __restrict__ expert_of_row, const int* __restrict__ expert_row0,
        const int* __restrict__ sf_offsets, int K, int num_k_tiles) {
    const int row = blockIdx.y, g = blockIdx.x, lane = threadIdx.x;  // one warp/(row,group)
    const int e = expert_of_row[row];
    if (e < 0) return;
    const int c0 = g * 16, Kh = K / 2, in_stride = FUSE_SILU ? K * 2 : K;
    const T* r = input + (long)row * in_stride;
    auto val = [&](int c) -> float {
        return FUSE_SILU ? mq_silu_mul(float(r[c]), float(r[K + c])) : float(r[c]);
    };
    float amax = 0.0f;
    for (int c = c0 + lane; c < c0 + 16; c += 32) amax = fmaxf(amax, fabsf(val(c)));
    amax = tms::warp_max_f(amax);
    const float gs = global_scale[e];
    const float sf_f = gs * amax / 6.0f;
    const uint8_t sf_code = tmq::e4m3_encode(sf_f);
    const float inv = sf_f > 0.0f ? gs / e4m3_decode(sf_code) : 0.0f;  // 1/(sf/gs)
    // lane 0 writes the swizzled scale byte
    if (lane == 0) {
        const int padded_groups = num_k_tiles * 4;
        const int lr = row - expert_row0[e];
        out_sc[(size_t)sf_offsets[e] * padded_groups + nvfp4_sf_offset(lr, g, num_k_tiles)] = sf_code;
    }
    // pack 2 codes/byte for cols [c0, c0+16); 8 bytes, lanes 0..7 own one byte pair
    for (int p = lane; p < 8; p += 32) {
        const int c = c0 + p * 2;
        const uint8_t lo = e2m1_encode(val(c) * inv), hi = e2m1_encode(val(c + 1) * inv);
        out_codes[(size_t)row * Kh + c / 2] = uint8_t((hi << 4) | lo);
    }
}

// ===========================================================================
// Scored top-k routing (MetalForge moe_topk_score / sigmoid / softplus_sqrt):
// score each expert logit (MODE 0 sigmoid, 1 sqrt(softplus)), masked top-k
// select (smaller-id ties), weight = scored * routed_scaling_factor / (renorm ?
// sum-of-selected-scored : 1). One warp per token. Complements the softmax-renorm
// tmoe::moe_route_topk (the Mixtral rule). Generic E, K<=16.
// ===========================================================================
__device__ __forceinline__ float moe_score(int mode, float x) {
    if (mode == 0) return 1.0f / (1.0f + expf(-x));
    const float y = x > 20.0f ? x : logf(1.0f + expf(x));   // softplus, threshold-guarded
    return sqrtf(y);
}
template <typename T>
__global__ void moe_route_scored(const T* __restrict__ logits, int* __restrict__ topk_ids,
                                 float* __restrict__ topk_w, int E, int K, int mode,
                                 int renormalize, float routed_scaling_factor) {
    const long base = (long)blockIdx.x * E;
    const int lane = threadIdx.x;
    int chosen[16];
    float chosen_v[16];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = idx; v = moe_score(mode, float(logits[base + idx])); valid = true;
    };
    tms::masked_topk(cand, E, K, lane, -3.4028234663852886e38f, chosen, chosen_v);
    float sum = 0.0f;
    if (renormalize) for (int k = 0; k < K; ++k) sum += chosen_v[k];
    const float denom = (renormalize && sum > 0.0f) ? sum : 1.0f;
    const float scale = routed_scaling_factor / denom;
    if (lane == 0) {
        const long ob = (long)blockIdx.x * K;
        for (int k = 0; k < K; ++k) {
            topk_ids[ob + k] = chosen[k];
            topk_w[ob + k] = chosen_v[k] * scale;
        }
    }
}

} // namespace tmoeq
