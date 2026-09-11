// SPDX-License-Identifier: Apache-2.0
#pragma once
// Qualified GLM53 SM120 BF16 prefill partials. This is the unchanged kernel
// from mhc_prefill_tc_probe SHA45d5e817; only its namespace is different.
// FP32 dot accumulation order differs from scalar partials. See the independent
// accuracy contract and full census before changing its arithmetic.
#include "mhc_ampere.cuh"
#include "bf16_decode_gemm.cuh"

namespace tms::glm53_mhc_prefill_tc {
using namespace tms::dsv4_mhc;
using namespace tms::decode_gemm;
constexpr int H = 4096, FLATS = 512, ROW = 520, TILE = 32;
constexpr int BYTES = (MIXES + TILE) * ROW * sizeof(__nv_bfloat16);
static_assert(HC == 4 && SPLITS == 32 && MIXES == 24);
static_assert(ROW % 8 == 0 && BYTES < 64 * 1024);

template <bool FUSED>
__global__ void __launch_bounds__(256) partials_tc(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const float* __restrict__ post,
    const float* __restrict__ comb,
    const __nv_bfloat16* __restrict__ fn,
    __nv_bfloat16* __restrict__ residual_out,
    float* __restrict__ partial, int tokens) {
    extern __shared__ __align__(16) unsigned char raw[];
    auto* ft = reinterpret_cast<__nv_bfloat16*>(raw);
    auto* vt = ft + MIXES * ROW;
    const int split = blockIdx.x, first = blockIdx.y * TILE;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int dim_split = split * 128;

    // Stage the original four stream slices without a persistent repack.
    for (int vec = tid; vec < MIXES * FLATS / 8; vec += 256) {
        const int output = vec / (FLATS / 8);
        const int flat = (vec % (FLATS / 8)) * 8;
        const int stream = flat / 128, dim = dim_split + flat % 128;
        cp_async16(smem_u32(ft + output * ROW + flat),
                   fn + output * (HC * H) + stream * H + dim, true);
    }
    cp_async_commit();

    // The same vectorized fused post-mix expression/order as partials_prefill.
    // Eight threads per token, two 8-wide chunks per thread/stream. Invalid
    // token rows are explicitly zeroed because warp MMA reads full tiles.
    const int t = tid >> 3, g = tid & 7, token = first + t;
    auto* vr = vt + t * ROW;
    if (token < tokens) {
        if constexpr (FUSED) {
            float pm[HC], cm[HC * HC];
#pragma unroll
            for (int i = 0; i < HC; ++i) pm[i] = post[token * HC + i];
#pragma unroll
            for (int i = 0; i < HC * HC; ++i) cm[i] = comb[token * HC * HC + i];
#pragma unroll
            for (int c = 0; c < 2; ++c) {
                const int dim = dim_split + g * 16 + c * 8;
                const uint4 xv = *reinterpret_cast<const uint4*>(x + size_t(token) * H + dim);
                uint4 rv[HC];
#pragma unroll
                for (int i = 0; i < HC; ++i)
                    rv[i] = *reinterpret_cast<const uint4*>(
                        residual + (size_t(token) * HC + i) * H + dim);
                const auto* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
#pragma unroll
                for (int out_stream = 0; out_stream < HC; ++out_stream) {
                    __align__(16) __nv_bfloat16 rounded[8];
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        float value = pm[out_stream] * float(xb[j]);
#pragma unroll
                        for (int in_stream = 0; in_stream < HC; ++in_stream) {
                            const auto* rb = reinterpret_cast<const __nv_bfloat16*>(&rv[in_stream]);
                            value += cm[in_stream * HC + out_stream] * float(rb[j]);
                        }
                        rounded[j] = __float2bfloat16_rn(value);
                    }
                    *reinterpret_cast<uint4*>(
                        residual_out + (size_t(token) * HC + out_stream) * H + dim) =
                        *reinterpret_cast<const uint4*>(rounded);
                    *reinterpret_cast<uint4*>(vr + out_stream * 128 + g * 16 + c * 8) =
                        *reinterpret_cast<const uint4*>(rounded);
                }
            }
        } else {
#pragma unroll
            for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int dim = dim_split + g * 16 + c * 8;
                    *reinterpret_cast<uint4*>(vr + stream * 128 + g * 16 + c * 8) =
                        *reinterpret_cast<const uint4*>(
                            residual + (size_t(token) * HC + stream) * H + dim);
                }
            }
        }
    } else {
#pragma unroll
        for (int stream = 0; stream < HC; ++stream) {
#pragma unroll
            for (int c = 0; c < 2; ++c)
                *reinterpret_cast<uint4*>(vr + stream * 128 + g * 16 + c * 8) =
                    make_uint4(0, 0, 0, 0);
        }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Six warps cover 2 token tiles x 3 mix tiles. Use the proven BF16
    // decode-GEMM fragment mapping; 520-element rows align every ldmatrix
    // address to 16 bytes. The seventh warp preserves the old square-sum
    // order; no BF16-rounded norm or approximate reduction is introduced.
    if (warp < 6) {
        const int m0 = (warp / 3) * 16, n0 = (warp % 3) * 8;
        const int mat = lane >> 3, row = lane & 7;
        float acc[4] = {};
#pragma unroll 4
        for (int k = 0; k < FLATS; k += 16) {
            uint32_t a[4], b[2];
            ldmatrix_x4(a, smem_u32(vt + (m0 + row + 8 * (mat & 1)) * ROW + k + 8 * (mat >> 1)));
            ldmatrix_x2(b, smem_u32(ft + (n0 + row) * ROW + k + 8 * (mat & 1)));
            mma_bf16_16816(acc, a, b[0], b[1]);
        }
        const int r0 = first + m0 + (lane >> 2), col = n0 + (lane & 3) * 2;
        if (r0 < tokens) {
            float* dst = partial + (size_t(r0) * SPLITS + split) * (MIXES + 1);
            dst[col] = acc[0];
            dst[col + 1] = acc[1];
        }
        if (r0 + 8 < tokens) {
            float* dst = partial + (size_t(r0 + 8) * SPLITS + split) * (MIXES + 1);
            dst[col] = acc[2];
            dst[col + 1] = acc[3];
        }
    } else if (warp == 6) {
        float sum = 0.0f;
        const auto* src = vt + lane * ROW;
#pragma unroll 8
        for (int k = 0; k < FLATS; k += 2) {
            const float2 v = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(src + k));
            sum += v.x * v.x;
            sum += v.y * v.y;
        }
        if (first + lane < tokens)
            partial[(size_t(first + lane) * SPLITS + split) * (MIXES + 1) + MIXES] = sum;
    }
}
}  // namespace tms::glm53_mhc_prefill_tc
