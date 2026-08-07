#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// quantize_tq2_0 — on-device pack into llama.cpp's native ternary TQ2_0 GGUF
// blocks (the plans' upstream export route), bit-compatible with ggml's
// quantize_row_tq2_0_ref (read from the reference tree 2026-07-07):
//
//   per 256-element block: d = ABSMAX (stored half, LAST: {uint8 qs[64]; half d});
//   code = lround(x/d) + 1 in {0,1,2}; element 128j + 32n + m -> qs[32j+m] bits 2n.
//
// Emits BOTH the packed blocks (for qgemm/qgemv/moe "tq2_0" and parity tooling)
// and the dequantized bf16 tensor from the SAME half-rounded scale.
// Rounding: metal::round = round-half-away == lroundf. On per-tensor-baked
// {-s,0,+s} input the ratios are exactly {-1,0,+1}, so codes AND the block scale
// reproduce the exporter's output bit-for-bit (the §8.2 preserve regime).
//
// Batched over an expert axis via grid.y ((E,N,K); E=1 = plain 2-D). One
// simdgroup per row; lane L owns bytes 32j+L of each block (8 elements:
// j in {0,1} x n in {0,3}), so absmax is one simd_max and each lane packs its
// two bytes with no cross-lane traffic. K % 256 == 0.
// ---------------------------------------------------------------------------

template <typename T>
kernel void quantize_tq2_0(device const T *W     [[buffer(0)]],
                           device uchar   *wq    [[buffer(1)]],
                           device bf16    *w_deq [[buffer(2)]],
                           constant int   &K     [[buffer(3)]],
                           constant int   &N     [[buffer(4)]],
                           uint3 tgid [[threadgroup_position_in_grid]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    const uint row = tgid.x, e = tgid.y;
    const long base = ((long)e * N + row) * K;
    const int  nblocks = K / 256;
    device uchar* row_blocks = wq + ((long)e * N + row) * nblocks * 66;

    for (int b = 0; b < nblocks; ++b) {
        const long bbase = base + (long)b * 256;
        float x[2][4];                                   // [j][n] = W[128j + 32n + lane]
        float amax = 0.0f;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 2; ++j) {
            #pragma clang loop unroll(full)
            for (int n = 0; n < 4; ++n) {
                x[j][n] = float(W[bbase + 128 * j + 32 * n + (int)lane]);
                amax = max(amax, fabs(x[j][n]));
            }
        }
        const float d = simd_max(amax);                  // ggml: d = absmax, no epsilon
        // ggml multiplies by the CORRECTLY-ROUNDED f32 reciprocal (id = 1.0f/d);
        // fast-math rcp is a ulp off, which flips codes exactly at the x == d/2
        // ties bf16-grid inputs hit. One fma Newton step re-rounds it.
        const float id0 = d > 0.0f ? 1.0f / d : 0.0f;
        const float id  = metal::fma(metal::fma(-id0, d, 1.0f), id0, id0);
        const half  dh  = half(d);                       // GGML_FP32_TO_FP16(d)

        device uchar* blk = row_blocks + (long)b * 66;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 2; ++j) {
            uint q = 0;
            #pragma clang loop unroll(full)
            for (int n = 0; n < 4; ++n) {
                // lroundf = round HALF AWAY; fast-math metal::round sends the
                // exact +/-0.5 ties (hit by bf16-grid inputs) to 0 — hand-roll it.
                // |v| <= 1 so floor(|v| + 0.5) is exact float arithmetic.
                const float v = x[j][n] * id;
                const int xi = (v < 0.0f ? -1 : 1) * int(metal::floor(metal::fabs(v) + 0.5f));
                q |= uint(xi + 1) << (2 * n);
                w_deq[bbase + 128 * j + 32 * n + (int)lane] = bf16(float(dh) * float(xi));
            }
            blk[32 * j + (int)lane] = uchar(q);
        }
        if (lane == 0) { ((device half*)(blk + 64))[0] = dh; }   // 66*b + 64: 2-aligned
    }
}

#define instantiate_quantize_tq2_0(type_name, T)                                \
  template [[host_name("quantize_tq2_0_" #type_name)]] [[kernel]] void         \
  quantize_tq2_0<T>(device const T *W [[buffer(0)]],                           \
                    device uchar *wq [[buffer(1)]],                            \
                    device bf16 *w_deq [[buffer(2)]],                          \
                    constant int &K [[buffer(3)]],                             \
                    constant int &N [[buffer(4)]],                             \
                    uint3 tgid [[threadgroup_position_in_grid]],               \
                    uint lane [[thread_index_in_simdgroup]]);

instantiate_quantize_tq2_0(float32, float)
instantiate_quantize_tq2_0(float16, half)
instantiate_quantize_tq2_0(bfloat16, bf16)
