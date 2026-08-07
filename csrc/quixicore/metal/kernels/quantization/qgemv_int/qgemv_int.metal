#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Integer-path quantized GEMV (batch-1 decode), int8xint8 -> int32 accumulate, then scale.
// Apple has no int8 simdgroup_matrix, so this is the GEMV (per-lane idot4 + simd_sum) shape.
// Prefill stays on the dequant-to-half MMA (half tensor cores beat a hand-rolled int dot).

// ---- W8A8 / SmoothQuant: int8 weights (per-channel scale) x int8 activations (per-token scale).
//   out[n] = w_scale[n] * a_scale * sum_k Wq[n,k] * Xq[k]   (scales factor out of the int32 sum). ----
kernel void qgemv_w8a8(
    device   half*  D       [[buffer(0)]],   // (N, 1) output
    device   char*  Wq      [[buffer(1)]],   // (N, K) int8 weights, row-major
    device   char*  Xq      [[buffer(2)]],   // (K, 1) int8 activations
    device   half*  w_scale [[buffer(3)]],   // (N,) per-channel weight scale
    device   half*  a_scale [[buffer(4)]],   // (1,) per-tensor activation scale (one decode token)
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    // Two rows per simdgroup: X is loaded once (uint4 = 16 int8) and dotted against both weight
    // rows — halves the X register loads and doubles per-threadgroup work at half the grid.
    const int row0 = tgid.x * 2;
    const bool two = row0 + 1 < N;
    device const uint4* w0 = (device const uint4*)(Wq + (uint)row0 * K);
    device const uint4* w1 = (device const uint4*)(Wq + (uint)(row0 + (two ? 1 : 0)) * K);
    device const uint4* xv = (device const uint4*)Xq;
    int acc0 = 0, acc1 = 0;
    for (int u = (int)lane; u < K / 16; u += 32) {                         // 16-byte loads, 4x dp4a
        const uint4 x = xv[u];
        const uint4 a = w0[u];
        acc0 += idot4(a.x, x.x) + idot4(a.y, x.y) + idot4(a.z, x.z) + idot4(a.w, x.w);
        const uint4 b = w1[u];
        acc1 += idot4(b.x, x.x) + idot4(b.y, x.y) + idot4(b.z, x.z) + idot4(b.w, x.w);
    }
    for (int k = (K & ~15) + (int)lane * 4; k + 4 <= K; k += 128) {        // K%16 tail (K%4==0)
        const uint x = ((device const uint*)Xq)[k / 4];
        acc0 += idot4(((device const uint*)Wq)[((uint)row0 * K + k) / 4], x);
        acc1 += idot4(((device const uint*)Wq)[((uint)(row0 + (two ? 1 : 0)) * K + k) / 4], x);
    }
    acc0 = metal::simd_sum(acc0);                                          // int32 reduce
    acc1 = metal::simd_sum(acc1);
    if (lane == 0) {
        D[row0] = half(float(acc0) * float(w_scale[row0]) * float(a_scale[0]));
        if (two) D[row0 + 1] = half(float(acc1) * float(w_scale[row0 + 1]) * float(a_scale[0]));
    }
}

// ---- BitNet W2A8: ternary 2-bit weights (per-group absmean scale) x int8 activations.
//   out[n] = a_scale * sum_g gscale[g] * sum_{k in g} (code-1) * Xq[k]   (per-group int32 sums). ----
kernel void qgemv_w2a8(
    device   half*  D       [[buffer(0)]],   // (N, 1) output
    device   uchar* Wq      [[buffer(1)]],   // (N, K/32) bitnet blocks
    device   char*  Xq      [[buffer(2)]],   // (K, 1) int8 activations
    device   half*  a_scale [[buffer(3)]],   // (1,) per-tensor activation scale
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / bitnet::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * bitnet::block_bytes;
    // Block-major walk (same shape as the dequant qgemv): each lane owns 8 contiguous codes of a
    // block (one 2-byte read), 8 blocks in flight per simdgroup iteration; the group scale is
    // applied ONCE per 8-code span to the integer subtotal. The old one-block-per-iteration walk
    // did a float multiply per element.
    constexpr int CPL = 8;                              // codes per lane
    constexpr int LPB = bitnet::block_k / CPL;          // 4 lanes per block
    constexpr int BPI = 32 / LPB;                       // 8 blocks per iteration
    const int b_off = (int)lane / LPB;
    const int col0  = ((int)lane % LPB) * CPL;
    float lane_acc = 0.0f;
    for (int g = b_off; g < bpr; g += BPI) {
        device const uchar* base = row_base + (uint)g * bitnet::block_bytes;
        const ushort codes = *(device const ushort*)(base + 2 + (col0 >> 2));  // 8 x 2-bit codes
        device const char* x = Xq + g * bitnet::block_k + col0;
        int isum = 0;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            isum += (int)((codes >> (2 * i)) & 3) * (int)x[i];   // code in {0,1,2}
        }
        int ixsum = 0;                                            // subtract the -1 bias: code-1
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) ixsum += (int)x[i];
        lane_acc += float(isum - ixsum) * float(bitnet::gscale(base));
    }
    const float facc = metal::simd_sum(lane_acc);
    if (lane == 0) D[row] = half(facc * float(a_scale[0]));
}

// W2A8 v2: one whole BitNet block per lane, streaming contiguous weight blocks
// and applying each group scale once per 32-code subtotal.
kernel void qgemv_w2a8_v2(
    device   half*  D       [[buffer(0)]],
    device   uchar* Wq      [[buffer(1)]],
    device   char*  Xq      [[buffer(2)]],
    device   half*  a_scale [[buffer(3)]],
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int row = tgid.x;
    const int bpr = K / bitnet::block_k;
    device const uchar* row_base = Wq + (uint)(row * bpr) * bitnet::block_bytes;
    float lane_acc = 0.0f;
    for (int g = (int)lane; g < bpr; g += 32) {
        device const uchar* base = row_base + (uint)g * bitnet::block_bytes;
        const half gscale = ((device const half*)base)[0];
        const uint4 x0 = ((device const uint4*)(Xq + g * 32))[0];
        const uint4 x1 = ((device const uint4*)(Xq + g * 32))[1];
        int isum = 0, xsum = 0;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const uint b2 = (uint)((device const ushort*)(base + 2))[j];
            const uint lo = b2 & 0xFFu, hi = (b2 >> 8) & 0xFFu;
            const uint s0 = (lo | (lo << 6) | (lo << 12) | (lo << 18)) & 0x03030303u;
            const uint s1 = (hi | (hi << 6) | (hi << 12) | (hi << 18)) & 0x03030303u;
            const uint xa = (j < 2) ? x0[2 * j] : x1[2 * (j - 2)];
            const uint xb = (j < 2) ? x0[2 * j + 1] : x1[2 * (j - 2) + 1];
            isum += idot4(s0, xa) + idot4(s1, xb);
            xsum += idot4(0x01010101u, xa) + idot4(0x01010101u, xb);
        }
        lane_acc += float(isum - xsum) * float(gscale);
    }
    const float facc = metal::simd_sum(lane_acc);
    if (lane == 0) D[row] = half(facc * float(a_scale[0]));
}

}
