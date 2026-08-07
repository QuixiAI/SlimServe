#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Integer-accumulate PREFILL GEMM (M>1), int8xint8 -> int32, then scale. Apple has no int8
// simdgroup_matrix, so this is a tiled int dot (idot4 + simd_sum), NOT a tensor-core MMA — it is
// the bit-exact int32 path (decode = exact, scaled once), expected SLOWER than the dequant-to-half
// MMA (which uses the half tensor cores). One simdgroup per output row, looping the M columns.
// Activations are token-major (M,K) so each token's K-vector is contiguous for idot4.

// ---- W8A8 / SmoothQuant prefill: int8 weight (per-channel scale) x int8 act (per-token scale). ----
kernel void qgemm_w8a8(
    device   half*  D       [[buffer(0)]],   // (N, M) output
    device   char*  Wq      [[buffer(1)]],   // (N, K) int8 weights, row-major
    device   char*  Xq      [[buffer(2)]],   // (M, K) int8 activations, token-major
    device   half*  w_scale [[buffer(3)]],   // (N,) per-channel
    device   half*  a_scale [[buffer(4)]],   // (M,) per-token
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    const constant int &M [[buffer(7)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int n = tgid.x;
    device const uint* wrow = (device const uint*)(Wq + (uint)n * K);
    const float wsc = float(w_scale[n]);
    for (int m = 0; m < M; m++) {
        device const uint* xrow = (device const uint*)(Xq + (uint)m * K);
        int acc = 0;
        for (int u = (int)lane; u < K / 4; u += 32) acc += idot4(wrow[u], xrow[u]);
        acc = metal::simd_sum(acc);
        if (lane == 0) D[(uint)n * M + m] = half(float(acc) * wsc * float(a_scale[m]));
    }
}

// ---- azp-aware W8A8: asymmetric int8 activations Xq = round(x/s_a) + azp[m]. Correction:
//   y[n,m] = s_w[n]*s_a[m] * ( sum_k W[n,k]*Xq[m,k]  -  azp[m] * sum_k W[n,k] )
// with w_rowsum (N,) int32 host-precomputed (tk.quant.quantize_w8a8 packer). ----
kernel void qgemm_w8a8_azp(
    device   half*  D       [[buffer(0)]],   // (N, M) output
    device   char*  Wq      [[buffer(1)]],   // (N, K) int8 weights, row-major
    device   char*  Xq      [[buffer(2)]],   // (M, K) int8 activations, token-major
    device   half*  w_scale [[buffer(3)]],   // (N,) per-channel
    device   float* a_scale [[buffer(4)]],   // (M,) per-token (azp quantizer emits f32)
    device   const int* w_rowsum [[buffer(5)]],  // (N,)
    device   const int* azp      [[buffer(6)]],  // (M,)
    const constant int &N [[buffer(7)]],
    const constant int &K [[buffer(8)]],
    const constant int &M [[buffer(9)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int n = tgid.x;
    device const uint* wrow = (device const uint*)(Wq + (uint)n * K);
    const float wsc = float(w_scale[n]);
    const int rsum = w_rowsum[n];
    for (int m = 0; m < M; m++) {
        device const uint* xrow = (device const uint*)(Xq + (uint)m * K);
        int acc = 0;
        for (int u = (int)lane; u < K / 4; u += 32) acc += idot4(wrow[u], xrow[u]);
        acc = metal::simd_sum(acc);
        if (lane == 0) {
            D[(uint)n * M + m] = half(float(acc - azp[m] * rsum) * wsc * a_scale[m]);
        }
    }
}

// ---- BitNet W2A8 prefill: ternary 2-bit weight (per-group absmean) x int8 act (per-token scale). ----
kernel void qgemm_w2a8(
    device   half*  D       [[buffer(0)]],   // (N, M)
    device   uchar* Wq      [[buffer(1)]],   // (N, K/32) bitnet blocks
    device   char*  Xq      [[buffer(2)]],   // (M, K) int8 token-major
    device   half*  a_scale [[buffer(3)]],   // (M,)
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int n = tgid.x;
    const int bpr = K / bitnet::block_k;
    device const uchar* row_base = Wq + (uint)(n * bpr) * bitnet::block_bytes;
    for (int m = 0; m < M; m++) {
        device const char* xrow = Xq + (uint)m * K;
        float lane_acc = 0.0f;
        for (int g = 0; g < bpr; g++) {
            device const uchar* base = row_base + (uint)g * bitnet::block_bytes;
            const int k = (int)lane;
            const int prod = bitnet::code(base, k) * (int)xrow[g * bitnet::block_k + k];
            lane_acc += float(prod) * float(bitnet::gscale(base));
        }
        const float facc = metal::simd_sum(lane_acc);
        if (lane == 0) D[(uint)n * M + m] = half(facc * float(a_scale[m]));
    }
}

}
