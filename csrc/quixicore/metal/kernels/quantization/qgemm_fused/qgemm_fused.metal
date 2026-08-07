#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// K2 (docs/new-kernels.md §3, built on request past its profiling gate) — fused
// per-token int8 activation quant + BitNet W2A8 GEMM. quantize_per_token_int8 +
// qgemm_w2a8 composed cost one full (M, K) int8 round-trip through device memory;
// here the codes live in threadgroup memory for exactly one token.
//
// One threadgroup (4 simdgroups, 128 threads) per token m:
//   phase 1: cooperative absmax over x[m,:], s = amax/127, encode into tg memory
//   phase 2: warps stride the N output rows, each simdgroup one row per iteration
//            (the same lane-per-k block walk as qgemm_w2a8)
// Output D (M, N) half, epilogue scale = half(s) — the same half-rounded grid as
// the composed path, so the two are bit-comparable. K <= 8192 (threadgroup array).
// ---------------------------------------------------------------------------

#define QGF_K_MAX 8192

template <typename T>
kernel void qgemm_w2a8_fused(device half        *D  [[buffer(0)]],   // (M, N)
                             device const uchar *Wq [[buffer(1)]],   // (N, K/32) bitnet blocks
                             device const T     *X  [[buffer(2)]],   // (M, K)
                             constant int &N [[buffer(3)]],
                             constant int &K [[buffer(4)]],
                             uint3 tgid [[threadgroup_position_in_grid]],
                             uint3 tid  [[thread_position_in_threadgroup]],
                             uint  warp [[simdgroup_index_in_threadgroup]],
                             uint  lane [[thread_index_in_simdgroup]]) {
    threadgroup char  xq[QGF_K_MAX];
    threadgroup float warp_amax[4];
    const long xbase = (long)tgid.x * K;

    // phase 1: per-token absmax -> int8 codes in threadgroup memory
    float amax = 0.0f;
    for (int i = (int)tid.x; i < K; i += 128) { amax = max(amax, fabs(float(X[xbase + i]))); }
    amax = simd_max(amax);
    if (lane == 0) { warp_amax[warp] = amax; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = max(max(warp_amax[0], warp_amax[1]), max(warp_amax[2], warp_amax[3]));
    const float s   = amax / 127.0f;
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;
    for (int i = (int)tid.x; i < K; i += 128) { xq[i] = tk_int8_encode(float(X[xbase + i]) * inv); }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // phase 2: each simdgroup walks output rows with stride 4
    const int bpr = K / bitnet::block_k;
    const float sh = float(half(s));                 // the composed path's f16 a_scale grid
    for (int n = (int)warp; n < N; n += 4) {
        device const uchar* row = Wq + (long)n * bpr * bitnet::block_bytes;
        float lane_acc = 0.0f;
        for (int g = 0; g < bpr; ++g) {
            device const uchar* base = row + (long)g * bitnet::block_bytes;
            const int prod = bitnet::code(base, (int)lane)
                           * (int)xq[g * bitnet::block_k + (int)lane];
            lane_acc += float(prod) * float(bitnet::gscale(base));
        }
        const float acc = metal::simd_sum(lane_acc);
        if (lane == 0) { D[(long)tgid.x * N + n] = half(acc * sh); }
    }
}

#define instantiate_qgemm_w2a8_fused(type_name, T)                              \
  template [[host_name("qgemm_w2a8_fused_" #type_name)]] [[kernel]] void        \
  qgemm_w2a8_fused<T>(device half *D [[buffer(0)]],                             \
                      device const uchar *Wq [[buffer(1)]],                     \
                      device const T *X [[buffer(2)]],                          \
                      constant int &N [[buffer(3)]],                            \
                      constant int &K [[buffer(4)]],                            \
                      uint3 tgid [[threadgroup_position_in_grid]],              \
                      uint3 tid [[thread_position_in_threadgroup]],              \
                      uint warp [[simdgroup_index_in_threadgroup]],             \
                      uint lane [[thread_index_in_simdgroup]]);

instantiate_qgemm_w2a8_fused(float32, float)
instantiate_qgemm_w2a8_fused(float16, half)
instantiate_qgemm_w2a8_fused(bfloat16, bf16)
