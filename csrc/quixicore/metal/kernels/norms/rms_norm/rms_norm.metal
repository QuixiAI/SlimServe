#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// RMSNorm (forward), bf16 I/O, fp32 compute.
//
//   y = x * rsqrt(mean(x^2) + eps) * weight
//
// Like LayerNorm but with no mean-subtraction and no bias. One simdgroup (32
// lanes) processes one row of length D; the whole row fits in registers, so
// there is no threadgroup memory and no barrier (cross-lane reduction is via
// simd shuffles inside `sum`).
// ---------------------------------------------------------------------------
template <int D>
kernel void rms_norm(device   bf16  *x      [[buffer(0)]],
                     device   bf16  *weight [[buffer(1)]],
                     device   bf16  *o      [[buffer(2)]],
                     constant uint  &M      [[buffer(3)]],   // total rows = prod(shape[:-1])
                     constant float &eps    [[buffer(4)]],
                     uint3 blockIdx [[threadgroup_position_in_grid]],
                     uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D)
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weight
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                   // naive layout, fp32 compute
    vecD xv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);  // bf16 -> fp32 on the fly
    load(wv, gl_w, {0, 0, 0,   0}, laneId);

    // mean of squares
    float ms = 0.f;
    mul(sq, xv, xv);
    sum(ms, sq, laneId);
    ms /= (float)D;
    float inv = metal::rsqrt(ms + eps);      // scalar rsqrt

    // normalize, scale
    mul(xv, xv, inv);                        // * 1/rms (vec - scalar)
    mul(xv, xv, wv);                         // * weight (channelwise vec - vec)
    store(gl_o, xv, {0, 0, row, 0}, laneId); // fp32 -> bf16 on the fly
}

#define instantiate_rms_norm(DVAL)                                            \
  template [[host_name("rms_norm_" #DVAL)]] [[kernel]] void                   \
  rms_norm<DVAL>(device   bf16  *x      [[buffer(0)]],                        \
                 device   bf16  *weight [[buffer(1)]],                        \
                 device   bf16  *o      [[buffer(2)]],                        \
                 constant uint  &M      [[buffer(3)]],                        \
                 constant float &eps    [[buffer(4)]],                        \
                 uint3 blockIdx [[threadgroup_position_in_grid]],             \
                 uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_rms_norm(256);
instantiate_rms_norm(512);
instantiate_rms_norm(768);
instantiate_rms_norm(1024);

// Dynamic-D RMSNorm for wider or non-instantiated widths. One simdgroup handles
// a row with two strided passes over bf16_4 chunks. Requires D % 4 == 0.
kernel void rms_norm_dyn(device   bf16  *x      [[buffer(0)]],
                         device   bf16  *weight [[buffer(1)]],
                         device   bf16  *o      [[buffer(2)]],
                         constant uint  &M      [[buffer(3)]],
                         constant float &eps    [[buffer(4)]],
                         constant int   &D      [[buffer(5)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    const long base = (long)blockIdx.x * D;
    const int nchunks = D / 4;
    float ss = 0.0f;
    for (int c = (int)laneId; c < nchunks; c += 32) {
        const float4 v = float4(((device const bf16_4*)(x + base))[c]);
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = metal::simd_sum(ss);
    const float inv = metal::rsqrt(ss / (float)D + eps);
    for (int c = (int)laneId; c < nchunks; c += 32) {
        const float4 v = float4(((device const bf16_4*)(x + base))[c]) * inv;
        const float4 w = float4(((device const bf16_4*)(weight))[c]);
        ((device bf16_4*)(o + base))[c] = bf16_4(v * w);
    }
}

// RMSNorm backward, dX only (dW = sum_rows dY*x*rstd is a cheap framework reduction). Per row, with
// m = dY*W and s = sum_j m_j x_j:  dX_i = rstd*m_i - (rstd^3 * s / D) * x_i  (Liger's factorization).
// rstd (rows,) is precomputed in the framework. One simdgroup per row; any D; T templated.
template <typename T>
kernel void rms_norm_bwd_dx(device const T     *x    [[buffer(0)]],   // (rows, D)
                            device const T     *w    [[buffer(1)]],   // (D,)
                            device const T     *dy   [[buffer(2)]],   // (rows, D)
                            device const float *rstd [[buffer(3)]],   // (rows,)
                            device T           *dx   [[buffer(4)]],   // (rows, D)
                            constant int &D          [[buffer(5)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    const float r = rstd[row];
    float s = 0.0f;
    for (int j = (int)lane; j < D; j += 32) {
        s += (float(dy[base + j]) * float(w[j])) * float(x[base + j]);
    }
    s = metal::simd_sum(s);
    const float c = r * r * r * s / float(D);
    for (int j = (int)lane; j < D; j += 32) {
        const float m = float(dy[base + j]) * float(w[j]);
        dx[base + j] = T(r * m - c * float(x[base + j]));
    }
}

// Fully-fused RMSNorm backward: one simdgroup per row computes rstd IN-KERNEL (sum x^2) alongside the
// dX combine sum (sum dY*W*x) in a single pass, writes dX, and accumulates dweight[j] += dY*x*rstd via
// a device float atomic (dweight (D,) fp32 zeroed first). Removes the framework rstd + dweight passes
// (one read of x/dY/W instead of three). Wave-7 #1: measure the dweight-atomic contention vs mx.fast.
template <typename T>
kernel void rms_norm_bwd_fused(device const T     *x    [[buffer(0)]],   // (rows, D)
                               device const T     *w    [[buffer(1)]],   // (D,)
                               device const T     *dy   [[buffer(2)]],   // (rows, D)
                               device T           *dx   [[buffer(3)]],   // (rows, D)
                               device metal::atomic_float *dweight [[buffer(4)]],  // (D,) zeroed
                               constant int   &D        [[buffer(5)]],
                               constant float &eps      [[buffer(6)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    float ssq = 0.0f, s = 0.0f;                       // sum x^2 (for rstd) and sum dY*W*x (for dX)
    for (int j = (int)lane; j < D; j += 32) {
        const float xv = float(x[base + j]);
        ssq += xv * xv;
        s   += (float(dy[base + j]) * float(w[j])) * xv;
    }
    ssq = metal::simd_sum(ssq);
    s   = metal::simd_sum(s);
    const float r = metal::rsqrt(ssq / float(D) + eps);
    const float c = r * r * r * s / float(D);
    for (int j = (int)lane; j < D; j += 32) {
        const float xv = float(x[base + j]);
        const float m  = float(dy[base + j]) * float(w[j]);
        dx[base + j] = T(r * m - c * xv);
        atomic_add_float(dweight, (long)j, float(dy[base + j]) * xv * r);   // dweight[j] += dY*x*rstd
    }
}

#define instantiate_rms_norm_bwd(type_name, T)                                 \
  template [[host_name("rms_norm_bwd_dx_" #type_name)]] [[kernel]] void         \
  rms_norm_bwd_dx<T>(device const T *x [[buffer(0)]], device const T *w [[buffer(1)]], \
    device const T *dy [[buffer(2)]], device const float *rstd [[buffer(3)]],  \
    device T *dx [[buffer(4)]], constant int &D [[buffer(5)]],                  \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("rms_norm_bwd_fused_" #type_name)]] [[kernel]] void      \
  rms_norm_bwd_fused<T>(device const T *x [[buffer(0)]], device const T *w [[buffer(1)]], \
    device const T *dy [[buffer(2)]], device T *dx [[buffer(3)]],               \
    device metal::atomic_float *dweight [[buffer(4)]], constant int &D [[buffer(5)]], \
    constant float &eps [[buffer(6)]],                                         \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_rms_norm_bwd(float32, float)
instantiate_rms_norm_bwd(float16, half)
instantiate_rms_norm_bwd(bfloat16, bf16)

}
