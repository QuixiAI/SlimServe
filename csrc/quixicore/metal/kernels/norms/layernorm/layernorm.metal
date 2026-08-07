#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// LayerNorm (forward), bf16 I/O, fp32 compute.
//
// One simdgroup (32 lanes) processes one row of length D. The whole row fits
// in registers (D/32 floats per lane), so there is no threadgroup memory, no
// async pipeline, and no threadgroup_barrier: every cross-lane exchange happens
// inside the warp-level `sum` reduction via simd shuffles.
//
//   y = (x - mean(x)) * rsqrt(var(x) + eps) * weight + bias
//
// This is a faithful-but-minimal port of ThunderKittens kernels/layernorm:
// dropout and the fused residual add are intentionally dropped for v1.
// ---------------------------------------------------------------------------
template <int D>
kernel void layernorm(device   bf16  *x      [[buffer(0)]],
                      device   bf16  *weight [[buffer(1)]],
                      device   bf16  *bias   [[buffer(2)]],
                      device   bf16  *o      [[buffer(3)]],
                      constant uint  &M      [[buffer(4)]],   // total rows = prod(shape[:-1])
                      constant float &eps    [[buffer(5)]],
                      uint3 blockIdx [[threadgroup_position_in_grid]],
                      uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D) — rows runtime, cols compile-time
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weight / bias
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                   // naive layout, fp32 compute
    vecD xv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);  // bf16 -> fp32 on the fly
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);

    // mean
    float mean = 0.f;
    sum(mean, xv, laneId);
    mean /= (float)D;
    sub(xv, xv, mean);                       // x - mean (vec - scalar, broadcast)

    // variance
    float var = 0.f;
    mul(sq, xv, xv);                         // (x - mean)^2
    sum(var, sq, laneId);
    var /= (float)D;
    float inv = metal::rsqrt(var + eps);     // scalar rsqrt (no substrate op needed)

    // normalize, scale, shift
    mul(xv, xv, inv);                        // * 1/std (vec - scalar)
    mul(xv, xv, wv);                         // * weight (channelwise vec - vec)
    add(xv, xv, bv);                         // + bias   (channelwise vec - vec)
    store(gl_o, xv, {0, 0, row, 0}, laneId); // fp32 -> bf16 on the fly
}

#define instantiate_layernorm(DVAL)                                            \
  template [[host_name("layernorm_" #DVAL)]] [[kernel]] void                   \
  layernorm<DVAL>(device   bf16  *x      [[buffer(0)]],                        \
                  device   bf16  *weight [[buffer(1)]],                        \
                  device   bf16  *bias   [[buffer(2)]],                        \
                  device   bf16  *o      [[buffer(3)]],                        \
                  constant uint  &M      [[buffer(4)]],                        \
                  constant float &eps    [[buffer(5)]],                        \
                  uint3 blockIdx [[threadgroup_position_in_grid]],             \
                  uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_layernorm(256);
instantiate_layernorm(512);
instantiate_layernorm(768);
instantiate_layernorm(1024);

// Dynamic-D BF16 LayerNorm for Swin patch embedding/final patch merging and
// other widths divisible by four. One simdgroup owns a row; device traffic is
// vectorized as bf16_4 while reductions remain fp32.
[[host_name("layernorm_dyn_bfloat16")]]
kernel void layernorm_dyn(device const bf16 *x [[buffer(0)]],
                          device const bf16 *weight [[buffer(1)]],
                          device const bf16 *bias [[buffer(2)]],
                          device bf16 *o [[buffer(3)]],
                          constant uint &M [[buffer(4)]],
                          constant float &eps [[buffer(5)]],
                          constant int &D [[buffer(6)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint laneId [[thread_index_in_simdgroup]]) {
    const long base = (long)blockIdx.x * D;
    const int nchunks = D / 4;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int c = int(laneId); c < nchunks; c += 32) {
        const float4 value = float4(((device const bf16_4 *)(x + base))[c]);
        sum += value.x + value.y + value.z + value.w;
        sumsq += metal::dot(value, value);
    }
    sum = metal::simd_sum(sum);
    sumsq = metal::simd_sum(sumsq);
    const float mean = sum / float(D);
    const float variance = metal::max(sumsq / float(D) - mean * mean, 0.0f);
    const float inverse = metal::rsqrt(variance + eps);
    for (int c = int(laneId); c < nchunks; c += 32) {
        const float4 value = float4(((device const bf16_4 *)(x + base))[c]);
        const float4 scale = float4(((device const bf16_4 *)weight)[c]);
        const float4 shift = float4(((device const bf16_4 *)bias)[c]);
        ((device bf16_4 *)(o + base))[c] = bf16_4((value - mean) * inverse * scale + shift);
    }
}

// LayerNorm backward, dX only (dW = sum_rows dY*x_hat, dB = sum_rows dY are framework reductions).
// With g = dY*W and x_hat = (x-mean)*rstd:  dX_i = rstd*(g_i - mean_j(g) - x_hat_i*mean_j(g*x_hat)).
// mean/rstd (rows,) precomputed in the framework. One simdgroup per row; any D; T templated.
template <typename T>
kernel void layernorm_bwd_dx(device const T     *x    [[buffer(0)]],   // (rows, D)
                             device const T     *w    [[buffer(1)]],   // (D,)
                             device const T     *dy   [[buffer(2)]],   // (rows, D)
                             device const float *mean [[buffer(3)]],   // (rows,)
                             device const float *rstd [[buffer(4)]],   // (rows,)
                             device T           *dx   [[buffer(5)]],   // (rows, D)
                             constant int &D          [[buffer(6)]],
                             uint row  [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    const float mu = mean[row];
    const float r = rstd[row];
    float s1 = 0.0f, s2 = 0.0f;
    for (int j = (int)lane; j < D; j += 32) {
        const float g = float(dy[base + j]) * float(w[j]);
        const float xhat = (float(x[base + j]) - mu) * r;
        s1 += g;
        s2 += g * xhat;
    }
    s1 = metal::simd_sum(s1) / float(D);
    s2 = metal::simd_sum(s2) / float(D);
    for (int j = (int)lane; j < D; j += 32) {
        const float g = float(dy[base + j]) * float(w[j]);
        const float xhat = (float(x[base + j]) - mu) * r;
        dx[base + j] = T(r * (g - s1 - xhat * s2));
    }
}

// Fully-fused LayerNorm backward: one simdgroup per row computes mean+rstd in-kernel, writes dX, and
// accumulates dweight[j] += dY*x_hat and dbias[j] += dY via device float atomics (both (D,) fp32,
// zeroed first) — one fused kernel instead of the framework mean/rstd + dX kernel + framework
// dweight/dbias passes. Wave-7 #1.
template <typename T>
kernel void layernorm_bwd_fused(device const T *x    [[buffer(0)]],   // (rows, D)
                                device const T *w    [[buffer(1)]],   // (D,)
                                device const T *dy   [[buffer(2)]],   // (rows, D)
                                device T       *dx   [[buffer(3)]],   // (rows, D)
                                device metal::atomic_float *dweight [[buffer(4)]],  // (D,) zeroed
                                device metal::atomic_float *dbias   [[buffer(5)]],  // (D,) zeroed
                                constant int   &D    [[buffer(6)]],
                                constant float &eps  [[buffer(7)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * D;
    float sx = 0.0f, sxx = 0.0f;                       // mean + variance
    for (int j = (int)lane; j < D; j += 32) {
        const float xv = float(x[base + j]);
        sx += xv; sxx += xv * xv;
    }
    sx = metal::simd_sum(sx); sxx = metal::simd_sum(sxx);
    const float mu = sx / float(D);
    const float r  = metal::rsqrt(sxx / float(D) - mu * mu + eps);
    float s1 = 0.0f, s2 = 0.0f;                        // sum g, sum g*x_hat
    for (int j = (int)lane; j < D; j += 32) {
        const float g = float(dy[base + j]) * float(w[j]);
        const float xhat = (float(x[base + j]) - mu) * r;
        s1 += g; s2 += g * xhat;
    }
    s1 = metal::simd_sum(s1) / float(D);
    s2 = metal::simd_sum(s2) / float(D);
    for (int j = (int)lane; j < D; j += 32) {
        const float dyv = float(dy[base + j]);
        const float g = dyv * float(w[j]);
        const float xhat = (float(x[base + j]) - mu) * r;
        dx[base + j] = T(r * (g - s1 - xhat * s2));
        atomic_add_float(dweight, (long)j, dyv * xhat);   // dweight[j] += dY*x_hat
        atomic_add_float(dbias,   (long)j, dyv);          // dbias[j]   += dY
    }
}

#define instantiate_layernorm_bwd(type_name, T)                                \
  template [[host_name("layernorm_bwd_dx_" #type_name)]] [[kernel]] void         \
  layernorm_bwd_dx<T>(device const T *x [[buffer(0)]], device const T *w [[buffer(1)]], \
    device const T *dy [[buffer(2)]], device const float *mean [[buffer(3)]],   \
    device const float *rstd [[buffer(4)]], device T *dx [[buffer(5)]],         \
    constant int &D [[buffer(6)]],                                             \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("layernorm_bwd_fused_" #type_name)]] [[kernel]] void      \
  layernorm_bwd_fused<T>(device const T *x [[buffer(0)]], device const T *w [[buffer(1)]], \
    device const T *dy [[buffer(2)]], device T *dx [[buffer(3)]],               \
    device metal::atomic_float *dweight [[buffer(4)]],                          \
    device metal::atomic_float *dbias [[buffer(5)]], constant int &D [[buffer(6)]], \
    constant float &eps [[buffer(7)]],                                         \
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_layernorm_bwd(float32, float)
instantiate_layernorm_bwd(float16, half)
instantiate_layernorm_bwd(bfloat16, bf16)

}
