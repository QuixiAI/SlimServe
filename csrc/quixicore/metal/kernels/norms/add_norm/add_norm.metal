#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Fused residual-add + normalization, bf16 I/O, fp32 compute.
//
// Every decoder block boundary computes `norm(x + residual)` and then feeds the
// *summed* residual (x + residual) into the next block. Fusing the add into the
// norm avoids an extra materialized add + global read/write of the hidden state.
//
// Two outputs:
//   o       = norm(x + residual) * weight (+ bias for LayerNorm)
//   res_out = x + residual                (the value the next block reads)
//
// Same register-resident, one-simdgroup-per-row structure as rms_norm/layernorm:
// the whole row of length D lives in registers (D/32 fp32 per lane), so there is
// no threadgroup memory and no barrier — the only cross-lane exchange is the
// warp-level `sum` reduction via simd shuffles. D ∈ {256,512,768,1024}.
//
// Ref: vLLM fused_add_rms_norm (layernorm_quant_kernels.cu), ONNX Runtime
// SkipLayerNorm (skip_layer_norm_impl.cu).
// ---------------------------------------------------------------------------
template <int D>
kernel void rms_norm_add(device   bf16  *x        [[buffer(0)]],
                         device   bf16  *residual  [[buffer(1)]],
                         device   bf16  *weight    [[buffer(2)]],
                         device   bf16  *o         [[buffer(3)]],
                         device   bf16  *res_out   [[buffer(4)]],
                         constant uint  &M         [[buffer(5)]],   // total rows = prod(shape[:-1])
                         constant float &eps       [[buffer(6)]],
                         uint3 blockIdx [[threadgroup_position_in_grid]],
                         uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D)
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weight
    row_gl gl_x(x,        nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o,        nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                   // naive layout, fp32 compute
    vecD xv, rv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);  // bf16 -> fp32 on the fly
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);

    add(xv, xv, rv);                         // x + residual
    store(gl_ro, xv, {0, 0, row, 0}, laneId); // write back x + residual (fp32 -> bf16)

    // mean of squares over the summed residual
    float ms = 0.f;
    mul(sq, xv, xv);
    sum(ms, sq, laneId);
    ms /= (float)D;
    float inv = metal::rsqrt(ms + eps);      // scalar rsqrt

    mul(xv, xv, inv);                        // * 1/rms (vec - scalar)
    mul(xv, xv, wv);                         // * weight (channelwise vec - vec)
    store(gl_o, xv, {0, 0, row, 0}, laneId); // fp32 -> bf16 on the fly
}

template <int D>
kernel void layernorm_add(device   bf16  *x        [[buffer(0)]],
                          device   bf16  *residual  [[buffer(1)]],
                          device   bf16  *weight    [[buffer(2)]],
                          device   bf16  *bias      [[buffer(3)]],
                          device   bf16  *o         [[buffer(4)]],
                          device   bf16  *res_out   [[buffer(5)]],
                          constant uint  &M         [[buffer(6)]],
                          constant float &eps       [[buffer(7)]],
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x,        nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o,        nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight,   nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,     nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);

    add(xv, xv, rv);                         // x + residual
    store(gl_ro, xv, {0, 0, row, 0}, laneId); // write back x + residual

    // mean
    float mean = 0.f;
    sum(mean, xv, laneId);
    mean /= (float)D;
    sub(xv, xv, mean);                       // x - mean

    // variance
    float var = 0.f;
    mul(sq, xv, xv);
    sum(var, sq, laneId);
    var /= (float)D;
    float inv = metal::rsqrt(var + eps);

    mul(xv, xv, inv);                        // * 1/std
    mul(xv, xv, wv);                         // * weight
    add(xv, xv, bv);                         // + bias
    store(gl_o, xv, {0, 0, row, 0}, laneId);
}

// Exact materialized residual-add + LayerNorm semantics for one-token decode.
// The sum is rounded to T before statistics, matching a framework `x +
// residual` tensor followed by LayerNorm. Unlike the register-tiled fast path,
// this accepts dynamic D and both fp32 and bf16.
template <typename T>
kernel void decode_layernorm_add(
    device const T *input [[buffer(0)]],
    device const T *residual [[buffer(1)]],
    device const T *weight [[buffer(2)]],
    device const T *bias [[buffer(3)]],
    device T *normalized [[buffer(4)]],
    device T *summed [[buffer(5)]],
    constant int &rows [[buffer(6)]],
    constant int &dimension [[buffer(7)]],
    constant float &epsilon [[buffer(8)]],
    uint row [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    if (int(row) >= rows) return;
    const long base = (long)row * dimension;
    float sum = 0.0f;
    float sumsq = 0.0f;
    for (int index = int(lane); index < dimension; index += 32) {
        const T rounded = T(float(input[base + index]) + float(residual[base + index]));
        summed[base + index] = rounded;
        const float value = float(rounded);
        sum += value;
        sumsq += value * value;
    }
    sum = metal::simd_sum(sum);
    sumsq = metal::simd_sum(sumsq);
    const float mean = sum / float(dimension);
    const float variance = metal::max(sumsq / float(dimension) - mean * mean, 0.0f);
    const float inverse = metal::rsqrt(variance + epsilon);
    for (int index = int(lane); index < dimension; index += 32) {
        const float value = float(summed[base + index]);
        normalized[base + index] =
            T((value - mean) * inverse * float(weight[index]) + float(bias[index]));
    }
}

#define instantiate_decode_layernorm_add(type_name, T)                        \
  template [[host_name("decode_layernorm_add_" #type_name)]] [[kernel]]     \
  void decode_layernorm_add<T>(device const T *input [[buffer(0)]],          \
    device const T *residual [[buffer(1)]], device const T *weight [[buffer(2)]], \
    device const T *bias [[buffer(3)]], device T *normalized [[buffer(4)]],  \
    device T *summed [[buffer(5)]], constant int &rows [[buffer(6)]],        \
    constant int &dimension [[buffer(7)]], constant float &epsilon [[buffer(8)]], \
    uint row [[threadgroup_position_in_grid]],                               \
    uint lane [[thread_index_in_simdgroup]]);

instantiate_decode_layernorm_add(float32, float)
instantiate_decode_layernorm_add(bfloat16, bf16)

// ---------------------------------------------------------------------------
// fp8 e4m3 epilogue variants: emit uint8 codes = e4m3(norm(x+residual)*weight / scale)
// directly from the register-resident normed vector (no bf16 round-trip). res_out is
// still written in bf16. Two scale modes: static per-tensor (inv_scale passed in, matches
// vLLM fused_add_rms_norm_static_fp8) and dynamic per-row (absmax/448, scale output).
// The rv_fl naive layout maps element xv[w][0] to global column w*32+laneId (same map
// as store()), so codes[row*D + w*32+laneId] = tk_e4m3_encode(xv[w][0]*inv_scale).
// ---------------------------------------------------------------------------
template <int D>
kernel void rms_norm_add_fp8(device   bf16  *x         [[buffer(0)]],
                             device   bf16  *residual  [[buffer(1)]],
                             device   bf16  *weight    [[buffer(2)]],
                             device   uchar *codes     [[buffer(3)]],
                             device   bf16  *res_out   [[buffer(4)]],
                             constant uint  &M         [[buffer(5)]],
                             constant float &eps       [[buffer(6)]],
                             constant float &inv_scale [[buffer(7)]],
                             uint3 blockIdx [[threadgroup_position_in_grid]],
                             uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float ms = 0.f;
    mul(sq, xv, xv); sum(ms, sq, laneId); ms /= (float)D;
    mul(xv, xv, metal::rsqrt(ms + eps));
    mul(xv, xv, wv);

    device uchar *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) {
        op[w * 32 + laneId] = tk_e4m3_encode(xv[w][0] * inv_scale);
    }
}

template <int D>
kernel void rms_norm_add_fp8_dyn(device   bf16  *x        [[buffer(0)]],
                                 device   bf16  *residual [[buffer(1)]],
                                 device   bf16  *weight   [[buffer(2)]],
                                 device   uchar *codes    [[buffer(3)]],
                                 device   bf16  *res_out  [[buffer(4)]],
                                 device   float *scale    [[buffer(5)]],
                                 constant uint  &M        [[buffer(6)]],
                                 constant float &eps      [[buffer(7)]],
                                 uint3 blockIdx [[threadgroup_position_in_grid]],
                                 uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float ms = 0.f;
    mul(sq, xv, xv); sum(ms, sq, laneId); ms /= (float)D;
    mul(xv, xv, metal::rsqrt(ms + eps));
    mul(xv, xv, wv);

    float amax = 0.f;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) amax = metal::max(amax, metal::fabs(xv[w][0]));
    amax = metal::simd_max(amax);
    const float s = amax / 448.0f;
    const float inv_scale = s > 0.f ? 1.f / s : 0.f;
    if (laneId == 0) scale[row] = s;

    device uchar *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) {
        op[w * 32 + laneId] = tk_e4m3_encode(xv[w][0] * inv_scale);
    }
}

// int8 sibling of rms_norm_add_fp8_dyn (dynamic per-row symmetric int8 for W8A8).
template <int D>
kernel void rms_norm_add_int8_dyn(device   bf16  *x        [[buffer(0)]],
                                 device   bf16  *residual [[buffer(1)]],
                                 device   bf16  *weight   [[buffer(2)]],
                                 device   char  *codes    [[buffer(3)]],
                                 device   bf16  *res_out  [[buffer(4)]],
                                 device   float *scale    [[buffer(5)]],
                                 constant uint  &M        [[buffer(6)]],
                                 constant float &eps      [[buffer(7)]],
                                 uint3 blockIdx [[threadgroup_position_in_grid]],
                                 uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float ms = 0.f;
    mul(sq, xv, xv); sum(ms, sq, laneId); ms /= (float)D;
    mul(xv, xv, metal::rsqrt(ms + eps));
    mul(xv, xv, wv);

    float amax = 0.f;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) amax = metal::max(amax, metal::fabs(xv[w][0]));
    amax = metal::simd_max(amax);
    const float s = amax / 127.0f;
    const float inv_scale = s > 0.f ? 1.f / s : 0.f;
    if (laneId == 0) scale[row] = s;

    device char *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) {
        op[w * 32 + laneId] = tk_int8_encode(xv[w][0] * inv_scale);
    }
}

// ---------------------------------------------------------------------------
// Per-block (per-128-group) dynamic quant tail: the activation-side layout the
// block-quant expert GEMMs (moe_grouped_gemm_*_q) read directly — norm output is
// grouped into 128-wide blocks, each with its own scale (rows, D/128) row-major.
//
// The rv_fl<D> row layout maps xv[w][0] -> global column w*32 + laneId. With the
// canonical block size G=128 (== 4*32), each lane's w-th element lives in block
// w/4 INDEPENDENT of lane, so a per-block absmax is a simd_max over WPB=4
// consecutive w values — register-resident, no threadgroup scratch (unlike the
// reference's group_max[256]). Requires D % 128 == 0. codes bound as uchar*;
// int8 writes the two's-complement byte (host allocates the int8 array).
// ---------------------------------------------------------------------------
template <int D, bool INT8>
METAL_FUNC void per_block_quant_tail(thread rv_fl<D>& xv, device uchar* codes,
                                     device float* scale, int row, uint laneId, int ue8m0) {
    constexpr int GG = 128;
    constexpr int WPB = GG / 32;          // 4
    constexpr int NB = D / GG;
    static_assert(D % GG == 0, "per-block quant requires D % 128 == 0");
    const float QMAX = INT8 ? 127.0f : 448.0f;
    device float *sc = scale + (long)row * NB;
    device uchar *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int b = 0; b < NB; ++b) {
        float bmax = 0.f;
        #pragma clang loop unroll(full)
        for (int wi = 0; wi < WPB; ++wi) bmax = metal::max(bmax, metal::fabs(xv[b * WPB + wi][0]));
        bmax = metal::simd_max(bmax);
        float s_b = bmax / QMAX;
        if (!INT8 && ue8m0 != 0 && bmax > 0.f) {
            s_b = metal::exp2(metal::ceil(metal::log2(metal::max(bmax, 1e-10f) / 448.0f)));
        }
        const float inv = s_b > 0.f ? 1.f / s_b : 0.f;
        #pragma clang loop unroll(full)
        for (int wi = 0; wi < WPB; ++wi) {
            const int idx = (b * WPB + wi) * 32 + (int)laneId;
            const float v = xv[b * WPB + wi][0] * inv;
            op[idx] = INT8 ? (uchar)tk_int8_encode(v) : tk_e4m3_encode(v);
        }
        if (laneId == 0) sc[b] = s_b;
    }
}

// rms_norm(x + residual) * weight -> per-128-block fp8 (INT8=false) or int8 (INT8=true).
template <int D, bool INT8>
kernel void rms_norm_add_per_block(device   bf16  *x        [[buffer(0)]],
                                   device   bf16  *residual [[buffer(1)]],
                                   device   bf16  *weight   [[buffer(2)]],
                                   device   uchar *codes    [[buffer(3)]],
                                   device   bf16  *res_out  [[buffer(4)]],
                                   device   float *scale    [[buffer(5)]],
                                   constant uint  &M        [[buffer(6)]],
                                   constant float &eps      [[buffer(7)]],
                                   constant int   &ue8m0    [[buffer(8)]],
                                   uint3 blockIdx [[threadgroup_position_in_grid]],
                                   uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float ms = 0.f;
    mul(sq, xv, xv); sum(ms, sq, laneId); ms /= (float)D;
    mul(xv, xv, metal::rsqrt(ms + eps));
    mul(xv, xv, wv);
    per_block_quant_tail<D, INT8>(xv, codes, scale, row, laneId, ue8m0);
}

template <int D>
kernel void layernorm_add_fp8(device   bf16  *x         [[buffer(0)]],
                              device   bf16  *residual  [[buffer(1)]],
                              device   bf16  *weight    [[buffer(2)]],
                              device   bf16  *bias      [[buffer(3)]],
                              device   uchar *codes     [[buffer(4)]],
                              device   bf16  *res_out   [[buffer(5)]],
                              constant uint  &M         [[buffer(6)]],
                              constant float &eps       [[buffer(7)]],
                              constant float &inv_scale [[buffer(8)]],
                              uint3 blockIdx [[threadgroup_position_in_grid]],
                              uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float mean = 0.f; sum(mean, xv, laneId); mean /= (float)D; sub(xv, xv, mean);
    float var = 0.f; mul(sq, xv, xv); sum(var, sq, laneId); var /= (float)D;
    mul(xv, xv, metal::rsqrt(var + eps));
    mul(xv, xv, wv); add(xv, xv, bv);

    device uchar *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) {
        op[w * 32 + laneId] = tk_e4m3_encode(xv[w][0] * inv_scale);
    }
}

template <int D>
kernel void layernorm_add_fp8_dyn(device   bf16  *x        [[buffer(0)]],
                                  device   bf16  *residual [[buffer(1)]],
                                  device   bf16  *weight   [[buffer(2)]],
                                  device   bf16  *bias     [[buffer(3)]],
                                  device   uchar *codes    [[buffer(4)]],
                                  device   bf16  *res_out  [[buffer(5)]],
                                  device   float *scale    [[buffer(6)]],
                                  constant uint  &M        [[buffer(7)]],
                                  constant float &eps      [[buffer(8)]],
                                  uint3 blockIdx [[threadgroup_position_in_grid]],
                                  uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float mean = 0.f; sum(mean, xv, laneId); mean /= (float)D; sub(xv, xv, mean);
    float var = 0.f; mul(sq, xv, xv); sum(var, sq, laneId); var /= (float)D;
    mul(xv, xv, metal::rsqrt(var + eps));
    mul(xv, xv, wv); add(xv, xv, bv);

    float amax = 0.f;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) amax = metal::max(amax, metal::fabs(xv[w][0]));
    amax = metal::simd_max(amax);
    const float s = amax / 448.0f;
    const float inv_scale = s > 0.f ? 1.f / s : 0.f;
    if (laneId == 0) scale[row] = s;

    device uchar *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) {
        op[w * 32 + laneId] = tk_e4m3_encode(xv[w][0] * inv_scale);
    }
}

// int8 sibling of layernorm_add_fp8_dyn (dynamic per-row symmetric int8).
template <int D>
kernel void layernorm_add_int8_dyn(device   bf16  *x        [[buffer(0)]],
                                   device   bf16  *residual [[buffer(1)]],
                                   device   bf16  *weight   [[buffer(2)]],
                                   device   bf16  *bias     [[buffer(3)]],
                                   device   char  *codes    [[buffer(4)]],
                                   device   bf16  *res_out  [[buffer(5)]],
                                   device   float *scale    [[buffer(6)]],
                                   constant uint  &M        [[buffer(7)]],
                                   constant float &eps      [[buffer(8)]],
                                   uint3 blockIdx [[threadgroup_position_in_grid]],
                                   uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float mean = 0.f; sum(mean, xv, laneId); mean /= (float)D; sub(xv, xv, mean);
    float var = 0.f; mul(sq, xv, xv); sum(var, sq, laneId); var /= (float)D;
    mul(xv, xv, metal::rsqrt(var + eps));
    mul(xv, xv, wv); add(xv, xv, bv);

    float amax = 0.f;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) amax = metal::max(amax, metal::fabs(xv[w][0]));
    amax = metal::simd_max(amax);
    const float s = amax / 127.0f;
    const float inv_scale = s > 0.f ? 1.f / s : 0.f;
    if (laneId == 0) scale[row] = s;

    device char *op = codes + (long)row * D;
    #pragma clang loop unroll(full)
    for (int w = 0; w < vecD::outer_dim; ++w) {
        op[w * 32 + laneId] = tk_int8_encode(xv[w][0] * inv_scale);
    }
}

// layernorm(x + residual) * weight + bias -> per-128-block fp8/int8.
template <int D, bool INT8>
kernel void layernorm_add_per_block(device   bf16  *x        [[buffer(0)]],
                                    device   bf16  *residual [[buffer(1)]],
                                    device   bf16  *weight   [[buffer(2)]],
                                    device   bf16  *bias     [[buffer(3)]],
                                    device   uchar *codes    [[buffer(4)]],
                                    device   bf16  *res_out  [[buffer(5)]],
                                    device   float *scale    [[buffer(6)]],
                                    constant uint  &M        [[buffer(7)]],
                                    constant float &eps      [[buffer(8)]],
                                    constant int   &ue8m0    [[buffer(9)]],
                                    uint3 blockIdx [[threadgroup_position_in_grid]],
                                    uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;
    using row_gl = gl<bf16, 1, 1, -1, D>;
    using vec_gl = gl<bf16, 1, 1,  1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_r(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_ro(res_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_b(bias,   nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, rv, wv, bv, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(rv, gl_r, {0, 0, row, 0}, laneId);
    load(wv, gl_w, {0, 0, 0,   0}, laneId);
    load(bv, gl_b, {0, 0, 0,   0}, laneId);
    add(xv, xv, rv);
    store(gl_ro, xv, {0, 0, row, 0}, laneId);
    float mean = 0.f; sum(mean, xv, laneId); mean /= (float)D; sub(xv, xv, mean);
    float var = 0.f; mul(sq, xv, xv); sum(var, sq, laneId); var /= (float)D;
    mul(xv, xv, metal::rsqrt(var + eps));
    mul(xv, xv, wv); add(xv, xv, bv);
    per_block_quant_tail<D, INT8>(xv, codes, scale, row, laneId, ue8m0);
}

#define instantiate_rms_norm_add(DVAL)                                          \
  template [[host_name("rms_norm_add_" #DVAL)]] [[kernel]] void                 \
  rms_norm_add<DVAL>(device   bf16  *x        [[buffer(0)]],                    \
                     device   bf16  *residual  [[buffer(1)]],                   \
                     device   bf16  *weight    [[buffer(2)]],                   \
                     device   bf16  *o         [[buffer(3)]],                   \
                     device   bf16  *res_out   [[buffer(4)]],                   \
                     constant uint  &M         [[buffer(5)]],                   \
                     constant float &eps       [[buffer(6)]],                   \
                     uint3 blockIdx [[threadgroup_position_in_grid]],           \
                     uint  laneId   [[thread_index_in_simdgroup]]);

#define instantiate_layernorm_add(DVAL)                                         \
  template [[host_name("layernorm_add_" #DVAL)]] [[kernel]] void                \
  layernorm_add<DVAL>(device   bf16  *x        [[buffer(0)]],                   \
                      device   bf16  *residual  [[buffer(1)]],                  \
                      device   bf16  *weight    [[buffer(2)]],                  \
                      device   bf16  *bias      [[buffer(3)]],                  \
                      device   bf16  *o         [[buffer(4)]],                  \
                      device   bf16  *res_out   [[buffer(5)]],                  \
                      constant uint  &M         [[buffer(6)]],                  \
                      constant float &eps       [[buffer(7)]],                  \
                      uint3 blockIdx [[threadgroup_position_in_grid]],          \
                      uint  laneId   [[thread_index_in_simdgroup]]);

#define instantiate_rms_norm_add_fp8(DVAL)                                      \
  template [[host_name("rms_norm_add_fp8_" #DVAL)]] [[kernel]] void             \
  rms_norm_add_fp8<DVAL>(device bf16 *x [[buffer(0)]],                          \
                         device bf16 *residual [[buffer(1)]],                   \
                         device bf16 *weight [[buffer(2)]],                     \
                         device uchar *codes [[buffer(3)]],                     \
                         device bf16 *res_out [[buffer(4)]],                    \
                         constant uint &M [[buffer(5)]],                        \
                         constant float &eps [[buffer(6)]],                     \
                         constant float &inv_scale [[buffer(7)]],               \
                         uint3 blockIdx [[threadgroup_position_in_grid]],       \
                         uint laneId [[thread_index_in_simdgroup]]);            \
  template [[host_name("rms_norm_add_fp8_dyn_" #DVAL)]] [[kernel]] void         \
  rms_norm_add_fp8_dyn<DVAL>(device bf16 *x [[buffer(0)]],                      \
                             device bf16 *residual [[buffer(1)]],               \
                             device bf16 *weight [[buffer(2)]],                 \
                             device uchar *codes [[buffer(3)]],                 \
                             device bf16 *res_out [[buffer(4)]],                \
                             device float *scale [[buffer(5)]],                 \
                             constant uint &M [[buffer(6)]],                    \
                             constant float &eps [[buffer(7)]],                 \
                             uint3 blockIdx [[threadgroup_position_in_grid]],   \
                             uint laneId [[thread_index_in_simdgroup]]);        \
  template [[host_name("rms_norm_add_int8_dyn_" #DVAL)]] [[kernel]] void        \
  rms_norm_add_int8_dyn<DVAL>(device bf16 *x [[buffer(0)]],                     \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device char *codes [[buffer(3)]],                 \
                              device bf16 *res_out [[buffer(4)]],               \
                              device float *scale [[buffer(5)]],                \
                              constant uint &M [[buffer(6)]],                   \
                              constant float &eps [[buffer(7)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);       \
  template [[host_name("rms_norm_add_per_block_fp8_" #DVAL)]] [[kernel]] void   \
  rms_norm_add_per_block<DVAL, false>(device bf16 *x [[buffer(0)]],             \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device uchar *codes [[buffer(3)]],                \
                              device bf16 *res_out [[buffer(4)]],               \
                              device float *scale [[buffer(5)]],                \
                              constant uint &M [[buffer(6)]],                   \
                              constant float &eps [[buffer(7)]],                \
                              constant int &ue8m0 [[buffer(8)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);       \
  template [[host_name("rms_norm_add_per_block_int8_" #DVAL)]] [[kernel]] void  \
  rms_norm_add_per_block<DVAL, true>(device bf16 *x [[buffer(0)]],              \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device uchar *codes [[buffer(3)]],                \
                              device bf16 *res_out [[buffer(4)]],               \
                              device float *scale [[buffer(5)]],                \
                              constant uint &M [[buffer(6)]],                   \
                              constant float &eps [[buffer(7)]],                \
                              constant int &ue8m0 [[buffer(8)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);

#define instantiate_layernorm_add_fp8(DVAL)                                     \
  template [[host_name("layernorm_add_fp8_" #DVAL)]] [[kernel]] void            \
  layernorm_add_fp8<DVAL>(device bf16 *x [[buffer(0)]],                         \
                          device bf16 *residual [[buffer(1)]],                  \
                          device bf16 *weight [[buffer(2)]],                    \
                          device bf16 *bias [[buffer(3)]],                      \
                          device uchar *codes [[buffer(4)]],                    \
                          device bf16 *res_out [[buffer(5)]],                   \
                          constant uint &M [[buffer(6)]],                       \
                          constant float &eps [[buffer(7)]],                    \
                          constant float &inv_scale [[buffer(8)]],              \
                          uint3 blockIdx [[threadgroup_position_in_grid]],      \
                          uint laneId [[thread_index_in_simdgroup]]);           \
  template [[host_name("layernorm_add_fp8_dyn_" #DVAL)]] [[kernel]] void        \
  layernorm_add_fp8_dyn<DVAL>(device bf16 *x [[buffer(0)]],                     \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device bf16 *bias [[buffer(3)]],                  \
                              device uchar *codes [[buffer(4)]],                \
                              device bf16 *res_out [[buffer(5)]],               \
                              device float *scale [[buffer(6)]],                \
                              constant uint &M [[buffer(7)]],                   \
                              constant float &eps [[buffer(8)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);       \
  template [[host_name("layernorm_add_int8_dyn_" #DVAL)]] [[kernel]] void       \
  layernorm_add_int8_dyn<DVAL>(device bf16 *x [[buffer(0)]],                    \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device bf16 *bias [[buffer(3)]],                  \
                              device char *codes [[buffer(4)]],                 \
                              device bf16 *res_out [[buffer(5)]],               \
                              device float *scale [[buffer(6)]],                \
                              constant uint &M [[buffer(7)]],                   \
                              constant float &eps [[buffer(8)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);       \
  template [[host_name("layernorm_add_per_block_fp8_" #DVAL)]] [[kernel]] void  \
  layernorm_add_per_block<DVAL, false>(device bf16 *x [[buffer(0)]],            \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device bf16 *bias [[buffer(3)]],                  \
                              device uchar *codes [[buffer(4)]],                \
                              device bf16 *res_out [[buffer(5)]],               \
                              device float *scale [[buffer(6)]],                \
                              constant uint &M [[buffer(7)]],                   \
                              constant float &eps [[buffer(8)]],                \
                              constant int &ue8m0 [[buffer(9)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);       \
  template [[host_name("layernorm_add_per_block_int8_" #DVAL)]] [[kernel]] void \
  layernorm_add_per_block<DVAL, true>(device bf16 *x [[buffer(0)]],             \
                              device bf16 *residual [[buffer(1)]],              \
                              device bf16 *weight [[buffer(2)]],                \
                              device bf16 *bias [[buffer(3)]],                  \
                              device uchar *codes [[buffer(4)]],                \
                              device bf16 *res_out [[buffer(5)]],               \
                              device float *scale [[buffer(6)]],                \
                              constant uint &M [[buffer(7)]],                   \
                              constant float &eps [[buffer(8)]],                \
                              constant int &ue8m0 [[buffer(9)]],                \
                              uint3 blockIdx [[threadgroup_position_in_grid]],  \
                              uint laneId [[thread_index_in_simdgroup]]);

instantiate_rms_norm_add(256);
instantiate_rms_norm_add(512);
instantiate_rms_norm_add(768);
instantiate_rms_norm_add(1024);

instantiate_layernorm_add(256);
instantiate_layernorm_add(512);
instantiate_layernorm_add(768);
instantiate_layernorm_add(1024);

instantiate_rms_norm_add_fp8(256);
instantiate_rms_norm_add_fp8(512);
instantiate_rms_norm_add_fp8(768);
instantiate_rms_norm_add_fp8(1024);

instantiate_layernorm_add_fp8(256);
instantiate_layernorm_add_fp8(512);
instantiate_layernorm_add_fp8(768);
instantiate_layernorm_add_fp8(1024);

}
