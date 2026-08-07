#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// rms_norm_residual_next (forward), bf16 I/O, fp32 compute.
//
// The residual-stream seam between two transformer sub-blocks, fused into one
// pass. Given a sub-block projection `x` (e.g. an attention or FFN output), its
// post-norm weight, the running residual, and the NEXT block's pre-norm weight:
//
//   pinv = rsqrt(mean(x^2) + eps)                 // post-RMSNorm of the projection
//   res  = residual + x * pinv * post_weight      // add the post-normed projection in
//   rinv = rsqrt(mean(res^2) + eps)               // pre-RMSNorm of the new residual
//   next = res * rinv * next_weight               // the next block's normed input
//
// Returns the updated residual `res` and the next block's input `next`. This is
// the post-norm/residual-add/pre-norm sandwich (Gemma-style) collapsed from three
// launches (two RMSNorms + an add, each a full (M,D) round-trip) into one: x, the
// residual, and both weights are read once and the two normalized outputs written
// once. One simdgroup (32 lanes) owns one row of width D — the whole row lives in
// registers, so there is no threadgroup memory and both reductions are simd
// shuffles inside `sum`. Shape-keyed by the hidden width D in {256,512,768,1024}.
// ---------------------------------------------------------------------------
template <int D>
kernel void rms_norm_residual_next(
    device   bf16  *x           [[buffer(0)]],   // (M, D) sub-block projection
    device   bf16  *post_weight [[buffer(1)]],   // (D,)   post-RMSNorm weight
    device   bf16  *residual    [[buffer(2)]],   // (M, D) running residual (read)
    device   bf16  *next_weight [[buffer(3)]],   // (D,)   next-block pre-RMSNorm weight
    device   bf16  *res_out     [[buffer(4)]],   // (M, D) residual + post-normed projection
    device   bf16  *next_out    [[buffer(5)]],   // (M, D) next block's normed input
    constant uint  &M           [[buffer(6)]],
    constant float &eps         [[buffer(7)]],
    uint3 blockIdx [[threadgroup_position_in_grid]],
    uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D)
    using vec_gl = gl<bf16, 1, 1,  1, D>;   // (1, D) — weights
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_res(residual, nullptr, nullptr, M, nullptr);
    row_gl gl_res_out(res_out, nullptr, nullptr, M, nullptr);
    row_gl gl_next_out(next_out, nullptr, nullptr, M, nullptr);
    vec_gl gl_pw(post_weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_nw(next_weight, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;
    vecD xv, pw, rv, nw, sq;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    load(pw, gl_pw, {0, 0, 0, 0}, laneId);
    load(rv, gl_res, {0, 0, row, 0}, laneId);
    load(nw, gl_nw, {0, 0, 0, 0}, laneId);

    // post-RMSNorm the projection, then add it into the residual
    float pss = 0.f;
    mul(sq, xv, xv);
    sum(pss, sq, laneId);
    pss /= (float)D;
    mul(xv, xv, metal::rsqrt(pss + eps));   // * 1/rms(x)
    mul(xv, xv, pw);                        // * post_weight
    add(rv, rv, xv);                        // res = residual + post-normed projection
    store(gl_res_out, rv, {0, 0, row, 0}, laneId);

    // pre-RMSNorm the new residual for the next block
    float rss = 0.f;
    mul(sq, rv, rv);
    sum(rss, sq, laneId);
    rss /= (float)D;
    mul(rv, rv, metal::rsqrt(rss + eps));   // * 1/rms(res)
    mul(rv, rv, nw);                        // * next_weight
    store(gl_next_out, rv, {0, 0, row, 0}, laneId);
}

#define instantiate_rms_norm_residual_next(DVAL)                                \
  template [[host_name("rms_norm_residual_next_" #DVAL)]] [[kernel]] void       \
  rms_norm_residual_next<DVAL>(                                                 \
      device   bf16  *x           [[buffer(0)]],                                \
      device   bf16  *post_weight [[buffer(1)]],                                \
      device   bf16  *residual    [[buffer(2)]],                                \
      device   bf16  *next_weight [[buffer(3)]],                                \
      device   bf16  *res_out     [[buffer(4)]],                                \
      device   bf16  *next_out    [[buffer(5)]],                                \
      constant uint  &M           [[buffer(6)]],                                \
      constant float &eps         [[buffer(7)]],                                \
      uint3 blockIdx [[threadgroup_position_in_grid]],                          \
      uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_rms_norm_residual_next(256);
instantiate_rms_norm_residual_next(512);
instantiate_rms_norm_residual_next(768);
instantiate_rms_norm_residual_next(1024);

}  // namespace mittens
