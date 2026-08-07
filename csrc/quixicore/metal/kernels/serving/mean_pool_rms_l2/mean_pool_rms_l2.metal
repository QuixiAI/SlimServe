#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// mean_pool_rms_l2 (forward), bf16 I/O, fp32 compute.
//
// Sequence-pool an (M, D) block of token states into one D-vector embedding:
//
//   p = (1/M) * sum_r x[r]            // mean pool over the M rows
//   n = p * rsqrt(mean(p^2) + eps) * weight   // RMSNorm (no bias, no mean-sub)
//   y = n * rsqrt(sum(n^2) + tiny)    // L2-normalize to the unit sphere
//
// This is the terminal pooling+normalize step of an embedding forward pass. It
// is shape-keyed by the hidden width D (a variant per instantiated D); any model
// with that width uses it. One simdgroup (32 lanes) owns the whole D-vector in
// registers, so there is no threadgroup memory and no barrier — the mean is a
// register accumulation over the M rows and both reductions are simd shuffles
// inside `sum`. Grid is (1,1,1): one pooled sequence per launch.
// ---------------------------------------------------------------------------
template <int D>
kernel void mean_pool_rms_l2(device   bf16  *x      [[buffer(0)]],   // (M, D) token states
                             device   bf16  *weight [[buffer(1)]],   // (D,)   RMSNorm weight
                             device   bf16  *o      [[buffer(2)]],   // (1, D) pooled embedding
                             constant uint  &M      [[buffer(3)]],   // number of pooled rows
                             constant float &eps    [[buffer(4)]],   // RMSNorm epsilon
                             uint3 blockIdx [[threadgroup_position_in_grid]],
                             uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");

    using row_gl = gl<bf16, 1, 1, -1, D>;    // (M, D)
    using vec_gl = gl<bf16, 1, 1,  1, D>;    // (1, D) — weight / output
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    vec_gl gl_w(weight, nullptr, nullptr, nullptr, nullptr);
    vec_gl gl_o(o, nullptr, nullptr, nullptr, nullptr);

    using vecD = rv_fl<D>;                    // naive layout, fp32 compute
    vecD acc, xv, wv, sq;

    // mean pool over the M rows (bf16 -> fp32 on the fly)
    zero(acc);
    for (uint r = 0; r < M; r++) {
        load(xv, gl_x, {0, 0, (int)r, 0}, laneId);
        add(acc, acc, xv);
    }
    mul(acc, acc, 1.0f / (float)M);

    // RMSNorm(acc, weight)
    float ms = 0.f;
    mul(sq, acc, acc);
    sum(ms, sq, laneId);
    ms /= (float)D;
    mul(acc, acc, metal::rsqrt(ms + eps));    // * 1/rms  (vec - scalar)
    load(wv, gl_w, {0, 0, 0, 0}, laneId);
    mul(acc, acc, wv);                        // * weight (channelwise)

    // L2-normalize
    float l2 = 0.f;
    mul(sq, acc, acc);
    sum(l2, sq, laneId);
    mul(acc, acc, metal::rsqrt(l2 + 1e-12f));

    store(gl_o, acc, {0, 0, 0, 0}, laneId);   // fp32 -> bf16 on the fly
}

// Batched mask-aware form. mask[b,t] != 0 includes a token. Empty rows emit
// exactly zero after the RMS/L2 stages instead of dividing by zero.
template <int D>
kernel void masked_mean_pool_rms_l2(
    device const bf16 *x [[buffer(0)]],
    device const int *mask [[buffer(1)]],
    device const bf16 *weight [[buffer(2)]],
    device bf16 *o [[buffer(3)]],
    constant uint &T [[buffer(4)]],
    constant float &eps [[buffer(5)]],
    uint batch [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  using row_gl = gl<bf16, 1, 1, -1, D>;
  using vec_gl = gl<bf16, 1, 1, 1, D>;
  row_gl gl_x(const_cast<device bf16 *>(x + (long)batch * T * D), nullptr, nullptr, T, nullptr);
  vec_gl gl_w(const_cast<device bf16 *>(weight), nullptr, nullptr, nullptr, nullptr);
  vec_gl gl_o(o + (long)batch * D, nullptr, nullptr, nullptr, nullptr);
  using vecD = rv_fl<D>;
  vecD acc, xv, wv, sq;
  zero(acc);
  uint count = 0;
  for (uint t = 0; t < T; ++t) {
    if (mask[(long)batch * T + t] != 0) {
      load(xv, gl_x, {0, 0, int(t), 0}, lane);
      add(acc, acc, xv);
      ++count;
    }
  }
  if (count != 0) mul(acc, acc, 1.0f / float(count));
  float ms = 0.0f;
  mul(sq, acc, acc); sum(ms, sq, lane); ms /= float(D);
  mul(acc, acc, metal::rsqrt(ms + eps));
  load(wv, gl_w, {0, 0, 0, 0}, lane); mul(acc, acc, wv);
  float l2 = 0.0f;
  mul(sq, acc, acc); sum(l2, sq, lane);
  mul(acc, acc, metal::rsqrt(l2 + 1e-12f));
  store(gl_o, acc, {0, 0, 0, 0}, lane);
}

#define instantiate_mean_pool_rms_l2(DVAL)                                       \
  template [[host_name("mean_pool_rms_l2_" #DVAL)]] [[kernel]] void              \
  mean_pool_rms_l2<DVAL>(device   bf16  *x      [[buffer(0)]],                    \
                         device   bf16  *weight [[buffer(1)]],                    \
                         device   bf16  *o      [[buffer(2)]],                    \
                         constant uint  &M      [[buffer(3)]],                    \
                         constant float &eps    [[buffer(4)]],                    \
                         uint3 blockIdx [[threadgroup_position_in_grid]],         \
                         uint  laneId   [[thread_index_in_simdgroup]]);            \
  template [[host_name("masked_mean_pool_rms_l2_" #DVAL)]] [[kernel]] void       \
  masked_mean_pool_rms_l2<DVAL>(device const bf16 *x [[buffer(0)]],               \
                         device const int *mask [[buffer(1)]],                    \
                         device const bf16 *weight [[buffer(2)]],                 \
                         device bf16 *o [[buffer(3)]], constant uint &T [[buffer(4)]],\
                         constant float &eps [[buffer(5)]],                       \
                         uint batch [[threadgroup_position_in_grid]],             \
                         uint lane [[thread_index_in_simdgroup]]);

instantiate_mean_pool_rms_l2(256);
instantiate_mean_pool_rms_l2(512);
instantiate_mean_pool_rms_l2(768);
instantiate_mean_pool_rms_l2(1024);

}  // namespace mittens
