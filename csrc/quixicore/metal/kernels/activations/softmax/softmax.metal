#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Softmax over the last axis, bf16 I/O, fp32 compute.
//
//   y = exp(x - max(x)) / sum(exp(x - max(x)))
//
// One simdgroup (32 lanes) per row of length D; the row fits in registers, so
// the max/sum reductions use simd shuffles (no threadgroup memory/barrier).
// This is the standalone version of the row-softmax used inline in attn_fwd.
// ---------------------------------------------------------------------------
template <int D>
kernel void softmax(device   bf16 *x [[buffer(0)]],
                    device   bf16 *o [[buffer(1)]],
                    constant uint &M [[buffer(2)]],   // total rows = prod(shape[:-1])
                    uint3 blockIdx [[threadgroup_position_in_grid]],
                    uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;   // (M, D)
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);

    using vecD = rv_fl<D>;
    vecD xv;
    load(xv, gl_x, {0, 0, row, 0}, laneId);

    float m = 0.f;
    max(m, xv, laneId);          // row max (broadcast to all lanes)
    sub(xv, xv, m);              // x - max  (vec - scalar)
    exp(xv, xv);                 // exp      (vec map)
    float s = 0.f;
    sum(s, xv, laneId);          // row sum
    div(xv, xv, s);             // normalize (vec - scalar)
    store(gl_o, xv, {0, 0, row, 0}, laneId);
}

#define instantiate_softmax(DVAL)                                             \
  template [[host_name("softmax_" #DVAL)]] [[kernel]] void                    \
  softmax<DVAL>(device   bf16 *x [[buffer(0)]],                               \
                device   bf16 *o [[buffer(1)]],                               \
                constant uint &M [[buffer(2)]],                               \
                uint3 blockIdx [[threadgroup_position_in_grid]],              \
                uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_softmax(256);
instantiate_softmax(512);
instantiate_softmax(768);
instantiate_softmax(1024);

}
