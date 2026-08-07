#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// GELU activation (tanh approximation), bf16 I/O, fp32 compute. Matches
// mx.nn.gelu_approx / F.gelu(approximate='tanh'):
//   y = 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
//
// Elementwise; one simdgroup per row of length D (rv_fl<D> in registers).
// ---------------------------------------------------------------------------
template <int D>
kernel void gelu(device   bf16 *x [[buffer(0)]],
                 device   bf16 *o [[buffer(1)]],
                 constant uint &M [[buffer(2)]],
                 uint3 blockIdx [[threadgroup_position_in_grid]],
                 uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % TILE_DIM == 0, "D must be divisible by 8");
    const int row = blockIdx.x;

    using row_gl = gl<bf16, 1, 1, -1, D>;
    row_gl gl_x(x, nullptr, nullptr, M, nullptr);
    row_gl gl_o(o, nullptr, nullptr, M, nullptr);

    using vecD = rv_fl<D>;
    vecD xv;
    load(xv, gl_x, {0, 0, row, 0}, laneId);
    gelu(xv, xv);                            // tanh-approx GELU (vec map)
    store(gl_o, xv, {0, 0, row, 0}, laneId);
}

#define instantiate_gelu(DVAL)                                                \
  template [[host_name("gelu_" #DVAL)]] [[kernel]] void                       \
  gelu<DVAL>(device   bf16 *x [[buffer(0)]],                                  \
             device   bf16 *o [[buffer(1)]],                                  \
             constant uint &M [[buffer(2)]],                                  \
             uint3 blockIdx [[threadgroup_position_in_grid]],                 \
             uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_gelu(256);
instantiate_gelu(512);
instantiate_gelu(768);
instantiate_gelu(1024);

// GELU backward (tanh approximation): dx = dy * gelu'(x). With k=sqrt(2/pi), a=0.044715,
// inner = k*(x + a*x^3), t = tanh(inner):  gelu'(x) = 0.5*(1+t) + 0.5*x*(1-t^2)*k*(1+3a*x^2).
// One thread per 4 elements (vec4 loads/stores when the 4-block fits; scalar tail otherwise) — the
// same packed_four pattern glu_bwd/embedding use, which roughly doubled bf16 elementwise bandwidth
// vs the scalar per-element access. Any shape; T templated (fp32/fp16/bf16).
METAL_FUNC float gelu_bwd_grad(float xv) {
    const float k = 0.7978845608028654f, a = 0.044715f;
    const float inner = k * (xv + a * xv * xv * xv);
    const float t = metal::precise::tanh(inner);
    const float dinner = k * (1.0f + 3.0f * a * xv * xv);
    return 0.5f * (1.0f + t) + 0.5f * xv * (1.0f - t * t) * dinner;
}
template <typename T>
kernel void gelu_bwd(device const T *x  [[buffer(0)]],
                     device const T *dy [[buffer(1)]],
                     device T       *dx [[buffer(2)]],
                     constant int &n    [[buffer(3)]],
                     uint gid [[thread_position_in_grid]]) {
    typedef typename base_types::packing<T>::packed_four T4;
    const uint base = gid * 4;
    if (base + 4 <= (uint)n) {
        float4 xv  = float4(((device const T4*)(x  + base))[0]);
        float4 dyv = float4(((device const T4*)(dy + base))[0]);
        float4 r;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) r[i] = dyv[i] * gelu_bwd_grad(xv[i]);
        ((device T4*)(dx + base))[0] = T4(r);
    } else {
        for (uint i = base; i < (uint)n; ++i) dx[i] = T(float(dy[i]) * gelu_bwd_grad(float(x[i])));
    }
}

#define instantiate_gelu_bwd(type_name, T)                                     \
  template [[host_name("gelu_bwd_" #type_name)]] [[kernel]] void                \
  gelu_bwd<T>(device const T *x [[buffer(0)]], device const T *dy [[buffer(1)]], \
    device T *dx [[buffer(2)]], constant int &n [[buffer(3)]],                  \
    uint gid [[thread_position_in_grid]]);

instantiate_gelu_bwd(float32, float)
instantiate_gelu_bwd(float16, half)
instantiate_gelu_bwd(bfloat16, bf16)

}
