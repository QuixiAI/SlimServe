#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// GLU_* constants, glu_tanh/glu_erf_approx/glu_gelu_* and glu_eval now come from the
// substrate header include/common/glu_eval.metal (shared with the fused act->quant
// epilogues in kernels/act_quant/). Only the backward-only derivative helpers live here.

template <typename T>
METAL_FUNC float to_float(T x) {
    return float(x);
}

// Exact analytic derivative of glu_erf_approx (even function: d/dx of an odd approx). With
// t = 1/(1+P|x|), poly = A1 t + A2 t^2 + .. + A5 t^5, and dt/d|x| = -P t^2:
//   d/dx [1 - poly * exp(-x^2)] = exp(-x^2) * (2|x| poly + P t^2 * dpoly/dt).
// Used by glu_grad mode 4 so the geglu_erf backward is the *exact* derivative of glu_eval mode 4
// (not the true-Gaussian derivative of the ideal erf, which the approximate forward never equals).
METAL_FUNC float glu_erf_approx_deriv(float x) {
    const float ax = metal::abs(x);
    const float t = 1.0f / (1.0f + GLU_ERF_P * ax);
    const float poly = (((((GLU_ERF_A5 * t + GLU_ERF_A4) * t) + GLU_ERF_A3) * t + GLU_ERF_A2) * t + GLU_ERF_A1) * t;
    const float dpoly_dt = (((5.0f * GLU_ERF_A5 * t + 4.0f * GLU_ERF_A4) * t + 3.0f * GLU_ERF_A3) * t
                            + 2.0f * GLU_ERF_A2) * t + GLU_ERF_A1;
    return metal::exp(-ax * ax) * (2.0f * ax * poly + GLU_ERF_P * t * t * dpoly_dt);
}




// One thread per 4 elements (vec4 loads/stores; the scalar version measured 103 GB/s vs the
// ~400 GB/s the same shape reaches vectorized). The last thread of a ragged n takes the
// scalar tail.
template <typename T, int MODE>
kernel void glu(device const T *x [[buffer(0)]],
                device const T *gate [[buffer(1)]],
                device T *out [[buffer(2)]],
                constant uint &n [[buffer(3)]],
                constant float &alpha [[buffer(4)]],
                constant float &limit [[buffer(5)]],
                uint tid [[thread_position_in_grid]]) {
    using T4 = metal::vec<T, 4>;
    const uint base = tid * 4;
    if (base + 4 <= n) {
        const float4 x0 = float4(((device const T4*)(x + base))[0]);
        const float4 x1 = float4(((device const T4*)(gate + base))[0]);
        float4 r;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) r[i] = glu_eval(MODE, x0[i], x1[i], alpha, limit);
        ((device T4*)(out + base))[0] = T4(r);
    } else {
        for (uint i = base; i < n; ++i)
            out[i] = T(glu_eval(MODE, to_float(x[i]), to_float(gate[i]), alpha, limit));
    }
}

// Backward of out = act(a)*b: db = dc*act(a); da = dc*b*act'(a). a=x (gate half), b=gate (up half).
// Each mode's act' is the exact derivative of its glu_eval branch (geglu_erf differentiates the A&S
// erf approximation itself via glu_erf_approx_deriv, so the forward/backward pair is bit-consistent).
METAL_FUNC void glu_grad(int mode, float a, float b, float dc, float alpha, float limit,
                         thread float &da, thread float &db) {
    if (mode == 0) {                       // reglu: act = relu(a)
        const float m = a > 0.0f ? 1.0f : 0.0f;
        db = dc * a * m;
        da = dc * b * m;
        return;
    }
    if (mode == 1) {                       // geglu (tanh)
        const float inner = GLU_SQRT_2_OVER_PI * a * (1.0f + GLU_GELU_COEF_A * a * a);
        const float t = glu_tanh(inner);
        const float dz = GLU_SQRT_2_OVER_PI * (1.0f + 3.0f * GLU_GELU_COEF_A * a * a);
        db = dc * (0.5f * a * (1.0f + t));
        da = dc * b * (0.5f * (1.0f + t) + 0.5f * a * (1.0f - t * t) * dz);
        return;
    }
    if (mode == 2) {                       // swiglu: act = a*sigmoid(a)
        const float s = 1.0f / (1.0f + metal::exp(-a));
        db = dc * (a * s);
        da = dc * b * (s * (1.0f + a * (1.0f - s)));
        return;
    }
    if (mode == 3) {                       // swiglu_oai (clamped swish * (1+clamp(b)))
        const float x0 = metal::min(a, limit);
        const float x1 = metal::max(metal::min(b, limit), -limit);
        const float s0 = 1.0f / (1.0f + metal::exp(-x0 * alpha));
        const float f = x0 * s0;
        const float ind_a = a < limit ? 1.0f : 0.0f;                  // d min(a,limit)/da
        const float ind_b = (b < limit && b > -limit) ? 1.0f : 0.0f;  // d clamp(b)/db
        db = dc * f * ind_b;
        da = dc * (1.0f + x1) * (s0 + x0 * alpha * s0 * (1.0f - s0)) * ind_a;
        return;
    }
    if (mode == 4) {                       // geglu_erf: act = 0.5 a (1+erf_approx(a/sqrt2))
        const float u = a * GLU_SQRT_2_INV;
        const float e = glu_erf_approx(u);
        const float de = glu_erf_approx_deriv(u) * GLU_SQRT_2_INV;   // d erf_approx(a/sqrt2) / da
        db = dc * (0.5f * a * (1.0f + e));
        da = dc * b * (0.5f * (1.0f + e) + 0.5f * a * de);
        return;
    }
    if (mode == 5) {                       // geglu_quick: act = a*sigmoid(1.702 a)
        const float s = 1.0f / (1.0f + metal::exp(GLU_GELU_QUICK_COEF * a));
        db = dc * (a * s);
        da = dc * b * (s + a * (-GLU_GELU_QUICK_COEF) * s * (1.0f - s));
        return;
    }
    // mode 6: sigmoid gate: out = sigmoid(a) * b
    const float s = 1.0f / (1.0f + metal::exp(-a));
    db = dc * s;
    da = dc * b * s * (1.0f - s);
}

template <typename T, int MODE>
kernel void glu_bwd(device const T *x [[buffer(0)]],       // a (gate half)
                    device const T *gate [[buffer(1)]],    // b (up half)
                    device const T *dc [[buffer(2)]],      // upstream grad
                    device T *da [[buffer(3)]],            // grad wrt x
                    device T *db [[buffer(4)]],            // grad wrt gate
                    constant uint &n [[buffer(5)]],
                    constant float &alpha [[buffer(6)]],
                    constant float &limit [[buffer(7)]],
                    uint tid [[thread_position_in_grid]]) {
    using T4 = metal::vec<T, 4>;
    const uint base = tid * 4;
    if (base + 4 <= n) {
        const float4 a4 = float4(((device const T4*)(x + base))[0]);
        const float4 b4 = float4(((device const T4*)(gate + base))[0]);
        const float4 c4 = float4(((device const T4*)(dc + base))[0]);
        float4 da4, db4;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
            float dai, dbi;
            glu_grad(MODE, a4[i], b4[i], c4[i], alpha, limit, dai, dbi);
            da4[i] = dai; db4[i] = dbi;
        }
        ((device T4*)(da + base))[0] = T4(da4);
        ((device T4*)(db + base))[0] = T4(db4);
    } else {
        for (uint i = base; i < n; ++i) {
            float dai, dbi;
            glu_grad(MODE, to_float(x[i]), to_float(gate[i]), to_float(dc[i]), alpha, limit, dai, dbi);
            da[i] = T(dai); db[i] = T(dbi);
        }
    }
}

#define instantiate_glu(MODE_NAME, MODE_ID, type_name, T)                    \
  template [[host_name("glu_" #MODE_NAME "_" #type_name)]] [[kernel]] void  \
  glu<T, MODE_ID>(device const T *x [[buffer(0)]],                           \
                  device const T *gate [[buffer(1)]],                        \
                  device T *out [[buffer(2)]],                               \
                  constant uint &n [[buffer(3)]],                            \
                  constant float &alpha [[buffer(4)]],                       \
                  constant float &limit [[buffer(5)]],                       \
                  uint tid [[thread_position_in_grid]]);                     \
  template [[host_name("glu_bwd_" #MODE_NAME "_" #type_name)]] [[kernel]] void \
  glu_bwd<T, MODE_ID>(device const T *x [[buffer(0)]],                       \
                  device const T *gate [[buffer(1)]], device const T *dc [[buffer(2)]], \
                  device T *da [[buffer(3)]], device T *db [[buffer(4)]],    \
                  constant uint &n [[buffer(5)]], constant float &alpha [[buffer(6)]], \
                  constant float &limit [[buffer(7)]], uint tid [[thread_position_in_grid]]);

#define instantiate_glu_mode(MODE_NAME, MODE_ID)                             \
  instantiate_glu(MODE_NAME, MODE_ID, float32, float)                        \
  instantiate_glu(MODE_NAME, MODE_ID, float16, half)                         \
  instantiate_glu(MODE_NAME, MODE_ID, bfloat16, bf16)

instantiate_glu_mode(reglu, 0)
instantiate_glu_mode(geglu, 1)
instantiate_glu_mode(swiglu, 2)
instantiate_glu_mode(swiglu_oai, 3)
instantiate_glu_mode(geglu_erf, 4)
instantiate_glu_mode(geglu_quick, 5)
instantiate_glu_mode(sigmoid, 6)

} // namespace mittens
