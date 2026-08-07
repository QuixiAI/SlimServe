#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Per-tensor FP8 (e4m3) FAKE-quant — moe_train_plan §7.5 mode b / §8.7 Q-T4:
// the eval-side quantize-dequantize of attention/embed/lm_head that measures the
// FP8-cast delta (mode b − mode a1) without relying on torch-MPS float8 support.
//
//   pass 1: quant_tensor_absmax (quant_rt.metal, reused) -> atomic absmax
//   pass 2 (this kernel): s = absmax/448;  x_fq = e4m3_decode(e4m3_encode(x/s)) * s
//
// Same two-pass shape as quantize_per_tensor_fp8, but emitting the DEQUANTIZED
// tensor in the input dtype (an eval drop-in) instead of codes. scale_out records
// the per-tensor scale for the model card / Q-T4 bookkeeping.
// ---------------------------------------------------------------------------

// e4m3 grid value of a/s under round-to-nearest-EVEN (a = |x| > 0, s > 0).
// Deliberately NOT tk_e4m3_encode: that encoder rounds half away from zero, while
// torch.float8_e4m3fn (the mode-b oracle and the runtimes' convention) is RNE —
// on bf16-grid inputs the halfway points are hit constantly and the two diverge.
// The device's fast-math division is off by ulps exactly at those ties, so the
// division only picks the CANDIDATE; the round decision compares a against the
// halfway point exactly via fma ((k+0.5)*step is exact, fma is single-rounded).
METAL_FUNC float fq_e4m3_rne_div(float a, float s) {
    const float q = a / s;                           // q <= 448·(1+ulp) by construction
    int e;
    (void)metal::frexp(q, e);
    int E = e - 1;                                   // q in [2^E, 2^{E+1})
    if (E < -6) { E = -6; }                          // subnormal floor: step 2^-9
    const float step = metal::ldexp(1.0f, E - 3);    // 3 mantissa bits
    float k = metal::floor(q / step);                // candidate (exact: pow2 shift)
    const float r = metal::fma(-(k + 0.5f) * step, s, a);   // sign(a - halfway*s), exact
    if (r > 0.0f || (r == 0.0f && metal::fmod(k, 2.0f) == 1.0f)) { k += 1.0f; }
    return k * step;                                 // rollover to 2^{E+1} is exact
}

template <typename T>
kernel void fake_quant_fp8(device const T    *x         [[buffer(0)]],
                           device const uint *scale_u   [[buffer(1)]],   // P3 orderable absmax
                           device T          *x_fq      [[buffer(2)]],
                           device float      *scale_out [[buffer(3)]],
                           constant int      &n         [[buffer(4)]],
                           uint tid [[thread_position_in_grid]]) {
    using T4 = vec<T, 4>;
    const float amax = orderable_uint_to_float(scale_u[0]);
    // fast-math division is off by an ulp exactly when amax/448 is representable —
    // which moves the whole grid off torch's. One fma Newton step re-rounds it.
    const float s0 = amax / 448.0f;
    const float s = metal::fma(metal::fma(-s0, 448.0f, amax), 1.0f / 448.0f, s0);
    const float inv = s > 0.0f ? 1.0f / s : 0.0f;    // only gates the s == 0 case
    if (tid == 0) { scale_out[0] = s; }
    const long base = (long)tid * 4;
    if (base + 4 <= (long)n) {
        const float4 v = float4(((device const T4*)(x + base))[0]);
        float4 dq;
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) {
            const float a = metal::fabs(v[j]);
            dq[j] = (inv > 0.0f && a > 0.0f)
                        ? metal::copysign(fq_e4m3_rne_div(a, s), v[j]) * s : 0.0f;
        }
        ((device T4*)(x_fq + base))[0] = T4(dq);
    } else {
        for (long i = base; i < (long)n; ++i) {
            const float xi = float(x[i]);
            const float a = metal::fabs(xi);
            x_fq[i] = T((inv > 0.0f && a > 0.0f)
                            ? metal::copysign(fq_e4m3_rne_div(a, s), xi) * s : 0.0f);
        }
    }
}

#define instantiate_fake_quant_fp8(type_name, T)                                \
  template [[host_name("fake_quant_fp8_" #type_name)]] [[kernel]] void          \
  fake_quant_fp8<T>(device const T *x [[buffer(0)]],                            \
                    device const uint *scale_u [[buffer(1)]],                   \
                    device T *x_fq [[buffer(2)]],                               \
                    device float *scale_out [[buffer(3)]],                      \
                    constant int &n [[buffer(4)]],                              \
                    uint tid [[thread_position_in_grid]]);

instantiate_fake_quant_fp8(float32, float)
instantiate_fake_quant_fp8(float16, half)
instantiate_fake_quant_fp8(bfloat16, bf16)
