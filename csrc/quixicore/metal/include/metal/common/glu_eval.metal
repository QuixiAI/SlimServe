/**
 * @file
 * @brief Shared gated-activation forward math (single definition for glu.metal and the fused
 * act->quant epilogues in act_quant.metal). glu_eval modes: 0 reglu, 1 geglu-tanh, 2 swiglu,
 * 3 swiglu_oai (clamped, alpha-scaled sigmoid, (1+up)), 4 geglu-erf,
 * 5 geglu-quick, 6 sigmoid gate.
 * The backward-only derivative helpers stay in kernels/glu/glu.metal.
 */

#pragma once
#include <metal_stdlib>

namespace mittens {

constant float GLU_GELU_COEF_A = 0.044715f;
constant float GLU_GELU_QUICK_COEF = -1.702f;
constant float GLU_SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
constant float GLU_SQRT_2_INV = 0.70710678118654752440084436210484f;

constant float GLU_ERF_P = 0.3275911f;
constant float GLU_ERF_A1 = 0.254829592f;
constant float GLU_ERF_A2 = -0.284496736f;
constant float GLU_ERF_A3 = 1.421413741f;
constant float GLU_ERF_A4 = -1.453152027f;
constant float GLU_ERF_A5 = 1.061405429f;

METAL_FUNC float glu_tanh(float x) {
    return 1.0f - 2.0f / (metal::exp(x + x) + 1.0f);
}

METAL_FUNC float glu_erf_approx(float x) {
    const float sx = x < 0.0f ? -1.0f : 1.0f;
    x = metal::abs(x);
    const float t = 1.0f / (1.0f + GLU_ERF_P * x);
    const float poly = (((((GLU_ERF_A5 * t + GLU_ERF_A4) * t) + GLU_ERF_A3) * t + GLU_ERF_A2) * t + GLU_ERF_A1) * t;
    const float y = 1.0f - poly * metal::exp(-x * x);
    return sx * y;
}

METAL_FUNC float glu_gelu_tanh(float x) {
    const float inner = GLU_SQRT_2_OVER_PI * x * (1.0f + GLU_GELU_COEF_A * x * x);
    return 0.5f * x * (1.0f + glu_tanh(inner));
}

METAL_FUNC float glu_gelu_erf(float x) {
    return 0.5f * x * (1.0f + glu_erf_approx(x * GLU_SQRT_2_INV));
}

METAL_FUNC float glu_eval(int mode, float x0, float x1, float alpha, float limit) {
    if (mode == 0) {
        return x0 * x1 * (x0 > 0.0f ? 1.0f : 0.0f);
    }
    if (mode == 1) {
        return glu_gelu_tanh(x0) * x1;
    }
    if (mode == 2) {
        return (x0 / (1.0f + metal::exp(-x0))) * x1;
    }
    if (mode == 3) {
        x0 = metal::min(x0, limit);
        x1 = metal::max(metal::min(x1, limit), -limit);
        return (x0 / (1.0f + metal::exp(-x0 * alpha))) * (1.0f + x1);
    }
    if (mode == 4) {
        return glu_gelu_erf(x0) * x1;
    }
    if (mode == 5) {
        return (x0 * (1.0f / (1.0f + metal::exp(GLU_GELU_QUICK_COEF * x0)))) * x1;
    }
    return (1.0f / (1.0f + metal::exp(-x0))) * x1;
}

} // namespace mittens
