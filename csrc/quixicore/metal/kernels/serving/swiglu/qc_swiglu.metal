// SPDX-License-Identifier: Apache-2.0
// Fused SwiGLU activations for the Metal serving path, bitwise-mirroring
// the eager torch chains they replace (MPS silu and
// sigmoid on fp16/bf16/fp32 are fp32-internal — silu is the DIVISION
// form x/(1+exp(-x)) with Metal precise::exp/divide — rounded once to
// the storage dtype; elementwise binary mul/add round in the storage
// dtype).
//
// Two forms, selected by oai_form:
//   0: out = silu(clamp?(gate)) * clamp?(up)
//      (apply_moe_activation Metal SILU branch: F.silu(gate) * up)
//   1: out = gate_c * sigmoid(alpha * gate_c) * (up + beta)
//      (SiluAndMulWithClamp.forward_native / SWIGLUOAI_UNINTERLEAVE)
// has_clamp applies gate <= limit and -limit <= up <= limit, matching
// silu_and_mul_with_clamp and the ds4 matvec_*_mid_worker reference.
//
// x is [T, 2N] contiguous (gate cols [0,N), up cols [N,2N)); y is [T,N].

#include <metal_stdlib>

template <typename T>
METAL_FUNC void qc_swiglu_body(device const T *x, device T *y, int N,
                               int has_clamp, float limit, int oai_form,
                               float alpha, float beta, uint gid, int total) {
    if ((int)gid >= total) { return; }
    const long row = (long)gid / N;
    const int col = (int)((long)gid - row * N);
    {
        #pragma clang fp reassociate(off) contract(off)
        T g = x[row * 2 * N + col];
        T u = x[row * 2 * N + N + col];
        if (has_clamp) {
            // Ternary keeps torch.clamp's NaN passthrough and avoids the
            // ambiguous min/clamp overloads for bfloat.
            const T lim = T(limit);
            const T nlim = T(-limit);
            g = (g > lim) ? lim : g;
            u = (u > lim) ? lim : ((u < nlim) ? nlim : u);
        }
        if (oai_form == 0) {
            const T s = T(metal::precise::divide(
                float(g), 1.0f + metal::precise::exp(-float(g))));
            y[gid] = s * u;
        } else {
            const T a = T(alpha) * g;
            const T s = T(metal::precise::divide(
                1.0f, 1.0f + metal::precise::exp(-float(a))));
            const T t1 = g * s;
            const T t2 = u + T(beta);
            y[gid] = t1 * t2;
        }
    }
}

#define instantiate_qc_swiglu(suffix, T)                                     \
  [[host_name("qc_swiglu_" #suffix)]] kernel void qc_swiglu_##suffix(        \
      device const T *x [[buffer(0)]], device T *y [[buffer(1)]],            \
      constant int &N [[buffer(2)]], constant int &has_clamp [[buffer(3)]],  \
      constant float &limit [[buffer(4)]],                                   \
      constant int &oai_form [[buffer(5)]],                                  \
      constant float &alpha [[buffer(6)]],                                   \
      constant float &beta [[buffer(7)]],                                    \
      constant int &total [[buffer(8)]],                                     \
      uint gid [[thread_position_in_grid]]) {                                \
    qc_swiglu_body<T>(x, y, N, has_clamp, limit, oai_form, alpha, beta, gid, \
                      total);                                                \
  }

instantiate_qc_swiglu(float16, half)
instantiate_qc_swiglu(bfloat16, bfloat)
instantiate_qc_swiglu(float32, float)
