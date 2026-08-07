#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Monarch FFT convolution forward.  An N = S*S point FFT-conv is factored into S×S
// complex tiles (the Monarch / Cooley–Tukey decomposition), so the whole thing is
// complex matmuls (the new complex_mma) + pointwise complex multiplies + transposes:
//
//   x = xᵀ ; x = x·F ; x = xᵀ ; x = x ⊙ tw_fft ; x = x·F ;        (forward FFT)
//   x = x ⊙ k_f ;                                                   (pointwise conv in freq)
//   x = x·Finv ; x = xᵀ ; x = x ⊙ tw_ifft ; x = x·Finv ; x = xᵀ    (inverse FFT)
//   out = Re(x)
//
// Mirrors monarch_conv() in the ThunderKittens fftconv reference. One simdgroup per
// (batch*head). Complex arrays carry a leading size-2 (real,imag) axis. F/Finv/tw* are
// shared (one S×S tile); k_f is per-head. Output is the real part, (BH, S, S).

// ---- small complex helpers on crt tiles ----
template<typename CRT>
static METAL_FUNC void cmplx_copy(thread CRT& d, thread const CRT& s) {
    copy(d.real, s.real);
    copy(d.imag, s.imag);
}
template<typename CRTD, typename CRTS>
static METAL_FUNC void cmplx_transpose(thread CRTD& d, thread const CRTS& s, const int lane) {
    transpose_sep(d.real, s.real, lane);
    transpose_sep(d.imag, s.imag, lane);
}
// d = a ⊙ b  (elementwise complex), d must be distinct from a and b.
template<typename CRT>
static METAL_FUNC void cmplx_mul(thread CRT& d, thread const CRT& a, thread const CRT& b) {
    typename CRT::component t;
    mul(d.real, a.real, b.real);   // ar·br
    mul(t,      a.imag, b.imag);   // ai·bi
    sub(d.real, d.real, t);        // dr = ar·br − ai·bi
    mul(d.imag, a.real, b.imag);   // ar·bi
    mul(t,      a.imag, b.real);   // ai·br
    add(d.imag, d.imag, t);        // di = ar·bi + ai·br
}

template<typename T, int S>
kernel void fftconv(
    device   T*   OUT  [[buffer(0)]],   // (BH, S, S)        real output
    device   T*   X    [[buffer(1)]],   // (2, BH, S, S)     complex input
    device   T*   F    [[buffer(2)]],   // (2, S, S)         FFT matrix
    device   T*   TWF  [[buffer(3)]],   // (2, S, S)         fwd twiddle factors
    device   T*   FINV [[buffer(4)]],   // (2, S, S)         inverse-FFT matrix
    device   T*   TWI  [[buffer(5)]],   // (2, S, S)         inv twiddle factors
    device   T*   KF   [[buffer(6)]],   // (2, H, S, S)      per-head kernel FFT
    const constant int &BH [[buffer(7)]],
    const constant int &H  [[buffer(8)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    const int bh = tgid.x;
    const int head = bh % H;
    const int SS = S * S;

    using gl_t = gl<T, 1, 1, -1, -1>;
    gl_t gXr(X,            nullptr, nullptr, BH * S, S);
    gl_t gXi(X + BH * SS,  nullptr, nullptr, BH * S, S);
    gl_t gFr(F,            nullptr, nullptr, S, S);
    gl_t gFi(F + SS,       nullptr, nullptr, S, S);
    gl_t gWFr(TWF,         nullptr, nullptr, S, S);
    gl_t gWFi(TWF + SS,    nullptr, nullptr, S, S);
    gl_t gIr(FINV,         nullptr, nullptr, S, S);
    gl_t gIi(FINV + SS,    nullptr, nullptr, S, S);
    gl_t gWIr(TWI,         nullptr, nullptr, S, S);
    gl_t gWIi(TWI + SS,    nullptr, nullptr, S, S);
    gl_t gKr(KF,           nullptr, nullptr, H * S, S);
    gl_t gKi(KF + H * SS,  nullptr, nullptr, H * S, S);
    gl_t gOut(OUT,         nullptr, nullptr, BH * S, S);

    using C = crt<T, S, S, ducks::rt_layout::row>;
    C x, y, m;

    load(x.real, gXr, {0, 0, bh, 0}, lane);
    load(x.imag, gXi, {0, 0, bh, 0}, lane);

    // ---- forward FFT ----
    cmplx_transpose(y, x, lane);                                   // x = xᵀ
    cmplx_copy(x, y);
    load(m.real, gFr, {0, 0, 0, 0}, lane); load(m.imag, gFi, {0, 0, 0, 0}, lane);
    complex_mm_AB(y, x, m);                                        // x = x·F
    cmplx_copy(x, y);
    cmplx_transpose(y, x, lane);                                   // x = xᵀ
    cmplx_copy(x, y);
    load(m.real, gWFr, {0, 0, 0, 0}, lane); load(m.imag, gWFi, {0, 0, 0, 0}, lane);
    cmplx_mul(y, x, m);                                            // x = x ⊙ tw_fft
    cmplx_copy(x, y);
    load(m.real, gFr, {0, 0, 0, 0}, lane); load(m.imag, gFi, {0, 0, 0, 0}, lane);
    complex_mm_AB(y, x, m);                                        // x = x·F
    cmplx_copy(x, y);

    // ---- pointwise convolution in frequency domain ----
    load(m.real, gKr, {0, 0, head, 0}, lane); load(m.imag, gKi, {0, 0, head, 0}, lane);
    cmplx_mul(y, x, m);                                            // x = x ⊙ k_f
    cmplx_copy(x, y);

    // ---- inverse FFT ----
    load(m.real, gIr, {0, 0, 0, 0}, lane); load(m.imag, gIi, {0, 0, 0, 0}, lane);
    complex_mm_AB(y, x, m);                                        // x = x·Finv
    cmplx_copy(x, y);
    cmplx_transpose(y, x, lane);                                   // x = xᵀ
    cmplx_copy(x, y);
    load(m.real, gWIr, {0, 0, 0, 0}, lane); load(m.imag, gWIi, {0, 0, 0, 0}, lane);
    cmplx_mul(y, x, m);                                            // x = x ⊙ tw_ifft
    cmplx_copy(x, y);
    load(m.real, gIr, {0, 0, 0, 0}, lane); load(m.imag, gIi, {0, 0, 0, 0}, lane);
    complex_mm_AB(y, x, m);                                        // x = x·Finv
    cmplx_copy(x, y);
    cmplx_transpose(y, x, lane);                                   // x = xᵀ

    store(gOut, y.real, {0, 0, bh, 0}, lane);                      // out = Re(x)
}

#define instantiate_fftconv(S)                                              \
   template [[host_name("fftconv_" #S)]] [[kernel]]                         \
   void fftconv<float, S>(                                                  \
     device float* OUT [[buffer(0)]], device float* X [[buffer(1)]],        \
     device float* F [[buffer(2)]], device float* TWF [[buffer(3)]],        \
     device float* FINV [[buffer(4)]], device float* TWI [[buffer(5)]],     \
     device float* KF [[buffer(6)]],                                        \
     const constant int &BH [[buffer(7)]], const constant int &H [[buffer(8)]], \
     uint3 tgid [[threadgroup_position_in_grid]],                          \
     uint lane [[thread_index_in_simdgroup]]);

instantiate_fftconv(16);
instantiate_fftconv(32);

}
