#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

#define PARAMS(T) \
    device T* y [[buffer(1)]], \
    device T* x [[buffer(0)]], \
    device T* out [[buffer(2)]], \
    constant const int& ndim1 [[buffer(3)]], \
    constant const int& ndim2 [[buffer(4)]], \
    uint tid [[thread_position_in_grid]]

// Elementwise out = x + y over a contiguous (ndim1, ndim2) buffer — flat, one thread per
// 4 elements (vec4 loads/stores), last thread takes the scalar tail. This began life as the
// 8x8-register-tile smoke test of the TK port; that layout measured 0.34x of mx add (64
// elements per 32-thread group through 2-element fragment gathers). The register-tile
// load/add/store path remains covered by the Xcode primitive tests and every MMA kernel.
template <typename T>
[[kernel]] void add_rt(PARAMS(T)) {
    using T4 = metal::vec<T, 4>;
    const ulong n = (ulong)ndim1 * (ulong)ndim2;
    const ulong base = (ulong)tid * 8;           // two vec4s -> 16-byte transactions for half/bf16
    if (base + 8 <= n) {
        const T4 a0 = ((device const T4*)(x + base))[0];
        const T4 a1 = ((device const T4*)(x + base))[1];
        const T4 b0 = ((device const T4*)(y + base))[0];
        const T4 b1 = ((device const T4*)(y + base))[1];
        ((device T4*)(out + base))[0] = a0 + b0;
        ((device T4*)(out + base))[1] = a1 + b1;
    } else {
        for (ulong i = base; i < n; ++i) out[i] = x[i] + y[i];
    }
}

#define instantiate_add_custom(type_name, T)                           \
  template [[host_name("add_rt_" #type_name)]] [[kernel]] void         \
  add_rt<T>(PARAMS(T));

instantiate_add_custom(float32, float);
instantiate_add_custom(float16, half);
instantiate_add_custom(bfloat16, bf16);

}
