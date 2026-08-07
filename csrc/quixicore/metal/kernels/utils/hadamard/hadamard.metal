#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// Walsh-Hadamard transform over the final axis. LPR lanes cooperate on one row
// (32/LPR rows per simdgroup); each lane owns E = D/LPR CONSECUTIVE elements.
// The low log2(E) index bits are butterflied locally in registers, the log2(LPR)
// lane bits via simd_shuffle_xor (masks < LPR never leave the row's lane group) —
// no threadgroup memory, no barriers. (FWHT stages act on independent index bits,
// so stage order is free.) D=64 uses LPR=8: E=8 gives 16-byte loads and 4 rows per
// simdgroup — with LPR=32 each lane moved only 4 bytes per row and the simdgroup
// starved (measured 0.4-0.5x of a plain matmul against H).
template <typename T, int D, int LPR>
kernel void hadamard(device const T *x [[buffer(0)]],
                     device T *out [[buffer(1)]],
                     constant float &scale [[buffer(2)]],
                     constant int &nrows [[buffer(3)]],
                     uint tg   [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    constexpr int E = D / LPR;          // consecutive elements per lane
    constexpr int RSG = 32 / LPR;       // rows per simdgroup
    static_assert(E >= 1 && RSG >= 1, "bad LPR");
    using T4 = metal::vec<T, 4>;
    const int rsg = (int)lane / LPR;
    const int lin = (int)lane % LPR;
    const long row = (long)tg * RSG + rsg;
    const bool live = row < (long)nrows;

    float v[E];
    if (live) {
        const long base = row * D + (long)lin * E;
        if (E % 4 == 0) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < E; i += 4) {
                const T4 t = ((device const T4*)(x + base + i))[0];
                v[i] = float(t.x); v[i + 1] = float(t.y);
                v[i + 2] = float(t.z); v[i + 3] = float(t.w);
            }
        } else {
            #pragma clang loop unroll(full)
            for (int i = 0; i < E; ++i) v[i] = float(x[base + i]);
        }
    } else {
        #pragma clang loop unroll(full)
        for (int i = 0; i < E; ++i) v[i] = 0.0f;   // dead rows still ride the shuffles
    }

    // local butterflies over the in-lane index bits
    #pragma clang loop unroll(full)
    for (int h = 1; h < E; h <<= 1) {
        #pragma clang loop unroll(full)
        for (int i = 0; i < E; ++i) {
            if ((i & h) == 0) {
                const float a = v[i], b = v[i + h];
                v[i] = a + b;
                v[i + h] = a - b;
            }
        }
    }
    // cross-lane butterflies over the log2(LPR) lane bits (stay within the row's group)
    #pragma clang loop unroll(full)
    for (int m = 1; m < LPR; m <<= 1) {
        const bool upper = (lin & m) != 0;
        #pragma clang loop unroll(full)
        for (int i = 0; i < E; ++i) {
            const float p = metal::simd_shuffle_xor(v[i], (ushort)m);
            v[i] = upper ? (p - v[i]) : (v[i] + p);
        }
    }
    if (live) {
        const long base = row * D + (long)lin * E;
        #pragma clang loop unroll(full)
        for (int i = 0; i < E; ++i) out[base + i] = T(v[i] * scale);
    }
}

#define instantiate_hadamard(type_name, T, DVAL, LPRVAL)                   \
  template [[host_name("hadamard_" #type_name "_" #DVAL)]] [[kernel]] void \
  hadamard<T, DVAL, LPRVAL>(device const T *x [[buffer(0)]],               \
                    device T *out [[buffer(1)]],                            \
                    constant float &scale [[buffer(2)]],                    \
                    constant int &nrows [[buffer(3)]],                      \
                    uint tg [[threadgroup_position_in_grid]],               \
                    uint lane [[thread_index_in_simdgroup]]);

#define instantiate_hadamard_type(type_name, T) \
  instantiate_hadamard(type_name, T, 64, 8)     \
  instantiate_hadamard(type_name, T, 128, 16)   \
  instantiate_hadamard(type_name, T, 256, 32)   \
  instantiate_hadamard(type_name, T, 512, 32)

instantiate_hadamard_type(float32, float)
instantiate_hadamard_type(float16, half)
instantiate_hadamard_type(bfloat16, bf16)
