// SPDX-License-Identifier: Apache-2.0
// Fused DFlash2 grouped dynamic convolution (drafter block conv).
//
// Replaces the ~10-op torch chain in qwen3_dflash2._grouped_conv
// (unflatten, coefficient add, per-tap pad + three muls + add, arange
// position mask, flatten) with one dispatch. The conv is block-local:
// query blocks are [block_size] consecutive rows per request, and tap k
// of row t reads row t-k only when the row's position within its block
// is >= k (no cross-block leakage; the torch form zero-pads and masks).
//
// Numerics mirror the eager torch chain: every binary op computes in
// fp32 and rounds once to the storage dtype (MPS elementwise semantics),
// in the same association order:
//   c_tap = round(base[tap] + delta[tap])
//   acc   = round(c_0 * x[t])
//   acc   = round(acc + round(c_tap * x[t - tap]))   (masked taps skipped:
//           the torch chain multiplies that term by 0.0 and adds exact 0)
//
// x is [T, H] rows; delta rows are [taps, G] with a caller-supplied row
// stride (the [T, 2, taps, G] projection view passes side slices without
// a contiguous copy); base is [taps, H] contiguous; out is [T, H].

#include <metal_stdlib>

template <typename T>
METAL_FUNC void qc_dflash_conv_body(device const T *x,
                                    device const T *delta,
                                    device const T *base, device T *out,
                                    int H, int num_groups, int group_size,
                                    int taps, int block_size,
                                    int delta_row_stride, uint gid,
                                    int total) {
    if ((int)gid >= total) { return; }
    const int t = (int)((long)gid / H);
    const int h = (int)((long)gid - (long)t * H);
    const int g = h / group_size;
    const int pos = (block_size & (block_size - 1)) == 0
                        ? (t & (block_size - 1))
                        : (t % block_size);
    {
        #pragma clang fp reassociate(off) contract(off)
        const float d0 = float(delta[(long)t * delta_row_stride + g]);
        const T c0 = T(float(base[h]) + d0);
        T acc = T(float(c0) * float(x[gid]));
        for (int tap = 1; tap < taps; ++tap) {
            if (pos < tap || t < tap) { continue; }
            const float dt =
                float(delta[(long)t * delta_row_stride + tap * num_groups + g]);
            const T ct = T(float(base[tap * H + h]) + dt);
            const T prod = T(float(ct) * float(x[(long)(t - tap) * H + h]));
            acc = T(float(acc) + float(prod));
        }
        out[gid] = acc;
    }
}

#define instantiate_qc_dflash_conv(suffix, T)                                \
  [[host_name("qc_dflash_conv_" #suffix)]] kernel void                       \
  qc_dflash_conv_##suffix(                                                   \
      device const T *x [[buffer(0)]],                                       \
      device const T *delta [[buffer(1)]],                                   \
      device const T *base [[buffer(2)]], device T *out [[buffer(3)]],       \
      constant int &H [[buffer(4)]],                                         \
      constant int &num_groups [[buffer(5)]],                                \
      constant int &group_size [[buffer(6)]],                                \
      constant int &taps [[buffer(7)]],                                      \
      constant int &block_size [[buffer(8)]],                                \
      constant int &delta_row_stride [[buffer(9)]],                          \
      constant int &total [[buffer(10)]],                                    \
      uint gid [[thread_position_in_grid]]) {                                \
    qc_dflash_conv_body<T>(x, delta, base, out, H, num_groups, group_size,   \
                           taps, block_size, delta_row_stride, gid, total);  \
  }

instantiate_qc_dflash_conv(float16, half)
instantiate_qc_dflash_conv(bfloat16, bfloat)
instantiate_qc_dflash_conv(float32, float)
