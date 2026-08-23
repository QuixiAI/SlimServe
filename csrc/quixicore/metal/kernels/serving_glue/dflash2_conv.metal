// DFlash 2 block-local dynamic two-tap convolution.
//
// out[t, c] = (base[side, 0, c] + coeff[t, side, 0, c/group]) * x[t, c]
//           + (base[side, 1, c] + coeff[t, side, 1, c/group]) * x[t-1, c]
//
// The predecessor term is zero at the first row of every 1+N draft block.
// This is the exact torch-native _apply_two_tap_conv contract, collapsed from
// repeat_interleave + roll + clone/zero + elementwise ops into one dispatch.
#include <metal_stdlib>

namespace mittens {

template <typename T>
kernel void dflash2_two_tap_conv(
    device const T* x [[buffer(0)]],       // (tokens, hidden)
    device const T* coeffs [[buffer(1)]],  // (tokens, 2 sides, 2 taps, groups)
    device const T* base [[buffer(2)]],    // (2 sides, 2 taps, hidden)
    device T* out [[buffer(3)]],           // (tokens, hidden)
    constant int& tokens [[buffer(4)]], constant int& hidden [[buffer(5)]],
    constant int& groups [[buffer(6)]], constant int& group_size [[buffer(7)]],
    constant int& block_size [[buffer(8)]], constant int& side [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  const uint total = uint(tokens) * uint(hidden);
  if (gid >= total) return;

  const int t = int(gid / uint(hidden));
  const int c = int(gid - uint(t) * uint(hidden));
  const int group = c / group_size;
  const long coeff_row = ((long)t * 2 + side) * 2 * groups;
  const long base_row = (long)side * 2 * hidden;

  const T x0 = x[gid];
  const bool has_prev = (t % block_size) != 0;
  const T x1 = has_prev ? x[(long)(t - 1) * hidden + c] : T(0.0f);
  // Preserve the torch-native expression's dtype boundaries: base+dyn,
  // each product, and the final sum are separate T-valued tensor ops there.
  // Rounding at the same three sites keeps bf16/fp16 model numerics stable.
  const T w0 = T(float(base[base_row + c]) + float(coeffs[coeff_row + group]));
  const T w1 = T(float(base[base_row + hidden + c]) +
                 float(coeffs[coeff_row + groups + group]));
  const T y0 = T(float(w0) * float(x0));
  const T y1 = T(float(w1) * float(x1));
  out[gid] = T(float(y0) + float(y1));
}

#define instantiate_dflash2_conv(tname, T)                                    \
  template [[host_name("dflash2_two_tap_conv_" #tname)]] [[kernel]] void      \
  dflash2_two_tap_conv<T>(                                                    \
      device const T* x [[buffer(0)]], device const T* coeffs [[buffer(1)]],  \
      device const T* base [[buffer(2)]], device T* out [[buffer(3)]],        \
      constant int& tokens [[buffer(4)]], constant int& hidden [[buffer(5)]], \
      constant int& groups [[buffer(6)]],                                     \
      constant int& group_size [[buffer(7)]],                                 \
      constant int& block_size [[buffer(8)]],                                 \
      constant int& side [[buffer(9)]], uint gid [[thread_position_in_grid]]);

instantiate_dflash2_conv(float32, float) instantiate_dflash2_conv(float16, half)
    instantiate_dflash2_conv(bfloat16, bfloat)

}  // namespace mittens
