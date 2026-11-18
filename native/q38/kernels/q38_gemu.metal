#include <metal_stdlib>
using namespace metal;

// q38 engine GEMV kernels for the uniform planar weight formats u4 and u5.
//
//   value = (n - L/2 + 0.5) * s          n in [0, L), L = 16 (u4) or 32 (u5)
//   s     = gscale * 2^(-c / 16)         one u8 code c per group of 16 weights
//
// Layout per output row: `.w` u8 [K/2], byte j = n4(2j+1) << 4 | n4(2j) with
// n4 the top four bits of n (the same nibble layout as NVFP4); u5 adds `.b`
// u8 [K/8], bit q of byte u = the low bit of n for weight 8u + q; `.scale` u8
// [K/16].
//
// Why these formats (2026-11-09, held-out output error under GPTQ on the MLP
// projections): the E2M1 grid of NVFP4 spends its levels badly on weights that
// are close to Gaussian inside a group. 16 uniform levels are 17% better at
// the same 4.5 bits a weight, and 32 uniform levels (5.5 bits) match FP8 E4M3
// rounding (8 bits) at 69% of the bytes.
//
// Decode needs no integer-to-float conversion: n goes into the top mantissa
// bits of a half with a fixed exponent, 0x3C00 | n4 << 6 | b << 5 = 1 + n/32
// (u4: 1 + n/16), and subtracting the grid centre (exact in half) leaves
// (n - L/2 + 0.5) / L in [-0.5, 0.5). The rest is the NVFP4 kernel's shape
// (kernels/q38_gemv.metal): 4 weight rows per simdgroup, 1 <= M <= R1
// activation rows per pass, 16-term group dots in half, cross-group sums in
// fp32. Inputs are always half * 2^-4 rows (the XH convention); packs with
// these formats run the half-activation path at every M.
//
// Modes: PAIR fused gate/up + SwiGLU (weight has 2N rows); DH with PAIR writes
// the output as half * 2^-4 as well (for a down projection that reads half).
// dispatch (N / 8, ceil(M / R1)), 64 threads; PAIR: (N / 4, ...). N % 8 == 0,
// K % 16 == 0.

template <short R1, bool PAIR, bool DH, bool B5, short GW = 16>
static inline void q38_u_body(device bfloat* D, device const uchar* Wq, device const half* X,
                              device const uchar* WS, device const float* GS, const int N,
                              const int K, const int M, device const uchar* Wb, uint3 tgid,
                              ushort sgitg, ushort lane, const int pair_off) {
  constexpr short NR = 4;
  constexpr short NSG = 2;
  constexpr short WPG = 16 / GW;
  const int sg_index = int(tgid.x) * NSG + sgitg;
  const int first_row = sg_index * (PAIR ? NR / 2 : NR);
  const int col0 = int(tgid.y) * R1;
  const int groups = K / 16;
  // Every load is base pointer + 32-bit index (see q38_gemv.metal).
  device const uint* Wu = (device const uint*)Wq;
  int widx[NR], sidx[NR], bidx[NR];
#pragma clang loop unroll(full)
  for (short r = 0; r < NR; ++r) {
    const int rr = first_row + (PAIR ? (r < NR / 2 ? r : r - NR / 2 + pair_off) : int(r));
    widx[r] = rr * (K / 8);
    sidx[r] = rr * groups;
    bidx[r] = rr * (K / (GW == 16 ? 16 : 8));  // in ushorts (whole group) or bytes (half group)
  }
  const half off = B5 ? -0.484375h : -0.46875h;  // -(L/2 - 0.5) / L

  float sumf[NR][R1];
#pragma clang loop unroll(full)
  for (short r = 0; r < NR; ++r)
#pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) sumf[r][j] = 0.f;
  int xq[R1];  // row offsets in 4-element units
#pragma clang loop unroll(full)
  for (short j = 0; j < R1; ++j) {
    const int c = (col0 + j < M) ? (col0 + j) : (M - 1);  // pad columns clamp; stores are guarded
    xq[j] = c * (K / 4);
  }

  for (int h = int(lane); h < groups * WPG; h += 32) {
    const int g = h / WPG;
    half w[NR][GW];
    float sc[NR];
#pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r) {
      const uint bb = !B5 ? 0u
                      : GW == 16 ? uint(((device const ushort*)Wb)[bidx[r] + h])
                                 : uint(Wb[bidx[r] + h]);
#pragma clang loop unroll(full)
      for (short u = 0; u < GW / 8; ++u) {
        const uint v = Wu[widx[r] + h * (GW / 8) + u];
        const short col = 8 * u;
        // pair t = weights (t, t + 4) of this word: nibbles 16 bits apart
        uint q0 = (v << 6) & 0x03C003C0u;
        uint q1 = (v << 2) & 0x03C003C0u;
        uint q2 = (v >> 2) & 0x03C003C0u;
        uint q3 = (v >> 6) & 0x03C003C0u;
        if (B5) {
          const uint bu = (bb >> (8 * u)) & 0xFFu;
          const uint R = bu | (bu << 12);  // bit t stays, bit t + 4 lands 16 above it
          q0 |= (R << 5) & 0x00200020u;
          q1 |= (R << 4) & 0x00200020u;
          q2 |= (R << 3) & 0x00200020u;
          q3 |= (R << 2) & 0x00200020u;
        }
        // Without the exponent bits the field is a subnormal half, n * 2^-19
        // (u4: n * 2^-18); one fused multiply-add per weight gives the same
        // (n + 0.5 - L/2) / L exactly and saves the OR per pair.
        const half2 h0 = as_type<half2>(q0);
        const half2 h1 = as_type<half2>(q1);
        const half2 h2 = as_type<half2>(q2);
        const half2 h3 = as_type<half2>(q3);
        w[r][col + 0] = fma(h0.x, 16384.0h, off);
        w[r][col + 1] = fma(h1.x, 16384.0h, off);
        w[r][col + 2] = fma(h2.x, 16384.0h, off);
        w[r][col + 3] = fma(h3.x, 16384.0h, off);
        w[r][col + 4] = fma(h0.y, 16384.0h, off);
        w[r][col + 5] = fma(h1.y, 16384.0h, off);
        w[r][col + 6] = fma(h2.y, 16384.0h, off);
        w[r][col + 7] = fma(h3.y, 16384.0h, off);
      }
      sc[r] = exp2(-0.0625f * float(WS[sidx[r] + g]));
    }
#pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) {
      // A barrier between columns pins the order the compiler emits them in:
      // every load of the iteration is then issued up front and waited for
      // once. Measured 2026-11-17, ms per step with / without: 3 rows 32.4 /
      // 32.8, 5 rows 43.2 / 45.6, 6: 48.5 / 50.6, 7: 53.9 / 56.4, 8: 59.8 /
      // 63.0; 4 rows 39.4 / 38.3 (left out), 2 rows no change.
      if ((R1 == 3 || R1 >= 5) && j > 0) simdgroup_barrier(mem_flags::mem_none);
      device const half4* xh4 = (device const half4*)X;
      half xs[GW];
#pragma clang loop unroll(full)
      for (short i = 0; i < GW / 4; ++i) {
        const half4 x4 = xh4[xq[j] + h * (GW / 4) + i];
        xs[4 * i + 0] = x4[0];
        xs[4 * i + 1] = x4[1];
        xs[4 * i + 2] = x4[2];
        xs[4 * i + 3] = x4[3];
      }
#pragma clang loop unroll(full)
      for (short r = 0; r < NR; ++r) {
        if (GW == 8) {
          // fp32 here: the 8-term half sum scored KL 0.01496 against 0.01471
          // (16-term half sum of the narrow kernels: 0.01480) for 1.2 ms a step
          float gsum = 0.0f;
#pragma clang loop unroll(full)
          for (short i = 0; i < GW; ++i) gsum += float(xs[i]) * float(w[r][i]);
          sumf[r][j] += gsum * sc[r];
        } else {
          half gsum = 0.0h;
#pragma clang loop unroll(full)
          for (short i = 0; i < GW; ++i) gsum += xs[i] * w[r][i];
          sumf[r][j] += float(gsum) * sc[r];
        }
      }
    }
  }

  // undo the 2^-4 of the activations and the 1/L of the decode
  const float gs = GS[0] * 16.0f * (B5 ? 32.0f : 16.0f);
  if (PAIR) {
#pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j)
#pragma clang loop unroll(full)
      for (short r = 0; r < NR / 2; ++r) {
        const float gate = simd_sum(sumf[r][j]) * gs;
        const float up = simd_sum(sumf[r + NR / 2][j]) * gs;
        if (lane == 0 && col0 + j < M) {
          const ulong idx = (ulong)(col0 + j) * (ulong)N + first_row + r;
          const float act = gate / (1.0f + exp(-gate)) * up;
          if (DH)
            ((device half*)D)[idx] = half(act * 0.0625f);
          else
            D[idx] = bfloat(act);
        }
      }
    return;
  }
#pragma clang loop unroll(full)
  for (short j = 0; j < R1; ++j)
#pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r) {
      const float tot = simd_sum(sumf[r][j]) * gs;
      if (lane == 0 && col0 + j < M) D[(ulong)(col0 + j) * (ulong)N + first_row + r] = bfloat(tot);
    }
}

#define Q38U_ARGS                                                               \
  device bfloat* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
  device const half* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],     \
  device const float* GS [[buffer(4)]], const constant int& N [[buffer(5)]],    \
  const constant int& K [[buffer(6)]], const constant int& M [[buffer(7)]],     \
  device const uchar* Wb [[buffer(8)]], uint3 tgid [[threadgroup_position_in_grid]], \
  ushort sgitg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// q38_u<b>_mvx_c<R1>   plain GEMV
// q38_u<b>_gux_c<R1>   fused gate/up + SwiGLU, half out
// q38_u<b>_guxb_c<R1>  fused gate/up + SwiGLU, BF16 out
#define Q38U_MV(B, R1)                                                          \
  [[host_name("q38_u" #B "_mvx_c" #R1)]] kernel void q38u##B##_mv_##R1(Q38U_ARGS) { \
    q38_u_body<R1, false, false, B == 5>(D, Wq, X, WS, GS, N, K, M, Wb, tgid, sgitg, lane, 0); \
  }
#define Q38U_GU(B, R1)                                                          \
  [[host_name("q38_u" #B "_gux_c" #R1)]] kernel void q38u##B##_gux_##R1(Q38U_ARGS) { \
    q38_u_body<R1, true, true, B == 5>(D, Wq, X, WS, GS, N, K, M, Wb, tgid, sgitg, lane, N); \
  }                                                                             \
  [[host_name("q38_u" #B "_guxb_c" #R1)]] kernel void q38u##B##_guxb_##R1(Q38U_ARGS) { \
    q38_u_body<R1, true, false, B == 5>(D, Wq, X, WS, GS, N, K, M, Wb, tgid, sgitg, lane, N); \
  }
#define Q38U_ALL(B) \
  Q38U_MV(B, 1) Q38U_MV(B, 2) Q38U_MV(B, 3) Q38U_MV(B, 4) \
  Q38U_GU(B, 1) Q38U_GU(B, 2) Q38U_GU(B, 3) Q38U_GU(B, 4)
Q38U_ALL(4)
Q38U_ALL(5)

// Wide variants (5..8 rows in one pass); see q38_gemv.metal.
#define Q38U_WIDE(B, R1, GW)                                                    \
  [[host_name("q38_u" #B "_mvx_w" #GW "c" #R1)]] kernel void q38u##B##_mvw##GW##_##R1(Q38U_ARGS) { \
    q38_u_body<R1, false, false, B == 5, GW>(D, Wq, X, WS, GS, N, K, M, Wb, tgid, sgitg, lane, 0); \
  }                                                                             \
  [[host_name("q38_u" #B "_gux_w" #GW "c" #R1)]] kernel void q38u##B##_guxw##GW##_##R1(Q38U_ARGS) { \
    q38_u_body<R1, true, true, B == 5, GW>(D, Wq, X, WS, GS, N, K, M, Wb, tgid, sgitg, lane, N); \
  }                                                                             \
  [[host_name("q38_u" #B "_guxb_w" #GW "c" #R1)]] kernel void q38u##B##_guxbw##GW##_##R1(Q38U_ARGS) { \
    q38_u_body<R1, true, false, B == 5, GW>(D, Wq, X, WS, GS, N, K, M, Wb, tgid, sgitg, lane, N); \
  }
#define Q38U_WIDE_ALL(B) \
  Q38U_WIDE(B, 5, 16) Q38U_WIDE(B, 6, 16) Q38U_WIDE(B, 7, 16) Q38U_WIDE(B, 8, 16) \
  Q38U_WIDE(B, 5, 8) Q38U_WIDE(B, 6, 8) Q38U_WIDE(B, 7, 8) Q38U_WIDE(B, 8, 8)
Q38U_WIDE_ALL(4)
Q38U_WIDE_ALL(5)
