#include <metal_stdlib>
using namespace metal;

// q38 engine GEMV kernels: BF16 activations, NVFP4 weights in the q38 pack
// layout (u8 [N, K/2] E2M1 nibbles, low nibble = even column; u8 [N, K/16]
// E4M3 group scales; one fp32 global multiplier). They are not bit-identical
// to the QuixiCore kernels and do not try to be: the engine's contract is
// parity with the CPU oracle within tolerance, which frees the arithmetic.
//
// One body, 1 <= M <= R1 activation rows per pass, 4 weight rows per
// simdgroup, weights read ONCE for all rows. The 16-term group dot runs in
// half: E2M1 weights are exact in half, the group sum is widened to fp32
// before the group scale is applied and the cross-group sum stays fp32. Half
// halves the live register state, which is what lets 4 activation rows share
// one pass (the fp32 QuixiCore kernel is register-bound at 2, so M = 4 read
// every weight twice). The nibble decode yields value * 2^-14 (a subnormal
// half for the 0.5 code), so weights are rescaled by 2^14 before any half
// arithmetic; activations are pre-scaled by 2^-4 so a 16-term sum cannot
// overflow below |x| ~ 1e4. Both are exact power-of-two scalings.
// Measured dead ends (2026-11-09, M1 Ultra): fp32 group dot (slower at M>=3),
// 8 weight rows or 8 columns per pass (register blow-up, 2x slower; 6 and 8
// columns re-measured with half inputs: 8 rows in 51 ms vs 33 ms as 2 x 4), leaving
// weights subnormal and scaling x instead (no faster, less accurate; again
// with half inputs scaled 2^6: 33.4 -> 32.2 ms at M = 3, nothing at M = 4, and
// the activation range shrinks to |x| < 1023).
//
// Modes (compile-time):
//   PAIR  fused MLP front half: the 4 rows are 2 gate rows and their 2 up
//         rows `pair_off` rows further down; output is silu(gate) * up.
//   NORM  the input row is residual * (1 + w), written by a RES kernel; this
//         kernel finishes the Gemma RMSNorm with rsqrt(mean(x^2) + eps), with
//         sum(x^2) read from the accumulator the producing kernel filled
//         (`Pin`, two words per row). Removes the standalone norm dispatch.
//   XH    the input rows are `half`, already scaled by 2^-4 (written by
//         q38_norm_add, q38_gdn_core or a PAIR+XH kernel), instead of BF16.
//         Converting BF16 -> scaled half per group and per column was ~1.5 of
//         the ~10 ALU ops per weight at M = 3 (5 ms of a 39 ms step).
//   DH    (PAIR only) the output is written as half * 2^-4 too, for a down
//         projection that is itself an XH kernel; without it the output is
//         BF16 (a down projection in another format).
//   RES   the output is a residual-stream row: D = W x + residual, plus the
//         same row times the next norm's (1 + w) in `Hs`, and the
//         sum of squares of what was written is accumulated into `Pout`
//         (host-zeroed, two words per row) for the next NORM kernel.
//
// dispatch (N / 8, ceil(M / R1)), 64 threads; PAIR: (N / 4, ...) with N the
// number of outputs. N % 8 == 0, K % 16 == 0.

// Sum of squares of a residual row, accumulated across the producing
// kernel's simdgroups as 64-bit fixed point (units of 2^-22, rounded; 2^-10
// truncated could be up to 1.25 off a row total) in two atomic
// words [lo, hi]. Integer adds make the total exact and independent of the
// order the simdgroups finish in, so outputs stay reproducible. (A first
// version left one float partial per simdgroup for the consumer to sum; 1280
// loads in every consumer simdgroup cost 14 ms a step at M = 2.)
static inline void q38_ss_add(device atomic_uint* acc, float ss) {
  const ulong u = ulong(ss * 4194304.0f + 0.5f);
  const uint lo = uint(u & 0xFFFFFFFFul);
  const uint old = atomic_fetch_add_explicit(&acc[0], lo, memory_order_relaxed);
  const uint carry = (old > 0xFFFFFFFFu - lo) ? 1u : 0u;
  atomic_fetch_add_explicit(&acc[1], uint(u >> 32) + carry, memory_order_relaxed);
}
static inline float q38_ss_load(device const atomic_uint* acc) {
  const uint lo = atomic_load_explicit(&acc[0], memory_order_relaxed);
  const uint hi = atomic_load_explicit(&acc[1], memory_order_relaxed);
  return (float(hi) * 4294967296.0f + float(lo)) * (1.0f / 4194304.0f);
}

template <short R1, bool PAIR, bool NORM, bool RES, bool XH = false, bool DH = false, short GW = 16>
static inline void q38_nvfp4_body(
    device bfloat* D, device const uchar* Wq, device const bfloat* X, device const uchar* WS,
    device const float* GS, const int N, const int K, const int M, device const bfloat* NW,
    device bfloat* Hs, device const atomic_uint* Pin, const int npart_in, const float eps, device const bfloat* Rsd,
    device atomic_uint* Pout, uint3 tgid, ushort sgitg, ushort lane, const int pair_off) {
  constexpr short NR = 4;
  constexpr short NSG = 2;
  constexpr short WPG = 16 / GW;  // lane iterations per 16-weight scale group
  const int sg_index = int(tgid.x) * NSG + sgitg;
  const int first_row = sg_index * (PAIR ? NR / 2 : NR);
  const int col0 = int(tgid.y) * R1;
  const int groups = K / 16;
  // Every load is base pointer + 32-bit index: the GPU's load instruction takes
  // that form directly, where 64-bit pointer arithmetic costs 3 to 4 integer
  // instructions per address and two registers per pointer (measured
  // 2026-11-17: 8 rows 68.2 -> 62.6 ms, 4 rows 39.5 -> 38.3). Largest index:
  // the LM head's 635 MB / 4.
  device const uint* Wu = (device const uint*)Wq;
  int widx[NR], sidx[NR];
#pragma clang loop unroll(full)
  for (short r = 0; r < NR; ++r) {
    const int rr = first_row + (PAIR ? (r < NR / 2 ? r : r - NR / 2 + pair_off) : int(r));
    widx[r] = rr * (K / 8);
    sidx[r] = rr * groups;
  }

  float sumf[NR][R1];
#pragma clang loop unroll(full)
  for (short r = 0; r < NR; ++r)
#pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) sumf[r][j] = 0.f;
  // pad columns clamp to the last live row; their stores are guarded
  int cols[R1];
  int xq[R1];  // row offsets in 4-element units
  float xsc[R1];
#pragma clang loop unroll(full)
  for (short j = 0; j < R1; ++j) {
    cols[j] = (col0 + j < M) ? (col0 + j) : (M - 1);
    xq[j] = cols[j] * (K / 4);
    xsc[j] = 0.0625f;
  }
  if (NORM) {
#pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) {
      const float ss = q38_ss_load(Pin + 2 * cols[j]);
      xsc[j] = 0.0625f * rsqrt(ss / float(K) + eps);
    }
  }

  for (int h = int(lane); h < groups * WPG; h += 32) {
    const int g = h / WPG;
    half w[NR][GW];
    float sc[NR];
#pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r) {
#pragma clang loop unroll(full)
      for (short u = 0; u < GW / 8; ++u) {
        const uint v = Wu[widx[r] + h * (GW / 8) + u];
        const short col = 8 * u;
        const half2 le = as_type<half2>(((v << 9) & 0x0E000E00u) | ((v << 12) & 0x80008000u));
        const half2 lo = as_type<half2>(((v << 1) & 0x0E000E00u) | ((v << 4) & 0x80008000u));
        const half2 he = as_type<half2>(((v << 5) & 0x0E000E00u) | ((v << 8) & 0x80008000u));
        const half2 ho = as_type<half2>(((v >> 3) & 0x0E000E00u) | (v & 0x80008000u));
        w[r][col + 0] = le.x * 16384.0h;
        w[r][col + 1] = he.x * 16384.0h;
        w[r][col + 2] = lo.x * 16384.0h;
        w[r][col + 3] = ho.x * 16384.0h;
        w[r][col + 4] = le.y * 16384.0h;
        w[r][col + 5] = he.y * 16384.0h;
        w[r][col + 6] = lo.y * 16384.0h;
        w[r][col + 7] = ho.y * 16384.0h;
      }
      const uint sb = WS[sidx[r] + g];
      const half sh = as_type<half>(ushort(((sb & 0x7F) << 7) | ((sb & 0x80) << 8)));
      sc[r] = 256.0f * float(sh);
    }
#pragma clang loop unroll(full)
    for (short j = 0; j < R1; ++j) {
      // A barrier between columns pins the order the compiler emits them in:
      // every load of the iteration is then issued up front and waited for
      // once. Measured 2026-11-17, ms per step with / without: 3 rows 32.4 /
      // 32.8, 5 rows 43.2 / 45.6, 6: 48.5 / 50.6, 7: 53.9 / 56.4, 8: 59.8 /
      // 63.0; 4 rows 39.4 / 38.3 (left out), 2 rows no change.
      if ((R1 == 3 || R1 >= 5) && j > 0) simdgroup_barrier(mem_flags::mem_none);
      half xs[GW];
      if (XH) {
        device const half4* xh4 = (device const half4*)X;
#pragma clang loop unroll(full)
        for (short i = 0; i < GW / 4; ++i) {
          const half4 x4 = xh4[xq[j] + h * (GW / 4) + i];
          xs[4 * i + 0] = x4[0];
          xs[4 * i + 1] = x4[1];
          xs[4 * i + 2] = x4[2];
          xs[4 * i + 3] = x4[3];
        }
      } else {
        device const bfloat4* xp4 = (device const bfloat4*)X;
#pragma clang loop unroll(full)
        for (short i = 0; i < GW / 4; ++i) {
          const float4 x4 = float4(xp4[xq[j] + h * (GW / 4) + i]) * xsc[j];
          xs[4 * i + 0] = half(x4[0]);
          xs[4 * i + 1] = half(x4[1]);
          xs[4 * i + 2] = half(x4[2]);
          xs[4 * i + 3] = half(x4[3]);
        }
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

  const float gs = GS[0] * 16.0f;
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
  for (short j = 0; j < R1; ++j) {
    float ss = 0.f;
#pragma clang loop unroll(full)
    for (short r = 0; r < NR; ++r) {
      float tot = simd_sum(sumf[r][j]) * gs;
      const ulong idx = (ulong)(col0 + j) * (ulong)N + first_row + r;
      if (RES) {
        if (lane == 0 && col0 + j < M) {
          tot += float(Rsd[idx]);
          ss += tot * tot;
          D[idx] = bfloat(tot);
          Hs[idx] = bfloat(tot * (1.0f + float(NW[first_row + r])));
        }
      } else if (lane == 0 && col0 + j < M) {
        D[idx] = bfloat(tot);
      }
    }
    if (RES && lane == 0 && col0 + j < M) q38_ss_add(Pout + 2 * (col0 + j), ss);
  }
}

#define Q38_ARGS_BASE                                                           \
  device bfloat* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
  device const bfloat* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],   \
  device const float* GS [[buffer(4)]], const constant int& N [[buffer(5)]],    \
  const constant int& K [[buffer(6)]], const constant int& M [[buffer(7)]]
#define Q38_ARGS_NORM                                                           \
  , device const atomic_uint* Pin [[buffer(9)]], const constant float& eps [[buffer(11)]]
#define Q38_ARGS_RES                                                            \
  , device const bfloat* Rsd [[buffer(12)]], device atomic_uint* Pout [[buffer(13)]],  \
  device const bfloat* NW [[buffer(14)]], device bfloat* Hs [[buffer(15)]]
#define Q38_ARGS_IDS                                                            \
  , uint3 tgid [[threadgroup_position_in_grid]],                                \
  ushort sgitg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]

// plain:  q38_nvfp4_mv_h_r4c<R1>
#define Q38_MV(name, R1)                                                        \
  [[host_name(name)]] kernel void q38_mv_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) {     \
    q38_nvfp4_body<R1, false, false, false>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, 0, 0.f, \
                                            nullptr, nullptr, tgid, sgitg, lane, 0); \
  }
// norm-folding consumer:  q38_nvfp4_mvn_c<R1>
#define Q38_MVN(name, R1)                                                       \
  [[host_name(name)]] kernel void q38_mvn_##R1(Q38_ARGS_BASE Q38_ARGS_NORM Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, false, true, false>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, Pin, 0, eps, \
                                           nullptr, nullptr, tgid, sgitg, lane, 0); \
  }
// residual producer:  q38_nvfp4_mvr_c<R1>
#define Q38_MVR(name, R1)                                                       \
  [[host_name(name)]] kernel void q38_mvr_##R1(Q38_ARGS_BASE Q38_ARGS_RES Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, false, false, true>(D, Wq, X, WS, GS, N, K, M, NW, Hs, nullptr, 0, 0.f, \
                                           Rsd, Pout, tgid, sgitg, lane, 0);    \
  }
// fused gate/up + SwiGLU (N = outputs per row, weight has 2N rows):  q38_nvfp4_gu_h_c<R1>
#define Q38_GU(name, R1)                                                        \
  [[host_name(name)]] kernel void q38_gu_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) {     \
    q38_nvfp4_body<R1, true, false, false>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, 0, 0.f, \
                                           nullptr, nullptr, tgid, sgitg, lane, N); \
  }
// the same reading a raw residual row:  q38_nvfp4_gun_c<R1>
#define Q38_GUN(name, R1)                                                       \
  [[host_name(name)]] kernel void q38_gun_##R1(Q38_ARGS_BASE Q38_ARGS_NORM Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, true, true, false>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, Pin, 0, eps, \
                                          nullptr, nullptr, tgid, sgitg, lane, N); \
  }

// half-input variants:  q38_nvfp4_mvx_c<R1>, q38_nvfp4_gux_c<R1> (half out), q38_nvfp4_guxb_c<R1> (BF16 out)
#define Q38_MVX(name, R1)                                                       \
  [[host_name(name)]] kernel void q38_mvx_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) {    \
    q38_nvfp4_body<R1, false, false, false, true>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, 0, \
                                                  0.f, nullptr, nullptr, tgid, sgitg, lane, 0); \
  }
#define Q38_GUX(name, R1)                                                       \
  [[host_name(name)]] kernel void q38_gux_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) {    \
    q38_nvfp4_body<R1, true, false, false, true, true>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, \
                                                       0, 0.f, nullptr, nullptr, tgid, sgitg, lane, N); \
  }
#define Q38_GUXB(name, R1)                                                      \
  [[host_name(name)]] kernel void q38_guxb_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) {   \
    q38_nvfp4_body<R1, true, false, false, true, false>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, \
                                                        0, 0.f, nullptr, nullptr, tgid, sgitg, lane, N); \
  }

Q38_MVX("q38_nvfp4_mvx_c1", 1)
Q38_MVX("q38_nvfp4_mvx_c2", 2)
Q38_MVX("q38_nvfp4_mvx_c3", 3)
Q38_MVX("q38_nvfp4_mvx_c4", 4)
Q38_GUX("q38_nvfp4_gux_c1", 1)
Q38_GUX("q38_nvfp4_gux_c2", 2)
Q38_GUX("q38_nvfp4_gux_c3", 3)
Q38_GUX("q38_nvfp4_gux_c4", 4)
Q38_GUXB("q38_nvfp4_guxb_c1", 1)
Q38_GUXB("q38_nvfp4_guxb_c2", 2)
Q38_GUXB("q38_nvfp4_guxb_c3", 3)
Q38_GUXB("q38_nvfp4_guxb_c4", 4)
Q38_MV("q38_nvfp4_mv_h_r4c1", 1)
Q38_MV("q38_nvfp4_mv_h_r4c2", 2)
Q38_MV("q38_nvfp4_mv_h_r4c3", 3)
Q38_MV("q38_nvfp4_mv_h_r4c4", 4)
Q38_MVN("q38_nvfp4_mvn_c1", 1)
Q38_MVN("q38_nvfp4_mvn_c2", 2)
Q38_MVN("q38_nvfp4_mvn_c3", 3)
Q38_MVN("q38_nvfp4_mvn_c4", 4)
Q38_MVR("q38_nvfp4_mvr_c1", 1)
Q38_MVR("q38_nvfp4_mvr_c2", 2)
Q38_MVR("q38_nvfp4_mvr_c3", 3)
Q38_MVR("q38_nvfp4_mvr_c4", 4)
Q38_GU("q38_nvfp4_gu_h_c1", 1)
Q38_GU("q38_nvfp4_gu_h_c2", 2)
Q38_GU("q38_nvfp4_gu_h_c3", 3)
Q38_GU("q38_nvfp4_gu_h_c4", 4)
Q38_GUN("q38_nvfp4_gun_c1", 1)
Q38_GUN("q38_nvfp4_gun_c2", 2)
Q38_GUN("q38_nvfp4_gun_c3", 3)
Q38_GUN("q38_nvfp4_gun_c4", 4)

// Wide variants: 5..8 activation rows in one pass. GW = weights a lane takes
// per iteration (16 = one scale group, 8 = half a group, which halves the
// decoded-weight register state).
#define Q38_WIDE(R1, GW)                                                        \
  [[host_name("q38_nvfp4_mvx_w" #GW "c" #R1)]] kernel void q38_mvx_w##GW##_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, false, false, false, true, false, GW>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, 0, \
                                                  0.f, nullptr, nullptr, tgid, sgitg, lane, 0); \
  }                                                                             \
  [[host_name("q38_nvfp4_mv_h_w" #GW "c" #R1)]] kernel void q38_mv_w##GW##_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, false, false, false, false, false, GW>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, 0, \
                                                  0.f, nullptr, nullptr, tgid, sgitg, lane, 0); \
  }                                                                             \
  [[host_name("q38_nvfp4_gux_w" #GW "c" #R1)]] kernel void q38_gux_w##GW##_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, true, false, false, true, true, GW>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, \
                                                       0, 0.f, nullptr, nullptr, tgid, sgitg, lane, N); \
  }                                                                             \
  [[host_name("q38_nvfp4_guxb_w" #GW "c" #R1)]] kernel void q38_guxb_w##GW##_##R1(Q38_ARGS_BASE Q38_ARGS_IDS) { \
    q38_nvfp4_body<R1, true, false, false, true, false, GW>(D, Wq, X, WS, GS, N, K, M, nullptr, nullptr, nullptr, \
                                                        0, 0.f, nullptr, nullptr, tgid, sgitg, lane, N); \
  }
Q38_WIDE(5, 16) Q38_WIDE(6, 16) Q38_WIDE(7, 16) Q38_WIDE(8, 16)
Q38_WIDE(5, 8) Q38_WIDE(6, 8) Q38_WIDE(7, 8) Q38_WIDE(8, 8)
