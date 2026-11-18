#include <metal_stdlib>
using namespace metal;

// q38 GEMM: many activation rows (prompt prefill) against one quantized
// projection, D[m][n] = sum_k W[n][k] * X[m][k].
//
// The GEMV family (q38_gemv.metal, q38_gemu.metal) keeps everything in
// registers and tops out at 4 activation rows per pass over the weights:
// 3.3e12 weight-row products a second on the M1 Ultra, 116 prompt tokens a
// second for this model. llama.cpp's Metal backend prefills the same model at
// 254 tokens a second (measured 2026-11-10) with the GPU's 8x8 matrix
// primitive, simdgroup_multiply_accumulate. This kernel is that tile
// structure with our own weight decode:
//
//   one threadgroup (128 threads, 4 simdgroups) owns a tile of 64 weight rows
//   x 32 activation rows and walks K in steps of 32. Per step each thread
//   decodes 16 weights (one scale group of one row) and loads 8 activations
//   into threadgroup memory as half 8x8 blocks; each simdgroup then does 8
//   multiply-accumulates per 8 columns of K into its 32 x 16 corner of the
//   tile (float accumulators).
//
// Weights go into the blocks with their group scale applied (the product is
// exact in half for NVFP4: 2 significant bits times 4; u4/u5 scales are
// 2^(-c/16), rounded to half, 2^-12 relative, far under the quantization
// step). Activations are half * 2^-4 as everywhere else in the engine.
//
// Layouts are the q38 pack's (see q38_gemv.metal / q38_gemu.metal). N % 64
// == 0, K % 32 == 0. dispatch (ceil(M / 32), N / 64) x 128 threads.

// 16 weights of scale group g of weight row `row`, times the group scale
// (NVFP4), or times 2^(8 - c/16) (u4/u5; the caller divides the 2^8 and the
// grid's 1/L back out).
template <int FMT>
static inline void q38_dq16(device const uchar* Wq, device const uchar* WS, device const uchar* Wb,
                            const ulong row, const int g, const int K, thread half* o) {
  const ulong wrow = (ulong)K / 2;
  const uint2 p = *(device const uint2*)(Wq + row * wrow + (ulong)g * 8);
  const uint sbyte = WS[row * (ulong)(K / 16) + (ulong)g];
  if (FMT == 1) {
    const half sh = as_type<half>(ushort(((sbyte & 0x7F) << 7) | ((sbyte & 0x80) << 8)));
    // E2M1 decode gives value * 2^-14 (subnormal for the 0.5 code) and the scale byte decodes to
    // scale * 2^-8: both are undone on the scale side
    // The weights enter the half blocks as value * scale * 2^-8 (at most 10.5): with the full scale
    // (up to 2688) the half products with large activations (post-SwiGLU rows reach the hundreds)
    // lose their low bits or overflow. The 2^8 comes back with the global scale.
    // One half multiply per weight: the E2M1 decode is value * 2^-14 (the 0.5 code is a
    // subnormal half; half arithmetic handles it, as in the GEMV kernels), the scale byte
    // decodes to scale * 2^-8, and (scale * 2^-8 * 2^14) fits a half. The block entry is
    // value * scale * 2^-8 (at most 10.5), exact; the 2^8 comes back with the global scale.
    const half hs = sh * 16384.0h;
#pragma clang loop unroll(full)
    for (short u = 0; u < 2; ++u) {
      const uint v = (u == 0) ? p.x : p.y;
      const short col = 8 * u;
      const half2 le = as_type<half2>(((v << 9) & 0x0E000E00u) | ((v << 12) & 0x80008000u)) * hs;
      const half2 lo = as_type<half2>(((v << 1) & 0x0E000E00u) | ((v << 4) & 0x80008000u)) * hs;
      const half2 he = as_type<half2>(((v << 5) & 0x0E000E00u) | ((v << 8) & 0x80008000u)) * hs;
      const half2 ho = as_type<half2>(((v >> 3) & 0x0E000E00u) | (v & 0x80008000u)) * hs;
      o[col + 0] = le.x;
      o[col + 1] = he.x;
      o[col + 2] = lo.x;
      o[col + 3] = ho.x;
      o[col + 4] = le.y;
      o[col + 5] = he.y;
      o[col + 6] = lo.y;
      o[col + 7] = ho.y;
    }
  } else {
    const bool B5 = FMT == 4;
    const half centre = B5 ? 1.484375h : 1.46875h;  // 1 + (L/2 - 0.5) / L
    const half s = half(exp2(8.0f - 0.0625f * float(sbyte)));
    const uint bb = B5 ? uint(*(device const ushort*)(Wb + row * ((ulong)K / 8) + (ulong)g * 2)) : 0u;
#pragma clang loop unroll(full)
    for (short u = 0; u < 2; ++u) {
      const uint v = (u == 0) ? p.x : p.y;
      const short col = 8 * u;
      uint q0 = (v << 6) & 0x03C003C0u;
      uint q1 = (v << 2) & 0x03C003C0u;
      uint q2 = (v >> 2) & 0x03C003C0u;
      uint q3 = (v >> 6) & 0x03C003C0u;
      if (B5) {
        const uint bu = (bb >> (8 * u)) & 0xFFu;
        const uint R = bu | (bu << 12);
        q0 |= (R << 5) & 0x00200020u;
        q1 |= (R << 4) & 0x00200020u;
        q2 |= (R << 3) & 0x00200020u;
        q3 |= (R << 2) & 0x00200020u;
      }
      const half2 h0 = as_type<half2>(q0 | 0x3C003C00u);
      const half2 h1 = as_type<half2>(q1 | 0x3C003C00u);
      const half2 h2 = as_type<half2>(q2 | 0x3C003C00u);
      const half2 h3 = as_type<half2>(q3 | 0x3C003C00u);
      o[col + 0] = (h0.x - centre) * s;
      o[col + 1] = (h1.x - centre) * s;
      o[col + 2] = (h2.x - centre) * s;
      o[col + 3] = (h3.x - centre) * s;
      o[col + 4] = (h0.y - centre) * s;
      o[col + 5] = (h1.y - centre) * s;
      o[col + 6] = (h2.y - centre) * s;
      o[col + 7] = (h3.y - centre) * s;
    }
  }
}

// Block element type of the 8x8 products: half. Float blocks (Q38_GEMM_F32) were measured
// 2026-11-10: 12% slower (179 vs 201 prompt tokens/s) and no better (KL of the rows after a
// 192-token big prefill 0.02092 float, 0.02088 half, 0.02092 through the 8-row GEMV path).
#ifdef Q38_GEMM_F32
typedef float q38_blk;
typedef float4 q38_blk4;
typedef float2x4 q38_blk2x4;
typedef simdgroup_float8x8 q38_m8;
#else
typedef half q38_blk;
typedef half4 q38_blk4;
typedef half2x4 q38_blk2x4;
typedef simdgroup_half8x8 q38_m8;
#endif

// XH: the activation rows are half * 2^-4 already (the engine's half copies)
// instead of BF16.
template <int FMT, bool XH>
static inline void q38_gemm_body(device bfloat* D, device const uchar* Wq, device const bfloat* X,
                                 device const uchar* WS, device const float* GS, const int N,
                                 const int K, const int M, device const uchar* Wb,
                                 threadgroup q38_blk* sa, threadgroup q38_blk* sb,
                                 uint3 tg, ushort tiitg, ushort sgitg) {
  const int r0 = int(tg.y) * 64;  // first weight row of the tile
  const int r1 = int(tg.x) * 32;  // first activation row of the tile
  const short nr1 = short(min(32, M - r1));
  const short arow = short(tiitg / 2), ag = short(tiitg % 2);
  const short brow = short(tiitg / 4), bk = short(tiitg % 4);
  const short lr1 = brow < nr1 ? brow : short(nr1 - 1);  // pad rows repeat the last live one; never stored
  device const bfloat* xrow = X + (ulong)(r1 + lr1) * (ulong)K + 8 * bk;

  simdgroup_float8x8 mc[8];
#pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);

  for (int k0 = 0; k0 < K; k0 += 32) {
    half w16[16];
    q38_dq16<FMT>(Wq, WS, Wb, (ulong)(r0 + arow), k0 / 16 + ag, K, w16);
    // the 8 activations as one 2 x 4 matrix: one store into the B block row
    q38_blk2x4 xb;
    if (XH) {
      device const half4* xp = (device const half4*)((device const half*)xrow + k0);
      xb = q38_blk2x4(q38_blk4(xp[0]), q38_blk4(xp[1]));
    } else {
      device const bfloat4* xp = (device const bfloat4*)(xrow + k0);
      xb = q38_blk2x4(q38_blk4(float4(xp[0]) * 0.0625f), q38_blk4(float4(xp[1]) * 0.0625f));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the previous step's loads are done
    {
      // A blocks: [k][row % 8], block index 8 * (k / 8) + row / 8
      const short sy = arow / 8, lx = arow % 8;
#pragma clang loop unroll(full)
      for (short i = 0; i < 16; ++i) {
        const short sx = 2 * ag + i / 8;
        *(sa + 64 * (8 * sx + sy) + 8 * (i % 8) + lx) = w16[i];
      }
      // B blocks: [row % 8][k], block index 4 * (k / 8) + row / 8
      *(threadgroup q38_blk2x4*)(sb + 64 * (4 * bk + brow / 8) + 8 * (brow % 8)) = xb;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const q38_blk* lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const q38_blk* lsmb = sb + 2 * 64 * (sgitg / 2);
#pragma clang loop unroll(full)
    for (short ik = 0; ik < 4; ++ik) {
      q38_m8 ma[4];
      q38_m8 mb[2];
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (short i = 0; i < 4; ++i) simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (short i = 0; i < 2; ++i) simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
      simdgroup_barrier(mem_flags::mem_none);
#pragma clang loop unroll(full)
      for (short i = 0; i < 8; ++i) simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
      lsma += 8 * 64;
      lsmb += 4 * 64;
    }
  }

  threadgroup float* sof = (threadgroup float*)sa;
  const float gs = GS[0] * 16.0f * (FMT == 1 ? 256.0f : (FMT == 4 ? 32.0f : 16.0f) * (1.0f / 256.0f));
#pragma clang loop unroll(full)
  for (short rd = 0; rd < 2; ++rd) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float* sc = sof + 32 * (sgitg & 1) + (8 * (sgitg >> 1)) * 64;
#pragma clang loop unroll(full)
    for (short i = 0; i < 4; ++i) simdgroup_store(mc[4 * rd + i], sc + 8 * i, 64, 0, false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
    for (short j = 0; j < 8; ++j) {
      const short e = short(tiitg) * 8 + j;
      const short ts = e / 64, n = e % 64;
      const short t = 16 * (ts / 8) + 8 * rd + ts % 8;
      if (t < nr1) D[(ulong)(r1 + t) * (ulong)N + (ulong)(r0 + n)] = bfloat(sof[e] * gs);
    }
  }
}

// Measured and rejected, 2026-11-10 (2048-token prefill, tokens/s; this kernel 213):
//   64 x 64 tile, 256 threads (half decode, half load)         181..187
//   64 x 64 tile, 128 threads, 16 accumulators per simdgroup   186
//   64 columns of K per step (half the barriers)               172
//   two block sets used alternately (one barrier per step)     202 against 208
//   a separate 8 KB staging array for the output (14 KB)       201 against 208
//   a 16 KB staging array (22 KB of threadgroup memory)        137
//   float blocks instead of half                               179 against 201
// With 3 of the 4 multiply rounds removed the kernel still takes 56% of its
// time: decode, block stores and weight traffic weigh about as much as the
// products, and nothing above that shares them across more rows came out
// ahead.

#define Q38_GEMM_KERNEL(NAME, FMT, XH)                                                            \
  [[host_name(NAME)]] kernel void q38_gemm_##FMT##_##XH(                                          \
      device bfloat* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],                       \
      device const bfloat* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],                 \
      device const float* GS [[buffer(4)]], const constant int& N [[buffer(5)]],                  \
      const constant int& K [[buffer(6)]], const constant int& M [[buffer(7)]],                   \
      device const uchar* Wb [[buffer(8)]], uint3 tg [[threadgroup_position_in_grid]],            \
      ushort tiitg [[thread_index_in_threadgroup]],                                               \
      ushort sgitg [[simdgroup_index_in_threadgroup]]) {                                          \
    threadgroup q38_blk sa[2048];                                    \
    threadgroup q38_blk sb[1024];                                    \
    q38_gemm_body<FMT, XH>(D, Wq, X, WS, GS, N, K, M, Wb, sa, sb, tg, tiitg, sgitg);              \
  }

Q38_GEMM_KERNEL("q38_gemm_nvfp4", 1, false)
Q38_GEMM_KERNEL("q38_gemm_nvfp4_xh", 1, true)
Q38_GEMM_KERNEL("q38_gemm_u4", 3, false)
Q38_GEMM_KERNEL("q38_gemm_u4_xh", 3, true)
Q38_GEMM_KERNEL("q38_gemm_u5", 4, false)
Q38_GEMM_KERNEL("q38_gemm_u5_xh", 4, true)
