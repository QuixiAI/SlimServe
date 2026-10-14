#include <metal_stdlib>
#include "tk.metal"

namespace mittens {

// Planar-layout quantized GEMM for the compressed-tensors serving path
// (prefill / batch-verify M > 8): D (M, N) = X (M, K) @ dequant(W)^T with
// the weights read DIRECTLY from the checkpoint's planar buffers - no
// bf16 materialization at load and no per-call dequant copy (the two
// paths these kernels replace; the operator directive is that weights
// stay in native quantized form always).
//
// Geometry is the classic llama.cpp kernel_mul_mm shape, proven in-repo
// by moe_mm_id (llama.cpp mul_mm_id port) and the qgemm family: one
// threadgroup computes a 64-row (N) x 32-column (M) output tile, 128
// threads in 4 simdgroups, K walked in 32-wide steps. Per step each
// thread decodes 16 weights into a transposed 8x8-blocked threadgroup
// tile (sa) and stages 8 activation elements (sb, converted to half -
// half tiles measured +25-30% over bfloat staging in the UPDATE 26
// campaign); each simdgroup then runs 8 simdgroup_float8x8 MMAs per
// 8-deep K slice for its 32x16 slab of the tile.
//
// The weight decode bodies are the measured winners from the qgemv
// campaign (qgemv.metal:1452-1479 is the rejection list - the llama.cpp
// constant-table LUT idiom serializes on this GPU at ~135 GB/s vs
// 452-529 for these select-free bit constructs):
// - NVFP4: four half2 bit-pattern constructs decode all 8 nibbles of a
//   uint; the group's e4m3 scale byte folds with the 2^22 rebias
//   (2^14 E2M1 x 2^8 E4M3, exact) BEFORE staging, because simdgroup MMA
//   accumulates across whole K-steps and cannot fold per-16 group scales
//   after the dot. Scaled magnitudes are <= 6 x 448, comfortably half.
//   The per-tensor global scale folds once in the fp32 epilogue.
// - FP8ch: two half2 constructs per uint decode 4 e4m3 bytes; patterns
//   stage unscaled (<= 448 x 2^-8); the 2^8 rebias and the per-CHANNEL
//   fp32 scale fold per output row in the epilogue.
//
// Bounds: N and M tails are clamp-read / store-guarded (llama.cpp
// pattern), so any N, M work. K % 32 == 0 is a host guard.

template<typename T>
kernel void qgemm_nvfp4_planar(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   const uchar* Wq [[buffer(1)]],   // (N, K/2) packed e2m1
    device   const T*     X  [[buffer(2)]],   // (M, K) activations, row-major
    device   const uchar* WS [[buffer(3)]],   // (N, K/16) e4m3 group scales
    device   const float* GS [[buffer(4)]],   // (1,) global multiplier
    const constant int &N [[buffer(5)]],
    const constant int &K [[buffer(6)]],
    const constant int &M [[buffer(7)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  // One 8 KB backing allocation: sa | sb during the K loop, re-carved as
  // the fp32 output staging (sc) in the epilogue (moe_mm_id pattern).
  threadgroup metal::half2x4 shmv[512];
  threadgroup half *sa = (threadgroup half *)shmv;              // 64 x 32
  threadgroup half *sb = (threadgroup half *)shmv + 64 * 32;    // 32 x 32
  threadgroup float *sc = (threadgroup float *)shmv;            // 32 x 64

  constexpr short NR0 = 64;   // weight rows (N) per threadgroup
  constexpr short NR1 = 32;   // activation columns (M) per threadgroup
  constexpr short NK = 32;    // K step

  const int r0 = int(tgid.x) * NR0;
  const int c0 = int(tgid.y) * NR1;
  const short nlive = (N - r0 < NR0) ? short(N - r0) : NR0;
  const short mlive = (M - c0 < NR1) ? short(M - c0) : NR1;

  // Weight staging geometry: thread -> (row lr, K-half il0); 16 weights
  // per thread per step = exactly one NVFP4 group (one uint2 + one scale
  // byte). Rows clamp-read for the N tail (dead rows never stored).
  const short lr = tiitg / 2;
  const short il0 = tiitg % 2;
  const int wrow = (r0 + lr < N) ? (r0 + lr) : (N - 1);
  const int groups = K / 16;
  device const uchar* w0 = Wq + (ulong)wrow * (ulong)(K / 2) + 8 * il0;
  device const uchar* s0 = WS + (ulong)wrow * (ulong)groups + il0;

  // Activation staging geometry: thread -> (column lc, 8-elem K span iy).
  // Columns clamp-read for the M tail.
  const short lc = tiitg / 4;
  const short iy = 8 * (tiitg % 4);
  const int xcol = (c0 + lc < M) ? (c0 + lc) : (M - 1);
  device const T* xa = X + (ulong)xcol * (ulong)K + iy;

  metal::simdgroup_half8x8 ma[4];
  metal::simdgroup_half8x8 mb[2];
  metal::simdgroup_float8x8 mc[8];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    mc[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  }

  for (int loop_k = 0; loop_k < K; loop_k += NK) {
    // --- decode 16 NVFP4 weights (one group), scale folded pre-MMA ---
    const uint2 p = *(device const uint2*)(w0 + loop_k / 2);
    const uint sbyte = s0[loop_k / 16];
    const half sh = as_type<half>(ushort(((sbyte & 0x7F) << 7) |
                                         ((sbyte & 0x80) << 8)));
    const float gscale = 4194304.0f * float(sh);   // 2^22 rebias, exact
    half w[16];
    #pragma clang loop unroll(full)
    for (short u = 0; u < 2; ++u) {
      const uint v = (u == 0) ? p.x : p.y;
      const short col = 8 * u;
      const half2 le = as_type<half2>(((v << 9) & 0x0E000E00u) |
                                      ((v << 12) & 0x80008000u));
      const half2 lo = as_type<half2>(((v << 1) & 0x0E000E00u) |
                                      ((v << 4) & 0x80008000u));
      const half2 he = as_type<half2>(((v << 5) & 0x0E000E00u) |
                                      ((v << 8) & 0x80008000u));
      const half2 ho = as_type<half2>(((v >> 3) & 0x0E000E00u) |
                                      (v & 0x80008000u));
      w[col + 0] = half(float(le.x) * gscale);
      w[col + 1] = half(float(he.x) * gscale);
      w[col + 2] = half(float(lo.x) * gscale);
      w[col + 3] = half(float(ho.x) * gscale);
      w[col + 4] = half(float(le.y) * gscale);
      w[col + 5] = half(float(he.y) * gscale);
      w[col + 6] = half(float(lo.y) * gscale);
      w[col + 7] = half(float(ho.y) * gscale);
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    // Scatter into transposed 8x8 blocks (moe_mm_id/llama.cpp layout).
    #pragma clang loop unroll(full)
    for (short i = 0; i < 16; i++) {
      const short sx = 2 * il0 + i / 8;
      const short sy = lr / 8;
      const short lx = lr % 8;
      const short ly = i % 8;
      const short ib = 8 * sx + sy;
      *(sa + 64 * ib + 8 * ly + lx) = w[i];
    }

    // Stage 8 activation elements, converted to half.
    {
      const short sx = tiitg % 4;
      const short sy = lc / 8;
      const short ly = lc % 8;
      const short ib = 4 * sx + sy;
      device const metal::vec<T, 4>* xp =
          (device const metal::vec<T, 4>*)(xa + loop_k);
      const metal::vec<T, 4> x0 = xp[0];
      const metal::vec<T, 4> x1 = xp[1];
      threadgroup half* dst = sb + 64 * ib + 8 * ly;
      dst[0] = half(x0[0]); dst[1] = half(x0[1]);
      dst[2] = half(x0[2]); dst[3] = half(x0[3]);
      dst[4] = half(x1[0]); dst[5] = half(x1[1]);
      dst[6] = half(x1[2]); dst[7] = half(x1[3]);
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);
    #pragma clang loop unroll(full)
    for (short ik = 0; ik < NK / 8; ik++) {
      metal::simdgroup_barrier(metal::mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 4; i++) {
        metal::simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
      }
      metal::simdgroup_barrier(metal::mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 2; i++) {
        metal::simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
      }
      metal::simdgroup_barrier(metal::mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; i++) {
        metal::simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4],
                                             mc[i]);
      }
      lsma += 8 * 64;
      lsmb += 4 * 64;
    }
  }

  // Epilogue: stage fp32 through shmem (reused), then guarded writeback
  // with the global scale folded once.
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  threadgroup float *temp_str =
      sc + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    metal::simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                           NR0, 0, false);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  const float gs = GS[0];
  for (short j = sgitg; j < mlive; j += 4) {
    device T *Dr = D + (ulong)(c0 + j) * (ulong)N + r0;
    threadgroup const float *src = sc + j * NR0;
    for (short i = tiisg; i < nlive; i += 32) {
      Dr[i] = T(src[i] * gs);
    }
  }
}

// FP8 per-channel twin: raw e4m3 bytes (N, K), fp32 per-channel scales
// (N,). Patterns stage unscaled; the 2^8 rebias and the channel scale
// fold per output row in the epilogue (matches qgemv_fp8ch numerics).
template<typename T>
kernel void qgemm_fp8ch(
    device   T*     D  [[buffer(0)]],   // (M, N) output, row-major
    device   const uchar* Wq [[buffer(1)]],   // (N, K) e4m3 bytes
    device   const T*     X  [[buffer(2)]],   // (M, K) activations
    device   const float* WS [[buffer(3)]],   // (N,) per-channel scales
    const constant int &N [[buffer(4)]],
    const constant int &K [[buffer(5)]],
    const constant int &M [[buffer(6)]],
    uint3  tgid  [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  threadgroup metal::half2x4 shmv[512];
  threadgroup half *sa = (threadgroup half *)shmv;
  threadgroup half *sb = (threadgroup half *)shmv + 64 * 32;
  threadgroup float *sc = (threadgroup float *)shmv;

  constexpr short NR0 = 64;
  constexpr short NR1 = 32;
  constexpr short NK = 32;

  const int r0 = int(tgid.x) * NR0;
  const int c0 = int(tgid.y) * NR1;
  const short nlive = (N - r0 < NR0) ? short(N - r0) : NR0;
  const short mlive = (M - c0 < NR1) ? short(M - c0) : NR1;

  const short lr = tiitg / 2;
  const short il0 = tiitg % 2;
  const int wrow = (r0 + lr < N) ? (r0 + lr) : (N - 1);
  device const uchar* w0 = Wq + (ulong)wrow * (ulong)K + 16 * il0;

  const short lc = tiitg / 4;
  const short iy = 8 * (tiitg % 4);
  const int xcol = (c0 + lc < M) ? (c0 + lc) : (M - 1);
  device const T* xa = X + (ulong)xcol * (ulong)K + iy;

  metal::simdgroup_half8x8 ma[4];
  metal::simdgroup_half8x8 mb[2];
  metal::simdgroup_float8x8 mc[8];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    mc[i] = metal::make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  }

  for (int loop_k = 0; loop_k < K; loop_k += NK) {
    // --- decode 16 e4m3 bytes (one uint4), unscaled patterns ---
    const uint4 a = *(device const uint4*)(w0 + loop_k);
    half w[16];
    #pragma clang loop unroll(full)
    for (short u = 0; u < 4; ++u) {
      const uint av = a[u];
      const half2 ae = as_type<half2>(((av << 7) & 0x3F803F80u) |
                                      ((av << 8) & 0x80008000u));
      const half2 ao = as_type<half2>(((av >> 1) & 0x3F803F80u) |
                                      (av & 0x80008000u));
      w[4 * u + 0] = ae.x;
      w[4 * u + 1] = ao.x;
      w[4 * u + 2] = ae.y;
      w[4 * u + 3] = ao.y;
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (short i = 0; i < 16; i++) {
      const short sx = 2 * il0 + i / 8;
      const short sy = lr / 8;
      const short lx = lr % 8;
      const short ly = i % 8;
      const short ib = 8 * sx + sy;
      *(sa + 64 * ib + 8 * ly + lx) = w[i];
    }

    {
      const short sx = tiitg % 4;
      const short sy = lc / 8;
      const short ly = lc % 8;
      const short ib = 4 * sx + sy;
      device const metal::vec<T, 4>* xp =
          (device const metal::vec<T, 4>*)(xa + loop_k);
      const metal::vec<T, 4> x0 = xp[0];
      const metal::vec<T, 4> x1 = xp[1];
      threadgroup half* dst = sb + 64 * ib + 8 * ly;
      dst[0] = half(x0[0]); dst[1] = half(x0[1]);
      dst[2] = half(x0[2]); dst[3] = half(x0[3]);
      dst[4] = half(x1[0]); dst[5] = half(x1[1]);
      dst[6] = half(x1[2]); dst[7] = half(x1[3]);
    }

    threadgroup_barrier(metal::mem_flags::mem_threadgroup);

    threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);
    #pragma clang loop unroll(full)
    for (short ik = 0; ik < NK / 8; ik++) {
      metal::simdgroup_barrier(metal::mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 4; i++) {
        metal::simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
      }
      metal::simdgroup_barrier(metal::mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 2; i++) {
        metal::simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
      }
      metal::simdgroup_barrier(metal::mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; i++) {
        metal::simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4],
                                             mc[i]);
      }
      lsma += 8 * 64;
      lsmb += 4 * 64;
    }
  }

  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  threadgroup float *temp_str =
      sc + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    metal::simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4),
                           NR0, 0, false);
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);

  for (short j = sgitg; j < mlive; j += 4) {
    device T *Dr = D + (ulong)(c0 + j) * (ulong)N + r0;
    threadgroup const float *src = sc + j * NR0;
    for (short i = tiisg; i < nlive; i += 32) {
      // 256 = the folded 2^8 decode rebias (exact power of two).
      Dr[i] = T(src[i] * (256.0f * WS[r0 + i]));
    }
  }
}

#define instantiate_qgemm_nvfp4_planar(name, T)                               \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_nvfp4_planar<T>(                                                \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const uchar* WS [[buffer(3)]],   \
     device const float* GS [[buffer(4)]],                                    \
     const constant int &N [[buffer(5)]], const constant int &K [[buffer(6)]],\
     const constant int &M [[buffer(7)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort tiitg [[thread_index_in_threadgroup]],                            \
     ushort tiisg [[thread_index_in_simdgroup]],                              \
     ushort sgitg [[simdgroup_index_in_threadgroup]]);

instantiate_qgemm_nvfp4_planar("qgemm_nvfp4_planar", half);
instantiate_qgemm_nvfp4_planar("qgemm_nvfp4_planar_bfloat16", bf16);

#define instantiate_qgemm_fp8ch(name, T)                                      \
   template [[host_name(name)]] [[kernel]]                                    \
   void qgemm_fp8ch<T>(                                                       \
     device T* D [[buffer(0)]], device const uchar* Wq [[buffer(1)]],         \
     device const T* X [[buffer(2)]], device const float* WS [[buffer(3)]],   \
     const constant int &N [[buffer(4)]], const constant int &K [[buffer(5)]],\
     const constant int &M [[buffer(6)]],                                     \
     uint3 tgid [[threadgroup_position_in_grid]],                             \
     ushort tiitg [[thread_index_in_threadgroup]],                            \
     ushort tiisg [[thread_index_in_simdgroup]],                              \
     ushort sgitg [[simdgroup_index_in_threadgroup]]);

instantiate_qgemm_fp8ch("qgemm_fp8ch", half);
instantiate_qgemm_fp8ch("qgemm_fp8ch_bfloat16", bf16);

}  // namespace mittens
