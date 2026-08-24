#include <metal_stdlib>
#include "tk.metal"

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Tiled MoE GEMM for prefill widths (llama.cpp kernel_mul_mm_id port).
//
// Two phases:
//   qc_moe_mm_map0_<topk>  — one threadgroup, one thread per expert: builds a
//     dense per-expert list of slot ids (ids[e][0..tpe[e]) = token*topk + slot)
//     from the (tokens, topk) router output. Entries past tpe[e] are stale
//     garbage and are never read (the GEMM early-exits on tpe[e]).
//   qc_moe_mm_id_iq2_xxs_w64 — 64x32 output tiles, 128 threads / 4
//     simdgroups, 8x8 simdgroup MMA over a dequantized A tile (64 rows x
//     32 k, iq2_xxs) and two gathered 32-slot B halves (fp16 activations)
//     per 64-slot queue entry. fp32 accumulate, half store, scatter rows
//     through the ids list so the output lands in the vec kernels' flat
//     (token, slot) row order. qc_moe_mm_id_q2_K[_soa] is the down twin.
//
// The GEMV path stays the decode answer; this kernel only wins once enough
// tokens share each expert's weight tile (host threshold: tokens >= 32).
// ---------------------------------------------------------------------------

// Raw 66-byte iq2_xxs superblock (256 weights): fp16 scale + 32 packed u16.
struct block_iq2_xxs_t {
  half d;
  ushort qs[32];
};
static_assert(sizeof(block_iq2_xxs_t) == 66, "iq2_xxs block must pack to 66B");

// --------------------------------- map0 ------------------------------------

template <short NE20>
kernel void qc_moe_mm_map0(
    device const int *topk_ids [[buffer(0)]],  // (tokens, NE20) i32
    device uint *tpe           [[buffer(1)]],  // (E) rows per expert
    device int *ids            [[buffer(2)]],  // (E, tokens) dense slot lists
    const constant int &tokens [[buffer(3)]],
    device uint *work          [[buffer(4)]],  // (cap) expert<<16 | slot
    device uint *wcount        [[buffer(5)]],  // (1) live work items
    device uint *work64        [[buffer(6)]],  // (cap64) 64-slot tiles
    device uint *wcount64      [[buffer(7)]],  // (1)
    ushort tpitg [[thread_position_in_threadgroup]],
    ushort ntg   [[threads_per_threadgroup]]) {
  threadgroup ushort sids[256 * NE20];  // E <= 256 host-checked (DSV4: 256)
  threadgroup uint tcnt[256];
  threadgroup uint toff[256];

  const short ide = tpitg;  // this thread's expert id
  uint n_all = 0;
  device int *ids_e = ids + (int)ide * tokens;

  for (int i21 = 0; i21 < tokens; i21 += ntg) {
    // Cooperative stage of ntg tokens' routes. Negative ids cast to large
    // ushorts and never match an expert (< 256), so they are silently
    // dropped.
    if (i21 + tpitg < tokens) {
      device const int *src = topk_ids + (i21 + tpitg) * NE20;
      threadgroup ushort *s = sids + tpitg * NE20;
      #pragma clang loop unroll(full)
      for (short i20 = 0; i20 < NE20; i20++) {
        s[i20] = (ushort)src[i20];
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short t = 0; t < ntg; t++) {
      if (i21 + t >= tokens) {
        break;
      }
      threadgroup const ushort *s = sids + t * NE20;
      // Branchless membership scan; duplicate expert ids within one token's
      // top-k would corrupt sel, but the DSV4 router emits unique ids.
      short sel = 0;
      #pragma clang loop unroll(full)
      for (short i20 = 0; i20 < NE20; i20++) {
        sel += (s[i20] == (ushort)ide) * (i20 + 1);
      }
      ids_e[n_all] = (i21 + t) * NE20 + sel - 1;
      n_all += sel > 0;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  tpe[ide] = n_all;

  // Work queue (ds4 map0 shape): serial-prefix the per-expert 32-slot tile
  // counts and emit (expert << 16 | first_slot) items so the tile GEMMs
  // dispatch exactly the non-empty tiles. tokens < 65536 host-checked for
  // the 16-bit slot pack.
  tcnt[ide] = (n_all + 31) / 32;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ide == 0) {
    uint acc = 0;
    for (short e2 = 0; e2 < (short)ntg; ++e2) {
      toff[e2] = acc;
      acc += tcnt[e2];
    }
    wcount[0] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint base = toff[ide];
  for (uint t = 0; t < tcnt[ide]; ++t) {
    work[base + t] = ((uint)ide << 16) | (uint)(t * 32);
  }

  // Second queue at 64-slot granularity for the dual-half tile kernels
  // (one A dequant amortized over two 32-slot halves). Same prefix scheme;
  // the barrier before the tcnt overwrite orders it after every thread's
  // 32-wide emit loop above.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  tcnt[ide] = (n_all + 63) / 64;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ide == 0) {
    uint acc = 0;
    for (short e2 = 0; e2 < (short)ntg; ++e2) {
      toff[e2] = acc;
      acc += tcnt[e2];
    }
    wcount64[0] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint b64 = toff[ide];
  for (uint t = 0; t < tcnt[ide]; ++t) {
    work64[b64 + t] = ((uint)ide << 16) | (uint)(t * 64);
  }
}

typedef decltype(qc_moe_mm_map0<2>) qc_moe_mm_map0_t;
template [[host_name("qc_moe_mm_map0_2")]] kernel qc_moe_mm_map0_t
    qc_moe_mm_map0<2>;
template [[host_name("qc_moe_mm_map0_4")]] kernel qc_moe_mm_map0_t
    qc_moe_mm_map0<4>;
template [[host_name("qc_moe_mm_map0_6")]] kernel qc_moe_mm_map0_t
    qc_moe_mm_map0<6>;
template [[host_name("qc_moe_mm_map0_8")]] kernel qc_moe_mm_map0_t
    qc_moe_mm_map0<8>;

// ------------------------------ tiled GEMM ---------------------------------

// 16-value iq2_xxs dequant granule (llama.cpp dequantize_iq2_xxs with the
// grid/sign tables staged to threadgroup memory). il in [0, 16); float math,
// half store.
static inline void mm_dequant16_iq2_xxs(device const block_iq2_xxs_t *xb,
                                        short il,
                                        threadgroup const ulong *svalues,
                                        threadgroup const uchar *ssigns,
                                        thread half4x4 &reg) {
  const float d = (float)xb->d;
  const short ib32 = il / 2;
  il = il % 2;
  device const ushort *q2 = xb->qs + 4 * ib32;
  const uint aux32_g = (uint)q2[0] | ((uint)q2[1] << 16);
  const uint aux32_s = (uint)q2[2] | ((uint)q2[3] << 16);
  thread const uchar *aux8 = (thread const uchar *)&aux32_g;
  const float dl = d * (0.5f + (float)(aux32_s >> 28)) * 0.25f;

  threadgroup const uchar *grid =
      (threadgroup const uchar *)(svalues + aux8[2 * il + 0]);
  uchar signs = ssigns[(aux32_s >> (14 * il)) & 127];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    reg[i / 4][i % 4] =
        (half)(dl * grid[i] * ((signs & kmask_iq2xs[i]) ? -1.f : 1.f));
  }
  grid = (threadgroup const uchar *)(svalues + aux8[2 * il + 1]);
  signs = ssigns[(aux32_s >> (14 * il + 7)) & 127];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; ++i) {
    reg[2 + i / 4][i % 4] =
        (half)(dl * grid[i] * ((signs & kmask_iq2xs[i]) ? -1.f : 1.f));
  }
}

// --------------------- dual-half (64-slot) iq2_xxs tile ---------------------
// Consumes the 64-slot map0 queue: one A (weight) dequant pass feeds two
// 32-slot B halves, halving the per-tile dequant+stage cost when an
// expert's slots fit one 64-tile. Each half runs the 64x32-tile MMA with
// per-half operand values and accumulation order, so per-slot outputs are
// independent of how slots split across halves. Empty second halves
// (expert count <= 32) skip their B stage and MMAs uniformly.
kernel void qc_moe_mm_id_iq2_xxs_w64(
    device half *D              [[buffer(0)]],  // (tokens*topk, N)
    device const uchar *Wq      [[buffer(1)]],  // (E, N, K/256*66) AoS
    device const half *X        [[buffer(2)]],  // (tokens, K)
    device const uint *tpe      [[buffer(3)]],  // (E)
    device const int *ids       [[buffer(4)]],  // (E, tokens)
    const constant int &N       [[buffer(5)]],
    const constant int &K       [[buffer(6)]],
    const constant int &tokens  [[buffer(7)]],
    const constant int &topk    [[buffer(8)]],
    device const uint *work     [[buffer(9)]],   // map0 64-slot tile queue
    device const uint *wcount   [[buffer(10)]],  // (1)
    uint3 tgid   [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  // One backing allocation: sa | sb during the k loop, re-carved as the fp32
  // output staging (sc) in the epilogue — the post-loop barrier orders the
  // reuse. The small footprint lets multiple threadgroups co-reside per
  // core so the per-k-step barriers overlap across threadgroups instead of
  // stalling the core.
  threadgroup half2x4 shmv[512];  // 8 KB, 8B-aligned for the B stores
  threadgroup half *sa = (threadgroup half *)shmv;
  threadgroup half *sb = (threadgroup half *)shmv + 64 * 32;
  threadgroup float *sc = (threadgroup float *)shmv;
  threadgroup ulong svalues[256];
  threadgroup uchar ssigns[128];

  if (tgid.x >= wcount[0]) {
    return;
  }
  const uint witem = work[tgid.x];
  const int im = (int)(witem >> 16);        // expert
  const int r1 = (int)(witem & 0xFFFFu);    // first slot (64-aligned)

  for (int i = tiitg; i < 256; i += 128) svalues[i] = iq2xxs_grid[i];
  for (int i = tiitg; i < 128; i += 128) ssigns[i] = ksigns_iq2xs[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr int NR0 = 64;
  constexpr int NR1 = 32;  // slots per half
  constexpr int NK = 32;
  constexpr int NL0 = 2;
  constexpr int NL1 = 4;
  constexpr short nl = 16;

  const int r0 = tgid.y * NR0;

  const int neh1 = (int)tpe[im];

  const short nr0 = (N - r0 < NR0) ? (short)(N - r0) : (short)NR0;
  const short nr1a = (neh1 - r1 < NR1) ? (short)(neh1 - r1) : (short)NR1;
  const int rem_b = neh1 - r1 - NR1;
  const short nr1b =
      rem_b <= 0 ? (short)0 : (rem_b < NR1 ? (short)rem_b : (short)NR1);
  const bool has_b = nr1b > 0;

  const short lr0 = ((short)(tiitg / NL0) < nr0) ? (short)(tiitg / NL0)
                                                 : (short)(nr0 - 1);
  const short lr1a = ((short)(tiitg / NL1) < nr1a) ? (short)(tiitg / NL1)
                                                   : (short)(nr1a - 1);
  const short lr1b =
      has_b ? (((short)(tiitg / NL1) < nr1b) ? (short)(tiitg / NL1)
                                             : (short)(nr1b - 1))
            : (short)0;

  const short il0 = tiitg % NL0;
  short il = il0;

  const int bpr = K / 256;
  device const block_iq2_xxs_t *xq =
      (device const block_iq2_xxs_t *)(Wq + (long)im * N * bpr * 66 +
                                       (long)(r0 + lr0) * bpr * 66);

  const short iy = 8 * (tiitg % NL1);
  const int ida = ids[im * tokens + r1 + lr1a];
  device const half *ya = X + (long)(ida / topk) * K + iy;
  device const half *yb = ya;
  if (has_b) {
    const int idb = ids[im * tokens + r1 + NR1 + lr1b];
    yb = X + (long)(idb / topk) * K + iy;
  }

  simdgroup_half8x8 ma[4];
  simdgroup_half8x8 mb[2];
  simdgroup_float8x8 mc0[8];
  simdgroup_float8x8 mc1[8];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    mc0[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    mc1[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  }

  // Dead-block cull: this simdgroup owns slots [sg_s0, sg_s0+16) of each
  // half (two 8-slot mb blocks). Blocks entirely past the live count carry
  // only padding lanes — skip their loads and MMAs (uniform per simdgroup;
  // the ik loop has no threadgroup barriers). Tail tiles get
  // proportionally cheaper; live outputs are untouched, dead accumulators
  // stay zero and are never scattered.
  const short sg_s0 = (short)((sgitg / 2) * 16);
  const short live_a =
      metal::clamp((short)(nr1a - sg_s0), (short)0, (short)16);
  const short live_b =
      metal::clamp((short)(nr1b - sg_s0), (short)0, (short)16);
  const short nmb_a = (short)((live_a + 7) / 8);  // 0..2 live mb blocks
  const short nmb_b = (short)((live_b + 7) / 8);

  for (int loop_k = 0; loop_k < K; loop_k += NK) {
    half4x4 temp_a;
    mm_dequant16_iq2_xxs(xq, il, svalues, ssigns, temp_a);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (short i = 0; i < 16; i++) {
      const short sx = 2 * il0 + i / 8;
      const short sy = (tiitg / NL0) / 8;
      const short lx = (tiitg / NL0) % 8;
      const short ly = i % 8;
      const short ib = 8 * sx + sy;
      *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
    }

    {
      const short sx = tiitg % NL1;
      const short sy = (tiitg / NL1) / 8;
      const short ly = (tiitg / NL1) % 8;
      const short ib = 4 * sx + sy;
      *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
          *(device const half2x4 *)ya;
      if (has_b) {
        *(threadgroup half2x4 *)(sb + 1024 + 64 * ib + 8 * ly) =
            *(device const half2x4 *)yb;
      }
    }

    il = (il + 2 < nl) ? il + 2 : il % 2;
    xq = (il < 2) ? xq + 1 : xq;
    ya += NK;
    yb += NK;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);

    #pragma clang loop unroll(full)
    for (short ik = 0; ik < NK / 8; ik++) {
      simdgroup_barrier(mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 4; i++) {
        simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
      }
      // Keep the fully-live path (nmb == 2, the common case) fully
      // unrolled; tail simdgroups take the narrow branch.
      if (nmb_a == 2) {
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; i++) {
          simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) {
          simdgroup_multiply_accumulate(mc0[i], mb[i / 4], ma[i % 4], mc0[i]);
        }
      } else if (nmb_a == 1) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(mb[0], lsmb, 8, 0, false);
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 4; i++) {
          simdgroup_multiply_accumulate(mc0[i], mb[0], ma[i], mc0[i]);
        }
      }
      if (nmb_b == 2) {
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 2; i++) {
          simdgroup_load(mb[i], lsmb + 1024 + 64 * i, 8, 0, false);
        }
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 8; i++) {
          simdgroup_multiply_accumulate(mc1[i], mb[i / 4], ma[i % 4], mc1[i]);
        }
      } else if (nmb_b == 1) {
        simdgroup_barrier(mem_flags::mem_none);
        simdgroup_load(mb[0], lsmb + 1024, 8, 0, false);
        simdgroup_barrier(mem_flags::mem_none);
        #pragma clang loop unroll(full)
        for (short i = 0; i < 4; i++) {
          simdgroup_multiply_accumulate(mc1[i], mb[0], ma[i], mc1[i]);
        }
      }
      lsma += 8 * 64;
      lsmb += 4 * 64;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *temp_str =
      sc + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    simdgroup_store(mc0[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0,
                    false);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short j = sgitg; j < nr1a; j += 4) {
    const int idj = ids[im * tokens + r1 + j];
    device half *Dr = D + (long)idj * N + r0;
    threadgroup const float *C = sc + j * NR0;
    for (short i = tiisg; i < nr0; i += 32) {
      Dr[i] = (half)C[i];
    }
  }

  if (has_b) {
    // sc is reused for the second half; the barrier orders the restage
    // after every simdgroup's half-a scatter reads.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (short i = 0; i < 8; i++) {
      simdgroup_store(mc1[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0,
                      0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = sgitg; j < nr1b; j += 4) {
      const int idj = ids[im * tokens + r1 + NR1 + j];
      device half *Dr = D + (long)idj * N + r0;
      threadgroup const float *C = sc + j * NR0;
      for (short i = tiisg; i < nr0; i += 32) {
        Dr[i] = (half)C[i];
      }
    }
  }
}

// ------------------------- q2_K down tile GEMM -----------------------------

// 16-value q2_K dequant granule (llama.cpp dequantize_q2_K with the block
// split into caller-provided scale/qs pointers so AoS and SoA share it).
static inline void mm_dequant16_q2_K(device const uchar *sc_p,
                                     device const uchar *qs_p, float d,
                                     float dmin, short il,
                                     thread half4x4 &reg) {
  const uchar sc8 = sc_p[il];
  // 4-byte-aligned in both layouts (AoS: 84*blk + 16; SoA: 64*blk), which is
  // enough for uchar4, so the 16 per-value byte loads collapse to four uchar4
  // loads. Same bytes, same float math and order — bit-identical outputs.
  // Do not widen past uchar4: the AoS base is not 16-byte aligned.
  device const uchar4 *q4 =
      (device const uchar4 *)(qs_p + 32 * (il / 8) + 16 * (il & 1));
  const uchar4 qv0 = q4[0], qv1 = q4[1], qv2 = q4[2], qv3 = q4[3];
  thread const uchar4 qv[4] = {qv0, qv1, qv2, qv3};
  const short il2 = (il / 2) % 4;
  const float coef =
      il2 > 1 ? (il2 > 2 ? 1.f / 64.f : 1.f / 16.f) : (il2 > 0 ? 1.f / 4.f : 1.f);
  const uchar mask = il2 > 1 ? (il2 > 2 ? 192 : 48) : (il2 > 0 ? 12 : 3);
  const float dl = d * (sc8 & 0xF) * coef;
  const float ml = dmin * (sc8 >> 4);
  #pragma clang loop unroll(full)
  for (short i = 0; i < 16; ++i) {
    reg[i / 4][i % 4] = (half)(dl * (qv[i / 4][i % 4] & mask) - ml);
  }
}

// Down-projection twin of the iq2_xxs tile GEMM: B rows are the per-slot
// activations (row = id = token*topk + slot; llama.cpp's ne11 == ne20 down
// semantics), so X is (tokens*topk, K). q2_K has no grid tables, so no
// threadgroup staging (14336 B total). SOA selects the load-time-repacked
// plane layout (per expert: qs plane @0, 64 B/blk | scales plane @N*nb*64,
// 16 B/blk | d,dmin plane @N*nb*80, 4 B/blk); AoS reads raw 84-byte blocks
// {scales[16], qs[64], d, dmin}.
template <bool SOA>
kernel void qc_moe_mm_id_q2_K_impl(
    device half *D              [[buffer(0)]],  // (tokens*topk, N)
    device const uchar *Wq      [[buffer(1)]],  // (E, N, K/256*84) AoS or SoA
    device const half *X        [[buffer(2)]],  // (tokens*topk, K) per-slot
    device const uint *tpe      [[buffer(3)]],  // (E)
    device const int *ids       [[buffer(4)]],  // (E, tokens)
    const constant int &N       [[buffer(5)]],
    const constant int &K       [[buffer(6)]],
    const constant int &tokens  [[buffer(7)]],
    const constant int &topk    [[buffer(8)]],
    device const uint *work     [[buffer(9)]],   // map0 tile queue
    device const uint *wcount   [[buffer(10)]],  // (1)
    uint3 tgid   [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]) {
  threadgroup half sa[64 * 32];
  threadgroup half2x4 sbv[128];
  threadgroup half *sb = (threadgroup half *)sbv;
  threadgroup float sc[32 * 64];

  if (tgid.x >= wcount[0]) {
    return;
  }
  const uint witem = work[tgid.x];
  const int im = (int)(witem >> 16);
  const int r1 = (int)(witem & 0xFFFFu);

  constexpr int NR0 = 64;
  constexpr int NR1 = 32;
  constexpr int NK = 32;
  constexpr int NL0 = 2;
  constexpr int NL1 = 4;
  constexpr short nl = 16;

  const int r0 = tgid.y * NR0;

  const int neh1 = (int)tpe[im];

  const short nr0 = (N - r0 < NR0) ? (short)(N - r0) : (short)NR0;
  const short nr1 = (neh1 - r1 < NR1) ? (short)(neh1 - r1) : (short)NR1;

  const short lr0 = ((short)(tiitg / NL0) < nr0) ? (short)(tiitg / NL0)
                                                 : (short)(nr0 - 1);
  const short lr1 = ((short)(tiitg / NL1) < nr1) ? (short)(tiitg / NL1)
                                                 : (short)(nr1 - 1);

  const short il0 = tiitg % NL0;
  short il = il0;

  const int bpr = K / 256;
  // Block cursor instead of a struct pointer: AoS and SoA both address
  // blocks by linear index (row*bpr + advancing k-block).
  long blk = (long)(r0 + lr0) * bpr;
  device const uchar *w_e;   // AoS: 84-byte blocks
  device const uchar *qs_pl, *sc_pl, *dm_pl;  // SoA planes
  if (SOA) {
    device const uchar *e_base = Wq + (long)im * N * bpr * 84;
    qs_pl = e_base;
    sc_pl = e_base + (long)N * bpr * 64;
    dm_pl = e_base + (long)N * bpr * 80;
    w_e = e_base;
  } else {
    w_e = Wq + (long)im * N * bpr * 84;
    qs_pl = sc_pl = dm_pl = w_e;
  }

  const int id = ids[im * tokens + r1 + lr1];
  const short iy = 8 * (tiitg % NL1);
  device const half *y = X + (long)id * K + iy;

  simdgroup_half8x8 ma[4];
  simdgroup_half8x8 mb[2];
  simdgroup_float8x8 mc[8];
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    mc[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
  }

  // Per-block dequant scalars cached across the block's 8 k-steps (il < 2
  // exactly on a block's first step; uniform across the threadgroup).
  // No per-ik dead-block cull here: rejected, the branch in the unrolled
  // MMA loop costs more than it saves at 32-wide (optimization_status
  // 2026-08-14 v8a).
  float d_r = 0.f, dmin_r = 0.f;
  device const uchar *scp_r = w_e;
  device const uchar *qsp_r = w_e;

  for (int loop_k = 0; loop_k < K; loop_k += NK) {
    half4x4 temp_a;
    if (il < 2) {
      if (SOA) {
        device const half *dm = (device const half *)(dm_pl + blk * 4);
        d_r = (float)dm[0];
        dmin_r = (float)dm[1];
        scp_r = sc_pl + blk * 16;
        qsp_r = qs_pl + blk * 64;
      } else {
        device const uchar *b = w_e + blk * 84;
        device const half *dm = (device const half *)(b + 80);
        d_r = (float)dm[0];
        dmin_r = (float)dm[1];
        scp_r = b;
        qsp_r = b + 16;
      }
    }
    mm_dequant16_q2_K(scp_r, qsp_r, d_r, dmin_r, il, temp_a);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    #pragma clang loop unroll(full)
    for (short i = 0; i < 16; i++) {
      const short sx = 2 * il0 + i / 8;
      const short sy = (tiitg / NL0) / 8;
      const short lx = (tiitg / NL0) % 8;
      const short ly = i % 8;
      const short ib = 8 * sx + sy;
      *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
    }

    {
      const short sx = tiitg % NL1;
      const short sy = (tiitg / NL1) / 8;
      const short ly = (tiitg / NL1) % 8;
      const short ib = 4 * sx + sy;
      *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
          *(device const half2x4 *)y;
    }

    il = (il + 2 < nl) ? il + 2 : il % 2;
    blk += (il < 2) ? 1 : 0;
    y += NK;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup const half *lsma = sa + 4 * 64 * (sgitg % 2);
    threadgroup const half *lsmb = sb + 2 * 64 * (sgitg / 2);

    #pragma clang loop unroll(full)
    for (short ik = 0; ik < NK / 8; ik++) {
      simdgroup_barrier(mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 4; i++) {
        simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
      }
      simdgroup_barrier(mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 2; i++) {
        simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
      }
      simdgroup_barrier(mem_flags::mem_none);
      #pragma clang loop unroll(full)
      for (short i = 0; i < 8; i++) {
        simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
      }
      lsma += 8 * 64;
      lsmb += 4 * 64;
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup float *temp_str = sc + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0;
  #pragma clang loop unroll(full)
  for (short i = 0; i < 8; i++) {
    simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0 * (i / 4), NR0, 0,
                    false);
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (short j = sgitg; j < nr1; j += 4) {
    const int idj = ids[im * tokens + r1 + j];
    device half *Dr = D + (long)idj * N + r0;
    threadgroup const float *C = sc + j * NR0;
    for (short i = tiisg; i < nr0; i += 32) {
      Dr[i] = (half)C[i];
    }
  }
}

typedef decltype(qc_moe_mm_id_q2_K_impl<false>) qc_moe_mm_id_q2_K_t;
template [[host_name("qc_moe_mm_id_q2_K")]] kernel qc_moe_mm_id_q2_K_t
    qc_moe_mm_id_q2_K_impl<false>;
template [[host_name("qc_moe_mm_id_q2_K_soa")]] kernel qc_moe_mm_id_q2_K_t
    qc_moe_mm_id_q2_K_impl<true>;

// No q2_K w64 twin (regressed on the SoA layout) and no fused pair+SwiGLU
// tile (measured negative twice) — see optimization_status 2026-08-13/14
// before re-trying either.
