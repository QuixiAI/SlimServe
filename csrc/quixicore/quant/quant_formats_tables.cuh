/**
 * @file
 * @brief Tranche 2 of the CUDA quant format layer: GGUF k-quants and i-quants
 * (table/codebook formats). Same contract as quant_formats.cuh:
 * `{block_k, block_bytes, dequant(base, col) -> float}`, decode in fp32,
 * bit-exact vs quant.py's dequantize_*. Layouts mirror ggml-common.h;
 * decoders mirror ThunderMittens dequant.metal (which mirrors ggml-metal).
 *
 * Formats: q2_K q3_K q4_K q5_K q6_K (256-superblock k-quants with packed
 * sub-scales) and iq4_nl iq4_xs iq2_xxs iq2_xs iq3_xxs iq1_s (non-linear
 * codebook / E8-lattice grid formats). Tables in quant_tables.cuh
 * (__constant__; indices are data-dependent so hot kernels may want smem).
 */
#pragma once
#include "quant_formats.cuh"
#include "quant_tables.cuh"

namespace tmq {

// ---- q2_K : { u8 scales[16]; u8 qs[64]; half d; half dmin; } - 84 bytes, 256/block.
//   16 sub-blocks of 16; scales byte = 4-bit dl-scale | 4-bit min. value = d*sc*q - dmin*m. ----
struct q2_K {
    static constexpr int block_k = 256, block_bytes = 84;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* scales = base;
        const uint8_t* qs = base + 16;
        const float d = half_at(base + 80), dmin = half_at(base + 82);
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int q = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        return d * float(scales[is] & 0xF) * float(q) - dmin * float(scales[is] >> 4);
    }
};

// ---- q3_K : { u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d; } - 110 bytes, 256/block.
//   low 2 bits in qs, high bit in hmask; 16 6-bit signed scales 4-way packed. value = d*(sc-32)*q3. ----
struct q3_K {
    static constexpr int block_k = 256, block_bytes = 110;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* hmask = base;
        const uint8_t* qs = base + 32;
        const uint8_t* sca = base + 96;
        const float d = half_at(base + 108);
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int low2 = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        const int hb = (hmask[sub * 16 + l] & (1 << (chunk * 4 + sidx))) ? 1 : 0;
        const int q3v = (low2 | (hb << 2)) - 4;
        const int w = is >> 2, b = is & 3;
        int s;
        if      (w == 0) s = (sca[b] & 0xF)            | ((sca[8 + b] & 3) << 4);
        else if (w == 1) s = (sca[4 + b] & 0xF)        | (((sca[8 + b] >> 2) & 3) << 4);
        else if (w == 2) s = ((sca[b] >> 4) & 0xF)     | (((sca[8 + b] >> 4) & 3) << 4);
        else             s = ((sca[4 + b] >> 4) & 0xF) | (((sca[8 + b] >> 6) & 3) << 4);
        return d * float(s - 32) * float(q3v);
    }
};

// ---- q4_K : { half d; half dmin; u8 scales[12]; u8 qs[128]; } - 144 bytes, 256/block.
//   8 sub-blocks of 32; 6-bit scale+min via get_scale_min_k4. value = (d*sc)*nib - (dmin*m). ----
struct q4_K {
    static constexpr int block_k = 256, block_bytes = 144;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base, 0), dmin = half_at(base, 1);
        const uint8_t* scales = base + 4;
        const uint8_t* qs     = base + 16;
        const int chunk = col / 64, pos = col % 64;
        int sub, nib;
        if (pos < 32) { sub = chunk * 2;     nib = qs[chunk * 32 + pos]        & 0x0F; }
        else          { sub = chunk * 2 + 1; nib = qs[chunk * 32 + (pos - 32)] >> 4;   }
        int sc, m;
        if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
        else {
            sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
            m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
        }
        return d * float(sc) * float(nib) - dmin * float(m);
    }
};

// ---- q5_K : { half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]; } - 176 bytes.
//   8 sub-blocks of 32; scale/min as q4_K; 5-bit q = nibble | (qh bit)<<4. ----
struct q5_K {
    static constexpr int block_k = 256, block_bytes = 176;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base, 0), dmin = half_at(base, 1);
        const uint8_t* sca = base + 4;
        const uint8_t* qh  = base + 16;
        const uint8_t* qs  = base + 48;
        const int chunk = col >> 6, pos = col & 63, sub = pos >> 5, l = pos & 31;
        const int is = 2 * chunk + sub;
        const int nib = sub ? (qs[chunk * 32 + l] >> 4) : (qs[chunk * 32 + l] & 0xF);
        const int hb = (qh[l] & (1 << (2 * chunk + sub))) ? 1 : 0;
        const int q = nib + hb * 16;
        int sc, mn;
        if (is < 4) { sc = sca[is] & 63; mn = sca[is + 4] & 63; }
        else {
            sc = (sca[is + 4] & 0x0F) | ((sca[is - 4] >> 6) << 4);
            mn = (sca[is + 4] >> 4)   | ((sca[is]     >> 6) << 4);
        }
        return d * float(sc) * float(q) - dmin * float(mn);
    }
};

// ---- q6_K : { u8 ql[128]; u8 qh[64]; int8 scales[16]; half d; } - 210 bytes, 256/block.
//   16 sub-blocks of 16; 6-bit q = (4 low in ql | 2 high in qh) - 32; int8 scales. ----
struct q6_K {
    static constexpr int block_k = 256, block_bytes = 210;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* ql = base;
        const uint8_t* qh = base + 128;
        const int8_t* sca = reinterpret_cast<const int8_t*>(base + 192);
        const float d = half_at(base + 208);
        const int chunk = col >> 7, pos = col & 127, group = pos >> 5, l = pos & 31;
        const int ql_byte = ql[chunk * 64 + l + 32 * (group & 1)];
        const int nib = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0xF);
        const int hbits = (qh[chunk * 32 + l] >> (2 * group)) & 3;
        const int q = (nib | (hbits << 4)) - 32;
        const int sc_idx = chunk * 8 + (l >> 4) + group * 2;
        return d * float(int(sca[sc_idx])) * float(q);
    }
};

// ---- iq4_nl : { half d; u8 qs[16]; } - 18 bytes, 32/block. value = d * kvalues_iq4nl[nib]
//   (q4_0-style nibble layout). ----
struct iq4_nl {
    static constexpr int block_k = 32, block_bytes = 18;
    __device__ static float dequant(const uint8_t* base, int col) {
        const uint8_t* qs = base + 2;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return half_at(base) * float(kvalues_iq4nl[nib]);
    }
};

// ---- iq4_xs : 256-superblock IQ4_NL. { half d; u16 scales_h; u8 scales_l[4]; u8 qs[128]; }
//   - 136 bytes. 8 sub-blocks of 32; 6-bit sub-scale ls - 32. value = d*ls * codebook[nib]. ----
struct iq4_xs {
    static constexpr int block_k = 256, block_bytes = 136;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint16_t scales_h = *reinterpret_cast<const uint16_t*>(base + 2);
        const uint8_t* scales_l = base + 4;
        const uint8_t* qs = base + 8;
        const int ib = col >> 5, local = col & 31;
        const int sl = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const int sh = (scales_h >> (2 * ib)) & 0x3;
        const int ls = (sl | (sh << 4)) - 32;
        const float dl = d * float(ls);
        const int nib = (local < 16) ? (qs[16 * ib + local] & 0x0F)
                                     : (qs[16 * ib + (local - 16)] >> 4);
        return dl * float(kvalues_iq4nl[nib]);
    }
};

// ---- iq2_xxs : E8-lattice 2.0625 bpw. { half d; u16 qs[32]; } - 66 bytes, 256/block.
//   Per 32: 4 u16 grid indices + 4 u16 signs/scale; grid entry = 8 packed u8 magnitudes. ----
struct iq2_xxs {
    static constexpr int block_k = 256, block_bytes = 66;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(base + 2);
        const int ib32 = col >> 5, p = col & 31, sub = p >> 3, elem = p & 7;
        const uint16_t* q2 = qs + 4 * ib32;
        const uint32_t aux_g = uint32_t(q2[0]) | (uint32_t(q2[1]) << 16);
        const uint32_t aux_s = uint32_t(q2[2]) | (uint32_t(q2[3]) << 16);
        const uint32_t g = (aux_g >> (8 * sub)) & 0xff;
        const uint32_t gv = uint32_t((iq2xxs_grid[g] >> (8 * elem)) & 0xffULL);
        const uint8_t signs = ksigns_iq2xs[(aux_s >> (7 * sub)) & 127];
        const float dl = d * (0.5f + float((aux_s >> 28) & 0xf)) * 0.25f;
        const float sgn = (signs & kmask_iq2xs[elem]) ? -1.0f : 1.0f;
        return dl * float(gv) * sgn;
    }
};

// ---- iq2_xs : E8-lattice 2.3125 bpw. { half d; u16 qs[32]; u8 scales[8]; } - 74 bytes.
//   u16 = 9-bit grid index | 7-bit signs; 4-bit per-half scale. ----
struct iq2_xs {
    static constexpr int block_k = 256, block_bytes = 74;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(base + 2);
        const uint8_t* scales = base + 66;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, sub2 = (p & 15) >> 3, elem = p & 7;
        const uint16_t idx16 = qs[4 * ib32 + 2 * il + sub2];
        const uint32_t g = idx16 & 511;
        const uint8_t signs = ksigns_iq2xs[idx16 >> 9];
        const int sc = (scales[ib32] >> (4 * il)) & 0xF;
        const float dl = d * (0.5f + float(sc)) * 0.25f;
        const uint32_t gv = uint32_t((iq2xs_grid[g] >> (8 * elem)) & 0xffULL);
        const float sgn = (signs & kmask_iq2xs[elem]) ? -1.0f : 1.0f;
        return dl * float(gv) * sgn;
    }
};

// ---- iq3_xxs : E8-lattice 3.0625 bpw. { half d; u8 qs[96]; } - 98 bytes. First 64 bytes =
//   8-bit grid indices (8 per 32); next 32 = u16 signs/scale. Grid entry = u32 of 4 magnitudes. ----
struct iq3_xxs {
    static constexpr int block_k = 256, block_bytes = 98;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint8_t* qs = base + 2;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15, r = w >> 2, i = w & 3;
        const uint8_t* q3 = qs + 8 * ib32;
        const uint16_t* gas = reinterpret_cast<const uint16_t*>(qs + 64) + 2 * ib32;
        const uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
        const uint32_t gv = (iq3xxs_grid[q3[4 * il + r]] >> (8 * i)) & 0xff;
        const uint8_t signs = ksigns_iq2xs[(aux32 >> (14 * il + 7 * (r >> 1))) & 127];
        const float dl = d * (0.5f + float(aux32 >> 28)) * 0.5f;
        const float sgn = (signs & kmask_iq2xs[i + 4 * (r & 1)]) ? -1.0f : 1.0f;
        return dl * float(gv) * sgn;
    }
};

// ---- iq1_s : 1.5625 bpw. { half d; u8 qs[32]; u16 qh[8]; } - 50 bytes. Grid index = qs byte |
//   high bits from qh; 3-bit scale + sign bit give dl and the ml offset (value = dl*nib + ml). ----
struct iq1_s {
    static constexpr int block_k = 256, block_bytes = 50;
    __device__ static float dequant(const uint8_t* base, int col) {
        const float d = half_at(base);
        const uint8_t* qs = base + 2;
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(base + 34);
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15;
        const int which = w >> 2, i = w & 3;
        const uint8_t* qsp = qs + 4 * ib32 + 2 * il;
        const uint16_t qhv = qh[ib32];
        const float dl = d * float(2 * ((qhv >> 12) & 7) + 1);
        const float ml = dl * ((qhv & 0x8000) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA));
        const uint32_t h = uint32_t(qhv >> (6 * il));
        const uint32_t gi = (which >> 1) == 0 ? (qsp[0] | ((h << 8) & 0x700))
                                              : (qsp[1] | ((h << 5) & 0x700));
        const uint32_t b = (iq1s_grid_gpu[gi] >> (8 * i)) & 0xff;
        const uint32_t nib = (which & 1) ? (b >> 4) : (b & 0xF);
        return dl * float(nib) + ml;
    }
};

// ---- dequant8 span specializations for the k-quants (perf pass): an
// 8-span never crosses a sub-block, so the packed sub-scale/min decode runs
// once per span instead of once per element. Bit-identical values. ----
template<>
__device__ __forceinline__ void dequant8<q2_K>(const uint8_t* base, int col0, float w[8]) {
    const uint8_t* scales = base;
    const uint8_t* qs = base + 16;
    const float d = half_at(base + 80), dmin = half_at(base + 82);
    const int chunk = col0 >> 7, pos = col0 & 127, sidx = pos >> 5, sub = (pos >> 4) & 1;
    const int l0 = pos & 15;
    const int is = chunk * 8 + sidx * 2 + sub;
    const float dl = d * float(scales[is] & 0xF);
    const float ml = dmin * float(scales[is] >> 4);
    const uint8_t* qp = qs + chunk * 32 + sub * 16 + l0;
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = dl * float((qp[i] >> (2 * sidx)) & 3) - ml;
}
template<>
__device__ __forceinline__ void dequant8<q3_K>(const uint8_t* base, int col0, float w[8]) {
    const uint8_t* hmask = base;
    const uint8_t* qs = base + 32;
    const uint8_t* sca = base + 96;
    const float d = half_at(base + 108);
    const int chunk = col0 >> 7, pos = col0 & 127, sidx = pos >> 5, sub = (pos >> 4) & 1;
    const int l0 = pos & 15;
    const int is = chunk * 8 + sidx * 2 + sub;
    const int ww = is >> 2, b = is & 3;
    int s;
    if      (ww == 0) s = (sca[b] & 0xF)            | ((sca[8 + b] & 3) << 4);
    else if (ww == 1) s = (sca[4 + b] & 0xF)        | (((sca[8 + b] >> 2) & 3) << 4);
    else if (ww == 2) s = ((sca[b] >> 4) & 0xF)     | (((sca[8 + b] >> 4) & 3) << 4);
    else              s = ((sca[4 + b] >> 4) & 0xF) | (((sca[8 + b] >> 6) & 3) << 4);
    const float ds = d * float(s - 32);
    const uint8_t* qp = qs + chunk * 32 + sub * 16 + l0;
    const uint8_t* hp = hmask + sub * 16 + l0;
    const int hbit = 1 << (chunk * 4 + sidx);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int low2 = (qp[i] >> (2 * sidx)) & 3;
        w[i] = ds * float((low2 | ((hp[i] & hbit) ? 4 : 0)) - 4);
    }
}
template<>
__device__ __forceinline__ void dequant8<q4_K>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base, 0), dmin = half_at(base, 1);
    const uint8_t* scales = base + 4;
    const uint8_t* qs = base + 16;
    const int chunk = col0 / 64, pos = col0 % 64;
    const int sub = chunk * 2 + (pos >= 32 ? 1 : 0);
    const int sh = (pos < 32) ? 0 : 4;
    const uint8_t* qp = qs + chunk * 32 + ((pos < 32) ? pos : pos - 32);
    int sc, m;
    if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
    else {
        sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
        m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
    }
    const float dsc = d * float(sc), dm = dmin * float(m);
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = dsc * float((qp[i] >> sh) & 0x0F) - dm;
}
template<>
__device__ __forceinline__ void dequant8<q5_K>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base, 0), dmin = half_at(base, 1);
    const uint8_t* sca = base + 4;
    const uint8_t* qh = base + 16;
    const uint8_t* qs = base + 48;
    const int chunk = col0 >> 6, pos = col0 & 63, sub = pos >> 5, l0 = pos & 31;
    const int is = 2 * chunk + sub;
    int sc, mn;
    if (is < 4) { sc = sca[is] & 63; mn = sca[is + 4] & 63; }
    else {
        sc = (sca[is + 4] & 0x0F) | ((sca[is - 4] >> 6) << 4);
        mn = (sca[is + 4] >> 4)   | ((sca[is]     >> 6) << 4);
    }
    const float dsc = d * float(sc), dm = dmin * float(mn);
    const uint8_t* qp = qs + chunk * 32 + l0;
    const uint8_t* hp = qh + l0;
    const int sh = sub ? 4 : 0;
    const int hbit = 1 << (2 * chunk + sub);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int q = ((qp[i] >> sh) & 0xF) + ((hp[i] & hbit) ? 16 : 0);
        w[i] = dsc * float(q) - dm;
    }
}
template<>
__device__ __forceinline__ void dequant8<q6_K>(const uint8_t* base, int col0, float w[8]) {
    const uint8_t* ql = base;
    const uint8_t* qh = base + 128;
    const int8_t* sca = reinterpret_cast<const int8_t*>(base + 192);
    const float d = half_at(base + 208);
    const int chunk = col0 >> 7, pos = col0 & 127, group = pos >> 5, l0 = pos & 31;
    const float dsc = d * float(int(sca[chunk * 8 + (l0 >> 4) + group * 2]));
    const uint8_t* qlp = ql + chunk * 64 + l0 + 32 * (group & 1);
    const uint8_t* qhp = qh + chunk * 32 + l0;
    const int sh = (group & 2) ? 4 : 0;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        const int nib = (qlp[i] >> sh) & 0xF;
        const int q = (nib | (((qhp[i] >> (2 * group)) & 3) << 4)) - 32;
        w[i] = dsc * float(q);
    }
}

// ---- dequant8 span specializations for the table formats (perf pass): a
// col0-aligned 8-span shares ONE grid entry (iq2*/iq1) or one sign byte +
// two u32 entries (iq3), and one sub-scale — so the divergent __constant__
// lookups drop 8x (the gemv was 1.2-7 GB/s from per-element re-lookups).
// Bit-identical values (same decode math, same fp32 op order per element). ----
template<>
__device__ __forceinline__ void dequant8<iq4_nl>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base);
    const uint8_t* qs = base + 2;
    const int sh = (col0 < 16) ? 0 : 4;
    const int qo = (col0 < 16) ? col0 : col0 - 16;
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = d * float(kvalues_iq4nl[(qs[qo + i] >> sh) & 0x0F]);
}
template<>
__device__ __forceinline__ void dequant8<iq4_xs>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base);
    const uint16_t scales_h = *reinterpret_cast<const uint16_t*>(base + 2);
    const uint8_t* scales_l = base + 4;
    const uint8_t* qs = base + 8;
    const int ib = col0 >> 5, local = col0 & 31;
    const int sl = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int sh = (scales_h >> (2 * ib)) & 0x3;
    const float dl = d * float((sl | (sh << 4)) - 32);
    const int shf = (local < 16) ? 0 : 4;
    const uint8_t* qp = qs + 16 * ib + ((local < 16) ? local : local - 16);
    #pragma unroll
    for (int i = 0; i < 8; i++) w[i] = dl * float(kvalues_iq4nl[(qp[i] >> shf) & 0x0F]);
}
template<>
__device__ __forceinline__ void dequant8<iq2_xxs>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base);
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(base + 2);
    const int ib32 = col0 >> 5, sub = (col0 & 31) >> 3;
    const uint16_t* q2 = qs + 4 * ib32;
    const uint32_t aux_g = uint32_t(q2[0]) | (uint32_t(q2[1]) << 16);
    const uint32_t aux_s = uint32_t(q2[2]) | (uint32_t(q2[3]) << 16);
    const unsigned long long grid = iq2xxs_grid[(aux_g >> (8 * sub)) & 0xff];   // one lookup
    const uint8_t signs = ksigns_iq2xs[(aux_s >> (7 * sub)) & 127];
    const float dl = d * (0.5f + float((aux_s >> 28) & 0xf)) * 0.25f;
    #pragma unroll
    for (int e = 0; e < 8; e++)
        w[e] = dl * float(uint32_t((grid >> (8 * e)) & 0xffULL))
                  * ((signs & kmask_iq2xs[e]) ? -1.0f : 1.0f);
}
template<>
__device__ __forceinline__ void dequant8<iq2_xs>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base);
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(base + 2);
    const uint8_t* scales = base + 66;
    const int ib32 = col0 >> 5, p = col0 & 31, il = p >> 4, sub2 = (p & 15) >> 3;
    const uint16_t idx16 = qs[4 * ib32 + 2 * il + sub2];
    const unsigned long long grid = iq2xs_grid[idx16 & 511];                    // one lookup
    const uint8_t signs = ksigns_iq2xs[idx16 >> 9];
    const float dl = d * (0.5f + float((scales[ib32] >> (4 * il)) & 0xF)) * 0.25f;
    #pragma unroll
    for (int e = 0; e < 8; e++)
        w[e] = dl * float(uint32_t((grid >> (8 * e)) & 0xffULL))
                  * ((signs & kmask_iq2xs[e]) ? -1.0f : 1.0f);
}
template<>
__device__ __forceinline__ void dequant8<iq3_xxs>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base);
    const uint8_t* qs = base + 2;
    const int ib32 = col0 >> 5, p = col0 & 31, il = p >> 4, r0 = (p & 15) >> 2;
    const uint8_t* q3 = qs + 8 * ib32;
    const uint16_t* gas = reinterpret_cast<const uint16_t*>(qs + 64) + 2 * ib32;
    const uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
    const uint32_t g0 = iq3xxs_grid[q3[4 * il + r0]];                           // two lookups
    const uint32_t g1 = iq3xxs_grid[q3[4 * il + r0 + 1]];
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (14 * il + 7 * (r0 >> 1))) & 127];
    const float dl = d * (0.5f + float(aux32 >> 28)) * 0.5f;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        w[i]     = dl * float((g0 >> (8 * i)) & 0xff)
                      * ((signs & kmask_iq2xs[i + 4 * (r0 & 1)]) ? -1.0f : 1.0f);
        w[4 + i] = dl * float((g1 >> (8 * i)) & 0xff)
                      * ((signs & kmask_iq2xs[i + 4 * ((r0 + 1) & 1)]) ? -1.0f : 1.0f);
    }
}
template<>
__device__ __forceinline__ void dequant8<iq1_s>(const uint8_t* base, int col0, float w[8]) {
    const float d = half_at(base);
    const uint8_t* qs = base + 2;
    const uint16_t* qh = reinterpret_cast<const uint16_t*>(base + 34);
    const int ib32 = col0 >> 5, p = col0 & 31, il = p >> 4, w0 = p & 15;
    const uint8_t* qsp = qs + 4 * ib32 + 2 * il;
    const uint16_t qhv = qh[ib32];
    const float dl = d * float(2 * ((qhv >> 12) & 7) + 1);
    const float ml = dl * ((qhv & 0x8000) ? (-1.0f - IQ1S_DELTA) : (-1.0f + IQ1S_DELTA));
    const uint32_t h = uint32_t(qhv >> (6 * il));
    const uint32_t gi = (w0 < 8) ? (qsp[0] | ((h << 8) & 0x700))
                                 : (qsp[1] | ((h << 5) & 0x700));
    const uint32_t grid = iq1s_grid_gpu[gi];                                    // one lookup
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t b = (grid >> (8 * i)) & 0xff;
        w[i]     = dl * float(b & 0xF) + ml;
        w[4 + i] = dl * float(b >> 4) + ml;
    }
}

}  // namespace tmq
