/**
 * @file
 * @brief In-register dequantization of quantized weight blocks into half tiles.
 *
 * "Marlin's method" on Apple: quantized weights are stored block-wise (a small fp16 scale +
 * packed low-precision codes), dequantized to `half` here, then fed to a standard
 * `simdgroup_matrix` MMA. The dequant math is pure IEEE-fp16 bit arithmetic, valid on Metal
 * `half` verbatim. Block layouts mirror llama.cpp's GGUF formats (ggml-common.h); the dequant
 * constants follow llama.cpp (ggml-metal.metal) and Marlin (dequant.h).
 *
 * A "format" is a small struct exposing `block_k` (weights per block), `block_bytes`, and a
 * `dequant(device const uchar* base, int col) -> half` for the weight at column `col` of the
 * block starting at byte `base`. `dequant_into_shared<FMT>` cooperatively dequantizes a tile.
 */
#pragma once
#include "../../../../common/common.metal"
#include "../../../../types/types.metal"
#include "dequant_tables.metal"   // GGUF i-quant lattice/codebook constant tables (namespace mittens)

namespace mittens {

// ---- integer dot primitive (the "idot/imma" unlock) ------------------------------------------
// Apple's simdgroup_matrix has no integer path, so activation-quantized kernels accumulate in
// int32 on the ALUs. idot4 is the dp4a equivalent (4 packed signed int8 per uint -> int32),
// modeled on BitNet's __dp4a; a per-lane int8 GEMV then simd_sum-reduces (see qgemv_w8a8/_w2a8).
METAL_FUNC int idot4(uint a, uint b) {
    int s = 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; i++) {
        s += (int)(char)((a >> (8 * i)) & 0xffu) * (int)(char)((b >> (8 * i)) & 0xffu);
    }
    return s;
}

// ---- codebook / lookup-table dequant primitive (the second new style) -------------------------
// Packed bits index a constant table instead of doing bit arithmetic. kvalues_iq4nl is GGUF's
// 16-entry non-linear fp codebook (ggml-common.h); a nibble indexes it, then * the block scale.
constant const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// ---- iq4_nl : { half d; uint8 qs[16]; }  — 18 bytes, 32 weights, value = d * kvalues_iq4nl[nib]
//   (q4_0-style nibble layout: col<16 -> low nibble of qs[col], else high nibble of qs[col-16]). ----
struct iq4_nl {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return d * half(kvalues_iq4nl[nib]);
    }
};

// ---- iq4_xs : 256-superblock IQ4_NL. { half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }
//   = 136 bytes. 8 sub-blocks of 32; each has a 6-bit scale ls = (4 low bits in scales_l | 2 high
//   bits in scales_h) − 32, so value = d·ls · kvalues_iq4nl[nibble]. (ggml-common.h block_iq4_xs.) ----
struct iq4_xs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 136;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        const ushort scales_h = ((device const ushort*)(base + 2))[0];
        device const uchar* scales_l = base + 4;       // 4 bytes
        device const uchar* qs = base + 8;             // 128 bytes
        const int ib = col >> 5;                       // sub-block 0..7
        const int local = col & 31;
        const int sl = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
        const int sh = (scales_h >> (2 * ib)) & 0x3;
        const int ls = (sl | (sh << 4)) - 32;          // 6-bit signed sub-scale
        const half dl = d * half(ls);
        const int nib = (local < 16) ? (qs[16 * ib + local] & 0x0F)
                                     : (qs[16 * ib + (local - 16)] >> 4);
        return dl * half(kvalues_iq4nl[nib]);
    }
};

// ---- iq2_xxs : E8-lattice 2.0625 bpw. { half d; uint16 qs[32]; } = 66 bytes, 256 weights.
//   Per block-of-32 (4 groups of 8): 4 uint16 = grid indices (aux_g) + signs/scale (aux_s). Each
//   8-bit grid index selects an iq2xxs_grid entry (8 packed uint8 magnitudes); a 7-bit ksigns index
//   gives the 8 signs; the top 4 bits of aux_s give the sub-scale. (ggml-metal dequantize_iq2_xxs.) ----
struct iq2_xxs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 66;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const ushort* qs = (device const ushort*)(base + 2);
        const int ib32 = col >> 5, p = col & 31, sub = p >> 3, elem = p & 7;
        device const ushort* q2 = qs + 4 * ib32;
        const uint aux_g = (uint)q2[0] | ((uint)q2[1] << 16);
        const uint aux_s = (uint)q2[2] | ((uint)q2[3] << 16);
        const uint g = (aux_g >> (8 * sub)) & 0xff;
        const uint gv = (uint)((iq2xxs_grid[g] >> (8 * elem)) & 0xffUL);
        const uchar signs = ksigns_iq2xs[(aux_s >> (7 * sub)) & 127];
        const half dl = d * (0.5h + half((aux_s >> 28) & 0xf)) * 0.25h;
        const half sgn = (signs & kmask_iq2xs[elem]) ? -1.0h : 1.0h;
        return dl * half(gv) * sgn;
    }
};

// ---- iq2_xs : E8-lattice 2.3125 bpw. { half d; uint16 qs[32]; uint8 scales[8]; } = 74 bytes, 256
//   weights. Each uint16: low 9 bits = iq2xs_grid index (512), high 7 = ksigns index; 4-bit
//   per-half scale from scales[ib32]. (ggml-metal dequantize_iq2_xs.) ----
struct iq2_xs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 74;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const ushort* qs = (device const ushort*)(base + 2);
        device const uchar* scales = base + 66;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, sub2 = (p & 15) >> 3, elem = p & 7;
        const ushort idx16 = qs[4 * ib32 + 2 * il + sub2];
        const uint g = idx16 & 511;
        const uchar signs = ksigns_iq2xs[idx16 >> 9];
        const int sc = (scales[ib32] >> (4 * il)) & 0xF;
        const half dl = d * (0.5h + half(sc)) * 0.25h;
        const uint gv = (uint)((iq2xs_grid[g] >> (8 * elem)) & 0xffUL);
        const half sgn = (signs & kmask_iq2xs[elem]) ? -1.0h : 1.0h;
        return dl * half(gv) * sgn;
    }
};

// ---- iq3_xxs : E8-lattice 3.0625 bpw. { half d; uint8 qs[96]; } = 98 bytes, 256 weights. First
//   64 bytes of qs = 8-bit grid indices (8 per block-of-32); the next 32 = uint16 sign/scale (gas).
//   Each iq3xxs_grid entry is a uint32 of 4 magnitudes. (ggml-metal dequantize_iq3_xxs.) ----
struct iq3_xxs {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 98;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15, r = w >> 2, i = w & 3;
        device const uchar* q3 = qs + 8 * ib32;
        device const ushort* gas = (device const ushort*)(qs + 64) + 2 * ib32;
        const uint aux32 = (uint)gas[0] | ((uint)gas[1] << 16);
        const uint gv = (iq3xxs_grid[q3[4 * il + r]] >> (8 * i)) & 0xff;
        const uchar signs = ksigns_iq2xs[(aux32 >> (14 * il + 7 * (r >> 1))) & 127];
        const half dl = d * (0.5h + half(aux32 >> 28)) * 0.5h;
        const half sgn = (signs & kmask_iq2xs[i + 4 * (r & 1)]) ? -1.0h : 1.0h;
        return dl * half(gv) * sgn;
    }
};

// ---- iq1_s : 1.5625 bpw. { half d; uint8 qs[32]; uint16 qh[8]; } = 50 bytes, 256 weights. Per
//   half: two iq1s_grid_gpu entries (index = qs byte | high bits from qh); 3-bit scale + a sign in
//   qh give dl and the ml offset (value = dl·nibble + ml). (ggml-metal dequantize_iq1_s.) ----
struct iq1_s {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 50;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        device const ushort* qh = (device const ushort*)(base + 34);
        const int ib32 = col >> 5, p = col & 31, il = p >> 4, w = p & 15;
        const int which = w >> 2, i = w & 3;        // which: 0/1 -> grid1 lo/hi, 2/3 -> grid2 lo/hi
        device const uchar* qsp = qs + 4 * ib32 + 2 * il;
        const ushort qhv = qh[ib32];
        const half dl = d * half(2 * ((qhv >> 12) & 7) + 1);
        const half ml = dl * ((qhv & 0x8000) ? half(-1.0h - IQ1S_DELTA) : half(-1.0h + IQ1S_DELTA));
        const uint h = (uint)(qhv >> (6 * il));
        const uint gi = (which >> 1) == 0 ? (qsp[0] | ((h << 8) & 0x700))
                                          : (qsp[1] | ((h << 5) & 0x700));
        const uint b = (iq1s_grid_gpu[gi] >> (8 * i)) & 0xff;
        const uint nib = (which & 1) ? (b >> 4) : (b & 0xF);
        return dl * half(nib) + ml;
    }
};

// ---- q8_0 : { half d; int8 qs[32]; }  — 34 bytes, 32 weights/block, value = d * q ----
struct q8_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        const char q = ((device const char*)(base + 2))[col];  // signed int8 codes at offset 2
        return d * half(q);
    }
    // integer path: the raw int8 code and the per-group (block) scale, kept separate.
    static METAL_FUNC int  code(device const uchar* base, int col) { return (int)((device const char*)(base + 2))[col]; }
    static METAL_FUNC half gscale(device const uchar* base)        { return ((device const half*)base)[0]; }
};

// ---- q4_0 : { half d; uint8 qs[16]; } — 18 bytes, 32 weights/block. Nibble packing (ggml):
//   weight col<16 -> qs[col]&0xF ; col>=16 -> qs[col-16]>>4 ; value = d * (nibble - 8). ----
struct q4_0 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];      // fp16 scale at offset 0
        device const uchar* qs = base + 2;                 // 16 packed-nibble bytes
        const int nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return d * half(nib - 8);
    }
};

// ---- q4_K : { half d; half dmin; uint8 scales[12]; uint8 qs[128]; } — 144 bytes, 256/block.
//   8 sub-blocks of 32; each has a 6-bit scale `sc` and 6-bit min `m` (packed in `scales`,
//   extracted GGUF-style). value = (d*sc)*nibble - (dmin*m). ----
struct q4_K {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 144;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d    = ((device const half*)base)[0];
        const half dmin = ((device const half*)base)[1];
        device const uchar* scales = base + 4;
        device const uchar* qs     = base + 16;
        const int chunk = col / 64;        // 0..3
        const int pos   = col % 64;        // 0..63
        int sub, nib;
        if (pos < 32) { sub = chunk * 2;     nib = qs[chunk * 32 + pos]        & 0x0F; }
        else          { sub = chunk * 2 + 1; nib = qs[chunk * 32 + (pos - 32)] >> 4;   }
        // get_scale_min_k4(sub): unpack the 6-bit scale `sc` and min `m`
        uchar sc, m;
        if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
        else {
            sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
            m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
        }
        return d * half(sc) * half(nib) - dmin * half(m);
    }
};

// ---- kU4B8 : GPTQ/Marlin grouped int4, group=128. { half scale; uint8 qs[64]; } — 66 bytes.
//   unsigned 4-bit with bias 8; value = scale * (nibble - 8). Nibble packing like q4_0 (col<64 ->
//   qs[col]&0xF ; col>=64 -> qs[col-64]>>4). Larger group than q4_0 (more compression). ----
struct kU4B8 {
    constant static constexpr const int block_k     = 128;
    constant static constexpr const int block_bytes = 66;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return scale * half(nib - 8);
    }
};

// ---- kU4 : AWQ grouped int4, group=128, per-group zero-point. { half scale; half zp;
//   uint8 qs[64]; } — 68 bytes. value = scale * (nibble - zp). ----
struct kU4 {
    constant static constexpr const int block_k     = 128;
    constant static constexpr const int block_bytes = 68;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        const half zp    = ((device const half*)base)[1];
        device const uchar* qs = base + 4;
        const int nib = (col < 64) ? (qs[col] & 0x0F) : (qs[col - 64] >> 4);
        return scale * (half(nib) - zp);
    }
};

// ---- hqq : HQQ int4 + per-group zero-point, group 64 (a thin kU4 variant at a finer group size).
//   { half scale; half zp; uint8 qs[32]; } — 36 bytes. value = scale*(nibble - zp). ----
struct hqq {
    constant static constexpr const int block_k = 64, block_bytes = 36;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        const half zp    = ((device const half*)base)[1];
        device const uchar* qs = base + 4;
        const int nib = (col < 32) ? (qs[col] & 0x0F) : (qs[col - 32] >> 4);
        return scale * (half(nib) - zp);
    }
};

// ---- float-code decoders (bit tricks; widen to half) ----
// Normal codes shift the exponent/mantissa fields into the fp16 field positions, then rescale by
// a power-of-two constant to fix the bias difference — exact, no exp2. Subnormal codes (e == 0)
// take a select computed in NORMAL half arithmetic: the shifted bit pattern would be a subnormal
// half, and the offline `xcrun metal` build (tk_torch) flushes subnormal arithmetic to zero
// (fast-math FTZ) while MLX's metallib does not — the select keeps both backends exact. The
// encoders clamp NaN/inf codes so they never reach the decoders.
// fp8 e4m3 (1-4-3, bias 7): normal = as_half(code<<7) * 2^(15-7); subnormal = m * 2^-9.
METAL_FUNC half tk_e4m3_decode(uchar v) {
    const ushort h = ushort(v & 0x7F) << 7;
    const half mag = (v & 0x78) ? as_type<half>(h) * 256.0h : half(v & 0x7) * 0.001953125h;
    return (v & 0x80) ? -mag : mag;
}
// fp4 e2m1 (1-2-1, bias 1): normal = as_half(code<<9) * 2^(15-1); subnormal values are 0 / 0.5.
METAL_FUNC half tk_e2m1_decode(uint nib) {
    const ushort h = ushort(nib & 0x7) << 9;
    const half mag = (nib & 0x6) ? as_type<half>(h) * 16384.0h : ((nib & 1) ? 0.5h : 0.0h);
    return (nib & 0x8) ? -mag : mag;
}

// E8M0 stores the biased IEEE exponent directly. For codes 1..254, placing
// the byte in a float exponent field reconstructs 2^(code-127) exactly; code
// zero is the one subnormal exception (2^-127). Code 255 maps to infinity,
// matching exp2(128). The half helper preserves the prior half-rounded scale
// contract without issuing a transcendental exp2 instruction.
METAL_FUNC float tk_e8m0_decode_f32(uint code) {
    const uint bits = code == 0 ? 0x00400000u : (code << 23);
    return as_type<float>(bits);
}

METAL_FUNC half tk_e8m0_decode(uint code) {
    return half(tk_e8m0_decode_f32(code));
}
// fp8 e5m2 (1-5-2, bias 15): e5m2 IS truncated fp16 — value = as_half(code << 8), a pure bitcast
// (e5m2 subnormals are genuine fp16 subnormals; constructing the bits directly involves no
// arithmetic, so FTZ cannot flush the decode itself).
METAL_FUNC half tk_e5m2_decode(uchar v) {
    return as_type<half>(ushort(ushort(v) << 8));
}

// --- Quantize/pack: round-to-nearest float -> fp8 / int8 codes (inverse of the
//     decoders above; standard OCP layouts so tk_e4m3_decode/tk_e5m2_decode invert them). ---

// float -> e4m3 (1-4-3, bias 7, max finite 448). NaN/overflow clamp to ±448 (0x7E).
// Encode float -> fp4 e2m1 code (4 bits: sign + {0,.5,1,1.5,2,3,4,6}). Nearest-decoded-value
// with ties -> the LOWEST code index, matching the host packer's np.argmin over the same
// code order (tk/quant.py _nearest with _E2M1_CODES = arange(16): positives 0..7 first).
METAL_FUNC uchar tk_e2m1_encode(float v) {
    float best = INFINITY;
    uchar bc = 0;
    #pragma clang loop unroll(full)
    for (int c = 0; c < 16; ++c) {
        const float d = metal::abs(v - float(tk_e2m1_decode((uint)c)));
        if (d < best) { best = d; bc = (uchar)c; }
    }
    return bc;
}

METAL_FUNC uchar tk_e4m3_encode(float x) {
    const uint sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = metal::fabs(x);
    if (a >= 448.0f) {
        return uchar(sign | 0x7Eu);              // clamp to max finite
    }
    if (!(a > 1.52587890625e-05f)) {             // < 2^-16 (half the smallest subnormal 2^-9? guard) -> 0
        // smallest e4m3 subnormal is 2^-9; round-to-zero below 2^-10
        if (a < 9.765625e-04f) {                 // 2^-10
            return uchar(sign);
        }
    }
    // The IEEE exponent is the E4M3 exponent before rebiasing. For normal outputs the top
    // three fraction bits are the mantissa and bit 19 is the half-ULP rounding bit, so encoding
    // needs no frexp/ldexp/divide sequence. Quantized inference values overwhelmingly take this
    // path; retain the explicit arithmetic path only for E4M3 subnormals.
    const uint bits = as_type<uint>(a);
    const int E = int((bits >> 23) & 0xFFu) - 127;
    if (E < -6) {
        int mant = int(metal::round(a * 512.0f)); // a / 2^-9
        if (mant <= 0) return uchar(sign);
        if (mant >= 8) return uchar(sign | (1u << 3)); // promote to smallest normal
        return uchar(sign | uint(mant));
    }
    const uint fraction = bits & 0x7FFFFFu;
    int mant = int((fraction + 0x80000u) >> 20); // nearest, halfway rounds up like metal::round
    int exp = E + 7;
    if (mant >= 8) { mant = 0; exp += 1; }
    if (exp >= 15 && mant >= 7) return uchar(sign | 0x7Eu);
    if (exp > 15) return uchar(sign | 0x7Eu);
    return uchar(sign | (uint(exp) << 3) | uint(mant));
}

// float -> e5m2 (1-5-2, bias 15, max finite 57344). Overflow clamp to ±57344 (0x7B).
METAL_FUNC uchar tk_e5m2_encode(float x) {
    const uint sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = metal::fabs(x);
    if (a >= 57344.0f) {
        return uchar(sign | 0x7Bu);
    }
    if (a < 3.0517578125e-05f / 2.0f) {          // < 2^-16 (half smallest subnormal) -> 0
        return uchar(sign);
    }
    const uint bits = as_type<uint>(a);
    const int E = int((bits >> 23) & 0xFFu) - 127;
    if (E < -14) {
        int mant = int(metal::round(a * 65536.0f)); // a / 2^-16
        if (mant <= 0) return uchar(sign);
        if (mant >= 4) return uchar(sign | (1u << 2));
        return uchar(sign | uint(mant));
    }
    const uint fraction = bits & 0x7FFFFFu;
    int mant = int((fraction + 0x100000u) >> 21);
    int exp = E + 15;
    if (mant >= 4) { mant = 0; exp += 1; }
    if (exp >= 31) return uchar(sign | 0x7Bu);
    return uchar(sign | (uint(exp) << 2) | uint(mant));
}

// float -> symmetric int8 in [-127, 127] (round to nearest, clamp). Returns as uchar bits.
METAL_FUNC char tk_int8_encode(float x) {
    float r = metal::round(x);
    r = metal::clamp(r, -127.0f, 127.0f);
    return char(int(r));
}
// fp6 e3m2 (1-3-2, bias 3): normal = as_half(code<<8) * 2^(15-3); subnormal = m * 2^-4.
METAL_FUNC half tk_e3m2_decode(uint c) {
    const ushort h = ushort(c & 0x1F) << 8;
    const half mag = (c & 0x1C) ? as_type<half>(h) * 4096.0h : half(c & 0x3) * 0.0625h;
    return (c & 0x20) ? -mag : mag;
}
// fp6 e2m3 (1-2-3, bias 1): normal = as_half(code<<7) * 2^(15-1); subnormal = m * 0.125.
METAL_FUNC half tk_e2m3_decode(uint c) {
    const ushort h = ushort(c & 0x1F) << 7;
    const half mag = (c & 0x18) ? as_type<half>(h) * 16384.0h : half(c & 0x7) * 0.125h;
    return (c & 0x20) ? -mag : mag;
}

// ---- fp8_e4m3 : per-group (32) half-scaled fp8. { half scale; uint8 qs[32]; } — 34 bytes.
//   value = scale * e4m3(q). ----
struct fp8_e4m3 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        return ((device const half*)base)[0] * tk_e4m3_decode((base + 2)[col]);
    }
};

// ---- fp4_e2m1 : per-group (32) half-scaled fp4 (nibbles, q4_0-style packing). 18 bytes. ----
struct fp4_e2m1 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 18;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* qs = base + 2;
        const uint nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return ((device const half*)base)[0] * tk_e2m1_decode(nib);
    }
};

// ---- mxfp8 : OCP microscaling — 32-element block, e8m0 power-of-two block scale + fp8 e4m3.
//   { uint8 e8m0; uint8 qs[32]; } — 33 bytes. value = 2^(e8m0-127) * e4m3(q). ----
struct mxfp8 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 33;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = metal::exp2(half((int)base[0] - 127));
        return scale * tk_e4m3_decode((base + 1)[col]);
    }
};

// ---- nvfp4 : 16-element block, fp8 e4m3 block scale + fp4 e2m1 codes (nibbles).
//   { uint8 e4m3_scale; uint8 qs[8]; } — 9 bytes. value = e4m3(scale) * e2m1(nib). ----
struct nvfp4 {
    constant static constexpr const int block_k     = 16;
    constant static constexpr const int block_bytes = 9;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = tk_e4m3_decode(base[0]);
        device const uchar* qs = base + 1;
        const uint nib = (col < 8) ? (qs[col] & 0x0F) : (qs[col - 8] >> 4);
        return scale * tk_e2m1_decode(nib);
    }
};

// ---- mxfp4 : OCP microscaling — 32-element block, e8m0 power-of-two block scale + fp4 e2m1 codes
//   (nibbles). { uint8 e8m0; uint8 qs[16]; } — 17 bytes. value = 2^(e8m0-127) * e2m1(nib). ----
struct mxfp4 {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 17;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = metal::exp2(half((int)base[0] - 127));
        device const uchar* qs = base + 1;
        const uint nib = (col < 16) ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        return scale * tk_e2m1_decode(nib);
    }
};

// ---- bitnet : BitNet b1.58 ternary weights {-1,0,+1}, group 32, per-group absmean scale.
//   2-bit codes packed 4/byte (code in {0,1,2} -> value scale*(code-1)). { half scale; uint8 qs[8]; }
//   = 10 bytes. (BitNet's GPU kernel uses int8×int2 dp4a; Apple has no int matmul, so we dequant
//   ternary -> half and use the standard simdgroup MMA, like every other format here.) ----
struct bitnet {
    constant static constexpr const int block_k     = 32;
    constant static constexpr const int block_bytes = 10;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = ((device const half*)base)[0];
        device const uchar* qs = base + 2;                 // 8 bytes, 4 ternary codes each
        const uint code = (qs[col >> 2] >> ((col & 3) * 2)) & 0x3;
        return scale * half((int)code - 1);                // 0->-1, 1->0, 2->+1
    }
    // integer path (W2A8): the ternary code in {-1,0,+1} and the per-group absmean scale.
    static METAL_FUNC int code(device const uchar* base, int col) {
        device const uchar* qs = base + 2;
        return (int)((qs[col >> 2] >> ((col & 3) * 2)) & 0x3) - 1;
    }
    static METAL_FUNC half gscale(device const uchar* base) { return ((device const half*)base)[0]; }
};

// ---- tq2_0 : llama.cpp/ggml native ternary (TQ2_0 GGUF). 256-element block,
//   per-block absmax half scale, 2-bit codes in {0,1,2} -> d*(code-1).
//   ggml layout { uint8 qs[64]; half d; } = 66 bytes: scale last, and element
//   128j + 32n + m lives in qs[32j + m] at bits 2n.
struct tq2_0 {
    constant static constexpr const int block_k     = 256;
    constant static constexpr const int block_bytes = 66;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)(base + 64))[0];
        const uint q = (base[((col >> 7) << 5) + (col & 31)] >> (((col >> 5) & 3) * 2)) & 0x3;
        return d * half((int)q - 1);
    }
    static METAL_FUNC int code(device const uchar* base, int col) {
        return (int)((base[((col >> 7) << 5) + (col & 31)] >> (((col >> 5) & 3) * 2)) & 0x3) - 1;
    }
    static METAL_FUNC half gscale(device const uchar* base) {
        return ((device const half*)(base + 64))[0];
    }
};

// ============================ Phase 4: float sub-formats =========================================
// ---- e5m2 : per-group (32) half-scaled fp8 e5m2. { half scale; uint8 qs[32]; } — 34 bytes. ----
struct e5m2 {
    constant static constexpr const int block_k = 32, block_bytes = 34;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        return ((device const half*)base)[0] * tk_e5m2_decode((base + 2)[col]);
    }
};

// ---- fp8_block : 128x128 block-scaled fp8 e4m3 (compressed-tensors). Laid out as a per-row k-block
//   of 128 with the (128-row x 128-col) tile scale replicated into each row's scale slot, so the
//   per-row dequant reads the shared block scale. { half scale; uint8 qs[128]; } — 130 bytes. ----
struct fp8_block {
    constant static constexpr const int block_k = 128, block_bytes = 130;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        return ((device const half*)base)[0] * tk_e4m3_decode((base + 2)[col]);
    }
};

// ---- fp8_raw : codes-only fp8 e4m3 (no per-block scale), 128/block. For fp8_block2d, where the
//   128x128 tile scale is a SEPARATE buffer (storage-optimal — no per-row scale replication). ----
struct fp8_raw {
    constant static constexpr const int block_k = 128, block_bytes = 128;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        return tk_e4m3_decode(base[col]);
    }
};

// ---- mxfp6 (e3m2 / e2m3) : OCP microscaling 6-bit. { uint8 e8m0; uint8 codes[24]; } — 25 bytes,
//   32 weights. 4 six-bit codes pack into 3 bytes (little-endian 24-bit groups). scale = 2^(e-127). ----
template<bool E3M2>
struct mxfp6 {
    constant static constexpr const int block_k = 32, block_bytes = 25;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half scale = metal::exp2(half((int)base[0] - 127));
        const int g = col >> 2, within = col & 3;
        device const uchar* p = base + 1 + 3 * g;
        const uint val = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16);
        const uint c = (val >> (6 * within)) & 0x3F;
        return scale * (E3M2 ? tk_e3m2_decode(c) : tk_e2m3_decode(c));
    }
};
using mxfp6_e3m2 = mxfp6<true>;
using mxfp6_e2m3 = mxfp6<false>;

// ============================ Phase 3: GGUF k-quant + legacy fan-out ============================
// Byte layouts match ggml-common.h; per-column decoders mirror the ggml CPU dequantize_row_* refs.

// ---- q4_1 : { half d; half m; uint8 qs[16]; } — 20 bytes, 32/block. value = d*nibble + m. ----
struct q4_1 {
    constant static constexpr const int block_k = 32, block_bytes = 20;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0], m = ((device const half*)base)[1];
        device const uchar* qs = base + 4;
        const int nib = (col < 16) ? (qs[col] & 0xF) : (qs[col - 16] >> 4);
        return d * half(nib) + m;
    }
};

// ---- q5_0 : { half d; uint8 qh[4]; uint8 qs[16]; } — 22 bytes. value = d*(q-16), q = nibble |
//   (5th bit = bit `col` of the qh uint32). ----
struct q5_0 {
    constant static constexpr const int block_k = 32, block_bytes = 22;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0];
        const uint qh = (uint)base[2] | ((uint)base[3] << 8) | ((uint)base[4] << 16) | ((uint)base[5] << 24);
        device const uchar* qs = base + 6;
        const int nib = (col < 16) ? (qs[col] & 0xF) : (qs[col - 16] >> 4);
        const int q = nib | (((qh >> col) & 1) << 4);
        return d * half(q - 16);
    }
};

// ---- q5_1 : { half d; half m; uint8 qh[4]; uint8 qs[16]; } — 24 bytes. value = d*q + m. ----
struct q5_1 {
    constant static constexpr const int block_k = 32, block_bytes = 24;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0], m = ((device const half*)base)[1];
        const uint qh = (uint)base[4] | ((uint)base[5] << 8) | ((uint)base[6] << 16) | ((uint)base[7] << 24);
        device const uchar* qs = base + 8;
        const int nib = (col < 16) ? (qs[col] & 0xF) : (qs[col - 16] >> 4);
        const int q = nib | (((qh >> col) & 1) << 4);
        return d * half(q) + m;
    }
};

// ---- q2_K : { uint8 scales[16]; uint8 qs[64]; half d; half dmin; } — 84 bytes, 256/block.
//   16 sub-blocks of 16; scales byte = 4-bit dl-scale | 4-bit min. value = d*sc*q - dmin*m. ----
struct q2_K {
    constant static constexpr const int block_k = 256, block_bytes = 84;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* scales = base; device const uchar* qs = base + 16;
        const half d = ((device const half*)(base + 80))[0], dmin = ((device const half*)(base + 82))[0];
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int q = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        return d * half(scales[is] & 0xF) * half(q) - dmin * half(scales[is] >> 4);
    }
};

// ---- q3_K : { uint8 hmask[32]; uint8 qs[64]; uint8 scales[12]; half d; } — 110 bytes, 256/block.
//   low 2 bits in qs, high bit in hmask; 16 6-bit signed scales packed (kmask). value = d*(sc-32)*q3. ----
struct q3_K {
    constant static constexpr const int block_k = 256, block_bytes = 110;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* hmask = base; device const uchar* qs = base + 32; device const uchar* sca = base + 96;
        const half d = ((device const half*)(base + 108))[0];
        const int chunk = col >> 7, pos = col & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l = pos & 15;
        const int is = chunk * 8 + sidx * 2 + sub;
        const int low2 = (qs[chunk * 32 + sub * 16 + l] >> (2 * sidx)) & 3;
        const int hb = (hmask[sub * 16 + l] & (1 << (chunk * 4 + sidx))) ? 1 : 0;
        const int q3v = (low2 | (hb << 2)) - 4;
        const int w = is >> 2, b = is & 3; int s;
        if (w == 0)      s = (sca[b] & 0xF)        | ((sca[8 + b] & 3) << 4);
        else if (w == 1) s = (sca[4 + b] & 0xF)    | (((sca[8 + b] >> 2) & 3) << 4);
        else if (w == 2) s = ((sca[b] >> 4) & 0xF) | (((sca[8 + b] >> 4) & 3) << 4);
        else             s = ((sca[4 + b] >> 4) & 0xF) | (((sca[8 + b] >> 6) & 3) << 4);
        return d * half(s - 32) * half(q3v);
    }
};

// ---- q5_K : { half d; half dmin; uint8 scales[12]; uint8 qh[32]; uint8 qs[128]; } — 176 bytes.
//   8 sub-blocks of 32; 6-bit scale+min (get_scale_min_k4, as q4_K); 5-bit q = nibble | (qh bit)<<4. ----
struct q5_K {
    constant static constexpr const int block_k = 256, block_bytes = 176;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        const half d = ((device const half*)base)[0], dmin = ((device const half*)(base + 2))[0];
        device const uchar* sca = base + 4; device const uchar* qh = base + 16; device const uchar* qs = base + 48;
        const int chunk = col >> 6, pos = col & 63, sub = pos >> 5, l = pos & 31;
        const int is = 2 * chunk + sub;
        const int nib = sub ? (qs[chunk * 32 + l] >> 4) : (qs[chunk * 32 + l] & 0xF);
        const int hb = (qh[l] & (1 << (2 * chunk + sub))) ? 1 : 0;
        const int q = nib + hb * 16;
        int sc, mn;
        if (is < 4) { sc = sca[is] & 63; mn = sca[is + 4] & 63; }
        else { sc = (sca[is + 4] & 0xF) | ((sca[is - 4] >> 6) << 4); mn = (sca[is + 4] >> 4) | ((sca[is] >> 6) << 4); }
        return d * half(sc) * half(q) - dmin * half(mn);
    }
};

// ---- q6_K : { uint8 ql[128]; uint8 qh[64]; int8 scales[16]; half d; } — 210 bytes, 256/block.
//   16 sub-blocks of 16; 6-bit q = (4 low in ql | 2 high in qh) - 32; int8 scales. value = d*sc*q. ----
struct q6_K {
    constant static constexpr const int block_k = 256, block_bytes = 210;
    static METAL_FUNC half dequant(device const uchar* base, int col) {
        device const uchar* ql = base; device const uchar* qh = base + 128;
        device const char* sca = (device const char*)(base + 192);
        const half d = ((device const half*)(base + 208))[0];
        const int chunk = col >> 7, pos = col & 127, group = pos >> 5, l = pos & 31;
        const int ql_byte = ql[chunk * 64 + l + 32 * (group & 1)];
        const int nib = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0xF);
        const int hbits = (qh[chunk * 32 + l] >> (2 * group)) & 3;
        const int q = (nib | (hbits << 4)) - 32;
        const int sc_idx = chunk * 8 + (l >> 4) + group * 2;
        return d * half((int)sca[sc_idx]) * half(q);
    }
};

// ================================ span decode (8 contiguous cols) ===============================
// w[0..7] = dequant(base, col0 .. col0+7), with col0 % 8 == 0. The generic version just loops —
// the Metal compiler CSEs the simple scale reads. Formats with a branchy sub-block scale unpack
// (k-quants, grouped int4, fp4 nibbles) specialize so the unpack runs ONCE per span instead of
// per element; every span of 8 stays inside one sub-block/nibble-half for all supported layouts,
// so the extraction mode is uniform across the span.
template<typename FMT>
METAL_FUNC void tk_dequant8(device const uchar* base, int col0, thread half* w) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = FMT::dequant(base, col0 + i);
}

// FP32 companion used by kernels whose public contract requires dequantization
// and all following scale/epilogue arithmetic to happen before the output dtype
// is rounded.  The generic implementation widens the format's native decoder;
// formats below specialize the path to avoid an intermediate half rounding.
// Keep the final cast at the consuming kernel's store site.
template<typename FMT>
METAL_FUNC void tk_dequant8_f32(device const uchar* base, int col0,
                                thread float* w) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = float(FMT::dequant(base, col0 + i));
}

template<>
METAL_FUNC void tk_dequant8_f32<q4_0>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half*)base)[0]);
    device const uchar* qs = base + 2;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int col = col0 + i;
        const int nib = col < 16 ? (qs[col] & 0x0F) : (qs[col - 16] >> 4);
        w[i] = d * float(nib - 8);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<q8_0>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half*)base)[0]);
    device const char* qs = (device const char*)(base + 2);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = d * float(qs[col0 + i]);
}

template<>
METAL_FUNC void tk_dequant8_f32<q6_K>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    device const uchar* ql = base;
    device const uchar* qh = base + 128;
    device const char* sca = (device const char*)(base + 192);
    const float d = float(((device const half*)(base + 208))[0]);
    const int chunk = col0 >> 7;
    const int pos = col0 & 127;
    const int group = pos >> 5;
    const int lane0 = pos & 31;
    const float dsc = d * float(sca[chunk * 8 + (lane0 >> 4) + group * 2]);
    device const uchar* q = ql + chunk * 64 + lane0 + 32 * (group & 1);
    device const uchar* h = qh + chunk * 32 + lane0;
    const int hshift = 2 * group;
    const bool hi = (group & 2) != 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int nib = hi ? (q[i] >> 4) : (q[i] & 0x0F);
        const int qv = (nib | (((h[i] >> hshift) & 3) << 4)) - 32;
        w[i] = dsc * float(qv);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<q4_K>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half*)base)[0]);
    const float dmin = float(((device const half*)base)[1]);
    device const uchar* scales = base + 4;
    device const uchar* qs = base + 16;
    const int chunk = col0 >> 6, pos = col0 & 63;
    const bool hi = pos >= 32;
    const int sub = chunk * 2 + (hi ? 1 : 0);
    int sc, mn;
    if (sub < 4) {
        sc = scales[sub] & 63;
        mn = scales[sub + 4] & 63;
    } else {
        sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
        mn = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
    }
    const float dl = d * float(sc), ml = dmin * float(mn);
    device const uchar* q = qs + chunk * 32 + (hi ? pos - 32 : pos);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = dl * float(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - ml;
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<q5_K>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half*)base)[0]);
    const float dmin = float(((device const half*)(base + 2))[0]);
    device const uchar* scales = base + 4;
    device const uchar* qh = base + 16;
    device const uchar* qs = base + 48;
    const int chunk = col0 >> 6, pos = col0 & 63;
    const int sub = pos >> 5, lane0 = pos & 31;
    const int scale_index = 2 * chunk + sub;
    int sc, mn;
    if (scale_index < 4) {
        sc = scales[scale_index] & 63;
        mn = scales[scale_index + 4] & 63;
    } else {
        sc = (scales[scale_index + 4] & 0x0F) |
            ((scales[scale_index - 4] >> 6) << 4);
        mn = (scales[scale_index + 4] >> 4) |
            ((scales[scale_index] >> 6) << 4);
    }
    const float dl = d * float(sc), ml = dmin * float(mn);
    const uchar high_mask = uchar(1u << scale_index);
    device const uchar* q = qs + chunk * 32 + lane0;
    device const uchar* h = qh + lane0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int q5 = (sub ? (q[i] >> 4) : (q[i] & 0x0F)) +
            ((h[i] & high_mask) ? 16 : 0);
        w[i] = dl * float(q5) - ml;
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<q2_K>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    device const uchar* scales = base;
    device const uchar* qs = base + 16;
    const float d = float(((device const half*)(base + 80))[0]);
    const float dmin = float(((device const half*)(base + 82))[0]);
    const int chunk = col0 >> 7, pos = col0 & 127;
    const int scale_index = pos >> 5, sub = (pos >> 4) & 1, lane0 = pos & 15;
    const uchar scale_byte = scales[chunk * 8 + scale_index * 2 + sub];
    const float dl = d * float(scale_byte & 0x0F);
    const float ml = dmin * float(scale_byte >> 4);
    device const uchar* q = qs + chunk * 32 + sub * 16 + lane0;
    const int shift = 2 * scale_index;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = dl * float((q[i] >> shift) & 3) - ml;
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<q3_K>(device const uchar* base, int col0,
                                     thread float* w) {
    #pragma clang fp reassociate(off)
    device const uchar* high_mask = base;
    device const uchar* qs = base + 32;
    device const uchar* scales = base + 96;
    const float d = float(((device const half*)(base + 108))[0]);
    const int chunk = col0 >> 7, pos = col0 & 127;
    const int scale_index = pos >> 5, sub = (pos >> 4) & 1, lane0 = pos & 15;
    const int index = chunk * 8 + scale_index * 2 + sub;
    const int word = index >> 2, byte = index & 3;
    int scale;
    if (word == 0) {
        scale = (scales[byte] & 0xF) | ((scales[8 + byte] & 3) << 4);
    } else if (word == 1) {
        scale = (scales[4 + byte] & 0xF) |
            (((scales[8 + byte] >> 2) & 3) << 4);
    } else if (word == 2) {
        scale = ((scales[byte] >> 4) & 0xF) |
            (((scales[8 + byte] >> 4) & 3) << 4);
    } else {
        scale = ((scales[4 + byte] >> 4) & 0xF) |
            (((scales[8 + byte] >> 6) & 3) << 4);
    }
    const float dsc = d * float(scale - 32);
    device const uchar* q = qs + chunk * 32 + sub * 16 + lane0;
    device const uchar* h = high_mask + sub * 16 + lane0;
    const int shift = 2 * scale_index;
    const uchar high_bit = uchar(1u << (chunk * 4 + scale_index));
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int q3 = (((q[i] >> shift) & 3) |
            (((h[i] & high_bit) ? 1 : 0) << 2)) - 4;
        w[i] = dsc * float(q3);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<iq4_xs>(device const uchar* base, int col0,
                                       thread float* w) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half*)base)[0]);
    const ushort scales_h = ((device const ushort*)(base + 2))[0];
    device const uchar* scales_l = base + 4;
    device const uchar* qs = base + 8;
    const int block = col0 >> 5, local = col0 & 31;
    const int low = (scales_l[block >> 1] >> (4 * (block & 1))) & 0x0F;
    const int high = (scales_h >> (2 * block)) & 0x3;
    const float dl = d * float((low | (high << 4)) - 32);
    const bool hi = local >= 16;
    device const uchar* q = qs + 16 * block + (hi ? local - 16 : local);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = dl * float(kvalues_iq4nl[hi ? (q[i] >> 4) : (q[i] & 0x0F)]);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<iq4_nl>(device const uchar* base, int col0,
                                       thread float* w) {
    #pragma clang fp reassociate(off)
    const float d = float(((device const half*)base)[0]);
    const bool hi = col0 >= 16;
    device const uchar* q = base + 2 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = d * float(kvalues_iq4nl[hi ? (q[i] >> 4) : (q[i] & 0x0F)]);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<kU4B8>(device const uchar* base, int col0,
                                      thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = float(((device const half*)base)[0]);
    const bool hi = col0 >= 64;
    device const uchar* q = base + 2 + (hi ? col0 - 64 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = scale * float(int(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - 8);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<kU4>(device const uchar* base, int col0,
                                    thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = float(((device const half*)base)[0]);
    const float zero = float(((device const half*)base)[1]);
    const bool hi = col0 >= 64;
    device const uchar* q = base + 4 + (hi ? col0 - 64 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = scale * (float(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - zero);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<hqq>(device const uchar* base, int col0,
                                    thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = float(((device const half*)base)[0]);
    const float zero = float(((device const half*)base)[1]);
    const bool hi = col0 >= 32;
    device const uchar* q = base + 4 + (hi ? col0 - 32 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = scale * (float(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - zero);
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<fp8_e4m3>(device const uchar* base, int col0,
                                         thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = float(((device const half*)base)[0]);
    device const uchar* q = base + 2 + col0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = scale * float(tk_e4m3_decode(q[i]));
}

template<>
METAL_FUNC void tk_dequant8_f32<mxfp8>(device const uchar* base, int col0,
                                      thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = tk_e8m0_decode_f32(base[0]);
    device const uchar* q = base + 1 + col0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) w[i] = scale * float(tk_e4m3_decode(q[i]));
}

template<>
METAL_FUNC void tk_dequant8_f32<nvfp4>(device const uchar* base, int col0,
                                      thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = float(tk_e4m3_decode(base[0]));
    const bool hi = col0 >= 8;
    device const uchar* q = base + 1 + (hi ? col0 - 8 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = scale * float(tk_e2m1_decode(hi ? uint(q[i] >> 4) : uint(q[i] & 0x0F)));
    }
}

template<>
METAL_FUNC void tk_dequant8_f32<mxfp4>(device const uchar* base, int col0,
                                      thread float* w) {
    #pragma clang fp reassociate(off)
    const float scale = tk_e8m0_decode_f32(base[0]);
    const bool hi = col0 >= 16;
    device const uchar* q = base + 1 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        w[i] = scale * float(tk_e2m1_decode(hi ? uint(q[i] >> 4) : uint(q[i] & 0x0F)));
    }
}

template<>
METAL_FUNC void tk_dequant8<q4_K>(device const uchar* base, int col0, thread half* w) {
    const half d    = ((device const half*)base)[0];
    const half dmin = ((device const half*)base)[1];
    device const uchar* scales = base + 4;
    device const uchar* qs     = base + 16;
    const int chunk = col0 >> 6, pos = col0 & 63;
    const bool hi   = pos >= 32;
    const int sub   = chunk * 2 + (hi ? 1 : 0);
    uchar sc, m;
    if (sub < 4) { sc = scales[sub] & 63; m = scales[sub + 4] & 63; }
    else {
        sc = (scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4);
        m  = (scales[sub + 4] >> 4)   | ((scales[sub]     >> 6) << 4);
    }
    const half dl = d * half(sc), ml = dmin * half(m);
    device const uchar* q = qs + chunk * 32 + (hi ? pos - 32 : pos);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = dl * half(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - ml;
}

template<>
METAL_FUNC void tk_dequant8<q5_K>(device const uchar* base, int col0, thread half* w) {
    const half d = ((device const half*)base)[0], dmin = ((device const half*)(base + 2))[0];
    device const uchar* sca = base + 4;
    device const uchar* qh  = base + 16;
    device const uchar* qs  = base + 48;
    const int chunk = col0 >> 6, pos = col0 & 63, sub = pos >> 5, l0 = pos & 31;
    const int is = 2 * chunk + sub;
    int sc, mn;
    if (is < 4) { sc = sca[is] & 63; mn = sca[is + 4] & 63; }
    else {
        sc = (sca[is + 4] & 0x0F) | ((sca[is - 4] >> 6) << 4);
        mn = (sca[is + 4] >> 4)   | ((sca[is]     >> 6) << 4);
    }
    const half dl = d * half(sc), ml = dmin * half(mn);
    const uchar hmask = uchar(1u << is);
    device const uchar* q = qs + chunk * 32 + l0;
    device const uchar* h = qh + l0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int q5 = (sub ? (q[i] >> 4) : (q[i] & 0x0F)) + ((h[i] & hmask) ? 16 : 0);
        w[i] = dl * half(q5) - ml;
    }
}

template<>
METAL_FUNC void tk_dequant8<q6_K>(device const uchar* base, int col0, thread half* w) {
    device const uchar* ql = base;
    device const uchar* qh = base + 128;
    device const char* sca = (device const char*)(base + 192);
    const half d = ((device const half*)(base + 208))[0];
    const int chunk = col0 >> 7, pos = col0 & 127, group = pos >> 5, l0 = pos & 31;
    const half dsc = d * half((int)sca[chunk * 8 + (l0 >> 4) + group * 2]);
    device const uchar* q = ql + chunk * 64 + l0 + 32 * (group & 1);
    device const uchar* h = qh + chunk * 32 + l0;
    const int hshift = 2 * group;
    const bool hi = (group & 2) != 0;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int nib = hi ? (q[i] >> 4) : (q[i] & 0x0F);
        const int qv  = (nib | (((h[i] >> hshift) & 3) << 4)) - 32;
        w[i] = dsc * half(qv);
    }
}

template<>
METAL_FUNC void tk_dequant8<q2_K>(device const uchar* base, int col0, thread half* w) {
    device const uchar* scales = base;
    device const uchar* qs = base + 16;
    const half d = ((device const half*)(base + 80))[0], dmin = ((device const half*)(base + 82))[0];
    const int chunk = col0 >> 7, pos = col0 & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l0 = pos & 15;
    const uchar sb = scales[chunk * 8 + sidx * 2 + sub];
    const half dl = d * half(sb & 0x0F), ml = dmin * half(sb >> 4);
    device const uchar* q = qs + chunk * 32 + sub * 16 + l0;
    const int shift = 2 * sidx;
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = dl * half((q[i] >> shift) & 3) - ml;
}

template<>
METAL_FUNC void tk_dequant8<q3_K>(device const uchar* base, int col0, thread half* w) {
    device const uchar* hmask = base;
    device const uchar* qs = base + 32;
    device const uchar* sca = base + 96;
    const half d = ((device const half*)(base + 108))[0];
    const int chunk = col0 >> 7, pos = col0 & 127, sidx = pos >> 5, sub = (pos >> 4) & 1, l0 = pos & 15;
    const int is = chunk * 8 + sidx * 2 + sub;
    const int wi = is >> 2, b = is & 3;
    int s;
    if (wi == 0)      s = (sca[b] & 0xF)        | ((sca[8 + b] & 3) << 4);
    else if (wi == 1) s = (sca[4 + b] & 0xF)    | (((sca[8 + b] >> 2) & 3) << 4);
    else if (wi == 2) s = ((sca[b] >> 4) & 0xF) | (((sca[8 + b] >> 4) & 3) << 4);
    else              s = ((sca[4 + b] >> 4) & 0xF) | (((sca[8 + b] >> 6) & 3) << 4);
    const half dsc = d * half(s - 32);
    device const uchar* q = qs + chunk * 32 + sub * 16 + l0;
    device const uchar* h = hmask + sub * 16 + l0;
    const int shift = 2 * sidx;
    const uchar hbit = uchar(1u << (chunk * 4 + sidx));
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int q3v = (((q[i] >> shift) & 3) | (((h[i] & hbit) ? 1 : 0) << 2)) - 4;
        w[i] = dsc * half(q3v);
    }
}

template<>
METAL_FUNC void tk_dequant8<iq4_xs>(device const uchar* base, int col0, thread half* w) {
    const half d = ((device const half*)base)[0];
    const ushort scales_h = ((device const ushort*)(base + 2))[0];
    device const uchar* scales_l = base + 4;
    device const uchar* qs = base + 8;
    const int ib = col0 >> 5, local = col0 & 31;
    const int sl = (scales_l[ib >> 1] >> (4 * (ib & 1))) & 0x0F;
    const int sh = (scales_h >> (2 * ib)) & 0x3;
    const half dl = d * half((sl | (sh << 4)) - 32);
    const bool hi = local >= 16;
    device const uchar* q = qs + 16 * ib + (hi ? local - 16 : local);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = dl * half(kvalues_iq4nl[hi ? (q[i] >> 4) : (q[i] & 0x0F)]);
}

template<>
METAL_FUNC void tk_dequant8<iq4_nl>(device const uchar* base, int col0, thread half* w) {
    const half d = ((device const half*)base)[0];
    const bool hi = col0 >= 16;
    device const uchar* q = base + 2 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = d * half(kvalues_iq4nl[hi ? (q[i] >> 4) : (q[i] & 0x0F)]);
}

template<>
METAL_FUNC void tk_dequant8<kU4B8>(device const uchar* base, int col0, thread half* w) {
    const half s = ((device const half*)base)[0];
    const bool hi = col0 >= 64;
    device const uchar* q = base + 2 + (hi ? col0 - 64 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = s * half((hi ? (q[i] >> 4) : (q[i] & 0x0F)) - 8);
}

template<>
METAL_FUNC void tk_dequant8<kU4>(device const uchar* base, int col0, thread half* w) {
    const half s = ((device const half*)base)[0], zp = ((device const half*)base)[1];
    const bool hi = col0 >= 64;
    device const uchar* q = base + 4 + (hi ? col0 - 64 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = s * (half(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - zp);
}

template<>
METAL_FUNC void tk_dequant8<hqq>(device const uchar* base, int col0, thread half* w) {
    const half s = ((device const half*)base)[0], zp = ((device const half*)base)[1];
    const bool hi = col0 >= 32;
    device const uchar* q = base + 4 + (hi ? col0 - 32 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = s * (half(hi ? (q[i] >> 4) : (q[i] & 0x0F)) - zp);
}

template<>
METAL_FUNC void tk_dequant8<nvfp4>(device const uchar* base, int col0, thread half* w) {
    const half s = tk_e4m3_decode(base[0]);
    const bool hi = col0 >= 8;
    device const uchar* q = base + 1 + (hi ? col0 - 8 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = s * tk_e2m1_decode(hi ? uint(q[i] >> 4) : uint(q[i] & 0x0F));
}

template<>
METAL_FUNC void tk_dequant8<mxfp4>(device const uchar* base, int col0, thread half* w) {
    const half s = metal::exp2(half((int)base[0] - 127));
    const bool hi = col0 >= 16;
    device const uchar* q = base + 1 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = s * tk_e2m1_decode(hi ? uint(q[i] >> 4) : uint(q[i] & 0x0F));
}

template<>
METAL_FUNC void tk_dequant8<q4_1>(device const uchar* base, int col0, thread half* w) {
    const half d = ((device const half*)base)[0], m = ((device const half*)base)[1];
    const bool hi = col0 >= 16;
    device const uchar* q = base + 4 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i)
        w[i] = d * half(hi ? (q[i] >> 4) : (q[i] & 0x0F)) + m;
}

template<>
METAL_FUNC void tk_dequant8<q5_0>(device const uchar* base, int col0, thread half* w) {
    const half d = ((device const half*)base)[0];
    const uint qh = (uint)base[2] | ((uint)base[3] << 8) | ((uint)base[4] << 16) | ((uint)base[5] << 24);
    const bool hi = col0 >= 16;
    device const uchar* q = base + 6 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int qv = (hi ? (q[i] >> 4) : (q[i] & 0x0F)) | (((qh >> (col0 + i)) & 1) << 4);
        w[i] = d * half(qv - 16);
    }
}

template<>
METAL_FUNC void tk_dequant8<q5_1>(device const uchar* base, int col0, thread half* w) {
    const half d = ((device const half*)base)[0], m = ((device const half*)base)[1];
    const uint qh = (uint)base[4] | ((uint)base[5] << 8) | ((uint)base[6] << 16) | ((uint)base[7] << 24);
    const bool hi = col0 >= 16;
    device const uchar* q = base + 8 + (hi ? col0 - 16 : col0);
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int qv = (hi ? (q[i] >> 4) : (q[i] & 0x0F)) | (((qh >> (col0 + i)) & 1) << 4);
        w[i] = d * half(qv) + m;
    }
}

// Cooperatively dequantize an (BN x BK) weight tile into a shared half tile. `kb` is the K-tile
// index in units of BK (the MMA K-step). The quant grouping (FMT::block_k) is DECOUPLED from BK:
// each tile column maps to its quant block via the global K index, so large blocks (e.g. q4_K's
// 256) work with a small BK. Requires FMT::block_k % BK == 0 and K % FMT::block_k == 0.
//   Packed layout: block(n, b) starts at byte (n*(K/FMT::block_k) + b) * FMT::block_bytes.
//   `group_threads` = total threads in the threadgroup; `threadIdx` = flat thread index.
template<typename FMT, int BN, int BK>
METAL_FUNC void dequant_into_shared(threadgroup st<half, BN, BK>& dst,
                                    device const uchar* Wq, int N, int K,
                                    int by, int kb, int group_threads, uint threadIdx) {
    static_assert(BK % 8 == 0, "dequant_into_shared: BK must be a multiple of 8");
    const int blocks_per_row = K / FMT::block_k;
    // one 8-col span per step: tk_dequant8 unpacks the block/sub-block scales once per span
    // instead of once per element (a span never straddles a quant block: block_k % 8 == 0).
    constexpr int SPANS_PER_ROW = BK / 8;
    for (int s = (int)threadIdx; s < BN * SPANS_PER_ROW; s += group_threads) {
        const int row  = s / SPANS_PER_ROW;
        const int tcol = (s % SPANS_PER_ROW) * 8;
        const int grow = by * BN + row;
        const int gk   = kb * BK + tcol;                 // global K column
        const int blk  = gk / FMT::block_k;              // which quant block
        const int cib  = gk % FMT::block_k;              // column within that block
        device const uchar* base = Wq + (uint)(grow * blocks_per_row + blk) * FMT::block_bytes;
        half w[8];
        tk_dequant8<FMT>(base, cib, w);
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) dst[int2(row, tcol + i)] = w[i];
    }
}

// A row-layout fragment lane owns adjacent pairs at columns
// {c,c+1,c+8,c+9,c+16,c+17,c+24,c+25}. Most formats use the scalar fallback below, but
// NVFP4 maps the span onto two 16-value blocks. Keep the helper format-gated so another
// layout cannot silently inherit that mapping.
template<typename FMT>
struct tk_dequant_row8_s8 {
    constant static constexpr const bool enabled = false;
    static METAL_FUNC void run(device const uchar* b0, device const uchar* b1,
                               int c, thread half* w) {
        w[0] = FMT::dequant(b0, c);      w[1] = FMT::dequant(b0, c + 1);
        w[2] = FMT::dequant(b0, c + 8);  w[3] = FMT::dequant(b0, c + 9);
        w[4] = FMT::dequant(b1, c);      w[5] = FMT::dequant(b1, c + 1);
        w[6] = FMT::dequant(b1, c + 8);  w[7] = FMT::dequant(b1, c + 9);
    }
};

template<>
struct tk_dequant_row8_s8<nvfp4> {
    constant static constexpr const bool enabled = true;
    static METAL_FUNC void run(device const uchar* b0, device const uchar* b1,
                               int c, thread half* w) {
        const half s0 = tk_e4m3_decode(b0[0]);
        const half s1 = tk_e4m3_decode(b1[0]);
        const uchar q00 = b0[1 + c], q01 = b0[2 + c];
        const uchar q10 = b1[1 + c], q11 = b1[2 + c];
        w[0] = s0 * tk_e2m1_decode(q00 & 0x0F);
        w[1] = s0 * tk_e2m1_decode(q01 & 0x0F);
        w[2] = s0 * tk_e2m1_decode(q00 >> 4);
        w[3] = s0 * tk_e2m1_decode(q01 >> 4);
        w[4] = s1 * tk_e2m1_decode(q10 & 0x0F);
        w[5] = s1 * tk_e2m1_decode(q11 & 0x0F);
        w[6] = s1 * tk_e2m1_decode(q10 >> 4);
        w[7] = s1 * tk_e2m1_decode(q11 >> 4);
    }
};

// Dequantize an (RT::rows x RT::cols) weight tile DIRECTLY into the simdgroup register fragment —
// no threadgroup round-trip, no barrier (Marlin's "zero-shuffle" idea on Apple). Each lane fills
// only its own 2 elements per 8x8 subtile, using the substrate's lane->(row,col) fragment map
// (mirrors load(rt, gl) in global_to_register.metal): thread_elements()[0]/[1] = the weights at
// (row = by*rows + i*8 + simd_y, col = kb*cols + j*8 + simd_x [+1]). `by`/`kb` are tile-block indices.
template<typename FMT, typename RT>
METAL_FUNC void dequant_into_register(thread RT& dst, device const uchar* Wq, int N, int K,
                                      int by, int kb, uint laneid) {
    const int qid    = (int)laneid / 4;
    const int simd_y = (qid & 4) + ((int)laneid / 2) % 4;
    const int simd_x = (qid & 2) * 2 + ((int)laneid % 2) * 2;
    const int bpr = K / FMT::block_k;
    if (tk_dequant_row8_s8<FMT>::enabled && RT::cols == 32 && RT::width == 4) {
        const int gc0 = kb * RT::cols + simd_x;
        const int blk0 = gc0 / FMT::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < RT::height; i++) {
            const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y;
            device const uchar* b0 = Wq + (uint)(grow * bpr + blk0) * FMT::block_bytes;
            device const uchar* b1 = b0 + FMT::block_bytes;
            half w[8];
            tk_dequant_row8_s8<FMT>::run(b0, b1, simd_x, w);
            #pragma clang loop unroll(full)
            for (int j = 0; j < RT::width; j++) {
                dst.tiles[i][j].data.thread_elements()[0] =
                    (typename RT::dtype)(float)w[2 * j];
                dst.tiles[i][j].data.thread_elements()[1] =
                    (typename RT::dtype)(float)w[2 * j + 1];
            }
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < RT::height; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < RT::width; j++) {
            const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y;
            const int gc   = kb * RT::cols + j * mittens::TILE_DIM + simd_x;
            const int blk0 = gc / FMT::block_k,       cib0 = gc % FMT::block_k;
            const int blk1 = (gc + 1) / FMT::block_k, cib1 = (gc + 1) % FMT::block_k;
            device const uchar* b0 = Wq + (uint)(grow * bpr + blk0) * FMT::block_bytes;
            device const uchar* b1 = Wq + (uint)(grow * bpr + blk1) * FMT::block_bytes;
            // cast half->RT::dtype (RT may be bf16, e.g. quantized-KV attention's V tile)
            dst.tiles[i][j].data.thread_elements()[0] = (typename RT::dtype)(float)FMT::dequant(b0, cib0);
            dst.tiles[i][j].data.thread_elements()[1] = (typename RT::dtype)(float)FMT::dequant(b1, cib1);
        }
    }
}

// Decode the 4 columns {c, c+8, c+16, c+24} of ONE quant block with a single scale unpack.
// This is the column pattern of the col-layout fragment fill below (one lane's 4 width-subtile
// slots at a fixed row): when FMT::block_k >= 32 all four land in the same block, because the
// K tile base is 32-aligned and c <= 6. Specializations amortize the per-block scale decode
// (fatal for the e8m0 formats, whose scale is an exp2) 4x over the naive per-element path.
template<typename FMT>
struct tk_dequant_cols4_s8 {
    static METAL_FUNC void run(device const uchar* base, int c, thread half* w) {
        w[0] = FMT::dequant(base, c);
        w[1] = FMT::dequant(base, c + 8);
        w[2] = FMT::dequant(base, c + 16);
        w[3] = FMT::dequant(base, c + 24);
    }
};
template<> struct tk_dequant_cols4_s8<q8_0> {
    static METAL_FUNC void run(device const uchar* base, int c, thread half* w) {
        const half d = ((device const half*)base)[0];
        device const char* qs = (device const char*)(base + 2);
        w[0] = d * half(qs[c]);      w[1] = d * half(qs[c + 8]);
        w[2] = d * half(qs[c + 16]); w[3] = d * half(qs[c + 24]);
    }
};
template<> struct tk_dequant_cols4_s8<fp8_e4m3> {
    static METAL_FUNC void run(device const uchar* base, int c, thread half* w) {
        const half d = ((device const half*)base)[0];
        device const uchar* qs = base + 2;
        w[0] = d * tk_e4m3_decode(qs[c]);      w[1] = d * tk_e4m3_decode(qs[c + 8]);
        w[2] = d * tk_e4m3_decode(qs[c + 16]); w[3] = d * tk_e4m3_decode(qs[c + 24]);
    }
};
template<> struct tk_dequant_cols4_s8<mxfp8> {
    static METAL_FUNC void run(device const uchar* base, int c, thread half* w) {
        const half d = metal::exp2(half((int)base[0] - 127));   // one exp2 for all 4
        device const uchar* qs = base + 1;
        w[0] = d * tk_e4m3_decode(qs[c]);      w[1] = d * tk_e4m3_decode(qs[c + 8]);
        w[2] = d * tk_e4m3_decode(qs[c + 16]); w[3] = d * tk_e4m3_decode(qs[c + 24]);
    }
};
template<> struct tk_dequant_cols4_s8<mxfp4> {
    static METAL_FUNC void run(device const uchar* base, int c, thread half* w) {
        // c <= 6, so cols {c, c+8} are low nibbles and {c+16, c+24} the high nibbles of the
        // SAME two bytes qs[c], qs[c+8] — two byte loads + one exp2 cover all four weights.
        // This span already amortizes scale expansion 4x. The native half exp2
        // remains faster than float-bit reconstruction plus half conversion at
        // the priority 32-row MoE decode shape.
        const half d = metal::exp2(half((int)base[0] - 127));
        const uchar b0 = (base + 1)[c], b1 = (base + 1)[c + 8];
        w[0] = d * tk_e2m1_decode(b0 & 0x0F); w[1] = d * tk_e2m1_decode(b1 & 0x0F);
        w[2] = d * tk_e2m1_decode(b0 >> 4);   w[3] = d * tk_e2m1_decode(b1 >> 4);
    }
};
template<> struct tk_dequant_cols4_s8<kU4> {
    static METAL_FUNC void run(device const uchar* base, int c, thread half* w) {
        // group 128: a 32-aligned window of width <= 30 never crosses the lo/hi nibble split
        // at col 64, so all four columns share one shift.
        const half scale = ((device const half*)base)[0];
        const half zp    = ((device const half*)base)[1];
        device const uchar* qs = base + 4;
        const int cm = (c < 64) ? c : c - 64;
        const int sh = (c < 64) ? 0 : 4;
        w[0] = scale * (half((qs[cm]      >> sh) & 0x0F) - zp);
        w[1] = scale * (half((qs[cm + 8]  >> sh) & 0x0F) - zp);
        w[2] = scale * (half((qs[cm + 16] >> sh) & 0x0F) - zp);
        w[3] = scale * (half((qs[cm + 24] >> sh) & 0x0F) - zp);
    }
};

// Two-block companion for 16-value formats.  In the col-layout fragment map, nvfp4 columns
// {c,c+8} are the low/high nibbles of byte c in the first block and {c+16,c+24} are the
// corresponding byte in the second block.  Two scale decodes and two byte loads cover the span.
template<typename FMT>
struct tk_dequant_cols4_s8x2 {
    constant static constexpr const bool enabled = false;
    static METAL_FUNC void run(device const uchar* b0, device const uchar* b1,
                               int c, thread half* w) {
        w[0] = FMT::dequant(b0, c);      w[1] = FMT::dequant(b0, c + 8);
        w[2] = FMT::dequant(b1, c);      w[3] = FMT::dequant(b1, c + 8);
    }
};

template<>
struct tk_dequant_cols4_s8x2<nvfp4> {
    constant static constexpr const bool enabled = true;
    static METAL_FUNC void run(device const uchar* b0, device const uchar* b1,
                               int c, thread half* w) {
        const half s0 = tk_e4m3_decode(b0[0]);
        const half s1 = tk_e4m3_decode(b1[0]);
        const uchar q0 = b0[1 + c], q1 = b1[1 + c];
        w[0] = s0 * tk_e2m1_decode(q0 & 0x0F);
        w[1] = s0 * tk_e2m1_decode(q0 >> 4);
        w[2] = s1 * tk_e2m1_decode(q1 & 0x0F);
        w[3] = s1 * tk_e2m1_decode(q1 >> 4);
    }
};

// Pair-of-rows column decoder with simdgroup scale broadcast. In a col-layout
// fragment, eight lanes cover the same row pair and different c values across
// one 32-column block. MXFP8 therefore needs only two E8M0 expansions for the
// pair; the other lanes receive those scales by shuffle and decode their own
// E4M3 codes.
template<typename FMT>
struct tk_dequant_cols4_s8_pair_shuffle {
    constant static constexpr const bool enabled = false;
    static METAL_FUNC void run(device const uchar*, device const uchar*,
                               int, ushort, thread half*, thread half*) {}
};

template<>
struct tk_dequant_cols4_s8_pair_shuffle<mxfp8> {
    constant static constexpr const bool enabled = true;
    static METAL_FUNC void run(device const uchar* b0,
                               device const uchar* b1,
                               int c, ushort leader,
                               thread half* w0, thread half* w1) {
        half d0 = 0.0h, d1 = 0.0h;
        if (c == 0) {
            d0 = metal::exp2(half((int)b0[0] - 127));
            d1 = metal::exp2(half((int)b1[0] - 127));
        }
        d0 = metal::simd_shuffle(d0, leader);
        d1 = metal::simd_shuffle(d1, leader);
        device const uchar* q0 = b0 + 1;
        device const uchar* q1 = b1 + 1;
        w0[0] = d0 * tk_e4m3_decode(q0[c]);
        w0[1] = d0 * tk_e4m3_decode(q0[c + 8]);
        w0[2] = d0 * tk_e4m3_decode(q0[c + 16]);
        w0[3] = d0 * tk_e4m3_decode(q0[c + 24]);
        w1[0] = d1 * tk_e4m3_decode(q1[c]);
        w1[1] = d1 * tk_e4m3_decode(q1[c + 8]);
        w1[2] = d1 * tk_e4m3_decode(q1[c + 16]);
        w1[3] = d1 * tk_e4m3_decode(q1[c + 24]);
    }
};

// Col-layout companion to dequant_into_register, for feeding mma_ABt's B operand
// (rt<T, M, K, col>) straight from a row-major (N, K)-packed weight — the quantized
// A @ W^T path (grouped expert GEMMs). Mirrors the col-layout load in
// global_to_register.metal exactly: the x/y lane derivations swap, and a lane's two
// thread elements are VERTICALLY adjacent — logical (row, col) and (row+1, col) —
// i.e. two different packed rows sharing one quant-block column. `by` indexes the
// logical-row (N) tile block, `kb` the K tile block, as in the row variant.
// For block_k >= 32 the width loop collapses to one tk_dequant_cols4_s8 span per
// (subtile-row, element): the 4 width-subtile columns {sx, sx+8, sx+16, sx+24} of a
// 32-aligned K tile share a quant block, so the scale is unpacked once per span.
template<typename FMT, typename RT>
METAL_FUNC void dequant_into_register_col(thread RT& dst, device const uchar* Wq, int N, int K,
                                          int by, int kb, uint laneid) {
    const int qid    = (int)laneid / 4;
    const int simd_x = (qid & 4) + ((int)laneid / 2) % 4;
    const int simd_y = (qid & 2) * 2 + ((int)laneid % 2) * 2;
    const int bpr = K / FMT::block_k;
    if (tk_dequant_cols4_s8x2<FMT>::enabled && RT::cols == 32 && RT::width == 4) {
        const int gc0 = kb * RT::cols + simd_x;
        const int blk0 = gc0 / FMT::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < RT::height; i++) {
            #pragma clang loop unroll(full)
            for (int el = 0; el < 2; el++) {
                const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y + el;
                device const uchar* b0 = Wq + (uint)(grow * bpr + blk0) * FMT::block_bytes;
                device const uchar* b1 = b0 + FMT::block_bytes;
                half w[4];
                tk_dequant_cols4_s8x2<FMT>::run(b0, b1, simd_x, w);
                #pragma clang loop unroll(full)
                for (int j = 0; j < RT::width; j++)
                    dst.tiles[i][j].data.thread_elements()[el] =
                        (typename RT::dtype)(float)w[j];
            }
        }
        return;
    }
    if (FMT::block_k >= 32 && RT::width == 4) {   // constexpr-foldable fast path
        const int gc0  = kb * RT::cols + simd_x;
        const int blk  = gc0 / FMT::block_k;
        const int cib0 = gc0 % FMT::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < RT::height; i++) {
            #pragma clang loop unroll(full)
            for (int el = 0; el < 2; el++) {
                const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y + el;
                device const uchar* base = Wq + (uint)(grow * bpr + blk) * FMT::block_bytes;
                half w[4];
                tk_dequant_cols4_s8<FMT>::run(base, cib0, w);
                #pragma clang loop unroll(full)
                for (int j = 0; j < RT::width; j++)
                    dst.tiles[i][j].data.thread_elements()[el] = (typename RT::dtype)(float)w[j];
            }
        }
        return;
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < RT::height; i++) {
        #pragma clang loop unroll(full)
        for (int j = 0; j < RT::width; j++) {
            const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y;
            const int gc   = kb * RT::cols + j * mittens::TILE_DIM + simd_x;
            const int blk = gc / FMT::block_k, cib = gc % FMT::block_k;
            device const uchar* b0 = Wq + (uint)(grow * bpr + blk) * FMT::block_bytes;
            device const uchar* b1 = Wq + (uint)((grow + 1) * bpr + blk) * FMT::block_bytes;
            dst.tiles[i][j].data.thread_elements()[0] = (typename RT::dtype)(float)FMT::dequant(b0, cib);
            dst.tiles[i][j].data.thread_elements()[1] = (typename RT::dtype)(float)FMT::dequant(b1, cib);
        }
    }
}

// SwiGLU-specific column decoder route. The paired scale-broadcast strategy
// wins when gate and up double the decoder work, but regresses the rectangular
// large-row projection; keep it behind a distinct call site.
template<typename FMT, typename RT>
METAL_FUNC void dequant_into_register_col_swiglu(
    thread RT& dst, device const uchar* Wq, int N, int K,
    int by, int kb, uint laneid) {
    if (tk_dequant_cols4_s8_pair_shuffle<FMT>::enabled &&
        FMT::block_k >= 32 && RT::width == 4) {
        const int qid = (int)laneid / 4;
        const int simd_x = (qid & 4) + ((int)laneid / 2) % 4;
        const int simd_y = (qid & 2) * 2 + ((int)laneid % 2) * 2;
        const int bpr = K / FMT::block_k;
        const int gc0 = kb * RT::cols + simd_x;
        const int blk = gc0 / FMT::block_k;
        const int cib0 = gc0 % FMT::block_k;
        const ushort leader = ushort(((simd_y & 4) ? 8 : 0) +
                                     ((simd_y & 2) ? 1 : 0));
        #pragma clang loop unroll(full)
        for (int i = 0; i < RT::height; i++) {
            const int grow = by * RT::rows + i * mittens::TILE_DIM + simd_y;
            device const uchar* b0 =
                Wq + (uint)(grow * bpr + blk) * FMT::block_bytes;
            device const uchar* b1 = b0 + (uint)bpr * FMT::block_bytes;
            half w0[4], w1[4];
            tk_dequant_cols4_s8_pair_shuffle<FMT>::run(
                b0, b1, cib0, leader, w0, w1);
            #pragma clang loop unroll(full)
            for (int j = 0; j < RT::width; j++) {
                dst.tiles[i][j].data.thread_elements()[0] =
                    (typename RT::dtype)(float)w0[j];
                dst.tiles[i][j].data.thread_elements()[1] =
                    (typename RT::dtype)(float)w1[j];
            }
        }
        return;
    }
    dequant_into_register_col<FMT>(dst, Wq, N, K, by, kb, laneid);
}

} // namespace mittens
