#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Fused LM-head + sampling: pick a decode token WITHOUT materializing the (T, V)
// logits. Two stages (the paged-attn-v2 partition/reduce shape):
//   lm_head_*_partials : grid (num_vtiles, T). One simdgroup owns a TILE_V slice
//     of the vocab for token t. Each lane owns the tile's vocab rows
//     v = base + lane + 32*r and computes the full dot logit = <W[v,:], h[t,:]>
//     serially (no per-vocab reduction), applies invtemp (+ bias), and for
//     stochastic modes adds Gumbel noise indexed by the GLOBAL vocab id v so the
//     fused draw equals the unfused sampler's. h[t,:] is read from global — one
//     small K-vector reused across the tile, served from cache.
//   lm_head_*_reduce : grid (T,). Combine the per-tile partials into the final id.
//
// argmax and categorical share the kernels (a runtime use_gumbel flag); top-k has
// its own pair (k partials per tile, Gumbel-max among the merged candidates).
// TILE_V must be a multiple of 32. W is (V, K) row-major, dtype T (fp16/bf16/f32).
// ---------------------------------------------------------------------------

constant float LMH_NEG_INF = -3.4028234663852886e38f;
constant int LMH_MAX_K = 64;

// Sequential fused-sampler rows benefit from decoding an entire q4_0 block at
// once. Keep the legacy half-rounded weight contract here: the scale/code
// product is rounded to half before the fp32 dot accumulation, exactly as in
// q4_0::dequant and tk_dequant8<q4_0>.
template <typename FMT, typename T>
METAL_FUNC float lmh_sampler_quant_dot(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / FMT::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base = wrow + (long)block * FMT::block_bytes;
        for (int col0 = 0; col0 < FMT::block_k; col0 += 8) {
            half values[8];
            tk_dequant8<FMT>(base, col0, values);
            const int input_base = block * FMT::block_k + col0;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                result += float(values[i]) * float(hrow[input_base + i]);
            }
        }
    }
    return result;
}

template <typename T>
METAL_FUNC float lmh_sampler_quant_dot_q4(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / q4_0::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base = wrow + (long)block * q4_0::block_bytes;
        const half scale = ((device const half *)base)[0];
        device const uchar *codes = base + 2;
        const int input_base = block * q4_0::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            const half low = scale * half(int(packed & 0x0f) - 8);
            const half high = scale * half(int(packed >> 4) - 8);
            result += float(low) * float(hrow[input_base + i]);
            result += float(high) * float(hrow[input_base + i + 16]);
        }
    }
    return result;
}

#define specialize_lmh_sampler_q4(T)                                      \
  template <> METAL_FUNC float lmh_sampler_quant_dot<q4_0, T>(            \
    device const T *hrow, device const uchar *wrow, int K) {              \
    return lmh_sampler_quant_dot_q4<T>(hrow, wrow, K);                     \
  }

specialize_lmh_sampler_q4(float)
specialize_lmh_sampler_q4(half)
specialize_lmh_sampler_q4(bf16)

// NVFP4 has one E4M3 scale and eight packed bytes per 16-value block. Decode
// the complete block in one pass so the sequential vocab-row dot does not
// reload the same scale and byte once for each 8-value half.
template <typename T>
METAL_FUNC float lmh_sampler_quant_dot_nvfp4(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / nvfp4::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base = wrow + (long)block * nvfp4::block_bytes;
        const half scale = tk_e4m3_decode(base[0]);
        device const uchar *codes = base + 1;
        const int input_base = block * nvfp4::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = codes[i];
            const half low = scale * tk_e2m1_decode(uint(packed & 0x0f));
            const half high = scale * tk_e2m1_decode(uint(packed >> 4));
            result += float(low) * float(hrow[input_base + i]);
            result += float(high) * float(hrow[input_base + i + 8]);
        }
    }
    return result;
}

#define specialize_lmh_sampler_nvfp4(T)                                  \
  template <> METAL_FUNC float lmh_sampler_quant_dot<nvfp4, T>(           \
    device const T *hrow, device const uchar *wrow, int K) {              \
    return lmh_sampler_quant_dot_nvfp4<T>(hrow, wrow, K);                  \
  }

specialize_lmh_sampler_nvfp4(float)
specialize_lmh_sampler_nvfp4(half)
specialize_lmh_sampler_nvfp4(bf16)

// MXFP4's 32 values share one E8M0 scale. Decode all sixteen packed
// bytes together so the sampler expands that scale once per block. The
// scale/code product remains half-rounded to match mxfp4::dequant.
template <typename T>
METAL_FUNC float lmh_sampler_quant_dot_mxfp4(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / mxfp4::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base =
            wrow + (long)block * mxfp4::block_bytes;
        const half scale = tk_e8m0_decode(base[0]);
        device const uchar *codes = base + 1;
        const int input_base = block * mxfp4::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            const half low = scale * tk_e2m1_decode(uint(packed & 0x0f));
            const half high = scale * tk_e2m1_decode(uint(packed >> 4));
            result += float(low) * float(hrow[input_base + i]);
            result += float(high) * float(hrow[input_base + i + 16]);
        }
    }
    return result;
}

#define specialize_lmh_sampler_mxfp4(T)                                  \
  template <> METAL_FUNC float lmh_sampler_quant_dot<mxfp4, T>(           \
    device const T *hrow, device const uchar *wrow, int K) {              \
    return lmh_sampler_quant_dot_mxfp4<T>(hrow, wrow, K);                  \
  }

specialize_lmh_sampler_mxfp4(float)
specialize_lmh_sampler_mxfp4(half)
specialize_lmh_sampler_mxfp4(bf16)

// emit functor for the Family-B masked_topk_local merge in the top-k/top-p partials kernels: writes
// each round's winner into the per-tile (part_val, part_id) partials on lane 0 (-1 id for an empty
// round). Shared by lm_head_topk_partials / lm_head_topk_partials_q / lm_head_topp_partials_q.
struct lmh_part_emit {
    device float *part_val;
    device int   *part_id;
    long pbase;
    uint lane;
    METAL_FUNC void operator()(int kk, float gbest, int gid) {
        if (lane == 0) {
            part_val[pbase + kk] = gbest;
            part_id[pbase + kk]  = (gbest == LMH_NEG_INF) ? -1 : gid;
        }
    }
};

// ---- argmax / categorical ----
// Grid (num_vtiles, num_tok), one simdgroup per (vocab tile, token) — max parallelism (the GEMV is
// compute-bound here, so parallelism beats W-reuse). Each lane owns the tile's vocab rows
// v = v0 + lane + 32*r and computes <W[v], h[t]> with VEC4 loads (4x fewer load instructions than
// the old scalar dot), then a cross-lane argmax. K % 4 == 0 -> the vec4 path; else a scalar tail.
template <typename T>
kernel void lm_head_argcat_partials(device const T     *h          [[buffer(0)]],   // (num_tok, K)
                                    device const T     *W          [[buffer(1)]],   // (V, K)
                                    device float       *part_val   [[buffer(2)]],   // (num_tok, num_vtiles)
                                    device int         *part_id    [[buffer(3)]],   // (num_tok, num_vtiles)
                                    device const float *bias       [[buffer(4)]],   // (V,) or dummy
                                    constant int   &V          [[buffer(5)]],
                                    constant int   &K          [[buffer(6)]],
                                    constant int   &TILE_V     [[buffer(7)]],
                                    constant int   &num_vtiles [[buffer(8)]],
                                    constant float &invtemp    [[buffer(9)]],
                                    constant uint  &seed       [[buffer(10)]],
                                    constant int   &use_gumbel [[buffer(11)]],
                                    constant int   &use_bias   [[buffer(12)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int nk4 = (K % 4 == 0) ? K / 4 : 0;               // vec4 rows must be 4-element aligned
    device const T *hrow = h + (long)t * K;
    device const metal::vec<T, 4> *h4 = (device const metal::vec<T, 4>*)hrow;

    float best = LMH_NEG_INF;
    int   bi   = (v0 + (int)lane < v1) ? v0 + (int)lane : v0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const T *wrow = W + (long)v * K;
        float dot_acc = 0.0f;
        device const metal::vec<T, 4> *w4 = (device const metal::vec<T, 4>*)wrow;
        for (int j = 0; j < nk4; ++j) dot_acc += dot(float4(w4[j]), float4(h4[j]));
        for (int i = nk4 * 4; i < K; ++i) dot_acc += float(wrow[i]) * float(hrow[i]);
        float ls = dot_acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, (uint)t, (uint)v);
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        part_val[(long)t * num_vtiles + vtile] = best;
        part_id[(long)t * num_vtiles + vtile]  = bi;
    }
}

kernel void lm_head_argcat_reduce(device const float *part_val   [[buffer(0)]],
                                  device const int   *part_id    [[buffer(1)]],
                                  device int         *out_idx    [[buffer(2)]],
                                  constant int &num_vtiles       [[buffer(3)]],
                                  uint  t    [[threadgroup_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
    const long base = (long)t * num_vtiles;
    float best = LMH_NEG_INF;
    int   bi   = 0x7fffffff;
    for (int j = (int)lane; j < num_vtiles; j += 32) {
        const float v = part_val[base + j];
        const int   id = part_id[base + j];
        if (v > best || (v == best && id < bi)) { best = v; bi = id; }
    }
    simd_argmax(best, bi);
    if (lane == 0) out_idx[t] = bi;
}

// ---- argmax / categorical over QUANTIZED weights (q8_0 / q4_0) ----
// Same geometry as lm_head_argcat_partials, but W is a packed (V, K/block_k, block_bytes) uchar
// tensor dequantized on the fly (tk_dequant8<FMT> unpacks 8 columns/span). h dtype = T (f16/bf16/f32).
template <typename FMT, typename T>
kernel void lm_head_argcat_partials_q(device const T     *h          [[buffer(0)]],  // (num_tok, K)
                                      device const uchar *Wq         [[buffer(1)]],  // packed (V,K)
                                      device float       *part_val   [[buffer(2)]],
                                      device int         *part_id    [[buffer(3)]],
                                      device const float *bias       [[buffer(4)]],
                                      constant int   &V          [[buffer(5)]],
                                      constant int   &K          [[buffer(6)]],
                                      constant int   &TILE_V     [[buffer(7)]],
                                      constant int   &num_vtiles [[buffer(8)]],
                                      constant float &invtemp    [[buffer(9)]],
                                      constant uint  &seed       [[buffer(10)]],
                                      constant int   &use_gumbel [[buffer(11)]],
                                      constant int   &use_bias   [[buffer(12)]],
                                      uint2 tgid [[threadgroup_position_in_grid]],
                                      uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;                       // quant blocks per weight row

    float best = LMH_NEG_INF;
    int   bi   = (v0 + (int)lane < v1) ? v0 + (int)lane : v0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const uchar *wq_row = Wq + (long)v * bpr * FMT::block_bytes;
        const float dot_acc = lmh_sampler_quant_dot<FMT>(hrow, wq_row, K);
        float ls = dot_acc * invtemp;
        if (use_bias) ls += bias[v];
        if (use_gumbel) ls += rng_gumbel(seed, (uint)t, (uint)v);
        if (ls > best || (ls == best && v < bi)) { best = ls; bi = v; }
    }
    simd_argmax(best, bi);
    if (lane == 0) {
        part_val[(long)t * num_vtiles + vtile] = best;
        part_id[(long)t * num_vtiles + vtile]  = bi;
    }
}

#define instantiate_lm_head_q(hn, FMT, T)                                          \
  template [[host_name("lm_head_argcat_partials_q_" hn)]] [[kernel]] void          \
  lm_head_argcat_partials_q<FMT, T>(device const T *h [[buffer(0)]],               \
    device const uchar *Wq [[buffer(1)]], device float *part_val [[buffer(2)]],    \
    device int *part_id [[buffer(3)]], device const float *bias [[buffer(4)]],     \
    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],                  \
    constant int &TILE_V [[buffer(7)]], constant int &num_vtiles [[buffer(8)]],    \
    constant float &invtemp [[buffer(9)]], constant uint &seed [[buffer(10)]],     \
    constant int &use_gumbel [[buffer(11)]], constant int &use_bias [[buffer(12)]], \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_q("q8_0_float32", q8_0, float)
instantiate_lm_head_q("q8_0_float16", q8_0, half)
instantiate_lm_head_q("q8_0_bfloat16", q8_0, bf16)
instantiate_lm_head_q("q4_0_float32", q4_0, float)
instantiate_lm_head_q("q4_0_float16", q4_0, half)
instantiate_lm_head_q("q4_0_bfloat16", q4_0, bf16)
instantiate_lm_head_q("mxfp8_float32", mxfp8, float)
instantiate_lm_head_q("mxfp8_float16", mxfp8, half)
instantiate_lm_head_q("mxfp8_bfloat16", mxfp8, bf16)
instantiate_lm_head_q("nvfp4_float32", nvfp4, float)
instantiate_lm_head_q("nvfp4_float16", nvfp4, half)
instantiate_lm_head_q("nvfp4_bfloat16", nvfp4, bf16)
instantiate_lm_head_q("mxfp4_float32", mxfp4, float)
instantiate_lm_head_q("mxfp4_float16", mxfp4, half)
instantiate_lm_head_q("mxfp4_bfloat16", mxfp4, bf16)

// Q6_K f32 specialization. Unlike the generic quantized LM head, this keeps
// the exact GGUF dequant product in fp32 instead of first
// rounding each weight through half. It shares the standard argcat ABI and
// reduce kernel, so batch rows, bias, temperature, and deterministic Gumbel
// sampling follow the normal QuixiCore contract.
METAL_FUNC float lmh_f16_bits_to_f32(ushort value) {
    const uint sign = uint(value & 0x8000) << 16;
    const uint exponent = (value >> 10) & 0x1f;
    const uint mantissa = value & 0x03ff;
    if (exponent == 0) {
        const float magnitude = ldexp(float(mantissa), -24);
        return sign != 0 ? -magnitude : magnitude;
    }
    if (exponent == 31) {
        return as_type<float>(sign | 0x7f800000 | (mantissa << 13));
    }
    return as_type<float>(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

METAL_FUNC float lmh_q6_k_dot(device const uchar *row_weights,
                              device const float *input,
                              int blocks_per_row) {
    float sum = 0.0f;
    for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
        device const uchar *block = row_weights + (long)block_index * 210;
        device const uchar *ql = block;
        device const uchar *qh = block + 128;
        device const char *scales = (device const char *)(block + 192);
        const ushort d_bits = ushort(block[208]) | (ushort(block[209]) << 8);
        const float d = lmh_f16_bits_to_f32(d_bits);
        for (int chunk = 0; chunk < 2; ++chunk) {
            for (int group = 0; group < 4; ++group) {
                for (int item = 0; item < 32; ++item) {
                    const uchar ql_byte = ql[chunk * 64 + item + 32 * (group & 1)];
                    const uint nibble = (group & 2) ? (ql_byte >> 4) : (ql_byte & 0x0f);
                    const uint high = (qh[chunk * 32 + item] >> (2 * group)) & 3;
                    const int quant = int(nibble | (high << 4)) - 32;
                    const int scale_index = chunk * 8 + (item >> 4) + group * 2;
                    const int column =
                        block_index * 256 + chunk * 128 + group * 32 + item;
                    sum += d * float(int(scales[scale_index])) * float(quant) * input[column];
                }
            }
        }
    }
    return sum;
}

[[host_name("lm_head_argcat_partials_q_q6_K_float32")]]
kernel void lm_head_argcat_partials_q_q6_K_float32(
    device const float *h [[buffer(0)]],
    device const uchar *Wq [[buffer(1)]],
    device float *part_val [[buffer(2)]],
    device int *part_id [[buffer(3)]],
    device const float *bias [[buffer(4)]],
    constant int &V [[buffer(5)]],
    constant int &K [[buffer(6)]],
    constant int &TILE_V [[buffer(7)]],
    constant int &num_vtiles [[buffer(8)]],
    constant float &invtemp [[buffer(9)]],
    constant uint &seed [[buffer(10)]],
    constant int &use_gumbel [[buffer(11)]],
    constant int &use_bias [[buffer(12)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int vtile = int(tgid.x);
    const int token = int(tgid.y);
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int blocks_per_row = K / 256;
    const int row_bytes = blocks_per_row * 210;
    device const float *hrow = h + (long)token * K;
    float best = LMH_NEG_INF;
    int best_id = v0 + int(lane) < v1 ? v0 + int(lane) : v0;
    for (int vocab = v0 + int(lane); vocab < v1; vocab += 32) {
        float value = lmh_q6_k_dot(Wq + (long)vocab * row_bytes, hrow, blocks_per_row);
        value *= invtemp;
        if (use_bias) value += bias[vocab];
        if (use_gumbel) value += rng_gumbel(seed, uint(token), uint(vocab));
        if (value > best || (value == best && vocab < best_id)) {
            best = value;
            best_id = vocab;
        }
    }
    simd_argmax(best, best_id);
    if (lane == 0) {
        const long offset = (long)token * num_vtiles + vtile;
        part_val[offset] = best;
        part_id[offset] = best_id;
    }
}

// ---- top-k ----
template <typename T>
kernel void lm_head_topk_partials(device const T     *h          [[buffer(0)]],
                                  device const T     *W          [[buffer(1)]],
                                  device float       *part_val   [[buffer(2)]],   // (num_tok, num_vtiles, K)
                                  device int         *part_id    [[buffer(3)]],
                                  device const float *bias       [[buffer(4)]],
                                  constant int   &V          [[buffer(5)]],
                                  constant int   &K          [[buffer(6)]],
                                  constant int   &TILE_V     [[buffer(7)]],
                                  constant int   &num_vtiles [[buffer(8)]],
                                  constant int   &topk       [[buffer(9)]],
                                  constant int   &use_bias   [[buffer(10)]],
                                  uint2 tgid [[threadgroup_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;

    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int nk4 = (K % 4 == 0) ? K / 4 : 0;
    device const metal::vec<T, 4> *h4 = (device const metal::vec<T, 4>*)hrow;
    constexpr int MAX_PER_LANE = 2048 / 32;   // TILE_V <= 2048
    float mine_val[MAX_PER_LANE];
    int   mine_id[MAX_PER_LANE];
    bool  used[MAX_PER_LANE];
    int   nmine = 0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const T *wrow = W + (long)v * K;
        device const metal::vec<T, 4> *w4 = (device const metal::vec<T, 4>*)wrow;
        float dot_acc = 0.0f;
        for (int j = 0; j < nk4; ++j) dot_acc += dot(float4(w4[j]), float4(h4[j]));
        for (int i = nk4 * 4; i < K; ++i) dot_acc += float(wrow[i]) * float(hrow[i]);
        float ls = dot_acc;
        if (use_bias) ls += bias[v];
        mine_val[nmine] = ls;
        mine_id[nmine]  = v;
        ++nmine;
    }
    const long pbase = ((long)t * num_vtiles + vtile) * topk;
    lmh_part_emit emit{part_val, part_id, pbase, lane};   // Family-B merge -> per-tile top-k partials
    masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF, emit);
}

kernel void lm_head_topk_reduce(device const float *part_val   [[buffer(0)]],   // (num_tok, num_vtiles, K)
                                device const int   *part_id    [[buffer(1)]],
                                device int         *out_idx    [[buffer(2)]],
                                constant int &num_vtiles       [[buffer(3)]],
                                constant int &topk             [[buffer(4)]],
                                constant uint &seed            [[buffer(5)]],
                                constant float &invtemp        [[buffer(6)]],
                                uint  t    [[threadgroup_position_in_grid]],
                                uint  lane [[thread_index_in_simdgroup]]) {
    const int ncand = num_vtiles * topk;
    const long base = (long)t * ncand;
    int   chosen_id[LMH_MAX_K];
    float chosen_val[LMH_MAX_K];
    // K masked-argmax rounds over the merged per-tile partial winners (Family-A helper).
    stored_cand cand{part_val, part_id, base};
    masked_topk(cand, ncand, topk, lane, LMH_NEG_INF, chosen_id, chosen_val);
    // Gumbel-max among the k winners (global vocab id in the noise stream).
    float best = LMH_NEG_INF;
    int   bi   = chosen_id[0];
    for (int kk = 0; kk < topk; ++kk) {
        if (chosen_id[kk] < 0) continue;
        const float g = rng_gumbel(seed, (uint)t, (uint)chosen_id[kk]);
        const float p = chosen_val[kk] * invtemp + g;
        if (p > best || (p == best && chosen_id[kk] < bi)) { best = p; bi = chosen_id[kk]; }
    }
    if (lane == 0) out_idx[t] = bi;
}

// Top-p (nucleus) reduce over the merged over-selected candidate pool (num_vtiles*topk per token).
// The nucleus tokens are selected from the pool (the top-k' contains them for the peaked LM-head
// distribution), but the cumulative mass is measured against the TRUE full-vocab normalizer Z built
// from the per-tile logsumexps (part_lse) emitted by lm_head_topp_partials_q — so the nucleus cutoff
// is exact, not the pool-only approximation. Mirrors top_p_sample: bisect the tempered-logit
// threshold L for cumulative mass >= p, then Gumbel-max over {ls >= L} (noise indexed by global id).
kernel void lm_head_topp_reduce(device const float *part_val [[buffer(0)]],   // (T, num_vtiles, topk)
                                device const int   *part_id  [[buffer(1)]],
                                device int         *out_idx  [[buffer(2)]],
                                constant int   &num_vtiles   [[buffer(3)]],
                                constant int   &topk         [[buffer(4)]],
                                constant float &p            [[buffer(5)]],
                                constant uint  &seed         [[buffer(6)]],
                                constant float &invtemp      [[buffer(7)]],
                                device const float *part_lse [[buffer(8)]],   // (T, num_vtiles)
                                uint  t    [[threadgroup_position_in_grid]],
                                uint  lane [[thread_index_in_simdgroup]]) {
    const int ncand = num_vtiles * topk;
    const long base = (long)t * ncand;
    float mx = LMH_NEG_INF;
    for (int j = (int)lane; j < ncand; j += 32) {
        if (part_id[base + j] >= 0) { mx = max(mx, part_val[base + j] * invtemp); }
    }
    mx = simd_max(mx);
    // true full-vocab Z from the per-tile tempered logsumexps: Z = sum_tiles exp(part_lse - mx)
    const long lbase = (long)t * num_vtiles;
    float Z = 0.0f;
    for (int vt = (int)lane; vt < num_vtiles; vt += 32) {
        const float pl = part_lse[lbase + vt];
        if (pl > LMH_NEG_INF) { Z += exp(pl - mx); }
    }
    Z = simd_sum(Z);
    float lo = mx - 40.0f, hi = mx;             // bisect L: largest L with mass{ls>=L} >= p
    for (int it = 0; it < 32; ++it) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int j = (int)lane; j < ncand; j += 32) {
            const float ls = part_val[base + j] * invtemp;
            if (part_id[base + j] >= 0 && ls >= mid) { sm += exp(ls - mx); }
        }
        sm = simd_sum(sm) / Z;
        if (sm >= p) { lo = mid; } else { hi = mid; }
    }
    const float L = lo;
    float best = LMH_NEG_INF;
    int   bi   = -1;
    for (int j = (int)lane; j < ncand; j += 32) {
        const int id = part_id[base + j];
        const float ls = part_val[base + j] * invtemp;
        if (id < 0 || ls < L) { continue; }
        const float pert = ls + rng_gumbel(seed, (uint)t, (uint)id);
        if (pert > best || (pert == best && id < bi)) { best = pert; bi = id; }
    }
    float gbest = best;
    int   gid   = (bi < 0) ? 0x7fffffff : bi;
    simd_argmax(gbest, gid);
    if (lane == 0) { out_idx[t] = (gbest == LMH_NEG_INF) ? -1 : gid; }
}

// Merge quantized LM-head tile candidates into exact per-beam log-probability
// candidates. Each tile supplied its full logsumexp and top-2BM tokens, so the
// merged pool contains the row's true top-2BM while the normalizer covers V.
kernel void lm_head_beam_reduce(
    device const float *part_val   [[buffer(0)]], // (BR,num_vtiles,2BM)
    device const int   *part_id    [[buffer(1)]],
    device const float *part_lse   [[buffer(2)]], // (BR,num_vtiles)
    device const float *cum        [[buffer(3)]], // (BR,)
    device float       *cand_score [[buffer(4)]], // (BR,2BM)
    device int         *cand_token [[buffer(5)]],
    constant int &num_vtiles       [[buffer(6)]],
    constant int &two_bm           [[buffer(7)]],
    uint row  [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int ncand = num_vtiles * two_bm;
    const long base = (long)row * ncand;
    int chosen_id[LMH_MAX_K];
    float chosen_val[LMH_MAX_K];
    stored_cand candidates{part_val, part_id, base};
    masked_topk(candidates, ncand, two_bm, lane, LMH_NEG_INF,
                chosen_id, chosen_val);

    const long lbase = (long)row * num_vtiles;
    float lmax = LMH_NEG_INF;
    for (int tile = (int)lane; tile < num_vtiles; tile += 32) {
        lmax = max(lmax, part_lse[lbase + tile]);
    }
    const float gmax = simd_max(lmax);
    float lsum = 0.0f;
    for (int tile = (int)lane; tile < num_vtiles; tile += 32) {
        const float tile_lse = part_lse[lbase + tile];
        if (tile_lse > LMH_NEG_INF) lsum += exp(tile_lse - gmax);
    }
    const float lse = gmax + log(simd_sum(lsum));
    if (lane == 0) {
        const long obase = (long)row * two_bm;
        const float cumr = cum[row];
        for (int k = 0; k < two_bm; ++k) {
            cand_token[obase + k] = chosen_id[k];
            cand_score[obase + k] = cumr + chosen_val[k] - lse;
        }
    }
}

// Quantized top-k partials: same per-lane top-k as lm_head_topk_partials, but W is packed
// (V, K/block_k, block_bytes) uchar dequantized on the fly via tk_dequant8<FMT>. Feeds the SAME
// lm_head_topk_reduce (global merge + Gumbel-max), so the fused quant top-k path avoids ever
// materializing the (T,V) logits — the point of quant decode.
template <typename FMT, typename T>
kernel void lm_head_topk_partials_q(device const T     *h          [[buffer(0)]],
                                    device const uchar *Wq         [[buffer(1)]],  // packed (V,K)
                                    device float       *part_val   [[buffer(2)]],
                                    device int         *part_id    [[buffer(3)]],
                                    device const float *bias       [[buffer(4)]],
                                    constant int   &V          [[buffer(5)]],
                                    constant int   &K          [[buffer(6)]],
                                    constant int   &TILE_V     [[buffer(7)]],
                                    constant int   &num_vtiles [[buffer(8)]],
                                    constant int   &topk       [[buffer(9)]],
                                    constant int   &use_bias   [[buffer(10)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y;
    device const T *hrow = h + (long)t * K;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    constexpr int MAX_PER_LANE = 2048 / 32;   // TILE_V <= 2048
    float mine_val[MAX_PER_LANE];
    int   mine_id[MAX_PER_LANE];
    bool  used[MAX_PER_LANE];
    int   nmine = 0;
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const uchar *wq_row = Wq + (long)v * bpr * FMT::block_bytes;
        const float dot_acc = lmh_sampler_quant_dot<FMT>(hrow, wq_row, K);
        float ls = dot_acc;
        if (use_bias) ls += bias[v];
        mine_val[nmine] = ls;
        mine_id[nmine]  = v;
        ++nmine;
    }
    const long pbase = ((long)t * num_vtiles + vtile) * topk;
    lmh_part_emit emit{part_val, part_id, pbase, lane};   // Family-B merge -> per-tile top-k partials
    masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF, emit);
}

#define instantiate_lm_head_topk_q(hn, FMT, T)                                     \
  template [[host_name("lm_head_topk_partials_q_" hn)]] [[kernel]] void             \
  lm_head_topk_partials_q<FMT, T>(device const T *h [[buffer(0)]],                  \
    device const uchar *Wq [[buffer(1)]], device float *part_val [[buffer(2)]],     \
    device int *part_id [[buffer(3)]], device const float *bias [[buffer(4)]],      \
    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],                   \
    constant int &TILE_V [[buffer(7)]], constant int &num_vtiles [[buffer(8)]],     \
    constant int &topk [[buffer(9)]], constant int &use_bias [[buffer(10)]],        \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_topk_q("q8_0_float32", q8_0, float)
instantiate_lm_head_topk_q("q8_0_float16", q8_0, half)
instantiate_lm_head_topk_q("q8_0_bfloat16", q8_0, bf16)
instantiate_lm_head_topk_q("q4_0_float32", q4_0, float)
instantiate_lm_head_topk_q("q4_0_float16", q4_0, half)
instantiate_lm_head_topk_q("q4_0_bfloat16", q4_0, bf16)
instantiate_lm_head_topk_q("mxfp8_float32", mxfp8, float)
instantiate_lm_head_topk_q("mxfp8_float16", mxfp8, half)
instantiate_lm_head_topk_q("mxfp8_bfloat16", mxfp8, bf16)
instantiate_lm_head_topk_q("nvfp4_float32", nvfp4, float)
instantiate_lm_head_topk_q("nvfp4_float16", nvfp4, half)
instantiate_lm_head_topk_q("nvfp4_bfloat16", nvfp4, bf16)
instantiate_lm_head_topk_q("mxfp4_float32", mxfp4, float)
instantiate_lm_head_topk_q("mxfp4_float16", mxfp4, half)
instantiate_lm_head_topk_q("mxfp4_bfloat16", mxfp4, bf16)

// Quantized top-p partials: identical top-k selection to lm_head_topk_partials_q, but ALSO emits a
// per-tile tempered log-sum-exp (part_lse) over EVERY dequantized logit in the tile (not just the
// top-k). The reduce merges these tile-lses into the true full-vocab normalizer Z, so the nucleus
// threshold is exact (not the pool-only approximation of the shared top-k partials). Needs invtemp
// because the softmax normalizer is temperature-dependent.
template <typename FMT, typename T>
kernel void lm_head_topp_partials_q(device const T     *h          [[buffer(0)]],
                                    device const uchar *Wq         [[buffer(1)]],  // packed (V,K)
                                    device float       *part_val   [[buffer(2)]],
                                    device int         *part_id    [[buffer(3)]],
                                    device const float *bias       [[buffer(4)]],
                                    constant int   &V          [[buffer(5)]],
                                    constant int   &K          [[buffer(6)]],
                                    constant int   &TILE_V     [[buffer(7)]],
                                    constant int   &num_vtiles [[buffer(8)]],
                                    constant int   &topk       [[buffer(9)]],
                                    constant int   &use_bias   [[buffer(10)]],
                                    constant float &invtemp    [[buffer(11)]],
                                    device float       *part_lse   [[buffer(12)]],  // (T, num_vtiles)
                                    constant int   &rows_per_tg [[buffer(13)]],
                                    constant int   &total_rows  [[buffer(14)]],
                                    uint2 tgid [[threadgroup_position_in_grid]],
                                    uint  simdgroup [[simdgroup_index_in_threadgroup]],
                                    uint  lane [[thread_index_in_simdgroup]]) {
    const int vtile = (int)tgid.x;
    const int t     = (int)tgid.y * rows_per_tg + (int)simdgroup;
    if (t >= total_rows) return;
    device const T *hrow = h + (long)t * K;
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int bpr = K / FMT::block_k;
    constexpr int MAX_PER_LANE = 2048 / 32;   // TILE_V <= 2048
    float mine_val[MAX_PER_LANE];
    int   mine_id[MAX_PER_LANE];
    bool  used[MAX_PER_LANE];
    int   nmine = 0;
    float lmax = LMH_NEG_INF, lsum = 0.0f;    // streaming tempered logsumexp over this lane's logits
    for (int v = v0 + (int)lane; v < v1; v += 32) {
        device const uchar *wq_row = Wq + (long)v * bpr * FMT::block_bytes;
        const float dot_acc = lmh_sampler_quant_dot<FMT>(hrow, wq_row, K);
        float ls = dot_acc;
        if (use_bias) ls += bias[v];
        const float tl = ls * invtemp;                    // tempered logit
        if (tl > lmax) { lsum = lsum * exp(lmax - tl) + 1.0f; lmax = tl; }
        else           { lsum += exp(tl - lmax); }
        mine_val[nmine] = ls;
        mine_id[nmine]  = v;
        ++nmine;
    }
    // per-tile lse = logsumexp over all lanes' tempered logits
    const float gmax = simd_max(lmax);
    float gsum = simd_sum((lmax == LMH_NEG_INF) ? 0.0f : lsum * exp(lmax - gmax));
    if (lane == 0) {
        part_lse[(long)t * num_vtiles + vtile] = (gsum > 0.0f) ? (gmax + log(gsum)) : LMH_NEG_INF;
    }
    const long pbase = ((long)t * num_vtiles + vtile) * topk;
    lmh_part_emit emit{part_val, part_id, pbase, lane};   // Family-B merge -> per-tile top-k partials
    masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF, emit);
}

#define instantiate_lm_head_topp_q(hn, FMT, T)                                     \
  template [[host_name("lm_head_topp_partials_q_" hn)]] [[kernel]] void             \
  lm_head_topp_partials_q<FMT, T>(device const T *h [[buffer(0)]],                  \
    device const uchar *Wq [[buffer(1)]], device float *part_val [[buffer(2)]],     \
    device int *part_id [[buffer(3)]], device const float *bias [[buffer(4)]],      \
    constant int &V [[buffer(5)]], constant int &K [[buffer(6)]],                   \
    constant int &TILE_V [[buffer(7)]], constant int &num_vtiles [[buffer(8)]],     \
    constant int &topk [[buffer(9)]], constant int &use_bias [[buffer(10)]],        \
    constant float &invtemp [[buffer(11)]], device float *part_lse [[buffer(12)]],  \
    constant int &rows_per_tg [[buffer(13)]], constant int &total_rows [[buffer(14)]], \
    uint2 tgid [[threadgroup_position_in_grid]],                                    \
    uint simdgroup [[simdgroup_index_in_threadgroup]],                              \
    uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_topp_q("q8_0_float32", q8_0, float)
instantiate_lm_head_topp_q("q8_0_float16", q8_0, half)
instantiate_lm_head_topp_q("q8_0_bfloat16", q8_0, bf16)
instantiate_lm_head_topp_q("q4_0_float32", q4_0, float)
instantiate_lm_head_topp_q("q4_0_float16", q4_0, half)
instantiate_lm_head_topp_q("q4_0_bfloat16", q4_0, bf16)
instantiate_lm_head_topp_q("mxfp8_float32", mxfp8, float)
instantiate_lm_head_topp_q("mxfp8_float16", mxfp8, half)
instantiate_lm_head_topp_q("mxfp8_bfloat16", mxfp8, bf16)
instantiate_lm_head_topp_q("nvfp4_float32", nvfp4, float)
instantiate_lm_head_topp_q("nvfp4_float16", nvfp4, half)
instantiate_lm_head_topp_q("nvfp4_bfloat16", nvfp4, bf16)
instantiate_lm_head_topp_q("mxfp4_float32", mxfp4, float)
instantiate_lm_head_topp_q("mxfp4_float16", mxfp4, half)
instantiate_lm_head_topp_q("mxfp4_bfloat16", mxfp4, bf16)

#define instantiate_lm_head(type_name, T)                                          \
  template [[host_name("lm_head_argcat_partials_" #type_name)]] [[kernel]] void     \
  lm_head_argcat_partials<T>(device const T *h [[buffer(0)]], device const T *W [[buffer(1)]], \
    device float *part_val [[buffer(2)]], device int *part_id [[buffer(3)]],        \
    device const float *bias [[buffer(4)]], constant int &V [[buffer(5)]],          \
    constant int &K [[buffer(6)]], constant int &TILE_V [[buffer(7)]],              \
    constant int &num_vtiles [[buffer(8)]], constant float &invtemp [[buffer(9)]],  \
    constant uint &seed [[buffer(10)]], constant int &use_gumbel [[buffer(11)]],    \
    constant int &use_bias [[buffer(12)]],                                          \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("lm_head_topk_partials_" #type_name)]] [[kernel]] void        \
  lm_head_topk_partials<T>(device const T *h [[buffer(0)]], device const T *W [[buffer(1)]], \
    device float *part_val [[buffer(2)]], device int *part_id [[buffer(3)]],        \
    device const float *bias [[buffer(4)]], constant int &V [[buffer(5)]],          \
    constant int &K [[buffer(6)]], constant int &TILE_V [[buffer(7)]],              \
    constant int &num_vtiles [[buffer(8)]], constant int &topk [[buffer(9)]],       \
    constant int &use_bias [[buffer(10)]],                                          \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head(float32, float)
instantiate_lm_head(float16, half)
instantiate_lm_head(bfloat16, bf16)

// ---- Constrained greedy LM head ----
// Fuses dense projection, grammar-mask lookup, greedy selection, and the
// selected-token log-probability. The normalizer intentionally includes masked
// tokens because the reference applies its grammar mask after log_softmax.
template <typename T>
kernel void lm_head_constrained_partials(
    device const T *h [[buffer(0)]],
    device const T *W [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const uchar *forbidden [[buffer(3)]],
    device const int *previous [[buffer(4)]],
    device float *part_max [[buffer(5)]],
    device float *part_sum [[buffer(6)]],
    device float *part_best [[buffer(7)]],
    device int *part_id [[buffer(8)]],
    constant int &V [[buffer(9)]],
    constant int &K [[buffer(10)]],
    constant int &TILE_V [[buffer(11)]],
    constant int &num_vtiles [[buffer(12)]],
    constant int &use_bias [[buffer(13)]],
    constant int &eos_id [[buffer(14)]],
    constant int &forbid_eos [[buffer(15)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int vtile = int(tgid.x);
    const int token = int(tgid.y);
    const int v0 = vtile * TILE_V;
    const int v1 = min(v0 + TILE_V, V);
    const int nk4 = K % 4 == 0 ? K / 4 : 0;
    device const T *hrow = h + (long)token * K;
    device const metal::vec<T, 4> *h4 = (device const metal::vec<T, 4> *)hrow;
    float mine[8];
    int nmine = 0;
    float local_max = LMH_NEG_INF;
    float valid_best = LMH_NEG_INF;
    int valid_id = 0x7fffffff;
    const int prev = previous[token];
    for (int vocab = v0 + int(lane); vocab < v1; vocab += 32) {
        device const T *wrow = W + (long)vocab * K;
        device const metal::vec<T, 4> *w4 = (device const metal::vec<T, 4> *)wrow;
        float dot_acc = 0.0f;
        for (int j = 0; j < nk4; ++j) {
            dot_acc += dot(float4(w4[j]), float4(h4[j]));
        }
        for (int j = nk4 * 4; j < K; ++j) {
            dot_acc += float(wrow[j]) * float(hrow[j]);
        }
        T rounded = T(dot_acc);
        if (use_bias) rounded = rounded + T(bias[vocab]);
        const float logit = float(rounded);
        mine[nmine++] = logit;
        local_max = max(local_max, logit);
        bool allowed = false;
        if (prev >= 0 && prev < V) {
            allowed = forbidden[(long)prev * V + vocab] == 0 &&
                      !(forbid_eos && vocab == eos_id);
        }
        if (allowed &&
            (logit > valid_best || (logit == valid_best && vocab < valid_id))) {
            valid_best = logit;
            valid_id = vocab;
        }
    }
    const float tile_max = simd_max(local_max);
    float local_sum = 0.0f;
    for (int j = 0; j < nmine; ++j) local_sum += exp(mine[j] - tile_max);
    const float tile_sum = simd_sum(local_sum);
    simd_argmax(valid_best, valid_id);
    if (lane == 0) {
        const long offset = (long)token * num_vtiles + vtile;
        part_max[offset] = tile_max;
        part_sum[offset] = tile_sum;
        part_best[offset] = valid_best;
        part_id[offset] = valid_id;
    }
}

kernel void lm_head_constrained_reduce(
    device const float *part_max [[buffer(0)]],
    device const float *part_sum [[buffer(1)]],
    device const float *part_best [[buffer(2)]],
    device const int *part_id [[buffer(3)]],
    device int *out_token [[buffer(4)]],
    device float *out_logprob [[buffer(5)]],
    constant int &num_vtiles [[buffer(6)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)token * num_vtiles;
    float maximum = LMH_NEG_INF;
    float best = LMH_NEG_INF;
    int best_id = 0x7fffffff;
    for (int tile = int(lane); tile < num_vtiles; tile += 32) {
        maximum = max(maximum, part_max[base + tile]);
        const float candidate = part_best[base + tile];
        const int candidate_id = part_id[base + tile];
        if (candidate > best || (candidate == best && candidate_id < best_id)) {
            best = candidate;
            best_id = candidate_id;
        }
    }
    maximum = simd_max(maximum);
    float normalizer = 0.0f;
    for (int tile = int(lane); tile < num_vtiles; tile += 32) {
        normalizer += part_sum[base + tile] * exp(part_max[base + tile] - maximum);
    }
    normalizer = simd_sum(normalizer);
    simd_argmax(best, best_id);
    if (lane == 0) {
        const bool found = best_id != 0x7fffffff;
        out_token[token] = found ? best_id : -1;
        out_logprob[token] = found ? best - (maximum + log(normalizer)) : -INFINITY;
    }
}

#define instantiate_lm_head_constrained(type_name, T)                         \
  template [[host_name("lm_head_constrained_partials_" #type_name)]] [[kernel]] void \
  lm_head_constrained_partials<T>(device const T *h [[buffer(0)]],           \
    device const T *W [[buffer(1)]], device const float *bias [[buffer(2)]], \
    device const uchar *forbidden [[buffer(3)]], device const int *previous [[buffer(4)]], \
    device float *part_max [[buffer(5)]], device float *part_sum [[buffer(6)]], \
    device float *part_best [[buffer(7)]], device int *part_id [[buffer(8)]], \
    constant int &V [[buffer(9)]], constant int &K [[buffer(10)]],           \
    constant int &TILE_V [[buffer(11)]], constant int &num_vtiles [[buffer(12)]], \
    constant int &use_bias [[buffer(13)]], constant int &eos_id [[buffer(14)]], \
    constant int &forbid_eos [[buffer(15)]],                                 \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_lm_head_constrained(float32, float)
instantiate_lm_head_constrained(float16, half)
instantiate_lm_head_constrained(bfloat16, bf16)

// ---------------------------------------------------------------------------
// Generic masked and sparse-candidate LM heads.
// ---------------------------------------------------------------------------
constant int LMH_MASKED_MAX_K = 8;

template <typename T>
METAL_FUNC float lmh_dense_dot_f32(
    device const T *hrow, device const T *wrow, int K) {
    const int nk4 = K % 4 == 0 ? K / 4 : 0;
    device const metal::vec<T, 4> *h4 =
        (device const metal::vec<T, 4> *)hrow;
    device const metal::vec<T, 4> *w4 =
        (device const metal::vec<T, 4> *)wrow;
    float result = 0.0f;
    for (int j = 0; j < nk4; ++j) result += dot(float4(w4[j]), float4(h4[j]));
    for (int j = nk4 * 4; j < K; ++j) result += float(wrow[j]) * float(hrow[j]);
    return result;
}

template <typename FMT, typename T>
METAL_FUNC float lmh_quant_dot_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / FMT::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base = wrow + (long)block * FMT::block_bytes;
        for (int col0 = 0; col0 < FMT::block_k; col0 += 8) {
            float values[8];
            tk_dequant8_f32<FMT>(base, col0, values);
            const int input_base = block * FMT::block_k + col0;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                result += values[i] * float(hrow[input_base + i]);
            }
        }
    }
    return result;
}

// q4_0 packs both 16-value halves behind one fp16 scale. Decode the complete
// block here so the sequential LM-head row dot loads that scale and each packed
// byte once, instead of re-entering the generic 8-value decoder four times.
template <typename T>
METAL_FUNC float lmh_quant_dot_q4_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / 32;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base = wrow + (long)block * 18;
        const float scale = float(((device const half *)base)[0]);
        device const uchar *codes = base + 2;
        const int input_base = block * 32;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            result += scale * float(int(packed & 0x0f) - 8) *
                float(hrow[input_base + i]);
            result += scale * float(int(packed >> 4) - 8) *
                float(hrow[input_base + i + 16]);
        }
    }
    return result;
}

#define specialize_lmh_quant_dot_q4(T)                                      \
  template <> METAL_FUNC float lmh_quant_dot_f32<q4_0, T>(                  \
    device const T *hrow, device const uchar *wrow, int K) {                \
    return lmh_quant_dot_q4_f32<T>(hrow, wrow, K);                           \
  }

specialize_lmh_quant_dot_q4(float)
specialize_lmh_quant_dot_q4(half)
specialize_lmh_quant_dot_q4(bf16)

// FP32-dequant companion for masked and sparse-candidate projection. As in
// the sampling path, one pass consumes the scale and both nibbles in each
// packed byte while preserving the no-intermediate-half-rounding contract.
template <typename T>
METAL_FUNC float lmh_quant_dot_nvfp4_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / nvfp4::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base = wrow + (long)block * nvfp4::block_bytes;
        const float scale = float(tk_e4m3_decode(base[0]));
        device const uchar *codes = base + 1;
        const int input_base = block * nvfp4::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 8; ++i) {
            const uchar packed = codes[i];
            const float low = scale * float(tk_e2m1_decode(uint(packed & 0x0f)));
            const float high = scale * float(tk_e2m1_decode(uint(packed >> 4)));
            result += low * float(hrow[input_base + i]);
            result += high * float(hrow[input_base + i + 8]);
        }
    }
    return result;
}

#define specialize_lmh_quant_dot_nvfp4(T)                                  \
  template <> METAL_FUNC float lmh_quant_dot_f32<nvfp4, T>(                 \
    device const T *hrow, device const uchar *wrow, int K) {                \
    return lmh_quant_dot_nvfp4_f32<T>(hrow, wrow, K);                        \
  }

specialize_lmh_quant_dot_nvfp4(float)
specialize_lmh_quant_dot_nvfp4(half)
specialize_lmh_quant_dot_nvfp4(bf16)

// FP32-dequant MXFP4 companion for masked and candidate projection. One pass
// expands the E8M0 scale once and consumes both nibbles of each packed byte.
template <typename T>
METAL_FUNC float lmh_quant_dot_mxfp4_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / mxfp4::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base =
            wrow + (long)block * mxfp4::block_bytes;
        const float scale = tk_e8m0_decode_f32(base[0]);
        device const uchar *codes = base + 1;
        const int input_base = block * mxfp4::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 16; ++i) {
            const uchar packed = codes[i];
            const float low =
                scale * float(tk_e2m1_decode(uint(packed & 0x0f)));
            const float high =
                scale * float(tk_e2m1_decode(uint(packed >> 4)));
            result += low * float(hrow[input_base + i]);
            result += high * float(hrow[input_base + i + 16]);
        }
    }
    return result;
}

#define specialize_lmh_quant_dot_mxfp4(T)                                  \
  template <> METAL_FUNC float lmh_quant_dot_f32<mxfp4, T>(                 \
    device const T *hrow, device const uchar *wrow, int K) {                \
    return lmh_quant_dot_mxfp4_f32<T>(hrow, wrow, K);                        \
  }

specialize_lmh_quant_dot_mxfp4(float)
specialize_lmh_quant_dot_mxfp4(half)
specialize_lmh_quant_dot_mxfp4(bf16)

// FP32 companion for masked/candidate projection. One complete block shares
// the reconstructed E8M0 scale while retaining one-final-rounding semantics.
template <typename T>
METAL_FUNC float lmh_quant_dot_mxfp8_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / mxfp8::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base =
            wrow + (long)block * mxfp8::block_bytes;
        const float scale = tk_e8m0_decode_f32(base[0]);
        device const uchar *codes = base + 1;
        const int input_base = block * mxfp8::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < mxfp8::block_k; ++i) {
            result += scale * float(tk_e4m3_decode(codes[i])) *
                float(hrow[input_base + i]);
        }
    }
    return result;
}

#define specialize_lmh_quant_dot_mxfp8(T)                                  \
  template <> METAL_FUNC float lmh_quant_dot_f32<mxfp8, T>(                \
    device const T *hrow, device const uchar *wrow, int K) {               \
    return lmh_quant_dot_mxfp8_f32<T>(hrow, wrow, K);                       \
  }

specialize_lmh_quant_dot_mxfp8(float)
specialize_lmh_quant_dot_mxfp8(half)
specialize_lmh_quant_dot_mxfp8(bf16)

// The masked projection kernel has enough independent row work to hide a
// constant-codebook lookup and benefits from removing E4M3 bit arithmetic.
// Keep this LUT local to that path: the same strategy regresses QGEMV,
// candidate projection, and the large-row MoE SwiGLU path.
constant ushort lmh_e4m3_half_lut[256] = {
    0x0000u, 0x1800u, 0x1c00u, 0x1e00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u,
    0x2400u, 0x2480u, 0x2500u, 0x2580u, 0x2600u, 0x2680u, 0x2700u, 0x2780u,
    0x2800u, 0x2880u, 0x2900u, 0x2980u, 0x2a00u, 0x2a80u, 0x2b00u, 0x2b80u,
    0x2c00u, 0x2c80u, 0x2d00u, 0x2d80u, 0x2e00u, 0x2e80u, 0x2f00u, 0x2f80u,
    0x3000u, 0x3080u, 0x3100u, 0x3180u, 0x3200u, 0x3280u, 0x3300u, 0x3380u,
    0x3400u, 0x3480u, 0x3500u, 0x3580u, 0x3600u, 0x3680u, 0x3700u, 0x3780u,
    0x3800u, 0x3880u, 0x3900u, 0x3980u, 0x3a00u, 0x3a80u, 0x3b00u, 0x3b80u,
    0x3c00u, 0x3c80u, 0x3d00u, 0x3d80u, 0x3e00u, 0x3e80u, 0x3f00u, 0x3f80u,
    0x4000u, 0x4080u, 0x4100u, 0x4180u, 0x4200u, 0x4280u, 0x4300u, 0x4380u,
    0x4400u, 0x4480u, 0x4500u, 0x4580u, 0x4600u, 0x4680u, 0x4700u, 0x4780u,
    0x4800u, 0x4880u, 0x4900u, 0x4980u, 0x4a00u, 0x4a80u, 0x4b00u, 0x4b80u,
    0x4c00u, 0x4c80u, 0x4d00u, 0x4d80u, 0x4e00u, 0x4e80u, 0x4f00u, 0x4f80u,
    0x5000u, 0x5080u, 0x5100u, 0x5180u, 0x5200u, 0x5280u, 0x5300u, 0x5380u,
    0x5400u, 0x5480u, 0x5500u, 0x5580u, 0x5600u, 0x5680u, 0x5700u, 0x5780u,
    0x5800u, 0x5880u, 0x5900u, 0x5980u, 0x5a00u, 0x5a80u, 0x5b00u, 0x5b80u,
    0x5c00u, 0x5c80u, 0x5d00u, 0x5d80u, 0x5e00u, 0x5e80u, 0x5f00u, 0x5f80u,
    0x8000u, 0x9800u, 0x9c00u, 0x9e00u, 0xa000u, 0xa100u, 0xa200u, 0xa300u,
    0xa400u, 0xa480u, 0xa500u, 0xa580u, 0xa600u, 0xa680u, 0xa700u, 0xa780u,
    0xa800u, 0xa880u, 0xa900u, 0xa980u, 0xaa00u, 0xaa80u, 0xab00u, 0xab80u,
    0xac00u, 0xac80u, 0xad00u, 0xad80u, 0xae00u, 0xae80u, 0xaf00u, 0xaf80u,
    0xb000u, 0xb080u, 0xb100u, 0xb180u, 0xb200u, 0xb280u, 0xb300u, 0xb380u,
    0xb400u, 0xb480u, 0xb500u, 0xb580u, 0xb600u, 0xb680u, 0xb700u, 0xb780u,
    0xb800u, 0xb880u, 0xb900u, 0xb980u, 0xba00u, 0xba80u, 0xbb00u, 0xbb80u,
    0xbc00u, 0xbc80u, 0xbd00u, 0xbd80u, 0xbe00u, 0xbe80u, 0xbf00u, 0xbf80u,
    0xc000u, 0xc080u, 0xc100u, 0xc180u, 0xc200u, 0xc280u, 0xc300u, 0xc380u,
    0xc400u, 0xc480u, 0xc500u, 0xc580u, 0xc600u, 0xc680u, 0xc700u, 0xc780u,
    0xc800u, 0xc880u, 0xc900u, 0xc980u, 0xca00u, 0xca80u, 0xcb00u, 0xcb80u,
    0xcc00u, 0xcc80u, 0xcd00u, 0xcd80u, 0xce00u, 0xce80u, 0xcf00u, 0xcf80u,
    0xd000u, 0xd080u, 0xd100u, 0xd180u, 0xd200u, 0xd280u, 0xd300u, 0xd380u,
    0xd400u, 0xd480u, 0xd500u, 0xd580u, 0xd600u, 0xd680u, 0xd700u, 0xd780u,
    0xd800u, 0xd880u, 0xd900u, 0xd980u, 0xda00u, 0xda80u, 0xdb00u, 0xdb80u,
    0xdc00u, 0xdc80u, 0xdd00u, 0xdd80u, 0xde00u, 0xde80u, 0xdf00u, 0xdf80u
};

template <typename FMT, typename T>
METAL_FUNC float lmh_masked_quant_dot_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    return lmh_quant_dot_f32<FMT>(hrow, wrow, K);
}

template <typename T>
METAL_FUNC float lmh_masked_quant_dot_mxfp8_f32(
    device const T *hrow, device const uchar *wrow, int K) {
    const int blocks = K / mxfp8::block_k;
    float result = 0.0f;
    for (int block = 0; block < blocks; ++block) {
        device const uchar *base =
            wrow + (long)block * mxfp8::block_bytes;
        const float scale = tk_e8m0_decode_f32(base[0]);
        device const uchar *codes = base + 1;
        const int input_base = block * mxfp8::block_k;
        #pragma clang loop unroll(full)
        for (int i = 0; i < mxfp8::block_k; ++i) {
            result += scale *
                float(as_type<half>(lmh_e4m3_half_lut[codes[i]])) *
                float(hrow[input_base + i]);
        }
    }
    return result;
}

#define specialize_lmh_masked_quant_dot_mxfp8(T)                           \
  template <> METAL_FUNC float lmh_masked_quant_dot_f32<mxfp8, T>(         \
    device const T *hrow, device const uchar *wrow, int K) {               \
    return lmh_masked_quant_dot_mxfp8_f32<T>(hrow, wrow, K);                \
  }

specialize_lmh_masked_quant_dot_mxfp8(float)
specialize_lmh_masked_quant_dot_mxfp8(half)
specialize_lmh_masked_quant_dot_mxfp8(bf16)

METAL_FUNC void lmh_lse_update(float value, thread float &maximum,
                               thread float &sum) {
    if (value > maximum) {
        sum = maximum == LMH_NEG_INF ? 1.0f : sum * exp(maximum - value) + 1.0f;
        maximum = value;
    } else {
        sum += exp(value - maximum);
    }
}

template <typename T>
kernel void lm_head_masked_partials(
    device const T *h [[buffer(0)]],
    device const T *W [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const uint *allow_mask [[buffer(3)]],
    device float *part_val [[buffer(4)]],
    device int *part_id [[buffer(5)]],
    device float *part_max [[buffer(6)]],
    device float *part_sum [[buffer(7)]],
    constant int &V [[buffer(8)]],
    constant int &K [[buffer(9)]],
    constant int &tile_v [[buffer(10)]],
    constant int &num_vtiles [[buffer(11)]],
    constant int &topk [[buffer(12)]],
    constant int &use_bias [[buffer(13)]],
    constant int &normalize_allowed [[buffer(14)]],
    constant int &mask_words [[buffer(15)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int vtile = int(tgid.x), token = int(tgid.y);
    const int v0 = vtile * tile_v, v1 = min(v0 + tile_v, V);
    device const T *hrow = h + (long)token * K;
    float mine_val[8];
    int mine_id[8];
    bool used[8];
    int nmine = 0;
    float local_max = LMH_NEG_INF, local_sum = 0.0f;
    for (int vocab = v0 + int(lane); vocab < v1; vocab += 32) {
        const bool allowed =
            (allow_mask[(long)token * mask_words + (vocab >> 5)] &
             (1u << (vocab & 31))) != 0;
        if (normalize_allowed && !allowed) continue;
        float logit = lmh_dense_dot_f32(hrow, W + (long)vocab * K, K);
        if (use_bias != 0) logit += bias[vocab];
        if (!normalize_allowed || allowed) lmh_lse_update(logit, local_max, local_sum);
        if (allowed) {
            mine_val[nmine] = logit;
            mine_id[nmine] = vocab;
            ++nmine;
        }
    }
    const float tile_max = simd_max(local_max);
    const float tile_sum = simd_sum(
        local_max == LMH_NEG_INF ? 0.0f : local_sum * exp(local_max - tile_max));
    if (lane == 0) {
        const long tile_offset = (long)token * num_vtiles + vtile;
        part_max[tile_offset] = tile_max;
        part_sum[tile_offset] = tile_sum;
    }
    const long base = ((long)token * num_vtiles + vtile) * topk;
    lmh_part_emit emit{part_val, part_id, base, lane};
    masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF, emit);
}

template <typename FMT, typename T>
kernel void lm_head_masked_partials_q(
    device const T *h [[buffer(0)]],
    device const uchar *W [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device const uint *allow_mask [[buffer(3)]],
    device float *part_val [[buffer(4)]],
    device int *part_id [[buffer(5)]],
    device float *part_max [[buffer(6)]],
    device float *part_sum [[buffer(7)]],
    constant int &V [[buffer(8)]],
    constant int &K [[buffer(9)]],
    constant int &tile_v [[buffer(10)]],
    constant int &num_vtiles [[buffer(11)]],
    constant int &topk [[buffer(12)]],
    constant int &use_bias [[buffer(13)]],
    constant int &normalize_allowed [[buffer(14)]],
    constant int &mask_words [[buffer(15)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int vtile = int(tgid.x), token = int(tgid.y);
    const int v0 = vtile * tile_v, v1 = min(v0 + tile_v, V);
    const int row_bytes = (K / FMT::block_k) * FMT::block_bytes;
    device const T *hrow = h + (long)token * K;
    float mine_val[8];
    int mine_id[8];
    bool used[8];
    int nmine = 0;
    float local_max = LMH_NEG_INF, local_sum = 0.0f;
    for (int vocab = v0 + int(lane); vocab < v1; vocab += 32) {
        const bool allowed =
            (allow_mask[(long)token * mask_words + (vocab >> 5)] &
             (1u << (vocab & 31))) != 0;
        if (normalize_allowed && !allowed) continue;
        float logit = lmh_masked_quant_dot_f32<FMT>(
            hrow, W + (long)vocab * row_bytes, K);
        if (use_bias != 0) logit += bias[vocab];
        if (!normalize_allowed || allowed) lmh_lse_update(logit, local_max, local_sum);
        if (allowed) {
            mine_val[nmine] = logit;
            mine_id[nmine] = vocab;
            ++nmine;
        }
    }
    const float tile_max = simd_max(local_max);
    const float tile_sum = simd_sum(
        local_max == LMH_NEG_INF ? 0.0f : local_sum * exp(local_max - tile_max));
    if (lane == 0) {
        const long tile_offset = (long)token * num_vtiles + vtile;
        part_max[tile_offset] = tile_max;
        part_sum[tile_offset] = tile_sum;
    }
    const long base = ((long)token * num_vtiles + vtile) * topk;
    lmh_part_emit emit{part_val, part_id, base, lane};
    masked_topk_local(mine_val, mine_id, used, nmine, topk, LMH_NEG_INF, emit);
}

kernel void lm_head_masked_reduce(
    device const float *part_val [[buffer(0)]],
    device const int *part_id [[buffer(1)]],
    device const float *part_max [[buffer(2)]],
    device const float *part_sum [[buffer(3)]],
    device int *out_id [[buffer(4)]],
    device float *out_logprob [[buffer(5)]],
    constant int &num_vtiles [[buffer(6)]],
    constant int &topk [[buffer(7)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    const int candidates = num_vtiles * topk;
    const long candidate_base = (long)token * candidates;
    int chosen[LMH_MASKED_MAX_K];
    float chosen_val[LMH_MASKED_MAX_K];
    stored_cand candidate{part_val, part_id, candidate_base};
    masked_topk(candidate, candidates, topk, lane, LMH_NEG_INF, chosen, chosen_val);
    const long tile_base = (long)token * num_vtiles;
    float maximum = LMH_NEG_INF;
    for (int tile = int(lane); tile < num_vtiles; tile += 32) {
        maximum = max(maximum, part_max[tile_base + tile]);
    }
    maximum = simd_max(maximum);
    float sum = 0.0f;
    for (int tile = int(lane); tile < num_vtiles; tile += 32) {
        const float tile_max = part_max[tile_base + tile];
        if (tile_max != LMH_NEG_INF) {
            sum += part_sum[tile_base + tile] * exp(tile_max - maximum);
        }
    }
    sum = simd_sum(sum);
    const float lse = sum > 0.0f ? maximum + log(sum) : INFINITY;
    if (lane == 0) {
        const long output_base = (long)token * topk;
        for (int i = 0; i < topk; ++i) {
            out_id[output_base + i] = chosen[i];
            out_logprob[output_base + i] =
                chosen[i] >= 0 ? chosen_val[i] - lse : -INFINITY;
        }
    }
}

METAL_FUNC void lmh_insert_candidate(
    float value, int id, int topk, thread float *values, thread int *ids) {
    int position = topk;
    for (int i = 0; i < topk; ++i) {
        if (value > values[i] || (value == values[i] && id < ids[i])) {
            position = i;
            break;
        }
    }
    if (position == topk) return;
    for (int i = topk - 1; i > position; --i) {
        values[i] = values[i - 1];
        ids[i] = ids[i - 1];
    }
    values[position] = value;
    ids[position] = id;
}

struct lmh_candidate_emit {
    device int *out_id;
    device float *out_logprob;
    long base;
    float lse;
    uint lane;
    METAL_FUNC void operator()(int k, float value, int id) {
        if (lane == 0) {
            const bool valid = value != LMH_NEG_INF && id != 0x7fffffff;
            out_id[base + k] = valid ? id : -1;
            out_logprob[base + k] = valid ? value - lse : -INFINITY;
        }
    }
};

template <typename T>
kernel void lm_head_candidates(
    device const T *h [[buffer(0)]],
    device const T *W [[buffer(1)]],
    device const int *candidate_ids [[buffer(2)]],
    device const int *offsets [[buffer(3)]],
    device const float *bias [[buffer(4)]],
    device int *out_id [[buffer(5)]],
    device float *out_logprob [[buffer(6)]],
    constant int &V [[buffer(7)]],
    constant int &K [[buffer(8)]],
    constant int &topk [[buffer(9)]],
    constant int &use_bias [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const T *hrow = h + (long)token * K;
    float mine_val[LMH_MASKED_MAX_K];
    int mine_id[LMH_MASKED_MAX_K];
    bool used[LMH_MASKED_MAX_K];
    for (int i = 0; i < topk; ++i) {
        mine_val[i] = LMH_NEG_INF;
        mine_id[i] = 0x7fffffff;
    }
    float local_max = LMH_NEG_INF, local_sum = 0.0f;
    const int begin = offsets[token], end = offsets[token + 1];
    for (int index = begin + int(lane); index < end; index += 32) {
        const int vocab = candidate_ids[index];
        if (vocab < 0 || vocab >= V) continue;
        float logit = lmh_dense_dot_f32(hrow, W + (long)vocab * K, K);
        if (use_bias != 0) logit += bias[vocab];
        lmh_lse_update(logit, local_max, local_sum);
        lmh_insert_candidate(logit, vocab, topk, mine_val, mine_id);
    }
    const float maximum = simd_max(local_max);
    const float sum = simd_sum(
        local_max == LMH_NEG_INF ? 0.0f : local_sum * exp(local_max - maximum));
    const float lse = sum > 0.0f ? maximum + log(sum) : INFINITY;
    lmh_candidate_emit emit{
        out_id, out_logprob, (long)token * topk, lse, lane};
    masked_topk_local(
        mine_val, mine_id, used, topk, topk, LMH_NEG_INF, emit);
}

template <typename FMT, typename T>
kernel void lm_head_candidates_q(
    device const T *h [[buffer(0)]],
    device const uchar *W [[buffer(1)]],
    device const int *candidate_ids [[buffer(2)]],
    device const int *offsets [[buffer(3)]],
    device const float *bias [[buffer(4)]],
    device int *out_id [[buffer(5)]],
    device float *out_logprob [[buffer(6)]],
    constant int &V [[buffer(7)]],
    constant int &K [[buffer(8)]],
    constant int &topk [[buffer(9)]],
    constant int &use_bias [[buffer(10)]],
    uint token [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    device const T *hrow = h + (long)token * K;
    const int row_bytes = (K / FMT::block_k) * FMT::block_bytes;
    float mine_val[LMH_MASKED_MAX_K];
    int mine_id[LMH_MASKED_MAX_K];
    bool used[LMH_MASKED_MAX_K];
    for (int i = 0; i < topk; ++i) {
        mine_val[i] = LMH_NEG_INF;
        mine_id[i] = 0x7fffffff;
    }
    float local_max = LMH_NEG_INF, local_sum = 0.0f;
    const int begin = offsets[token], end = offsets[token + 1];
    for (int index = begin + int(lane); index < end; index += 32) {
        const int vocab = candidate_ids[index];
        if (vocab < 0 || vocab >= V) continue;
        float logit = lmh_quant_dot_f32<FMT>(
            hrow, W + (long)vocab * row_bytes, K);
        if (use_bias != 0) logit += bias[vocab];
        lmh_lse_update(logit, local_max, local_sum);
        lmh_insert_candidate(logit, vocab, topk, mine_val, mine_id);
    }
    const float maximum = simd_max(local_max);
    const float sum = simd_sum(
        local_max == LMH_NEG_INF ? 0.0f : local_sum * exp(local_max - maximum));
    const float lse = sum > 0.0f ? maximum + log(sum) : INFINITY;
    lmh_candidate_emit emit{
        out_id, out_logprob, (long)token * topk, lse, lane};
    masked_topk_local(
        mine_val, mine_id, used, topk, topk, LMH_NEG_INF, emit);
}

#define instantiate_lm_head_masked_dense(type_name, T)                       \
  template [[host_name("lm_head_masked_partials_dense_" #type_name)]]       \
  [[kernel]] void lm_head_masked_partials<T>(                                \
    device const T *h [[buffer(0)]], device const T *W [[buffer(1)]],        \
    device const float *bias [[buffer(2)]], device const uint *mask [[buffer(3)]], \
    device float *part_val [[buffer(4)]], device int *part_id [[buffer(5)]], \
    device float *part_max [[buffer(6)]], device float *part_sum [[buffer(7)]], \
    constant int &V [[buffer(8)]], constant int &K [[buffer(9)]],            \
    constant int &tile_v [[buffer(10)]], constant int &num_vtiles [[buffer(11)]], \
    constant int &topk [[buffer(12)]], constant int &use_bias [[buffer(13)]], \
    constant int &normalize_allowed [[buffer(14)]],                          \
    constant int &mask_words [[buffer(15)]],                                 \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("lm_head_candidates_dense_" #type_name)]] [[kernel]] \
  void lm_head_candidates<T>(                                                \
    device const T *h [[buffer(0)]], device const T *W [[buffer(1)]],        \
    device const int *candidate_ids [[buffer(2)]], device const int *offsets [[buffer(3)]], \
    device const float *bias [[buffer(4)]], device int *out_id [[buffer(5)]], \
    device float *out_logprob [[buffer(6)]], constant int &V [[buffer(7)]],  \
    constant int &K [[buffer(8)]], constant int &topk [[buffer(9)]],         \
    constant int &use_bias [[buffer(10)]],                                   \
    uint token [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

#define instantiate_lm_head_masked_q(fmt_name, FMT, type_name, T)            \
  template [[host_name("lm_head_masked_partials_" fmt_name "_" #type_name)]] \
  [[kernel]] void lm_head_masked_partials_q<FMT, T>(                         \
    device const T *h [[buffer(0)]], device const uchar *W [[buffer(1)]],    \
    device const float *bias [[buffer(2)]], device const uint *mask [[buffer(3)]], \
    device float *part_val [[buffer(4)]], device int *part_id [[buffer(5)]], \
    device float *part_max [[buffer(6)]], device float *part_sum [[buffer(7)]], \
    constant int &V [[buffer(8)]], constant int &K [[buffer(9)]],            \
    constant int &tile_v [[buffer(10)]], constant int &num_vtiles [[buffer(11)]], \
    constant int &topk [[buffer(12)]], constant int &use_bias [[buffer(13)]], \
    constant int &normalize_allowed [[buffer(14)]],                          \
    constant int &mask_words [[buffer(15)]],                                 \
    uint2 tgid [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]); \
  template [[host_name("lm_head_candidates_" fmt_name "_" #type_name)]]    \
  [[kernel]] void lm_head_candidates_q<FMT, T>(                              \
    device const T *h [[buffer(0)]], device const uchar *W [[buffer(1)]],    \
    device const int *candidate_ids [[buffer(2)]], device const int *offsets [[buffer(3)]], \
    device const float *bias [[buffer(4)]], device int *out_id [[buffer(5)]], \
    device float *out_logprob [[buffer(6)]], constant int &V [[buffer(7)]],  \
    constant int &K [[buffer(8)]], constant int &topk [[buffer(9)]],         \
    constant int &use_bias [[buffer(10)]],                                   \
    uint token [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

#define instantiate_lm_head_masked_type(type_name, T)                        \
  instantiate_lm_head_masked_dense(type_name, T)                             \
  instantiate_lm_head_masked_q("q4_0", q4_0, type_name, T)                  \
  instantiate_lm_head_masked_q("q8_0", q8_0, type_name, T)                  \
  instantiate_lm_head_masked_q("q6_K", q6_K, type_name, T)                  \
  instantiate_lm_head_masked_q("mxfp8", mxfp8, type_name, T)                \
  instantiate_lm_head_masked_q("nvfp4", nvfp4, type_name, T)                \
  instantiate_lm_head_masked_q("mxfp4", mxfp4, type_name, T)

instantiate_lm_head_masked_type(float32, float)
instantiate_lm_head_masked_type(float16, half)
instantiate_lm_head_masked_type(bfloat16, bf16)
