#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// Value-typed bf16 loads: the half specialization rounds
// half->float(exact)->bf16(RNE), bit-identical to an eager .to(bfloat16)
// cast kernel — the fp16 serving path binds fp16 tensors directly and the
// math sees the same values the cast-then-load pipeline produced.
template <typename T> inline bf16 mla_val_bf16(device const T *p, long i);
template <> inline bf16 mla_val_bf16(device const bf16 *p, long i) {
    return p[i];
}
template <> inline bf16 mla_val_bf16(device const half *p, long i) {
    return bf16(float(p[i]));
}

// Exact 2^(e-127) from a UE8M0 scale byte: a float with exponent field e and
// zero mantissa. Fast-math metal::exp2 is ~2 ulps low at negative integer
// inputs (measured on M1 Ultra at -O2; see the insert kernels below), which
// would scale every dequantized cache value below what the encoder used.
// Valid for e in [1, 254]; the insert kernels only emit ~[105, 253].
inline float mla_ue8m0_scale(int e) {
    return as_type<float>((uint)e << 23);
}

// The half instantiation reads the fp16 kv_score GEMM output directly
// (bit-exact: see mla_val_bf16 above).
template <typename T>
kernel void dsv4_save_partial_states(
        device const T *kv [[buffer(0)]],
        device const T *score [[buffer(1)]],
        device const bf16 *ape [[buffer(2)]],
        device const int *positions [[buffer(3)]],
        device float *state_cache [[buffer(4)]],
        device const long *slot_mapping [[buffer(5)]],
        constant int &num_tokens [[buffer(6)]],
        constant int &head_size [[buffer(7)]],
        constant int &block_size [[buffer(8)]],
        constant int &block_stride [[buffer(9)]],
        constant int &token_stride [[buffer(10)]],
        constant int &state_width [[buffer(11)]],
        constant int &compress_ratio [[buffer(12)]],
        constant int &in_stride [[buffer(13)]],
        uint token [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (int(token) >= num_tokens) { return; }
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long block = slot / block_size;
    const long offset = slot - block * block_size;
    const long cache_base = block * block_stride + offset * token_stride;
    // in_stride lets kv/score bind as row-strided views of the fused
    // kv_score GEMM output (no eager .contiguous() copies); values are
    // identical to the packed layout.
    const long input_base = long(token) * in_stride;
    const long ape_base = long(positions[token] % compress_ratio) * head_size;
    // score+ape adds in bf16 by design: the Triton reference
    // (_save_partial_states_kernel) also adds the bf16 loads before the fp32
    // store, so widening the operands here would break bit parity with it.
    for (int dim = int(tid); dim < head_size; dim += 256) {
        state_cache[cache_base + dim] = mla_val_bf16(kv, input_base + dim);
        state_cache[cache_base + state_width + dim] =
            mla_val_bf16(score, input_base + dim) + ape[ape_base + dim];
    }
}

#define instantiate_dsv4_save_partial_states(T, NAME)                          \
  template [[host_name(NAME)]] [[kernel]] void dsv4_save_partial_states<T>(    \
      device const T *kv [[buffer(0)]],                                        \
      device const T *score [[buffer(1)]],                                     \
      device const bf16 *ape [[buffer(2)]],                                    \
      device const int *positions [[buffer(3)]],                               \
      device float *state_cache [[buffer(4)]],                                 \
      device const long *slot_mapping [[buffer(5)]],                           \
      constant int &num_tokens [[buffer(6)]],                                  \
      constant int &head_size [[buffer(7)]],                                   \
      constant int &block_size [[buffer(8)]],                                  \
      constant int &block_stride [[buffer(9)]],                                \
      constant int &token_stride [[buffer(10)]],                               \
      constant int &state_width [[buffer(11)]],                                \
      constant int &compress_ratio [[buffer(12)]],                             \
      constant int &in_stride [[buffer(13)]],                                  \
      uint token [[threadgroup_position_in_grid]],                             \
      uint tid [[thread_position_in_threadgroup]]);

instantiate_dsv4_save_partial_states(bf16, "dsv4_save_partial_states");
instantiate_dsv4_save_partial_states(half, "dsv4_save_partial_states_half");

// ---------------------------------------------------------------------------
// DeepSeek Multi-head Latent Attention (MLA) — preprocessing kernels.
//
// P1: mla_q_norm_rope — the fused Q-path. Per (token, head): optional RMSNorm over
// the full head dim (no-weight for the V4/V3.2 Q-norm, weighted for a kv_a-style
// norm, or none), then GPT-J *interleaved* RoPE on the last `rope_dim` dims (the
// `nope` prefix passes through), bf16 store. Head layout: head_dim = nope_dim +
// rope_dim (e.g. 192 = 128+64 for V2/V3, or 512 = 448+64 for V4).
//
// One warp (32 lanes) per (token, head). Each lane owns head_dim/32 CONTIGUOUS
// elements — even (head_dim % 64 == 0), so every interleaved pair (g, g+1) with g
// even is resident in a single lane (no cross-lane shuffle), and the full-head
// sum-of-squares is a per-lane contiguous sum + one simd_sum. nope_dim even ⇒ a
// pair never straddles the nope/rope boundary.
//
// cos/sin are separate (max_pos, rope_dim/2) bf16 tables (the ThunderMittens RoPE
// convention), indexed by positions[token]. Golden: rmsnorm_no_weight +
// apply_rope_gptj_last_k in vLLM's test_fused_deepseek_v4_qnorm_rope_kv_insert.
//
// The half instantiations read fp16 activations directly, rounding
// half->float(exact)->bf16(RNE) at load (bit-exact vs a cast-then-load
// pipeline; same contract as mla_val_bf16).
// ---------------------------------------------------------------------------
template <typename T> inline float mla_load_bf16(device const T *p, long i);
template <> inline float mla_load_bf16(device const bf16 *p, long i) {
    return float(p[i]);
}
template <> inline float mla_load_bf16(device const half *p, long i) {
    return float(bf16(float(p[i])));
}

template <int D, typename QT>
kernel void mla_q_norm_rope(device const QT  *q          [[buffer(0)]],
                            device const bf16 *cosb        [[buffer(1)]],
                            device const bf16 *sinb        [[buffer(2)]],
                            device const int  *positions   [[buffer(3)]],
                            device bf16       *out         [[buffer(4)]],
                            constant int &num_heads        [[buffer(5)]],
                            constant int &nope_dim         [[buffer(6)]],
                            constant int &rope_dim         [[buffer(7)]],
                            constant int &norm_mode        [[buffer(8)]],   // 0 none,1 rms,2 rms+w
                            constant float &eps            [[buffer(9)]],
                            device const QT  *norm_weight [[buffer(10)]],  // (D,), read iff mode 2
                            uint3 blockIdx [[threadgroup_position_in_grid]],
                            uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D % 64 == 0, "mla_q_norm_rope needs head_dim divisible by 64");
    constexpr int PER_LANE = D / 32;              // contiguous, even
    const int row = blockIdx.x;                   // (token, head) flattened
    const int token = row / num_heads;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;
    const long base = (long)row * D + (long)laneId * PER_LANE;

    // Full-head RMS (no-weight) if requested.
    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        for (int k = 0; k < PER_LANE; ++k) { const float v = mla_load_bf16(q, base + k); ss += v * v; }
        ss = simd_sum(ss);
        rms = metal::rsqrt(ss / (float)D + eps);
    }

    const long wbase = (long)laneId * PER_LANE;   // norm_weight index for this lane's chunk
    const long csbase = (long)pos * rope_half;
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = (int)laneId * PER_LANE + k;   // even global index (start of a pair)
        float v0 = mla_load_bf16(q, base + k) * rms;
        float v1 = mla_load_bf16(q, base + k + 1) * rms;
        if (norm_mode == 2) {
            v0 *= mla_load_bf16(norm_weight, wbase + k);
            v1 *= mla_load_bf16(norm_weight, wbase + k + 1);
        }
        if (g0 >= nope_dim) {
            const int p = (g0 - nope_dim) / 2;       // rope pair index
            const float c = float(cosb[csbase + p]);
            const float s = float(sinb[csbase + p]);
            out[base + k]     = bf16(v0 * c - v1 * s);
            out[base + k + 1] = bf16(v0 * s + v1 * c);
        } else {
            out[base + k]     = bf16(v0);
            out[base + k + 1] = bf16(v1);
        }
    }
}

#define instantiate_mla_q_norm_rope_t(DVAL, QT, NAME)                          \
  template [[host_name(NAME)]] [[kernel]] void                                 \
  mla_q_norm_rope<DVAL, QT>(device const QT *q [[buffer(0)]],                  \
                        device const bf16 *cosb [[buffer(1)]],                 \
                        device const bf16 *sinb [[buffer(2)]],                 \
                        device const int  *positions [[buffer(3)]],            \
                        device bf16       *out [[buffer(4)]],                  \
                        constant int &num_heads [[buffer(5)]],                 \
                        constant int &nope_dim [[buffer(6)]],                  \
                        constant int &rope_dim [[buffer(7)]],                  \
                        constant int &norm_mode [[buffer(8)]],                 \
                        constant float &eps [[buffer(9)]],                     \
                        device const QT *norm_weight [[buffer(10)]],           \
                        uint3 blockIdx [[threadgroup_position_in_grid]],       \
                        uint  laneId   [[thread_index_in_simdgroup]]);
#define instantiate_mla_q_norm_rope(DVAL)                                      \
  instantiate_mla_q_norm_rope_t(DVAL, bf16, "mla_q_norm_rope_" #DVAL)

instantiate_mla_q_norm_rope(128);
instantiate_mla_q_norm_rope(192);
instantiate_mla_q_norm_rope(256);
instantiate_mla_q_norm_rope(512);
// fp16-input variant used by fp16 serving (rounds to bf16 in-register).
// Only D=512 exists for half input (the DSV4 serving shape); the launcher
// builds the name from the runtime head_dim, so a half input with any other
// head_dim is a PSO-not-found at dispatch.
instantiate_mla_q_norm_rope_t(512, half, "mla_q_norm_rope_half_512");

// ---------------------------------------------------------------------------
// P2: mla_kv_insert — classic bf16 latent KV-insert (concat_and_cache_mla). One warp per token
// writes into a paged cache kv_cache[num_blocks, block_size, LATENT + rope_dim] (MQA — one shared
// latent per token, no head axis): the compressed latent kv_c (LATENT, optionally kv_a-RMSNormed)
// at [0:LATENT], and interleaved-RoPE'd k_pe (rope_dim) at [LATENT:LATENT+rope_dim]. Clone-then-
// insert: the caller pre-populates the cache; this kernel overwrites only the mapped slots.
// LATENT % 64 == 0; rope_dim/2 <= 32 (one pair per lane).
// ---------------------------------------------------------------------------
template <int LATENT>
kernel void mla_kv_insert(device const bf16 *kv_c        [[buffer(0)]],   // (T, LATENT)
                          device const bf16 *k_pe        [[buffer(1)]],   // (T, rope_dim)
                          device const bf16 *cosb        [[buffer(2)]],
                          device const bf16 *sinb        [[buffer(3)]],
                          device const int  *positions   [[buffer(4)]],
                          device const long *slot_mapping [[buffer(5)]],
                          device bf16       *kv_cache    [[buffer(6)]],    // (nb, bs, LATENT+rope)
                          constant int &block_size       [[buffer(7)]],
                          constant int &rope_dim         [[buffer(8)]],
                          constant int &norm_mode        [[buffer(9)]],    // 0 none, 2 weighted
                          constant float &eps            [[buffer(10)]],
                          device const bf16 *norm_weight [[buffer(11)]],   // (LATENT,), mode 2
                          uint3 blockIdx [[threadgroup_position_in_grid]],
                          uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(LATENT % 64 == 0, "mla_kv_insert needs LATENT divisible by 64");
    constexpr int LPL = LATENT / 32;                 // latent elements per lane (even)
    const int token = blockIdx.x;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long block = slot / block_size;
    const long off = slot % block_size;
    const int row_width = LATENT + rope_dim;
    const long dst = ((block * block_size + off)) * (long)row_width;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;

    // Latent: optional RMSNorm over LATENT, then write to [0:LATENT].
    const long lbase = (long)token * LATENT + (long)laneId * LPL;
    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        for (int k = 0; k < LPL; ++k) { const float v = float(kv_c[lbase + k]); ss += v * v; }
        ss = simd_sum(ss);
        rms = metal::rsqrt(ss / (float)LATENT + eps);
    }
    for (int k = 0; k < LPL; ++k) {
        float v = float(kv_c[lbase + k]) * rms;
        if (norm_mode == 2) { v *= float(norm_weight[laneId * LPL + k]); }
        kv_cache[dst + laneId * LPL + k] = bf16(v);
    }

    // RoPE key: interleaved rotate on rope_dim, write to [LATENT:LATENT+rope_dim].
    if ((int)laneId < rope_half) {
        const long rbase = (long)token * rope_dim + (long)laneId * 2;
        const float e = float(k_pe[rbase]);
        const float o = float(k_pe[rbase + 1]);
        const float c = float(cosb[(long)pos * rope_half + laneId]);
        const float s = float(sinb[(long)pos * rope_half + laneId]);
        kv_cache[dst + LATENT + laneId * 2]     = bf16(e * c - o * s);
        kv_cache[dst + LATENT + laneId * 2 + 1] = bf16(e * s + o * c);
    }
}

#define instantiate_mla_kv_insert(LVAL)                                        \
  template [[host_name("mla_kv_insert_" #LVAL)]] [[kernel]] void               \
  mla_kv_insert<LVAL>(device const bf16 *kv_c [[buffer(0)]],                   \
                      device const bf16 *k_pe [[buffer(1)]],                   \
                      device const bf16 *cosb [[buffer(2)]],                   \
                      device const bf16 *sinb [[buffer(3)]],                   \
                      device const int  *positions [[buffer(4)]],              \
                      device const long *slot_mapping [[buffer(5)]],           \
                      device bf16       *kv_cache [[buffer(6)]],               \
                      constant int &block_size [[buffer(7)]],                  \
                      constant int &rope_dim [[buffer(8)]],                    \
                      constant int &norm_mode [[buffer(9)]],                   \
                      constant float &eps [[buffer(10)]],                      \
                      device const bf16 *norm_weight [[buffer(11)]],           \
                      uint3 blockIdx [[threadgroup_position_in_grid]],         \
                      uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_mla_kv_insert(128);
instantiate_mla_kv_insert(256);
instantiate_mla_kv_insert(512);

// ---------------------------------------------------------------------------
// P3: mla_kv_insert_fp8 — DeepSeek-V4/V3.2 packed KV-insert. The 512-wide latent is [448 NoPE |
// 64 RoPE]. NoPE is quantized to e4m3 fp8 with a per-64-block UE8M0 (power-of-2) scale; RoPE gets
// interleaved RoPE and stays bf16. Per token: data_cache (…, 576 bytes) = 448 code bytes ‖ 128
// bytes (64 bf16 rope); scale_cache (…, 8 bytes) = 7 UE8M0 exponent bytes + 1 pad. One warp/token,
// each lane owning 16 contiguous latent elems: lanes 0..27 = 7 NoPE blocks (4 lanes = one 64-block,
// reduced via simd_shuffle_xor), lanes 28..31 = the 64 RoPE dims. UE8M0: exponent =
// ceil(log2(absmax/448)); scale_byte = exponent+127; code = e4m3(x·2^-exponent) (matches vLLM).
// ---------------------------------------------------------------------------
kernel void mla_kv_insert_fp8(device const bf16 *kv          [[buffer(0)]],   // (T, 512)
                              device const bf16 *cosb        [[buffer(1)]],   // (P, 32)
                              device const bf16 *sinb        [[buffer(2)]],
                              device const int  *positions   [[buffer(3)]],
                              device const long *slot_mapping [[buffer(4)]],
                              device uchar *data_cache       [[buffer(5)]],   // (nb, bs, 576)
                              device uchar *scale_cache      [[buffer(6)]],   // (nb, bs, 8)
                              constant int &block_size       [[buffer(7)]],
                              uint3 blockIdx [[threadgroup_position_in_grid]],
                              uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int LAT = 512, NOPE = 448, PER_LANE = 16, NOPE_LANES = NOPE / PER_LANE;  // 28
    constexpr float FP8_MAX = 448.0f;
    const int token = blockIdx.x;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long dslot = (slot / block_size) * block_size + (slot % block_size);
    const long dst_data = dslot * 576;
    const long dst_scale = dslot * 8;
    const int pos = positions[token];
    const long kbase = (long)token * LAT + (long)laneId * PER_LANE;

    float v[PER_LANE];
    for (int k = 0; k < PER_LANE; ++k) { v[k] = float(kv[kbase + k]); }

    // Per-64-block absmax = max over the 4 lanes in this lane's block (unconditional — the shuffle
    // is convergent; RoPE lanes 28..31 form their own harmless group we ignore).
    float amax = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) { amax = metal::max(amax, metal::fabs(v[k])); }
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 1));
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 2));
    const float exponent = metal::ceil(metal::log2(metal::max(amax, 1e-4f) / FP8_MAX));
    // Fast-math exp2 lands ~2 ulps low at negative integer inputs (measured on
    // M1 Ultra; see the indexer compress kernels), which shifts e4m3 rounding
    // ties vs the exact fp32 reference. Build 2^-e from the float bit pattern
    // and derive the stored scale byte from the same clamped ei.
    const int ei = metal::clamp((int)exponent, -126, 126);
    const float inv_scale = as_type<float>((uint)((127 - ei) << 23));

    if ((int)laneId < NOPE_LANES) {
        for (int k = 0; k < PER_LANE; ++k) {
            data_cache[dst_data + laneId * PER_LANE + k] = tk_e4m3_encode(v[k] * inv_scale);
        }
        if ((laneId & 3) == 0) {   // first lane of each 4-lane (64-elem) block writes its scale byte
            scale_cache[dst_scale + laneId / 4] = (uchar)(ei + 127);
        }
    } else {
        // RoPE dims [448,512): this lane holds a 16-wide contiguous slice (8 pairs).
        const int rl = ((int)laneId - NOPE_LANES) * PER_LANE;   // rope-local start: 0,16,32,48
        device bf16 *rope_out = (device bf16 *)(data_cache + dst_data + NOPE);
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (rl + j) / 2;                          // rope pair index 0..31
            const float c = float(cosb[(long)pos * 32 + p]);
            const float s = float(sinb[(long)pos * 32 + p]);
            rope_out[rl + j]     = bf16(v[j] * c - v[j + 1] * s);
            rope_out[rl + j + 1] = bf16(v[j] * s + v[j + 1] * c);
        }
    }
    if (laneId == 0) { scale_cache[dst_scale + 7] = 0; }   // pad byte
}

// vLLM allocates the DeepSeek-V4 cache as one 584-byte token slot rather than
// the split 576-byte data and 8-byte scale arrays used by the standalone
// QuixiCore API. Keep the shader math identical and only change the address
// calculation so the serving path can update its cache in place.
kernel void mla_kv_insert_fp8_packed(
        device const bf16 *kv           [[buffer(0)]],
        device const bf16 *cosb         [[buffer(1)]],
        device const bf16 *sinb         [[buffer(2)]],
        device const int  *positions    [[buffer(3)]],
        device const long *slot_mapping [[buffer(4)]],
        device uchar      *kv_cache     [[buffer(5)]],
        constant int &block_size        [[buffer(6)]],
        constant int &cache_block_stride [[buffer(7)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint laneId [[thread_index_in_simdgroup]]) {
    constexpr int LAT = 512, NOPE = 448, PER_LANE = 16;
    constexpr int NOPE_LANES = NOPE / PER_LANE;
    constexpr int SLOT_BYTES = 584, DATA_BYTES = 576;
    constexpr float FP8_MAX = 448.0f;
    const int token = blockIdx.x;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long base = (slot / block_size) * (long)cache_block_stride +
                      (slot % block_size) * SLOT_BYTES;
    const int pos = positions[token];
    const long kbase = (long)token * LAT + (long)laneId * PER_LANE;

    float v[PER_LANE];
    for (int k = 0; k < PER_LANE; ++k) { v[k] = float(kv[kbase + k]); }

    float amax = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) {
        amax = metal::max(amax, metal::fabs(v[k]));
    }
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 1));
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 2));
    const float exponent = metal::ceil(
        metal::log2(metal::max(amax, 1e-4f) / FP8_MAX));
    // Exact 2^-e from the bit pattern — fast-math exp2 is ~2 ulps low at
    // negative integer inputs (see mla_kv_insert_fp8 / indexer kernels).
    const int ei = metal::clamp((int)exponent, -126, 126);
    const float inv_scale = as_type<float>((uint)((127 - ei) << 23));

    if ((int)laneId < NOPE_LANES) {
        for (int k = 0; k < PER_LANE; ++k) {
            kv_cache[base + laneId * PER_LANE + k] =
                tk_e4m3_encode(v[k] * inv_scale);
        }
        if ((laneId & 3) == 0) {
            kv_cache[base + DATA_BYTES + laneId / 4] = (uchar)(ei + 127);
        }
    } else {
        const int rl = ((int)laneId - NOPE_LANES) * PER_LANE;
        device bf16 *rope_out = (device bf16 *)(kv_cache + base + NOPE);
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (rl + j) / 2;
            const float c = float(cosb[(long)pos * 32 + p]);
            const float s = float(sinb[(long)pos * 32 + p]);
            rope_out[rl + j] = bf16(v[j] * c - v[j + 1] * s);
            rope_out[rl + j + 1] = bf16(v[j] * s + v[j + 1] * c);
        }
    }
    if (laneId == 0) { kv_cache[base + DATA_BYTES + 7] = 0; }
}

// fp16-input twin of mla_kv_insert_fp8_packed. Two deltas only: each
// element is rounded half->bf16 (RNE) at load (bit-exact, see
// mla_load_bf16), and src_stride lets kv bind as a row-strided view of
// the fused projection output (no eager .contiguous()).
kernel void mla_kv_insert_fp8_packed_half(
        device const half *kv           [[buffer(0)]],
        device const bf16 *cosb         [[buffer(1)]],
        device const bf16 *sinb         [[buffer(2)]],
        device const int  *positions    [[buffer(3)]],
        device const long *slot_mapping [[buffer(4)]],
        device uchar      *kv_cache     [[buffer(5)]],
        constant int &block_size        [[buffer(6)]],
        constant int &cache_block_stride [[buffer(7)]],
        constant int &src_stride        [[buffer(8)]],
        uint3 blockIdx [[threadgroup_position_in_grid]],
        uint laneId [[thread_index_in_simdgroup]]) {
    constexpr int NOPE = 448, PER_LANE = 16;
    constexpr int NOPE_LANES = NOPE / PER_LANE;
    constexpr int SLOT_BYTES = 584, DATA_BYTES = 576;
    constexpr float FP8_MAX = 448.0f;
    const int token = blockIdx.x;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long base = (slot / block_size) * (long)cache_block_stride +
                      (slot % block_size) * SLOT_BYTES;
    const int pos = positions[token];
    const long kbase = (long)token * src_stride + (long)laneId * PER_LANE;

    float v[PER_LANE];
    for (int k = 0; k < PER_LANE; ++k) { v[k] = mla_load_bf16(kv, kbase + k); }

    // amax reduces over a 4-lane group only (xor-1 + xor-2): 4 lanes x
    // PER_LANE dims = one 64-dim fp8 scale group; the (laneId & 3) == 0
    // lane below writes that group's exponent byte.
    float amax = 0.0f;
    for (int k = 0; k < PER_LANE; ++k) {
        amax = metal::max(amax, metal::fabs(v[k]));
    }
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 1));
    amax = metal::max(amax, metal::simd_shuffle_xor(amax, 2));
    const float exponent = metal::ceil(
        metal::log2(metal::max(amax, 1e-4f) / FP8_MAX));
    // Exact 2^-e from the bit pattern — fast-math exp2 is ~2 ulps low at
    // negative integer inputs (see mla_kv_insert_fp8 / indexer kernels).
    const int ei = metal::clamp((int)exponent, -126, 126);
    const float inv_scale = as_type<float>((uint)((127 - ei) << 23));

    if ((int)laneId < NOPE_LANES) {
        for (int k = 0; k < PER_LANE; ++k) {
            kv_cache[base + laneId * PER_LANE + k] =
                tk_e4m3_encode(v[k] * inv_scale);
        }
        if ((laneId & 3) == 0) {
            kv_cache[base + DATA_BYTES + laneId / 4] = (uchar)(ei + 127);
        }
    } else {
        const int rl = ((int)laneId - NOPE_LANES) * PER_LANE;
        device bf16 *rope_out = (device bf16 *)(kv_cache + base + NOPE);
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (rl + j) / 2;
            const float c = float(cosb[(long)pos * 32 + p]);
            const float s = float(sinb[(long)pos * 32 + p]);
            rope_out[rl + j] = bf16(v[j] * c - v[j + 1] * s);
            rope_out[rl + j + 1] = bf16(v[j] * s + v[j + 1] * c);
        }
    }
    if (laneId == 0) { kv_cache[base + DATA_BYTES + 7] = 0; }
}

// ---------------------------------------------------------------------------
// P4: mla_decode — MLA absorb-path latent flash-decode (MQA). The query is the absorbed
// [ql_nope(LATENT) ‖ q_pe(rope)] = QK-wide vector (ql_nope = q_nope @ W_UK_T, done by the caller);
// the paged cache kv_cache[nb, bs, QK] stores one shared latent per token = [latent(LATENT) ‖
// k_pe(rope)]. Score is the full QK-wide dot (latent + rope), but the value accumulate is over the
// LATENT part only (rope carries no value) — an asymmetric dot(QK)/accumulate(LATENT) decode. Output
// o (…, LATENT) is then W_UV-up-projected by the caller. One simdgroup per (head, batch); the striped
// lane map (d = lane + 32*i) puts the latent in i<LATENT/32 and the rope in the tail, so the AV loop
// is just the first LATENT/32 iterations. Absorb-path == MHA path algebraically.
// ---------------------------------------------------------------------------
template <int LATENT, int ROPE>
kernel void mla_decode(device const bf16 *q            [[buffer(0)]],   // (B, N, LATENT+ROPE)
                       device const bf16 *kv_cache     [[buffer(1)]],   // (nb, bs, LATENT+ROPE)
                       device const int  *block_table  [[buffer(2)]],
                       device const int  *context_lens [[buffer(3)]],
                       device bf16       *out          [[buffer(4)]],   // (B, N, LATENT)
                       constant int &block_size        [[buffer(5)]],
                       constant int &block_table_stride [[buffer(6)]],
                       constant float &scale           [[buffer(7)]],
                       constant int &num_heads         [[buffer(8)]],
                       uint3 tgid [[threadgroup_position_in_grid]],
                       uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int QK = LATENT + ROPE;
    constexpr int VPL_QK = QK / 32;        // query values per lane (dot width)
    constexpr int VPL_AV = LATENT / 32;    // latent values per lane (accumulate width)
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int context_len = context_lens[batch];
    const long q_base = ((long)batch * num_heads + head) * QK;

    float qv[VPL_QK], acc[VPL_AV];
    for (int i = 0; i < VPL_QK; ++i) { qv[i] = float(q[q_base + lane + 32 * i]); }
    for (int i = 0; i < VPL_AV; ++i) { acc[i] = 0.0f; }

    float m = -3.4028234663852886e38f, l = 0.0f;
    for (int t = 0; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long cache_base = ((long)block * block_size + slot) * QK;   // MQA: no head axis

        float partial = 0.0f;
        for (int i = 0; i < VPL_QK; ++i) {
            partial += qv[i] * float(kv_cache[cache_base + lane + 32 * i]);
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL_AV; ++i) {      // value = the latent part only
            acc[i] = acc[i] * alpha + beta * float(kv_cache[cache_base + lane + 32 * i]);
        }
        l = l * alpha + beta;
        m = new_m;
    }

    const long out_base = ((long)batch * num_heads + head) * LATENT;
    for (int i = 0; i < VPL_AV; ++i) {
        out[out_base + lane + 32 * i] = l == 0.0f ? bf16(0) : bf16(acc[i] / l);
    }
}

#define instantiate_mla_decode(LVAL, RVAL)                                     \
  template [[host_name("mla_decode_" #LVAL "_" #RVAL)]] [[kernel]] void         \
  mla_decode<LVAL, RVAL>(device const bf16 *q [[buffer(0)]],                    \
                         device const bf16 *kv_cache [[buffer(1)]],             \
                         device const int  *block_table [[buffer(2)]],          \
                         device const int  *context_lens [[buffer(3)]],         \
                         device bf16       *out [[buffer(4)]],                   \
                         constant int &block_size [[buffer(5)]],                \
                         constant int &block_table_stride [[buffer(6)]],        \
                         constant float &scale [[buffer(7)]],                   \
                         constant int &num_heads [[buffer(8)]],                 \
                         uint3 tgid [[threadgroup_position_in_grid]],           \
                         uint  lane [[thread_index_in_simdgroup]]);

instantiate_mla_decode(512, 64);

// ---------------------------------------------------------------------------
// P4v2: mla_decode_partition — the P4 decode with a v2-style sequence-partition grid axis
// (grid (H, B, P), one simdgroup per (head, partition)), emitting paged_attention_v2-style
// partials (per-partition normalized acc, max logit, exp sum) combined by the existing
// paged_attention_reduce<bf16, LATENT> kernel. Occupancy at long context, like v2's 2-5x
// over v1. A multi-head threadgroup-staged variant (4 heads sharing each staged token block)
// was tried and REJECTED: 1.5-1.6x slower at 32 heads — the cache already serves the
// cross-head reuse, and the per-8-token barriers cost more than the reads they save (the
// same lesson as paged_attention_gqa_staged and gemm_staged).
// ---------------------------------------------------------------------------
template <int LATENT, int ROPE>
kernel void mla_decode_partition(
    device const bf16 *q            [[buffer(0)]],   // (B, N, QK)
    device const bf16 *kv_cache     [[buffer(1)]],   // (nb, bs, QK)
    device const int  *block_table  [[buffer(2)]],
    device const int  *context_lens [[buffer(3)]],
    device float      *tmp_out      [[buffer(4)]],   // (B, N, P, LATENT)
    device float      *max_logits   [[buffer(5)]],   // (B, N, P)
    device float      *exp_sums     [[buffer(6)]],   // (B, N, P)
    constant int &block_size        [[buffer(7)]],
    constant int &block_table_stride [[buffer(8)]],
    constant float &scale           [[buffer(9)]],
    constant int &num_heads         [[buffer(10)]],
    constant int &num_partitions    [[buffer(11)]],
    constant int &partition_size    [[buffer(12)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int QK = LATENT + ROPE;
    constexpr int VPL_QK = QK / 32;        // query values per lane (dot width)
    constexpr int VPL_AV = LATENT / 32;    // latent values per lane (accumulate width)
    constexpr float MLA_NEG_INF = -3.4028234663852886e38f;   // must match paged_attention_reduce
    const int head  = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part  = (int)tgid.z;
    const int context_len = context_lens[batch];
    const int t_beg = part * partition_size;
    const int t_end = metal::min(context_len, t_beg + partition_size);

    float qv[VPL_QK], acc[VPL_AV];
    const long q_base = ((long)batch * num_heads + head) * QK;
    for (int i = 0; i < VPL_QK; ++i) qv[i] = float(q[q_base + lane + 32 * i]);
    for (int i = 0; i < VPL_AV; ++i) acc[i] = 0.0f;
    float m = MLA_NEG_INF, l = 0.0f;

    for (int t = t_beg; t < t_end; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long cache_base = ((long)block * block_size + slot) * QK;   // MQA: no head axis

        float partial = 0.0f;
        for (int i = 0; i < VPL_QK; ++i)
            partial += qv[i] * float(kv_cache[cache_base + lane + 32 * i]);
        const float score = metal::simd_sum(partial) * scale;
        const float new_m = metal::max(m, score);
        const float alpha = l == 0.0f ? 0.0f : metal::exp(m - new_m);
        const float beta = metal::exp(score - new_m);
        for (int i = 0; i < VPL_AV; ++i)      // value = the latent part only
            acc[i] = acc[i] * alpha + beta * float(kv_cache[cache_base + lane + 32 * i]);
        l = l * alpha + beta;
        m = new_m;
    }

    const long stat = ((long)batch * num_heads + head) * num_partitions + part;
    const long ob = stat * LATENT;
    for (int i = 0; i < VPL_AV; ++i)
        tmp_out[ob + lane + 32 * i] = l == 0.0f ? 0.0f : acc[i] / l;
    if (lane == 0) {
        max_logits[stat] = l == 0.0f ? MLA_NEG_INF : m;
        exp_sums[stat] = l;
    }
}

#define instantiate_mla_decode_partition(LVAL, RVAL)                            \
  template [[host_name("mla_decode_partition_" #LVAL "_" #RVAL)]] [[kernel]]     \
  void mla_decode_partition<LVAL, RVAL>(                                         \
      device const bf16 *q [[buffer(0)]],                                       \
      device const bf16 *kv_cache [[buffer(1)]],                                \
      device const int  *block_table [[buffer(2)]],                             \
      device const int  *context_lens [[buffer(3)]],                            \
      device float      *tmp_out [[buffer(4)]],                                 \
      device float      *max_logits [[buffer(5)]],                              \
      device float      *exp_sums [[buffer(6)]],                                \
      constant int &block_size [[buffer(7)]],                                   \
      constant int &block_table_stride [[buffer(8)]],                           \
      constant float &scale [[buffer(9)]],                                      \
      constant int &num_heads [[buffer(10)]],                                   \
      constant int &num_partitions [[buffer(11)]],                              \
      constant int &partition_size [[buffer(12)]],                              \
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_mla_decode_partition(512, 64);

// ---------------------------------------------------------------------------
// P4a: mla_decode_fp8 — DeepSeek-V4 dense latent decode over the UE8M0-packed cache (P3). The V4
// latent is 512 = [448 NoPE | 64 RoPE], and (unlike classic MLA) BOTH the score and the value are
// over the full 512 (rope included), scale = 512^-0.5. Per cached token this dequantizes the 448
// NoPE (e4m3_decode * 2^(scale_byte-127), per-64 UE8M0 block) and reads the 64 bf16 RoPE, then does
// the online-softmax decode. Output o (B, num_heads, 512); the inverse-RoPE of o[448:512] + the
// grouped wo_a/wo_b projection are the caller's (phase 4d). MQA: one shared latent per token.
// ---------------------------------------------------------------------------
kernel void mla_decode_fp8(device const bf16 *q            [[buffer(0)]],   // (B, N, 512)
                           device const uchar *data_cache  [[buffer(1)]],   // (nb, bs, 576)
                           device const uchar *scale_cache [[buffer(2)]],   // (nb, bs, 8)
                           device const int  *block_table  [[buffer(3)]],
                           device const int  *context_lens [[buffer(4)]],
                           device bf16       *out          [[buffer(5)]],   // (B, N, 512)
                           constant int &block_size        [[buffer(6)]],
                           constant int &block_table_stride [[buffer(7)]],
                           constant float &scale           [[buffer(8)]],
                           constant int &num_heads         [[buffer(9)]],
                           uint3 tgid [[threadgroup_position_in_grid]],
                           uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;   // 16 per lane
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int context_len = context_lens[batch];
    const long q_base = ((long)batch * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    for (int i = 0; i < VPL; ++i) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }

    float m = -3.4028234663852886e38f, l = 0.0f;
    for (int t = 0; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long dslot = (long)block * block_size + slot;
        const long dbase = dslot * 576;      // packed data bytes
        const long sbase = dslot * 8;         // UE8M0 scale bytes
        device const bf16 *rope = (device const bf16 *)(data_cache + dbase + NOPE);

        float lat[VPL];
        float partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const uchar code = data_cache[dbase + d];
                const int e = (int)scale_cache[sbase + d / 64];
                lat[i] = float(tk_e4m3_decode(code)) * mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) { acc[i] = acc[i] * alpha + beta * lat[i]; }
        l = l * alpha + beta;
        m = new_m;
    }

    const long out_base = ((long)batch * num_heads + head) * LATENT;
    for (int i = 0; i < VPL; ++i) {
        out[out_base + lane + 32 * i] = l == 0.0f ? bf16(0) : bf16(acc[i] / l);
    }
}

// ---------------------------------------------------------------------------
// P4b: mla_decode_fp8_sparse — DeepSeek-V4 sparse latent decode. Same dequant-on-read V4 math as
// mla_decode_fp8, but each query attends only the caller-provided token positions
// indices[batch, 0:topk_length[batch]] (the Lightning Indexer's top-k set) instead of the whole
// context — a gather-by-index decode. indices entries < 0 are skipped.
// ---------------------------------------------------------------------------
kernel void mla_decode_fp8_sparse(device const bf16 *q            [[buffer(0)]],
                                  device const uchar *data_cache  [[buffer(1)]],
                                  device const uchar *scale_cache [[buffer(2)]],
                                  device const int  *block_table  [[buffer(3)]],
                                  device const int  *indices      [[buffer(4)]],   // (B, max_topk)
                                  device const int  *topk_length  [[buffer(5)]],   // (B,)
                                  device bf16       *out          [[buffer(6)]],
                                  constant int &block_size        [[buffer(7)]],
                                  constant int &block_table_stride [[buffer(8)]],
                                  constant float &scale           [[buffer(9)]],
                                  constant int &num_heads         [[buffer(10)]],
                                  constant int &max_topk          [[buffer(11)]],
                                  uint3 tgid [[threadgroup_position_in_grid]],
                                  uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int len = topk_length[batch];
    const long q_base = ((long)batch * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    for (int i = 0; i < VPL; ++i) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }

    float m = -3.4028234663852886e38f, l = 0.0f;
    for (int j = 0; j < len; ++j) {
        const int t = indices[batch * max_topk + j];
        if (t < 0) { continue; }
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long dslot = (long)block * block_size + slot;
        const long dbase = dslot * 576;
        const long sbase = dslot * 8;
        device const bf16 *rope = (device const bf16 *)(data_cache + dbase + NOPE);

        float lat[VPL];
        float partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const uchar code = data_cache[dbase + d];
                const int e = (int)scale_cache[sbase + d / 64];
                lat[i] = float(tk_e4m3_decode(code)) * mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) { acc[i] = acc[i] * alpha + beta * lat[i]; }
        l = l * alpha + beta;
        m = new_m;
    }

    const long out_base = ((long)batch * num_heads + head) * LATENT;
    for (int i = 0; i < VPL; ++i) {
        out[out_base + lane + 32 * i] = l == 0.0f ? bf16(0) : bf16(acc[i] / l);
    }
}

// DeepSeek-V4 combines a compressed long-range cache with the uncompressed
// sliding-window cache. The vLLM metadata builder supplies physical slot ids
// for both lists, so this kernel can consume both cache allocations without a
// gather/copy workspace and merge them in one exact online softmax.
//
// The half-output instantiation writes the fp16 serving buffer directly:
// each element is rounded float->bf16(RNE) first, then widened/converted
// to half exactly as a bf16-store + cast-copy chain would.
template <typename T> inline T mla_store_bf16(float v);
template <> inline bf16 mla_store_bf16<bf16>(float v) { return bf16(v); }
template <> inline half mla_store_bf16<half>(float v) {
    return half(float(bf16(v)));
}

template <typename OT>
kernel void mla_decode_fp8_sparse_two_cache_packed(
        device const bf16 *q             [[buffer(0)]],
        device const uchar *compressed   [[buffer(1)]],
        device const int *compressed_idx [[buffer(2)]],
        device const int *compressed_len [[buffer(3)]],
        device const uchar *swa          [[buffer(4)]],
        device const int *swa_idx        [[buffer(5)]],
        device const int *swa_len        [[buffer(6)]],
        device const float *sinks        [[buffer(7)]],
        device OT *out                   [[buffer(8)]],
        constant int &num_heads          [[buffer(9)]],
        constant int &compressed_width   [[buffer(10)]],
        constant int &swa_width          [[buffer(11)]],
        constant float &scale            [[buffer(12)]],
        constant int &compressed_block_size [[buffer(13)]],
        constant int &compressed_block_stride [[buffer(14)]],
        constant int &swa_block_size [[buffer(15)]],
        constant int &swa_block_stride [[buffer(16)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    constexpr int SLOT_BYTES = 584, DATA_BYTES = 576;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const long q_base = ((long)batch * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    for (int i = 0; i < VPL; ++i) {
        qv[i] = float(q[q_base + lane + 32 * i]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    const int compressed_count = compressed_len[batch];
    for (int j = 0; j < compressed_count; ++j) {
        const int slot = compressed_idx[batch * compressed_width + j];
        if (slot < 0) { continue; }
        const long base = (slot / compressed_block_size) *
                              (long)compressed_block_stride +
                          (slot % compressed_block_size) * SLOT_BYTES;
        device const bf16 *rope =
            (device const bf16 *)(compressed + base + NOPE);
        float lat[VPL], partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const int e = (int)compressed[base + DATA_BYTES + d / 64];
                lat[i] = float(tk_e4m3_decode(compressed[base + d])) *
                         mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) {
            acc[i] = acc[i] * alpha + beta * lat[i];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    const int slen = swa_len[batch];
    for (int j = 0; j < slen; ++j) {
        const int slot = swa_idx[batch * swa_width + j];
        if (slot < 0) { continue; }
        const long base = (slot / swa_block_size) * (long)swa_block_stride +
                          (slot % swa_block_size) * SLOT_BYTES;
        device const bf16 *rope = (device const bf16 *)(swa + base + NOPE);
        float lat[VPL], partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const int e = (int)swa[base + DATA_BYTES + d / 64];
                lat[i] = float(tk_e4m3_decode(swa[base + d])) *
                         mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) {
            acc[i] = acc[i] * alpha + beta * lat[i];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    const float sink = sinks[head];
    if (isfinite(sink)) {
        const float new_m = max(m, sink);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        l = l * alpha + exp(sink - new_m);
        for (int i = 0; i < VPL; ++i) { acc[i] *= alpha; }
    }

    const long out_base = ((long)batch * num_heads + head) * LATENT;
    for (int i = 0; i < VPL; ++i) {
        out[out_base + lane + 32 * i] =
            l == 0.0f ? mla_store_bf16<OT>(0.0f) : mla_store_bf16<OT>(acc[i] / l);
    }
}

// Dense-causal prefill FA support: decode a list of 584-byte fp8 cache
// slots into a contiguous half [n, 512] scratch (the decode kernels' exact
// fp32 materialization rounded to half — fp8 values scale-multiplied are
// exactly representable in half at serving scales, bf16 rope -> half is
// exact in range). One simdgroup per slot; -1 slots write zeros.
kernel void mla_prefill_dequant_slots(
        device const uchar *cache   [[buffer(0)]],
        device const int *slots     [[buffer(1)]],  // (n)
        device half *out            [[buffer(2)]],  // (n, 512)
        constant int &block_size    [[buffer(3)]],
        constant long &block_stride [[buffer(4)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    constexpr int SLOT_BYTES = 584, DATA_BYTES = 576;
    const int j = (int)tgid.x;
    const int slot = slots[j];
    device half *dst = out + (long)j * LATENT;
    if (slot < 0) {
        for (int i = 0; i < VPL; ++i) { dst[lane + 32 * i] = half(0.0f); }
        return;
    }
    const long base = (slot / block_size) * block_stride +
                      (slot % block_size) * SLOT_BYTES;
    device const bf16 *rope = (device const bf16 *)(cache + base + NOPE);
    for (int i = 0; i < VPL; ++i) {
        const int d = lane + 32 * i;
        float v;
        if (d < NOPE) {
            const int e = (int)cache[base + DATA_BYTES + d / 64];
            v = float(tk_e4m3_decode(cache[base + d])) *
                mla_ue8m0_scale(e);
        } else {
            v = float(rope[d - NOPE]);
        }
        dst[d] = half(v);
    }
}

// Dense-causal prefill FA (simdgroup-MMA): one threadgroup = 8 query
// tokens x 1 head over a single request slice. Phase A walks the shared
// compressed-candidate prefix with per-row causal lens; phase B walks the
// SWA band axis with per-row [lo, hi) masks; both axes are pre-decoded
// half [n, 512] scratches (K == V == the 512 latent). Per 32-column
// block: the 4 simdgroups each accumulate an S partial over their 128-dim
// k slice via MMA (K^T loaded transposed straight from the scratch, no
// staging), a TG-wide reduce + per-row online-softmax stats produce a
// half P tile and per-row alphas, then each simdgroup rescales its
// register O slice with a diagonal-matrix MMA (ggml flash_attn trick)
// and accumulates P.V over its 128-dim V slice. ULP class vs the decode
// walk (P rounds to half; block-level reassociation).
//
// Contract: tail 32-column blocks still load full 8x8 K/V fragments before
// jn masks the scores, so kc/ks must be padded to a 32-row multiple plus a
// spare block (the SlimServe host pads the dequant slot lists with -1 —
// metal.py _pad_slots — and deepseek_v4_prefill_fa checks nc/ns % 32). The dequant
// kernel zero-fills -1-slot rows, and every score a padded row produces is
// masked (jn) before the softmax.
template <typename OT>
kernel void mla_prefill_fa_mma(
        device const bf16 *q       [[buffer(0)]],   // (T, H, 512) slice
        device const half *kc      [[buffer(1)]],   // (nc, 512)
        device const half *ks      [[buffer(2)]],   // (ns, 512)
        device const int *lens_c   [[buffer(3)]],   // (T) causal cols in kc
        device const int *lo_s     [[buffer(4)]],   // (T) band start in ks
        device const int *hi_s     [[buffer(5)]],   // (T) band end in ks
        device const float *sinks  [[buffer(6)]],   // (H)
        device OT *out             [[buffer(7)]],   // (T, H, 512)
        constant int &T            [[buffer(8)]],
        constant int &num_heads    [[buffer(9)]],
        constant int &nc           [[buffer(10)]],
        constant int &ns           [[buffer(11)]],
        constant float &scale      [[buffer(12)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr int QT = 8;    // query tokens per threadgroup
    constexpr int C = 32;    // columns per block
    constexpr int DK = 512;
    constexpr int NSG = 4;   // simdgroups; each owns a 128-dim slice
    constexpr int SLICE = DK / NSG;  // 128

    const int t0 = (int)tgid.x * QT;
    const int h = (int)tgid.y;

    threadgroup half sq[QT * DK];          // 8 KB staged queries
    threadgroup float sS[NSG * QT * C];    // 4 KB score partials
    threadgroup half sP[QT * C];           // 512 B probability tile
    threadgroup float sAlpha[QT];
    threadgroup float sM[QT];
    threadgroup float sL[QT];
    threadgroup float sDiag[8 * 8];        // 256 B diagonal for rescale

    // Stage queries (half; bf16 -> half is exact in range).
    for (int idx = (int)tiitg; idx < QT * DK; idx += 128) {
        const int r = idx / DK, d = idx % DK;
        const int t = t0 + r;
        sq[idx] = t < T
                      ? half(float(q[((long)t * num_heads + h) * DK + d]))
                      : half(0.0f);
    }
    if (tiitg < QT) {
        sM[tiitg] = -3.4028234663852886e38f;
        sL[tiitg] = 0.0f;
    }
    if (tiitg < 64) { sDiag[tiitg] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // O slice accumulator: [QT x SLICE] fp32 = 16 8x8 frags.
    simdgroup_float8x8 O[SLICE / 8];
    #pragma clang loop unroll(full)
    for (short i = 0; i < SLICE / 8; ++i) {
        O[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    }

    // Tile-wide column ranges. Causal lens are monotonic within the slice.
    const int t_last = metal::min(t0 + QT - 1, T - 1);
    const int cmax = metal::min(lens_c[t_last], nc);
    const int s_lo = metal::max(lo_s[t0], 0);
    const int s_hi = metal::min(hi_s[t_last], ns);

    for (int phase = 0; phase < 2; ++phase) {
        device const half *kv = phase == 0 ? kc : ks;
        const int c_begin = phase == 0 ? 0 : s_lo;
        const int c_end = phase == 0 ? cmax : s_hi;
        for (int jb = c_begin; jb < c_end; jb += C) {
            const int jn = metal::min(c_end - jb, C);

            // --- S partial over this simdgroup's 128-dim k slice ---
            {
                simdgroup_float8x8 Sf[C / 8];
                #pragma clang loop unroll(full)
                for (short cfr = 0; cfr < C / 8; ++cfr) {
                    Sf[cfr] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                }
                simdgroup_half8x8 qa;
                simdgroup_half8x8 kb;
                const int d0 = (int)sgitg * SLICE;
                #pragma clang loop unroll(full)
                for (short kk = 0; kk < SLICE / 8; ++kk) {
                    simdgroup_load(qa, sq + d0 + kk * 8, DK, 0, false);
                    #pragma clang loop unroll(full)
                    for (short cfr = 0; cfr < C / 8; ++cfr) {
                        // transposed load: K^T tile from row-major kv
                        simdgroup_load(
                            kb, kv + (long)(jb + cfr * 8) * DK + d0 + kk * 8,
                            DK, 0, true);
                        simdgroup_multiply_accumulate(Sf[cfr], qa, kb,
                                                      Sf[cfr]);
                    }
                }
                threadgroup float *dst = sS + (int)sgitg * QT * C;
                #pragma clang loop unroll(full)
                for (short cfr = 0; cfr < C / 8; ++cfr) {
                    simdgroup_store(Sf[cfr], dst + cfr * 8, C, 0, false);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- reduce partials, mask, per-row online stats, P tile ---
            for (int idx = (int)tiitg; idx < QT * C; idx += 128) {
                const int r = idx / C, c = idx % C;
                const int t = t0 + r;
                float v = sS[idx] + sS[QT * C + idx] + sS[2 * QT * C + idx] +
                          sS[3 * QT * C + idx];
                v *= scale;
                const int col = jb + c;
                bool ok = (t < T) && (c < jn);
                if (phase == 0) {
                    ok = ok && col < lens_c[t];
                } else {
                    ok = ok && col >= lo_s[t] && col < hi_s[t];
                }
                sS[idx] = ok ? v : -3.4028234663852886e38f;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tiitg < QT) {
                const int r = (int)tiitg;
                float tm = -3.4028234663852886e38f;
                for (int c = 0; c < C; ++c) {
                    tm = max(tm, sS[r * C + c]);
                }
                float alpha = 1.0f;
                if (tm == -3.4028234663852886e38f) {
                    // no live columns: P row zero, O unchanged
                    for (int c = 0; c < C; ++c) {
                        sP[r * C + c] = half(0.0f);
                    }
                } else {
                    const float new_m = max(sM[r], tm);
                    alpha = sL[r] == 0.0f ? 0.0f : exp(sM[r] - new_m);
                    float bsum = 0.0f;
                    for (int c = 0; c < C; ++c) {
                        const float sc = sS[r * C + c];
                        const float b =
                            sc == -3.4028234663852886e38f ? 0.0f
                                                          : exp(sc - new_m);
                        sP[r * C + c] = half(b);
                        bsum += b;
                    }
                    sL[r] = sL[r] * alpha + bsum;
                    sM[r] = new_m;
                }
                sDiag[r * 8 + r] = alpha;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // --- O rescale (diagonal MMA) + P.V accumulate ---
            {
                simdgroup_float8x8 Df;
                simdgroup_load(Df, sDiag, 8, 0, false);
                const simdgroup_float8x8 Z =
                    make_filled_simdgroup_matrix<float, 8, 8>(0.f);
                #pragma clang loop unroll(full)
                for (short o = 0; o < SLICE / 8; ++o) {
                    simdgroup_float8x8 t2;
                    simdgroup_multiply_accumulate(t2, Df, O[o], Z);
                    O[o] = t2;
                }
                simdgroup_half8x8 pf;
                simdgroup_half8x8 vf;
                const int d0 = (int)sgitg * SLICE;
                #pragma clang loop unroll(full)
                for (short cfr = 0; cfr < C / 8; ++cfr) {
                    simdgroup_load(pf, sP + cfr * 8, C, 0, false);
                    #pragma clang loop unroll(full)
                    for (short o = 0; o < SLICE / 8; ++o) {
                        simdgroup_load(
                            vf, kv + (long)(jb + cfr * 8) * DK + d0 + o * 8,
                            DK, 0, false);
                        simdgroup_multiply_accumulate(O[o], pf, vf, O[o]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // Sink fold + final 1/l normalization as one diagonal MMA.
    if (tiitg < QT) {
        const int r = (int)tiitg;
        float mr = sM[r], lr = sL[r];
        const float sink = sinks[h];
        float alpha = 1.0f;
        if (isfinite(sink)) {
            const float new_m = max(mr, sink);
            alpha = lr == 0.0f ? 0.0f : exp(mr - new_m);
            lr = lr * alpha + exp(sink - new_m);
        }
        sDiag[r * 8 + r] = lr == 0.0f ? 0.0f : alpha / lr;
        sL[r] = lr;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    {
        simdgroup_float8x8 Df;
        simdgroup_load(Df, sDiag, 8, 0, false);
        const simdgroup_float8x8 Z =
            make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        #pragma clang loop unroll(full)
        for (short o = 0; o < SLICE / 8; ++o) {
            simdgroup_float8x8 t2;
            simdgroup_multiply_accumulate(t2, Df, O[o], Z);
            O[o] = t2;
        }
    }

    // Write out through the shared staging (sS reused as [QT x SLICE]
    // f32, 4 KB): the four simdgroups take turns staging their V-dim
    // slice, then all 128 threads copy that slice's rows out.
    threadgroup float *sO = sS;
    for (short sl = 0; sl < NSG; ++sl) {
        if ((short)sgitg == sl) {
            #pragma clang loop unroll(full)
            for (short o = 0; o < SLICE / 8; ++o) {
                simdgroup_store(O[o], sO + o * 8, SLICE, 0, false);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int idx = (int)tiitg; idx < QT * SLICE; idx += 128) {
            const int r = idx / SLICE, d = idx % SLICE;
            const int t = t0 + r;
            if (t < T) {
                out[((long)t * num_heads + h) * DK + sl * SLICE + d] =
                    mla_store_bf16<OT>(sO[idx]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

#define instantiate_mla_prefill_fa_mma(OT, NAME)                              \
  template [[host_name(NAME)]] [[kernel]] void mla_prefill_fa_mma<OT>(         \
      device const bf16 *q [[buffer(0)]],                                      \
      device const half *kc [[buffer(1)]],                                     \
      device const half *ks [[buffer(2)]],                                     \
      device const int *lens_c [[buffer(3)]],                                  \
      device const int *lo_s [[buffer(4)]],                                    \
      device const int *hi_s [[buffer(5)]],                                    \
      device const float *sinks [[buffer(6)]],                                 \
      device OT *out [[buffer(7)]],                                            \
      constant int &T [[buffer(8)]],                                           \
      constant int &num_heads [[buffer(9)]],                                   \
      constant int &nc [[buffer(10)]],                                         \
      constant int &ns [[buffer(11)]],                                         \
      constant float &scale [[buffer(12)]],                                    \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint tiitg [[thread_index_in_threadgroup]],                              \
      uint sgitg [[simdgroup_index_in_threadgroup]]);

instantiate_mla_prefill_fa_mma(bf16, "mla_prefill_fa_mma");
instantiate_mla_prefill_fa_mma(half, "mla_prefill_fa_mma_out_half");

// Prefill-width twin of mla_decode_fp8_sparse_two_cache_packed. The decode
// kernel gives every (head, token) simdgroup its own serial candidate walk,
// so at chunk widths each 584-byte slot is re-read and fp8-decoded once
// per head. Here one 256-thread threadgroup owns 16 heads of one token:
// candidate slots are staged through threadgroup memory as the SAME fp32
// values the decode kernel materializes (8 slots x 512, 16 KB), then each
// simdgroup walks the staged tile for its two heads with the decode
// kernel's exact per-candidate order, lane-dim mapping (lane + 32*i),
// simd_sum tree, and online-softmax update — outputs are BIT-IDENTICAL to
// the fused decode kernel, per-token dequant drops 32x per threadgroup.
template <typename OT>
kernel void mla_prefill_fp8_sparse_two_cache_packed(
        device const bf16 *q             [[buffer(0)]],
        device const uchar *compressed   [[buffer(1)]],
        device const int *compressed_idx [[buffer(2)]],
        device const int *compressed_len [[buffer(3)]],
        device const uchar *swa          [[buffer(4)]],
        device const int *swa_idx        [[buffer(5)]],
        device const int *swa_len        [[buffer(6)]],
        device const float *sinks        [[buffer(7)]],
        device OT *out                   [[buffer(8)]],
        constant int &num_heads          [[buffer(9)]],
        constant int &compressed_width   [[buffer(10)]],
        constant int &swa_width          [[buffer(11)]],
        constant float &scale            [[buffer(12)]],
        constant int &compressed_block_size [[buffer(13)]],
        constant int &compressed_block_stride [[buffer(14)]],
        constant int &swa_block_size [[buffer(15)]],
        constant int &swa_block_stride [[buffer(16)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint tiitg   [[thread_index_in_threadgroup]],
        uint sgitg   [[simdgroup_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    constexpr int SLOT_BYTES = 584, DATA_BYTES = 576;
    constexpr int TILE = 8;      // staged slots per round
    constexpr int ROWS = 2;      // heads per simdgroup
    const int batch = (int)tgid.y;
    // head0 assumes exactly 256 threads / 8 simdgroups per threadgroup
    // (the launcher's fixed dispatch shape: grid.x = heads/16).
    const int head0 = (int)tgid.x * 16 + (int)sgitg * ROWS;

    threadgroup float sres[TILE][LATENT];  // 16 KB

    float qv[ROWS][VPL], acc[ROWS][VPL];
    float m[ROWS], l[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        const long q_base = ((long)batch * num_heads + head0 + r) * LATENT;
        for (int i = 0; i < VPL; ++i) {
            qv[r][i] = float(q[q_base + lane + 32 * i]);
            acc[r][i] = 0.0f;
        }
        m[r] = -3.4028234663852886e38f;
        l[r] = 0.0f;
    }

    // Both cache walks share the staging loop; PASS 0 = compressed list,
    // PASS 1 = SWA list, in the decode kernel's order. Each staged slot is
    // the decode kernel's exact fp32 materialization, and each row's
    // per-candidate walk keeps its order, lane-dim mapping, simd_sum tree
    // and online-softmax update — outputs BIT-IDENTICAL to the decode
    // kernel, with the fp8 decode shared across 16 heads instead of run
    // per head. No tile-level softmax / dual-candidate variants: both
    // measured slower (optimization_status 2026-08-13).
    for (int pass = 0; pass < 2; ++pass) {
        device const uchar *cache = pass == 0 ? compressed : swa;
        device const int *idx = pass == 0 ? compressed_idx : swa_idx;
        const int width = pass == 0 ? compressed_width : swa_width;
        const int count = pass == 0 ? compressed_len[batch] : swa_len[batch];
        const int bsize = pass == 0 ? compressed_block_size : swa_block_size;
        const long bstride =
            pass == 0 ? compressed_block_stride : swa_block_stride;
        for (int j0 = 0; j0 < count; j0 += TILE) {
            const int jn = metal::min(count - j0, TILE);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            {
                const int js = (int)tiitg / 32;        // staged slot 0..7
                const int d0 = ((int)tiitg % 32) * 16; // 16 dims per thread
                if (js < jn) {
                    const int slot = idx[batch * width + j0 + js];
                    if (slot >= 0) {
                        const long base = (slot / bsize) * bstride +
                                          (slot % bsize) * SLOT_BYTES;
                        device const bf16 *rope =
                            (device const bf16 *)(cache + base + NOPE);
                        for (int d = d0; d < d0 + 16; ++d) {
                            float v;
                            if (d < NOPE) {
                                const int e =
                                    (int)cache[base + DATA_BYTES + d / 64];
                                v = float(tk_e4m3_decode(cache[base + d])) *
                                    mla_ue8m0_scale(e);
                            } else {
                                v = float(rope[d - NOPE]);
                            }
                            sres[js][d] = v;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (int js = 0; js < jn; ++js) {
                const int slot = idx[batch * width + j0 + js];
                if (slot < 0) { continue; }
                threadgroup const float *kv = sres[js];
                for (int r = 0; r < ROWS; ++r) {
                    float partial = 0.0f;
                    float lat[VPL];
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < VPL; ++i) {
                        lat[i] = kv[lane + 32 * i];
                        partial += qv[r][i] * lat[i];
                    }
                    const float score = simd_sum(partial) * scale;
                    const float new_m = max(m[r], score);
                    const float alpha =
                        l[r] == 0.0f ? 0.0f : exp(m[r] - new_m);
                    const float beta = exp(score - new_m);
                    #pragma clang loop unroll(full)
                    for (int i = 0; i < VPL; ++i) {
                        acc[r][i] = acc[r][i] * alpha + beta * lat[i];
                    }
                    l[r] = l[r] * alpha + beta;
                    m[r] = new_m;
                }
            }
        }
    }

    for (int r = 0; r < ROWS; ++r) {
        const float sink = sinks[head0 + r];
        if (isfinite(sink)) {
            const float new_m = max(m[r], sink);
            const float alpha = l[r] == 0.0f ? 0.0f : exp(m[r] - new_m);
            l[r] = l[r] * alpha + exp(sink - new_m);
            for (int i = 0; i < VPL; ++i) { acc[r][i] *= alpha; }
        }
        const long out_base = ((long)batch * num_heads + head0 + r) * LATENT;
        for (int i = 0; i < VPL; ++i) {
            out[out_base + lane + 32 * i] =
                l[r] == 0.0f ? mla_store_bf16<OT>(0.0f)
                             : mla_store_bf16<OT>(acc[r][i] / l[r]);
        }
    }
}

#define instantiate_mla_prefill_fp8_sparse_two_cache_packed(OT, NAME)          \
  template [[host_name(NAME)]] [[kernel]] void                                 \
  mla_prefill_fp8_sparse_two_cache_packed<OT>(                                 \
      device const bf16 *q [[buffer(0)]],                                      \
      device const uchar *compressed [[buffer(1)]],                            \
      device const int *compressed_idx [[buffer(2)]],                          \
      device const int *compressed_len [[buffer(3)]],                          \
      device const uchar *swa [[buffer(4)]],                                   \
      device const int *swa_idx [[buffer(5)]],                                 \
      device const int *swa_len [[buffer(6)]],                                 \
      device const float *sinks [[buffer(7)]],                                 \
      device OT *out [[buffer(8)]],                                            \
      constant int &num_heads [[buffer(9)]],                                   \
      constant int &compressed_width [[buffer(10)]],                           \
      constant int &swa_width [[buffer(11)]],                                  \
      constant float &scale [[buffer(12)]],                                    \
      constant int &compressed_block_size [[buffer(13)]],                      \
      constant int &compressed_block_stride [[buffer(14)]],                    \
      constant int &swa_block_size [[buffer(15)]],                             \
      constant int &swa_block_stride [[buffer(16)]],                           \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint tiitg   [[thread_index_in_threadgroup]],                            \
      uint sgitg   [[simdgroup_index_in_threadgroup]],                         \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_mla_prefill_fp8_sparse_two_cache_packed(
    bf16, "mla_prefill_fp8_sparse_two_cache_packed");
instantiate_mla_prefill_fp8_sparse_two_cache_packed(
    half, "mla_prefill_fp8_sparse_two_cache_packed_out_half");

#define instantiate_mla_decode_fp8_sparse_two_cache_packed(OT, NAME)           \
  template [[host_name(NAME)]] [[kernel]] void                                 \
  mla_decode_fp8_sparse_two_cache_packed<OT>(                                  \
      device const bf16 *q [[buffer(0)]],                                      \
      device const uchar *compressed [[buffer(1)]],                            \
      device const int *compressed_idx [[buffer(2)]],                          \
      device const int *compressed_len [[buffer(3)]],                          \
      device const uchar *swa [[buffer(4)]],                                   \
      device const int *swa_idx [[buffer(5)]],                                 \
      device const int *swa_len [[buffer(6)]],                                 \
      device const float *sinks [[buffer(7)]],                                 \
      device OT *out [[buffer(8)]],                                            \
      constant int &num_heads [[buffer(9)]],                                   \
      constant int &compressed_width [[buffer(10)]],                           \
      constant int &swa_width [[buffer(11)]],                                  \
      constant float &scale [[buffer(12)]],                                    \
      constant int &compressed_block_size [[buffer(13)]],                      \
      constant int &compressed_block_stride [[buffer(14)]],                    \
      constant int &swa_block_size [[buffer(15)]],                             \
      constant int &swa_block_stride [[buffer(16)]],                           \
      uint3 tgid [[threadgroup_position_in_grid]],                             \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_mla_decode_fp8_sparse_two_cache_packed(
    bf16, "mla_decode_fp8_sparse_two_cache_packed");
instantiate_mla_decode_fp8_sparse_two_cache_packed(
    half, "mla_decode_fp8_sparse_two_cache_packed_out_half");

// Split-K twin of the serving two-cache sparse decode (VLLM_QC_MLA_SPLITK):
// grid gains a partition axis (H, B, P); partition p walks slice
// [p*partition_size, ...) of the VIRTUAL concatenation
// [compressed candidates ++ SWA candidates] with the same per-slot decode
// and online softmax, emitting paged-v2 partials (normalized acc, max
// logit, exp sum) for paged_attention_reduce<*, 512>. The attention sink
// is NOT applied here — the reduce owns it, exactly once per (batch,
// head). ULP class vs the fused kernel: the cross-partition LSE merge
// reassociates the softmax sum.
kernel void mla_decode_fp8_sparse_two_cache_packed_partition(
        device const bf16 *q             [[buffer(0)]],
        device const uchar *compressed   [[buffer(1)]],
        device const int *compressed_idx [[buffer(2)]],
        device const int *compressed_len [[buffer(3)]],
        device const uchar *swa          [[buffer(4)]],
        device const int *swa_idx        [[buffer(5)]],
        device const int *swa_len        [[buffer(6)]],
        device float *tmp_out            [[buffer(7)]],   // (B, H, P, 512)
        device float *max_logits         [[buffer(8)]],   // (B, H, P)
        device float *exp_sums           [[buffer(9)]],   // (B, H, P)
        constant int &num_heads          [[buffer(10)]],
        constant int &compressed_width   [[buffer(11)]],
        constant int &swa_width          [[buffer(12)]],
        constant float &scale            [[buffer(13)]],
        constant int &compressed_block_size [[buffer(14)]],
        constant int &compressed_block_stride [[buffer(15)]],
        constant int &swa_block_size [[buffer(16)]],
        constant int &swa_block_stride [[buffer(17)]],
        constant int &num_partitions [[buffer(18)]],
        constant int &partition_size [[buffer(19)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    constexpr int SLOT_BYTES = 584, DATA_BYTES = 576;
    constexpr float MLA_NEG_INF = -3.4028234663852886e38f;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part = (int)tgid.z;
    const long q_base = ((long)batch * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    for (int i = 0; i < VPL; ++i) {
        qv[i] = float(q[q_base + lane + 32 * i]);
        acc[i] = 0.0f;
    }
    float m = MLA_NEG_INF, l = 0.0f;

    const int clen = compressed_len[batch];
    const int total = clen + swa_len[batch];
    const int j_beg = part * partition_size;
    const int j_end = metal::min(total, j_beg + partition_size);
    for (int j = j_beg; j < j_end; ++j) {
        int slot;
        device const uchar *cache;
        int bs;
        long bstride;
        if (j < clen) {
            slot = compressed_idx[batch * compressed_width + j];
            cache = compressed;
            bs = compressed_block_size;
            bstride = compressed_block_stride;
        } else {
            slot = swa_idx[batch * swa_width + (j - clen)];
            cache = swa;
            bs = swa_block_size;
            bstride = swa_block_stride;
        }
        if (slot < 0) { continue; }
        const long base = (slot / bs) * bstride + (slot % bs) * SLOT_BYTES;
        device const bf16 *rope = (device const bf16 *)(cache + base + NOPE);
        float lat[VPL], partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const int e = (int)cache[base + DATA_BYTES + d / 64];
                lat[i] = float(tk_e4m3_decode(cache[base + d])) *
                         mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) {
            acc[i] = acc[i] * alpha + beta * lat[i];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    const long stat = ((long)batch * num_heads + head) * num_partitions + part;
    const long ob = stat * LATENT;
    for (int i = 0; i < VPL; ++i) {
        tmp_out[ob + lane + 32 * i] = l == 0.0f ? 0.0f : acc[i] / l;
    }
    if (lane == 0) {
        max_logits[stat] = l == 0.0f ? MLA_NEG_INF : m;
        exp_sums[stat] = l;
    }
}

// ---------------------------------------------------------------------------
// P4a-v2 / P4b-v2: partitioned variants of the fp8 dense/sparse decodes — the same
// sequence-partition upgrade that gave the bf16 mla_decode 1.7-6.3x (grid gains a partition
// axis; per-partition online-softmax partials are combined by paged_attention_reduce<bf16,512>).
// The dense kernel partitions the token range; the sparse kernel partitions the top-k INDEX LIST
// (indices are arbitrary token positions, so no block alignment is needed).
// ---------------------------------------------------------------------------
kernel void mla_decode_fp8_partition(
        device const bf16 *q            [[buffer(0)]],   // (B, N, 512)
        device const uchar *data_cache  [[buffer(1)]],   // (nb, bs, 576)
        device const uchar *scale_cache [[buffer(2)]],   // (nb, bs, 8)
        device const int  *block_table  [[buffer(3)]],
        device const int  *context_lens [[buffer(4)]],
        device float      *tmp_out      [[buffer(5)]],   // (B, N, P, 512)
        device float      *max_logits   [[buffer(6)]],   // (B, N, P)
        device float      *exp_sums     [[buffer(7)]],   // (B, N, P)
        constant int &block_size        [[buffer(8)]],
        constant int &block_table_stride [[buffer(9)]],
        constant float &scale           [[buffer(10)]],
        constant int &num_heads         [[buffer(11)]],
        constant int &num_partitions    [[buffer(12)]],
        constant int &partition_size    [[buffer(13)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    constexpr float MLA_NEG_INF = -3.4028234663852886e38f;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part = (int)tgid.z;
    const int context_len = context_lens[batch];
    const int t_beg = part * partition_size;
    const int t_end = min(context_len, t_beg + partition_size);
    const long q_base = ((long)batch * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    for (int i = 0; i < VPL; ++i) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }

    float m = MLA_NEG_INF, l = 0.0f;
    for (int t = t_beg; t < t_end; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long dslot = (long)block * block_size + slot;
        const long dbase = dslot * 576;
        const long sbase = dslot * 8;
        device const bf16 *rope = (device const bf16 *)(data_cache + dbase + NOPE);

        float lat[VPL];
        float partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const uchar code = data_cache[dbase + d];
                const int e = (int)scale_cache[sbase + d / 64];
                lat[i] = float(tk_e4m3_decode(code)) * mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) { acc[i] = acc[i] * alpha + beta * lat[i]; }
        l = l * alpha + beta;
        m = new_m;
    }

    const long stat = ((long)batch * num_heads + head) * num_partitions + part;
    const long ob = stat * LATENT;
    for (int i = 0; i < VPL; ++i) {
        tmp_out[ob + lane + 32 * i] = l == 0.0f ? 0.0f : acc[i] / l;
    }
    if (lane == 0) {
        max_logits[stat] = l == 0.0f ? MLA_NEG_INF : m;
        exp_sums[stat] = l;
    }
}

kernel void mla_decode_fp8_sparse_partition(
        device const bf16 *q            [[buffer(0)]],
        device const uchar *data_cache  [[buffer(1)]],
        device const uchar *scale_cache [[buffer(2)]],
        device const int  *block_table  [[buffer(3)]],
        device const int  *indices      [[buffer(4)]],   // (B, max_topk)
        device const int  *topk_length  [[buffer(5)]],   // (B,)
        device float      *tmp_out      [[buffer(6)]],   // (B, N, P, 512)
        device float      *max_logits   [[buffer(7)]],   // (B, N, P)
        device float      *exp_sums     [[buffer(8)]],   // (B, N, P)
        constant int &block_size        [[buffer(9)]],
        constant int &block_table_stride [[buffer(10)]],
        constant float &scale           [[buffer(11)]],
        constant int &num_heads         [[buffer(12)]],
        constant int &max_topk          [[buffer(13)]],
        constant int &num_partitions    [[buffer(14)]],
        constant int &partition_size    [[buffer(15)]],
        uint3 tgid [[threadgroup_position_in_grid]],
        uint  lane [[thread_index_in_simdgroup]]) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    constexpr float MLA_NEG_INF = -3.4028234663852886e38f;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int part = (int)tgid.z;
    const int len = topk_length[batch];
    const int j_beg = part * partition_size;               // partition of the top-k index list
    const int j_end = min(len, j_beg + partition_size);
    const long q_base = ((long)batch * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    for (int i = 0; i < VPL; ++i) { qv[i] = float(q[q_base + lane + 32 * i]); acc[i] = 0.0f; }

    float m = MLA_NEG_INF, l = 0.0f;
    for (int j = j_beg; j < j_end; ++j) {
        const int t = indices[batch * max_topk + j];
        if (t < 0) { continue; }
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long dslot = (long)block * block_size + slot;
        const long dbase = dslot * 576;
        const long sbase = dslot * 8;
        device const bf16 *rope = (device const bf16 *)(data_cache + dbase + NOPE);

        float lat[VPL];
        float partial = 0.0f;
        for (int i = 0; i < VPL; ++i) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const uchar code = data_cache[dbase + d];
                const int e = (int)scale_cache[sbase + d / 64];
                lat[i] = float(tk_e4m3_decode(code)) * mla_ue8m0_scale(e);
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VPL; ++i) { acc[i] = acc[i] * alpha + beta * lat[i]; }
        l = l * alpha + beta;
        m = new_m;
    }

    const long stat = ((long)batch * num_heads + head) * num_partitions + part;
    const long ob = stat * LATENT;
    for (int i = 0; i < VPL; ++i) {
        tmp_out[ob + lane + 32 * i] = l == 0.0f ? 0.0f : acc[i] / l;
    }
    if (lane == 0) {
        max_logits[stat] = l == 0.0f ? MLA_NEG_INF : m;
        exp_sums[stat] = l;
    }
}

// Single-buffer bf16 copy (clone-then-insert prologue for the MLA cache).
kernel void mla_cache_clone(device const bf16 *src [[buffer(0)]],
                            device bf16       *dst [[buffer(1)]],
                            constant ulong &n      [[buffer(2)]],
                            uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid < n) { dst[tid] = src[tid]; }
}

// Single-buffer uchar copy (clone prologue for the packed fp8 data/scale caches).
kernel void mla_cache_clone_u8(device const uchar *src [[buffer(0)]],
                               device uchar       *dst [[buffer(1)]],
                               constant ulong &n       [[buffer(2)]],
                               uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid < n) { dst[tid] = src[tid]; }
}
