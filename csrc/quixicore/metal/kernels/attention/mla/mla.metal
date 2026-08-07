#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

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
// ---------------------------------------------------------------------------
template <int D>
kernel void mla_q_norm_rope(device const bf16 *q          [[buffer(0)]],
                            device const bf16 *cosb        [[buffer(1)]],
                            device const bf16 *sinb        [[buffer(2)]],
                            device const int  *positions   [[buffer(3)]],
                            device bf16       *out         [[buffer(4)]],
                            constant int &num_heads        [[buffer(5)]],
                            constant int &nope_dim         [[buffer(6)]],
                            constant int &rope_dim         [[buffer(7)]],
                            constant int &norm_mode        [[buffer(8)]],   // 0 none,1 rms,2 rms+w
                            constant float &eps            [[buffer(9)]],
                            device const bf16 *norm_weight [[buffer(10)]],  // (D,), read iff mode 2
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
        for (int k = 0; k < PER_LANE; ++k) { const float v = float(q[base + k]); ss += v * v; }
        ss = simd_sum(ss);
        rms = metal::rsqrt(ss / (float)D + eps);
    }

    const long wbase = (long)laneId * PER_LANE;   // norm_weight index for this lane's chunk
    const long csbase = (long)pos * rope_half;
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = (int)laneId * PER_LANE + k;   // even global index (start of a pair)
        float v0 = float(q[base + k]) * rms;
        float v1 = float(q[base + k + 1]) * rms;
        if (norm_mode == 2) {
            v0 *= float(norm_weight[wbase + k]);
            v1 *= float(norm_weight[wbase + k + 1]);
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

#define instantiate_mla_q_norm_rope(DVAL)                                      \
  template [[host_name("mla_q_norm_rope_" #DVAL)]] [[kernel]] void             \
  mla_q_norm_rope<DVAL>(device const bf16 *q [[buffer(0)]],                    \
                        device const bf16 *cosb [[buffer(1)]],                 \
                        device const bf16 *sinb [[buffer(2)]],                 \
                        device const int  *positions [[buffer(3)]],            \
                        device bf16       *out [[buffer(4)]],                  \
                        constant int &num_heads [[buffer(5)]],                 \
                        constant int &nope_dim [[buffer(6)]],                  \
                        constant int &rope_dim [[buffer(7)]],                  \
                        constant int &norm_mode [[buffer(8)]],                 \
                        constant float &eps [[buffer(9)]],                     \
                        device const bf16 *norm_weight [[buffer(10)]],         \
                        uint3 blockIdx [[threadgroup_position_in_grid]],       \
                        uint  laneId   [[thread_index_in_simdgroup]]);

instantiate_mla_q_norm_rope(128);
instantiate_mla_q_norm_rope(192);
instantiate_mla_q_norm_rope(256);
instantiate_mla_q_norm_rope(512);

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
    const float inv_scale = metal::exp2(-exponent);

    if ((int)laneId < NOPE_LANES) {
        for (int k = 0; k < PER_LANE; ++k) {
            data_cache[dst_data + laneId * PER_LANE + k] = tk_e4m3_encode(v[k] * inv_scale);
        }
        if ((laneId & 3) == 0) {   // first lane of each 4-lane (64-elem) block writes its scale byte
            const int e = metal::clamp((int)exponent + 127, 0, 255);
            scale_cache[dst_scale + laneId / 4] = (uchar)e;
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
                lat[i] = float(tk_e4m3_decode(code)) * metal::exp2((float)(e - 127));
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
                lat[i] = float(tk_e4m3_decode(code)) * metal::exp2((float)(e - 127));
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
                lat[i] = float(tk_e4m3_decode(code)) * metal::exp2((float)(e - 127));
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
                lat[i] = float(tk_e4m3_decode(code)) * metal::exp2((float)(e - 127));
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
