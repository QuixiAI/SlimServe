#pragma once
// DeepSeek MLA decode (paged, MQA latent cache), CUDA/SM86 port of ThunderMittens
// kernels/mla core: mla_q_norm_rope (GPT-J interleaved RoPE, optional RMSNorm),
// mla_kv_insert_fp8 (V4 packed cache: 448 NoPE -> e4m3 with per-64 UE8M0 scales,
// 64 RoPE bf16; 576B data + 8B scale rows), mla_decode (bf16 absorb path:
// QK=LATENT+ROPE dot, LATENT-only accumulate), mla_decode_fp8 (dequant-on-read).
// Partitioned + sparse variants and the bf16 insert are the follow-up; the
// D=512 v2 reducer already exists in paged_attn_v2.cu.
//
// Build:
//   /usr/local/cuda/bin/nvcc mla.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -I../quant -o mla.out
#include "quant_formats.cuh"   // tmq::e4m3_decode
#include "tm_warp.cuh"
#include <cuda_bf16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using bf16 = __nv_bfloat16;

namespace tms {


// float -> e4m3 RNE encode (same as quant_rt's, local copy to keep this file standalone)
__device__ __forceinline__ uint8_t e4m3_encode(float x) {
    const unsigned sign = (x < 0.0f) ? 0x80u : 0x00u;
    float a = fabsf(x);
    if (!(a < 448.0f)) return uint8_t(sign | 0x7Eu);
    if (a < 0.0009765625f) return uint8_t(sign | unsigned(int(rintf(a * 512.0f))));
    int e; frexpf(a, &e);
    const int E = e - 1;
    if (E < -6) {
        int mant = int(rintf(a * 512.0f));
        if (mant >= 8) return uint8_t(sign | (1u << 3));
        return uint8_t(sign | unsigned(mant));
    }
    const float two_m = a / exp2f(float(E));
    int mant = int(rintf((two_m - 1.0f) * 8.0f));
    int exp = E + 7;
    if (mant >= 8) { mant = 0; exp += 1; }
    if (exp > 15 || (exp == 15 && mant > 6)) return uint8_t(sign | 0x7Eu);
    return uint8_t(sign | (unsigned(exp) << 3) | unsigned(mant));
}

// ---- fused Q path: optional RMSNorm over full head dim + GPT-J interleaved RoPE on the
// last rope_dim dims. One warp per (token, head); lane owns D/32 CONTIGUOUS elements so
// every interleaved pair lives in one lane. norm_mode: 0 none, 1 rms, 2 rms*weight. ----
template <int D>
__global__ void mla_q_norm_rope(const bf16* q, const bf16* cosb, const bf16* sinb,
                                const int* positions, bf16* out, int num_heads,
                                int nope_dim, int rope_dim, int norm_mode, float eps,
                                const bf16* norm_weight) {
    static_assert(D % 64 == 0, "head_dim must be divisible by 64");
    constexpr int PER_LANE = D / 32;
    const int row = blockIdx.x;                  // (token, head)
    const int token = row / num_heads;
    const int lane = threadIdx.x;
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;
    const int64_t base = int64_t(row) * D + lane * PER_LANE;

    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        #pragma unroll
        for (int k = 0; k < PER_LANE; k++) { const float v = float(q[base + k]); ss += v * v; }
        ss = warp_sum_f(ss);
        rms = rsqrtf(ss / float(D) + eps);
    }
    const int64_t wbase = int64_t(lane) * PER_LANE;
    const int64_t csbase = int64_t(pos) * rope_half;
    #pragma unroll
    for (int k = 0; k < PER_LANE; k += 2) {
        const int g0 = lane * PER_LANE + k;      // even global index (pair start)
        float v0 = float(q[base + k]) * rms;
        float v1 = float(q[base + k + 1]) * rms;
        if (norm_mode == 2) {
            v0 *= float(norm_weight[wbase + k]);
            v1 *= float(norm_weight[wbase + k + 1]);
        }
        if (g0 >= nope_dim) {                    // interleaved rotate within the pair
            const int p = (g0 - nope_dim) / 2;
            const float c = float(cosb[csbase + p]), s = float(sinb[csbase + p]);
            const float r0 = v0 * c - v1 * s, r1 = v0 * s + v1 * c;
            v0 = r0; v1 = r1;
        }
        out[base + k] = bf16(v0);
        out[base + k + 1] = bf16(v1);
    }
}

// ---- classic bf16 latent KV-insert (vLLM concat_and_cache_mla; TM mla.metal 102-161).
// One warp per token writes kv_cache (nb, bs, LATENT+rope_dim) — MQA, no head axis:
// the compressed latent (optionally kv_a-RMSNormed; norm_mode 0 none / 2 weighted) at
// [0:LATENT], interleaved-RoPE'd k_pe at [LATENT:+rope_dim]. Negative slot skips
// (clone-then-insert: caller pre-populates, this overwrites mapped slots only).
// LATENT % 64 == 0; rope_dim/2 <= 32 (one pair per lane). ----
template <int LATENT>
__global__ void mla_kv_insert(const bf16* kv_c, const bf16* k_pe, const bf16* cosb,
                              const bf16* sinb, const int* positions,
                              const int64_t* slot_mapping, bf16* kv_cache,
                              int block_size, int rope_dim, int norm_mode, float eps,
                              const bf16* norm_weight) {
    static_assert(LATENT % 64 == 0, "mla_kv_insert needs LATENT divisible by 64");
    constexpr int LPL = LATENT / 32;                 // latent elements per lane (even)
    const int token = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    (void)block_size;                                // slot*(row) == (block*bs+off)*(row)
    const int64_t dst = slot * int64_t(LATENT + rope_dim);
    const int pos = positions[token];
    const int rope_half = rope_dim / 2;

    const int64_t lbase = int64_t(token) * LATENT + int64_t(lane) * LPL;
    float rms = 1.0f;
    if (norm_mode != 0) {
        float ss = 0.0f;
        #pragma unroll
        for (int k = 0; k < LPL; ++k) { const float v = float(kv_c[lbase + k]); ss += v * v; }
        ss = warp_sum_f(ss);
        rms = rsqrtf(ss / float(LATENT) + eps);
    }
    #pragma unroll
    for (int k = 0; k < LPL; ++k) {
        float v = float(kv_c[lbase + k]) * rms;
        if (norm_mode == 2) v *= float(norm_weight[lane * LPL + k]);
        kv_cache[dst + lane * LPL + k] = bf16(v);
    }
    if (lane < rope_half) {
        const int64_t rbase = int64_t(token) * rope_dim + int64_t(lane) * 2;
        const float e = float(k_pe[rbase]);
        const float o = float(k_pe[rbase + 1]);
        const float c = float(cosb[int64_t(pos) * rope_half + lane]);
        const float s = float(sinb[int64_t(pos) * rope_half + lane]);
        kv_cache[dst + LATENT + lane * 2]     = bf16(e * c - o * s);
        kv_cache[dst + LATENT + lane * 2 + 1] = bf16(e * s + o * c);
    }
}

// ---- V4 packed insert: kv (T, 512) -> data_cache (nb, bs, 576) + scale_cache (nb, bs, 8).
// NoPE [0,448): e4m3 with per-64-elem UE8M0 power-of-2 scale; RoPE [448,512): GPT-J
// interleaved rotate, stored bf16. Lane owns 16 contiguous elems; 4 lanes = one 64-block. ----
__global__ void mla_kv_insert_fp8(const bf16* kv, const bf16* cosb, const bf16* sinb,
                                  const int* positions, const int64_t* slot_mapping,
                                  uint8_t* data_cache, uint8_t* scale_cache, int block_size) {
    constexpr int LAT = 512, NOPE = 448, PER_LANE = 16, NOPE_LANES = NOPE / PER_LANE;  // 28
    const int token = blockIdx.x, lane = threadIdx.x;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;
    const int64_t dslot = (slot / block_size) * block_size + slot % block_size;
    const int64_t dst_data = dslot * 576, dst_scale = dslot * 8;
    const int pos = positions[token];
    const int64_t kbase = int64_t(token) * LAT + lane * PER_LANE;

    float v[PER_LANE];
    #pragma unroll
    for (int k = 0; k < PER_LANE; k++) v[k] = float(kv[kbase + k]);

    float amax = 0.0f;
    #pragma unroll
    for (int k = 0; k < PER_LANE; k++) amax = fmaxf(amax, fabsf(v[k]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    const float exponent = ceilf(log2f(fmaxf(amax, 1e-4f) / 448.0f));
    const float inv_scale = exp2f(-exponent);

    if (lane < NOPE_LANES) {
        #pragma unroll
        for (int k = 0; k < PER_LANE; k++)
            data_cache[dst_data + lane * PER_LANE + k] = e4m3_encode(v[k] * inv_scale);
        if ((lane & 3) == 0) {
            const int e = min(max(int(exponent) + 127, 0), 255);
            scale_cache[dst_scale + lane / 4] = uint8_t(e);
        }
    } else {
        const int rl = (lane - NOPE_LANES) * PER_LANE;   // rope-local start: 0,16,32,48
        bf16* rope_out = reinterpret_cast<bf16*>(data_cache + dst_data + NOPE);
        #pragma unroll
        for (int j = 0; j < PER_LANE; j += 2) {
            const int p = (rl + j) / 2;
            const float c = float(cosb[int64_t(pos) * 32 + p]);
            const float s = float(sinb[int64_t(pos) * 32 + p]);
            rope_out[rl + j]     = bf16(v[j] * c - v[j + 1] * s);
            rope_out[rl + j + 1] = bf16(v[j] * s + v[j + 1] * c);
        }
    }
    if (lane == 0) scale_cache[dst_scale + 7] = 0;
}

// ---- bf16 absorb-path decode: score over QK=LATENT+ROPE, accumulate over LATENT only ----
template <int LATENT, int ROPE>
__global__ void mla_decode(const bf16* q, const bf16* kv_cache, const int* block_table,
                           const int* context_lens, bf16* out,
                           int block_size, int bt_stride, float scale, int num_heads) {
    constexpr int QK = LATENT + ROPE, VQK = QK / 32, VAV = LATENT / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int context_len = context_lens[batch];
    const int64_t q_base = (int64_t(batch) * num_heads + head) * QK;

    float qv[VQK], acc[VAV];
    #pragma unroll
    for (int i = 0; i < VQK; i++) qv[i] = float(q[q_base + lane + 32 * i]);
    #pragma unroll
    for (int i = 0; i < VAV; i++) acc[i] = 0.0f;
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * QK;   // MQA: no head axis
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VQK; i++) partial += qv[i] * float(kv_cache[base + lane + 32 * i]);
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VAV; i++)
            acc[i] = acc[i] * alpha + beta * float(kv_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    const int64_t out_base = (int64_t(batch) * num_heads + head) * LATENT;
    #pragma unroll
    for (int i = 0; i < VAV; i++)
        out[out_base + lane + 32 * i] = (l == 0.0f) ? bf16(0.0f) : bf16(acc[i] / l);
}

// ---- V4 fp8 decode: dequant-on-read over the packed 576/8 cache, score+value over 512 ----
__global__ void mla_decode_fp8(const bf16* q, const uint8_t* data_cache, const uint8_t* scale_cache,
                               const int* block_table, const int* context_lens, bf16* out,
                               int block_size, int bt_stride, float scale, int num_heads) {
    constexpr int LATENT = 512, NOPE = 448, VPL = LATENT / 32;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int context_len = context_lens[batch];
    const int64_t q_base = (int64_t(batch) * num_heads + head) * LATENT;

    float qv[VPL], acc[VPL];
    #pragma unroll
    for (int i = 0; i < VPL; i++) qv[i] = float(q[q_base + lane + 32 * i]);
    #pragma unroll
    for (int i = 0; i < VPL; i++) acc[i] = 0.0f;
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = 0; t < context_len; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t dslot = int64_t(block) * block_size + slot;
        const int64_t dbase = dslot * 576, sbase = dslot * 8;
        const bf16* rope = reinterpret_cast<const bf16*>(data_cache + dbase + NOPE);

        float lat[VPL], partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            if (d < NOPE) {
                const int e = int(scale_cache[sbase + d / 64]);
                lat[i] = tmq::e4m3_decode(data_cache[dbase + d]) * exp2f(float(e - 127));
            } else {
                lat[i] = float(rope[d - NOPE]);
            }
            partial += qv[i] * lat[i];
        }
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++) acc[i] = acc[i] * alpha + beta * lat[i];
        l = l * alpha + beta;
        m = nm;
    }
    const int64_t out_base = (int64_t(batch) * num_heads + head) * LATENT;
    #pragma unroll
    for (int i = 0; i < VPL; i++)
        out[out_base + lane + 32 * i] = (l == 0.0f) ? bf16(0.0f) : bf16(acc[i] / l);
}

// ---- bf16 partitioned decode (TM mla.metal 338-421): grid gains a partition
// axis (H, B, P); per-partition online-softmax partials (tmp_out normalized,
// max_logits/exp_sums stats) combine via paged_attention_reduce<bf16, LATENT>.
// Empty partitions emit NEG_INF/0 so the reducer skips them. ----
template <int LATENT, int ROPE>
__global__ void mla_decode_partition(const bf16* q, const bf16* kv_cache,
                                     const int* block_table, const int* context_lens,
                                     float* tmp_out, float* max_logits, float* exp_sums,
                                     int block_size, int bt_stride, float scale,
                                     int num_heads, int num_partitions, int partition_size) {
    constexpr int QK = LATENT + ROPE, VQK = QK / 32, VAV = LATENT / 32;
    constexpr float MLA_NEG_INF = -3.4028234663852886e38f;
    const int head = blockIdx.x, batch = blockIdx.y, part = blockIdx.z, lane = threadIdx.x;
    const int context_len = context_lens[batch];
    const int t_beg = part * partition_size;
    const int t_end = min(context_len, t_beg + partition_size);
    const int64_t q_base = (int64_t(batch) * num_heads + head) * QK;

    float qv[VQK], acc[VAV];
    #pragma unroll
    for (int i = 0; i < VQK; i++) qv[i] = float(q[q_base + lane + 32 * i]);
    #pragma unroll
    for (int i = 0; i < VAV; i++) acc[i] = 0.0f;
    float m = MLA_NEG_INF, l = 0.0f;

    for (int t = t_beg; t < t_end; t++) {
        const int col = t / block_size, slot = t - col * block_size;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t base = (int64_t(block) * block_size + slot) * QK;
        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < VQK; i++) partial += qv[i] * float(kv_cache[base + lane + 32 * i]);
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VAV; i++)
            acc[i] = acc[i] * alpha + beta * float(kv_cache[base + lane + 32 * i]);
        l = l * alpha + beta;
        m = nm;
    }
    const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
    const int64_t ob = stat * LATENT;
    #pragma unroll
    for (int i = 0; i < VAV; i++)
        tmp_out[ob + lane + 32 * i] = (l == 0.0f) ? 0.0f : acc[i] / l;
    if (lane == 0) {
        max_logits[stat] = (l == 0.0f) ? MLA_NEG_INF : m;
        exp_sums[stat] = l;
    }
}

// ---- V4 fp8 decode, generalized (TM mla.metal 497-718): SPARSE walks the
// caller's top-k index list indices[batch, 0:topk_length[batch]] (entries < 0
// skipped — Lightning Indexer gather); PART adds the partition axis (dense
// partitions the token range, sparse partitions the INDEX LIST) and emits v2
// partials for paged_attention_reduce<bf16,512>. <false,false> is the plain
// dense decode (mla_decode_fp8 kept above as the validated original). ----
// Geometry is templated so one kernel serves several MLA shapes:
//   QW   score width  = q width = elements per cache slot
//   VW   value width  = output width (the leading VW dims are the value; any
//        trailing QW-VW dims contribute to the score only, e.g. a rope tail)
//   NFP8 leading elements stored as fp8; the remaining QW-NFP8 are bf16
//   SMODE 0 = per-64 e8 exponent byte from scale_cache (original)
//         1 = single per-tensor scale passed in kv_scale (vLLM fp8 MLA cache)
// Defaults reproduce the original 512/512/448/e8 instantiation exactly.
//   DeepSeek-style : <SPARSE, PART>                       448 fp8 + 64 bf16 rope
//   GLM-5.2-Vision : <SPARSE, PART, 576, 512, 576, 1>     576 fp8, value = 512
template <bool SPARSE, bool PART, int QW = 512, int VW = 512, int NFP8 = 448,
          int SMODE = 0>
__global__ void mla_decode_fp8_v(const bf16* q, const uint8_t* data_cache,
                                 const uint8_t* scale_cache, const int* block_table,
                                 const int* context_lens,                  // dense
                                 const int* indices, const int* topk_length, int max_topk,  // sparse
                                 bf16* out,                                // !PART
                                 float* tmp_out, float* max_logits, float* exp_sums,  // PART
                                 int block_size, int bt_stride, float scale, int num_heads,
                                 int num_partitions, int partition_size,
                                 float kv_scale = 1.0f,
                                 // Split-q: when q_pe2 != nullptr, `q` holds the nope part
                                 // in HEAD-MAJOR [H, B, VW] layout (the natural bmm output,
                                 // pre-transpose) and q_pe2 the rope tail [B, H, QW-VW] with
                                 // pe_stride elements between heads (>= QW-VW: a strided
                                 // split view is read in place). Reads the two source
                                 // buffers directly instead of a per-layer torch.cat;
                                 // values and order are identical.
                                 const bf16* q_pe2 = nullptr, int pe_stride = 0) {
    constexpr int NOPE = NFP8, VPL = VW / 32, QPL = QW / 32;
    constexpr int SLOT_BYTES = NFP8 + 2 * (QW - NFP8);
    static_assert(QW % 32 == 0 && VW % 32 == 0 && VW <= QW, "bad MLA geometry");
    constexpr float MLA_NEG_INF = -3.4028234663852886e38f;
    const int head = blockIdx.x, batch = blockIdx.y, lane = threadIdx.x;
    const int part = PART ? blockIdx.z : 0;
    const int len = SPARSE ? topk_length[batch] : context_lens[batch];
    // Balance the split by the request's ACTUAL length, not the fixed
    // partition_size: with j_beg = part * partition_size every context
    // shorter than partition_size lands entirely in partition 0 and the
    // other P-1 warps exit -- at short contexts that is one warp per
    // (head, token) again, exactly what the partition axis was added to
    // avoid. ceil(len / num_partitions) keeps all P partitions busy at
    // every length; the reduce is boundary-agnostic.
    const int part_span =
        PART ? (len + num_partitions - 1) / num_partitions : len;
    const int j_beg = PART ? part * part_span : 0;
    const int j_end = PART ? min(len, j_beg + part_span) : len;
    const int64_t q_base = (int64_t(batch) * num_heads + head) * QW;
    const bool splitq = (q_pe2 != nullptr);
    const int64_t qn_base = (int64_t(head) * gridDim.y + batch) * VW;
    const int64_t pe_base = (int64_t(batch) * num_heads + head) * pe_stride;

    // All-fp8 slots (NFP8 == QW, the GLM geometry) take a vectorized path:
    // the scalar loop reads the row as QPL rounds of one-byte-per-lane loads
    // -- each round a 32-byte uncoalesced transaction at full memory latency,
    // measured 6.75 us per row. Four bytes per lane per round turns that into
    // QW/128 coalesced 128-byte rounds, and dims [0, VW) land exactly in the
    // full rounds while the tail round is rope-only score work.
    constexpr bool VECFP8 = (NFP8 == QW) && (QW % 4 == 0) && (VW % 128 == 0);
    constexpr int VR = VW / 128;              // full 4-byte rounds (V dims)
    constexpr int TAIL = (QW - VW) / 64;      // 2-byte tail rounds (rope dims)

    float qv[QPL], acc[VPL];
    if constexpr (VECFP8) {
        #pragma unroll
        for (int i = 0; i < VR; i++) {
            const bf16* src = splitq ? &q[qn_base + 4 * lane + 128 * i]
                                     : &q[q_base + 4 * lane + 128 * i];
            const uint2 qw = *reinterpret_cast<const uint2*>(src);
            const bf16* qb = reinterpret_cast<const bf16*>(&qw);
            #pragma unroll
            for (int k = 0; k < 4; k++) qv[4 * i + k] = float(qb[k]);
        }
        #pragma unroll
        for (int i = 0; i < TAIL; i++) {
            const bf16* src = splitq ? &q_pe2[pe_base + 2 * lane + 64 * i]
                                     : &q[q_base + VW + 2 * lane + 64 * i];
            const uint32_t qw = *reinterpret_cast<const uint32_t*>(src);
            const bf16* qb = reinterpret_cast<const bf16*>(&qw);
            qv[4 * VR + 2 * i] = float(qb[0]);
            qv[4 * VR + 2 * i + 1] = float(qb[1]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < QPL; i++) {
            const int d = lane + 32 * i;
            qv[i] = float(splitq ? (d < VW ? q[qn_base + d]
                                           : q_pe2[pe_base + d - VW])
                                 : q[q_base + d]);
        }
    }
    #pragma unroll
    for (int i = 0; i < VPL; i++) acc[i] = 0.0f;
    float m = MLA_NEG_INF, l = 0.0f;

    for (int j = j_beg; j < j_end; j++) {
        const int t = SPARSE ? indices[batch * max_topk + j] : j;
        if (SPARSE && t < 0) continue;
        const int col = t / block_size, slot = t - col * block_size;
        // A sparse index list is caller-supplied and can be ragged or stale
        // (padding, a profiling pass, a request shorter than max_topk). Guard the
        // block-table read as well as its contents; without this an index past
        // the request's table walks off the end and then dereferences the cache
        // with a garbage block id.
        if (col < 0 || col >= bt_stride) continue;
        const int block = block_table[batch * bt_stride + col];
        if (block < 0) continue;
        const int64_t dslot = int64_t(block) * block_size + slot;
        const int64_t dbase = dslot * SLOT_BYTES, sbase = dslot * (NFP8 / 64);
        const bf16* rope = reinterpret_cast<const bf16*>(data_cache + dbase + NOPE);

        float lat[QPL], partial = 0.0f;
        if constexpr (VECFP8) {
            #pragma unroll
            for (int i = 0; i < VR; i++) {
                const uint32_t w = *reinterpret_cast<const uint32_t*>(
                    data_cache + dbase + 4 * lane + 128 * i);
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const int d = 4 * lane + 128 * i + k;
                    float dq = tmq::e4m3_decode(uint8_t(w >> (8 * k)));
                    if (SMODE == 0) {
                        const int e = int(scale_cache[sbase + d / 64]);
                        dq *= exp2f(float(e - 127));
                    } else {
                        dq *= kv_scale;
                    }
                    lat[4 * i + k] = dq;
                    partial += qv[4 * i + k] * dq;
                }
            }
            #pragma unroll
            for (int i = 0; i < TAIL; i++) {
                const uint16_t w = *reinterpret_cast<const uint16_t*>(
                    data_cache + dbase + VW + 2 * lane + 64 * i);
                #pragma unroll
                for (int k = 0; k < 2; k++) {
                    const int d = VW + 2 * lane + 64 * i + k;
                    float dq = tmq::e4m3_decode(uint8_t(w >> (8 * k)));
                    if (SMODE == 0) {
                        const int e = int(scale_cache[sbase + d / 64]);
                        dq *= exp2f(float(e - 127));
                    } else {
                        dq *= kv_scale;
                    }
                    lat[4 * VR + 2 * i + k] = dq;
                    partial += qv[4 * VR + 2 * i + k] * dq;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < QPL; i++) {
                const int d = lane + 32 * i;
                if (d < NOPE) {
                    const float dq = tmq::e4m3_decode(data_cache[dbase + d]);
                    if (SMODE == 0) {
                        const int e = int(scale_cache[sbase + d / 64]);
                        lat[i] = dq * exp2f(float(e - 127));
                    } else {
                        lat[i] = dq * kv_scale;
                    }
                } else {
                    lat[i] = float(rope[d - NOPE]);
                }
                partial += qv[i] * lat[i];
            }
        }
        const float score = warp_sum_f(partial) * scale;
        const float nm = fmaxf(m, score);
        const float alpha = (l == 0.0f) ? 0.0f : expf(m - nm);
        const float beta = expf(score - nm);
        #pragma unroll
        for (int i = 0; i < VPL; i++) acc[i] = acc[i] * alpha + beta * lat[i];
        l = l * alpha + beta;
        m = nm;
    }
    if (PART) {
        const int64_t stat = (int64_t(batch) * num_heads + head) * num_partitions + part;
        // Empty partitions write only their skip sentinel: the reduce never
        // reads tmp_out for a partition whose max_logit is NEG_INF, and at
        // short sparse contexts most partitions are empty -- zero-filling
        // their 512-float tiles was pure wasted store traffic.
        if (l == 0.0f) {
            if (lane == 0) {
                max_logits[stat] = MLA_NEG_INF;
                exp_sums[stat] = 0.0f;
            }
            return;
        }
        const int64_t ob = stat * VW;
        if constexpr (VECFP8) {
            // acc[4i+k] holds dim 4*lane + 128*i + k; store float4 to the
            // canonical layout so the unchanged reduce kernel reads it as-is.
            #pragma unroll
            for (int i = 0; i < VR; i++) {
                float4 v;
                v.x = acc[4 * i] / l;     v.y = acc[4 * i + 1] / l;
                v.z = acc[4 * i + 2] / l; v.w = acc[4 * i + 3] / l;
                *reinterpret_cast<float4*>(&tmp_out[ob + 4 * lane + 128 * i]) = v;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VPL; i++)
                tmp_out[ob + lane + 32 * i] = acc[i] / l;
        }
        if (lane == 0) {
            max_logits[stat] = m;
            exp_sums[stat] = l;
        }
    } else {
        const int64_t out_base = (int64_t(batch) * num_heads + head) * VW;
        if constexpr (VECFP8) {
            #pragma unroll
            for (int i = 0; i < VR; i++) {
                bf16 v[4];
                #pragma unroll
                for (int k = 0; k < 4; k++)
                    v[k] = (l == 0.0f) ? bf16(0.0f) : bf16(acc[4 * i + k] / l);
                *reinterpret_cast<uint2*>(&out[out_base + 4 * lane + 128 * i]) =
                    *reinterpret_cast<const uint2*>(v);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < VPL; i++)
                out[out_base + lane + 32 * i] =
                    (l == 0.0f) ? bf16(0.0f) : bf16(acc[i] / l);
        }
    }
}

// tlen[row] = last valid (>= 0) position + 1 in the top-k index list, in one
// launch. Replaces the (idx >= 0) * arange -> amax ATen chain, which cost
// three kernels per layer per decode step. One block per row.
__global__ void sparse_topk_tlen(const int* __restrict__ idx,
                                 int* __restrict__ tlen, int topk) {
    const int row = blockIdx.x;
    const int* r = idx + (size_t)row * topk;
    int m = 0;
    if (topk % 4 == 0) {
        const int4* r4 = reinterpret_cast<const int4*>(r);
        for (int p = threadIdx.x; p < topk / 4; p += blockDim.x) {
            const int4 v = r4[p];
            const int last = v.w >= 0 ? 4 : v.z >= 0 ? 3 : v.y >= 0 ? 2
                             : v.x >= 0 ? 1 : 0;
            if (last) m = max(m, 4 * p + last);
        }
    } else {
        for (int p = threadIdx.x; p < topk; p += blockDim.x)
            if (r[p] >= 0) m = max(m, p + 1);
    }
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        m = max(m, __shfl_xor_sync(0xffffffff, m, mask));
    __shared__ int warp_max[8];
    if ((threadIdx.x & 31) == 0) warp_max[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        int v = threadIdx.x < (blockDim.x >> 5) ? warp_max[threadIdx.x] : 0;
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1)
            v = max(v, __shfl_xor_sync(0xffffffff, v, mask));
        if (threadIdx.x == 0) tlen[row] = v;
    }
}

}  // namespace tms

