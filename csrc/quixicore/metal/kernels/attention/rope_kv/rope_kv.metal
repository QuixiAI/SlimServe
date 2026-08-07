#include "tk.metal"
#include <metal_stdlib>

namespace mittens {

// ---------------------------------------------------------------------------
// Fused RoPE + paged-KV insert, bf16 I/O, fp32 compute.
//
// A serving prefill/decode step rotates the new K (split-half / GPT-NeoX RoPE,
// matching mx.fast.rope(traditional=False)) and writes the rotated K together
// with V straight into the paged KV cache at slot_mapping[token] — fusing what
// was three separate kernels (rotary + two cache scatters).
//
//   k1 = k[:D/2], k2 = k[D/2:]                  (per (token, kv_head) row)
//   ko1 = k1*cos - k2*sin ;  ko2 = k2*cos + k1*sin
//   key_cache[slot, kv_head]   = [ko1, ko2]     (rotated)
//   value_cache[slot, kv_head] = v              (unrotated)
//
// cos/sin are precomputed (P, D/2) and indexed by positions[token]. The paged
// cache (num_blocks, block_size, num_kv_heads, D) is contiguous, so it flattens
// to (R, D) and the destination row is
//   dst = (block * block_size + block_offset) * num_kv_heads + kv_head.
//
// One simdgroup (32 lanes) handles one (token, kv_head) row, flattened to
// (M = num_tokens * num_kv_heads, D). slot_mapping[token] < 0 skips the token.
// Q-RoPE is the existing `rotary` kernel (Q is not written to the cache).
//
// Ref: vLLM fused_*_qknorm_rope_kv_insert kernels (warp = one (token, head-slot)).
// ---------------------------------------------------------------------------
template <typename T, int D>
kernel void rope_kv_insert(device   T    *k            [[buffer(0)]],
                           device   T    *v            [[buffer(1)]],
                           device   T    *cosb         [[buffer(2)]],
                           device   T    *sinb         [[buffer(3)]],
                           device   const int  *positions    [[buffer(4)]],
                           device   const long *slot_mapping [[buffer(5)]],
                           device   T    *key_cache    [[buffer(6)]],
                           device   T    *value_cache  [[buffer(7)]],
                           constant int  &num_kv_heads [[buffer(8)]],
                           constant int  &block_size   [[buffer(9)]],
                           uint3 blockIdx [[threadgroup_position_in_grid]],
                           uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");

    const int row = blockIdx.x;                  // (token, kv_head) flattened
    const int token = row / num_kv_heads;
    const int kv_head = row % num_kv_heads;

    const long slot = slot_mapping[token];
    if (slot < 0) {
        return;                                  // padding token — skip
    }
    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const long dst_row =
        (block * block_size + block_offset) * (long)num_kv_heads + (long)kv_head;
    const int pos = positions[token];

    using row_gl = gl<T, 1, 1, -1, D>;           // (M, D) source / (R, D) cache
    using cs_gl  = gl<T, 1, 1, -1, D2>;          // (P, D/2)
    row_gl gl_k(k,           nullptr, nullptr, 1, nullptr);
    row_gl gl_v(v,           nullptr, nullptr, 1, nullptr);
    row_gl gl_kc(key_cache,  nullptr, nullptr, 1, nullptr);
    row_gl gl_vc(value_cache,nullptr, nullptr, 1, nullptr);
    cs_gl  gl_c(cosb,        nullptr, nullptr, 1, nullptr);
    cs_gl  gl_s(sinb,        nullptr, nullptr, 1, nullptr);

    using vecH = rv_fl<D2>;
    vecH k1, k2, cv, sv, o1, o2, tmp;
    load(k1, gl_k, {0, 0, row, 0}, laneId);      // first half of K
    load(k2, gl_k, {0, 0, row, 1}, laneId);      // second half of K
    load(cv, gl_c, {0, 0, pos, 0}, laneId);
    load(sv, gl_s, {0, 0, pos, 0}, laneId);

    // ko1 = k1*cos - k2*sin
    mul(o1, k1, cv);
    mul(tmp, k2, sv);
    sub(o1, o1, tmp);
    // ko2 = k2*cos + k1*sin
    mul(o2, k2, cv);
    mul(tmp, k1, sv);
    add(o2, o2, tmp);

    // Rotated K -> paged cache row.
    store(gl_kc, o1, {0, 0, (int)dst_row, 0}, laneId);
    store(gl_kc, o2, {0, 0, (int)dst_row, 1}, laneId);

    // V (unrotated) -> paged cache row.
    using vecD = rv_fl<D>;
    vecD vv;
    load(vv, gl_v, {0, 0, row, 0}, laneId);
    store(gl_vc, vv, {0, 0, (int)dst_row, 0}, laneId);
}

// Same as rope_kv_insert, but RMSNorms K over the head dim before RoPE (fused
// Q/K-norm + RoPE + insert, the reference pattern). norm over all D of the K row:
// rms = rsqrt(mean(k^2)+eps); k = k*rms*w  (or k*rms*(1+w) for gemma=1). bf16.
template <typename T, int D>
kernel void rope_kv_insert_norm(device   T    *k            [[buffer(0)]],
                                device   T    *v            [[buffer(1)]],
                                device   T    *cosb         [[buffer(2)]],
                                device   T    *sinb         [[buffer(3)]],
                                device   const int  *positions    [[buffer(4)]],
                                device   const long *slot_mapping [[buffer(5)]],
                                device   T    *key_cache    [[buffer(6)]],
                                device   T    *value_cache  [[buffer(7)]],
                                device   T    *norm_weight  [[buffer(8)]],
                                constant int  &num_kv_heads [[buffer(9)]],
                                constant int  &block_size   [[buffer(10)]],
                                constant float &eps         [[buffer(11)]],
                                constant int  &gemma        [[buffer(12)]],
                                uint3 blockIdx [[threadgroup_position_in_grid]],
                                uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");
    const int row = blockIdx.x;
    const int token = row / num_kv_heads;
    const int kv_head = row % num_kv_heads;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const long dst_row =
        (block * block_size + block_offset) * (long)num_kv_heads + (long)kv_head;
    const int pos = positions[token];

    using row_gl = gl<T, 1, 1, -1, D>;
    using cs_gl  = gl<T, 1, 1, -1, D2>;
    row_gl gl_k(k,            nullptr, nullptr, 1, nullptr);
    row_gl gl_v(v,            nullptr, nullptr, 1, nullptr);
    row_gl gl_kc(key_cache,   nullptr, nullptr, 1, nullptr);
    row_gl gl_vc(value_cache, nullptr, nullptr, 1, nullptr);
    cs_gl  gl_c(cosb,         nullptr, nullptr, 1, nullptr);
    cs_gl  gl_s(sinb,         nullptr, nullptr, 1, nullptr);
    cs_gl  gl_w(norm_weight,  nullptr, nullptr, 1, nullptr);   // (D,) split into halves

    using vecH = rv_fl<D2>;
    vecH k1, k2, w1, w2, cv, sv, o1, o2, tmp, sq;
    load(k1, gl_k, {0, 0, row, 0}, laneId);
    load(k2, gl_k, {0, 0, row, 1}, laneId);
    load(w1, gl_w, {0, 0, 0,   0}, laneId);
    load(w2, gl_w, {0, 0, 0,   1}, laneId);

    // RMSNorm over the full head dim D.
    float ss1 = 0.f, ss2 = 0.f;
    mul(sq, k1, k1); sum(ss1, sq, laneId);
    mul(sq, k2, k2); sum(ss2, sq, laneId);
    const float rms = metal::rsqrt((ss1 + ss2) / (float)D + eps);
    mul(k1, k1, rms); mul(k2, k2, rms);
    if (gemma) { add(w1, w1, 1.0f); add(w2, w2, 1.0f); }   // gemma uses (1 + weight)
    mul(k1, k1, w1); mul(k2, k2, w2);

    // RoPE on the normed halves.
    load(cv, gl_c, {0, 0, pos, 0}, laneId);
    load(sv, gl_s, {0, 0, pos, 0}, laneId);
    mul(o1, k1, cv); mul(tmp, k2, sv); sub(o1, o1, tmp);
    mul(o2, k2, cv); mul(tmp, k1, sv); add(o2, o2, tmp);
    store(gl_kc, o1, {0, 0, (int)dst_row, 0}, laneId);
    store(gl_kc, o2, {0, 0, (int)dst_row, 1}, laneId);

    using vecD = rv_fl<D>;
    vecD vv;
    load(vv, gl_v, {0, 0, row, 0}, laneId);
    store(gl_vc, vv, {0, 0, (int)dst_row, 0}, laneId);
}

#define instantiate_rope_kv_insert_norm(type_name, T, DVAL)                    \
  template [[host_name("rope_kv_insert_norm_" #type_name "_" #DVAL)]] [[kernel]] void \
  rope_kv_insert_norm<T, DVAL>(device T *k [[buffer(0)]],                       \
                            device T *v [[buffer(1)]],                          \
                            device T *cosb [[buffer(2)]],                       \
                            device T *sinb [[buffer(3)]],                       \
                            device const int *positions [[buffer(4)]],          \
                            device const long *slot_mapping [[buffer(5)]],      \
                            device T *key_cache [[buffer(6)]],                  \
                            device T *value_cache [[buffer(7)]],                \
                            device T *norm_weight [[buffer(8)]],                \
                            constant int &num_kv_heads [[buffer(9)]],           \
                            constant int &block_size [[buffer(10)]],            \
                            constant float &eps [[buffer(11)]],                 \
                            constant int &gemma [[buffer(12)]],                 \
                            uint3 blockIdx [[threadgroup_position_in_grid]],    \
                            uint laneId [[thread_index_in_simdgroup]]);

#define instantiate_rope_kv_insert(type_name, T, DVAL)                         \
  template [[host_name("rope_kv_insert_" #type_name "_" #DVAL)]] [[kernel]] void \
  rope_kv_insert<T, DVAL>(device   T    *k            [[buffer(0)]],           \
                       device   T    *v            [[buffer(1)]],              \
                       device   T    *cosb         [[buffer(2)]],             \
                       device   T    *sinb         [[buffer(3)]],             \
                       device   const int  *positions    [[buffer(4)]],       \
                       device   const long *slot_mapping [[buffer(5)]],       \
                       device   T    *key_cache    [[buffer(6)]],             \
                       device   T    *value_cache  [[buffer(7)]],             \
                       constant int  &num_kv_heads [[buffer(8)]],             \
                       constant int  &block_size   [[buffer(9)]],             \
                       uint3 blockIdx [[threadgroup_position_in_grid]],        \
                       uint  laneId   [[thread_index_in_simdgroup]]);

// Q-path: rotate (optionally weighted-RMSNorm, gemma-(1+w)) Q into a contiguous q_out. Q is
// (num_tokens, num_q_heads, D) flattened to (M, D); out row = in row (Q is NOT paged/scattered).
// Split-half RoPE, positions-indexed like the K insert — the Q companion of rope_kv_insert.
template <typename T, int D>
kernel void rope_q(device   T    *q            [[buffer(0)]],
                   device   T    *cosb         [[buffer(1)]],
                   device   T    *sinb         [[buffer(2)]],
                   device   const int *positions [[buffer(3)]],
                   device   T    *q_out        [[buffer(4)]],
                   constant int  &num_heads    [[buffer(5)]],
                   constant int  &do_norm      [[buffer(6)]],   // 0 none, 1 weighted RMSNorm
                   constant int  &gemma        [[buffer(7)]],
                   constant float &eps         [[buffer(8)]],
                   device   T    *norm_weight  [[buffer(9)]],   // (D,), read iff do_norm
                   uint3 blockIdx [[threadgroup_position_in_grid]],
                   uint  laneId   [[thread_index_in_simdgroup]]) {
    constexpr int D2 = D / 2;
    static_assert(D2 % TILE_DIM == 0, "D/2 must be divisible by 8");
    const int row = blockIdx.x;
    const int token = row / num_heads;
    const int pos = positions[token];

    using row_gl = gl<T, 1, 1, -1, D>;
    using cs_gl  = gl<T, 1, 1, -1, D2>;
    row_gl gl_q(q, nullptr, nullptr, 1, nullptr);
    row_gl gl_o(q_out, nullptr, nullptr, 1, nullptr);
    cs_gl  gl_c(cosb, nullptr, nullptr, 1, nullptr);
    cs_gl  gl_s(sinb, nullptr, nullptr, 1, nullptr);
    cs_gl  gl_w(norm_weight, nullptr, nullptr, 1, nullptr);

    using vecH = rv_fl<D2>;
    vecH q1, q2, cv, sv, o1, o2, tmp, w1, w2, sq;
    load(q1, gl_q, {0, 0, row, 0}, laneId);
    load(q2, gl_q, {0, 0, row, 1}, laneId);
    if (do_norm) {
        float ss1 = 0.f, ss2 = 0.f;
        mul(sq, q1, q1); sum(ss1, sq, laneId);
        mul(sq, q2, q2); sum(ss2, sq, laneId);
        const float rms = metal::rsqrt((ss1 + ss2) / (float)D + eps);
        mul(q1, q1, rms); mul(q2, q2, rms);
        load(w1, gl_w, {0, 0, 0, 0}, laneId);
        load(w2, gl_w, {0, 0, 0, 1}, laneId);
        if (gemma) { add(w1, w1, 1.0f); add(w2, w2, 1.0f); }
        mul(q1, q1, w1); mul(q2, q2, w2);
    }
    load(cv, gl_c, {0, 0, pos, 0}, laneId);
    load(sv, gl_s, {0, 0, pos, 0}, laneId);
    mul(o1, q1, cv); mul(tmp, q2, sv); sub(o1, o1, tmp);
    mul(o2, q2, cv); mul(tmp, q1, sv); add(o2, o2, tmp);
    store(gl_o, o1, {0, 0, row, 0}, laneId);
    store(gl_o, o2, {0, 0, row, 1}, laneId);
}

#define instantiate_rope_q(type_name, T, DVAL)                                 \
  template [[host_name("rope_q_" #type_name "_" #DVAL)]] [[kernel]] void        \
  rope_q<T, DVAL>(device T *q [[buffer(0)]],                                    \
                  device T *cosb [[buffer(1)]],                                 \
                  device T *sinb [[buffer(2)]],                                 \
                  device const int *positions [[buffer(3)]],                    \
                  device T *q_out [[buffer(4)]],                                \
                  constant int &num_heads [[buffer(5)]],                        \
                  constant int &do_norm [[buffer(6)]],                          \
                  constant int &gemma [[buffer(7)]],                            \
                  constant float &eps [[buffer(8)]],                            \
                  device T *norm_weight [[buffer(9)]],                          \
                  uint3 blockIdx [[threadgroup_position_in_grid]],              \
                  uint laneId [[thread_index_in_simdgroup]]);

#define instantiate_rope_kv_all(type_name, T)                                  \
  instantiate_rope_kv_insert(type_name, T, 64)                                 \
  instantiate_rope_kv_insert(type_name, T, 128)                               \
  instantiate_rope_kv_insert_norm(type_name, T, 64)                            \
  instantiate_rope_kv_insert_norm(type_name, T, 128)                           \
  instantiate_rope_q(type_name, T, 64)                                         \
  instantiate_rope_q(type_name, T, 128)

instantiate_rope_kv_all(float32, float)
instantiate_rope_kv_all(float16, half)
instantiate_rope_kv_all(bfloat16, bf16)

}
