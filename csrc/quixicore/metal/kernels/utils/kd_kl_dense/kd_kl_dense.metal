#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// DENSE-teacher KD-KL fwd/bwd (train_plan §5.1 / §9.1 A6b — the full-KL arm of
// the A6b-vs-A6c ablation): loss = KL( softmax(t·invtemp) ‖ softmax(s·invtemp) )
// per row, with LIVE teacher logits (no top-k cache). Fused so the (T, V)
// log-softmax / probability tensors of the PyTorch chunked path are never
// materialized — both logit rows are streamed, per the chunked-losses mandate's
// "or fused equivalents" clause.
//
//   fwd:  loss = Σ_v p_t (log p_t − log q),  p_t = softmax(zt), q = softmax(zs),
//         zt = t·invtemp, zs = s·invtemp; emits both LSEs for the backward.
//   bwd:  d loss / d s_v = go · invtemp · (q_v − p_t,v)
//
// Temperature convention matches kd_kl_topk: pass invtemp = 1/τ, caller applies
// α·τ² to loss / grad_out. One simdgroup per row; grid (Tn, 1, 1), 32 threads.
// ---------------------------------------------------------------------------

constant float KDD_NEG_INF = -3.4028234663852886e38f;

template <typename T>
kernel void kd_kl_dense_fwd(device const T *t_logits [[buffer(0)]],  // (Tn, V)
                            device const T *s_logits [[buffer(1)]],  // (Tn, V)
                            device float   *loss     [[buffer(2)]],  // (Tn,)
                            device float   *lse_t_out[[buffer(3)]],  // (Tn,)
                            device float   *lse_s_out[[buffer(4)]],  // (Tn,)
                            constant int   &V        [[buffer(5)]],
                            constant float &invtemp  [[buffer(6)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    // pass 1: online (max, sumexp) for teacher and student simultaneously
    float mt = KDD_NEG_INF, lt = 0.0f, ms = KDD_NEG_INF, ls = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float zt = float(t_logits[base + i]) * invtemp;
        const float zs = float(s_logits[base + i]) * invtemp;
        float nm = max(mt, zt);
        lt = lt * exp(mt - nm) + exp(zt - nm); mt = nm;
        nm = max(ms, zs);
        ls = ls * exp(ms - nm) + exp(zs - nm); ms = nm;
    }
    const float Mt = simd_max(mt), Ms = simd_max(ms);
    lt = simd_sum(lt * exp(mt - Mt));
    ls = simd_sum(ls * exp(ms - Ms));
    const float lse_t = Mt + log(lt), lse_s = Ms + log(ls);

    // pass 2: Σ p_t · ((zt − lse_t) − (zs − lse_s))
    float acc = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float zt = float(t_logits[base + i]) * invtemp;
        const float zs = float(s_logits[base + i]) * invtemp;
        acc += exp(zt - lse_t) * ((zt - lse_t) - (zs - lse_s));
    }
    acc = simd_sum(acc);
    if (lane == 0) { loss[row] = acc; lse_t_out[row] = lse_t; lse_s_out[row] = lse_s; }
}

template <typename T>
kernel void kd_kl_dense_bwd(device const T     *t_logits [[buffer(0)]],
                            device const T     *s_logits [[buffer(1)]],
                            device const float *lse_t_in [[buffer(2)]],
                            device const float *lse_s_in [[buffer(3)]],
                            device const float *grad_out [[buffer(4)]],  // (Tn,)
                            device T           *grad_s   [[buffer(5)]],  // (Tn, V)
                            constant int   &V       [[buffer(6)]],
                            constant float &invtemp [[buffer(7)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const float lse_t = lse_t_in[row], lse_s = lse_s_in[row];
    const float go = grad_out[row] * invtemp;
    for (int i = (int)lane; i < V; i += 32) {
        const float q  = exp(float(s_logits[base + i]) * invtemp - lse_s);
        const float pt = exp(float(t_logits[base + i]) * invtemp - lse_t);
        grad_s[base + i] = T((q - pt) * go);
    }
}

// ---------------------------------------------------------------------------
// FUSED CE + dense-KD (the heal loss, one kernel pair). CE and KD both stream
// the same student logits, and their backwards write into the same grad tensor;
// separately that costs 4 kernel launches + an autograd grad-add pass over
// (T, V). Fused: fwd is a SINGLE pass — three online LSEs (student @ temp 1
// for CE, student @ 1/tau + teacher @ 1/tau for KD), the CE target gather, AND
// the KL sum via online weighted accumulators (flash-attention-style rescale):
//   KL = (S1 - S2)/L + lse_s - lse_t,  S1 = Σ exp(zt-m)·zt, S2 = Σ exp(zt-m)·zs,
//   L = Σ exp(zt-m)   [derivation: Σ p_t·((zt-lse_t)-(zs-lse_s)) with Σ p_t = 1]
// bwd emits the COMBINED grad in a single pass:
//   grad = go_ce*(softmax(s) - onehot) + go_kd*invtemp*(softmax(s/tau) - softmax(t/tau))
// targets < 0 (ignore_index) contribute no CE loss/grad; KD applies to all rows.
// ---------------------------------------------------------------------------

template <typename T>
kernel void kd_ce_fused_fwd(device const T   *t_logits [[buffer(0)]],
                            device const T   *s_logits [[buffer(1)]],
                            device const int *targets  [[buffer(2)]],
                            device float     *ce       [[buffer(3)]],
                            device float     *kd       [[buffer(4)]],
                            device float     *lse_sr   [[buffer(5)]],  // student, temp 1
                            device float     *lse_st   [[buffer(6)]],  // student, 1/tau
                            device float     *lse_t    [[buffer(7)]],  // teacher, 1/tau
                            constant int   &V       [[buffer(8)]],
                            constant float &invtemp [[buffer(9)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const int tgt = targets[row];
    // vec2 loads (128 B per simdgroup fetch) with float2 online accumulators,
    // merged pairwise before the simd reduce; scalar tail covers odd V.
    float2 m_sr = KDD_NEG_INF, l_sr = 0.0f;
    float2 m_st = KDD_NEG_INF, l_st = 0.0f;
    float2 m_t  = KDD_NEG_INF, l_t  = 0.0f;
    float2 s1 = 0.0f, s2 = 0.0f;         // Σ exp(zt-m_t)·zt, Σ exp(zt-m_t)·zs
    float tgt_logit = 0.0f;
    const int V2 = (V % 2 == 0) ? V / 2 : 0;
    device const vec<T, 2> *t2 = (device const vec<T, 2> *)(t_logits + base);
    device const vec<T, 2> *s2p = (device const vec<T, 2> *)(s_logits + base);
    for (int i = (int)lane; i < V2; i += 32) {
        const float2 zs_raw = float2(s2p[i]);
        const float2 zs = zs_raw * invtemp;
        const float2 zt = float2(t2[i]) * invtemp;
        float2 nm = max(m_sr, zs_raw);
        l_sr = l_sr * exp(m_sr - nm) + exp(zs_raw - nm); m_sr = nm;
        nm = max(m_st, zs);
        l_st = l_st * exp(m_st - nm) + exp(zs - nm); m_st = nm;
        nm = max(m_t, zt);
        const float2 r = exp(m_t - nm), w = exp(zt - nm);
        l_t = l_t * r + w;
        s1  = s1  * r + w * zt;
        s2  = s2  * r + w * zs;
        m_t = nm;
        if (2 * i == tgt)     { tgt_logit = zs_raw.x; }
        if (2 * i + 1 == tgt) { tgt_logit = zs_raw.y; }
    }
    for (int i = 2 * V2 + (int)lane; i < V; i += 32) {
        const float zs_raw = float(s_logits[base + i]);
        const float zs = zs_raw * invtemp;
        const float zt = float(t_logits[base + i]) * invtemp;
        float nm = max(m_sr.x, zs_raw);
        l_sr.x = l_sr.x * exp(m_sr.x - nm) + exp(zs_raw - nm); m_sr.x = nm;
        nm = max(m_st.x, zs);
        l_st.x = l_st.x * exp(m_st.x - nm) + exp(zs - nm); m_st.x = nm;
        nm = max(m_t.x, zt);
        const float r = exp(m_t.x - nm), w = exp(zt - nm);
        l_t.x = l_t.x * r + w;
        s1.x  = s1.x  * r + w * zt;
        s2.x  = s2.x  * r + w * zs;
        m_t.x = nm;
        if (i == tgt) { tgt_logit = zs_raw; }
    }
    // pairwise .x/.y merge -> per-lane scalars, then the usual simd reduce
    const float msr = max(m_sr.x, m_sr.y), mst = max(m_st.x, m_st.y);
    const float mt = max(m_t.x, m_t.y);
    const float lsr = l_sr.x * exp(m_sr.x - msr) + l_sr.y * exp(m_sr.y - msr);
    const float lst = l_st.x * exp(m_st.x - mst) + l_st.y * exp(m_st.y - mst);
    const float2 rp = exp(m_t - mt);
    const float ltm = dot(l_t, rp), s1m = dot(s1, rp), s2m = dot(s2, rp);
    const float Msr = simd_max(msr), Mst = simd_max(mst), Mt = simd_max(mt);
    const float Lsr = simd_sum(lsr * exp(msr - Msr));
    const float Lst = simd_sum(lst * exp(mst - Mst));
    const float rt = exp(mt - Mt);
    const float L  = simd_sum(ltm * rt);
    const float S1 = simd_sum(s1m * rt);
    const float S2 = simd_sum(s2m * rt);
    const float LSR = Msr + log(Lsr), LST = Mst + log(Lst), LT = Mt + log(L);
    tgt_logit = simd_sum(tgt_logit);                 // exactly one lane holds it
    if (lane == 0) {
        ce[row] = (tgt >= 0) ? (LSR - tgt_logit) : 0.0f;
        kd[row] = (S1 - S2) / L + LST - LT;
        lse_sr[row] = LSR; lse_st[row] = LST; lse_t[row] = LT;
    }
}

template <typename T>
kernel void kd_ce_fused_bwd(device const T     *t_logits [[buffer(0)]],
                            device const T     *s_logits [[buffer(1)]],
                            device const int   *targets  [[buffer(2)]],
                            device const float *lse_sr   [[buffer(3)]],
                            device const float *lse_st   [[buffer(4)]],
                            device const float *lse_t    [[buffer(5)]],
                            device const float *go_ce    [[buffer(6)]],
                            device const float *go_kd    [[buffer(7)]],
                            device T           *grad_s   [[buffer(8)]],
                            constant int   &V       [[buffer(9)]],
                            constant float &invtemp [[buffer(10)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const int tgt = targets[row];
    const float LSR = lse_sr[row], LST = lse_st[row], LT = lse_t[row];
    const float gce = (tgt >= 0) ? go_ce[row] : 0.0f;
    const float gkd = go_kd[row] * invtemp;
    // vec2 main loop: 128 B per simdgroup fetch (full cache line); scalar tail
    // covers odd V. Row base is vec2-aligned whenever V is even.
    const int V2 = (V % 2 == 0) ? V / 2 : 0;
    device const vec<T, 2> *t2 = (device const vec<T, 2> *)(t_logits + base);
    device const vec<T, 2> *s2 = (device const vec<T, 2> *)(s_logits + base);
    device vec<T, 2> *g2 = (device vec<T, 2> *)(grad_s + base);
    for (int i = (int)lane; i < V2; i += 32) {
        const float2 zs_raw = float2(s2[i]);
        const float2 p_raw = exp(zs_raw - LSR);
        const float2 q  = exp(zs_raw * invtemp - LST);
        const float2 pt = exp(float2(t2[i]) * invtemp - LT);
        float2 g = gce * p_raw + gkd * (q - pt);
        if (2 * i == tgt)     { g.x -= gce; }
        if (2 * i + 1 == tgt) { g.y -= gce; }
        g2[i] = vec<T, 2>(g);
    }
    for (int i = 2 * V2 + (int)lane; i < V; i += 32) {
        const float zs_raw = float(s_logits[base + i]);
        const float p_raw = exp(zs_raw - LSR);
        const float q  = exp(zs_raw * invtemp - LST);
        const float pt = exp(float(t_logits[base + i]) * invtemp - LT);
        float g = gce * (p_raw - (i == tgt ? 1.0f : 0.0f)) + gkd * (q - pt);
        grad_s[base + i] = T(g);
    }
}

#define instantiate_kd_ce_fused(type_name, T)                                        \
  template [[host_name("kd_ce_fused_fwd_" #type_name)]] [[kernel]] void              \
  kd_ce_fused_fwd<T>(device const T *t_logits [[buffer(0)]],                         \
                     device const T *s_logits [[buffer(1)]],                         \
                     device const int *targets [[buffer(2)]],                        \
                     device float *ce [[buffer(3)]], device float *kd [[buffer(4)]], \
                     device float *lse_sr [[buffer(5)]],                             \
                     device float *lse_st [[buffer(6)]],                             \
                     device float *lse_t [[buffer(7)]],                              \
                     constant int &V [[buffer(8)]],                                  \
                     constant float &invtemp [[buffer(9)]],                          \
                     uint row [[threadgroup_position_in_grid]],                      \
                     uint lane [[thread_index_in_simdgroup]]);                       \
  template [[host_name("kd_ce_fused_bwd_" #type_name)]] [[kernel]] void              \
  kd_ce_fused_bwd<T>(device const T *t_logits [[buffer(0)]],                         \
                     device const T *s_logits [[buffer(1)]],                         \
                     device const int *targets [[buffer(2)]],                        \
                     device const float *lse_sr [[buffer(3)]],                       \
                     device const float *lse_st [[buffer(4)]],                       \
                     device const float *lse_t [[buffer(5)]],                        \
                     device const float *go_ce [[buffer(6)]],                        \
                     device const float *go_kd [[buffer(7)]],                        \
                     device T *grad_s [[buffer(8)]],                                 \
                     constant int &V [[buffer(9)]],                                  \
                     constant float &invtemp [[buffer(10)]],                         \
                     uint row [[threadgroup_position_in_grid]],                      \
                     uint lane [[thread_index_in_simdgroup]]);

instantiate_kd_ce_fused(float32, float)
instantiate_kd_ce_fused(float16, half)
instantiate_kd_ce_fused(bfloat16, bf16)

#define instantiate_kd_kl_dense(type_name, T)                                        \
  template [[host_name("kd_kl_dense_fwd_" #type_name)]] [[kernel]] void              \
  kd_kl_dense_fwd<T>(device const T *t_logits [[buffer(0)]],                         \
                     device const T *s_logits [[buffer(1)]],                         \
                     device float *loss [[buffer(2)]],                               \
                     device float *lse_t_out [[buffer(3)]],                          \
                     device float *lse_s_out [[buffer(4)]],                          \
                     constant int &V [[buffer(5)]],                                  \
                     constant float &invtemp [[buffer(6)]],                          \
                     uint row [[threadgroup_position_in_grid]],                      \
                     uint lane [[thread_index_in_simdgroup]]);                       \
  template [[host_name("kd_kl_dense_bwd_" #type_name)]] [[kernel]] void              \
  kd_kl_dense_bwd<T>(device const T *t_logits [[buffer(0)]],                         \
                     device const T *s_logits [[buffer(1)]],                         \
                     device const float *lse_t_in [[buffer(2)]],                     \
                     device const float *lse_s_in [[buffer(3)]],                     \
                     device const float *grad_out [[buffer(4)]],                     \
                     device T *grad_s [[buffer(5)]],                                 \
                     constant int &V [[buffer(6)]],                                  \
                     constant float &invtemp [[buffer(7)]],                          \
                     uint row [[threadgroup_position_in_grid]],                      \
                     uint lane [[thread_index_in_simdgroup]]);

instantiate_kd_kl_dense(float32, float)
instantiate_kd_kl_dense(float16, half)
instantiate_kd_kl_dense(bfloat16, bf16)
