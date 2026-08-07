#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// The "sampler zoo" logit/prob transforms (metal-forge/vLLM semantics on the TM sampling
// substrate): one simdgroup (32 lanes) per row, vocab strided by 32, fp32 math, out-of-place,
// masked tokens = SMPT_NEG_INF. Every logit transform takes `invtemp` and writes TEMPERED
// logits (the apply_penalty contract; invtemp = 1 reproduces the untempered reference).
// Params are launch-uniform scalars (TM convention; per-row variants are additive later).
// ---------------------------------------------------------------------------

constant float SMPT_NEG_INF = -3.4028234663852886e38f;
constant int SMPT_MAX_K = 64;

// Gemma-style final-logit softcap. This is deliberately separate from the
// attention score softcap because it operates on materialized LM-head logits.
template <typename T>
kernel void logits_softcap(device const T *logits [[buffer(0)]],
                           device T *out [[buffer(1)]],
                           constant uint &n [[buffer(2)]],
                           constant float &cap [[buffer(3)]],
                           uint tid [[thread_position_in_grid]]) {
    using T4 = metal::vec<T, 4>;
    const uint base = tid * 4;
    if (base + 4 <= n) {
        const float4 value = float4(((device const T4*)(logits + base))[0]);
        float4 result;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
            result[i] = cap * metal::tanh(value[i] / cap);
        }
        ((device T4*)(out + base))[0] = T4(result);
    } else {
        for (uint i = base; i < n; ++i) {
            const float value = float(logits[i]);
            out[i] = T(cap * metal::tanh(value / cap));
        }
    }
}

// Reusable scalar-bounds clamp for clippable projections and activation
// stabilization. Bounds may be infinite, matching framework clamp semantics.
template <typename T>
kernel void value_clip(device const T *x [[buffer(0)]],
                       device T *out [[buffer(1)]],
                       constant uint &n [[buffer(2)]],
                       constant float &min_value [[buffer(3)]],
                       constant float &max_value [[buffer(4)]],
                       uint tid [[thread_position_in_grid]]) {
    using T4 = metal::vec<T, 4>;
    const uint base = tid * 4;
    if (base + 4 <= n) {
        const float4 value = float4(((device const T4*)(x + base))[0]);
        float4 result;
        #pragma clang loop unroll(full)
        for (int i = 0; i < 4; ++i) {
            result[i] = metal::clamp(value[i], min_value, max_value);
        }
        ((device T4*)(out + base))[0] = T4(result);
    } else {
        for (uint i = base; i < n; ++i) {
            out[i] = T(metal::clamp(float(x[i]), min_value, max_value));
        }
    }
}

// quadratic / smoothing sampling: diff = ls - max; diff -= diff^2 (s*diff - k);
// out = ls - diff' with k = factor(3-curve)/2, s = factor(curve-1)/2. factor == 0 -> copy.
template <typename T>
kernel void quadratic_transform(device const T *logits [[buffer(0)]],
                                device T       *out    [[buffer(1)]],
                                constant int   &V      [[buffer(2)]],
                                constant float &factor [[buffer(3)]],
                                constant float &curve  [[buffer(4)]],
                                constant float &invtemp [[buffer(5)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float mx = SMPT_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) mx = max(mx, float(logits[base + i]) * invtemp);
    mx = simd_max(mx);
    const float k = factor * (3.0f - curve) / 2.0f;
    const float s = factor * (curve - 1.0f) / 2.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (factor == 0.0f) { out[base + i] = T(ls); continue; }
        float diff = ls - mx;
        diff -= diff * diff * (s * diff - k);
        out[base + i] = T(metal::isfinite(diff) ? ls - (ls - mx - diff) : ls);
    }
}

// top-nsigma: threshold = max - nsigma * sample-stddev (Bessel). Finite logits assumed
// (compose BEFORE other -inf masks).
template <typename T>
kernel void top_nsigma_mask(device const T *logits [[buffer(0)]],
                            device T       *out    [[buffer(1)]],
                            constant int   &V      [[buffer(2)]],
                            constant float &nsigma [[buffer(3)]],
                            constant float &invtemp [[buffer(4)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float mx = SMPT_NEG_INF, sum = 0.0f, sumsq = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        mx = max(mx, ls);
        sum += ls;
        sumsq += ls * ls;
    }
    mx = simd_max(mx);
    sum = simd_sum(sum);
    sumsq = simd_sum(sumsq);
    const float var = max((sumsq - sum * sum / (float)V) / (float)(V - 1), 0.0f);
    const float thresh = mx - nsigma * metal::sqrt(var);
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        out[base + i] = (ls < thresh) ? T(SMPT_NEG_INF) : T(ls);
    }
}

// top-A in log space (exact): prob < top_a * pmax^2  <=>  ls - mx < log(top_a) - log(Z).
// The argmax row (ls == mx) is always kept. top_a <= 0 -> keep all.
template <typename T>
kernel void top_a_mask(device const T *logits [[buffer(0)]],
                       device T       *out    [[buffer(1)]],
                       constant int   &V      [[buffer(2)]],
                       constant float &top_a  [[buffer(3)]],
                       constant float &invtemp [[buffer(4)]],
                       uint row  [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float mx = SMPT_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) mx = max(mx, float(logits[base + i]) * invtemp);
    mx = simd_max(mx);
    float z = 0.0f;
    for (int i = (int)lane; i < V; i += 32) z += metal::exp(float(logits[base + i]) * invtemp - mx);
    z = simd_sum(z);
    const float log_thr = (top_a > 0.0f) ? metal::log(top_a) - metal::log(z) : SMPT_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        out[base + i] = (ls < mx && ls - mx < log_thr) ? T(SMPT_NEG_INF) : T(ls);
    }
}

// epsilon cutoff: drop tokens with prob < eps (the argmax and its ties always survive).
template <typename T>
kernel void epsilon_cutoff_mask(device const T *logits [[buffer(0)]],
                                device T       *out    [[buffer(1)]],
                                constant int   &V      [[buffer(2)]],
                                constant float &eps    [[buffer(3)]],
                                constant float &invtemp [[buffer(4)]],
                                uint row  [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float mx = SMPT_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) mx = max(mx, float(logits[base + i]) * invtemp);
    mx = simd_max(mx);
    float z = 0.0f;
    for (int i = (int)lane; i < V; i += 32) z += metal::exp(float(logits[base + i]) * invtemp - mx);
    z = simd_sum(z);
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        const bool drop = ls < mx && metal::exp(ls - mx) / z < eps;
        out[base + i] = drop ? T(SMPT_NEG_INF) : T(ls);
    }
}

// eta cutoff: eps_eff = min(eta, sqrt(eta) * exp(sum p log p)); entropy via the typical_p
// trick: sum p log p = S1/Z - mx - log Z with S1 = sum exp(ls - mx) * ls.
template <typename T>
kernel void eta_cutoff_mask(device const T *logits [[buffer(0)]],
                            device T       *out    [[buffer(1)]],
                            constant int   &V      [[buffer(2)]],
                            constant float &eta    [[buffer(3)]],
                            constant float &invtemp [[buffer(4)]],
                            uint row  [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float mx = SMPT_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) mx = max(mx, float(logits[base + i]) * invtemp);
    mx = simd_max(mx);
    float z = 0.0f, s1 = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        const float e = metal::exp(ls - mx);
        z += e;
        s1 += e * ls;
    }
    z = simd_sum(z);
    s1 = simd_sum(s1);
    const float sum_plogp = s1 / z - mx - metal::log(z);      // = -entropy
    const float eps_eff = min(eta, metal::sqrt(eta) * metal::exp(sum_plogp));
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        const bool drop = ls < mx && metal::exp(ls - mx) / z < eps_eff;
        out[base + i] = drop ? T(SMPT_NEG_INF) : T(ls);
    }
}

// XTC (exclude top choices): with probability `probability` (on-device coin), remove every
// token with prob >= threshold EXCEPT the least-probable such token. Comparisons run in the
// exp(ls - mx) domain against threshold * Z (no per-token division).
template <typename T>
kernel void xtc_mask(device const T *logits    [[buffer(0)]],
                     device T       *out       [[buffer(1)]],
                     constant int   &V         [[buffer(2)]],
                     constant float &threshold [[buffer(3)]],
                     constant float &probability [[buffer(4)]],
                     constant float &invtemp   [[buffer(5)]],
                     constant uint  &seed      [[buffer(6)]],
                     uint row  [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    // per-row coin at a counter outside the token-index range (can't collide with draws)
    const bool apply = rng_uniform(seed, row, 0xFFFFFFFFu) < probability;
    if (!apply) {
        for (int i = (int)lane; i < V; i += 32) {
            out[base + i] = T(float(logits[base + i]) * invtemp);
        }
        return;
    }
    float mx = SMPT_NEG_INF;
    for (int i = (int)lane; i < V; i += 32) mx = max(mx, float(logits[base + i]) * invtemp);
    mx = simd_max(mx);
    float z = 0.0f;
    for (int i = (int)lane; i < V; i += 32) z += metal::exp(float(logits[base + i]) * invtemp - mx);
    z = simd_sum(z);
    const float ethr = threshold * z;                          // eligibility in e-domain
    int count = 0;
    float emin = INFINITY;
    for (int i = (int)lane; i < V; i += 32) {
        const float e = metal::exp(float(logits[base + i]) * invtemp - mx);
        if (e >= ethr) {
            count += 1;
            emin = min(emin, e);
        }
    }
    count = simd_sum(count);
    emin = simd_min(emin);
    for (int i = (int)lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        const float e = metal::exp(ls - mx);
        const bool remove = count > 1 && e >= ethr && e > emin;
        out[base + i] = remove ? T(SMPT_NEG_INF) : T(ls);
    }
}

// skew sampling over PROBS: out_i = pow(cdf_i, e) - pow(cdf_{i-1}, e) with e = exp(skew) on
// the INDEX-ORDER running CDF (the metal-forge contract; NOTE this diverges from exllamav2's
// sorted-CDF skew — an exact sorted variant needs a sort and is deferred). Lane-contiguous
// chunks + simd_prefix_exclusive_sum of chunk sums; serial pow-walk inside the chunk.
template <typename T>
kernel void skew_transform(device const T *probs [[buffer(0)]],
                           device T       *out   [[buffer(1)]],
                           constant int   &V     [[buffer(2)]],
                           constant float &skew  [[buffer(3)]],
                           uint row  [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    const int chunk = (V + 31) / 32;
    const int lo = (int)lane * chunk;
    const int hi = min(lo + chunk, V);
    float csum = 0.0f;
    for (int i = lo; i < hi; ++i) csum += float(probs[base + i]);
    const float cdf0 = simd_prefix_exclusive_sum(csum);
    const float e = metal::exp(skew);
    float cum = cdf0;
    float prev = metal::pow(max(cdf0, 0.0f), e);
    for (int i = lo; i < hi; ++i) {
        cum += float(probs[base + i]);
        const float t = metal::pow(max(cum, 0.0f), e);
        out[base + i] = T(t - prev);
        prev = t;
    }
}

// top-k renormalized PROBS (spec-decode distribution utility): keep the top-k probabilities
// (ties -> smaller id, the masked_topk rule), renormalize to sum 1, zero elsewhere. k <= 64.
template <typename T>
kernel void top_k_renorm_probs(device const T *probs [[buffer(0)]],
                               device T       *out   [[buffer(1)]],
                               constant int   &V     [[buffer(2)]],
                               constant int   &K     [[buffer(3)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    int chosen[SMPT_MAX_K];
    float chosen_val[SMPT_MAX_K];
    indexed_cand<T> cand{probs, base};
    const int k = min(K, V);
    masked_topk(cand, V, k, lane, SMPT_NEG_INF, chosen, chosen_val);
    float kept_sum = 0.0f;
    for (int j = 0; j < k; ++j) {
        if (chosen[j] >= 0) kept_sum += chosen_val[j];
    }
    const float thr = chosen_val[k - 1];
    const int thr_id = chosen[k - 1];
    const float inv = kept_sum > 0.0f ? 1.0f / kept_sum : 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float p = float(probs[base + i]);
        const bool keep = (k >= V) || p > thr || (p == thr && i <= thr_id);
        out[base + i] = keep ? T(p * inv) : T(0.0f);
    }
}

// top-p renormalized PROBS: 32-iteration bisection on the probability threshold (the TM
// no-sort idiom, deliberately tighter than metal-forge's 5 iterations), then rescale the
// kept mass to 1. top_p >= 1 keeps everything.
template <typename T>
kernel void top_p_renorm_probs(device const T *probs [[buffer(0)]],
                               device T       *out   [[buffer(1)]],
                               constant int   &V     [[buffer(2)]],
                               constant float &top_p [[buffer(3)]],
                               uint row  [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    float total = 0.0f, pmax = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float p = float(probs[base + i]);
        total += p;
        pmax = max(pmax, p);
    }
    total = simd_sum(total);
    pmax = simd_max(pmax);
    const float target = top_p * total;
    float thr = 0.0f;
    if (top_p < 1.0f && target > 0.0f) {
        float lo = 0.0f, hi = pmax;
        for (int it = 0; it < 32; ++it) {
            const float mid = 0.5f * (lo + hi);
            float mass = 0.0f;
            for (int i = (int)lane; i < V; i += 32) {
                const float p = float(probs[base + i]);
                if (p >= mid) mass += p;
            }
            mass = simd_sum(mass);
            if (mass >= target) lo = mid; else hi = mid;
        }
        thr = lo;
    }
    float kept = 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float p = float(probs[base + i]);
        if (p >= thr) kept += p;
    }
    kept = simd_sum(kept);
    const float inv = kept > 0.0f ? 1.0f / kept : 0.0f;
    for (int i = (int)lane; i < V; i += 32) {
        const float p = float(probs[base + i]);
        out[base + i] = (p >= thr) ? T(p * inv) : T(0.0f);
    }
}

// no-repeat-ngram: ban every token that would complete an already-seen ngram_size-gram
// (per-lane over history start positions; concurrent -inf scatters are benign). n >= 2.
template <typename T>
kernel void no_repeat_ngram_mask(device const T *logits [[buffer(0)]],
                                 device const int *prev [[buffer(1)]],   // (rows, L)
                                 device const int *lens [[buffer(2)]],   // (rows,)
                                 device T       *out    [[buffer(3)]],
                                 constant int   &V      [[buffer(4)]],
                                 constant int   &L      [[buffer(5)]],
                                 constant int   &ngram  [[buffer(6)]],
                                 constant float &invtemp [[buffer(7)]],
                                 uint row  [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const long base = (long)row * V;
    for (int i = (int)lane; i < V; i += 32) {
        out[base + i] = T(float(logits[base + i]) * invtemp);
    }
    simdgroup_barrier(mem_flags::mem_device);
    const int len = min(lens[row], L);
    const int n = ngram;
    if (n < 2 || len < n) { return; }
    device const int *h = prev + (long)row * L;
    // suffix = the last n-1 tokens; a match at start s bans h[s + n - 1]
    for (int s = (int)lane; s + n - 1 < len; s += 32) {
        bool match = true;
        for (int j = 0; j < n - 1; ++j) {
            if (h[s + j] != h[len - (n - 1) + j]) { match = false; break; }
        }
        if (match) {
            const int banned = h[s + n - 1];
            if (banned >= 0 && banned < V) {
                out[base + banned] = T(SMPT_NEG_INF);
            }
        }
    }
}

// DRY repetition penalty (faithful transcription of the metal-forge/vLLM-community loop; see
// dry_penalty_logits in the reference). The outer occurrence walk stays simdgroup-uniform;
// the O(max_ngram) inner match-length unwind is parallelized across lanes via a
// first-violation simd_min — exact, since the serial loop stops at the first violation.
// TM divergences: launch-uniform scalar params and ONE shared breakers list (NB, pad -1)
// instead of per-row parameter arrays; the penalty applies to the TEMPERED logit.
template <typename T>
kernel void dry_penalty(device const T *logits  [[buffer(0)]],
                        device const int *prev  [[buffer(1)]],   // (rows, L)
                        device const int *lens  [[buffer(2)]],   // (rows,)
                        device const int *breakers [[buffer(3)]], // (NB,) pad -1
                        device T       *out     [[buffer(4)]],
                        constant int   &V       [[buffer(5)]],
                        constant int   &L       [[buffer(6)]],
                        constant int   &NB      [[buffer(7)]],
                        constant float &multiplier [[buffer(8)]],
                        constant float &dbase   [[buffer(9)]],
                        constant int   &allowed [[buffer(10)]],
                        constant int   &range   [[buffer(11)]],
                        constant int   &max_ngram [[buffer(12)]],
                        constant int   &max_occ [[buffer(13)]],
                        constant int   &early_exit [[buffer(14)]],
                        constant float &invtemp [[buffer(15)]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    const long vbase = (long)row * V;
    for (int i = (int)lane; i < V; i += 32) {
        out[vbase + i] = T(float(logits[vbase + i]) * invtemp);
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (multiplier == 0.0f) { return; }
    const int len = min(lens[row], L);
    if (len < 2) { return; }
    device const int *h = prev + (long)row * L;

    // uniform helper: is tok a sequence breaker?
    const int last = h[len - 1];
    bool last_is_breaker = false;
    for (int b = 0; b < NB; ++b) {
        if (breakers[b] == last) { last_is_breaker = true; }
    }
    if (last_is_breaker) { return; }

    const int start_idx = range > 0 ? max(0, len - range) : 0;
    int curr_max_ngram = -1;
    const int ngram_cap = min(len - start_idx, max_ngram + 1);
    for (int gi = 0; gi < ngram_cap; ++gi) {
        const int tok = h[len - gi - 1];
        bool brk = false;
        for (int b = 0; b < NB; ++b) {
            if (breakers[b] == tok) { brk = true; }
        }
        if (brk) { break; }
        curr_max_ngram = gi;
    }
    if (curr_max_ngram <= allowed) { return; }

    int seen = 0;
    for (int idx = len - 2; idx >= start_idx; --idx) {
        if (h[idx] != last) { continue; }
        if (seen >= max_occ) { break; }
        ++seen;

        const int max_unwind = min(idx - start_idx, curr_max_ngram);
        // parallel first-violation scan: lane l tests offsets l+1, l+33, ...
        int my_viol = max_unwind + 1;
        for (int o = (int)lane + 1; o <= max_unwind; o += 32) {
            const int tok = h[idx - o];
            bool viol = tok != h[len - o - 1];
            for (int b = 0; b < NB && !viol; ++b) {
                if (breakers[b] == tok) { viol = true; }
            }
            if (viol) { my_viol = min(my_viol, o); break; }   // later offsets in this lane are moot
        }
        const int fv = simd_min(my_viol);
        const int match_len = min(fv - 1, max_unwind);
        if (match_len <= 0) { continue; }

        const int next_token = h[idx + 1];
        if (next_token >= 0 && next_token < V) {
            const int new_len = match_len + 1;
            if (lane == 0) {
                const float penalty = multiplier * metal::pow(dbase, float(new_len - allowed));
                const float tempered = float(logits[vbase + next_token]) * invtemp;
                out[vbase + next_token] = T(min(float(out[vbase + next_token]),
                                                tempered - penalty));
            }
            if (new_len >= early_exit) { break; }
        }
    }
}

#define instantiate_transforms(type_name, T)                                              \
  template [[host_name("logits_softcap_" #type_name)]] [[kernel]] void                   \
  logits_softcap<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],    \
      constant uint &n [[buffer(2)]], constant float &cap [[buffer(3)]],                   \
      uint tid [[thread_position_in_grid]]);                                               \
  template [[host_name("value_clip_" #type_name)]] [[kernel]] void                       \
  value_clip<T>(device const T *x [[buffer(0)]], device T *out [[buffer(1)]],              \
      constant uint &n [[buffer(2)]], constant float &min_value [[buffer(3)]],             \
      constant float &max_value [[buffer(4)]],                                             \
      uint tid [[thread_position_in_grid]]);                                               \
  template [[host_name("quadratic_transform_" #type_name)]] [[kernel]] void               \
  quadratic_transform<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],\
      constant int &V [[buffer(2)]], constant float &factor [[buffer(3)]],                \
      constant float &curve [[buffer(4)]], constant float &invtemp [[buffer(5)]],         \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("top_nsigma_mask_" #type_name)]] [[kernel]] void                   \
  top_nsigma_mask<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],   \
      constant int &V [[buffer(2)]], constant float &nsigma [[buffer(3)]],                \
      constant float &invtemp [[buffer(4)]],                                              \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("top_a_mask_" #type_name)]] [[kernel]] void                        \
  top_a_mask<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],        \
      constant int &V [[buffer(2)]], constant float &top_a [[buffer(3)]],                 \
      constant float &invtemp [[buffer(4)]],                                              \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("epsilon_cutoff_mask_" #type_name)]] [[kernel]] void               \
  epsilon_cutoff_mask<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],\
      constant int &V [[buffer(2)]], constant float &eps [[buffer(3)]],                   \
      constant float &invtemp [[buffer(4)]],                                              \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("eta_cutoff_mask_" #type_name)]] [[kernel]] void                   \
  eta_cutoff_mask<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],   \
      constant int &V [[buffer(2)]], constant float &eta [[buffer(3)]],                   \
      constant float &invtemp [[buffer(4)]],                                              \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("xtc_mask_" #type_name)]] [[kernel]] void                          \
  xtc_mask<T>(device const T *logits [[buffer(0)]], device T *out [[buffer(1)]],          \
      constant int &V [[buffer(2)]], constant float &threshold [[buffer(3)]],             \
      constant float &probability [[buffer(4)]], constant float &invtemp [[buffer(5)]],   \
      constant uint &seed [[buffer(6)]],                                                  \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("skew_transform_" #type_name)]] [[kernel]] void                    \
  skew_transform<T>(device const T *probs [[buffer(0)]], device T *out [[buffer(1)]],     \
      constant int &V [[buffer(2)]], constant float &skew [[buffer(3)]],                  \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("top_k_renorm_probs_" #type_name)]] [[kernel]] void                \
  top_k_renorm_probs<T>(device const T *probs [[buffer(0)]], device T *out [[buffer(1)]], \
      constant int &V [[buffer(2)]], constant int &K [[buffer(3)]],                       \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("top_p_renorm_probs_" #type_name)]] [[kernel]] void                \
  top_p_renorm_probs<T>(device const T *probs [[buffer(0)]], device T *out [[buffer(1)]], \
      constant int &V [[buffer(2)]], constant float &top_p [[buffer(3)]],                 \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("no_repeat_ngram_mask_" #type_name)]] [[kernel]] void              \
  no_repeat_ngram_mask<T>(device const T *logits [[buffer(0)]],                           \
      device const int *prev [[buffer(1)]], device const int *lens [[buffer(2)]],         \
      device T *out [[buffer(3)]], constant int &V [[buffer(4)]],                         \
      constant int &L [[buffer(5)]], constant int &ngram [[buffer(6)]],                   \
      constant float &invtemp [[buffer(7)]],                                              \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);\
  template [[host_name("dry_penalty_" #type_name)]] [[kernel]] void                       \
  dry_penalty<T>(device const T *logits [[buffer(0)]],                                    \
      device const int *prev [[buffer(1)]], device const int *lens [[buffer(2)]],         \
      device const int *breakers [[buffer(3)]], device T *out [[buffer(4)]],              \
      constant int &V [[buffer(5)]], constant int &L [[buffer(6)]],                       \
      constant int &NB [[buffer(7)]], constant float &multiplier [[buffer(8)]],           \
      constant float &dbase [[buffer(9)]], constant int &allowed [[buffer(10)]],          \
      constant int &range [[buffer(11)]], constant int &max_ngram [[buffer(12)]],         \
      constant int &max_occ [[buffer(13)]], constant int &early_exit [[buffer(14)]],      \
      constant float &invtemp [[buffer(15)]],                                             \
      uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]);

instantiate_transforms(float32, float)
instantiate_transforms(float16, half)
instantiate_transforms(bfloat16, bf16)
