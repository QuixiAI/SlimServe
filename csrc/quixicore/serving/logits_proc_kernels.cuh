/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's logits-processor "tail-cutoff zoo"
 * (sample_top_p.metal) that our sampler stack didn't have: top-nsigma, top-a,
 * epsilon/eta cutoff, XTC, quadratic transform (warp-per-row, strided vocab —
 * the apply_penalty skeleton), skew transform (CDF^exp over probs, block scan),
 * and the serial no-repeat-ngram / DRY penalties. All deterministic (they mask
 * or transform logits/probs; the downstream draw stays on our rng_gumbel path).
 *
 * Per-row scalar params (nsigma[row], top_a[row], ...) match MetalForge.
 */
#pragma once
#include "tm_warp.cuh"                   // warp_sum_f/max_f/min_f, block_inclusive_scan_f
#include <cuda_fp16.h>

namespace tmlp {

#define LP_NEG_INF (-3.4028234663852886e38f)

// ---- top-nsigma: threshold = max - nsigma*sqrt(var); keep value >= threshold.
// var = (Sxx - Sx^2/n)/(n-1), clamped >= 0. Raw logits (no softmax). ----
__global__ void top_nsigma_mask(const float* logits, float* out, int V, const float* nsigma) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    float mx = LP_NEG_INF, sx = 0.0f, sxx = 0.0f;
    for (int i = lane; i < V; i += 32) { const float v = logits[base+i]; mx = fmaxf(mx,v); sx += v; sxx += v*v; }
    mx = tms::warp_max_f(mx); sx = tms::warp_sum_f(sx); sxx = tms::warp_sum_f(sxx);
    const float var = fmaxf((sxx - sx*sx/V) / float(V-1), 0.0f);
    const float thr = mx - nsigma[row] * sqrtf(var);
    for (int i = lane; i < V; i += 32) out[base+i] = logits[base+i] < thr ? LP_NEG_INF : logits[base+i];
}

// ---- top-a: threshold = (1/Z)^2 * top_a; drop prob < threshold. ----
__global__ void top_a_mask(const float* logits, float* out, int V, const float* top_a) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    float mx = LP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, logits[base+i]);
    mx = tms::warp_max_f(mx);
    float Z = 0.0f;
    for (int i = lane; i < V; i += 32) Z += expf(logits[base+i]-mx);
    Z = tms::warp_sum_f(Z);
    const float top_prob = 1.0f / Z;
    const float thr = top_prob*top_prob*top_a[row];
    for (int i = lane; i < V; i += 32) {
        const float p = expf(logits[base+i]-mx)/Z;
        out[base+i] = p < thr ? LP_NEG_INF : logits[base+i];
    }
}

// ---- epsilon cutoff: drop iff value<max && prob<eps[row]. ----
__global__ void epsilon_cutoff_mask(const float* logits, float* out, int V, const float* eps) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    float mx = LP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, logits[base+i]);
    mx = tms::warp_max_f(mx);
    float Z = 0.0f;
    for (int i = lane; i < V; i += 32) Z += expf(logits[base+i]-mx);
    Z = fmaxf(tms::warp_sum_f(Z), 1e-20f);
    const float thr = eps[row];
    for (int i = lane; i < V; i += 32) {
        const float v = logits[base+i], p = expf(v-mx)/Z;
        out[base+i] = (v < mx && p < thr) ? LP_NEG_INF : v;
    }
}

// ---- eta cutoff: eps = min(eta, sqrt(eta)*exp(sum p*logp)); drop value<max && prob<eps. ----
__global__ void eta_cutoff_mask(const float* logits, float* out, int V, const float* eta) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    float mx = LP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, logits[base+i]);
    mx = tms::warp_max_f(mx);
    float Z = 0.0f;
    for (int i = lane; i < V; i += 32) Z += expf(logits[base+i]-mx);
    Z = fmaxf(tms::warp_sum_f(Z), 1e-20f);
    const float logZ = logf(Z);
    float negH = 0.0f;                             // sum p * logp
    for (int i = lane; i < V; i += 32) {
        const float sh = logits[base+i]-mx, p = expf(sh)/Z;
        if (p > 0.0f) negH += p * (sh - logZ);
    }
    negH = tms::warp_sum_f(negH);
    const float e = eta[row];
    const float eps = fminf(e, sqrtf(e) * expf(negH));
    for (int i = lane; i < V; i += 32) {
        const float v = logits[base+i], p = expf(v-mx)/Z;
        out[base+i] = (v < mx && p < eps) ? LP_NEG_INF : v;
    }
}

// ---- XTC: remove high-prob tokens except the least-probable eligible one.
// eligible = prob>=thr; keep_prob = min eligible prob; remove iff
// eligible>1 && prob>=thr && prob>keep_prob. ----
__global__ void xtc_mask(const float* logits, float* out, int V, const float* thresholds,
                         const int* apply_xtc) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    if (!apply_xtc[row]) { for (int i = lane; i < V; i += 32) out[base+i] = logits[base+i]; return; }
    float mx = LP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, logits[base+i]);
    mx = tms::warp_max_f(mx);
    float Z = 0.0f;
    for (int i = lane; i < V; i += 32) Z += expf(logits[base+i]-mx);
    Z = tms::warp_sum_f(Z);
    const float thr = thresholds[row];
    int cnt = 0; float keep_prob = 3.4028234663852886e38f;
    for (int i = lane; i < V; i += 32) {
        const float p = expf(logits[base+i]-mx)/Z;
        if (p >= thr) { cnt++; keep_prob = fminf(keep_prob, p); }
    }
    cnt = tms::warp_sum_i32(cnt);
    cnt = __shfl_sync(0xffffffffu, cnt, 0);        // broadcast the reduced count
    keep_prob = tms::warp_min_f(keep_prob);
    for (int i = lane; i < V; i += 32) {
        const float v = logits[base+i], p = expf(v-mx)/Z;
        const bool remove = cnt > 1 && p >= thr && p > keep_prob;
        out[base+i] = remove ? LP_NEG_INF : v;
    }
}

// ---- quadratic transform: diff=value-max; diff -= diff^2*(s*diff - k),
// k=factor*(3-curve)/2, s=factor*(curve-1)/2; out = value-diff (if finite). ----
__global__ void quadratic_transform(const float* logits, float* out, int V,
                                    const float* factors, const float* curves) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    const float factor = factors[row];
    if (factor == 0.0f) { for (int i = lane; i < V; i += 32) out[base+i] = logits[base+i]; return; }
    float mx = LP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, logits[base+i]);
    mx = tms::warp_max_f(mx);
    const float curve = curves[row];
    const float k = factor*(3.0f-curve)*0.5f, s = factor*(curve-1.0f)*0.5f;
    for (int i = lane; i < V; i += 32) {
        const float v = logits[base+i];
        float diff = v - mx;
        diff -= diff*diff*(s*diff - k);
        out[base+i] = isfinite(diff) ? (v - diff) : v;
    }
}

// ---- skew transform (on PROBS): out[t] = CDF(t)^exp - CDF(t-1)^exp,
// exp = exp(skew[row]). Each thread owns a CONTIGUOUS vocab chunk; block scan
// of chunk totals gives the CDF start. 256-thread block. ----
__global__ void skew_transform(const float* probs, float* out, int V, const float* skews) {
    __shared__ float sg[8];
    const int row = blockIdx.x, tid = threadIdx.x, nt = blockDim.x;
    const long base = (long)row * V;
    const float ex = expf(skews[row]);
    const int chunk = (V + nt - 1) / nt;
    const int c0 = tid * chunk, c1 = min(c0 + chunk, V);
    float mine = 0.0f;
    for (int t = c0; t < c1; ++t) mine += probs[base + t];
    float total;
    const float incl = tms::block_inclusive_scan_f(mine, tid, nt, sg, total);
    float cumulative = incl - mine;                // exclusive start for this chunk
    float previous = powf(cumulative, ex);
    for (int t = c0; t < c1; ++t) {
        cumulative += probs[base + t];
        const float tr = powf(cumulative, ex);
        out[base + t] = tr - previous;
        previous = tr;
    }
}

// ---- no-repeat-ngram: ban any token that would complete a repeat of the last
// (ngram-1) history tokens. history (rows, hist_stride) int; hist_len[row].
// Copy logits->out (strided), then lane 0 scans history for prefix matches and
// bans the following token. ngram <= 8. ----
__global__ void no_repeat_ngram_mask(const float* logits, float* out, int V,
                                     const int* history, const int* hist_len, int hist_stride,
                                     int ngram) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    for (int i = lane; i < V; i += 32) out[base+i] = logits[base+i];
    __syncwarp();
    if (lane != 0) return;
    const int H = hist_len[row], p = ngram - 1;
    if (H < p) return;
    const int* h = history + (long)row * hist_stride;
    for (int start = 0; start + p < H; ++start) {   // does h[start..start+p-1] == suffix?
        bool match = true;
        for (int j = 0; j < p; ++j) if (h[start + j] != h[H - p + j]) { match = false; break; }
        if (match) { const int ban = h[start + p]; if (ban >= 0 && ban < V) out[base + ban] = LP_NEG_INF; }
    }
}

// ---- DRY penalty: for the longest suffix of history ending at the last token
// (respecting sequence_breakers), penalize the token(s) that followed earlier
// occurrences by multiplier * base^(match_len - allowed_length). Simplified vLLM
// DRY: scan for matches of the growing suffix; penalize out[next]. One warp;
// lane 0 walks. history (rows, hist_stride); breakers is a sorted set per row
// omitted here (pass breaker_flag[token]!=0 to stop a match). ----
__global__ void dry_penalty(const float* logits, float* out, int V,
                            const int* history, const int* hist_len, int hist_stride,
                            const uint8_t* is_breaker, const float* multiplier,
                            const float* base_, const int* allowed_length, int max_ngram,
                            int max_occurrences) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const long base = (long)row * V;
    for (int i = lane; i < V; i += 32) out[base+i] = logits[base+i];
    __syncwarp();
    if (lane != 0) return;
    const int H = hist_len[row];
    if (H < 1) return;
    const int* h = history + (long)row * hist_stride;
    const int last = h[H - 1];
    const float mult = multiplier[row], bs = base_[row];
    const int allow = allowed_length[row];
    int occ = 0;
    // for each earlier occurrence of `last`, measure the backward match length
    for (int p = H - 2; p >= 0 && occ < max_occurrences; --p) {
        if (h[p] != last) continue;
        int mlen = 1;
        while (mlen < max_ngram && p - mlen >= 0 && (H - 1 - mlen) >= 0
               && h[p - mlen] == h[H - 1 - mlen]
               && (is_breaker == nullptr || !is_breaker[h[p - mlen]])) ++mlen;
        const int next = (p + 1 < H) ? h[p + 1] : -1;
        if (next >= 0 && next < V && mlen + 1 > allow) {
            const float pen = mult * powf(bs, float(mlen + 1 - allow));
            out[base + next] = fminf(out[base + next], logits[base + next] - pen);
        }
        ++occ;
    }
}

} // namespace tmlp
