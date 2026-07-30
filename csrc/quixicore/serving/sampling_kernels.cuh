#pragma once
// Sampling kernels, CUDA/SM86 port of ThunderMittens kernels/sampling (sampler
// half; beam + spec-decode live in spec_beam.cu). One warp per logits row, vocab
// looped stride-32. Every stochastic draw is a pure function of (seed, row, vocab
// id) via tm_rng.cuh - bit-identical on host, so oracles are exact.
//
//   argmax, sample_categorical (Gumbel-max), top_k_sample (K masked-argmax
//   rounds, K<=64), top_p_sample (32-step logit-threshold bisection, no sort),
//   min_p_sample, typical_p_sample (entropy + surprise-threshold bisection),
//   penalty_histogram + apply_penalty (temperature/rep/presence/freq/bias/
//   min-length EOS mask, vLLM order), apply_token_bitmask (grammar),
//   apply_bad_words.
//
// Build:
//   /usr/local/cuda/bin/nvcc sampling.cu -std=c++20 -O2 -DKITTENS_SM86 \
//     -gencode arch=compute_86,code=sm_86 -I../quant -o sampling.out
#include "tm_rng.cuh"
#include "tm_warp.cuh"
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

using namespace tmq;

namespace tms {

#define SMP_NEG_INF (-3.4028234663852886e38f)
#define SAMPLE_MAX_K 64

// argmax with ties toward the smaller id (numpy semantics)

template <typename T>
__global__ void argmax_k(const T* logits, int* out_idx, int V) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    float best = SMP_NEG_INF;
    int bi = lane < V ? lane : 0;
    for (int i = lane; i < V; i += 32) {
        const float v = float(logits[base + i]);
        if (v > best || (v == best && i < bi)) { best = v; bi = i; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[row] = bi;
}

template <typename T>
__global__ void sample_categorical(const T* logits, int* out_idx, int V, unsigned seed, float invtemp) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    float best = SMP_NEG_INF;
    int bi = lane < V ? lane : 0;
    for (int i = lane; i < V; i += 32) {
        const float p = float(logits[base + i]) * invtemp + rng_gumbel(seed, row, i);
        if (p > best || (p == best && i < bi)) { best = p; bi = i; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[row] = bi;
}

template <typename T>
__global__ void top_k_sample(const T* logits, int* out_idx, int V, int K, unsigned seed, float invtemp) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    int chosen_id[SAMPLE_MAX_K];
    float chosen_logit[SAMPLE_MAX_K];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = idx; v = float(logits[base + idx]); valid = true;
    };
    masked_topk(cand, V, K, lane, SMP_NEG_INF, chosen_id, chosen_logit);
    float best = SMP_NEG_INF;
    int bi = chosen_id[0];
    for (int j = 0; j < K; j++) {
        const float p = chosen_logit[j] * invtemp + rng_gumbel(seed, row, chosen_id[j]);
        if (p > best || (p == best && chosen_id[j] < bi)) { best = p; bi = chosen_id[j]; }
    }
    if (lane == 0) out_idx[row] = bi;
}

__global__ void penalty_histogram(const int* prev_tokens, int* counts, int V, int L, int TL,
                                  const int* parent_ids) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TL) return;
    const int row = tid / L, col = tid - row * L;
    const int tok = prev_tokens[int64_t(parent_ids[row]) * L + col];
    if (tok >= 0 && tok < V) atomicAdd(&counts[int64_t(row) * V + tok], 1);
}

template <typename T>
__global__ void apply_penalty(const T* logits, const int* counts, T* out, int V,
                              float invtemp, float rep, float presence, float freq,
                              const float* bias, int eos_id, int min_length, int gen_len) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    const bool mask_eos = (eos_id >= 0) && (gen_len < min_length);
    for (int v = lane; v < V; v += 32) {
        float ls = float(logits[base + v]) * invtemp;
        const int c = counts[base + v];
        if (c > 0) {
            ls = (ls < 0.0f) ? (ls * rep) : (ls / rep);
            ls -= presence;
            ls -= freq * float(c);
        }
        ls += bias[v];
        if (mask_eos && v == eos_id) ls = SMP_NEG_INF;
        out[base + v] = T(ls);
    }
}

template <typename T>
__global__ void top_p_sample(const T* logits, int* out_idx, int V, float p, unsigned seed, float invtemp) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    float mx = SMP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, float(logits[base + i]) * invtemp);
    mx = warp_max_f(mx);
    float Z = 0.0f;
    for (int i = lane; i < V; i += 32) Z += expf(float(logits[base + i]) * invtemp - mx);
    Z = warp_sum_f(Z);
    float lo = mx - 40.0f, hi = mx;
    for (int it = 0; it < 32; it++) {
        const float mid = 0.5f * (lo + hi);
        float sm = 0.0f;
        for (int i = lane; i < V; i += 32) {
            const float ls = float(logits[base + i]) * invtemp;
            if (ls >= mid) sm += expf(ls - mx);
        }
        sm = warp_sum_f(sm) / Z;
        if (sm >= p) lo = mid; else hi = mid;
    }
    const float L = lo;
    float best = SMP_NEG_INF;
    int bi = lane < V ? lane : 0;
    for (int i = lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (ls < L) continue;
        const float pert = ls + rng_gumbel(seed, row, i);
        if (pert > best || (pert == best && i < bi)) { best = pert; bi = i; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[row] = bi;
}

template <typename T>
__global__ void min_p_sample(const T* logits, int* out_idx, int V, float min_p, unsigned seed, float invtemp) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    float m = SMP_NEG_INF;
    for (int i = lane; i < V; i += 32) m = fmaxf(m, float(logits[base + i]) * invtemp);
    m = warp_max_f(m);
    const float thresh = m + logf(min_p);
    float best = SMP_NEG_INF;
    int bi = lane < V ? lane : 0;
    for (int i = lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (ls < thresh) continue;
        const float pert = ls + rng_gumbel(seed, row, i);
        if (pert > best || (pert == best && i < bi)) { best = pert; bi = i; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[row] = bi;
}

template <typename T>
__global__ void typical_p_sample(const T* logits, int* out_idx, int V, float typ_p, unsigned seed, float invtemp) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    float mx = SMP_NEG_INF;
    for (int i = lane; i < V; i += 32) mx = fmaxf(mx, float(logits[base + i]) * invtemp);
    mx = warp_max_f(mx);
    float Z = 0.0f, S1 = 0.0f;
    for (int i = lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        const float e = expf(ls - mx);
        Z += e; S1 += e * ls;
    }
    Z = warp_sum_f(Z); S1 = warp_sum_f(S1);
    const float logZ = logf(Z);
    const float H = mx + logZ - S1 / Z;
    float smax = 0.0f;
    for (int i = lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        smax = fmaxf(smax, fabsf((mx + logZ - ls) - H));
    }
    smax = warp_max_f(smax);
    float lo = 0.0f, hi = smax;
    for (int it = 0; it < 32; it++) {
        const float mid = 0.5f * (lo + hi);
        float mass = 0.0f;
        for (int i = lane; i < V; i += 32) {
            const float ls = float(logits[base + i]) * invtemp;
            if (fabsf((mx + logZ - ls) - H) <= mid) mass += expf(ls - mx);
        }
        mass = warp_sum_f(mass) / Z;
        if (mass >= typ_p) hi = mid; else lo = mid;
    }
    const float tau = hi;
    float best = SMP_NEG_INF;
    int bi = lane < V ? lane : 0;
    for (int i = lane; i < V; i += 32) {
        const float ls = float(logits[base + i]) * invtemp;
        if (fabsf((mx + logZ - ls) - H) > tau) continue;
        const float pert = ls + rng_gumbel(seed, row, i);
        if (pert > best || (pert == best && i < bi)) { best = pert; bi = i; }
    }
    warp_argmax(best, bi);
    if (lane == 0) out_idx[row] = bi;
}

template <typename T>
__global__ void apply_token_bitmask(const T* logits, const uint32_t* bitmask, T* out, int V, int words) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    const uint32_t* bm = bitmask + int64_t(row) * words;
    for (int v = lane; v < V; v += 32) {
        const bool allow = (bm[v >> 5] >> (v & 31)) & 1;
        out[base + v] = allow ? logits[base + v] : T(SMP_NEG_INF);
    }
}

template <typename T>
__global__ void apply_bad_words(const T* logits, T* out, const int* bad_ids, const int* bad_lens,
                                int V, int max_bad) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    for (int v = lane; v < V; v += 32) out[base + v] = logits[base + v];
    __syncwarp();
    const int n = bad_lens[row];
    for (int j = lane; j < n && j < max_bad; j += 32) {
        const int id = bad_ids[int64_t(row) * max_bad + j];
        if (id >= 0 && id < V) out[base + id] = T(SMP_NEG_INF);
    }
}

}  // namespace tms

