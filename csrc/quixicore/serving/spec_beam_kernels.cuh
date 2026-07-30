#pragma once
// Beam-search advance + speculative decoding, CUDA/SM86 port of the second half
// of ThunderMittens kernels/sampling:
//   beam_topk_partials - per beam: ONE vocab pass doing online LSE + per-lane
//     top-2BM local sets, merged by masked-argmax rounds (Family-B); candidate
//     score = cum_log_prob + (logit - lse)
//   beam_select        - per batch: top-BM over the BM*2BM candidates -> next
//     token / parent beam / new cumulative logprob (TRT-LLM 2-pass, BM<=16)
//   spec_verify_linear - vLLM rejection sampling: accept iff u*p_draft <= p_target;
//     first reject -> residual (p_t-p_d)+ Gumbel sample; all accept -> bonus
//   spec_verify_tree + build_dynamic_tree - TRT-LLM dynamicTree target-only
//     rejection (lane-0 serial walk, cooperative terminal Gumbel sample)
//   spec_compact       - pack accepted+terminal tokens with cu offsets (the
//     chunked block scan) + absolute KV positions
//   spec_update_kv_meta
//
// Build:
//   /usr/local/cuda/bin/nvcc spec_beam.cu -std=c++20 --extended-lambda -O2 \
//     -DKITTENS_SM86 -gencode arch=compute_86,code=sm_86 -I../quant -o spec_beam.out
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

template <typename T>
__global__ void beam_topk_partials(const T* logits, const float* cum_log_probs,
                                   float* cand_score, int* cand_token, int V, int two_bm) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const int64_t base = int64_t(row) * V;
    float m = SMP_NEG_INF, l = 0.0f;
    float lv[SAMPLE_MAX_K];
    int li[SAMPLE_MAX_K];
    for (int k = 0; k < two_bm; k++) { lv[k] = SMP_NEG_INF; li[k] = -1; }
    float minv = SMP_NEG_INF;
    int minp = 0;
    for (int i = lane; i < V; i += 32) {
        const float x = float(logits[base + i]);
        const float nm = fmaxf(m, x);
        l = l * expf(m - nm) + expf(x - nm);
        m = nm;
        if (x > minv) {
            lv[minp] = x; li[minp] = i;
            minv = lv[0]; minp = 0;
            for (int k = 1; k < two_bm; k++) if (lv[k] < minv) { minv = lv[k]; minp = k; }
        }
    }
    const float M = warp_max_f(m);
    l = warp_sum_f(l * expf(m - M));
    const float lse = M + logf(l);
    const float cumr = cum_log_probs[row];
    const int64_t obase = int64_t(row) * two_bm;
    bool used[SAMPLE_MAX_K];
    masked_topk_local(lv, li, used, two_bm, two_bm, SMP_NEG_INF,
        [&](int kk, float gbest, int gid) {
            if (lane == 0) {
                cand_token[obase + kk] = (gbest == SMP_NEG_INF) ? -1 : gid;
                cand_score[obase + kk] = (gbest == SMP_NEG_INF) ? SMP_NEG_INF : cumr + (gbest - lse);
            }
        });
}

__global__ void beam_select(const float* cand_score, const int* cand_token,
                            int* next_token, int* parent_beam, float* new_cum,
                            int BM, int two_bm) {
    const int b = blockIdx.x, lane = threadIdx.x;
    const int ncand = BM * two_bm;
    const int64_t row0 = int64_t(b) * BM;
    int chosen[16];
    float chosen_sc[16];
    auto cand = [&](int idx, int& id, float& v, bool& valid) {
        id = idx; v = cand_score[row0 * two_bm + idx]; valid = true;
    };
    masked_topk(cand, ncand, BM, lane, SMP_NEG_INF, chosen, chosen_sc);
    if (lane == 0) {
        for (int k = 0; k < BM; k++) {
            const int gc = chosen[k];
            const int i = gc / two_bm, j = gc - i * two_bm;
            next_token[row0 + k] = cand_token[(row0 + i) * two_bm + j];
            parent_beam[row0 + k] = i;
            new_cum[row0 + k] = chosen_sc[k];
        }
    }
}

#define SPEC_PLACEHOLDER (-1)

__global__ void spec_verify_linear(const int* draft_tokens, const float* draft_probs,
                                   const float* target_probs, const int* bonus_tokens,
                                   const float* accept_u, int* out_tokens, int* accepted_cnt,
                                   int S, int V, unsigned seed) {
    const int b = blockIdx.x, lane = threadIdx.x;
    int rejected_at = S;
    for (int i = 0; i < S; i++) {
        const int dt = draft_tokens[b * S + i];
        const int64_t tbase = (int64_t(b) * (S + 1) + i) * V;
        const int64_t dbase = (int64_t(b) * S + i) * V;
        const float p_t = target_probs[tbase + dt];
        const float p_d = draft_probs[dbase + dt];
        const float u = accept_u[b * S + i];
        const bool accept = (p_d <= 0.0f) ? true : (u * p_d <= p_t);
        if (accept) {
            if (lane == 0) out_tokens[b * (S + 1) + i] = dt;
            continue;
        }
        float best = SMP_NEG_INF;
        int bi = 0;
        for (int v = lane; v < V; v += 32) {
            const float r = fmaxf(0.0f, target_probs[tbase + v] - draft_probs[dbase + v]);
            const float logit = (r > 0.0f) ? logf(r) : SMP_NEG_INF;
            const float g = logit + rng_gumbel(seed, unsigned(b * S + i), unsigned(v));
            if (g > best || (g == best && v < bi)) { best = g; bi = v; }
        }
        warp_argmax(best, bi);
        if (lane == 0) out_tokens[b * (S + 1) + i] = bi;
        rejected_at = i;
        break;
    }
    if (rejected_at == S) {
        if (lane == 0) out_tokens[b * (S + 1) + S] = bonus_tokens[b];
    } else {
        for (int i = rejected_at + 1; i <= S; i++)
            if (lane == 0) out_tokens[b * (S + 1) + i] = SPEC_PLACEHOLDER;
    }
    if (lane == 0) accepted_cnt[b] = rejected_at;
}

__global__ void spec_verify_tree(const int* draft_tokens, const float* target_probs,
                                 const int* retrieve_next_token, const int* retrieve_next_sibling,
                                 int* accept_index, int* accept_token, int* accept_num,
                                 int N, int V, unsigned seed, const int* tree_valid) {
    __shared__ int s_num_accepted, s_last, s_term;
    const int b = blockIdx.x, lane = threadIdx.x;
    const int64_t nbase = int64_t(b) * N;
    for (int i = lane; i < N; i += 32) {
        accept_index[nbase + i] = -1;
        accept_token[nbase + i] = -1;
    }
    __syncwarp();
    if (lane == 0) {
        if (tree_valid[b] == 0) {
            accept_index[nbase] = 0;
            s_num_accepted = 0; s_last = 0; s_term = 1;
        } else {
            int last = 0, num_acc = 0, term = 0;
            accept_index[nbase] = 0;
            for (int j = 1; j < N; j++) {
                const int firstChild = retrieve_next_token[nbase + last];
                if (firstChild == -1) { term = 1; break; }
                const float coin = rng_uniform(seed, unsigned(b), unsigned(j));
                float probAcc = 0.0f;
                bool accepted = false;
                int child = firstChild;
                const float* parentProbs = target_probs + (nbase + last) * int64_t(V);
                while (child != -1) {
                    const int tok = draft_tokens[int64_t(b) * (N - 1) + (child - 1)];
                    probAcc += parentProbs[tok];
                    if (coin <= probAcc) {
                        accept_token[nbase + num_acc] = tok;
                        num_acc += 1;
                        accept_index[nbase + num_acc] = child;
                        last = child;
                        accepted = true;
                        break;
                    }
                    child = retrieve_next_sibling[nbase + child];
                }
                if (!accepted) { term = 2; break; }
            }
            s_num_accepted = num_acc; s_last = last; s_term = term;
        }
    }
    __syncwarp();
    const int term = s_term, last = s_last, num_acc = s_num_accepted;
    if (term != 0) {
        const float* tp = target_probs + (nbase + last) * int64_t(V);
        const int firstChild = (term == 2) ? retrieve_next_token[nbase + last] : -1;
        float best = SMP_NEG_INF;
        int bi = -1;
        for (int v = lane; v < V; v += 32) {
            const float pv = tp[v];
            if (pv <= 0.0f) continue;
            if (term == 2) {
                bool tried = false;
                for (int c = firstChild; c != -1; c = retrieve_next_sibling[nbase + c])
                    if (draft_tokens[int64_t(b) * (N - 1) + (c - 1)] == v) { tried = true; break; }
                if (tried) continue;
            }
            const float g = logf(pv) + rng_gumbel(seed + 0x2545F491u, unsigned(b), unsigned(v));
            if (g > best || (g == best && v < bi)) { best = g; bi = v; }
        }
        float gb = best;
        int gi = (bi < 0) ? 0x7fffffff : bi;
        warp_argmax(gb, gi);
        if (lane == 0) accept_token[nbase + num_acc] = (gb == SMP_NEG_INF) ? -1 : gi;
    }
    if (lane == 0) accept_num[b] = num_acc;
}

__global__ void build_dynamic_tree(const int* parents, int* retrieve_next_token,
                                   int* retrieve_next_sibling, int* positions, int N) {
    const int b = blockIdx.x, lane = threadIdx.x;
    const int64_t nb = int64_t(b) * N;
    for (int i = lane; i < N; i += 32) {
        retrieve_next_token[nb + i] = -1;
        retrieve_next_sibling[nb + i] = -1;
    }
    __syncwarp();
    for (int c = lane; c < N; c += 32) {
        int d = 0, x = c;
        while (parents[nb + x] >= 0) { x = parents[nb + x]; d += 1; }
        positions[nb + c] = d;
        const int p = parents[nb + c];
        if (c == 0 || p < 0) continue;
        int sib = -1;
        for (int c2 = c + 1; c2 < N; c2++) if (parents[nb + c2] == p) { sib = c2; break; }
        retrieve_next_sibling[nb + c] = sib;
        bool isFirst = true;
        for (int c2 = 1; c2 < c; c2++) if (parents[nb + c2] == p) { isFirst = false; break; }
        if (isFirst) retrieve_next_token[nb + p] = c;
    }
}

__global__ void spec_compact(const int* out_tokens, const int* accepted_cnt, const int* seq_lens,
                             int* packed_tokens, int* packed_pos, int* cu_accepted, int B, int Sp1) {
    __shared__ int sg_sums[8];
    const unsigned tid = threadIdx.x, nthreads = blockDim.x;
    const int chunk = (B + int(nthreads) - 1) / int(nthreads);
    int lo = int(tid) * chunk; if (lo > B) lo = B;
    int hi = lo + chunk;       if (hi > B) hi = B;
    int local_vlen = 0;
    for (int b = lo; b < hi; b++) local_vlen += accepted_cnt[b] + 1;
    int total = 0;
    const int base = block_exclusive_scan_i32(local_vlen, tid, nthreads, sg_sums, total);
    if (tid == 0) cu_accepted[B] = total;
    int run = base;
    for (int b = lo; b < hi; b++) {
        const int vlen = accepted_cnt[b] + 1;
        const int sl = seq_lens[b];
        cu_accepted[b] = run;
        for (int j = 0; j < vlen; j++) {
            packed_tokens[run + j] = out_tokens[int64_t(b) * Sp1 + j];
            packed_pos[run + j] = sl + j;
        }
        run += vlen;
    }
    const int cap = B * Sp1;
    for (int i = int(tid); i < cap; i += int(nthreads))
        if (i >= total) { packed_tokens[i] = -1; packed_pos[i] = -1; }
}

__global__ void spec_update_kv_meta(const int* seq_lens, const int* accepted_cnt,
                                    int* new_seq_lens, int B) {
    const int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < B) new_seq_lens[gid] = seq_lens[gid] + accepted_cnt[gid] + 1;
}

}  // namespace tms

