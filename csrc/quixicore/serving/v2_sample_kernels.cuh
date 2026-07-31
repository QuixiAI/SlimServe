#pragma once
// Native CUDA ports of the V2 model-runner sampler / spec-decode Triton
// kernels (vllm/v1/worker/gpu/sample/{gumbel,logprob,penalties,
// prompt_logprob}.py and spec_decode/rejection_sampler_utils.py).
//
// Bit-exactness contract: outputs match the Triton kernels bitwise. That
// pins three things:
//   * RNG: Triton's tl.rand/tl.randint are Philox4x32-10 with Triton's key
//     schedule and its int32->[0,1) mapping (|x|-via-complement * 2^-31ish).
//   * Scalar math lowering: tl.exp == mul by log2e + ex2.approx.f32 (no ftz),
//     `/` == div.full.f32, tl.log/log1p == libdevice logf/log1pf. Plain CUDA
//     expf()/`/` do NOT match; use the tt_* helpers below.
//   * Reduction trees: fp32 sums replicate Triton's fold (per-thread
//     sizePerThread=4 x 2 reps, warp shfl.bfly 16..1, cross-warp pairwise).
//     Max/argmax(tie->lowest index) are order-independent and just need the
//     same tie rule.
// fma contraction: Triton emits separate mul/add for a*b+-c; __fmul_rn /
// __fsub_rn guard the spots nvcc would otherwise contract.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace tmv2s {

#define V2S_NEG_INF (-CUDART_INF_F)
// Smallest positive tl.rand output (matches _TL_RAND_MIN / triton's scale).
#define V2S_TL_RAND_SCALE 4.6566127342e-10f
#define V2S_TL_RAND64_SCALE 5.421010862427522170037e-20
#define V2S_F64_TINY 2.2250738585072014e-308

// ---------------------------------------------------------------------------
// Triton-lowering-exact scalar ops
// ---------------------------------------------------------------------------

__device__ __forceinline__ float tt_exp(float x) {
    float y = x * 1.4426950408889634f;  // 0f3FB8AA3B, same constant as triton
    asm("ex2.approx.f32 %0, %0;" : "+f"(y));
    return y;
}

__device__ __forceinline__ float tt_div(float a, float b) {
    float c;
    asm("div.full.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

template <typename T>
__device__ __forceinline__ float to_f32(T v) { return float(v); }
template <>
__device__ __forceinline__ float to_f32<__half>(__half v) { return __half2float(v); }
template <>
__device__ __forceinline__ float to_f32<__nv_bfloat16>(__nv_bfloat16 v) {
    return __bfloat162float(v);
}

// ---------------------------------------------------------------------------
// Philox4x32-10, bit-exact with triton.language.random
// ---------------------------------------------------------------------------

struct philox_out { uint32_t c0, c1, c2, c3; };

__device__ __forceinline__ philox_out tt_philox(uint32_t c0, uint32_t c1,
                                                uint32_t c2, uint32_t c3,
                                                uint32_t k0, uint32_t k1) {
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        const uint32_t p0 = c0, p2 = c2;
        c0 = __umulhi(0xCD9E8D57u, p2) ^ c1 ^ k0;
        c2 = __umulhi(0xD2511F53u, p0) ^ c3 ^ k1;
        c1 = 0xCD9E8D57u * p2;
        c3 = 0xD2511F53u * p0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return {c0, c1, c2, c3};
}

// tl.randint(seed, offset). Pass seed/offset with the same extension triton
// applies: int64 seed as-is; a uint32 seed (chained gumbel seed) zero-extended;
// int32 offsets zero-filled in the high counter word.
__device__ __forceinline__ philox_out tt_randint4x(uint64_t seed, uint64_t off) {
    return tt_philox(uint32_t(off), uint32_t(off >> 32), 0u, 0u,
                     uint32_t(seed), uint32_t(seed >> 32));
}

__device__ __forceinline__ uint32_t tt_randint(uint64_t seed, uint64_t off) {
    return tt_randint4x(seed, off).c0;
}

__device__ __forceinline__ float tt_uniform(uint32_t r) {
    int32_t x = int32_t(r);
    x = (x < 0) ? (-x - 1) : x;
    return float(x) * V2S_TL_RAND_SCALE;
}

// tl_rand32(seed, off, includes_zero=False)
__device__ __forceinline__ float tt_rand_nz(uint64_t seed, uint64_t off) {
    return fmaxf(tt_uniform(tt_randint(seed, off)), V2S_TL_RAND_SCALE);
}

// tl_rand64(seed, off, includes_zero=False)
__device__ __forceinline__ double tt_rand64_nz(uint64_t seed, uint64_t off) {
    const philox_out p = tt_randint4x(seed, off);
    const uint64_t r = (uint64_t(p.c1) << 32) | uint64_t(p.c0);
    const double u = double(r) * V2S_TL_RAND64_SCALE;
    return fmax(u, V2S_F64_TINY);
}

// fp32 gumbel noise; matches gumbel.py's -log(-log1p(-u)) tail trick.
__device__ __forceinline__ float tt_gumbel32(uint64_t seed, uint64_t off) {
    const float u = tt_rand_nz(seed, off);
    return -logf(-log1pf(-u));
}

__device__ __forceinline__ double tt_gumbel64(uint64_t seed, uint64_t off) {
    const double u = tt_rand64_nz(seed, off);
    return -log(-log(u));
}

// ---------------------------------------------------------------------------
// Reduction helpers (triton shfl.bfly trees)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_bfly_sum_f32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

__device__ __forceinline__ float warp_bfly_max_f32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

__device__ __forceinline__ int warp_bfly_sum_i32(int v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

// Triton max(..., return_indices=True) tie rule: equal values keep the
// lowest index. Order-independent, so any tree works.
template <typename VT>
__device__ __forceinline__ void argmax_combine(VT& v, int& i, VT ov, int oi) {
    if (!(v > ov || (v == ov && i < oi))) { v = ov; i = oi; }
}

template <typename VT>
__device__ __forceinline__ void warp_bfly_argmax(VT& v, int& i) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const VT ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        argmax_combine(v, i, ov, oi);
    }
}

// ---------------------------------------------------------------------------
// gumbel.py: _temperature_kernel
// ---------------------------------------------------------------------------

__global__ void v2_temperature_k(float* __restrict__ logits, int64_t logits_stride,
                                 const int32_t* __restrict__ expanded_idx_mapping,
                                 const float* __restrict__ temperature, int V) {
    const int64_t token_idx = blockIdx.x;
    const int64_t req = expanded_idx_mapping[token_idx];
    const float temp = temperature[req];
    if (temp == 0.0f || temp == 1.0f) return;
    float* row = logits + token_idx * logits_stride;
    for (int i = blockIdx.y * 8192 + threadIdx.x,
             e = min(int(blockIdx.y + 1) * 8192, V);
         i < e; i += blockDim.x)
        row[i] = tt_div(row[i], temp);
}

// ---------------------------------------------------------------------------
// gumbel.py: _gumbel_sample_kernel (also gumbel_block_argmax)
// One warp-group block per (token, 1024-wide vocab block).
// ---------------------------------------------------------------------------

template <typename T, typename VT, bool APPLY_TEMP, bool HAS_PROCESSED>
__global__ void v2_gumbel_sample_k(
        int64_t* __restrict__ local_argmax, int64_t la_stride,
        VT* __restrict__ local_max, int64_t lm_stride,
        float* __restrict__ processed_logits, int64_t pl_stride,
        const int64_t* __restrict__ pl_col,  // null / [1] / per-token
        int per_token_col,
        const T* __restrict__ logits, int64_t logits_stride,
        const int32_t* __restrict__ expanded_idx_mapping,
        const int64_t* __restrict__ seeds,
        const int64_t* __restrict__ pos_ptr,
        const float* __restrict__ temp_ptr, int V) {
    const int64_t token_idx = blockIdx.x;
    const int block_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int64_t req_state_idx = expanded_idx_mapping[token_idx];
    const bool is_valid_req = req_state_idx >= 0;
    const float temp = is_valid_req ? temp_ptr[req_state_idx] : 0.0f;

    int64_t col = 0;
    if (HAS_PROCESSED && pl_col != nullptr)
        col = per_token_col ? pl_col[token_idx] : pl_col[0];

    uint32_t gumbel_seed = 0;
    if (temp != 0.0f) {
        const int64_t seed = is_valid_req ? seeds[req_state_idx] : 0;
        const int64_t pos = pos_ptr[token_idx];
        gumbel_seed = tt_randint(uint64_t(seed), uint64_t(pos));
    }

    VT best = VT(V2S_NEG_INF);
    int best_g = 0x7fffffff;
    const T* row = logits + token_idx * logits_stride;
    for (int j = tid; j < 1024; j += blockDim.x) {
        const int g = block_idx * 1024 + j;
        const bool in_range = g < V;
        float v = in_range ? to_f32(row[g]) : V2S_NEG_INF;
        if (APPLY_TEMP && temp != 0.0f) v = tt_div(v, temp);
        if (HAS_PROCESSED && in_range && is_valid_req)
            processed_logits[req_state_idx * pl_stride + col * V + g] = v;
        VT vv = VT(v);
        if (temp != 0.0f) {
            VT noise;
            if constexpr (sizeof(VT) == 8)
                noise = VT(tt_gumbel64(uint64_t(gumbel_seed), uint64_t(uint32_t(g))));
            else
                noise = VT(tt_gumbel32(uint64_t(gumbel_seed), uint64_t(uint32_t(g))));
            vv = in_range ? vv + noise : VT(V2S_NEG_INF);
        }
        argmax_combine(best, best_g, vv, g);
    }
    warp_bfly_argmax(best, best_g);

    __shared__ VT s_v[32];
    __shared__ int s_i[32];
    const int lane = tid & 31, warp = tid >> 5, nwarps = blockDim.x >> 5;
    if (lane == 0) { s_v[warp] = best; s_i[warp] = best_g; }
    __syncthreads();
    if (tid == 0) {
        for (int w = 1; w < nwarps; ++w) argmax_combine(best, best_g, s_v[w], s_i[w]);
        local_argmax[token_idx * la_stride + block_idx] = int64_t(best_g);
        local_max[token_idx * lm_stride + block_idx] = best;
    }
}

// ---------------------------------------------------------------------------
// logprob.py: _topk_log_softmax_kernel
// Must run with 128 threads: the fp32 sum replicates triton's num_warps=4
// tree (per-thread [4t..4t+3] + [512+4t..512+4t+3] fold, bfly 16..1,
// cross-warp (p0+p2)+(p1+p3)).
// ---------------------------------------------------------------------------

__device__ __forceinline__ float block128_triton_sum(float thread_acc) {
    __shared__ float s_part[4];
    float v = warp_bfly_sum_f32(thread_acc);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_part[warp] = v;
    __syncthreads();
    float total;
    if (threadIdx.x == 0) {
        total = (s_part[0] + s_part[2]) + (s_part[1] + s_part[3]);
        s_part[0] = total;
    }
    __syncthreads();
    total = s_part[0];
    __syncthreads();
    return total;
}

__device__ __forceinline__ float block128_max(float thread_acc) {
    __shared__ float s_part[4];
    float v = warp_bfly_max_f32(thread_acc);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_part[warp] = v;
    __syncthreads();
    float total;
    if (threadIdx.x == 0) {
        total = fmaxf(fmaxf(s_part[0], s_part[2]), fmaxf(s_part[1], s_part[3]));
        s_part[0] = total;
    }
    __syncthreads();
    total = s_part[0];
    __syncthreads();
    return total;
}

// Element -> thread mapping of the vectorized specialization Triton compiles
// for a 1024-wide chunk with 128 threads: fp32 loads are v4 (two reps of 4
// contiguous per thread), 2-byte dtypes load 8 contiguous per thread in one
// rep. The in-thread fold order must follow the register order.
template <typename T>
__device__ __forceinline__ int v2_topk_elem(int base, int t, int i) {
    if (sizeof(T) == 4) return base + (i >> 2) * 512 + t * 4 + (i & 3);
    return base + t * 8 + i;
}

template <typename T>
__global__ void v2_topk_log_softmax_k(
        float* __restrict__ out,
        const T* __restrict__ logits, int64_t logits_stride,
        const int64_t* __restrict__ topk_ids,
        int topk, int V) {
    const int64_t req_idx = blockIdx.x;
    const T* row = logits + req_idx * logits_stride;
    const int t = threadIdx.x;

    float max_val = V2S_NEG_INF;
    for (int base = 0; base < V; base += 1024) {
        float m = V2S_NEG_INF;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int j = v2_topk_elem<T>(base, t, i);
            m = fmaxf(m, j < V ? to_f32(row[j]) : V2S_NEG_INF);
        }
        max_val = fmaxf(max_val, block128_max(m));
    }

    float se = 0.0f;
    for (int base = 0; base < V; base += 1024) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int j = v2_topk_elem<T>(base, t, i);
            // triton: load other=0.0, exp, then zero masked lanes/groups
            const float x = j < V ? to_f32(row[j]) : 0.0f;
            float e = tt_exp(x - max_val);
            acc += (j < V) ? e : 0.0f;
        }
        se += block128_triton_sum(acc);
    }
    const float lse = logf(se);

    for (int k = t; k < topk; k += blockDim.x) {
        const int64_t tok = topk_ids[req_idx * topk + k];
        const float o = (to_f32(row[tok]) - max_val) - lse;
        out[req_idx * topk + k] = o;
    }
}

// ---------------------------------------------------------------------------
// logprob.py: _ranks_kernel (integer count, order-free)
// ---------------------------------------------------------------------------

template <typename T>
__global__ void v2_ranks_k(int64_t* __restrict__ out,
                           const T* __restrict__ logits, int64_t logits_stride,
                           const int64_t* __restrict__ token_ids, int V) {
    const int64_t req_idx = blockIdx.x;
    const T* row = logits + req_idx * logits_stride;
    const float x = to_f32(row[token_ids[req_idx]]);
    // triton pads each 8192-block with -inf and compares unmasked, so the
    // padding is counted whenever x == -inf; replicate that.
    const int padded = (V + 8191) / 8192 * 8192;
    int n = 0;
    for (int j = threadIdx.x; j < padded; j += blockDim.x) {
        const float v = j < V ? to_f32(row[j]) : V2S_NEG_INF;
        n += (v >= x) ? 1 : 0;
    }
    n = warp_bfly_sum_i32(n);
    __shared__ int s_part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_part[warp] = n;
    __syncthreads();
    if (threadIdx.x == 0) {
        const int nwarps = blockDim.x >> 5;
        for (int w = 1; w < nwarps; ++w) n += s_part[w];
        out[req_idx] = int64_t(n);
    }
}

// ---------------------------------------------------------------------------
// logprob.py: _fill_logprob_token_ids_kernel
// ---------------------------------------------------------------------------

__global__ void v2_fill_logprob_token_ids_k(
        int64_t* __restrict__ out_token_ids, int64_t out_tid_stride,
        bool* __restrict__ out_valid_mask, int64_t out_mask_stride,
        const int64_t* __restrict__ sampled_token_ids,
        const int32_t* __restrict__ topk_indices, int64_t topk_stride,
        const int32_t* __restrict__ expanded_idx_mapping,
        const int32_t* __restrict__ num_per_req_token_ids,
        const int32_t* __restrict__ per_req_token_ids, int64_t per_req_stride,
        int num_topk) {
    const int64_t batch_idx = blockIdx.x;
    if (threadIdx.x == 0) {
        out_token_ids[batch_idx * out_tid_stride] = sampled_token_ids[batch_idx];
        out_valid_mask[batch_idx * out_mask_stride] = true;
    }
    const int64_t req_state_idx = expanded_idx_mapping[batch_idx];
    const int num_custom = num_per_req_token_ids[req_state_idx];

    const int32_t* src;
    int limit;
    if (num_custom > 0) {
        src = per_req_token_ids + req_state_idx * per_req_stride;
        limit = num_custom;
    } else {
        src = topk_indices + batch_idx * topk_stride;
        limit = num_topk;
    }
    int64_t* tid_base = out_token_ids + batch_idx * out_tid_stride + 1;
    bool* mask_base = out_valid_mask + batch_idx * out_mask_stride + 1;
    for (int c = threadIdx.x; c < limit; c += blockDim.x) {
        tid_base[c] = int64_t(src[c]);
        mask_base[c] = true;
    }
}

// ---------------------------------------------------------------------------
// penalties.py: _penalties_kernel (elementwise). Triton contracts the
// `logits -= penalty * count` pairs into fma.rn with a negated multiplier;
// mirror that exactly.
// ---------------------------------------------------------------------------

__global__ void v2_penalties_k(
        float* __restrict__ logits, int64_t logits_stride,
        const int32_t* __restrict__ expanded_idx_mapping,
        const int32_t* __restrict__ token_ids,
        const int32_t* __restrict__ expanded_local_pos,
        const float* __restrict__ repetition_penalty,
        const float* __restrict__ frequency_penalty,
        const float* __restrict__ presence_penalty,
        const int32_t* __restrict__ prompt_bin_mask, int64_t pbm_stride,
        const int32_t* __restrict__ output_bin_counts, int64_t obc_stride,
        int V) {
    const int64_t token_idx = blockIdx.x;
    const int64_t req_state_idx = expanded_idx_mapping[token_idx];
    const float rep = repetition_penalty[req_state_idx];
    const float freq = frequency_penalty[req_state_idx];
    const float pres = presence_penalty[req_state_idx];
    const bool use_rep = rep != 1.0f;
    if (!use_rep && freq == 0.0f && pres == 0.0f) return;

    const int pos = expanded_local_pos[token_idx];
    const int64_t start_idx = token_idx - pos;
    float* row = logits + token_idx * logits_stride;
    const int32_t* counts_row = output_bin_counts + req_state_idx * obc_stride;
    const int32_t* pbm_row = prompt_bin_mask + req_state_idx * pbm_stride;

    for (int j = blockIdx.y * 8192 + threadIdx.x,
             e = min(int(blockIdx.y + 1) * 8192, V);
         j < e; j += blockDim.x) {
        float v = row[j];
        int count = counts_row[j];
        for (int prev = 0; prev < pos; ++prev)
            count += (j == token_ids[start_idx + prev + 1]) ? 1 : 0;
        const bool out_mask = count > 0;
        if (use_rep) {
            const bool prompt_bit = (pbm_row[j >> 5] >> (j & 31)) & 1;
            const float scale = (prompt_bit || out_mask) ? rep : 1.0f;
            v *= (v > 0.0f) ? tt_div(1.0f, scale) : scale;
        }
        v = __fmaf_rn(-freq, float(count), v);
        v = __fmaf_rn(-pres, out_mask ? 1.0f : 0.0f, v);
        row[j] = v;
    }
}

// ---------------------------------------------------------------------------
// penalties.py: _bincount_kernel (integer atomics, order-free)
// ---------------------------------------------------------------------------

__global__ void v2_bincount_k(
        const int32_t* __restrict__ expanded_idx_mapping,
        const int32_t* __restrict__ all_token_ids, int64_t ati_stride,
        const int32_t* __restrict__ prompt_len,
        const int32_t* __restrict__ prefill_len,
        int32_t* __restrict__ prompt_bin_mask, int64_t pbm_stride,
        int32_t* __restrict__ output_bin_counts, int64_t obc_stride) {
    const int token_idx = blockIdx.x;
    const int block_idx = blockIdx.y;
    const int64_t req_state_idx = expanded_idx_mapping[token_idx];
    const int pf_len = prefill_len[req_state_idx];
    if (block_idx * 1024 >= pf_len) return;
    const int p_len = prompt_len[req_state_idx];
    const int32_t* tok_row = all_token_ids + req_state_idx * ati_stride;

    if (block_idx * 1024 < p_len) {
        for (int b = block_idx * 1024 + threadIdx.x,
                 e = min((block_idx + 1) * 1024, p_len);
             b < e; b += blockDim.x) {
            const int tok = tok_row[b];
            atomicOr(&prompt_bin_mask[req_state_idx * pbm_stride + (tok >> 5)],
                     1 << (tok & 31));
        }
    }
    if ((block_idx + 1) * 1024 >= p_len) {
        for (int b = block_idx * 1024 + threadIdx.x,
                 e = min((block_idx + 1) * 1024, pf_len);
             b < e; b += blockDim.x) {
            if (b < p_len) continue;
            atomicAdd(&output_bin_counts[req_state_idx * obc_stride + tok_row[b]], 1);
        }
    }
}

// ---------------------------------------------------------------------------
// prompt_logprob.py: _prompt_logprobs_token_ids_kernel
// ---------------------------------------------------------------------------

__global__ void v2_prompt_logprobs_token_ids_k(
        int64_t* __restrict__ out,
        const int32_t* __restrict__ query_start_loc,
        const int32_t* __restrict__ idx_mapping,
        const int32_t* __restrict__ num_computed_tokens,
        const int32_t* __restrict__ all_token_ids, int64_t ati_stride) {
    const int batch_idx = blockIdx.x;
    const int64_t req_state_idx = idx_mapping[batch_idx];
    const int query_start = query_start_loc[batch_idx];
    const int query_len = query_start_loc[batch_idx + 1] - query_start;
    const int nct = num_computed_tokens[req_state_idx];
    const int32_t* row = all_token_ids + req_state_idx * ati_stride;
    for (int b = threadIdx.x; b < query_len; b += blockDim.x)
        out[query_start + b] = int64_t(row[nct + 1 + b]);
}

// ---------------------------------------------------------------------------
// rejection_sampler_utils.py helpers. One warp per request; the per-block
// stats tensors are <= 32 wide after padding (gate native path on that).
// Lanes >= nb contribute exact neutrals (+0.0 / -inf), which leaves the
// triton bfly tree's bits unchanged.
// ---------------------------------------------------------------------------

// Padded stats width, the constexpr triton specializes on.
__device__ __forceinline__ int v2_padded_width(int nb) {
    int p = 1;
    while (p < nb) p <<= 1;
    return p;
}

__device__ __forceinline__ float v2_global_lse(
        const float* __restrict__ local_max, int64_t lm_stride,
        const float* __restrict__ local_sumexp, int64_t ls_stride,
        int64_t logit_idx, int nb, int lane) {
    // Width-P tensor over 32 lanes: lane l holds element l % P (replicated),
    // reduced with bfly offsets P/2..1. ptxas fuses the first sum step into
    // fma(exp, s, shuffled); later steps are plain adds. Replicate both.
    const int P = v2_padded_width(nb);
    const int el = lane & (P - 1);
    const float m = el < nb ? local_max[logit_idx * lm_stride + el]
                            : V2S_NEG_INF;
    const float s = el < nb ? local_sumexp[logit_idx * ls_stride + el] : 0.0f;
    float gmax = m;
    for (int off = P >> 1; off > 0; off >>= 1)
        gmax = fmaxf(gmax, __shfl_xor_sync(0xffffffffu, gmax, off));
    const float e = tt_exp(m - gmax);
    float acc = s * e;
    bool first = true;
    for (int off = P >> 1; off > 0; off >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, acc, off);
        acc = first ? __fmaf_rn(e, s, other) : acc + other;
        first = false;
    }
    return gmax + logf(acc);
}

__device__ __forceinline__ int64_t v2_global_target_argmax(
        const float* __restrict__ local_max, int64_t lm_stride,
        const int64_t* __restrict__ local_argmax, int64_t la_stride,
        int64_t logit_idx, int nb, int lane) {
    const int P = v2_padded_width(nb);
    const int el = lane & (P - 1);
    float v = el < nb ? local_max[logit_idx * lm_stride + el] : V2S_NEG_INF;
    int i = el;
    for (int off = P >> 1; off > 0; off >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, off);
        const int oi = __shfl_xor_sync(0xffffffffu, i, off);
        argmax_combine(v, i, ov, oi);
    }
    return local_argmax[logit_idx * la_stride + i];
}

// DS: draft_sampled element type. Serving passes input_ids gathers (int32);
// tests and some callers pass int64. Triton widens on load either way.
template <bool HAS_DRAFT, typename DS>
__global__ void v2_rejection_k(
        int64_t* __restrict__ sampled, int64_t sampled_stride,
        int32_t* __restrict__ num_sampled,
        float* __restrict__ target_rejected_lse,
        float* __restrict__ draft_rejected_lse,
        const float* __restrict__ target_logits, int64_t target_stride,
        const int64_t* __restrict__ t_local_argmax, int64_t t_la_stride,
        const float* __restrict__ t_local_max, int64_t t_lm_stride,
        const float* __restrict__ t_local_sumexp, int64_t t_ls_stride,
        const DS* __restrict__ draft_sampled,
        const float* __restrict__ draft_logits, int64_t d_stride0,
        int64_t d_stride1,
        const float* __restrict__ d_local_max, int64_t d_lm_stride,
        const float* __restrict__ d_local_sumexp, int64_t d_ls_stride,
        const int32_t* __restrict__ cu_num_logits,
        const int32_t* __restrict__ idx_mapping,
        const float* __restrict__ temp_ptr,
        const int64_t* __restrict__ seed_ptr,
        const int64_t* __restrict__ pos_ptr,
        int vocab_num_blocks) {
    const int req_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int64_t req_state_idx = idx_mapping[req_idx];
    const int64_t start_idx = cu_num_logits[req_idx];
    const int num_draft_tokens = cu_num_logits[req_idx + 1] - int(start_idx) - 1;
    const int64_t seed = seed_ptr[req_state_idx];
    const float temp = temp_ptr[req_state_idx];
    const bool is_greedy = temp == 0.0f;

    int64_t accepted_length = 0;
    float target_lse = 0.0f, draft_lse = 0.0f;
    bool accepted = true;
    for (int i = 0; i < num_draft_tokens; ++i) {
        const int64_t logit_idx = start_idx + i;
        int64_t d_tok = int64_t(draft_sampled[logit_idx + 1]);
        const int64_t pos = pos_ptr[logit_idx];
        const float u = tt_rand_nz(uint64_t(seed), uint64_t(pos));
        if (accepted) {
            if (is_greedy) {
                const int64_t target_argmax = v2_global_target_argmax(
                    t_local_max, t_lm_stride, t_local_argmax, t_la_stride,
                    logit_idx, vocab_num_blocks, lane);
                accepted = accepted && (target_argmax == d_tok);
                if (lane == 0)
                    sampled[req_idx * sampled_stride + i] =
                        accepted ? d_tok : target_argmax;
            } else {
                const bool is_valid_draft = d_tok >= 0;
                d_tok = d_tok > 0 ? d_tok : 0;
                target_lse = v2_global_lse(t_local_max, t_lm_stride,
                                           t_local_sumexp, t_ls_stride,
                                           logit_idx, vocab_num_blocks, lane);
                const float target_logprob =
                    target_logits[logit_idx * target_stride + d_tok] - target_lse;
                float draft_logprob;
                if (HAS_DRAFT) {
                    draft_lse = v2_global_lse(d_local_max, d_lm_stride,
                                              d_local_sumexp, d_ls_stride,
                                              logit_idx, vocab_num_blocks, lane);
                    draft_logprob =
                        draft_logits[req_state_idx * d_stride0 + i * d_stride1 +
                                     d_tok] -
                        draft_lse;
                } else {
                    draft_logprob = 0.0f;
                    draft_lse = 0.0f;
                }
                accepted = accepted && (target_logprob > logf(u) + draft_logprob);
                accepted = accepted && is_valid_draft;
                if (lane == 0) sampled[req_idx * sampled_stride + i] = d_tok;
            }
            accepted_length += accepted ? 1 : 0;
        }
    }
    if (lane == 0) {
        num_sampled[req_idx] = int32_t(accepted_length);
        target_rejected_lse[req_idx] = target_lse;
        draft_rejected_lse[req_idx] = draft_lse;
    }
}

// ---------------------------------------------------------------------------
// rejection_sampler_utils.py: _resample_kernel (no block verification).
// One warp per (req, 1024-wide vocab block); argmax is order-free.
// ---------------------------------------------------------------------------

template <bool HAS_DRAFT, typename VT, typename DS>
__global__ void v2_resample_k(
        int64_t* __restrict__ rl_argmax, int64_t rla_stride,
        VT* __restrict__ rl_max, int64_t rlm_stride,
        const float* __restrict__ target_logits, int64_t target_stride,
        const float* __restrict__ target_rejected_lse,
        const float* __restrict__ draft_logits, int64_t d_stride0,
        int64_t d_stride1,
        const float* __restrict__ draft_rejected_lse,
        const int32_t* __restrict__ rejected_step,
        const int32_t* __restrict__ cu_num_logits,
        const int32_t* __restrict__ expanded_idx_mapping,
        const DS* __restrict__ draft_sampled,
        const float* __restrict__ temp_ptr,
        const int64_t* __restrict__ seed_ptr,
        const int64_t* __restrict__ pos_ptr,
        int V) {
    const int req_idx = blockIdx.x;
    const int resample_idx = rejected_step[req_idx];
    const int64_t start_idx = cu_num_logits[req_idx];
    const int64_t end_idx = cu_num_logits[req_idx + 1];
    const int64_t rtok = start_idx + resample_idx;
    const int64_t req_state_idx = expanded_idx_mapping[rtok];
    const float temp = temp_ptr[req_state_idx];
    const bool is_bonus = rtok == end_idx - 1;
    if (temp == 0.0f && !is_bonus) return;

    const int block_idx = blockIdx.y;
    const bool is_valid_req = req_state_idx >= 0;
    uint32_t gumbel_seed = 0;
    if (temp != 0.0f) {
        const int64_t seed = is_valid_req ? seed_ptr[req_state_idx] : 0;
        const int64_t pos = pos_ptr[rtok];
        gumbel_seed = tt_randint(uint64_t(seed), uint64_t(pos));
    }
    const float t_lse = target_rejected_lse[req_idx];
    const float d_lse = HAS_DRAFT ? draft_rejected_lse[req_idx] : 0.0f;
    const int64_t rejected_draft =
        (!is_bonus && !HAS_DRAFT) ? int64_t(draft_sampled[rtok + 1]) : 0;

    VT best = VT(V2S_NEG_INF);
    int best_g = 0x7fffffff;
    for (int j = threadIdx.x; j < 1024; j += 32) {
        const int g = block_idx * 1024 + j;
        const bool in_range = g < V;
        const float tl_v =
            in_range ? target_logits[rtok * target_stride + g] : V2S_NEG_INF;
        float residual;
        if (is_bonus) {
            residual = tl_v;
        } else if (HAS_DRAFT) {
            const float dl_v =
                in_range
                    ? draft_logits[req_state_idx * d_stride0 +
                                   int64_t(resample_idx) * d_stride1 + g]
                    : V2S_NEG_INF;
            const float tlp = tl_v - t_lse;
            const float dlp = dl_v - d_lse;
            const float ratio = tt_exp(dlp - tlp);
            residual = (ratio < 1.0f) ? tlp + log1pf(-ratio) : V2S_NEG_INF;
        } else {
            residual = (int64_t(g) != rejected_draft) ? tl_v : V2S_NEG_INF;
        }
        VT vv = VT(residual);
        if (temp != 0.0f) {
            VT noise;
            if constexpr (sizeof(VT) == 8)
                noise = VT(tt_gumbel64(uint64_t(gumbel_seed), uint64_t(uint32_t(g))));
            else
                noise = VT(tt_gumbel32(uint64_t(gumbel_seed), uint64_t(uint32_t(g))));
            vv = in_range ? vv + noise : VT(V2S_NEG_INF);
        }
        argmax_combine(best, best_g, vv, g);
    }
    warp_bfly_argmax(best, best_g);
    if (threadIdx.x == 0) {
        rl_argmax[int64_t(req_idx) * rla_stride + block_idx] = int64_t(best_g);
        rl_max[int64_t(req_idx) * rlm_stride + block_idx] = best;
    }
}

// ---------------------------------------------------------------------------
// structured_outputs.py: _apply_grammar_bitmask_kernel. Words beyond
// bitmask_stride read as 0 (masked-load default), i.e. those tokens get -inf
// like the Triton original.
// ---------------------------------------------------------------------------

template <typename T>
__global__ void v2_grammar_bitmask_k(
        T* __restrict__ logits, int64_t logits_stride,
        const int32_t* __restrict__ logits_indices,
        const int32_t* __restrict__ bitmask, int64_t bitmask_stride, int V) {
    const int bitmask_idx = blockIdx.x;
    const int64_t logits_idx = logits_indices[bitmask_idx];
    T* row = logits + logits_idx * logits_stride;
    const int32_t* bm = bitmask + int64_t(bitmask_idx) * bitmask_stride;
    // Only ever stores -inf; dtype-generic because serving logits arrive
    // bf16 here while the sampler's fp32 view is a separate call site.
    const T neg_inf = T(V2S_NEG_INF);
    for (int j = blockIdx.y * 8192 + threadIdx.x,
             e = min(int(blockIdx.y + 1) * 8192, V);
         j < e; j += blockDim.x) {
        const int w = j >> 5;
        const int32_t word = w < bitmask_stride ? bm[w] : 0;
        if (((word >> (j & 31)) & 1) == 0) row[j] = neg_inf;
    }
}

// ---------------------------------------------------------------------------
// min_p.py: _min_p_kernel (max is order-free; threshold uses libdevice logf)
// ---------------------------------------------------------------------------

__global__ void v2_min_p_k(float* __restrict__ logits, int64_t logits_stride,
                           const int32_t* __restrict__ expanded_idx_mapping,
                           const float* __restrict__ min_p_ptr, int V) {
    const int64_t token_idx = blockIdx.x;
    const float min_p = min_p_ptr[expanded_idx_mapping[token_idx]];
    if (min_p == 0.0f) return;
    float* row = logits + token_idx * logits_stride;
    float m = V2S_NEG_INF;
    for (int j = threadIdx.x; j < V; j += blockDim.x) m = fmaxf(m, row[j]);
    m = warp_bfly_max_f32(m);
    __shared__ float s_part[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    if (lane == 0) s_part[warp] = m;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < nwarps; ++w) m = fmaxf(m, s_part[w]);
        s_part[0] = m;
    }
    __syncthreads();
    const float threshold = s_part[0] + logf(min_p);
    for (int j = threadIdx.x; j < V; j += blockDim.x)
        if (row[j] < threshold) row[j] = V2S_NEG_INF;
}

// ---------------------------------------------------------------------------
// logit_bias.py: _bias_kernel. 1024 threads, one list slot per thread;
// __syncthreads() reproduces the tl.debug_barrier() ordering.
// ---------------------------------------------------------------------------

__global__ void v2_logit_bias_k(
        float* __restrict__ logits, int64_t logits_stride, int V,
        const int32_t* __restrict__ expanded_idx_mapping,
        const int32_t* __restrict__ num_allowed_token_ids,
        const int32_t* __restrict__ allowed_token_ids, int64_t allowed_stride,
        const int32_t* __restrict__ num_logit_bias,
        const int32_t* __restrict__ bias_token_ids, int64_t bias_ids_stride,
        const float* __restrict__ bias, int64_t bias_stride,
        const int64_t* __restrict__ pos_ptr,
        const int32_t* __restrict__ min_lens,
        const int32_t* __restrict__ num_stop_token_ids,
        const int32_t* __restrict__ stop_token_ids, int64_t stop_stride) {
    const int64_t token_idx = blockIdx.x;
    const int64_t req_state_idx = expanded_idx_mapping[token_idx];
    float* row = logits + token_idx * logits_stride;
    const int c = threadIdx.x;

    const int num_allowed = num_allowed_token_ids[req_state_idx];
    if (num_allowed > 0) {
        int32_t tok = 0;
        float saved = 0.0f;
        if (c < num_allowed) {
            tok = allowed_token_ids[req_state_idx * allowed_stride + c];
            saved = row[tok];
        }
        __syncthreads();
        for (int j = c; j < V; j += blockDim.x) row[j] = V2S_NEG_INF;
        __syncthreads();
        if (c < num_allowed) row[tok] = saved;
    }
    __syncthreads();

    const int num_bias = num_logit_bias[req_state_idx];
    if (num_bias > 0 && c < num_bias) {
        const int32_t tok = bias_token_ids[req_state_idx * bias_ids_stride + c];
        row[tok] += bias[req_state_idx * bias_stride + c];
    }
    __syncthreads();

    const int num_stop = num_stop_token_ids[req_state_idx];
    const int64_t pos = pos_ptr[token_idx];
    const int min_len = min_lens[req_state_idx];
    if (num_stop > 0 && pos + 1 < min_len && c < num_stop) {
        const int32_t tok = stop_token_ids[req_state_idx * stop_stride + c];
        row[tok] = V2S_NEG_INF;
    }
}

// ---------------------------------------------------------------------------
// bad_words.py: _bad_words_kernel (one thread per bad word)
// ---------------------------------------------------------------------------

__global__ void v2_bad_words_k(
        float* __restrict__ logits, int64_t logits_stride,
        const int32_t* __restrict__ expanded_idx_mapping,
        const int32_t* __restrict__ bad_word_token_ids, int64_t bw_tokens_stride,
        const int32_t* __restrict__ bad_word_offsets, int64_t bw_offsets_stride,
        const int32_t* __restrict__ num_bad_words_ptr,
        const int32_t* __restrict__ all_token_ids, int64_t ati_stride,
        const int32_t* __restrict__ prompt_len_ptr,
        const int32_t* __restrict__ total_len_ptr,
        const int32_t* __restrict__ input_ids,
        const int32_t* __restrict__ expanded_local_pos) {
    const int64_t token_idx = blockIdx.x;
    const int64_t req_state_idx = expanded_idx_mapping[token_idx];
    const int num_bad_words = num_bad_words_ptr[req_state_idx];

    const int pos = expanded_local_pos[token_idx];
    const int64_t cur_req_first_pos = token_idx - pos;
    const int prompt_len = prompt_len_ptr[req_state_idx];
    const int output_len = total_len_ptr[req_state_idx] - prompt_len;
    const int effective_len = output_len + pos;

    const int32_t* offs = bad_word_offsets + req_state_idx * bw_offsets_stride;
    const int32_t* toks = bad_word_token_ids + req_state_idx * bw_tokens_stride;
    const int32_t* output_base =
        all_token_ids + req_state_idx * ati_stride + prompt_len;

    for (int bw = threadIdx.x; bw < num_bad_words; bw += blockDim.x) {
        const int start = offs[bw];
        const int end = offs[bw + 1];
        const int prefix_len = end - start - 1;
        if (prefix_len > effective_len) continue;
        const int32_t last_token = toks[end - 1];
        bool match = true;
        for (int i = 0; i < prefix_len; ++i) {
            const int actual_pos = effective_len - prefix_len + i;
            int32_t actual;
            if (actual_pos >= output_len)
                actual = input_ids[cur_req_first_pos + (actual_pos - output_len)];
            else
                actual = output_base[actual_pos];
            match = match && (toks[start + i] == actual);
        }
        if (match) logits[token_idx * logits_stride + last_token] = V2S_NEG_INF;
    }
}

// ---------------------------------------------------------------------------
// rejection_sampler_utils.py: _compute_local_logits_stats_kernel.
// 128 threads over an 8192-wide block; the fp32 sumexp uses the same
// vectorized fold as v2_topk_log_softmax (per-thread 16 reps x 4 contiguous,
// bfly 16..1, cross-warp (p0+p2)+(p1+p3)), so route only for V % 16 == 0.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void v2_block_max_sumexp_8192(
        const float* __restrict__ row, int base, int V,
        float& out_max, float& out_sumexp) {
    const int t = threadIdx.x;
    float m = V2S_NEG_INF;
#pragma unroll
    for (int rep = 0; rep < 16; ++rep)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int j = base + rep * 512 + t * 4 + k;
            m = fmaxf(m, j < V ? row[j] : V2S_NEG_INF);
        }
    const float gmax = block128_max(m);
    float acc = 0.0f;
#pragma unroll
    for (int rep = 0; rep < 16; ++rep)
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int j = base + rep * 512 + t * 4 + k;
            const float x = j < V ? row[j] : V2S_NEG_INF;
            acc += tt_exp(x - gmax);
        }
    const float total = block128_triton_sum(acc);
    out_max = gmax;
    out_sumexp = gmax > V2S_NEG_INF ? total : 0.0f;
}

template <typename VT>
__device__ __forceinline__ void v2_block_argmax_8192(
        const float* __restrict__ row, int base, int V,
        VT& out_val, int& out_idx) {
    const int t = threadIdx.x;
    VT best = VT(V2S_NEG_INF);
    int best_j = 0x7fffffff;
    for (int j = base + t; j < base + 8192; j += blockDim.x) {
        const VT v = VT(j < V ? row[j] : V2S_NEG_INF);
        argmax_combine(best, best_j, v, j);
    }
    warp_bfly_argmax(best, best_j);
    __shared__ VT s_v[32];
    __shared__ int s_i[32];
    const int lane = t & 31, warp = t >> 5, nwarps = blockDim.x >> 5;
    if (lane == 0) { s_v[warp] = best; s_i[warp] = best_j; }
    __syncthreads();
    if (t == 0)
        for (int w = 1; w < nwarps; ++w) argmax_combine(best, best_j, s_v[w], s_i[w]);
    out_val = best;
    out_idx = best_j;
    __syncthreads();
}

template <bool HAS_DRAFT>
__global__ void v2_local_logits_stats_k(
        int64_t* __restrict__ t_local_argmax, int64_t t_la_stride,
        float* __restrict__ t_local_max, int64_t t_lm_stride,
        float* __restrict__ t_local_sumexp, int64_t t_ls_stride,
        float* __restrict__ d_local_max, int64_t d_lm_stride,
        float* __restrict__ d_local_sumexp, int64_t d_ls_stride,
        const float* __restrict__ target_logits, int64_t target_stride,
        const float* __restrict__ draft_logits, int64_t d_stride0,
        int64_t d_stride1,
        const int32_t* __restrict__ expanded_idx_mapping,
        const int32_t* __restrict__ expanded_local_pos,
        const float* __restrict__ temp_ptr, int V, int num_speculative_steps) {
    const int64_t logit_idx = blockIdx.x;
    const int draft_step_idx = expanded_local_pos[logit_idx];
    if (draft_step_idx >= num_speculative_steps) return;

    const int64_t req_state_idx = expanded_idx_mapping[logit_idx];
    const float temp = temp_ptr[req_state_idx];
    const int block_idx = blockIdx.y;
    const int base = block_idx * 8192;
    const float* t_row = target_logits + logit_idx * target_stride;

    if (temp == 0.0f) {
        float value;
        int idx;
        v2_block_argmax_8192(t_row, base, V, value, idx);
        if (threadIdx.x == 0) {
            t_local_argmax[logit_idx * t_la_stride + block_idx] = int64_t(idx);
            t_local_max[logit_idx * t_lm_stride + block_idx] = value;
        }
    } else {
        float m, se;
        v2_block_max_sumexp_8192(t_row, base, V, m, se);
        if (threadIdx.x == 0) {
            t_local_max[logit_idx * t_lm_stride + block_idx] = m;
            t_local_sumexp[logit_idx * t_ls_stride + block_idx] = se;
        }
        if (HAS_DRAFT) {
            __syncthreads();
            const float* d_row = draft_logits + req_state_idx * d_stride0 +
                                 int64_t(draft_step_idx) * d_stride1;
            v2_block_max_sumexp_8192(d_row, base, V, m, se);
            if (threadIdx.x == 0) {
                d_local_max[logit_idx * d_lm_stride + block_idx] = m;
                d_local_sumexp[logit_idx * d_ls_stride + block_idx] = se;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// rejection_sampler_utils.py: _insert_resampled_kernel. One warp per request;
// the num_sampled increment happens before the greedy early-out, like triton.
// ---------------------------------------------------------------------------

template <typename VT>
__global__ void v2_insert_resampled_k(
        int64_t* __restrict__ sampled, int64_t sampled_stride,
        int32_t* __restrict__ num_sampled_ptr,
        const int64_t* __restrict__ rl_argmax, int64_t rla_stride,
        const VT* __restrict__ rl_max, int64_t rlm_stride,
        int resample_num_blocks,
        const int32_t* __restrict__ cu_num_logits,
        const int32_t* __restrict__ expanded_idx_mapping,
        const float* __restrict__ temp_ptr) {
    const int req_idx = blockIdx.x;
    const int lane = threadIdx.x;
    const int num_sampled = num_sampled_ptr[req_idx];
    const int64_t start_idx = cu_num_logits[req_idx];
    const int64_t end_idx = cu_num_logits[req_idx + 1];
    const int64_t resample_token_idx = start_idx + num_sampled;
    const int64_t req_state_idx = expanded_idx_mapping[resample_token_idx];
    __syncwarp();
    if (lane == 0) num_sampled_ptr[req_idx] = num_sampled + 1;

    const float temp = temp_ptr[req_state_idx];
    const bool is_bonus = resample_token_idx == end_idx - 1;
    if (temp == 0.0f && !is_bonus) return;

    VT best = VT(V2S_NEG_INF);
    int best_i = 0x7fffffff;
    for (int b = lane; b < resample_num_blocks; b += 32) {
        const VT v = rl_max[int64_t(req_idx) * rlm_stride + b];
        argmax_combine(best, best_i, v, b);
    }
    warp_bfly_argmax(best, best_i);
    if (lane == 0)
        sampled[int64_t(req_idx) * sampled_stride + num_sampled] =
            rl_argmax[int64_t(req_idx) * rla_stride + best_i];
}

// ---------------------------------------------------------------------------
// rejection_sampler.py: _flatten_sampled_kernel
// ---------------------------------------------------------------------------

__global__ void v2_flatten_sampled_k(
        int64_t* __restrict__ flat_sampled,
        const int64_t* __restrict__ sampled, int64_t sampled_stride,
        const int32_t* __restrict__ num_sampled,
        const int32_t* __restrict__ cu_num_logits) {
    const int req_idx = blockIdx.x;
    const int64_t start_idx = cu_num_logits[req_idx];
    const int n = num_sampled[req_idx];
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        flat_sampled[start_idx + i] =
            sampled[int64_t(req_idx) * sampled_stride + i];
}

}  // namespace tmv2s
