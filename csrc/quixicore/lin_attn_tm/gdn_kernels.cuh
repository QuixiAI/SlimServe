/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's Gated DeltaNet linear attention
 * (gdn_linear_attention.metal). Per-token SERIAL recurrence (no chunking) with
 * a scalar decay gate g_t and the DeltaNet write-key correction:
 *
 *   S      *= g_t                         # scalar decay, broadcast over Dk
 *   kv_mem  = <k_t, S>                     # warp-sum over Dk
 *   delta   = (v_t[dv] - kv_mem) * beta_t
 *   S      += k_t * delta                  # rank-1 write-key update
 *   y_t[dv] = <q_t, S>                     # warp-sum over Dk
 *
 * Layout: state_pool [max_seqs, Hv, Dv, Dk]; q,k packed (T, Hk, Dk); v,y
 * (T, Hv, Dv); g,beta (T, Hv). GQA hk = hv/(Hv/Hk). varlen via cu_seqlens;
 * paged recurrent state via slot_mapping. One warp per (req, hv, dv); lane
 * owns Dk/32 state elements (Dk up to 256). Structural twin of
 * lin_attn_causal's column-owner serial scan (tm_linattn_kernels.cuh).
 */
#pragma once
#include "../serving/tm_warp.cuh"        // warp_sum_f
#include <cuda_fp16.h>
#include <cuda_bf16.h>

namespace tmgdn {

template <typename T> __device__ __forceinline__ float gf(T v);
template <> __device__ __forceinline__ float gf<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float gf<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float gf<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T fg(float v);
template <> __device__ __forceinline__ float          fg<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         fg<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  fg<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// grid (Dv, 1, num_requests*Hv); block 32. State loaded from / written back to
// state_pool[slot] (paged); zero-init the pool for a fresh sequence.
template <typename T>
__global__ void gdn_linear_attention(const T* __restrict__ q, const T* __restrict__ k,
        const T* __restrict__ v, const T* __restrict__ g, const T* __restrict__ beta,
        T* __restrict__ state_pool, const int* __restrict__ cu_seqlens,
        const int* __restrict__ slot_mapping, T* __restrict__ y,
        int num_requests, int Hk, int Hv, int Dk, int Dv) {
    const int req = blockIdx.z / Hv, hv = blockIdx.z % Hv;
    const int dv = blockIdx.x, lane = threadIdx.x;
    if (req >= num_requests || dv >= Dv) return;
    const int hk = hv / (Hv / Hk);
    const int seq_start = cu_seqlens[req], seq_end = cu_seqlens[req + 1];
    const int seq_len = seq_end - seq_start;
    const int slot = slot_mapping[req];
    T* sp = state_pool + (((long)slot * Hv + hv) * Dv + dv) * Dk;

    const int npt = Dk / 32;                       // state elems per lane (<=8)
    float state[8];
    #pragma unroll
    for (int i = 0; i < npt; ++i) {
        const int s = npt * lane + i;
        state[i] = (s < Dk) ? gf(sp[s]) : 0.0f;
    }
    const T* q_ = q + (long)seq_start * Hk * Dk + hk * Dk;
    const T* k_ = k + (long)seq_start * Hk * Dk + hk * Dk;
    const T* v_ = v + (long)seq_start * Hv * Dv + hv * Dv;
    const T* g_ = g + (long)seq_start * Hv;
    const T* b_ = beta + (long)seq_start * Hv;
    T* y_ = y + (long)seq_start * Hv * Dv + hv * Dv;

    for (int t = 0; t < seq_len; ++t) {
        const float gv = gf(g_[hv]);
        float kv = 0.0f;
        #pragma unroll
        for (int i = 0; i < npt; ++i) {
            state[i] *= gv;
            kv += state[i] * gf(k_[npt * lane + i]);
        }
        kv = tms::warp_sum_f(kv);
        const float delta = (gf(v_[dv]) - kv) * gf(b_[hv]);
        float out = 0.0f;
        #pragma unroll
        for (int i = 0; i < npt; ++i) {
            state[i] += gf(k_[npt * lane + i]) * delta;
            out += state[i] * gf(q_[npt * lane + i]);
        }
        out = tms::warp_sum_f(out);
        if (lane == 0) y_[dv] = fg<T>(out);
        q_ += Hk * Dk; k_ += Hk * Dk; v_ += Hv * Dv; y_ += Hv * Dv; g_ += Hv; b_ += Hv;
    }
    #pragma unroll
    for (int i = 0; i < npt; ++i) {
        const int s = npt * lane + i;
        if (s < Dk) sp[s] = fg<T>(state[i]);
    }
}

} // namespace tmgdn
