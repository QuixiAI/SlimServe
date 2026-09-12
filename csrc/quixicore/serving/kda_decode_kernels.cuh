#pragma once
// KDA (Kimi Delta Attention) single-token decode recurrence, CUDA port of
// the vendored Triton `fused_recurrent_kda_packed_decode_kernel`
// (vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py).
//
// Per (token, head): S[v, k] (fp32, K x V per head) is decayed by exp(gate[k]),
// v' = beta * (v - S . k), S += v' k^T, out = S . q; q and k are L2-normalized
// and q scaled by K^-0.5; gate = lower_bound * sigmoid(a * (raw_g + bias))
// (or -a * softplus(raw_g + bias) without a lower bound), beta = sigmoid(raw).
//
// Motive (GLM-5.3 TP4 x DP2 profile, 2026-09-12): the Triton kernel moved
// the state at 0.56 TB/s effective (115 us per layer at 32 tokens x 16
// heads, 64 MB read + write). Layout here: one block of 8 warps per
// (token, head); each warp owns one state row at a time with a 128-bit
// load/store per lane (512 B per warp op, fully coalesced), 16 rows per
// warp; k, q and the gate live in registers per lane after one shared
// staging pass. Row dots are warp shuffles.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tms::kda {

constexpr int THREADS = 256;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float sigmoidf(float x) { return 1.0f / (1.0f + __expf(-x)); }

// K == V == 128, lane owns columns [4*lane, +4).
template <int K, int V>
__global__ void __launch_bounds__(THREADS) kda_decode_kernel(
    const __nv_bfloat16* __restrict__ mixed,   // [N, stride_mixed] q | k | v per head
    const __nv_bfloat16* __restrict__ raw_g,   // [N, stride_g] per head K
    const __nv_bfloat16* __restrict__ raw_beta,// [N, stride_beta] per head
    const float* __restrict__ A_log,           // [H]
    const float* __restrict__ dt_bias,         // [H*K]
    float* __restrict__ state,                 // [cache, H, V, K] (stride_state per cache slot)
    const int* __restrict__ state_indices,     // [N]
    __nv_bfloat16* __restrict__ out,           // [N, H, V]
    int H, int64_t stride_mixed, int64_t stride_g, int64_t stride_beta,
    int64_t stride_state, float scale, float lower_bound, int use_lower_bound) {
    static_assert(K == 128 && V == 128, "GLM-5.3 KDA geometry");
    const int nh = blockIdx.x;
    const int n = nh / H, h = nh - n * H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __nv_bfloat16* o_row = out + (int64_t(n) * H + h) * V;
    const int state_idx = state_indices[n];
    if (state_idx <= 0) {
        for (int i = tid; i < V; i += THREADS) o_row[i] = __float2bfloat16_rn(0.0f);
        return;
    }
    __shared__ float s_q[K], s_k[K], s_v[V], s_gexp[K];
    __shared__ float s_red[2];
    const __nv_bfloat16* row = mixed + int64_t(n) * stride_mixed;
    if (tid < K) {
        const float q = float(row[h * K + tid]);
        const float k = float(row[H * K + h * K + tid]);
        s_q[tid] = q; s_k[tid] = k;
        const float g = float(raw_g[int64_t(n) * stride_g + h * K + tid]) + dt_bias[h * K + tid];
        const float a = __expf(A_log[h]);
        float gate;
        if (use_lower_bound) {
            gate = lower_bound * sigmoidf(a * g);
        } else {
            const float sp = g > 20.0f ? g : __logf(1.0f + __expf(g));
            gate = -a * sp;
        }
        s_gexp[tid] = __expf(gate);
    } else if (tid < K + V) {
        s_v[tid - K] = float(row[2 * H * K + h * V + (tid - K)]);
    }
    __syncthreads();
    // L2 norms of q and k (warps 0 and 1 each reduce 128 values: 4 per lane).
    if (warp < 2) {
        const float* src = warp == 0 ? s_q : s_k;
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float x = src[4 * lane + i]; ss += x * x; }
        ss = warp_sum(ss);
        if (lane == 0) s_red[warp] = rsqrtf(ss + 1e-6f);   // 1/sqrt(sum + 1e-6)
    }
    __syncthreads();
    const float inv_q = s_red[0] * scale, inv_k = s_red[1];
    const float beta = sigmoidf(float(raw_beta[int64_t(n) * stride_beta + h]));
    float qn[4], kn[4], ge[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        qn[i] = s_q[4 * lane + i] * inv_q;
        kn[i] = s_k[4 * lane + i] * inv_k;
        ge[i] = s_gexp[4 * lane + i];
    }
    float* S = state + int64_t(state_idx) * stride_state + int64_t(h) * V * K;
    // 16 rows per warp, one row per iteration, lane-contiguous float4.
    for (int r = warp; r < V; r += THREADS / 32) {
        float4* p = reinterpret_cast<float4*>(S + int64_t(r) * K + 4 * lane);
        float4 s4 = *p;
        float s[4] = {s4.x * ge[0], s4.y * ge[1], s4.z * ge[2], s4.w * ge[3]};
        float dot_k = s[0] * kn[0] + s[1] * kn[1] + s[2] * kn[2] + s[3] * kn[3];
        dot_k = warp_sum(dot_k);
        const float vp = (s_v[r] - dot_k) * beta;
#pragma unroll
        for (int i = 0; i < 4; ++i) s[i] += vp * kn[i];
        float dot_q = s[0] * qn[0] + s[1] * qn[1] + s[2] * qn[2] + s[3] * qn[3];
        dot_q = warp_sum(dot_q);
        *p = make_float4(s[0], s[1], s[2], s[3]);
        if (lane == 0) o_row[r] = __float2bfloat16_rn(dot_q);
    }
}

}  // namespace tms::kda

// ---- Speculative rows: T tokens per (request, head) with the state in registers.
// MODE 0: forward with today's contract (output per row, state stored into
//         slot[t] after every row).
// MODE 1: forward, outputs only (deferred commit; nothing stored).
// MODE 2: commit - replay rows [0, new_accepted), no outputs, store the state
//         into slot[t] for t == new_accepted - 1 and for t == boundary_row.
// The state is read once per (request, head) at slot[prev_accepted - 1] and
// each state row lives in one warp's registers across the T rows, so the
// traffic is one state read plus the mode's stores (2026-09-12: the Triton
// forward is issue-bound at 88 us without its stores and 126 us with them
// at 32 requests x 4 rows).
namespace tms::kda {

constexpr int SPEC_MAX_T = 8;

template <int K, int V, int MODE, int TT>
__global__ void __launch_bounds__(THREADS) kda_spec_kernel(
        const __nv_bfloat16* __restrict__ q,       // [N, H, K] via stride_tok
        const __nv_bfloat16* __restrict__ k,
        const __nv_bfloat16* __restrict__ v,       // [N, H, V]
        const __nv_bfloat16* __restrict__ raw_g,   // [N, H, K] via stride_g
        const __nv_bfloat16* __restrict__ raw_beta,// [N, H] via stride_beta
        const float* __restrict__ A_log, const float* __restrict__ dt_bias,
        float* __restrict__ state, const int* __restrict__ cu_seqlens,
        const int* __restrict__ state_indices,     // [R, S] (stride_idx)
        const int* __restrict__ prev_accepted, const int* __restrict__ new_accepted,
        const int* __restrict__ boundary_row,
        __nv_bfloat16* __restrict__ out,           // [N, H, V] via stride_out
        int H, int64_t stride_tok, int64_t stride_g, int64_t stride_beta,
        int64_t stride_out, int64_t stride_state, int stride_idx,
        float scale, float lower_bound, int use_lower_bound) {
    static_assert(K == 128 && V == 128, "GLM-5.3 KDA geometry");
    const int rh = blockIdx.x;
    const int r = rh / H, h = rh - r * H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int bos = cu_seqlens[r], eos = cu_seqlens[r + 1];
    int len = eos - bos;
    if (len <= 0) return;
    if (MODE == 2) len = min(len, new_accepted[r]);
    if (len <= 0) return;
    const int init_col = max(prev_accepted[r] - 1, 0);
    const int state_idx = state_indices[r * stride_idx + init_col];
    if (state_idx <= 0) {
        if (MODE != 2) {
            for (int t = 0; t < len; ++t) {
                __nv_bfloat16* o = out + (int64_t(bos + t) * stride_out) + int64_t(h) * V;
                for (int i = tid; i < V; i += THREADS) o[i] = __float2bfloat16_rn(0.0f);
            }
        }
        return;
    }
    const int brow = (MODE == 2) ? boundary_row[r] : -1;
    __shared__ __align__(16) float s_q[TT][K];
    __shared__ __align__(16) float s_k[TT][K];
    __shared__ __align__(16) float s_v[TT][V];
    __shared__ __align__(16) float s_ge[TT][K];
    __shared__ float s_beta[TT], s_inv[2][TT];
    const float a = __expf(A_log[h]);
    for (int i = tid; i < len * K; i += THREADS) {
        const int t = i / K, d = i - t * K;
        const int64_t tok = int64_t(bos + t);
        s_q[t][d] = float(q[tok * stride_tok + int64_t(h) * K + d]);
        s_k[t][d] = float(k[tok * stride_tok + int64_t(h) * K + d]);
        s_v[t][d] = float(v[tok * stride_tok + int64_t(h) * V + d]);
        const float gg = float(raw_g[tok * stride_g + int64_t(h) * K + d]) + dt_bias[h * K + d];
        float gate;
        if (use_lower_bound) {
            gate = lower_bound * sigmoidf(a * gg);
        } else {
            const float sp = gg > 20.0f ? gg : __logf(1.0f + __expf(gg));
            gate = -a * sp;
        }
        s_ge[t][d] = __expf(gate);
    }
    if (tid < len) s_beta[tid] = sigmoidf(float(raw_beta[int64_t(bos + tid) * stride_beta + h]));
    __syncthreads();
    // L2 norms: pair p = (t = p/2, q or k), warps loop over the 2*len pairs.
    for (int p = warp; p < 2 * len; p += THREADS / 32) {
        const int t = p >> 1;
        const float* src = (p & 1) ? s_k[t] : s_q[t];
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float x = src[4 * lane + i]; ss += x * x; }
        ss = warp_sum(ss);
        if (lane == 0) s_inv[p & 1][t] = rsqrtf(ss + 1e-6f);
    }
    __syncthreads();
    // Lane mapping: G lanes share one state row (128 / G columns each), so a
    // row's two reductions per token cost log2(G) shuffles instead of 5 and
    // the per-token q/k/gate slices come from shared memory as warp
    // broadcasts (all row groups read the same columns).
    constexpr int G = 4;
    constexpr int CPL = K / G;               // columns per lane
    constexpr int ROWS_PER_PASS = 32 / G;    // rows a warp advances per pass
    constexpr int WARPS = THREADS / 32;
    static_assert(V % (WARPS * ROWS_PER_PASS) == 0, "rows per warp");
    const int sub = lane % G, rsel = lane / G;
    // Column chunk j of this lane: the G lanes of a row interleave their 16 B
    // chunks so one warp instruction covers a contiguous 4 * G * 4 B run per
    // row (full 32 B sectors on both loads and stores).
    auto col = [&](int j) { return j * (4 * G) + sub * 4; };
    // Normalized q (with scale) and k, and exp(gate), written back to smem once.
    for (int i = tid; i < len * K; i += THREADS) {
        const int t = i / K, d = i - t * K;
        s_q[t][d] *= s_inv[0][t] * scale;
        s_k[t][d] *= s_inv[1][t];
    }
    __syncthreads();
    const float* S0 = state + int64_t(state_idx) * stride_state + int64_t(h) * V * K;
    for (int row0 = warp * (V / WARPS); row0 < (warp + 1) * (V / WARPS); row0 += ROWS_PER_PASS) {
        const int row = row0 + rsel;
        float st[CPL];
        {
            const float* src = S0 + int64_t(row) * K;
#pragma unroll
            for (int j = 0; j < CPL / 4; ++j) { const float4 x = *reinterpret_cast<const float4*>(src + col(j)); st[4 * j] = x.x; st[4 * j + 1] = x.y; st[4 * j + 2] = x.z; st[4 * j + 3] = x.w; }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < len) {
                float dot_k = 0.0f;
#pragma unroll
                for (int j = 0; j < CPL / 4; ++j) {
                    const float4 gg = *reinterpret_cast<const float4*>(&s_ge[t][col(j)]);
                    const float4 k4 = *reinterpret_cast<const float4*>(&s_k[t][col(j)]);
                    st[4 * j] *= gg.x; st[4 * j + 1] *= gg.y; st[4 * j + 2] *= gg.z; st[4 * j + 3] *= gg.w;
                    dot_k += st[4 * j] * k4.x + st[4 * j + 1] * k4.y + st[4 * j + 2] * k4.z + st[4 * j + 3] * k4.w;
                }
#pragma unroll
                for (int o = 1; o < G; o <<= 1) dot_k += __shfl_xor_sync(0xffffffffu, dot_k, o);
                const float vp = (s_v[t][row] - dot_k) * s_beta[t];
#pragma unroll
                for (int j = 0; j < CPL / 4; ++j) {
                    const float4 k4 = *reinterpret_cast<const float4*>(&s_k[t][col(j)]);
                    st[4 * j] += vp * k4.x; st[4 * j + 1] += vp * k4.y; st[4 * j + 2] += vp * k4.z; st[4 * j + 3] += vp * k4.w;
                }
                if (MODE != 2) {
                    float dot_q = 0.0f;
#pragma unroll
                    for (int j = 0; j < CPL / 4; ++j) {
                        const float4 q4 = *reinterpret_cast<const float4*>(&s_q[t][col(j)]);
                        dot_q += st[4 * j] * q4.x + st[4 * j + 1] * q4.y + st[4 * j + 2] * q4.z + st[4 * j + 3] * q4.w;
                    }
#pragma unroll
                    for (int o = 1; o < G; o <<= 1) dot_q += __shfl_xor_sync(0xffffffffu, dot_q, o);
                    if (sub == 0) out[int64_t(bos + t) * stride_out + int64_t(h) * V + row] = __float2bfloat16_rn(dot_q);
                }
                const bool do_store = (MODE == 0) || (MODE == 2 && (t == len - 1 || t == brow));
                if (do_store) {
                    const int slot = state_indices[r * stride_idx + t];
                    if (slot > 0) {
                        float* dst = state + int64_t(slot) * stride_state + int64_t(h) * V * K + int64_t(row) * K;
#pragma unroll
                        for (int j = 0; j < CPL / 4; ++j) *reinterpret_cast<float4*>(dst + col(j)) = make_float4(st[4 * j], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]);
                    }
                }
            }
        }
    }
}

}  // namespace tms::kda
