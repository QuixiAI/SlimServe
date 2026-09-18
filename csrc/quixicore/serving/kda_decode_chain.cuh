// KDA decode chain: the per-layer decode glue of GLM-5.3's KDA layers in two
// launches instead of four (kda_gate_pair, causal_conv1d_update,
// fused_recurrent_kda, layer_norm_gated), both under programmatic dependent
// launch. Rows are speculative or plain decode: R requests of up to TT rows
// each (cu_seqlens), the conv state rolled by the accepted count like the
// Triton update kernel, the SSM state read once per (request, head) at the
// accepted column and stored after every row into that row's column.
//
// Kernel 1 (pre), one block per (request, head), 256 threads:
//   conv taps + SiLU on the head's q/k/v channels (bf16-rounded like the
//   Triton output), conv-state roll over the columns the Triton update
//   touches (width - 1 + rows - 1 per request: plain decode rolls three
//   columns, a speculative request of `len` rows rolls len + 2 and leaves the
//   rest of the history untouched), the f_b / g_b gate GEMVs for the head
//   (weights read once per block, T rows reuse them, outputs bf16-rounded),
//   the gate exp(lower_bound * sigmoid(a * (g + bias))), q/k L2 norms.
//   Writes the per-row operands (fp32) and g2 (bf16) to scratch.
// Kernel 2 (rec), SPLIT blocks per (request, head), each owning V/SPLIT state
//   rows in registers across the rows: delta-rule update and read-out,
//   per-row state stores, bf16-rounded outputs to scratch plus their sum of
//   squares; the last of the SPLIT blocks applies the gated RMS norm
//   (y * rsqrt(mean(y^2) + eps) * w * sigmoid(g2)) and writes bf16 output.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tms::kda_chain {

constexpr int HD = 128;            // head dim (K == V)
constexpr int PRE_THREADS = 256;
constexpr int MAX_T = 8;           // rows per request
constexpr int CONV_W = 4;
constexpr int MAX_STATE_LEN = CONV_W - 1 + MAX_T - 1;  // columns a request can touch

__device__ __forceinline__ void chain_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}
__device__ __forceinline__ void chain_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
template <typename T> __device__ __forceinline__ float ldf(const T* p);
template <> __device__ __forceinline__ float ldf<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ldf<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ void stf(T* p, float v);
template <> __device__ __forceinline__ void stf<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void stf<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16_rn(v); }

// Scratch written by kernel 1, read by kernel 2 (per token n, head h):
//   ops[((n*H + h)*4 + part)*128 + d], part 0 q (normalised, scaled), 1 k
//   (normalised), 2 v, 3 exp(gate); beta_sig[n*H + h]; g2[(n*H + h)*128 + d] bf16.
struct ChainScratch {
    float* ops;
    float* beta_sig;
    __nv_bfloat16* g2;
    float* o;        // [N, H, 128] fp32 (bf16-rounded read-out)
    float* sumsq;    // [N, H]
    int* counter;    // [R * H]
};

template <typename CT, int TT>
__global__ void __launch_bounds__(PRE_THREADS) kda_chain_pre_kernel(
        const __nv_bfloat16* __restrict__ mixed,    // [N, stride_tok] q | k | v channels
        CT* __restrict__ conv_state,                // [slots, dim, state_len] via strides
        const float* __restrict__ conv_w,           // [dim, 4]
        const int* __restrict__ conv_idx,           // [R] via stride_cidx
        const __nv_bfloat16* __restrict__ f_a,      // [N, stride_fa] (128 wide)
        const __nv_bfloat16* __restrict__ g_a,      // [N, stride_ga]
        const __nv_bfloat16* __restrict__ f_w,      // [H*128, 128]
        const __nv_bfloat16* __restrict__ g_w,      // [H*128, 128]
        const __nv_bfloat16* __restrict__ raw_beta, // [N, stride_beta] per head
        const float* __restrict__ A_log, const float* __restrict__ dt_bias,
        const int* __restrict__ cu_seqlens,         // [R+1] or null (one row per request)
        const int* __restrict__ accepted,           // [R] or null
        ChainScratch sc, int H,
        int64_t stride_tok, int64_t stride_fa, int64_t stride_ga, int64_t stride_beta,
        int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, int stride_cidx,
        float scale, float lower_bound, int use_lower_bound) {
    const int r = blockIdx.x / H, h = blockIdx.x - r * H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ __align__(16) float s_x[TT][3 * HD];   // conv outputs (bf16-rounded): q | k | v
    __shared__ __align__(16) float s_fa[TT][HD];
    __shared__ __align__(16) float s_ga[TT][HD];
    __shared__ float s_inv[2][TT];
    chain_launch_dependents();
    chain_wait();
    const int bos = cu_seqlens ? cu_seqlens[r] : r;
    const int len = cu_seqlens ? cu_seqlens[r + 1] - bos : 1;
    if (len <= 0) return;
    if (tid == 0) sc.counter[blockIdx.x] = 0;
    if (tid < len) sc.sumsq[(bos + tid) * H + h] = 0.0f;
    // ---- conv: channel c of this head (q: c < 128, k: < 256, v: < 384) ----
    {
        // Columns touched: width - 1 for a plain row, len + width - 2 for a
        // speculative request (the Triton update's varlen-revised state_len).
        const int state_len = (cu_seqlens ? len : 1) + CONV_W - 2;
        const int offset = accepted ? accepted[r] - 1 : 0;
        const int slot = conv_idx[r * stride_cidx];
        for (int c = tid; c < 3 * HD; c += PRE_THREADS) {
            const int ch = (c / HD) * H * HD + h * HD + (c % HD);
            CT* st = conv_state + int64_t(slot) * cs_seq + int64_t(ch) * cs_dim;
            const float4 w4 = *reinterpret_cast<const float4*>(conv_w + ch * CONV_W);
            float c0 = ldf(st + (offset + 0) * cs_tok);
            float c1 = ldf(st + (offset + 1) * cs_tok);
            float c2 = ldf(st + (offset + 2) * cs_tok);
            float x[TT];
#pragma unroll
            for (int t = 0; t < TT; ++t) x[t] = (t < len) ? __bfloat162float(mixed[int64_t(bos + t) * stride_tok + ch]) : 0.0f;
            float keep[MAX_STATE_LEN];
            for (int t2 = 0; t2 + len < state_len; ++t2) keep[t2] = ldf(st + (offset + t2 + 1) * cs_tok);
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t < len) {
                    float acc = 0.0f;
                    acc += c0 * w4.x; acc += c1 * w4.y; acc += c2 * w4.z; acc += x[t] * w4.w;
                    c0 = c1; c1 = c2; c2 = x[t];
                    acc = acc / (1.0f + __expf(-acc));
                    s_x[t][c] = bf16r(acc);
                }
            }
            // roll: new[t2] = old[offset + t2 + 1] while t2 + len < state_len, then the raw rows
            for (int t2 = 0; t2 < state_len; ++t2) {
                const float v = (t2 + len < state_len) ? keep[t2] : x[t2 - (state_len - len)];
                stf(st + t2 * cs_tok, v);
            }
        }
    }
    // ---- stage f_a / g_a rows ----
    for (int i = tid; i < len * HD; i += PRE_THREADS) {
        const int t = i / HD, d = i - t * HD;
        s_fa[t][d] = __bfloat162float(f_a[int64_t(bos + t) * stride_fa + d]);
        s_ga[t][d] = __bfloat162float(g_a[int64_t(bos + t) * stride_ga + d]);
    }
    __syncthreads();
    // ---- gate GEMVs: threads 0..127 -> g1 row h*128+d, 128..255 -> g2 ----
    {
        const bool is_g2 = tid >= HD;
        const int d = is_g2 ? tid - HD : tid;
        const __nv_bfloat16* wrow = (is_g2 ? g_w : f_w) + (int64_t(h) * HD + d) * HD;
        const float (*a)[HD] = is_g2 ? s_ga : s_fa;
        float acc[TT];
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
#pragma unroll 4
        for (int k0 = 0; k0 < HD; k0 += 8) {
            const uint4 u = *reinterpret_cast<const uint4*>(wrow + k0);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&u);
            float w[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) { const float2 f = __bfloat1622float2(w2[j]); w[2 * j] = f.x; w[2 * j + 1] = f.y; }
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t < len) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[t] += w[j] * a[t][k0 + j];
                }
            }
        }
        const float av = __expf(A_log[h]);
        const float bias = dt_bias[h * HD + d];
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < len) {
                const float o = bf16r(acc[t]);
                const int64_t nh = int64_t(bos + t) * H + h;
                if (is_g2) {
                    sc.g2[nh * HD + d] = __float2bfloat16_rn(o);
                } else {
                    const float gg = o + bias;
                    float gate;
                    if (use_lower_bound) {
                        gate = lower_bound * sigm(av * gg);
                    } else {
                        const float sp = gg > 20.0f ? gg : __logf(1.0f + __expf(gg));
                        gate = -av * sp;
                    }
                    sc.ops[(nh * 4 + 3) * HD + d] = __expf(gate);
                }
            }
        }
    }
    // ---- q / k L2 norms: pair p = (t, q|k) ----
    for (int p = warp; p < 2 * len; p += PRE_THREADS / 32) {
        const int t = p >> 1;
        const float* src = s_x[t] + ((p & 1) ? HD : 0);
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float v = src[4 * lane + i]; ss += v * v; }
        ss = wsum(ss);
        if (lane == 0) s_inv[p & 1][t] = 1.0f / sqrtf(ss + 1e-6f);
    }
    __syncthreads();
    for (int i = tid; i < len * HD; i += PRE_THREADS) {
        const int t = i / HD, d = i - t * HD;
        const int64_t nh = int64_t(bos + t) * H + h;
        float* op = sc.ops + nh * 4 * HD;
        op[0 * HD + d] = s_x[t][d] * s_inv[0][t] * scale;
        op[1 * HD + d] = s_x[t][HD + d] * s_inv[1][t];
        op[2 * HD + d] = s_x[t][2 * HD + d];
    }
    if (tid < len) sc.beta_sig[int64_t(bos + tid) * H + h] = sigm(__bfloat162float(raw_beta[int64_t(bos + tid) * stride_beta + h]));
}

// SPLIT blocks per (request, head); RB = 128 / SPLIT state rows per block, 4
// lanes per row (32 columns each, interleaved 16 B chunks), one pass per warp.
template <int TT, int SPLIT>
__global__ void __launch_bounds__(32 * (HD / SPLIT / 8)) kda_chain_rec_kernel(
        ChainScratch sc,
        float* __restrict__ state,                  // [slots, H, V, K]
        const int* __restrict__ state_indices,      // [R, S] via stride_idx
        const int* __restrict__ cu_seqlens, const int* __restrict__ accepted,
        const __nv_bfloat16* __restrict__ norm_w,   // [128]
        __nv_bfloat16* __restrict__ out,            // [N, stride_out] -> head h at h*128
        int H, int64_t stride_state, int stride_idx, int64_t stride_out, float eps) {
    constexpr int RB = HD / SPLIT;
    constexpr int WARPS = RB / 8;
    constexpr int THREADS = 32 * WARPS;
    constexpr int G = 4, CPL = HD / G;
    const int rh = blockIdx.x / SPLIT, vs = blockIdx.x - rh * SPLIT;
    const int r = rh / H, h = rh - r * H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int sub = lane % G, rsel = lane / G;
    __shared__ __align__(16) float s_q[TT][HD];
    __shared__ __align__(16) float s_k[TT][HD];
    __shared__ __align__(16) float s_ge[TT][HD];
    __shared__ float s_v[TT][RB];
    __shared__ float s_beta[TT], s_ss[TT];
    __shared__ int s_last;
    chain_launch_dependents();
    chain_wait();
    const int bos = cu_seqlens ? cu_seqlens[r] : r;
    const int len = cu_seqlens ? cu_seqlens[r + 1] - bos : 1;
    if (len <= 0) return;
    for (int i = tid; i < len * HD; i += THREADS) {
        const int t = i / HD, d = i - t * HD;
        const float* op = sc.ops + (int64_t(bos + t) * H + h) * 4 * HD;
        s_q[t][d] = op[d]; s_k[t][d] = op[HD + d]; s_ge[t][d] = op[3 * HD + d];
        if (d < RB) s_v[t][d] = op[2 * HD + vs * RB + d];
    }
    if (tid < len) { s_beta[tid] = sc.beta_sig[int64_t(bos + tid) * H + h]; s_ss[tid] = 0.0f; }
    __syncthreads();
    const int init_col = accepted ? max(accepted[r] - 1, 0) : 0;
    const int state_idx = state_indices[r * stride_idx + init_col];
    const int row = vs * RB + warp * 8 + rsel;
    auto col = [&](int j) { return j * (4 * G) + sub * 4; };
    float* orow = sc.o + (int64_t(bos) * H + h) * HD;   // + t*H*HD
    if (state_idx <= 0) {
        for (int t = 0; t < len; ++t)
            for (int d = tid; d < RB; d += THREADS) orow[int64_t(t) * H * HD + vs * RB + d] = 0.0f;
    } else {
        float st[CPL];
        const float* S0 = state + int64_t(state_idx) * stride_state + int64_t(h) * HD * HD + int64_t(row) * HD;
#pragma unroll
        for (int j = 0; j < CPL / 4; ++j) {
            const float4 x = *reinterpret_cast<const float4*>(S0 + col(j));
            st[4 * j] = x.x; st[4 * j + 1] = x.y; st[4 * j + 2] = x.z; st[4 * j + 3] = x.w;
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
                const float vp = (s_v[t][row - vs * RB] - dot_k) * s_beta[t];
                float dot_q = 0.0f;
#pragma unroll
                for (int j = 0; j < CPL / 4; ++j) {
                    const float4 k4 = *reinterpret_cast<const float4*>(&s_k[t][col(j)]);
                    const float4 q4 = *reinterpret_cast<const float4*>(&s_q[t][col(j)]);
                    st[4 * j] += vp * k4.x; st[4 * j + 1] += vp * k4.y; st[4 * j + 2] += vp * k4.z; st[4 * j + 3] += vp * k4.w;
                    dot_q += st[4 * j] * q4.x + st[4 * j + 1] * q4.y + st[4 * j + 2] * q4.z + st[4 * j + 3] * q4.w;
                }
#pragma unroll
                for (int o = 1; o < G; o <<= 1) dot_q += __shfl_xor_sync(0xffffffffu, dot_q, o);
                const float y = bf16r(dot_q);
                float part = (sub == 0) ? y * y : 0.0f;
                if (sub == 0) orow[int64_t(t) * H * HD + row] = y;
                part = wsum(part);
                if (lane == 0) atomicAdd(&s_ss[t], part);
                const int slot = state_indices[r * stride_idx + t];
                if (slot > 0) {
                    float* dst = state + int64_t(slot) * stride_state + int64_t(h) * HD * HD + int64_t(row) * HD;
#pragma unroll
                    for (int j = 0; j < CPL / 4; ++j)
                        *reinterpret_cast<float4*>(dst + col(j)) = make_float4(st[4 * j], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]);
                }
            }
        }
    }
    __syncthreads();
    if (tid < len) atomicAdd(&sc.sumsq[int64_t(bos + tid) * H + h], s_ss[tid]);
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = (atomicAdd(&sc.counter[rh], 1) == SPLIT - 1) ? 1 : 0;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // ---- gated RMS norm over the head's 128 outputs, all rows of the request ----
    for (int i = tid; i < len * HD; i += THREADS) {
        const int t = i / HD, d = i - t * HD;
        const int64_t nh = int64_t(bos + t) * H + h;
        const float ss = __ldcg(sc.sumsq + nh);
        const float rstd = 1.0f / sqrtf(ss / float(HD) + eps);
        const float y = __ldcg(sc.o + nh * HD + d);
        const float g = __bfloat162float(sc.g2[nh * HD + d]);
        const float w = __bfloat162float(norm_w[d]);
        out[int64_t(bos + t) * stride_out + int64_t(h) * HD + d] = __float2bfloat16_rn(y * rstd * w * sigm(g));
    }
}

}  // namespace tms::kda_chain
