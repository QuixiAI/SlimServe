// KDA decode block: the per-layer decode glue of GLM-5.3's KDA layers
// (kda_gate_pair, causal_conv1d_update, fused_recurrent_kda, layer_norm_gated)
// in ONE launch per layer. A (request, head) pair is served by 1024 threads:
// a thread-block cluster of CL blocks of 1024/CL threads. The phases stay as
// wide as the four kernels they replace and hand their results across the
// cluster through distributed shared memory, so there is no global scratch, no
// atomic counter and no grid-wide barrier:
//
//   phase A (every block, its slice of the head): conv taps + SiLU on 3*128/CL
//     of the head's q/k/v channels with the Triton update's exact history roll,
//     2*128/CL of the f_b / g_b gate GEMV outputs (4 lanes per output, the
//     weights read once, T rows reuse them), the gate exp and beta sigmoid;
//     the block's SSM state rows are loaded into registers at the very top so
//     that read overlaps the whole phase;
//   cluster.sync();
//   gather: q, k and exp(gate) of every row from the peers, this block's V rows
//     and output-gate columns, the q/k L2 norms;
//   phase B (this block's 128/CL state rows in registers, 8 lanes per row):
//     delta-rule update and read-out for every row of the request, the state
//     stored into each row's column, the read-out and its sum of squares kept
//     in shared memory;
//   cluster.sync();
//   the gated RMS norm of this block's columns with the cluster-wide sum of
//     squares; cluster.sync() before exit (peers still read this block's smem).
//
// CL is chosen per call: a cluster of 8 x 128 threads spreads one request's
// 16 heads over 128 SMs (the c1 case: the gate weights, conv history and
// state stream from 128 SMs at once), while this part keeps at most ~2
// cluster blocks resident per SM whatever their size (measured
// cudaOccupancyMaxActiveClusters: 46 clusters of 8, 94 of 4 on 188 SMs). The
// launcher picks the widest of CL 8 and CL 4 whose active-cluster capacity
// holds the whole batch in one wave and declines larger batches, which the
// Triton chain serves.
//
// Rows are speculative or plain decode: R requests of up to TT rows each
// (cu_seqlens), conv state rolled by the accepted count like the Triton update
// kernel, the SSM state (fp32 or bf16, loaded to fp32 and stored through its
// dtype) read once per (request, head) at the accepted column.
// Everything is bf16-rounded where the Triton chain materialises bf16, so the
// outputs match it within the rounding of the gate GEMV's summation order
// (tests/kernels/test_kda_decode_block.py).
#pragma once
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace tms::kda_block {

constexpr int HD = 128;                  // head dim (K == V)
constexpr int MAX_T = 8;                 // rows per request
constexpr int CONV_W = 4;
constexpr int PAIR_THREADS = 1024;       // threads per (request, head), over the cluster
constexpr int LPR = 8;                   // lanes per state row
constexpr int CPL = HD / LPR;            // 16 state columns per lane

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
// Four consecutive state columns (16 B fp32 or 8 B bf16), loaded to / stored from fp32.
template <typename T> __device__ __forceinline__ float4 ld4(const T* p);
template <> __device__ __forceinline__ float4 ld4<float>(const float* p) { return *reinterpret_cast<const float4*>(p); }
template <> __device__ __forceinline__ float4 ld4<__nv_bfloat16>(const __nv_bfloat16* p) {
    const uint2 u = *reinterpret_cast<const uint2*>(p);
    const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
    const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
    return make_float4(a.x, a.y, b.x, b.y);
}
template <typename T> __device__ __forceinline__ void st4(T* p, float4 v);
template <> __device__ __forceinline__ void st4<float>(float* p, float4 v) { *reinterpret_cast<float4*>(p) = v; }
template <> __device__ __forceinline__ void st4<__nv_bfloat16>(__nv_bfloat16* p, float4 v) {
    uint2 u;
    *reinterpret_cast<__nv_bfloat162*>(&u.x) = __float22bfloat162_rn(make_float2(v.x, v.y));
    *reinterpret_cast<__nv_bfloat162*>(&u.y) = __float22bfloat162_rn(make_float2(v.z, v.w));
    *reinterpret_cast<uint2*>(p) = u;
}

template <int CL> struct Geom {
    static constexpr int THREADS = PAIR_THREADS / CL;
    static constexpr int CONV_PER = 3 * HD / CL;    // conv channels per block
    static constexpr int GATE_PER = 2 * HD / CL;    // gate outputs per block (f then g)
    static constexpr int ROWS = HD / CL;            // state rows per block
    static_assert(THREADS == 4 * GATE_PER, "4 lanes per gate output");
    static_assert(THREADS == ROWS * LPR, "one state row per 8 lanes");
    static_assert(CONV_PER <= THREADS, "one conv channel per thread");
    static_assert(THREADS % 32 == 0 && THREADS >= 64, "whole warps");
};

__device__ __forceinline__ void block_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void block_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
}

template <typename T>
__device__ __forceinline__ const T* peer(cooperative_groups::cluster_group& cluster, const T* p, int rank) {
    return cluster.map_shared_rank(p, rank);
}

template <typename CT, typename ST, int TT, int CL>
__global__ void __launch_bounds__(Geom<CL>::THREADS) kda_block_kernel(
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
        ST* __restrict__ state,                     // [slots, H, V, K] fp32 or bf16
        const int* __restrict__ state_indices,      // [R, S] via stride_idx
        const __nv_bfloat16* __restrict__ norm_w,   // [128]
        __nv_bfloat16* __restrict__ out,            // [N, stride_out] -> head h at h*128
        int H, int64_t stride_tok, int64_t stride_fa, int64_t stride_ga, int64_t stride_beta,
        int64_t cs_seq, int64_t cs_dim, int64_t cs_tok, int stride_cidx,
        int64_t stride_state, int stride_idx, int64_t stride_out,
        float scale, float lower_bound, int use_lower_bound, float eps, int pdl) {
    using G = Geom<CL>;
    constexpr int THREADS = G::THREADS, CONV_PER = G::CONV_PER, GATE_PER = G::GATE_PER, ROWS = G::ROWS;
    constexpr int KEEP = TT + CONV_W - 2;            // conv history columns a request can touch
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    const int b = int(cluster.block_rank());
    const int rh = blockIdx.x / CL;
    const int r = rh / H, h = rh - r * H;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    __shared__ float s_x[TT][CONV_PER];              // this block's conv outputs (bf16-rounded)
    __shared__ float s_g[TT][GATE_PER];              // this block's gate outputs: exp(gate) | g2 (bf16-rounded)
    // The staged gate inputs (f_a, g_a) are dead once the GEMV is done, before
    // the first cluster.sync(); the gathered q / k reuse their buffers.
    __shared__ __align__(16) float s_ab[2][TT][HD];
    float (*s_fa)[HD] = s_ab[0];
    float (*s_ga)[HD] = s_ab[1];
    float (*s_q)[HD] = s_ab[0];                      // gathered, normalised and scaled
    float (*s_k)[HD] = s_ab[1];                      // gathered, normalised
    __shared__ __align__(16) float s_ge[TT][HD];     // gathered exp(gate)
    __shared__ float s_v[TT][ROWS];                  // this block's V rows
    __shared__ float s_g2[TT][ROWS];                 // this block's output-gate columns
    __shared__ float s_y[TT][ROWS];                  // this block's read-outs (bf16-rounded)
    __shared__ float s_beta[TT], s_inv[2][TT], s_ss[TT];
    if (pdl) block_wait();
    const int bos = cu_seqlens ? cu_seqlens[r] : r;
    const int len = cu_seqlens ? cu_seqlens[r + 1] - bos : 1;
    if (len <= 0) return;                            // per request: the whole cluster leaves together
    // ---- the block's state rows, loaded first so the read overlaps phase A ----
    const int sub = lane & (LPR - 1), rloc = warp * 4 + (lane >> 3), row = b * ROWS + rloc;
    auto col = [&](int j) { return j * (LPR * 4) + sub * 4; };
    const int init_col = accepted ? max(accepted[r] - 1, 0) : 0;
    const int state_idx = state_indices[r * stride_idx + init_col];
    float st[CPL];
    if (state_idx > 0) {
        const ST* S0 = state + int64_t(state_idx) * stride_state + int64_t(h) * HD * HD + int64_t(row) * HD;
#pragma unroll
        for (int j = 0; j < CPL / 4; ++j) {
            const float4 x = ld4(S0 + col(j));
            st[4 * j] = x.x; st[4 * j + 1] = x.y; st[4 * j + 2] = x.z; st[4 * j + 3] = x.w;
        }
    } else {
#pragma unroll
        for (int j = 0; j < CPL; ++j) st[j] = 0.0f;
    }
    // ---- gate GEMV weights: output o = b*GATE_PER + tid/4, lanes kq = tid%4 take 32 k each ----
    const int o = b * GATE_PER + (tid >> 2), kq = tid & 3;
    const bool is_g2 = o >= HD;
    const int od = is_g2 ? o - HD : o;
    uint4 wq[4];
    {
        const __nv_bfloat16* wrow = (is_g2 ? g_w : f_w) + (int64_t(h) * HD + od) * HD + kq * 32;
#pragma unroll
        for (int j = 0; j < 4; ++j) wq[j] = *reinterpret_cast<const uint4*>(wrow + 8 * j);
    }
    // ---- stage f_a / g_a rows (all threads) ----
    for (int i = tid; i < len * HD; i += THREADS) {
        const int t = i / HD, d = i - t * HD;
        s_fa[t][d] = __bfloat162float(f_a[int64_t(bos + t) * stride_fa + d]);
        s_ga[t][d] = __bfloat162float(g_a[int64_t(bos + t) * stride_ga + d]);
    }
    if (tid < len) { s_beta[tid] = sigm(__bfloat162float(raw_beta[int64_t(bos + tid) * stride_beta + h])); s_ss[tid] = 0.0f; }
    // ---- conv: channel c = b*CONV_PER + tid of this head (q: c < 128, k: < 256, v: < 384) ----
    if (tid < CONV_PER) {
        const int c = b * CONV_PER + tid;
        const int state_len = (cu_seqlens ? len : 1) + CONV_W - 2;
        const int offset = accepted ? accepted[r] - 1 : 0;
        const int slot = conv_idx[r * stride_cidx];
        const int ch = (c / HD) * H * HD + h * HD + (c % HD);
        CT* stp = conv_state + int64_t(slot) * cs_seq + int64_t(ch) * cs_dim;
        const float4 w4 = *reinterpret_cast<const float4*>(conv_w + ch * CONV_W);
        float c0 = ldf(stp + (offset + 0) * cs_tok);
        float c1 = ldf(stp + (offset + 1) * cs_tok);
        float c2 = ldf(stp + (offset + 2) * cs_tok);
        float x[TT];
#pragma unroll
        for (int t = 0; t < TT; ++t) x[t] = (t < len) ? __bfloat162float(mixed[int64_t(bos + t) * stride_tok + ch]) : 0.0f;
        float keep[KEEP];
#pragma unroll
        for (int t2 = 0; t2 < KEEP; ++t2) keep[t2] = (t2 + len < state_len) ? ldf(stp + (offset + t2 + 1) * cs_tok) : 0.0f;
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            if (t < len) {
                float acc = 0.0f;
                acc += c0 * w4.x; acc += c1 * w4.y; acc += c2 * w4.z; acc += x[t] * w4.w;
                c0 = c1; c1 = c2; c2 = x[t];
                acc = acc / (1.0f + __expf(-acc));
                s_x[t][tid] = bf16r(acc);
            }
        }
        // roll: new[t2] = old[offset + t2 + 1] while t2 + len < state_len, then the raw rows
#pragma unroll
        for (int t2 = 0; t2 < KEEP; ++t2) {
            if (t2 < state_len) {
                float v = keep[t2];
                if (t2 + len >= state_len) {
                    const int xi = t2 - (state_len - len);
#pragma unroll
                    for (int t = 0; t < TT; ++t) if (t == xi) v = x[t];
                }
                stf(stp + t2 * cs_tok, v);
            }
        }
    }
    __syncthreads();
    // ---- gate GEMVs from the staged rows ----
    {
        const float (*a)[HD] = is_g2 ? s_ga : s_fa;
        float acc[TT];
#pragma unroll
        for (int t = 0; t < TT; ++t) acc[t] = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&wq[j]);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float2 f = __bfloat1622float2(w2[q]);
                const int k0 = kq * 32 + 8 * j + 2 * q;
#pragma unroll
                for (int t = 0; t < TT; ++t) {
                    if (t < len) acc[t] += f.x * a[t][k0] + f.y * a[t][k0 + 1];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TT; ++t) {
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 1);
            acc[t] += __shfl_xor_sync(0xffffffffu, acc[t], 2);
        }
        if (kq == 0) {
            const float av = __expf(A_log[h]);
            const float bias = dt_bias[h * HD + od];
#pragma unroll
            for (int t = 0; t < TT; ++t) {
                if (t < len) {
                    const float ov = bf16r(acc[t]);
                    float g;
                    if (is_g2) {
                        g = ov;
                    } else {
                        const float gg = ov + bias;
                        float gate;
                        if (use_lower_bound) {
                            gate = lower_bound * sigm(av * gg);
                        } else {
                            const float sp = gg > 20.0f ? gg : __logf(1.0f + __expf(gg));
                            gate = -av * sp;
                        }
                        g = __expf(gate);
                    }
                    s_g[t][tid >> 2] = g;
                }
            }
        }
    }
    cluster.sync();                          // every block's conv and gate slices are visible
    // ---- gather the head's operands from the peers ----
    for (int i = tid; i < len * HD; i += THREADS) {
        const int t = i / HD, d = i - t * HD;
        const int cq = d, ck = HD + d;
        s_q[t][d] = peer(cluster, &s_x[0][0], cq / CONV_PER)[t * CONV_PER + cq % CONV_PER];
        s_k[t][d] = peer(cluster, &s_x[0][0], ck / CONV_PER)[t * CONV_PER + ck % CONV_PER];
        s_ge[t][d] = peer(cluster, &s_g[0][0], d / GATE_PER)[t * GATE_PER + d % GATE_PER];
    }
    for (int i = tid; i < len * ROWS; i += THREADS) {
        const int t = i / ROWS, j = i - t * ROWS;
        const int cv = 2 * HD + b * ROWS + j, og = HD + b * ROWS + j;
        s_v[t][j] = peer(cluster, &s_x[0][0], cv / CONV_PER)[t * CONV_PER + cv % CONV_PER];
        s_g2[t][j] = peer(cluster, &s_g[0][0], og / GATE_PER)[t * GATE_PER + og % GATE_PER];
    }
    __syncthreads();
    // ---- q / k L2 norms: pair p = (t, q|k), one warp each ----
    for (int p = warp; p < 2 * len; p += THREADS / 32) {
        const int t = p >> 1;
        const float* src = (p & 1) ? s_k[t] : s_q[t];
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float v = src[4 * lane + i]; ss += v * v; }
        ss = wsum(ss);
        if (lane == 0) s_inv[p & 1][t] = 1.0f / sqrtf(ss + 1e-6f);
    }
    __syncthreads();
    for (int i = tid; i < len * HD; i += THREADS) {
        const int t = i / HD, d = i - t * HD;
        s_q[t][d] *= s_inv[0][t] * scale;
        s_k[t][d] *= s_inv[1][t];
    }
    __syncthreads();
    // ---- recurrence over this block's rows: row = b*ROWS + warp*4 + lane/8, 16 interleaved columns per lane ----
    if (state_idx <= 0) {
        for (int i = tid; i < len * ROWS; i += THREADS) s_y[i / ROWS][i % ROWS] = 0.0f;
    } else {
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
                for (int o2 = 1; o2 < LPR; o2 <<= 1) dot_k += __shfl_xor_sync(0xffffffffu, dot_k, o2);
                const float vp = (s_v[t][rloc] - dot_k) * s_beta[t];
                float dot_q = 0.0f;
#pragma unroll
                for (int j = 0; j < CPL / 4; ++j) {
                    const float4 k4 = *reinterpret_cast<const float4*>(&s_k[t][col(j)]);
                    const float4 q4 = *reinterpret_cast<const float4*>(&s_q[t][col(j)]);
                    st[4 * j] += vp * k4.x; st[4 * j + 1] += vp * k4.y; st[4 * j + 2] += vp * k4.z; st[4 * j + 3] += vp * k4.w;
                    dot_q += st[4 * j] * q4.x + st[4 * j + 1] * q4.y + st[4 * j + 2] * q4.z + st[4 * j + 3] * q4.w;
                }
#pragma unroll
                for (int o2 = 1; o2 < LPR; o2 <<= 1) dot_q += __shfl_xor_sync(0xffffffffu, dot_q, o2);
                const float y = bf16r(dot_q);
                if (sub == 0) s_y[t][rloc] = y;
                float part = (sub == 0) ? y * y : 0.0f;
                part = wsum(part);
                if (lane == 0) atomicAdd(&s_ss[t], part);
                const int slot = state_indices[r * stride_idx + t];
                if (slot > 0) {
                    ST* dst = state + int64_t(slot) * stride_state + int64_t(h) * HD * HD + int64_t(row) * HD;
#pragma unroll
                    for (int j = 0; j < CPL / 4; ++j)
                        st4(dst + col(j), make_float4(st[4 * j], st[4 * j + 1], st[4 * j + 2], st[4 * j + 3]));
                }
            }
        }
    }
    if (pdl) block_launch_dependents();              // after the main work: the dependent's CTAs must not squat on the SMs
    cluster.sync();                          // every block's s_ss partials are complete
    // ---- gated RMS norm of this block's columns with the cluster-wide sum of squares ----
    for (int i = tid; i < len * ROWS; i += THREADS) {
        const int t = i / ROWS, j = i - t * ROWS;
        float ss = 0.0f;
#pragma unroll
        for (int p = 0; p < CL; ++p) ss += peer(cluster, &s_ss[0], p)[t];
        const float rstd = 1.0f / sqrtf(ss / float(HD) + eps);
        const int d = b * ROWS + j;
        const float w = __bfloat162float(norm_w[d]);
        out[int64_t(bos + t) * stride_out + int64_t(h) * HD + d] =
            __float2bfloat16_rn(s_y[t][j] * rstd * w * sigm(s_g2[t][j]));
    }
    cluster.sync();  // peers keep their shared memory until everyone has read it
}

}  // namespace tms::kda_block
