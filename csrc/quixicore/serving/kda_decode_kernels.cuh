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
