/**
 * @file
 * @brief CUDA/SM86 port of MetalForge's M6 misc kernels: the random-sign
 * Fast Walsh-Hadamard rotation at the core of TurboQuant (turboquant.metal),
 * GPTQ/act-order column permute (layout.metal), and per-LoRA MoE alignment
 * (moe.metal moe_lora_align).
 *
 * The FWHT rotation is the reusable QuaRot/SpinQuant-style transform: with a
 * per-index sign vector S and the FWHT matrix H, forward(x) = (1/sqrt(D)) H S x
 * and inverse(y) = S (1/sqrt(D)) H y, so inverse(forward(x)) == x (H^2 = D*I,
 * S^2 = I). One warp per row; lane owns D/32 contiguous elements (low index bits
 * butterflied in registers, the 5 lane bits via __shfl_xor). The sign vector is
 * a runtime buffer (D floats, +-1) rather than MetalForge's embedded 960-entry
 * FWHT_SIGNS tables. The DeepSeek sub-8-bit K-uniform / V-centroid cache codec
 * (tq_encode) that wraps this rotation is included below, also taking the sign
 * vector as a runtime buffer (pass the key-42 table for MetalForge byte-compat).
 */
#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace tm6 {

// log2(HEAD_SIZE): FWHT butterfly stage count. 64->6,128->7,256->8,512->9.
template <int HS> __host__ __device__ constexpr int fwht_stages() {
    int n = 0, x = HS; while (x > 1) { x >>= 1; ++n; } return n;
}

template <typename T> __device__ __forceinline__ float t6(T v);
template <> __device__ __forceinline__ float t6<float>(float v)                 { return v; }
template <> __device__ __forceinline__ float t6<__half>(__half v)               { return __half2float(v); }
template <> __device__ __forceinline__ float t6<__nv_bfloat16>(__nv_bfloat16 v) { return __bfloat162float(v); }
template <typename T> __device__ __forceinline__ T f6(float v);
template <> __device__ __forceinline__ float          f6<float>(float v)         { return v; }
template <> __device__ __forceinline__ __half         f6<__half>(float v)        { return __float2half(v); }
template <> __device__ __forceinline__ __nv_bfloat16  f6<__nv_bfloat16>(float v) { return __float2bfloat16(v); }

// Random-sign FWHT rotation, one warp per row. D in {64,128,256,512}; lane owns
// E=D/32 contiguous elements at [lane*E, lane*E+E). INVERSE=false applies the
// sign before the transform, true after (fused with the 1/sqrt(D) norm).
template <typename T, int D, bool INVERSE>
__global__ void fwht_rotate(const T* __restrict__ x, T* __restrict__ out,
                            const float* __restrict__ sign, int nrows) {
    constexpr int E = D / 32;
    const long row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= nrows) return;
    const long base = row * D + (long)lane * E;
    const float inv_sqrt = rsqrtf(float(D));

    float v[E];
    #pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] = t6(x[base + i]);
        if (!INVERSE) v[i] *= sign[lane * E + i];          // sign before (forward)
    }
    // in-lane butterflies over the low log2(E) index bits
    #pragma unroll
    for (int h = 1; h < E; h <<= 1) {
        #pragma unroll
        for (int i = 0; i < E; ++i) if ((i & h) == 0) {
            const float a = v[i], b = v[i + h];
            v[i] = a + b; v[i + h] = a - b;
        }
    }
    // cross-lane butterflies over the 5 lane bits (32 lanes = 2^5)
    #pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
        const bool upper = (lane & m) != 0;
        #pragma unroll
        for (int i = 0; i < E; ++i) {
            const float p = __shfl_xor_sync(0xffffffffu, v[i], m);
            v[i] = upper ? (p - v[i]) : (v[i] + p);
        }
    }
    #pragma unroll
    for (int i = 0; i < E; ++i) {
        v[i] *= inv_sqrt;
        if (INVERSE) v[i] *= sign[lane * E + i];           // sign after (inverse)
        out[base + i] = f6<T>(v[i]);
    }
}

// GPTQ/act-order column permute (MetalForge permute_cols_16bit): 16-bit gather
// output[r, c] = input[r, perm[c]]. grid (cols, rows); one thread per element.
template <typename T>
__global__ void permute_cols(const T* __restrict__ input, T* __restrict__ output,
                             const int* __restrict__ perm, int rows, int cols) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y;
    if (row >= rows || col >= cols) return;
    output[(long)row * cols + col] = input[(long)row * cols + perm[col]];
}

// Per-LoRA MoE alignment (MetalForge moe_lora_align_int32): one block per
// active LoRA adapter. Histograms token->expert (filtered by token_lora_mapping
// == lora_id), block-pads the per-expert segments, scatters routing-row indices
// into sorted_base, and fills expert_base with the owning expert per padded
// block. sorted/expert bases are per-LoRA slices. num_experts <= 256.
static __global__ void moe_lora_align(const int* __restrict__ topk_ids,
        const int* __restrict__ token_lora_mapping, const int* __restrict__ lora_ids,
        const uint8_t* __restrict__ adapter_enabled, int* __restrict__ sorted_token_ids,
        int* __restrict__ expert_ids, int* __restrict__ num_tokens_post_pad,
        int max_loras, int num_experts, int assignments, int topk, int block_size,
        int sorted_capacity_per_lora, int expert_capacity_per_lora) {
    const int lora_idx = blockIdx.x;
    const int tid = threadIdx.x, nt = blockDim.x;
    if (lora_idx >= max_loras || num_experts <= 0 || num_experts > 256) return;
    const int lora_id = lora_ids[lora_idx];
    if (lora_id < 0 || !adapter_enabled[lora_id]) return;

    int* sorted_base = sorted_token_ids + (long)lora_id * sorted_capacity_per_lora;
    int* expert_base = expert_ids + (long)lora_id * expert_capacity_per_lora;
    __shared__ int counts[257];
    __shared__ int offsets[257];

    for (int i = tid; i < sorted_capacity_per_lora; i += nt) sorted_base[i] = assignments;
    for (int i = tid; i < expert_capacity_per_lora; i += nt) expert_base[i] = -1;
    if (tid == 0) num_tokens_post_pad[lora_id] = 0;
    for (int e = tid; e <= num_experts; e += nt) counts[e] = 0;
    __syncthreads();

    for (int i = tid; i < assignments; i += nt) {
        if (token_lora_mapping[i / topk] != lora_id) continue;
        int e = topk_ids[i];
        if (e < 0 || e >= num_experts) e = num_experts;
        atomicAdd(&counts[e], 1);
    }
    __syncthreads();

    if (tid == 0) {
        int pad = 0;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = pad;
            pad += ((counts[e] + block_size - 1) / block_size) * block_size;
        }
        offsets[num_experts] = pad;
        num_tokens_post_pad[lora_id] = pad;
    }
    __syncthreads();
    for (int e = tid; e <= num_experts; e += nt) counts[e] = 0;
    __syncthreads();

    for (int i = tid; i < assignments; i += nt) {
        if (token_lora_mapping[i / topk] != lora_id) continue;
        int e = topk_ids[i];
        if (e < 0 || e >= num_experts) e = num_experts;
        const int dst = offsets[e] + atomicAdd(&counts[e], 1);
        if (dst >= 0 && dst < sorted_capacity_per_lora) sorted_base[dst] = i;
    }
    __syncthreads();

    for (int e = tid; e < num_experts; e += nt) {
        const int sb = offsets[e] / block_size, nb = (offsets[e+1] - offsets[e]) / block_size;
        for (int b = 0; b < nb && sb + b < expert_capacity_per_lora; ++b) expert_base[sb + b] = e;
    }
}

// ============================ TurboQuant cache codec ========================
// Fused encode-and-cache (MetalForge tq_encode): for one (token, kv_head), quant
// the K row with per-32-group asymmetric-uniform quant (signed q8_0 / unsigned
// sub-8-bit) and the V row with a forward random-sign FWHT rotation -> per-32
// RMS scale -> Lloyd-Max centroid searchsorted. Writes fp8/sub-byte codes plus
// the fp16 scale/zero caches at slot_mapping[token]. Sign vector is a runtime
// buffer (HEAD_SIZE floats, +-1) — pass MetalForge's key-42 FWHT_SIGNS table for
// byte-compatibility with its caches. Decode: K = (idx+zp)*scale; V = inverse
// fwht_rotate over centroids[idx]*scale (the forward rotation here and
// fwht_rotate<INVERSE=true> are a matched pair over the same sign vector).
//
// One block per (token, kv_head); blockDim.x = HEAD_SIZE. Each warp (32 lanes)
// owns exactly one 32-element scale group, so min/max/RMS reduce over the warp.

// Forward FWHT on a per-thread scalar (element index = t). Stages 0-4 use warp
// shuffle; stages 5+ cross warps via shared memory. Returns (1/sqrt(HS))*H*(S.x).
template <int HS>
__device__ __forceinline__ float tg_forward_fwht(float x, float* buf, const float* sign, int t) {
    constexpr int NS = fwht_stages<HS>();
    const float inv_sqrt = rsqrtf((float)HS);
    x *= sign[t];
    #pragma unroll
    for (int s = 0; s < 5 && s < NS; ++s) {
        const unsigned mask = 1u << s;
        const float p = __shfl_xor_sync(0xffffffffu, x, mask);
        x = (t & mask) ? (p - x) : (x + p);
    }
    if (NS > 5) {
        buf[t] = x; __syncthreads();
        for (int s = 5; s < NS; ++s) {
            const unsigned mask = 1u << s;
            const float me = buf[t], p = buf[t ^ mask];
            __syncthreads();
            buf[t] = (t & mask) ? (p - me) : (me + p);
            __syncthreads();
        }
        x = buf[t];
    }
    return x * inv_sqrt;
}

// Assemble one output byte from staged sub-8-bit indices (bits in [1,8]).
__device__ __forceinline__ unsigned tq_pack_byte(const uint8_t* idx_buf, int bit_start, int bits) {
    const int first_e = bit_start / bits, last_e = (bit_start + 7) / bits;
    const unsigned mask = (1u << bits) - 1u;
    unsigned byte = 0;
    for (int e = first_e; e <= last_e; ++e) {
        const int shift = e * bits - bit_start;
        const unsigned v = (unsigned)idx_buf[e] & mask;
        byte |= (shift >= 0) ? (v << shift) : (v >> (-shift));
    }
    return byte & 0xFFu;
}

template <typename T, int HS>
__global__ void tq_encode(const T* __restrict__ key, const T* __restrict__ value,
        uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
        __half* __restrict__ key_scale_cache, __half* __restrict__ value_scale_cache,
        __half* __restrict__ key_zero_cache, const int64_t* __restrict__ slot_mapping,
        const float* __restrict__ v_centroids, const float* __restrict__ sign,
        int num_kv_heads, int block_size, int k_bits, int k_signed, int v_bits) {
    constexpr int SG = 32;
    const int t = threadIdx.x, token = blockIdx.x, kvh = blockIdx.y;
    const int sid = t >> 5, lane = t & 31;
    const int64_t slot = slot_mapping[token];
    if (slot < 0) return;

    const int block_idx = int(slot / block_size), block_off = int(slot % block_size);
    constexpr int head_dim = HS, scale_groups = HS / SG;
    const int k_packed = (head_dim * k_bits + 7) / 8;
    const int v_packed = (head_dim * v_bits + 7) / 8;

    const int64_t src_base = ((int64_t)token * num_kv_heads + kvh) * head_dim;
    const float k_val = t6(key[src_base + t]);
    const float v_val = t6(value[src_base + t]);

    __shared__ float fwht_buf[HS];
    __shared__ uint8_t k_idx_buf[HS];
    __shared__ uint8_t v_idx_buf[HS];
    __shared__ float v_boundaries[255];

    const int num_centroids = 1 << v_bits;
    for (int i = t; i < num_centroids - 1; i += HS)
        v_boundaries[i] = 0.5f * (v_centroids[i] + v_centroids[i + 1]);

    const int64_t token_base = ((int64_t)block_idx * block_size + block_off) * num_kv_heads;
    const int64_t kc_base = (token_base + kvh) * k_packed;
    const int64_t vc_base = (token_base + kvh) * v_packed;
    const int64_t scale_base = (token_base + kvh) * scale_groups;

    // ---- K encode: per-warp min/max, asymmetric uniform quant (fp16 domain) ----
    float k_min = k_val, k_max = k_val;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        k_min = fminf(k_min, __shfl_xor_sync(0xffffffffu, k_min, o));
        k_max = fmaxf(k_max, __shfl_xor_sync(0xffffffffu, k_max, o));
    }
    __half k_scale_h; float k_zp_f; int k_idx_i;
    if (k_signed) {
        const int max_val = (1 << (k_bits - 1)) - 1;
        k_scale_h = __hdiv(__float2half(k_max - k_min), __float2half(2.0f * max_val));
        const __half k_sum_h = __float2half(k_max + k_min);
        k_zp_f = rintf(__half2float(__hdiv(k_sum_h, __hmul(__float2half(2.0f), k_scale_h))));
        k_idx_i = int(rintf(__half2float(__hdiv(__float2half(k_val), k_scale_h)) - k_zp_f));
        k_idx_i = max(-max_val, min(k_idx_i, max_val));
    } else {
        const int max_val = (1 << k_bits) - 1;
        k_scale_h = __hdiv(__float2half(k_max - k_min), __float2half((float)max_val));
        k_zp_f = rintf(__half2float(__hdiv(__float2half(k_min), k_scale_h)));
        k_idx_i = int(rintf(__half2float(__hdiv(__float2half(k_val), k_scale_h)) - k_zp_f));
        k_idx_i = max(0, min(k_idx_i, max_val));
    }
    if (lane == 0) {
        key_scale_cache[scale_base + sid] = k_scale_h;
        key_zero_cache[scale_base + sid] = __float2half(k_zp_f);
    }
    if (k_bits == 8) {
        key_cache[kc_base + t] = (uint8_t)((unsigned)k_idx_i & 0xFFu);
    } else {
        k_idx_buf[t] = (uint8_t)((unsigned)k_idx_i & ((1u << k_bits) - 1u));
        __syncthreads();
        if (t < k_packed) key_cache[kc_base + t] = (uint8_t)tq_pack_byte(k_idx_buf, t * 8, k_bits);
    }

    // ---- V encode: forward FWHT -> per-warp RMS scale -> centroid searchsorted ----
    const float v_rot = tg_forward_fwht<HS>(v_val, fwht_buf, sign, t);
    const __half v_rot_h = __float2half(v_rot);
    float v_sq = __half2float(v_rot_h) * __half2float(v_rot_h);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v_sq += __shfl_xor_sync(0xffffffffu, v_sq, o);
    const __half v_scale_h = __float2half(sqrtf(v_sq * (1.0f / (float)SG)));
    const float v_norm = __half2float(__hdiv(v_rot_h, v_scale_h));

    __syncthreads();                                  // publish v_boundaries preload
    int v_idx = 0;
    for (int i = 0; i < num_centroids - 1; ++i) if (v_norm > v_boundaries[i]) ++v_idx;
    v_idx_buf[t] = (uint8_t)(v_idx & ((1 << v_bits) - 1));
    if (lane == 0) value_scale_cache[scale_base + sid] = v_scale_h;
    __syncthreads();
    if (t < v_packed) value_cache[vc_base + t] = (uint8_t)tq_pack_byte(v_idx_buf, t * 8, v_bits);
}

} // namespace tm6
