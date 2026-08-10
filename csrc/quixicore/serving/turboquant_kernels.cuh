#pragma once
// TurboQuant KV cache, native CUDA port of the Triton kernels in
// vllm/v1/attention/ops/triton_turboquant_{store,decode}.py.
//
// Bit-exactness contract (vs the Triton originals on the same GPU):
//   * store kernels are bitwise: `/` is div.full.f32, fp32->fp8e4b15 is
//     cvt.rz.f16.f32 then clamp-to-0x3F00 / shift / +0x80 round (the exact
//     PTX Triton emits), fp16 downcasts are cvt.rn.f16.f32, float->int is
//     cvt.rzi, and the min/max reductions are order-independent.
//   * decode stage1 pins Triton's op order for BLOCK_D == 64 / num_warps=1:
//     element d lives on lane d%32, the two per-lane products combine as
//     fma(q_lo, k_lo, q_hi*k_hi), cross-lane sums walk shfl.bfly 16..1, the
//     exp argument is fma(raw_sum, scale, -new_max), and the 4-token value
//     fold is fma(p3,v3, fma(p2,v2, fma(p0,v0, p1*v1))).
//   * stage2 and the full-dequant kernel are elementwise with a sequential
//     split loop, bitwise given equal inputs (fma spots pinned).
//
// Cache slot layout per (position, head), byte-packed:
//   FP8 keys: [key fp8e4b15 (D)                | value | v_scale f16 | v_zero f16]
//   MSE keys: [key idx (MSE_BYTES) | norm f16  | value | v_scale f16 | v_zero f16]
// value is D 3- or 4-bit codes (VAL_DATA_BYTES bytes).
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "tm_warp.cuh"
#include <cstdint>

#ifndef CUDART_INF_F_TQ
#define CUDART_INF_F_TQ __int_as_float(0x7f800000)
#endif

namespace tms {

// ---- Triton-lowering-exact scalar ops ----
__device__ __forceinline__ float tq_div(float a, float b) {
    float c;
    asm("div.full.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}
__device__ __forceinline__ float tq_exp(float x) {
    float y = x * 1.4426950408889634f;  // 0f3FB8AA3B, triton's log2(e)
    asm("ex2.approx.f32 %0, %0;" : "+f"(y));
    return y;
}
// fp32 -> fp8e4b15 byte, replicating Triton's Fp32->Fp8E4M3B15 lowering:
// round-toward-zero to f16, clamp |x| to 1.75 (0x3F00), then take the high
// byte of (abs<<1)+0x80 with the sign OR'd back in.
__device__ __forceinline__ uint8_t tq_f32_to_e4b15(float x) {
    uint16_t h;
    asm("cvt.rz.f16.f32 %0, %1;" : "=h"(h) : "f"(x));
    const uint32_t s = h & 0x8000u;
    uint32_t a = h & 0x7fffu;
    if (a > 0x3F00u) a = 0x3F00u;  // min.f16 vs 1.75 == integer min here
    return uint8_t((s | ((a << 1) + 0x80u)) >> 8);
}
// fp8e4b15 -> f16 bits is an exact left-shift-by-7 expansion.
__device__ __forceinline__ uint16_t tq_e4b15_to_f16_bits(uint8_t b) {
    return uint16_t(((b & 0x80u) << 8) | ((b & 0x7fu) << 7));
}
__device__ __forceinline__ float tq_e4b15_to_f32(uint8_t b) {
    return __half2float(__ushort_as_half(tq_e4b15_to_f16_bits(b)));
}
__device__ __forceinline__ float tq_load_f16(const uint8_t* p) {
    return __half2float(__ushort_as_half(uint16_t(p[0] | (p[1] << 8))));
}
__device__ __forceinline__ void tq_store_f16(uint8_t* p, float x) {
    const uint16_t u = __half_as_ushort(__float2half_rn(x));
    p[0] = uint8_t(u & 0xFF);
    p[1] = uint8_t(u >> 8);
}
__device__ __forceinline__ float tq_to_f32(__half v) { return __half2float(v); }
__device__ __forceinline__ float tq_to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ float tq_to_f32(float v) { return v; }

// bfly trees over 32 lanes; min/max/sum orders match Triton's warp reduce.
__device__ __forceinline__ float tq_warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, off));
    return v;
}

// ---- shared value-quantization tail (uniform 3/4-bit + scale/zero) ----
// One warp; sq is a >=D-byte shared scratch. Mirrors _store_quantized_value.
template <typename TV>
__device__ __forceinline__ void tq_store_value_q(
        const TV* value, uint8_t* cache, int64_t base, int64_t slot_base,
        int D, int kps, int vqb, int val_data_bytes, int lane, uint8_t* sq) {
    float vals[16];
    float vmin = CUDART_INF_F_TQ, vmax = -CUDART_INF_F_TQ;
    for (int d = lane, i = 0; d < D; d += 32, i++) {
        const float x = tq_to_f32(value[base + d]);
        vals[i] = x;
        vmin = fminf(vmin, x);
        vmax = fmaxf(vmax, x);
    }
    vmin = warp_min_f(vmin);
    vmax = warp_max_f(vmax);
    const float levels = (vqb == 3) ? 7.0f : 15.0f;
    const float v_scale = fmaxf(tq_div(vmax - vmin, levels), 1e-8f);
    const int qmax = (vqb == 3) ? 7 : 15;
    for (int d = lane, i = 0; d < D; d += 32, i++) {
        int q = int(tq_div(vals[i] - vmin, v_scale) + 0.5f);
        q = min(max(q, 0), qmax);
        sq[d] = uint8_t(q);
    }
    __syncwarp();
    const int64_t vbase = slot_base + kps;
    if (vqb == 4) {
        for (int j = lane; j < val_data_bytes; j += 32)
            cache[vbase + j] = uint8_t((sq[2 * j] & 0xF) | ((sq[2 * j + 1] & 0xF) << 4));
    } else {
        for (int g = lane; g < D / 8; g += 32) {
            uint32_t packed = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                packed |= uint32_t(sq[8 * g + i] & 0x7) << (3 * i);
            cache[vbase + 3 * g] = uint8_t(packed & 0xFF);
            cache[vbase + 3 * g + 1] = uint8_t((packed >> 8) & 0xFF);
            cache[vbase + 3 * g + 2] = uint8_t((packed >> 16) & 0xFF);
        }
    }
    if (lane == 0) {
        tq_store_f16(cache + vbase + val_data_bytes, v_scale);
        tq_store_f16(cache + vbase + val_data_bytes + 2, vmin);
    }
}

// ---- store, FP8-key path (_tq_fused_store_fp8) ----
// grid: (N*H) warps, block 32. key/value are [N*H, D] contiguous.
template <typename T, typename SlotT>
__global__ void tq_store_fp8(const T* __restrict__ key, const T* __restrict__ value,
                             uint8_t* __restrict__ cache,
                             const SlotT* __restrict__ slot_mapping,
                             int64_t stride_block, int64_t stride_pos, int64_t stride_head,
                             int D, int H, int block_size,
                             int kps, int vqb, int val_data_bytes) {
    __shared__ uint8_t sq[512];
    const int pid = blockIdx.x;
    const int token = pid / H, head = pid % H;
    const int64_t slot = int64_t(slot_mapping[token]);
    if (slot < 0) return;
    const int64_t slot_base = (slot / block_size) * stride_block
                            + (slot % block_size) * stride_pos
                            + int64_t(head) * stride_head;
    const int64_t base = int64_t(pid) * D;
    const int lane = threadIdx.x;
    for (int d = lane; d < D; d += 32)
        cache[slot_base + d] = tq_f32_to_e4b15(tq_to_f32(key[base + d]));
    tq_store_value_q(value, cache, base, slot_base, D, kps, vqb, val_data_bytes,
                     lane, sq);
}

// ---- store, MSE-key path (_tq_fused_store_mse) ----
// y/norms/value are the launcher's fp32 tensors (normalize + rotation GEMM
// stay outside, in cuBLAS, exactly like the Triton launcher).
template <typename SlotT>
__global__ void tq_store_mse(const float* __restrict__ y, const float* __restrict__ norms,
                             const float* __restrict__ value,
                             const float* __restrict__ midpoints,
                             uint8_t* __restrict__ cache,
                             const SlotT* __restrict__ slot_mapping,
                             int64_t stride_block, int64_t stride_pos, int64_t stride_head,
                             int D, int H, int block_size,
                             int mse_bits, int mse_bytes, int n_centroids,
                             int kps, int vqb, int val_data_bytes) {
    __shared__ uint8_t sq[512];
    const int pid = blockIdx.x;
    const int token = pid / H, head = pid % H;
    const int64_t slot = int64_t(slot_mapping[token]);
    if (slot < 0) return;
    const int64_t slot_base = (slot / block_size) * stride_block
                            + (slot % block_size) * stride_pos
                            + int64_t(head) * stride_head;
    const int64_t base = int64_t(pid) * D;
    const int lane = threadIdx.x;

    // binary-search bucketize, identical branch structure to the Triton loop
    for (int d = lane; d < D; d += 32) {
        const float yv = y[base + d];
        int lo = 0, hi = n_centroids - 1;
        for (int it = 0; it < mse_bits; it++) {
            const int mid = (lo + hi) >> 1;
            const float mv = midpoints[min(mid, n_centroids - 2)];
            if (yv >= mv) lo = mid + 1; else hi = mid;
        }
        sq[d] = uint8_t(min(lo, n_centroids - 1));
    }
    __syncwarp();
    if (mse_bits == 4) {
        for (int j = lane; j < mse_bytes; j += 32)
            cache[slot_base + j] = uint8_t((sq[2 * j] & 0xF) | ((sq[2 * j + 1] & 0xF) << 4));
    } else {  // mse_bits == 3
        for (int g = lane; g < D / 8; g += 32) {
            uint32_t packed = 0;
            #pragma unroll
            for (int i = 0; i < 8; i++)
                packed |= uint32_t(sq[8 * g + i] & 0x7) << (3 * i);
            cache[slot_base + 3 * g] = uint8_t(packed & 0xFF);
            cache[slot_base + 3 * g + 1] = uint8_t((packed >> 8) & 0xFF);
            cache[slot_base + 3 * g + 2] = uint8_t((packed >> 16) & 0xFF);
        }
    }
    if (lane == 0) tq_store_f16(cache + slot_base + mse_bytes, norms[pid]);
    __syncwarp();
    tq_store_value_q(value, cache, base, slot_base, D, kps, vqb, val_data_bytes,
                     lane, sq);
}

// ---- per-token value dequant used by stage1 / full dequant ----
__device__ __forceinline__ void tq_load_value_sz(const uint8_t* cache, int64_t vbase,
                                                 int val_data_bytes,
                                                 float& v_scale, float& v_zero) {
    v_scale = tq_load_f16(cache + vbase + val_data_bytes);
    v_zero = tq_load_f16(cache + vbase + val_data_bytes + 2);
}
__device__ __forceinline__ float tq_value_idx(const uint8_t* cache, int64_t vbase,
                                              int vqb, int d) {
    if (vqb == 4) {
        const int raw = cache[vbase + (d >> 1)];
        return float((raw >> ((d & 1) * 4)) & 0xF);
    }
    const int bit = d * 3;
    const int raw = cache[vbase + (bit >> 3)] | (cache[vbase + (bit >> 3) + 1] << 8);
    return float((raw >> (bit & 7)) & 0x7);
}
__device__ __forceinline__ int tq_mse_idx(const uint8_t* cache, int64_t slot_base,
                                          int mse_bits, int d) {
    const int bit = d * mse_bits;
    const int raw = cache[slot_base + (bit >> 3)] | (cache[slot_base + (bit >> 3) + 1] << 8);
    return (raw >> (bit & 7)) & ((1 << mse_bits) - 1);
}

// ---- decode stage 1 (_tq_decode_stage1) ----
// grid (B, Hq, NUM_KV_SPLITS), block 32. TQ = q element type (fp16/bf16 for
// the fp8-key path, fp32 for the rotated MSE query). Element d -> lane d%32,
// register i = d/32; VPL = D/32 <= 16.
template <typename TQ>
__global__ void tq_decode_stage1(const TQ* __restrict__ q_rot,
                                 const uint8_t* __restrict__ cache,
                                 const int* __restrict__ block_table,
                                 const int* __restrict__ seq_lens,
                                 const float* __restrict__ centroids,
                                 float* __restrict__ mid_o,
                                 int64_t stride_qb, int64_t stride_qh,
                                 int64_t stride_cb, int64_t stride_cp, int64_t stride_ch,
                                 int64_t stride_bt,
                                 int64_t stride_mb, int64_t stride_mh, int64_t stride_ms,
                                 int D, int block_size, int num_splits, int kv_group,
                                 int mse_bits, int mse_bytes, int kps, int vqb,
                                 int val_data_bytes, float scale,
                                 int key_fp8, int norm_correction, int window) {
    const int bid = blockIdx.x, hid = blockIdx.y, sid = blockIdx.z;
    const int kv_head = hid / kv_group;
    const int seq_len = seq_lens[bid];
    int kv_start = 0;
    if (window > 0) kv_start = max(seq_len - window, 0);
    const int split_len = (seq_len - kv_start + num_splits - 1) / num_splits;
    const int split_start = kv_start + split_len * sid;
    const int split_end = min(split_start + split_len, seq_len);
    if (split_start >= split_end) return;

    const int lane = threadIdx.x;
    const int VPL = D / 32;  // <= 16 (checked in the launcher)
    float qv[16];
    const int64_t q_base = int64_t(bid) * stride_qb + int64_t(hid) * stride_qh;
    for (int i = 0; i < VPL; i++) qv[i] = tq_to_f32(q_rot[q_base + lane + 32 * i]);

    float m_prev = -CUDART_INF_F_TQ, l_prev = 0.0f;
    float acc[16] = {};
    const int64_t bt_base = int64_t(bid) * stride_bt;

    for (int start_n = split_start; start_n < split_end; start_n += 4) {
        float p[4], v[4][16], raw[4];
        bool valid[4];
        int64_t sb[4];
        for (int t = 0; t < 4; t++) {
            const int kv = start_n + t;
            valid[t] = kv < split_end;
            raw[t] = 0.0f;
            if (!valid[t]) continue;
            const int64_t bnum = int64_t(block_table[bt_base + kv / block_size]);
            sb[t] = bnum * stride_cb + int64_t(kv % block_size) * stride_cp
                  + int64_t(kv_head) * stride_ch;
            float partial;
            if (key_fp8) {
                const float k0 = tq_e4b15_to_f32(cache[sb[t] + lane]);
                if (VPL == 2) {
                    // Triton's D=64 pairing: fma(q_lo, k_lo, q_hi * k_hi)
                    const float k1 = tq_e4b15_to_f32(cache[sb[t] + lane + 32]);
                    partial = __fmaf_rn(qv[0], k0, __fmul_rn(qv[1], k1));
                } else {
                    partial = __fmul_rn(qv[0], k0);
                    for (int i = 1; i < VPL; i++)
                        partial = __fmaf_rn(qv[i],
                                            tq_e4b15_to_f32(cache[sb[t] + lane + 32 * i]),
                                            partial);
                }
                raw[t] = tq_warp_sum(partial);
            } else {
                float c[16];
                for (int i = 0; i < VPL; i++)
                    c[i] = centroids[tq_mse_idx(cache, sb[t], mse_bits, lane + 32 * i)];
                if (norm_correction) {
                    float nsq = __fmul_rn(c[0], c[0]);
                    for (int i = 1; i < VPL; i++)
                        nsq = __fmaf_rn(c[i], c[i], nsq);
                    nsq = tq_warp_sum(nsq);
                    const float inv = tq_div(1.0f, sqrtf(nsq + 1e-16f));
                    for (int i = 0; i < VPL; i++) c[i] = __fmul_rn(c[i], inv);
                }
                if (VPL == 2) {
                    partial = __fmaf_rn(qv[0], c[0], __fmul_rn(qv[1], c[1]));
                } else {
                    partial = __fmul_rn(qv[0], c[0]);
                    for (int i = 1; i < VPL; i++)
                        partial = __fmaf_rn(qv[i], c[i], partial);
                }
                const float term1 = tq_warp_sum(partial);
                const float vec_norm = tq_load_f16(cache + sb[t] + mse_bytes);
                raw[t] = __fmul_rn(vec_norm, term1);
            }
        }
        // online softmax, Triton's fold: max(((s0,s1),s2),s3) then m_prev,
        // exp argument re-derived as fma(raw, scale, -new_max)
        float sc[4];
        for (int t = 0; t < 4; t++)
            sc[t] = valid[t] ? __fmul_rn(raw[t], scale) : -CUDART_INF_F_TQ;
        const float n_e_max =
            fmaxf(fmaxf(fmaxf(fmaxf(sc[0], sc[1]), sc[2]), sc[3]), m_prev);
        const float re_scale = tq_exp(m_prev - n_e_max);
        for (int t = 0; t < 4; t++)
            p[t] = tq_exp(valid[t] ? __fmaf_rn(raw[t], scale, -n_e_max)
                                   : -CUDART_INF_F_TQ);
        // values (masked tokens contribute exact zeros, like Triton's
        // masked loads: idx=0, scale=0, zero=0 -> value 0 and p=0)
        for (int t = 0; t < 4; t++) {
            if (!valid[t]) {
                for (int i = 0; i < VPL; i++) v[t][i] = 0.0f;
                continue;
            }
            const int64_t vbase = sb[t] + kps;
            float v_scale, v_zero;
            tq_load_value_sz(cache, vbase, val_data_bytes, v_scale, v_zero);
            for (int i = 0; i < VPL; i++)
                v[t][i] = __fmaf_rn(tq_value_idx(cache, vbase, vqb, lane + 32 * i),
                                    v_scale, v_zero);
        }
        for (int i = 0; i < VPL; i++) {
            float s = __fmaf_rn(p[0], v[0][i], __fmul_rn(p[1], v[1][i]));
            s = __fmaf_rn(p[2], v[2][i], s);
            s = __fmaf_rn(p[3], v[3][i], s);
            acc[i] = __fmaf_rn(acc[i], re_scale, s);
        }
        l_prev = __fmaf_rn(l_prev, re_scale,
                           __fadd_rn(__fadd_rn(__fadd_rn(p[0], p[1]), p[2]), p[3]));
        m_prev = n_e_max;
    }

    const int64_t out_base = int64_t(bid) * stride_mb + int64_t(hid) * stride_mh
                           + int64_t(sid) * stride_ms;
    const float safe_l = l_prev > 0.0f ? l_prev : 1.0f;
    for (int i = 0; i < VPL; i++)
        mid_o[out_base + lane + 32 * i] = tq_div(acc[i], safe_l);
    if (lane == 0) mid_o[out_base + D] = m_prev + logf(safe_l);
}

// ---- decode stage 2 (_fwd_kernel_stage2) ----
// grid (B, Hq), block 32; sequential log-sum-exp merge across splits.
template <typename TO>
__global__ void tq_decode_stage2(const float* __restrict__ mid_o, TO* __restrict__ out,
                                 float* __restrict__ lse,
                                 const int* __restrict__ seq_lens,
                                 int64_t stride_mb, int64_t stride_mh, int64_t stride_ms,
                                 int64_t stride_ob, int64_t stride_oh, int64_t stride_lb,
                                 int num_splits, int D) {
    const int bid = blockIdx.x, hid = blockIdx.y;
    const int seq_len = seq_lens[bid];
    const int lane = threadIdx.x;
    const int VPL = D / 32;
    float acc[16] = {};
    float e_sum = 0.0f, e_max = -CUDART_INF_F_TQ;
    const int64_t base = int64_t(bid) * stride_mb + int64_t(hid) * stride_mh;
    const int kv_per_split = (seq_len + num_splits - 1) / num_splits;
    for (int s = 0; s < num_splits; s++) {
        const int ks = kv_per_split * s;
        if (min(ks + kv_per_split, seq_len) <= ks) continue;
        const int64_t sbase = base + int64_t(s) * stride_ms;
        const float tlogic = mid_o[sbase + D];
        const float nmax = fmaxf(tlogic, e_max);
        const float old_scale = tq_exp(e_max - nmax);
        const float el = tq_exp(tlogic - nmax);
        for (int i = 0; i < VPL; i++)
            acc[i] = __fmaf_rn(acc[i], old_scale,
                               __fmul_rn(el, mid_o[sbase + lane + 32 * i]));
        e_sum = __fmaf_rn(e_sum, old_scale, el);
        e_max = nmax;
    }
    const int64_t obase = int64_t(bid) * stride_ob + int64_t(hid) * stride_oh;
    for (int i = 0; i < VPL; i++)
        out[obase + lane + 32 * i] = TO(tq_div(acc[i], e_sum));
    if (lane == 0) lse[int64_t(bid) * stride_lb + hid] = e_max + logf(e_sum);
}

// ---- full K/V dequant to fp16 (_tq_full_dequant_kv) ----
// grid (positions, B * Hk), block 32; elementwise per (position, head).
__global__ void tq_full_dequant_kv(const uint8_t* __restrict__ cache,
                                   const int* __restrict__ block_table,
                                   const float* __restrict__ centroids,
                                   __half* __restrict__ k_out, __half* __restrict__ v_out,
                                   int64_t stride_ko_b, int64_t stride_ko_h, int64_t stride_ko_s,
                                   int64_t stride_vo_b, int64_t stride_vo_h, int64_t stride_vo_s,
                                   int64_t stride_cb, int64_t stride_cp, int64_t stride_ch,
                                   int64_t stride_bt,
                                   int D, int block_size, int num_kv_heads,
                                   int mse_bytes, int kps, int vqb, int val_data_bytes,
                                   int mse_bits, int key_fp8, int norm_correction) {
    const int pos = blockIdx.x, bh = blockIdx.y;
    const int bid = bh / num_kv_heads, hid = bh % num_kv_heads;
    const int64_t bnum = int64_t(block_table[int64_t(bid) * stride_bt + pos / block_size]);
    const int64_t slot_base = bnum * stride_cb + int64_t(pos % block_size) * stride_cp
                            + int64_t(hid) * stride_ch;
    const int lane = threadIdx.x;
    const int VPL = D / 32;
    const int64_t ko_base = int64_t(bid) * stride_ko_b + int64_t(hid) * stride_ko_h
                          + int64_t(pos) * stride_ko_s;
    const int64_t vo_base = int64_t(bid) * stride_vo_b + int64_t(hid) * stride_vo_h
                          + int64_t(pos) * stride_vo_s;
    if (key_fp8) {
        for (int i = 0; i < VPL; i++) {
            const int d = lane + 32 * i;
            k_out[ko_base + d] =
                __ushort_as_half(tq_e4b15_to_f16_bits(cache[slot_base + d]));
        }
    } else {
        // D/32 values per lane. D=512 needs all 16 entries; the original
        // four-entry scratch silently overran thread-local storage when the
        // native full-dequant path was used for DSV4 draft attention.
        float c[16];
        for (int i = 0; i < VPL; i++)
            c[i] = centroids[tq_mse_idx(cache, slot_base, mse_bits, lane + 32 * i)];
        if (norm_correction) {
            float nsq = __fmul_rn(c[0], c[0]);
            for (int i = 1; i < VPL; i++) nsq = __fmaf_rn(c[i], c[i], nsq);
            nsq = tq_warp_sum(nsq);
            const float inv = tq_div(1.0f, sqrtf(nsq + 1e-16f));
            for (int i = 0; i < VPL; i++) c[i] = __fmul_rn(c[i], inv);
        }
        const float vec_norm = tq_load_f16(cache + slot_base + mse_bytes);
        for (int i = 0; i < VPL; i++)
            k_out[ko_base + lane + 32 * i] = __float2half_rn(__fmul_rn(vec_norm, c[i]));
    }
    const int64_t vbase = slot_base + kps;
    float v_scale, v_zero;
    tq_load_value_sz(cache, vbase, val_data_bytes, v_scale, v_zero);
    for (int i = 0; i < VPL; i++) {
        const int d = lane + 32 * i;
        v_out[vo_base + d] = __float2half_rn(
            __fmaf_rn(tq_value_idx(cache, vbase, vqb, d), v_scale, v_zero));
    }
}

}  // namespace tms
