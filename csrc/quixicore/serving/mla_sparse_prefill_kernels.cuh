#pragma once
// Sparse NoPE-MLA attention for PREFILL chunks over the paged fp8 latent
// (GLM-5.3-Flash: 512-wide latent, e4m3 slots with one per-tensor scale,
// top-2048 keys per query selected as 4-token POOLS by the compact indexer).
//
// Motive (2026-09-12 prefill profile): with no prefill path the sparse
// backend ran the per-token DECODE kernel for prefill chunks - 39% of prefill
// time, every query streaming its own 2048 keys (~1 MB per token per layer).
// Consecutive queries of one request select heavily overlapping pools, so a
// group of GQ = 4 queries attends on tensor cores over the UNION of their
// pools with a per-query mask:
//
//   prep kernel  : per group, sort the (physical pool slot, query, token-bit)
//                  entries of its 4 index lists, merge into a union pool list
//                  and a 4-bit validity per (query, pool).
//   attn kernel  : CTA per group, 8 warps = 4 queries x 2 halves of the 512
//                  output dims; tiles of KT = 8 pools (32 keys) dequantized
//                  e4m3 -> bf16 into shared memory once (read by Q.K^T through
//                  ldmatrix and by P.V through ldmatrix.trans); the warp pair
//                  of a query splits the 512-dim Q.K^T reduction and swaps
//                  partial scores through shared memory; online softmax in
//                  fp32 (flash-attention-2 fragment layout); kv_scale is
//                  folded into q for the scores and applied to O at the end.
//   Semantics: attention over the SET of a query's selected keys. A token
//   listed twice (the indexer's always-selected tail can repeat a pool's
//   token) counts once here, twice in the per-token decode kernel.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace tms::sparse_prefill {

constexpr int GQ = 4;          // queries per group
constexpr int MAXU = 2112;     // max union pools per group (4 x 528 list slots)
constexpr int KT = 8;          // pools per key tile
constexpr int KEYS = KT * 4;   // 32 keys per tile
constexpr int D = 512;         // latent width
constexpr int LDK = D + 8;     // bf16 smem row stride (1040 B)
constexpr int THREADS = 256;

// ---------------------------------------------------------------- prep ----
// One block per group. entries = (phys_pool << 6) | (q << 4) | (1 << bit) packed
// so equal (pool, q) merge by OR of the low 4 bits after sorting.
template <int NLIST>
__global__ void __launch_bounds__(THREADS) sparse_prefill_prep_kernel(
        const int* __restrict__ indices,     // [T, W] logical token positions, -1 pad
        const int* __restrict__ topk_len,    // [T] valid prefix length of each list
        const int* __restrict__ bt,          // [T, max_blocks] block table per token
        int T, int W, int max_blocks, int block_size,
        int* __restrict__ pools,             // [G, MAXU] physical slot of each union pool's first token
        int* __restrict__ qmask,             // [G, GQ, MAXU] 4-bit token validity per (query, pool), zeroed by the caller
        int* __restrict__ counts) {          // [G]
    const int g = blockIdx.x;
    const int t0 = g * GQ;
    const int nq = min(GQ, T - t0);
    extern __shared__ unsigned keys_smem[];   // NLIST
    const int tid = threadIdx.x;
    // 1. gather keys, compacted to pool RUNS: the indexer lists a pool's 4
    //    tokens consecutively, so an entry emits a key only when it starts a
    //    run (its pool differs from the previous entry's); the run's token
    //    bits are gathered by scanning forward. Scattered lists degrade to
    //    one key per token (still correct). Keys are compacted with a block
    //    scan so the sort covers only the emitted keys.
    __shared__ int s_scan0[THREADS];
    __shared__ int s_nkeys;
    if (tid == 0) s_nkeys = 0;
    __syncthreads();
    const int total_entries = GQ * W;
    for (int chunk = 0; chunk < total_entries; chunk += THREADS) {
        const int e = chunk + tid;
        unsigned key = 0xFFFFFFFFu;
        bool emit = false;
        if (e < total_entries) {
            const int q = e / W, j = e - q * W;
            if (q < nq && j < topk_len[t0 + q]) {
                const int idx = indices[(size_t)(t0 + q) * W + j];
                if (idx >= 0) {
                    const int blk = idx / block_size, off = idx - blk * block_size;
                    if (blk < max_blocks) {
                        const int phys = bt[(size_t)(t0 + q) * max_blocks + blk] * block_size + off;
                        const int pool = phys >> 2;
                        bool start = true;
                        if (j > 0) {
                            const int pidx = indices[(size_t)(t0 + q) * W + j - 1];
                            if (pidx >= 0) {
                                const int pblk = pidx / block_size, poff = pidx - pblk * block_size;
                                if (pblk < max_blocks) {
                                    const int pphys = bt[(size_t)(t0 + q) * max_blocks + pblk] * block_size + poff;
                                    start = (pphys >> 2) != pool;
                                }
                            }
                        }
                        if (start) {
                            unsigned bits = 1u << (phys & 3);
                            for (int k = 1; k < 4 && j + k < topk_len[t0 + q]; ++k) {
                                const int nidx = indices[(size_t)(t0 + q) * W + j + k];
                                if (nidx < 0) break;
                                const int nblk = nidx / block_size, noff = nidx - nblk * block_size;
                                if (nblk >= max_blocks) break;
                                const int nphys = bt[(size_t)(t0 + q) * max_blocks + nblk] * block_size + noff;
                                if ((nphys >> 2) != pool) break;
                                bits |= 1u << (nphys & 3);
                            }
                            key = (unsigned(pool) << 6) | (unsigned(q) << 4) | bits;
                            emit = true;
                        }
                    }
                }
            }
        }
        // compact
        s_scan0[tid] = emit ? 1 : 0;
        __syncthreads();
        for (int off = 1; off < THREADS; off <<= 1) {
            const int add = (tid >= off) ? s_scan0[tid - off] : 0;
            __syncthreads();
            s_scan0[tid] += add;
            __syncthreads();
        }
        if (emit) {
            const int pos = s_nkeys + s_scan0[tid] - 1;
            if (pos < NLIST) keys_smem[pos] = key;
        }
        __syncthreads();
        if (tid == THREADS - 1) s_nkeys += s_scan0[tid];
        __syncthreads();
    }
    const int nkeys = min(s_nkeys, NLIST);
    // sort size: smallest power of two >= nkeys (sentinel-filled)
    int nsort = 1;
    while (nsort < nkeys) nsort <<= 1;
    for (int i = nkeys + tid; i < nsort; i += THREADS) keys_smem[i] = 0xFFFFFFFFu;
    __syncthreads();
    // 2. bitonic sort over nsort
    for (int k = 2; k <= nsort; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < nsort; i += THREADS) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const unsigned a = keys_smem[i], b = keys_smem[ixj];
                    const bool up = ((i & k) == 0);
                    if ((a > b) == up) { keys_smem[i] = b; keys_smem[ixj] = a; }
                }
            }
            __syncthreads();
        }
    }
    // 3. unique pools in parallel: flag pool starts, block-scan the flags for
    //    positions, write the pool list at the starts and OR every entry's
    //    token bits into its (query, pool) mask.
    __shared__ int s_scan[THREADS];
    __shared__ int s_base;
    if (tid == 0) s_base = 0;
    __syncthreads();
    for (int chunk = 0; chunk < nsort; chunk += THREADS) {
        const int i = chunk + tid;
        const unsigned key = (i < nsort) ? keys_smem[i] : 0xFFFFFFFFu;
        const bool valid = key != 0xFFFFFFFFu;
        const unsigned pool = key >> 6;
        const bool start = valid && (i == 0 || (keys_smem[i - 1] >> 6) != pool);
        // inclusive scan of `start` over the chunk
        int v = start ? 1 : 0;
        s_scan[tid] = v;
        __syncthreads();
        for (int off = 1; off < THREADS; off <<= 1) {
            const int add = (tid >= off) ? s_scan[tid - off] : 0;
            __syncthreads();
            s_scan[tid] += add;
            __syncthreads();
        }
        const int pos = s_base + s_scan[tid] - 1;   // position of this entry's pool (if start) ...
        __syncthreads();
        if (tid == THREADS - 1) s_base += s_scan[tid];
        // an entry that is not a start belongs to the most recent start at or before it:
        // its pool position = s_base_before + (number of starts at indices <= i) - 1 = pos as computed
        if (valid && pos >= 0 && pos < MAXU) {
            if (start) pools[(size_t)g * MAXU + pos] = int(pool << 2);
            const int q = (key >> 4) & 3;
            atomicOr(qmask + ((size_t)g * GQ + q) * MAXU + pos, int(key & 0xF));
        }
        __syncthreads();
    }
    if (tid == 0) counts[g] = min(s_base, MAXU);
}

// ------------------------------------------------------------ attention ----
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* smem) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}
__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<unsigned*>(&v);
}
// e4m3 byte -> bf16 bits: value * 2^0 exactly (exponent bias 7 -> 127: add 120 to the exponent field).
__device__ __forceinline__ __nv_bfloat16 e4m3_to_bf16(uint8_t b) {
    const unsigned s = (b & 0x80u) << 8;
    const unsigned em = b & 0x7Fu;
    unsigned bits;
    if (em == 0) bits = s;   // zero
    else if ((em & 0x78u) == 0) {   // subnormal: mant/8 * 2^-6
        const float f = float(em) * (1.0f / 512.0f);
        __nv_bfloat16 h = __float2bfloat16_rn(f);
        bits = (*reinterpret_cast<unsigned short*>(&h)) | s;
    } else {
        bits = s | (((em >> 3) + 120u) << 7) | ((em & 7u) << 4);
    }
    unsigned short u = (unsigned short)bits;
    return *reinterpret_cast<__nv_bfloat16*>(&u);
}

// smem: Q [GQ*16 rows][LDK] bf16 (66.5 KB) + K tile [KEYS][LDK] bf16 (33 KB) + S exchange [8 warps][16][KEYS] fp32 (16 KB)
constexpr int Q_ROWS = GQ * 16;
constexpr int SMEM_Q = Q_ROWS * LDK * 2;
constexpr int SMEM_K = KEYS * LDK * 2;
constexpr int SMEM_S = 8 * 16 * KEYS * 4;
constexpr int SMEM_TOTAL = SMEM_Q + SMEM_K + SMEM_S;

__global__ void __launch_bounds__(THREADS, 1) sparse_prefill_attn_kernel(
        const __nv_bfloat16* __restrict__ q,     // [T, H, 512] bf16
        const uint8_t* __restrict__ data,        // fp8 slots, 512 B each, slot s at data + s * 512
        const int* __restrict__ pools,           // [G, MAXU]
        const int* __restrict__ qmask,           // [G, GQ, MAXU]
        const int* __restrict__ counts,          // [G]
        __nv_bfloat16* __restrict__ out,         // [T, H, 512] bf16
        int T, int H, float q_scale /* softmax_scale * kv_scale * log2(e) */, float kv_scale,
        int block_size, int64_t page_stride_bytes /* bytes between consecutive pages of the (packed) cache */) {
    static_assert(GQ * 16 == 64, "8 warps = 4 queries x 2 dim halves at 16 heads");
    const int g = blockIdx.x;
    const int t0 = g * GQ;
    const int nq = min(GQ, T - t0);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int mq = warp & 3;         // query within group (its 16 heads = 16 rows)
    const int half = warp >> 2;      // output dims [256*half, +256), Q.K^T dims likewise
    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* qs = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* ks = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_Q);
    float* sx = reinterpret_cast<float*>(smem_raw + SMEM_Q + SMEM_K);   // [8][16][KEYS]
    if (H != 16) return;   // this build covers the TP4 head count
    // Q -> smem (rows: query m, head h -> row 16*m + h), pre-scaled by q_scale.
    for (int e = tid; e < Q_ROWS * (D / 8); e += THREADS) {
        const int r = e / (D / 8), c8 = (e - r * (D / 8)) * 8;
        const int m = r >> 4, h = r & 15;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (m < nq) v = *reinterpret_cast<const uint4*>(q + ((size_t)(t0 + m) * H + h) * D + c8);
        const __nv_bfloat16* pb = reinterpret_cast<const __nv_bfloat16*>(&v);
        uint4 o; __nv_bfloat16* po = reinterpret_cast<__nv_bfloat16*>(&o);
#pragma unroll
        for (int k = 0; k < 8; ++k) po[k] = __float2bfloat16_rn(__bfloat162float(pb[k]) * q_scale);
        *reinterpret_cast<uint4*>(qs + r * LDK + c8) = o;
    }
    __syncthreads();
    const int npools = counts[g];
    const int ntiles = (npools + KT - 1) / KT;
    // per-thread accumulators: C fragments for 32 n-tiles of 8 dims (256 dims) x 1 m-tile
    float o[32][4];
#pragma unroll
    for (int j = 0; j < 32; ++j) { o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f; }
    float m_row[2] = {-1e30f, -1e30f}, l_row[2] = {0.0f, 0.0f};   // rows g and g+8 of this warp's m-tile
    const int gr = lane >> 2, t2 = (lane & 3) * 2;
    for (int tile = 0; tile < ntiles; ++tile) {
        // ---- stage K tile: 8 pools x 4 slots x 512 B fp8 -> bf16 rows [key][dim]
        __syncthreads();   // previous tile fully consumed
        for (int e = tid; e < KEYS * (D / 16); e += THREADS) {
            const int key = e / (D / 16), c16 = (e - key * (D / 16)) * 16;
            const int p = tile * KT + (key >> 2);
            uint4 raw = make_uint4(0, 0, 0, 0);
            if (p < npools) {
                const int slot = pools[(size_t)g * MAXU + p] + (key & 3);
                const int pg = slot / block_size, off = slot - pg * block_size;
                raw = *reinterpret_cast<const uint4*>(data + (size_t)pg * page_stride_bytes + (size_t)off * D + c16);
            }
            const uint8_t* rb = reinterpret_cast<const uint8_t*>(&raw);
            uint4 o0, o1;
            __nv_bfloat16* p0 = reinterpret_cast<__nv_bfloat16*>(&o0);
            __nv_bfloat16* p1 = reinterpret_cast<__nv_bfloat16*>(&o1);
#pragma unroll
            for (int k = 0; k < 8; ++k) { p0[k] = e4m3_to_bf16(rb[k]); p1[k] = e4m3_to_bf16(rb[8 + k]); }
            *reinterpret_cast<uint4*>(ks + key * LDK + c16) = o0;
            *reinterpret_cast<uint4*>(ks + key * LDK + c16 + 8) = o1;
        }
        __syncthreads();
        // ---- partial S over this warp's 256 dims: 4 n-tiles (32 keys), 16 k16 steps
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f; }
#pragma unroll 4
        for (int kk = 0; kk < 16; ++kk) {
            unsigned a[4];
            {
                const int r = mq * 16 + (lane & 15);
                const int c = half * 256 + kk * 16 + (lane >> 4) * 8;
                ldmatrix_x4(a, qs + r * LDK + c);
            }
#pragma unroll
            for (int j = 0; j < 4; j += 2) {
                unsigned b[4];
                const int qd = lane >> 3;
                const int r = j * 8 + (qd >> 1) * 8 + (lane & 7);
                const int c = half * 256 + kk * 16 + (qd & 1) * 8;
                ldmatrix_x4(b, ks + r * LDK + c);
                mma_bf16(s[j], a, b[0], b[1]);
                mma_bf16(s[j + 1], a, b[2], b[3]);
            }
        }
        // ---- exchange partials with the sibling warp (same query, other half)
        float* mine = sx + (warp * 16) * KEYS;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = j * 8 + t2;
            mine[gr * KEYS + c] = s[j][0]; mine[gr * KEYS + c + 1] = s[j][1];
            mine[(gr + 8) * KEYS + c] = s[j][2]; mine[(gr + 8) * KEYS + c + 1] = s[j][3];
        }
        __syncthreads();
        const float* other = sx + ((warp ^ 4) * 16) * KEYS;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = j * 8 + t2;
            s[j][0] += other[gr * KEYS + c]; s[j][1] += other[gr * KEYS + c + 1];
            s[j][2] += other[(gr + 8) * KEYS + c]; s[j][3] += other[(gr + 8) * KEYS + c + 1];
        }
        // ---- mask (per query pool validity), online softmax (base 2: q_scale carries log2 e)
        float tmax[2] = {-1e30f, -1e30f};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const int key = j * 8 + t2 + v;
                const int p = tile * KT + (key >> 2);
                bool ok = (p < npools) && (mq < nq);
                if (ok) {
                    const int mk = qmask[((size_t)g * GQ + mq) * MAXU + p];
                    ok = (mk >> (key & 3)) & 1;
                }
                if (!ok) { s[j][v] = -1e30f; s[j][2 + v] = -1e30f; }
                tmax[0] = fmaxf(tmax[0], s[j][v]); tmax[1] = fmaxf(tmax[1], s[j][2 + v]);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 1));
            tmax[r] = fmaxf(tmax[r], __shfl_xor_sync(0xffffffffu, tmax[r], 2));
        }
        float alpha[2], rsum[2] = {0.0f, 0.0f};
        float mnew[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mnew[r] = fmaxf(m_row[r], tmax[r]);
            alpha[r] = exp2f(m_row[r] - mnew[r]);
        }
        unsigned pa[2][4];   // P as A fragments for two k16 steps (keys 0-15, 16-31)
        const bool live0 = mnew[0] > -1e29f, live1 = mnew[1] > -1e29f;   // any valid key so far
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float p0 = live0 ? exp2f(s[j][0] - mnew[0]) : 0.0f, p1 = live0 ? exp2f(s[j][1] - mnew[0]) : 0.0f;
            float p2 = live1 ? exp2f(s[j][2] - mnew[1]) : 0.0f, p3 = live1 ? exp2f(s[j][3] - mnew[1]) : 0.0f;
            rsum[0] += p0 + p1; rsum[1] += p2 + p3;
            // C-fragment (rows gr / gr+8, cols j*8+t2..+1) -> A fragment of k16 step j/2:
            // a0 = (row gr, k 0-7 region), a1 = (row gr+8, k 0-7), a2 = (row gr, k 8-15), a3 = (row gr+8, k 8-15)
            const int step = j >> 1, hi = j & 1;
            pa[step][hi * 2 + 0] = pack_bf16x2(p0, p1);
            pa[step][hi * 2 + 1] = pack_bf16x2(p2, p3);
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            rsum[r] += __shfl_xor_sync(0xffffffffu, rsum[r], 1);
            rsum[r] += __shfl_xor_sync(0xffffffffu, rsum[r], 2);
            l_row[r] = l_row[r] * alpha[r] + rsum[r];
            m_row[r] = mnew[r];
        }
        // rescale O
#pragma unroll
        for (int j = 0; j < 32; ++j) { o[j][0] *= alpha[0]; o[j][1] *= alpha[0]; o[j][2] *= alpha[1]; o[j][3] *= alpha[1]; }
        // ---- O += P.V : B = V^T from [key][dim] via ldmatrix.trans; n-tiles = 32 x 8 dims of this half
        // Fix the A-fragment ordering: mma A regs are {a0: rows 0-7 k0-7, a1: rows 8-15 k0-7, a2: rows 0-7 k8-15, a3: rows 8-15 k8-15}
        unsigned afr[2][4];
#pragma unroll
        for (int st = 0; st < 2; ++st) { afr[st][0] = pa[st][0]; afr[st][1] = pa[st][1]; afr[st][2] = pa[st][2]; afr[st][3] = pa[st][3]; }
#pragma unroll
        for (int st = 0; st < 2; ++st) {
#pragma unroll
            for (int j = 0; j < 32; j += 2) {
                unsigned b[4];
                // trans load: 4 8x8 matrices covering keys [16*st, +16) x dims [8*j, +16)
                const int qd = lane >> 3;
                const int key = st * 16 + (qd & 1) * 8 + (lane & 7);
                const int dim = half * 256 + j * 8 + (qd >> 1) * 8;
                ldmatrix_x4_trans(b, ks + key * LDK + dim);
                mma_bf16(o[j], afr[st], b[0], b[1]);
                mma_bf16(o[j + 1], afr[st], b[2], b[3]);
            }
        }
    }
    // ---- epilogue: O / l * kv_scale -> out[t0+mq][head=row][dims]
    if (mq >= nq) return;
    const float inv0 = (l_row[0] > 0.0f) ? kv_scale / l_row[0] : 0.0f;
    const float inv1 = (l_row[1] > 0.0f) ? kv_scale / l_row[1] : 0.0f;
#pragma unroll
    for (int j = 0; j < 32; ++j) {
        const int dim = half * 256 + j * 8 + t2;
        __nv_bfloat16* o0 = out + ((size_t)(t0 + mq) * H + gr) * D + dim;
        __nv_bfloat16* o1 = out + ((size_t)(t0 + mq) * H + gr + 8) * D + dim;
        *reinterpret_cast<__nv_bfloat162*>(o0) = __floats2bfloat162_rn(o[j][0] * inv0, o[j][1] * inv0);
        *reinterpret_cast<__nv_bfloat162*>(o1) = __floats2bfloat162_rn(o[j][2] * inv1, o[j][3] * inv1);
    }
}

}  // namespace tms::sparse_prefill
