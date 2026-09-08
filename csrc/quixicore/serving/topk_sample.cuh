// Small-k sampling on fp32 logits rows: top-k (k <= 32) with ties at the k-th
// value kept, top-p over the kept set (drop from the smallest while the
// cumulative mass at or below a token is <= 1 - p, the largest always kept),
// softmax over what is left, then the exponential-noise argmax. The masks are
// those of vLLM's apply_top_k_top_p_pytorch; the noise comes from the caller
// (torch's seeded generators), one draw per candidate.
//
// Two launches: (1) NB blocks per row each radix-select the top K of their
// slice of the vocabulary (shared-memory 11-bit histograms, 3 passes) and
// write K candidates; (2) one block per row radix-selects the top K of the
// NB*K candidates, sorts them in one warp, applies the masks and samples.
// Written for the GLM-5.3-Flash vocabulary (154880) where the fused Triton
// top-k/top-p kernel runs one program per row (124 us) plus a full-vocabulary
// softmax (48 us) per decode step.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>

namespace tms::topk_sample {

constexpr int K = 32;               // candidate window; every request's top_k <= K
constexpr int NB = 16;              // blocks per row in the candidate pass
constexpr int THREADS = 1024;       // candidate pass block
constexpr int MERGE_THREADS = 512;  // merge pass block = NB * K candidates
constexpr int RADIX_BITS = 11;
constexpr int BINS = 1 << RADIX_BITS;

// Order-preserving key: larger float -> larger key. NaN is not expected.
__device__ __forceinline__ uint32_t key_of(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Bits of `key` selected by pass p (0..2): [31:21], [20:10], [9:0].
__device__ __forceinline__ int pass_shift(int p) { return p == 0 ? 21 : (p == 1 ? 10 : 0); }
__device__ __forceinline__ int pass_bits(int p) { return p == 2 ? 10 : RADIX_BITS; }

// Block-wide: hist[BINS] holds counts; find the highest bin b such that the
// count of elements in bins above b is < need <= that count + hist[b].
// Returns b and leaves in *need the number still to take from bin b.
// Bins are scanned from the top with an exclusive prefix over reversed order.
template <int NT>
__device__ __forceinline__ int select_bin(uint32_t* hist, int* need_io, int nbins,
                                          int* scan_warp /* NT/32 */, int* scan_out) {
    constexpr int ITEMS_MAX = BINS / NT;  // bins per thread in the widest pass
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int items = nbins / NT;
    // Thread t owns reversed bins [t*items, (t+1)*items): reversed r -> bin nbins-1-r.
    uint32_t local[ITEMS_MAX];
    uint32_t sum = 0;
#pragma unroll
    for (int i = 0; i < ITEMS_MAX; ++i) {
        if (i < items) { local[i] = hist[nbins - 1 - (tid * items + i)]; sum += local[i]; }
    }
    // Inclusive warp scan of thread sums.
    uint32_t incl = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const uint32_t v = __shfl_up_sync(0xffffffffu, incl, o);
        if (lane >= o) incl += v;
    }
    if (lane == 31) scan_warp[warp] = int(incl);
    __syncthreads();
    if (warp == 0) {
        int v = lane < NT / 32 ? scan_warp[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, v, o);
            if (lane >= o) v += u;
        }
        if (lane < NT / 32) scan_warp[lane] = v;  // inclusive over warps
    }
    __syncthreads();
    const uint32_t warp_excl = warp == 0 ? 0u : uint32_t(scan_warp[warp - 1]);
    uint32_t above = warp_excl + incl - sum;  // elements in reversed bins before this thread's first
    const int need = *need_io;
#pragma unroll
    for (int i = 0; i < ITEMS_MAX; ++i) {
        if (i < items) {
            const uint32_t here = local[i];
            if (above < uint32_t(need) && uint32_t(need) <= above + here) {
                scan_out[0] = nbins - 1 - (tid * items + i);
                scan_out[1] = need - int(above);
            }
            above += here;
        }
    }
    __syncthreads();
    const int bin = scan_out[0];
    if (tid == 0) *need_io = scan_out[1];
    __syncthreads();
    return bin;
}

// Radix-select the K-th largest key among the elements yielded by acc(i) for
// i in [0, n) (threads stride over i). Returns the threshold key; *need_out is
// how many elements equal to it belong to the top K (ties beyond are cut).
template <int NT, typename Acc>
__device__ __forceinline__ uint32_t select_threshold(Acc acc, int n, int k, uint32_t* hist,
                                                     int* need_sh, int* scan_warp, int* scan_out) {
    const int tid = threadIdx.x;
    uint32_t prefix = 0;  // selected high bits so far
    int prefix_bits = 0;
    if (tid == 0) *need_sh = k;
    for (int p = 0; p < 3; ++p) {
        const int shift = pass_shift(p), bits = pass_bits(p), nbins = 1 << bits;
        for (int b = tid; b < nbins; b += NT) hist[b] = 0;
        __syncthreads();
        for (int i = tid; i < n; i += NT) {
            const uint32_t key = acc(i);
            if (prefix_bits == 0 || (key >> (shift + bits)) == prefix) {
                atomicAdd(&hist[(key >> shift) & (nbins - 1)], 1u);
            }
        }
        __syncthreads();
        const int bin = select_bin<NT>(hist, need_sh, nbins, scan_warp, scan_out);
        prefix = (prefix << bits) | uint32_t(bin);
        prefix_bits += bits;
        __syncthreads();
    }
    return prefix;
}

// Pass 1: NB blocks per row, each writes its slice's top K (value, index) into
// cand_val / cand_idx [B, NB*K]. Slots past the slice's element count (never,
// for V >= NB*K) are -inf / -1.
__global__ void __launch_bounds__(THREADS) candidates_kernel(
    const float* __restrict__ logits, long row_stride, int V,
    float* __restrict__ cand_val, int* __restrict__ cand_idx) {
    __shared__ uint32_t hist[BINS];
    __shared__ int need_sh, scan_warp[THREADS / 32], scan_out[2], out_count, tie_taken;
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    const int chunk = (V + NB - 1) / NB;
    const int start = part * chunk, end = min(V, start + chunk), n = max(0, end - start);
    const float* x = logits + long(row) * row_stride + start;
    float* out_v = cand_val + (long(row) * NB + part) * K;
    int* out_i = cand_idx + (long(row) * NB + part) * K;
    if (tid == 0) { out_count = 0; tie_taken = 0; }
    const int k = min(K, n);
    uint32_t thresh = 0;
    if (k > 0) {
        thresh = select_threshold<THREADS>([&](int i) { return key_of(x[i]); }, n, k, hist,
                                           &need_sh, scan_warp, scan_out);
    }
    __syncthreads();
    const int need = need_sh;
    for (int i = tid; i < n; i += THREADS) {
        const float v = x[i];
        const uint32_t key = key_of(v);
        int slot = -1;
        if (key > thresh) {
            slot = atomicAdd(&out_count, 1);
        } else if (key == thresh) {
            if (atomicAdd(&tie_taken, 1) < need) slot = atomicAdd(&out_count, 1);
        }
        if (slot >= 0 && slot < K) { out_v[slot] = v; out_i[slot] = start + i; }
    }
    __syncthreads();
    for (int s = out_count + tid; s < K; s += THREADS) { out_v[s] = -INFINITY; out_i[s] = -1; }
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Pass 2: one block per row. Top K of the NB*K candidates, sorted descending
// in warp 0, then the masks and the sample.
__global__ void __launch_bounds__(MERGE_THREADS) merge_sample_kernel(
    const float* __restrict__ cand_val, const int* __restrict__ cand_idx,
    const int* __restrict__ top_k, const float* __restrict__ top_p,
    const float* __restrict__ noise, long* __restrict__ out) {
    constexpr int N = NB * K;
    static_assert(N == MERGE_THREADS, "one candidate per thread");
    __shared__ uint32_t hist[BINS];
    __shared__ float sv[N];
    __shared__ int si[N];
    __shared__ float tv[K];
    __shared__ int ti[K];
    __shared__ int need_sh, scan_warp[MERGE_THREADS / 32], scan_out[2], out_count, tie_taken;
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    sv[tid] = cand_val[long(row) * N + tid];
    si[tid] = cand_idx[long(row) * N + tid];
    if (tid == 0) { out_count = 0; tie_taken = 0; }
    __syncthreads();
    const uint32_t thresh = select_threshold<MERGE_THREADS>(
        [&](int i) { return key_of(sv[i]); }, N, K, hist, &need_sh, scan_warp, scan_out);
    __syncthreads();
    {
        const uint32_t key = key_of(sv[tid]);
        int slot = -1;
        if (key > thresh) slot = atomicAdd(&out_count, 1);
        else if (key == thresh && atomicAdd(&tie_taken, 1) < need_sh) slot = atomicAdd(&out_count, 1);
        if (slot >= 0 && slot < K) { tv[slot] = sv[tid]; ti[slot] = si[tid]; }
    }
    __syncthreads();
    if (tid >= 32) return;
    // Warp 0: bitonic sort of the K = 32 candidates, descending by value.
    float v = tv[lane];
    int idx = ti[lane];
    for (int size = 2; size <= 32; size <<= 1) {
        for (int stride = size >> 1; stride > 0; stride >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, stride);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, stride);
            const bool upper = (lane & stride) != 0;         // this lane is the higher index of the pair
            const bool descending = ((lane & size) == 0);    // block direction
            // Descending block: the lower index keeps the larger value, the
            // upper index the smaller; ascending block: the reverse.
            const bool take_other = descending ? (upper ? ov < v : ov > v) : (upper ? ov > v : ov < v);
            if (take_other) { v = ov; idx = oi; }
        }
    }
    // Now v is sorted descending across lanes 0..31.
    const int k = min(max(top_k[row], 1), K);
    const float kth = __shfl_sync(0xffffffffu, v, k - 1);
    bool keep = (lane < k) || (v >= kth);
    if (top_p != nullptr) {
        const float p = top_p[row];
        const float m = __shfl_sync(0xffffffffu, v, 0);  // lane 0 is the row maximum and always kept
        float e = keep ? __expf(v - m) : 0.0f;
        const float z = warp_sum(e);
        const float prob = e / z;
        // S_j = mass at or below this token = sum over lanes >= j, summed from
        // the smallest token up like the reference's ascending cumsum.
        float asc = prob;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const float u = __shfl_down_sync(0xffffffffu, asc, o);
            if (lane + o < 32) asc += u;
        }
        const bool drop = (asc <= 1.0f - p) && lane > 0;
        keep = keep && !drop;
    }
    const float m2 = __shfl_sync(0xffffffffu, v, 0);
    const float e2 = keep ? __expf(v - m2) : 0.0f;
    const float z2 = warp_sum(e2);
    const float score = keep ? (e2 / z2) / noise[long(row) * K + lane] : -1.0f;
    const float best = warp_max(score);
    const unsigned ballot = __ballot_sync(0xffffffffu, score == best);
    const int winner = ballot ? __ffs(ballot) - 1 : 0;  // lane 0 is always kept
    const int token = __shfl_sync(0xffffffffu, idx, winner);  // all lanes take part
    if (lane == 0) out[row] = long(token);
}

}  // namespace tms::topk_sample
