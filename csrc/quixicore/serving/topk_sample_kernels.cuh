#pragma once
// Fused top-k / top-p / Gumbel-max sampling for the V2 sampler, for requests
// whose top_k is at most 32 (the candidate window K). Four launches replace
// the Triton top-k/top-p mask pass plus the Gumbel argmax pass:
//
//   1. candidates_kernel:        NB blocks per row, each radix-selects its
//                                slice's top K (value, id) and counts the ties
//                                it had to omit at its local threshold;
//   2. cutoff_kernel:            the global k-th value from the NB*K
//                                candidates, every omitted local tie accounted
//                                for, then the nucleus cutoff and its
//                                deterministic token-id boundary;
//   3. sample_partitions_kernel: every retained vocabulary token, including
//                                ties that fit in neither window, races with
//                                the sampler's own (seed, pos, token)-keyed
//                                Philox noise; one winner per partition;
//   4. finish_kernel:            the argmax over the partitions.
//
// The candidate window only finds the threshold; it never caps the retained
// mass or the sampled ids: there are at most k-1 values strictly above the
// global k-th value and every such value survives a local top-32.
//
// Semantics. Top-k retains all ties at the k-th value. Top-p orders the
// retained tokens ascending by (logit, id) and drops the prefix whose
// cumulative mass is <= 1-p, always keeping at least one token; this defined
// tie order replaces torch.sort's unstable equal-key order. The sample is the
// exponential race argmax_i p_i / E_i with E_i = -log(1 - U_i) (fp32) or
// -log(U_i) (fp64), U_i the uniform the Gumbel path draws for the same
// (seed, pos, token): the same distribution as argmax_i (logit_i + G_i) with
// G_i = -log(E_i), and the same per-request stream, so seeded requests stay
// reproducible across boots.
//
// Preconditions: 1 <= top_k <= K, 0 <= top_p <= 1, no NaN or +inf logits and
// at least one finite logit per row. -inf masks and signed zeros are fine.
// Rows that violate this (the sampler warmups feed uninitialized logits) have
// unspecified sampling semantics but always return an in-range id.
#include "v2_sample_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <cfloat>
#include <climits>

namespace tmv2s::topk_sample {

constexpr int K = 32;               // candidate window; every request's top_k <= K
constexpr int NB = 16;              // blocks per row in the candidate pass
constexpr int THREADS = 1024;       // candidate pass block
constexpr int MERGE_THREADS = 512;  // merge pass block = NB * K candidates
constexpr int RADIX_BITS = 11;
constexpr int BINS = 1 << RADIX_BITS;

// Order-preserving key: larger float -> larger key. NaN is not expected.
__device__ __forceinline__ uint32_t key_of(float f) {
    if (f == 0.0f) return 0x80000000u;  // Canonicalize signed zeros.
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float value_of(uint32_t key) {
    return __uint_as_float((key & 0x80000000u) ? (key & 0x7fffffffu) : ~key);
}

struct RowCutoff {
    float maximum;
    float value;
    int first_id;   // First retained token ID when the cutoff is above kth.
    int drop_ties;  // Number of lowest-ID kth ties to drop, or -1 for first_id.
};

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
    const float* __restrict__ logits, int64_t row_stride, int V,
    float* __restrict__ cand_val, int* __restrict__ cand_idx,
    float* __restrict__ part_threshold, int* __restrict__ part_omitted) {
    __shared__ uint32_t hist[BINS];
    __shared__ int need_sh, scan_warp[THREADS / 32], scan_out[2], out_count, tie_taken;
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    const int chunk = (V + NB - 1) / NB;
    const int start = part * chunk, end = min(V, start + chunk), n = max(0, end - start);
    const float* x = logits + int64_t(row) * row_stride + start;
    float* out_v = cand_val + (int64_t(row) * NB + part) * K;
    int* out_i = cand_idx + (int64_t(row) * NB + part) * K;
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
    if (tid == 0) {
        part_threshold[int64_t(row) * NB + part] = value_of(thresh);
        part_omitted[int64_t(row) * NB + part] = tie_taken - need;
    }
}

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// Pass 2: find the global kth value. Account for every omitted local tie,
// then determine the nucleus cutoff and its deterministic token-ID boundary.
__global__ void __launch_bounds__(MERGE_THREADS) cutoff_kernel(
    const float* __restrict__ cand_val, const int* __restrict__ cand_idx,
    const float* __restrict__ part_threshold, const int* __restrict__ part_omitted,
    const int* __restrict__ top_k, const float* __restrict__ top_p,
    RowCutoff* __restrict__ cutoffs, int* __restrict__ tie_prefix) {
    constexpr int N = NB * K;
    static_assert(N == MERGE_THREADS, "one candidate per thread");
    __shared__ uint32_t hist[BINS];
    __shared__ float sv[N];
    __shared__ int si[N];
    __shared__ float tv[K];
    __shared__ int ti[K];
    __shared__ float kth_sh;
    __shared__ int part_ties[NB];
    __shared__ int need_sh, scan_warp[MERGE_THREADS / 32], scan_out[2], out_count, tie_taken;
    const int row = blockIdx.x, tid = threadIdx.x, lane = tid & 31;
    sv[tid] = cand_val[int64_t(row) * N + tid];
    si[tid] = cand_idx[int64_t(row) * N + tid];
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
    if (tid < K) {
        // Descending (value, ID), the reverse of the documented ascending
        // nucleus ordering. All values strictly above kth are present.
        float v = tv[lane];
        int idx = ti[lane];
        for (int size = 2; size <= K; size <<= 1) {
            for (int stride = size >> 1; stride > 0; stride >>= 1) {
                const float ov = __shfl_xor_sync(0xffffffffu, v, stride);
                const int oi = __shfl_xor_sync(0xffffffffu, idx, stride);
                const bool greater = ov > v || (ov == v && oi > idx);
                const bool less = ov < v || (ov == v && oi < idx);
                const bool upper = (lane & stride) != 0;
                const bool descending = (lane & size) == 0;
                if (descending ? (upper ? less : greater) : (upper ? greater : less)) {
                    v = ov; idx = oi;
                }
            }
        }
        tv[lane] = v; ti[lane] = idx;
        const int k = min(max(top_k[row], 1), K);
        const float kth = __shfl_sync(0xffffffffu, v, k - 1);
        if (lane == 0) kth_sh = kth;
    }
    __syncthreads();
    // Each warp owns exactly one original partition's 32 candidate slots.
    const int part = tid / K;
    const unsigned eq = __ballot_sync(0xffffffffu, si[tid] >= 0 && sv[tid] == kth_sh);
    if (lane == 0) {
        part_ties[part] = __popc(eq) +
            (part_threshold[int64_t(row) * NB + part] == kth_sh
                 ? part_omitted[int64_t(row) * NB + part] : 0);
    }
    __syncthreads();
    if (tid >= K) return;

    int count = lane < NB ? part_ties[lane] : 0;
    int inclusive = count;
#pragma unroll
    for (int offset = 1; offset < K; offset <<= 1) {
        const int other = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += other;
    }
    if (lane < NB) tie_prefix[int64_t(row) * NB + lane] = inclusive - count;
    const int ties = __shfl_sync(0xffffffffu, inclusive, K - 1);
    const float v = tv[lane], kth = kth_sh, maximum = tv[0];
    if (ties <= 0 || !isfinite(maximum)) {
        // No probability distribution exists for this row. Avoid NaN-to-int
        // conversion and undefined cutoff arithmetic. Pass 3's empty-set
        // argmax returns token zero, safe for dummy startup consumers.
        if (lane == 0) cutoffs[row] = {0.0f, __int_as_float(0x7fffffff), 0, -1};
        return;
    }
    const bool above = v > kth;
    const int n_above = __popc(__ballot_sync(0xffffffffu, above));
    // Only this one-warp cutoff calculation needs FP64. With unbounded
    // ties, rounding 1-p or a cumulative probability in FP32 can move the
    // ID boundary by a whole token even for a uniform distribution.
    const double e = above ? exp(double(v) - double(maximum)) : 0.0;
    const double tie_e = exp(double(kth) - double(maximum));
    const double tie_mass = double(ties) * tie_e;
    const double z = warp_sum(e) + tie_mass;
    const double p = top_p ? double(top_p[row]) : 1.0;
    const double budget = (1.0 - p) * z;
    int drop = tie_e > 0.0
        ? int(fmin(double(ties), floor(budget / tie_e))) : ties;
    drop = min(max(drop, 0), ties);
    if (n_above == 0) drop = min(drop, ties - 1);  // Always retain a maximum.

    RowCutoff result;
    result.maximum = maximum;
    if (drop < ties) {
        result.value = kth;
        result.drop_ties = drop;
        result.first_id = 0;
        if (n_above + ties <= K) {
            // Common case: all cutoff IDs fit in the window. Avoid a full
            // vocabulary prefix scan just to decide a tiny tie group.
            const int last_kept = n_above + ties - drop - 1;
            result.first_id = __shfl_sync(0xffffffffu, ti[lane], last_kept);
            result.drop_ties = -1;
        }
    } else {
        double asc = e;
#pragma unroll
        for (int offset = 1; offset < K; offset <<= 1) {
            const double other = __shfl_down_sync(0xffffffffu, asc, offset);
            if (lane + offset < K) asc += other;
        }
        const bool keep = above && (lane == 0 || (p > 0.0 && asc + tie_mass > budget));
        const unsigned kept = __ballot_sync(0xffffffffu, keep);
        const int last_kept = K - 1 - __clz(kept);
        result.value = __shfl_sync(0xffffffffu, v, last_kept);
        result.first_id = __shfl_sync(0xffffffffu, ti[lane], last_kept);
        result.drop_ties = -1;
    }
    if (lane == 0) cutoffs[row] = result;
}

template <typename Score>
__device__ __forceinline__ void warp_argmax(Score& score, int& id) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        const Score other = __shfl_down_sync(0xffffffffu, score, offset);
        const int other_id = __shfl_down_sync(0xffffffffu, id, offset);
        if ((threadIdx.x & 31) + offset < 32 &&
            (other > score || (other == score && other_id < id))) {
            score = other; id = other_id;
        }
    }
}

// The exponential variate behind the Gumbel path's noise for one token:
// gumbel.py draws U and adds -log(-log(1 - U)) (fp32; -log(-log U) in fp64),
// so p / E with E = -log(1 - U) (resp. -log U) ranks the tokens identically.
template <typename Score>
__device__ __forceinline__ Score race_noise(uint32_t gumbel_seed, int id) {
    if constexpr (sizeof(Score) == 8) {
        return -log(tt_rand64_nz(uint64_t(gumbel_seed), uint64_t(uint32_t(id))));
    } else {
        return -log1pf(-tt_rand_nz(uint64_t(gumbel_seed), uint64_t(uint32_t(id))));
    }
}

constexpr int SAMPLE_THREADS = 256;

// Pass 3: sample every retained vocabulary token, including ties which did
// not fit in either candidate window. A full tie prefix is needed only when
// the nucleus partially drops a tie group larger than the window.
template <typename Score>
__global__ void __launch_bounds__(SAMPLE_THREADS) sample_partitions_kernel(
    const float* __restrict__ logits, int64_t row_stride, int V,
    const int32_t* __restrict__ expanded_idx_mapping,
    const int64_t* __restrict__ seeds, const int64_t* __restrict__ pos_ptr,
    const RowCutoff* __restrict__ cutoffs, const int* __restrict__ tie_prefix,
    Score* __restrict__ part_score, int* __restrict__ part_id) {
    constexpr int WARPS = SAMPLE_THREADS / 32;
    __shared__ int scan[WARPS];
    __shared__ Score scores[WARPS];
    __shared__ int ids[WARPS];
    const int row = blockIdx.y, part = blockIdx.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid / 32;
    const int chunk = (V + NB - 1) / NB;
    const int start = part * chunk, end = min(V, start + chunk);
    const RowCutoff cutoff = cutoffs[row];
    // The request's stream position, exactly as v2_gumbel_sample_k keys it.
    const int64_t req = expanded_idx_mapping[row];
    const int64_t seed = req >= 0 ? seeds[req] : 0;
    const uint32_t gumbel_seed = tt_randint(uint64_t(seed), uint64_t(pos_ptr[row]));
    int preceding = tie_prefix[int64_t(row) * NB + part];
    Score best = Score(-1);
    int best_id = min(start + tid, V - 1);
    for (int base = start; base < end; base += SAMPLE_THREADS) {
        const int id = base + tid;
        const bool valid = id < end;
        const float v = valid ? logits[int64_t(row) * row_stride + id] : -INFINITY;
        bool keep = valid && (v > cutoff.value ||
            (v == cutoff.value && (cutoff.drop_ties >= 0 || id >= cutoff.first_id)));
        if (cutoff.drop_ties > 0) {
            const unsigned eq = __ballot_sync(0xffffffffu, valid && v == cutoff.value);
            const int local_rank = __popc(eq & ((1u << lane) - 1u));
            if (lane == 0) scan[warp] = __popc(eq);
            __syncthreads();
            if (warp == 0) {
                int incl = lane < WARPS ? scan[lane] : 0;
#pragma unroll
                for (int offset = 1; offset < WARPS; offset <<= 1) {
                    const int other = __shfl_up_sync(0xffffffffu, incl, offset);
                    if (lane >= offset) incl += other;
                }
                if (lane < WARPS) scan[lane] = incl;
            }
            __syncthreads();
            const int rank = preceding + (warp ? scan[warp - 1] : 0) + local_rank;
            if (v == cutoff.value && rank < cutoff.drop_ties) keep = false;
            preceding += scan[WARPS - 1];
            __syncthreads();
        }
        // Normalizing probabilities again would multiply every score by
        // the same positive scalar and cannot change the sampled token.
        const Score score = keep
            ? Score(__expf(v - cutoff.maximum)) / race_noise<Score>(gumbel_seed, id)
            : Score(-1);
        if (score > best || (score == best && id < best_id)) {
            best = score; best_id = id;
        }
    }
    warp_argmax(best, best_id);
    if (lane == 0) { scores[warp] = best; ids[warp] = best_id; }
    __syncthreads();
    if (warp == 0) {
        best = lane < WARPS ? scores[lane] : Score(-1);
        best_id = lane < WARPS ? ids[lane] : INT_MAX;
        warp_argmax(best, best_id);
        if (lane == 0) {
            part_score[int64_t(row) * NB + part] = best;
            part_id[int64_t(row) * NB + part] = best_id;
        }
    }
}

template <typename Score>
__global__ void finish_kernel(const Score* part_score, const int* part_id, int64_t* out) {
    const int row = blockIdx.x, lane = threadIdx.x;
    Score best = lane < NB ? part_score[int64_t(row) * NB + lane] : Score(-1);
    int id = lane < NB ? part_id[int64_t(row) * NB + lane] : INT_MAX;
    warp_argmax(best, id);
    if (lane == 0) out[row] = int64_t(id);
}

}  // namespace tmv2s::topk_sample
