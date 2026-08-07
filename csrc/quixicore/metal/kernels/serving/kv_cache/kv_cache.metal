#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

template <typename T>
kernel void kv_cache_zero(device T *key_cache [[buffer(0)]],
                          device T *value_cache [[buffer(1)]],
                          constant ulong &n [[buffer(2)]],
                          uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= n) {
        return;
    }
    key_cache[tid] = T(0);
    value_cache[tid] = T(0);
}

template <typename T>
kernel void kv_cache_scatter(device const T *key [[buffer(0)]],
                             device const T *value [[buffer(1)]],
                             device const long *slot_mapping [[buffer(2)]],
                             device T *key_cache [[buffer(3)]],
                             device T *value_cache [[buffer(4)]],
                             constant int &num_heads [[buffer(5)]],
                             constant int &head_size [[buffer(6)]],
                             constant int &block_size [[buffer(7)]],
                             uint token [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint tptg [[threads_per_threadgroup]]) {
    const long slot = slot_mapping[token];
    if (slot < 0) {
        return;
    }

    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const int row_elems = num_heads * head_size;
    const long src_base = (long)token * row_elems;
    const long dst_base =
        ((block * block_size + block_offset) * num_heads) * head_size;

    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        key_cache[dst_base + i] = key[src_base + i];
        value_cache[dst_base + i] = value[src_base + i];
    }
}

template <typename T>
kernel void kv_cache_gather(device const T *key_cache [[buffer(0)]],
                            device const T *value_cache [[buffer(1)]],
                            device T *key_out [[buffer(2)]],
                            device T *value_out [[buffer(3)]],
                            device const int *block_table [[buffer(4)]],
                            device const int *cu_seq_lens [[buffer(5)]],
                            constant int &num_tokens [[buffer(6)]],
                            constant int &num_seqs [[buffer(7)]],
                            constant int &block_size [[buffer(8)]],
                            constant int &block_table_stride [[buffer(9)]],
                            constant int &num_heads [[buffer(10)]],
                            constant int &head_size [[buffer(11)]],
                            uint token [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint tptg [[threads_per_threadgroup]]) {
    if ((int)token >= num_tokens) {
        return;
    }

    int lo = 0;
    int hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= (int)token) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    const int batch = lo;
    const int local_token = (int)token - cu_seq_lens[batch];
    const int table_col = local_token / block_size;
    const int slot = local_token % block_size;
    const int block = block_table[batch * block_table_stride + table_col];
    const int row_elems = num_heads * head_size;
    const long out_base = (long)token * row_elems;

    if (block < 0) {
        for (int i = (int)tid; i < row_elems; i += (int)tptg) {
            key_out[out_base + i] = T(0);
            value_out[out_base + i] = T(0);
        }
        return;
    }

    const long cache_base =
        (((long)block * block_size + slot) * num_heads) * head_size;
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        key_out[out_base + i] = key_cache[cache_base + i];
        value_out[out_base + i] = value_cache[cache_base + i];
    }
}

template <typename T>
kernel void kv_cache_clone(device const T *key_cache [[buffer(0)]],
                           device const T *value_cache [[buffer(1)]],
                           device T *key_out [[buffer(2)]],
                           device T *value_out [[buffer(3)]],
                           constant ulong &n [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
    // vec4 the full-cache clone (n = num_blocks*block_size*H_KV*D is a multiple of 4); 8/16-byte
    // transactions instead of 2-byte scalar loads. Tail handles any non-multiple-of-4 remainder.
    typedef typename base_types::packing<T>::packed_four T4;
    const ulong i = (ulong)gid * 4;
    if (i + 4 <= n) {
        *(device T4*)(key_out + i)   = *(device const T4*)(key_cache + i);
        *(device T4*)(value_out + i) = *(device const T4*)(value_cache + i);
    } else {
        for (ulong j = i; j < n; ++j) { key_out[j] = key_cache[j]; value_out[j] = value_cache[j]; }
    }
}

// Copy KV blocks src->dst. Reads from the ORIGINAL cache (key_src/value_src) and writes the CLONE
// (key_dst/value_dst) so a block that is simultaneously a source and a destination (a beam-reorder
// chain) always reads the pre-reorder value — deterministic and race-free regardless of pair order.
template <typename T>
kernel void kv_cache_copy_blocks(device const T *key_src [[buffer(0)]],
                                 device const T *value_src [[buffer(1)]],
                                 device T *key_dst [[buffer(2)]],
                                 device T *value_dst [[buffer(3)]],
                                 device const long *block_mapping [[buffer(4)]],
                                 constant int &numel_per_block [[buffer(5)]],
                                 uint pair [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tptg [[threads_per_threadgroup]]) {
    const long src_block = block_mapping[2 * pair];
    const long dst_block = block_mapping[2 * pair + 1];
    if (src_block < 0 || dst_block < 0) {
        return;
    }

    const long src_base = src_block * numel_per_block;
    const long dst_base = dst_block * numel_per_block;
    // vec4 the block copy (numel_per_block = block_size*H_KV*D is a multiple of 4, and both bases are
    // multiples of it -> 4-aligned); scalar tail covers any non-multiple-of-4 remainder.
    typedef typename base_types::packing<T>::packed_four T4;
    const int N4 = (numel_per_block & 3) == 0 ? numel_per_block : 0;
    for (int i = (int)tid * 4; i < N4; i += (int)tptg * 4) {
        *(device T4*)(key_dst + dst_base + i)   = *(device const T4*)(key_src + src_base + i);
        *(device T4*)(value_dst + dst_base + i) = *(device const T4*)(value_src + src_base + i);
    }
    for (int i = N4 + (int)tid; i < numel_per_block; i += (int)tptg) {
        key_dst[dst_base + i] = key_src[src_base + i];
        value_dst[dst_base + i] = value_src[src_base + i];
    }
}

// Build the (src,dst) block-copy pairs for a beam KV reorder ON-DEVICE — removes the host readback
// of parent_beam/block_table that the pure-Python builder needed (a per-step decode sync). Emits a
// FIXED-SIZE, deterministic pairs buffer (no atomics/scan): slot gid = gb*max_blocks + c holds the
// copy for global beam gb (= b*BM + k) at block column c, or the sentinel (-1,-1) which the
// downstream kv_cache_copy_blocks kernel already skips. A child beam k with parent p (== per-batch-
// local parent_beam[b,k]) copies parent block bt[b*BM+p, c] -> bt[gb, c] for c < ceil(sl/block_size);
// p == k (kept its own history) or an out-of-range/negative slot -> sentinel. One thread per slot.
// Wave-7 #6 (measure-first): a scan + atomic-cursor compaction of only the real pairs was rejected --
// this kernel is overhead-bound (~130 us floor, flat from 2k to 262k slots), so compacting the output
// cannot beat the launch/eval floor and would only add atomic contention + nondeterministic ordering
// while the downstream kv_cache_copy_blocks already skips sentinels cheaply. Kept as-is.
kernel void beam_build_copy_pairs(device const int  *parent_beam [[buffer(0)]],   // (B, BM)
                                  device const int  *block_table [[buffer(1)]],   // (B*BM, max_blocks)
                                  device const int  *seq_lens    [[buffer(2)]],   // (B*BM,)
                                  device long       *pairs       [[buffer(3)]],   // (B*BM*max_blocks, 2)
                                  constant int      &BM          [[buffer(4)]],
                                  constant int      &max_blocks  [[buffer(5)]],
                                  constant int      &block_size  [[buffer(6)]],
                                  constant int      &n_slots     [[buffer(7)]],
                                  uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= n_slots) return;
    const int gb = (int)gid / max_blocks;             // global beam = b*BM + k
    const int c  = (int)gid % max_blocks;             // block column
    const int b  = gb / BM;
    const int k  = gb % BM;
    long src = -1, dst = -1;
    const int p = parent_beam[b * BM + k];
    if (p != k) {                                     // p == k -> beam kept its own history, no move
        const int nblk = (seq_lens[gb] + block_size - 1) / block_size;
        if (c < nblk) {
            const int s = block_table[(b * BM + p) * max_blocks + c];
            const int d = block_table[gb * max_blocks + c];
            if (s >= 0 && d >= 0) { src = (long)s; dst = (long)d; }
        }
    }
    pairs[2 * (long)gid]     = src;
    pairs[2 * (long)gid + 1] = dst;
}

// Zero-copy beam KV reorder: emit a new block table where each child beam's rows point at its PARENT
// beam's physical blocks (no KV copy) — new_block_table[b*BM+k] = block_table[b*BM+parent_beam[b,k]].
// Caveat: children then SHARE physical blocks with the parent, so the cache manager must refcount /
// copy-on-write a block before any beam mutates it (out of scope here). One threadgroup per beam row.
kernel void beam_remap_block_table(device const int *block_table     [[buffer(0)]],  // (B*BM, max_blocks)
                                   device const int *parent_beam     [[buffer(1)]],  // (B, BM)
                                   device int       *new_block_table [[buffer(2)]],  // (B*BM, max_blocks)
                                   constant int &BM         [[buffer(3)]],
                                   constant int &max_blocks [[buffer(4)]],
                                   uint  row      [[threadgroup_position_in_grid]],
                                   uint  lid      [[thread_position_in_threadgroup]],
                                   uint  nthreads [[threads_per_threadgroup]]) {
    const int b = (int)row / BM;
    const int parent = parent_beam[row];               // per-batch-local parent index in [0, BM)
    const long dst = (long)row * max_blocks;
    const long src = (long)(b * BM + parent) * max_blocks;
    for (int c = (int)lid; c < max_blocks; c += (int)nthreads) {
        new_block_table[dst + c] = block_table[src + c];
    }
}

template <typename T>
kernel void kv_cache_scales(device const T *key [[buffer(0)]],
                            device const T *value [[buffer(1)]],
                            device float *key_scale [[buffer(2)]],
                            device float *value_scale [[buffer(3)]],
                            constant ulong &n [[buffer(4)]],
                            uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float shared_key[256];
    threadgroup float shared_value[256];

    float key_max = 0.0f;
    float value_max = 0.0f;
    for (ulong i = tid; i < n; i += 256) {
        key_max = max(key_max, abs(float(key[i])));
        value_max = max(value_max, abs(float(value[i])));
    }

    shared_key[tid] = key_max;
    shared_value[tid] = value_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_key[tid] = max(shared_key[tid], shared_key[tid + stride]);
            shared_value[tid] = max(shared_value[tid], shared_value[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        key_scale[0] = shared_key[0] / 240.0f;
        value_scale[0] = shared_value[0] / 240.0f;
    }
}

template <typename T, int D>
kernel void paged_attention(device const T *q [[buffer(0)]],
                            device const T *key_cache [[buffer(1)]],
                            device const T *value_cache [[buffer(2)]],
                            device const int *block_table [[buffer(3)]],
                            device const int *context_lens [[buffer(4)]],
                            device T *out [[buffer(5)]],
                            constant int &block_size [[buffer(6)]],
                            constant int &block_table_stride [[buffer(7)]],
                            constant float &scale [[buffer(8)]],
                            constant int &num_heads [[buffer(9)]],
                            constant int &num_kv_heads [[buffer(10)]],
                            device const float *alibi_slopes [[buffer(11)]],  // (num_heads,)
                            constant int &use_alibi [[buffer(12)]],            // 0 = off
                            device const int *block_mask [[buffer(13)]],       // (batch, [mask_heads,] max_blocks)
                            constant int &use_mask [[buffer(14)]],             // 0 = dense
                            constant int &window [[buffer(15)]],               // >0 = sliding window
                            constant int &mask_heads [[buffer(16)]],           // 1 = per-batch, H = per-head
                            uint3 tgid [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;       // query head (grid x ranges over num_heads)
    const int batch = (int)tgid.y;
    // GQA/MQA: each KV head is shared by (num_heads / num_kv_heads) query heads.
    // When num_kv_heads == num_heads this is plain MHA (kv_head == head).
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const long row_base = ((long)batch * num_heads + head) * D;
    // Sliding window: the decode query (at position context_len) attends the `window` most recent
    // keys [context_len-window, context_len). window <= 0 disables it (attend all).
    const int t_start = (window > 0) ? max(0, context_len - window) : 0;

    float qv[VALUES_PER_LANE];
    float acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }

    float m = -3.4028234663852886e38f;
    float l = 0.0f;

    for (int t = t_start; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) {
            continue;
        }
        // Block-sparse: skip whole KV blocks this query doesn't attend to. mask_heads == 1:
        // the mask shares the block_table's (batch, max_blocks) layout; mask_heads == H:
        // per-query-head selectivity (MInference vertical+slash masks).
        if (use_mask && block_mask[((long)batch * mask_heads + (mask_heads > 1 ? head : 0)) *
                                       block_table_stride + block_col] == 0) {
            continue;
        }

        const long cache_base =
            (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            partial += qv[i] * float(key_cache[cache_base + d]);
        }

        // ALiBi: linear per-head position bias; key t is at distance (context_len-1 - t) from
        // the (implicit) most-recent query position, so bias = slope * (t - context_len + 1) <= 0.
        float score = simd_sum(partial) * scale;
        if (use_alibi) { score += alibi_slopes[head] * float(t - context_len + 1); }
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);

        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            acc[i] = acc[i] * alpha + beta * float(value_cache[cache_base + d]);
        }
        l = l * alpha + beta;
        m = new_m;
    }

    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T(acc[i] / l);
    }
}

// vLLM x-packed cache layout: same decode, but the caches use vLLM's memory order so a
// ThunderMittens decode can read a vLLM KV cache directly. key_cache is
// (num_blocks, num_kv_heads, head_size/x, block_size, x) — the head dim is split into x-wide
// chunks with block_size in between (x = 16/sizeof(dtype) for coalesced 16-byte loads);
// value_cache is (num_blocks, num_kv_heads, head_size, block_size). x is passed at runtime.
template <typename T, int D>
kernel void paged_attention_xcache(device const T *q [[buffer(0)]],
                                   device const T *key_cache [[buffer(1)]],
                                   device const T *value_cache [[buffer(2)]],
                                   device const int *block_table [[buffer(3)]],
                                   device const int *context_lens [[buffer(4)]],
                                   device T *out [[buffer(5)]],
                                   constant int &block_size [[buffer(6)]],
                                   constant int &block_table_stride [[buffer(7)]],
                                   constant float &scale [[buffer(8)]],
                                   constant int &num_heads [[buffer(9)]],
                                   constant int &num_kv_heads [[buffer(10)]],
                                   constant int &x [[buffer(11)]],
                                   uint3 tgid [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;

    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const long row_base = ((long)batch * num_heads + head) * D;
    const int dh = D / x;   // head_size / x (number of x-wide chunks)

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }

    float m = -3.4028234663852886e38f, l = 0.0f;
    for (int t = 0; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }

        const long kv_hbase = (long)block * num_kv_heads + kv_head;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            // key_cache[block][kv_head][d/x][slot][d%x]
            const long kidx = (((kv_hbase * dh + d / x) * block_size + slot) * x) + (d % x);
            partial += qv[i] * float(key_cache[kidx]);
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            // value_cache[block][kv_head][d][slot]
            const long vidx = (kv_hbase * D + d) * block_size + slot;
            acc[i] = acc[i] * alpha + beta * float(value_cache[vidx]);
        }
        l = l * alpha + beta;
        m = new_m;
    }

    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T(acc[i] / l);
    }
}

// GQA KV-reuse staged decode: one threadgroup per (kv_head, batch) with `group_size`
// simdgroups (one query head each). Each KV token vector is staged into threadgroup memory
// ONCE and reused by every query head sharing that kv_head, amortizing the cache bandwidth
// by group_size (the flashinfer decode structure). Math is identical to paged_attention above,
// so the output is bit-for-bit equal; the two kernels differ only in memory-traffic shape.
template <typename T, int D>
kernel void paged_attention_gqa_staged(
    device const T *q [[buffer(0)]],
    device const T *key_cache [[buffer(1)]],
    device const T *value_cache [[buffer(2)]],
    device const int *block_table [[buffer(3)]],
    device const int *context_lens [[buffer(4)]],
    device T *out [[buffer(5)]],
    constant int &block_size [[buffer(6)]],
    constant int &block_table_stride [[buffer(7)]],
    constant float &scale [[buffer(8)]],
    constant int &num_heads [[buffer(9)]],
    constant int &num_kv_heads [[buffer(10)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_id [[simdgroup_index_in_threadgroup]],
    uint3 ntg [[threads_per_threadgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;
    threadgroup float sh_k[D];
    threadgroup float sh_v[D];

    const int kv_head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int group_size = num_heads / num_kv_heads;
    const int head = kv_head * group_size + (int)simd_id;   // query head this simdgroup serves
    const int context_len = context_lens[batch];
    const long row_base = ((long)batch * num_heads + head) * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    const int tid_flat = (int)lane + 32 * (int)simd_id;
    const int nthreads = (int)ntg.x * (int)ntg.y;

    for (int t = 0; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // prior iteration done reading sh_k/sh_v
        if (block >= 0) {
            const long cache_base =
                (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
            for (int idx = tid_flat; idx < D; idx += nthreads) {
                sh_k[idx] = float(key_cache[cache_base + idx]);
                sh_v[idx] = float(value_cache[cache_base + idx]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // sh_k/sh_v populated
        if (block < 0) { continue; }

        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            partial += qv[i] * sh_k[(int)lane + 32 * i];
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            acc[i] = acc[i] * alpha + beta * sh_v[(int)lane + 32 * i];
        }
        l = l * alpha + beta;
        m = new_m;
    }

    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T(acc[i] / l);
    }
}

// --- fp8 KV cache: store e4m3 codes (uint8) with per-tensor K/V scales. ---

kernel void kv_cache_zero_u8(device uchar *key_cache [[buffer(0)]],
                             device uchar *value_cache [[buffer(1)]],
                             constant ulong &n [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if ((ulong)tid >= n) { return; }
    key_cache[tid] = 0;     // e4m3(0) == 0x00
    value_cache[tid] = 0;
}

template <typename T>
kernel void kv_cache_scatter_fp8(device const T *key [[buffer(0)]],
                                 device const T *value [[buffer(1)]],
                                 device const long *slot_mapping [[buffer(2)]],
                                 device uchar *key_cache [[buffer(3)]],
                                 device uchar *value_cache [[buffer(4)]],
                                 constant int &num_heads [[buffer(5)]],
                                 constant int &head_size [[buffer(6)]],
                                 constant int &block_size [[buffer(7)]],
                                 device const float *k_scale [[buffer(8)]],
                                 device const float *v_scale [[buffer(9)]],
                                 constant int &fmt [[buffer(10)]],   // 0 = e4m3, 1 = e5m2
                                 uint token [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tptg [[threads_per_threadgroup]]) {
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }
    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const int row_elems = num_heads * head_size;
    const long src_base = (long)token * row_elems;
    const long dst_base = ((block * block_size + block_offset) * num_heads) * head_size;
    // k_scale/v_scale are (num_heads,) per-head arrays; element i lives in head i/head_size.
    // (Per-tensor callers pass a broadcast array with every entry equal.)
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        const int h = i / head_size;
        const float inv_k = k_scale[h] > 0.0f ? 1.0f / k_scale[h] : 0.0f;
        const float inv_v = v_scale[h] > 0.0f ? 1.0f / v_scale[h] : 0.0f;
        const float kq = float(key[src_base + i]) * inv_k;
        const float vq = float(value[src_base + i]) * inv_v;
        key_cache[dst_base + i]   = fmt == 1 ? tk_e5m2_encode(kq) : tk_e4m3_encode(kq);
        value_cache[dst_base + i] = fmt == 1 ? tk_e5m2_encode(vq) : tk_e4m3_encode(vq);
    }
}

template <typename T, int D, int FMT>
kernel void paged_attention_fp8(device const T *q [[buffer(0)]],
                                device const uchar *key_cache [[buffer(1)]],
                                device const uchar *value_cache [[buffer(2)]],
                                device const int *block_table [[buffer(3)]],
                                device const int *context_lens [[buffer(4)]],
                                device T *out [[buffer(5)]],
                                constant int &block_size [[buffer(6)]],
                                constant int &block_table_stride [[buffer(7)]],
                                constant float &scale [[buffer(8)]],
                                constant int &num_heads [[buffer(9)]],
                                constant int &num_kv_heads [[buffer(10)]],
                                device const float *k_scale [[buffer(11)]],
                                device const float *v_scale [[buffer(12)]],
                                constant int &fmt [[buffer(13)]],   // 0 = e4m3, 1 = e5m2
                                constant int &window [[buffer(14)]], // >0 = sliding window
                                uint3 tgid [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const float ks = k_scale[kv_head], vs = v_scale[kv_head];
    const float score_scale = scale * ks;
    const long row_base = ((long)batch * num_heads + head) * D;
    // Sliding window: attend only the `window` most recent keys.
    const int t_start = (window > 0) ? max(0, context_len - window) : 0;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;

    for (int t = t_start; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int slot = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long cache_base =
            (((long)block * block_size + slot) * num_kv_heads + kv_head) * D;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const uchar kcode = key_cache[cache_base + d];
            const float kdec = FMT == 1 ? float(tk_e5m2_decode(kcode)) : float(tk_e4m3_decode(kcode));
            partial += qv[i] * kdec;
        }
        const float score = simd_sum(partial) * score_scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const uchar vcode = value_cache[cache_base + d];
            const float vdec = FMT == 1 ? float(tk_e5m2_decode(vcode)) : float(tk_e4m3_decode(vcode));
            acc[i] = acc[i] * alpha + beta * vdec;
        }
        l = l * alpha + beta;
        m = new_m;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T((acc[i] / l) * vs);
    }
}

// ---------------------------------------------------------------------------
// fp8 KV gather + upconvert: the read path for a paged fp8 prefix cache. Reads
// e4m3/e5m2 code bytes from the paged cache and dequantizes to OUT_T (bf16) via
// tk_e4m3/e5m2_decode(code) * scale[kv_head] — per-kv_head scales so it round-trips
// exactly with kv_cache_scatter_fp8. Same worklist as kv_cache_gather (one TG per
// token, cu_seq_lens binary search, block<0 zero-fill). fmt 0=e4m3, 1=e5m2.
// ---------------------------------------------------------------------------
template <typename OUT_T>
kernel void kv_cache_gather_fp8(device const uchar *key_cache [[buffer(0)]],
                                device const uchar *value_cache [[buffer(1)]],
                                device OUT_T *key_out [[buffer(2)]],
                                device OUT_T *value_out [[buffer(3)]],
                                device const int *block_table [[buffer(4)]],
                                device const int *cu_seq_lens [[buffer(5)]],
                                device const float *k_scale [[buffer(6)]],
                                device const float *v_scale [[buffer(7)]],
                                constant int &num_tokens [[buffer(8)]],
                                constant int &num_seqs [[buffer(9)]],
                                constant int &block_size [[buffer(10)]],
                                constant int &block_table_stride [[buffer(11)]],
                                constant int &num_heads [[buffer(12)]],
                                constant int &head_size [[buffer(13)]],
                                constant int &fmt [[buffer(14)]],
                                uint token [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint tptg [[threads_per_threadgroup]]) {
    if ((int)token >= num_tokens) { return; }
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= (int)token) lo = mid; else hi = mid - 1;
    }
    const int batch = lo;
    const int local_token = (int)token - cu_seq_lens[batch];
    const int table_col = local_token / block_size;
    const int slot = local_token % block_size;
    const int block = block_table[batch * block_table_stride + table_col];
    const int row_elems = num_heads * head_size;
    const long out_base = (long)token * row_elems;
    if (block < 0) {
        for (int i = (int)tid; i < row_elems; i += (int)tptg) {
            key_out[out_base + i] = OUT_T(0);
            value_out[out_base + i] = OUT_T(0);
        }
        return;
    }
    const long cache_base = (((long)block * block_size + slot) * num_heads) * head_size;
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        const int kv_head = i / head_size;
        const uchar kc = key_cache[cache_base + i];
        const uchar vc = value_cache[cache_base + i];
        const float kd = fmt == 1 ? float(tk_e5m2_decode(kc)) : float(tk_e4m3_decode(kc));
        const float vd = fmt == 1 ? float(tk_e5m2_decode(vc)) : float(tk_e4m3_decode(vc));
        key_out[out_base + i] = OUT_T(kd * k_scale[kv_head]);
        value_out[out_base + i] = OUT_T(vd * v_scale[kv_head]);
    }
}

// Incremental per-tensor KV scale update (running max). Seeds from the existing scale and
// only raises it: new = max(old, absmax(k or v)/240). Single 256-thread threadgroup
// (grid-stride), same reduction as kv_cache_scales — no atomics needed for the scalar.
template <typename T>
kernel void kv_cache_scale_update(device const T *key [[buffer(0)]],
                                  device const T *value [[buffer(1)]],
                                  device const float *old_key_scale [[buffer(2)]],
                                  device const float *old_value_scale [[buffer(3)]],
                                  device float *new_key_scale [[buffer(4)]],
                                  device float *new_value_scale [[buffer(5)]],
                                  constant ulong &n [[buffer(6)]],
                                  uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float shared_key[256];
    threadgroup float shared_value[256];
    float key_max = 0.0f, value_max = 0.0f;
    for (ulong i = tid; i < n; i += 256) {
        key_max = max(key_max, abs(float(key[i])));
        value_max = max(value_max, abs(float(value[i])));
    }
    shared_key[tid] = key_max;
    shared_value[tid] = value_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_key[tid] = max(shared_key[tid], shared_key[tid + stride]);
            shared_value[tid] = max(shared_value[tid], shared_value[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        new_key_scale[0] = max(old_key_scale[0], shared_key[0] / 240.0f);
        new_value_scale[0] = max(old_value_scale[0], shared_value[0] / 240.0f);
    }
}

#define instantiate_kv_cache_scatter_fp8(type_name, T)                        \
  template [[host_name("kv_cache_scatter_fp8_" #type_name)]] [[kernel]] void  \
  kv_cache_scatter_fp8<T>(device const T *key [[buffer(0)]],                  \
                          device const T *value [[buffer(1)]],                \
                          device const long *slot_mapping [[buffer(2)]],      \
                          device uchar *key_cache [[buffer(3)]],              \
                          device uchar *value_cache [[buffer(4)]],            \
                          constant int &num_heads [[buffer(5)]],              \
                          constant int &head_size [[buffer(6)]],              \
                          constant int &block_size [[buffer(7)]],             \
                          device const float *k_scale [[buffer(8)]],          \
                          device const float *v_scale [[buffer(9)]],          \
                          constant int &fmt [[buffer(10)]],                   \
                          uint token [[threadgroup_position_in_grid]],        \
                          uint tid [[thread_position_in_threadgroup]],        \
                          uint tptg [[threads_per_threadgroup]]);             \
  template [[host_name("kv_cache_gather_fp8_" #type_name)]] [[kernel]] void   \
  kv_cache_gather_fp8<T>(device const uchar *key_cache [[buffer(0)]],         \
                         device const uchar *value_cache [[buffer(1)]],       \
                         device T *key_out [[buffer(2)]],                     \
                         device T *value_out [[buffer(3)]],                   \
                         device const int *block_table [[buffer(4)]],         \
                         device const int *cu_seq_lens [[buffer(5)]],         \
                         device const float *k_scale [[buffer(6)]],           \
                         device const float *v_scale [[buffer(7)]],           \
                         constant int &num_tokens [[buffer(8)]],              \
                         constant int &num_seqs [[buffer(9)]],                \
                         constant int &block_size [[buffer(10)]],             \
                         constant int &block_table_stride [[buffer(11)]],     \
                         constant int &num_heads [[buffer(12)]],              \
                         constant int &head_size [[buffer(13)]],              \
                         constant int &fmt [[buffer(14)]],                    \
                         uint token [[threadgroup_position_in_grid]],         \
                         uint tid [[thread_position_in_threadgroup]],         \
                         uint tptg [[threads_per_threadgroup]]);              \
  template [[host_name("kv_cache_scale_update_" #type_name)]] [[kernel]] void \
  kv_cache_scale_update<T>(device const T *key [[buffer(0)]],                 \
                           device const T *value [[buffer(1)]],               \
                           device const float *old_key_scale [[buffer(2)]],   \
                           device const float *old_value_scale [[buffer(3)]], \
                           device float *new_key_scale [[buffer(4)]],         \
                           device float *new_value_scale [[buffer(5)]],       \
                           constant ulong &n [[buffer(6)]],                   \
                           uint tid [[thread_position_in_threadgroup]]);

#define instantiate_paged_attention_fp8(fmt_name, FMT, type_name, T, DVAL)    \
  template [[host_name("paged_attention_fp8_" #fmt_name "_" #type_name "_" #DVAL)]] \
  [[kernel]] void paged_attention_fp8<T, DVAL, FMT>(                          \
      device const T *q [[buffer(0)]],                                        \
      device const uchar *key_cache [[buffer(1)]],                            \
      device const uchar *value_cache [[buffer(2)]],                          \
      device const int *block_table [[buffer(3)]],                            \
      device const int *context_lens [[buffer(4)]],                          \
      device T *out [[buffer(5)]],                                            \
      constant int &block_size [[buffer(6)]],                                 \
      constant int &block_table_stride [[buffer(7)]],                         \
      constant float &scale [[buffer(8)]],                                    \
      constant int &num_heads [[buffer(9)]],                                  \
      constant int &num_kv_heads [[buffer(10)]],                             \
      device const float *k_scale [[buffer(11)]],                             \
      device const float *v_scale [[buffer(12)]],                             \
      constant int &fmt [[buffer(13)]],                                       \
      constant int &window [[buffer(14)]],                                    \
      uint3 tgid [[threadgroup_position_in_grid]],                            \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kv_cache_scatter_fp8(float32, float)
instantiate_kv_cache_scatter_fp8(float16, half)
instantiate_kv_cache_scatter_fp8(bfloat16, bf16)
instantiate_paged_attention_fp8(e4m3, 0, float32, float, 64)
instantiate_paged_attention_fp8(e4m3, 0, float32, float, 128)
instantiate_paged_attention_fp8(e4m3, 0, float16, half, 64)
instantiate_paged_attention_fp8(e4m3, 0, float16, half, 128)
instantiate_paged_attention_fp8(e4m3, 0, bfloat16, bf16, 64)
instantiate_paged_attention_fp8(e4m3, 0, bfloat16, bf16, 128)
instantiate_paged_attention_fp8(e5m2, 1, float32, float, 64)
instantiate_paged_attention_fp8(e5m2, 1, float32, float, 128)
instantiate_paged_attention_fp8(e5m2, 1, float16, half, 64)
instantiate_paged_attention_fp8(e5m2, 1, float16, half, 128)
instantiate_paged_attention_fp8(e5m2, 1, bfloat16, bf16, 64)
instantiate_paged_attention_fp8(e5m2, 1, bfloat16, bf16, 128)

// ---------------------------------------------------------------------------
// Q8_0 KV cache.
//
// QuixiCore owns the tensor ABI rather than exposing the packed 34-byte GGML
// struct: codes are int8 (nb, bs, H, D), scales are fp16
// (nb, bs, H, D/32).  A logical Q8_0 block is still exactly 32 consecutive
// head-dimension values with one fp16 delta.  Codes are formed with the fp32
// delta (amax/127); reads use the stored fp16 delta.
// ---------------------------------------------------------------------------

kernel void kv_cache_zero_q8_0(device char *key_codes [[buffer(0)]],
                               device half *key_scales [[buffer(1)]],
                               device char *value_codes [[buffer(2)]],
                               device half *value_scales [[buffer(3)]],
                               constant ulong &code_n [[buffer(4)]],
                               constant ulong &scale_n [[buffer(5)]],
                               uint gid [[thread_position_in_grid]]) {
    const ulong i = (ulong)gid;
    if (i < code_n) {
        key_codes[i] = 0;
        value_codes[i] = 0;
    }
    if (i < scale_n) {
        key_scales[i] = 0.0h;
        value_scales[i] = 0.0h;
    }
}

template <typename T>
kernel void kv_cache_scatter_q8_0(device const T *key [[buffer(0)]],
                                  device const T *value [[buffer(1)]],
                                  device const long *slot_mapping [[buffer(2)]],
                                  device char *key_codes [[buffer(3)]],
                                  device half *key_scales [[buffer(4)]],
                                  device char *value_codes [[buffer(5)]],
                                  device half *value_scales [[buffer(6)]],
                                  constant int &num_heads [[buffer(7)]],
                                  constant int &head_size [[buffer(8)]],
                                  constant int &block_size [[buffer(9)]],
                                  uint3 tgid [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
#pragma METAL fp math_mode(safe)
    const int group = (int)tgid.x;
    const int head = (int)tgid.y;
    const int token = (int)tgid.z;
    const long slot = slot_mapping[token];
    if (slot < 0) { return; }

    const long block = slot / block_size;
    const long block_offset = slot % block_size;
    const int groups_per_head = head_size / 32;
    const int d = group * 32 + (int)lane;
    const long src = ((long)token * num_heads + head) * head_size + d;
    const long dst = ((block * block_size + block_offset) * num_heads + head) * head_size + d;
    const long scale_dst =
        ((block * block_size + block_offset) * num_heads + head) * groups_per_head + group;

    const float k = float(key[src]);
    const float v = float(value[src]);
    const float kamax = metal::simd_max(metal::fabs(k));
    const float vamax = metal::simd_max(metal::fabs(v));
    const float kd = kamax / 127.0f;
    const float vd = vamax / 127.0f;
    const float kinv = kd > 0.0f ? 1.0f / kd : 0.0f;
    const float vinv = vd > 0.0f ? 1.0f / vd : 0.0f;
    key_codes[dst] = tk_int8_encode(k * kinv);
    value_codes[dst] = tk_int8_encode(v * vinv);
    if (lane == 0) {
        key_scales[scale_dst] = half(kd);
        value_scales[scale_dst] = half(vd);
    }
}

template <typename OUT_T>
kernel void kv_cache_gather_q8_0(device const char *key_codes [[buffer(0)]],
                                 device const half *key_scales [[buffer(1)]],
                                 device const char *value_codes [[buffer(2)]],
                                 device const half *value_scales [[buffer(3)]],
                                 device OUT_T *key_out [[buffer(4)]],
                                 device OUT_T *value_out [[buffer(5)]],
                                 device const int *block_table [[buffer(6)]],
                                 device const int *cu_seq_lens [[buffer(7)]],
                                 constant int &num_tokens [[buffer(8)]],
                                 constant int &num_seqs [[buffer(9)]],
                                 constant int &block_size [[buffer(10)]],
                                 constant int &block_table_stride [[buffer(11)]],
                                 constant int &num_heads [[buffer(12)]],
                                 constant int &head_size [[buffer(13)]],
                                 uint token [[threadgroup_position_in_grid]],
                                 uint tid [[thread_position_in_threadgroup]],
                                 uint tptg [[threads_per_threadgroup]]) {
    if ((int)token >= num_tokens) { return; }
    int lo = 0, hi = num_seqs;
    while (lo < hi) {
        const int mid = (lo + hi + 1) / 2;
        if (cu_seq_lens[mid] <= (int)token) lo = mid; else hi = mid - 1;
    }
    const int batch = lo;
    const int local_token = (int)token - cu_seq_lens[batch];
    const int block_col = local_token / block_size;
    const int block_offset = local_token % block_size;
    const int block = block_table[batch * block_table_stride + block_col];
    const int row_elems = num_heads * head_size;
    const int groups_per_head = head_size / 32;
    const long out_base = (long)token * row_elems;
    if (block < 0) {
        for (int i = (int)tid; i < row_elems; i += (int)tptg) {
            key_out[out_base + i] = OUT_T(0);
            value_out[out_base + i] = OUT_T(0);
        }
        return;
    }
    const long code_base = ((long)block * block_size + block_offset) * row_elems;
    const long scale_base =
        ((long)block * block_size + block_offset) * num_heads * groups_per_head;
    for (int i = (int)tid; i < row_elems; i += (int)tptg) {
        const int head = i / head_size;
        const int d = i - head * head_size;
        const long si = scale_base + (long)head * groups_per_head + d / 32;
        key_out[out_base + i] = OUT_T(float(key_codes[code_base + i]) * float(key_scales[si]));
        value_out[out_base + i] =
            OUT_T(float(value_codes[code_base + i]) * float(value_scales[si]));
    }
}

kernel void kv_cache_clone_q8_0(device const char *key_codes [[buffer(0)]],
                                device const half *key_scales [[buffer(1)]],
                                device const char *value_codes [[buffer(2)]],
                                device const half *value_scales [[buffer(3)]],
                                device char *key_codes_out [[buffer(4)]],
                                device half *key_scales_out [[buffer(5)]],
                                device char *value_codes_out [[buffer(6)]],
                                device half *value_scales_out [[buffer(7)]],
                                constant ulong &code_n [[buffer(8)]],
                                constant ulong &scale_n [[buffer(9)]],
                                uint gid [[thread_position_in_grid]]) {
    const ulong i = (ulong)gid;
    if (i < code_n) {
        key_codes_out[i] = key_codes[i];
        value_codes_out[i] = value_codes[i];
    }
    if (i < scale_n) {
        key_scales_out[i] = key_scales[i];
        value_scales_out[i] = value_scales[i];
    }
}

kernel void kv_cache_copy_blocks_q8_0(
    device const char *key_codes [[buffer(0)]],
    device const half *key_scales [[buffer(1)]],
    device const char *value_codes [[buffer(2)]],
    device const half *value_scales [[buffer(3)]],
    device char *key_codes_out [[buffer(4)]],
    device half *key_scales_out [[buffer(5)]],
    device char *value_codes_out [[buffer(6)]],
    device half *value_scales_out [[buffer(7)]],
    device const long *block_mapping [[buffer(8)]],
    constant int &codes_per_block [[buffer(9)]],
    constant int &scales_per_block [[buffer(10)]],
    uint pair [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tptg [[threads_per_threadgroup]]) {
    const long src_block = block_mapping[2 * pair];
    const long dst_block = block_mapping[2 * pair + 1];
    if (src_block < 0 || dst_block < 0) { return; }
    const long src_code = src_block * codes_per_block;
    const long dst_code = dst_block * codes_per_block;
    for (int i = (int)tid; i < codes_per_block; i += (int)tptg) {
        key_codes_out[dst_code + i] = key_codes[src_code + i];
        value_codes_out[dst_code + i] = value_codes[src_code + i];
    }
    const long src_scale = src_block * scales_per_block;
    const long dst_scale = dst_block * scales_per_block;
    for (int i = (int)tid; i < scales_per_block; i += (int)tptg) {
        key_scales_out[dst_scale + i] = key_scales[src_scale + i];
        value_scales_out[dst_scale + i] = value_scales[src_scale + i];
    }
}

template <typename T, int D>
kernel void paged_attention_q8_0(device const T *q [[buffer(0)]],
                                 device const char *key_codes [[buffer(1)]],
                                 device const half *key_scales [[buffer(2)]],
                                 device const char *value_codes [[buffer(3)]],
                                 device const half *value_scales [[buffer(4)]],
                                 device const int *block_table [[buffer(5)]],
                                 device const int *context_lens [[buffer(6)]],
                                 device T *out [[buffer(7)]],
                                 constant int &block_size [[buffer(8)]],
                                 constant int &block_table_stride [[buffer(9)]],
                                 constant float &scale [[buffer(10)]],
                                 constant int &num_heads [[buffer(11)]],
                                 constant int &num_kv_heads [[buffer(12)]],
                                 constant int &window [[buffer(13)]],
                                 uint3 tgid [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    constexpr int VALUES_PER_LANE = D / 32;
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int kv_head = head / (num_heads / num_kv_heads);
    const int context_len = context_lens[batch];
    const int t_start = window > 0 ? max(0, context_len - window) : 0;
    const long row_base = ((long)batch * num_heads + head) * D;

    float qv[VALUES_PER_LANE], acc[VALUES_PER_LANE];
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        qv[i] = float(q[row_base + d]);
        acc[i] = 0.0f;
    }
    float m = -3.4028234663852886e38f, l = 0.0f;
    for (int t = t_start; t < context_len; ++t) {
        const int block_col = t / block_size;
        const int block_offset = t - block_col * block_size;
        const int block = block_table[batch * block_table_stride + block_col];
        if (block < 0) { continue; }
        const long code_base =
            (((long)block * block_size + block_offset) * num_kv_heads + kv_head) * D;
        const long scale_base =
            (((long)block * block_size + block_offset) * num_kv_heads + kv_head) * VALUES_PER_LANE;
        float partial = 0.0f;
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const float kd = float(key_codes[code_base + d]) * float(key_scales[scale_base + i]);
            partial += qv[i] * kd;
        }
        const float score = simd_sum(partial) * scale;
        const float new_m = max(m, score);
        const float alpha = l == 0.0f ? 0.0f : exp(m - new_m);
        const float beta = exp(score - new_m);
        for (int i = 0; i < VALUES_PER_LANE; ++i) {
            const int d = (int)lane + 32 * i;
            const float vd =
                float(value_codes[code_base + d]) * float(value_scales[scale_base + i]);
            acc[i] = acc[i] * alpha + beta * vd;
        }
        l = l * alpha + beta;
        m = new_m;
    }
    for (int i = 0; i < VALUES_PER_LANE; ++i) {
        const int d = (int)lane + 32 * i;
        out[row_base + d] = l == 0.0f ? T(0) : T(acc[i] / l);
    }
}

#define instantiate_kv_cache_q8_0(type_name, T)                                  \
  template [[host_name("kv_cache_scatter_q8_0_" #type_name)]] [[kernel]] void  \
  kv_cache_scatter_q8_0<T>(device const T *key [[buffer(0)]],                    \
                           device const T *value [[buffer(1)]],                  \
                           device const long *slot_mapping [[buffer(2)]],        \
                           device char *key_codes [[buffer(3)]],                 \
                           device half *key_scales [[buffer(4)]],                \
                           device char *value_codes [[buffer(5)]],               \
                           device half *value_scales [[buffer(6)]],              \
                           constant int &num_heads [[buffer(7)]],                \
                           constant int &head_size [[buffer(8)]],                \
                           constant int &block_size [[buffer(9)]],               \
                           uint3 tgid [[threadgroup_position_in_grid]],          \
                           uint lane [[thread_index_in_simdgroup]]);             \
  template [[host_name("kv_cache_gather_q8_0_" #type_name)]] [[kernel]] void   \
  kv_cache_gather_q8_0<T>(device const char *key_codes [[buffer(0)]],            \
                          device const half *key_scales [[buffer(1)]],           \
                          device const char *value_codes [[buffer(2)]],          \
                          device const half *value_scales [[buffer(3)]],         \
                          device T *key_out [[buffer(4)]],                       \
                          device T *value_out [[buffer(5)]],                     \
                          device const int *block_table [[buffer(6)]],           \
                          device const int *cu_seq_lens [[buffer(7)]],           \
                          constant int &num_tokens [[buffer(8)]],                \
                          constant int &num_seqs [[buffer(9)]],                  \
                          constant int &block_size [[buffer(10)]],               \
                          constant int &block_table_stride [[buffer(11)]],       \
                          constant int &num_heads [[buffer(12)]],                \
                          constant int &head_size [[buffer(13)]],                \
                          uint token [[threadgroup_position_in_grid]],           \
                          uint tid [[thread_position_in_threadgroup]],           \
                          uint tptg [[threads_per_threadgroup]]);

#define instantiate_paged_attention_q8_0(type_name, T, DVAL)                     \
  template [[host_name("paged_attention_q8_0_" #type_name "_" #DVAL)]]       \
  [[kernel]] void paged_attention_q8_0<T, DVAL>(                                \
      device const T *q [[buffer(0)]], device const char *key_codes [[buffer(1)]],\
      device const half *key_scales [[buffer(2)]],                              \
      device const char *value_codes [[buffer(3)]],                             \
      device const half *value_scales [[buffer(4)]],                            \
      device const int *block_table [[buffer(5)]],                              \
      device const int *context_lens [[buffer(6)]], device T *out [[buffer(7)]],\
      constant int &block_size [[buffer(8)]],                                   \
      constant int &block_table_stride [[buffer(9)]],                           \
      constant float &scale [[buffer(10)]], constant int &num_heads [[buffer(11)]],\
      constant int &num_kv_heads [[buffer(12)]], constant int &window [[buffer(13)]],\
      uint3 tgid [[threadgroup_position_in_grid]],                              \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kv_cache_q8_0(float32, float)
instantiate_kv_cache_q8_0(float16, half)
instantiate_kv_cache_q8_0(bfloat16, bf16)
instantiate_paged_attention_q8_0(float32, float, 64)
instantiate_paged_attention_q8_0(float32, float, 128)
instantiate_paged_attention_q8_0(float16, half, 64)
instantiate_paged_attention_q8_0(float16, half, 128)
instantiate_paged_attention_q8_0(bfloat16, bf16, 64)
instantiate_paged_attention_q8_0(bfloat16, bf16, 128)

#define instantiate_kv_cache_type(type_name, T)                               \
  template [[host_name("kv_cache_zero_" #type_name)]] [[kernel]] void        \
  kv_cache_zero<T>(device T *key_cache [[buffer(0)]],                         \
                   device T *value_cache [[buffer(1)]],                       \
                   constant ulong &n [[buffer(2)]],                           \
                   uint tid [[thread_position_in_grid]]);                     \
  template [[host_name("kv_cache_scatter_" #type_name)]] [[kernel]] void     \
  kv_cache_scatter<T>(device const T *key [[buffer(0)]],                      \
                      device const T *value [[buffer(1)]],                    \
                      device const long *slot_mapping [[buffer(2)]],          \
                      device T *key_cache [[buffer(3)]],                      \
                      device T *value_cache [[buffer(4)]],                    \
                      constant int &num_heads [[buffer(5)]],                  \
                      constant int &head_size [[buffer(6)]],                  \
                      constant int &block_size [[buffer(7)]],                 \
                      uint token [[threadgroup_position_in_grid]],            \
                      uint tid [[thread_position_in_threadgroup]],            \
                      uint tptg [[threads_per_threadgroup]]);                 \
  template [[host_name("kv_cache_gather_" #type_name)]] [[kernel]] void      \
  kv_cache_gather<T>(device const T *key_cache [[buffer(0)]],                 \
                     device const T *value_cache [[buffer(1)]],               \
                     device T *key_out [[buffer(2)]],                         \
                     device T *value_out [[buffer(3)]],                       \
                     device const int *block_table [[buffer(4)]],             \
                     device const int *cu_seq_lens [[buffer(5)]],             \
                     constant int &num_tokens [[buffer(6)]],                  \
                     constant int &num_seqs [[buffer(7)]],                    \
                     constant int &block_size [[buffer(8)]],                  \
                     constant int &block_table_stride [[buffer(9)]],          \
                     constant int &num_heads [[buffer(10)]],                  \
                     constant int &head_size [[buffer(11)]],                  \
                     uint token [[threadgroup_position_in_grid]],             \
                     uint tid [[thread_position_in_threadgroup]],             \
                     uint tptg [[threads_per_threadgroup]]);                  \
  template [[host_name("kv_cache_clone_" #type_name)]] [[kernel]] void       \
  kv_cache_clone<T>(device const T *key_cache [[buffer(0)]],                  \
                    device const T *value_cache [[buffer(1)]],                \
                    device T *key_out [[buffer(2)]],                          \
                    device T *value_out [[buffer(3)]],                        \
                    constant ulong &n [[buffer(4)]],                          \
                    uint tid [[thread_position_in_grid]]);                    \
  template [[host_name("kv_cache_copy_blocks_" #type_name)]] [[kernel]] void \
  kv_cache_copy_blocks<T>(device const T *key_src [[buffer(0)]],             \
                          device const T *value_src [[buffer(1)]],           \
                          device T *key_dst [[buffer(2)]],                    \
                          device T *value_dst [[buffer(3)]],                  \
                          device const long *block_mapping [[buffer(4)]],     \
                          constant int &numel_per_block [[buffer(5)]],        \
                          uint pair [[threadgroup_position_in_grid]],         \
                          uint tid [[thread_position_in_threadgroup]],        \
                          uint tptg [[threads_per_threadgroup]]);             \
  template [[host_name("kv_cache_scales_" #type_name)]] [[kernel]] void      \
  kv_cache_scales<T>(device const T *key [[buffer(0)]],                       \
                     device const T *value [[buffer(1)]],                     \
                     device float *key_scale [[buffer(2)]],                   \
                     device float *value_scale [[buffer(3)]],                 \
                     constant ulong &n [[buffer(4)]],                         \
                     uint tid [[thread_position_in_threadgroup]]);

#define instantiate_paged_attention_type(type_name, T, DVAL)                 \
  template [[host_name("paged_attention_" #type_name "_" #DVAL)]]            \
  [[kernel]] void paged_attention<T, DVAL>(                                  \
      device const T *q [[buffer(0)]],                                       \
      device const T *key_cache [[buffer(1)]],                               \
      device const T *value_cache [[buffer(2)]],                             \
      device const int *block_table [[buffer(3)]],                           \
      device const int *context_lens [[buffer(4)]],                          \
      device T *out [[buffer(5)]],                                           \
      constant int &block_size [[buffer(6)]],                                \
      constant int &block_table_stride [[buffer(7)]],                        \
      constant float &scale [[buffer(8)]],                                   \
      constant int &num_heads [[buffer(9)]],                                 \
      constant int &num_kv_heads [[buffer(10)]],                             \
      device const float *alibi_slopes [[buffer(11)]],                       \
      constant int &use_alibi [[buffer(12)]],                                \
      device const int *block_mask [[buffer(13)]],                           \
      constant int &use_mask [[buffer(14)]],                                 \
      constant int &window [[buffer(15)]],                                   \
      constant int &mask_heads [[buffer(16)]],                               \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);

#define instantiate_paged_attention_staged(type_name, T, DVAL)                \
  template [[host_name("paged_attention_gqa_staged_" #type_name "_" #DVAL)]]  \
  [[kernel]] void paged_attention_gqa_staged<T, DVAL>(                        \
      device const T *q [[buffer(0)]],                                       \
      device const T *key_cache [[buffer(1)]],                               \
      device const T *value_cache [[buffer(2)]],                             \
      device const int *block_table [[buffer(3)]],                           \
      device const int *context_lens [[buffer(4)]],                          \
      device T *out [[buffer(5)]],                                           \
      constant int &block_size [[buffer(6)]],                                \
      constant int &block_table_stride [[buffer(7)]],                        \
      constant float &scale [[buffer(8)]],                                   \
      constant int &num_heads [[buffer(9)]],                                 \
      constant int &num_kv_heads [[buffer(10)]],                             \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]],                               \
      uint simd_id [[simdgroup_index_in_threadgroup]],                       \
      uint3 ntg [[threads_per_threadgroup]]);

#define instantiate_paged_attention_xcache(type_name, T, DVAL)                \
  template [[host_name("paged_attention_xcache_" #type_name "_" #DVAL)]]      \
  [[kernel]] void paged_attention_xcache<T, DVAL>(                            \
      device const T *q [[buffer(0)]],                                       \
      device const T *key_cache [[buffer(1)]],                               \
      device const T *value_cache [[buffer(2)]],                             \
      device const int *block_table [[buffer(3)]],                           \
      device const int *context_lens [[buffer(4)]],                          \
      device T *out [[buffer(5)]],                                           \
      constant int &block_size [[buffer(6)]],                                \
      constant int &block_table_stride [[buffer(7)]],                        \
      constant float &scale [[buffer(8)]],                                   \
      constant int &num_heads [[buffer(9)]],                                 \
      constant int &num_kv_heads [[buffer(10)]],                             \
      constant int &x [[buffer(11)]],                                        \
      uint3 tgid [[threadgroup_position_in_grid]],                           \
      uint lane [[thread_index_in_simdgroup]]);

instantiate_kv_cache_type(float32, float)
instantiate_kv_cache_type(float16, half)
instantiate_kv_cache_type(bfloat16, bf16)

instantiate_paged_attention_type(float32, float, 64)
instantiate_paged_attention_type(float32, float, 128)
instantiate_paged_attention_type(float16, half, 64)
instantiate_paged_attention_type(float16, half, 128)
instantiate_paged_attention_type(bfloat16, bf16, 64)
instantiate_paged_attention_type(bfloat16, bf16, 128)

instantiate_paged_attention_staged(float32, float, 64)
instantiate_paged_attention_staged(float32, float, 128)
instantiate_paged_attention_staged(float16, half, 64)
instantiate_paged_attention_staged(float16, half, 128)
instantiate_paged_attention_staged(bfloat16, bf16, 64)
instantiate_paged_attention_staged(bfloat16, bf16, 128)

instantiate_paged_attention_xcache(float32, float, 64)
instantiate_paged_attention_xcache(float32, float, 128)
instantiate_paged_attention_xcache(float16, half, 64)
instantiate_paged_attention_xcache(float16, half, 128)
instantiate_paged_attention_xcache(bfloat16, bf16, 64)
instantiate_paged_attention_xcache(bfloat16, bf16, 128)
