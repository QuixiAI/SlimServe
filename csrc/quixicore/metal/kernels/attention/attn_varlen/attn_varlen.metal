#include "tk.metal"
#include <metal_stdlib>
namespace mittens {

// Varlen / paged-prefill flash attention: causal attention over ragged packed queries that read
// K/V straight from the paged KV cache (no dense (B,H,N,D) materialization), with prefix support
// (context_len >= q_len), GQA, and D in {64,128}.
//
// Layout decisions (see kernels/attn_varlen/attn_varlen.cpp for the host worklist builder):
//  - Q and O are HEAD-MAJOR packed: (H, total_padded, D). The register-tile loader hardcodes
//    row_stride == cols, so rows of a tile must be D-contiguous; head-major gives exactly that.
//    Each sequence is padded to a multiple of 8 rows, so the 8-row tiles never straddle two
//    sequences and tile gx owns packed rows [8*gx, 8*gx+8).
//  - The paged cache is (num_blocks, block_size, H_KV, D). A tile's 8 KV rows are strided by
//    H_KV*D there, so they can't be tile-loaded directly -> they are staged into a threadgroup
//    st<bf16,8,D> first (the attn_q pattern). block_size % 8 == 0 (asserted host-side) keeps an
//    8-aligned KV tile inside a single block.
//
// Per tile the worklist supplies: tile_seq[gx] = batch b, tile_local0[gx] = the tile's first row
// as a query index within sequence b. A query at tile-local row r sits at absolute position
// past + local0 + r (past = context_len[b] - q_len[b], the cached prefix) and attends keys
// [0, past+local0+r]. The boundary KV tile uses make_causal_shifted with shift = past+local0-kv0.
constant constexpr const int TNV = 8;

template <int D>
kernel void attn_varlen_prefill(device   bf16     *q_hm         [[buffer(0)]],  // (H, total_padded, D)
                                device   bf16     *key_cache    [[buffer(1)]],  // (nb, bs, H_KV, D)
                                device   bf16     *value_cache  [[buffer(2)]],
                                device const int  *block_table  [[buffer(3)]],  // (B, max_blocks)
                                device const int  *context_lens [[buffer(4)]],  // (B,)
                                device const int  *tile_seq     [[buffer(5)]],  // (n_tiles,)
                                device const int  *tile_local0  [[buffer(6)]],  // (n_tiles,)
                                device const int  *seq_qlen     [[buffer(7)]],  // (B,)
                                device   bf16     *o_hm         [[buffer(8)]],  // (H, total_padded, D)
                                constant int      &total_padded [[buffer(9)]],
                                constant int      &H            [[buffer(10)]],
                                constant int      &H_KV         [[buffer(11)]],
                                constant int      &block_size   [[buffer(12)]],
                                constant int      &bt_stride    [[buffer(13)]],
                                constant float    &scale        [[buffer(14)]],
                                constant float    &softcap      [[buffer(15)]],
                                device const float *sinks       [[buffer(16)]],
                                constant int      &has_sink     [[buffer(17)]],
                                uint3 blockIdx [[threadgroup_position_in_grid]],
                                uint  tid      [[thread_index_in_threadgroup]],
                                uint  laneId   [[thread_index_in_simdgroup]]) {
    static_assert(D == 64 || D == 128, "D must be 64 or 128");
    using global_layout = gl<bfloat, 1, -1, -1, D>;
    global_layout gl_q(q_hm, nullptr, H, total_padded, nullptr);
    global_layout gl_o(o_hm, nullptr, H, total_padded, nullptr);
    using rt_qkv = rt_bf<TNV, D>;
    using rt_k_t = rt_bf<TNV, D, ducks::rt_layout::col>;
    using rt_att = rt_fl<TNV, TNV>;
    using rt_o   = rt_fl<TNV, D>;
    using rv_att = rt_fl<TNV, TNV>::col_vec;

    const int gx   = (int)blockIdx.x;   // tile index
    const int head = (int)blockIdx.y;
    const int b    = tile_seq[gx];
    if (b < 0) { return; }              // sentinel tile past n_tiles (fully device-resident path)
    const int local0 = tile_local0[gx];
    const int ctx  = context_lens[b];
    const int past = ctx - seq_qlen[b];             // cached prefix length (>= 0)
    const int kv_head = head / (H / H_KV);          // GQA/MQA

    int kv_limit = past + local0 + TNV;             // exclusive upper bound over the tile's rows
    if (kv_limit > ctx) kv_limit = ctx;

    threadgroup st<half, TNV, D> sK, sV;
    rt_qkv q_reg; rt_k_t k_reg; rt_qkv v_reg; rt_att att_block; rt_o o_reg;
    rv_att max_vec_last, max_vec, norm_vec;

    load(q_reg, gl_q, {0, head, gx, 0}, laneId);
    const bool capped = softcap > 0.0f;
    const float sink_l2 = (has_sink != 0) ? sinks[head] * 1.44269504089f : 0.0f;
    if (has_sink != 0) { zero(max_vec); add(max_vec, max_vec, sink_l2); }
    else               { neg_infty(max_vec); }
    zero(norm_vec); zero(o_reg);
    // softcap active: keep raw scale (tanh is nonlinear); log2(e) applies after the cap.
    const bf16 q_mul = (bf16)(capped ? scale : scale * 1.44269504089f);
    mul(q_reg, q_reg, q_mul);

    for (int kv0 = 0; kv0 < kv_limit; kv0 += TNV) {
        const int block_col = kv0 / block_size;
        const int slot0 = kv0 - block_col * block_size;
        const int blk = block_table[b * bt_stride + block_col];
        // Stage the 8-row KV tile from the paged cache into threadgroup memory.
        for (int idx = (int)tid; idx < TNV * D; idx += 32) {
            const int s = idx / D, d = idx - s * D;
            if (blk < 0) { sK[int2(s, d)] = (half)0; sV[int2(s, d)] = (half)0; continue; }
            const long crow = (((long)blk * block_size + slot0 + s) * H_KV + kv_head) * D;
            sK[int2(s, d)] = (half)key_cache[crow + d];
            sV[int2(s, d)] = (half)value_cache[crow + d];
        }
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);

        load(k_reg, sK, laneId);                                    // shared -> col reg (K^T)
        zero(att_block);
        mma_ABt(att_block, q_reg, k_reg, att_block);
        if (capped) {   // BEFORE the causal mask (tanh would compress -1e30 to -softcap)
            mul(att_block, att_block, 1.0f / softcap);
            tanh(att_block, att_block);
            mul(att_block, att_block, softcap * 1.44269504089f);
        }
        // Boundary tile: mask keys past each query's causal horizon. shift = past+local0-kv0;
        // shift >= 7 means the whole tile is in-horizon (no-op).
        const int shift = past + local0 - kv0;
        if (shift <= 6) {   // shift >= 7 => whole tile within every row's causal horizon (no-op)
            float nb = -1e30f;
            make_causal_shifted(att_block, att_block, laneId, shift, nb);
        }
        copy(max_vec_last, max_vec, laneId);
        row_max(max_vec, att_block, max_vec, laneId);
        sub(max_vec_last, max_vec_last, max_vec); exp2(max_vec_last, max_vec_last);
        sub_row(att_block, att_block, max_vec); exp2(att_block, att_block);
        mul(norm_vec, norm_vec, max_vec_last);
        row_sum(norm_vec, att_block, norm_vec, laneId);
        mul_row(o_reg, o_reg, max_vec_last);
        load(v_reg, sV, laneId);                                    // shared -> row reg
        mma_AB(o_reg, att_block, v_reg, o_reg);
        threadgroup_barrier(metal::mem_flags::mem_threadgroup);     // before sK/sV reuse
    }
    if (has_sink != 0) {
        rv_att sink_term;
        copy(sink_term, max_vec, laneId);
        mul(sink_term, sink_term, -1.0f);
        add(sink_term, sink_term, sink_l2);
        exp2(sink_term, sink_term);
        add(norm_vec, norm_vec, sink_term);
    }
    div_row(o_reg, o_reg, norm_vec);
    store(gl_o, o_reg, {0, head, gx, 0}, laneId);
}

#define instantiate_attn_varlen(D)                                                     \
  template [[host_name("attn_varlen_prefill_" #D)]] [[kernel]] void                    \
  attn_varlen_prefill<D>(device bf16 *q_hm [[buffer(0)]], device bf16 *key_cache [[buffer(1)]], \
    device bf16 *value_cache [[buffer(2)]], device const int *block_table [[buffer(3)]], \
    device const int *context_lens [[buffer(4)]], device const int *tile_seq [[buffer(5)]], \
    device const int *tile_local0 [[buffer(6)]], device const int *seq_qlen [[buffer(7)]], \
    device bf16 *o_hm [[buffer(8)]], constant int &total_padded [[buffer(9)]],         \
    constant int &H [[buffer(10)]], constant int &H_KV [[buffer(11)]],                 \
    constant int &block_size [[buffer(12)]], constant int &bt_stride [[buffer(13)]],   \
    constant float &scale [[buffer(14)]], constant float &softcap [[buffer(15)]],      \
    device const float *sinks [[buffer(16)]], constant int &has_sink [[buffer(17)]],   \
    uint3 blockIdx [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]], \
    uint laneId [[thread_index_in_simdgroup]]);

instantiate_attn_varlen(64);
instantiate_attn_varlen(128);

// On-device varlen prefill scheduler: from a device cu_seqlens (B+1) build the per-8-row-tile
// worklist that attn_varlen_prefill consumes, with no host loop. One threadgroup (B <= 256); a
// threadgroup exclusive prefix-sum turns per-seq tile counts and padded lengths into tile offsets
// and pad offsets. Emits qlens (B), pad_off (B+1, exclusive over padded), tile_seq/tile_local0
// (max_tiles; sentinel -1 past n_tiles so the attn kernel skips unused tiles), and n_tiles (1).
// max_tiles is a host upper bound (>= sum ceil(qlen/8)); Metal cannot size a grid from device data.
[[host_name("varlen_build_worklist")]]
kernel void varlen_build_worklist(device const int *cu_seqlens [[buffer(0)]],   // (B+1,)
                                  device int *qlens       [[buffer(1)]],         // (B,)
                                  device int *pad_off     [[buffer(2)]],         // (B+1,)
                                  device int *tile_seq    [[buffer(3)]],         // (max_tiles,)
                                  device int *tile_local0 [[buffer(4)]],         // (max_tiles,)
                                  device int *n_tiles     [[buffer(5)]],         // (1,)
                                  constant int &B         [[buffer(6)]],
                                  constant int &max_tiles [[buffer(7)]],
                                  uint tid [[thread_index_in_threadgroup]],
                                  uint nthreads [[threads_per_threadgroup]]) {
    threadgroup int sg_sums[8];    // nthreads / 32 <= 8 (nthreads <= 256)
    // CHUNKED single-threadgroup scan: each thread owns a CONTIGUOUS chunk of batches, so B is not
    // capped by the thread count (the old one-thread-per-batch form capped at nthreads<=256). Two
    // passes over the (cheap-to-recompute) chunk: (1) per-thread local tile/pad totals, (2) a
    // threadgroup exclusive scan over those totals gives each thread its base offset, (3) re-walk the
    // chunk emitting tile_seq/tile_local0 + pad_off from the base. No per-chunk storage.
    const int chunk = (B + (int)nthreads - 1) / (int)nthreads;
    int lo = (int)tid * chunk; if (lo > B) { lo = B; }
    int hi = lo + chunk;       if (hi > B) { hi = B; }
    int local_tiles = 0, local_pad = 0;
    for (int b = lo; b < hi; ++b) {
        const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
        qlens[b] = qlen;
        const int nt = (qlen + 7) / 8;
        local_tiles += nt;
        local_pad += nt * 8;
    }
    int total_tiles = 0;
    const int base_tile = mittens::threadgroup_exclusive_scan_i32(local_tiles, tid, nthreads,
                                                                  sg_sums, total_tiles);
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);   // reuse sg_sums for scan #2
    int total_padded = 0;
    const int base_pad = mittens::threadgroup_exclusive_scan_i32(local_pad, tid, nthreads, sg_sums,
                                                                total_padded);
    if (tid == 0) { pad_off[B] = total_padded; n_tiles[0] = total_tiles; }
    int run_tile = base_tile, run_pad = base_pad;
    for (int b = lo; b < hi; ++b) {
        const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
        const int nt = (qlen + 7) / 8;
        pad_off[b] = run_pad;
        for (int t = 0; t < nt; ++t) {
            tile_seq[run_tile + t] = b;
            tile_local0[run_tile + t] = t * 8;
        }
        run_tile += nt;
        run_pad += nt * 8;
    }
    // sentinel-fill unused slots [total_tiles, max_tiles) (disjoint from the emit ranges above)
    for (int i = (int)tid; i < max_tiles; i += (int)nthreads) {
        if (i >= total_tiles) { tile_seq[i] = -1; tile_local0[i] = 0; }
    }
}

// Device-resident varlen Q pad/gather: build the padded head-major layout q_hm (H, total_padded, D)
// that attn_varlen_prefill consumes from packed Q (total_q, H, D). Grids over PADDED positions p so
// every row is written (real rows gather packed token cu_seqlens[b]+local; pad rows write 0) — no
// host per-batch pad+transpose, no pre-zero. One threadgroup per padded position; threads stride D.
[[host_name("varlen_q_pad_gather")]]
kernel void varlen_q_pad_gather(device const bf16 *q_packed   [[buffer(0)]],   // (total_q, H, D)
                                device const int  *cu_seqlens [[buffer(1)]],   // (B+1,)
                                device const int  *pad_off    [[buffer(2)]],   // (B+1,)
                                device bf16       *q_hm       [[buffer(3)]],   // (H, total_padded, D)
                                constant int &B [[buffer(4)]], constant int &H [[buffer(5)]],
                                constant int &D [[buffer(6)]], constant int &total_padded [[buffer(7)]],
                                uint p [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint nthreads [[threads_per_threadgroup]]) {
    threadgroup int s_valid, s_t;
    if (lid == 0) {
        int b = B - 1;
        for (int k = 0; k < B; ++k) { if ((int)p < pad_off[k + 1]) { b = k; break; } }
        const int i = (int)p - pad_off[b];
        const int qlen = cu_seqlens[b + 1] - cu_seqlens[b];
        s_valid = i < qlen ? 1 : 0;
        s_t = cu_seqlens[b] + i;
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const int valid = s_valid, t = s_t;
    for (int h = 0; h < H; ++h) {
        for (int d = (int)lid; d < D; d += (int)nthreads) {
            q_hm[((long)h * total_padded + (int)p) * D + d] =
                valid ? q_packed[((long)t * H + h) * D + d] : bf16(0.0f);
        }
    }
}

// Inverse: gather the head-major output o_hm (H, total_padded, D) back into packed (total_q, H, D).
[[host_name("varlen_o_regather")]]
kernel void varlen_o_regather(device const bf16 *o_hm       [[buffer(0)]],   // (H, total_padded, D)
                              device const int  *cu_seqlens [[buffer(1)]],   // (B+1,)
                              device const int  *pad_off    [[buffer(2)]],   // (B+1,)
                              device bf16       *o_packed   [[buffer(3)]],   // (total_q, H, D)
                              constant int &B [[buffer(4)]], constant int &H [[buffer(5)]],
                              constant int &D [[buffer(6)]], constant int &total_padded [[buffer(7)]],
                              uint t [[threadgroup_position_in_grid]],
                              uint lid [[thread_position_in_threadgroup]],
                              uint nthreads [[threads_per_threadgroup]]) {
    threadgroup int s_p;
    if (lid == 0) {
        int b = B - 1;
        for (int k = 0; k < B; ++k) { if ((int)t < cu_seqlens[k + 1]) { b = k; break; } }
        s_p = pad_off[b] + ((int)t - cu_seqlens[b]);
    }
    metal::threadgroup_barrier(metal::mem_flags::mem_threadgroup);
    const int p = s_p;
    for (int h = 0; h < H; ++h) {
        for (int d = (int)lid; d < D; d += (int)nthreads) {
            o_packed[((long)t * H + h) * D + d] = o_hm[((long)h * total_padded + p) * D + d];
        }
    }
}

}
