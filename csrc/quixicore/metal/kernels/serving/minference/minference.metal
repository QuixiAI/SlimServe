#include "tk.metal"
#include <metal_stdlib>

using namespace metal;

namespace mittens {

// ---------------------------------------------------------------------------
// MInference block-mask builders (decode-focused): convert per-head vertical column
// indexes + slash diagonal offsets into the per-head block mask
// (batch, num_heads, max_blocks) consumed by paged_attention_block_sparse. For the decode
// query at position ctx-1: vertical col c marks block c / bs; slash offset o (distance from
// the diagonal) marks block (ctx-1 - o) / bs. -1 entries = padding. The reference's serial
// two-pointer CSR merge is deliberately skipped — the block mask IS our consumer format
// (CSR builder deferred until a prefill block-sparse consumer exists).
// nnz caps (vertical_topk / slash_topk <= nnz array width) give the _mergehead-style
// per-call budget without a second kernel.
// ---------------------------------------------------------------------------

kernel void minference_build_block_mask(
    device const int *vertical_indexes [[buffer(0)]],   // (B, H, nnz_v), -1 pad
    device const int *slash_indexes    [[buffer(1)]],   // (B, H, nnz_s), -1 pad
    device const int *context_lens     [[buffer(2)]],   // (B,)
    device int *block_mask             [[buffer(3)]],   // (B, H, max_blocks) 0/1
    constant int &num_heads  [[buffer(4)]],
    constant int &nnz_v      [[buffer(5)]],
    constant int &nnz_s      [[buffer(6)]],
    constant int &vertical_topk [[buffer(7)]],          // use first k verticals (<= nnz_v)
    constant int &slash_topk [[buffer(8)]],             // use first k slashes (<= nnz_s)
    constant int &block_size [[buffer(9)]],
    constant int &max_blocks [[buffer(10)]],
    constant int &last_n_blocks [[buffer(11)]],         // always-attend recent blocks (>=1)
    uint3 tgid [[threadgroup_position_in_grid]],
    uint lane  [[thread_index_in_simdgroup]]) {
    const int head = (int)tgid.x;
    const int batch = (int)tgid.y;
    const int ctx = context_lens[batch];
    const int qpos = ctx - 1;
    const int nblocks = metal::min((ctx + block_size - 1) / block_size, max_blocks);
    device int *mask = block_mask + ((long)batch * num_heads + head) * max_blocks;
    for (int i = (int)lane; i < max_blocks; i += 32) {
        mask[i] = 0;
    }
    simdgroup_barrier(mem_flags::mem_device);
    if (ctx <= 0) { return; }
    // recent blocks (the local window every MInference config keeps)
    for (int i = (int)lane; i < metal::min(last_n_blocks, nblocks); i += 32) {
        mask[nblocks - 1 - i] = 1;
    }
    const long vb = ((long)batch * num_heads + head) * nnz_v;
    const int nv = metal::min(vertical_topk, nnz_v);
    for (int i = (int)lane; i < nv; i += 32) {
        const int c = vertical_indexes[vb + i];
        if (c >= 0 && c < ctx) {
            mask[c / block_size] = 1;
        }
    }
    const long sb = ((long)batch * num_heads + head) * nnz_s;
    const int ns = metal::min(slash_topk, nnz_s);
    for (int i = (int)lane; i < ns; i += 32) {
        const int o = slash_indexes[sb + i];
        if (o >= 0 && o <= qpos) {
            mask[(qpos - o) / block_size] = 1;
        }
    }
}

} // namespace mittens
