#include "tk.metal"
#include <metal_stdlib>

using namespace metal;
using namespace mittens;

// ---------------------------------------------------------------------------
// Token embedding + multimodal span merge (GPU-resident, no host preprocessing bubble).
//
//   embedding_lookup:  out[t] = scale * table[token_ids[t]]  (+ pos_table[t] if use_pos).
//                      A padded/negative token id emits zeros. Pure gather.
//   merge_multimodal_spans:  out[t] = src[t] >= 0 ? modal[src[t]] : text[t].  The per-row src map
//                      encodes which text positions are replaced by image/audio/video embeddings
//                      (built host-side from span offsets/lengths); the kernel is a per-row select.
// Both run ONE THREADGROUP PER TOKEN (t = threadgroup id), threads striding over D — this hoists
// the row base out of the inner loop (no per-element t=gid/D, d=gid%D integer divides) and lets the
// contiguous row be read/written as vec4 (packed_four) when D%4==0 (rows are then 4-aligned); a
// scalar loop covers the D%4!=0 case. T templated (fp16/bf16/fp32).
// Ref: FasterTransformer gpt_kernels invokeInputIdsEmbeddingLookupPosEncoding.
// ---------------------------------------------------------------------------

template <typename T>
kernel void embedding_lookup(device const int *token_ids [[buffer(0)]],  // (num_tok,)
                             device const T   *table     [[buffer(1)]],  // (vocab, D)
                             device const T   *pos_table [[buffer(2)]],  // (num_tok, D) or dummy
                             device T         *out       [[buffer(3)]],  // (num_tok, D)
                             constant int   &D           [[buffer(4)]],
                             constant int   &vocab       [[buffer(5)]],
                             constant int   &n_tok       [[buffer(6)]],
                             constant float &scale       [[buffer(7)]],
                             constant int   &use_pos     [[buffer(8)]],
                             uint  t        [[threadgroup_position_in_grid]],
                             uint  lid      [[thread_position_in_threadgroup]],
                             uint  nthreads [[threads_per_threadgroup]]) {
    typedef typename base_types::packing<T>::packed_four T4;
    const int tok = token_ids[t];
    const bool valid = tok >= 0 && tok < vocab;
    device const T *trow = table + (long)(valid ? tok : 0) * D;
    device const T *prow = pos_table + (long)t * D;
    device T       *orow = out + (long)t * D;
    const int D4 = ((D & 3) == 0) ? D : 0;   // vec4 body only when rows are 4-aligned
    for (int d = (int)lid * 4; d < D4; d += (int)nthreads * 4) {
        float4 v = valid ? float4(*(device const T4*)(trow + d)) * scale : float4(0.0f);
        if (use_pos) v += float4(*(device const T4*)(prow + d));
        *(device T4*)(orow + d) = T4(v);
    }
    for (int d = D4 + (int)lid; d < D; d += (int)nthreads) {   // scalar tail / D%4!=0 path
        float v = valid ? float(trow[d]) * scale : 0.0f;
        if (use_pos) v += float(prow[d]);
        orow[d] = T(v);
    }
}

// Two-table embedding preparation used by BERT-family inputs and other
// token+segment schemes:
//   out[t] = token_scale * token_table[token_ids[t]] + type_table[type_ids[t]].
// Each gather is independently bounds checked; an invalid id contributes zero.
template <typename T>
kernel void embedding_lookup_types(
    device const int *token_ids [[buffer(0)]],
    device const int *type_ids [[buffer(1)]],
    device const T *token_table [[buffer(2)]],
    device const T *type_table [[buffer(3)]],
    device T *out [[buffer(4)]],
    constant int &D [[buffer(5)]],
    constant int &token_vocab [[buffer(6)]],
    constant int &type_vocab [[buffer(7)]],
    constant float &token_scale [[buffer(8)]],
    uint t [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint nthreads [[threads_per_threadgroup]]) {
  typedef typename base_types::packing<T>::packed_four T4;
  const int token_id = token_ids[t];
  const int type_id = type_ids[t];
  const bool token_valid = token_id >= 0 && token_id < token_vocab;
  const bool type_valid = type_id >= 0 && type_id < type_vocab;
  device const T *token_row = token_table + (long)(token_valid ? token_id : 0) * D;
  device const T *type_row = type_table + (long)(type_valid ? type_id : 0) * D;
  device T *out_row = out + (long)t * D;
  const int D4 = ((D & 3) == 0) ? D : 0;
  for (int d = int(lid) * 4; d < D4; d += int(nthreads) * 4) {
    float4 v = token_valid ? float4(*(device const T4 *)(token_row + d)) * token_scale
                           : float4(0.0f);
    if (type_valid) v += float4(*(device const T4 *)(type_row + d));
    *(device T4 *)(out_row + d) = T4(v);
  }
  for (int d = D4 + int(lid); d < D; d += int(nthreads)) {
    float v = token_valid ? float(token_row[d]) * token_scale : 0.0f;
    if (type_valid) v += float(type_row[d]);
    out_row[d] = T(v);
  }
}

// out[t] = (src[t] >= 0) ? modal[src[t]] : text[t]. src (num_tok,) int: -1 keeps the text embedding,
// >=0 gathers row src[t] of the modal embeddings. One threadgroup per token (row select + copy).
template <typename T>
kernel void merge_multimodal_spans(device const T   *text  [[buffer(0)]],   // (num_tok, D)
                                   device const T   *modal [[buffer(1)]],   // (num_modal, D)
                                   device const int *src   [[buffer(2)]],   // (num_tok,)
                                   device T         *out   [[buffer(3)]],   // (num_tok, D)
                                   constant int &D         [[buffer(4)]],
                                   constant int &n_tok     [[buffer(5)]],
                                   constant int &n_modal   [[buffer(6)]],
                                   uint  t        [[threadgroup_position_in_grid]],
                                   uint  lid      [[thread_position_in_threadgroup]],
                                   uint  nthreads [[threads_per_threadgroup]]) {
    typedef typename base_types::packing<T>::packed_four T4;
    const int sm = src[t];
    const bool use_modal = sm >= 0 && sm < n_modal;
    device const T *srow = use_modal ? (modal + (long)sm * D) : (text + (long)t * D);
    device T       *orow = out + (long)t * D;
    const int D4 = ((D & 3) == 0) ? D : 0;
    for (int d = (int)lid * 4; d < D4; d += (int)nthreads * 4) {
        *(device T4*)(orow + d) = *(device const T4*)(srow + d);
    }
    for (int d = D4 + (int)lid; d < D; d += (int)nthreads) {
        orow[d] = srow[d];
    }
}

// Build the multimodal `src` map on-device (the input to merge_multimodal_spans), removing the host
// loop over image/audio spans. Span k covers text positions [span_offsets[k], +span_lengths[k]) and
// maps them to modal rows [modal_starts[k], +span_lengths[k]); so for a token t inside span k at
// offset o, src[t] = modal_starts[k] + o, else -1 (keep the text embedding). One thread per token
// (scans the few spans). Ref: the host span->modal index list built in the merge_multimodal docstring.
kernel void build_multimodal_src(device const int *span_offsets [[buffer(0)]],  // (num_spans,)
                                 device const int *span_lengths [[buffer(1)]],  // (num_spans,)
                                 device const int *modal_starts [[buffer(2)]],  // (num_spans,)
                                 device int       *src          [[buffer(3)]],  // (num_tok,)
                                 constant int &num_spans        [[buffer(4)]],
                                 constant int &num_tok          [[buffer(5)]],
                                 uint gid [[thread_position_in_grid]]) {
    if ((int)gid >= num_tok) { return; }
    const int t = (int)gid;
    int s = -1;
    for (int k = 0; k < num_spans; ++k) {
        const int o = t - span_offsets[k];
        if (o >= 0 && o < span_lengths[k]) { s = modal_starts[k] + o; break; }
    }
    src[gid] = s;
}

// Zero a float buffer (the gradient accumulator, before the atomic scatter-add). One thread/elem.
kernel void embedding_zero_f32(device float *p [[buffer(0)]],
                               constant int  &n [[buffer(1)]],
                               uint gid [[thread_position_in_grid]]) {
    if ((int)gid < n) p[gid] = 0.0f;
}

// Embedding backward (atomic scatter-add): dtable[token_ids[t]*D + d] += scale * dY[t*D + d].
// dtable (vocab, D) fp32 must be zeroed first (embedding_zero_f32). Tokens sharing an id scatter
// into the same row concurrently, so the accumulate is a relaxed device float atomic-add (P1a).
// A padding / out-of-range id contributes nothing. One threadgroup per token, threads stride D.
template <typename T>
kernel void embedding_backward(device const int    *token_ids [[buffer(0)]],  // (num_tok,)
                               device const T      *dY        [[buffer(1)]],  // (num_tok, D)
                               device metal::atomic_float *dtable [[buffer(2)]],  // (vocab, D) zeroed
                               constant int   &D           [[buffer(3)]],
                               constant int   &vocab       [[buffer(4)]],
                               constant int   &n_tok       [[buffer(5)]],
                               constant float &scale       [[buffer(6)]],
                               uint  t        [[threadgroup_position_in_grid]],
                               uint  lid      [[thread_position_in_threadgroup]],
                               uint  nthreads [[threads_per_threadgroup]]) {
    const int tok = token_ids[t];
    if (tok < 0 || tok >= vocab) return;
    const long trow = (long)tok * D;
    const long drow = (long)t * D;
    for (int d = (int)lid; d < D; d += (int)nthreads) {
        atomic_add_float(dtable, trow + d, float(dY[drow + d]) * scale);
    }
}

// Embedding backward (sorted-segment, atomic-free): the tokens are pre-sorted by id on the host
// (perm = argsort(token_ids), sorted_ids = token_ids[perm]). One threadgroup per sorted position;
// only the SEGMENT-START position of each id (sorted_ids[i-1] != id) does work, summing dY[perm[j]]
// over its equal-id run and writing dtable[id] once — so every output row is written by exactly one
// threadgroup and no atomics are needed. dtable must be zeroed first (ids absent from token_ids keep
// their zero). Negative / out-of-range ids sort to the front and are skipped. Wins over the atomic
// scatter when a few ids are heavily duplicated (high atomic contention). Ref: PyTorch
// embedding_dense_backward sorted-index path.
template <typename T>
kernel void embedding_backward_sorted(device const int *sorted_ids [[buffer(0)]],  // (num_tok,) asc
                                      device const int *perm       [[buffer(1)]],  // (num_tok,)
                                      device const T   *dY         [[buffer(2)]],  // (num_tok, D)
                                      device float     *dtable     [[buffer(3)]],  // (vocab, D) zeroed
                                      constant int   &D           [[buffer(4)]],
                                      constant int   &vocab       [[buffer(5)]],
                                      constant int   &n_tok       [[buffer(6)]],
                                      constant float &scale       [[buffer(7)]],
                                      uint  i        [[threadgroup_position_in_grid]],
                                      uint  lid      [[thread_position_in_threadgroup]],
                                      uint  nthreads [[threads_per_threadgroup]]) {
    const int id = sorted_ids[i];
    if (id < 0 || id >= vocab) return;                     // padding / out-of-range: no contribution
    if ((int)i > 0 && sorted_ids[i - 1] == id) return;     // only the segment start accumulates
    const long orow = (long)id * D;
    for (int d = (int)lid; d < D; d += (int)nthreads) {
        float acc = 0.0f;
        for (int j = (int)i; j < n_tok && sorted_ids[j] == id; ++j) {
            acc += float(dY[(long)perm[j] * D + d]);
        }
        dtable[orow + d] = acc * scale;                    // sole writer of this row -> no atomics
    }
}

#define instantiate_embedding(type_name, T)                                        \
  template [[host_name("embedding_backward_sorted_" #type_name)]] [[kernel]] void   \
  embedding_backward_sorted<T>(device const int *sorted_ids [[buffer(0)]],          \
    device const int *perm [[buffer(1)]], device const T *dY [[buffer(2)]],         \
    device float *dtable [[buffer(3)]], constant int &D [[buffer(4)]],              \
    constant int &vocab [[buffer(5)]], constant int &n_tok [[buffer(6)]],           \
    constant float &scale [[buffer(7)]], uint i [[threadgroup_position_in_grid]],   \
    uint lid [[thread_position_in_threadgroup]], uint nthreads [[threads_per_threadgroup]]); \
  template [[host_name("embedding_backward_" #type_name)]] [[kernel]] void          \
  embedding_backward<T>(device const int *token_ids [[buffer(0)]],                  \
    device const T *dY [[buffer(1)]], device metal::atomic_float *dtable [[buffer(2)]], \
    constant int &D [[buffer(3)]], constant int &vocab [[buffer(4)]],               \
    constant int &n_tok [[buffer(5)]], constant float &scale [[buffer(6)]],         \
    uint t [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], \
    uint nthreads [[threads_per_threadgroup]]);                                     \
  template [[host_name("embedding_lookup_" #type_name)]] [[kernel]] void            \
  embedding_lookup<T>(device const int *token_ids [[buffer(0)]],                    \
    device const T *table [[buffer(1)]], device const T *pos_table [[buffer(2)]],   \
    device T *out [[buffer(3)]], constant int &D [[buffer(4)]],                     \
    constant int &vocab [[buffer(5)]], constant int &n_tok [[buffer(6)]],           \
    constant float &scale [[buffer(7)]], constant int &use_pos [[buffer(8)]],       \
    uint t [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], \
    uint nthreads [[threads_per_threadgroup]]);                                     \
  template [[host_name("embedding_lookup_types_" #type_name)]] [[kernel]] void      \
  embedding_lookup_types<T>(device const int *token_ids [[buffer(0)]],              \
    device const int *type_ids [[buffer(1)]], device const T *token_table [[buffer(2)]],\
    device const T *type_table [[buffer(3)]], device T *out [[buffer(4)]],           \
    constant int &D [[buffer(5)]], constant int &token_vocab [[buffer(6)]],          \
    constant int &type_vocab [[buffer(7)]], constant float &token_scale [[buffer(8)]],\
    uint t [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]],\
    uint nthreads [[threads_per_threadgroup]]);                                     \
  template [[host_name("merge_multimodal_spans_" #type_name)]] [[kernel]] void      \
  merge_multimodal_spans<T>(device const T *text [[buffer(0)]],                     \
    device const T *modal [[buffer(1)]], device const int *src [[buffer(2)]],       \
    device T *out [[buffer(3)]], constant int &D [[buffer(4)]],                     \
    constant int &n_tok [[buffer(5)]], constant int &n_modal [[buffer(6)]],         \
    uint t [[threadgroup_position_in_grid]], uint lid [[thread_position_in_threadgroup]], \
    uint nthreads [[threads_per_threadgroup]]);

instantiate_embedding(float32, float)
instantiate_embedding(float16, half)
instantiate_embedding(bfloat16, bf16)
