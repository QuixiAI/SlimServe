// tk_cuda serving/decode bindings: torch wrappers over the validated W4/W5
// kernels (kernels/serving/*_kernels.cuh). Registered into the _C module by
// init_serving(m), called from tm_cuda_ext.cu's PYBIND11_MODULE.
#include "kv_cache_kernels.cuh"
#include "paged_attn_v2_kernels.cuh"
#include "rope_kv_kernels.cuh"
#include "attn_q_kernels.cuh"
#include "mla_kernels.cuh"
#include "beam_xcache_kernels.cuh"
#include "attn_varlen_kernels.cuh"
#include "slot_mapping_kernels.cuh"
#include "indexer_logits_mma.cuh"
#include "indexer_paged_logits.cuh"
#include "sampling_kernels.cuh"
#include "spec_beam_kernels.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

using namespace tms;
namespace py = pybind11;

#define CK(x) TORCH_CHECK(x.is_cuda() && x.is_contiguous(), #x " must be contiguous CUDA")
static cudaStream_t stream() { return at::cuda::getCurrentCUDAStream(); }
static const half* hp(const torch::Tensor& t) { return reinterpret_cast<const half*>(t.data_ptr()); }
static half* hpm(torch::Tensor& t) { return reinterpret_cast<half*>(t.data_ptr()); }
static const __nv_bfloat16* bp(const torch::Tensor& t) { return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr()); }
static __nv_bfloat16* bpm(torch::Tensor& t) { return reinterpret_cast<__nv_bfloat16*>(t.data_ptr()); }

// ---- kv cache management ----
static void py_kv_scatter(torch::Tensor key, torch::Tensor value, torch::Tensor slot_mapping,
                          torch::Tensor key_cache, torch::Tensor value_cache, int64_t block_size) {
    CK(key); CK(value); CK(slot_mapping); CK(key_cache); CK(value_cache);
    const int T = key.size(0), H = key.size(1), D = key.size(2);
    kv_cache_scatter<half><<<T, 128, 0, stream()>>>(hp(key), hp(value),
        slot_mapping.data_ptr<int64_t>(), hpm(key_cache), hpm(value_cache), H, D, int(block_size));
}
static std::tuple<torch::Tensor, torch::Tensor> py_kv_gather(
        torch::Tensor key_cache, torch::Tensor value_cache, torch::Tensor block_table,
        torch::Tensor cu_seq_lens, int64_t num_tokens, int64_t H, int64_t D, int64_t block_size) {
    CK(key_cache); CK(value_cache); CK(block_table); CK(cu_seq_lens);
    auto k = torch::empty({num_tokens, H, D}, key_cache.options());
    auto v = torch::empty({num_tokens, H, D}, key_cache.options());
    const int B = cu_seq_lens.numel() - 1;
    kv_cache_gather<half><<<int(num_tokens), 128, 0, stream()>>>(hp(key_cache), hp(value_cache),
        hpm(k), hpm(v), block_table.data_ptr<int>(), cu_seq_lens.data_ptr<int>(),
        int(num_tokens), B, int(block_size), int(block_table.size(1)), int(H), int(D));
    return {k, v};
}
static void py_copy_blocks(torch::Tensor ks, torch::Tensor vs, torch::Tensor kd, torch::Tensor vd,
                           torch::Tensor pairs, int64_t block_elems) {
    CK(ks); CK(vs); CK(kd); CK(vd); CK(pairs);
    const int n = pairs.size(0);
    kv_cache_copy_blocks<half><<<n, 128, 0, stream()>>>(hp(ks), hp(vs), hpm(kd), hpm(vd),
        pairs.data_ptr<int64_t>(), int(block_elems));
}

// ---- decode attention ----
static torch::Tensor py_paged_attention(torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
        torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale, int64_t num_kv_heads,
        c10::optional<torch::Tensor> alibi, c10::optional<torch::Tensor> block_mask, int64_t window) {
    CK(q); CK(kc); CK(vc); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    auto out = torch::empty_like(q);
    const float* al = alibi ? alibi->data_ptr<float>() : nullptr;
    const int* bm = block_mask ? block_mask->data_ptr<int>() : nullptr;
    dim3 grid(H, B);
    #define LAUNCH(DD) paged_attention<half, DD><<<grid, 32, 0, stream()>>>(hp(q), hp(kc), hp(vc), \
        bt.data_ptr<int>(), ctx.data_ptr<int>(), hpm(out), int(block_size), int(bt.size(1)), \
        float(scale), H, int(num_kv_heads), al, al ? 1 : 0, bm, bm ? 1 : 0, int(window))
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}
static torch::Tensor py_paged_attention_v2(torch::Tensor q, torch::Tensor kc, torch::Tensor vc,
        torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale, int64_t num_kv_heads,
        int64_t partition_size, int64_t max_context, int64_t window) {
    CK(q); CK(kc); CK(vc); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    const int P = int((max_context + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, D}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    auto out = torch::empty_like(q);
    dim3 pg(H, B, P), rg(H, B);
    #define LAUNCH(DD) do { \
        paged_attention_partition<half, DD><<<pg, 32, 0, stream()>>>(hp(q), hp(kc), hp(vc), \
            bt.data_ptr<int>(), ctx.data_ptr<int>(), tmp.data_ptr<float>(), ml.data_ptr<float>(), \
            es.data_ptr<float>(), int(block_size), int(bt.size(1)), float(scale), H, \
            int(num_kv_heads), P, int(partition_size), int(window)); \
        paged_attention_reduce<half, DD><<<rg, 32, 0, stream()>>>(tmp.data_ptr<float>(), \
            ml.data_ptr<float>(), es.data_ptr<float>(), hpm(out), H, P); } while (0)
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}

// ---- rope + insert ----
static void py_rope_kv_insert(torch::Tensor k, torch::Tensor v, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, torch::Tensor slots, torch::Tensor kc, torch::Tensor vc,
        int64_t num_kv_heads, int64_t block_size, c10::optional<torch::Tensor> norm_weight,
        bool gemma, double eps) {
    CK(k); CK(v); CK(cosb); CK(sinb); CK(positions); CK(slots); CK(kc); CK(vc);
    const int M = k.size(0), D = k.size(1);
    const half* w = norm_weight ? hp(*norm_weight) : nullptr;
    #define LAUNCH(DD, NORM) rope_kv_insert<half, DD, NORM><<<M, 32, 0, stream()>>>(hp(k), hp(v), \
        hp(cosb), hp(sinb), positions.data_ptr<int>(), slots.data_ptr<int64_t>(), hpm(kc), hpm(vc), \
        w, int(num_kv_heads), int(block_size), gemma ? 1 : 0, float(eps))
    if (D == 64)  { if (w) LAUNCH(64, true);  else LAUNCH(64, false); }
    else if (D == 128) { if (w) LAUNCH(128, true); else LAUNCH(128, false); }
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
}
static torch::Tensor py_rope_q(torch::Tensor q, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, int64_t num_heads, c10::optional<torch::Tensor> norm_weight,
        bool gemma, double eps) {
    CK(q); CK(cosb); CK(sinb); CK(positions);
    const int M = q.size(0), D = q.size(1);
    auto out = torch::empty_like(q);
    const half* w = norm_weight ? hp(*norm_weight) : nullptr;
    #define LAUNCH(DD, NORM) rope_q<half, DD, NORM><<<M, 32, 0, stream()>>>(hp(q), hp(cosb), \
        hp(sinb), positions.data_ptr<int>(), hpm(out), w, int(num_heads), gemma ? 1 : 0, float(eps))
    if (D == 64)  { if (w) LAUNCH(64, true);  else LAUNCH(64, false); }
    else if (D == 128) { if (w) LAUNCH(128, true); else LAUNCH(128, false); }
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}

// ---- quantized-KV attention ----
static torch::Tensor py_attn_q(torch::Tensor q, torch::Tensor Kq, torch::Tensor Vq,
                               const std::string& fmt, bool causal, double scale,
                               int64_t B, int64_t H, int64_t N, int64_t D) {
    CK(q); CK(Kq); CK(Vq);
    auto out = torch::empty_like(q);
    dim3 grid{unsigned(N), unsigned(H), unsigned(B)};
    #define LAUNCH(FMT, DD, C) attn_q<tmq::FMT, DD, C><<<grid, 32, 0, stream()>>>(hp(q), \
        Kq.data_ptr<uint8_t>(), Vq.data_ptr<uint8_t>(), hpm(out), int(N), int(H), float(scale))
    #define DISPATCH(FMT) do { \
        if (D == 64)  { if (causal) LAUNCH(FMT, 64, true);  else LAUNCH(FMT, 64, false);  } \
        else if (D == 128) { if (causal) LAUNCH(FMT, 128, true); else LAUNCH(FMT, 128, false); } \
        else TORCH_CHECK(false, "D must be 64/128"); return out; } while (0)
    if (fmt == "q8_0") DISPATCH(q8_0);
    if (fmt == "q4_0") DISPATCH(q4_0);
    if (fmt == "fp8_e4m3") DISPATCH(fp8_e4m3);
    TORCH_CHECK(false, "attn_q format must be q8_0/q4_0/fp8_e4m3");
    #undef DISPATCH
    #undef LAUNCH
}

// ---- MLA ----
static torch::Tensor py_mla_decode(torch::Tensor q, torch::Tensor kv_cache, torch::Tensor bt,
                                   torch::Tensor ctx, int64_t block_size, double scale) {
    CK(q); CK(kv_cache); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "mla_decode expects QK=576");
    auto out = torch::empty({B, H, 512}, q.options());
    mla_decode<512, 64><<<dim3(H, B), 32, 0, stream()>>>(bp(q), bp(kv_cache), bt.data_ptr<int>(),
        ctx.data_ptr<int>(), bpm(out), int(block_size), int(bt.size(1)), float(scale), H);
    return out;
}
static torch::Tensor py_mla_decode_fp8(torch::Tensor q, torch::Tensor data, torch::Tensor scl,
                                       torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale) {
    CK(q); CK(data); CK(scl); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    auto out = torch::empty_like(q);
    mla_decode_fp8<<<dim3(H, B), 32, 0, stream()>>>(bp(q), data.data_ptr<uint8_t>(),
        scl.data_ptr<uint8_t>(), bt.data_ptr<int>(), ctx.data_ptr<int>(), bpm(out),
        int(block_size), int(bt.size(1)), float(scale), H);
    return out;
}
static void py_mla_kv_insert_fp8(torch::Tensor kv, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, torch::Tensor slots, torch::Tensor data, torch::Tensor scl,
        int64_t block_size) {
    CK(kv); CK(cosb); CK(sinb); CK(positions); CK(slots); CK(data); CK(scl);
    mla_kv_insert_fp8<<<int(kv.size(0)), 32, 0, stream()>>>(bp(kv), bp(cosb), bp(sinb),
        positions.data_ptr<int>(), slots.data_ptr<int64_t>(), data.data_ptr<uint8_t>(),
        scl.data_ptr<uint8_t>(), int(block_size));
}
static torch::Tensor py_mla_q_norm_rope(torch::Tensor q, torch::Tensor cosb, torch::Tensor sinb,
        torch::Tensor positions, int64_t num_heads, int64_t nope_dim, int64_t rope_dim,
        int64_t norm_mode, double eps, c10::optional<torch::Tensor> norm_weight) {
    CK(q); CK(cosb); CK(sinb); CK(positions);
    const int M = q.size(0), D = q.size(1);
    auto out = torch::empty_like(q);
    const __nv_bfloat16* w = norm_weight ? bp(*norm_weight) : nullptr;
    #define LAUNCH(DD) mla_q_norm_rope<DD><<<M, 32, 0, stream()>>>(bp(q), bp(cosb), bp(sinb), \
        positions.data_ptr<int>(), bpm(out), int(num_heads), int(nope_dim), int(rope_dim), \
        int(norm_mode), float(eps), w)
    if (D == 128) LAUNCH(128); else if (D == 192) LAUNCH(192);
    else if (D == 256) LAUNCH(256); else if (D == 512) LAUNCH(512);
    else TORCH_CHECK(false, "D must be 128/192/256/512");
    #undef LAUNCH
    return out;
}

// ---- samplers ----
static torch::Tensor py_sample(torch::Tensor logits, const std::string& mode, int64_t seed,
                               double temperature, double param, int64_t K) {
    CK(logits);
    TORCH_CHECK(logits.scalar_type() == torch::kFloat, "logits must be fp32");
    const int T = logits.size(0), V = logits.size(1);
    auto out = torch::empty({T}, logits.options().dtype(torch::kInt));
    const float invtemp = temperature > 0 ? float(1.0 / temperature) : 1.0f;
    const float* lp = logits.data_ptr<float>();
    int* op = out.data_ptr<int>();
    if (mode == "argmax")      argmax_k<float><<<T, 32, 0, stream()>>>(lp, op, V);
    else if (mode == "categorical") sample_categorical<float><<<T, 32, 0, stream()>>>(lp, op, V, unsigned(seed), invtemp);
    else if (mode == "top_k")  top_k_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, int(K), unsigned(seed), invtemp);
    else if (mode == "top_p")  top_p_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, float(param), unsigned(seed), invtemp);
    else if (mode == "min_p")  min_p_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, float(param), unsigned(seed), invtemp);
    else if (mode == "typical_p") typical_p_sample<float><<<T, 32, 0, stream()>>>(lp, op, V, float(param), unsigned(seed), invtemp);
    else TORCH_CHECK(false, "unknown sample mode ", mode);
    return out;
}
static torch::Tensor py_apply_penalties(torch::Tensor logits, torch::Tensor prev_tokens,
        torch::Tensor parent_ids, double temperature, double rep, double presence, double freq,
        torch::Tensor bias, int64_t eos_id, int64_t min_length, int64_t gen_len) {
    CK(logits); CK(prev_tokens); CK(parent_ids); CK(bias);
    const int T = logits.size(0), V = logits.size(1), L = prev_tokens.size(1);
    auto counts = torch::zeros({T, V}, logits.options().dtype(torch::kInt));
    auto out = torch::empty_like(logits);
    penalty_histogram<<<(T * L + 255) / 256, 256, 0, stream()>>>(prev_tokens.data_ptr<int>(),
        counts.data_ptr<int>(), V, L, T * L, parent_ids.data_ptr<int>());
    apply_penalty<float><<<T, 32, 0, stream()>>>(logits.data_ptr<float>(), counts.data_ptr<int>(),
        out.data_ptr<float>(), V, temperature > 0 ? float(1.0 / temperature) : 1.0f,
        float(rep), float(presence), float(freq), bias.data_ptr<float>(),
        int(eos_id), int(min_length), int(gen_len));
    return out;
}
static torch::Tensor py_apply_token_bitmask(torch::Tensor logits, torch::Tensor bitmask) {
    CK(logits); CK(bitmask);
    const int T = logits.size(0), V = logits.size(1);
    auto out = torch::empty_like(logits);
    apply_token_bitmask<float><<<T, 32, 0, stream()>>>(logits.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(bitmask.data_ptr<int>()), out.data_ptr<float>(),
        V, int(bitmask.size(1)));
    return out;
}
static torch::Tensor py_apply_bad_words(torch::Tensor logits, torch::Tensor bad_ids, torch::Tensor bad_lens) {
    CK(logits); CK(bad_ids); CK(bad_lens);
    const int T = logits.size(0), V = logits.size(1);
    auto out = torch::empty_like(logits);
    apply_bad_words<float><<<T, 32, 0, stream()>>>(logits.data_ptr<float>(), out.data_ptr<float>(),
        bad_ids.data_ptr<int>(), bad_lens.data_ptr<int>(), V, int(bad_ids.size(1)));
    return out;
}

// ---- beam + spec ----
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_beam_advance(
        torch::Tensor logits, torch::Tensor cum_log_probs, int64_t BM) {
    CK(logits); CK(cum_log_probs);
    const int rows = logits.size(0), V = logits.size(1), two_bm = 2 * int(BM);
    const int B = rows / int(BM);
    auto sc = torch::empty({rows, two_bm}, logits.options());
    auto tok = torch::empty({rows, two_bm}, logits.options().dtype(torch::kInt));
    auto nt = torch::empty({B, int(BM)}, tok.options());
    auto par = torch::empty({B, int(BM)}, tok.options());
    auto nc = torch::empty({B, int(BM)}, logits.options());
    beam_topk_partials<float><<<rows, 32, 0, stream()>>>(logits.data_ptr<float>(),
        cum_log_probs.data_ptr<float>(), sc.data_ptr<float>(), tok.data_ptr<int>(), V, two_bm);
    beam_select<<<B, 32, 0, stream()>>>(sc.data_ptr<float>(), tok.data_ptr<int>(),
        nt.data_ptr<int>(), par.data_ptr<int>(), nc.data_ptr<float>(), int(BM), two_bm);
    return {nt, par, nc};
}
static std::tuple<torch::Tensor, torch::Tensor> py_spec_verify_linear(
        torch::Tensor draft, torch::Tensor draft_probs, torch::Tensor target_probs,
        torch::Tensor bonus, torch::Tensor accept_u, int64_t seed) {
    CK(draft); CK(draft_probs); CK(target_probs); CK(bonus); CK(accept_u);
    const int B = draft.size(0), S = draft.size(1), V = draft_probs.size(2);
    auto out = torch::empty({B, S + 1}, draft.options());
    auto cnt = torch::empty({B}, draft.options());
    spec_verify_linear<<<B, 32, 0, stream()>>>(draft.data_ptr<int>(), draft_probs.data_ptr<float>(),
        target_probs.data_ptr<float>(), bonus.data_ptr<int>(), accept_u.data_ptr<float>(),
        out.data_ptr<int>(), cnt.data_ptr<int>(), S, V, unsigned(seed));
    return {out, cnt};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_build_dynamic_tree(torch::Tensor parents) {
    CK(parents);
    const int B = parents.size(0), N = parents.size(1);
    auto nt = torch::empty_like(parents);
    auto ns = torch::empty_like(parents);
    auto pos = torch::empty_like(parents);
    build_dynamic_tree<<<B, 32, 0, stream()>>>(parents.data_ptr<int>(), nt.data_ptr<int>(),
        ns.data_ptr<int>(), pos.data_ptr<int>(), N);
    return {nt, ns, pos};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_spec_verify_tree(
        torch::Tensor draft, torch::Tensor target_probs, torch::Tensor nt, torch::Tensor ns,
        torch::Tensor tree_valid, int64_t seed) {
    CK(draft); CK(target_probs); CK(nt); CK(ns); CK(tree_valid);
    const int B = nt.size(0), N = nt.size(1), V = target_probs.size(2);
    auto ai = torch::empty({B, N}, nt.options());
    auto at = torch::empty({B, N}, nt.options());
    auto an = torch::empty({B}, nt.options());
    spec_verify_tree<<<B, 32, 0, stream()>>>(draft.data_ptr<int>(), target_probs.data_ptr<float>(),
        nt.data_ptr<int>(), ns.data_ptr<int>(), ai.data_ptr<int>(), at.data_ptr<int>(),
        an.data_ptr<int>(), N, V, unsigned(seed), tree_valid.data_ptr<int>());
    return {ai, at, an};
}
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> py_spec_compact(
        torch::Tensor out_tokens, torch::Tensor accepted_cnt, torch::Tensor seq_lens) {
    CK(out_tokens); CK(accepted_cnt); CK(seq_lens);
    const int B = out_tokens.size(0), Sp1 = out_tokens.size(1);
    auto pt = torch::empty({B * Sp1}, out_tokens.options());
    auto pp = torch::empty({B * Sp1}, out_tokens.options());
    auto cu = torch::empty({B + 1}, out_tokens.options());
    spec_compact<<<1, 256, 0, stream()>>>(out_tokens.data_ptr<int>(), accepted_cnt.data_ptr<int>(),
        seq_lens.data_ptr<int>(), pt.data_ptr<int>(), pp.data_ptr<int>(), cu.data_ptr<int>(), B, Sp1);
    return {pt, pp, cu};
}
static torch::Tensor py_spec_update_kv_meta(torch::Tensor seq_lens, torch::Tensor accepted_cnt) {
    CK(seq_lens); CK(accepted_cnt);
    const int B = seq_lens.numel();
    auto out = torch::empty_like(seq_lens);
    spec_update_kv_meta<<<(B + 255) / 256, 256, 0, stream()>>>(seq_lens.data_ptr<int>(),
        accepted_cnt.data_ptr<int>(), out.data_ptr<int>(), B);
    return out;
}

// ---- Step-4 additions: bf16 MLA insert, partitioned/sparse MLA decodes,
// gqa_staged decode, dense sliding-window attention ----
static void py_mla_kv_insert(torch::Tensor kv_c, torch::Tensor k_pe, torch::Tensor cosb,
        torch::Tensor sinb, torch::Tensor positions, torch::Tensor slots,
        torch::Tensor kv_cache, int64_t block_size, int64_t norm_mode, double eps,
        c10::optional<torch::Tensor> norm_weight) {
    CK(kv_c); CK(k_pe); CK(cosb); CK(sinb); CK(positions); CK(slots); CK(kv_cache);
    const int T = kv_c.size(0), LATENT = kv_c.size(1), rope_dim = k_pe.size(1);
    const __nv_bfloat16* w = norm_weight ? bp(*norm_weight) : nullptr;
    #define LAUNCH(L) mla_kv_insert<L><<<T, 32, 0, stream()>>>(bp(kv_c), bp(k_pe), bp(cosb), \
        bp(sinb), positions.data_ptr<int>(), slots.data_ptr<int64_t>(), bpm(kv_cache), \
        int(block_size), rope_dim, int(norm_mode), float(eps), w)
    if (LATENT == 128) LAUNCH(128); else if (LATENT == 256) LAUNCH(256);
    else if (LATENT == 512) LAUNCH(512);
    else TORCH_CHECK(false, "LATENT must be 128/256/512");
    #undef LAUNCH
}
static torch::Tensor py_mla_decode_partition(torch::Tensor q, torch::Tensor kv_cache,
        torch::Tensor bt, torch::Tensor ctx, int64_t block_size, double scale,
        int64_t partition_size, int64_t max_context) {
    CK(q); CK(kv_cache); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "mla_decode_partition expects QK=576");
    const int P = int((max_context + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    auto out = torch::empty({B, H, 512}, q.options());
    mla_decode_partition<512, 64><<<dim3(H, B, P), 32, 0, stream()>>>(bp(q), bp(kv_cache),
        bt.data_ptr<int>(), ctx.data_ptr<int>(), tmp.data_ptr<float>(), ml.data_ptr<float>(),
        es.data_ptr<float>(), int(block_size), int(bt.size(1)), float(scale), H, P,
        int(partition_size));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}
static torch::Tensor py_mla_decode_fp8_partition(torch::Tensor q, torch::Tensor data,
        torch::Tensor scl, torch::Tensor bt, torch::Tensor ctx, int64_t block_size,
        double scale, int64_t partition_size, int64_t max_context) {
    CK(q); CK(data); CK(scl); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1);
    const int P = int((max_context + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    auto out = torch::empty_like(q);
    mla_decode_fp8_v<false, true><<<dim3(H, B, P), 32, 0, stream()>>>(bp(q),
        data.data_ptr<uint8_t>(), scl.data_ptr<uint8_t>(), bt.data_ptr<int>(),
        ctx.data_ptr<int>(), nullptr, nullptr, 0, nullptr, tmp.data_ptr<float>(),
        ml.data_ptr<float>(), es.data_ptr<float>(), int(block_size), int(bt.size(1)),
        float(scale), H, P, int(partition_size));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}
// Native replacement for the DSA indexer's Triton metadata kernel.
static void py_indexer_metadata(torch::Tensor query_start_loc,
        torch::Tensor uncompressed_seq_lens, torch::Tensor cu_compressed_seq_lens,
        torch::Tensor row_start_cu, torch::Tensor token_to_seq,
        torch::Tensor cu_ks, torch::Tensor cu_ke, int64_t query_slice_start,
        int64_t query_slice_stop, int64_t dcp_rank, int64_t dcp_world,
        int64_t dcp_interleave, int64_t compress_ratio) {
    CK(query_start_loc); CK(uncompressed_seq_lens); CK(cu_compressed_seq_lens);
    CK(row_start_cu); CK(token_to_seq); CK(cu_ks); CK(cu_ke);
    const int num_reqs = (int)query_start_loc.size(0) - 1;
    if (num_reqs <= 0) return;
    indexer_metadata<<<num_reqs, 256, 0, stream()>>>(
        query_start_loc.data_ptr<int>(), uncompressed_seq_lens.data_ptr<int>(),
        cu_compressed_seq_lens.data_ptr<int>(), row_start_cu.data_ptr<int>(),
        token_to_seq.data_ptr<int>(), cu_ks.data_ptr<int>(), cu_ke.data_ptr<int>(),
        (int)query_slice_start, (int)query_slice_stop, (int)dcp_rank,
        (int)dcp_world, (int)dcp_interleave, (int)compress_ratio);
}

// Native replacement for vLLM's Triton _compute_slot_mapping_kernel.
static void py_compute_slot_mapping(torch::Tensor query_start_loc,
        torch::Tensor positions, torch::Tensor block_table,
        torch::Tensor slot_mapping, int64_t num_tokens, int64_t max_num_tokens,
        int64_t block_size, int64_t kv_cache_block_size,
        int64_t blocks_per_kv_block, int64_t cp_world, int64_t cp_rank,
        int64_t cp_interleave, int64_t pad_id) {
    CK(query_start_loc); CK(positions); CK(block_table); CK(slot_mapping);
    TORCH_CHECK(positions.scalar_type() == torch::kLong, "positions must be int64");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kLong, "slot_mapping int64");
    const int num_reqs = (int)query_start_loc.size(0) - 1;
    compute_slot_mapping<<<num_reqs + 1, 256, 0, stream()>>>(
        (long)num_tokens, (long)max_num_tokens, query_start_loc.data_ptr<int>(),
        reinterpret_cast<const long*>(positions.data_ptr()),
        block_table.data_ptr<int>(), (int)block_table.size(1), (int)block_size,
        reinterpret_cast<long*>(slot_mapping.data_ptr()),
        (int)kv_cache_block_size, (int)blocks_per_kv_block, (int)cp_world,
        (int)cp_rank, (int)cp_interleave, (long)pad_id);
}

// DSA indexer decode logits over the paged fp8 K cache. Replaces
// fp8_paged_mqa_logits_torch, which loops the batch in Python and calls
// .item() -- a host sync that CUDA graph capture rejects outright.
static torch::Tensor py_fp8_paged_mqa_logits(torch::Tensor q,
        torch::Tensor kv_cache, torch::Tensor weights,
        torch::Tensor context_lens, torch::Tensor block_tables,
        int64_t max_model_len) {
    CK(q); CK(kv_cache); CK(weights); CK(context_lens); CK(block_tables);
    TORCH_CHECK(weights.scalar_type() == torch::kFloat, "weights must be fp32");
    TORCH_CHECK(context_lens.scalar_type() == torch::kInt, "context_lens int32");
    TORCH_CHECK(block_tables.scalar_type() == torch::kInt, "block_tables int32");
    constexpr int D = 128, NT = 64, NWARP = 4;
    const int B = (int)q.size(0), H = (int)q.size(2);
    TORCH_CHECK(q.size(3) == D, "indexer head_dim must be ", D);
    TORCH_CHECK(q.size(1) == 1, "paged logits path is next_n==1 (decode)");
    TORCH_CHECK(H <= 16 * NWARP, "H exceeds heads per block");
    const int block_size = (int)kv_cache.size(1);

    auto out = torch::full({B, (long)max_model_len},
                           -std::numeric_limits<float>::infinity(),
                           q.options().dtype(torch::kFloat));
    const size_t smem = tms::indexer_paged_mqa_logits_smem<D, NT, NWARP>();
    auto kern = tms::indexer_paged_mqa_logits<D, NT, NWARP>;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    dim3 grid((unsigned)((max_model_len + NT - 1) / NT), (unsigned)B);
    kern<<<grid, 32 * NWARP, smem, stream()>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(kv_cache.data_ptr()),
        weights.data_ptr<float>(), context_lens.data_ptr<int>(),
        block_tables.data_ptr<int>(), out.data_ptr<float>(), H, block_size,
        (int)block_tables.size(1), (int)max_model_len);
    return out;
}

// Non-paged (prefill) indexer logits: the tensor-core kernel from
// indexer_logits_mma.cuh, which was validated standalone but never bound.
static torch::Tensor py_fp8_mqa_logits(torch::Tensor q, torch::Tensor k,
        torch::Tensor kscale, torch::Tensor weights, torch::Tensor ks,
        torch::Tensor ke) {
    CK(q); CK(k); CK(kscale); CK(weights); CK(ks); CK(ke);
    constexpr int D = 128, NT = 64, NWARP = 4;
    const int M = (int)q.size(0), H = (int)q.size(1), N = (int)k.size(0);
    TORCH_CHECK(q.size(2) == D, "indexer head_dim must be ", D);
    TORCH_CHECK(H <= 16 * NWARP, "H exceeds heads per block");
    auto out = torch::empty({M, N}, q.options().dtype(torch::kFloat));
    const size_t smem = tms::indexer_mqa_logits_mma_smem<D, NT, NWARP>();
    auto kern = tms::indexer_mqa_logits_mma<D, NT, NWARP>;
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)smem);
    dim3 grid((unsigned)((N + NT - 1) / NT), (unsigned)M);
    kern<<<grid, 32 * NWARP, smem, stream()>>>(
        reinterpret_cast<const uint8_t*>(q.data_ptr()),
        reinterpret_cast<const uint8_t*>(k.data_ptr()),
        kscale.data_ptr<float>(), weights.data_ptr<float>(),
        ks.data_ptr<int>(), ke.data_ptr<int>(), out.data_ptr<float>(), M, N, H);
    return out;
}

// GLM-5.2-Vision, bf16 cache. NFP8=0 means every element is read as bf16, so a
// slot is 576 bf16 = 1152 B. Ampere (sm80) has no native fp8e4nv, so vLLM cannot
// store an fp8 KV cache there -- this is the geometry that actually runs on A100.
static torch::Tensor py_mla_decode_bf16_sparse_glm(torch::Tensor q, torch::Tensor kv,
        torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale) {
    CK(q); CK(kv); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "GLM MLA expects q width 576, got ", q.size(2));
    const int max_topk = indices.size(1);
    auto out = torch::empty({B, H, 512}, q.options());
    mla_decode_fp8_v<true, false, 576, 512, 0, 0><<<dim3(H, B), 32, 0, stream()>>>(
        bp(q), reinterpret_cast<const uint8_t*>(kv.data_ptr()), nullptr,
        bt.data_ptr<int>(), nullptr, indices.data_ptr<int>(),
        topk_length.data_ptr<int>(), max_topk, bpm(out), nullptr, nullptr, nullptr,
        int(block_size), int(bt.size(1)), float(scale), H, 1, 0, 1.0f);
    return out;
}

// GLM-5.2-Vision geometry: q/cache slot are 576 fp8 elements, the value is the
// leading 512, and the cache carries one per-tensor kv_scale (vLLM's fp8 MLA
// layout) rather than per-64 e8 exponents. `indices` are request-local logical
// token positions resolved through `bt`, which is exactly what vLLM's
// topk_indices_buffer already holds -- no global-index conversion needed.
static torch::Tensor py_mla_decode_fp8_sparse_glm(torch::Tensor q, torch::Tensor data,
        torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale, double kv_scale,
        int64_t partition_size) {
    CK(q); CK(data); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    TORCH_CHECK(q.size(2) == 576, "GLM MLA expects q width 576, got ", q.size(2));
    const int max_topk = indices.size(1);
    auto out = torch::empty({B, H, 512}, q.options());
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false, 576, 512, 576, 1><<<dim3(H, B), 32, 0, stream()>>>(
            bp(q), data.data_ptr<uint8_t>(), nullptr, bt.data_ptr<int>(), nullptr,
            indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, bpm(out),
            nullptr, nullptr, nullptr, int(block_size), int(bt.size(1)), float(scale), H, 1, 0,
            float(kv_scale));
        return out;
    }
    // Partitioned: the unpartitioned launch is one 32-thread warp per
    // (head, token) walking the whole index list serially -- 64 warps across
    // 108 SMs at B=1, which profiled at 48% of ALL decode GPU time. Splitting
    // the list over blockIdx.z multiplies the exposed parallelism by P and the
    // reduce merges the online-softmax partials exactly.
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true, 576, 512, 576, 1><<<dim3(H, B, P), 32, 0, stream()>>>(
        bp(q), data.data_ptr<uint8_t>(), nullptr, bt.data_ptr<int>(), nullptr,
        indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size),
        float(kv_scale));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}

static torch::Tensor py_mla_decode_fp8_sparse(torch::Tensor q, torch::Tensor data,
        torch::Tensor scl, torch::Tensor bt, torch::Tensor indices, torch::Tensor topk_length,
        int64_t block_size, double scale, int64_t partition_size) {
    CK(q); CK(data); CK(scl); CK(bt); CK(indices); CK(topk_length);
    const int B = q.size(0), H = q.size(1);
    const int max_topk = indices.size(1);
    auto out = torch::empty_like(q);
    if (partition_size <= 0) {
        mla_decode_fp8_v<true, false><<<dim3(H, B), 32, 0, stream()>>>(bp(q),
            data.data_ptr<uint8_t>(), scl.data_ptr<uint8_t>(), bt.data_ptr<int>(), nullptr,
            indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, bpm(out),
            nullptr, nullptr, nullptr, int(block_size), int(bt.size(1)), float(scale), H, 1, 0);
        return out;
    }
    const int P = int((max_topk + partition_size - 1) / partition_size);
    auto opts = q.options().dtype(torch::kFloat);
    auto tmp = torch::empty({B, H, P, 512}, opts);
    auto ml = torch::empty({B, H, P}, opts);
    auto es = torch::empty({B, H, P}, opts);
    mla_decode_fp8_v<true, true><<<dim3(H, B, P), 32, 0, stream()>>>(bp(q),
        data.data_ptr<uint8_t>(), scl.data_ptr<uint8_t>(), bt.data_ptr<int>(), nullptr,
        indices.data_ptr<int>(), topk_length.data_ptr<int>(), max_topk, nullptr,
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(),
        int(block_size), int(bt.size(1)), float(scale), H, P, int(partition_size));
    paged_attention_reduce<__nv_bfloat16, 512><<<dim3(H, B), 32, 0, stream()>>>(
        tmp.data_ptr<float>(), ml.data_ptr<float>(), es.data_ptr<float>(), bpm(out), H, P);
    return out;
}
static torch::Tensor py_paged_attention_gqa_staged(torch::Tensor q, torch::Tensor kc,
        torch::Tensor vc, torch::Tensor bt, torch::Tensor ctx, int64_t block_size,
        double scale, int64_t num_kv_heads) {
    CK(q); CK(kc); CK(vc); CK(bt); CK(ctx);
    const int B = q.size(0), H = q.size(1), D = q.size(2);
    const int gs = H / int(num_kv_heads);
    auto out = torch::empty_like(q);
    dim3 grid{unsigned(num_kv_heads), unsigned(B)};
    #define LAUNCH(DD) paged_attention_gqa_staged<half, DD><<<grid, 32 * gs, 0, stream()>>>( \
        hp(q), hp(kc), hp(vc), bt.data_ptr<int>(), ctx.data_ptr<int>(), hpm(out), \
        int(block_size), int(bt.size(1)), float(scale), H, int(num_kv_heads))
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}
static torch::Tensor py_attn_window(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                                    double scale, int64_t window) {
    CK(q); CK(k); CK(v);
    const int B = q.size(0), H = q.size(1), N = q.size(2), D = q.size(3);
    auto out = torch::empty_like(q);
    dim3 grid{unsigned(N), unsigned(B * H)};
    #define LAUNCH(DD) attn_window<half, DD><<<grid, 32, 0, stream()>>>( \
        hp(q), hp(k), hp(v), hpm(out), N, float(scale), int(window))
    if (D == 64) LAUNCH(64); else if (D == 128) LAUNCH(128);
    else TORCH_CHECK(false, "D must be 64/128");
    #undef LAUNCH
    return out;
}

void init_serving(py::module_& m) {
    m.def("mla_kv_insert", &py_mla_kv_insert, py::arg("kv_c"), py::arg("k_pe"), py::arg("cos"),
          py::arg("sin"), py::arg("positions"), py::arg("slot_mapping"), py::arg("kv_cache"),
          py::arg("block_size"), py::arg("norm_mode") = 0, py::arg("eps") = 1e-6,
          py::arg("norm_weight") = py::none());
    m.def("mla_decode_partition", &py_mla_decode_partition);
    m.def("mla_decode_fp8_partition", &py_mla_decode_fp8_partition);
    m.def("indexer_metadata", &py_indexer_metadata,
          py::arg("query_start_loc"), py::arg("uncompressed_seq_lens"),
          py::arg("cu_compressed_seq_lens"), py::arg("row_start_cu"),
          py::arg("token_to_seq"), py::arg("cu_ks"), py::arg("cu_ke"),
          py::arg("query_slice_start"), py::arg("query_slice_stop"),
          py::arg("dcp_rank"), py::arg("dcp_world"),
          py::arg("dcp_interleave"), py::arg("compress_ratio"));
    m.def("fp8_paged_mqa_logits", &py_fp8_paged_mqa_logits,
          py::arg("q"), py::arg("kv_cache"), py::arg("weights"),
          py::arg("context_lens"), py::arg("block_tables"),
          py::arg("max_model_len"));
    m.def("fp8_mqa_logits", &py_fp8_mqa_logits,
          py::arg("q"), py::arg("k"), py::arg("kscale"), py::arg("weights"),
          py::arg("ks"), py::arg("ke"));
    m.def("compute_slot_mapping", &py_compute_slot_mapping,
          py::arg("query_start_loc"), py::arg("positions"),
          py::arg("block_table"), py::arg("slot_mapping"),
          py::arg("num_tokens"), py::arg("max_num_tokens"),
          py::arg("block_size"), py::arg("kv_cache_block_size"),
          py::arg("blocks_per_kv_block"), py::arg("cp_world"),
          py::arg("cp_rank"), py::arg("cp_interleave"), py::arg("pad_id"));
    m.def("mla_decode_bf16_sparse_glm", &py_mla_decode_bf16_sparse_glm, py::arg("q"),
          py::arg("kv"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"));
    m.def("mla_decode_fp8_sparse_glm", &py_mla_decode_fp8_sparse_glm, py::arg("q"),
          py::arg("data"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"),
          py::arg("kv_scale"), py::arg("partition_size") = 0);
    m.def("mla_decode_fp8_sparse", &py_mla_decode_fp8_sparse, py::arg("q"), py::arg("data"),
          py::arg("scale_cache"), py::arg("block_table"), py::arg("indices"),
          py::arg("topk_length"), py::arg("block_size"), py::arg("scale"),
          py::arg("partition_size") = 0);
    m.def("paged_attention_gqa_staged", &py_paged_attention_gqa_staged);
    m.def("attn_window", &py_attn_window, py::arg("q"), py::arg("k"), py::arg("v"),
          py::arg("scale"), py::arg("window") = 0);
    m.def("kv_scatter", &py_kv_scatter);
    m.def("kv_gather", &py_kv_gather);
    m.def("copy_blocks", &py_copy_blocks);
    m.def("paged_attention", &py_paged_attention,
          py::arg("q"), py::arg("key_cache"), py::arg("value_cache"), py::arg("block_table"),
          py::arg("context_lens"), py::arg("block_size"), py::arg("scale"), py::arg("num_kv_heads"),
          py::arg("alibi") = py::none(), py::arg("block_mask") = py::none(), py::arg("window") = 0);
    m.def("paged_attention_v2", &py_paged_attention_v2,
          py::arg("q"), py::arg("key_cache"), py::arg("value_cache"), py::arg("block_table"),
          py::arg("context_lens"), py::arg("block_size"), py::arg("scale"), py::arg("num_kv_heads"),
          py::arg("partition_size"), py::arg("max_context"), py::arg("window") = 0);
    m.def("rope_kv_insert", &py_rope_kv_insert,
          py::arg("k"), py::arg("v"), py::arg("cos"), py::arg("sin"), py::arg("positions"),
          py::arg("slot_mapping"), py::arg("key_cache"), py::arg("value_cache"),
          py::arg("num_kv_heads"), py::arg("block_size"), py::arg("norm_weight") = py::none(),
          py::arg("gemma") = false, py::arg("eps") = 1e-6);
    m.def("rope_q", &py_rope_q,
          py::arg("q"), py::arg("cos"), py::arg("sin"), py::arg("positions"), py::arg("num_heads"),
          py::arg("norm_weight") = py::none(), py::arg("gemma") = false, py::arg("eps") = 1e-6);
    m.def("attn_q", &py_attn_q);
    m.def("mla_decode", &py_mla_decode);
    m.def("mla_decode_fp8", &py_mla_decode_fp8);
    m.def("mla_kv_insert_fp8", &py_mla_kv_insert_fp8);
    m.def("mla_q_norm_rope", &py_mla_q_norm_rope,
          py::arg("q"), py::arg("cos"), py::arg("sin"), py::arg("positions"), py::arg("num_heads"),
          py::arg("nope_dim"), py::arg("rope_dim"), py::arg("norm_mode") = 0, py::arg("eps") = 1e-6,
          py::arg("norm_weight") = py::none());
    m.def("sample", &py_sample, py::arg("logits"), py::arg("mode"), py::arg("seed") = 0,
          py::arg("temperature") = 1.0, py::arg("param") = 0.9, py::arg("k") = 40);
    m.def("apply_penalties", &py_apply_penalties);
    m.def("apply_token_bitmask", &py_apply_token_bitmask);
    m.def("apply_bad_words", &py_apply_bad_words);
    m.def("beam_advance", &py_beam_advance);
    m.def("spec_verify_linear", &py_spec_verify_linear);
    m.def("build_dynamic_tree", &py_build_dynamic_tree);
    m.def("spec_verify_tree", &py_spec_verify_tree);
    m.def("spec_compact", &py_spec_compact);
    m.def("spec_update_kv_meta", &py_spec_update_kv_meta);
}
