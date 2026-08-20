#pragma once

// QuixiCore XPU op ABI.
//
// This is the framework-agnostic launch surface for the backend: every op is a
// free function that takes a sycl::queue plus raw device pointers and shape
// metadata. It is the XPU analogue of the Metal backend's `tk::launch_*` layer.
//
// Both callers use the SAME entry points:
//   * the native C++ helpers (which allocate USM and manage a queue), and
//   * the PyTorch-XPU binding (which passes a Torch tensor's data pointer and
//     the tensor's own sycl::queue).
//
// Contract names and semantics are shared with the other QuixiCore backends;
// Intel-specific layout / subgroup / XMX choices stay inside the kernel
// variants under kernels/<family>/<operation>/variants/.
//
// Requires a SYCL toolchain; only compiled under QUIXICORE_XPU_ENABLE_SYCL.

#include <cstddef>
#include <cstdint>

#include <sycl/sycl.hpp>

#include "quixicore/xpu/runtime.hpp"
#include "quixicore/xpu/variants.hpp"

namespace quixicore::xpu::ops {

// KV-cache element encoding for the paged attention ops.
enum class KvCacheDType {
  same,      // cache elements are the activation dtype
  fp8_e4m3,  // u8 e4m3 with device-scalar k/v scales
  fp8_e5m2,
};


// ----------------------------------------------------------------------------
// activations
// ----------------------------------------------------------------------------

// GELU approximation selector. `erf` is the exact Gaussian error function form
// 0.5*x*(1+erf(x/sqrt2)); `tanh` is the tanh approximation used by many LLMs.
enum class GeluApprox {
  erf,
  tanh,
};

// Elementwise GELU over `n` contiguous elements. `in` and `out` are device
// pointers of dtype `dt` (may alias for in-place). Computes in fp32.
//
// `variant` selects the native SYCL or vendor (oneDNN) implementation; both are
// shipped and produce results within the same contract tolerance. When
// `blocking` is true the call waits for completion; otherwise the caller owns
// synchronization.
void gelu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          GeluApprox approx = GeluApprox::erf, Variant variant = Variant::sycl,
          bool blocking = true);

// Fused SwiGLU + activation quantization modes.
enum class GluQuantMode {
  group_fp8,  // per-`group` dynamic e4m3 with fp32 scales [rows, d/group]
  mxfp4,      // per-32-group fp32 power-of-two scale + packed e2m1 nibbles
};

// Fused SwiGLU + quantize. x is [rows, 2*d] (gate half then value half, the
// glu layout) of dtype dt; y = silu(gate) * value in fp32. out_q is u8
// [rows, d] (group_fp8) or [rows, d/2] (mxfp4, element 2i in the low
// nibble); out_scales is fp32 [rows, d/group] (group must divide d; mxfp4
// forces group 32). Scale rules match norm_quant's dynamic modes. Encode
// steps are integer-exact. Out-of-envelope shapes are rejected without
// launching.
void glu_quant(sycl::queue& q, const void* x, std::uint8_t* out_q,
               float* out_scales, std::size_t rows, std::size_t d,
               std::size_t group, GluQuantMode mode, DType dt,
               Variant variant = Variant::sycl, bool blocking = true);

// Numerically stable softmax over the last axis of a [rows, dim] row-major
// tensor: subtract the row max, exponentiate, normalize by the row sum. `x`,
// `out` are device pointers of dtype `dt` ([rows*dim]); exp/sum accumulate in
// fp32.
void softmax(sycl::queue& q, const void* x, void* out, std::size_t rows,
             std::size_t dim, DType dt, Variant variant = Variant::sycl,
             bool blocking = true);

// SiLU / swish: out[i] = x[i] * sigmoid(x[i]). Elementwise over `n`.
void silu(sycl::queue& q, const void* in, void* out, std::size_t n, DType dt,
          Variant variant = Variant::sycl, bool blocking = true);

// GELU backward: grad_in[i] = grad_out[i] * gelu'(x[i]). `approx` selects the
// erf or tanh derivative to match the forward. Elementwise over `n`.
void gelu_backward(sycl::queue& q, const void* grad_out, const void* x,
                   void* grad_in, std::size_t n, DType dt,
                   GeluApprox approx = GeluApprox::erf,
                   Variant variant = Variant::sycl, bool blocking = true);

// Gated linear unit variants. Input `x` is [rows, 2*d] row-major (gate half then
// value half); output is [rows, d]: out[r,i] = act(x[r,i]) * x[r,d+i], where act
// is silu (swiglu), gelu (geglu), relu (reglu), or sigmoid (glu).
enum class GluMode {
  swiglu,
  geglu,
  reglu,
  glu,
};

void glu(sycl::queue& q, const void* x, void* out, std::size_t rows,
         std::size_t d, DType dt, GluMode mode = GluMode::swiglu,
         Variant variant = Variant::sycl, bool blocking = true);


// GEGLU with a fused f16 output. Input `x` is [rows, 2*d] row-major (gate half
// then value half, the same layout as glu); `out` is [rows, d] and always f16:
//   out[r,i] = f16( gelu_tanh(x[r,i]) * x[r,d+i] )
// The gate uses the tanh GELU approximation (matching the embeddinggemma.c FFN
// source); compute is fp32. This is the f16-context variant of glu (cf.
// attention_f16ctx), folding the dt->f16 convert a downstream f16 GEMM needs
// into the activation. Shape: GEGLU -> f16.
void glu_gelu_f16(sycl::queue& q, const void* x, void* out, std::size_t rows,
                  std::size_t d, DType dt, Variant variant = Variant::sycl,
                  bool blocking = true);

// ----------------------------------------------------------------------------
// attention
// ----------------------------------------------------------------------------

// Rotary position embedding (RoPE), NeoX half-split form. `x`, `out` are
// [tokens, n_heads, head_dim] row-major of dtype `dt`; token t uses position
// (pos0 + t). Rotates pairs (i, i + head_dim/2). head_dim must be even.
void rope(sycl::queue& q, const void* x, void* out, std::size_t tokens,
          std::size_t n_heads, std::size_t head_dim, float base,
          std::size_t pos0, DType dt, Variant variant = Variant::sycl,
          bool blocking = true);

// LSE-weighted merge of two partial attention results (split-KV / prefix +
// suffix; consumes paged_attention_prefill's LSE). out_a/out_b/out are
// [rows, d] of dtype dt, lse_a/lse_b/lse_out (nullable) [rows] f32. A
// partition at -inf contributes nothing; all-empty rows give zero output
// and -inf lse. Graph-capture-safe.
void merge_attn_states(sycl::queue& q, const void* out_a, const float* lse_a,
                       const void* out_b, const float* lse_b, void* out,
                       float* lse_out, std::size_t rows, std::size_t d,
                       DType dt, Variant variant = Variant::sycl,
                       bool blocking = true);

// Multimodal rotary embedding (M-RoPE, Qwen2-VL family). positions is
// [n_sections, tokens] int64 — each rotary pair takes its cos/sin row from
// the position axis owning it per the cumulative `sections` widths
// ([n_sections] int32 device array summing to rot_dim/2, <= 4 sections).
// cos_sin_cache is [max_pos, rot_dim] of dtype dt, cos half then sin half.
// query [tokens, n_heads, head_size] and key [tokens, n_kv_heads, head_size]
// rotate IN PLACE over rot_dim (NeoX half-split or GPT-J interleaved); the
// tail past rot_dim is untouched. Graph-capture-safe.
void mrope(sycl::queue& q, void* query, void* key, const void* cos_sin_cache,
           const std::int64_t* positions, const std::int32_t* sections,
           std::size_t n_sections, std::size_t tokens, std::size_t n_heads,
           std::size_t n_kv_heads, std::size_t head_size, std::size_t rot_dim,
           bool neox, DType dt, Variant variant = Variant::sycl,
           bool blocking = true);

// Positioned RoPE with an explicit per-token position array (contract
// rotary_positioned): the single-section form of mrope — positions is
// [tokens] int64, both mask conventions, same in-place layout rules.
void rotary_positioned(sycl::queue& q, void* query, void* key,
                       const void* cos_sin_cache,
                       const std::int64_t* positions, std::size_t tokens,
                       std::size_t n_heads, std::size_t n_kv_heads,
                       std::size_t head_size, std::size_t rot_dim, bool neox,
                       DType dt, Variant variant = Variant::sycl,
                       bool blocking = true);

// Flash-style scaled dot-product attention (online softmax; no materialized
// score matrix). Q is [n_heads, seq_q, d]; K, V are [n_kv_heads, seq_k, d]
// (GQA: q head h uses kv head h / (n_heads/n_kv_heads)); O is [n_heads, seq_q,
// d], dtype dt. scale = 1/sqrt(d). `causal` masks key positions k > q (aligned
// at the sequence end when seq_q==seq_k). fp32 accumulation. head_dim d <= 128.
void attention(sycl::queue& q, const void* Q, const void* K, const void* V,
               void* O, std::size_t n_heads, std::size_t n_kv_heads,
               std::size_t seq_q, std::size_t seq_k, std::size_t d, bool causal,
               DType dt, Variant variant = Variant::sycl, bool blocking = true);

// Flash-style SDPA with a fused f16 context store. Same contract and math as
// attention() above, but writes the output twice in one epilogue: O in dtype dt
// and O_f16 in f16 (same [n_heads, seq_q, d] layout). This folds the ctx->f16
// convert a downstream attention-output GEMM needs into the attention pass,
// removing a standalone O-sized convert kernel. O_f16 equals the f16 rounding of
// O. Shape: online-attention + fused f16 context store.
void attention_f16ctx(sycl::queue& q, const void* Q, const void* K,
                      const void* V, void* O, void* O_f16, std::size_t n_heads,
                      std::size_t n_kv_heads, std::size_t seq_q,
                      std::size_t seq_k, std::size_t d, bool causal, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// Symmetric sliding-window attention (SWA). Same contract, layout, and flash
// online-softmax math as attention() above, but the mask is a SYMMETRIC band
// rather than causal: query qi attends key positions in [center - window/2,
// center + window/2] clamped to [0, seq_k), where center = qi + (seq_k - seq_q)
// end-aligns the query into the key axis (center == qi when seq_q == seq_k).
// Unlike causal sliding-window attention the band looks BOTH forward and
// backward within +-window/2 -- the mask EmbeddingGemma's local encoder layers
// use in its alternating global/local stack. window == 0 means dense (attend
// all keys), identical to attention() with causal == false. head_dim d <= 256.
// Shape: symmetric sliding-window, GQA, D<=256.
void attn_swa(sycl::queue& q, const void* Q, const void* K, const void* V,
              void* O, std::size_t n_heads, std::size_t n_kv_heads,
              std::size_t seq_q, std::size_t seq_k, std::size_t d,
              std::size_t window, DType dt, Variant variant = Variant::sycl,
              bool blocking = true);


// Fused per-head QK-norm + RoPE (+ optional f16 convert). For every (token,
// head) of Q and K: RMS-normalize the head-dim vector by its learned weight,
// scale the query heads by `query_scale` (key heads by 1), then apply NeoX
// half-split RoPE at position (pos0 + token). Q is [tokens, n_head, head_dim],
// K is [tokens, n_head_kv, head_dim] row-major (GQA when n_head_kv < n_head),
// both dtype dt and updated in place; `q_weight` / `k_weight` are [head_dim].
// When `Q_f16` / `K_f16` are non-null (f16, same layout) the rotated result is
// also written there (fused convert for a downstream f16 QK GEMM); pass null to
// skip. head_dim must be even; fp32 accumulation. Collapses the per-head
// rms_norm(Q) + rms_norm(K) + query-scale + rope(Q) + rope(K) chain into one
// launch. Shape: per-head RMSNorm + query-scale + RoPE.
void qk_norm_rope(sycl::queue& q, void* Q, void* K, const void* q_weight,
                  const void* k_weight, void* Q_f16, void* K_f16,
                  std::size_t tokens, std::size_t n_head, std::size_t n_head_kv,
                  std::size_t head_dim, float base, std::size_t pos0,
                  float query_scale, float eps, DType dt,
                  Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// optimizers
// ----------------------------------------------------------------------------

// Fused AdamW, in-place. `p` (params), `m`, `v` (moments) are updated; `g` is
// the gradient. All are [n] of dtype `dt`. `step` is 1-based (bias correction).
void adamw(sycl::queue& q, void* p, const void* g, void* m, void* v,
           std::size_t n, float lr, float beta1, float beta2, float eps,
           float weight_decay, int step, DType dt,
           Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// sampling
// ----------------------------------------------------------------------------

// Greedy argmax over the last axis. `logits` is [rows, vocab] of dtype `dt`;
// `out` is [rows] int32 (lowest index on ties).
void argmax(sycl::queue& q, const void* logits, int* out, std::size_t rows,
            std::size_t vocab, DType dt, Variant variant = Variant::sycl,
            bool blocking = true);

// Categorical sampling from temperature-scaled softmax(logits). `logits`
// [rows, vocab] dt; `out` [rows] int32. `seed` drives the stateless RNG (one
// uniform per row). temperature -> 0 reduces to argmax.
void sample_categorical(sycl::queue& q, const void* logits, int* out,
                        std::size_t rows, std::size_t vocab, float temperature,
                        std::uint32_t seed, DType dt,
                        Variant variant = Variant::sycl, bool blocking = true);

// Top-k sampling: restrict to the k highest logits, softmax over them
// (temperature), then sample. `out` [rows] int32 is always one of the row's
// top-k tokens.
void top_k_sample(sycl::queue& q, const void* logits, int* out, std::size_t rows,
                  std::size_t vocab, int k, float temperature,
                  std::uint32_t seed, DType dt, Variant variant = Variant::sycl,
                  bool blocking = true);

// ----------------------------------------------------------------------------
// serving (kv cache, embedding)
// ----------------------------------------------------------------------------

// Embedding lookup: out[t, :] = table[ids[t], :]. `table` is [vocab, dim] dtype
// `dt`, `ids` is [n] int32, `out` is [n, dim] dtype `dt`.
void embedding_lookup(sycl::queue& q, const void* table, const int* ids,
                      void* out, std::size_t n, std::size_t dim, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// Top-k probability renormalization (contract top_k_renorm): keep the k
// largest probabilities per row (ties at the k-th value included), zero the
// rest, renormalize to sum 1. probs/out are [rows, vocab] of dtype dt
// (out-of-place; may alias). k <= 64. Rows with zero kept mass pass through.
void top_k_renorm(sycl::queue& q, const void* probs, void* out,
                  std::size_t rows, std::size_t vocab, int k, DType dt,
                  Variant variant = Variant::sycl, bool blocking = true);

// Top-p (nucleus) renormalization (contract top_p_renorm): keep the minimal
// high-probability set whose mass reaches top_p (threshold binary search,
// ties included), renormalize. Same layout rules as top_k_renorm.
void top_p_renorm(sycl::queue& q, const void* probs, void* out,
                  std::size_t rows, std::size_t vocab, float top_p, DType dt,
                  Variant variant = Variant::sycl, bool blocking = true);

// KV-cache scatter: cache[slots[t], :] = src[t, :]. `cache` is [max_slots, row],
// `src` is [n, row], `slots` is [n] int32 (a negative slot skips the row).
// `row` = n_heads * head_dim (or any contiguous row width), dtype `dt`.
void kv_cache_scatter(sycl::queue& q, void* cache, const void* src,
                      const int* slots, std::size_t n, std::size_t row, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// KV-cache gather: out[i, :] = cache[idx[i], :]. Inverse of scatter.
void kv_cache_gather(sycl::queue& q, const void* cache, const int* idx,
                     void* out, std::size_t n, std::size_t row, DType dt,
                     Variant variant = Variant::sycl, bool blocking = true);

// Paged KV-cache write — the write side of paged_attention_* (vLLM
// reshape_and_cache_flash semantics). key/value are [n_tokens, n_kv_heads, d]
// of dtype dt; caches are the paged layout the attention ops read
// ([n_pages, page_size, n_kv_heads, d], page_stride_elems pitch);
// slot_mapping[t] is the FLAT slot (page * page_size + offset), int64, < 0
// skips. kv_dt fp8_e4m3 divides by the device-scalar k/v scales before the
// integer-exact e4m3 encode (the attention decode multiplies them back).
// Graph-capture-safe.
void kv_cache_scatter_paged(
    sycl::queue& q, const void* key, const void* value, void* k_cache,
    void* v_cache, const std::int64_t* slot_mapping, std::size_t n_tokens,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    KvCacheDType kv_dt, DType dt, Variant variant = Variant::sycl,
    bool blocking = true);

// Paged KV-cache gather — inverse of kv_cache_scatter_paged with optional
// fp8 dequant (x device-scalar scales); slots [n] int64 flat, < 0 gathers
// zeros. k_out/v_out are [n, n_kv_heads, d] of dtype dt.
void kv_cache_gather_paged(
    sycl::queue& q, const void* k_cache, const void* v_cache, void* k_out,
    void* v_out, const std::int64_t* slots, std::size_t n,
    std::size_t n_kv_heads, std::size_t d, std::size_t page_size,
    std::size_t page_stride_elems, const float* k_scale, const float* v_scale,
    KvCacheDType kv_dt, DType dt, Variant variant = Variant::sycl,
    bool blocking = true);

// Sentence-embedding pooling head: masked mean-pool over each sequence's tokens
// with a per-token RMSNorm (learned weight) folded in, then L2-normalize. `x` is
// [total_tokens, dim] row-major; sequence s owns rows [offsets[s], offsets[s+1])
// (`offsets` is [batch+1] int32, monotonic). `weight` is [dim], `out` is
// [batch, dim], all dtype dt. For sequence s over token range [a, b):
//   r_t   = x[t] * rsqrt(mean_d(x[t,d]^2) + eps) * weight   (RMSNorm, per token)
//   m     = (1/(b-a)) * sum_t r_t                           (masked mean)
//   out[s]= m * rsqrt(sum_d m[d]^2)                         (L2; 0-vector passes)
// `dim` is the shape key {256,512,768,1024}; fp32 accumulation. An empty
// sequence (b==a) yields a zero vector. Shape: masked-mean + per-token RMSNorm +
// L2 pooling head. Native-only.
void pool_mean_rms_l2(sycl::queue& q, const void* x, const void* weight,
                      const int* offsets, void* out, std::size_t batch,
                      std::size_t dim, float eps, DType dt,
                      Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// utils
// ----------------------------------------------------------------------------

// Inverted dropout: out[i] = uniform(seed,i) < p ? 0 : in[i]/(1-p), over `n`.
void dropout(sycl::queue& q, const void* in, void* out, std::size_t n, float p,
             std::uint32_t seed, DType dt, Variant variant = Variant::sycl,
             bool blocking = true);

// Per-row cross-entropy loss from logits: loss[r] = logsumexp(logits[r,:]) -
// logits[r, target[r]]. `logits` [rows, vocab] dt, `target` [rows] int32,
// `loss` [rows] fp32. fp32 accumulation.
void cross_entropy(sycl::queue& q, const void* logits, const int* target,
                   float* loss, std::size_t rows, std::size_t vocab, DType dt,
                   Variant variant = Variant::sycl, bool blocking = true);

// Fast Walsh-Hadamard transform (unnormalized) over each row: out[r,:] =
// H_n * in[r,:]. `in`/`out` [rows, n] dt, n a power of two (<= 2048).
void hadamard(sycl::queue& q, const void* in, void* out, std::size_t rows,
              std::size_t n, DType dt, Variant variant = Variant::sycl,
              bool blocking = true);

// Batched-gather LoRA matvecs (contract lora_apply). Per token row b the
// adapter is lora_idx[b] int32 (-1 or out-of-range = no adapter: shrink
// writes zeros, expand leaves the destination untouched).
//   shrink: out[b, r] = scale * sum_h in[b, h] * A[idx, r, h], out fp32
//           [batch, rank]; A is [n_loras, rank, hidden] of dtype dt.
//   expand: out[b, out_offset + j] (+)= sum_r in[b, r] * B[idx, j, r];
//           in is the shrink output (fp32 [batch, rank]), B is
//           [n_loras, out_dim, rank] of dtype dt, out rows have out_stride
//           elements (the slice form covers fused-QKV split destinations),
//           and `accumulate` adds into the existing output.
// fp32 math. Graph-capture-safe.
void lora_shrink(sycl::queue& q, const void* in, const void* w,
                 const std::int32_t* lora_idx, float* out, std::size_t batch,
                 std::size_t hidden, std::size_t rank, std::size_t n_loras,
                 float scale, DType dt, Variant variant = Variant::sycl,
                 bool blocking = true);

void lora_expand(sycl::queue& q, const float* in, const void* w,
                 const std::int32_t* lora_idx, void* out, std::size_t batch,
                 std::size_t rank, std::size_t out_dim, std::size_t n_loras,
                 std::size_t out_offset, std::size_t out_stride,
                 bool accumulate, DType dt, Variant variant = Variant::sycl,
                 bool blocking = true);

// ----------------------------------------------------------------------------
// moe
// ----------------------------------------------------------------------------

// MoE top-k routing. `router_logits` [n_tokens, n_experts] dtype dt. Selects the
// top-k experts per token and softmax-normalizes over the selected k. Outputs
// `expert_ids` [n_tokens, k] int32 and `expert_weights` [n_tokens, k] fp32.
// Router gating modes: softmax over the selected logits (default),
// sigmoid scores (Laguna-style, optionally renormalized), or
// sqrt(softplus) scores. All are monotonic in the logit so top-k selection
// (lowest-index tie-break) is identical; only the weights differ.
// routed_scaling multiplies the final weights.
enum class MoeGating { softmax, sigmoid, softplus_sqrt };

void moe_route_topk(sycl::queue& q, const void* router_logits, int* expert_ids,
                    float* expert_weights, std::size_t n_tokens,
                    std::size_t n_experts, int k, DType dt,
                    MoeGating gating = MoeGating::softmax,
                    bool renormalize = true, float routed_scaling = 1.0f,
                    Variant variant = Variant::sycl, bool blocking = true);

// Expert-sorted permutation of routed hidden rows (contract
// moe_route_top_k_prefix_sum_permute): gathers hidden [n_tokens, H] into
// permuted [n_valid_pairs, H] sorted by expert, fills rows_per_expert [E]
// (the moe_grouped_qgemm segmentation input) and row_map [n_tokens*top_k]
// (-1 for invalid expert ids — the EP-safe skip). cursors is caller scratch
// [E] int32. Rows within one expert land in nondeterministic order; the
// GEMM is row-symmetric and the unpermute reads through row_map, so end
// results are order-independent. Allocation-free, no host sync.
void moe_permute(sycl::queue& q, const void* hidden, const int* topk_ids,
                 void* permuted, std::int32_t* rows_per_expert,
                 std::int32_t* row_map, std::int32_t* cursors,
                 std::size_t n_tokens, std::size_t top_k,
                 std::size_t hidden_dim, std::size_t n_experts, DType dt,
                 Variant variant = Variant::sycl, bool blocking = true);

// Inverse weighted reduce (contract moe_unpermute_weighted_reduce):
// out[t,:] = sum_j topk_weights[t,j] * permuted[row_map[t*top_k+j], :] in
// fp32, skipping -1 rows.
void moe_unpermute_weighted_reduce(sycl::queue& q, const void* permuted,
                                   const std::int32_t* row_map,
                                   const float* topk_weights, void* out,
                                   std::size_t n_tokens, std::size_t top_k,
                                   std::size_t hidden_dim, DType dt,
                                   Variant variant = Variant::sycl,
                                   bool blocking = true);

// Grouped-GEMM weight formats (see moe_grouped_qgemm).
enum class MoeWeightFormat {
  w16,         // W [E, N, K] in the activation dtype
  int4_group,  // W [E, N, K/2] packed nibbles + f16 scales [E, N, K/group]
  nvfp4,       // W [E, N, K/2] e2m1 + u8 e4m3 scales [E, N, K/16] LINEAR
               // (de-swizzled) + per-expert f32 global scale (fp32 epilogue)
};

// Segmented per-expert GEMM with fused weight dequant on the native DPAS
// building block (contract grouped_gemm / moe_grouped_qgemm). A [M_total, K]
// act_dt with rows SORTED BY EXPERT; rows_per_expert [E] device int32 (zeros
// allowed, sum == M_total, E <= 256, never read on host); C [M_total, N]
// act_dt. K % 16 == 0; act_dt f16/bf16 (DPAS operands). The permute prologue
// (row sorting) is caller-owned. Graph-capture-safe: static worst-case grid,
// on-device segment walk, no allocation.
void moe_grouped_qgemm(sycl::queue& q, const void* A, const void* W,
                       const void* scales, const float* global_scales, void* C,
                       const std::int32_t* rows_per_expert, std::size_t M_total,
                       std::size_t N, std::size_t K, std::size_t E,
                       std::size_t group, MoeWeightFormat fmt, DType act_dt,
                       Variant variant = Variant::sycl, bool blocking = true);

// Grouped GEMM + SwiGLU (contract moe_grouped_qswiglu), composite v1:
// W is [E, 2I, K] with gate rows [0, I) and up rows [I, 2I); the GEMM lands
// in caller scratch [M_total, 2I] (the glu layout) and ops::glu applies
// silu(gate) * up into C [M_total, I]. A fused single-kernel form is the
// recorded follow-up (keep only if it beats the composite on B60).
void moe_grouped_qswiglu(sycl::queue& q, const void* A, const void* W,
                         const void* scales, const float* global_scales,
                         void* scratch_2i, void* C,
                         const std::int32_t* rows_per_expert,
                         std::size_t M_total, std::size_t I, std::size_t K,
                         std::size_t E, std::size_t group, MoeWeightFormat fmt,
                         DType act_dt, Variant variant = Variant::sycl,
                         bool blocking = true);

// Decode-oriented routed MoE with ModelOpt NVFP4 weights. `hidden` is [M,K]
// dtype `act_dt`; ids/weights are [M,top_k] int32/f32. Weight layouts are
// w13 [E,2I,K/2], w13_scales [E,2I,K/16], w2 [E,K,I/2], and w2_scales
// [E,K,I/16], with one fp32 global scale per expert/projection. The output is
// fp32 [M,K]. Invalid expert ids (<0 or >=E) are skipped. The call initializes
// `out_f32` to zero before accumulating routed outputs.
void nvfp4_moe_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *out_f32, std::size_t M, std::size_t E,
                     std::size_t top_k, std::size_t K, std::size_t I, DType act_dt,
                     bool multiply_router_weight = true, Variant variant = Variant::sycl,
                     bool blocking = true);

// Higher-occupancy two-stage form of `nvfp4_moe_fused`. The caller supplies
// fp32 scratch [M*top_k,2I]. This is the preferred batch-1 graph-replay path;
// the fused form can be faster once M*top_k already fills the device.
void nvfp4_moe_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                     const float *topk_weights, const void *w13, const void *w13_scales,
                     const float *w13_global_scales, const void *w2, const void *w2_scales,
                     const float *w2_global_scales, float *scratch_f32, float *out_f32,
                     std::size_t M, std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                     DType act_dt, bool multiply_router_weight = true,
                     Variant variant = Variant::sycl, bool blocking = true);

// ReLU²-ungated NVFP4 MoE (NemotronH-style experts). Same contract as
// `nvfp4_moe_fused` except w1 is a SINGLE up-projection [E,I,K/2] with scales
// [E,I,K/16] (not gate+up), and the activation is relu(g)^2 with no gate
// multiply. Output fp32 [M,K], zero-initialized by the call.
void nvfp4_moe_relu2_fused(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w1, const void *w1_scales,
                           const float *w1_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *out_f32, std::size_t M,
                           std::size_t E, std::size_t top_k, std::size_t K, std::size_t I,
                           DType act_dt, bool multiply_router_weight = true,
                           Variant variant = Variant::sycl, bool blocking = true);

// Higher-occupancy two-stage form of `nvfp4_moe_relu2_fused`. The caller
// supplies fp32 scratch [M*top_k, I] (the up-projection buffer).
void nvfp4_moe_relu2_split(sycl::queue &q, const void *hidden, const int *topk_ids,
                           const float *topk_weights, const void *w1, const void *w1_scales,
                           const float *w1_global_scales, const void *w2, const void *w2_scales,
                           const float *w2_global_scales, float *scratch_f32, float *out_f32,
                           std::size_t M, std::size_t E, std::size_t top_k, std::size_t K,
                           std::size_t I, DType act_dt, bool multiply_router_weight = true,
                           Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// linear_attention
// ----------------------------------------------------------------------------

// In-place L2 normalization of q/k head vectors (the Gated DeltaNet
// pre-step): each [dk] vector of the [tokens, heads, dk] tensor is divided
// by sqrt(sum(x^2) + eps). Run on q and k before gated_delta_rule_varlen.
void gdn_l2norm_qk(sycl::queue& q, void* qk, std::size_t tokens,
                   std::size_t heads, std::size_t dk, float eps, DType dt,
                   Variant variant = Variant::sycl, bool blocking = true);

// Varlen gated delta rule (Gated DeltaNet), exact recurrence, prefill +
// decode in one call. Q, K are [T, Hk, dk] (pre-L2-normalized) and V,
// core_out [T, Hv, dv] of dtype act_dt, packed by cu_seqlens [batch+1];
// b and a are [T, Hv] f32 raw gate projections (beta = sigmoid(b),
// g = exp(-exp(A_log[hv]) * softplus(a + dt_bias[hv]))); A_log, dt_bias
// (nullptr = 0) are [Hv] f32. ssm_state [nslots, Hv, dv, dk] of state_dt is
// read at state_indices[seq] (zeros when has_initial_state[seq] is false or
// the pointer is null-slot) and written back IN PLACE after the last token;
// null/out-of-range slots emit zero output and touch no state. GQA maps
// hv -> hv / (Hv/Hk). dk <= 128; per-token order is decay, kv read, rank-1
// update, output (matching the vLLM general kernel). The width-4 conv stage
// is causal_conv1d_prefill/decode with token-major strides; the chunked-DPAS
// prefill pipeline is the deferred perf variant.
void gated_delta_rule_varlen(
    sycl::queue& q, const void* Q, const void* K, const void* V,
    const float* b, const float* a, const float* A_log, const float* dt_bias,
    void* ssm_state, void* core_out, const std::int32_t* cu_seqlens,
    const std::int32_t* state_indices, const bool* has_initial_state,
    std::size_t batch, std::size_t Hk, std::size_t dk, std::size_t Hv,
    std::size_t dv, std::size_t nslots, DType act_dt, DType state_dt,
    Variant variant = Variant::sycl, bool blocking = true);

// Non-causal linear attention. Q, K, V are [n_heads, seq, dim] dtype dt; O is
// [n_heads, seq, dim]. Per head: KV = sum_t K[t]^T V[t] (dim x dim), z = sum_t
// K[t] (dim), O[t] = (Q[t] @ KV) / (Q[t] . z + eps). fp32 accumulation. dim must
// be <= 64 for the SLM path. Native-only.
void linear_attn(sycl::queue& q, const void* Q, const void* K, const void* V,
                 void* O, std::size_t n_heads, std::size_t seq, std::size_t dim,
                 DType dt, Variant variant = Variant::sycl, bool blocking = true);

// Qwen3.5/Qwen3.6 Gated DeltaNet decode core for the non-interleaved layout.
// Fixed model dimensions are q/k=16x128, v/z=32x128, conv width=4. Inputs are
// projected_qkvz [B,12288] and projected_ba [B,64]. The call mutates conv_state
// ([slots,3,8192] or [slots,8192,3]) and ssm_state [slots,32,128,128], writes
// scratch `mixed_qkv` [B,8192], and returns core/z [B,32,128]. State indices in
// [0,slots) are valid and must be unique within a decode batch. Negative or
// out-of-range indices leave state untouched and produce zero core output;
// `z` always passes through from the projected input.
void qwen_gdn_decode(sycl::queue &q, const void *projected_qkvz, const void *projected_ba,
                     void *conv_state, void *ssm_state, const void *conv_weight,
                     const void *conv_bias, const float *A_log, const void *dt_bias,
                     const int *state_indices, void *mixed_qkv, void *core_out, void *z_out,
                     std::size_t batch, std::size_t state_slots, bool conv_dim_first, DType act_dt,
                     DType state_dt, DType dt_bias_dt, Variant variant = Variant::sycl,
                     bool blocking = true);

// ----------------------------------------------------------------------------
// ssm (state-space / Mamba)
// ----------------------------------------------------------------------------

// Mamba selective scan (S6), forward. Per channel c and state s the recurrence
// is h_s = exp(delta*A[c,s]) * h_s + delta*B[t,s]*u[c,t]; y[c,t] = sum_s C[t,s]*h_s
// + D[c]*u[c,t]. Shapes: u,delta [n_chan, seq]; A [n_chan, state]; B,C [seq,
// state] (shared across channels); D [n_chan]; y [n_chan, seq]. All dtype dt
// except the recurrence runs in fp32. `state` <= 16. Native-only (sequential).
void selective_scan(sycl::queue& q, const void* u, const void* delta,
                    const void* A, const void* B, const void* C, const void* D,
                    void* y, std::size_t n_chan, std::size_t seq,
                    std::size_t state, DType dt, Variant variant = Variant::sycl,
                    bool blocking = true);

// DeepSeek-V4 manifold hyper-connections, post stage (contract
// dsv4_hc_post): out[t,o,h] = sum_i comb_res_mix[t,i,o] * residual[t,i,h]
// + post_mix[t,o] * x[t,h]. residual/out [tokens, n_streams, hidden] dtype
// dt (n_streams <= 8; 4 for DSV4), x [tokens, hidden], mixes fp32. The pre
// (Sinkhorn-gated) and comb stages are deferred pending a DSV4 bring-up —
// see the D-wave ledger.
void dsv4_hc_post(sycl::queue& q, const float* comb_res_mix,
                  const void* residual, const float* post_mix, const void* x,
                  void* out, std::size_t tokens, std::size_t n_streams,
                  std::size_t hidden, DType dt,
                  Variant variant = Variant::sycl, bool blocking = true);

// Mamba-2 SSD decode (selective-state-update, scalar-A-per-head), one token per
// sequence. state is [nslots, nheads, headdim, dstate] of dtype state_dt with
// element strides s0..s3 (strided serving views are accepted); x and out are
// [batch, nheads, headdim] of dtype act_dt; dt_raw [batch, nheads], A and
// dt_bias [nheads], D [nheads, headdim] (nullptr = skip) are f32; B and C are
// [batch, ngroups, dstate] of dtype act_dt. Per sequence b the state is read
// from slot src_indices[b] and written to dst_indices[b] (copy-on-write; equal
// src/dst updates in place). A negative or out-of-range src emits zero output
// and touches no state; a negative dst skips the state write (read-only step).
// dt_softplus applies softplus(dt_raw + dt_bias). Recurrence in fp32.
// Graph-capture-safe: no host sync, no allocation, fixed launch geometry.
void ssd_decode(sycl::queue& q, void* state, const void* x, const float* dt_raw,
                const float* A, const void* B, const void* C, const float* D,
                const float* dt_bias, const std::int32_t* src_indices,
                const std::int32_t* dst_indices, void* out, bool dt_softplus,
                std::size_t batch, std::size_t nheads, std::size_t headdim,
                std::size_t dstate, std::size_t ngroups, std::size_t nslots,
                std::int64_t s0, std::int64_t s1, std::int64_t s2,
                std::int64_t s3, DType act_dt, DType state_dt,
                Variant variant = Variant::sycl, bool blocking = true);

// Varlen Mamba-2 SSD prefill selective scan (sequential-over-tokens variant).
// x and out are [T, nheads, headdim] of dtype act_dt where T is the packed
// token count; dt_raw [T, nheads], A and dt_bias [nheads], D [nheads, headdim]
// (nullptr = skip) are f32; B and C are [T, ngroups, dstate] act_dt;
// cu_seqlens is [batch+1] int32 packed token offsets. initial_states
// [batch, nheads, headdim, dstate] of dtype state_dt seeds each sequence
// (nullptr = zero); the final state per sequence is written to varlen_states
// (same shape). The caller owns the cache gather/scatter of those states —
// that split keeps the kernel free of slot indirection and capture-safe.
// dt_h = clamp(softplus(dt_raw + dt_bias), dt_lo, dt_hi); recurrence in fp32
// with each lane's state row of dstate held in shared local memory. Requires
// headdim*dstate*4 bytes of SLM per work-group and headdim lanes; shapes
// outside the device envelope are rejected without launching.
void ssd_prefill(sycl::queue& q, const void* x, const float* dt_raw,
                 const float* A, const void* B, const void* C, const float* D,
                 const float* dt_bias, const std::int32_t* cu_seqlens,
                 const void* initial_states, void* out, void* varlen_states,
                 bool dt_softplus, float dt_lo, float dt_hi, std::size_t batch,
                 std::size_t nheads, std::size_t headdim, std::size_t dstate,
                 std::size_t ngroups, DType act_dt, DType state_dt,
                 Variant variant = Variant::sycl, bool blocking = true);

// Depthwise causal conv1d decode update (Mamba-2 conv stage), one token per
// sequence. conv_state is [nslots, dim, state_len] of dtype state_dt with
// element strides cs0..cs2 (both the dim-major and the len-major serving
// layouts are just stride choices); x and out are [batch, dim] of dtype act_dt;
// weight is [dim, kernel], bias [dim] (nullptr = none), both act_dt. Per row b
// the state at slot indices[b] is read and updated in place (shift left one,
// append x); a negative or out-of-range slot emits zero output and touches no
// state. Requires kernel == state_len + 1 and kernel <= 8 (register window);
// shapes outside that envelope are rejected without launching. fp32 math,
// optional SiLU. Graph-capture-safe.
void causal_conv1d_decode(sycl::queue& q, void* conv_state, const void* x,
                          const void* weight, const void* bias,
                          const std::int32_t* indices, void* out, bool silu,
                          std::size_t batch, std::size_t dim,
                          std::size_t state_len, std::size_t kernel,
                          std::size_t nslots, std::int64_t cs0,
                          std::int64_t cs1, std::int64_t cs2, DType act_dt,
                          DType state_dt, Variant variant = Variant::sycl,
                          bool blocking = true);

// Varlen depthwise causal conv1d prefill. x and out are DIM-MAJOR
// [dim, total_tokens] of dtype act_dt with element strides xs0/xs1 and
// os0/os1 (the packed serving transpose layout); weight [dim, kernel] and
// bias [dim] (nullptr = none) act_dt; conv_state [nslots, dim, state_len]
// state_dt with strides cs0..cs2; cu_seqlens [batch+1] int32; indices [batch]
// int32 cache slots (<0 = null: zero output rows, no state touch); has_init
// [batch] bools (nullptr = all false) select whether the earliest taps of a
// sequence read the slot's existing window or zeros. Two chained kernels:
// output pass, then state write-back (last state_len samples of
// [initial_state | seq_x]); the dependency guarantees old-window reads precede
// the writes, and the returned/waited event is the write-back's. Empty
// sequences leave their slot untouched. Same kernel <= 8 envelope as
// causal_conv1d_decode. fp32 math, optional SiLU. Graph-capture-safe.
void causal_conv1d_prefill(sycl::queue& q, void* conv_state, const void* x,
                           const void* weight, const void* bias,
                           const std::int32_t* cu_seqlens,
                           const std::int32_t* indices, const bool* has_init,
                           void* out, bool silu, std::size_t total_tokens,
                           std::size_t batch, std::size_t dim,
                           std::size_t state_len, std::size_t kernel,
                           std::size_t nslots, std::int64_t xs0,
                           std::int64_t xs1, std::int64_t os0, std::int64_t os1,
                           std::int64_t cs0, std::int64_t cs1, std::int64_t cs2,
                           DType act_dt, DType state_dt,
                           Variant variant = Variant::sycl,
                           bool blocking = true);

// Split-KV paged attention decode (one query token per sequence; contract
// decode_cache_attention). Q and O are [batch, n_heads, d] of dtype dt; the
// KV caches are [n_pages, page_size, n_kv_heads, d] with page_stride_elems
// the element pitch between pages (>= page_size*n_kv_heads*d); block_table
// [batch, max_pages] int32 with -1 pages skipped; seq_lens [batch] context
// lengths. tmp_out [batch, n_heads, splits, d], exp_sums and max_logits
// [batch, n_heads, splits] are CALLER-OWNED fp32 workspaces sized for the
// maximum shapes (no allocation — graph-capture-safe); two chained
// submissions (partials, then an LSE merge that also folds the optional
// per-head attention sinks into the denominator; empty splits are -inf
// guarded). window_left >= 0 keeps only the trailing window_left+1 keys.
// fp8 KV decodes through exact codecs with device-scalar k/v scales
// (nullptr = 1). d <= 256. Correctness-first per-work-item shape; the
// DPAS-tiled variant is the recorded throughput lever.
void paged_attention_decode(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* tmp_out, float* exp_sums, float* max_logits,
    const std::int32_t* block_table, const std::int32_t* seq_lens,
    std::size_t batch, std::size_t n_heads, std::size_t n_kv_heads,
    std::size_t d, std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, int num_kv_splits, float sm_scale,
    int window_left, const float* sinks, const float* k_scale,
    const float* v_scale, DType dt, KvCacheDType kv_dt,
    Variant variant = Variant::sycl, bool blocking = true);

// Varlen paged/dense prefill FMHA forward (contract paged_attention_advanced
// + mixed_prefill_decode_attention). Q and O are [total_q, n_heads, d]
// packed by cu_seqlens_q [batch+1]; cu_seqlens_k gives per-sequence context
// lengths. block_table as in decode; a null block_table reads a contiguous
// dense cache [batch, max_seqlen_k, n_kv_heads, d]. Causal masking is
// end-aligned; window_left/right bound the band (-1 unbounded; with
// causal=false this expresses the symmetric window). A null-able per-batch
// is_prefill u8 mask skips decode rows so mixed batches share the launch.
// Optional lse [total_q, n_heads] stores m + log(l) with sinks included.
// d <= 256; total_q is the packed token count (host-known).
void paged_attention_prefill(
    sycl::queue& q, const void* Q, const void* k_cache, const void* v_cache,
    void* O, float* lse, const std::int32_t* block_table,
    const std::int32_t* cu_seqlens_q, const std::int32_t* cu_seqlens_k,
    const std::uint8_t* is_prefill, std::size_t total_q, std::size_t batch,
    std::size_t n_heads, std::size_t n_kv_heads, std::size_t d,
    std::size_t page_size, std::size_t max_pages,
    std::size_t page_stride_elems, std::size_t max_seqlen_k, float sm_scale,
    bool causal, int window_left, int window_right, const float* sinks,
    const float* k_scale, const float* v_scale, DType dt, KvCacheDType kv_dt,
    Variant variant = Variant::sycl, bool blocking = true);

// DeepSeek-style fp8 MQA indexer logits (sparse-attention index scoring):
// logits[s,kv] = sum_h head_weights[s,h] * relu((q[s,h,:] . kv[kv,:]) *
// kv_scales[kv]), masked to -inf outside [ks[s], ke[s]). q_fp8 [S,H,D] and
// kv_fp8 [Skv,D] are e4m3 bytes (decoded to bf16 for the DPAS — this
// generation has no fp8 MMA); head_weights [S,H], kv_scales [Skv], logits
// [S,Skv] f32. D must be a multiple of 16 (rejected otherwise); any H
// (zero-padded head tiles). Runs on the native joint_matrix building block
// (kernels/common/xmx_tile.hpp). Graph-capture-safe.
void mqa_logits(sycl::queue& q, const std::uint8_t* q_fp8,
                const std::uint8_t* kv_fp8, const float* kv_scales,
                const float* head_weights, const std::int32_t* ks,
                const std::int32_t* ke, float* logits, std::size_t S,
                std::size_t H, std::size_t D, std::size_t Skv,
                Variant variant = Variant::sycl, bool blocking = true);

// ----------------------------------------------------------------------------
// collectives (multi-GPU)
// ----------------------------------------------------------------------------

// Bytes for one rank's peer-visible all-reduce region ([signal | staging]);
// staging must hold the largest payload the group will reduce. Regions must be
// zero-filled once before the first collective.
std::size_t all_reduce_region_bytes(std::size_t staging_bytes);

// Capturable P2P sum all-reduce (world 2..8). regions[r] is rank r's region
// base as mapped in THIS rank's address space (same-context USM in-process, or
// an IPC-mapped pointer cross-process; the fd exchange is integration-owned);
// regions[rank] is our own. inp/out are device pointers of dtype dt with numel
// elements (out-of-place). Accumulation is fp32 in FIXED rank order 0..world-1
// so every rank produces bitwise-identical results. Payloads below
// twoshot_min_bytes use the one-shot algorithm (latency-bound), larger ones
// reduce-scatter + all-gather. The flag handshake self-increments a
// device-resident generation counter, so the call records into a SYCL command
// graph and survives replay; no host sync, no allocation. Out-of-envelope
// rank/world are rejected without launching. Every rank in the group must
// launch the collective concurrently or the (bounded) spins expire.
void all_reduce(sycl::queue& q, const void* inp, void* out,
                void* const* regions, int rank, int world, std::size_t numel,
                DType dt, std::size_t twoshot_min_bytes = 65536,
                Variant variant = Variant::sycl, bool blocking = true);

// Sum all-reduce across ALL visible Intel GPUs (the 4x B60). `in_per_gpu` is a
// host buffer [n_gpus * count] where GPU g's contribution is at offset g*count;
// `out` is host [count] = the elementwise sum across GPUs. Orchestrates the
// capturable `all_reduce` in-process: one shared SYCL context, per-GPU regions,
// every rank's collective launched concurrently. Returns the number of GPUs
// used (capability-gated: 0 if none). Native path; a oneCCL vendor variant is
// the production route (deferred).
std::size_t all_reduce_sum(const float* in_per_gpu, float* out,
                           std::size_t count);

// ----------------------------------------------------------------------------
// matmul
// ----------------------------------------------------------------------------

// Dense GEMM: C[M,N] = A[M,K] * B[K,N], all row-major, fp32 accumulation. `a`,
// `b`, `c` are device pointers of dtype `dt`. The vendor variant is oneDNN
// matmul (XMX/DPAS-backed); the native SYCL variant is an SLM-tiled baseline.
void dense_gemm(sycl::queue& q, const void* a, const void* b, void* c,
                std::size_t M, std::size_t N, std::size_t K, DType dt,
                Variant variant = Variant::best, bool blocking = true);

// ----------------------------------------------------------------------------
// quantization
// ----------------------------------------------------------------------------

// Quantize a weight matrix W [N, K] (dtype dt) to symmetric int4, group-wise:
// per group of `group` columns, scale = groupmax(|W|)/7, q = round(W/scale)
// clamped to [-8,7], packed 2-per-byte along K (low nibble = even k). Outputs
// `w_packed` [N*K/2] uint8 and `scales` [N, K/group] fp16 — the exact layout
// qgemv_int4 consumes (so quantize -> qgemv round-trips). Native.
void quantize_int4_group(sycl::queue& q, const void* w, void* w_packed,
                         void* scales, std::size_t N, std::size_t K,
                         std::size_t group, DType dt,
                         Variant variant = Variant::sycl, bool blocking = true);

// int4 group-quantized GEMV (batch-1 decode), Marlin/Metal-style dequant on the
// fly. Weight W is [N, K] symmetric signed int4 packed 2-per-byte along K (low
// nibble = even k); `scales` is [N, K/group] fp16 (half); activation `x` is [K]
// and output `y` is [N], both of dtype `act_dt`. Accumulates in fp32:
//   y[n] = sum_k (int4(W[n,k]) * scales[n, k/group]) * x[k]
// K must be even and a multiple of `group`.
void qgemv_int4(sycl::queue& q, const void* w_packed, const void* scales,
                const void* x, void* y, std::size_t N, std::size_t K,
                std::size_t group, DType act_dt, Variant variant = Variant::sycl,
                bool blocking = true);

// W4A16 GEMM on the Xe tensor engine (DPAS via SYCL joint_matrix):
// C[M,N] = A[M,K] . dequant(W)^T. A/C are 16-bit float (`act_dt` in {f16, bf16}
// -- "a16"); W is [N,K] int4 group-quantized with the SAME encoding
// qgemv_int4 / quantize_int4_group use: `w_packed` is 2 nibbles/byte (low nibble
// = even k, high = odd k, signed two's-complement), `scales` is f16 [N, K/group],
// dequant(W)[n,k] = s4(nibble) * scales[n, k/group]. The weight tile is
// dequantized on the fly into SLM and multiplied by the activation tile on the
// int4-weight tensor path (DPAS), fp32 accumulation. Small-M (M<=32) decode-GEMM
// shape (the batched analogue of qgemv_int4). K must be even and `group` must
// divide K; any M/N are handled by edge masking. act_dt f32 is unsupported.
void w4a16_gemm(sycl::queue& q, const void* A, const void* w_packed,
                const void* scales, void* C, std::size_t M, std::size_t N,
                std::size_t K, std::size_t group, DType act_dt,
                Variant variant = Variant::sycl, bool blocking = true);

// fp8 format selector (OCP / NVIDIA fp8).
enum class Fp8Kind {
  e4m3,
  e5m2,
};

// fp8 GEMM: C[M,N] = A_fp8[M,K] @ B_fp8[K,N], scaled by a single global `scale`.
// A, B are opaque fp8 bytes (1 byte/elem) of kind `kind`; C is `out_dt`
// (f32/f16/bf16). best-routing (measured on B60): M=1 -> native SYCL decode
// GEMV (weight-memory-bound fast path); M>1 -> oneDNN matmul. If the vendor
// fp8 matmul is unsupported the call reports it (no silent wrong result).
void fp8_gemm(sycl::queue& q, const void* a_fp8, const void* b_fp8, void* c,
              std::size_t M, std::size_t N, std::size_t K, Fp8Kind kind,
              float scale, DType out_dt, Variant variant = Variant::best,
              bool blocking = true);

// FP8 weight-only GEMM: C[M,N] = A[M,K] @ dequant(W[N,K])^T. Activations and
// output use `act_dt`; W stores raw e4m3/e5m2 bytes in checkpoint-native [N,K]
// layout. `weight_scale` is an fp32 device pointer to [N] (per-channel) or [1].
// `best` routes M=1 to native decode GEMV and M>1 to oneDNN when present.
void fp8_gemm_w8a16(sycl::queue &q, const void *activations, const void *weight_fp8,
                    const float *weight_scale, void *out, std::size_t M, std::size_t N,
                    std::size_t K, Fp8Kind kind, bool per_channel, DType act_dt,
                    Variant variant = Variant::best, bool blocking = true);

// fp8 codecs: f32 -> fp8 (out is 1 byte/elem) and fp8 -> f32, both over `n`
// contiguous elements. Vendor-backed (oneDNN reorder). Useful for quantizing
// activations/weights and for exact round-trip references.
// TurboQuant KV-cache codec, format version 2 (specs/formats/turboquant.md).
// key/value are [num_tokens, num_kv_heads, head_size] of dtype dt; caches are
// slot-indexed: key_cache [slot, heads, ceil(head_size*key_bits/8)] u8,
// value_cache likewise with value_bits, and the three scale caches
// [slot, heads, head_size/32] fp16. slot_mapping[token] < 0 skips the token.
// Keys (key_bits < 8) take the rotated Lloyd-Max path (caller-supplied
// ascending centroids [2^key_bits] and +-1 signs [head_size]); key_bits == 8
// selects the unrotated saturating e4m3 byte path (no scales/table). Values
// are per-group uniform with fp16 scale and zero; decode is
// (code + zero) * scale, and value_signed applies two's-complement at 8 bits.
// The fp16 rounding chain is load-bearing: the codec is shared verbatim with
// the host oracle and codes must match bit for bit. head_size in {64,128,256};
// bits in [2,8]; out-of-envelope shapes are rejected without launching.
void turboquant_encode(sycl::queue& q, const void* key, const void* value,
                       std::uint8_t* key_cache, std::uint8_t* value_cache,
                       void* key_scale_cache, void* value_scale_cache,
                       void* value_zero_cache,
                       const std::int64_t* slot_mapping, const float* centroids,
                       const float* signs, std::size_t num_tokens,
                       std::size_t num_kv_heads, std::size_t head_size,
                       int key_bits, int value_bits, bool value_signed,
                       DType dt, Variant variant = Variant::sycl,
                       bool blocking = true);

// Inverse of turboquant_encode for a gathered slot list; k_out/v_out are
// [num_slots, heads, head_size] fp32. A negative slot decodes to zeros.
void turboquant_decode(sycl::queue& q, const std::uint8_t* key_cache,
                       const std::uint8_t* value_cache,
                       const void* key_scale_cache,
                       const void* value_scale_cache,
                       const void* value_zero_cache, const std::int64_t* slots,
                       const float* centroids, const float* signs, float* k_out,
                       float* v_out, std::size_t num_slots,
                       std::size_t num_kv_heads, std::size_t head_size,
                       int key_bits, int value_bits, bool value_signed,
                       Variant variant = Variant::sycl, bool blocking = true);

void fp8_encode(sycl::queue& q, const float* in, void* out_fp8, std::size_t n,
                Fp8Kind kind, bool blocking = true);
void fp8_decode(sycl::queue& q, const void* in_fp8, float* out, std::size_t n,
                Fp8Kind kind, bool blocking = true);

// mxfp4 GEMV (OCP microscaling FP4), native decode. Weight W is [N, K] of e2m1
// (fp4) elements packed 2/byte along K, with one e8m0 (power-of-two) block scale
// per 32 elements: `block_scales` is [N, K/32] uint8. Activation `x` is [K] and
// output `y` is [N], dtype `act_dt`. Dequant in fp32:
//   w = e2m1(nibble) * 2^(e8m0 - 127);  y[n] = sum_k w * x[k]
// K must be a multiple of 32. Proves mxfp4 (not a hardware feature) runs
// natively on Intel via a hand-written decoder.
void mxfp4_gemv(sycl::queue& q, const void* w_packed, const void* block_scales,
                const void* x, void* y, std::size_t N, std::size_t K,
                DType act_dt, Variant variant = Variant::sycl,
                bool blocking = true);

// nvfp4 GEMV (NVIDIA FP4), native decode. Weight W is [N, K] of e2m1 (fp4)
// packed 2/byte, with one e4m3 (fp8) block scale per 16 elements
// (`block_scales` is [N, K/16] uint8) and a per-tensor fp32 `global_scale`.
// Activation `x` is [K], output `y` is [N], dtype `act_dt`. Dequant in fp32:
//   w = e2m1(nibble) * e4m3(block_scale) * global_scale;  y[n] = sum_k w * x[k]
// K must be a multiple of 32. Proves nvfp4 decodes natively on Intel.
void nvfp4_gemv(sycl::queue& q, const void* w_packed, const void* block_scales,
                float global_scale, const void* x, void* y, std::size_t N,
                std::size_t K, DType act_dt, Variant variant = Variant::sycl,
                bool blocking = true);

// Batched NVFP4 W4A16 GEMM with checkpoint-native packed W [N,K/2]. The
// decode path submits one optimized GEMV per activation row; this beat the
// decode-once M-tiled alternative at every measured serving shape M=1,4,8.
void nvfp4_gemm(sycl::queue &q, const void *w_packed, const void *block_scales, float global_scale,
                const void *x, void *y, std::size_t M, std::size_t N, std::size_t K, DType act_dt,
                Variant variant = Variant::sycl, bool blocking = true);

// GGUF (llama.cpp) block-quant GEMV, native decode from the authentic on-disk
// block layout. `w_blocks` is row-major [N rows], each row = K/32 blocks laid
// consecutively; a block is { fp16 scale d; quants } — q8_0: 34 bytes (32 int8),
// q4_0: 18 bytes (32 int4 packed, dequant (nibble-8)*d). Activation `x` is [K],
// output `y` is [N], dtype `act_dt`. K must be a multiple of 32.
enum class GgufType {
  q8_0,
  q4_0,
  q6_K,
  q4_K,
  q5_K,
  q2_K,
  q3_K,
  iq4_nl,
  q4_1,
  q5_0,
  q5_1,
  iq4_xs,
  iq2_xxs,
  iq2_xs,
  iq3_xxs,
  iq1_s,
};

void gguf_gemv(sycl::queue& q, const void* w_blocks, const void* x, void* y,
               std::size_t N, std::size_t K, GgufType type, DType act_dt,
               Variant variant = Variant::sycl, bool blocking = true);

// GGUF routed GEMV / dequantize on ggml type ids (8 = Q8_0, 10 = Q2_K,
// 16 = IQ2_XXS), the DeepSeek-V4 serving formats. `w` is [E, N, row] on-disk
// blocks (E = 1 for a dense matrix); y[r, :] = W[expert_ids[r]] . x[r / top_k]
// for r < R (expert_ids null => dense, top_k = 1; negative id => zero row).
// `iq2xxs_grid_dev` is the 256-entry uint64 iq2xxs grid resident on the
// device (kernels/quantization/gguf_gemv/gguf_iq_tables.hpp), read only for
// IQ2_XXS. Native only.
bool gguf_routed_supports(int ggml_type);
void gguf_routed_gemv(sycl::queue& q, const void* w, const void* x, void* y,
                      const std::int32_t* expert_ids, std::size_t R,
                      std::size_t N, std::size_t K, std::size_t top_k,
                      int ggml_type, DType act_dt,
                      const std::uint64_t* iq2xxs_grid_dev, bool blocking = true);
void gguf_dequantize(sycl::queue& q, const void* w, void* out, std::size_t N,
                     std::size_t K, int ggml_type, DType out_dt,
                     const std::uint64_t* iq2xxs_grid_dev, bool blocking = true);

// Per-token symmetric int8 activation quantization. `x` [rows, dim] dtype dt ->
// `q` [rows, dim] int8 + `scale` [rows] fp32, where scale = rowmax(|x|)/127 and
// q = round(x/scale). Feeds qgemm_int8 (the w8a8 path). Native.
void act_quant_int8(sycl::queue& q, const void* x, signed char* q_out,
                    float* scale, std::size_t rows, std::size_t dim, DType dt,
                    Variant variant = Variant::sycl, bool blocking = true);

// int8 w8a8 GEMM: C[M,N] = (A_int8[M,K] @ B_int8[K,N]) * a_scale[M] * b_scale[N].
// A, B are int8 device pointers; a_scale (per-row/token) and b_scale (per-col/
// channel) are fp32 [M] and [N]; C is `out_dt` (f32/f16/bf16). Accumulates int32.
// The vendor variant is oneDNN int8 matmul (XMX/DPAS); the native variant is an
// SLM-tiled int8 baseline.
void qgemm_int8(sycl::queue& q, const void* a_int8, const void* b_int8,
                const void* a_scale, const void* b_scale, void* c, std::size_t M,
                std::size_t N, std::size_t K, DType out_dt,
                Variant variant = Variant::best, bool blocking = true);

// ----------------------------------------------------------------------------
// norms
// ----------------------------------------------------------------------------

// RMSNorm over the last axis of a [rows, dim] row-major tensor:
//   out[r, i] = x[r, i] * rsqrt(mean_i(x[r, :]^2) + eps) * weight[i]
// `x`, `out` are device pointers of dtype `dt` ([rows*dim]); `weight` is [dim]
// of dtype `dt`. Reduction accumulates in fp32. Deterministic family.
void rms_norm(sycl::queue& q, const void* x, const void* weight, void* out,
              std::size_t rows, std::size_t dim, float eps, DType dt,
              Variant variant = Variant::sycl, bool blocking = true);

// Fused residual add + RMSNorm. For each row, normalize the unrounded fp32 sum
// of x and residual, update residual in place with the storage-dtype sum, and
// write the normalized/weighted result to out. `out` must not alias `residual`.
void fused_add_rms_norm(sycl::queue &q, const void *x, void *residual, const void *weight,
                        void *out, std::size_t rows, std::size_t dim, float eps, DType dt,
                        Variant variant = Variant::sycl, bool blocking = true);


// Fused residual-add + double RMSNorm with f16 convert. Extends
// fused_add_rms_norm to the transformer layer boundary: `projection` is a
// sublayer output, RMS-normalized by `post_weight` and added into the
// `residual` stream (updated in place); the updated residual is then
// RMS-normalized by `next_weight` for the next layer and written to `next_out`
// as f16. Per row over `dim`, fp32 accumulation:
//   pinv     = rsqrt(mean_d(projection^2) + eps)
//   residual = residual + projection * post_weight * pinv
//   rinv     = rsqrt(mean_d(residual^2) + eps)
//   next_out = f16(residual * next_weight * rinv)
// `projection`, `post_weight`, `residual`, `next_weight` are dtype dt;
// `next_out` is always f16 and must not alias `residual`. Collapses the
// post-norm, residual add, next pre-norm, and f16 convert into one launch.
// Shape: residual-add + double RMSNorm -> f16.
// Fused RMSNorm + activation quantization modes.
enum class NormQuantMode {
  static_fp8,   // e4m3 with a caller-provided device-scalar scale
  dynamic_fp8,  // e4m3 with a per-row scale = max(absmax/448, 1/(448*512)),
                // written to out_scales[row]
  mxfp4,        // per-32-group fp32 power-of-two scale
                // exp2(ceil(log2(max(absmax/6, 1e-10)))) in out_scales
                // [rows, hidden/32], packed e2m1 nibbles (element 2i low)
};

// Fused RMSNorm + quantize. x is [rows, hidden] of dtype dt, weight [hidden];
// out_q is u8 [rows, hidden] (fp8 modes) or [rows, hidden/2] (mxfp4). With
// residual non-null the call adds it IN PLACE first (variance uses the
// dtype-rounded read-back) — that form satisfies the residual_rms_norm_quant
// contract. static_scale is a device scalar (static_fp8 only); out_scales is
// the scale output (dynamic/mxfp4 only). The e4m3/e2m1 encode steps are
// integer-exact; hidden must be divisible by 32 for mxfp4 (rejected
// otherwise).
void norm_quant(sycl::queue& q, const void* x, void* residual,
                const void* weight, std::uint8_t* out_q,
                const float* static_scale, float* out_scales, std::size_t rows,
                std::size_t hidden, float eps, NormQuantMode mode, DType dt,
                Variant variant = Variant::sycl, bool blocking = true);

// Gated group-RMSNorm (the Mamba-2 mixer output norm). x, gate, out are
// [rows, hidden] of dtype dt; weight is [hidden]. y = x * silu(gate) in fp32;
// with rms_norm true, each of n_groups contiguous slices of the hidden dim is
// RMS-normalized independently (variance over hidden/n_groups elements), the
// result is rounded to dt and multiplied by weight (the torch rounding
// order); rms_norm false returns y directly and ignores weight (may be
// nullptr). Single-device semantics: the tensor-parallel n_groups==1
// cross-rank variance reduction is integration-owned. hidden must be
// divisible by n_groups.
void group_rms_norm_gated(sycl::queue& q, const void* x, const void* gate,
                          const void* weight, void* out, std::size_t rows,
                          std::size_t hidden, std::size_t n_groups, float eps,
                          bool rms_norm, DType dt,
                          Variant variant = Variant::sycl,
                          bool blocking = true);

void rms_residual_next(sycl::queue &q, const void *projection, const void *post_weight,
                       void *residual, const void *next_weight, void *next_out,
                       std::size_t rows, std::size_t dim, float eps, DType dt,
                       Variant variant = Variant::sycl, bool blocking = true);

// LayerNorm over the last axis of a [rows, dim] row-major tensor:
//   out[r, i] = (x[r, i] - mean) * rsqrt(var + eps) * weight[i] + bias[i]
// with mean/var over x[r, :]. `bias` may be null to skip the shift. `weight`,
// `bias` are [dim] of dtype `dt`. Reduction accumulates in fp32.
void layernorm(sycl::queue& q, const void* x, const void* weight,
               const void* bias, void* out, std::size_t rows, std::size_t dim,
               float eps, DType dt, Variant variant = Variant::sycl,
               bool blocking = true);

}  // namespace quixicore::xpu::ops
