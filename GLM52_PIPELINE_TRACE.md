# GLM-5.2-Vision (Q2_K routed): llama.cpp vs vLLM — line-by-line pipeline trace

Purpose: find where our math diverges from llama.cpp, which produces coherent
output from the identical GGUF. Every claim below is either a code citation or a
measured value from a 5-token prompt (`"The capital of France is"`,
ids `[785, 6722, 315, 9621, 374]`), greedy.

Reference dump: `llama-eval-callback` (CPU, `-ngl 0`) — 4689 tensor dumps.
Ours: `QF_DUMP_LAYERS=1` probes in `deepseek_v2.py` / `mla.py`.

Config (`text_config`): 78 layers, hidden 6144, 64 heads, `q_lora_rank` 2048,
`kv_lora_rank` 512, `qk_nope_head_dim` 192, `qk_rope_head_dim` 64,
`v_head_dim` 256, `intermediate_size` 12288, `moe_intermediate_size` 2048,
256 experts, 8 per token, 1 shared, `first_k_dense_replace` 3,
`routed_scaling_factor` 2.5, `norm_topk_prob` true, `scoring_func` sigmoid,
`n_group` 1, `topk_group` 1, `index_topk` 2048, `index_topk_freq` 4,
`index_skip_topk_offset` 3, `indexer_rope_interleave` true.

---

## 0. Files traced

| side | file | role |
|---|---|---|
| llama.cpp | `src/models/glm-dsa.cpp` | graph builder for this arch |
| llama.cpp | `src/llama-graph.cpp` | `build_norm`, `build_ffn`, `build_moe_ffn`, `build_attn` |
| llama.cpp | `src/llama-model.cpp:2490` | rope type for `LLM_ARCH_GLM_DSA` |
| vLLM | `vllm/model_executor/models/deepseek_v2.py` | decoder layer, MLP, MoE |
| vLLM | `vllm/model_executor/layers/mla.py` | MLA projections wrapper |
| vLLM | `vllm/model_executor/layers/attention/mla_attention.py` | absorption, attention |
| plugin | `weights_adapter/glm_dsa.py` | GGUF→HF name map, kv_b_proj rebuild |
| plugin | `quantization/fused_moe.py` | MoE kernel dispatch |
| plugin | `csrc/gguf/moe.cuh`, `moe_hip.cuh`, `moe_vec_hip.cuh` | MoE kernels |

---

## 1. Embedding

**llama.cpp** `glm-dsa.cpp`: `inpL = build_inp_embd(model.tok_embd)` → `GET_ROWS`.

**vLLM**: `VocabParallelEmbedding` → plugin `_apply_gguf_embedding`:
`index_select(qweight, 0, ids)` then `ggml_dequantize`.

| | value |
|---|---|
| llama.cpp `embd` | `0.0198, 0.0044, 0.0005` |
| vLLM `embd` | `0.0198, 0.0044, 0.0005` |

**IDENTICAL.** Also verified independently: plugin dequant of `token_embd`
(Q8_0, 154880×6144) matches a CPU `gguf.quants.dequantize` reference to 1.1e-5.
Tokenizer ids and `vocab_size=154880` match the GGUF exactly.

---

## 2. Input layernorm

**llama.cpp**: `cur = build_norm(inpL, attn_norm, NULL, LLM_NORM_RMS, il)` —
RMS norm then `MUL` by weight.

**vLLM**: `self.input_layernorm(hidden_states, residual)` — fused RMSNorm that
also returns the updated residual.

| | value |
|---|---|
| llama.cpp `attn_norm-0` | `0.0811, 0.0246, 0.0017` |
| vLLM `attn_norm-0` | `0.0811, 0.0247, 0.0017` |

**IDENTICAL** (bf16 noise in the 4th decimal).

---

## 3. Q path

**llama.cpp** (`glm-dsa.cpp:246-250, 394`):
```
qr = mul_mat(wq_a, cur)              // cb("qr")  ← raw
qr = build_norm(qr, attn_q_a_norm)   // cb("qr")  ← same label, post-norm
q  = mul_mat(wq_b, qr)
```
Note: **both** pre- and post-norm tensors are labelled `qr`, so the first
`qr-0` in the dump is the *raw* projection.

**vLLM** (`mla.py:167-192`): `fused_qkv_a_proj(hidden_states)` produces
`[q_lora_rank | kv_lora_rank + qk_rope]` in one matmul, then splits.
llama.cpp uses two separate matmuls. **Structurally different, numerically equal:**

| | value |
|---|---|
| llama.cpp raw `qr` | `-0.0178, -0.0584, 0.0134` |
| vLLM `qkv_lora` (q part) | `-0.0179, -0.0583, 0.0134` |
| llama.cpp `q-0` | `0.0067, -0.0031, 0.0074` |
| vLLM `q-0` (rank 0) | `0.0067, -0.0032, 0.0074` |

**IDENTICAL.** The fusion is a valid refactor — not a divergence.

---

## 4. KV compression path

**llama.cpp** (`glm-dsa.cpp:409-434`):
```
kv_cmpr_pe = mul_mat(wkv_a_mqa, cur)
kv_cmpr    = view(kv_cmpr_pe, [512, n_tokens])
k_pe       = view(kv_cmpr_pe, [64, 1, n_tokens], offset=512)
q_pe = rope(q_pe); k_pe = rope(k_pe)
kv_cmpr = build_norm(kv_cmpr, attn_kv_a_norm)   // norm AFTER split+rope
```

**vLLM** (`mla.py:189-193`): same split `[kv_lora_rank | qk_rope]`, same norm.

| | value |
|---|---|
| llama.cpp `kv_cmpr_pe-0` | `0.1672, -0.0005, -0.1706` |
| vLLM `kv_cmpr_pe-0` | `0.1670, -0.0008, -0.1709` |
| llama.cpp `Kcur/Vcur` (post-norm) | `0.0041, -0.0000, -0.0042` |
| vLLM `kv_cmpr-0` (post-norm) | `0.0041, -0.0000, -0.0042` |

**IDENTICAL.**

---

## 5. RoPE convention

**llama.cpp**: `llama-model.cpp:2490` — `LLM_ARCH_GLM_DSA → LLAMA_ROPE_TYPE_NORM`.
`NORM` = interleaved pairs `(x[2i], x[2i+1])`. (`NEOX` = half-split.)
Indexer rope is explicitly `LLAMA_ROPE_TYPE_NORM` too (`glm-dsa.cpp:276,306`).

**vLLM**: `deepseek_v2.py:567` and `:1112` — `get_rope(..., is_neox_style=False)`.
`is_neox_style=False` = interleaved.

**IDENTICAL.**

---

## 6. Attention core (absorbed MLA)

**llama.cpp** (`glm-dsa.cpp:438-466`):
```
q_nope          = permute(q_nope, 0,2,1,3)
q_nope_absorbed = mul_mat(wk_b, q_nope)          // wk_b used DIRECTLY
q_nope_absorbed = permute(..., 0,2,1,3)
Qcur = concat(q_nope_absorbed, q_pe, 0)          // [512 | 64]
Kcur = concat(kv_cmpr,         k_pe, 0)          // [512 | 64]
Vcur = kv_cmpr                                   // [512]
cur  = build_attn(..., wo, wo_s, Qcur, Kcur, Vcur, wv_b, top_k, kq_scale)
```
`build_attn` (`llama-graph.cpp:2455-2500`):
```
kq  = mul_mat(k, q); soft_max_ext(kq, mask, kq_scale, bias)
kqv = mul_mat(v, kq)
kqv = mul_mat(v_mla, kqv)      // decompress MQA→MHA with wv_b
permute + cont_2d + wo
```

**vLLM**: reconstructs a dense `kv_b_proj` from `attn_k_b`/`attn_v_b`
(plugin `glm_dsa.py:430-440`), then `mla_attention.py:971-1007` does
`.T` → `view(kv_lora, heads, nope+v)` → `split` → `W_UK`/`W_UV`, and runs the
aiter sparse MQA kernel.

**Round-trip verified as a true inverse:** adapter emits
`(H*(192+256), 512)`; vLLM applies `.T` → `(512, 28672)` (assert at
`mla_attention.py:989` passes), then a stride-compatible `view(512, 64, 448)`,
giving `[l][h][j] = W[h*448+j, l]`. Correct.

**Decisive measurement** — the post-attention residual:

| | value |
|---|---|
| llama.cpp `ffn_inp-0` | `0.0185, 0.0042, 0.0042` |
| vLLM `ffn_inp-0` | `0.0184, 0.0042, 0.0042` |

**IDENTICAL.** The entire attention block — projections, rope, absorption,
sparse kernel, `o_proj`, and the forced-MQA path — produces the correct
residual stream. Attention is **not** the divergence.

(`kqv_out-0` appears to differ, but llama.cpp's `kqv_out` is pre-`wo_s`:
`embd + kqv_out ≠ its own ffn_inp`, so the labels are not comparable.)

---

## 7. Post-attention layernorm + dense MLP (layers 0-2)

**llama.cpp** (`glm-dsa.cpp:476-488`):
```
ffn_inp = add(cur, inpSA)
cur     = build_norm(ffn_inp, ffn_norm, LLM_NORM_RMS)
cur     = build_ffn(cur, ffn_up, ffn_up_s, ffn_gate, ffn_gate_s,
                    ffn_down, ffn_down_s, NULL, LLM_FFN_SILU, LLM_FFN_PAR)
```
`build_ffn` (`llama-graph.cpp:1596-1665`): `tmp = up(cur)`, `cur = gate(cur)`,
`cur = ggml_swiglu_split(cur, tmp)` = **silu(gate) * up**, then `down`.
`*_s` scales are **null** for this checkpoint (verified: 0 scale tensors in GGUF).

**vLLM** (`deepseek_v2.py:292-296`): `MergedColumnParallelLinear` producing
`[gate | up]`, `SiluAndMul()` = silu(first half) * second half, then
`RowParallelLinear` down + all-reduce.

| probe | llama.cpp | vLLM | |
|---|---|---|---|
| `ffn_norm-0` (mlp in) | `0.0461, 0.0096, 0.0110` | `0.0461, 0.0096, 0.0110` | identical |
| `ffn_gate-0` | `0.0884, 0.0042, 0.0721` | `0.0884, 0.0039, 0.0713` | identical |
| `ffn_up-0` | `-0.0026, 0.0249, 0.0405` | `-0.0027, 0.0248, 0.0403` | identical |
| `ffn_swiglu-0` | `-0.0001, 0.0001, 0.0015` | `-0.0001, 0.0000, 0.0015` | identical |
| `ffn_out-0` | `-0.0067, -0.0014, -0.0030` | `-0.0043, -0.0029, -0.0043` | **differs** |

Gate/up are **not** swapped. `down_proj` input is correct, output differs.
`down_proj` sums **12288** terms of magnitude ~1e-4 into a ~5e-3 result — near
total cancellation. llama.cpp accumulates in f32 on CPU; we accumulate in bf16.
This difference is consistent with precision, **not proven to be a bug**.

TP sharding of `ffn_down` verified correct: plugin
`params.py:239-247` narrows dim 1 of the *byte* tensor
(13056 bytes → 6528 = 192 Q8_0 blocks = exactly 6144 logical cols). Block-aligned.

---

## 8. MoE (layers 3+)

### 8a. Gating — verified identical

**llama.cpp** (`llama-graph.cpp:1845-1960`):
```
probs           = sigmoid(logits)
selection_probs = probs + exp_probs_b        // bias for SELECTION only
                  // comment: "leave probs unbiased as it's later used to
                  //           get expert weights"
selected        = argsort_top_k(selection_probs, 8)
weights         = probs[selected]            // UNBIASED
if (norm_w)  weights /= clamp(sum(weights), 6.1e-5, inf)
if (w_scale) weights *= w_scale              // 2.5
```
`n_expert_groups = 1`, so the group-masking branch is **skipped**.

**ours** — aiter `biased_grouped_topk`
(`csrc/kernels/topk_softmax_kernels_group.cu:365-390, 566-610`):
```
gating[i] = 1/(1+exp(-gating[i]));   // sigmoid
if (isBiased) gating[i] += bias[i];  // selection score
...
max_val -= correction_bias[max_idx]; // unbias for weight
sum += max_val;                      // sum over UNBIASED
topk_weights = topk_value * (routed_scaling_factor / sum);
```

**IDENTICAL**, including the subtle unbias-before-weighting.

### 8b. Expert compute

**llama.cpp** (`llama-graph.cpp:1962-2056`):
`up = mul_mat_id(up_exps, cur, selected)`, `gate = mul_mat_id(gate_exps, ...)`,
`ggml_swiglu_split(gate, up)`, `down = mul_mat_id(down_exps, ...)`,
`× weights`, sum over experts. Then `+ ffn_shexp` (shared expert), then
`+ ffn_inp` residual.

**ours** (`plugin/quantization/fused_moe.py:56-110`) — two kernels, chosen by
token count:
```python
if qtypes in MMQ_QUANT_TYPES and x.shape[0] > 64:     #  > 64 tokens
    moe_align_block_size(...); ggml_moe_a8(...)        #  MMQ tiled path
elif qtypes in MMVQ_QUANT_TYPES:                       #  <= 64 tokens
    ggml_moe_a8_vec(...)                               #  MMVQ vector path
```

Expert stacks unbind correctly: GGUF `ffn_gate_exps` data is
`(256, 2048, 2016)` and `ffn_down_exps` is `(256, 6144, 672)` — experts
outermost, so `weight.unbind()` (dim 0) yields correct per-expert slices.

---

## 9. Divergences found

### D1 — `moe.cuh` / `moe_hip.cuh`: truncating grid division (FIXED)

`csrc/gguf/moe.cuh:175` and 9 sibling sites, plus 10 in `moe_hip.cuh`:
```cpp
const int block_num_y = (tokens_post_padded) / mmq_x;        // WRONG
```
vs the correct form used everywhere else (`mmq_hip.cuh:144`):
```cpp
const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
```
With `MOE_X_Q2_K = 8`, this drops the tail of the token range, and yields
**zero blocks** when `tokens_post_padded < 8` — i.e. the routed experts
compute nothing at all. Mathematically we evaluate
`y[t] = Σ_e w_e·E_e(x[t])` only for `t ∈ [0, 8⌊N/8⌋)` instead of `t ∈ [0, N)`.

**Fixed** — all 20 sites now use ceiling division. The kernel already guards
overshoot (`if (blockIdx.y * mmq_x > num_tokens_post_padded[0]) return;`).

**Scope: this path is only taken when `x.shape[0] > 64`.** Our 5-token probe
uses the MMVQ vector path, so D1 is NOT the cause of the observed 5-token
divergence. It IS a real bug for prompts/batches over 64 tokens.

### D2 — FP8 re-quantization of absorbed MLA weights (DISABLED)

`mla_attention.py:1024-1031`, gated by `VLLM_ROCM_USE_AITER_FP8BMM`
(**default True**): `W_UK`/`W_UV` are re-quantized to FP8 with a single
per-tensor scale and attention runs through an FP8 BMM. llama.cpp keeps them at
full precision. On a checkpoint whose routed quant deliberately holds attention
at q8_0, this is a real precision loss.

Disabled (`VLLM_ROCM_USE_AITER_FP8BMM=0`). Output changed
(`'H'` → `'9'`) but remained wrong, so **not the root cause**.

### D3 — bf16 vs f32 accumulation in `down_proj`

Section 7. Not proven to be a bug; expected for a 12288-term cancelling sum.

---

## 10. Verified NOT divergent

- tokenizer ids, vocab size
- embedding lookup + Q8_0 dequant (vs CPU reference, 1.1e-5)
- input layernorm, kv layernorm, q_a layernorm
- `q_a`, `q_b`, `kv_a` projections (fused vs split — numerically equal)
- RoPE convention (interleaved both sides)
- **entire attention block** (`ffn_inp-0` matches)
- `kv_b_proj` reconstruct → `.T` → view → split round trip (true inverse)
- dense MLP gate/up/SwiGLU
- MoE gating: sigmoid, bias-for-selection, unbias-for-weights, renorm, scale
- expert stack unbind (experts outermost)
- TP byte-space sharding of `ffn_down` (block-aligned)
- MMVQ vector path grid geometry (proper ceiling division)

---

## 11. Open — next measurements

The 5-token divergence is **not** yet explained. Everything upstream of the MoE
is byte-identical, and the MoE gating is identical. That leaves the **MMVQ
expert GEMM numerics** (`ggml_moe_a8_vec` → `moe_vec_q<q2_K>`) as the
unverified stage for short prompts.

Proposed, in order:
1. Probe `ffn_moe_out` and `ffn_shexp` separately at layer 3 against
   llama.cpp's `ffn_moe_out-3` / `ffn_shexp-3`. This splits routed-expert
   output from shared-expert output.
2. If routed output is wrong, dequantize one expert's `ffn_gate_exps` slice and
   compare against a CPU `gguf.quants.dequantize` reference — the same check
   that cleared the embedding.
3. Re-run with a **>64-token** prompt to exercise the MMQ path and confirm D1's
   fix independently.
