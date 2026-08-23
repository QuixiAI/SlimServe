# Qwen3.8-27B + DFlash 2 on Metal: extracted reference maps

2026-08-19. Companion to HANDOFF.md (staged plan) and notebook entries
(1)-(3). Everything below is extracted from primary sources: the unsloth /
incoai / z-lab configs, HF GGUF metadata, llama.cpp master (arch `qwen35`),
and llama.cpp PR #27342 (`dflash2-pr` branch in ~/llama.cpp, commit
5ecbe1ac1). Verify against the downloaded GGUFs before coding the parsers.

## Target GGUF (arch `qwen35`) -- from llama.cpp src/models/qwen35.cpp

Hparam keys (`qwen35.` prefix):

- `attention.layer_norm_rms_epsilon`; `rope.dimension_sections` (MRoPE,
  4 entries); `ssm.conv_kernel` (4), `ssm.inner_size`, `ssm.state_size`
  (128 = head dim for both K and V), `ssm.time_step_rank` (48 = n_v_heads),
  `ssm.group_count` (16 = n_k_heads); `nextn_predict_layers` (MTP count;
  SKIP those layers); `attention.recurrent_layers` (bool array) or
  fallback `full_attention_interval` (default 4): layer i is recurrent iff
  `(i + 1) % interval != 0` -- i.e. layers 3, 7, ..., 63 are full attention
  (16 of 64), the rest gated-deltanet.
- Derived dims (27B): key_dim = 128*16 = 2048, value_dim = 128*48 = 6144,
  conv_dim = key_dim*2 + value_dim = 10240.

Per-layer tensors, recurrent (gated-deltanet) layers:

| GGUF name | shape | role |
|---|---|---|
| `blk.N.attn_norm.weight` | {5120} | pre norm |
| `blk.N.attn_qkv.weight` | {5120, 2*2048+6144} | fused qkv |
| `blk.N.attn_gate.weight` | {5120, 6144} | output gate (swish) |
| `blk.N.ssm_conv1d.weight` | {4, 10240} | short causal conv |
| `blk.N.ssm_dt.bias` | {48} | dt bias |
| `blk.N.ssm_a` | {48} | decay (no-scan form) |
| `blk.N.ssm_beta.weight` | {5120, 48} | beta head |
| `blk.N.ssm_alpha.weight` | {5120, 48} | alpha head |
| `blk.N.ssm_norm.weight` | {128} | per-head RMS on V out |
| `blk.N.ssm_out.weight` | {6144, 5120} | out proj |
| `blk.N.attn_post_norm.weight` | {5120} | post norm |
| `blk.N.ffn_{gate,up,down}.weight` | 5120x17408 | SwiGLU |

Full-attention layers: `attn_norm`, fused qkv via create_tensor_qkv with
**Q width = n_embd_head_k * n_head * 2** (the x2 is the attention output
gate, `attn_output_gate: true`), `attn_output.weight`
{n_embd_head_k*n_head, 5120}, `attn_q_norm`/`attn_k_norm` {256} (per
head_dim), `attn_post_norm`, same FFN. GQA 24/4, head_dim 256,
partial_rotary_factor 0.25, interleaved MRoPE sections [11, 11, 10].

MTP/NextN block (appended beyond layer 63): `blk.64.*` plus
`blk.64.nextn.{eh_proj,enorm,hnorm,embed_tokens,shared_head_head,
shared_head_norm}` -- DO NOT LOAD (DFlash 2 replaces MTP; llama.cpp gates
these behind `ml.load_mtp`).

Top level: `token_embd.weight` {5120, 248320}, `output_norm.weight`,
`output.weight` (optional; falls back to tied embeddings).

DUMPED 2026-08-19 (866 tensors; deltas vs the llama.cpp-derived map):

- Full-attention layers ship SPLIT projections: `attn_q` {5120, 12288}
  (24 heads x 256 x 2 -- the output gate rides fused inside Q),
  `attn_k`/`attn_v` {5120, 1024}, `attn_output` {6144, 5120},
  `attn_q_norm`/`attn_k_norm` {256}. llama.cpp's create_tensor_qkv
  accepts both fused and split; this artifact is split.
- Linear layers ship FUSED `attn_qkv` {5120, 10240} (= 2*2048 + 6144)
  plus `attn_gate` {5120, 6144} and the ssm_* set exactly as mapped
  (`ssm_conv1d` {4, 10240} F32, `ssm_a`/`ssm_dt.bias` {48} F32,
  `ssm_alpha`/`ssm_beta` {5120, 48} Q8_0, `ssm_norm` {128} F32,
  `ssm_out` {6144, 5120}).
- The pre-FFN norm is named `post_attention_norm` (not ffn_norm).
- `blk.64.*` is the MTP block incl. `nextn.{eh_proj,enorm,hnorm,
  shared_head_norm}` -- skip on load (no nextn embed/head: tied).
- Metadata: block_count 65 (INCLUDES the 1 MTP layer;
  nextn_predict_layers 1), full_attention_interval 4 (no
  recurrent_layers array), rope.dimension_sections [11, 11, 10, 0],
  rope.dimension_count 64, tokenizer pre "qwen35". Real file size
  9,828,981,664 B, sha256 fd4730dd8aad...4032ddb0 (the tree-API numbers
  first recorded in notebook (1) were wrong).

**QUANT MIX (UD-Q2_K_XL) -- the Metal kernel matrix problem.** The
artifact spreads 15 tensor types across the serving path: F32 x360,
IQ3_XXS x112, Q8_0 x98 (incl. ssm_alpha/beta), IQ2_S x67, IQ3_S x57,
IQ2_XXS x48, IQ2_XS x34, Q4_K x21, IQ1_S x20 (ffn only), IQ4_XS x19,
Q2_K x16 (incl. token_embd), Q6_K x6, Q3_K x5, Q5_K x2, IQ1_M x1.
The Muse-campaign Metal kernels cover Q4_K/Q5_K/Q6_K (+ the DSV4 path's
IQ2_XXS/Q2_K); IQ1_S/IQ1_M/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS/Q8_0/Q3_K
coverage on the Metal GEMV/GEMM path must be audited per kernel.
Bring-up plan: dequantize unsupported types to fp16 at load (correctness
first, memory-priced), then add native decode per format in
bytes-moved-per-step order. llama.cpp's ggml-metal kernels serve this
exact artifact and are the porting reference.

Vision: mmproj-F16.gguf, DUMPED 2026-08-19 (ground truth, arch `clip`,
`clip.projector_type = qwen3vl_merger`):

- Keys: `clip.vision.{projection_dim 5120, image_size 768, patch_size 16,
  embedding_length 1152, feed_forward_length 4304, block_count 27,
  attention.head_count 16, attention.layer_norm_epsilon 1e-6,
  spatial_merge_size 2, image_mean/std [0.5]*3,
  is_deepstack_layers all-false}`, `clip.use_gelu = true`.
- Tower blocks `v.blk.N.`: `attn_qkv.{weight,bias}` {1152, 3456} (fused,
  WITH bias), `attn_out.{weight,bias}`, `ffn_up`/`ffn_down` (+bias, gelu,
  NO gate), `ln1`/`ln2` `.{weight,bias}` -- LayerNorm, not RMS. Weights
  F16, biases/norms F32 (dequant not needed; tower loads as-is).
- Top level: `v.patch_embd.weight` {16,16,3,1152} + `.weight.1` (second
  temporal tap) + bias; `v.position_embd.weight` {1152, 2304} (48x48);
  `v.post_ln.{weight,bias}`; merger MLP `mm.0.{weight,bias}` {4608,4608}
  + GELU + `mm.2.{weight,bias}` {4608, 5120} where 4608 = 1152 * 4
  (2x2 spatial-merge concat).

## Drafter GGUF (arch `dflash`, DFlash 2) -- from llama.cpp PR #27342

Discriminator inside arch "dflash" (now three-way in our fork's parsers):

- `dflash.expert_count` present -> DeepSeek-V4 DSpark-style drafter (MoE).
- `dflash.selector_rank` present (or tensor `selector_hidden.weight`
  exists) -> **DFlash 2**.
- neither -> Muse-Glimmer dense DFlash 1.

Metadata, DUMPED 2026-08-19 (ground truth): `dflash.block_count 5`,
`context_length 262144`, `embedding_length 5120`, `feed_forward_length
17408`, `attention.head_count 32`, `head_count_kv 8`,
`attention.causal = False` (EXPLICIT -- all layers non-causal despite
`sliding_window 2048` + `sliding_window_pattern [True]*5`; plumb through
dflash_config.causal, which overrides per-layer causality in
qwen3_dflash._dflash_layer_causal), `rope.freq_base 1e7`,
`key_length/value_length 128`, `block_size 8` (TOTAL rows incl. the
anchor: llama.cpp walks i in [1, 8) = 7 drafted tokens, matching the
profile's num_speculative_tokens 7), `conv_kernel_size 2`,
`conv_group_size 16`, `selector_rank 256`, `selector_top_k 16`,
`target_layers [6, 20, 34, 48, 62]` (1-BASED, subtract 1 like Muse).
Tokenizer: full gpt2 BPE with `tokenizer.ggml.pre = "qwen35"`,
`mask_token_id 248070`, vision-aware chat template -- so this dflash
file CAN reach tokenizer construction; the Muse hard-assumption in
tokenizers/registry.py must be discriminated.

Tensor inventory (81 tensors: 32 F32, 45 Q4_K, 4 Q6_K): per layer
`blk.N.{attn_q 5120x4096, attn_k 5120x1024, attn_v 5120x1024,
attn_output 4096x5120, attn_q_norm/attn_k_norm 128, attn_norm,
ffn_gate/up 5120x17408, ffn_down 17408x5120, ffn_norm}` -- SPLIT
q/k/v, same shape family as the Muse drafter -- plus `attn_conv_base`
{5120,2,2} F32, `attn_conv_proj.weight` {5120,1280} Q4_K,
`ffn_conv_base`, `ffn_conv_proj.weight` (1280 = 2 taps x 2 sides x 320
groups). Top level: `fc.weight` {25600,5120} (5x5120 concat encoder),
`enc.output_norm.weight`, `output_norm.weight`, `selector_hidden.weight`
{5120,256}, `selector_predecessor.weight`/`selector_successor.weight`
{256,248320} both Q4_K. NO embed/lm_head (shared with the target, like
Muse).

New tensors:

| GGUF name | shape | role |
|---|---|---|
| `blk.N.attn_conv_base` | {5120, 2, 2} | base kernel, [tap][side] |
| `blk.N.attn_conv_proj.weight` | {5120, 2*2*320} | dynamic coeff proj |
| `blk.N.ffn_conv_base` | {5120, 2, 2} | same for FFN pair |
| `blk.N.ffn_conv_proj.weight` | {5120, 1280} | " |
| `selector_predecessor.weight` | {256, 248320} | A: token embedding |
| `selector_successor.weight` | {256, 248320} | B: token embedding |
| `selector_hidden.weight` | {5120, 256} | H: context gate |

(320 = 5120 / conv_group_size 16 groups; llama.cpp asserts
n_embd % group_size == 0 and n_embd >= top_k*(top_k+1).)

## DFlash 2 forward semantics (llama.cpp PR, src/models/dflash.cpp)

Per drafter layer, with `noise_norm` = the pre-attention normalized hidden
and `cur` the running residual stream:

1. `attn_dynamic = attn_conv_proj @ noise_norm` -- computed ONCE, reused
   for both sides. Reshaped {n_groups, kernel, 2 sides} per token.
2. Pre-attn: `noise_norm = conv(noise_norm, attn_dynamic, base, side=0)`.
3. Post-attn: `cur = conv(cur, attn_dynamic, base, side=1)`.
4. `ffn_dynamic = ffn_conv_proj @ cur` (pre-FFN stream), then pre-FFN
   side 0 on the FFN input, post-FFN side 1 on the FFN output.

conv(x, dyn, base, side), block-local over each 1+N draft block:

```
for tap in {0, 1}:
    weight = base[:, tap, side] + repeat_per_group(dyn[group, tap, side])
    out   += weight * shift_within_block(x, tap)   # tap 1: predecessor row
```

shift zero-pads at block row 0; row 0 IS the anchor (last verified token)
in the 1+N layout, so position 1's tap-1 reads the verified token exactly
as the blog's figure 4 requires.

## Selector semantics

Graph side (replaces logits as the drafter output -- "DFlash2 never
consumes raw logits"):

- `candidates = top_k(logits, 16)` per position; `U = logits[candidates]`.
- `gate = selector_hidden @ hidden_state` (rank 256) per position.
- For pos = 1..block-1: `A = selector_predecessor[prev ids]` (pos 1: the
  anchor token id), `scores[a, b] = <A(a) * gate_pos, B(b)> + U_pos[b]`
  -> a 16x16 matrix per position (pos 1's rows identical).
- Packed per position: `[candidate_ids as f32 (16), scores (256)]`,
  padded to n_embd -- the "lattice".

Host walk (common/speculative.cpp): predecessor := 0; per position, at
T=0 take argmax over `scores[predecessor, :]`; at T>0 softmax
`(s - max)/T` over the 16, sample, and KEEP the 16-way distribution per
position for lossless rejection sampling. Emit the chosen candidate id;
its index becomes the next predecessor.

## vLLM integration points (fork, scouted 2026-08-19)

- `vllm/transformers_utils/gguf_config_parser.py:34-83`: add `qwen35`
  branch (target) and a third dflash probe branch on
  `dflash.selector_rank`.
- `vllm/model_executor/model_loader/gguf_loader.py:59-101`: same two
  dispatch additions; new adapters under
  `model_loader/gguf_adapters/` (target text+vision, drafter).
- Target LM half: the fork vendors the full upstream GDN layer
  (`layers/mamba/gdn/qwen_gdn_linear_attn.py`, Triton fla ops under
  `vllm/third_party/flash_linear_attention`) -- algorithm in-tree, but
  kernels are CUDA/Triton. Metal needs equivalents of
  chunk_gated_delta_rule (prefill), fused recurrent decode, and
  causal_conv1d update. llama.cpp qwen35.cpp + llama-memory-recurrent
  are the platform-portable reference.
- Drafter class: subclass `qwen3_dflash.py::DFlashQwen3*` (Muse
  precedent: override precompute_and_store_context_kv and draft
  sampling for Metal); add the 4 conv applications + 2 dynamic
  projections per layer and the selector head.
- Speculator: `v1/worker/gpu/spec_decode/dflash/speculator.py` draft
  sampling is top-1 only; add a selector path (top-16 + pair scores +
  walk) gated on the drafter config carrying selector_rank. Keep method
  string "dflash" (pydantic Literal + extra=forbid makes new methods
  expensive; upstream uses "dflash" for DFlash 2 too).
- Registry: arch name must start with "DFlash"
  (`models/registry.py:387-393` + EAGLEConfig prefix rule).
- Fused Metal `dflash_sample_greedy` is gated m in [9,17]; block 8 = 8
  rows is outside AND DFlash 2 needs the selector anyway -- eager
  selector first, fuse into the step command buffer later.
- TOKENIZER TRAP (`vllm/tokenizers/registry.py:228-237`): arch "dflash"
  is hard-assumed to be Muse-Glimmer's drafter ("any dflash file
  reaching tokenizer construction is Muse-Glimmer's") and routed to the
  Muse BPE builder. If the qwen38 DFlash 2 drafter GGUF ever reaches
  tokenizer construction, that assumption breaks -- extend the branch
  with the selector_rank/expert_count discriminator or a tokenizer.ggml
  presence check. Target arch "qwen35" falls through to the generic
  `build_tokenizer_from_gguf`; verify its pre-tokenizer split against
  the file's `tokenizer.ggml.pre` once downloaded.

## GDN speculative-decode state rollback (design, pre-implementation)

Speculation on this target is not just batching: the 48 recurrent layers
must not commit state for rejected draft tokens. The Triton path's
mechanism (fused_recurrent.py, IS_SPEC_DECODING + INPLACE_FINAL_STATE):

- The GDN metadata builder gives each spec request a row of per-position
  state slots (`spec_state_indices_tensor[req, pos]`); during the verify
  forward the kernel STORES the running state after EVERY token into that
  position's slot (conv state analogously via causal_conv1d_update's
  num_accepted_tokens/query_start_loc form).
- The next step loads its initial state from slot
  `spec_state_indices[req, num_accepted_tokens - 1]` -- i.e. the state
  right after the last accepted token; rejected positions' slots are
  simply never read again.
- llama.cpp calls the same requirement `rs_rollback`
  (llm_arch_supports_rs_rollback lists QWEN35).

MPS plan: extend `_forward_core_mps`'s scan to (a) accept a per-token
state-slot row and scatter the fp32 state after each position (the scan
already materializes it -- the store is a per-position copy, batched
across heads), (b) same for the conv tail window, (c) honor
num_accepted_tokens when loading initial states. Cost estimate per
verify step: 8 positions x 48 layers x (48h x 128 x 128 fp32) ~ 12 MB/req
of state traffic -- noise next to the weight pass. Implement AFTER plain
decode reaches greedy parity; the current spec-mask NotImplementedError
is the guard.

## Fused GDN decode step (design, pre-implementation)

Measured: plain decode ~400 ms/step vs a ~46 ms bandwidth floor at current
residency (notebook (19)) -- dispatch/python-bound. Structure of one decode
step today: 64 layers x (norm, in_proj_qkvz GEMV, in_proj_ba GEMV, conv
update ~4 small ops, scan step ~8 small ops incl. fp32 state read/write,
gated norm, out_proj GEMV, residual) + 16 full-attn layers' attention +
final norm + lm_head ~= 1000+ host dispatches/step plus engine overhead.
The Muse campaign killed the same disease with muse_step: one Metal
command buffer emitting the whole forward via a host-side emit chain
(qc_metal_serving.mm), reusing the eager kernels' variant tables.

Plan (after spec correctness lands, since the fused step must reproduce
the spec paths too):

1. Persistent GPU-resident GDN state: fp32 ssm state (48h x 128 x 128 per
   layer per slot) and conv tail (10240 x 3) already live in the mamba
   cache tensors -- the fused step reads/writes them in place; no new
   allocation.
2. New kernels needed (small, all elementwise/GEMV-class):
   qwen_gdn_step (conv-tap update + l2norm + gating + single-token delta
   rule + gated RMS norm, one dispatch per layer covering all 48 heads;
   fp32 accumulate) and a fused residual/norm site. The GEMVs reuse the
   existing qgemv/qgemm_sm routes via the emit chain like muse_step's
   emit_matvec.
3. Full-attention layers: reuse the muse fused-step attention site;
   head_dim 256 needs the paged fast path extended (currently 64/128) or
   the SDPA site -- measure first, the 16 layers may be minor at decode.
4. Step ledger targets: quantized bytes/step ~9.2 GiB post-native-IQ
   (~20 ms at stream rate) + GDN state traffic ~0.6 GiB (~1.3 ms) +
   dispatch (goal: one command buffer, <1 ms host) => plain decode
   ceiling ~30-40 tok/s, matching llama.cpp's 35.67 which serves the
   same bytes with the same hardware.
5. Order of perf work: (a) fused GDN step emit chain (the 8.7x), (b)
   native IQ1_M/IQ2_S/IQ3_S decode (the 2.3x bytes), (c) head_dim-256
   paged attention, (d) spec verify-band tuning (M=8 falls below the
   muse sm-band's 9-row floor; either extend qgemm_sm variants to M=8
   or pad).
