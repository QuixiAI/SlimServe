# GGUF quant analyses

Per-model notes on how each GGUF build spends its bits, and what that
implies for serving it here.

## DeepSeek v4 Flash 0731 antirez

```text
   Variant        Routed-expert quantization            Size    Positioning
  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━━━━━
   IQ2XXS+Q2K     gate/up IQ2_XXS, down Q2_K, every     80.764 GiB   smallest
                  layer
  ─────────────  ───────────────────────────────────  ───────────  ───────────────
   Hybrid         layers 0-36 as above; 37-42 all       90.889 GiB   better compact
                  Q4_K
  ─────────────  ───────────────────────────────────  ───────────  ───────────────
   MXFP4          gate/up/down all MXFP4               145.264 GiB  FP4 throughput
  ─────────────  ───────────────────────────────────  ───────────  ───────────────
   Q4K            gate/up/down all Q4_K                153.327 GiB  quality first
```

### 1. IQ2XXS-w2Q2K — 80.8 GiB

  The routed experts use:

- ffn_gate_exps and ffn_up_exps: IQ2_XXS, effectively 2.0625 bits/weight
- ffn_down_exps, also called W2: Q2_K, effectively 2.625 bits/weight

  IQ2_XXS is an extremely compact importance-aware quantization. Q2_K retains
  slightly more information for the down projection, where
  quantization damage can be especially visible.

  This is the memory-saving version. Expect the largest reduction in reasoning,
  coding accuracy, and difficult long-context behavior compared with
  the 4-bit variants.

### 2. Layers37–42Q4K hybrid — 90.9 GiB

  This is the same layout as the 80.8 GiB model for layers 0–36, but the final
  six zero-indexed layers—37 through 42—use Q4_K for all routed-
  expert matrices.

  So:

- Layers 0–36: gate/up IQ2_XXS, down Q2_K
- Layers 37–42: gate/up/down Q4_K

  It spends about 10.1 GiB more to preserve the model’s final transformation
  stages. It should recover some quality versus the full IQ2/Q2 version
  without approaching the 153 GiB footprint of full Q4_K.

  fixed is only present in the filename; the GGUF metadata does not describe
  what was fixed. The file itself parses correctly.

### 3. MXFP4 experts — 145.3 GiB

  Every routed-expert matrix uses MXFP4:

- 4-bit E2M1 floating-point values
- One shared E8M0 scale per block of 32 values
- Effective storage: 4.25 bits/weight

  MXFP4 is designed around efficient FP4 execution. It can be the
  throughput-oriented choice when the serving hardware and kernel backend have a
  strong MXFP4 path. On hardware without efficient FP4 support, its theoretical
  advantage can disappear because of conversion or emulation
  overhead.

  Unlike the other three, this file contains no recorded importance-matrix
  metadata.

### 4. Q4_K experts — 153.3 GiB

  Every routed-expert gate, up, and down tensor uses Q4_K, effectively 4.5
  bits/weight.

  Q4_K uses sub-block scales and offsets and this file was created with an
  importance matrix. Of these four, it is the safest quality-first
  choice, assuming sufficient memory.

  It is about:

- 8.1 GiB larger than MXFP4
- 62.4 GiB larger than the hybrid
- 72.6 GiB larger than IQ2/Q2

---

## Kimi K3 DwarfStar Q2 — antirez

`Kimi-K3-IQ2_XXS-Q2_K.gguf`, 858,760,729,248 bytes (799.8 GiB), shipped as five
160 GiB parts to be concatenated in lexical order. 2736 tensors, GGUF v3.

Read from the header of part 1 — GGUF puts all metadata and the full tensor
index at the front, so the layout below needs no reassembly. Summing the
inventory reproduces the stated byte count exactly, which is the check that the
reading is right.

### Shape

```text
   general.architecture      kimi-k3          (unmapped; no adapter claims it)
   block_count               93
   embedding_length          7168
   expert_count              896              routed
   expert_used_count         16               top-k
   expert_feed_forward_length 3072
   routed_hidden_length      3584
   context_length            1048576
   vocabulary_size           163840
   vision                    false
```text

### One model, two quant strategies, two naming conventions

Only the routed experts and the shared experts are quantized below 8 bits.
Every attention matrix, every norm, the embedding and the LM head are BF16.

```text
   Tensor group                        Type      Count  Shape
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━  ━━━━━  ━━━━━━━━━━━━━━━━━━━
   blk.N.ffn_gate_exps / ffn_up_exps   IQ2_XXS     184  [3584, 3072, 896]
   blk.N.ffn_down_exps                 Q2_K         92  [3072, 3584, 896]
   ..shared_experts.{gate,up,down}     MXFP4       276  [7168, 6144]
   ..routed_expert_up_proj             Q8_0         92  [3584, 7168]
   layer 0 mlp.{gate,up}_proj          Q8_0          2  [7168, 33792]
   attention, norms, embed, lm_head     BF16       1584
   small params (biases, conv, A_log)   F32         506
```

The split follows the same reasoning as the DeepSeek-V4 IQ2XXS build: gate and
up take the most aggressive format, down keeps more precision because
quantization damage there is the most visible.

Two things are unusual and both matter for loading.

**Mixed naming.** The 3D routed-expert stacks use ggml's `blk.N.ffn_*_exps`
convention; everything else uses HF names (`language_model.model.layers.N.*`).
A name map has to handle both, and the `language_model.` prefix suggests the
converter was written for a multimodal wrapper even though `vision` is false.

**Tiktoken vocab.** There is no `tokenizer.ggml.tokens`/`merges`. The vocabulary
is a single `tokenizer.kimi-k3.tiktoken` blob of base64 token/rank lines, so the
byte-BPE reconstruction used for GLM-5.2 and DeepSeek-V4 does not apply.

### Hybrid attention

The layer stack is not uniform, and the per-layer tensor sets say so:

- **69 layers (0–90)** carry `q_conv1d`, `k_conv1d`, `v_conv1d`, `A_log`,
  `dt_bias`, `f_a_proj`, `f_b_proj`, `o_norm` — a gated linear/SSD attention.
- **24 layers (3–92)** carry `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`,
  `kv_b_proj` — DeepSeek-style MLA.
- All 93 share `g_proj` and `o_proj`.

69 + 24 = 93, so it is roughly a 3:1 interleave of linear attention to full MLA.
Layer 0 is dense (`mlp.{gate,up,down}_proj` at 33792); layers 1–92 are MoE.

Every layer also has scalar residual gates — `mlp_res_proj` and
`self_attention_res_proj`, both `[7168, 1]` — with a matching pair at the model
level. Nothing in the GLM or DeepSeek-V4 adapters corresponds to these.

### What it would take to serve

The quant side is already done. IQ2_XXS, Q2_K and MXFP4 are exactly the three
formats this repo now has dequant, vector and MMQ tile kernels for.

The pairing is also the one the recent MoE work exists for: w1 is IQ2_XXS
(tile width 8) and w2 is Q2_K (tile width 4). Before the alignment fix those two
widths could not share one `moe_align_block_size` layout — the mismatch read the
wrong expert per tile and produced fluent nonsense. It works now because the row
alignment is the LCM of the two widths and each kernel gets an `expert_ids`
expanded to its own tile count.

Three real blockers remain, in order of severity:

1. **896 experts against a 255-expert kernel limit.** `moe_q` bails with
   `if (exp_idx > 255 || exp_idx < 0) return;`. Experts 256–895 would be skipped
   silently — no crash, just most of the model missing. This is a hard blocker
   and the first thing to fix; the guard predates any model with more than 256
   experts and the bound looks like a sentinel check that quietly became a
   capacity limit.

2. **No `kimi-k3` adapter, config builder or registry entry.** The same five
   pieces the `deepseek4` work needed, plus a tiktoken vocabulary path that
   neither existing model required.

3. **Hybrid SSD/MLA attention.** The linear-attention layers are a different
   operator from anything currently served here. Whether `vllm/models/` already
   has a Kimi K3 implementation is the thing to check before estimating this.

### Footprint

799.8 GiB of weights needs 5 MI300X at minimum for the weights alone, and
realistically 8 with a usable KV pool — this is the first model discussed here
that does not fit the 2–4 GPU envelope. At 896 experts with top-16 routing,
expert-parallel placement is worth considering before tensor-parallel.

---

## GLM-5.2 UD routed — antirez / Unsloth

`~/models/antirez-GLM-5.2-gguf`, three unsharded builds of the model this
fork is tuned for. Architecture `glm-dsa`, 1809 tensors each, and the same
structure throughout: the builds differ *only* in how the routed experts are
stored.

```text
   Build                            Routed experts        Size    imatrix
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━
   UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K IQ2_XXS, blk 78 Q2_K  196.6 GiB   yes
   UD-Q2_K_RoutedQ2K                Q2_K                  244.0 GiB   yes
   UD-Q4_K_RoutedQ4K                Q4_K                  404.4 GiB   yes
```

### Shape and routing

```text
   general.architecture     glm-dsa          256x22B, Zai Org GLM 5.2
   quantized_by             Unsloth
   expert_count             256
   expert_used_count        8
   expert_shared_count      1
   expert_feed_forward_length 2048
   feed_forward_length      12288            dense layers
   leading_dense_block_count 3               layers 0-2 are dense
   expert_gating_func       2                sigmoid
   expert_weights_scale     2.5
   attention.head_count     64
   attention.head_count_kv  1                MLA
   rope.freq_base           8000000
   vocab_size               154880
   nextn_predict_layers     1
```

### Where the bits go

Only the routed experts are below 8 bits. Everything else — attention, the
shared expert, norms, embeddings, output — is Q8_0 or F32, identically in all
three files:

```text
   Group                              IQ2 build   Q2_K build   Q4_K build
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━━━━━
   ffn_{gate,up,down}_exps (76 lay)   IQ2_XXS x225  Q2_K x228   Q4_K x228
                                      + Q2_K x3
   everything else per layer          Q8_0 x870, F32 x708
   globals                            Q8_0 x2, F32 x1
```

Routed experts live on layers 3–78, which is `leading_dense_block_count = 3`
and 76 MoE layers, three tensors each: 228. Expert stacks are
`[6144, 2048, 256]` for gate/up and `[2048, 6144, 256]` for down.

`blk78Q2K` in the first filename is literal and small: layer 78 — the last MoE
layer — keeps its three expert tensors in Q2_K while the other 75 layers go to
IQ2_XXS. Three tensors out of 228. It is the same idea as DeepSeek-V4's
`Layers37-42Q4K` hybrid but at a much smaller dose, spending a little to
protect the final transformation stage.

All three carry imatrix provenance in the metadata —
`unsloth_calibration_GLM-5.2.txt`, 88 chunks, 1002 entries — and
`imatrix_unsloth.gguf_file` (1.1 GiB) sits beside them, so these are
reproducible rather than opaque.

### Serving

Nothing to do: this is the tuned path. `glm-dsa` is what `GlmDsaGGUFAdapter`
and `build_config_from_gguf` were written for, the tokenizer is the ordinary
`gpt2` BPE with the `glm4` pre-tokenizer that `build_bpe_tokenizer` handles,
and `run-glm-optimized.sh` already selects between these three by `--quant`.

Kernel coverage is complete for all three. IQ2_XXS gained an MMQ tile path
recently, so the smallest build no longer falls back to the vector kernel at
every batch size; Q2_K and Q4_K already had one. The mixed IQ2_XXS/Q2_K layer
78 is also the heterogeneous tile-width case — IQ2_XXS uses width 8 and Q2_K
width 4 — which works because row alignment is now the LCM of the two rather
than whichever type came first.

One practical note: the run script does not point here. It serves the sharded
copies under `GLM-5.2-Vision-GGUF/antirez-routed/`, which are the same three
models split into 5, 6 and 10 parts. Both sets are present, so roughly 845 GiB
is duplicated. That is affordable today — 74 TiB free — but it is worth knowing
that deleting this directory would not affect serving, and that these unsharded
files are the more convenient ones to inspect.
