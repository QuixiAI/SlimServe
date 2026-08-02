# DeepSeek-V4-Flash GGUF: what the adapter has to map

Read from the 0731 MXFP4 release
(`DeepSeek-V4-Flash-MXFP4Experts-...-mxfp4-0731.gguf`, 145.3 GiB, 1328 tensors).
`general.architecture` is **`deepseek4`**, now claimed by
`gguf_adapters/deepseek4.py`. The model code was already here
(`vllm/models/deepseek_v4/`, including `amd/rocm.py`); loading was the gap, and
this file is why the numbers below are what they are.

Everything below is read from the file, not inferred, so it does not need
rediscovering. `~/ds4` is a complete DeepSeek-V4-Flash engine in C with its own
ROCm backend and is the reference for the architecture semantics.

## Config: GGUF key to the field the model reads

The fields are the ones `vllm/models/deepseek_v4/**` actually dereferences.
Every GGUF key below is written without its `deepseek4.` prefix.

| model field | GGUF key | value |
| --- | --- | ---: |
| `hidden_size` | `embedding_length` | 4096 |
| `num_hidden_layers` | `block_count` | 43 |
| `vocab_size` | `vocab_size` | 129280 |
| `num_attention_heads` | `attention.head_count` | 64 |
| `n_routed_experts` | `expert_count` | 256 |
| `n_shared_experts` | `expert_shared_count` | 1 |
| `num_experts_per_tok` | `expert_used_count` | 6 |
| `moe_intermediate_size` | `expert_feed_forward_length` | 2048 |
| `norm_topk_prob` | `expert_weights_norm` | true |
| `rms_norm_eps` | `attention.layer_norm_rms_epsilon` | 1e-6 |
| `hc_mult` | `hyper_connection.count` | 4 |
| `hc_eps` | `hyper_connection.epsilon` | 1e-6 |
| `hc_sinkhorn_iters` | `hyper_connection.sinkhorn_iterations` | 20 |
| `num_hash_layers` | `hash_layer_count` | 3 |
| `index_topk` | `attention.indexer.top_k` | 512 |
| `swiglu_limit` | `swiglu_clamp_exp` | 10.0 |
| `rope_theta` | `rope.freq_base` | 10000 |
| `compress_rope_theta` | `attention.compress_rope_freq_base` | 160000 |

Also present and needed for MLA / RoPE: `attention.q_lora_rank` 1024,
`attention.key_length` and `value_length` 512, `attention.head_count_kv` 1,
`attention.output_lora_rank` 1024, `attention.output_group_count` 8,
`attention.sliding_window` 128, `rope.dimension_count` 64, yarn scaling
(`factor` 16, `original_context_length` 65536 → `context_length` 1048576),
`expert_gating_func` 4 (= sqrtsoftplus), `expert_weights_scale` 1.5, and
`nextn_predict_layers` 1 — which is a lie about this file: it carries no MTP
tensors, so the config sets `num_nextn_predict_layers` to 0 and the nextn head
comes from the separate DSpark drafter GGUF.

## Tensors, and the parts that are NOT uniform across layers

Names follow the ordinary `blk.N.*` convention, so the map is mechanical — but
four groups are present on only some layers, and assuming 43 everywhere is the
easy way to get a confusing load failure:

| tensor | layers |
| --- | --- |
| `attn_compressor_{kv,gate,norm}.weight` | 41 — **missing on 0, 1** |
| `indexer.*`, `indexer_compressor_*` | 21 — **even layers 2..42 only** |
| `exp_probs_b.bias` | 40 — **missing on 0, 1, 2** |
| `ffn_gate_tid2eid.weight` | 3 — **layers 0, 1, 2 only** (the hash layers) |

So layers 0–2 are the hash layers (they carry `ffn_gate_tid2eid` and no
`exp_probs_b`), and the sparse indexer runs on even layers from 2 up. Present on
all 43: `attn_{norm,kv,kv_a_norm,q_a,q_a_norm,q_b,output_a,output_b,sinks}`,
`ffn_{norm,gate_inp,gate_exps,up_exps,down_exps}`,
`ffn_{gate,up,down}_shexp`,
and `hc_{attn,ffn}_{base,fn,scale}`. Global: `token_embd`, `output`,
`output_norm`, `output_hc_{base,fn,scale}`.

Note `attn_sinks` (attention sinks) and the `hc_*` triples per layer — the
hyper-connection mix/scale/base — have no analogue in the GLM adapter.

## Quantization

1328 tensors: 492 F32, 359 F16, 345 Q8_0, **129 MXFP4**, 3 I32. The MXFP4 ones
are the routed experts, stored per layer as `[4096, 2048, 256]` (gate/up) and
`[2048, 4096, 256]` (down) — expert-major, so the MoE grouped path indexes the
last dim.

MXFP4 reading is fully implemented: dequant, the q8_1 vector paths and the MMQ
tile paths (`ggml-common.h`, `vecdotq.cuh`, `mmvq.cuh`, `moe_vec.cuh`,
`mmq.cuh`, `moe.cuh`, and `MXFP4_QUANT_TYPES` in the gguf quant utils, which is
now in all three of `DEQUANT`/`MMVQ`/`MMQ`).

The MoE tile path is 1.5x the vec path at 32 routed tokens and 2.4x at 512, and
loses below ~8, so the existing `_moe_vec_row_limit` crossover does the right
thing unchanged. No kernel work is left for this model: the experts have both
paths and the DSA indexer is already covered.

## Already covered

The DSA indexer is `head_count` 64, `key_length` 128 — exactly the second
specialization of `fp8_mqa_logits` in `csrc/quixicore/rocm/`, already verified
bitwise at that shape. No indexer kernel work is needed.

`vllm/models/deepseek_v4/` is present and its ROCm backend (`amd/rocm.py`) is a
real implementation, not a stub: sparse-MLA attention, ragged top-k/SWA index
metadata, CUDA-graph-safe buffers, aiter prefill/decode dispatch. It does still
carry its own Triton kernels for index packing, which the Triton purge has not
reached.

## The five changes loading needed — all done

Kept because the reasoning is not recoverable from the result, and because two
of them were pre-existing gaps rather than new work:

1. **The adapter** (`deepseek4.py`). Static rename tables, since transformers
   has no entry for this architecture and the default adapter's
   introspect-the-HF-model trick cannot work. Unlike `glm_dsa` nothing needs
   assembling: every one of the 1328 tensors is a pure rename, because each
   fused module is fed the pre-fusion shard names `stacked_params_mapping`
   expects. Only `attn_output_a` and `output.weight` are dequantized.
2. **`_prepare_adapter` hardcoded `GlmDsaGGUFAdapter`.** There was no adapter
   registry — `BaseGGUFWeightsAdapter.matches()` is declared and still has no
   call site anywhere. An adapter claiming `deepseek4` was unreachable until
   that function dispatched on architecture, which it now does.
3. **`gguf_config_parser` had the same problem** one layer up: it called a
   builder hand-written for `glm-dsa.*` keys that returns a `Glm5vConfig`, with
   no branch on `general.architecture`. Also dispatched now.
4. **`DeepseekV4Config` was registered but did not exist.** The GLM-only
   specialization deleted the module and left `config.py` and
   `configs/__init__.py` pointing at it, so any non-GGUF config path raised
   ImportError. Restored from that commit's parent.
5. **No registry entry** mapped an architecture to `vllm.models.deepseek_v4`,
   though `_resolve_module_name` names that package as its example of the
   layout. Restored, with the DSpark draft and MTP entries.

The model code reads `hf_config` by plain attribute access, so it needs no
particular config class — an object carrying the fields in the table above is
enough, which is how the GLM path already works.

## What only showed up by running it

- **`block_size` must be 256.** The DeepSeek-V4 sparse MLA backend reports
  `[256]`; the GLM value of 64 fails at KV-cache setup with "no common block
  size for 64", which does not name the backend that rejected it.
- **`ffn_gate_tid2eid` is I32**, and the GGUF weight iterator treated anything
  outside F32/BF16/F16 as quantized. Its scalar type tag was emitted under a
  name with no "weight" to replace, so the tag landed on the table's own name
  and the loader saw a 0-d tensor for a `[129280, 6]` parameter.
- **`attn_output_a` cannot stay packed**: the ROCm inv-rope path reads
  `wo_a.weight` directly to build a per-group einsum operand.
- **`output.weight` cannot either**: `lm_head` is built unquantized, and
  renaming a quantized tensor to `.qweight` also drops it out of the model's
  `head.weight -> lm_head.weight` suffix rule.
- **The chat template must not be used.** DeepSeek-V4 renders prompts through
  `deepseek_v4_encoding.encode_messages`, and the GGUF's own jinja template
  binds `messages` itself, so rendering it raises "got multiple values for
  keyword argument 'messages'".
