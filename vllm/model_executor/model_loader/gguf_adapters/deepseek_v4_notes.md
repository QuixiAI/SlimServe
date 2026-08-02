# DeepSeek-V4-Flash GGUF: what the adapter has to map

Read from the 0731 MXFP4 release
(`DeepSeek-V4-Flash-MXFP4Experts-...-mxfp4-0731.gguf`, 145.3 GiB, 1328 tensors).
`general.architecture` is **`deepseek4`**, which no adapter currently claims —
`default.py` maps only `deepseek2/v2/v3/mtp` and `glm_dsa.py` claims
`glm_moe_dsa`. The model code exists (`vllm/models/deepseek_v4/`, including
`amd/rocm.py`); the loader is the gap.

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
`expert_gating_func` 4, `expert_weights_scale` 1.5, `nextn_predict_layers` 1.

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
