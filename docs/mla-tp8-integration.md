# MLA backend integration map for Kimi K3 at TP8

## 1. Backend interface Kimi K3's 24 MLA layers use

Model path: `KimiK3ForConditionalGeneration` → registry `/home/hotaisle/SlimServe/vllm/model_executor/models/registry.py:302-309` → `vllm/models/kimi_k3/amd/linear.py`.

- Layer construction: `KimiDecoderLayer.__init__` picks `KimiGatedDeltaNetAttention` for KDA layers, else `KimiMLAAttention` — `vllm/models/kimi_k3/amd/linear.py:502-519`.
- `KimiMLAAttention` (`:328`) builds projections and `MultiHeadLatentAttentionWrapper` at `:451` with `self.num_local_heads = num_heads // tp_size` (`:357`) and `assert num_heads % tp_size == 0` (`:361`).
- Wrapper: `vllm/model_executor/layers/mla.py:56-205`. It creates `MLAAttention(num_heads=num_local_heads, ...)` at `:109`. Its `forward` (`:129-205`) does fused qkv_a → q_b → `q.view(-1, heads, qk_head_dim=192)`, splits `kv_lora` into `kv_c (512)` + `k_pe (64, unsqueeze(1))`, **skips RoPE** (`mla_use_nope=True`, `rotary_emb=None`), calls `self.mla_attn(...)` with `output_shape=(T, num_heads*v_head_dim)`, then K3's output gate `attn_out * sigmoid(g_proj(h))` and `o_proj`.
- Backend selection: `MLAAttention.__init__` → `get_attn_backend(head_size=576, dtype, kv_cache_dtype, use_mla=True, use_sparse=False, num_heads=num_local_heads)` at `vllm/model_executor/layers/attention/mla_attention.py:430`. On ROCm with `VLLM_ROCM_USE_AITER=1` the priority list is `[ROCM_AITER_MLA, TRITON_MLA, ROCM_AITER_TRITON_MLA]` (`vllm/platforms/rocm.py:413-419`); without AITER it is `[TRITON_MLA]` (`:420-423`). `ROCM_AITER_TRITON_MLA` does not exist in this tree (no `aiter_triton_mla.py` under `vllm/v1/attention/backends/mla/`), so it ImportErrors out of the list.
- Enum/paths: `vllm/v1/attention/backends/registry.py:53` (`ROCM_AITER_MLA`), `:84` (`TRITON_MLA`). A drop-in kernel adds one enum entry + backend class and inserts it in `_get_backend_priorities`.

Backend class surface a replacement must implement (subclass `MLACommonBackend`, `mla_attention.py:1373-1410`):
`get_name`, `get_impl_cls`, `get_builder_cls`, `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size)`, `get_kv_cache_stride_order`, `get_supported_head_sizes`, `get_supported_kernel_block_sizes`, `is_mla() -> True`, plus `supported_dtypes` / `supported_kv_cache_dtypes` used by `validate_configuration` (`vllm/v1/attention/backend.py:320-360` — note it takes **no** `num_heads`).

Per-step metadata: `MLACommonMetadataBuilder.build` (`mla_attention.py:1966-2090`), driven by `CommonAttentionMetadata` (`vllm/v1/attention/backend.py:412-492`). It splits decode/prefill with `split_decodes_and_prefills(decode_threshold=reorder_batch_threshold)`, builds `MLACommonPrefillMetadata` (block_table slice from `num_decodes:`, `query_start_loc`, chunked-context workspace) and calls `_build_decode(block_table_tensor[:num_decodes], seq_lens[:num_decodes], max_seq_len, query_start_loc_cpu/device[:num_decodes+1], num_decode_tokens, dcp_tot_seq_lens)` (`:2062-2070`). Base decode metadata is just `block_table`, `seq_lens`, `dcp_tot_seq_lens` (`:1448-1452`). Builder class vars a replacement sets: `_cudagraph_support`, `query_len_support`, `reorder_batch_threshold` (AITER: `UNIFORM_BATCH` / `UNIFORM`, `rocm_aiter_mla.py:139-141`; Triton: `UNIFORM_SINGLE_TOKEN_DECODE` / `SINGLE_ONLY`, `triton_mla.py:51-53`).

Two kernel entry points on the impl:

- Prefill (MHA, post-`kv_b_proj` decompression): `MLACommonBaseImpl.forward_mha` (`mla_attention.py:2513`), called at `:795`. AITER overrides it for an FP8 ASM path (`rocm_aiter_mla.py:850-916`), default is `flash_attn_varlen_func` via `_flash_attn_varlen_diff_headdims`. **A decode-only replacement can inherit `forward_mha` unchanged.**
- Decode (MQA over the latent cache): `forward_mqa` (abstract at `mla_attention.py:2678`, called at `:894`).

Dispatch in `MLAAttention.forward_impl` (`:681-970`): decode rows are the first `num_decode_tokens` rows (batch is reordered so decodes lead); `q` is split `[qk_nope 128, qk_rope 64]` at `:817`, nope is `bmm`-absorbed with `W_UK_T` into `[N,B,512]` and transposed to `[B,N,512]` (`:855-869`), concatenated (or left as a tuple) with `q_pe` `[B,N,64]` (`:876-880`), passed to `forward_mqa`, and the returned `[B,N,512]` goes through `_v_up_proj` into `output[:num_mqa_tokens]`.

## 2. Tensor shapes / dtypes / layouts at the boundary (K3, TP8)

| Object | Shape | dtype | notes |
|---|---|---|---|
| `q` (decode) | tuple `([B,12,512], [B,12,64])` or cat `[B,12,576]` | bf16 (`model_config.dtype`); fp8 if `supports_quant_query_input` and fp8 cache | `B = attn_metadata.num_decode_tokens`; ql_nope is post-`W_UK` latent, q_pe is un-roped (NoPE model) |
| `o` (return) | `[B,12,512]` (`kv_lora_rank`) | `decode.attn_out_dtype` = `model_config.dtype` | second return is `lse` `[B,12]` or `None` (needed only if DCP>1) |
| `kv_c_and_k_pe_cache` | `[num_blocks, block_size, 576]` | bf16 (`auto`), or fp8 view | `576 = 512 kv_lora_rank + 64 qk_rope`; `num_kv_heads=1`, so identical on every rank |
| `block_table` | `[num_decodes, max_blocks_per_req]` | int32 | `decode.block_table` |
| `seq_lens` | `[num_decodes]` | int32 | total context incl. current token |
| `slot_mapping` | `[num_actual_tokens]` | int64 | flat index `block_id*block_size + offset`; consumed only by `concat_and_cache_mla`, not by the decode kernel |
| `query_start_loc` | `[num_reqs+1]` | int32 | device + cpu mirrors |

KV spec: `MLAAttention.get_kv_cache_spec` → `MLAAttentionSpec(block_size=cache_config.block_size, num_kv_heads=1, head_size=576, dtype=…)` (`mla_attention.py:1154-1165`, spec at `vllm/v1/kv_cache_interface.py:391`). Stride order `(0,1,2)`.

Block size: the profile requests `block_size: 64`, but K3 is hybrid, so `unify_kv_cache_spec_page_size` (`vllm/v1/core/kv_cache_utils.py:1063-1100`) scales the MLA block up by `ratio = max_page_size // mla_page_size` to match the fixed KDA state page — the observed value is 960 tokens at `kv_cache_dtype=auto` and 1920 at fp8 (documented in `slimserve/profiles.json`, k3-6 notes). The kernel block size is then chosen by `select_common_block_size` (`vllm/v1/worker/utils.py:250-315`) from `get_supported_kernel_block_sizes()`. AITER declares `MultipleOf(1)` (`rocm_aiter_mla.py:70-74`) and flattens pages; Triton MLA declares `MultipleOf(16)` (`triton_mla.py:98`) and indexes pages natively. **A replacement kernel must handle block_size ≈ 960 (or declare a divisor and let the runner split).**

AITER-specific decode metadata (`AiterMLADecodeMetadata`, `rocm_aiter_mla.py:90-106`): `paged_kv_indptr [num_reqs+1]` = cumsum(seq_lens); `paged_kv_indices [sum(seq_lens)]` = per-token flat slot ids produced by `_expand_page_indices_kernel` (`:582-635`); `paged_kv_last_page_len` = all ones; `qo_indptr [num_reqs+1]` = `arange` for pure decode; `max_qo_len`; `attn_out_dtype`; `has_persistent_metadata`. Split/reduce work metadata (`work_meta_data`, `work_indptr`, `work_info_set`, `reduce_indptr`, `reduce_final_map`, `reduce_partial_map`) is sized once by `get_mla_metadata_info_v1` (`:192-243`) and filled per step by `get_mla_metadata_v1` (`:522-543`) **only when `max_qo_len == 1`**.

## 3. What AITER's decode call receives and returns

`vllm/v1/attention/backends/mla/rocm_aiter_mla.py:958-969` calls:

```python
rocm_aiter_ops.mla_decode_fwd(
    mla_padded_q,                                   # [B, max(16,N), 576]
    kv_buffer,                                      # kv_c_and_k_pe_cache.unsqueeze(2): [num_blocks, block_size, 1, 576]
    o,                                              # [B, max(16,N), 512] out param
    self.scale,                                     # 1/sqrt(192)
    attn_metadata.decode.qo_indptr,                 # [num_reqs+1] int32
    attn_metadata.decode.max_qo_len,                # int, 1 for pure decode
    attn_metadata.decode.paged_kv_indptr,           # [num_reqs+1] int32, cumsum(seq_lens)
    attn_metadata.decode.paged_kv_indices,          # [sum(seq_lens)] int32, per-token flat slots
    attn_metadata.decode.paged_kv_last_page_len,    # [num_reqs] int32, all 1
    **mla_kwargs,                                   # q_scale, kv_scale, and the 6 work/reduce tensors
)
```

Wrapper signature — `vllm/_aiter_ops.py:2321-2340`:

```python
@staticmethod
def mla_decode_fwd(q, kv_buffer, o, sm_scale, qo_indptr, max_seqlen_qo,
                   kv_indptr=None, kv_indices=None, kv_last_page_lens=None,
                   logit_cap=0.0, q_scale=None, kv_scale=None,
                   work_meta_data=None, work_indptr=None, work_info_set=None,
                   reduce_indptr=None, reduce_final_map=None, reduce_partial_map=None)
```

Custom-op impl — `vllm/_aiter_ops.py:442-507` — reshapes `kv_buffer.view(-1, 1, 1, q.shape[-1])` (page_size collapses to 1 token/page, which is why indices are pre-expanded) and forwards to `aiter.mla.mla_decode_fwd(q, kv_flat, o, qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, max_seqlen_qo, sm_scale=…, logit_cap=…, [q_scale, kv_scale], [work_*/reduce_*])`. Returns `None`; `o` is written in place. Semantics: causal MQA of `q[b,h,:]` against the 576-wide latent rows selected by `kv_indices[kv_indptr[b]:kv_indptr[b+1]]`, softmax over `sm_scale * q·k`, output = weighted sum of the first 512 dims. `q_scale`/`kv_scale` are per-tensor fp8 dequant scales (`layer._q_scale`, `layer._k_scale`), only forwarded when the installed AITER exposes them (`_check_aiter_mla_fp8_support`, `:414-439`).

Head padding wrappers: `get_mla_padded_q` repeats q to 16 heads when `N < 16` (`:667-674`), `get_mla_unpadded_o` strides the result back (`:677-682`). These only work when `16 % N == 0` — 12 fails.

## 4. Kimi K3 numbers

Derived from tensor shapes in `vllm/transformers_utils/gguf_kimi_k3.py:216-253` (docstring `:15-21`), fed into `KimiLinearConfig` (`vllm/transformers_utils/configs/kimi_linear.py`):

| field | value | source |
|---|---|---|
| `hidden_size` | 7168 | `embedding_length` |
| `num_attention_heads` | 96 | `attention.head_count`, or `q_proj[1] // kda_head_dim` (`:216-219`) |
| `q_lora_rank` | 1536 | `q_a_proj.weight[1]` (`:221`) |
| `kv_lora_rank` | 512 | `kv_a_layernorm.weight[0]` (`:222`) |
| `qk_rope_head_dim` | 64 | `kv_a_proj_with_mqa[1] - kv_lora_rank` (`:225`) |
| `qk_nope_head_dim` | 128 | `q_b_proj[1]//96 - 64` (`:226-228`) |
| `v_head_dim` | 128 | `kv_b_proj[1]//96 - 128` (`:229`) |
| KDA `head_dim` | 128, 96 heads | `o_norm.weight[0]` (`:217`) |
| layers | 93 = 24 MLA + 69 KDA | classified by tensor presence (`:198-212`); confirmed `slimserve/profiles.json:141` |
| experts | 896 routed, top-8, latent `routed_expert_hidden_size` 3584 | `:245-249, :83-86` |
| vocab | 163584 base + 256 special = 163840 | `:113-119` |

Derived: `qk_head_dim = 192`, `head_size (cache) = 576`, `scale = 192**-0.5`.

Per-rank (MLA and KDA both shard the same 96 heads):

| | heads/rank | q decode `[B,N,576]` | o `[B,N,512]` | o_proj in | AITER ok? |
|---|---|---|---|---|---|
| TP2 | 48 | `[B,48,576]` | `[B,48,512]` | 6144 | yes (48 % 16 == 0) |
| TP6 | 16 | `[B,16,576]` | `[B,16,512]` | 2048 | yes |
| TP8 | **12** | `[B,12,576]` | `[B,12,512]` | 1536 | **no** (12 ∤ 16, 16 ∤ 12) |

`kv_lora_rank`, `qk_nope/rope`, `v_head_dim` are per-head and never sharded; the KV cache is replicated across ranks.

## 5. Where the assertion lives, and what else moves at TP8

- Assertion: `AiterMLAHelper.check_num_heads_validity` (`rocm_aiter_mla.py:646-653`), predicate `is_valid_num_heads` (`:654-660`), invoked from `AiterMLAImpl.__init__:714` and `rocm_aiter_mla_sparse.py:749`. Fires at layer construction, after backend selection.
- Selection does **not** filter on heads on ROCm: `get_attn_backend(..., num_heads=…)` (`vllm/v1/attention/selector.py:111,173,188`) → `RocmPlatform.get_attn_backend_cls`/`get_valid_backends` accept `num_heads` (`vllm/platforms/rocm.py:497,545,577`) but never consult it; `validate_configuration` has no `num_heads` parameter (`vllm/v1/attention/backend.py:320-338`). Contrast CUDA, which does use it (`vllm/platforms/cuda.py:107`). So the fix is either a new backend ahead of AITER in `_get_backend_priorities`, or `--attention-backend TRITON_MLA`.
- The builder already anticipates <16 heads: `self._num_attention_heads = max(16, self.num_heads)` (`rocm_aiter_mla.py:191-194`) — metadata sizing survives, the impl assert does not.

Other TP8 surfaces, checked:

- **KDA / GDN** (`vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py`): `assert num_heads % tp_size == 0` at `:249`; TP8 → `local_num_heads=12`, `local_projection_size=1536`, `local_output_size = 4*1536+128+12 = 6284`, `in_proj_padding = -6284 % 16 = 4` (`:266-271`), zeroed at `:287-288`. State shapes divide cleanly (`MambaStateShapeCalculator.kda_state_shape`, `vllm/model_executor/layers/mamba/mamba_utils.py:271-294`: conv_dim `3*12288/8`, recurrent `(12,128,128)`). Triton KDA kernels are head-count-generic. No blocker.
- **MoE**: `KimiMoE.__init__` (`vllm/models/kimi_k3/amd/linear.py:158-…`) — with EP, 896/8 = 112 experts per rank (already validated in the shipped `k3-8` profile, `slimserve/profiles.json:369-397`); GGUF EP sharding is wired in `vllm/model_executor/model_loader/gguf_loader.py:87-125` (`compute_local_expert_ids`, rejects EPLB at `:104`). Under plain TP8 (no EP) watch `min_moe_intermediate_per_partition = 256` at `linear.py:184-193`, which silently pads `moe_intermediate_size` and zero-fills `w13/w2` (`:280-296`). Also relevant: the MMQ expert-id/stride fixes recorded in `HANDOFF.md:198-224` require a locally built `_C_stable_libtorch`.
- **Vocab / embedding**: `VocabParallelEmbedding` uses `padding_size = lcm(64, tp_size)` (`vllm/model_executor/layers/vocab_parallel_embedding.py:251-260`); 163840 is already a multiple of 64 and of 8 → no change at TP8.
- **Other projections at TP8**: q_b_proj out 18432/8 = 2304; kv_b_proj out 24576/8 = 3072; o_proj in 12288/8 = 1536; g_proj out 12288/8 = 1536 — all integral, and all multiples of 256 where GGUF quant blocks matter except q_b (2304 = 9×256, fine).
- **attn_res** (`vllm/models/kimi_k3/amd/ops/attn_res.py:88-120`) operates on full `hidden_size` with replicated weights — TP-independent.

## 6. Existing non-AITER MLA decode reference

`vllm/v1/attention/backends/mla/triton_mla.py` — `TritonMLAImpl.forward_mqa` (`:208-286`):

```python
def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer) -> tuple[Tensor, Tensor | None]
```

It reads `q_num_heads = q.shape[1]` with no constraint, allocates `o [B,q_num_heads,512]` and `lse [B,q_num_heads]`, unsqueezes the cache to `[num_blocks, block_size, 1, 576]`, slices `kv_c_cache = […, :512]`, sets `PAGE_SIZE = cache.size(1)` (native paging — no index expansion), and calls:

```python
decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o, lse,
                     attn_metadata.decode.block_table, attn_metadata.decode.seq_lens,
                     attn_logits, num_kv_splits, self.scale, PAGE_SIZE,
                     k_scale=layer._k_scale, v_scale=layer._k_scale, is_mla=True)
```

(`vllm/v1/attention/ops/triton_decode_attention.py:756`). The grouped stage-1 kernel tiles heads as `BLOCK_H = 16` with `VALID_BLOCK_H = min(BLOCK_H, kv_group_num)` and a `mask_h` predicate (`:317-322`), and its grid is `(batch, cdiv(head_num, min(BLOCK_H, kv_group_num)), num_kv_splits)` (`:513-518`), so **12 heads works unmodified** — one masked tile per request. It also sets HIP-specific `waves_per_eu/matrix_instr_nonkdim/kpack` (`:527`), returns a real `lse` (so it satisfies `can_return_lse_for_decode = True`), and reserves its split workspace at max in the builder (`triton_mla.py:59-80`). This is the correctness oracle to diff a HIP kernel against, and it is already reachable today via `--attention-backend TRITON_MLA` or by unsetting `VLLM_ROCM_USE_AITER`.

Non-references: `rocm_aiter_mla_sparse.py` and `quixicore_mla_sparse.py` are sparse/indexer paths (the latter is an sm80 CUDA kernel, `csrc/quixicore/serving/mla_kernels.cuh`); `csrc/libtorch_stable/attention/mla/` is CUTLASS SM100 only. There is no existing HIP MLA decode kernel in this tree.

## 7. Minimum drop-in checklist

1. New backend class subclassing `MLACommonBackend`; add to `AttentionBackendEnum` and to the ROCm MLA priority list ahead of `ROCM_AITER_MLA`.
2. Builder subclassing `MLACommonMetadataBuilder`; override `_build_decode` only if the kernel needs extra metadata; declare `query_len_support`/`_cudagraph_support` (K3 serves under `FULL_DECODE_ONLY`, so the decode path must be capturable — `UNIFORM_SINGLE_TOKEN_DECODE`/`SINGLE_ONLY` at minimum).
3. Impl subclassing `MLACommonImpl`, implementing only `forward_mqa` (inherit `forward_mha`, `do_kv_cache_update`, `_v_up_proj`).
4. Kernel contract: q `[B,N,576]` bf16 → o `[B,N,512]`; latent cache `[num_blocks, block_size, 576]`; int32 `block_table`/`seq_lens`; `sm_scale = 192**-0.5`; causal over `seq_lens[b]` tokens; return `lse` only if DCP is ever enabled.
5. Declare `get_supported_kernel_block_sizes()` compatible with the hybrid-inflated MLA block (960 at `kv_cache_dtype=auto`).
6. Validate against `TritonMLAImpl` at N=12, B∈{1,…,max_num_seqs}, seq_lens spanning block boundaries.
