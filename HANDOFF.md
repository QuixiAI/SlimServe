> Policy update (2026-09-10): TurboQuant is prohibited in every profile,
> including draft KV. FP8 is the only permitted KV cache quantization.
> `qwen38-nvfp4-1-tq` is removed; `qwen38-nvfp4-1` now enables vision on
> Metal. Earlier TurboQuant directives and measurements below are historical.
> See `perf/optimization_status.md` for current validation evidence.

<!--
Campaign handoffs in this file. Each is current for its own campaign;
none supersedes another. The first two met on this file in the 2026-08-28
merge of origin/main and were joined rather than reconciled.

  0. GLM-5.3-Flash Q2 on Metal (glm53f-q2-1)      -- first section (plan, 2026-09-10)
  1. NVFP4-on-Metal campaign (M1 Ultra / M5 Max)  -- second section
  2. MI300X GGUF profile record                   -- third section
  3. GLM-5.3-Flash on 8x A100 (glm53f-*)          -- fourth section, at EOF
-->

# HANDOFF — GLM-5.3-Flash Q2 on Metal (`glm53f-q2-1`), campaign plan 2026-09-10 and status log (PR #30 open, 2026-09-14)

Read this first when resuming the campaign. Companion documents:
`perf/perf.md` (method), `perf/metal_m1ultra_campaign_v2.md` and
`perf/metal_m1ultra_retrospective.md` (the DSV4 Metal campaign and its
ceiling analysis), `perf/prefill_handoff.md` (prefill campaign + boot
protocol), `perf/qwen38_metal_design.md` (the GDN-hybrid bring-up: closest
analog to the KDA layers), the `glm53f-*` A100 sections below and
`vllm/model_executor/models/glm5_next.py` (the model as it serves on A100
today). Every number here is quoted from a cited source or is derived
arithmetic marked as such.

## Mission

Serve antirez's own Q2 GGUF of GLM-5.3-Flash through a SlimServe profile
(`glm53f-q2-1`: this exact quant, this exact hardware, Apple Metal on the
Mac Studio M1 Ultra 128 GiB) and beat his ds4 engine on the same box, the
way DSV4 0731 was beaten (SlimServe 34.9 tok/s decode / 371 tok/s prefill
vs ds4 25.5 / 277, `perf/baseline_status.md` UPDATE 30 and 26).

The campaign runs the way the DSV4 and Qwen3.8 Metal campaigns ran:

1. **Discovery + bring-up + decode** in one session: register the profile,
   clock the base recipe (ds4, best config), bring the profile to
   correctness, pin baselines, run byte-ordered decode kernel waves, then
   speculation. Ends with a compaction handoff written into this section.
2. **Prefill** in a fresh session from that handoff.
3. **Cleanup + gates + PR.** The PR description opens with the base-recipe
   clock on this box (ds4 decode / prefill at its best config) against the
   `glm53f-q2-1` numbers, then the ledger of what was kept and rejected.
4. **Port** the kernels to QuixiCore-Metal (`csrc/quixicore/metal/README.md`
   sync contract). Kernels are developed here first because that library
   connects to nothing; SlimServe is where they are wired.

Speculation policy: **whichever speculator is fastest on this hardware
ships.** The candidates are the native MTP head (in the GGUF), the
published DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2`), and any other
drafter that appears for this model. Measure each under the same gate and
pin the winner. TurboQuant is prohibited (policy at the top of this file);
FP8 is the only permitted KV quantization; f16 latent pages are fine.

Every optimization ships with a `VLLM_QC_*` kill switch, an oracle test, a
liveness breadcrumb, and a notebook entry in `perf/optimization_status.md`;
pins go to `perf/baseline_status.md`; raw artifacts under
`perf/results/YYYY-MM-DD/<run>/`; never cite `/tmp`. Compaction handoffs
are appended to this section at every milestone.

## Repo state

- Branch `glm53f-metal-campaign`, cut from `main` at 55fdb54bd
  (2026-09-11), worktree `~/Code/slimserve/SlimServe-glm53f`. The main
  checkout stays on its own branch because a live server runs from it.
- `main` already serves this model on 8x A100 (`glm53f-nvfp4-8`, c1 153.6
  tok/s with DFlash2 k=3; see the A100 section below). The Metal campaign
  is a port of a serving architecture onto Metal plus antirez's GGUF
  layout, not a bring-up from nothing.
- Nothing in the `glm5_next` python path routes to Metal yet (§"Missing").
- PR #24 (`glm53-flash-sm120`) is NOT an ancestor of main and diverges in
  the four core model files; only its plan structure and gotcha list are
  used here.
- ds4 reference: upstream `6289c516` (2026-09-08) cloned to
  `~/.local/scratch/ds4-upstream`. The `~/ds4` and `~/Code/ds4` checkouts
  are single-commit snapshots from 2026-08-09 that predate GLM 5.3 support;
  do not use them for this model.
- Model files land in `~/models/antirez-glm-5.3-flash-gguf/`.

## Target

### The artifact

| item | value |
| --- | --- |
| file | `antirez/glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf` |
| bytes | 96,505,816,384 (89.88 GiB) |
| sha256 | `e81fd6241c6e55a64e1e14e47a3eab61a173fa8d7e4b5c1d1848827119705b32` |
| GGUF | v3, 1,412 tensors, 50 KV pairs, `general.architecture = glm5-next`, source revision `84c6a6aa` of `zai-org/GLM-5.3-Flash` |
| layout | single file, ds4's own tensor names (`kda_q`, `hc_attn_fn`, `indexer.pool_gate`, `nextn.eh_proj`); not loadable by llama.cpp |
| sidecar | `GLM-5.3-Flash-Vision-Encoder.gguf`, 1.05 GiB (vision deferred) |

Quant map (tallied from the header):

| tensor group | quant | GiB |
| --- | --- | ---: |
| routed `ffn_gate_exps` + `ffn_up_exps`, 43 layers | IQ2_XXS (imatrix) | 49.89 |
| routed `ffn_down_exps` | Q2_K | 31.75 |
| KDA `kda_q`, `kda_k` | Q4_K | 1.20 |
| KDA `kda_v`, `kda_output`, `f_a/f_b/g_a/g_b/beta` | Q8_0 | ~2.5 |
| sparse MLA `attn_q_a/q_b/kv_a_mqa/k_b/v_b/output` | Q8_0 | ~1.5 |
| shared experts, dense FFN (layers 0-2), `token_embd`, `output` | Q8_0 | ~2.5 |
| indexer `attn_q_b/attn_k/proj/pool_gate/pool_ape`, `hc_*_fn`, `nextn.eh_proj` | BF16 | 0.30 |
| router `ffn_gate_inp`, norms, convs, `dt_bias`, `a_log` | F32 | 0.20 |

The expert recipe is byte-for-byte the DSV4 0731 recipe (IQ2_XXS gate/up,
Q2_K down), so the `ggml_moe_a8_vec*` / `ggml_moe_mm_id` Metal suite
applies unchanged except for top-k.

### The model (config.json, cross-checked against the GGUF KV)

- 45 trunk layers + 1 MTP block (layer 45). Hidden 4096, vocab 154,880,
  context 1,048,576, no RoPE anywhere in the trunk.
- Strict 3:1 pattern. Sparse-MLA layers at 3, 7, 11, 15, 19, 23, 27, 31,
  35, 39, 43 (11); every other layer is KDA (34). The MTP block is a
  sparse-MLA + MoE layer.
- KDA: 64 heads x 128, short conv 4, low-rank forget gate (`f_a` 4096->128,
  `f_b` 128->8192, `dt_bias`, `a_log[64]`), `gate_lower_bound` -5.0, one
  beta per head, low-rank output gate (`g_a/g_b`), per-head gated RMSNorm,
  out 8192->4096. Decay is PER KEY CHANNEL:
  `alpha = exp(-5 * sigmoid(exp(a_log_h) * (f + dt_bias)))`; recurrence
  `S = (I - beta k k^T) Diag(alpha) S + beta k v^T`, `o = S^T q`, q/k
  L2-normalized in kernel. This is not Qwen's GDN (scalar decay per head).
- Sparse MLA (NoPE): 64 heads, `qk_nope` 256, `qk_rope` 0, `v_head` 256,
  `q_lora` 1536, `kv_lora` 512, absorbed `k_b [256,512,64]` /
  `v_b [512,256,64]`. Latent row = 512 per token per layer.
- Indexer: 32 heads x 128, `index_topk` 2048, `index_kpool` 4 (512 pools
  selected, expanded to 2048 positions, plus the incomplete tail pool always
  selected: `selected_limit` 2051). Pool = softmax(gate(t)+ape[m])-weighted
  sum of LayerNorm'd k over 4 tokens; score = sum_h w_h relu(q_h . pk).
  Not DSV4's compressor and not DSV4's fp8 per-token indexer.
- mHC: `hc_mult` 4, 20 Sinkhorn iterations, `hc_eps` 1e-6; shapes identical
  to DSV4. Head is an unweighted mean over the 4 streams; post multiplier
  2.0.
- MoE: 288 routed, top-8, 1 shared (2048), `moe_intermediate` 2048, layers
  0-2 dense (12288), sigmoid scoring + `noaux_tc` bias top-8, renormalized,
  `routed_scaling_factor` 2.5, `swiglu_limit` 10.0 (clamped SwiGLU).
- MTP: one nextn block (`eh_proj` 8192->4096, enorm/hnorm, full DSA+MoE
  layer). Shares embeddings and head.
- Tokenizer: `glm4` pre-tokenizer, EOS set {154820, 154827, 154829}, bos
  `[gMASK]` 154822, template opens `[gMASK]<sop>`, reasoning-effort system
  line, `<think>` blocks; vLLM parsers `glm45` (reasoning) / `glm47` (tools).
- The "169B" in the `glm53f-*` profile summaries is a stale label. Expert
  params alone are 25.2M x 288 x 42 = ~305B; the public 320B figure is
  right (PR #24 plan `docs/glm53-flash-sm120-plan.md:26-36`).

### Memory budget on the box (derived)

Metal working set on this M1 Ultra is ~115 GiB with
`iogpu.wired_limit_mb=122880`. DSV4 served at ~103 GiB total demand; the
thrash edge is ~115.

| component | size | note |
| --- | ---: | --- |
| weights, pinned | 89.9 GiB | vs DSV4's 80.8 + 6.5 drafter = 87.3; 2.6 GiB tighter |
| `kv_b_proj` dequant (adapter emits bf16) | ~0.4 GiB | 11 x 64 x 512 x 512 x 2 B (glm_dsa precedent) |
| KDA recurrent state | 136 MiB / seq | 34 x 64 x 128 x 128 x 4 B fp32; x2 if MTP keeps a rollback copy |
| KDA conv state | 9.6 MiB / seq | 34 x 3 x 8192 x 3 x 4 B |
| MLA latent cache | 11 KiB / token | 11 layers x 512 x 2 B (f16); +1 KiB/token for the MTP layer |
| indexer pool cache | 0.7 KiB / token | 11 x 128 x 2 B / 4 |
| KV pool at 262,144 tokens, 1 seq | ~3.1 GiB | vs DSV4's 9 GiB for the same length |
| DFlash2 drafter, if it wins | ~2 GiB | 1B bf16 |

An 8 GiB fixed KV pool (two full 262k requests, or 32 x 16k) lands total
demand at ~100-101 GiB, under the DSV4 figure. `max_num_seqs` stays modest:
each sequence carries 146 MiB of fp32 recurrent state on top of KV.

### Per-token bytes and the ceiling (derived from the census)

| traffic, autoregressive batch 1 | GB/token | source |
| --- | ---: | --- |
| routed experts, 8 x 7.07 MB x 42 layers | 2.4 | same expert bytes as DSV4 (6.75 MiB each) |
| shared expert Q8, 42 layers | 1.1 | |
| KDA projections (Q4_K q/k, Q8 v/out, gates), 34 layers | 3.7 | ~105 MB/layer |
| sparse MLA projections Q8, 11 layers | 1.5 | |
| head/embed/dense-FFN/router | 1.0 | |
| KDA state read+write | 0.3 | 136 MiB x 2 |
| **total** | **~10.0** | DSV4 was ~9 GB AR, ~22.5 GB per k=5 spec step |

Effective-bandwidth ladder from the DSV4 retrospective, applied to 10 GB:

| regime | GB/s | tok/s |
| --- | ---: | ---: |
| DSV4 in-situ at campaign end | 172-183 | 17-18 |
| ds4's measured kernel efficiency on DSV4 | 283 | 28 |
| llama.cpp-class sustained | ~400 | 40 |
| best single kernel measured on this box (q8_0 GEMV) | 738 | 74 |

Two structural differences cut in our favor. 38% of the bytes are dense
M=1 GEMVs over Q4_K/Q8_0 (the KDA projections), and dense GEMV on this box
already runs at 438-738 GB/s, whereas IQ2_XXS expert GEMV tops out at
255-340. And decode cost is context-flat: 34 of 45 layers hold no KV, and
the 11 that do attend to at most 2,051 positions. DSV4's speculation
multiplier came from reading dense weights once per verify batch; the same
lever applies, with the caveat that a recurrent mixer runs k+1 sequential
steps per verify (Qwen finding, `perf/optimization_status.md` 2026-08-20),
so k stays small.

### The bar

- **ds4 is the bar and must be measured here.** No resident-Q2
  GLM-5.3-Flash tok/s is published anywhere (blog, README, PERFORMANCE.md,
  speed-bench CSVs, 717 commit bodies checked). Closest: DeepSeek Flash Q2
  on an M2 Ultra at 410 prefill / 23.2 decode; GLM-5.3-Flash Q4_K streamed
  from SSD on an M5 Max at 11.9-14.9 decode. Measure `6289c516` on this box
  with clean memory, `--mtp` off and on (`--mtp-timing` prints acceptance),
  `--power 100`; the bar is its best config.
- llama.cpp: `glm5next` is still a draft PR (#27752); the Unsloth GGUFs
  (UD-IQ2_XXS 101.84 GB) are a different, larger file. Optional quality
  cross-check only. MLX: only 512 GB Ultra numbers exist.
- Our CUDA record for scale: `glm53f-nvfp4-8` c1 153.6 tok/s (DFlash2 k=3),
  non-spec 109.9.

## What exists (reuse) and what is missing (the campaign)

### In-tree on `main`

- **Model graph.** `glm5_next.py` composes GLM-5.3-Flash from layers this
  fork already serves: `KimiGatedDeltaNetAttention` (Kimi-K3 KDA), the NoPE
  `MultiHeadLatentAttentionWrapper` path, `DeepseekV2MoE` routing, the fused
  `dsv4_mhc_*` ops, `Glm5NextPooledIndexer`. Loader, stacked params mapping,
  EAGLE-3 aux taps for DFlash2, MTP block class, vision wrapper.
- **mHC on Metal.** `dsv4_mhc.metal` exposes `fused_post_pre`, `pre`,
  `post`, `hc_head` with `rms_eps / pre_eps / sink_eps / post_mult /
  sinkhorn_repeat` as runtime buffers; `layers/mhc.py` already probes them
  on Metal. GLM passes `post_mult=2.0`, no fused norm weight, mean head.
  Binding and parity, not a kernel.
- **MoE.** `ggml_moe_a8_vec{,_swiglu,_sum}` multi-row IQ2_XXS/Q2_K GEMVs,
  `ggml_moe_mm_id` tile GEMM for prefill, `moe_finalize`. `map0<NE20>` is
  instantiated for {2,4,6,8}; host asserts were raised to E=256 for DSV4
  and must go to 288. `qc_swiglu` carries a clamp limit; confirm 10.0.
- **Linear-attention infrastructure.** `gdn.metal` (`gdn_recur`,
  `gdn_recur_spec` with `slot_table[r, t]` checkpointing, `gdn_short_conv`,
  `gdn_fused_prepare`, `gdn_gated_rmsnorm_f32`) and
  `serving_glue/gdn_step.metal` (fused conv + scan + gated norm,
  per-position loop inside, MAX_S 16). Hybrid pool contracts: blocks-first
  `kv_cache_stride_order`, `state_stride` on every recur/conv kernel,
  `mamba_cache_dtype=float32`, `VLLM_SSM_CONV_STATE_LAYOUT=DS`, prefix
  caching off, V2 runner, 64-bit `kv_cache_gather_range`, steady-cache
  metadata. All model-agnostic. The scan body is scalar-decay and is not.
- **Sparse MLA / indexer bodies.** `mla_decode_fp8_sparse` (+ split-K
  partition, LSE reduce), `mla_prefill_fa_mma`, `paged_attn_v2`, the
  DSV4-shaped `serving/indexer/*`, and the sync-free torch producer
  `deepseek_v4/metal_indexer.py` as a template.
- **Speculation plumbing.** DFlash V1/V2 speculator, seeded MPS rejection
  sampler, `num_speculative_tokens_per_batch_size`, `qwen3_5_mtp` precedent
  for a native MTP head, `gdn_recur_spec` state rollback.
- **Everything else in the step.** `residency_pin`, `add_rms_norm`,
  `rms_norm_add_dyn`, lm_head Q8 argcat partials, sampler suite,
  marshalling memos, `qc_metal_serving.mm` encoder/PSO/ring conventions,
  phase/sync/op-census profilers, muse single-CB glue.
- **GGUF loading.** `gguf_adapters/glm_dsa.py` (GLM-5.2: absorbed
  `attn_k_b`/`attn_v_b` rebuilt into a dequantized bf16 `kv_b_proj`,
  indexer halves dequantized, `nextn.*` mapped and dropped unless a
  speculator asks) and `gguf_kimi_k3.py` (config derived from tensor
  shapes, layer types from `A_log` presence).

### Missing on Metal

1. **GGUF adapter for `glm5-next`** with ds4's tensor names: KDA
   `kda_{q,k,v,beta,f_a,f_b,g_a,g_b}` -> `in_proj_qkvgfab` shards,
   `kda_*_conv` -> fused `conv1d`, `a_log`/`dt_bias`/`o_norm`; MLA absorbed
   `k_b`/`v_b` -> dequant `kv_b_proj` (~0.4 GiB bf16); indexer BF16 tensors
   -> `wq_b`, `wk`, `weights_proj`, `pool_gate`, `pool_ape`, `k_norm` w+b;
   `hc_*_{fn,base,scale}`; nextn block; config from the 50 KV pairs;
   tokenizer + chat template from the header.
2. **KDA core with per-channel decay**: decode step (conv-4, SiLU, gates,
   L2 norm, recurrence with `Diag(alpha)`, gated RMSNorm), spec-verify
   variant with slot-table checkpoints, prefill scan. No torch MPS fallback
   exists (`gdn_mps_fallback.py` is Qwen-only, scalar gate); the layer
   asserts `backend == "triton"`. The oracle is a new native port of
   `fused_recurrent_kda_fwd`.
3. **NoPE 512-latent sparse MLA** decode and prefill, 64 heads at TP1,
   top-2048(+tail) selection lists, f16 latent pages. `metal_mla_sparse.py`
   is the stub naming the four missing ops and is why `glm52-xxs-1` is
   `in-progress`. The NoPE bf16 variant is the simpler of the two shapes.
4. **Pooled indexer** on Metal: pool update (LN k, gate softmax + APE, f16
   pool cache, f32 tail side-buffer), scoring, top-512 pools, expand +
   always-select-tail. Four Triton kernels on CUDA; nothing on Metal.
5. **Router** E=288, sigmoid + bias, top-8, renorm, scale 2.5 (DSV4's
   `dsv4_router_topk` is sqrtsoftplus top-6). Must be total over NaN.
6. **SwiGLU clamp** 10.0 in the fused expert and dense paths.
7. **MTP state rollback** for a recurrent model (slot-table checkpoints as
   `gdn_recur_spec` does; ds4 snapshots/restores 146 MiB per cycle).
8. **Platform routing** in the python path: `torch.cuda.current_device()`
   in three files, `torch.cuda.Stream()` for the mHC projection stream,
   `is_cuda()` / sm80 gates in the indexer and mHC ops, KDA backend assert.
9. **Vision** (24-layer ViT, patch 14, temporal 2, merge 2): deferred;
   `language_model_only` on Metal until a text record exists.

## Research digest: copy this, avoid that

### ds4's GLM-5.3-Flash implementation (6289c516) — copy the shape, beat the kernels

- KDA decode is one fused kernel per (row, head): grid `(n_rows, 64)`, 128
  threads = 4 simdgroups, each simdgroup owns value rows, each lane 4 key
  columns; conv shift in place, SiLU, decay, beta, L2 norm (q additionally
  x 128^-0.5), the recurrence via `simd_sum(dot(h,k4))`, then gated
  RMSNorm; fp32 state `[head][value 128][key 128]`. Right fusion boundary,
  matches our `gdn_step.metal`. Upstream of it he issues 8 separate matmul
  dispatches per layer (q, k, v, f_a, f_b, beta, g_a, g_b); we fold those
  into one merged `in_proj` GEMV plus an in-kernel `f_b/g_b` bmm. His fused
  BF16 QKV path is M3-Ultra-only and does not apply to the Q2 file.
- **KDA prefill in ds4 is serial per token**: `prepare` runs 64
  threadgroups (one per head) looping over tokens, then a recurrence over
  64 x 32 (head, value-group), then per-token output; fixed 2048-token
  chunk. The prefill differentiator: a chunked WY/UT-transform scan with
  elementwise decays (chunk 64) on simdgroup MMA, or at minimum a
  wide-parallel recurrent kernel overlapped with the MoE tile GEMM.
- Sparse MLA is absorbed-only (`full_kv_cache=false`), f16 latent cache, q
  projected through Q8 `k_b`, `v_b` applied after. Below 2,051 visible
  tokens every token is selected and he runs a dense path. Do the same for
  bring-up: dense NoPE MLA over the whole latent cache is correct for short
  contexts and needs no indexer.
- Indexer: `kernel_glm53_indexer_pool_update` (one 128-d row per 4 tokens
  into an f16 pool cache, tail in an f32 side buffer) and
  `kernel_glm53_expand_pool_selection` (512 pool ids -> 2048 + <=3 tail).
  Copy the cache layout; score with a tiled kernel.
- MTP: draft block in the main GGUF; "commits up to two tokens per cycle";
  snapshots every KDA layer's conv + recurrent state before a speculative
  cycle and restores on rejection; opportunistic acceptance by default,
  `--mtp-exact-sampling` for the true distribution; batching disables MTP.
  Acceptance not published. Slot-table checkpoints avoid the copy.
- GLM requires `--power 100` in ds4; expect the same power sensitivity (the
  Qwen campaign recorded 1.28 tok/s on a 20 W charger).
- His QA fixture for the Q2 file: NLL 0.458 avg, 89/100 first-token match,
  7.37 greedy-prefix vs the FP8 reference on 100 Z.AI cases. Reproduce the
  metric against `glm53f-nvfp4-8` to certify the adapter.

### DSV4 Metal campaign (retrospective + notebook), ranked by step time recovered

1. Residency: windowed madvise during load, resident sweep,
   `MTLResidencySet` pin, `iogpu.wired_limit_mb`. Watch `vm_stat` on every
   boot; the 89.9 GiB file will pin at ~94 GiB of Metal buffers.
2. Transient-allocation churn adjacent to the hot CB chain (the 2.07x
   cos_sin memo). Rank by `aten::empty` census, not kernel count. GLM has
   no rope, so that bug cannot recur, but the class will: mHC stream
   splits, indexer staging, KDA gate reshapes.
3. Boundary/width metadata bugs found by bracket counting (c128 fix,
   -50 ms). GLM's analog: the 4-token pool boundary and the 2051 tail rule.
4. Fused single-dispatch kernels for eager chains: mHC (already bound),
   router, norms, SwiGLU, pool update.
5. Weight-stationary small-M GEMV (`qgemv_mb`) and multi-row MoE GEMV for
   the verify band.
6. Drain kill: every per-step blocking H2D; drains are conserved, the win
   arrives only when the last one dies.
7. Split-K sparse MLA at low simdgroup counts (64 heads x B=1).
8. Prefill: tiled MoE GEMM, dense-causal MMA FA, single-chunk scheduling
   (`max_num_batched_tokens` = 2048 + seqs x spec slots), transposed wide
   Q8 GEMM.

Do not redo (measured, rejected): launch-count reduction as a thesis; the
native step tape; IQ2_XXS SoA/pair repack (2-4% slower on Apple LSU);
cross-threadgroup weight dedup for verify (2x regression); second-queue
weight prefetch; BM=128 staged tiles; tile-level softmax; Metal-4 tensor
ops on M1; threadgroup staging of cache-resident activations; trusting
phaseprof absolutes or isolated-stream microbenches; spec block k>5 for
DSpark-class drafters.

### Qwen3.8 GDN-hybrid campaign (the KDA analog)

- Wire the linear-attention state as fp32 page-packed pools with explicit
  `state_stride`; force blocks-first stride order on the attention side
  before the first boot (the page-layout collision NaN'd 5 of 6 boots).
- Put the per-position verify loop inside one kernel with slot-table
  checkpointing from day one (`gdn_recur_spec` contract: load from
  `slot_table[r, num_accepted-1]`, store every position).
- At c1 the GDN layer is launch-bound, not bandwidth-bound: fusion buys
  2-3%, the merged projection GEMV is the cost. For KDA the projections
  are 105 MB/layer, so the GEMV is the layer.
- Data-dependent-shape ops drain the whole MPS queue (boolean masks,
  `repeat_interleave` with tensor repeats, `.to("cpu")` on lengths): hunt
  with cProfile, not bracketed profilers.
- Spec on a recurrent mixer: mean acceptance ~3 on prose regardless of k,
  verify is k+1 sequential steps per layer; batch-adaptive schedule;
  drafter math bit-exact or measured neutral.
- MPS `index_select` on non-contiguous sources is ~20x slow and wraps at
  2^31 elements; use `kv_cache_gather_range`.
- Every metallib / `.so` rebuild re-gates the DSV4 and Qwen anchors
  bit-exact (26 consecutive so far). Rebuild `_quixicore_C` after any merge
  touching `csrc/quixicore`; rm-then-cp then `codesign -f -s -`.

### CUDA `glm53f` campaigns (PR #24 and `main`)

- Downcast small F32 tensors are a real quality bug (router bias, `A_log`,
  `dt_bias`, `hc_*`): the RedHatAI checkpoint lost ~0.03 nats. antirez's
  GGUF keeps them F32 (census above); verify at load anyway.
- Index-emitting kernels must be total over NaN/inf (vLLM warm-ups push
  NaN activations; the CUDA router once selected "expert 288").
- Indexer bugs were found by parity tests, not benchmarks: tail-store race
  dropping the query's own token; KV-indexed vs query-row request map
  after a cached prefix. Port `tests/kernels/test_glm5_next_indexer_prefill.py`
  and `tests/glm5_next/test_pooled_indexer_parity.py` to the Metal path
  before serving.
- No token-exact greedy gate exists on CUDA (routing flips under
  nondeterministic reductions). Metal is deterministic, so sha anchors are
  available here and are the primary gates.
- MTP head port worked (84% greedy acceptance at k=1) but regressed
  throughput on CUDA (eager draft ~3 ms/step, expert-byte tax per
  speculative row); DFlash2 k=3 won there (+40% c1). Metal's cost model
  differs (dense bytes dominate; launch cost per draft step is the tax).
  Measure both here.
- `glm53f-*` naming is the operator's decision of 2026-09-06; plain
  `glm53` means the 743B dense-MLA GLM-5.3.

## Method and gates (the handbook, unchanged)

- **Harness.** `benchmarks/benchmark_dsv4_exact.py` against the frozen
  `perf/results/harness_assets/m2_source.txt`; legs 1000-in/8-out (fast
  sha), 1000/2000 at offsets 1,2,3,5,7,11,13 (step ms = (wall - prefill) /
  draft steps), 2500-in/64-out (long-ctx anchor), c4/c8 as TPS gates.
  Greedy sha anchors, plus a seeded temp 1.0 / top-p 0.95 / top-k 20 /
  seed 42 arm to match the A100 `glm53f` protocol and a GSM8K acceptance
  arm once speculation is on.
- **Gate classes.** Bit-exact = identical shas and spec counters plus a
  liveness proof (breadcrumb / PSO name); identical shas after a ULP-class
  change mean "not engaged". ULP = determinism x2, coherent text, paired
  step ms, means over >= 5 offsets, and a `VLLM_QC_*=0` reversion sentinel
  reproducing the prior sha. Trajectory-lottery rule: never decide on a
  single-offset tok/s.
- **Boot protocol.** `sysctl iogpu.wired_limit_mb=122880` (resets on
  restart), boot, wait for startup, primer (8 tok), one ~1000-token
  single-chunk request with decode, only then multi-chunk prefill. Kill
  both EngineCore and api_server on `kIOGPUCommandBufferCallbackError*`.
- **Correctness.** Per-layer cosine vs a reference (>= 0.9997 was the Qwen
  bar); planted-string needle at 3.3-4.5k, then 16k/32k/128k/262k;
  antirez's NLL/first-token fixture reproduced against `glm53f-nvfp4-8`;
  acceptance tracked in every gate once speculation is on.
- **Roofline discipline.** Prove the wall before fusing (the DSV4
  diagnosis was wrong four times). Order: `vm_stat` -> op census by
  `aten::empty` -> xctrace first-duration column -> bracket counts.

## Roadmap

### Session 1 — discovery, bring-up, decode

**Discovery.**

1. Download the Q2 file and the vision sidecar; verify bytes and sha256.
   Register `sources.glm53f-q2` in `slimserve/profiles.json` (repo
   `antirez/glm-5.3-flash-gguf`, quant tag `Q2`, `min_memory_bytes.metal`
   137438953472) and profile `glm53f-q2-1` with a `metal` variant,
   `status: in-progress`.
2. Dump the header: full tensor census, all 50 KV pairs, chat template
   presence, tokenizer arrays. Confirm the artifact table above.
3. Clock the base recipe: ds4 `6289c516` on this box with clean memory,
   1000/2000 greedy c1 decode and 2048 prefill, `--mtp` off and on with
   `--mtp-timing`, `--power 100`. Record in `perf/baseline_status.md` as
   the GLM-5.3-Flash Metal bar. This number opens the PR description.
   Recipe (binaries in `~/.local/scratch/ds4-upstream`, built with `make`):
   (a) exact-token legs through the server, as the DSV4 bar was measured:
   `./ds4-server --metal -m ~/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q2.gguf --ctx 3008 --tokens 2000 --host 127.0.0.1 --port 18080 --power 100`
   then `benchmarks/benchmark_dsv4_exact.py --url http://127.0.0.1:18080/v1/completions --served-model-name <name from /v1/models> --input-tokens 1000 --output-tokens 2000 --prompt-offset 1` (greedy; `--metrics-url none`); this is the no-MTP arm. `--mtp` is a `ds4` CLI flag only
   (`ds4_cli.c:2024`; neither `ds4-server` nor `ds4-bench` parse it, and
   session-batched serving disables MTP), so the MTP arm runs through
   the CLI on the same 1000-token prompt text:
   `./ds4 --metal -m <gguf> --ctx 3008 --prompt-file <prompt.txt> -n 2000 --temp 0 --mtp --mtp-timing --power 100`
   and the no-MTP CLI twin without `--mtp` for an apples-to-apples pair;
   take decode tok/s from the CLI's timing output and acceptance from
   `--mtp-timing`. (b) frontier sweep:
   `./ds4-bench --metal -m <gguf> --prompt-file speed-bench/promessi_sposi.txt --ctx-start 2048 --ctx-max 2048 --gen-tokens 128 --csv perf/results/<date>/glm53f-q2-bar/ds4_bench_2048.csv`
   for ds4's own prefill/decode definition at the 2048 frontier. Memory
   must be clean: no other server resident (the box holds one 90 GiB
   model at a time).
4. Memory census on our side: expected pinned bytes, KV pool sizing, KDA
   state per seq; pick `max_num_seqs` and the fixed `kv_cache_memory_bytes`.
5. Blocker list with file:line for every CUDA-only gate in the python path
   and every missing op, plus the KDA parameterization diff against
   `gdn_step.metal` written out as equations.
6. Reference-dump recipe: per-layer hidden states and KDA state for a fixed
   64-token prompt from the A100 `glm53f-nvfp4-8` box (approximate), and a
   CPU transformers run of the FP8 checkpoint's first 4 layers (exact).

**Bring-up to correctness** (slow is fine; both prior campaigns started at
0.1-2 tok/s).

1. Adapter `gguf_adapters/glm5_next.py` for `glm5-next`. Config from the
   header (`gguf_kimi_k3.py` pattern). MTP block mapped and dropped unless
   a speculator asks. Vision tensors ignored.
2. Platform routing: device-neutral buffer allocation; mHC through the
   Metal `dsv4_mhc` bindings with `post_mult=2.0` and a mean head; KDA
   backend selector with a `metal` branch; indexer and MLA backends select
   Metal implementations; `language_model_only` on Metal.
3. KDA torch-native MPS oracle (`kda_mps_fallback.py`): decode step,
   spec-verify with per-position state, recurrent prefill. Exact port of
   `fused_recurrent_kda_fwd` math; the correctness reference for every
   later kernel.
4. Hybrid pool for 34 KDA groups + 11 MLA groups + 11 indexer groups:
   blocks-first stride order, fp32 state, `state_stride`, prefix caching
   off, V2 runner. Reuse `mamba_hybrid.py` and `attn_utils` as Qwen does.
5. Dense NoPE MLA fallback for contexts under 2,051 tokens (torch gather
   over the f16 latent pages + SDPA), no indexer. First tokens.
6. Indexer torch producer (`metal_indexer.py` pattern, fixed-shape tiles):
   pool update, scoring, top-512, expand + tail. Longer contexts.
7. Router / SwiGLU clamp / MoE: torch router first (sigmoid + bias, top-8,
   renorm, x2.5), `qc_swiglu` with limit 10, `ggml_moe_a8_vec*` with
   NE20=8 and E=288 asserts, `ggml_moe_mm_id` for prefill.
8. Gates: greedy coherent text; per-layer cosine vs the reference dumps;
   needle at 3.3-4.5k; antirez's NLL fixture within noise of the A100
   record; profile boots through `slimserve glm53f-q2-1 --serve`.

**Baseline pinning.** Residency pin on (`Pinned N allocations`), wired
limit set, `vm_stat` clean. Pin 8-tok sha, off1-2000 sha/counters/step ms,
2500x64 sha; c4/c8 TPS; the seeded arm. Re-measure ds4 the same day.
Restate per-token bytes from the load census and the ceiling ladder.

**Decode kernel waves, byte-ordered.**

- W1 KDA. Merged `in_proj_qkvgfab` GEMV over mixed Q4_K/Q8_0 shards in one
  dispatch (or one per quant type), `_mb` twins for M<=8; fused
  `kda_step.metal` (conv shift + SiLU + f_b/g_b bmm + decay + beta + L2
  norm + recurrence + gated norm) with `slot_table` checkpoints and
  `resume_slot`, per-channel `Diag(alpha)`, fp32 state. Oracle: the
  fallback, float64, determinism x2, batched rows bit-identical to looped
  batch-1. Expected largest single win (34 layers, 3.7 GB/token).
- W2 MoE + router. `glm_router_topk` E=288 sigmoid+bias top-8 (bitwise via
  a split boundary if MPS transcendentals disagree, as `dsv4_router` does);
  shared-expert + sum fold into `moe_finalize`; NE20=8 tile set and the
  8-slot multi-row path; SwiGLU clamp in the fused kernel.
- W3 Sparse MLA NoPE + indexer. `mla_decode_f16_sparse_nope`: absorbed q
  through Q8 `k_b`, latent attention over the 2,051-row selection list,
  `v_b` after; split-K partition at 64 heads x B=1 with LSE reduce; f16
  latent pages. Indexer kernels: pool update, tiled scoring (32 heads x 128
  against pooled keys), top-512, expand+tail. Parity tests ported from the
  CUDA tree.
- W4 Glue. Bind what the eager chain still does in torch: norms
  (`rms_norm_add_dyn`), lm_head Q8 argcat partials at vocab 154,880,
  embedding lookup, sampler. Marshalling-memo audit by `aten::empty`
  census; data-dependent-shape hunt with cProfile; drain kill.
- W5 Sync/async. `VLLM_METAL_ASYNC_SCHED`, steady-cache metadata for the
  hybrid builder, bound-mode attention lens. Park by the "<2% relocation"
  rule.

Exit: paired step ms at or under ds4's autoregressive step at the same
draw; op census closes to measurement; every wave gated per the handbook.

**Speculation — whichever is fastest ships.**

- Native MTP head (in the GGUF; `qwen3_5_mtp` / `kimi_k3` MTP precedent;
  the A100 proposer `spec_decode/glm5_next_mtp.py` is platform neutral).
  k=1-2. The draft is one DSA+MoE layer: ~60 MB of expert bytes plus Q8
  projections per draft token against a 10 GB target step. KDA rollback
  through the W1 slot table, no snapshot copies.
- DFlash2 `incoai/GLM-5.3-Flash-DFlash2` (1B bf16, ~2 GiB; the A100
  record's choice, k=3; aux taps already in `glm5_next`; the Metal DFlash
  V2 speculator exists from Qwen).
- Any other published drafter for this model, same gate.
- Verify-band kernels: `_mb` twins for the KDA projections and MLA at
  M=k+1, `ggml_moe_a8_vec` multi-row at 8 x (k+1) slots. k sweep on paired
  step ms; batch-adaptive schedule.
- Hard rule: spec c1 > plain c1, c4/c8 within noise of plain, acceptance in
  every gate. Pin the winner as canonical; plain path behind a switch.

**Compaction handoff** appended here: pins, tree state, open items ranked,
verification recipes, traps that cost time.

### Session 2 — prefill (new session from the handoff)

- MoE tile GEMM (`ggml_moe_mm_id`, NE20=8) and Q8 wide GEMMs: exist - but
  the tile route was gated on fp16 activations and never engaged for the
  bf16 GLM model until W11 (2026-09-12); prefill 84 -> 182 t/s at 2500.
- KDA prefill already runs on Metal through the fused `kda_step`
  (prepare + `kda_recur` sequential scan + gated rmsnorm); ~78 ms/layer at
  2500 tokens is the scan, the rest of the layer is the projections.
- KDA chunked prefill, the differentiator over ds4's serial scan: chunk 64
  WY/UT transform with elementwise cumulative decays on simdgroup MMA, fp32
  in the inverse ("keep FP32 in the WY inverse", CUDA plan). Fallback if
  the chunk kernel slips: a (head, value-group)-parallel recurrent kernel
  like ds4's, overlapped with the MoE GEMM in the same CB.
- Indexer prefill: pooled keys once per request, tiled scoring, dense path
  under 2,051 tokens. Sparse MLA prefill: head-batched MMA over the
  selection list (`quixicore_mla_sparse_prefill.py` design,
  `mla_prefill_fa_mma` precedent).
- Single-chunk scheduling: `max_num_batched_tokens` = 2048 + `max_num_seqs`
  x (k+1). Boot-ramp protocol stays mandatory.
- Walls at 512/1000/2048/3000 via disjoint id windows; phase-bracket split
  and xctrace ground truth before fusing.

Exit: 2048-token prefill at or above ds4's on this box.

### Session 3 — long context, cleanup, PR

- Needle ladder 16k/32k/128k/262k; KV budget re-sized from measured
  bytes/token; env-switch audit; delete bit-exact-proven kill switches,
  keep ULP sentinels; comment hygiene.
- Domain-sliced commits: kernels + metallib; python serving path; adapter;
  profile + harness; tests; perf docs. Rebuild `_quixicore_C`; re-gate the
  DSV4 and Qwen anchors bit-exact; profile `status` flipped to supported.
- PR description opens with the base-recipe clock (ds4 best config on this
  box: decode / prefill) against `glm53f-q2-1`, then the retained/rejected
  ledger.

### Session 4 — decode (research digest + wave plan, 2026-09-14)

Directive: decode first; prefill only as a by-product. Bar: ds4 exact-token
gate 22.52 tok/s, and ds4 PR #964 (open, bit-exact, +27-35% decode on its
own workload) would put it near 29-30. Goal (clarified by the user
2026-09-14): exceed ds4's speeds by AS MUCH AS POSSIBLE - the +50% figure
was only the DSV4 profile's example, not a fixed gate for GLM. Working
targets: >= 40 tok/s off1-2000, probe >= 50, and keep going.

**Where the K=1 step goes (measured 2026-09-13/14, `w14-budget/`).** The
phase profiler is sync-inflated (258 ms/step under it vs 54 real), so the
budget is built from the per-kernel microbench at the exact GLM shapes
(`kbench.py`, GLM env, `kbench_q8nr.log`) against the measured step:

| class | M=1 (no-spec) | M=2 (K=1 verify) | GB/s now | note |
| --- | ---: | ---: | ---: | --- |
| routed experts w13 iq2_xxs pair-swiglu, 43 layers | 7.38 | 13.60 | 202-220 | scales linearly with T: NOT bandwidth-bound |
| routed experts w2 q2_K sum, 43 layers | 4.08 | 5.83 | 232-325 | same |
| dense GEMVs (34 KDA + 11 MLA + 43 shexp + head), q8_0 NR on | 15.92 | 15.30 | 740-950 large; 250-370 q4_K; 20-70 tiny | q4_K q\|k 4.35 ms at ~300 GB/s; f_b\|g_b 1.1 ms at 70 GB/s; shexp 2.3 ms at ~480 |
| router F32 (43) + indexer bf16 (11) | 1.5 | 1.5 | | |
| **GEMV total** | **~29** | **~36 (+~2 drafter)** | | |
| step measured | 43 | ~54 | | non-GEMV = 14-16 ms (KDA prepare/recur/norm, MLA sparse decode + indexer, mHC x90, sampling, launch gaps) |

Weight floor: 10.60 GB/token = 15.1 ms at 700 GB/s (13.3 at 800).

**Research digest (what to copy).** ds4 PR #964 (findings:
`scratchpad/ds4_pr964_findings.md`): its exact +30% came from launch
removal and fusion, not from GEMV bandwidth (its dense kernels were already
at 700+): mHC producer fused (+5.7%), KDA gate pairing, HC-expand epilogues
folded into the producing GEMV, shared-down + HC fusion, router top-8 in
one simdgroup with an exact ordered select, KDA inputs (q|k|v|f_a|g_a|beta)
in ONE dispatch, KDA decode values interleaved x4 rows, exact phased DSA
kernels; 637 dispatches/token remain (~2.9 ms). Its routed MoE at Q4_K runs
~600 GB/s. llama.cpp master (`kernels/mul_mv.metal`): iq2_xxs kernel
unchanged in design (threadgroup codebook, byte lookups; N_R0 4 / N_SG 2,
row-split for short rows), q2_K 4/2, mul_mv_ext (float4x4 dequant, nxpsg
threads per row) for K-quants at ne11 4..8; nothing there beats our
geometry. vLLM/SGLang: MTP K>1 (SGLang GLM-5 accept length 2.76 at K=3);
vLLM GLM-5.3 MTP acceptance bug #55605. Conclusion: the expert kernels are
the one place we are far below everyone (200 vs 600 GB/s); the rest is
launch/fusion work of the kind ds4 did, plus MTP K=2 which nobody on Metal
has made pay.

**Waves (decode).**

- W15 expert GEMVs (largest lever: ~8 ms no-spec, ~11 ms at K=1).
  `qgemv_moe_mr_iq2_xxs_swiglu` and `qgemv_moe_mr_q2_K_sum` from 200-325
  GB/s to >= 550. Diagnose ALU vs memory first (streaming-only twin of each
  kernel). Candidates, keeping the per-lane chunk order so results stay
  bit-identical: codebook rows as 8-byte loads and sign application by
  sign-bit xor (llama.cpp #28301 / ds4), 8-16 byte weight loads, more rows
  per simdgroup and per threadgroup (amortize the 2 KB codebook fill that is
  paid 4096x per layer), token-batched walk that streams each selected
  expert row once for all T tokens (K=1 verify has T=2, K=2 T=3), and
  one-threadgroup-per-expert-row-tile grids. Same for q2_K down (ushort
  loads -> uint4, NR0 8, token loop inside). Target: 43-layer expert time
  <= 4.5 ms at T=1, <= 7 ms at T=2.
- W16 non-GEMV budget with a real GPU timeline (`xctrace` Metal System
  Trace per-encoder intervals; the phase profiler cannot do this), then
  fuse in ds4's order: KDA in_proj to ONE launch incl. f_b|g_b (f_b/g_b are
  1.1 ms of pure launch floor), prepare + gated norm folds, mHC sites, MLA
  decode's indexer chain (idx_wq_b/idx_wk/idx_op/idx_insert = 4 launches x
  11), absorb bmm (k_b/v_b per-head at 240 GB/s), q_a|kv_a_mqa merge,
  shexp gate|up pair-swiglu (one launch, no separate qc_swiglu), router
  F32 GEMV + top-k in one kernel. Target: non-GEMV 14-16 -> <= 9 ms.
- W17 q4_K kda q|k: 4.35 ms at ~300 GB/s -> >= 650 (1.9 ms). Same
  arithmetic order as `qgemv_q4k_nr` (bit-exact): 8-byte q loads, more
  blocks in flight, or a load-time repack of the 144-B block into a
  lane-contiguous layout (same values, one 16-B load per lane per row).
- W18 MTP K=2: pos-1 acceptance is broken on prose (0.03-0.42 vs pos-0
  0.84; `w9-mtp/accept_by_pos_k2_post.log`) and the step is ~94 ms vs the
  ~70 the T=3 verify + second draft step should cost. Fix the second draft
  step (attention metadata / slot mapping / recycled hidden for the pooled
  indexer and sparse MLA at draft_index=1), then measure tok/cycle x
  step. Only pays after W15 (expert cost is linear in T).
- W19 sampling + drafter overhead: `sample_tokens` is 17 ms/call
  sync-inflated at K=1 (drafter 12 of it); count launches per step (add a
  dispatch counter to the Metal encoder), remove host syncs.

Gate per wave: tf short/long + 8tok / off1-2000 / 2500x64 pins
(7dd30ea193a6 / a88022b375ea / f80d14780458) when bit-exact; a numerics
change re-pins through needles. DSV4/Qwen anchors are unaffected by
construction (GLM-only env block / per-layer flags); re-gate only if a
shared kernel body changes.

### Port

Port the finished kernels to QuixiCore-Metal per the sync contract in
`csrc/quixicore/metal/README.md` (develop here first; port finished pieces
upstream; drift check against the port branch).

## Risks

| risk | mitigation |
| --- | --- |
| Memory: 89.9 GiB weights + fp32 KDA state per seq leaves ~10 GiB for KV and transients under the thrash edge | fixed KV pool, small `max_num_seqs`, `vm_stat` on every boot, residency pin, no vision |
| KDA numerics: per-channel decay, bounded gate, L2 norms; bf16 state is a known quality hazard | fp32 state, float64 oracle, per-layer parity vs reference dumps before any fusion |
| Adapter: ds4-private tensor naming, absorbed k_b/v_b, no llama.cpp cross-check for this file | header census first, NLL fixture vs the A100 record, Unsloth GGUF as an optional second reference |
| Indexer/tail/pool-boundary bugs invisible to throughput | port the CUDA parity tests; bracket counts on pool updates |
| Sparse MLA is the historical Metal blocker (GLM 5.2 never served) | NoPE bf16 shape is simpler; dense fallback under 2,051 tokens ships first |
| Spec on a recurrent mixer amortizes poorly | small k, batch-adaptive schedule, judge on step rate not acceptance luck |
| ds4 bar unknown until measured | measured first, MTP on and off; the bar is its best config |
| Power/thermal sensitivity | AC power, rested box, record wattage in every entry |

## Status log

- 2026-09-10: plan written; branch cut from main 55fdb54bd; Q2 download
  started into `~/models/antirez-glm-5.3-flash-gguf/`; ds4 `6289c516`
  cloned to `~/.local/scratch/ds4-upstream` and built (`make`, Metal).
  Nothing served.
- 2026-09-10: profile registered. `sources.glm53f-q2` (quant tag `Q2`,
  sha-pinned, `min_memory_bytes.metal` 128 GiB, DFlash2 registered as the
  unmeasured candidate drafter) and `glm53f-q2-1` (`status: in-progress`,
  8 GiB fixed KV pool, `max_num_seqs` 8, `max_model_len` 131072,
  `mamba_ssm_cache_dtype` float32, `language_model_only`, glm45/glm47
  parsers). `tests/slimserve/test_profiles.py` 68 passed. The CLI gate on
  in-progress records now honours `SLIMSERVE_SERVE_IN_PROGRESS=1`
  (warns instead of dying) so the campaign can boot the record.
- 2026-09-10: header census from the first 64 MB of the download
  (`perf/results/2026-09-10/glm53f-q2-discovery/gguf_header_census.txt`,
  plus the embedded 8,617-char chat template as `chat_template.jinja`).
  Confirms the artifact table: 1,412 tensors, 50 KV pairs (all model
  hparams present under `glm5-next.*`, including `layer_types`,
  `swiglu_limit` 10, `expert_weights_scale` 2.5, `gate_lower_bound` -5,
  indexer `pool_size` 4 / `top_k` 2048), 12 sparse-MLA blocks (11 trunk +
  nextn), hc tensors on the 45 trunk blocks only, `ffn_gate_inp` /
  `exp_probs_b` on 43 blocks (42 MoE + nextn), `eot` 154827, `eom` 154829,
  `unk` 154820. Total tensor bytes 89.87 GiB.
- 2026-09-10: blocker list with file:line saved to
  `perf/results/2026-09-10/glm53f-q2-discovery/metal_blockers.md`. Two
  findings change the bring-up: `Glm5NextForCausalLM` is not in the model
  registry (only the ConditionalGeneration wrapper), so the GGUF config
  emits that name with `language_model_only`; and the tiled MoE prefill
  GEMM silently caps at 256 experts (`qc_metal_serving.mm:3673`,
  `gguf/fused_moe.py:686,:756`), so 288 experts drop to the per-slot GEMV
  path until both caps are raised. The V2 model runner is forced for
  `Glm5Next*` (`vllm/config/vllm.py:82`), as for Qwen3.8 on Metal.
- 2026-09-10: KDA torch-native path landed (bring-up item 3).
  `vllm/model_executor/layers/mamba/gdn/kda_mps_fallback.py` (gate both
  forms, L2 norm + K^-0.5 scale, per-key-channel decay recurrence, gated
  sigmoid RMSNorm, conv decode/prefill/spec, recurrent decode/prefill/spec
  with slot-table resume/store, `KdaVarlenPlan` one-host-copy metadata);
  `kimi_gdn_linear_attn.py` gained `VLLM_METAL_KDA` (unset/1 = native on
  non-CUDA platforms, 0 = Triton everywhere, `native` = native everywhere
  for CUDA A/B) and a `_forward_native` dispatch placed before
  `_get_kda_kernels()`, so no Triton module is touched on Metal; the
  `assert backend == "triton"` is gone. `tests/model_executor/test_kda_native.py`
  58 passed on CPU and MPS (float64 reference, prefill == token-by-token
  decode, spec rollback == decode over the accepted prefix, MPS
  determinism x2). Known deltas vs CUDA: fp32 everywhere (the chunk kernel
  rounds l2norm to bf16), NULL rows zeroed for all positions, slot 0 never
  written. Prefill is a sequential per-token torch loop (~7 launches per
  token per layer): the fused Metal `kda_step` kernel is decode wave W1.
- 2026-09-10: `glm5-next` GGUF loading landed (bring-up item 1).
  `vllm/transformers_utils/gguf_glm5_next.py` builds the config from the
  `glm5-next.*` keys alone (tensor shapes only verify) and emits
  `architectures=["Glm5NextForConditionalGeneration"]` (the vision
  sub-config is a default; `language_model_only` makes the tower a
  StageMissingLayer); a vendored `configs/glm5_next.py` mirror covers a
  transformers without `glm5_next` (this venv is 5.15.0).
  `gguf_adapters/glm5_next.py` (`Glm5NextGGUFAdapter`) maps every tensor
  in the census and raises on missing/extra/wrong-shape; policy: routed
  experts stay IQ2_XXS/Q2_K packed, Q8_0/Q4_K linears stay packed
  (KDA `in_proj_qkvgfab` is a 6-shard mixed Q4_K/Q8_0 stack), absorbed
  `attn_k_b`+`attn_v_b` -> one dequantized bf16 `kv_b_proj` [32768,512],
  `token_embd` dequantized (embed is built without quant_config), indexer
  BF16 and all F32 vectors pass through (router bias, `A_log`, `dt_bias`
  never downcast), nextn dropped unless the speculator is `glm5_next_mtp`.
  Wiring: `gguf_loader.py`, `gguf_config_parser.py`,
  `tokenizers/registry.py` (glm4 BPE split, keeps the file's own chat
  template), `entrypoints/openai/model_parsers.py` ("glm5-next" ->
  glm45/glm47), `models/registry.py` (+`Glm5NextForCausalLM` as a
  text-only fallback). `tests/model_executor/test_glm5_next_gguf.py` 16
  passed; existing adapter tests unchanged except a PRE-EXISTING failure
  in `test_kimi_k3_ep.py` (`RoutedExperts` has no `swiglu_limit`,
  `gguf/fused_moe.py:1194`), which is also the first place to look when
  wiring GLM's `swiglu_limit` 10 through the GGUF MoE path.
  Open: whether the multimodal registry tries to build a processor from
  the .gguf path under `language_model_only` (first boot will tell);
  `f_b/g_b` kept packed Q8_0 (1 MiB/layer to dequantize if a kernel wants
  bf16).
- Oracle note: ds4 itself is the adapter oracle for this file.
  `./ds4 --dump-logprobs FILE` writes greedy top-logprobs for a prompt and
  `--first-token-test` runs an exact CPU whole-model pass on the first
  token; compare our first-token logits and greedy continuation against
  those before trusting any per-layer parity built on our own dequant.
- 2026-09-10: **upstream's tracked `vllm/quixicore_metal.metallib` does not
  load on this box.** Commit b47509dd1 rebuilt it under `cmake/metal.cmake`'s
  `-std=metal4.0` (for the `mpp::tensor_ops` M5 path in `qgemm_sm.metal`,
  guarded by `__HAVE_TENSOR__`); macOS 15.6 refuses it with
  `MTLLibraryErrorDomain Code=1 "This library is using language version 4.0
  which is not supported on this OS."`, which surfaces as
  `quixicore: failed to load metallib` on the first kernel launch (EngineCore
  dies on the primer request, server answers 500). Fix on this box: rebuild
  from source with `-std=metal3.1` (`perf/results/harness_assets/build_metallib.sh`,
  run under bash, ~19 s, 97 sources, 1862 functions, loads clean); the
  worktree carries that rebuild as an uncommitted binary change. The
  Metal-4 kernels are compiled out under 3.1, which is correct for the M1
  Ultra (no neural accelerators). Raw: `perf/results/2026-09-10/dsv4-regate-worktree/{boot_metal4_fail.log,regate_metal4_fail.log}`.
  For the PR: the tracked metallib must stay loadable on macOS 15 (build it
  with 3.1, or gate the 4.0 build on the host SDK) or every non-macOS-26
  Metal box is dead on arrival.
- 2026-09-10: **DSV4 anchors re-gated bit-exact on the worktree stack**
  (main 55fdb54bd + campaign edits + metal3.1-rebuilt metallib + locally
  built `_quixicore_C`): `dsv4-xxs-1` primer ok, 8tok `573db39598e7` 5/15/3,
  off1-2000 `bb83cc3054a3` 1582/2110/422 wall 57.99 s (34.5 tok/s),
  2500x64 `73f41acf8ca0` 40/120/24 wall 5.24 s. All three equal the prior
  pins, so the FP8-draft-KV policy commit did not roll this profile's
  trajectory. Raw: `perf/results/2026-09-10/dsv4-regate-worktree/`.
  Monitor lesson: liveness must match `slimserve.cli dsv4-xxs-1` too, since
  the CLI does not exec into `api_server` until after the model loads.
- 2026-09-10: `swiglu_limit` wired for GLM (bring-up item). ds4 applies
  `silu(min(g, L)) * clamp(u, -L, L)` with `glm5-next.swiglu_limit` = 10 on
  the dense FFNs, the shared expert and the routed experts (same
  `DS4_SWIGLU_CLAMP_EXP` as DSV4); upstream `glm5_next.py` applied none
  (plain `SiluAndMul` via `DeepseekV2MoE`/`DeepseekV2MLP`). Now
  `DeepseekV2MLP(swiglu_limit=)` -> `SiluAndMulWithClamp`, `DeepseekV2MoE`
  passes `getattr(config, "swiglu_limit", None)` to the shared expert and
  to `FusedMoE`, and the GLM dense layers pass it too; DeepSeek configs
  without the field are unchanged. The Metal GEMV (`qgemv_moe_mr_swiglu`
  `has_clamp`) and the unfused `act()` clamp already implement exactly
  that formula, so the GGUF MoE path needs no kernel change.
- 2026-09-10: tiled MoE prefill cap raised for 288 experts:
  `QC_MOE_MAP0_MAX_E = 512` in `moe_mm_id.metal` (threadgroup arrays were
  hard-sized 256; map0 is one thread per expert, so 288 fits the 1024
  limit), host check `E <= 512` in `qc_metal_serving.mm`, python gates
  `w1.shape[0]/w2.shape[0] <= 512` in `gguf/fused_moe.py`. Compiles under
  metal3.1; NOT yet rebuilt into the metallib/.so (next rebuild batches it
  with the routing agent's kernels, then DSV4 re-gates again).
- 2026-09-10: **Metal routing for `glm5_next` landed (bring-up items 2/4/5/6,
  routing agent).** Backend: `MLAAttention` -> `MetalPlatform.get_attn_backend_cls`
  (`platforms/metal.py:220-233`, `use_mla and use_sparse` -> `METAL_MLA_SPARSE`,
  plain MLA raises) -> rewritten `v1/attention/backends/mla/metal_mla_sparse.py`
  (504 lines): no prefill backend on Metal so every token goes through
  `forward_mqa`; base `do_kv_cache_update` overridden by a torch insert; KV
  layout `(num_blocks, block_size, 512)` bf16/f16 identical to QuixiCore's so
  the hybrid pool / packed slab / tiers see the same pages; prefill requests
  with seq_len <= 2051 run dense causal over the latent prefix (equal to the
  fully-selected sparse result), longer rows use the gathered path; PAD (-1)
  gathers zeroed; PAD inserts land on block 0 row 0. **fp8 latent pages are
  NOT implemented** (auto/bf16/f16 only; `VllmConfig()` on this box defaults
  to fp8_e4m3 - the profile pins `kv_cache_dtype auto`, which needs a
  recorded exception or an e4m3 LUT path later). Indexer:
  `glm5_next_indexer.py` (+254) torch-native producer gated by
  `VLLM_METAL_GLM_INDEXER` (unset/1 native on MPS, `native` everywhere, `0`
  Triton); top-k tie order is torch.topk's (sets identical); pool keys
  rounded through bf16 where Triton does. `topk_indices_buffer` moved to
  `glm5_next_device()` in `glm5_next.py:422` / `glm5_next_mtp.py:99`.
  `glm5_next.py:740` guards the `transformers.models.glm5_next` import
  (venv transformers 5.15.0 lacks it; falls back to the vendored config,
  processors None - image requests would fail at processor construction;
  fine under `language_model_only`, revisit for vision). mHC: quixicore
  route, parity vs `mhc_pre_torch`/`mhc_post_torch` D=256/4096 T=1..130
  passes; no CUDA streams (`_overlap_kda/_overlap_router` False). Router:
  `GroupedTopk.forward_mps -> grouped_topk` matches a float64 reference,
  total over NaN rows. Smoke: `Glm5NextForCausalLM` (2 KDA + 1 DSA, hidden
  64, dense FFNs since unquantized FusedMoE has no Metal backend) on MPS:
  prefill+decode vs fresh prefill same argmax, cosine > 0.98, all finite.
  Tests: 5 new files 47 passed; related suite 565 passed / 528 skipped.
  Pre-existing collection errors untouched: `test_pooled_indexer_parity.py`,
  `test_mtp_adapter.py` (transformers glm5_next), `test_deepcontext_qualification.py`
  (pyarrow). Untested: MPS 32-bit index windowing on > 2^31-element slabs;
  torch-producer cost at 1M context.
- 2026-09-10: native stack rebuilt for the map0 cap: metallib (metal3.1,
  1862 fns, loads) + `_quixicore_C` (89 pybind fns, `ggml_moe_mm_id`
  present). DSV4 re-gate #2 on this build is owed before it is trusted;
  scheduled right after the ds4 bar + oracle (both need the memory).
- 2026-09-11: **ds4 BAR MEASURED (this box, clean memory, ds4 6289c516).**
  CLI no-MTP: prefill 143.03 / generation 18.19 t/s. CLI `--mtp`:
  prefill 143.11 / **generation 22.52 t/s** (90.1% accept, 1.90 tok/cycle,
  output byte-identical to no-MTP) - this is the decode number to beat.
  ds4-bench 2048/128: prefill 139.16 / gen 17.39 (steady 17.40).
  ds4-server: 1018-tok prefill 144.5 t/s, decode 19.75 t/s avg over 612
  tokens, 2518-tok prefill 132.9 t/s. Harness shas vs ds4-server are
  not comparable (it re-tokenizes to 1018 tokens). Full table in
  `perf/baseline_status.md` (top section); raw in
  `perf/results/2026-09-10/glm53f-q2-bar/`. Two run lessons: ds4 must
  run from its own directory (`metal/*.metal` resolved relative to
  cwd), and its wired residency drains for minutes after exit.
  Campaign targets restated against the bar: decode >= 35 tok/s c1
  would be 1.55x ds4-MTP; prefill >= 300 t/s @2048 would be 2.1x.
- 2026-09-11: ds4 oracle dumps captured (`perf/results/2026-09-10/glm53f-q2-oracle/`):
  `short_logprobs.json` (9-token raw prompt "The three most important ideas
  in distributed systems are", 32 greedy steps, top-8 logit+logprob per
  step; greedy text ": redundancy, redundancy, and redundancy. - Leslie
  Lamport ...") and `long_logprobs.json` (the 1000-token bar prompt, 64
  steps). `--first-token-test` is CPU-only and `ds4 --cpu` segfaults on
  this file (rc 139, `first_token.log`); not chased - the Metal dumps
  are the oracle. Our GGUF tokenizer reproduces ds4 exactly: short 9 /
  long 1000 tokens, ids [785, 2326, 1429, ...], no BOS added by
  `encode()` (raw completions are directly comparable).
  `compare_oracle.py --dump ... --prompt-file ... [--tokenizer GGUF]`
  hits `/v1/completions` greedy with logprobs and reports the matched
  greedy prefix, first divergence with both top-4 lists, and top-1
  logprob deltas. Correctness gate for the first boot: short dump >= 24/32
  greedy prefix (Q2 fp16-vs-fp32 accumulation will diverge eventually;
  a divergence with near-tied logprobs is fine, one with a confident
  ds4 top-1 missing from our top-8 is a bug).
- 2026-09-11: DSV4 re-gate #2 on the map0-cap rebuild: bit-exact
  (8tok 573db39598e7 5/15/3 1.79 s; off1-2000 bb83cc3054a3 1582/2110/422
  57.95 s; 2500x64 73f41acf8ca0 40/120/24 5.19 s). Raw:
  `perf/results/2026-09-10/dsv4-regate-worktree/r2/`. The native stack
  now in the worktree is trusted for the GLM first boot.
- 2026-09-11: **FIRST BOOT attempt 1** (`glm53f-q2-1 --no-spec`,
  `perf/results/2026-09-10/glm53f-q2-boot1/boot_attempt1_blocksize.log`):
  adapter mapped 1383/1412 tensors for the trunk (29 = nextn MTP block,
  dropped by design); load_weights 45.6 s, process_weights_after_loading
  33.9 s (GGUFMoEMethod 31.2 s over 42 layers); text-only mode; MLA
  prefill backend none -> top-k MQA path (expected on Metal);
  determine_available_memory RAN THE PROFILING FORWARD WITHOUT ERROR
  (boot +90 s -> +148 s); KV plan: hybrid aligner mamba page 4,341,760 B
  (fp32 KDA state), attention 1024 B/token -> block 4288 tokens, 10
  padding layers (29.4% pool waste), 421,813 tokens of KV in the 8 GiB
  pool (3.22x 131072). Died at `validate_block_size`: align mode needs
  block_size <= max_num_batched_tokens (2048). Fix: profile
  `max_num_batched_tokens` 2048 -> 4288 (one prefill chunk == one block;
  note added to the record; test_profiles 68 passed). bf16 KDA state
  would halve the block to 2144 but the fp32 state is a quality choice
  carried from the Qwen GDN campaign; revisit only with a measured
  reason. Monitor lesson: never end a monitor pipe with `cut` (block
  buffered, swallowed 30 min of output).
- 2026-09-11: **FIRST TOKENS.** Boot attempts 2-5
  (`perf/results/2026-09-10/glm53f-q2-boot1/boot_attempt{2,3,4}_*.log`):
  (2) EngineCore died on the first request with `kernel not found in
  metallib: qgemm_sm_t2_q4_K` - upstream main's K-quant small-M GEMM
  selection (tape verify band `qc_metal_serving.mm` ~:700, eager
  `quixicore/ops.py ggml_mul_mat_sm`, DFlash `dflash_sample_greedy`
  variant 16) assumes the M5 tensor-ops kernels, which are compiled out
  of a metal3.1 build. Fix: `has_kernel(name)` / `has_tensor_qgemm()`
  (cached probe for `qgemm_sm_t2_q4_K`) in the host, exposed through
  pybind and `quixicore_ops.has_kernel`; all three selectors fall back
  to the pre-tensor winners (paired-plane 11/12 for q4_K/q5_K, span 10
  for q6_K, 9 otherwise) when absent. DSV4 never reaches these branches
  (no K-quant linears, no DFlash greedy path), but re-gate #3 is owed
  for the .so rebuild. (3) `RuntimeError: x must be contiguous` from
  the KDA layer: `f_a`/`g_a` are split views of the fused qkvgfab
  projection and the 9-token request landed in the Metal small-M route;
  `gguf/linear.py _fused_mul_mat_gguf` now makes x contiguous on Metal
  (every quixicore host entry checks it). (4) first coherent greedy
  output: ": Consistency, Availability, and Partition Tolerance. These
  are known as the CAP theorem..." (9 prompt + 32 tokens, 13.2 s cold).
  (5) `prompt_logprobs` needed a torch fallback in
  `v1/worker/gpu/sample/prompt_logprob.py` (no Triton on Metal, no
  native `v2_prompt_logprobs_token_ids`).
- 2026-09-11: **CORRECTNESS vs ds4 (boot1 stack).** Greedy prefix vs
  the dumps: short 1/32 (step-1 near-tie swap: both engines have
  ' Cons'/' redundancy' as top-2, 0.18 nats apart, flipped), long 5/64
  (diverges at a ds4 tie of 0.004 nats). Teacher-forced (score ds4's
  own greedy tokens under our model, `teacher_forced.py`,
  `tf_{short,long}_boot1.json`): short 31/32 argmax agreement, mean
  |dlogprob| 0.049 max 0.67; long (after 1000 tokens) 52/64, mean 0.175
  max 0.97, every disagreement at a ds4 margin < 0.6 nats and the ds4
  token always in our top-8. Verdict: plausible Q2 numerics, no
  missing-token bug; the drift grows with context (0.05 -> 0.18) and is
  flagged: capture a ~300-token ds4 dump when ds4 next has the box and
  bisect (KDA fp32 recurrence vs MLA dense window vs MoE). Needle recall
  PASS at 1500 / 4000 / 6500 / 12000 tokens (`needle_boot1.log`): the
  sparse-MLA gathered path (> 2051), the indexer top-k and 3-chunk
  prefill at 4288 all recall the code word verbatim.
- 2026-09-11: **BRING-UP SPEED (no kernels yet).** Streaming 64-token
  prompt, 128 tokens: ttft 1.6 s, decode 3.04 tok/s (329 ms/step) x2
  identical. Needle prefill ~45-50 t/s (1500 tok 34 s, 12000 tok 244 s)
  - the KDA prefill is the sequential torch loop, Session 2 work.
  ds4 bar: 22.52 (MTP) / 18.19 decode, 143 prefill. Exact-token
  baseline gates running -> `perf/results/2026-09-11/glm53f-q2-baseline/`.
- 2026-09-11: **N0 BASELINE PINNED** (`perf/baseline_status.md` top
  section): 8tok `7dd30ea193a6` 20.9 s; off1-2000 `f9282de33fed` 2.94
  tok/s 680.4 s; 2500x64 `b6949a578f73` 70.3 s. Harness honored exact
  counts (1000/2000/2500). Gap to ds4-MTP: 7.5x decode. Next: phase
  split (`VLLM_QC_PHASE_PROF=1`, layer brackets mhc_attn_site / kda_attn
  / mla_attn / mhc_ffn_site / moe / dense_ffn added to `glm5_next.py`),
  then decode wave W1.
- 2026-09-11: **W1 FUSED KDA RETAINED: decode 3.04 -> 6.6 tok/s (329 -> 151
  ms/step).** Phase split at N0 put kda_attn at 232 of ~450 sync-inflated
  ms/step. New `kernels/linear_attention/kda/kda.metal` (fused prepare:
  conv ring + L2 norm + per-channel decay gate + beta; `kda_recur`: gdn_recur
  with a per-channel decay vector, varlen, fp32 page-packed pool in place,
  slot <= 0 null; sigmoid gated rmsnorm), host `kda_step` (one encode, three
  launches, ring_out scratch), `quixicore_ops.kda_step`, hook in
  `_forward_native` for pure non-spec batches (decode + varlen prefill; fresh
  prefill rows zero their pool rows, then load_initial=1). Spec batches and
  head_dim != 128 stay native. `VLLM_METAL_KDA_FUSED=0` pins the torch path.
  Parity: kernel state == float64 recurrence to 4e-8 (the MPS torch reference
  is 1.6e-3 off float64 - the torch path was the noisier one);
  `tests/kernels/test_kda_metal_fused.py` 6 passed, `test_kda_native.py` 58.
  Live: teacher-forced long 55/64 mean 0.124 (better than N0 52/64, 0.175),
  short 30/32; needles pass. Prefill also moved (needle 12k: 244 -> 159 s)
  since the varlen kernel replaces the per-token torch loop. Details in
  `perf/optimization_status.md` (2026-09-11 entry). Owed: DSV4 re-gate #3
  (extension rebuilt for `has_tensor_qgemm` + `kda_step`), W1 exact gates
  -> `perf/results/2026-09-11/glm53f-q2-w1-kda/gates/`.
- 2026-09-11: **W2 SPARSE LATENT MLA DECODE KERNEL RETAINED (small step:
  decode 6.6 -> 6.68 tok/s, off1-2000 6.08 -> 6.25).** New
  `kernels/attention/mla_sparse_latent/` (partition grid (H,R,P) online
  softmax over the indexer's top-k list against the paged bf16 latent +
  fp32 reduce), host `mla_sparse_latent_decode` (auto P = 1024/(R*H), cap
  32), `quixicore_ops.mla_sparse_latent_decode`, fast path in
  `metal_mla_sparse.sparse_attend_rows` (`VLLM_METAL_MLA_SPARSE_KERNEL=0`
  pins torch). Microbench R=1/H=64/W=2051: torch 0.55 ms -> 0.17 ms/layer.
  `tests/kernels/test_mla_sparse_latent_metal.py` 6 passed. Live gates
  (`glm53f-q2-w2-mla/gates/`): 8tok 7dd30ea193a6 (== N0/W1), off1-2000
  7656da8323ef 320.1 s, 2500x64 29b6e8772495 39.9 s; teacher-forced
  identical to W1 (55/64 long, 30/32 short); needles pass (12k 151 s).
  Lesson: the attention math was never the 5 ms/layer - the MLA bucket is
  torch glue (indexer producer, absorb bmm, projections, casts). Added
  fine-grained brackets (`mla_qkv_a`, `mla_q_b`, `mla_indexer`, `idx_*`,
  `mla_core`, `mla_absorb_bmm`, `mla_mqa`, `mla_vup`, `mla_o_proj`,
  `mhc_norm`) for the W3 profile. Also built + parity-tested (12 passed,
  `tests/kernels/test_mhc_fused_norm_metal.py`) the norm-fused mHC kernels
  `dsv4_mhc_fused_post_pre_norm_*` / `dsv4_mhc_pre_finalize_norm_*`
  (RMSNorm of the layer input in the epilogue; `_fuse_mhc_norm` now true on
  Metal, `VLLM_METAL_MHC_FUSE_NORM=0` pins the separate norm) - 90 fewer
  dispatches/step, measured in the W3 profiled boot. DSV4 re-gate #3 still
  owed (extension rebuilt again: `mla_sparse_latent_decode`, mHC norm).

- 2026-09-11: **W3 mHC SITES RETAINED: decode 6.68 -> 12.28 tok/s (81
  ms/step), off1-2000 6.25 -> 11.25 tok/s.** The fine profile put the 90
  mHC sites at ~1.1 ms each: the GLM path used the one-threadgroup-per-token
  monolith `dsv4_mhc_fused_post_pre` (0.73 ms/call at T=1, latency-starved
  on the 1.5 MB fn read) that DSV4 Metal had already rejected (2026-08-12).
  Host now encodes post + split pre (dots 25 simdgroups/token + finalize) in
  one encode (0.11 ms/call, bit-identical), monolith kernel deleted; the
  RMSNorm rides in `dsv4_mhc_pre_finalize_norm_*` (`_fuse_mhc_norm` true on
  Metal). Tests 20 passed; gates `glm53f-q2-w3-mhc/gates/` (8tok
  7dd30ea193a6 unchanged, off1-2000 e1b8cf9219b7 177.8 s, 2500x64
  12d98b9d33ab == W1); teacher-forced short 28/32 (0.069), long 56/64
  (0.155); needles pass. Note for DSV4: its own fused-op gate never took
  this host, so DSV4 numbers are unaffected; the shared `dsv4_mhc_pre` host
  only gained the optional norm arguments. DSV4 re-gate #3 still owed.

- 2026-09-11: **W4 fp32 ROUTING + METAL ROUTER KERNEL RETAINED: decode
  12.3 -> 14.9 tok/s (67 ms/step), off1-2000 11.26 -> 13.15; teacher-forced
  long 56/64 -> 60/64 (mean 0.155 -> 0.088).** Diagnostics first: cb-census
  GPU-busy 72/82 ms, ioreg GPU util 85-98% (GPU-side time, not host);
  `gemv_bench.py`: GEMVs ~37 ms/step, near their launch floor at 35 MB.
  `moe_router_topk` kernel (one simdgroup/token, bias top-k, unbiased
  weights, renorm, scale) routed inside `grouped_topk()`; it only engaged
  after GLM routing was switched to fp32 (`moe_router_dtype` in
  `gguf_glm5_next.py`, `force_fp32_compute` on Metal in `DeepseekV2MoE`) -
  the bf16 gate was also the source of the ds4 disagreement. A/B calibrated
  ~20 us per small MPS op (670 ops -> 14 ms). KDA prefill `fresh.any()`
  sync removed (ttft 0.88 -> 0.80 s). Gates `glm53f-q2-w4-router/gates/`
  (8tok 5f870096e207, off1-2000 29878bddab74 152.1 s, 2500x64 c4f607e5cc8a);
  tests 32 passed. Next: widened op census -> fuse remaining glue.

- 2026-09-11: **W5 KDA LAUNCH DIET RETAINED: decode 14.9 -> 15.85 tok/s
  (63 ms/step), off1-2000 13.15 -> 13.93; gates bit-identical to W4b.**
  Same-type in_proj shards coalesced at load (two GEMVs, not six), kda_step
  writes the attention slab directly (`out=`), o_proj result returned
  instead of copied, decode cu arange cached per step. Fused post+dots mHC
  pass measured slower and reverted. Found: CPU->MPS copy_ blocks until the
  GPU queue drains (the 1.9 ms/step MLA-metadata stall). W6 in flight:
  Metal indexer (pool logits kernel + torch top-k + expand kernel), paged
  row insert kernel for the indexer and latent caches, direct decode
  return in forward_mqa, device-built request ids in the MLA metadata.

- 2026-09-11: **W6 METAL INDEXER + ROW INSERT + METADATA RETAINED: decode
  15.85 -> 18.78 tok/s (53 ms/step), off1-2000 13.93 -> 16.65.** Kernels
  `glm5_indexer_pool_logits` / `glm5_indexer_expand_topk` /
  `paged_row_insert` (`kernels/attention/glm5_indexer/`), torch sorted
  top-k in between; latent + indexer inserts on the kernel; decode MQA
  returned directly; MLA metadata request ids built on device (the
  CPU->MPS copy_ stall). Tests 8 passed; gates 8tok/off1-2000 shas ==
  W5, 2500x64 5b6e19532bf8 (rolled, reduction order); teacher-forced
  unchanged (60/64); needles pass. We now match ds4 no-MTP (18.19) on the
  probe; ds4 MTP 22.52 remains the bar -> W7 remaining glue (census), then
  MTP speculation (the fused KDA path must learn the spec batch layout:
  `_forward_native` spec branch uses kda_conv_spec_update_native /
  kda_recurrent_spec_native).

- 2026-09-11: **W7 REMAINING GLUE DIET RETAINED BUT FLAT: decode 18.78 ->
  18.89 tok/s.** `kv_meta_prepare` (one launch for all six groups' block
  tables + slot mappings, `kernels/serving/kv_meta/`), `mamba_last_blocks`,
  `glm5_indexer_pack` (fused [wk|gate|weights] linear + norm/pack kernel),
  q2_K sum `accumulate=` folding the shared-expert add (runtime flag,
  bit-exact vs the bf16 add), GEMV `out=` column slices for the KDA
  hetero shards, int64 `paged_row_insert`. 19 tests pass; gates 8tok sha
  unchanged, off1/2500 rolled (pack LayerNorm fp32 order, near-tie flip),
  teacher-forced == W6, needles pass. **Metal System Trace
  (`w8-gemv/gputrace.sh` + `analyze_trace.py`): GPU busy 77%/step; one
  9.8 ms gap per step between the sampler blit and the next step's first
  compute (CPU scheduling/input prep with the GPU idle; Metal forces
  async_scheduling off, `VLLM_METAL_ASYNC_SCHED=1` opts in) + ~0.9 ms
  before the mamba metadata build.** Next: W8a async scheduling, W8b GEMV
  bandwidth (q8_0 llama.cpp-geometry kernel `qgemv_q8_0_nr` written,
  `VLLM_QC_Q8_NR_GEOM` sweep via `w8-gemv/bench_q8.py`, unbuilt), then MTP.

- 2026-09-11: **W8a ASYNC SCHEDULING RETAINED: decode 18.89 -> 21.94
  tok/s, off1-2000 16.63 -> 19.38 (103 s).** `platforms/metal.py` no
  longer forces `async_scheduling=False` (`VLLM_METAL_ASYNC_SCHED=0`
  pins sync). All three gate shas identical to W7; teacher-forced and
  needles unchanged. GPU busy 77% -> 95%. ds4 MTP 22.52 is 3% away on a
  no-spec decode. Next: W8b GEMV bandwidth sweep (q8_0 NR kernel,
  split-K q4_K NR `VLLM_QC_Q4K_NR_KSPLIT`, split-K q2_K sum
  `VLLM_QC_Q2K_SUM_KSPLIT`; all env-gated, defaults unchanged), then MTP.

- 2026-09-11: **W8b q8_0 GEMV RETAINED: decode 21.94 -> 23.26 tok/s,
  off1-2000 19.38 -> 20.38 (98 s). ds4's MTP bar 22.52 is beaten with NO
  speculation.** `qgemv_q8_0_nr` (llama.cpp 2x4 geometry, default route,
  `VLLM_QC_Q8_NR=0` / `VLLM_QC_Q8_NR_GEOM`): kda_v 375 -> 619 GB/s,
  attn_output 354 -> 787. Split-K q4_K/q2_K-sum variants measured slower
  and were removed. Gates rolled (decode numerics: fp32 int8 dots vs
  half-rounded products; parity test within 4 ulp of exact); teacher-forced
  is a prefill oracle and cannot see decode GEMV changes (byte-identical).
  DSV4 re-gate owed (shares the fp16 q8_0 route). Next: W9 MTP (nextn
  block registered as the Metal record's `speculator`, `model: "target"`,
  `glm5_next_mtp` K=1; engine.py resolves the draft to the target GGUF).

- 2026-09-11: **W9 GGUF MTP DRAFTER (K=1) RETAINED: decode 23.26 -> 31.95
  tok/s, off1-2000 20.38 -> 26.50 (75.5 s, 84.6% accept, 1.85 tok/cycle).
  ds4's MTP bar 22.52 beaten by 18% on its own workload.** Fixes on the
  way: speculator hc_mult widening gated to deepseek_v4; torch-native
  Metal replacements for the drafter's Triton input-prep kernels;
  `qgemv_q8_0_nr_mb` (M=2..4, 4x4, bit-identical rows to batch-1) for the
  verify pass (+4 tok/s); and the drafter feed - the nextn head wants the
  POST-final-norm hidden state (pre-norm mean drafted at 72%, emitting
  `<|code_suffix|>` at fixed content positions; ds4 accepts 100% on the
  same loop prompts; `VLLM_GLM_MTP_HIDDEN=pre` keeps the diagnostic).
  Gate shas: 8tok == W7, off1-2000 bf329ffc964c and 2500x64 105b4a274b22
  rolled (lossless greedy verify: identical across drafter feeds). tf and
  needles unchanged. Diag hooks: `VLLM_SPEC_TRACE=<file>` (runner +
  speculator), `w9-mtp/{loop_accept,loop_len,accept_by_pos,ref_draft}.py`.
  K=2 needs `max_num_batched_tokens` 4352 (Mamba align block grows with
  K); first pass (pre-norm feed) lost - re-measured with the fix (see
  optimization_status W9). K=2 measured and lost (22.88). DFlash2 arm
  PARKED after two boot blockers (mixed KV alignment fixed in attn_utils;
  hybrid group block sizes vs hash_block_size 8832 unresolved - see
  optimization_status W9). Wave recipe trimmed: tf + 3 gates (~3 min);
  needles only for attention/indexer/KV-paging changes or the pre-PR gate
  (`NEEDLES=1`). Next: K=1 step overhead (54 vs 43 ms: q4_K in_proj mb,
  MoE second-token expert bytes, draft cost), then Session 2 prefill;
  DSV4 + Qwen re-gates owed (q8_0 NR/mb routes).
- 2026-09-12: **W11 PREFILL: MoE TILE GEMM ROUTED FOR bf16 - prefill 84 ->
  182 t/s at 2500 tokens, 86 -> 185-197 at 1000; ds4's 143 beaten.** The
  tile route (`ggml_moe_mm_id`) had required fp16 activations; GLM runs
  bf16, so every GLM prefill ran the per-slot decode GEMVs (the profile's
  `moe_w13` label is the GEMV branch's). Fix in `fused_moe.py`: fp16 cast
  in, reduce in fp16, one cast out (`test_metal_moe_mm_bf16.py`). tf 28/32
  (0.0605) / 58/64 (0.0843) - near-tie flips, mean distance to ds4
  improved; gates rolled (8tok 7dd30ea193a6, off1-2000 cf3a15f8c2ef,
  2500x64 15be63c98eae 16.5 s). The KDA-prefill arm (`kda_recur_prefill`)
  is dead code: the fused `kda_step` block already runs prefill on
  `kda_recur`; revert at the next rebuild. Sub-phases added (kda_in_proj /
  kda_gate_proj / kda_core / kda_o_proj, moe_mm_w13 / moe_mm_w2). Benches:
  `w11-prefill/bench_moe_mm.log` (q2_K down tile 5 TFLOPS at 2500 tokens
  vs 9 at 1024), `bench_mla_prefill.log` (MHA-form MPS SDPA 2-2.6x over
  the torch-fp32 attend at >= 2048). Remaining 2500-token budget (13.75 s):
  KDA layers (~2.7 s sequential scan + projections), MoE tiles (~5.5 s),
  MLA (~1.5 s) - profile pending (`prof_chain.sh`). Next: q2_K w64 down
  tile, KDA in_proj route, MLA MHA SDPA under the 2,051 dense limit,
  chunked KDA scan; 2048-token clock on ds4's own definition.
- 2026-09-12: **W12 KDA PREPARE TOKEN-CHUNKED, BIT-EXACT: prefill 182 ->
  230 t/s at 2500 tokens, 231 at 2048 (ds4 143).** The fused `kda_step`'s
  prepare kernel (conv + norm + gate) ran one simdgroup per (request, row)
  over every token serially - 256 simdgroups for a 2500-token request.
  Grid y = 32-token chunks (history from the raw rows for chunks > 0; last
  chunk writes the pool) plus R value rows per `kda_recur` simdgroup
  (R=1 89 ms/layer -> R=4 5.7 -> R=8 4.8, bit-identical; the first rows
  bench was broken - wrapper dropped the arg - and wrongly read "neutral");
  `kda_step` 107 -> 13.4 ms/layer at 2500 tokens. Shas identical to W11,
  tf identical, 23 KDA tests pass.
  Sub-phase profile (`phaseprof_moe.txt`): MoE tiles are now ~50% of the
  2500-token prefill (w13 60 + w2 51 ms/layer), MLA ~25% (mqa 166
  ms/layer), KDA projections ~15%. Next: MLA hybrid SDPA prefill (written,
  parity test), then MoE tiles.
- 2026-09-12: **W13 MLA FRESH-PREFILL ROWS VIA MPS SDPA (hybrid form)
  RETAINED: 2048 tokens 231 -> 245-251 t/s, 2500 230 -> 240; tf 29/32 /
  62/64 (best yet); 8tok sha unchanged, long gates rolled.** MoE w64 tile
  software pipelining measured neutral and reverted (byte-identical
  kernel). Session 2 exit met (2048-token prefill 1.8x ds4). Remaining
  2500-token budget ~10.4 s: MoE tiles ~5.4 s, KDA projections ~2.3 s,
  MLA ~1.3 s. Next: Session 3 - env-switch audit, dead-code removal
  (`kda_recurrent_prefill_metal` route; `kda_recur` rows variants), DSV4
  + Qwen anchor re-gates, needles, PR (opens with the ds4 bar), port.
- 2026-09-12: **SESSION 3 IN PROGRESS.** (1) `kda_recur` rows default
  R=8 (`bench_kda_rows.log`: R=1 89 / R=2 44 / R=4 5.7 / R=8 4.8 ms per
  layer at 2500 tokens, torch-loop exact; the first bench was broken and
  the W12 entries were corrected). (2) Dead code out: the torch-side
  `kda_recurrent_prefill_metal` serving route + `VLLM_METAL_KDA_PREFILL`
  (the fused `kda_step` block owns prefill; the wrapper stays as the
  kernel's test harness), the `VLLM_GLM_MTP_HIDDEN=pre` diagnostic feed,
  the `VLLM_SPEC_TRACE` hooks (runner + speculator). Env switches left,
  by class - torch-fallback null switches (bit-close): VLLM_METAL_KDA,
  _KDA_FUSED, _GLM_INDEXER, _INDEXER_PACK, _MLA_SPARSE_KERNEL,
  _MHC_FUSE_NORM, _ROW_INSERT, _KV_META, _MAMBA_LB; numerics sentinels
  (ULP class): VLLM_METAL_MLA_PREFILL_SDPA, VLLM_METAL_ASYNC_SCHED
  (scheduling mode); kernel-geometry knobs (bit-identical): VLLM_QC_Q8_NR,
  _Q8_NR_GEOM, _Q8_NR_MB_GEOM, VLLM_QC_KDA_RECUR_ROWS;
  VLLM_QC_MOE_MM_MIN_TOKENS (tile crossover). (3) Port STAGED (not
  committed) in `~/Code/QuixiCore-Metal` branch `glm53f-metal-campaign-port`
  off `dsv4-m1ultra-serving-port`: new `kernels/attention/{glm5_indexer,
  mla_sparse_latent}`, `kernels/linear_attention/kda`, `kernels/moe/
  moe_router`, `kernels/serving/kv_meta`; updated `common/tk_launch.h`,
  `moe/moe_mm_id/moe_mm_id.metal`, `quantization/qgemv/qgemv.metal`,
  `serving/dsv4_mhc/dsv4_mhc.metal`; metallib builds there with the README
  recipe (15.8 MB). Pre-existing drift NOT ported (older campaigns'
  files differ between SlimServe main and the port branch: paged_attn_v2,
  kv_cache.metal + kv_cache_fp8_strided + mamba_align, muse_step,
  rejection_sample, sampling/{flatten_sampled,logit_bias}). (4) Gates on
  the final build: GLM final chain (probe + tf + 3 gates + needles), DSV4
  anchor r3 (`perf/results/2026-09-12/dsv4-regate-r3/regate.sh`), Qwen
  NVFP4 anchor (`perf/results/2026-09-12/qwen-regate/regate.sh`, pins c1
  1000x256 467b35c3 / 2500x64 d0e07ddd) - results appended below.
- 2026-09-12: **FINAL GLM GATE (R=8 build): bit-identical to W13** - 8tok
  7dd30ea193a6, off1-2000 a88022b375ea, 2500x64 f80d14780458; tf 29/32 /
  62/64 identical; needles PASS x4. Prefill 2048 tokens 7.94/8.01 s =
  **256-258 t/s**, 2500 tokens 10.40/10.57 s = **237-241 t/s**, 1000
  tokens 4.08/4.39 s (`prefill_probe_final.log`).
- 2026-09-12: **DSV4 ANCHOR ROLLED on the final build -> bisected -> fixed
  -> bit-exact again.** r3: 8tok / 2500x64 bit-exact, off1-2000
  394993781527 vs pin bb83cc3054a3 (deterministic 3/3, tie-flip at ~token
  100). One-boot-per-switch bisect (`dsv4-regate-r3/regate_env.sh`): only
  `VLLM_METAL_ASYNC_SCHED=0` restores the pin; `VLLM_QC_Q8_NR=0` and
  `VLLM_METAL_KV_META=0` do not. Cause = the W8a platform-wide async
  scheduling default (DSpark verify path under the async step pipeline).
  The q8_0 NR GEMV, the W8b ledger's named suspect, is exonerated on DSV4
  (kernel A/B vs the generic walk: 1-ulp differences at batch 1-4 as
  documented, equal 0.181-ulp mean error against an fp64 reference;
  `ab_q8nr.py/.log`). Fix: `platforms/metal.py` async back to opt-in
  (`VLLM_METAL_ASYNC_SCHED=1`, pre-campaign contract) + the opt-in in the
  glm53f-q2-1 Metal env block (qwen38-nvfp4-1 already had it); test
  `test_metal_config.py::test_async_scheduling_is_opt_in`; stale profile
  speculator note corrected. r4 on the fixed tree: DSV4 ALL BIT-EXACT
  (573db39598e7 / bb83cc3054a3 2/2 / 73f41acf8ca0; 28th consecutive; sync
  scheduler in the boot log). The env-switch class line above changes:
  VLLM_METAL_ASYNC_SCHED is a per-profile opt-in, not a sentinel. Qwen
  anchor: HEAD's profile (vision enabled, b47509dd1) cannot boot in this
  venv (Qwen3VLVideoProcessor needs torchvision) - pre-existing at HEAD;
  gated with `--language-model-only` appended to the resolved argv
  (`qwen-regate/boot_qwen_lang.py`), the configuration of its pins.
  GLM post-fix gate (`glm-regate-asyncfix/`, async enabled via the
  profile): 7dd30ea193a6 / a88022b375ea 28.77 tok/s / f80d14780458 -
  bit-identical to the final pins. Qwen anchor: campaign c1 1000x256
  0f6fbb440708 / 2500x64 1b2541ff56ae (2/2 each) = upstream HEAD booted
  the same way (`qwen-regate/head_null/`, HEAD python + HEAD-built
  binary): the campaign is bit-identical to HEAD on Qwen; the 08-25 pins
  (467b35c3 / d0e07ddd) are stale at HEAD (HEAD-side 09-07..10 changes) -
  flagged for the Qwen owners, not re-pinned here. **Session 3 status:
  cleanup done, gates done (GLM final + post-fix, DSV4 28th consecutive
  bit-exact, Qwen = HEAD), PR draft filled
  (`perf/results/2026-09-12/PR_DESCRIPTION.md`), port staged uncommitted;
  nothing committed - commits on request.**
- 2026-09-12: **Scope check on the other Metal profiles.** The campaign's
  shared-code changes must leave every non-GLM profile as it was. Verified
  on the two anchored profiles: dsv4-xxs-1 bit-exact vs its pins (after the
  async scoping fix), qwen38-nvfp4-1 bit-identical to upstream HEAD. The
  remaining two Metal profiles on this box cannot be checked: muse-kdyn-1
  (Muse-Glimmer-30B Q4_K_XL, 18.3 GiB) and qwen38-q2kxl-1 (UD-Q2_K_XL,
  9.2 GiB; only the retired Q4_K_M file is local) have no weights on the
  machine and no recorded Metal pins. The campaign-vs-HEAD gate for them is
  written (`perf/results/2026-09-12/other-profiles/regate.sh`: boots the
  profile as `slimserve <profile> --serve`, c1 1000x256 + 2500x64 x2, on
  the campaign tree and on a HEAD checkout) and runs once the weights are
  fetched (`slimserve <profile> --download-only`). Superseded the same day
  by the static scope audit below: no boot is needed to know whether a
  change reaches a profile.
- 2026-09-12: **SCOPE AUDIT + SCOPING DONE.** Every campaign feature a
  non-GLM profile could execute is now opt-in through the glm53f-q2-1 env
  block (async, q8_0 NR GEMV, kv_meta, mamba_last_blocks, router kernel,
  MoE shared-expert fold; hetero shard fusion is a per-layer flag from the
  KDA projection), and the pre-campaign mHC monolith kernel is restored
  for the norm-free DSV4 call. Audit table: optimization_status
  2026-09-12 "scope audit". metallib + .so rebuilt; 214 unit tests pass;
  the staged port builds with the same kernel files. Re-gates on this
  build: DSV4 r5 (default env = its pre-campaign path throughout) ALL
  BIT-EXACT 573db39598e7 / bb83cc3054a3 2/2 / 73f41acf8ca0, 29th
  consecutive; GLM with its opt-ins 7dd30ea193a6 / a88022b375ea 28.78
  tok/s / f80d14780458, bit-identical to the final pins. **Session 3
  status: cleanup, scoping, gates and PR draft done; port staged
  uncommitted; nothing committed - commits on request.**
- 2026-09-14: **SESSION 4 (decode) RESEARCH + BUDGET DONE; plan in the
  roadmap ("Session 4").** ds4 PR #964 fetched (branch `pr964`, findings
  `scratchpad/ds4_pr964_findings.md`), llama.cpp master mul_mv kernels
  read, K=1 / K=2 phase profiles and the per-kernel microbench at every
  GLM decode shape taken (`perf/results/2026-09-13/glm53f-q2-w14-budget/`:
  `prof_run.sh` + `boot_glm_prof.py` (GLM_K / GLM_MNBT overrides),
  `k1/`, `k2/`, `kbench.py`, `kbench_q8nr.log`). Headline: the routed
  expert kernels run at 200-325 GB/s and scale linearly with tokens
  (11.5 ms/token at T=1, 19.4 at T=2 = 36% of the K=1 step); every large
  q8_0 GEMV is at 740-950 GB/s with the NR route; q4_K q|k at ~300;
  non-GEMV 14-16 ms. Waves W15-W19 ordered by payoff: experts, non-GEMV
  fusions, q4_K, MTP K=2, sampling/drafter. Nothing committed.
- 2026-09-14: **W15 EXPERT KERNELS: NOTHING PROMOTED - ALU FLOOR.** 25
  env-gated variants of the iq2_xxs pair-swiglu and q2_K sum kernels
  (`w15-experts/`, optimization_status 2026-09-14): the ALU-only twin runs
  at exactly the production time and the streaming twin at 550-770 GB/s,
  so the routed-expert cost is the ~5 ops/weight dequant, not memory;
  every exact rewrite (8-byte codebook loads, float/half/constant codebook
  tables, sign-mask tables or register masks, magic-number conversion,
  2-chunk ILP, slot-parallel q2_K, all geometries) lands within +-10%,
  half-precision math (inexact) +10%. W15 closed; the experiment kernels
  stay env-gated until the pre-PR cleanup strips them. Next: W16 GPU
  timeline (xctrace) for the 14-16 ms non-GEMV budget.
- 2026-09-14: **W24 - THE DRAFTER RAN WITHOUT ATTENTION; FIXED. off1-2000
  27.57 -> 29.46 tok/s (+6.9%), ds4's MTP bar beaten +30.8%, gate shas
  UNCHANGED.** K=2 first (W23, `w23-k2/`): 1.98 tok/cycle at ~77 ms,
  off1-2000 25.51 - the xctrace decomposition puts the 3rd verify row at
  +14-16 ms of expert bytes plus a 4.1 ms host gap before draft step 2,
  so K>1 cannot pay for itself on real text even with a perfect 2nd
  position (needs >= 2.66 tok/cycle); K=1 stays. Chasing the broken 2nd
  position anyway: an env-gated per-cycle drafter trace
  (`autoregressive/_glm_trace.py`, DIAGNOSTIC, delete before the PR) +
  an offline fp32 dense replay of blk.45 dequantized from the GGUF
  (`w23-k2/ref_k2b.py`, `score_log.py`) showed the server's drafter
  below its own dense reference at BOTH positions (pos0 0.69 vs 0.78,
  pos1 0.32 vs 0.65 with the server's recycled hidden), and a per-module
  probe (`ref_probe.py`) found the attention output EXACTLY ZERO on every
  extend/decode row (whole-prompt prefill rows: cos 1.000 - they take the
  SDPA path). Root cause: `eagle/utils.load_eagle_model` re-points the
  shared `topk_indices_buffer` / `topk_len_buffer` on nn.Modules only;
  the sparse-MLA attention impl (`layer.impl`) had captured the draft's
  private buffers in __init__ and kept reading them (never written).
  Fix: re-point the impls too (generic; CUDA/ROCm MTP drafts on sparse
  models inherit it - flag in the PR). Verify (`w24-fix/chain_k1.sh`,
  profile boot): 8tok 7dd30ea193a6 / off1-2000 8a9753e18551 / 2500x64
  1d7d58486dc7 unchanged, tf long 0.083, needles 4/4; accept set off1
  full 0.56 -> 0.648, m2 1000 0.75 -> 0.892, m2 2000 0.980; probe 35.25
  at 1.997 tok/cycle. Queued: K=2 post-fix sweep (`w24-fix/k2`) and the
  W24a shared||routed concurrent-region A/B (`VLLM_METAL_MOE_OVERLAP=1`,
  `w24-fix/ovl_k1`). See `optimization_status.md` 2026-09-14 W23/W24.
  FOLLOW-THROUGH (same day): W24a overlap KEPT (+1.2%, bit-identical);
  K=2 post-fix is a wash on the equal-prefix accept set and its 32.05
  gate is a loop artifact (the M=3 numerics roll the text at char 41 into
  a x28 repeated bullet) - K=1 stays; W24b indexer identity bypass
  (<= 2048 context: no pool logits, no torch top-k) KEPT, +3.5% on the
  fixed-text loop probe, tf + needles held, off1-2000 rolled (this
  prompt re-rolls on any decode-numerics change; its speed number is not
  the headline). FINAL BUILD (both flags in the profile env,
  `w24-fix/final_k1/`): loop probe 36.97 tok/s at 2.000 tok/cycle
  (session start 34.5 at 1.945), accept set 30-37 tok/s, **PINS NOW
  7dd30ea193a6 / 393882a2ddaf / 1d7d58486dc7**. W24c staged for the
  next boot window (rebuild needed): in-kernel identity expand (drops the
  3 torch ops per MLA layer) + dual q_a|kv_a RMSNorm (env
  VLLM_METAL_MLA_DUAL_NORM=1), exactness test `w24-fix/kernels_test.py`,
  verify `w24-fix/run_followups5.sh`.
  W24c RESULT: KEPT - kernels ALL EXACT standalone, all three gates
  identical to the pins in situ, loop probe 36.9 -> 37.3 tok/s, accept set
  +1%; VLLM_METAL_MLA_DUAL_NORM=1 added to the profile env (the in-kernel
  identity expand is automatic once the metallib has it). NEXT = W25
  texture-unit codebook LUT for the iq2_xxs expert kernel (see
  optimization_status.md 2026-09-14, "NEXT (W25)").
  W25 RESULT: the texture unit works as a lookup engine ONLY for tables
  inside its L1: the 64 KB joint signed LUT is 3x slower, the 4 KB
  magnitude table with one texel read per 8 weights (`texm`) is -21..26%
  vs production at T=1/2/4, BIT-EXACT (14 variants compared). Promoted
  behind VLLM_METAL_MOE_IQ2_TEX=1 (`qgemv_iq2_xxs_moe_mr_swiglu_texm`);
  same trick staged for the q2_K down sum (VLLM_METAL_MOE_Q2K_TEX=1,
  two 4 KB masked-field tables). In-situ verify `w25-tex/tex_k1/`.
  W25 IN SITU: KEPT - all three gate pins identical, needles 4/4, loop
  probe 37.3 -> 39.1 tok/s (+5%), accept set +4.6%; the q2_K twin LOSES
  (its and+convert walk is cheaper than the texel reads) and stays off.
  VLLM_METAL_MOE_IQ2_TEX=1 is in the profile env. Queued: W26 KDA in_proj
  shard GEMVs in one concurrent region (VLLM_METAL_SHARD_OVERLAP=1,
  gguf/linear.py) and W26b router kernel inside the MoE region
  (VLLM_METAL_MOE_ROUTER_OVERLAP=1, moe_runner.py + gguf/fused_moe.py),
  each a single-variable verify (`w26-shard/`).
  W26 KEPT (marginal: +0.3..0.5% on every probe, bit-identical); W26b KEPT
  (+1.4% probe, +2.3% m2 2000, off1-2000 gate 34.12 tok/s, bit-identical).
  Both flags are in the profile env. The W25 experiment twins are stripped
  (only `texm` half/bf16 + the 4 KB `iq2xxs_mag` table + the lut_texture
  registry remain); FINAL chain `w26-shard/run_final.sh`: rebuild,
  standalone texm exactness, combined profile verify -> `final_k1/`,
  traced sweep -> `final_trace/budget.txt`.
  FINAL RESULT (18:14): all pins held (7dd30ea193a6 / 393882a2ddaf /
  1d7d58486dc7), needles 4/4, tf 0.083; loop probe 39.96 tok/s at 2.000
  tok/cycle (session start 34.5 = +15.8%), accept set 32.45-39.92 tok/s,
  off1-2000 gate 34.23 tok/s = ds4's MTP bar (22.52) +52%.
  PROFILE ENV NOW (glm53f-q2-1, metal): V2 runner, async sched, Q8 NR, KV
  meta, mamba LB, MoE router, MoE fold, shexp pair, KDA gate dual, MoE
  overlap, indexer identity, MLA dual norm, MoE iq2 tex, shard overlap,
  MoE router overlap. K stays 1, max_num_batched_tokens 4288.
  STILL OWED BEFORE A PR: delete the DIAGNOSTIC hooks (`spec_decode/
  autoregressive/_glm_trace.py`, the `# DIAGNOSTIC` lines in
  `autoregressive/speculator.py`), the W15/W20 expert-kernel twins and
  their env hooks (VLLM_QC_MOE_IQ2_VARIANT / VLLM_QC_MOE_Q2K_VARIANT), the
  unused `qgemv_q8_0_nr_pair_swiglu_router` kernel/host op, the W16d q4_K
  twins; re-gate the DSV4/Qwen anchors (every campaign feature is
  env-gated to this profile, the eagle/utils.py top-k buffer share is
  generic - flag it for the CUDA/ROCm owners); then the final gate +
  needles on the cleaned build. Nothing is committed.
- 2026-09-14: **CORRECTION - ds4's 90% off1 acceptance is a LOOP
  ARTIFACT.** Reading ds4's `--mtp-timing` bar run per cycle
  (`glm53f-q2-bar/C_cli.err`): its greedy 2000-token output degenerates
  into one repeated paragraph after ~300 tokens (66 copies; the no-MTP
  arm is byte-identical, so it is ds4's greedy path), and acceptance is
  0.47 / 0.61 over cycles 0-100 / 100-200 then 0.96-1.00 in the loop. Our
  off1-2000 output never loops and our drafter accepts 0.48-0.56 on the
  same prompt = the same as ds4 on real text. The indexer cannot be the
  cause below ~2048 tokens (top-2048 covers every token; the 64+300
  probe at 0.48 had no pruning). Apples-to-apples on non-looping text:
  ds4 ~17.9 tok/s (308 tokens / 200 x 86 ms) vs ours 27.0 = +51%; against
  ds4's recorded bar (loop included) +23%. The "+8 tok/s in acceptance"
  projection in the next entry is WITHDRAWN. Remaining levers: deeper
  drafting (K=2/3 with the single nextn block applied recursively -
  `w23-k2/chain_k.sh`), the step itself, and the shared||routed overlap.
  See `optimization_status.md` 2026-09-14 CORRECTION.
- 2026-09-14: **THE GOAL IS ACCEPTANCE-BOUND, NOT STEP-BOUND.** The
  off1-2000 gate (27.7 tok/s) runs at the SAME ~54 ms step as the m2
  probe (34.5 tok/s); the difference is MTP acceptance - off1 ~0.56
  tok accepted per draft vs m2 0.92 (`w22-concurrent/accept_probe.py`,
  `verify_ldd/accept.log`). ds4 `--mtp` on the SAME off1 gets 90%
  acceptance (1.90 tok/cycle). We already beat ds4 +23% (27.7 vs 22.52)
  on a 54 ms step vs ds4's 84 despite accepting ~50% vs 90%. Match ds4's
  acceptance at our step -> ~35 tok/s = **+56% with no kernel change**
  (the user clarified 2026-09-14 that the goal is to exceed ds4 by as much
  as possible; +50% was only the DSV4 example); the expert-kernel step floor (W15/W20, ~20.8 ms MoE) caps the
  step-time path at ~28-30 tok/s on off1. Acceptance falls with context
  (m2 0.92@64 -> 0.75@1000), pointing at the MTP block's draft-time
  sparse-MLA + pooled indexer (top-2048 over the growing KV), not a
  uniform numerics bug. **NEXT (top priority): teacher-force off1, compare
  our MTP draft argmax vs the target argmax position-by-position against
  ds4's, isolate the draft-time indexer/sparse-MLA/head defect, fix.**
  (`optimization_status` acceptance entry.)
- 2026-09-14: **W21 (router-into-pair) DEAD-END; W22 = LDD strided
  in-place KDA in_proj writes LANDED (step 55.6 -> 53.8 ms), concurrent
  regions a null.** Root cause both fusions missed: GLM's
  `MoERunner.is_internal_router` is true (the runner owns the gate), so
  `DeepseekV2MoE.forward` takes the `if` branch and the router-into-pair
  call + MoE concurrent-region wrapper (both in the `else`) never ran -
  the earlier "+0.3% token-identical" was the unchanged torch-gate
  fallback. Both reverted. KEPT: `qgemv_q8_0_nr_mb`/`qgemv_q4k_nr_mb` now
  take an output row stride (`LDD`, buffer 5) so the KDA hetero shards
  (q|k q4_K + v|f|g|beta q8_0) write column slices of one output in place
  - 68 torch copies/step gone, BIT-EXACT (`strided_test.py`), gates
  unchanged (7dd30ea193a6 / 8a9753e18551 / 1d7d58486dc7), needles 4/4.
  The MTLDispatchTypeConcurrent mechanism (`qc_concurrent_begin/end`,
  auto buffer-conflict barriers, 2-D range tracker; llama.cpp
  `ggml-metal-ops` / ds4 `g_batch_encoder_concurrent` scheme) is built
  and standalone-verified but gives no in-situ gain (bandwidth-bound
  shards don't overlap); left in `qc_metal_serving.mm` for a future
  runner-side shared||routed overlap. Profile `VLLM_METAL_ROUTER_PAIR` /
  `VLLM_METAL_CONCURRENT` removed; the `..._router` kernel is on the
  pre-PR cleanup list.
- 2026-09-14: **W16 VERIFY CHAIN PASSED: tf 29/32 + 62/64, needles 3/3,
  off1-2000 gate 26.50 -> 27.46 tok/s (+3.6%), probe 34.15; pins
  re-pinned 7dd30ea193a6 / 8a9753e18551 / 1d7d58486dc7** (8tok unchanged;
  the two long shas move because the sparse-MLA partition merge order
  changes with the valid-range split - `optimization_status` 2026-09-14
  verify entry, raw `w16-fusions/verify/`).
- 2026-09-14: **W16a/b/c LANDED (in-situ +0.5% each, verify chain
  passed, see above); W19 DRAM-RESIDENT BENCH; W20 EXPERT LUT CLOSED WITH
  HARDWARE NUMBERS; W16d q4_K CLOSED.** (`optimization_status` 2026-09-14 entries.)
  W16a shexp pair-SwiGLU in situ: 46 qc_swiglu launches gone, step 56.0 ->
  55.8 ms, probe 33.21/33.35 -> 33.39/33.55 (bit-exact). W16b f_b|g_b
  dual q8_0 GEMV (`qgemv_dual`, bit-exact, `VLLM_METAL_KDA_GATE_DUAL=1`
  in the profile env): 34 launches gone, step 55.6, probe 33.47/33.62.
  W16c sparse MLA: the indexer's expand kernel now writes each row's
  valid prefix length (`topk_len_buffer`, shared with the drafter like the
  indices) and `mla_sparse_latent_decode` partitions only that range
  (`VLLM_METAL_MLA_TLEN=0` bisects): R=2 valid=100 123 -> 39 us
  standalone, full width unchanged (L2-bound at ~1 TB/s; a heads-per-
  simdgroup twin spills and is off). NUMERICS: the fp32 partition merge
  order changes for decode rows -> gate pins re-pin through tf + needles
  (chain `w16-fusions/verify/chain.sh`). W19 (`w19-dram/dbench2.log`):
  the W14/W17 per-kernel numbers were SLC-inflated 1.3-1.6x; DRAM rates:
  q8_0 NR 490-630 (M=1) / 440-600 (M=2), q4_K 475 / 373, lm_head 755,
  ~17 us per launch of in-situ overhead across ~300 dense launches. W20
  (`w20-lut/`): pair-LUT iq2_xxs kernels (threadgroup table 192/305/549
  us vs production 189/319/607 at T=1/2/4; register table via
  simd_shuffle 15x slower). Metal GPU Counters (`counters_run.sh`,
  `parse_counters.py`): production ALU limiter 75% / F32 16%; the LUT
  kernel saturates nothing (latency-bound at 16 simdgroups/core).
  Measured (`tgload_bench.py`): divergent threadgroup loads 12.6-13.9
  lane-loads per core-cycle, general simd_shuffle 3.2, vs 128 ALU lane-ops
  -> a lookup costs ~10 ALU ops, so every table formulation lands at the
  production time. The expert budget (20.8 ms/step) is the format's decode
  cost on this GPU; a cheaper repack costs >= 1.45x bytes (+24 GB, no room).
  W16d: q4_K M=2 column-pair kernel 101 us DRAM; geometry twins and a
  hoisted-decode twin 104-329 us; closed. MTP K=2 (W18) is bounded by the
  same per-token expert cost (verify M=3 adds ~10 ms/step for ~0.75 more
  tokens/cycle): deprioritized. Next: router F32 GEMV as a side branch of
  the pair-SwiGLU dispatch (43 launches), merged latent+indexer KV insert
  (12), KDA in_proj launch merge, then the torch glue ("Compute Command"
  ~26/step). Cleanup before PR: strip `_x/_ilp2/_sp/qgemv_q4k_nr_x/
  qgemv_q4k_nr_mb_x/lut*/lean*/rlut*/qc_tgload_bench` kernels and the
  VLLM_QC_MOE_IQ2_VARIANT / VLLM_QC_MOE_Q2K_VARIANT / VLLM_QC_Q4K_VARIANT /
  VLLM_QC_Q4K_MB_VARIANT / VLLM_QC_MLA_HPG hooks.
- 2026-09-14: **W16 GPU TIMELINE (xctrace Metal System Trace) - the real
  K=1 step budget.** `w16-timeline/trace_run.sh` boots the profile, traces
  the EngineCore for 6 s mid-decode (probe 33.2-33.35 tok/s at 1.945
  tok/cycle over 600 tokens, 30.2 over 64), `parse_intervals.py` fits the
  per-op GPU cost from the 11,568 encoder intervals (least squares over
  the op names each interval lists; residual 16 us on 540 us intervals;
  `k1/budget.txt`). 56.0 ms/step, GPU busy 95.2%, 110 steps in window:
  routed MoE (router+w13 swiglu 14.7, w2 sum 6.1) 20.8 ms; q8_0 NR GEMVs
  197/step 10.3 ms (52 us each); KDA q|k q4_K at M=2 (`qc_mmvq_loop`)
  35/step 6.9 ms (198 us each; the kernel alone measures 95 us at N=16384
  M=2); sparse MLA decode 12/step 2.9 ms (241 us each at a ~100-token
  context - fixed cost, kernel measures far less standalone); `qc_swiglu`
  46/step 2.25 ms; f_b|g_b K=128 GEMVs 68/step 2.2 ms (33 us each vs 12.5
  standalone); drafter singleton 2.1; lm_head+final norm+sampling 1.9;
  rms_norm 30/step 1.8; paged_row_insert 24/step 1.7; indexer 12/step
  0.9; kda_step 34/step 0.56 (16 us each); mHC pre 0.55; mamba_align
  2/step 0.4; torch ops ~1.0. Every small kernel costs 20-40 us more in
  situ than standalone: `encode()` opens and ends a NEW compute encoder
  per quixicore op (~600 encoders/step) and torch kicks ~105 command
  buffers/step (16.6 us median gap). Levers, in order: (1) coalesce our
  dispatches into torch's stream encoder instead of one encoder per op;
  (2) fusions that remove dispatches (shexp gate|up pair-swiglu -86,
  f_b|g_b dual -34, MLA q_a|kv_a + dual norms + merged inserts -46,
  router GEMV into topk -43); (3) sparse MLA decode fixed cost; (4) q4_K
  M=2 route. W17 (q4_K M=1 vector loads): no gain, kernel is at its
  compute floor like the experts (`w17-q4k/`).

- 2026-09-14 (late): **PRE-PR CLEANUP DONE, MAIN MERGED, PR #30 OPEN**
  (`perf/optimization_status.md` 2026-09-14 cleanup entry has the full
  ledger). Stripped: the W15/W17/W20 twins and their VLLM_QC_*_VARIANT
  hooks, the router-pair kernel, `qc_tgload_bench`, the MLA heads-per-group
  twin, `concurrent_enabled`, `_glm_trace.py`; two stale tests fixed.
  Cleaned build: kernels bit-exact, 362 tests. Four commits (Eric Hartford)
  on `glm53f-metal-campaign`, then main 02fb15fc1 merged (328035d39): the
  measured record replaces main's placeholder on its `glm53f-gguf` source
  (text modality until the Metal vision path is qualified, DFlash2 as the
  source drafter, glm47 parsers); the W26b router region stays closed when
  main's expert-stats histogram is on. Merged build: 365 tests, kernels
  bit-exact, verify chain `w26-shard/merged_k1/` - probe 40.02 tok/s at
  2.000 tok/cycle, accept set 32.50-40.00, tf 29/32 + 62/64, needles 4/4,
  ALL PINS HELD (7dd30ea193a6 / 393882a2ddaf 34.27 tok/s / 1d7d58486dc7).
  Pushed to QuixiAI/SlimServe `glm53f-metal-campaign`, PR #30 against main.
  NOT in the PR: the tracked metallib (this box builds metal3.1; the
  maintainer rebuilds with cmake/metal.cmake). STILL OPEN after the PR: the
  QuixiCore-Metal port (`~/Code/QuixiCore-Metal` branch
  `glm53f-metal-campaign-port` predates the strip - re-sync from these
  sources), vision on Metal for this model (source modality flips to
  text+image with it), the native 1048576 context (needs a ~12 GiB latent
  pool per request), an fp8 latent path for the Metal sparse-MLA kernels,
  the Qwen HEAD-side re-pin. Next step-time levers (~1% each): fuse the two
  paged_row_insert dispatches per MLA layer with the identity expand; the
  drafter / sampling torch glue (~1.3 ms); rms_norm into the following
  GEMV's staging. Harness note: the model boot pins ~100 GiB and the
  harness kills background task trees at that point - boot the server
  detached (Popen start_new_session, `scratchpad/boot_detached.py` pattern)
  and run the verify steps as foreground calls.

- 2026-09-15: **REVIEW ROUNDS ON PR #30** (CodeRabbit x2, Astra/codex
  GPT-6 x1; notebook 2026-09-15 entry has the full ledger). 9 + 5 findings
  fixed, all kept, gates bit-identical throughout. The two that mattered:
  the KDA in_proj shard concurrent region accepted batches its GEMV route
  cannot write in place (TORCH_CHECK -> engine death at 3 concurrent
  requests), and `kda_fused_prepare` wrote the conv history that chunk 0 of
  the same dispatch reads (a race on long prefills; the writeback for
  multi-chunk requests is now its own `kda_conv_history_write` dispatch).
  VERIFYING THOSE EXPOSED A THIRD, WORSE ONE: the drafter's torch prefill
  prep gathered the runner's [max_num_reqs, 1] buffers 2-D, so every
  multi-request step with the speculator raised a broadcast error - ALL
  concurrent serving was dead and no gate saw it (they are all c=1). Fixed;
  `perf/results/2026-09-14/concurrency_check.py` is now a standing check
  (c=2/3/6 all served: 27.8 / 33.7 / 41.2 tok/s aggregate). Also fixed:
  unbounded per-prefill-length KDA scratch rings (~20 GiB), the fp16 q8_0
  geometry fallback dispatching out of bounds, the indexer + MLA windowed
  scatters' unordered duplicate writes, a NaN-total router top-k, absent
  GGUF token ids, the KDA prefill state contract, and six test-hygiene
  items. Build/verify helpers now live in `perf/results/2026-09-14/`
  (`build.sh`, `boot_detached.py`, `concurrency_check.py`). NEXT: the
  concurrent-serving path has never been profiled - c=6 aggregate 41.2
  tok/s against c=1 40.0 says batching buys almost nothing yet, which is
  the obvious next campaign after this PR lands.

# HANDOFF — GLM-5.3-Flash Q2 on Metal: CONCURRENT SERVING (campaign plan 2026-09-15)

Read this with the campaign section above it (same profile, same worktree,
same discipline); that one took single-stream decode and prefill past ds4
and is on PR #30. This one finishes the job. The profile record is gated
`in-progress` until it does.

## Mission

Make `glm53f-q2-1` serve concurrent requests at a throughput worth having,
pin concurrent exact-token gates for it, then flip the record to
`supported`. Single-stream performance must not regress and the three
existing gate shas must stay bit-identical.

Operator judgement, 2026-09-15: a profile whose batching buys nothing is
not finished work, whatever its single-stream numbers say. This is also
the repo's own standard - CLAUDE.md puts concurrent workloads under
correctness requirements, and treats flat scaling as evidence of a wrong
design rather than a local bug.

## State on arrival

- Branch `glm53f-metal-campaign` in worktree `~/Code/slimserve/SlimServe-glm53f`,
  8 commits ahead of `origin/main`, open as PR #30 (QuixiAI/SlimServe),
  reviewed by CodeRabbit (2 rounds) and Astra/codex GPT-6 (1 round), every
  finding fixed. HEAD e975d30aa. Branch from here; rebase onto main when
  #30 lands.
- Single-stream, on the final build: loop probe 39.96 tok/s at 2.000
  tok/cycle, off1-2000 gate 34.28 tok/s, prefill 237-258 t/s, needles 4/4,
  teacher-forced vs ds4 29/32 and 62/64. ds4 on the same box: 22.52 tok/s
  decode with MTP, 143 t/s prefill.
- Gate pins (must hold): 8tok `7dd30ea193a6`, off1-2000 `393882a2ddaf`,
  2500x64 `1d7d58486dc7`. `perf/results/2026-09-11/glm53f-q2-baseline/gate.sh <outdir>`.
- Concurrency WORKS but does not SCALE. It was dead until 2026-09-15 (the
  drafter's torch prefill prep crashed every multi-request step); that fix
  is in e975d30aa with the ledger in `perf/optimization_status.md`.

## The measurement you inherit, and its caveat

`perf/results/2026-09-14/concurrency_check.py <N> <max_tokens>` against a
live server, from `perf/results/2026-09-14/glm53f-q2-w26-shard/astra1b_k1/concurrency.log`:

| concurrency | aggregate tok/s | per-request decode |
| --- | --- | --- |
| 1 (decode probe) | 40.0 | 40.0 |
| 2 | 27.8 | ~14 |
| 3 | 33.7 | ~11 |
| 6 | 41.2 | ~7.1 |

CAVEAT, read it before optimizing anything: that script measures completion
tokens over wall clock INCLUDING each request's prefill and the scheduler
ramp, over a short 96-token window, so the low-concurrency rows are
depressed by TTFT and are not comparable to the c=1 decode probe. Do not
quote them as a scaling curve. They are a smoke test.

What they do support, because it survives the caveat: at c=6 each request
decodes at ~7.1 tok/s where one alone gets 40, so the step time grew ~6x
for 6x the rows. With the K=1 MTP drafter every request contributes 2 rows
per step, so c=6 is 12 rows in ~291 ms against 2 rows in 48.3 ms. That is
linear. Essentially nothing amortizes across rows.

FIRST TASK: replace that smoke test with a real measurement before
touching a kernel. `benchmarks/benchmark_dsv4_exact.py` already takes
`--concurrency 1..128`, builds that many DISTINCT prompts at strided
offsets, returns a per-request sha list, and reports `exact` only when
every request returned the expected token counts. So the existing gate
protocol extends to concurrency directly - see `gate.sh` for the invocation
and copy it. Measure with output tokens long enough (>= 300) that prefill
stops dominating, and record per-request decode alongside aggregate.

## Where to look, in order

1. **The routed MoE expert kernels, almost certainly the whole story.**
   They dispatch one threadgroup row per (token, route) pair -
   `grid.y = tokens * topk` in `launch_qgemv_moe_mr_swiglu` and
   `launch_qgemv_moe_mr_q2k_sum` (`csrc/quixicore/metal/kernels/common/tk_launch.h`).
   Every row reads its experts' weights for itself, so expert bytes scale
   linearly with rows in flight and nothing is shared. At the last trace
   with per-kernel attribution (W16, `perf/results/2026-09-14/glm53f-q2-w16-timeline/k1/budget.txt`)
   routed MoE was 20.8 ms of a 56.0 ms step, the largest single block, and
   it is the block that cannot amortize in the current form.
   The grouping machinery already exists for the prefill path:
   `launch_moe_mm_map0` builds per-expert slot lists (`ids (E, tokens)`,
   counts `tpe (E)`) for the tile GEMM `ggml_moe_mm_id`. But that route is
   gated at `VLLM_QC_MOE_MM_MIN_TOKENS` (default 32,
   `vllm/model_executor/layers/quantization/gguf/fused_moe.py`), and a K=1
   step at the record's `max_num_seqs` of 8 is 16 rows - below the
   threshold, so decode never takes it. The lever is an expert-grouped
   decode route: sort rows by expert, read each expert's weights once for
   all its rows. Start from the GLM 5.2 Ampere path (route/align,
   expert-contiguous gather, grouped GEMM, grouped SwiGLU, finalize) that
   CLAUDE.md points at, rather than inventing one. Measure the crossover
   honestly: at small row counts the gather may cost more than it saves.
2. **The dense projection batch kernels' row caps.** The q8_0 NR batch
   kernel covers M 2..4 and the q4_K NR twin 2/4/8-row chunks; above those
   the host loops or chunks, losing weight-stationarity. `_metal_shard_region_ok`
   in `vllm/model_executor/layers/quantization/gguf/linear.py` documents
   the exact limits (it was written to keep the concurrent region off the
   routes that cannot write strided output). c=8 at K=1 is 16 rows, past
   every cap. Widening the instantiated row counts is mechanical work with
   a real payoff if the MoE stops dominating.
3. **The record's own sizing.** `max_num_seqs` 8 and
   `max_num_batched_tokens` 4288 were chosen for the single-stream
   bring-up (4288 is the Mamba align-mode block, and align mode requires
   the batch cap to be at least one block - do not lower it blindly). The
   8 GiB KV pool was sized for two full-context requests, not for many
   concurrent ones. Re-sizing is part of the campaign, with the memory
   arithmetic redone: weights are 89.9 GiB of a ~115 GiB working set.
4. **Sparse MLA decode and the pooled indexer at higher row counts.** Both
   were only ever measured at 1-2 rows.

## Profiling gotcha you will hit immediately

The final build runs 77 concurrent-dispatch regions per step, and the
xctrace encoder-level fit can no longer attribute cost to individual
kernels inside them - `final_trace/budget.txt` shows everything collapsed
into one `qc_concurrent` row at 159.9% share. To get per-kernel attribution
back, boot with the region flags off (`VLLM_METAL_MOE_OVERLAP=0`,
`VLLM_METAL_SHARD_OVERLAP=0`, `VLLM_METAL_MOE_ROUTER_OVERLAP=0`), profile,
then re-enable. The trace harness is
`perf/results/2026-09-14/glm53f-q2-w24-fix/chain_kp2.sh` with
`parse_intervals.py`.

## Gates and discipline

- The three single-stream gate shas must stay bit-identical, or roll only
  on a deliberate numerics change recorded with its teacher-forced and
  needle check, exactly as the campaign section above did it.
- The c=1 decode probe must not regress:
  `perf/results/2026-09-11/glm53f-q2-w9-mtp/decode_probe_spec.py 600`.
- Add concurrent exact-token gates (see FIRST TASK) and pin them. They are
  the artifact that lets the record flip to `supported`.
- Keep `concurrency_check.py` as the cheap smoke check after every build:
  it is what catches a crash like the drafter one.
- Every retained or rejected change gets a notebook entry in
  `perf/optimization_status.md` with baseline, hypothesis, correctness,
  throughput, decision and raw artifact path; pins go to
  `perf/baseline_status.md`; raw logs under `perf/results/YYYY-MM-DD/<run>/`.
- Other profiles: every campaign feature is opt-in through this profile's
  env block and defaults to off. Keep it that way, and answer "does this
  reach another profile" by reading the code path, never by booting.

## Definition of done

1. A concurrent exact-token gate set exists, passes, and is pinned.
2. Batching earns its keep. Proposed bar, adjust with the operator: at
   least 2x the c=1 aggregate at c=4 and 3x at c=8. Perfect scaling is not
   available - top-8 of 288 experts means concurrent rows touch mostly
   different experts - but linear step growth is a design problem, not a
   ceiling.
3. Single-stream gates and probe unchanged.
4. The record's `status` / `status_reason` / `status_detail` come off, its
   notes state the concurrent numbers, and the PR description carries them.

## Mechanics

- Python: `PYTHONPATH=<worktree> ~/Code/slimserve/SlimServe/.venv/bin/python`.
  Lint `uvx ruff check`. Shell is fish: quote globs, prefer script files
  over heredocs (zsh/fish glob errors abort whole commands).
- Build: `bash perf/results/2026-09-14/build.sh` (metallib at `-std=metal3.1`
  for this macOS 15 box, then the extension, ~40 s). NEVER rebuild while a
  server runs. NEVER commit `vllm/quixicore_metal.metallib`: upstream's is
  a metal4.0 binary carrying the M5 tensor-ops kernels this box cannot
  build, and the PR asks the maintainer to regenerate it.
- Boot: `python perf/results/2026-09-14/boot_detached.py <outdir>`, then
  poll `/health`. Boot takes ~3 min. It must be DETACHED: the agent harness
  kills background task trees once the model pins ~100 GiB, which will kill
  a server started as a background task. Run the verification steps as
  foreground calls instead, each under ~10 minutes.
- The user has standing authorization to kill any local server when the
  campaign needs the memory; no need to ask.
- Commits: Eric Hartford is the sole author, no co-author or assistance
  trailers. Commit only when the user asks.

# HANDOFF — NVFP4-on-Metal campaign (updated 2026-08-25; CAMPAIGN COMPLETE through UPDATE 55 — PR #12 open, origin/main merged and re-gated bit-exact, QuixiCore-Metal port landed)

## Mission

Serve **Qwen3.8-27B from `unsloth/Qwen3.8-27B-NVFP4`** (Unsloth Dynamic
v3.0, safetensors, compressed-tensors) **on Metal (M1 Ultra)**, then
optimize it with the same rigor as the Q4_K campaign. Directive from the
boss via the user: SlimServe is **opinionated — fine-tuned for specific
quants per platform**. NVFP4 is the quant of record for this model
(~98% of BF16 quality). Kernels/plumbing written portably-minded;
Metal first. BF16 is NOT a serving target.

Additional directive (user, 2026-08-19): **use DFlash and TurboQuant
where possible.** ~~Resolved as MTP because "no DFlash drafter exists
for Qwen3.8"~~ — **that claim was FALSE when written** (never checked
online; caught by the user 2026-08-19). **DFlash 2 shipped ~2026-08-13
with a Qwen3.8-27B drafter**: `z-lab/Qwen3.8-27B-DFlash2` (mirror
`incoai/…`, Apache 2.0), 2B BF16, one 3.85 GB safetensors, block_size 8
(7 draft tokens/verify), LOSSLESS, claims 3.43x GSM8K c1. Config:
`DFlash2DraftModel` — 5 sliding-window(2048) qwen3 layers, hidden 5120 /
vocab 248320 / GQA 32-8 (all match target), non-causal block attention,
mask_token_id 248070, taps target hidden states at layers
[5,19,33,47,61], selector rank 256 top_k 16, two-tap grouped dynamic
conv (group 16). In-tree we have the DFlash **V1** stack (DFlashProposer
runner, qwen3_dflash.py, GGUF adapter/tests); V2 needs the selector,
dynamic convs, per-position candidates + path tracing, and a
`DFlash2DraftModel` registry entry. References: upstream vLLM PR #52816
and github.com/z-lab/dflash. **N5 = DFlash 2 integration (promoted,
now the active milestone); MTP (`qwen3_5_mtp`) demoted to fallback.**
**TurboQuant** applies directly — full Metal kernel port exists — and
is milestone N6.

## N14 Muse single-CB — COMPLETE, DEFAULT FLIPPED (2026-08-24 evening, UPDATEs 50-53)

- **STATUS: N14 DONE. The qwen38-nvfp4-1 profile now carries
  VLLM_QC_MUSE=1 (UPDATE 53); plain-boot validated c1 17.045 sha
  467b35c3; the box serves the new canonical. NEW PINS: c1 17.10 sha
  467b35c3 BIT-EXACT with the pre-muse pin (154.4% of the Q4_K bar,
  +4.1%) / c4 23.12 / c8 25.59 (eager fallback) / 2500x64 3.45 sha
  aa448847 (the one re-pinned leg; 30k needle EXACT). c1 now sits AT
  the N3-era modeled bandwidth ceiling (~17 tok/s). Kill switch
  VLLM_QC_MUSE=0 = UPDATE 49 behavior.**
- Machinery: muse_q38_init/layer_gdn/layer_attn/run (qc_metal_serving.mm)
  emit all 64 layers + final norm + the drafter's 5 aux taps into ONE
  command buffer; glue kernels in muse_step.metal; python wire-in
  vllm/model_executor/models/muse_q38_metal.py (lazy registration,
  eligibility = uniform pure-spec decode m<=8, serve/shadow modes with
  exact state snapshot/restore, VLLM_QC_MUSE_LAYERS cap-list bisection,
  VLLM_QC_MUSE_DEBUG=<layer> stage-dump isolation).
- Root causes found during bringup (15 gate rounds, all in
  optimization_status UPDATEs 51-52): GDN state pools are SHARED across
  layers with PER-LAYER slot windows; the 16 attention layers span 4 KV
  GROUPS with per-group block tables — muse takes per-layer/per-group
  tensor vectors for both. Bit-exactness seeds eliminated by emitting
  eager's own kernels (qc_swiglu, gemma norms) and per-op-rounding
  mirrors (sigmoid gate).
- CAUTION for future reading: an intermediate broken build measured
  "+26%" — fake speed from 12/16 attention layers reading wrong tiny
  page sets. The honest muse win at c1 is +4.0% so far; the muse-mode
  phase census (next) decides where the remaining time is.
- DSV4 anchors bit-exact through 23 consecutive re-gates today.

## Parked: RadixArk NVFP4-BF16-LMHead checkpoint (user decision 2026-08-24: stay on unsloth)

- `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (SGLang org, pushed 2026-08-24)
  = NVFP4 MLP (group 16) + fp8 attn/GDN projections + **bf16 lm_head**
  (exists because SGLang can't load unsloth's fp8 lm_head). ModelOpt
  format, MIXED_PRECISION, per-tensor fp8 scales, W4A4 + fp8-static-KV
  recipe. Compatibility read (verified from its config.json): same arch
  (Qwen3_5ForConditionalGeneration 48+16), our NVFP4/fp8ch GEMVs apply
  as-is; gaps = ModelOpt->Metal kernel routing (fork's modelopt backend
  hard-wires Marlin), per-tensor->per-row fp8 scale broadcast, new
  profile + own pins (different weights). We'd serve W4A16/W8A16 + bf16
  KV (per the KV decision). bf16 head = +1.27 GB, ~2x lm_head pass
  bandwidth, better logits fidelity. Est. ~0.5-1.5 days bring-up.
- Sibling repo `RadixArk/Qwen3.8-27B-DSpark`: a DSpark drafter for this
  model — natural comparison vs DFlash2 if this checkpoint is ever
  brought up.
- DECISION: noted and parked; the campaign continues on
  `unsloth/Qwen3.8-27B-NVFP4`.

## Commit plan for UPDATEs 5-53 (user GO 2026-08-25; landed as this series
## after the pre-PR audit + cleanup pass, UPDATE 54)

63 dirty/untracked files on `qwen38-bringup` (last commit b1ee83671),
plus `tests/slimserve/test_profiles.py` in group 7 (the source-key rename)
and `config/vllm.py` moved to group 5 (DFlash2 scope). Series (each
buildable, library-first):

1. **qwen3.5/qwen3-next bring-up** — models/{qwen3_5,qwen3_next}.py,
   transformers_utils/configs/{qwen3_5,qwen3_5_moe,qwen3_next}.py,
   gguf_qwen35.py, gguf_adapters/{__init__,qwen3_5}.py, gguf_loader,
   gguf_config_parser, tokenizers/registry, models/{registry,config}.py,
   mamba_mixer2.py, gdn_mps_fallback.py, platforms/{metal,metal_compat},
   vllm/config/vllm.py, layernorm.py.
2. **Metal serving kernels + bindings** — all csrc/quixicore changes
   (qgemv, paged_attn_v2, gdn, rms_norm, kv_cache, turboquant,
   dflash_conv/, qk_norm_rope_gate/, muse_step.metal, tk_launch.h,
   qc_metal_serving.mm), vllm/quixicore_metal.metallib, quixicore/ops.py,
   muse_qwen38_design.md.
3. **CT Metal GEMV routing** — kernels/linear/{__init__,metal_dequant,
   nvfp4/metal,scaled_mm/metal}.py, compressed_tensors scheme edits,
   fp8_utils.py.
4. **GDN + attention + runner serving path** — qwen_gdn_linear_attn.py,
   v1 backends/{gdn_attn,metal_attn}.py, worker/gpu/{attn_utils,
   model_runner,model_states/*,sample/*}.py, metal_phaseprof.py.
5. **DFlash2 speculation** — qwen3_dflash2.py, spec_decode/dflash2/,
   spec_decode/{__init__,speculator}.py, qwen3_dflash.py.
6. **TurboQuant serving** — backends/turboquant_attn.py,
   ops/turboquant_native.py.
7. **Muse wire-in + profiles + registry** — models/muse_q38_metal.py,
   slimserve/{profiles.json,registry.py} (incl. the VLLM_QC_MUSE=1 flip).
8. **Perf notebook + docs** — perf/{optimization_status,baseline_status}.md,
   HANDOFF.md.

PR description must carry the "PR notes" section below (KV cache
intentionally unquantized). After commit: the QuixiCore-Metal twin-port
batch (kernel files byte-identical: turboquant incl. splitk/reduce,
dflash_conv, qk_norm_rope_gate, kv_cache block_mult, muse_step glue,
qgemv/gdn/paged_attn deltas).

## PR notes (must appear in the PR description at submission)

- **KV cache is intentionally NOT quantized in the canonical profile**
  (user decision, 2026-08-24). The TurboQuant k8v4 KV path
  (`qwen38-nvfp4-1-tq`) was fully built, debugged, and measured — it
  wins c4 +2.4% / c8 +1.8% / 2500x64 +1.4% with 2.4x KV capacity and
  a 262k needle-exact result — but its V-reconstruction floor
  (cos ~0.9955 vs bf16) was judged an unacceptable default quality
  trade. It ships as the opt-in long-context/capacity profile only.

## Repo state snapshot (2026-08-19; historical — see the N-milestone
## sections above for current status)

- **N2 + N3 COMPLETE (2026-08-19)**: exact-token N2 baseline pinned
  (UPDATE 14 + baseline_status snapshot: c1 7.02 sha 8c58a4c6 3/3,
  c4 14.37, c8 15.96, 2500x64 2.55 sha d0e07ddd 3/3), then the first
  CT GEMV kernels shipped (UPDATE 15): `qgemv_fp8ch` (461–650 GB/s at
  serving shapes) + `qgemv_nvfp4_planar` v6 (426/447 GB/s), routed
  M==1-only (batch loop measured BELOW dense GEMM at M=4/8 — `_mb`
  weight-stationary twins are the open c4/c8 lever). Serving: **c1
  10.10/10.14 sha 2e567ea7 (+44% over N2, 90% of Q4_K's 11.17)**,
  2500x64 2.82/2.83 sha 95adbd97, c4/c8 unchanged by design; ramp-32
  indicative 14.50. float64 oracles ALL PASS; kill-switch boot
  (VLLM_QC_FP8CH=0 + VLLM_QC_NVFP4=0) reproduces N2 BIT-EXACT.
  DSV4 anchors re-gated ALL BIT-EXACT after BOTH rebuilds
  (anchor_regate_fp8ch/ and anchor_regate_nvfp4_final/).
  **N3 mb twins SHIPPED same day (UPDATE 16)**: qgemv_fp8ch_mb +
  qgemv_nvfp4_planar_mb (column-pair grid-split; rows bit-identical
  to looped batch-1), routed even M (NVFP4 <=8, FP8 <=4 by measured
  crossover), kill switches VLLM_QC_NVFP4_MB/VLLM_QC_FP8CH_MB.
  **c4 15.64 (+8.8%), c8 16.89 (+5.4%)**; c1/2500x64 shas bit-exact;
  null-check boot reproduces UPDATE 15; kNvfp4Nib table removed;
  anchors re-gated again (anchor_regate_ct_mb/).
  **N3d fp8ch v6-decode port SHIPPED same day (UPDATE 18)**: select-
  free E4M3 half2 bit-pattern decode + 2^8 epilogue fold + vec4 X
  staging; fp8ch mb bound 4 -> 8 (crossover moved). **c1 10.35
  (+2.9%) / c4 16.28 (+4.1%) / c8 17.54 (+3.9%) — c8 PASSES the Q4_K
  bar (100.7%)**; 2500x64 + c4 shas HELD; **c1 canonical sha rolled
  2e567ea7 -> 467b35c3** (deterministic; summation-order rounding —
  fast-math reassociation the old select chain blocked; per-element
  exact, attribution in UPDATE 18); DSV4 anchors re-gated ALL
  BIT-EXACT (anchor_regate_v6port/).
  **N3e nvfp4 v7 vectorized decode SHIPPED same day (UPDATE 19)**:
  half2 constructs decode all 8 nibbles per uint + 2^22-folded
  select-free group scale; gate_up 631 / down 662 GB/s; **idealized
  all-GEMV ~26.1 ms = AT the ~27 ms campaign target — N3 GEMV work is
  CLOSED**. **c1 10.70 (95.8% of bar) / c4 16.98 (102.3% CROSSED) /
  c8 18.25 (104.8% CROSSED)**; 2500x64 2.91 sha 95adbd97 held
  bit-exact; **c1 canonical sha now fa58598b** (roll predicted in
  advance, reassociation class, UPDATE 19); DSV4 anchors re-gated
  ALL BIT-EXACT (anchor_regate_nvfp4v7/, 5th of the day).
  **N4 census + lever 1 SHIPPED same day (UPDATE 20)**: census =
  host exonerated, split mlp 33.5 / gdn 25.3 / full_attn 17.5 /
  glue 23.7; ROOT CAUSE found — head-256 full attn had NO paged
  kernel (SDPA gather every decode step). Split-K partition/reduce
  @256 shipped (one serial encoder, target 1536, P<=64,
  `max_context` op arg), crossover-routed (batch >= 2 OR ctx >=
  2048 -> kernel; batch-1 short ctx keeps SDPA — the kernel's fixed
  per-call cost loses there, measured across 3 gate rounds).
  **c1 10.72 sha fa58598b bit-exact (96.0% of bar) / c4 18.54
  (111.7% CROSSED) / c8 21.26 (122.1% CROSSED)**; 2500x64 2.914
  sha aedef4ec (kernel route); ctx-32k decode now viable (10.4
  ms/call x16). Kill switch VLLM_QC_PA256=0, tuning
  VLLM_QC_PA256_SPLITK. DSV4 anchors ALL BIT-EXACT
  (anchor_regate_pa256/, 6th of the day). The expanded
  DFlash-verify attention path (q_len>1) now routes to the kernel
  — N5 prerequisite done.
  **N4 lever 2 SHIPPED same day (UPDATE 21)**: fused add+RMSNorm
  (`add_rms_norm`, one dispatch per residual seam x128/step,
  BIT-EXACT by oracle AND by serving shas — c1 fa58598b / 2500x64
  aedef4ec both held): c4 18.87 (+1.5%, 113.4% of bar) / c8 21.43
  (+0.8%, 123.0%) / 2500x64 2.94 (+1.1%); c1 flat 10.71 (95.9%).
  SwiGLU hypothesis CLOSED — already fused (qc_swiglu). Kill switch
  VLLM_QC_ADDNORM. Anchors ALL BIT-EXACT (anchor_regate_addnorm/,
  7th of the day). OPS LESSON: gate scripts kill-first ALWAYS (a
  restored box + a clean-box-assuming gate = EADDRINUSE + a
  half-booted second EngineCore poisoning every batch run).
  **N4 lever 3 SHIPPED 2026-08-20 (UPDATE 22)**: GDN dispatch fusion —
  `gdn_fused_prepare` (conv+silu+q/k-norm+v+gate in ONE dispatch,
  reading qkvz/ba projection rows in place; pure-decode routed) +
  `gdn_gated_rmsnorm_f32` (norm off the fp32 recurrence output, z in
  place) + container restructure: 12 -> 3 non-GEMV dispatches per GDN
  layer (~430 fewer/step). Oracle 20/20 BIT-EXACT incl. conv state
  pools; serving shas HELD (c1 fa58598b / 2500x64 aedef4ec):
  **c1 11.02/10.97 (+2.7%, 98.4% of bar) / c4 19.25 (+2.0%, 115.7%)
  / c8 21.68 (+1.2%, 124.5%) / 2500x64 2.97 (+0.9%)**. Kill switches
  VLLM_QC_GDN_FUSEPREP / VLLM_QC_GDN_FUSENORM. Anchors ALL BIT-EXACT
  (anchor_regate_gdnfuse/, 8th consecutive). CALIBRATION LESSON:
  ~2.4 ms actual vs ~13 ms census-share estimate — marginal dispatch
  cost at batch 1 is ~5 us (pipeline overlap); sync-prof shares are
  upper bounds for dispatch-elimination levers.
  **N4 CLOSED / base-decode CEILING DECLARED (UPDATE 23, 2026-08-20)**:
  xctrace Metal System Trace attach on the live box (no rebuild)
  showed ~240 CBs/step (torch-MPS per-op commits, commitAndContinue
  already default; qc encode() exonerated), channel 81% busy, and 73%
  of channel time in 1-3 ms CB-granularity footprints whose encoder
  payloads are ~10x smaller — the step is CB-scheduling granularity +
  the 26 ms GEMV floor, not unfused kernels. No single-kernel lever
  remains; the only sizeable lever is the Muse whole-step single-CB
  loop (multi-day, re-ranks under speculation). Campaign totals vs N2:
  c1 +57% / c4 +34% / c8 +36%.
  **N5 DFlash2 INTEGRATION COMPLETE (UPDATEs 24-25, 2026-08-20)**: the
  full pipeline is live on Metal (drafter -> selector walk -> GDN spec
  verify with per-position state checkpointing -> rejection), oracle
  24/24 bit-exact, greedy-DETERMINISTIC serving (b46e676c 2/2), and the
  no-spec/spec divergence was proven to be an EXACT LOGPROB TIE flip
  (' Use' vs ' Report' both -1.538660) — greedy-correct, not a bug.
  Acceptance 2.93 tokens/step (essay prose, k=7). BUT spec is currently
  SLOWER (c1 9.59 vs 11.02) because the verify target forward at M=8
  costs 234 ms/step — the UPDATE 11 weight-stationary batch-GEMV
  collapse (~111 GB/s at M=8 vs 533-633 at M=1), now inherited by the
  NVFP4/fp8ch kernels, plus dequant+dense fallback at c4/c8 (M>8).
  Canonical profile stays no-spec; the twin `qwen38-nvfp4-1-df2` carries
  spec.
  **N5b COMPLETE (UPDATE 26, 2026-08-20): mv_ext batch GEMVs**
  (`qgemv_fp8ch_mv4r` R1=4 / `qgemv_nvfp4_mv4r` R1=2 — NR=4 rows/SG,
  X via L1-served device loads, the llama.cpp kernel_mul_mv_ext
  precedent) route batches 3..8 incl. odd M (kill switches
  VLLM_QC_FP8CH_MV4R / VLLM_QC_NVFP4_MV4R; batch 2 keeps mb, M>8
  dense). Oracle: batch rows BIT-IDENTICAL to looped batch-1. Gate:
  **no-spec c1 11.06 fa58598b BIT-EXACT / c4 20.68 (+7.4%, 124.3% of
  bar) / c8 23.74 (+9.5%, 136.3%) / 2500x64 2.98 aedef4ec BIT-EXACT;
  SPEC c1 11.11 b46e676c HELD (+15.9%) — spec BEATS no-spec at c1 for
  the first time**; spec 2500x64 3.154 (+5.9% vs no-spec); spec c4/c8
  still dense-bound (spec = c1 feature). Anchors ALL BIT-EXACT (10th).
  KEY RECALIBRATION: M=8 batch work is FMA-ISSUE-bound on M1
  (~2.3-3.0e12 FMA/s; simdgroup_matrix is lane math, not tensor
  cores) — the UPDATE 25 ~21 tok/s projection was wrong; kernel
  headroom at M=8 was ~1.3-1.7x and is now spent.
  **k-SWEEP DONE (UPDATE 27, 2026-08-20): k=3 WINS — spec c1 13.65
  (sha 8c58a4c6 2/2) = +23% over both k=7 and no-spec; 2500x64 3.239.**
  Mean acceptance ~3.0 REGARDLESS of k (drafter horizon ~3 tokens on
  prose) — larger k is pure wasted verify FMAs. Twin profile pinned to
  k=3.
  **ADAPTIVE SPEC PROMOTED (UPDATE 28, 2026-08-21): the canonical
  profile `qwen38-nvfp4-1` is now speculative:true with
  num_speculative_tokens_per_batch_size=[[1,1,3],[2,8,0]]** — the
  fork's dynamic-SD feature (UPDATE 27's "no disable-by-batch-size" was
  WRONG) wired into the V2 runner: sample_tokens skips the drafter
  forward when the scheduler's per-step K is 0 and reports zero-width
  drafts; plus a gdn_attn.py:253 fix (`block_table[:, 0].contiguous()`
  — the strided column view of the [batch, k+1] spec mamba block table
  crashed gdn_short_conv at batch >= 2 with zero drafts). Promoted
  gate: **c1 13.651/13.62 sha 8c58a4c6 2/2 BIT-EXACT vs the k=3 twin
  (+23.4%, 122.2% of the Q4_K bar — c1 clears the bar for the first
  time) / c4 20.104 (97.2% of no-spec) / c8 23.275/23.26 (98.0%) /
  2500x64 3.229 f497f4a9**; anchors bit-exact 11th consecutive. NO-SPEC
  kernel anchors (fa58598b / aedef4ec) reproduce via speculative:false.
  Concurrency first-shas are composition-timeline dependent under
  adaptive spec — c4/c8 are TPS gates, not sha gates. Twin -df2 now
  identical to canonical (kept as historical gate id).
  **MUSE SCOPING + GEMMA NORMS (UPDATEs 29-30, 2026-08-21): the c1-spec
  step is CPU-DISPATCH-BOUND** (~220 ms/step = ~100 GPU + ~120 host;
  xctrace 46% busy; op census 6831 torch dispatches/step; all 101
  .item()/step are CPU-cheap). Biggest glue block fixed: GemmaRMSNorm
  (128 target norms/step, torch-native since bring-up) now routes to
  gemma_rms_norm{,_add}_dyn Metal kernels (exact ir semantics; oracle:
  res_out bitwise exact, out 99.98-100%). Gate: **c1 13.90/13.91
  8c58a4c6 HELD / c4 20.61/20.66 / c8 23.844/23.843 (ABOVE old no-spec
  23.74) / 2500x64 3.248 sha ROLLED f497f4a9 -> d0e07ddd (predicted ulp
  class, 2/2)**; anchors bit-exact 12th. CALIBRATION: removing 19% of
  dispatches bought ~2% — host cost is NOT aten-count-proportional;
  fusion-by-fusion has poor marginal returns.
  **SYNC FIX + ASYNC SCHED (UPDATEs 31-32, 2026-08-21): cProfile census
  (VLLM_QC_PYPROF=1 in metal_phaseprof.py) found postprocess_state's
  boolean-mask MPS scatter draining the WHOLE GPU queue every spec step
  (67 ms/call self-time; invisible to sync-bracketed censuses; explains
  the 46%-vs-81% busy split — no-spec takes the int index_fill_
  branch). Fix = trailing dump slot + torch.where scatter (sync-free,
  identical values): +3.3% c1, ALL shas bit-exact. Reading: sync
  scheduling re-books the wait at end-of-step => async re-ranked; dflash
  added to the config async whitelist (vllm.py) + VLLM_METAL_ASYNC_
  SCHED=1 promoted into BOTH qwen profile envs: c4 +2.3% / c8 +2.1% /
  c1 flat / 2500x64 -0.7% accepted, ALL shas bit-exact. CURRENT
  CANONICAL: c1 14.27 8c58a4c6 (127.8% of bar) / c4 21.45 / c8 24.52
  (140.8%) / 2500x64 3.30 d0e07ddd. Session cumulative vs no-spec
  start: c1 +29%.**
  **SYNC-HUNT CLOSED (UPDATEs 33-35, 2026-08-21): census rounds 3-4
  named the relocating wait each time.** (33) metal_attn: eager
  seq_lens D2H at build -> lazy property + bound-mode draft SDPA (GPU
  visibility mask from seq_lens_gpu; one-block back-off): 2500x64
  +1.7%, c1 sha ROLLED 8c58a4c6 -> 467b35c3 (predicted ulp class,
  deterministic 2/2). (34) gdn_attn spec branch: 5x GPU[cpu-bool-mask]
  -> CPU nonzero + async_tensor_h2d + index_select: **c1 +3.2% to
  14.788/14.801 BIT-EXACT**. (35) gdn_attn ~335 repeat_interleave (GPU
  repeats tensor, data-dependent even with output_size) -> scatter_add
  segment-id + index_select (CPU oracle incl. zero-length requests;
  plain seg[starts]=1 is WRONG there): gated **NEUTRAL** (c1
  14.713/14.710 467b35c3 bit-exact, all legs flat) — KEPT per the
  pre-set rule, and the hunt is **PARKED**: third consecutive
  relocation; under async each removed drain banks only its overlap
  headroom, the LAST drain absorbs the queue tail. **FINAL CANONICAL:
  c1 14.80 467b35c3 (132.5% of bar) / c4 21.61 / c8 24.65 (141.5%) /
  2500x64 3.378 d0e07ddd. Session vs no-spec start: c1 +33.8%.**
  Remaining parked sync sites: gdn_attn has_initial_state
  (prefill-only), metal_attn target-prefill lazy materialization.
  **N6 gate 1 PASSED 2026-08-21 (UPDATE 36; see the N6 section below):
  `qwen38-nvfp4-1-tq` quality-gated after a six-run debug arc that
  fixed a latent GQA head-mapping bug in tq_attention_combined. TPS
  tax 6-13%, KV 2.4x smaller; canonical default unchanged.
  N8 DONE (UPDATEs 38-39; see the N8 section below): Metal TQ dequant
  continuation route (tq_decode_combined) unlocked long context —
  262k needle EXACT (75 min prefill), 2500x64 gap -3.0% at the
  canonical sha, c1 pin bit-exact, anchors 14th consecutive.
  #16b DONE (UPDATE 40; see the #16b section below): split-K TQ
  decode attention (tq_attention_splitk/reduce) — kernel 12-33x, -tq
  c1 tax -11.6% -> -0.8%, and the -tq profile now BEATS canonical at
  c4 (+2.2%) / c8 (+2.0%) / 2500x64 (+2.1%); anchors 15th
  consecutive; new pins c1 228d0bf4 / tail-2100x32 1337d2f7, 2500x64
  still the canonical d0e07ddd. FLAGGED DECISION: canonical-default
  flip to -tq (quality call — k8v4 cos ~0.9955 floor vs bf16).
  fp8-KV comparison DEPRIORITIZED: its motivation was the TQ decode
  tax, which is now -0.8% with 2.4x capacity — fp8 (2x capacity, new
  kernel surface) is dominated.
  N9 DONE (UPDATEs 41-42; see the N9 section below): the mrope
  repeat_interleave queue-drain (65.8 ms/step, 64% of the host
  profile) fixed with a static-shape token->request map — canonical
  c1 14.80 -> 16.139 sha 467b35c3 BIT-EXACT (144.5% of the Q4_K
  bar) / c4 23.16 / c8 25.62 / 2500x64 3.454 d0e07ddd; -tq c1 16.08
  sha 228d0bf4 bit-exact (-0.3%). OPS: tmp cleaner destroyed the
  harness assets mid-gate; recovered byte-exact into
  perf/results/harness_assets/ (USE THAT PATH).
  N10 DONE (UPDATE 43; see the N10 section below): steady-cache
  metadata for the mamba-hybrid path — GDN builder rebuild (8.2
  ms/step) skipped on steady uniform all-spec decode steps
  (signature + 2-copy steady_decode_update; VLLM_QC_STEADY_META=0
  null proven = UPDATE 41). Canonical c1 16.139 -> 16.334/16.346 sha
  467b35c3 BIT-EXACT (+1.2%, 146.3% of the Q4_K bar); 2500x64
  d0e07ddd + -tq c1 228d0bf4 bit-exact (+0.8%); c4/c8 noise-flat;
  both needles exact. Two engine-killing bugs found on the way (see
  section): upstream eligibility deref'd is_prefilling=None, and the
  runner's per-step-fresh seq_lens_cpu_upper_bound froze in the
  cached cm — deterministic sha roll from silently truncated decode
  attention.
  N11a DONE (UPDATE 44; see the N11a section below): fused DFlash2
  grouped-conv kernel (qc_dflash_conv, ~200 drafter encodes/step
  collapsed to 20; kernel parity 37/37 BIT-EXACT vs eager so drafts
  are bit-identical). Canonical c1 16.334 -> 16.428/16.438 sha
  467b35c3 BIT-EXACT (+0.6%, 147.1% of the Q4_K bar); 2500x64
  d0e07ddd bit-exact; c4 23.11 / c8 25.62 noise-flat; DSV4 anchors
  bit-exact 17th consecutive after the metallib+.so rebuild;
  VLLM_QC_DFLASH_CONV=0 null proven = UPDATE 43. (Note: drafter_propose
  nests inside the sample_tokens bracket — 32.7 = 24.4 drafter + 6.5
  reject + glue; no unattributed sampling cost.)
  N11b DONE (UPDATE 45): pure-prefill causal SDPA reads the CPU bound
  (row-exact for all-prefill batches) instead of the queue-draining
  seq_lens D2H (census: 3.43 s / 12 calls, multi-chunk-prefill
  concentrated). VLLM_QC_SDPA_PREFILL_BOUND=0 null proven = U44; DSV4
  8tok + c1 + 2500x64 pins all bit-exact; leg TPS unchanged (+0.3%
  2500x64 — the +2-7% prediction missed, single-chunk prompt); the
  win books as long-context TTFT (multi-chunk prefills lose a
  hundreds-of-ms host stall per chunk).
  UPDATE 46 (-tq re-measure, all legs, U43-U45 stack): c1 16.18-16.24
  sha 228d0bf4 BIT-EXACT (-1.2% vs canonical) / 2500x64 3.51
  d0e07ddd (+1.4%) / c4 23.6-23.7 (+2.4%) / c8 26.05-26.09 (+1.8%)
  / needle exact — -tq beats canonical everywhere above c1 with 2.4x
  KV; flip decision data fully current (quality call). NEXT candidates:
  N11c TRIED AND REJECTED (UPDATE 47): drafter block attention through
  the expanded paged kernel lost on every leg (c1 -7.7%, acceptance
  1.59 -> 1.54 accepted/draft + kernel slower than one-request SDPA
  at draft shapes; 2500x64 forked trajectory). Quarantined opt-in
  VLLM_QC_DRAFT_BLOCK_PA=1. KEY LESSON: drafter math is NOT
  reduction-order-tolerant — the 16-way candidate selector amplifies
  ULP noise into acceptance loss; only bit-exact or measured-neutral
  drafter changes are safe.
  N12 TRIED AND REJECTED/PARKED (UPDATE 48): fused qk-norm-rope-gate
  kernel (qc_qk_norm_rope_gate, 1 dispatch per attn layer, bf16
  parity 8/8 at small T) forked the canonical trajectory at prefill —
  torch-MPS eager numerics are SIZE-DEPENDENT (~5 ppm single-ulp at
  T=1000, exact at decode shapes); win was c4/c8 ~+1%, not worth the
  full re-pin. Opt-in VLLM_QC_QKROPE=1; anchors bit-exact 18th.
  KEY LESSON: validate parity at SERVING shapes incl. prefill T.
  N13 DONE (UPDATE 49): kv_cache_scatter bound + routed (block_mult=2
  page-local; 5 ops -> 1 per attn layer, ~80 encodes/step) — all pins
  + anchors (19th) bit-exact, TPS flat within noise, retained as
  dispatch hygiene. MUSE SCOPING (this session): the muse_step
  machinery (muse_glimmer.py + muse_step_init/layer/run — whole-model
  single-encoder emit loop) exists for the DENSE Muse-Glimmer arch;
  adapting to qwen38 = emit variants for qgemv fp8ch/nvfp4 (mb/mv4r),
  gdn_fused_prepare/recur_spec/norm, PA256/splitk, residual adds —
  target-forward-only scope (sampling/drafter stay eager), est. 2-4
  focused days, AND it re-pins every sha by construction — a
  trajectory-re-pin decision FLAGGED FOR THE USER (same class as the
  -tq default flip). Encode-crumb tier is now exhausted: remaining
  levers are Muse (user-gated), drafter linear batching (thin), and
  the parked-dangerous items.
  DECISION RESOLVED (2026-08-24, user): -tq default flip REJECTED —
  the KV cache stays unquantized (bf16) for canonical. Quality call:
  the k8v4 V-reconstruction floor (cos ~0.9955) is not acceptable as
  the default, and the c4/c8/longctx wins don't override it. The -tq
  profile REMAINS registered as the opt-in long-context/capacity
  profile. **PR NOTE (user directive): this decision + rationale MUST
  appear in the PR description when UPDATEs 5-49 are submitted** —
  see "PR notes" below.
  DECISION RESOLVED (2026-08-24, user): Muse single-CB = **GO**. The
  user accepted the trajectory re-pin ("we've done it before" — the
  muse_glimmer precedent). Plan: build opt-in-gated so canonical
  stays bit-exact during bringup; full quality revalidation (needle
  262k + long decodes + acceptance before/after) then NEW pins at
  flip time. Commit of UPDATEs 5-49 still awaits the user's word and
  can land at any point before the flip.
  NEXT: N14 Muse single-CB bringup (ACTIVE);
  commit of UPDATEs 5-49 (user-gated). Mechanical fallback:
  drafter linear batching (~58 unquantized MPSGraph linears/step);
  gdn NSG-multi-row (demoted — dispatch amortization, small);
  twin-port batch to QuixiCore-Metal (commit-gated, now incl.
  tq_attention_splitk/reduce + dflash_conv). Parked stacks: mrope torch-native
  (16 full-attn layers), metal_attn index_put_ KV updates, ~80 MPSGraph
  unquantized linears (drafter + candidates), Muse single-CB step loop
  (re-rank post-N6), uniform-decode classification for zero-draft
  batches.
  OPS LESSON (UPDATE 15): a TERM'd server can hang in shutdown holding
  ~101 GB wired while ps-based checks false-negative — verify kills BY
  PID and check `vm_stat` wired (~2–3 GB idle) before any measurement.
- **N1 COMPLETE (2026-08-19)**: `qwen38-nvfp4-1` serves on :8000 (user
  had the Q4_K server taken down first — one server at a time on this
  box). Full record: perf/optimization_status.md **UPDATE 13**; raw
  gates in `perf/results/2026-08-19/n1_nvfp4_bringup/`. Gates: helper
  dequant bit-exact (CPU+MPS, full E4M3 range); Gate A structural (11
  asserts incl. **lm_head = W8A16Fp8**, silent-bf16 trap did not fire)
  + 6 load-path weight comparisons ALL BIT-EXACT vs CPU oracle of raw
  checkpoint bytes; Gate B live-server ramp + 3340-tok needle exact +
  greedy probes. Indicative (NOT harness): c1 decode **8.83 tok/s** on
  the bring-up path (dequant-once to bf16 at load, ~40 GB resident,
  F.linear apply; `VLLM_METAL_CT_DEQUANT=call` = per-call low-mem
  fallback). eos resolved: generation_config = [248046, 248044], chat
  stops on 248046 like the GGUF campaign.
- Three NEW blockers found at N1 (beyond the N0 list, all fixed):
  fork-wide `KernelConfig.linear/moe_backend="aiter"` default leaks to
  every platform (Metal check_and_update_config resets to auto);
  `RopeState.prepare_positions` is Triton and this config has
  uses_mrope=True (torch replacement in metal_compat); checkpoint
  `kv_cache_scheme` fp8 + auto flips KV to fp8 (attention.py:299) with
  no Metal dense/GQA fp8-KV path (profile pins kv_cache_dtype=bfloat16
  explicitly; fp8/TurboQuant KV is N6's call).
- `qwen38-1` (Q4_K_M GGUF) profile intact, not serving. Fresh-boot
  sanity was c1 11.195 tok/s sha 36ed113a. Stays the fallback Metal
  path until the decommission gate (**N7**); restore via
  `restore_qwen_m4c.sh` (session-a99b scratchpad).
- Branch `qwen38-bringup`. 39 uncommitted entries in SlimServe (all
  N1–N3 edits) + 3 in ~/Code/QuixiCore-Metal (byte-identical kernel
  twins; the N3 fp8ch/nvfp4 kernels still need their QuixiCore-Metal
  twin port once gated).
- Q4_K campaign records: `perf/optimization_status.md` UPDATEs 1–12,
  `perf/baseline_status.md` tail (M4b + M4c). Canonical Q4_K shas:
  c1 36ed113a, 2500x64 268721b3, ramp-32 0f9506fc. DSV4 anchors
  re-gated bit-exact twice on 2026-08-18
  (latest: `perf/results/2026-08-18/anchor_regate_q4k_mb/`).
- Q4_K decode budget (M4b census): c1 step 65.5 ms = 35.1 ms idealized
  GEMV + ~31 ms non-GEMV. Box empirical bandwidth roofline ~700–740
  GB/s; best Qwen kernel 633 GB/s (q4_K gate_up M=1).

## N0 discovery results (verified 2026-08-18/19)

Raw artifacts (`n0_checkpoint_findings.md`, `nvfp4_config.json`,
`nvfp4_index.json`, `nvfp4_shapes.json`) lived in a session scratchpad
that is not preserved; the durable findings are all inlined below.

### Checkpoint anatomy — MIXED FP8 + NVFP4 (the headline)

- HF sha 7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108. 23.44 GB total:
  `model.safetensors` **22.57 GB single file** (loader must not assume
  shards) + `model_mtp.safetensors` 849 MB. Arch
  `Qwen3_5ForConditionalGeneration` (VL; vision tower bf16),
  compressed-tensors `format: mixed-precision`, two config groups.
- **group_1 NVFP4** (W4A4 g16 in checkpoint; we run W4A16):
  `mlp.{gate,up,down}_proj` **layers 0–55 only**. Tensors:
  `weight_packed` U8 [N, K/2]; `weight_scale` F8_E4M3 [N, K/16]
  (gate/up [17408,320], down [5120,1088]); `weight_global_scale` F32 [1]
  — **NOTE the name: not `weight_scale_2`**; `input_global_scale` F32
  (ignored on Metal).
- **group_0 FP8 E4M3 per-channel** (W8A8-dynamic in checkpoint; we run
  W8A16): all 16 `self_attn` q[12288,5120] (attn_output_gate doubles q)
  / k[1024,5120] / v[1024,5120] / o[5120,6144]; all 48 `linear_attn`
  in_proj_qkv[10240,5120] / in_proj_z[6144,5120] / out_proj[5120,6144];
  **lm_head [248320,5120]**; `mlp` layers 56–63. Format: `weight`
  F8_E4M3 [N,K] + `weight_scale` **BF16 [N,1]**.
- **bf16 (ignore list)**: embed_tokens, vision tower, GDN internals
  (conv1d, A_log, dt_bias, in_proj_a/b [48,5120], norms), all
  layernorms, and the **entire `mtp.*`** (separate file).
- `kv_cache_scheme`: fp8 static per-tensor; `k_scale`/`v_scale` BF16
  scalars per attn layer — feeds N6.
- Config quirks: eos **248044** per config.json (previous handoff said
  248046 — recheck at N1); mrope_interleaved, partial_rotary 0.25,
  full_attention_interval 4; 64 layers = 48 GDN + 16 attn confirmed.
- **Decode-token read budget**: 56 NVFP4 MLP layers 8.42 GB + 8 FP8 MLP
  2.14 + 16 attn 1.68 + 48 GDN 5.54 + lm_head 1.27 ≈ **19.1 GB/tok**
  (**FP8 side 10.6 GB > NVFP4 side 8.4 GB**). Q4_K reads ~16.5 GB/tok
  → NVFP4 c1 bandwidth ceiling is BELOW Q4_K's. At the ~700 GB/s box
  roofline: GEMV floor ~27 ms + today's ~31 ms non-GEMV ⇒ ~17 tok/s
  hard cap. **Beating Q4_K at c1 requires #16 (non-GEMV)**; NVFP4's
  format wins live in batch/prefill ALU (cheap decode) and MTP.
- lm_head risk RESOLVED: FP8, 1.27 GB/tok ≈ today's Q6_K 1.04 GB.

### In-tree assets (better than the prior handoff knew)

- metallib ALREADY ships: `nvfp4` format struct
  (`dequant.metal:431` — interleaved 9-byte {scale, qs[8]}; the
  checkpoint is PLANAR → layout mismatch, do not use as-is),
  `qgemv_nvfp4`(+bf16/moe) pipelines (`qgemv.metal:1622`),
  `qgemm_nvfp4` / `qgemm_frag_nvfp4` (`qgemm.metal:71,223`), and an
  nvfp4 lm_head sampler family that is **DEAD CODE** (zero host
  bindings anywhere — do not count on it).
- `fp8_raw` struct (`dequant.metal:517`) is **byte-identical** to the
  checkpoint's planar [N,K] e4m3 weight. `tk_e4m3_decode`
  (`dequant.metal:271`): 3-op arithmetic decode, subnormal-exact under
  offline-compile FTZ.
- Per-channel-scale epilogue precedents: `qgemv_w8a8`
  (`qgemv_int.metal:12` — w_scale[n] epilogue, 2 rows/simdgroup;
  launcher-only, unbound) and `qgemm_fp8_scaled` (`qgemm.metal:340` —
  rank-1 epilogue; drop a_scale for W8A16 prefill).
- **TurboQuant Metal port is complete**: `turboquant.metal`
  encode/decode + combined encode/attention serving kernels,
  instantiated for head sizes **64/128/256/512** × f32/f16/bf16;
  native launchers `turboquant_native.py`; `TurboQuantAttentionBackend`
  selectable on Metal (`metal.py:213`). Metal mileage today = DSV4
  drafter KV only; main-KV on the hybrid V2 runner is new integration.

### Bring-up blockers, in hit order (all confirmed with file:line)

1. `vllm/platforms/metal.py:78` `supported_quantization = ["gguf"]` —
   add `"compressed-tensors"` (enforced via `config/model.py:1172`).
2. `vllm/model_executor/kernels/linear/__init__.py`: no METAL key in
   `_POSSIBLE_NVFP4_KERNELS` (:446) nor `_POSSIBLE_WFP8A16_KERNELS`
   (:379); and :922 force-selects Marlin when `use_a16` — gate that on
   is_cuda.
3. fp8 allocations raise on MPS (`Undefined type Float8_e4m3fn`,
   verified torch 2.13.0): `compressed_tensors_w4a4_nvfp4.py:73`
   (scale [N,K/16]) and `fp8_utils.py:1255`
   `create_fp8_weight_parameter` (the FULL [N,K] weight). Allocate
   uint8, `.view(fp8)` lazily (view works on MPS; `.to()` does not).
   LUT pattern for python-side decode:
   `vllm/models/deepseek_v4/metal_indexer.py:46` `_e4m3_lut`.
4. Scheme selection: FP8 layers **auto-select**
   `CompressedTensorsW8A16Fp8` (Metal's fake sm80 fails the ≥89 W8A8
   gate, `compressed_tensors.py:818-837`) — free and correct. NVFP4
   lands on W4A4 (`use_a16=False`, checkpoint has input scales,
   `:734-743`) — needs a Metal override to W4A16 semantics (or the
   Metal kernel simply ignores input scales).
5. `compressed_tensors_w8a16_fp8.py:145` transposes weight to (K,N);
   our qgemv family wants (N,K) row-major and the host binding asserts
   contiguity — intercept before the `.t()` on Metal.
6. TRAP: lm_head scheme exceptions are swallowed
   (`compressed_tensors.py:183`) → silent fallback to an UNQUANTIZED
   2.5 GB bf16 lm_head. Assert FP8 lm_head at N1. Never set
   `head_dtype` on Metal (`logits_processor.py:110` raises; `:132`
   materializes an fp32 copy).
7. Pure-torch oracle pieces (`nvfp4_emulation_utils.py`):
   `break_fp4_bytes` is clean; `dequantize_to_dtype:391` and
   `ref_nvfp4_quant:435` hit the fp8 `.to()` break — patch with the
   u8 LUT. Triton decorators at module import — verify the stub shim
   on macOS.

### Landscape (who else runs NVFP4 + the technique haul)

- Competition on Metal: **only MLX** (`fp_qmv_fast`; the vllm-metal
  project serves NVFP4-mlx checkpoints on it). llama.cpp has
  GGML_TYPE_NVFP4 but its Metal backend declines the type (CPU
  fallback), and its GGUF conversion drops the fp32 global scale (ours
  is more faithful). "Faster than anyone" = **beat MLX** on this
  checkpoint; llama.cpp Q4_K c1 20.9 remains the platform ceiling ref.
- E2M1→fp16 decode: MLX 3-op branchless
  `as_type<half>((n&7)<<9) * 2^14` + sign select (subnormal 0.5 exact);
  Marlin gets 2 ops per 2 values after load-time nibble pre-positioning
  at halfword tops. E4M3→fp16: `(s&127)<<7` then ×2^8, **exact**;
  decode scales as UNSIGNED (bit7 never set in valid checkpoints).
- **E2M1×E4M3 product is exactly representable in fp16** → proven
  structure (ggml mxfp4 Metal kernel ⊕ MLX fp_qmv): 1 thread = one
  16-value group per iteration, unscaled FMA tree, ONE scale multiply
  per 16 MACs hoisted onto the block partial sum, fp32 cross-block
  accumulation, simd_sum, global scale once per output element.
- **Fold every power-of-2 rebias into one fp32 epilogue constant**
  (global × 2^22 if both decodes are left raw) — the Marlin pattern.
- Traps seen in the wild: e4m3 SUBNORMAL scales occur in real
  checkpoints (caused llama.cpp's PPL=5.8M bug — decoder must handle);
  global-scale inversion conventions differ per backend (the CT scheme
  stores `layer.weight_global_scale = 1/max` — get it right once at
  load); TP sharding of scales along K (vLLM #41511).
- Batch tiers (ggml precedent, matches our UPDATE 11 cost model):
  mul_mv (M=1) / ext r1 (M=2–5) / simdgroup-matrix GEMM above.

## Milestones (revised at N0 close + DFlash/TurboQuant directive)

**N1 — Bring-up (correctness first, slow OK): ✅ DONE 2026-08-19 (see UPDATE 13)**
- Plumbing unblocks 1–7 above (explain-before-edit each).
- `MetalNvFp4LinearKernel` + `MetalWFp8A16LinearKernel` with bring-up
  apply: dequant-once-to-bf16 at load (~45 GB unified RAM — bring-up
  only) or per-call dequant; W4A16/W8A16 semantics. Both CT and
  ModelOpt configs normalize to the same attr names post-load
  (`weight`, `weight_scale`, `weight_global_scale`) — write the Metal
  kernels against those and serve both for free.
- Model/loader: compressed-tensors safetensors on the qwen3_5 chain
  (single-file; `mtp.*` index entries resolve to
  `model_mtp.safetensors` — keep out of base load). Fused-shard scheme
  checks are safe here (q/k/v all group_0; gate/up both group_1) but
  `should_ignore_layer` RAISES on mixed schemes across fused shards —
  watch it.
- Register source `qwen38-27b-nvfp4` + profile `qwen38-nvfp4-1`
  (platforms ["metal"], engine cloned from `qwen38-1`); slimserve owns
  the 23.4 GB fetch.
- Gates: greedy parity vs the pure-torch emulation oracle + retrieval
  needle test (UPDATE 3/4 methodology). Assert lm_head came out FP8
  (blocker-list trap 6).

**N2 — Exact-token baseline: ✅ DONE 2026-08-19 (UPDATE 14)**
- `benchmark_dsv4_exact.py` + m2_source.txt, c1/c4/c8 1000x256 +
  2500x64, pin new canonical shas + wall clocks,
  `perf/baseline_status.md` snapshot. Bar: Q4_K M4c
  (c1 11.17 / c4 16.60 / c8 17.42); llama.cpp Q4_K c1 20.9 ceiling ref.
- Pinned: c1 7.02 (sha 8c58a4c6 3/3) / c4 14.37 / c8 15.96 / 2500x64
  2.55 (sha d0e07ddd 3/3). c4/c8 request-0 shas are tie-carriers, NOT
  anchors.

**N3 — Metal GEMV kernels (FP8 FIRST, then NVFP4): ✅ kernels SHIPPED 2026-08-19 (UPDATE 15); `_mb` twins + geometry OPEN**
- **3a FP8-channel GEMV + `_mb` twin (do first)**: carries 10.6 GB/tok
  — more traffic than the NVFP4 side — with zero unpacking; must hit
  near-roofline. Planar e4m3 decode (`tk_e4m3_decode` / fp8_raw idea)
  × bf16 activations, per-row `w_scale[n]` epilogue (`qgemv_w8a8`
  structure, drop a_scale, fp16 FMA not idot). Covers attn qkvo, GDN
  qkv/z/out, mlp 56–63, and lm_head (grid-split for 248,320 rows).
- **3b NVFP4 planar GEMV + `_mb` twin**: NEW kernel — do NOT repack to
  the interleaved struct. Buffers D, Wq, Wsc, X + N, K, global-scale
  (setBytes); fresh lane geometry (1 thread = one g16 block/iter);
  scale-on-partial-sum; fp32 accumulation; folded fp32 epilogue.
  Microbench decode variants: MLX 3-op vs Marlin-repack 2-op vs 8 KB
  product-LUT (expect ALU to win on M1; repack is load-time-free).
  Keep the UPDATE 10/11 lessons: 4+ blocks in flight, no weight
  materialization, grid.y column pairs for batch; M-wide unrolled
  bodies i-cache-thrash; TG-mem accumulators only reach parity.
- Gates (all mandatory): float64 oracle (adapt `q4k_oracle*.py`),
  serving-shape microbench (adapt `mmvq_bench*.py`), serving A/B with
  kill-switch envs, DSV4 anchor re-gate on ANY metallib/.so rebuild.

**N4 — Prefill + the transferred levers:**
- Prefill dequant-then-matmul first; then evaluate `qgemm_frag_nvfp4`
  (needs a planar variant) and `qgemm_fp8_scaled` (drop a_scale) for
  fused. Remember DSV4 PREFILL v10: host aten fluff around a GEMM can
  exceed the kernel time — check the wrapper before the kernel.
- **#16 non-GEMV ~31 ms/step is now the c1-decisive lever**: gdn
  NSG-multi-row (llama.cpp ggml-metal.metal:2704), ba-GEMV fusion,
  paged-attn head-256 gap (see N6 overlap), norm/glue. #17 batch ALU:
  simdgroup_matrix at M=8 (may partially dissolve — NVFP4 decode is
  cheaper per column).

**N5 — DFlash 2 speculation (INTEGRATION COMPLETE 2026-08-20, UPDATEs
24-25; serving default deferred behind N5b):**
- SHIPPED: full PR #52816 port (qwen3_dflash2.py, dflash2/ speculator
  with torch-native MPS walk, registry/config/V2 forcing,
  draft_logits_spec hook; the "reconcile upstream hooks first" step
  dissolved — the hooks are the PR's own additions); compressed-tensors
  lm_head accepted in compute_candidates (drafter ships no embed/lm_head
  and ties to the target's fp8ch head); MPS GDN spec path (UPDATE 24:
  gdn_fused_prepare spec-rewind mode + gdn_recur_spec per-position
  checkpointing + _forward_core_metal_spec, oracle 24/24 bit-exact,
  non-spec re-gates all bit-exact incl. 9th consecutive DSV4 anchors);
  three MPS fixes in the DFlash1 context-KV precompute (torch-native
  rms_norm x2, RoPE forward_native, metal_attn.do_kv_cache_update).
- Profile: twin `qwen38-nvfp4-1-df2` (speculator on source
  qwen38-27b-nvfp4, z-lab/Qwen3.8-27B-DFlash2 pinned 50307d4c,
  method=dflash k=7; drafter resolves from ~/models/Qwen3.8-27B-DFlash2
  = symlinked HF snapshot). Canonical qwen38-nvfp4-1 stays no-spec.
- GATE RESULTS (UPDATE 25): greedy spec output DETERMINISTIC (c1 sha
  b46e676c 2/2, same sha at c4 request-0); divergence from the no-spec
  fa58598b proven to be an EXACT logprob tie (' Use' / ' Report' both
  -1.538660) flipped by batched-verify kernel numerics — greedy-correct,
  gate redefined as determinism + tie-flip-only + acceptance +
  throughput. Acceptance 2.93 tok/step at k=7 (essay prose). Throughput
  BELOW no-spec (c1 9.59 vs 11.02; c4/c8 worse; 2500x64 +2.7%): phase
  census puts target_forward at 234 ms/step (M=8) — the UPDATE 11
  weight-stationary batch-GEMV collapse, inherited by NVFP4/fp8ch, plus
  the M>8 dequant+dense fallback at c4/c8.
- **N5b DONE (UPDATE 26, 2026-08-20): mv_ext batch GEMVs** for M in
  [3, 8] (fp8ch R1=4 / nvfp4 R1=2; odd M native — the compute_candidates
  M=R*7 lm_head fix came free; simdgroup_matrix GEMM investigated and
  parked: M1's MAC phase is issue-bound at ~2.3e12 FMA/s, the "M=8
  collapse" was mostly FMA physics and the ~21 tok/s projection was
  wrong). Gate: spec c1 9.59 -> 11.11 (b46e676c HELD) — ABOVE no-spec
  11.06 for the first time; no-spec c4 +7.4% / c8 +9.5%. Remaining from
  the old list: revisit max_num_batched_tokens (spec drops
  max_num_scheduled_tokens to 2000); simdgroup GEMM for M in [9, 32]
  only if c4/c8 spec ever matters.
- k-sweep DONE (UPDATE 27): k=3 13.65 / k=5 11.70 / k=7 11.11 spec c1;
  acceptance ~3.0 at every k (drafter horizon, not verify quality) —
  twin pinned to k=3; promotion to canonical blocked only on a
  batch-adaptive spec gate. Probabilistic drafting: implemented
  torch-native on MPS (no Philox parity — same stance as gumbel_sample);
  greedy is the serving mode.
- Fallback (demoted): the model's own MTP head (`qwen3_5_mtp`,
  `model_mtp.safetensors` 849 MB bf16, num_speculative_tokens 2 per
  the GGUF-source precedent) — only if DFlash 2 underdelivers post-N5b.

**N6 — TurboQuant KV: GATE 1 PASSED 2026-08-21 (UPDATE 36).**
- Profile `qwen38-nvfp4-1-tq` = canonical + TURBOQUANT backend +
  turboquant_k8v4 on the target's 16 full-attn layers AND the DFlash2
  drafter (`speculative_overrides` pins BOTH drafter fields — an unset
  drafter dtype inherits engine-global TQ with a metal_attn backend =
  boot reshape crash; a bf16 drafter cannot pad into the TQ max page).
- Six-run debugging arc (notebook UPDATE 36) found the Metal TQ kernels
  had NEVER been production-exercised ("DSV4 drafter TQ" is a
  CUDA/ROCm-profile fact; the Metal .so binds ONLY encode +
  attention_metal). Fixes: .contiguous() V guard in _store_kv (fused-
  QKV slice), Metal routes ALL continuation prefill through the
  synthetic-decode path (no dequant symbol on Metal), and THE BUG —
  `tq_attention_combined` mapped q-head->kv-head as head % Hkv where
  vLLM's GQA convention is head / (Hq/Hkv). One-line fix in
  turboquant.metal; parity cos 0.9955 at hs64/128/256 ctx 1-2500
  (scratchpad 04ba9b90 tq_parity.py); DSV4 anchors BIT-EXACT 13th
  consecutive post-rebuild (anchor_regate_tqfix/). PENDING: twin-port
  turboquant.metal to QuixiCore-Metal with the qgemv/rms_norm batch.
- Gate: needle "739214" exact at 4.5k multi-chunk prefill, det 2/2
  (c1 0c92d6e3, 2500x64 4494d8e6 profile-local pins), TPS c1 13.12
  (-11.3%) / 2500x64 2.93 (-13.4%) / c4 19.73 (-8.7%) / c8 23.17
  (-6.0%). KV 4096 -> 1728 B/token (~2.4x, target + drafter).
  CANONICAL DEFAULT UNCHANGED (TPS still rules at 32k).
- Remaining N6 options: long-ctx legs + max-admission sizing where the
  2.4x pays (262k/1M); fp8-KV comparison (checkpoint ships static k/v
  scales) if TQ tax matters; perf lever to close the 2500x64 gap =
  native TQ dequant kernel or a slots-matrix-free continuation
  (current Metal continuation builds an O(q_len x max_ctx) int32 slots
  matrix per layer call in turboquant_native.py). #16b
  (tq_attention_combined_hs256 replacing the PA256/SDPA head-256
  route) still unevaluated.

**N7 — Opinionated cleanup: DECOMMISSION DONE 2026-08-21 (UPDATE 37).**
- Criteria met: every leg beats the Q4_K bar (c1 132.5% / c4 130.1% /
  c8 141.5% / 2500x64 103.3%) and canonical needle recall is exact at
  4.5k multi-chunk. Profile `qwen38-1` + source `qwen38-27b` (GGUF)
  removed from slimserve/profiles.json (zero dangling refs; canonical
  dry-run clean). q4_K kernels STAY in the shared QuixiCore-Metal
  library per the boss directive; local GGUF files untouched.
- Remaining tail: 262k/1M context sizing — informed by UPDATE 36's
  capacity data (canonical bf16 KV holds 158,038 tokens under the
  16 GiB budget and cannot fit one 262k request; the TQ profile holds
  292,882 and can). Long-context serving likely rides the -tq profile
  with max_model_len raised.

**N8 — 262k long-ctx sizing + Metal TQ dequant route (2026-08-21, UPDATE 38;
UPDATE 39 pending the in-flight gate).**
- Phase 1 (UPDATE 38): `--ctx 262144` boots and ADMITS on -tq (native
  max_position_embeddings 262144, no rope scaling; `slimserve --ctx`
  replaces max_model_len). But prefill through the synthetic-decode
  continuation is O(ctx^2) with a ~24x-redundant constant: measured 16k
  prefill 406 s; model T(C) ~= 6.8ms*C + 2.3e-6*C^2 -> 262k ~= 22 h.
  BLOCKED -> ported the CUDA continuation design to Metal.
- Implementation (worktree, uncommitted): `tq_decode_combined` kernel in
  turboquant.metal (combined-slot dequant, decode math lifted from the
  split tq_decode + tq_attention_combined; 12 instantiations);
  `launch_tq_decode_combined` in tk_launch.h; `turboquant_dequant_kv_metal`
  binding in qc_metal_serving.mm + ops.py wrapper; turboquant_attn.py:
  restored the >128 continuation threshold on Metal, Metal branch in
  `_continuation_prefill` (1-D slot list from the block table — no
  (q_len, ctx) slots matrix), Metal guard on the CUDA Pi inverse-rotation
  (Metal K is never rotated), `_metal_tiled_continuation_attention`
  (KV-tiled fp32 online softmax, tile 4096, engaged when seq_len > 8192 —
  masked SDPA would materialize 0.5-26 GB per layer call at 262k), and
  the continuation workspace reserve re-enabled on Metal. metallib
  13,094,352 B + .so rebuilt; parity PASS (scratchpad 04ba9b90
  tq_dequant_parity.py): dequant K cos 0.99998, V cos 0.9955 (= the known
  4-bit error), tiled-vs-SDPA same-KV cos 0.99999, raw-KV oracle 0.996.
- GATED 2026-08-21 17:37 (UPDATE 39), ALL PREDICTIONS HELD:
  DSV4 anchors BIT-EXACT (14th consecutive). TQ short-ctx: needle
  739214 exact; c1 13.078/13.082 sha 0c92d6e3 BIT-EXACT vs the
  UPDATE 36 pin (no continuation on that leg — change provably
  isolated); 2500x64 3.278/3.268 sha d0e07ddd = the CANONICAL sha
  (greedy tokens now match canonical on this leg), gap -13.4% ->
  -3.0%; c4 -7.2%; c8 -2.1%. 262k ladder ALL EXACT: 16k=111 s (3.7x
  vs synthetic-decode 406 s), 65k=577 s, 131k=1492 s, 262k=4493 s at
  259,888 prompt tokens. Fitted prefill T(C) ~= 6.4e-3*C + 3.9e-8*C^2
  (131k predicted 1489 vs 1492 measured). 1M PARKED (out of native
  rope spec 262144 + needs ~40 GiB KV budget). Raw:
  perf/results/2026-08-21/{anchor_regate_dequant, tq_dequant_gate,
  tq_longctx2}/. Canonical default UNCHANGED at 32k.

**#16b — split-K TQ decode attention (2026-08-21, UPDATE 40): DONE, the
-tq profile now beats canonical on 3 of 4 legs.**
- The #16b evaluation (tq256_bench.py, clean box) REJECTED the item as
  posed — monolithic `tq_attention_combined` is 12-19x SLOWER than
  PA256 — but root-caused the entire -tq c1 -11.6% tax: one
  threadgroup per (q-head, batch), two TG barriers per token, ~7 GB/s;
  batch 4 costs the same as batch 1 (occupancy-bound proof). Slots
  build and table width exonerated.
- Implementation (worktree, uncommitted): `tq_attention_splitk` +
  `tq_attention_reduce` in turboquant.metal (PA256-style (H,B,P) grid;
  one SIMDGROUP per token — lane covers HS/32 elements, dot = one
  simd_sum, ZERO token-loop barriers; FWHT-domain V partials, one
  staged cross-simdgroup merge; reduce folds sink + single inverse
  FWHT; 24 instantiations); launchers in tk_launch.h;
  `turboquant_attention_splitk_metal` host op (PA256-style sizing,
  target 1536 via VLLM_QC_TQ_SPLITK_TARGET, ring_out tq_sk_*, one
  serial-dispatch encoder, max_context arg) + ops.py wrapper; routing
  in turboquant_native.py Metal branch (head_size==256 AND no sliding
  window only — hs64/128 DSV4 drafter stays monolithic BY CONSTRUCTION
  so the anchors hold; VLLM_QC_TQ_SPLITK=0 kill-switch); max_context
  plumbed host-side from seq_lens_cpu at _decode_attention +
  _uniform_query_attention (spec-verify hot path) + the synthetic-
  decode continuation site. metallib 13,327,040 B + .so 960,272 B.
- Parity 10/10 PASS (tq_splitk_parity.py): vs monolithic rel
  1e-9..3e-6 (summation order only); quant floor vs SDPA 0.9954-0.9958
  unchanged; ragged/ctx=1/ctx=5/48-blocks/sinks/hs128 covered.
- Bench: kernel 12-33x; TQ decode now FASTER than canonical bf16
  PA256 everywhere (ctx1000 b4: 0.229 vs 0.380 ms; 32k b4: 6.15 vs
  12.38 — 432 vs 1024 B/token at ~220 GB/s effective).
- GATED 2026-08-21 19:23, ALL PREDICTIONS HELD: DSV4 anchors BIT-EXACT
  (15th consecutive, anchor_regate_tqsk/). -tq: needle 739214 exact;
  c1 14.688/14.685 sha 228d0bf4 2/2 (rolled as predicted; -11.6% tax
  -> -0.8%); 2500x64 3.459/3.448 sha d0e07ddd 2/2 (CANONICAL sha
  HELD, +2.1% ABOVE canonical); NEW tail 2100x32 leg 2.174/2.171 sha
  1337d2f7 2/2 (synthetic-decode continuation through splitk); c4
  22.11 (+2.2% above canonical); c8 25.15 (+2.0%); 16k needle exact
  at the 262k config, 110 s wall. Raw: perf/results/2026-08-21/
  {tq_splitk_bench, tq_splitk_gate, tq_splitk_longctx,
  anchor_regate_tqsk}/. FLAGGED DECISION: -tq now wins c4/c8/2500x64
  and trails c1 by 0.8% with 2.4x smaller KV — flipping the canonical
  default is a quality call (k8v4 cos ~0.9955 floor), not a TPS one.

**N9 — mrope repeat_interleave queue-drain (2026-08-24, UPDATEs 41-42):
DONE — canonical c1 14.80 -> 16.14 (+9.6%), all pins bit-exact.**
- Live-box xctrace found one ~21 ms GPU-idle gap per c1 step (host
  still encoding when the queue drains); a fresh cProfile census
  (canonical env — the 08-21 census predated async sched and its
  67 ms postprocess_state lead was DEAD, killed by a statement-timer
  probe: 0.055 ms/step today) attributed 65.8 ms/step (64% of host
  profile) to torch.repeat_interleave in _metal_prepare_rope_positions
  — the N1 torch replacement for the Triton mrope kernel.
  Data-dependent output shape -> full MPS queue drain per step; the
  same class gdn_attn.py fixed earlier.
- Fix (metal_compat.py, python-only): static-shape scatter+cumsum
  token->request map (200/200 equal vs repeat_interleave on MPS incl.
  zero-length requests); patched DefaultModelState.prepare_inputs
  passes num_tokens=input_batch.num_tokens (host int, no sync).
  VLLM_QC_ROPE_STATIC=0 = dynamic null path.
- OPS incident (UPDATE 42): the tmp cleaner swept the a99b scratchpad
  mid-gate — m2_source.txt/dsv4_gate.py/build recipes GONE. All
  recovered byte-exact (m2_source = cat of three perf docs at commit
  2ae908e42, 69,105 B, proven by a null boot reproducing c1 467b35c3
  + 2500x64 d0e07ddd bit-exact; dsv4_gate.py from the 43e3 transcript
  Write call; build scripts from the a99b transcript). STABLE HOME:
  perf/results/harness_assets/ — every chain now uses that path.
- GATED 2026-08-24 (rope_gate/ + anchor_leg_rope/): Phase 0 null
  validation bit-exact; canonical c1 16.139/16.138 sha 467b35c3
  BIT-EXACT 2/2 (144.5% of the Q4_K bar) / 2500x64 3.454/3.447 sha
  d0e07ddd BIT-EXACT / c4 23.16 (+7.2%) / c8 25.62 (+3.9%); -tq
  needle exact + c1 16.08 sha 228d0bf4 BIT-EXACT 2/2 (-0.3% vs
  canonical; -tq c4/c8/2500x64 not re-measured post-rope); DSV4 8tok
  anchor BIT-EXACT (python-only change, no binary rebuild).
- Remaining step structure: ~80 ms GPU + residual encode burst
  (~620 Metal encodes/step: ~357 MPSGraph + ~265 custom). Next
  levers: steady-cache metadata for the mamba-hybrid path (gdn
  builder ~30 ops x 10 calls/step; extend attn_utils steady path +
  GDN steady_decode_update per the DSV4 sparse_swa precedent),
  drafter MPSGraph linears, then Muse single-CB.

**N10 — steady-cache metadata for the mamba-hybrid path (2026-08-24,
UPDATE 43): DONE — c1 16.139 -> 16.334 (+1.2%), all pins bit-exact,
default ON on Metal (VLLM_QC_STEADY_META=0 null).**
- Post-rope census: gdn_attn build 8.2 ms/step (10 group builds x
  ~30 tensor ops + h2d each, rebuilt from scratch every step);
  upstream steady machinery never fires here (FULL-graph + opt-in
  gated, disqualified whenever model_specific_attn_metadata exists).
- Change (python-only, 4 files): ModelSpecificAttnMetadata.
  steady_signature() protocol (interface.py, None default);
  MambaHybridAttnMetadata returns ("mamba-spec", k) only at uniform
  all-spec decode (CPU checks; adaptive-k changes break the sig);
  attn_utils folds the sig into the steady tuple, refreshes
  cm.max_seq_len + cm.seq_lens_cpu_upper_bound in place on hits,
  calls steady_decode_update on supporting builders with fresh extra
  kwargs, rebuilds the rest WITH those kwargs; GDN steady_decode_update
  = re-copy spec_state_indices from the live block table + refresh
  num_accepted (2 copies replace ~30 ops).
- Two engine-killing findings (both now recorded lessons):
  (1) upstream eligibility deref'd is_prefilling=None on this path —
  crash on first request; a non-None model-specific signature is now
  the all-decode guarantee (CPU-side, no MPS sync) and the tensor
  check governs only the no-metadata case. Ramp helpers must verify
  GENERATED TEXT — an instant HTTP 500 satisfies curl and read as
  "ramped" while the engine was dead (round 1's false-positive
  restore).
  (2) model_runner allocates seq_lens_cpu_upper_bound FRESH each step
  (np.zeros + from_numpy), so the cached cm's view froze at
  cold-build content and metal/tq builders (rebuilt on every hit)
  derived decode max_context from it — newest tokens silently
  excluded from full attention. Symptom: deterministic sha roll
  (ac33cee5 both runs) with PASSING needle and +2.7% "speedup"
  bought by wrong math. seq_lens / query_start_loc / gathered block
  tables / slot mappings are in-place persistent buffers (audited);
  query_start_loc_cpu freezes too but is content-equal under the
  uniform sig.
- GATED round 3 (steady_gate/): canon needle PASS; c1 16.334/16.346
  sha 467b35c3 BIT-EXACT 2/2; 2500x64 3.444/3.452 sha d0e07ddd
  BIT-EXACT 2/2; c4 23.02 / c8 25.57 (-0.6%/-0.2%, noise — hits
  rare under composition churn); TQ needle PASS; -tq c1
  16.216/16.210 sha 228d0bf4 BIT-EXACT 2/2 (+0.8%). Null boot
  reproduced UPDATE 41 exactly (16.153, 467b35c3). Honest-gain note:
  the predicted +4-8% did not materialize — adaptive-k sig breaks
  make hits rarer than the per-step build cost implied.

**N11a — fused DFlash2 grouped conv (2026-08-24, UPDATE 44): DONE —
c1 16.334 -> 16.428 (+0.6%), everything bit-exact, default ON
(VLLM_QC_DFLASH_CONV=0 null).**
- Post-N10 phase census: drafter_propose 24.4 ms/step serialized with
  only ~5 ms of GPU weight reads — host-encode-bound. Each drafter
  layer ran 4 eager _grouped_conv chains (~10 MPS ops each incl. a
  torch.arange and an F.pad) = ~200 encodes/step.
- qc_dflash_conv (serving/dflash_conv/): one thread per element,
  block-local taps + position mask folded in; per-op fp32-round
  mirrors MPS elementwise semantics — kernel parity 37/37 BIT-EXACT
  (bf16/f16/f32, blocks 4/8/9, both projection side views via storage
  offset + row stride, no contiguous copy; check_mps_strided).
  Launcher tk_launch.h, binding qc_metal_serving.mm, ops.py wrapper,
  routing in qwen3_dflash2._grouped_conv (stale-.so hasattr guard,
  layout guards fall back to eager).
- Gate: DSV4 anchors bit-exact 17th consecutive (metallib+.so
  rebuild); null boot = UPDATE 43 exactly; conv-ON c1 16.428/16.438
  467b35c3 / 2500x64 3.454 d0e07ddd / needle exact / c4 23.11 c8
  25.62. Calibration reconfirmed: eager-chain profiler shares are
  queue-tail-inflated; dispatch elimination recovers ~5 us x count.
- CORRECTION (post-gate): drafter_propose nests INSIDE the
  sample_tokens bracket (model_runner.py:1651 under the :1522 wrap) —
  32.7 = 24.4 drafter + 6.5 sample_and_reject + ~1.8 glue. There is
  NO unattributed sampling cost; the drafter remains the top host
  block. Next instrument: VLLM_QC_OP_CENSUS per-step op counts —
  count x ~5 us is the honest currency for encode levers.

## Tooling that transfers (recovered to `perf/results/harness_assets/`;
session scratchpads are swept by the tmp cleaner — never cite them as
durable locations)

- `build_metallib.sh` (17s full rebuild), `build_qc_metal.sh` (.so),
  single-file iteration loop: `xcrun metal -std=metal3.1 -O2 -I
  csrc/quixicore/metal/include/metal -I csrc/quixicore/metal/kernels/common
  <one .metal> -o mini.metallib` (~10 s) + `qgemv_mb_bench.m`
  (standalone Metal harness) + `pipe_info.m` (register-footprint
  introspection via maxTotalThreadsPerThreadgroup).
- NOTE (`cmake/metal.cmake:43-49`): editing `dequant.metal` or anything
  under `include/metal/` is a SUBSTRATE edit — the full metallib build
  tracks it via the DEPENDS glob, but a stale hand-built mini.metallib
  will silently keep old shaders.
- `mmvq_bench.py` / `mmvq_bench_mm.py` (serving-route GEMV bench),
  `q4k_oracle.py` / `q4k_oracle_mb.py` (float64 + bit-identity oracles),
  `q4k_mb_gate_a.sh` / `_b.sh` (serving gate scripts),
  `anchor_regate_q4k_mb.sh` + `dsv4_gate.py` (DSV4 anchor re-gate),
  `restore_qwen_m4c.sh` (clean-boot restore).
- N0 artifacts (session-04ba9b90 scratchpad): `n0_checkpoint_findings.md`,
  `nvfp4_config.json`, `nvfp4_index.json`, `nvfp4_shapes.json`.
- Census instruments (env-gated, in-tree): `VLLM_SYNCPROF=1`,
  `VLLM_QC_PHASE_PROF=1` (layer brackets in qwen3_next.py).
- Artifact backups: `artifacts_backup_m4/`, `artifacts_backup_m4b/`.

## Standing ops constraints (non-negotiable)

- Explain-before-edit for every project-code change (symptom, root
  cause w/ evidence, exact change, why safe — then proceed).
- Commit only when the user asks.
- Interval logs are diagnostics; exact-token harness output only for
  claims. Record everything in perf/optimization_status.md per
  perf/perf.md; raw artifacts under perf/results/YYYY-MM-DD/<run-id>/.
- DSV4 anchors re-gate on ANY metallib/.so rebuild (UPDATE 30 pins:
  8tok 573db39598e7, off1-2000 bb83cc3054a3, 2500x64 f75e1d41ac3d).
- Boot ramp protocol before multi-chunk prefill (primer + 1000-tok
  w/ decode + multi-chunk throwaway). Kill = TERM EngineCore AND
  api_server pids, verify dead, wait for memory before reboot.
- Milestone ritual: inline copy-pasteable compaction handoff in the
  reply at every milestone. Hold turn for <10-min waits.
- Do NOT use the AskUserQuestion tool — it breaks the user's TUI; ask
  in plain text.
- Qwen model facts: 64 layers = 48 GDN + 16 full attn, hidden 5120,
  ffn 17408, vocab 248,320; eos 248044 per NVFP4 config.json (old
  handoff said 248046 — recheck against the served tokenizer at N1).
  Metal victim canary hardcodes block 30 (delete after soak); NANPROBE
  kit env-gated VLLM_QC_NANPROBE=1.

---

# PARALLEL CAMPAIGN HANDOFF (merged from main 2026-08-25): Qwen3.8-27B + DFlash 2 GGUF/vision Metal serving (M5 Max box)

# Qwen3.8-27B + DFlash 2 Metal Serving Handoff

Updated: 2026-08-23 00:58 (M5 Max MacBook Pro, 128 GB, ~460 GB/s measured
stream). Written so a fresh agent can take over cold. Read this, then
`perf/optimization_status.md` entries (19) onward, then
`perf/qwen38_metal_design.md` (every verified tensor map + mechanism).

## Mission and hard rules

- Serve profile `qwen38-q2kxl-1` on Metal: Qwen3.8-27B (unsloth UD-Q2_K_XL
  GGUF, 64-layer hybrid: 48 gated-deltanet linear-attention layers + 16
  full-attention layers, head_dim 256, interleaved MRoPE; plus the
  mmproj-F16 `qwen3vl_merger` vision tower) speculated by the Inco AI
  DFlash 2 drafter (z-lab Q4_K_M GGUF, block 8, top-16 path selector,
  two-tap convs).
- Bars: llama.cpp plain decode on this box/artifact = **35.67 tok/s**.
  Vendor DFlash 2 acceptance 4.80 is a GSM8K number; the llama.cpp
  dflash2-pr branch on the SAME GGUFs/settings gets 2.51 tok/step on our
  essay prompt and 4.74 on a GSM8K-style prompt (task-domain dependence;
  compare acceptance only on matched arms).
- **Speculation is always on and must be net-positive** (memory
  `spec-always-fastest`); slower-than-plain spec is a BUG, never a
  documented config. This gate is now closed: registered DFlash k=3 beats
  plain on both retained prompt arms and the matched exact-server workload.
- **Greedy / temperature 0 is banned stack-wide** (memory
  `no-greedy-benchmarks`; the user removed the flag on purpose). Validation
  and benches use the model's shipped sampling defaults from the GGUF
  (`general.sampling`: temp 1.0 / top_p 0.95 / top_k 20), seeded (42).
  Layer-level parity (cosine on activations) needs no sampling and stays
  the correctness instrument.
- Commit authorship: Eric Hartford sole author, no assistance trailers
  (the repo's signoff hook adds his Signed-off-by). Commit with
  `env SKIP=markdownlint-cli2 git commit ...` (the notebook's pre-existing
  line lengths fail markdownlint; its auto-fix also corrupts `+ ~15`-style
  lines and `_foo` identifiers -- never let it run on perf/).

## State of the tree

Committed on `main` (pushed): `fe960935f` "vision, DFlash 2 spec e2e,
native IQ decode, hybrid-pool layout fix (15 tok/s plain)" on top of
`39efaa7d9` (correct plain decode, layer parity). Prior campaigns: Muse
`ad8e8e937` (20.1 tok/s spec, Metal), DSV4 A100 `bad7cfd46` (A100 box).

UNCOMMITTED in the worktree is one tested optimization stack (preserve all
of it; do not treat the native pieces as abandoned experiments):

1. **Fused target GDN, complete and routed.**
   `csrc/quixicore/metal/kernels/serving_glue/gdn_step.metal`, the binding
   in `qc_metal_serving.mm`, `vllm/quixicore/ops.py`, and
   `qwen_gdn_linear_attn.py` implement decode and multi-position verify
   (convolution + recurrent scan, fp32 state in place, exact store/resume /
   rollback slots) plus a fused gated RMS norm. The torch-native oracle
   remains intact. Kill switch: `VLLM_QWEN38_FUSED_GDN=0`.
   Correctness: 147/147 exhaustive cases, all uniform/ragged/null/mixed
   plan cases, all gated-norm cases. Durable real-geometry tests are in
   `tests/model_executor/test_qwen_gdn_metal.py`.
2. **Verify-band quant MM, complete and routed.**
   `dequant.metal`, `qgemv.metal`, and the binding admit the target's
   Q2_K/Q3_K/IQ1/IQ2/IQ3/IQ4_XS formats to M={2,4,8,16,17} MM instead of
   repeated GEMV. Real-GGUF M=8/17 error <=0.2674%; sampled M=8 kernels
   are 2.4-5.0x faster than eight GEMVs. The eight-wide IQ decoder's
   difference from the scalar decoder is rounding-only (<0.1%).
3. **Seeded/vectorized MPS rejection, complete and routed.**
   `rejection_sampler_utils.py`, `qwen3_dflash2.py`, and `speculator.py`
   key all selector/accept/residual/bonus draws by (seed, position), remove
   the draft-logit double temperature divide, and batch rejection fully on
   MPS. Full-vocabulary Gumbel emission is now one keyed uniform plus CDF
   inverse sampling. Monte Carlo passes; the clean spec bench is seed-stable.
4. **Fused DFlash 2 convolution, built and routed.**
   `csrc/quixicore/metal/kernels/serving_glue/dflash2_conv.metal` replaces
   each repeat_interleave/roll/clone/elementwise graph with one dispatch.
   Both sides pass the torch reference at real 5120-hidden / 320-group BF16
   geometry; a same-process real-geometry microbench is 5.65x faster.
   Kill switch: `VLLM_QWEN38_FUSED_DFLASH2_CONV=0`. Powered end-to-end A/B
   retained it: fused essay/GSM medians 16.42/38.12 versus 15.75/36.22.
5. **64-bit hybrid KV gather, complete and routed.** MPS `index_select`
   silently wrapped signed 32-bit element offsets on Qwen's interleaved,
   strided K/V source. Requests crossing physical block 1271 therefore fed
   bad K/V into layer 19 and produced all-NaN target logits. The native
   `kv_cache_gather_range` carries the physical block stride and address math
   in 64 bits, gathers only live rows, and is exact at blocks 1186/1271/1580.
   A 20x64-token repeated run remains finite and seed-identical through the
   old failure window and allocator wrap.
6. **Profile/tests/build.** `qwen38-q2kxl-1` is supported, registers DFlash
   k=3, and exports `VLLM_USE_V2_MODEL_RUNNER=1`. The real server passed text
   and image. The final focused suite (including the 5 GiB >2^31-offset case)
   is 17/17; `tests/slimserve` is 58 passed/1 skipped. Final metallib SHA-256:
   `539035eb15dea29152e11503fc1ee08676d5dfe08b9ef4cc241283092e887d4c`;
   deployed extension SHA-256:
   `ede784f0d4ecf7a5111fc55374987661ea3bcc4c48602189c0a009cb88c4efdb`.
7. **Shared live validation.** Registry discovery finds `dsv4-xxs-1`,
   `muse-kdyn-1`, and Qwen on this machine. Qwen passes text+image. Muse now
   passes text+image after fixing its parser's new-turn state and the real
   split `" to"` / `"=self<|message|>"` streamed header; raw SSE cleanly
   separates `reasoning_content` and final `content`. The combined parser and
   SlimServe suite is 62 passed/1 skipped. The complete matrix is not green:
   DSV4 reached health but its first request ran at about 0.1 tok/s with 0/5
   drafted tokens accepted and was terminated after about 12 minutes.

**CURRENT STATUS:** no correctness or profile gate is blocking Qwen serving.
The old 20 W power blocker is closed; the retained numbers below were captured
on AC power after the charger change. Qwen's remaining gap is performance
versus the 35.67 tok/s llama.cpp plain reference, not production-path
correctness. Separately, the current-machine profile matrix is blocked by the
DSV4 Metal regression described below; do not present that shared matrix as a
pass.

## Measured numbers (seeded shipped defaults, V2 runner, in-process)

| Build | Plain essay | Plain GSM8K | Spec essay | Spec GSM8K |
| --- | ---: | ---: | ---: | ---: |
| campaign start (V1) | 2.5 | -- | -- | -- |
| V2 runner, fp16 dequants | 6.4 | -- | 4.0 | -- |
| + layout fix (strided gather penalty) | 2.2 | 2.0 | 1.0 | 2.2 |
| **fe960935f** (+ gather fix, native IQ) | **15.0** | **14.0** | 3.5-4.1 | 8.0-9.5 |
| + fused GDN/MM/vector rejection (pre-conv) | 16.15-16.40 | 15.81-16.11 | **15.03-15.60** | **36.06-36.66** |
| + powered stack, k=7 | 16.99-17.17 | 16.77-16.86 | 16.05-16.35 | 36.82-38.19 |
| **supported profile, k=3** | **16.99-17.17** | **16.77-16.86** | **23.06-23.74** | **34.33-35.25** |

Acceptance (Prometheus counters, essay, notebook (25)): 2.71 tokens/step,
0.244 draft rate -- beats the llama.cpp dflash2-pr branch (2.51/0.219).
Correctness: all 64 layers cos >= 0.9997 vs llama.cpp eval-callback;
corruption gauntlet 7/7 clean boots, 24/24 same-seed pairs identical;
fused GDN exhaustive harness 147/147 plus all plan shapes; rejection Monte
Carlo PASS; current focused durable suite 17/17; vision tower ~1e-3 vs
llama-mtmd-cli; real SlimServe text and image requests pass; Muse-Glimmer was
unregressed by a fresh profile-exact text+image smoke after its reasoning
parser repair.

The old 8-position x 48-layer Python GDN inversion and the >2^31 hybrid-cache
corruption are closed. At k=3, M=4 verification is a better Metal operating
point than the trained/upstream k=7/M=8 width: essay median rises from 16.18 to
23.27 tok/s while GSM remains 34.78. The exact registered server produced 128
input + 256 output tokens at 18.646 tok/s spec versus 15.913 plain (+17.2%).

## Open bugs / items, ranked

1. Remaining perf versus llama.cpp: Q4_K GEMV measures only ~95 GB/s on the
   5120x6144
   ssm_out shape (4.7x off floor, pre-existing); head_dim-256 paged
   attention fast path (the 16 full layers run SDPA; paged path is
   64/128 only); selector walk and the other five-layer drafter graphs.
   The k=3 top ledger is target 70.19 ms and inclusive sample/propose tail
   16.42 ms per step, so target bandwidth is again the primary wall.
2. Fix the DSV4 Metal profile regression exposed by the attempted complete
   live-smoke matrix. `dsv4-xxs-1` loaded 93.63 GiB and reached health, then
   spent about 12 minutes on the first tiny request at ~0.1 tok/s; the first
   draft had 0/5 accepted. This is grossly inconsistent with its historical
   33.684 tok/s baseline and must be isolated at first-step/verify granularity.
   Qwen and Muse both pass their registered text+image arms, but the full
   three-profile matrix remains failed until DSV4 completes.
3. Cosmetics/hygiene: env-gated diagnostics remain in
   `models/qwen3_5.py` (`_Qwen38DumpState`, layer-parity instrument) and
   `qwen3_dflash2.py` (`QWEN38_DFLASH_DUMP` recall@k dump) -- zero-cost
   unset; remove at campaign close. Stale `autostash` entry in `git
   stash` is from Aug 7 (DSV4 era), safe to drop. A stray token quirk
   appeared in sampled answers at temp 1.0 (both text and vision) --
   unattributed, low priority.

## Scripts and raw artifacts

Durable copies are under `perf/results/2026-08-22/qwen38-fused-gdn/`:
`consolidated_bench.py {plain|spec}` (essay + GSM8K arms, 3 seeded repeats,
prints BENCH_JSON), `collapse_probe.py`, `spec_profile.py`, and
`spec_profile_top.py`. Patterns: in-process `vllm.LLM(**engine_kwargs)`
from `slimserve.registry.resolve("qwen38-q2kxl-1","metal",1,None,2**37)`,
`__main__` guard (EngineCore spawns), `SamplingParams(temperature=1.0,
top_p=0.95, top_k=20, seed=42)`, `max_model_len` 8192 +
`gpu_memory_utilization` 0.45-0.6 for qwen38 (Muse smokes must use
PROFILE-EXACT kwargs -- a max_model_len override breaks its image
profiling).

Final raw data is under `perf/results/2026-08-23/qwen38-kv-gather/`:
`run_summary.json`, `exact_spec.json`, `exact_plain.json`, `smoke.json`, and
the real-server log. Shared-profile evidence is in `smoke-muse-final.json`,
`smoke-muse-final/muse-kdyn-1.log`, and `smoke-all/dsv4-xxs-1.log`. The exact
server harness command uses
`benchmarks/benchmark_dsv4_exact.py` with explicit `--temperature 1.0
--top-p 0.95 --top-k 20 --seed 42`; never rely on that harness's legacy greedy
default for Qwen.

## Ops gotchas (each cost real time)

- The Mac SLEEPS and kills background runs/agents: hold it awake
  (`mcp adrafinil keep_awake`, lid-closed included) for any long run.
- A 20 W charger at 1% battery throttles this workload catastrophically.
  Require a high-wattage supply and battery reserve before any baseline or
  phase-profile run; check `pmset -g batt` and the charger wattage first.
- Refreshing `vllm/quixicore_metal.metallib` / `_quixicore_C...so`: rm-then-cp
  + `codesign -f -s - <so>`; cp over the mapped inode SIGKILLs on dlopen.
- llama.cpp builds: `env -u LDFLAGS -u CPPFLAGS` (a custom-LLVM env poisons
  links); `~/llama.cpp/build-qwen38` (master, plain oracle),
  `~/llama.cpp-dflash2/build` (PR #27342 spec oracle); check a binary is
  fresh before trusting it (`strings ... | grep` a known-new symbol).
- Registry bytes/sha come from `curl -I` / the HF paths-info API, never
  from summarized pages (a wrong byte count masqueraded as a broken
  download for an hour).
- Regression smokes use profile-exact engine kwargs.

## Reference facts (verified; full maps in perf/qwen38_metal_design.md)

- GGUF arch strings: target `qwen35`, drafter `dflash` (three-way probe:
  `dflash.expert_count` -> DSV4 DSpark, `dflash.selector_rank` -> DFlash
  2, neither -> Muse) across config parser, tokenizer registry, loader.
- llama.cpp converter conventions undone at load: +1 fold in every norm
  weight except `linear_attn.norm` (we use GemmaRMSNorm -> subtract 1),
  GDN per-V-head tensors in TILED order (pairing i_k = i_hv % 16, cfg
  `gdn_tiled_v_head_layout`; the FLA Triton kernels still assume grouped
  if this GGUF ever runs on CUDA), `ssm_a` stored as -exp(A_log)
  (A_log = log(-ssm_a)), conv1d (dim,kernel)->(dim,1,kernel), MTP block
  `blk.64.*` unmapped, fused `attn_qkv` row-split into q/k/v shards
  (GGUF quantizes per output row), full-attn gate fused inside `attn_q`
  (per-head [q|gate], matches the vendored split).
- Hybrid shared block pool: attention views are restrided blocks-first
  (attn_utils `_update_hybrid_attention_mamba_layout`); metal_attn's SDPA
  path uses the native 64-bit range gather. Do not restore MPS `index_select`
  on the strided pages view: beyond 2^31 elements it silently reads the wrong
  address, even though its small-cache microbench is fast.
- Quant formats: all native on Metal (qgemv + qgemm tiles incl. IQ1_S,
  IQ1_M, IQ2_XS, IQ2_S, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL);
  `_DEQUANT_TYPES` is empty; only the Q2_K embed table dequantizes.
- Drafter: 5 layers all NON-causal (`dflash.attention.causal=False`),
  block 8 counts the anchor (7 drafted), target layers [5,19,33,47,61]
  0-based, selector A/B tables {248320,256} Q4_K dequantized at load. The
  registered Metal serving depth is deliberately k=3 after the powered sweep.

---

# Handoff: MI300X GGUF profile record (2026-08-25)

> **Status update (2026-08-28, added during the origin/main merge).** The
> "actual open problem" below is **resolved**; read this section as the
> investigation record it was, not as current state. The illegal-memory-access
> fault at `max_num_seqs: 64` was root-caused to the DFlash2 two-tap
> convolution using the checkpoint's trained block width (8) instead of the
> active `1 + num_speculative_tokens` serving layout, and fixed in
> `_resolve_serving_block_size` (`vllm/model_executor/models/qwen3_dflash2.py`).
> The `qwen38-q2kxl-1` mi300x record ships at 64 sequences with the fixed
> 96 GiB KV pool -- the "fallback that passes" was not needed. Measured exact
> workload: c1 77.23 tok/s, c8 194.21 tok/s on the V2 runner, 200.20 tok/s
> after type-aware imatrix routing. Full write-ups: the three entries dated
> 2026-08-25/26 at the end of `perf/optimization_status.md`.

## One-paragraph state

The Qwen3.8-27B GGUF path on MI300X **works** — correct text and vision output,
DFlash2 speculation live — and that is committed and pushed. What is **not**
finished is the `qwen38-q2kxl-1` mi300x profile *record*: it is uncommitted in
the worktree, and the config values I picked fail in ways I had not finished
bisecting when this session ended. A working configuration is known (see
"Fallback that passes"). The open question is whether the failures are bugs
worth fixing or shapes to back away from.

## What is committed and pushed (do not redo)

`main` @ `2989f4b28b`, in order:

- `cd7e983d07` — **GDN value-head layout fix.** llama.cpp stores per-value-head
  gated-deltanet tensors in ggml *tiled* order (value head `hv` pairs with key
  head `hv % H`, expanded with `ggml_repeat_4d`, see
  `~/llama.cpp/src/models/qwen35.cpp:443`); the FLA kernels expand with
  `repeat_interleave`, i.e. HF *grouped* order. Selected by platform:
  `gdn_core_honors_tiled = current_platform.is_metal()`. Metal keeps its native
  tiled scan; elsewhere the v-head axis is normalized around the recurrence.
  The same normalization also had to go into `_forward_core_decode_non_spec`,
  whose early `return` sits between the two reorder points — that is why
  prefill was perfect while every decoded token was garbage.
- `b4cc16492c` — **DFlash drafter KV layout fix.** The drafter hardcoded the
  split cache layout `(2, num_blocks, block_size, H, head_size)` it was written
  against on Metal. TRITON_ATTN and ROCM_AITER_FA *pack*:
  `(num_blocks, H, block_size, 2*head_size)`. `_store_kv_at_slots` now detects
  the layout and **raises on anything unrecognized rather than guessing**.
- `a4ee5caea6` — NVFP4 regression gate (that profile still passes).
- `2989f4b28b` — **profiles are one config per platform.** Five profiles used
  to span platforms via `platform_overrides`; each id now stores one record per
  platform under `variants`, each tagged with its own `platform`. The CLI still
  takes no platform (it detects one). `registry.variant(id, platform)` gets one
  record. Two tests enforce it. `platform_overrides` is retired.

Earlier in the session (also pushed): IQ2_XXS dense-dispatch fix, `default:`
guards on the GGUF kernel switches, ROCm build repair, compiled-startup fix.
See `perf/optimization_status.md` for the full write-ups.

## Uncommitted work in progress

`git status` shows two modified files:

- `slimserve/profiles.json` — adds the `mi300x` record to
  `qwen38-q2kxl-1.variants`, retitles the profile ("Qwen3.8-27B GGUF on 1 GPU",
  was "on one Mac" which is wrong for a two-platform id), and adds
  `min_gpus.mi300x = 1` to the `q2kxl` quant.
- `tests/slimserve/test_profiles.py` — the smoke matrix asserted every mi300x
  profile drafts with DSpark; this one legitimately uses DFlash2, so it needed
  an exception next to the existing NVFP4 one.

62/62 tests pass with these changes.

## The actual open problem

I sized the mi300x record from the `qwen38-nvfp4-1` mi300x record (same model,
same card, already tuned) rather than from the Metal record. That changed four
things at once versus the config known to work, and then I guessed at which one
broke it instead of bisecting. That was the wrong method and it cost four runs.

| run | config delta from Metal baseline | result |
| --- | --- | --- |
| `gguf-mi300x` | `gpu_memory_utilization: 0.9`, 262144 ctx, 64 seqs, FULL_DECODE_ONLY | **PASSED**, 268.6 s load |
| `gguf-mi300x2` | fixed 96 GiB KV pool, `max_num_batched_tokens: 8192` | illegal memory access |
| `gguf-mi300x3` | fixed pool, batched tokens back to 2048 | illegal memory access, truncated answer |
| `gguf-mi300x4` | fixed pool, `cudagraph_mode: NONE` | **different** failure: `invalid configuration argument` in `_dummy_sampler_run` → `apply_temperature` |
| `gguf-mi300x5` | `max_num_seqs: 16`, graph capture restored | **was still running at handoff** — read `perf/results/2026-08-25/gguf-mi300x5/smoke.json` |

Raw logs for every run: `perf/results/2026-08-25/gguf-mi300x*/`.

### What is ruled out

- **Memory pressure.** GPU 1 was idle; every run allocated a valid KV cache
  (~1.2M tokens, 4.6x concurrency for 262144-token requests).
- **CUDA graph capture.** Disabling it did not fix the fault, it produced a
  *different, earlier* one. So the decode-path GDN changes from `cd7e983d07`
  are not implicated in the memory fault.
- **The profiling forward.** I claimed this was the discriminator and was
  wrong — the earlier passing `gguf-spec3` run also skipped it (it used the
  Metal record's fixed `kv_cache_memory_bytes`).

### Live hypothesis

Something about the larger shapes trips a kernel launch-configuration limit.
`invalid configuration argument` from `apply_temperature` during sampler warmup
is a bad grid, and the shape driving that is `max_num_seqs` (I raised it 16 →
64). `max_model_len: 262144` may interact. **If confirmed this is likely a real
bug worth fixing rather than a value to retreat from** — but bisect one variable
at a time before concluding anything.

### Fallback that passes

If the bisection stalls and you need a landable record, `gguf-mi300x` passed:
`gpu_memory_utilization: 0.9`, 262144 ctx, 64 seqs, FULL_DECODE_ONLY,
`max_num_batched_tokens` default. Its one flaw is a 268.6 s boot, of which
**142 s is an 8192-token profiling forward** JIT-compiling Triton GDN kernels
(`profile_run: LM dummy run`, boot +80.2 s → +222.5 s). A fixed
`kv_cache_memory_bytes` skips that phase entirely — which is why I reached for
it, and where the trouble started. Do not ship a record that faults, and do not
ship the slow one without saying why in the notebook.

## Verification recipes

Smoke the profile as registered (this is the gate that matters):

```bash
HIP_VISIBLE_DEVICES=1 CUDA_VISIBLE_DEVICES=1 .venv/bin/python -c "
import sys; sys.argv=['smoke','--profile','qwen38-q2kxl-1','--max-tokens','64',
 '--log-dir','perf/results/<date>/<run>','--output','perf/results/<date>/<run>/smoke.json']
from slimserve.smoke import main; sys.exit(main())"
```

Decode/prefill self-consistency (no external oracle; catches decode-path bugs
that prefill-only checks miss — this is what found the fast-path bug):
`/tmp/.../scratchpad/decode_equiv.py` in this session, or re-derive: generate N
tokens incrementally, then re-prefill the growing sequence for each token, and
compare. They must match exactly.

Per-layer parity against llama.cpp:

```bash
# reference (writes ~2700 tensors with sums and corner values)
~/llama.cpp/build/bin/llama-eval-callback -m <gguf> -p "<prompt>" -n 1 --temp 0 -ngl 99 < /dev/null
# ours: env-gated per-layer dump, arm with a prompt starting "The three most", >= 11 tokens
VLLM_QWEN38_DEBUG_DUMP=<dir> .venv/bin/python <script>
```

`llama-cli` needs `--no-conversation --single-turn` with stdin closed; `-no-cnv`
is ignored in this build and it will loop emitting prompts.

## Traps that cost real time today

1. **Sums are permutation-invariant.** They cannot distinguish right-values-
   wrong-order, and on a near-zero residual dominated by cancellation they are
   actively misleading (`linear_attn_out` read 204 vs 360 while the elementwise
   corners matched to quantization noise). I built and published a wrong root
   cause on this and had to retract it. **Compare corner values.**
2. **A missing probe is evidence.** The decode step printing no `conv_out` while
   prefill printed it is what exposed the bypassed fast path. When two paths
   disagree, diff *which code each executes* before diffing numbers.
3. **`pgrep -f <script>` matches the Bash wrapper's own command line.** I hung
   three shells with this today, despite a memory note warning about it. Wait on
   explicit PIDs with `kill -0`.
4. **Don't change four things and then guess.** See the table above.
5. `llama.cpp`'s `attn_output-N` is the GDN core output *before* `out_proj`; the
   counterpart to a module-level hook is `linear_attn_out-N`.

## House rules that bit me

- A profile is **model x quant x platform x config**. Never widen a profile to a
  platform it was not tuned on; add that platform's own record. I used a
  temporary-widening script all session before being corrected — don't.
- Ask the user about facts he already knows (was this artifact validated, what
  did that campaign run) instead of spending GPU runs deriving them.

---

# HANDOFF — GLM-5.3-Flash on 8x A100 (`glm53f-nvfp4-4` / `glm53f-nvfp4-8`), updated 2026-09-10 (supersedes the 2026-09-08 state below)

## 2026-09-10 checkpoint - V2 runner, DFlash2 speculative record

- **Operator directives (2026-09-10):** (1) "Proceed" on TP scaling, MTP
  and the 1M qualification; (2) **V2 model runner only - V1 is
  deprecated** (banner in `vllm/v1/worker/gpu_model_runner.py`, warning at
  the selection point in `gpu_worker.py`; new architectures go in
  `DEFAULT_V2_MODEL_RUNNER_ARCHITECTURES` and V2 gets fixed, never V1);
  (3) try `incoai/GLM-5.3-Flash-DFlash2`; (4) MTP at 3-4 tokens rather
  than 2 (superseded by DFlash2 winning).
- **State:** `glm53f-nvfp4-8` runs on V2 (both GLM architectures are
  default-V2 now; V2 non-spec matches V1: c1 109.6 / c8 490.3) and is
  **speculative by default with DFlash2**: k=3 for 1-16 running requests,
  0 above (`num_speculative_tokens_per_batch_size [[1,16,3],[17,64,0]]`),
  `max_cudagraph_capture_size 256`. Through `slimserve glm53f-nvfp4-8
  --serve`, three warmed repeats: **c1 153.6 / c8 495.0 / c16 687.3 /
  c32 879.5 / c64 1074.5** vs non-spec 109.9 / 494.6 / 676.0 / 929.9 /
  ~1085 on the same tree. Canaries pass. Pool 3,082,532 tokens (the
  drafter costs ~1.2M tokens of pool). `--no-spec` is the non-spec path.
  Committed e6c876c9e. Forced-eviction tier acceptance, WildChat leg and
  the 1M-context leg on this record were launched right after (results
  in `perf/optimization_status.md`, 2026-09-10 entries; raw under
  `perf/results/2026-09-10/glm53f-final-spec/`, `glm53f-leg-spec/`,
  `glm53f-1m-leg/`).
- **How DFlash2 is wired (all committed):** `glm5_next` implements the
  EAGLE-3 aux-hidden-state interface (`_Glm5NextAuxTaps` mixin, listed
  BEFORE `SupportsEagle3` in the bases because the protocol ships concrete
  defaults); a tap at layer k is `glm5_mhc_post(mlp_out, residual,
  post_mix, res_mix).mean(dim=1)` - the mean over the four mHC streams of
  the layer's completed output (SGLang PR 36708 `hc_contract`). Taps
  (6, 15, 25, 34, 43) in the +1 convention. The drafter shares the
  target's embed_tokens/lm_head via `get_language_model()`. Source
  speculator on `glm53f-nvfp4` (revision bf582e4e); the V1-only
  checkpoint-MTP adapter is no longer registered (its code remains).
- **DFlash contract traps:** the V2 DFlash speculator drafts a FIXED
  block: per-batch schedules may pair k with 0 only (runner skips
  drafting); any partial k asserts. The compact indexer cache caps
  speculation at 5 tokens (8-row raw ring), so block-8's k=7 needs a
  ring-geometry change; k=5 gave +1.6% c1 over k=3 for -8% c8.
  c1 throughput is acceptance-luck at temperature 1.0 (per run 1.5-3.2);
  compare STEP RATES (tok/s / acceptance length): ~72 steps/s
  speculative vs ~110 plain in every arm, independent of capture size or
  schedule.
- **MTP (checkpoint head) findings, V1-only, for the record:** under
  FULL_DECODE_ONLY the drafter got NO graphs (proposer keys off the mixed
  mode) and ran eager; under FULL_AND_PIECEWISE k=3 gave +6.5% c1 and lost
  c8+; a speculative step still carried a fixed ~7 ms. Superseded.
- **TP scaling:** c1 is latency-bound (97% GPU util at 181 W; async
  scheduling on; ~1,500 kernels/token, ~850 on the main stream): the
  TP-invariant residue is the per-kernel latency floor, so the 1.5x
  TP8/TP4 ratio gate at c1 is a kernel-count problem, not a collective
  one; MTP/DFlash2 (verify batching) is what lifted c1. No ownership
  work was started. TP4 (`glm53f-nvfp4-4`) has not been re-measured on
  today's tree and still carries an unbooted tier config.
- **Scratch tooling (`~/.local/scratch/glm53/`):** `launch_profile.py`
  (registered argv + `--spec --k --schedule --additional --cudagraph-mode
  --capture` overrides), `dflash_arm.sh` / `mtp_arm.sh` (boot + validate +
  acceptance), `sweep_summary.py`, `final_gates.sh` (record flip + all
  gates), `leg_1m.sh` (two sessions to 1,040,000 tokens with
  `--require-target`, tiers + verify on), `census_launch.sh`.

# HANDOFF — GLM-5.3-Flash on 8x A100 (`glm53f-nvfp4-4` / `glm53f-nvfp4-8`), updated 2026-09-08

## One-paragraph state

GLM-5.3-Flash (`glm5_next`: 34 KDA linear layers + 11 pooled-indexer sparse
MLA layers + mHC residual streams + 288-expert NVFP4 MoE) serves on 8x A100
through `slimserve glm53f-nvfp4-8 --serve`. The record is maximized for this
box: model-default 1,048,576 context on three KV tiers (VRAM 3.17M tokens,
72 GiB/rank pinned host, 256 GiB/rank disk), `max_num_seqs` 64, EP off.
Text + image + tool canaries pass, the forced-eviction tier acceptance is
clean (0/106 byte mismatches, promotion from disk exercised), and the
WildChat deep-context leg passes (33/33 recall to 204K, 0 errors). Everything
described here is committed and pushed to `main`. The campaign's live problem
is **TP scaling**, which is below the repo's hard gate — see Open items #1.

**Naming:** the Flash records are `glm53f-*`. Plain `glm53` reads as the 743B
GLM-5.3, which shares GLM-5.2's dense-MLA architecture and is a different
model. Renamed 2026-09-06 (operator). **Raw artifact paths under
`perf/results/` predate the rename** and are still spelled `glm53-nvfp4-*`.

## Measured baselines (all through the real profiles)

Harness: `benchmarks/benchmark_dsv4_exact.py`, 1000 in / 300 out,
temperature 1.0 / top-p 0.95 / top-k 20, seed 42, warmed per concurrency.
Aggregate output tok/s.

| record | c1 | c8 | c16 | c32 | c64 |
|---|---|---|---|---|---|
| `glm53f-nvfp4-8` (maximized, 2026-09-06) | 83.8 | 402.6 | 562.1 | 750.0 | 931.9 |
| `glm53f-nvfp4-4` (2026-09-03, pre-tier)  | 73.8 | 332.1 | 464.6 | -- | -- |
| `glm52-q2k-8` (2026-09-06)               | 72.5 | 166.5 | 223.1 | -- | -- |
| `glm52-q2k-4` (2026-09-06)               | 31.7 | 93.7  | 115.6 | -- | -- |

GLM-5.3-Flash beats GLM-5.2 at every concurrency (+16% c1, ~2.5x c8/c16);
that is the architecture (only 11 of 45 layers touch KV) plus NVFP4 Marlin
MoE vs Q2_K. Raw: `perf/results/2026-09-06/` and `2026-09-03/`.

## Open items, ranked

1. **TP scaling is below the repo's hard gate — the live problem.**
   CLAUDE.md requires TP8 >= 1.5x TP4. Measured: 83.8 vs 73.8 at c1 = **+14%**
   (c8/c16 ~ +21%). GLM-5.2 on the same box is a healthy 2.3x, so this is
   specific to `glm5_next`, not the machine. Cause is measured, not guessed:
   per-rank work that does not shrink with rank count. TP8 profile at
   1000-token context (`perf/optimization_status.md`, 2026-09-03 entry) puts
   the TP-invariant residue at ~3.3 ms of ~12 ms/token: mHC 1.35, pooled
   indexer 0.72, custom allreduce 0.65, `direct_copy` 0.53.
   Ranked fixes (gain / effort), already written up in the notebook:
   1. ~150 `direct_copy` launches/token — `contiguous()`/view copies around
      the mHC op wrappers and q/idx staging in the sparse backend (~0.5 ms, easy).
   2. Pooled indexer at 65 us for a 250-pool context, should be ~15 us
      (q/APE preload dominates; one tile per program) (~0.5 ms).
   3. mHC **channel ownership** as DSV4 does it (`VLLM_DSV4_TP_OWNERSHIP`:
      each rank transitions 4096/TP channels, transition fused into the
      allreduce) — up to 1.2 ms, deep integration.
   4. M=1 GEMV chain: 280 cuBLAS launches/token is launch-bound; fusable per
      layer (q_a+kv_a already fused; KDA's in_proj/f_b/g_a/g_b are four).
   **Do NOT re-try the plain fused allreduce+mHC transition op** — it was
   implemented, measured (85.5/421 vs 84/413 = noise) and REJECTED 2026-09-03,
   because the mHC math stays replicated on every rank. DSV4's win comes from
   ownership (#3), not from merging the two kernels.

2. **`glm53f-nvfp4-4` carries an unvalidated tier config.** The record now
   states `host_tier_gb_per_rank: 72` / `nvme_tier_gb_per_rank: 256` (bumped
   alongside the -8 record on 2026-09-06) but has **never been booted with
   them** — its only profile validation (`perf/results/2026-09-03/
   glm53-nvfp4-4-baseline/`) predates the tier being enabled on it at all.
   It is also still at `util 0.85` / `max_num_seqs 16` (maximizing was scoped
   to the -8 record). Next: boot it, run the canaries + exact bench, and run
   the eviction acceptance; then decide whether to maximize it too.

3. **MTP speculative decoding is unported.** The checkpoint ships an MTP head
   at layer 45; `glm5_next.py` skips it at load. The upstream GLM-5.3 recipe
   runs `--speculative-config.method mtp --num_speculative_tokens 5`. This is
   the largest single-stream win still on the table. Follow the DSpark/MTP
   precedent already in-tree for other profiles.

4. **Tier restores are not exercised by the deep-context leg.** With 8
   sessions at ~180K against a 3.17M-token VRAM pool, nothing evicts, so the
   leg shows 228 boundary-state saves and ZERO restores (same as GLM-5.2's
   TP8 leg). That is expected, not a bug. Restore evidence comes only from
   `benchmarks/benchmark_kv_tier_eviction.py`, which forces eviction. If you
   want restores under a realistic leg, shrink the pool or raise session count.

5. **Follow-ups inherited, still open:** window-tail staging for DSV4's
   sliding-window groups (its tier stays write-only by design until then);
   MI300X connector generalization (issues #17/#18).

## Verification recipes (copy-paste)

```bash
# Fast gate after ANY change (no GPU contention, ~25 s total)
cd ~/SlimServe
.venv/bin/python -m pytest tests/slimserve/test_profiles.py -q          # 63
CUDA_VISIBLE_DEVICES=7 VLLM_KV_TIER_VERIFY=1 .venv/bin/python -m pytest \
  tests/v1/core/test_kv_tier_index{,_disk}.py \
  tests/v1/core/test_host_tier_connector{,_disk}.py \
  tests/v1/worker/test_kv_tier_{dma,dma_disk,nvme}.py \
  tests/v1/worker/test_kv_residency.py -q                               # 60
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest \
  tests/kernels/test_quixicore_sparse_mla_bf16.py \
  tests/glm5_next/test_pooled_indexer_parity.py -q                      # 14

# Full profile validation (boot + text/image/tool canaries + exact bench)
bash ~/.local/scratch/glm53/validate_today.sh glm53f-nvfp4-8

# Tier acceptance (the ONLY thing that proves restores)
export VLLM_KV_TIER_VERIFY=1 VLLM_KV_TIER_LOG_MISS=1 \
       SLIMSERVE_KV_TIER_DIR=/home/ubuntu/.local/scratch/kv-tier
bash ~/.local/scratch/glm53/tier_accept.sh <tag> <pool_tokens> 40000
# PASS = "0/N restores mismatched" AND non-zero "host-tier: hit for" lines.

# Deep-context leg (1.25 h)
RESULTS_DATE=$(date +%Y-%m-%d) RESULTS_TAG=glm53f-leg \
  bash ~/.local/scratch/a100-sweep/run_one.sh glm53f-nvfp4-8
```

## Traps that cost real time in this campaign (each one burned hours)

- **Rebuild `_quixicore_C` after any merge that touches `csrc/`.** A merge
  changed `post_update`'s pybind signature; the stale `.so` killed a boot at
  the sampler with "incompatible function arguments".
  `cmake --build build/temp.linux-x86_64-cpython-312 --target _quixicore_C -j$(nproc)`
  then copy the `.so` into `vllm/`.
- **Raw `api_server` launches need `--enable-prefix-caching` explicitly.**
  vLLM defaults it **OFF** for hybrid (mamba/GDN) models. A tier acceptance
  "passed 6/6" with ZERO tier activity — full re-prefill. `slimserve --serve`
  always passes it; check `vllm:cache_config_info` in `/metrics`.
- **Never A/B a compiled path with a private env var.** The compile cache key
  hashes `envs.compile_factors()` (a fixed VLLM_* list) + config + traced file
  contents. An env-gated branch inside a compiled forward does **not** change
  the key, so the "off" arm silently loads the "on" arm's graph. A/B via
  distinct code (git stash / worktree) instead. See memory
  `compile-cache-ab-hazard`.
- **Marker recall alone NEVER proves the tier.** Evidence is the connector's
  hit/restore/promotion counters plus `kv-tier verify` lines. Also: at
  temperature 1.0 the model may *refuse* the planted markers on policy
  ("I didn't store that code") — check `in_reasoning` and the reply text
  before calling it corruption.
- **The verify tool itself was wrong four times** while the data was bit-exact
  (host-slot-keyed digests, submission-time digests that predate the copy
  event, live-length lookup racing the same way, sha1 vs sha256 across the DMA
  and IO thread). Before trusting a mismatch count, run the 3-second
  `tests/v1/worker/test_kv_tier_dma_disk.py::test_verify_digest_survives_promotion`.
- **`safekill`, never `pkill -f`.** `~/.local/scratch/bin/safekill <pattern>`;
  a `-f` pattern your own command contains matches the shell wrapper and kills
  the turn. For GPU teardown, `nvidia-smi --query-compute-apps=pid` + kill by PID.
- **Never flatten a packed KV view.** `reshape(-1)` on the packed cross-layer
  slab silently COPIES the whole layer cache per call (this was costing
  GLM-5.2 most of its throughput). Address pages by block stride; the sparse
  entry points take `page_stride_bytes`.
- **Warm every concurrency before measuring.** An unwarmed arm read TP8 at
  10 tok/s purely from cold Triton autotune inside the measured window.

## Design notes worth reading before touching the tier

Hybrid page-size unification widens the smaller-page group's block: the
indexer group (512 B/token) gets 2176-token blocks beside the MLA group's
1088 at TP4 (576/1152 at TP8) while the scheduler hashes at the gcd. The
connector therefore keeps **per-group block ratios** — a group is staged and
restored only where its block completes, and resume boundaries align to the
ratios' lcm (`HostKVTierIndex.due(i)`, `_resume_align`). A 58-op restore at
TP4 = 36 MLA + 18 indexer + 4 KDA state pages; 106 at TP8 = 68 + 34 + 4.

## Scripts and raw artifacts

- Scripts: `~/.local/scratch/glm53/` — `validate_today.sh` (profile boot +
  canaries + bench), `tier_boot2.sh` (raw boot, `HOST_GB`/`NVME_GB`),
  `tier_accept.sh`, `max8_validate.sh`, `proflong.sh` (torch profiler at real
  context), `write_max_records.py`, `bench_nope.py` (kernel microbench).
- Leg runner: `~/.local/scratch/a100-sweep/run_one.sh`.
- Raw results: `perf/results/2026-09-0{3,4,6}/` (pre-rename `glm53-*` names).
- Notebook: `perf/optimization_status.md` (2026-09-03 through 09-06 entries);
  baselines: `perf/baseline_status.md`.
- Memory files that matter here: `glm53-flash-bringup`,
  `kv-tier-hybrid-lessons`, `compile-cache-ab-hazard`, `safekill-not-pkill`.

## 2026-09-08 implementation checkpoint (supersedes the open-work snapshot above)

- **ZG is CPU-only at the user's request.** Global defaults and all six
  workspace index manifests now use CPU; no model/schema change or full
  re-index was needed. The daemon has CUDA visibility disabled, affinity
  122-125, one embedding context, and an installed-backend patch to respect
  CPU affinity when choosing threads. NVIDIA reports only inference workers.
  An npm upgrade can overwrite the backend patch. MCP clients need reload
  after this daemon replacement; `zg query --mode server --refresh off`
  works meanwhile. GitHub issue creation returned 403; report is saved at
  `/home/ubuntu/.local/scratch/zg-bug-reports/04-cpu-affinity-threads.md`.
- **Implemented KDA output-norm fusion on CUDA.** The norm sits inside
  opaque `kda_attention`, where the default decomposed CustomOp dispatch
  cannot be fused by the enclosing compiler. Explicitly using its existing
  Triton CUDA kernel removes 102 direct-copy launches per decode token
  (145 -> 43), plus decomposed arithmetic. This supersedes the earlier
  guess that the copy class mainly came from mHC `.contiguous()` wrappers.
  ROCm dispatch, weights, profiles and KV layout are unchanged.
- Clean real-profile A/B, 1000 input / 300 output tokens, three warmed
  repeats per concurrency, median aggregate tok/s:

  | Profile | c1 before -> after | c8 before -> after | c16 before -> after |
  | --- | ---: | ---: | ---: |
  | `glm53f-nvfp4-4` | 75.59 -> 80.79 | 335.90 -> 349.57 | 469.68 -> 483.96 |
  | `glm53f-nvfp4-8` | 83.82 -> 90.30 | 402.07 -> 424.74 | 561.51 -> 584.44 |

- Both profiles pass text/reasoning/image/tool canaries and exact-token
  checks. Final fast gates: **174 passed**, including 24 output-norm and
  13 repaired pooled-indexer tests. TP4 now boots its registered 72 GiB
  host / 256 GiB disk tiers per rank, with 1,292,330 GPU-pool tokens.
- **Forced eviction passed on September 9.** Six markers recalled, six
  host-tier hits resuming at 39,168 tokens, zero mismatches across 1,392
  per-rank restore operations. Current TP4
  server PID 3089864, port 8400; benchmark uses the actual pool, 40K target,
  six markers and eight churn streams. Completed artifacts:

  ```bash
  tail -n 20 perf/results/2026-09-08/glm53f-tp4-kda-fused/tier/acceptance.log
  rg 'host-tier: hit for|restores mismatched' \
    perf/results/2026-09-08/glm53f-tp4-kda-fused/server.log
  ```

  No fresh disk-promotion acceptance is claimed. The full 1.25-hour WildChat leg and
  TP8 c32/c64 have not been rerun on this change.
- **Scaling is still wrong:** TP8/TP4 is only 1.118x / 1.215x / 1.208x;
  the 1.5x minimum is not met. Channel ownership, projection fusion and
  GLM-specific MTP remain open. A one-pool/program indexer experiment is
  quarantined under `benchmarks/`, not serving: it helps c1/1000-token
  microbenchmarks but regresses longer/batched shapes and failed one long
  logit-tolerance case. Do not promote it as the indexer solution.
- Use new `benchmarks/validate_glm5_next.py` against an already-running
  SlimServe server. Unlike the historical scratch runners it asserts
  canaries, refuses artifact reuse and does not kill unrelated GPU PIDs.
  Boot with the registered profile, then pass
  `--tokenizer /home/ubuntu/models/GLM-5.3-Flash-NVFP4 --out <fresh-dir>`.
  Wait for owned workers to fully exit before booting another TP profile.
- Raw data: `perf/results/2026-09-08/glm53f-tp{4,8}-kda-fused/`, clean
  reference folders and `glm53f-kda-norm/`. Full commands, repeat spreads,
  trace evidence, rejected experiments and caveats are in
  `perf/optimization_status.md`; stable throughput snapshot is in
  `perf/baseline_status.md`. This is a measured incremental win, not
  completion of the full optimization campaign.

## 2026-09-09 active checkpoint (supersedes September 8 live-process state)

- September 8's recorded server PIDs are no longer live. ZG remains CPU-only.
- Implemented and short-workload validated on both registered profiles:
  merged KDA g_a rows, paired f_b/g_b projection, and new split SIMT mHC
  transition with RMSNorm after the required BF16 rounding boundary. No
  quantized-weight repack or persistent duplicate cache. The failed padded
  TF32 mHC and single-pool indexer prototypes remain diagnostic-only.
- Non-speculative median tok/s, three warmed 1000-in/300-out repeats:
  TP4 c1/c8/c16 **86.71 / 365.29 / 501.80**; TP8 c1/c8/c16/c32/c64
  **97.08 / 445.12 / 611.13 / 790.03 / 970.80**. Text/reasoning/image/tool
  canaries pass on both. TP8/TP4 remains 1.120x/1.219x/1.218x: the scaling
  campaign is unfinished. Latest kernels still need long-workload acceptance.
- Fast combined gate before extra batch cases: 367 passed. Expanded mHC/
  paired-gate suite: 179 passed, adding 26 cases. MTP/CLI CPU contracts:
  69 passed, followed by four config/loader tests after adding the draft
  layer-count converter. These are not MTP serving acceptance.
- MTP adapter is now implemented behind **`--spec`**, using the source's
  existing registered k=1. Both GLM profiles remain non-speculative by
  default. It uses ordinary residual GLM NoPE sparse MLA + MoE, not mHC;
  its checkpoint experts are **block-128 FP8**, not the main model's NVFP4.
  Preserve root mixed-quant config, one-layer metadata, shared embeddings/
  head, and expanded pool/tail index width. Cross-draft index reuse is
  deliberately disabled until its state semantics are validated.
- MTP attempts02-04 exposed and fixed target-vs-draft config selection,
  GLM image-token dispatch, and independent MLA/indexer cache ownership.
  The dedicated proposer has separate persistent per-group slot mappings,
  correct group block tables, and explicit logits/recycle tuple handling.
  CPU contracts including Qwen precedent tests: 80 passed.
- Attempt05 passes loading, compilation and graph profiling but fails the
  full-context capacity check: 18.12 GiB KV needed versus 17.38 available
  at TP4 utilization 0.85. No MTP health or canary result yet. TP4 profile
  utilization is now 0.90; full context remains 1,048,576.
- A separate router correction enables direct BF16-input/FP32-output cuBLAS
  on SM80 H4096/E288 instead of BF16 logits followed by a cast. 24 targeted
  GPU tests pass, including changing-input CUDA graphs. The throughput
  snapshot above predates this precision correction and is historical.
  Current fresh non-spec reference is
  `perf/results/2026-09-09/glm53f-tp4-fp32-router-reference/`, port 8400.
  Check health/logs/process state before continuing. MTP must be compared
  against this corrected reference at the same profile memory budget.
- Raw non-spec A/B and traces: `perf/results/2026-09-09/glm53f-tp{4,8}-simt-mhc/`.
  Next: finish MTP bring-up or clearly quarantine a failed path, measure
  acceptance/rejection and real TPS, then run the full long-context and tier
  gates on the retained candidate. Channel ownership remains open.
- QuixiCore port is pending. Read-only review found no canonical operation
  for these kernels and stricter umbrella tolerances than the current
  model-specific tests. Do not silently loosen that contract or relabel
  Apache-2.0 kernel sources as MIT.

### 2026-09-09 later checkpoint (supersedes the live state above)

- Corrected non-spec TP4 reference finished: **87.40 / 368.44 / 502.91**
  median tok/s at c1/c8/c16, full context/utilization 0.90, all canaries and
  exact-token checks passing. Raw `glm53f-tp4-fp32-router-reference/`.
  Corrected TP8 has not been measured; the earlier matched TP4/TP8 matrix
  still demonstrates failure of the 1.5x scaling gate.
- MTP attempt06 reaches health at full context with a 1,234,509-token pool.
  Canaries/exact checks pass and draft counters are nonzero. Medians
  **92.23 / 357.81 / 486.43**: c1 repeat ranges overlap the reference,
  c8/c16 regress 2.9%/3.3%. **Do not promote k=1 to the defaults.**
- Greedy-repeatability diagnostics found different outputs/logprobs on MTP,
  corrected non-spec, native-mHC-only A/B, **and untouched c61ccdbb9**.
  The latter comparison is `glm53f-tp4-start-reference-repeatability/
  attempt03/`; attempts01/02 failed only on missing worktree runtime
  dependencies. Non-repeatability predates this pass, but its root cause
  and sensitive quality parity remain open. Do not call it solved or use
  greedy text equality as an already-established baseline guarantee.
- The temporary SIMT-disable diagnostic is removed. New SIMT and norm
  fusion are limited to SM80; other platforms retain explicit norm calls.
  Latest tests: 368 GLM/router/profile/sparse-MLA, 43 host-tier/index CPU,
  then 148 SM80 transition and 10 platform/MTP contracts. These overlap;
  do not add them into a unique-test total.
- Test servers from these experiments have been stopped. Before another
  boot, inspect process/GPU state. ZG stays CPU-only. The detached starting
  revision remains at `/home/ubuntu/.local/scratch/glm53f-start-reference`,
  with unchanged native/FlashAttention dependency symlinks; its tracked
  source is clean. Main worktree changes are uncommitted.
- Next real-profile command: `.venv/bin/slimserve glm53f-nvfp4-8 --serve
  -y --host 127.0.0.1 --port 8400`. Use a fresh artifact directory and
  `benchmarks/validate_glm5_next.py --tokenizer
  /home/ubuntu/models/GLM-5.3-Flash-NVFP4 --out <fresh>/validation
  --concurrency 1 8 16 32 64` for the corrected TP8 reference. Then finish
  sensitive quality/long-context/forced-eviction gates on the retained code.
  Channel ownership and the QuixiCore port remain unfinished.

### 2026-09-09 TP8 cache campaign checkpoint

- Corrected TP8 baseline is now measured: c1/c8/c16/c32 medians
  **98.99 / 448.56 / 607.58 / 804.05** tok/s, canaries/exact checks pass,
  3,172,516-token pool at 50.10 GiB KV/rank. Raw:
  `perf/results/2026-09-09/glm53f-tp8-corrected-reference/`.
- No serving process is left from that baseline. ZG is CPU-only. Goal
  remains active: optimize TP8 performance and KV at c1/c8/c16/c32.
- New compact pooled-indexer prototype is **not enabled in serving**:
  `vllm/model_executor/layers/glm5_next_pool_cache.py`. Cache completed BF16
  four-token pools plus an eight-row page-owned raw ring, logical row64
  instead of256 BF16 elements (4x smaller indexer allocation; aggregate
  capacity must be measured). Enforce at most five speculative tokens.
- Single-pool arithmetic variants failed tight parity near BF16 rounding
  boundaries and were removed. Matching the old scorer's tiled pooling
  passes 69 unit cases at original tolerance; exact pruned top-k checks
  and update+score microbenchmarks follow. See current performance notebook
  entry and `glm53f-compact-pool-cache/` raw artifacts.
- Next: gate parity/timing, integrate behind compiler-hashed additional
  config, then fresh registered-profile A/B and actual long/tier validation.
  Do not present a cache-size estimate or kernel microbenchmark as an
  end-to-end result. Scaling, sensitive quality and QuixiCore port stay open.

### 2026-09-09 compact-cache live checkpoint (supersedes preceding process state)

- Compact flag is now integrated and enabled in the working TP8 registry
  for validation, not yet long/tier qualified. TP4 remains raw-cache.
  GPU/config/HF selection gate: 101 passed; profile contracts: 65 passed.
- Candidate `glm53f-tp8-compact-pool/attempt01/`: canaries/exact checks pass;
  c1/c8/c16/c32 medians **99.07/473.79/660.65/893.77** tok/s, versus
  corrected raw **98.99/448.56/607.58/804.05**. c1 effectively unchanged,
  concurrent gains +5.6%/+8.7%/+11.2%. Actual KV pool **4,225,909 tokens**,
  +33.20% at the same 50.10 GiB/rank. All timed serving sources hash-checked.
- **Live API PID3316656, engine3317413**, port8400, workers3317655..3317662.
  Do not start another engine, edit serving source, or overlap GPU tests.
  Full c8 WildChat run is started after the short timings, targeting1M
  context for1.25hours; artifacts `<attempt01>/deepcontext_c8.{log,json}`.
  Benchmark PID3329002 (exec session62063); serving exec session7030.
  Check its process/log before launching anything. It is not a pass yet.
- Next: finish that exact long gate, then forced eviction/restore using
  pool4225909 and verify actual host/disk restore counters and byte checks.
  Kernel micro parity/simulated restore do not replace this. Scaling,
  sensitive quality, further decode optimization and QuixiCore port remain
  open; persistent performance/KV goal is not complete.

- Subsequent CPU-only work while that same live run owns all GPUs:
  ratio2/ratio8 incomplete-page/resume and disk promotion tests pass
  (**47 combined host/disk connector/index tests**). Raw:
  `glm53f-compact-pool-cache/ratio8-scheduler-disk.log`. No physical restore
  pass implied. Serving source hashes remain unchanged.
- Next decode prototype is quarantined under `benchmarks/`:
  `glm5_next_mhc_deferred.py` and its `benchmark_...` runner. Splits urgent
  pre-mix/RMSNorm from deferred Sinkhorn, overlaps the latter with real
  KDA/router projection, joins before returning. Both phases compile
  offline SM80; GPU parity/replay/timing not run. Do not enable yet.
  Stream-aware reference trace census is in `glm53f-mhc-deferred/` and
  reproducible with `benchmarks/summarize_glm5_next_trace.py`.

- Isolated next candidate now exists at
  `/home/ubuntu/.local/scratch/glm53f-next-candidate`: full current source
  snapshot plus **shared indexer logits scratch**. Explicit lazy model
  owner, no KV/weight sharing, nominal640MiB/rank saving. Additional config
  `glm5_next_shared_indexer_scratch` is enabled only there. Not GPU/live
  validated and not applied to main. CPU suite90passed/10CUDA-skipped;
  final ownership12passed. Artifacts and checked incremental patch:
  `perf/results/2026-09-09/glm53f-shared-indexer-scratch/`.
  Use candidate `.venv/bin/python -m slimserve.cli`, NOT its symlinked
  console script (the latter imports original editable source).
- Current compact-only long process3329002 remains running on the original
  server; last verified at32minutes elapsed, no logged server errors.
  Preserve it. After long/tier qualification, run the candidate's
  `tests/glm5_next/test_indexer_workspace_gpu.py` before any sharing A/B.

### 2026-09-09 06:06 UTC: long gate finished, forced restore running

- Supersedes preceding benchmark state: long benchmark3329002/session62063
  finished successfully. **633 turns, zero errors, 100/100 recall**,76.08min;
  max context530222, median509100.5. All8sessions stopped on time cap,
  NOT the1M target. DMA restore count remained0: no tier pass implied.
- Same frozen compact server3316656/engine3317413 still owns all8GPUs.
  New restore harness **session25585** uses realpool4225909, factor1.75
  (target7395341 actual filler tokens), c8fillers55K and six natural
  markerplants40/50/60/70/80/90K. Artifacts
  `glm53f-tp8-compact-pool/attempt01/tier_acceptance.{log,json}`.
  Check completion, physicalhost/diskreads, recall and byte verification
  before stopping server or launching candidate GPU tests.
- Harness now accounts actual server prompt usage and records fillers;
  12CPU tests pass. Main serving/profile source hashes unchanged.
- Added NVFP4 packed-layout CPU contracts (4pass/3CUDApending), not a
  serving kernel. Scratchworktree and mHC-overlap GPU gates still pending.
- Performance goal remains active; c1 flat and TP scaling below1.5x.

- Later CPU work: TP4/TP8 stream-aware census confirms essentially equal
  mHC1.223/1.216ms and worse allreduce .619/.723ms, despite smaller expert
  GEMMs. Raw TP4 census under`glm53f-mhc-deferred/`.
- Prepared pool scorer tile/program sweep (`benchmark_glm5_next_pool_cache.py
  --tune-score`), four tile shapes compile offline; no GPU timing yet.
  Existing scorer already bounds its program count; do not reinvent that.
- Added isolated singleton align experiment: immutable padding metadata
  plus current router IDs removes general expert sorting for M1/non-EP.
  Twelve CPU tests pass; actual Marlin/changed-route graph parity pending.
  `benchmark_glm5_next_singleton_align.py` cycles16 disjoint expert sets
  for timing to avoid L2-only numbers. No serving changes.
- Restore harness PID3358956/session25585 remains running; last progress
  64fillers/4194141tokens of7395341target. Frozen source hashes still pass.

### 2026-09-09 06:34 UTC: physical tier pass; server stopping for GPU gates

- Restore harness3358956/session25585 completed: **PASS6/6**,18/18 recalled
  after7,405,558 actual filler tokens. Six NVMe promotions;825 disk reads
  and825 restores per rank,6600 total,zero reported mismatches. Matched
  every issuance/verification record. Raw`attempt01/tier-mechanism-summary.json`.
- Compact-only source hashes still pass. API3316656/engine3317413 stopped;
  exact workers3317655..3317662 are finishing CUDA/disk teardown. Confirm
  no GPU processes before starting pending tests. Last nvidia-smi wait
  session53108 was still finishing; do not treat teardown as GPU-idle.
- Next: GPU regression/layout gates and isolated candidate scratch tests;
  then mHC overlap/scorer sweep/singleton alignment timings, separately.
  No new candidate applied to main. Full1M, scaling and goal remain open.

### 2026-09-09 06:50 UTC: scratch-only A/B booting; next candidates measured

- LIVE: detached scratch worktree API3377788/engine3378642, workers start
  at3378919. Serving session3318; validation PID3377872/session21202.
  Port8400; raw`glm53f-tp8-shared-scratch/attempt01/`. Only scratch-sharing
  flag differs from retained compact baseline. Source hashes/cwd verified.
  Preserve that worktree and shared binaries, no competing GPU tests.
- All old compact workers are gone. Main GLM suite412pass, then430pass
  after new opt-in singleton dispatch. Native repack3GPU contracts pass.
- ScratchGPU tests10pass after diagnosing unstable native top-k ordering:
  private-vs-private also reorders; test now requires bit-exact logits and
  identical sets, plus per-layer snapshots before overwrite. Original
  failures and control evidence retained. Updated checked incremental patch:
  `glm53f-shared-indexer-scratch/candidate-only-replay-controls.patch`.
- Main now has optional singleton alignment helper/Marlin wiring, enabled
  by`glm5_next_singleton_marlin_alignment`; **all profiles leave it OFF**.
  CPU30pass, fullGLM430pass; integrated public native MoE micro41.97→35.69us
  with exact changing-route graph parity. NOT in live scratch worktree.
- mHC overlap8shapes parity/replay pass, saves~3–4us per subgraph; not
  integrated in serving. Scorer192configurations parity pass; larger tiles
  help long contexts but regress small ones. Needs context-adaptive dispatch,
  not blanket change. Detailed raw paths/numbers in performance notebook.
- Next: finish scratch-only canaries/TPS/KV-capacity A/B. Then independently
  validate singleton serving opt-in and integrate measured mHC overlap;
  keep full1M quality, c1/scaling, long workloads and QuixiCore port open.

### 2026-09-09 07:06 UTC: scratch retained; singleton A/B now owns main/GPU state

- Scratch-only A/B finished: all canaries/exact checks pass, c1/c8/c16/c32
  medians **99.834/473.828/657.642/894.580** tok/s (effectively flat).
  Actual KV **4,278,924 tokens**,50.72GiB/rank:640MiB/rank reclaimed,
  +53,015 tokens/+1.2545% beyond compact-only. Retained for memory.
  Raw`glm53f-tp8-shared-scratch/attempt01/comparison.json`.
- Applied tested scratch changes to MAIN and updated TP8 profile notes.
  Main integration **107 CPU/profile tests +452 GLM GPU tests pass**.
  Old scratch API3377788 and its workers are gone; that worktree is idle.
- LIVE **MAIN TREE** singleton-alignment validation candidate:
  API3397536, engine3398115, workers3398359 onward, port8400.
  Serving session99280; harness3397610/session48367 waits for health then
  canaries and c1/c8/c16/c32 three repeats. Raw
  `glm53f-tp8-singleton-align/attempt01/`. Source hashes verified.
  **Main serving sources/profile and shared binaries are frozen now.**
- `glm5_next_singleton_marlin_alignment` is now TRUE in the working TP8
  profile solely for this real validation candidate; latest state supersedes
  prior OFF notes. Do not call it retained until A/B results justify it.
  Baseline is the completed shared-scratch-only run above, not raw cache.
- mHC overlap remains benchmark-only. Potential integration: model-owned
  stream, opaque transition+consumer projection, explicit join before any
  output escapes. KDA currently projects at its forward entry; allow a
  precomputed projection. DeepseekV2MoE.forward currently computes its gate
  internally; an external router result needs guarded plumbing for non-SP,
  non-internal-router paths. New allocations/capture lifetimes require
  their own GPU tests; persistent-buffer prototype alone does not prove it.
- Scorer adaptive dispatch remains unimplemented. Larger tiles are not a
  universal win. Full1M context/combined long-tiers, TP scaling, deeper
  kernel work, and the QuixiCore port remain open. Goal stays active.

### 2026-09-09 07:25 UTC: singleton retained; mHC overlap live A/B booting

- Singleton A/B finished: c1/c8/c16/c32 medians101.880/475.342/660.040/896.848,
  +2.049% c1 vs shared-scratch baseline, higher concurrencies effectively
  unchanged. Exact token counts/canaries pass; all source hashes match.
  Retained singleton flag. API3397536 and every worker exited; nvidia-smi
  confirmed no compute processes before subsequent GPU tests.
- Opaque fresh-allocation mHC projection candidate passed20tests and retains
  ~3–4us/subgraph savings over the original transition+projection. Integrated
  owned `glm5_next_mhc_project.py` with model/PP-owned stream, explicit join,
  strict SM80/BF16/H4096/HC4/noLoRA/unquantized-linear guards; optional KDA
  projection and MoE router-logits plumbing. Main GLM suite483passed.
  See `glm53f-mhc-deferred/` logs and notebook for initial fixture failures.
- LIVE MAIN TREE A/B: API3414693, server session72809, harness3414768/
  session91444, port8400. Raw`glm53f-tp8-mhc-overlap/attempt01/`.
  New flag `glm5_next_mhc_projection_overlap` TRUE in TP8 solely as a
  validation candidate. Baseline is retained singleton, not shared scratch.
  Registered profile otherwise unchanged; exact c1/8/16/32 three repeats.
  Nine source hashes saved. Freeze main serving sources/profile and shared
  native binaries, no competing GPU tests while this A/B runs.
- Check boot log for actual enabled KDA/router-site counts, then health,
  canaries, exact JSONs and KV capacity. Do not infer retention from startup.
  Adaptive scorer still unimplemented; full1M/combined tiers, c1/scaling,
  deeper kernels and QuixiCore port remain open. Goal stays active.

### 2026-09-09 07:47 UTC: KDA overlap retained; full mHC router A/B booting

- KDA-only A/B completed: medians c1/c8/c16/c32
  **102.763/476.888/657.832/899.056**, +0.867% c1 vs singleton; higher
  concurrencies within run spread. All exact1000/300 counts/canaries pass.
  Pool4,280,453 tokens/50.74GiB per rank. API3414693 and all workers exited,
  and nvidia-smi confirmed no compute processes before subsequent GPU tests.
- New owned `prepare_router_projection` uses MoERunner's existing external-
  gate interface, removing only its identical gate alias during construction.
  Parent MoE keeps weights/loader names; reject SP/transforms/shared-gate
  fusion/DBO/unsupported linears. New strict additional-config
  `glm5_next_mhc_router_projection_overlap` requires mHC overlap flag.
  No new runner API and no shared-expert sync bypass. CPU/profile76pass;
  complete GLM GPU suite **498pass** before boot.
- LIVE MAIN TREE: API3439513, engine3440417, workers3440672..3440679.
  Server session92894; harness3439584/session23493; port8400. Raw
  `glm53f-tp8-mhc-router-overlap/attempt01/`. All nine source hashes match.
  Actual boot confirms **34 KDA layers and42 router sites on all8 ranks**.
  Router flag TRUE solely as validation candidate; KDA-only checkpoint is
  its baseline. Finish health/canaries/exact c1/8/16/32 three repeats before
  retention decision. Freeze serving sources/profile/shared native binaries;
  no competing GPU tests. Startup/weight loading is not serving acceptance.
- Adaptive scorer remains BENCHMARK-ONLY. Largest BP128 tile fails strict
  changed-input tolerance; BP64 initially failed too. Static warp controls
  isolated a layout-dependent FP32 head reduction: explicit two16-head
  sums fixes BP64; tolerance unchanged. Four mixed-length graph tests pass.
  Final candidate shortBP16/32/64, medium/longBP64; all12 timing shapes pass
  raw-logit and top512 checks. Score-time baseline→candidate µs:
  c1 at1K/16K/131K 3.047→3.102/4.495→4.194/16.289→9.958;
  c8 3.525→3.599/10.024→9.305/51.902→50.988;
  c16 4.329→3.944/17.504→13.062/98.238→92.518;
  c32 5.561→4.844/32.672→24.037/187.580→174.002.
  Raw `glm53f-compact-pool-cache/adaptive-bp64-fixed-timing.*`; original
  failures/controls preserved separately. Do not enable failed BP128.
- New BENCHMARK-ONLY c1 update hypothesis: update+score18.96us at1K vs
  score3.05us; one CTA can complete a pool then write its raw ring in one
  launch, and skip pooling on the other3/4 steps. Prototype
  `benchmarks/glm5_next_singleton_pool_update.py` reuses original16-pool
  arithmetic and block barrier. Offline SM80compile passes BS64/4608,
  `singleton-update-offline.log`; **no GPU correctness/timing yet**.
  Pending `tests/glm5_next/test_singleton_pool_update_candidate.py` checks
  all phases, invalid slots, ring/page boundaries and untouched slab bytes.
- Next after live A/B: record full-router result, stop exact API, verify
  worker/GPU teardown, then singleton-update GPU gate/timing and adaptive
  scorer integration behind its own comparison. Full1M/combined long-tiers,
  TP scaling, deeper kernel work and QuixiCore port remain open.

### 2026-09-09 08:18 UTC: singleton cache update retained; long baseline live

- Supersedes the07:47 live-state checkpoint above. Router overlap FAILED
  its performance A/B: medians102.708/474.973/655.794/897.148 versus
  KDA-only102.763/476.888/657.832/899.056. All gates pass but no TPS win;
  separate router flag nowFALSE, helper documented diagnostic-only.
  Router API3439513 and all workers exited; all GPUs were clear before
  subsequent GPU tests. Raw `glm53f-tp8-mhc-router-overlap/attempt01/`.
- Singleton cache update now owned in `glm5_next_pool_cache.py`: one CTA
  skips incomplete pools, retains original16-pool arithmetic on completion,
  barriers then writes raw ring. Byte-exact phase/ring/page/invalid-slot/
  changed-graph tests pass atBS64/4608. BS4608 baseline→fused us byphase:
  15.668→1.585,15.561→1.611,15.610→1.576,15.794→14.509. Fivealternating
  graph repeats. Main integration502GPU+79CPU/config/profile tests pass.
- Singleton serving A/B finished exit0: c1/c8/c16/c32 medians
  **104.557/477.388/658.092/901.649** tok/s; c1+1.746% versus KDA-only,
  others effectively flat. All exact1000/300 counts and canaries pass,
  all11 source hashes match, pool unchanged4,280,453/50.74GiB/rank.
  Retain singleton-update flag. Raw `glm53f-tp8-singleton-update/attempt01/`
  including `comparison.json` with all raw repeats.
- LIVE MAIN TREE retained checkpoint: API3462018/engine3462609,
  workers3462853..3462860; server session12278, port8400. Old short
  harness3462091/session97652 finished. **New16K baseline session35518**:
  `validate_glm5_next.py --base-url http://127.0.0.1:8400 --tokenizer
  /home/ubuntu/models/GLM-5.3-Flash-NVFP4 --out
  perf/results/2026-09-09/glm53f-tp8-singleton-update/attempt01/context16k
  --concurrency 1 8 16 32 --repeats 3 --input-tokens 16384
  --output-tokens 300 --repeat-source`. CPU-only client; stdout/stderr
  `attempt01/context16k.log`. Canaries/warmup pass; matrix inprogress.
  Freeze serving/profile/native files; no competing GPU tests. Keep this
  engine for131K baseline next if16K passes. Stop only its exact API after
  harness finishes, then verify all workers and GPU allocations disappear.
- Adaptive scorer is now owned in `glm5_next_pool_score.py` and wired
  through independent strict flag/opaque schema, decodeR<=64/H32only,
  prefill unchanged; **profile flagFALSE** pending serving A/B. BP64
  grouped-head reduction passes6 mixed replay cases includingc64/synthetic1M;
  BP128 remains rejected. No tolerance relaxed. Both cache flags require
  compact layout. Benchmark copies now only wrap/compile owned kernels.
- Long-harness optionsinput/output/repeat-source preserve olddefaults;
  two CPUtests pass (`glm53f-compact-pool-cache/validation-shapes-cpu.log`).
  Exact timing includes prefix-alignment restore/recompute, not puredecode.
- Next projection experiment only underbenchmarks:
  `glm5_next_bf16_gemv_candidate.py` and `benchmark_glm5_next_bf16_gemv.py`.
  All9 SM80 grouped-row/full4096 BF16 variants compile offline; no GPUgate
  or timing yet. Readowned DSV4projection/router references first; runner
  compares cuBLAS/nativeDSV4/9variants with16disjoint matrices andchanged
  graph inputs. Do not run while serving ownsGPUs. Do not relax exactmHC
  projection tests based on this experiment's existing merged-KDA tolerance.
- ZG remainsCPU-only; MCPtransportclosed, CLIserver queries available.
  Full1M/combined long-tiers, TP scaling, deeper kernels andQuixiCoreCUDA
  port remain open. Goal ACTIVE; performance is not completely optimized.

- 08:22 correction: first16K harness(session35518) exited1 atc8 prompt
  construction, before submittingc8 requests. `--repeat-source` previously
  repeated/truncated to exactlyone prompt, leaving no second start window.
  Fixed benchmark-onlybuilder to include offset and a cycle of distinct
  start positions; validates empty source/badshapes.19 CPUtests pass,
  plus realGLMtokenizer32distinct exact prompts at16K and131K. No serving
  source changed. Partialc1 oldrun84.10/83.57/85.18 retained asdiagnostic.
  **Current16K harness3489454/session78741**, output
  `attempt01/context16k-fixed/` and `attempt01/context16k-fixed.log`. Same command
  and API3462018 asabove, onlyoutputpathchanged. Complete thismatrix,
  then131K baseline; compareadaptive with thesame fixedpromptbuilder.

- Pending benchmark-onlyKDAcopy elimination: unchanged ownedpackedkernel
  launched intooutputview, no serving imports. Files
  `benchmarks/glm5_next_kda_direct_output.py`,
  `benchmarks/benchmark_glm5_next_kda_direct_output.py`,
  `tests/glm5_next/test_kda_direct_output_candidate.py`.
  Twenty GPU cases only CPU-collected; not GPU-tested/timed. Run after GPU release.
  Integratinglater requiresoptionalout inownedwrapper andmixed-batch-safe
  callerdispatch; do not change arithmetic or speculative merge behavior.
- 16K exactbench includes2,560-token recompute past13,824 alignedprefix.
  A finer-grained compactindexer pool is a potentialKV/prefix-reuse lead.
  ReadexistingQwenmulti-pool precedent; itsone-blockchatdemand sizing isnot
  a drop-inGLMlong-context policy. No planner/servingcodechanged.

### 2026-09-09 08:32 UTC: 16K baseline complete; 131K matrix LIVE

- Corrected16K harness3489454/session78741 completed exit0. Medians
  c1/c8/c16/c32 **84.450/279.034/343.821/397.038** tok/s; all12 exact
  JSONs have16384/300 per request, all canaries pass,11serving+3client
  hashes match. `attempt01/context16k-summary.json` records rawrepeats.
- **Current harness session2986**, `attempt01/context131k/` and
  `attempt01/context131k.log`: same command asfixed16K, but
  `--input-tokens 131072 --output-tokens 300`. Same API3462018,
  engine3462609/workers3462853..3462860/server12278, port8400.
  Serving/profile/native code FROZEN; no competing GPU work. Prompt
  priming is much longer here. c32 is close to resident capacity; watch
  preemption/tiers and distinguish these from scorer/decode time.
- Finish131K before stopping exact API. After allworkers/GPU allocations
  disappear: pending KDA direct-output20GPUtests + microbenchmark, pending
  BF16 GEMV numerical/HBM microbenchmark. Adaptive scorer stillFALSE;
  its A/B must match short/16K/131K protocols and fixedclienthashes.
- QuixiCore port-review agent completed delta audit, no edits/GPUcalls.
  Narrow new payload is compact pool update+cached scorer and mHC urgent/
  deferred finalizer; do notport disabledadaptive or rejectedrouteroverlap.
  Scratch/alignment are ownership/metadata utilities. Preserve caller-owned
  outputs/streams, inactive-logit undefinedcolumns, exactwhole-slabupdate
  parity andApacheprovenance. Independentraw-logit/mathoracles atumbrella
  tolerances stillneeded; Transformers top-kparity isnot thatproof.
- Full1M/combinedlong-tier, TPscaling, deeperkernels andstandaloneport
  remain open. Goal ACTIVE. Do not describe performance as fully optimized.

### 2026-09-09 08:50 UTC: 131K still live; attribution helpers and prefix candidate

- Authoritative live state unchanged: API3462018/engine3462609,
  workers3462853..3462860; current harness3504709/session2986.
  `attempt01/context131k.log`: c1 finished60.21/48.90/49.39, c8
  117.57/119.27/117.64; c16 warmup10.89 and first timed133.37.
  These are partial observations, not final matrix medians. Real host-tier
  restores occurred (372/rank at one checkpoint), zero reported mismatches
  so far; final totals/coverage still need extraction. No preemption/server
  error seen at that checkpoint. Do not restart a healthy slow cold-prefill.
- All11 serving/profile and3 client/source hashes still match. **Do not edit
  the exact harness until131K completes**: it launches a fresh client process
  for every repeat. No competing GPU work or serving/native changes.
- Next before engine teardown: aligned-prefix baseline planned at124417
  and18433 input tokens,2000 output, c1/8/16/32,3repeats. Exact commands in
  `attempt01/aligned-decode-protocol.md`. One token beyond4608 boundaries
  should reduce the uncached tail, but prove this with counters; do not call
  it pure decode merely from length. Keep existing16K/131K results separate.
  Capacity corrected before launch: boot confirms Mamba align, four state
  groups/up to8 state blocks per request; actual8397-block slab. c32
  129025/2000 would need8480 blocks;124417/2000 needs8192. Keep131072/300
  stress unchanged (conservative8512 blocks). Actual admission/preemption
  still needs metrics; this arithmetic alone does not prove the bottleneck.
  IMPORTANT: current engine has `VLLM_KV_TIER_VERIFY=1`; restore completion
  includes per-page GPU-to-CPU reads plus CPU SHA1 in `kv_tier_dma._verify`.
  These are correctness-instrumented results, not uninstrumented production
  tier TPS. Quantify separately with a matched verification-off run; keep
  verification identical within any kernel A/B. c32 first repeat102.42;
  later checkpoint0preemptions/2running/21deferred/0capacity-waiting. Finish
  remaining repeats; do not attribute the whole gap from this one snapshot.
- New `benchmarks/serving_cache_metrics.py` parses model-filtered cache hits,
  new prefill tokens and preemptions. Missing isNone, invalid/reset counters
  fail. **Not integrated into exact client yet**. After131K, wire into the
  existing before/after metrics reads (no extra requests inside timer), add
  CPU wiring tests, save new aligned-client hashes and use them for both A/B
  arms. Current server exposes the required counters.
- New independent `benchmarks/glm5_next_pool_oracle.py`: CPUFP64 pooling
  semantics and score oracle, optional GPU comparison at unchanged umbrella
  BF16(0.002/0.002) andFP32(atol1e-6/rtol1e-5) contracts. GPU scorer oracle
  uses actual stored BF16 keys to separate scorer error from key rounding.
  GPU diagnostic NOT run. Combined oracle/metrics/prompt/harness CPU suite:
  **30passed**, raw `glm53f-compact-pool-cache/long-client-and-oracle-cpu.log`.
- New `test_pool_partial_prefix_candidate.py`: four GPU tests onlyCPU-
  collected. Clone a page containing a later suffix into a different physical
  page, resume at a4-token-aligned prefix, append changed rows, require raw
  scorer parity and untouched source/padding. Hypothesis: future keys are
  masked byvisible length and a new four-token pool replaces its own ring
  rows, so no historic raw-ring snapshot is needed at such a boundary.
  This is **not** scheduler/tier COW implementation or proof. A real path
  needs partial-hit lengths, bounded alias hashes with block-lifetime
  invalidation, copy-before-write and KDA boundary/tier agreement.
- Static multi-pool sizing is not automatically safe: existing `_chat_demand`
  assumes one attention block, and Mamba align mode needs up to2 state pages
  per active request. Fixed reservations can remove32long-request capacity.
  Partial-page COW is a potential alternative preserving the compact layout.
- Next boot should enable an inactive profiler configuration for targeted
  post-benchmark traces. Last full census predates these retained changes;
  profile both current decode and long-prefix tail prefill before attributing
  the remaining gap to a particular upstream kernel. Existing profiler config
  precedent is in `glm53f-tp8-corrected-reference/server.log`; this current
  engine has no profiler enabled. Do not call start_profile on it.

- 08:56 live update: c16 completed133.37/135.11/136.22; c32 warmup child
  PID3528341 is live under harness3504709/session2986. All8 GPU compute
  processes are still the owned TP workers; no ZG GPU allocation. Finish
  this exact matrix before changing client/source or launching GPU tests.

### 2026-09-09 09:06 UTC: tier duplicate-page evidence, no serving mutation

- 131K c32 warmup remains live in child3528341 under3504709/session2986;
  verify current process/log state before acting. GPU snapshot100% allranks,
  memory utilization15–16%, SM1410MHz; not simply idlewaiting ondisk.
- New CPUdiagnostic `benchmarks/measure_glm5_tier_duplicate_pages.py`:
  current index,129024-token identicalprefix, attention ratios1/8,4
  separate tailgroups. At actual11915host/42366disk capacities,32sequential
  lineages write8064attention pages for252distinct pages, plus128tails,
  total8192diskwrites. All32lineages resumable. NOT a servingconcurrency
  benchmark or measuredspeedup. FiveCPUtests pass. Raw
  `glm53f-compact-pool-cache/tier-duplicate-profile-capacity.{jsonl,log}`,
  `tier-duplicate-accounting-cpu.log`; smallerpressurecase also preserved.
- Why: Mamba GPU-hit requests do not adopt the tier lineage; dedup isonly
  withinowner. This protects branching. DoNOT undo it via sharedfirsthash
  owners or unconditionalMamba adoption. Potentialcontent-addressed immutable
  page store must retain independenttails and add correctrefcounts/pins
  through demotion,promotion,pendingwrites,evictionandreuse. Existingrelease
  helpers assumeexclusive slots. No suchstore implemented yet.
- Avoid a speculative EP detour: actualMoEintermediate2048, localTP8width256.
  Currentnaive dispatch requiresDP>1 orSP; EP alone neednotdoalltoall.
  VerifyresolvedSP/communication before re-proposing the earlierrejectedarm.

### 2026-09-09 09:24 UTC:131K completed; aligned124K baseline LIVE

- Goal remains active, NOT fully optimized. Completed131K parent3504709 /
  session2986 exited0; all12exact131072/300 and canaries pass. Medians
  c1/c8/c16/c32:49.386/117.642/135.111/97.858 tok/s. c32 regresses versus
  c16; source/client hashes matched to completion. Summary raw
  `glm53f-tp8-singleton-update/attempt01/context131k-summary.json`.
- Engine boot cumulative at that completion:0preemptions,160reported
  restore batches/37783ops per rank,0reported mismatches/errors. Verification
  skips missing expected digests; no complete-coverage claim. All results
  include `VLLM_KV_TIER_VERIFY=1` readback/hash overhead. Keep a separate
  verification-off production comparison in the next-run plan.
- After terminal only, integrated `serving_cache_metrics.py` into existing
  exact-client two HTTP snapshots, outside timer. New `metric_snapshot`
  returns spec/cache pair; old `metric_counters` compatibility retained.
  Unknown counters stayNone, invalid/reset fail.29CPUtests pass plus script
  --help smoke; raw compact-pool-cache/cache-metrics-wiring-cpu.log.
- **CURRENT LIVE**: same API3462018/engine3462609/workers3462853..3462860;
  new parent3562500/session89742, `attempt01/aligned124k.log`, output
  `attempt01/aligned124k/`.124417input/2000output,c1/8/16/32,3repeats.
  Canaries PASS; c1 warmup child3562509 at09:24. Verify current processes
  before acting. Serving/source and NEW aligned-client hashes frozen until
  all repeats finish. Hash files `source-sha256.txt` and
  `aligned-client-sha256.txt` (includes metrics helper). No competing GPUs.
- Read per-result cache_metrics: require prefill_requests matchesconcurrency
  before interpreting computed tokens. Alignment alone is not puredecode
  proof. Originalfull131Kstress preserved;124417shape corrected to8192
  conservative resident blocks within8397pool (old129025/2000 didnotfit).
- Next: finish alignedlong baseline, matching18433/2000 medium baseline;
  then stop exactowned API and waitallworkers/GPUcontexts clear. Run pending
  GPU independentpooloracle, partialprefixclone4cases, KDA direct-output20
  cases/timing and BF16GEMVdiagnostic. Enable adaptive scorer onlyafter a
  matching baseline and safety gates; freshboot should have inactiveprofiler
  enabled. No benchmark-only prototypes are currently serving paths.
- First aligned c1 repeats94.31/90.02 tok/s (partial, not finalmedian).
  r1 cache evidence:124416GPU-hit tokens,1computedprefill token,1finished
  prefill request,0externalhits/0preemptions; exact124417/2000. This validates
  the intended reuse for that request, not yet c8/c16/c32. NEWclient hashes
  stillmatch. Server/harness live; continue fromsession89742.

### 2026-09-09 09:37 UTC: aligned c8 done, row-shard diagnostic prepared

- Live parent3562500/session89742, sameAPI/engine/workers. c1 completed
  94.31/90.02/90.48; c8 completed303.74/318.52/307.60. Every c8 timed
  result records8computed prefill tokens/8requests,995328GPU-hit tokens,
  0externalhits/0preemptions. c16warmup child3572478 live at09:37; nothung.
  Source/client frozen and no competing GPUwork.
- New quarantined benchmark `benchmark_glm5_next_indexer_row_shard.py`
  targets replicated indexer score/topk: assign full query rows acrossTP
  ranks, gather512poolindices/query; no KV precision/layout change.11
  disjoint cache layers, exactlogits/selectedsets, changed-input CUDAgraphs,
  fixediteration counts (never independent time-budget NCCL loops),5
  alternating repeats/maxranktime.36CPU ownership/mappingtests pass;
  **GPUgate/timing notrun**. Short cases may beL2hot; no serving gainclaim.
- Run afterserver/releasesallGPUs, e.g. OMP_NUM_THREADS=1 .venv/bin/python
  -m torch.distributed.run --standalone --nproc-per-node=8 --module
  benchmarks.benchmark_glm5_next_indexer_row_shard. Capture stdout/stderr
  under compact-pool-cache. c1fallback staysreplicated. Short-context NCCL
  cost mayrejectthis; a long-contextonly route must respect CUDAgraph
  replaydynamic lengths, not Python trace-time branching.
- DCP/FP8 are notdrop-in knobs: sparse backendcurrentlyreturnsnoLSE and
  usesglobalindices/unshardedpages, whileDCPcallerrequiresLSEcombine;
  currentFP8pathassumes576latentwith64RoPE, notGLM5NoPE512. Ownedchanges
  are possible but notimplemented. Prioritize measured/proven paths.

### 2026-09-09 10:13 UTC: alignedlong complete; medium LIVE; sparseTC prepared

- **CURRENT LIVE**: parent3607040/session66802, outputattempt01/aligned18k,
  logaligned18k.log.18433input/2000output,c1/8/16/32,3repeats. Same
  API3462018/engine3462609/workers3462853..3462860/server session12278.
  Canariespass; c1first105.34. Source/client frozen untilparentterminal.
  Use existing source-sha256.txt/aligned-client-sha256.txt for verification.
- Prior parent3562500/session89742 completed0. Aligned124417/2000medians
 90.481/307.599/464.679/702.056; all12exact andcanariespass. All timed
  requestscompute1prompttoken,remainingtokensGPUhit,0externalhits/preemptions.
  c32warmup had3732480externalhit tokens, but timedrepeatsdidnot. Raw
  aligned124k-summary.json. VerificationstillONforoffloads, not production
  tierTPS; no comparisonofdifferentpromptshapesasoptimizationgain.
- New `benchmarks/glm5_next_sparse_tc_candidate.py`: NoPE512 BF16 Q/KV,
  tensor-core shared-head QK and value products, FP32 partitionsoftmax,
  twoBF16probability components (high+residual), no KV quantization. Changed
  arithmeticorder: NOTbitexactclaim. Negative/emptyselectedtiles guarded,
  strided64-bitpageaddressing, splits32/64/128.6SM80 variants compile,
  shared48/80/144KiB. Current source includesemptytile skip; raw
  sparse-tc-empty-skip-offline.{jsonl,log}, earliercompilealso preserved.
- `tests/glm5_next/test_sparse_tc_candidate.py`:60GPUcases collectedonly,
  **NOTRUN**. H8/16,c1/8/16/32,BS64/576,actualscale256^-0.5 +legacy512,
  holes/shuffledpages/emptyrows/changedgraphreplays. Existinggates preserved:
  normalizedmax5e-3,pointwiseBF16.002/.002,nativeabsolute<1e-3. Do notloosen.
- `benchmark_glm5_next_sparse_tc.py` measures11disjointlayercaches and
  independentrequests withactualpackedstride. c32/131Kallocatesabout48GB
  ononeidleGPU; requiresfreememoryratherthanshrinking. Run GPUgatesfirst,
  thenbenchmark; freshprofileA/B requiredforservinggain. No servingimports
  these prototypes. References inspected inownedCUDA,DS4ROCm,QCRocm,
  llama.cpp Ampere MMA; no broadvendoring.
- Existing adaptive-scorer tests now8cases: addedprefillR256/2048 with
 3requestcaches andbatchedvalid-columnchecks. CPUcollectedonly, extended
  suiteNOTGPUrun. Servingstillguardsadaptiveoff/prefilloriginal.
- Next aftermediumcomplete: stoponlyownedAPI,waitallworkersandGPUcontexts
  clear. GPUgates: independentpooloracle/partialprefix; sparseTC60;
  adaptiveextended8; KDA directoutput20; thenisolatedtimings(rowshardneeds
  all8GPUs, nooverlap). Prioritize measuredwins, onefactorat atime. Next
  serverboot shouldenableinactiveprofiler; separatelyquantifyverification
  offproductionperformance withmatchingbaseline. Full1M/scaling/combined
  tierquality/QuixiCoreportstillopen; goalNOTcomplete.
- KVreviewfoundexisting VLLM_PREFIX_CACHE_RETENTION_INTERVAL sparseMamba
  snapshotretention/reachable-boundarypolicy. No changeorproofthatthis
  causedwarmupmisses; inspect/reusebeforeinventingnewretention.

- 10:22livecheckpoint: medium parent3607040/session66802 nowc32warmup,
  child3614794. c1complete105.34/105.66/105.47; c8
 465.20/481.79/463.75; c16 720.53/744.84/738.14. Partialmatrix only.
  SparseTC CLIhelp smoke succeeds,60tests collected (notGPU-run), all6
  empty-skip SM80compile variants succeed. Exactmodule search findsno
  sparseTC/row-shard diagnostic imports in vllm orslimserve. Serving remains
  unchanged. Next safeaction: finishc32,recordmetrics/hashes,stopownedserver
  andwaitGPUclearbeforeisolatedgates. Do notaddmoreuntestedkernels first.

### 2026-09-09: serving stopped; prepared GPU gates and sparse-TC timing

- Supersedes the live checkpoints above: aligned18433/2000 completed all
  12 exact-token runs and canaries. Median c1/c8/c16/c32:
  105.471/465.203/738.142/1093.203 tok/s. Every timed request computes one
  prompt token, with zero external hits or preemptions. Raw
  `glm53f-tp8-singleton-update/attempt01/aligned18k-summary.json`.
- Owned API3462018 stopped after zero running/waiting requests. All workers
  exited and all eight GPU contexts were verified clear. ZG has no GPU context.
- Prepared GPU suite: **71 passed, one failed**, then `-x` stopped it.
  Partial-prefix clone4 and sparse-TC60 passed; adaptive7 passed, R2048
  prefill failed strict1e-6 parity at one value (absolute1.90735e-6).
  KDA direct-output20 did NOT run. Adaptive remains disabled in the profile.
  Raw `glm53f-compact-pool-cache/prepared-kernel-gates-gpu.log`.
- Independent FP64 pool oracle: scorer passes all six cases, but the
  BF16-rounded key reference fails a few values. Original failed diagnostic
  preserved. Added unrounded-reference errors and midpoint examples without
  changing `strict_pass` or its tolerances; diagnostic rerun pending.
- Sparse-TC isolated timing running on GPU0, session24511, raw
  `glm53f-compact-pool-cache/sparse-tc-timing-gpu.{jsonl,log}`. Do not overlap
  other GPU work. Early c1/1000 native68.85us vs split32 9.52us/layer;
  c8/1000 native84.08us vs split128 20.85us. These are kernel measurements,
  not serving speedups; full matrix and matched profile A/B remain pending.
- Goal remains active: kernel integration/serving A/B, fresh profiling,
  full1M/combined-tier correctness, TP scaling, and QuixiCore port remain open.

### 2026-09-09 10:42 UTC: sparse-TC candidate TP8 boot LIVE

- Timing session24511 completed0, full eight-case sparse-TC matrix passed.
  Selected TC32 at c1 / TC128 at c8+ is2.4–7.9x faster at the isolated
  attention operation. Full timings and caveats are in the notebook; not
  an end-to-end speedup claim.
- Kernel promoted to `vllm/quixicore/sparse_mla_tc.py`; benchmark re-exports
  it. Default-off backend option `glm5_next_sparse_tc_decode`, SM80/H8,
  non-spec pure decode1..32 only; other shapes/prefill native. Explicit
  FP32 partial scratch. GPU gates after promotion98pass (84candidate,
  14native); configuration/dispatch/profile CPU suite90pass.
- **PROFILE FLAG TRUE SOLELY FOR THIS VALIDATION CANDIDATE**, not retained
  as a serving win yet. Server session91412, API3628422, port8400; raw
  `perf/results/2026-09-09/glm53f-tp8-sparse-tc/attempt01/`. Boot command:
  `SLIMSERVE_KV_TIER_DIR=/home/ubuntu/.cache/slimserve/kv-tier VLLM_KV_TIER_VERIFY=1
  .venv/bin/python -m slimserve.cli glm53f-nvfp4-8 --serve --host 127.0.0.1
  --port 8400 --torch-profile-dir <attempt01>/traces -y`.
  Profiler configured but inactive during timing. No competing GPU work.
- Validator parent3628492/session39703 waits for health, then text/reasoning/
  image/tool canaries and exact1000/300 c1/8/16/32 three-repeat matrix.
  Raw `attempt01/short/`, progress `short.log`. Frozen16source/client hashes
  at `source-client-sha256.txt`. Compare with singleton-update short baseline,
  same verification-on tier settings. Fresh profile trace after timing;
  alignedlong A/B and quality gates still required.
- KDA direct-output20 GPU tests and isolated timing completed0: median
  allocate/copy -> direct us at c1/8/16/32:
  5.559->4.042 /10.712->9.444 /16.264->14.850 /28.264->26.862.
  Eight changed-input replays exact output/state,16disjoint layers, five
  alternating repeats. Not integrated: preserve one-factor attention A/B.
- Pool oracle rerun still fails both original rounded and extra unrounded
  diagnostic. Midpoint evidence recorded without changing its gate. Scorer
  all six passes; stronger independent pool conformance remains open.

### 2026-09-09 10:55 UTC: short win; long c1 REGRESSION under investigation

- Short validator3628492/session39703 completed0. Exact1000/300 medians
  109.654630/486.892838/677.277244/926.254877 at c1/8/16/32, gains
  +4.875%/+1.991%/+2.915%/+2.729% vs singleton-update. All12exact/canaries
  pass;16source/client hashes match; same4,280,453-token pool. Raw
  sparse-tc/attempt01/comparison.json. This qualifies ONLY the short result.
- Bounded profile session85056 completed0, endpoints200, stopped explicitly.
  Eight rank traces saved. Candidate decode contains11TCpart+11TCreduce
  calls,118.719us aggregate for those kernels in the sampled step.
- Old census assumed all90mHC markers on main stream; actual CUDA graph
  schedules49/21/0/20 across streams23/306/307/308. Fixed only the CPU
  summarizer to validate all90 and refuse unproven per-layer attribution.
  Three CPU tests pass. Raw faileddecode-census.log preserved, corrected
  decode-census.json anddecode-census-multistream.log. Step wall8327.551us,
  kernel union7876.431us, kernel sum8771.554us (overlap; not wall time).
- **CURRENT LIVE** aligned124417/2000 harness3644471/session92573,
  same API3628422/engine3629008/workers3629276..3629283/session91412.
  Raw attempt01/aligned124k, progressaligned124k.log. Canaries pass.
  c1 results77.977/76.24/75.28 vs baseline90.481median: REGRESSION.
  Timed counters still1computed prompt token,124416GPUhits,0externalhits/
  preemptions; no restore ops. GPUs1410MHz, no competing GPU processes.
  c8warmup currently in progress. Keep running; no competing GPU tests.
- Do not retain this as a universal win or discard the regression. After
  long matrix, run a matching short post-profiler control, then bounded
  long-context trace. Investigate real selected-KV/kernel behavior, profiler
  lifecycle overhead, and verification/offload/cache-history differences;
  they are hypotheses, not established causes. A fresh native baseline with
  matched lifecycle may be needed. Full1M/scaling/combined tiers still open.

### 2026-09-09: c16 complete; c32 long-prefix warmup LIVE

- Same live harness3644471/session92573, API3628422/engine3629008,
  workers3629276..3629283, serving session91412. No source/profile changes.
  c8long322.185/315.951/317.039 (median317.039 vs307.599, +3.07%).
  c16long469.02/468.46/511.76 (median469.02 vs464.679, ~+0.93%, wide
  spread). All finished exact checks pass. c1median76.2409 remains a
  regression versus90.4807. Do not retain the candidate universally.
- Current child3667298 is c32warmup, seed1/output32. At the latest live
  check,1running/23waiting after about4min priming. This is expected cold
  long-prefix work, not a terminal or blocked harness. Keep GPU exclusive.
- CPU stopped-profiler control costs9–17us per step, too small to explain
  millisecond c1regression. Added five RED lifecycle regression tests at
  `tests/v1/worker/test_profiler_annotation_lifecycle.py`, raw
  `attempt01/profiler-annotation-regression-before.log`: inactive/stopped/
  capped/delayed profilers still build annotations. Serving fix NOT applied
  while sources are frozen. After current benchmark/control captures,
  expose WorkerProfiler.is_running, call step() before checking it in
  Worker.annotate_profile, and return nullcontext before computing metadata
  unless running. Keep delayed start and iteration caps working; rerun tests.
- Eight-rank collective census recorded at`attempt01/collective-rank-census.json`:
 91calls/rank,706–767us aggregate per rank. Cross-rank timestamp skew is
  explicitly UNUSABLE: apparent finish skew70.74us exceeds typical5–8us
  barrier-containing kernel durations, indicating clock calibration trouble.
  Do not mistake it for rank imbalance. No collective code changed.
- Next after c32completes: record full long summary/hashes/counters, run
  short1000/300 c1 post-profiler control, then a bounded actual124K c1
  decode trace (no throughput claim for profiled requests). A matched fresh
  native/candidate lifecycle A/B may be required to isolate the regression.

### 2026-09-09 20:02 UTC: verification-OFF native baseline boot LIVE

- Supersedes all earlier live-process entries. Prior long harness92573
  completed0 at11:38. At19:45 the existing server was idle; all16 original
  source/client hashes verified unchanged.
  Full alignedlong medians76.240899/317.039233/469.016586/685.006942,
  all12exact/canaries. c32repeats735.399/685.007/539.294; all timed
  requests1computed token, remaining prefix GPU-hit,0externalhits/preemptions.
- Completed post-profiler short c1 control:106.088/106.692/106.258tok/s.
  Separate long trace: steady sampled decode10543.528us, sparseTC
  part+reduce148.257us/11layers. One-token cached-prefill annotated span
  1.071s is mostly WAITING, not1.071s of arithmetic. It contains~250host
  tensor-to-NumPy conversions consistent with synchronous whole-prefix
  offload verification. Text/IDs control excludes tokenization as dominant:
  tokenize0.19s; text32-output requests1.69–2.18s, IDs1.74–1.92s.
- Stopped old owned API3628422 only after idle; all workers/GPU contexts
  gone. Its server session91412 completed0. No model server remains from
  that arm. All old raw artifacts preserved under sparse-tc/attempt01.
- Implemented fixes after stopping: WorkerProfiler.is_running and an
  annotation early return AFTER step() (keeps delayed start/caps); shared
  zero-copy `_row_digest` for stable contiguous CPU buffers, strided-C-order
  fallback, exact existingSHA1. No async/lifetime changes. Five formerly-red
  profiler tests pass; CPU lifecycle/digest/IO14pass; additionalCPU15pass;
  verification-ON GPU DMA/disk/digest18pass. Lint/diff checks pass.
- Hash benchmark256disjoint6,488,064B rows: median1.035870s->0.956970s,
  exactchecksums, fivealternatingrepeats. PeakPython allocation6,488,337B
  ->758B perchecksum. This only removescopies, notexpensiveaudit scans;
  no servingTPSgainclaimed. Rawzero-copy-digest-cpu-timing.{jsonl,log}.
- **CURRENT LIVE** native-kernel production-timing baseline:
  API3722292/server session88175; shortvalidator3722367/session65941.
  `glm5_next_sparse_tc_decode=false` inprofile (NOT universallyretained),
  `VLLM_KV_TIER_VERIFY=0`, sameTP8/cache/tier/graph settings. Profiler
  configured but keep INACTIVE until alltimingcomplete. Twentynewsource/
  client hashes frozen. Raw
  `perf/results/2026-09-09/glm53f-tp8-verify-off-native/attempt01/`.
- Next: finishshort c1/8/16/32, thenaligned124417/2000 native baseline,
  followed by matchingverification-OFF sparseTC candidate. Do not compare
  OFFvsON asakernelgain. Do notmodifyservingsource or runGPUmicrobenchmarks
  duringtheactiveprofile. Full1M/combinedtier/scaling/QuixiCoreport remainopen.

### 2026-09-09 20:11 UTC: native short complete; aligned long baseline LIVE

- Performance is not completely optimized. User explicitly challenged the
  interruption; the existing optimization goal remains active. Do not mark
  completion while long-context, combined-tier, scaling and port gates remain.
- Native verification-OFF short validator session65941 completed0. Exact
  1000 input/300 output, three repeats at c1/c8/c16/c32; all12 exact results
  and text/reasoning/image/tool canaries pass. Medians:
  105.072241/480.207654/662.717723/905.221838 tok/s. All timed prompts were
  computed with zero prefix hits, external hits or preemptions. Raw
  `glm53f-tp8-verify-off-native/attempt01/short/` and `short-summary.json`.
- All20 frozen source/client hashes still match. Same healthy API3722292,
  engine3722840, workers3723094..3723101, server session88175. Pool remains
  4,280,453 tokens/50.74GiB per rank. Only these workers use GPU memory;
  ZG is not using GPU memory. No profiler start and no serving-source edits.
- **CURRENT LIVE** aligned124417-input/2000-output baseline:
  harness3757133, session70364, three repeats at c1/8/16/32, repeat-source.
  Output `glm53f-tp8-verify-off-native/attempt01/aligned124k/`, progress
  `aligned124k.log`. Preserve GPU exclusivity and frozen sources until done.
- CPU-only profiler/digest/trace regression rerun:15passed. Updated baseline
  notebook to remove stale claim that sparse TC is currently enabled;
  long verification-ON results do not qualify universal retention.
- Next: finish existing session70364 (do not restart it), record exact counts,
  cache counters, repeats and source hashes; then matching verification-OFF
  sparse-TC candidate. OFF-versus-ON is not a valid kernel speedup comparison.

### 2026-09-09 20:26 UTC: long c1/c8 complete; c16 live; rollback fix prepared

- CURRENT LIVE remains long native harness3757133/session70364, server
  session88175/API3722292/engine3722840/workers3723094..3723101. c16 warmup
  is progressing, not terminal. Do not restart on an observation timeout.
  Twenty source/client hashes still match; no serving-source changes.
- Long c1 exact repeats76.827485/76.490771/76.724874 (median76.724874).
  All three1computed token/124416GPU prefix hits,0external/preemptions.
  c8 repeats435.83/421.16/417.96 complete. Read raw JSON for precise c8
  values/counters. c1 is below the old native90.48 too, so the earlier
  regression is not established as a sparse-TC-specific issue. Continue
  the matched verification-OFF arms; no universal candidate retention yet.
- New concrete CPU index bug: `HostKVTierIndex.promote` rolls back with
  `new_slots + list(tail.values())` after partial tail allocation failure,
  although those tail slots already occur in `new_slots`. It frees slots
  twice (five free-list entries for four actual slots in the reproducer).
  Add only `new_slots` to the rollback call. **Serving file NOT patched**
  during the frozen benchmark. A separate in-memory exact one-line candidate
  passes51CPU index/disk/connector tests, including four new regression
  shapes with GLM's four tail groups. New regression is RED on production
  source until correction is applied; do not leave it RED at goal completion.
  Apply at a source-change boundary while preserving matching A/B provenance;
  normal tests and real tier checks still required afterward.
- Raw under verify-off-native/attempt01: partial-promotion-rollback-before.log,
  partial-promotion-rollback-expanded-inmemory-fix.log (51passed). No evidence
  yet that the throughput run triggered this partial-promotion failure.
- Extended CPU duplicate accounting to32independent prefix families across
  eight replay rounds at216blocks/124416tokens, actual host/disk capacities.
  62208attention copies for7776unique pages,1024independent tails. Potential
  cumulative copy reduction328.90GiB/rank; unique attention+tails53.17GiB
  in this synthetic workload. Not GPU-memory savings or real serving TPS.
  Current exclusive-slot index retains171/256trajectories,123disk-only.
  TenCPU tests pass. Raw tier-prefix-families-cpu.{json,log} and tests.log.
- Immutable sharing is still NOT implemented. Use canonical physical pages
  with reference counts and transfer/read pins, never shared Mamba lineage
  keys. See the new notebook entries for lifecycle gates and precise commands.
- Latest poll of existing session70364 is live: c16warmup complete,
  first timed repeat656.53tok/s. Precise c8median421.161393, all three exact
  with995328GPU-hit prompt tokens/8computed tokens,0external/preemptions.

### 2026-09-09 20:47 UTC: c32 timed baseline LIVE; immutable page pool added

- Same live native benchmark: session70364/harness3757133; server88175,
  API3722292/engine3722840/workers3723094..3723101. c16 med575.023210
  (repeats656.533018/575.023210/501.676357), all exact,1computed token per
  request,0external hits/preemptions. Significant repeat drift remains even
  with verification OFF. c32warmup completed107.54tok/s; timed runs live.
  All20frozen serving/client hashes match. Do not restart on observation lag.
- Implemented `vllm/v1/core/immutable_kv_pages.py`, a scheduler-local physical
  page pool with canonical immutable attention keys, private tails, per-owner
  lease handles, host/disk locations and reserved/submitted/completed tickets.
  Pending transfers and reads pin bytes even after the last owner releases.
  No serving imports or connector integration yet; it cannot affect this arm.
- New pool/accounting suite49CPU tests pass, including representative bytes,
  delayed IO/cancellation, private tails, slot reuse,8x600random lifecycle
  transitions, and profile-sized c1/c8/c16/c32family replay inventories.
  Raw `immutable-page-pool-final-cpu.log`; lint/diff checks pass.
- Extended `benchmarks/measure_glm5_tier_duplicate_pages.py` with optional
  `--implementation shared-pool`. Actual component inventory uses275/2200/
  4400/8800physical pages for8/64/128/256owners (1/8/16/32families,8rounds,
  216blocks,4private tails). c32total copies63232->8800 versus exclusive
  index simulation; all256owners retain ready copies. No serving TPS or
  full trajectory-resume claim. Raw `shared-page-pool-inventory.jsonl`.
- Integration prerequisite discovered: worker `get_finished` currently
  discards completed offload IDs; scheduler confirms host writes on the next
  step. New tickets require actual all-rank completion. Extend completion
  aggregation, cancellation and worker block pins; retain hash-chain/tail
  boundary checks and main-resident compatibility. Do not merely alias slot
  integers or call the new component a finished serving optimization.
- Live IO sample: O_DIRECT, virtual QEMU disk;441–519MB/s worker writes,
  5.69–11.25ms write awaits,7–42% guest busy, negligible dirty pages. This
  observation does NOT prove saturation or the c16 drift cause. Raw
  tier-io-observation.jsonl. No competing GPU work was launched.
- Next: finish existing c32timed repeats; snapshot metrics/hashes; same-settings
  sparse-TC arm. Known rollback regression remains RED on production source
  until the documented correction is applied at a provenance-safe boundary.
- While extending completion aggregation, review sequence namespaces:
  scheduler offloads increment from0, worker restores increment from1<<20.
  That finite offset is not a proof of disjoint IDs over long service life.
  No collision reproducer or production change has been made for this yet.

### 2026-09-09 23:03 UTC: native arm complete; cached-stream crash isolated and patched

- Supersedes previous LIVE entries: native harness70364 completed0 at20:52;
  idle server88175 was stopped gracefully at22:04. Native short medians
  105.072241/480.207654/662.717723/905.221838; aligned124417/2000 medians
  76.724874/421.161393/575.023210/682.066751 at c1/8/16/32. All24 timed
  results exact, all canaries pass. Long timed requests each1computed token,
  124416GPU hits,0external hits/preemptions. Long c32 repeats
  851.599794/682.066751/641.815969 show substantial lifecycle drift.
  Raw verify-off-native/attempt01/{short,aligned124k}-summary.json.
- Sparse-TC verify-OFF attempts01/02 BOTH failed before health, no TPS.
  All associated processes are gone. Attempt01 kernel journal shows libcuda
  segfaults; attempt02 faulthandler isolates `glm5_next_mhc_project.py`
  ExternalStream.wait_stream. Cached Inductor call contains raw stream pointer
  1054131040 from a previous process. This is our stream-plumbing defect,
  not evidence against the sparse-TC kernel. Do not erase compile caches or
  disable overlap to conceal it. Raw verify-off-sparse-tc/attempt{01,02}/.
- Implemented `glm5_mhc_project_runtime`: compiled op stores a stable string;
  actual invocation resolves the current engine's strongly-owned stream via
  ForwardContext.static_forward_context/no_compile_layers. Existing overlap,
  allocations and stream joins remain. Model no longer passes CUDA pointers
  through compiled graphs. Old direct-handle op is diagnostic-only.
- Applied the known partial-promotion rollback fix in kv_tier_index.py:
  free new_slots once, not new_slots plus already-included tail slots.
  Normal CPU suite75passed/13GPU-skipped, including all four formerly-red
  rollback shapes. GPU restart/parity suite session84055 is running alone;
  no model server currently live. Raw glm53f-tp8-runtime-stream-fix/.
- Added PageCompletionBarrier with atomic ticket batches, all-rank ACKs,
  failed-rank retention until remaining IO finishes, duplicate/conflict checks,
  and one monotonically increasing sequence namespace. Combined pool/barrier/
  inventory CPU suite67passed; raw sparse-tc/attempt02/page-completion-cpu.log.
  This component remains UNUSED by serving, not a live KV optimization.
- Next: finish GPU tests, real registered-profile boot AND cached restart,
  fresh matched native/TC source-frozen benchmark arms. Both serving fixes
  change provenance; pre-fix native results are historical, not a clean A/B
  against a post-fix candidate. Tier sharing integration,1M quality, scaling
  and used-kernel QuixiCore port remain open; optimization goal stays active.

### 2026-09-09 23:15 UTC: mHC first real boot passes; sequence collision fixed

- Runtime-stream regression:34GPU-enabled tests passed. Two fresh processes
  used different stream pointers929720144/1057870912; second got an Inductor
  disk-cache hit, four changed-input graph replays each were exact. Runtime
  operator parity covers c1/8/16/32/64/65 and BF16/FP32 projection. CPU
  pool/completion/dispatch/lifecycle follow-up90passed. No native build needed.
- Real registered profile first fixed boot reached health at23:07:55, then
  all text/reasoning/image/tool canaries and12exact1000/300 runs passed.
  c1/c8/c16/c32 medians104.568910/479.858271/667.168559/905.371495 tok/s.
  GPU pool unchanged4,280,453tokens/50.74GiB per rank; graph0.22GiB. All
  timed prompts fully computed,0hits/preemptions; profiler never activated.
  Real generated op contains 'language_model.model.mhc_projection_stream',
  not a pointer. Raw `glm53f-tp8-runtime-native/attempt01/`, summary JSON
  and generated-stream-call.txt.21frozen source/client hashes match.
- Validator34993 completed0; idle API3856294 gracefully stopped at23:13:50;
  server77004 completed0. All workers/GPU contexts verified gone by23:15.
  Next boot must exercise existing compile cache, not delete it.
- Reproduced another owned bug in actual HostTierConnector CPU methods:
  positive offload1048577 and first restore1048577 coexist; completing ONLY
  offload makes get_finished falsely acknowledge the unfinished restore.
  Fixed worker _seq to start0/decrement, so restore IDs are strictly negative
  and scheduler offload/write-through IDs positive. No finite-offset collision.
  Normal CPU47passed, including4new namespace regressions. In-memory first
  harness failed because recompiled __init__ lost super's class closure;
  corrected isolated candidate47passed, then normal source47passed. Raw
  `runtime-stream-fix/tier-sequence-{collision-before,tests-before,inmemory-fix,
  inmemory-candidate,normal-cpu}.log`. GPU negative-ID disk checks running.
- No serving TPS claim for sequence fix yet. New matched baseline will be
  `runtime-native/attempt02/` with all three correctness fixes. Initialboot
  attempt01 remains first-boot qualification, not the upcoming TC comparator.

### 2026-09-09 23:21 UTC: full cached restart passes; fresh native matrix LIVE

- All eight ranks explicitly logged Directly load AOT compilation at23:18:58
  from the SAME13e053b5... cache produced by attempt01. Graph-memory profiling
  and capture passed; API health23:19:47 and text/reasoning/image/tool canaries
  pass. No compile cache deletion or overlap disabling. This exercises the
  exact full-model restart lifecycle that failed before the stream fix.
- **CURRENT LIVE** native verification-OFF attempt02: server79338/API3873283,
  engine3873893/workers3874221..3874228. Combined validator session92002,
  shell3873357/current short harness3873359. It runs short1000/300 c1/8/16/32
  three repeats, THEN automatically aligned124417/2000 with repeat-source
  at the same concurrencies/repeats. Do not launch a duplicate long harness.
  Raw `perf/results/2026-09-09/glm53f-tp8-runtime-native/attempt02/`.
- All22serving/client hashes frozen and matching. No more serving-source
  edits or GPU microbenchmarks until this arm finishes. Profiler configured
  but MUST remain inactive during timings. Sparse-TC flag false for native.
  New negative-restore-ID normal CPU47pass and GPU/DMA/disk/digest21pass;
  all formerly-red namespace and rollback regressions are fixed.
- Automatic cached-start memory profiling chose4,304,412tokens (~51.02GiB
  KV/rank), vs first-boot4,280,453. Config/hot kernel settings unchanged;
  do not claim this lifecycle-dependent extra capacity as an optimization.
  Match/record candidate cache lifecycle and actual capacity/cache counters.
- Read-only GPU telemetry running: session46183/PID3875728, every2seconds
  clocks/power/temperature/utilization to attempt02/gpu-telemetry.csv. Stop
  this owned monitor after native timing/server shutdown; use the same query
  in candidate arm. Only model workers consume GPU memory; ZG stays CPU-only.
- Next: finish this baseline, then fixed-source sparse-TC comparator. The
  queued BF16 projection and row-sharded indexer GPU gates must wait for
  exclusive GPUs. Simple allreduce+mHC fusion was already rejected in the
  September3 notebook; channel ownership is the deeper unimplemented path.

### 2026-09-09 23:55 UTC: long native c16 stalls; c32 still warming

- Same healthy server79338/API3873283, engine3873893, workers3874221..28.
  Combined harness92002 now runs long validator3886304; no duplicate clients.
  Short complete:104.758619/481.682668/666.585207/905.489373 median tok/s.
  Long c1/c8 medians76.569775/432.752712. c16 repeats
  667.621484/378.663185/511.606749 (all exact,1 computed prompt token/request,
  124416 GPU hits/request,zero external hits/preemptions). c32 cold warmup
  remains active. Same22source hashes; no serving edits or new GPU jobs.
- c16 repeat2 includes roughly38seconds of all-GPU idle while requests are
  active, followed by normal decode. Need isolate the CPU/IO blocking path;
  do not call this a slow kernel or proven reclamation defect. Raw telemetry
  and exact JSON under runtime-native/attempt02.
- Two read-only nonblocking stack dumps and30seconds of5-Hz CPU sampling
  were taken during c32's untimed cold warmup, then stopped. Engine mostly
  waits on workers; worker0 samples often hit prepare_chunk_indices .tolist()
  synchronization in prefill. These samples do NOT capture the cached stall.
  Artifacts c32-cold-{engine,worker0}-stack.txt and *-stacks.raw.
- Next: complete frozen c32 timings; separately diagnose repeated cached c16
  on the SAME server using .venv/bin/py-spy (sudo -n needed for attachment).
  Trace all ranks and engine; keep diagnostic TPS separate from clean baseline.
  Monitor46183/PID3875728 still runs; stop only after this native arm ends.
  Source fixes, GPU candidate work and shared-page serving integration remain
  pending; performance is not fully optimized.

### 2026-09-10 00:08 UTC: native finished; CPU admission diagnostic LIVE

- Native validator92002 completed0. Long c32 med650.419160, repeats
  828.820734/650.419160/622.037196, all32tokens computed/batch with
  3,981,312GPU prefix hits and0external/preemptions. All24timed exact checks
  and canaries passed. Long medians c1/8/16/32:
  76.569775/432.752712/511.606749/650.419160. Serving22hashes unchanged.
- ALL-GPU idle observed spans c32r1/r2/r3:12.016/36.044/42.065seconds.
  Raw native attempt02/telemetry-summary.json. New summarizer5CPU tests pass.
- Same server79338/API3873283 remains live. Separate diagnostic session33410,
  validator3900014 (c16,124417/2000,two repeats plus normal warmups/canaries).
  py-spy session99095/PID3899992 samples API AND all descendants at10Hz,
  nonblocking/idle/threads,240seconds,speedscope. Raw
  perf/results/2026-09-10/glm53f-tp8-admission-diagnostic/attempt01/.
  Do not start another client or change serving code during this diagnosis.
  Passive telemetry46183/PID3875728 continues in native attempt02 CSV.
- New CPU disk-write-completion regression:5fail/3pass on current source.
  Earlier write errors are lost if final batch completion succeeds; isolated
  in-memory failed-batch tracking passes8tests. Apply only after diagnosis/
  source boundary, then normal tests and real IO injection. Not proven as the
  performance-stall cause. Logs native attempt02/write-completion-*.log.
- KDA prefill redundantly rebuilds already-prepared CPU chunk metadata using
  GPU .tolist(); mixed-batch row conventions differ, so a naive wiring fix is
  wrong. Future candidate, no serving changes. Performance goal remains active.

### 2026-09-10 00:24 UTC: fixes validated in isolation; fresh arm BOOTING

- Prior diagnostic33410 completed0 (c16 profiled660.23/546.22); py-spy99095
  stopped after240seconds/client completion via SIGINT to owned3899992.
  Raw85thread profiles,177194samples/2856errors,engine-stack-summary.json.
  Prior server79338/API3873283 stopped idle and completed0; all GPU contexts
  gone. Old passive monitor3875728 stopped.
- Fixed _busy to iterate lazily and _reclaim to skip empty host trajectories.
  CPU pressure reproducer (11915host/42366disk,216blocks,ratio8,withheld ACKs)
  c1/8/16/32 time before0.720/5.897/12.208/25.658s; after
  0.063/0.511/1.055/2.257s. All state fingerprints identical; no E2E gain yet.
- Sticky _write_failed IDs prevent early write failures from being confirmed
  when the final completion succeeds. Real GPU/O_DIRECT2fail before, pass
  after; later successful promotion preserves bytes. CPU56+60pass,GPU16pass
  (verificationON). Raw admission-diagnostic/attempt01/reclaim-*.jsonl and
  CPU/GPU test logs. All new formerly-red regressions fixed.
- CURRENT BOOT: server20772/API3903659/engine3904097/workers3904307..14;
  combined validator69787/shell3903731/current short3903733. Raw
  perf/results/2026-09-10/glm53f-tp8-reclaim-shortcircuit/attempt01/.
 22source/client hashes frozen, same native profile/verificationOFF/72GiB
  host/256GiB disk/1M context/CUDA graphs. All8ranks reuse13e053b5... AOT;
  pool4,304,412tokens exactly matches control. Health/capture pending here.
  Passive GPU monitor15354/PID3903743; GPU profiler MUST stay inactive.
  Validator runs short1000/300 c1/8/16/32 three repeats, then automatically
  long124417/2000 repeat-source same matrix. No duplicate client, serving
  source edits, or GPU microbenchmarks while this arm runs.
- Top-k comparison16shapes/3arms/4replay gates complete: global selector
  swap REJECTED (short/normal slower, narrow long faster but tie sets differ).
  Serving selector untouched. Raw2026-09-10/glm53f-pool-topk/selection.jsonl.
- Next: health and exact short/long matrix,compare idle spans against native
  attempt02. Sparse-TC matching,shared-page integration,KDA metadata,1M quality,
  TP scaling and used-kernel port remain open. Optimization goal ACTIVE.

### 2026-09-10 00:29 UTC: candidate short PASS; long matrix LIVE

- Reclaim-shortcircuit/attempt01 is healthy, cached AOT/full graph capture
  pass; exact same4,304,412-token GPU pool as native control. All canaries
  and12short exact runs pass. Medians c1/8/16/32:
  104.761441/480.099853/675.116340/904.888892tok/s, broadly unchanged from
  control. Short prompts fully computed,0hits/preemptions. Raw short-summary
  and per-run JSON; do not claim a short-workload gain from noise.
- Combined validator69787 now runs long PID3909589 (shell3903731),
 124417input/2000output/repeat-source,c1/8/16/32,three timed repeats plus
  normal warmups. Long canaries pass; throughput still pending. Do not
  duplicate it. Server20772/API3903659/engine3904097/workers3904307..14.
  Passive GPU telemetry15354/PID3903743.22source/client hashes still match.
- Finish this actual workload and compare idle spans with native attempt02.
  No serving edits or GPU microbenchmarks during the frozen arm. Profiler
  must stay inactive. Current11x claim is CPU pressure bookkeeping ONLY;
  real long-serving speedup not yet established. Goal remains ACTIVE.

### 2026-09-10 00:39 UTC: long c1/c8 complete; c16 warming; no source changes

- Current reclaim-shortcircuit/attempt01 long medians c1=76.783386,
  c8=444.799135tok/s (control76.569775/432.752712). c8 repeats
  450.431059/444.799135/439.605479: +2.78% median but still trending down.
  All6exact;1computed token/124416GPU hits per request,0external/preemptions.
  c16 cold warmup active. Do not claim c16/c32 improvement before measurement.
- Same server20772/API3903659/engine3904097/workers3904307..14. Combined
  validator69787/current long3909589,passive monitor15354/PID3903743.
 22source/client hashes match. No new GPU job, no serving edits, profilerOFF.
- KDA metadata candidate and benchmark added under benchmarks/ only; NOT
  imported by serving. Pure-prefill reuse of existing exact GDN chunk table,
  unchanged fused gate/cumulative KDA kernels. Mixed ordinary rows and spec
  explicitly fall back (row numbering differs);11CPU tests and CLI/Ruff pass.
  GPU exact output/state and timing gates are queued, NOT RUN. Raw
  admission-diagnostic/attempt01/kda-metadata-{cpu,cli}.log.
- After GPUs are genuinely free: CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1
  .venv/bin/python -m benchmarks.benchmark_glm5_next_kda_metadata. Benchmark
  uses fresh per-step cu_seqlens, within-step layer reuse, independent V
  buffers and4changed-input exact checks. No metadata serving-speedup claim.
  First finish the current long matrix and idle-span comparison; goal ACTIVE.

### 2026-09-10 00:52 UTC: long c16 stall reduction measured; c32 cold warmup LIVE

- Current reclaim-shortcircuit/attempt01 c16 repeats are
  676.465269/675.807024/669.671294 tok/s, median675.807024 versus control
  511.606749 (+32.1%). All exact, 1 computed token/124416 GPU prefix hits
  per request, zero external hits/preemptions. Walls47.305/47.351/47.785s.
  Passive telemetry has NO multi-sample all-GPU idle span in these windows;
  control r2/r3 had36.053/16.037s observed spans. This supports reduced
  admission bookkeeping stalls, NOT faster GPU kernels. Same GPU capacity.
- c32 cold warmup ongoing (22 waiting requests at00:52); do not duplicate
  the validator or mistake slow cold prefill for a hang. Same server20772,
  API3903659/engine3904097/workers3904307..14; validator69787/long3909589;
  monitor15354/PID3903743. All22 frozen source/client hashes still match.
  Profiler inactive; no new GPU work or serving edits until arm completion.
- Hash-first hybrid lookup V2 is quarantined in
  benchmarks/kv_tier_lookup_candidate.py, NOT imported by serving. Rejects
  nonmatching chains before scanning page readiness; preserves positive
  readiness, full-chain/tail checks, LRU tie-breaking and partial fallback.
  No new persistent metadata.23 differential/efficiency tests pass; isolated
  override also passes43 existing index/connector/compact tests plus17 disk
  connector/namespace/reclaim tests. GPU/E2E lookup qualification NOT RUN.
- CPU-only five alternating100-lookup repeats on pressure fixture: median
  baseline/candidate hit31969/250us, miss30766/28us, shared-prefix miss
  33719/29us, all-owner-shared-prefix miss32191/501us, all-tails-pending
  28.00/6.29us. CPU timings overlapped live serving; not serving TPS or a
  cross-version timing comparison. V1 had an unnecessary chain-copy cost
  for pending tails; V2 first-hash/tail-availability checks remove it.
  Raw admission-diagnostic/attempt01/lookup-v2-{cpu,existing-suite,
  existing-extended}.log, lookup-v2-timing.jsonl, lookup-v2-source.sha256;
  V1 source archived in lookup-v1-source.patch. No CPU test jobs remain.
- Next: finish c32, refresh telemetry-summary and full comparison, then
  choose the next source boundary. Lookup V2 and KDA metadata GPU gates are
  queued; never integrate either into this still-frozen serving arm.

### 2026-09-10 01:00 UTC: full-host-busy guard CPU-qualified; c32 still LIVE

- Same server20772/API3903659/engine3904097/workers3904307..14, validator
  69787/long3909589, monitor15354/PID3903743. c32 cold prompts still draining
  (7 waiting at00:58:46; successful responses/offloads through00:59:50).
  No new timed c32 result yet. Keep profiler inactive and serving files frozen.
- Additional quarantined benchmark candidate:
  benchmarks/kv_tier_full_busy_candidate.py, FullBusyIndex._alloc_slot.
  When no free slot exists and either len(_host_busy) or len(_pending_write)
  equals num_slots, every slot is protected; return None without scanning.
  Uses existing live collections, no persistent failure cache, new counters,
  ownership changes, or completion invalidation. Normal/main allocation
  behavior is unchanged; partial busy sets fall back to original allocator.
- Five focused CPU tests pass, including immediate recovery after just one
  completed trajectory, mixed ready/busy collections and free-slot behavior.
  Initial zero-capacity test assumed allocation could be attempted, but the
  existing constructor correctly rejects zero capacity; corrected that test
  expectation, not production code. Isolated combined lookup+full-busy override
  passes60 existing tier tests; Ruff passes. No serving integration yet.
- Three alternating CPU pressure repeats, median current-lazy-reclaim/guard
  seconds c1 0.063676/0.000447, c8 0.511554/0.003621,
  c16 1.084834/0.006853, c32 2.228115/0.015038. All allocations rejected and
  identical state fingerprints per concurrency across both arms. Same real-API
  11915-host/42366-disk pressure fixture; fixture copy excluded. CPU work
  overlaps live serving; do not extrapolate this 140–158x bookkeeping win to
  serving TPS. Raw admission-diagnostic/attempt01/full-busy-{cpu-v2,
  lookup-existing-suite}.log, full-busy-timing.jsonl/.log, full-busy-source.sha256.
  CPU jobs30189/34539/90548 are done; no new GPU jobs.
- After c32 and source/counter/telemetry recording, collect a bounded native
  c1 decode trace BEFORE stopping this server, to guide kernel work. Only
  then activate its already-configured8-iteration profiler; a diagnostic
  exact client can use --warmup-output-tokens 0 --input-tokens 124417
  --output-tokens 32 --repeat-source --allow-no-spec atc1. Explicitly stop
  profiling afterward. Profiled timing is NOT a comparison result. Then stop
  idle server and monitor, verify GPU contexts gone, run queued exclusive
  KDA metadata GPU gate. Hash-first/full-busy serving trial remains queued.

### 2026-09-10 01:08 UTC: FULL MATRIX PASS; new residual-stall diagnostic LIVE

- Combined validator69787 completed0; long3909589 gone. Long medians c1/8/
  c16/c32=76.783386/444.799135/675.807024/1003.841066 tok/s, versus control
  +0.28%/+2.78%/+32.10%/+54.34%. c32repeats1003.841066/1031.001635/
  960.592796. All24exact/canaries, timed long1computed token/124416GPU hits
  per request,0external/preemptions. Short unchanged. Same4,304,412tokens,
  sameAOT,22source/client hashes match. Profiler never active during timing.
  Retain current lazy-reclaim and sticky-disk-failure fixes; baseline updated.
- Current telemetry: c16 and c32r1/r2 have no multi-sample all-GPU idle span;
  c32r3 STILL has14.023s observed span. Do not say stalls fully eliminated.
  Raw aligned124k-summary.json, long-comparison.json, telemetry-summary.json.
  Cold warmup used7163restore ops/rank; timed-no-restore claim excludes that.
- New diagnostic, SAME server20772/API3903659/engine3904097/workers..307..314,
  monitor15354/PID3903743: engine-only10Hz110s nonblocking py-spy session
  58868, separate exact124417/2000 c32 client session64386,8-token priming.
  Raw current attempt/admission-postfix/engine-stacks.json, exact-c32.json/.log.
  Explicitly separate profiled diagnostic, not comparison TPS. No serving
  edits, GPU profiler still inactive. Check these sessions/processes; don't
  duplicate the request or assume sampling timeout means workload stopped.
- Next: inspect post-fix engine stacks for remaining admission work, then
  bounded c1 native long GPU trace (start/stop endpoints,8-iteration cap).
  Stop server only when idle; then exclusive KDA metadata GPU gate and next
  serving source boundary. Hash-first and full-busy candidates are NOT yet
  integrated. Stuck observation cell218 terminated after its actual process
  had disappeared; this did not stop any model or benchmark process.

### 2026-09-10 01:24 UTC: next admission optimizations integrated; fresh arm BOOTING

- Prior diagnostic64386 completed0: c32 exact1014.525903tok/s, same full GPU
  hits/0external/preemptions. CPU sampler58868 completed0,5020samples/95errors,
  100.4sampled engine seconds. Tier work6.1s: lookup3.7s (resumability3.5),
  staging2.0s (reclaim1.8). Inclusive times overlap and are not exact critical
  path durations. Raw reclaim-shortcircuit/attempt01/admission-postfix/
  engine-summary.json. This reinforces the queued lookup/full-busy changes.
- Native c1 long GPU diagnostic3012 completed0, eight traces at
  traces/*.1789002795*.pt.trace.json; native-long-census.json and rank-census
  saved. Rank0 sample10.813ms region/10.360ms kernel union;91allreduces1.437ms,
  11native sparse MLA0.753ms,11top-k0.728ms,37BF16GEMMs0.861ms. Across ranks
  collective sums0.625–1.531ms. These sums overlap; no cross-GPU timestamp
  alignment or E2E gain inferred. All profiling was AFTER benchmark completion.
- Attempted c32 GPU diagnostic39729 completed exact counts, but its8-iteration
  window only captured single-request admission/decode. `--tokens 32` census
  correctly rejected it. Saved *.1789002866*.traces are NOT c32 evidence.
  Next c32 capture must start after a long diagnostic actually has32running
  decode requests. CPU census now accepts --tokens, retains marker gates,
  five tests pass. No profile-wrapper fix implied by this insufficient capture.
- Old server20772/API3903659 stopped via SIGINT only after0running/0waiting;
  session completed0, all workers gone. Monitor3903743 stopped. All8GPUs
  reported0MiB/0% before exclusive tests. NVML query-compute-apps timed out;
  bounded per-GPU memory query worked and confirmed cleanup (not a GPU fault).
- KDA metadata GPU85844 completed0,7shapes/4changed-input checks exact output
  and state. Median synthetic34layer step baseline/candidate microseconds:
  singleton14084/13421,8singletons14383/13513,16singletons15298/13427,
  32singletons14853/14133,varied lengths14380/15073,1024tokens14607/13987,
  8192tokens30494/29534. Mixed-length regression/variability: DO NOT integrate
  yet. Raw admission-diagnostic/attempt01/kda-metadata-gpu.jsonl/.log and
  source hash. Pure-prefill routing only; mixed ordinary/spec still fallback.
- BF16 GEMV GPU97327 completed0:2widths,16disjointweights,11arms,8changed-input
  graph checks/arm,5alternatingtiming repeats. Width3336 cuBLAS22.788us,
  ownedDSV419.171us, groupedN4K1024 18.966us; width6416 cuBLAS36.139us,
  ownedDSV432.678us, best grouped33.714us. All pass0.008/0.008 gate, but NOT
  bit-exact (maxabs up to0.015625). Modest kernel win, no serving integration
  or mHC exact-gate relaxation. Raw2026-09-10/glm53f-bf16-gemv/timing.jsonl.
- Integrated hash-first hybrid lookup and full-busy _alloc_slot guard into
  vllm/v1/core/kv_tier_index.py AFTER server stop. No GPU kernels or profile
  settings changed. Benchmarks now retain frozen LegacyLookupIndex and
  LegacyAllocIndex references; tests exercise actual production implementations
  (do not use removed HashFirstIndex/FullBusyIndex names from earlier entries).
 96CPUtests pass; Ruff/diff checks pass. Integrated CPU benchmarks repeat
  matching state/parity: lookup hit22448/164us,miss22684/23us,all-shared miss
  22800/375us,pending19.62/4.97us; c32allocation2.235s/0.0163s. Raw next arm.
- CURRENT BOOT: raw perf/results/2026-09-10/glm53f-tp8-hash-first/attempt01/;
  server14511/API3953228 under shell3953226. Combined validator69680,
  shell3953304/current short3953305 waits for health, then short1000/300
  and automatically long124417/2000, c1/8/16/32,3repeats each. Monitor87230/
  PID3953313.22serving/client hashes frozen, source.patch and cpu-tests.log.
  Same registered native TP8 profile,verificationOFF,72GiBhost/256GiBdisk,
  1Mcontext,BF16KV,maxseq64,fullgraphs64, profiler configured but INACTIVE.
  Health/cache capacity/AOT reuse not yet verified at this checkpoint.
  No serving edits or GPU microbenchmarks while this arm is live. Goal ACTIVE.

### 2026-09-10 01:26 UTC: hash-first arm HEALTHY; matrix LIVE

- Server14511/API3953228, engine3953827, workers3954108..3954115.
  All8ranks reused13e053b5... AOT at01:25:14; full CUDA graph capture passed,
  same51.03GiB and exactly4,304,412GPU tokens. Startup/health complete.
- Validator69680/shell3953304/current short3953305: text/reasoning/image/
  tool canaries PASS; short c1 warmup64.77tok/s (not a comparison result).
  Three-repeat c1/8/16/32 short matrix now running, then same long matrix
  automatically. Monitor87230/PID3953313.22source/client hashes match.
  Profiler INACTIVE. Do not duplicate clients, edit serving files, or run
  GPU microbenchmarks until complete. New admission E2E benefit still pending.

### 2026-09-10 01:41 UTC: short complete, long c16 warming; strict depth gate added

- Hash-first arm short medians c1/8/16/32=105.119335/481.509238/
  665.244430/906.046103tok/s, all12exact/canaries, broadly unchanged versus
  reclaim-shortcircuit. Raw short-summary.json. Long c1 median76.588333,
  c8 median438.114136 (repeats444.742082/438.114136/433.332071).
  All6exact,1computed token/124416GPU hits per request,0external/preemptions.
  This does NOT establish a new serving gain; c16/c32 still pending.
- Long client now3964925 under shell3953304/session69680. Same server14511/
  API3953228/engine3953827/workers3954108..15; monitor87230/PID3953313.
  c16 cold warmup active.22source/client hashes match, profiler inactive.
  No serving edits or new GPU jobs during this turn.
- Strengthened the SEPARATE long-validation harness with --require-target:
  benchmarks/benchmark_wildchat_deepcontext.py now requires each session to
  perform a successful codename recall at actual prompt usage >=ctx-target.
  A prompt+reasoning-completion estimate is insufficient; if the first target
  probe is too shallow, growth continues. Strict markers contain session IDs
  to avoid collisions. Strict API errors stop as errors, not squeeze/ceiling
  successes; invalid/missing usage fails immediately. Raw records/qualification
  are saved before nonzero exit on time caps, missing sessions or failed gates.
  Default exploratory mode remains available; strict mode must be requested.
- Also fixed this harness's TTFT to recognize either reasoning or
  reasoning_content deltas. Nineteen CPU tests pass, including both streaming
  schemas, false depth from reasoning usage, every-session gating, missing
  usage, API errors, and CLI persistence/exit behavior. Ruff/diff checks pass.
  Raw current attempt/deepcontext-qualification-cpu-final.log, help.log,
  source.sha256 (all prefixed deepcontext-qualification- as named on disk).
  Historical633-turn/530K artifact correctly fails the new1M gate for all8
  sessions: deepcontext-historical-qualification.json. No new1M GPU run yet.
- After performance arms finish, queued actual-depth validation command:
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python
  benchmarks/benchmark_wildchat_deepcontext.py --parquet
  /home/ubuntu/.local/scratch/wildchat/data/train-00000-of-00014.parquet
  --base-url http://127.0.0.1:8400/v1 --concurrency 8 --ctx-target 1000000
  --require-target --max-hours 4 --seed 42 --out <new-run>/deepcontext_c8.json.
  Leave room under1048576 for the1024-token reasoning+recall response.
  This gate proves depth/recall only; observed physical tier restores and
  byte verification remain separate requirements. Goal remains ACTIVE.

### 2026-09-10 02:02 UTC: c16 complete; isolated warp histogram queued

- Hash-first arm long c16 repeats682.169657/679.444794/672.048175tok/s,
  median679.444794 (+0.54% versus reclaim-shortcircuit675.807024).
  All9 completed long checks exact, each request1computed token/124416GPU
  prefix hits, zero external hits/preemptions. The prior telemetry census
  through c16 observes no all-GPU idle spans in those three repeats. This
  does not establish a new serving win. c32 cold warmup remains live under
  client3964925/session69680; server and telemetry PIDs unchanged. All22
  frozen hashes match. No serving edits, profiler activation or GPU tests.
- Quarantined first-pass top-k histogram candidate, NOT serving-integrated:
  csrc/quixicore/glm53f_warp_histogram.cuh and glm53f_histogram_benchmark.cu,
  benchmarks/benchmark_glm5_next_warp_histogram.py. Same-bin warp lanes elect
  one atomicAdd(popcount) leader using match_any_sync. Hypothesis: narrow
  score distributions serialize shared histogram atomics; integer counts
  can be preserved while reducing atomic traffic. Existing full top-k
  alternatives regress other distributions, so no blanket selector swap.
- Offline nvcc SM80 shared-library build succeeded;12CPU oracle/fixture
  tests pass. Native GPU correctness/timing NOT RUN. The GPU gate includes
  exact CPU histograms, changed-input/length CUDA graph replays, poisoned
  padding and output guards; separate32-row cases exercise lengths0..4096,
  vector tails, partial warps,512-thread loop boundaries, signed zero and
  infinities. Count parity is NOT selected-ID parity or full top-k timing.
- Raw perf/results/2026-09-10/glm53f-warp-histogram/ contains libhistogram.so,
  build.log, cpu-oracle.log and source.sha256. Only after this serving arm
  and any post-matrix trace complete, stop the owned idle server and run:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m
  benchmarks.benchmark_glm5_next_warp_histogram --library
  /home/ubuntu/SlimServe/perf/results/2026-09-10/glm53f-warp-histogram/libhistogram.so.
  Capture full c32 decode only after32requests are actually decoding; the
  previous early-start trace captured singleton admission steps instead.

### 2026-09-10 02:11 UTC: hash-first matrix COMPLETE; post-matrix c32 trace LIVE

- Validator69680/long3964925 completed exit0; server3953228/3953827 and
  workers3954108..15 remain alive. Telemetry3953313 stopped withSIGINT
  after all timed runs.22frozen hashes match; profiler never activated in
  comparison. matrix-comparison.json records all24exact checks and full
  repeat arrays against reclaim-shortcircuit. Canaries pass.
- Short medians105.119335/481.509238/665.244430/906.046103; long medians
  76.588333/438.114136/679.444794/1056.612837. Long changes versus prior
  -0.25%/-1.50%/+0.54%/+5.26%. c32 repeats1056.612837/1067.498366/
  981.721960. First2 have no observed all-GPU idle spans; r3 has4.006s
  plus an isolated zero sample. Every long timed request1computed token,
  124416GPU hits, zero external/preemptions. Whole arm6916restores/rank
  during warmup/priming. Retain CPU additions, not a GPU kernel speedup;
  residual stalls and small c8/short-c16 regressions remain visible.
- POST-MATRIX diagnostic launched in session53161: same c32/124417prompt,
  4000outputs, warmup-output-tokens0, otherwise identical sampling. Artifacts
  native-long-c32-diagnostic.json/.log under current attempt; EXCLUDED from
  timing comparison. Trigger session58552 polls metrics once/sec until
  running32/waiting0 with increasing generation and unchanged prompt counts
  over2intervals, then POSTs start_profile. Profiler configured max8iterations.
  Trigger log native-long-c32-profile-trigger.jsonl. Need wait client finish,
  POST stop_profile if needed and verify actual32-token GPU annotation on
  every rank before calling this a c32 trace. No competing GPU microbenchmarks.

### 2026-09-10 02:15 UTC: c32 trace validated; server STOPPED for isolated GPU gate

- Diagnostic53161/client4013207 completed exit0:32x4000outputs exact,
  same1computed token/124416GPU prefix hits per request, zero external/
  preemptions. Trigger58552 completed0 and started profiling only after
  two stable32-running/0-waiting decode intervals; POST stop_profile also
  completed after client exit. All8rank traces contain a complete32-token
  annotation with90mHC sites; native-long-c32-rank-census.json/.log.
- Rank0 region22565.803us, union22233.528us, sum23577.094us across
  overlapping streams.84Marlin MoE calls5949.998us,11native sparse MLA
  2758.310us,91allreduces2102.506us,11cached pool logits1998.504us,
  90mHC partials1618.395us,11topK789.532us. Rank-local region range
  22462.661..22644.491us. Do not align timestamps across ranks or add
  overlapping kernel sums as wall time. MoE, sparse attention and scoring
  are larger c32 targets than the histogram alone.
- At confirmed0running/0waiting, SIGINT sent to owned API3953228.
  Server14511 exited0; API/engine/all8workers are gone. Teardown took
  several seconds; a bounded nvidia-smi query timed out during teardown,
  then all8devices confirmed0MiB/0% at02:15. No reset/forced kill used.
- Isolated warp-histogram GPU benchmark launched only after that clean
  release. Raw glm53f-warp-histogram/gpu.jsonl/.log. No serving process
  currently running. Await exact native gates and full36timing cases before
  deciding on any kernel integration; next matched sparse-TC arm remains
  queued, along with strict1M qualification and scaling work.

### 2026-09-10 02:18 UTC: histogram REJECTED; sparse-TC A/B preparing

- Histogram session77825 completed0: native32-row edge cases and all36
  timing cases pass exact histograms, poisoned padding, guards and changed
  graph replays. Candidate is SLOWER in every case:1.19x..9.04x baseline
  time. Reject for serving. No sampler/native-extension edits made.
  Disassembly shows baseline ATOMS.POPC.INC.32 versus explicit MATCH.ANY,
  votes and ATOMS.ADD in candidate. Extra software grouping is unhelpful
  here; no whole-selector speed claim. Raw gpu.jsonl/.log and disassembly.sass.
- First native attempt97000 failed the graph replay gate: benchmark cached
  current stream outside torch.cuda.graph, launching on the wrong stream
  and capturing an empty graph. Fixed BOTH benchmark launch sites to resolve
  current stream at launch. Failed logs/hashes preserved as gpu-failed-capture
  and source-failed-capture.sha256; corrected run contains all37records.
  This was a diagnostic harness bug, not evidence of a serving-kernel failure.
- Enabled ONLY glm5_next_sparse_tc_decode in TP8 registry for the next
  controlled A/B (pending retention; revert if rejected). All admission fixes,
  BF16 KV,tiers,graphs and other flags unchanged. Raw next arm
  perf/results/2026-09-10/glm53f-tp8-admission-sparse-tc/attempt01/.
  Fresh98GPU parity/native tests session47875, CPU dispatch/profile suite
  session74785. No server yet; wait these GPU tests before launch. Baseline
  is completed hash-first/attempt01, not older pre-admission-fix timings.

### 2026-09-10 02:20 UTC: sparse-TC matched arm BOOTING

- Fresh GPU suite98passed in6.81s; CPU dispatch/mHC/profile suite94passed
  in6.35s. GPU test processes exited0 and all8devices confirmed0MiB/0%.
- CURRENT server session21893/shell4019515/API4019529/engine4020072.
  Command: SLIMSERVE_KV_TIER_DIR=/home/ubuntu/.cache/slimserve/kv-tier
  VLLM_KV_TIER_VERIFY=0 PYTHONFAULTHANDLER=1 .venv/bin/python -m
  slimserve.cli glm53f-nvfp4-8 --serve --host 127.0.0.1 --port 8400
  --torch-profile-dir perf/results/2026-09-10/glm53f-tp8-admission-sparse-tc/attempt01/traces -y.
- Validator session14576/shell4020306/short4020307 waits for health, then
  short1000/300 c1/8/16/32 three repeats; aligned124417/2000 with repeat-source
  automatically follows under same shell. Telemetry81862/PID4020316 polls
  all GPUs every2s. Do not duplicate, enable profiling or run competing GPU
  work. No serving source edits while this arm lives.
- New22-file source-client-sha256 differs from baseline ONLY profiles.json;
  replacing the one sparse-TC true flag withfalse reproduces baseline profile
  hashccbb2615486a57fd7c3d2218b53415b35bc5f76f28146ddb390e4517ab7a832a.
  Health/AOT/graph capture/capacity/serving quality still pending. This is an
  experiment in progress, not validated TP8 serving or a retained default.

### 2026-09-10 02:34 UTC: sparse-TC short PASS; long c1 complete; MoE sweep prepared

- Sparse-TC arm reached health, text/reasoning/image/tool canaries PASS,
  FULL_DECODE_ONLY graph capture passed (0.22GiB). New AOT key
  e2bc7d0f7d2ee0b31734452b358caec1c4f604c5d8778807048ab569a57c4b03
  was compiled/saved in this boot, not yet a fresh-process cache-hit test.
- MEMORY TRADEOFF: GPU KV is4,280,453tokens/50.74GiB, versus native
  hash-first4,304,412tokens/51.03GiB:23,959fewer tokens (-0.56%). Same
  config/geometry except sparse-TC flag; do NOT claim unchanged capacity.
- Short1000/300 three-repeat c1/8/16/32 medians109.717290/489.634384/
  682.751705/928.972062tok/s: +4.37%/+1.69%/+2.63%/+2.53% versus
  hash-first. All12exact, no external hits/preemptions. short-comparison.json
  stores repeat arrays and explicit pool sizes. Full retention remains pending.
- Long client now4034488 under shell4020306/session14576; short4020307
  exited0 and pipeline advanced automatically. Long c1 repeats102.610098/
  102.235713/103.416627 (median102.610098 versus76.588333). All3exact,
  onecomputed token/124416GPU prefix hits, zero external/preemptions or
  observed all-GPU idle spans. c8 cold warmup active; c16/c32 pending.
  Server21893/API4019529/engine4020072 and telemetry81862/PID4020316
  remain live.22frozen hashes match. Profiler inactive; no new GPU jobs.
- Prepared CPU-only benchmarks/benchmark_glm5_next_marlin_tiles.py and
  tests/glm5_next/test_marlin_tile_plan.py:16fixture/config tests pass.
  Existing native Marlin tuning arguments, actual TP8 NVFP4 shapes
  gate/up4096->512/down256->4096,E288/top8/block8/BF16,17auto/tile/CTA
  choices, c1/8/16/32 and synthetic disjoint/shared/uniform routing.
  Uses one packed weight bank and16rotated route sets to avoid tiny hot
  expert-only L2 timing. Explicit touched-byte count, four changed graph
  replays with poisoned outputs/guards, finite/nonzero reference, and
  bit-exact flag; non-bit-exact candidates are NOT qualified. Both GEMMs
  timed separately; alignment/SwiGLU/reduce/fullmodel gates still required.
- No MoE GPU run or serving change. Queue after live arm ends/releasesGPUs:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m
  benchmarks.benchmark_glm5_next_marlin_tiles > <new-run>/gpu.jsonl.
  Raw preparation perf/results/2026-09-10/glm53f-marlin-tiles/:
  cpu-plan.log,help.log,source.sha256,native-launch-census.json.
  Native c1 trace has42Marlin launches at216CTAs/82,432B shared and42at
  432CTAs/40,704B; c32 all84use432CTAs/40,704B,128threads/122registers.
  Tile choice affects resource use, but real expert occupancy is NOT known
  from these traces; route fixtures are synthetic, not measured routing.

### 2026-09-10 02:36 UTC: sparse-TC long c8 complete; c16 warming

- Long c8 repeats439.879771/441.076611/439.705033, median439.879771,
  only+0.40% versus native438.114136. c1 gain+33.98% must not be
  generalized to c8. All6completed long checks exact, onecomputed token/
  124416GPU prefix hits per request, no external hits/preemptions or
  observed all-GPU idle spans. Raw long-partial-comparison.json and refreshed
  telemetry-summary.json. Short comparison remains +4.37/+1.69/+2.63/+2.53%.
- Long4034488/session14576 confirmed live; c16warming, c32pending.
  Server/monitor PIDs unchanged,22frozen hashes match, no profiler activation
  or additional GPU workloads. MoE tile preparation ends with16CPU tests
  passing in2.05s and matching source hashes; GPU gates remain queued.
  Full1M qualification, candidate cached restart and TP scaling remain open.

### 2026-09-10 03:01 UTC: c16 complete; fixed-K down kernel CPU/offline gates

- Sparse-TC long c16 median706.513174tok/s, repeats700.141336/
  706.513174/710.484107: +3.98% versus native679.444794. All9completed
  long checks exact,1computed token/124416GPU prefix hits per request,
  zero external/preemptions or observed all-GPU idle spans. Refreshed
  long-partial-comparison.json/telemetry-summary.json. c32 cold warmup
  remains live; two waiting requests at02:59:44. Server/client/monitor IDs
  unchanged. Profiler inactive; no serving edits or GPU microbenchmarks.
- New quarantined benchmarks/glm5_next_nvfp4_down_candidate.py implements
  fullK256/N4096/E288 BF16 NVFP4 down, reading existing packed Marlin
  weights/scales with expert-aligned8row metadata. One CTA per full-K/Ntile,
  no global partial-sum buffer/locks/weight repack. Integer exponent rebias
  reconstructs native tiny BF16 weights without FP32 subnormal arithmetic.
  Preserves accumulator->BF16 and (router*stored global)->BF16 BEFORE final
  multiply. Different MMA order still needs native/full-model quality gates.
- Literal native bit oracle covers16FP4codes x120valid scale bytes plus
  signed zeros/clipping; converter test enumerates all finite nonnegative
  E4M3 inputs. Counterexamples show why neither epilogue rounding can be
  removed. Initial scale-MSB shift6 was wrong; shift7 fixed after CPU test
  failed. Initial log/hash retained as cpu-contract-initial/source-initial.
- Offline SM80 compilation succeeded: Ntile64/128 require40,960/73,728B
  shared memory.32combined CPU tests pass (27MoE fixture/scheduler +5numeric
  contracts). NO native GPU execution/timing of this new kernel yet.
- Important scheduler correction: Marlin already uses full-K data-parallel
  tiles plus a small stream-K tail, with global reduction only for tiles
  split over CTAs. CPU mirroring shows c32 disjoint routing/current432CTA
  down launch has7776DP+416tail tiles but ZERO split tiles; c1 down256tiles
  are all split. Cannot claim allc32MoE cost is global reduction. Actual
  model routing remains unmeasured; synthetic route variety is deliberate.
- MoE sweep now supports --direct-down (off by default) to compare both new
  tiles alongside17existing Marlin configurations only for the down leg.
  Logs each arm and replay result; checks timed graph output against same
  kernel eager output after every repeat, and guards after timing. Finite
  non-bit-exact candidates remain diagnostics, NOT qualified replacements.
- Latest raw/source hashes for BOTH MoE benchmark and new kernel/tests:
  perf/results/2026-09-10/glm53f-nvfp4-direct-down/ (cpu-combined.log,
  offline.jsonl/.log,source.sha256,help.log). Prior marlin-tiles hashes describe
  the earlier pre-direct benchmark, not current source.
  After full serving matrix and any profile diagnostics finish/freeGPUs:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m
  benchmarks.benchmark_glm5_next_marlin_tiles --direct-down > <new-run>/gpu.jsonl
  2> <new-run>/gpu.log. Do not run while current server lives.

### 2026-09-10: sparse-TC full matrix complete; post-matrix c32 capture live

- All24exact checks pass, canaries pass,22source/client hashes match.
  Long c32 median1162.670308tok/s (+10.04% versus native1056.612837),
  repeats1162.670308/1195.071181/1035.037738. r3 has8.014s sampled all-GPU
  idle span; r1/r2 none. No claim of eliminating admission stalls.
  full-matrix-audit.json and refreshed telemetry-summary.json saved.
- Server4019529/engine4020072 remains healthy; long validator4034488 gone,
  metrics0running/0waiting confirmed before diagnostics. Telemetry4020316
  stays live. No serving edits made. Sparse-TC retained only as current
  candidate; cached restart/full1M/combined tier and scaling gates open.
- Postmatrix diagnostic session45847:32requests,124417prompt,4000output,
  repeat-source,warmup0, same benchmark_dsv4_exact and source. Trigger50167
  waits two32-running/0-waiting intervals with stableprompt/increasingdecode
  before POSTstart_profile. Raw tc-long-c32-diagnostic.json/.log,
  tc-long-c32-profile-trigger.jsonl, run-specific profile_when_decode.py.
  Diagnostic timings are NOT performance matrix results. Stop profiler
  after client exits; census must verify32token annotation and90mHC sites.

### 2026-09-10 03:13 UTC: c1/c32 trace gates passed; server gracefully stopping

- c32 diagnostic45847/trigger50167 both exited0; exact128000outputs,
  full GPU-prefix hits,0external/preemption. All8traces validate32tokens/
  90mHCsites. Rank0 region23372.728us versus native22565.803us, despite
  sparseTCpart980.772+reduce113.955us versus nativeMLA2758.310us.
  Marlin sum8089.021us versus5949.998us. Single sampled profile region
  is NOT an end-to-end speedup estimate or deterministic routing match.
- c1 diagnostic79510/trigger42807 both exited0; exact2000outputs, full
  prefix hits,0external/preemption. All8censuses pass1token/90mHCsites.
  Rank0 region12361.179us; TCpart77.643+reduce36.475us versus native
  MLA752.886us. Allreduce3840.856us versus prior1437.295us; must not
  explain the entire34% serving gain by this one trace. Raw filenames
  tc-long-c1-diagnostic and traces/*1789009934*.census.json; c32*1789009806*.
- Profiler stopped after each diagnostic. Confirmed0running/0waiting,
  then SIGINT4019529 and TERMtelemetry4020316. Await all8workers gone
  and0MiB GPUs before isolated MoE work. Fresh CPU32tests pass again in
  cpu-pre-gpu.log; source.sha256 matches both new benchmark/kernel/tests.

### 2026-09-10 03:20 UTC: isolated MoE sweep LIVE; optional sparse scratch prepared

- Server/API/engine/all8workers exited; all8GPUs confirmed0MiB/0% before
  new GPU work. MoE sweep session88173/shell4086897/PID4086899 remains
  LIVE (confirmed03:19), GPU0 only, no server or other GPU job. Command:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m
  benchmarks.benchmark_glm5_next_marlin_tiles --direct-down.
  Raw perf/results/2026-09-10/glm53f-marlin-tiles/gpu-attempt01/:
  gpu.jsonl,gpu.log,source.sha256,partial-summary.json. Do NOT duplicate.
- Seven complete cases so far (allc1routing/legs plusc8disjointgateup);
  c8disjointdown timing underway. Changed-input graph/guard/finite checks
  pass so far. No material bit-exact c1tile win: auto gateup~15.6us/down
  ~13.2us. Native down64x128/3CTA is~10.08us but NOTbitexact(max0.001953125).
  Directdown64/128~27.57/28.44us, NOTbitexact(max0.00390625), slower.
  Do not generalize partialc1results to allrouting/concurrency or serving.
- While GPU sweep lives, added optional fixed-address SparseTCWorkspace
  in vllm/quixicore/sparse_mla_workspace.py and optionalworkspace kwarg
  to sparse_mla_tc.sparse_tc_nope. NO backend/model integration; default
  private-allocation path unchanged. No edits to live MoE benchmark/kernel.
  Model-owned sequential calls only; concurrentstreams/microbatches need
  separateowners. Scratch excludes outputs and neverresizes aftercapture.
- For registered2080indices/H8 andc1..32splitpolicy, max544row-parttiles,
  exactly8,947,712B residentstorage.10CPUtests pass;5GPU graph/lifetime
  tests pending. New benchmark_glm5_next_sparse_scratch.py additionally
  captures allfour shapes into sharedgraphpool, replays interleavedshapes,
  compares private/shared bitexact and records allocation/peak/timing.
  IMPORTANT: graphallocator may already reuse private temporaries. This
  is a measured hypothesis, not a claimed11xallocation orKVcapacitygain.
- Raw prep perf/results/2026-09-10/glm53f-sparse-tc-shared-scratch/:
  cpu.log,help.log,source.sha256. Ruff/diffcheckpass. AFTER88173exits/GPUfree:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
  tests/glm5_next/test_sparse_tc_workspace.py
  Then twofreshprocesses benchmark_glm5_next_sparse_scratch (private default,
  --shared candidate), separate rawJSON/log. If memory/TPS not better,
  rejectsharedscratch; no productionwiring merely becauseCPUtestspass.
  Candidatecachedserverrestart, strict1Mqualification, rowshard8GPUtest,
  residualadmissionstalls and TPscaling remainopen. Goalactive.

### 2026-09-10 03:33 UTC: GPU queue serialized; routing diagnostic CLI ready

- MoE88173/PID4086899 stilllive (20/24cases at03:32, c32sharedgateup).
  Prior turn is progress: fullnativeGPU evidence plus scratch implementation.
  Current partial results show no material bitexact wins on disjoint routes;
  sharedc8gateup auto29.030938us versus128x64/2CTA15.733628us, but maxabs
  0.03125/notbitexact. Actual model route occupancy not yet measured.
- IMPORTANT LIVE QUEUE: scratchpipeline77297/shell4095546 waits for
  MoE4086899 exit, then requires24validrecords/sourcehashes/all8GPUs0MiB/0%,
  runs15scratchtests, privateallocation benchmark, freshprocess--shared.
  Script/rawlogs under sparse-tc-shared-scratch/run_after_moe.sh,pipeline.log.
  Rowshardpipeline96609/shell4096928 then waits for scratch4095546 exit,
  verifiesbotharmJSONgates/sourcehashes/idleGPUs and runs torchrun8ranks
  -m benchmarks.benchmark_glm5_next_indexer_row_shard. Raw under
  perf/results/2026-09-10/glm53f-indexer-row-shard/;36CPUtests pass.
  DO NOT start servers or other GPU work until BOTH queued pipelines finish.
  A failed prior gate aborts its successor; inspectlogs, do not duplicate.
- Added CLI --enable-return-routed-experts as explicit diagnostic override,
  preserving allregisteredmodel/quant/KV/tier/graph/spec settings; defaultoff.
  Uses existing BaseRouter.capture_fn and RoutedExpertsCapturer, no newGPU
  capture implementation.67profile/CLItests pass; actualdryrun succeeds.
  Flag adds GPU/hosttransit buffers, CPUslotstore and perstepD2H/API payload;
  such runs arediagnostics, notbaselineTPS. Freshcachednormalboot stillneeded.
- Exactclient --dump-responses DIR saves fullJSON aftertimer/metrics so
  routed_experts payload is not discarded.21clientCPUtests pass, including
  explicitout-of-timerwriteordering and unchangeddefaultbehavior. Full
  responseexpertIDs are per-token, NOT scheduler-step batchmembership;
  do not call token-offset alignment measured concurrentrouteoccupancy.
- Raw CLI/client prep glm53f-routing-capture/{profile-cpu.log,client-cpu.log,
  cli-cpu.log,dry-run.txt,source.sha256}. No actualrouting-enabledserver yet.
  MoEbenchmarkfourfilehashes unchanged; CLI/exactclient changed only after
  prior servingmatrix andpostmatrixdiagnostics had completed/exited.

### 2026-09-10 03:42 UTC: MoE/scratch rejected; row-sharding GPU win; cached restart LIVE

- All prior GPU pipelines completed0:88173MoE,77297scratch,96609rowshard.
  MoE24/24cases complete, fourchangedreplays and five timingrepeats each,
  guards/finite/replaystability passed. No bitexactnativeconfig improved
  by1%. Directdownboth tiles NOTbitexact inall12downcases and2.094..4.575x
  SLOWER than auto. Rejectdirectkernel and statictile tuning forserving.
  Fullraw marlin-tiles/gpu-attempt01/{gpu.jsonl,gpu.log,summary.json}.
- Scratch15GPU/CPUtests passed, bothfreshprocessallocationarms exact.
  Peakallocated120,365,568B IDENTICAL. Sharedadds8,947,712B persistent
  allocation, reservedbytes150,994,944->130,023,424 (-20MiB), not proof
  of a larger servingKVpool. Latency effectivelyunchanged; rejectforcapacity.
  Moved class AND optionalwrapper under benchmarks/glm5_next_sparse_scratch_candidate.py;
  production vllm/quixicore/sparse_mla_tc.py restored EXACT originalhash.
  No workspace class remainsunder vllm/quixicore. Diagnosticbenchmark/tests
  import quarantinedwrapper. Postcleanup99GPUtests pass in6.72s and two
  new allocationarms reproduceequalpeak/latency. Raw post-quarantine-gpu.log,
  private-quarantined.json,shared-quarantined.json,post-quarantine.sha256.
  Earlier source.sha256 describesprequarantinepaths; useposthash forcurrent.
- Rowshard12/12cases,8ranks, fourchangedgraphreplays each pass exactvalid
  logits and selectedsets. At131072context replicated->sharded us/layer:
  c8 72.720757->67.984291 (1.070x), c16 111.548973->72.802681(1.532x),
  c32 190.568268->80.019081(2.382x). c1unchangedfallback. At1024context
  c8/16/32 SLOWER~3.14..3.55x;16384c8/16slower,c32neutral. No serving
  integrationyet; kernels+NCCLgatheronly, commonquery/cacheupdate/expansion
  excluded. ShortcachescanbeL2hot. Fullsummary glm53f-indexer-row-shard/summary.json.
- Nextrowshardintegration must preserve graph-capture semantics: cannotbranch
  on a Pythoncontextlength duringcapture and expectit to change onreplay.
  A first boundedexperiment can use capture-stableR16/32 policy with measured
  shortcontexttax, leavingc1/c8unchanged; or build actualdynamicGPUgating/
  separategraphvariants. Gather512selectedpoolIDs beforeexpand2080tokens,
  preserveglobalrow_req IDs and compactcache strides; allTP ranks must use
  identicalcollectiveorder. Existing _pooled_select combines score/select/
  expand and needsafactor without changingbaselinearithmetic. RealA/Brequired.
- PostcleanupGPUsall0MiB/0% confirmed, no queuedjobsremain. CURRENT normal
  registeredTP8 sparseTCcachedrestart: server2088/shell4104771/API4104772,
  validator67687/PID4104846 (shortc1/8/16/32 three repeats afterhealth),
  telemetry34259/shell4104851. Raw glm53f-tp8-sparse-tc-cached-restart/attempt01/.
  Same servecommand/tiers/flags asmatchedTCarm; profilerconfiguredbutINACTIVE,
  routingcaptureOFF, sharedTCscratchOFF/removed, noMoEtileoverride.
 22source/client hashes pluscli.sha256 captured. Sourcefreeze/GPUexclusivity
  whilevalidatorruns. At03:42workers loading; cachedAOT/health/canaries/TPS
  notyet claimed. DO NOT duplicate/restart onobservationtimeouts.
  Strict1Mqualification/physicalcombinedtiers/TPscaling/rowshardserving remainopen.

### 03:44 UTC: cached AOT loaded on8ranks; KV capacity restored without scratch

- Current engine4105389, workers4105640..4105647, telemetry4104857.
  All8ranks explicitly "Directly load AOT compilation" keye2bc7d0f7d2ee0b3...
  at03:43:44. SameTCflagtrue, originalproductionwrapperhash restored.
- GPUKV now4,304,412tokens (same asnativebaseline), versus4,280,453
  onfirstTCcompileboot. The earlier−0.56% was an observedboot allocation
  difference, NOT a demonstrated permanentTCkernelcapacitytax. Cached
  versusnewcompile changesbootstate; causeofpeakdifference notisolated.
  Do NOT attribute restoredcapacity to rejectedsharedscratch (notpresent).
  Keep shortvalidator67687/PID4104846 running; health/fullgraph/canary/TPS
  verification stillneeded. Sourcefrozen, profilerinactive, no competingGPUjobs.

### 2026-09-10 03:49 UTC: cached restart fully passed short matrix

- Validator67687/PID4104846 completed0; all12exact checks and text/reasoning/
  image/toolcanaries pass. Three-repeatmedians c1/8/16/32:
 109.931906/492.040836/681.866542/933.963014tok/s. Fullgraphs captured0.22GiB;
  all8AOTcacheloads confirmed, GPUKV4,304,412tokens/51.03GiB. Activation
  peak0.94GiB versus1.16GiB onfirstTCcompileboot; totalnonKVusagealsochanged.
  Cause notisolated, but earliercapacitycost isnot persistent here.
- Source/clienthashes match; profilerneveractivated; metrics0running/0waiting.
  Short-summary.json/telemetry-summary.json saved. Telemetry4104857 stopped
  aftertimings. Server2088/API4104772/engine4105389/workers4105640..47
  remains HEALTHY/IDLE on8400. No otherGPUjobs orqueuedpipelines remain.
- Retain sparseTC flagfor testedTP8/H8/SM80/no-spec decode shapes: matched
  fullshort+124Kcomparison andcachedrestart passed. Full1M/combinedphysical
  tiers/TPscaling stillopen; retention isnot claimingallqualificationdone.
- Next actualoptimization: rowshard integration/A/B (GPUkernel+collective
  gatespassed12cases) with explicitshortcontexttradeoff and graph-safepolicy.
  ConsiderR16/32only first; c1/c8 unchanged. Before anyGPUisolatedtests or
  rebuiltserver, gracefullystoptheownedidleAPI andverifyallGPUsreleased.
  No routing-enabledcaptureboot yet. Historicalqueue scripts are completed
  artifacts, NOT commands to rerun afterscratchquarantinechangedtheirpaths.

### 2026-09-10 04:12 UTC: row-sharded indexer integrated; cached full matrix LIVE

- Status: serving candidate, NOT retained. The preceding status turn verified
  qualification completion; this turn audited all four exact JSONs and source
  hashes, stopped the idle qualification server and launched a cached matrix.
- Implementation in glm5_next_indexer.py factors unchanged score/topK into
  _pooled_topk. Each rank scores contiguous R/8 rows, preserving global
  row_req IDs/strided compact pages; gathers512int32 pool IDs per row before
  unchanged expansion. Runtime group resolution inside the opaque op avoids
  serializing communicator pointers. Strict SM80/TP8/DP1/PP1/H32/compact/
  no-spec gate; only puredecodeR16/32. c1/c8/prefill/other shapes unchanged.
  No Pythoncontext-length branch frozen at capture; adaptive scorer rejected.
  TP8 registry row-shard flag is true EXPERIMENTALLY, pending serving A/B.
- Integrated actual-vLLM-group eight-GPU benchmark:8cases, fivechanged graph
  replays, ragged/zero lengths, query/page/request remapping, expanded-set/
  bounds/guard and post-timing parity pass. At131072context c16 baseline
  115.306->74.465us/layer, c32 194.016->82.017us INCLUDING expansion. At1024,
  c16 12.223->32.662us and c32 12.817->32.260us: explicit shortcontext tax.
  c1/c8 fallback unchanged.112 initialCPU/104 unchanged-pathGPU/105 post-enable
  CPU gates pass. Raw glm53f-indexer-row-shard-integrated/ under2026-09-10.
- First real compile boot passes text/reasoning/image/tool canaries and four
  exact1000/300 checks:109.30/492.01/670.28/922.76tok/s atc1/8/16/32.
  One repeat: qualification, NOT retention evidence.22source/client hashes
  and CLI hash match. GPUKV4,280,453tokens; graph capture0.70GiB versus
  baseline0.22GiB. Additional graphmemory must be counted, no capacity win.
  AOT20013379fd5afd76b03c40f319a150c9f05cff6cf685a98fa6907c10a586247a.
  API4126249 stopped gracefully after0running/0waiting; all8workers gone
  and all8GPUs0MiB/0% verified before fresh launch.
- CURRENT cached-attempt01: server9671/API4139241; matrix48726/shell4139331/
  shortvalidator4139332; passive GPUtelemetry97235. Raw root
  perf/results/2026-09-10/glm53f-tp8-row-shard/cached-attempt01/.
  run_matrix.sh waits for health, runs canaries plus three exact repeats per
  c1/c8/c16/c32 at1000/300, then124417/2000 (--repeat-source), and checks
  hashes/all24exact outputs. Same registered fullcontext/BF16KV/72GiBhost+
  256GiBdisk perrank; SLIMSERVE_KV_TIER_DIR=/home/ubuntu/.cache/slimserve/kv-tier,
  verification0. Sources frozen, profiler NEVER activated, routingcaptureoff,
  no competing GPU jobs. Do not restart on observation timeout.
- CachedAOT/health/newcapacity/repeatedTPS pending at04:12. Short control is
  sparse-TC cached-restart109.931906/492.040836/681.866542/933.963014tok/s,
  4,304,412GPUKVtokens. Long TC control102.610098/439.879771/706.513174/
  1162.670308 was a firstcompileboot with4,280,453tokens; a cached no-row-
  shard long control may be needed for clean retention. Full1M/physical
  combined-tier validation and TPscaling remain open.

### 04:17 UTC: cached capacity confirmed; communicator-reuse candidate prepared

- All8ranks loaded AOT20013379... at04:12:20. GPUKV4,304,412tokens/
  51.03GiB, equal to retained cached sparse-TC control. Capture still0.70GiB
  versus0.22GiB; unchanged token count does not erase additional graphmemory.
  Cached serving canaries pass; short matrix running throughc32.
- Read owned cuda_communicator.py/pynccl.py: an existing caller-stream NCCL
  all_gather is available. Current serving rowshard uses PyTorch device_group.
  Hypothesis ONLY: reusing existing PyNccl can avoid another communicator's
  storage/internal-stream overhead. No attribution proven yet.
- New benchmark-only glm5_next_indexer_gather_candidate.py preserves score/
  select/expand and uses live group.device_communicator.pynccl_comm.all_gather.
  Fails closed on missing/disabled communicator, no production wiring or
  communicator pointers serialized.44 CPU plumbing/dispatch tests pass;
  GPU correctness/performance/memory all PENDING. Ruff/help pass.
- Integrated benchmark now supports --gather existing-pynccl (default remains
  process-group). Coordination barriers/max-rank timing reduction use CPU
  Gloo, so benchmark coordination does not initialize the very extra GPU
  communicator being measured. Added per-rank allocator/free-memory snapshots
  before case/warmup and after warmup/capture. Use separate fresh processes
  for both arms AFTER the serving matrix and GPU release. No GPUjob queued.
  Original integrated results used the old benchmark revision; new source
  hashes in perf/results/2026-09-10/glm53f-indexer-existing-pynccl/source.sha256
  distinguish this version. Existing serving source hashes remain unchanged.
- Next isolated commands after full matrix/server exit: torchrun8ranks
  -m benchmarks.benchmark_glm5_next_indexer_row_shard_integrated --gather
  process-group, then separate freshprocess --gather existing-pynccl, with
  separate JSONL/logs. Tests don't establish collective or model parity.

### 04:18 UTC: cached row-shard short matrix complete; long matrix LIVE

- All12short exact checks and canaries pass. Three-repeat c1/c8/c16/c32
  medians110.328039/492.317002/674.969529/924.313557tok/s. Against cached
  no-row-shard TC control: +0.36/+0.06/-1.01/-1.03%. Short-context penalty
  is measured, not a retained win. Same4,304,412KVtokens, graphcost0.70GiB.
- Raw cached-attempt01/short-summary.json, short-comparison.json and refreshed
  telemetry-summary.json. Serving hashes still match; profiler inactive.
  Long validator4151603 now live under matrix48726/shell4139331; long canaries
  pass,124417/2000 three repeats perconcurrency underway. API4139241 stilllive;
  monitor97235 stilllive. Leave source frozen and GPUs exclusive.
- No queued GPU microbenchmark. New existing-PyNccl candidate is CPU-gated
  preparation only; do not run its torchrun while this server owns GPUs.
  Full long matrix, fresh cached no-shard long control if required, actual
  gather memory/parity tests,1M/physicaltiers/TPscaling remain open.

### 04:21 UTC: serial communicator experiment queue LIVE

- Previous goal turn was progress: completed short matrix, measured the1%
  high-concurrency tax, and implemented/tested the benchmark-only candidate.
  Current long validator4151603 remains live; c1three repeats104.04/103.68/
  105.04tok/s, exact,1computedtoken/124416GPU-prefix hits perrequest and0
  external/preemptions. c8warmup is doing coldprefill; intervalTPS is NOT
  used as serving evidence. No allGPUidle spans in12short timing windows.
- NEW live queue session35561/shell4155597 waits for matrix4139331 exit,
  then requires24exactJSONs and serving hashes, saves longsummary/telemetry,
  verifies API4139241's identity and0running/0waiting before gracefulSIGINT.
  It requires APIexit and all8GPUs0MiB/0% before proceeding, stops the exact
  owned telemetry4139361, then runs two fresh eight-rank processes:
  process-group control, existing-pynccl candidate. Both require8parity/guard
  result records and unchanged benchmark hashes. Failure aborts successor.
  Script perf/results/2026-09-10/glm53f-indexer-existing-pynccl/run_after_matrix.sh;
  pipeline.log and per-armJSONL/logs in same root. DO NOT duplicate queue.
  Benchmark sources now frozen too. No other GPUjobs or automaticserverboot.
  Once complete inspect maxranklatency AND perrankfree/allocated/reserved
  snapshots; isolated numbers are not a modelTPS/servingKVcapacity claim.

### 04:28 UTC: long c1/c8 complete; matched cached control now required

- Long c8 repeats534.987973/522.988205/476.748025tok/s, median522.988205;
  all exact124417/2000, eachrequest124416GPU-prefix hits/1computedtoken,
  zeroexternal/preemptions. Old TC long control median439.879771 had the
  same workload/cachemetric counts but a firstcompileboot and older client
  (new dump-responses option is defaultoff). Nominalc8 dispatch is unchanged,
  so DO NOT attribute this difference to rowshardkernels. A fresh matched
  cached no-row-shard long control is REQUIRED, not merely optional, before
  retention. May need postmatrixc8 trace to verify actual captured rowshape.
- c1/c8 partial summary saved in cached-attempt01/long-partial-summary.json.
  c16warmup live at04:27:30 (1running/10waiting, coldprefill/warmup only),
  validator4151603 and queue4155597 confirmedlive. No observationtimeout or
  small intervalgenerationrate is evidence this process has stopped.
  Continue same matrix48726 and queue35561; source/GPUexclusivity unchanged.

### 04:41 UTC: long c16 complete; query-tiled prefill scorer prepared

- Previous turn was progress: c1/c8 long evidence established a required
  matched-control rerun, and the guarded communicator queue was started.
  Current serving sources and queued benchmark hashes remain unchanged.
  Long c16 repeats706.07/745.64/757.77tok/s, all exact; c32warmup now live.
  Continue matrix48726/validator4151603 and queue35561/shell4155597. No
  competing GPUwork, no profiler activation. Do NOT duplicate these jobs.
- Capture audit: actual boot contains an8-row graph; indexer takes
  md.num_decode_tokens and enables rowsharding onlyR16/32. This supports
  the unchanged-c8 hypothesis, but is not a live-step trace. Fresh cached
  no-shard control remains required for attribution, including c16/c32.
- New independent prefill hypothesis from owned _cached_pool_logits and
  _tile implementations: each query CTA reloads the same pooled keys.
  Quarantined benchmarks/glm5_next_pool_query_tile_candidate.py shares keys
  across2/4adjacentqueries of one request via a wider tensor-core N tile;
  device-visible request checks fall back to independent rows at boundaries.
  Preserves explicit two16-head reduction; no serving/indexer dispatch edits.
- CPU launch/strided-page contracts4pass;10GPU cases skipped because GPU
  validation is intentionally deferred. OfflineSM80compile passesBQ2/4,
  sharedmemory20,480/36,864B. Ruff/help pass. No claimed GPU parity or speedup.
  GPUtests require unchanged1e-6 native tolerance, selected512sets, guards,
  6changed graph replays and mixed/zero/ragged lengths atR5/64/256/2048,
  page64/576 and context2052/8196/131076. Do not loosen numeric oracle.
- benchmarks/benchmark_glm5_next_pool_query_tile.py prepared for isolated
  native/query2/query4 comparison:11distinctcachelayers, grouped/interleaved
  request layouts,R128/512/2048,8192/131072context,5alternating100mstiming
  repeats, strictper-layer parity/topK/guards andposttimingparity. This only
  times score, not query projection/cacheupdate/topK or end-to-end TPS.
  Raw perf/results/2026-09-10/glm53f-pool-query-tile/{cpu.log,offline.jsonl,
  offline.log,help.log,source.sha256}. NO new GPU queue for this experiment.
- AFTER current communicator queue exits and GPUs are verified free:
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
  tests/glm5_next/test_pool_query_tile_candidate.py
  Only if strictGPUgates pass, run -m benchmarks.benchmark_glm5_next_pool_query_tile
  in a separate isolated GPUprocess with rawJSONL/log. A failed numerical
  gate needs correction or rejection, not a serving integration.

### 04:55 UTC: row-shard full matrix passed; serial GPU transition underway

- Matrix48726 completed0, all24exact checks and both sets of canaries pass;
  22serving/client and CLI hashes match. Long medians c1/c8/c16/c32:
  104.040803/522.988205/745.644132/1183.014642tok/s. c32repeats1252.028020/
  1183.014642/1100.653862. Each timed long request1computedtoken/124416
  GPU-prefix hits,0external/preemptions. c32r3 has two isolatedallGPUzero
  samples and one2.003s observed zero span; other23timingwindows have none.
  Residual intermittent admission stalls remain. Not a clean attribution
  win: matched cached no-shard control still required, especially c8.
- Raw cached-attempt01/long-summary.json, short-comparison.json and final
  telemetry-summary.json. Graphmemory0.70GiB/KV4,304,412tokens; no profiler
  activation during any timing. Response hashes vary across same-config
  repeats, so identical token counts are not identical expert-route traces.
- Queue35561/shell4155597 audited results and idlemetrics, gracefully stopped
  API4139241 after identitycheck. Server9671completed0; workers are releasing
  remaining allocations. Queue waits forall8GPUs0MiB/0% before control and
  candidate; do not launch competing work or duplicate it. CPU-prefill
  candidate is still unqueued and not GPUvalidated. Benchmark hashes frozen.

### 2026-09-10 05:12 UTC: communicator memory win integrated; prefill tile corrected

- Previous goal turn was progress (prefill candidate implementation/offline
  gates). This turn completed the full rowshard matrix, both fresh gather
  arms, integrated the measured memory fix and GPU-gated/timed prefill tiling.
- Queue35561 completed0; all16eight-rank case records pass changedgraph
  expanded-set/bounds/guards. At firstR16 gather, ProcessGroupNCCL consumes
  507,510,784 non-allocator bytes extra on EVERY rank. Existing-PyNccl avoids
  exactly that amount (484MiB/rank,3.78125GiB across8), persistent through
  subsequent cases. Same allocator bytes; firstc1/c8 cases havezero difference.
  Raw glm53f-indexer-existing-pynccl/{summary.json,memory-by-rank.json,
  process-group.jsonl,existing-pynccl.jsonl,pipeline.log}.
- Latency largelyunchanged: at131K c16PG74.348us/direct74.519us; c32
  PG81.872/direct82.015us. At1K c16PG33.585/direct31.178us, c32PG32.381/
  direct32.329us. This is a memory win, not claimed servingTPS improvement.
- Serving _pooled_select now resolves live TP device_communicator, waits
  for init, requires activePyNccl, and calls all_gather on callerstream.
  No communicator pointer serialized in AOT; c1/c8/prefill fallback unchanged.
  57CPUtests pass. Integrated actual-serving eight-rank GPU test53721
  completed0/all8cases pass. New hashes/raw in existing-pynccl/serving-integration/.
  Benchmark retains historical PG control in pooled_select_process_group;
  --gather now supports serving(default), process-group, existing-pynccl.
  Old queue scripts/source.sha256 are historical; do NOT rerun them.
- Prefill initialGPUgate87885 failed2/14cases, BQ2/4 atR2048/context131076.
  Reproducer52816 shows same first mismatch: replay5/homogeneous, row423,
  pool21866, abs1.430511e-6. Original two16-head reduction was wrong for this
  wider layout. Replaced with explicit query-separated H32 reduction.
  Same strict1e-6 tolerance now passes failingcases64014 and full14tests.
  No oracle relaxation. Originalfailed logs/hashes remain; current manifest
  is reduce32-full.sha256 and source-reduce32/ snapshot, not initialsource.sha256.
- Prefill sweep72634 completed0,12cases, all11layers strictnativeparity,
  selected512sets/guards andposttimingparity; BOTH variants bitexact in all
  cases. At131K grouped cR128 native632.059/Q4437.181us, R5122495.699/
  1517.562us, R20489908.252/5836.418us (score only). At8K/R2048 grouped
  1057.274->730.809us. InterleavedQ4 SLOWER24-35%; Q2 about3-11%slower.
  Therefore do not blanket-enable. Next investigate mixed-group fallback
  occupancy/segmentation or justified metadata eligibility; realprofile
  prefill/TPS/strictreference gates required before retention.
  Raw glm53f-pool-query-tile/{reduce32-full-gpu.log,gpu-benchmark.jsonl,
  gpu-benchmark.log,summary.json}; no prefill serving wiring.
- All priorGPUworkers gone/all8GPUs0MiB before new REAL TP8 boot:
  CURRENT server31315/API14591, shortvalidator6436/shell14668/PID14669,
  telemetry62809/shell14675. Root glm53f-tp8-row-shard-pynccl/attempt01/.
  Only communicator implementation changed versus prior rowshard arm;
  registryflags/fullcontext/BF16KV/tiers unchanged. Profiler configured but
  NEVER activated. run_short.sh health/canaries/three repeats c1/8/16/32.
  22source/client hashes+CLI hash AND actual source-snapshot copies saved.
  No queued GPUjobs; prefillkernel NOT integrated. Freeze serving sources
  during this qualification. Check cachedAOT/newgraphmemory/KVcapacity/TPS.
  Matched cached no-row-shard long control,1M/physicaltiers/TPscaling stillopen.

### 05:13 UTC: real-profile graph memory restored; AOT recompiled, not a cache hit

- CurrentAPI14591 reached health, text/reasoning/image/tool canaries pass.
  GPUKV4,304,412tokens, graphcapture0.22GiB (down fromPG0.70GiB). Real boot
  confirms the graphmemory tax is removed, NOT a larger allocatedKVpool.
- All8ranks rejected oldAOT20013379... because source changed; recompiled
  and saved under SAMEkey at05:11:30,16.20s compiletotal. Do NOT describe
  this as a direct cachedAOTload. A subsequent unchanged-source cachedboot
  is stillneeded to qualify runtime communicator lookup across restarts.
- Short validator6436/PID14669 live; c1repeats110.03/110.46/109.77tok/s.
  Engine15152/workers15389..15396, telemetry14677/session62809. No other
  GPUjobs/queues. Currentservingsource snapshot/hashes preserved; freeze
  until three-repeatshort matrix completes. Prefilltile remainsbenchmark-only.

### 05:21 UTC: PyNccl short qualification passed; unchanged cached matrix LIVE

- FirstPyNccl realboot shortvalidator6436 completed0/all12exact checks and
  canaries pass. c1/8/16/32 medians110.031163/492.227398/676.794211/
  925.940706tok/s. No allGPUidle samples in12timingwindows. Sourcehashesmatch,
  graph0.22GiB, GPUKV4,304,412tokens. This qualifies communicator replacement
  within experimentalrowsharding; does NOT retain the rowsharding feature
  versus an unsharded control. Firstboot had recompiledAOT onsourcechange.
- Afteridlemetrics/APIidentitycheck, API14591 andtelemetry14677 stopped;
  allworkersgone andall8GPUs0MiB/0% verified. CURRENT unchanged cachedboot:
  server69175/API23773, matrix37303/shell23850/shortvalidator23851,
  telemetry62982/shell23864. Root glm53f-tp8-row-shard-pynccl/cached-attempt01/.
  run_matrix.sh runs canaries+3short and3long repeats per c1/8/16/32,
  then checksall24exact/hashes. No competing GPUjobs or queuedmicrobenchmarks.
  Same22source/client+CLI hashes asattempt01/source-snapshot. Profilerinactive.
  Freeze serving source; require8directAOTloads/graphmemory/KVpool/gates.
- Additional exact-source lead, notimplemented: sampler.cu:399's
  topKPerRowJob returns ascending indices WITHOUT reading logits when
  rowLen<=topK and multipleBlocksPerRow=false. top_k_per_row_prefill uses
  that non-multiple-block specialization, including ourdecodecaller. Thus
  selection-only scoring may be skipped when completepools<=512 (visible
  <=2051), while preserving existing expansion/tail order. Generic score
  APIs/oracles must still produce logits. Need poisoned-logit/nativeindex
  checks anddynamicgraph boundaries2047..2052/mixedlongrows before anychange.
  No serving edits made for this lead; new prefillsweep remainsquarantined.

### 05:23 UTC: unchanged-source AOT cache hit verified on all8ranks

- All8ranks directly loaded20013379... at05:21:15; no source-invalidated
  recompile thisboot. Engine24342; workers24605/24606/24607/24608/24613/
  24623/24624/24625. Graphcapture0.22GiB, KV4,304,412tokens. Cachedboot
  canaries pass andc1three repeats109.61/109.40/110.36tok/s completed.
- Keep69175server/API23773,37303matrix/shell23850 and62982telemetry live.
  Fullshort+long matrix is inprogress; no queue orcompetingGPUjob. Same
  frozen sources, no profiler activation. The prior qualification's short
  medians remain110.031163/492.227398/676.794211/925.940706tok/s.

### 2026-09-10 05:35 UTC: split-prefill CPU/offline gates ready; matrix still live

- Serving sources unchanged, all22 frozen hashes match. API23773/server69175,
  matrix37303/shell23850, long validator31691, telemetry62982 remain active.
  Short complete: 109.612908/492.535253/676.543965/926.373823tok/s, all12exact
  plus canaries. Long c1 median104.009008; c8 median474.709732, repeats
  443.433952/531.377361/474.709732. Both exact/cached as expected; no sampled
  all-GPU idle in those six timed windows. c16/c32 pending. Do not stop the
  serving job or launch GPU microbenchmarks during its matrix.
- Split prefill candidate separates wide homogeneous-query CTAs from one-row
  mixed-request CTAs; device predicates complementary, no host readback.
  SM80 offline four variants pass: mixed12KiB vs Q4wide36KiB shared/CTA.
  24CPU tests pass,20GPU tests skipped,lint pass. GPU performance/parity remains
  UNVALIDATED. Initial merged offline ASTSource constexpr failure fixed;
  failure log preserved. Raw glm53f-pool-query-split/. Existing measured merged
  snapshot remains glm53f-pool-query-tile/source-reduce32/.
- Next after matrix completion and deliberate GPU release: run
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q
  tests/glm5_next/test_pool_query_tile_candidate.py
  then ONLY if all strict gates pass, benchmark -m
  benchmarks.benchmark_glm5_next_pool_query_tile --variants query2 query4
  split2 split4 --layouts grouped interleaved mixed, saving raw artifacts.
  No GPU jobs queued. Matched cached no-row-shard long control remains the
  serving attribution priority; neither prefill variant is integrated.

### 2026-09-10 06:00 UTC: CRITICAL prefill mapping fix; old matrix interrupted

- The apparent query grouping investigation found a correctness bug:
  chunk.token_to_seq maps KV rows, but GLM used token_to_seq[:R] as query IDs.
  Cached multi-request or sliced prefill can read the wrong request's cache.
  Actual Triton CPU interpreter witness and4A100 failures establish it with
  both native/Triton metadata. Existing token-count/canary passes did not
  cover this case. Historical TPS is not a quality-qualified cached-prefill
  baseline. See new leading baseline_status warning and optimization entry.
- Fixed serving _prefill_query_requests: int32 index_select of token_to_seq
  at per-query cu_seqlen_ks; no host sync. Guard against unqualified localized
  DCP row bounds. No generic metadata/kernel change; no CMake rebuild.
  13targeted tests/161broaderGPUtests pass, raw and compact caches included.
  Raw glm53f-prefill-row-map/. 6CPU recall-oracle tests pass. Real profile
  quality and TPS still pending. Do not say fully qualified or optimized.
- Old server69175/API23773, validator31691/matrix37303, telemetry62982 and
  orphan c32 warmup42336 are STOPPED. API needed second interrupt plus orphan
  client termination to close outstanding HTTP requests. All8GPUs0MiB/0%
  verified before new boot. Old cached matrix has12short+9long results only;
  interrupted during c32 warmup. c1/8/16 long medians104.009008/474.709732/
  719.383057, no sampledallGPUidle. Preserve telemetry-interrupted.json.
- Cancelled serial candidate queue43251/shell42055 (exit143, no GPU stages
  ran). DO NOT rerun glm53f-prefill-serial/run_after_matrix.sh; staleIDs and
  interrupted matrix. Split and score-skip candidates remain parked/not
  GPUvalidated. Skip candidate has5CPUtests,11GPUskip,2offlineBScompiles;
  benchmark19cases preserves full1M logits stride. No production integration.
- CURRENT: server34015/API51217; matrix39384/shell52064; telemetry95112.
  Root glm53f-tp8-prefill-row-map/attempt01/. Booting real glm53f-nvfp4-8,
  unchanged row-shard experimental flag and allotherprofile settings;
  BF16mainKV/full1M/72GiBhost+256GiBdisk perrank. Profiler configured/inactive.
 26source/client hashes+CLI and actual snapshots frozen. Matrix waitshealth,
  runs new cached distinct-record recall at c1/8/16/32, then short/long exact
  matrices. No queued/competing GPU work. Keep sources frozen while live.
  First finish correctness requalification; then matched cached no-row-shard
  control, full1M, physicaltiers and TPscaling. Goal remains active.

### 06:10 UTC: SECOND correctness failure isolated; NEW matrix is stopped, API idle

- Update all prior CURRENT entries: server34015/API51217 is healthy/IDLE,
  engine51751/workers51990..51997, telemetry95112/PID52075 live. No benchmark,
  microbenchmark or queue is running. No profiler activation. Fixed-map boot
  has4,304,412GPUKVtokens/graph0.22GiB; all8AOTloads source-invalidated and
  recompiled20013379..., NOT directcachehit. Serving sources still frozen.
- New matrix39384 failed initial6000-token unaligned fixture: correct marker,
  but0cachehits (not cache evidence).44381 failed before HTTP because installed
  tokenizer tokenize=True returnsBatchEncoding.88707 failed the corrected
  aligned fixture's ACTUALQUALITY gate. No short/long matrix on this boot.
  All previous client processes terminal. Source snapshots/hashes updated for
  client-only fixes; original and intermediate copies/logs retained.
- Aligned canary: prime raw rendered-chat prefix9217tokens/1generated, then
  chat sameprefix+question12069tokens. Hits9216GPUcachetokens, computes2853,
  but generates400 `!` tokens in reasoning and null final content. This is
  not a token-budget issue to paper over. c8/16/32canaries not reached.
- Independent salted cold-v-cached reproducer30579 completed0 and proves:
  cold identical12069-token prompt→correct marker,65completiontokens;
  separately primed cached prompt→400exclamation corruption.0externalhits/
  preemptions throughout. Root cause unisolated; not evidence of diskrestore.
  Raw glm53f-prefill-row-map/cold-v-cached.jsonl/.log. Reproduce with
  CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 .venv/bin/python -m
  benchmarks.reproduce_glm5_next_cached_extend --salt <fresh-unique-salt>
  against currenthealthyAPI; eachrun MUST usefreshsalt for cold validity.
- Currentcorrectedfixtureclient copy: attempt01/isolation-aligned-client-v3.py;
  realfailure: isolation-aligned-v2/c1-recall.json.9CPUfixture/oracletests pass.
  Native/Triton query-map bug IS fixed (13targeted/161broaderGPUtests); this
  second resident-prefix-extend failure is NOT. Baseline warning updated.
- Next isolate first failing operation with actualstate/finite diagnostics:
  KDA/convolution initialstates, GPUcacheownership/copy, sparse attention.
  Localanchors kimi_gdn_linear_attn.py:643-725, gather_initial_states.py,
  gdn_attn.py:283, MambaManager, scheduler._mamba_block_aligned_split.
  Do not assume a cause, disable prefixcache, shrinkcontext, or optimize an
  unqualified path. Rowshard attribution, prefillcandidates,1M/tiers/scaling
  remain open. Goal active; previous turn made concrete correctness progress.

### 06:38 UTC: first bad boundary is layer3 sparse attention; attempt03 booting

- Supersedes earlier live-process entries. Attempt02 finite diagnostic API
  68619/workers69445..69452 terminated after the deliberate first-bad error.
  All eight GPU allocations verified absent before the next boot.
- Raw attempt02 proves cold control passes, then cached extend first becomes
  non-finite at layer3.attention on all ranks: 575/576 whole rows bad.
  layer3.pre_attention and KDA layers0..2 remain finite. Root cause inside
  sparse attention not yet isolated; do not assume KDA or tier restore.
- New opt-in diagnostics in glm5_next_debug.py, mla_attention.py and
  quixicore_mla_sparse.py compare all KV writes and dump bounded referenced
  BF16 pages/Q/indices/BT/output upon sparse-kernel failure. Eight diagnostic
  tests pass. Temporary diagnostics, not production optimizations.
- CURRENT attempt03 server session71999/API79193; client health wait62040.
  Raw perf/results/2026-09-10/glm53f-cached-extend-finite/attempt03/ with
  run_server.sh, run_reproducer.sh, source.sha256/source-snapshot. Fresh salt
  finite-attempt03-20260910-0637. Sources frozen; no GPU competitors/queues.
  Revalidate handles before actions. Need saved sparse-first-bad.pt evidence,
  then fix actual operation and run real cached quality gates before TPS.

### 06:54 UTC: graph-dispatch fix implemented; non-diagnostic TP8 qualification

- Attempts03/04 are TERMINAL, allGPUallocations gone before new boot. Raw
  cold controls pass; current KV writes and Q finite/exact all8ranks, but old
  prefix pages46/49 corrupted. Attempt04 write history proves they were
  correctly written inprime, then rows0..529 (542720bytes) overwritten while
  trailing46rows unchanged: exactly a TP8 KDA conv+SSM payload.
- Root mismatch: runner counts uniform1token asdecode even when GDN/KDA
  metadata treats it as a prompt tail/prefill and does NOT refresh captured
  decode state-index buffers. Graph replay can write stale slots now owned
  by MLA. GPUModelRunner._determine_batch_execution_and_padding now excludes
  FULL for hybrid prefills using the same computed<prompt CPU phase test.
  Genuine c1/8/16/32 decode and dummycapture overrides stillFULL. Before:
  9regressionsfail/6pass; after22dispatchtestspass. Live fix not yet qualified.
- Diagnostic files named sparse-first-bad.pt were overwritten by an already
  queued576-token batch after first1728-token cached batch failed; do not
  treat filename as first chronological snapshot. Writehistory/logs retained.
- CURRENT server26209/API95143, sourcefrozen, diagnosticsOFF. Root
  perf/results/2026-09-10/glm53f-hybrid-prefill-graph/attempt01/. run_quality.sh
  waitshealth, then freshsalt cold/cachedquality gate, then distinct-record
  cachedrecall c1/8/16/32 andsourcehashaudit. No competingGPUjobs. Preserve
  full1M/BF16/tiers/graphs/profile flags. Afterquality: exactmatrices, restart,
  physicaltiers/full1M/scaling and matchedrowshard control remain required.

### 07:01 UTC: cached recall PASSES57/57; clean probe-free restart next

- Server26209/API95143 STOPPED afterqualityclient82421 completed0. AllGPU
  memory verified0MiB. Original cold/cached repro bothcorrect (41/42tokens),
  cached9216hits+2853computed. Distinctrecords1/1,8/8,16/16,32/32 pass with
  expectedGPUhits,0external/preemptions. Sourcehashesmatch; noTPSrun yet.
- That boot reused diagnosticAOT45c089ab... withprobesruntimeoff. To avoid
  any residual probe nodes in performance results, removedalltemporary
  finitehooks andcompilerbranch plusdebugmodule/test. Savedsource snapshots
  retain them. GPUrunnerhybridprefilldispatchfix andregressiontests RETAINED.
- CURRENT regressiontest session37342 (GPU0 plusCPU); no server/client/GPU
  benchmark running. Newcleanboot script prepared at
  perf/results/2026-09-10/glm53f-hybrid-prefill-graph/attempt02/run_server.sh.
  Aftertests: bootit, freeze source snapshots/hashes, repeatcoldcached+57recall,
  then exactshort/124Kmatrices withpassivetelemetry. Do not runattempt01
  run_perf.sh (oldboot/artifact); no performance job has been queued.

### 07:04 UTC: clean regression suite35pass; new serial qualification running

- Regression37342 finished0:35tests pass (V1 hybrid dispatch, existingV2
  dispatch,13prefillquerymap tests including actualGPUmetadata). No native
  code changed. Temp probe hooks/module/tests fully removed; compilerpass
  andmla_attention.py have no remainingdiff fromtheir prediagnosticstate.
- CURRENT cleanserver51508/API102761 booting; worker0 PID103867, matrix9464/
  shell104454 waitshealth. Root glm53f-hybrid-prefill-graph/attempt02/.
  Source snapshots/hashes frozen. run_matrix.sh serially runs freshsalt
  cold/prime/cached gate,57distinctcachedrecalls, short1000/300 andlong
  124417/2000 three-repeat c1/8/16/32, thenhash/exact/telemetrysummaries.
  Passive nvidia-smi starts only afterquality andselfterminates withmatrix.
  No competingGPUwork/profiler. Previousattempt01quality57/57remainsvalid,
  but newcleanboothasnotreachedhealthorproducedTPS yet. Do not mixthose runs.
- Goalactive; thisturn fixed a demonstrated cached-prefix corruption bug,
  qualified it throughregisteredTP8atalltargetconcurrencies, cleaneddiagnostics,
  andstartedcleanrequalification. Nextinspectsame livehandles; do not restart
  on an observationtimeout. Full1M/physicaltiers/scaling/matchedrowshard and
  parkedprefillkernels remainopen aftercorrectness/performancequalification.

### 07:16 UTC: clean57/57recall+12shortexact PASS; longmatrix live

- Server51508/API102761/engine103620 live; workers103867,103868,103869,
  103870,103871,103875,103884,103888. Matrix9464/shell104454, longvalidator
  115285; passive nvidia-smi110566 managedby matrixEXITtrap. No competitor,
  no profiler. Sources frozen+hashesmatch. Raw hybrid-prefill-graph/attempt02.
- Probe-free AOTde6565114... freshcompiled. KV4,280,453tokens/graph0.22GiB,
  versuspriorcachedboot4,304,412. Do notattribute0.56% tofixyet; unchanged
  cachedrestart stillneeded. Originalcachedextend passes; all57distinctrecalls
  andtext/reasoning/image/toolcanariespass. Shortmedians108.756832/489.245287/
  675.183879/924.461716 atc1/8/16/32, all12exact. Everyshortrequest computes
  full1000tokens with0cachehits/preemptions. short-completed-summary.json.
- Long124417/2000matrix running: canariespass,c1r1=102.39tok/s.13window
  telemetry snapshot(12short+firstlong)allpass,no sampledallGPUidle. Do not
  regardlongdone. Wait samehandles, preservefrozen serving sources.
- CPU-only benchmarkgeometry preparationdone: poolskip/querytile/split
  benchmarkcache now actual4608tokens/eleven-column6,488,064bytestride,
  helper benchmarks/glm5_next_pool_layout.py. CPU59pass/47GPUskipped,lintpass.
  Score-elisionoffline64/576/4608pass12KiBshared. Query/split compile_only
  defaults4608 butnewgeometryoffline/GPU notrun. Raw pool-skip-geometry/.
  No servingwiring/GPUjobs. Earlier576microbench1.7xnotactualgeometryproof.
- Goalactive. Finishlongmatrix, then planunchangedcachedrestart+matched
  rowshardcontrol andGPUcandidategates duringa safe GPU-releaseinterval;
  strict1M/physicaltiers/scaling stillrequired. No queuedGPUcandidateprocess.

## 2026-09-10 (late): TP4 x DP2 decision, DP root cause, block-pool crash chase

- Operator decision: glm53f-nvfp4-8 serves TP4 x DP2 (prod is always c8+).
  Root cause of the old "DP2" c1 loss: with EP off vLLM flattens the MoE TP
  group across DP (1/8 of each expert per rank, all-gather across replicas
  every MoE layer, lockstep dummy steps, per-step DP token sync). Fixed by
  `--data-parallel-replicate-moe` (6629cd0d5). Prefix-affinity DP routing
  landed in the load balancer (VLLM_DP_PREFIX_AFFINITY). Record NOT flipped
  yet: needs the mx-tp4dp2 arm (replicated) to pass canaries + exact, then
  the full gate set through `slimserve --serve`.
- Block-pool crash on the speculative record under tier restores: first
  `get_new_blocks` ref_cnt assert (leg 1), then with fail-fast invariants
  `popleft_n` walked off the free list with num_free_blocks still positive
  (leg 2, no double free reported). Cross-pool touch/free checks added
  (7a453f9b3); leg 3 (`glm53f-leg-spec-instr2`) reproduces on them.
- GPU queue (scripts in ~/.local/scratch/glm53/): queue2.sh = instrumented
  leg -> mx-tp4dp2 (replicated) -> mx-tp4dp2-flat -> mx-tp4dp2-ep; then
  tp4_arm_chain.sh (mx-tp4 standalone) and k4_chain.sh (DFlash2 k=4 arms
  + paired k=3). Logs: queue2.log, matrix-chain.log, k4-chain.log. The 1M
  leg is parked (leg_1m.sh stub; real script leg_1m.real.sh) until the
  block-pool fault is fixed.

## 2026-09-11 12:00: record flipped to TP4 x DP2; block-pool fault fixed

- glm53f-nvfp4-8 = TP4 x DP2 with replicated MoE (9cb3c3c30). Serve-path
  c1 117.8 / c8 506.8 / c16 786.5 / c32 985.8 / c64 1158.0; canaries and
  tier acceptance pass (perf/results/2026-09-11/glm53f-final-dp2/).
- Block-pool leg deaths: negative count into get_new_blocks from
  allocate_external_computed_blocks (fixed 9c24bc5ad; see notebook).
- Running (queue10.sh -> queue10.log): DP2 WildChat leg
  (perf/results/2026-09-11/glm53f-leg-dp2/), then the 1M leg
  (leg_1m.real.sh dp2-1m). Then queue11: k=3/k=4 arms on the record with
  the compile-hash fix. Old queue scripts are dead; only queue10/11 run.

## 2026-09-11 20:00: TP4 x DP2 throughput program in flight

- Landed (all on main): multi-API-server entry (ad61a11a8), grouped
  indexer scoring for speculative rows (043effea6, bit-exact), fp8 main KV
  for the 512-wide NoPE sparse MLA path incl. the tensor-core Triton
  decode (7b0323050, parity passed), compile cache keyed by draft block
  size, harness credits recall in reasoning + flags budget exhaustion.
- 1M-context legs: both layouts reach 1.04M; every probe miss above 900K
  was the model exhausting its 1,024-token thinking budget with an empty
  answer, not lost recall (notebook 2026-09-11).
- GPU queue (scripts + logs in ~/.local/scratch/glm53/): queue14 = A/B
  arms dp2-base / dp2-api2 / dp2-numa / dp2-k3-all; queue15 = closing DP2
  1M leg (dp2-1m-c) with the reasoning-aware harness; queue16 = c32-per-
  replica torch profile (profile_dp2.sh; its first attempt hung on a bare
  `wait`, fixed); queue17 = fp8 main-KV arm dp2-fp8kv. Results land in
  perf/results/2026-09-10/glm53f-dflash2/<arm>/ and
  perf/results/2026-09-10/glm53f-1m-leg/dp2-1m-c/.
- Next after the profile: fused marlin MoE (launch list in
  fused_moe/experts/marlin_moe.py: gemm1, silu_and_mul, gemm2, moe_sum per
  layer), batched variant of the DSV4 fused mHC allreduce
  (`should_fuse_dsv4_mhc` is batch-1 only), sampler/launch fusions,
  host-resident main KV for GLM. Record flips (schedule, api2, numa, fp8
  KV) only on measured wins plus the gate set.

## 2026-09-11 23:40: program status

- Profile of the DP2 record at 32 req/replica (notebook): MoE GEMM 39%
  (at bandwidth), mHC partials 14%, dense GEMMs 12%, KDA 8%, allreduce 6%.
  Landed: token-batched mHC partials (97f13c4ee, 78 -> 31 us at T=32).
  Next kernels: KDA recurrent (0.56 TB/s effective, ~2x headroom;
  vllm/models/kimi_k3/amd/ops/third_party/kda/fused_recurrent.py), then
  spec at c32 once per-row costs are down.
- Thinking budget 2000 now on both glm53f records; harness credits recall
  in reasoning and sends probes a per-request budget.
- OPEN: cold prefill >= ~900K produced '!' garbage in the 1M leg (both
  layouts); a cold-prefill probe worker died silently at 200K on a
  torch.compile cache two concurrent boots had corrupted (cache set
  aside at ~/.cache/vllm/torch_compile_cache.corrupt-*). Rerun queued
  (queue22, results in perf/results/2026-09-11/glm53f-coldprefill2/).
- GPU queue: q19 NUMA arm -> q20 fp8-KV arm + FULL_AND_PIECEWISE arm ->
  q21 same-day base on the batched-partials tree -> q22 parity test +
  cold-prefill bisection. Logs ~/.local/scratch/glm53/queue*.log.

## 2026-09-12 12:10: fp8 main KV on the record; mHC warp-split kernel; roofline audit

- RECORD (glm53f-nvfp4-8/a100): TP4 x DP2 replicated MoE + fp8 (e4m3) main
  NoPE-MLA KV (glm5_next_main_kv_fp8, pool 2.97M tokens) + mHC warp-split
  partials. Exact medians c1 127.1 / c8 497.0 / c16 843.9 / c32 1023.3 /
  c64 1320.8 tok/s (perf/baseline_status.md, raw perf/results/2026-09-12/
  glm53f-mhcws-record/). Gates for the fp8 flip: canaries, eviction-restore
  6/6 with verify (0/88 mismatched), WildChat leg 737 turns / 0 errors /
  119/119 recall / 549 restores / 93.4% prefix hits.
- Fixed on the way (29e6fdf98): the host tier hashed at the smallest
  attention block while the scheduler hashes at the gcd of every group
  (fp8 widens MLA blocks to 2304 beside the 1152 KDA block), and the tier
  index never bound hashes at positions where no attention block completes;
  the scheduler's invalid-block handler assumed one KV group and killed the
  engine on the first fail-closed restore. tests/v1/core/
  test_scheduler_invalid_blocks_hybrid.py, test_host_tier_connector.py.
- Kernel pass (notebook 2026-09-12): mHC partials warp-split
  (partials_batched_ws, c8bf2d250): 31 -> 13.6 us at T=32, 98 -> 33 at
  T=128, serving +12.8% at c64. KDA CUDA decode kernel: env-gated, no lever
  (Triton at bandwidth from N=32). Roofline audit of the c64 128-token
  step: marlin NVFP4 MoE at the HBM roofline (167 distinct experts/layer,
  VLLM_MOE_EXPERT_STATS diagnostic), all-reduce 1stage +2.2% (rejected),
  dense bf16 GEMMs at 0.3-0.9 TB/s under cuBLAS - skinny split-K GEMM
  written (skinny_gemm_ampere.cuh, op skinny_gemm, tests 54/54) but load-
  bound at 0.65 TB/s; parked with the v2 design in the notebook.
- OPEN: KDA speculative path (1 read + 4 per-row state stores per layer,
  7% of the step) needs a deferred-commit runner design; the 1M-leg '!'
  garbage after >= 900K restores in two-session runs is unreproduced;
  registry tests fail on another session's glm53f-q2-1/metal and
  glm53f-gguf records (not this record); Nsight Compute is blocked
  (ERR_NVGPUCTRPERM). QuixiCore-CUDA port of the mHC family and the KDA
  kernel is grafted (kernels/serving/, tm_cuda_serving.cu) pending its
  compile check and commit.

## 2026-09-12 21:40: throughput program - record at fixed k=2, sustained metric, closed levers

- RECORD glm53f-nvfp4-8: DFlash2 fixed k=2 (b2cc96fea). Exact c1 128.0 /
  c8 567.1 / c16 875.6 / c32 1110.0 / c64 1413.8; sustained (vllm bench
  serve random 1000/300) c64 1227-1411, c128 1672-1674 output tok/s; leg
  785 turns / 0 errors / 126/126 recall. The per-batch draft schedule was
  inert under DP (engine falls back to fixed k); k=3 had been running
  everywhere. Dynamic schedules are now allowed under replicated-MoE DP,
  zero-draft ranges rejected at config time (they crash KV init).
- Primary metric is now sustained load: ~/.local/scratch/glm53/sustained.sh
  <outdir> <conc...> against a running :8400 server (reproducible within
  2%; the exact harness spreads 5-10% at c96+).
- Closed by measurement (notebook 2026-09-12): stream overlap of MoE
  weight streaming with compute (A100 time-slices grid-filling kernels,
  -2%), fp8 marlin dense weights (slower than cuBLAS), custom W8A16
  skinny GEMM (cuBLAS parity at best after 4 restructurings; committed,
  gated off, tests 113/113, kernel csrc/quixicore/serving/
  w8a16_gemm_ampere.cuh, method vllm/model_executor/layers/quantization/
  qc_w8a16.py, env VLLM_QC_DENSE_W8A16), k=4 drafts (-13%).
- NEXT: prefill-step profile (queue47) - continuous prefill is first-order
  in the sustained metric; then the DP replicas' per-replica dynamic k if
  the c8 point (4 per replica) prefers a longer draft.
- Rule re-learned the hard way: never cp the extension .so into vllm/ while
  any server is booting or running (killed the k=4 arm's boot).

## 2026-09-13 00:10: sparse prefill kernel on the record; program state

- RECORD glm53f-nvfp4-8 = fp8 main KV + mHC warp-split + fixed k=2 +
  sparse MLA prefill kernel (396300c18). Sustained c64 1508 / c128 1864
  output tok/s (program start: 1140-1157 / 1425); exact c8 567 / c16 876 /
  c32 1110 / c64 1414; leg 845 turns / 0 errors / 135/135 recall / 704K.
- The prefill kernel (csrc/quixicore/serving/mla_sparse_prefill_kernels.cuh,
  op mla_sparse_prefill_fp8, dispatch in the sparse backend's forward_mqa,
  flag glm5_next_sparse_prefill / env VLLM_QC_SPARSE_PREFILL) replaced the
  per-token decode kernel that was 39% of prefill time; groups of 4 queries
  over the union of their pools; duplicates in the index list count once.
- Remaining levers after this program: prefill MoE/mHC/NCCL shares (see the
  with-kernel profile in the notebook), the KDA deferred commit (~2%), the
  W8A16 dense path (parity, gated off), per-replica dynamic draft length.

## 2026-09-13 11:40: batch cap 128, balanced DP routing, chunk-size verdict, prefill GEMM parked

- RECORD glm53f-nvfp4-8: max_num_seqs 128 / capture 384, prefix-affinity
  router balanced by live-context footprint (single-home credit),
  max_num_batched_tokens 2048 explicit. Sustained c64 1538 / c128 1921 /
  c256 2414 (program start 1150 / 1425 / -); exact c1 136 / c8 572 / c16
  898 / c32 1199 / c64 1531; leg 811 turns / 0 errors / 128/128 recall.
  Baseline section "2026-09-13: ... record" in perf/baseline_status.md.
- ROUTER (vllm/v1/engine/dp_prefix_affinity.py, d6af7171b): sticky
  affinity had frozen a chance 5/3 session split for a whole leg (restores
  2x on one replica, 845 -> 671 turns, one recall miss). New terms:
  prompt tokens routed per replica over the last 64 requests x 0.1
  (VLLM_DP_PREFIX_AFFINITY_RECENT_WINDOW / _RECENT_PERMILLE) and only the
  longest recorded match is credited. Check on any leg: "Engine 000/001
  ... Running:" averages in the server log should be within ~0.2.
- CHUNK SIZE: 4096/8192-token chunks win the random benches (+5%
  sustained, +11% exact c64) but lose the deep-context leg by 13% of
  turns with the SAME pool and router; the host tier restores 4x fewer
  blocks and offloads 2x more above 2048 (misses are not the cause: 30 vs
  207). TOP OPEN ITEM: find why restores drop with larger chunks (suspect
  the aligned-boundary tail-state save when a prefill step crosses several
  block boundaries), then re-run the 4096 leg; if it matches 2048, take
  4096 (all numbers in the profile note). 4096 is the validated override
  for short-context high-throughput serving today.
- PREFILL MoE GEMM (csrc/quixicore/serving/nvfp4_moe_prefill_ampere.cuh,
  op nvfp4_moe_gemm, hook in fused_marlin_moe behind
  VLLM_QC_NVFP4_PREFILL_MOE_MIN_ROWS, default 0 = off): reads the Marlin
  layout in place, shared dequant per tile, four warp/stage configs,
  parity 14/14. Isolated: w13 -10% vs Marlin, w2 +13..25%; serving A/B at
  4096 chunks: no gain. PARKED. The notebook has the iteration table and
  the smem-traffic analysis; Marlin at large M is 8-warp latency-bound
  with 0.038 B/MAC of smem traffic - beating it needs a multistage
  CUTLASS-grade mainloop, not a shared-dequant tile.
- DRAFT LENGTH: per-replica schedules work under DP (guard relaxed, k=0
  ranges rejected) but every k>=3 arm lost to fixed k=2 at c64-c256;
  KDA deferred commit stays parked (a loss at k=2, kernels tested).
- INCIDENTS to not repeat: (1) a new code path defaulting ON before its
  .so was installed killed an arm's boot; (2) a unit test on GPU 7 while a
  server booted (memory profiling) killed that boot with a cuBLAS internal
  error - nothing touches the GPUs while a server boots; (3) `ps | grep |
  kill` and `pkill -f` on a script name killed my own shell twice - use
  `pgrep -f "^bash script.sh"` or safekill.
- Installed vllm/_quixicore_C*.so = build with the parked prefill GEMM
  (all other ops unchanged).
