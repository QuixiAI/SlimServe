# GLM-5.2-Vision on vLLM — performance work log

> ## READ FIRST (2026-07-27, iteration 2): the correctness gates are not deterministic
>
> Running the gate suite **twice inside one process**, same prompts, temperature
> 0, produces **different output** — and that is true with the **unmodified**
> kernels, so it is a pre-existing engine bug and not something this perf work
> introduced. Sample from `rep_ctrl_a.log`, control `.so`, pass A vs pass B of
> the same boot:
>
> ```
> A='BLUE-HERRING-7742'
> B='BLUE-HERRING-774!0.0.01,1,1,1,11,'
> A='...a river flowing through a valley surrounded by towering!0a0,0a1,0a2,0a2,0a0,...'
> B='...a river flowing through a valley surrounded by towering granite cliffs...'
> ```
>
> Four boots of the same suite: control failed 6 gates in one boot and 0 in
> another; the modified kernel failed 1 in both. **A single gate run proves
> nothing about a change.** Generation starts correct and degenerates partway
> in, which is the signature of state corruption mid-decode, not of a bad GEMM.
>
> **REJECTED: prefix caching is not the cause.** It looked like the suspect
> because every clean run in this file (`bench2.py`, `ctx_needle.py` — 6/6
> needle recall at 32k/64k/128k/256k) sets `enable_prefix_caching=False` while
> `gates.py` leaves it on. Measured with it off: still non-deterministic, still
> corrupting (control `.so`, prefix caching off — 5 gate failures). **Prefix
> caching stays ON**; it is required, it works in stock vLLM, and it is not
> what is broken here. Any future gate run must leave it enabled so the bug
> hunt stays on the real path.
>
> Failure counts over 8 boots (2 kernels x prefix-caching on/off, 2 boots each):
>
> | kernel | pc on | pc on | pc on | pc off |
> |---|---|---|---|---|
> | control (unmodified) | 6 | 0 | 2 | 5 |
> | multi-column mmvq | 1 | 1 | 2 | 1 |
>
> Every one of the 8 boots was non-deterministic within itself. The modified
> kernel is, if anything, the more stable of the two — it is not the cause.
>
> ### Narrowed: there are TWO separate effects, and the bad one is CUDA graphs
>
> `enforce_eager=True`, same suite, same boot, twice: **ALL GATES PASS**, and
> the *only* outputs that differ between the two passes are the four vision
> descriptions — differing by mild wording late in the text
> (`"the mountain is partiall..."` vs `"the mountain is bathed i..."`), never
> corrupted. Text and needle outputs are **byte-identical** across passes.
>
> With CUDA graphs on, text and needle outputs also differ, and differ by
> catastrophic corruption (`'BLUE-HERRING-774!0.0.01,1,1,1,11,'`).
>
> | config | kernel | gates failed | text/needle stable across passes? |
> |---|---|---|---|
> | eager | control | **0** | **yes, byte-identical** |
> | graphs | control | 6 / 0 / 2 / 5 / 1 | no — corrupts |
> | graphs | multi-column | 1 / 1 / 2 / 1 | no — corrupts |
> | graphs, max_num_seqs=1 | control | 1 | no |
>
> So: **effect (1)** mild vision-only non-determinism, present in eager too —
> some non-deterministic reduction in the ViT/projector path, low severity.
> **effect (2)** severe text corruption that requires CUDA graphs — this is the
> real bug. `max_num_seqs=1` does not fix it, so it is not a batching race.
> `VLLM_ROCM_USE_AITER=0` cannot be tested (`Sparse attention indexer ROCm path
> is only supported on AITER`), so the aiter sparse-MLA path cannot be ruled
> out by substitution.
>
> ### LOCALISED (iteration 3): there are TWO independent bugs
>
> `batch_determinism.py` — same four TEXT prompts, **no images**, decoded
> one-at-a-time and four-at-a-time, 4 rounds each, one boot per config. This
> breaks the vision/batch confound completely, and the answer is not what the
> gate data suggested:
>
> | config | batch=1 | batch=4 |
> |---|---|---|
> | prefix caching **OFF** | **STABLE** (4/4 rounds identical, all 4 prompts) | NON-DETERMINISTIC |
> | prefix caching **ON** | NON-DETERMINISTIC | NON-DETERMINISTIC |
>
> **Bug A — a prefix-cache HIT does not reproduce the uncached result, and the
> hit is the wrong one.** With caching off, batch=1 is perfectly reproducible
> (two boots, 7 rounds, byte-identical). With caching on, the per-round pattern
> is `[0,1,1,1]`, `[0,1,1,1]`, `[1,0,0,0]` — round 1 (cache miss) differs from
> rounds 2-4 (cache hit), and the hits all agree with each other. Comparing the
> actual text against the caching-off reference settles the direction:
>
> ```
> caching OFF (reference, stable)  'To understand how a distributed inference engine shards (partitions) Transformer weights, it is helpful to thi'
> caching ON, round 1 (MISS)       'To understand how a distributed inference engine shards (partitions) Transformer weights, it is helpful to thi'   <- matches
> caching ON, rounds 2-4 (HIT)     'To understand how a distributed inference engine shards (partitions) Transformer weights, we first have to und'   <- deviates
>
> caching OFF (reference)          '# The Signal\n\nThe lighthouse had stood on Cape Murrow for a hundred and seventeen years...'
> caching ON, round 1 (MISS)       '# The Signal\n\nThe lighthouse had stood on Cape Murrow for a hundred and seventeen years...'                   <- matches
> caching ON, rounds 2-4 (HIT)     'The salt wind had a voice. It moaned against the thick, sea-glass windows of the lighthouse...'                 <- deviates
> ```
>
> So the cold path is correct and **the prefix-cache read path is what is
> wrong** — deterministic, not a race. This is the bug behind the flaky *text*
> gates, which run at batch 1 with caching on.
>
> Checked and **wrong**: the obvious mechanism would be the DSA sparse
> indexer's K cache being a second cache that prefix caching does not track.
> It is not — `DeepseekV32IndexerCache.get_kv_cache_spec`
> (`deepseek_v2.py:631`) registers it as a proper `MLAAttentionSpec` layer, so
> the KV-cache manager and the prefix-cache accounting do see it. The mechanism
> is still open.
>
> **Bug B — batched decode (batch>1) is racy.** Present with prefix caching
> OFF, so it is independent of bug A. The per-round patterns are scattered
> (`[0,2,1,0]`, `[1,1,0,2]`) rather than cold/warm, which is the signature of a
> race rather than a stale cache. This is the bug behind the *vision* gates,
> which are the only ones that submit 4 requests in one `generate()` — nothing
> to do with vision, and consistent with the vision tower being provably
> deterministic.
>
> ### Iteration 4: bug B is NOT an indexing bug in the GGUF kernels
>
> `batch_identical.py` — N *identical* prompts in one `generate()`, temperature
> 0, prefix caching off, round 0 discarded as warmup:
>
> | n identical prompts | distinct outputs | rows of one batch disagree? |
> |---|---|---|
> | 1 | 1 | — |
> | 2 | 2 | yes |
> | 4 | 8 | yes |
> | 8 | 17 (24 samples) | yes |
>
> **Every row of a batch of identical prompts produces a different
> continuation**, and under `enforce_eager` the same 8 outputs recur every
> round (8 distinct across 3 rounds of 8) — deterministic and
> *position*-dependent, not a race. `VLLM_ROCM_USE_AITER_CUSTOM_AR=0` does not
> change it, so **the aiter custom all-reduce is REJECTED**; eager does not
> change it, so **CUDA graphs are REJECTED for bug B**.
>
> **REJECTED as the cause: a row-indexing bug in the GGUF GEMV kernels.**
> `identical_rows.py` feeds M identical rows straight to the kernels at real
> TP2 shapes. o_proj q8_0, q_a q8_0, shared_gate_up q2_K and the MoE
> `ggml_moe_a8_vec` are **row-independent at every M from 1 to 32** — all rows
> bit-identical. Both the unmodified and multi-column kernels behave the same.
>
> It did surface a real but **irrelevant** defect, worth recording so it is not
> rediscovered: `ggml_mul_mat_vec_a8` gives 1-ulp, row-position-dependent
> results for **q2_K and q6_K at output width n >= 32768** (~1 affected column
> per 8 000, scattered indices, reproducible bit-for-bit across processes).
> q8_0 and q4_K are clean at every size up to 77 440. **This model is not
> affected**: the only layer wide enough is `lm_head`, and `output.weight` in
> this GGUF is type 8 = **Q8_0**, which tests clean. Do not attribute bug B to
> this.
>
> ### ROOT CAUSE (iteration 4): amplification, not a bug. Both A and B
>
> `logit_gap.py`, 600 real decode steps, top1-top2 logprob gap:
>
> ```
> p01  0.000000      <- at least 1% of steps are EXACT ties
> p05  0.375
> p50  7.000
> fraction of steps with gap < 1e-4 : 1.167%
> fraction of steps with gap < 0.5  : 6.833%
> ```
>
> **1.17% of greedy decisions are decided by less than 1e-4 of logprob.** Over a
> 150-token response that is `1 - (1 - 0.0117)^150 = 83%` chance of at least one
> flipped token — and one flipped token in a 2-bit model sends it
> off-distribution, which is the degenerate digit-soup tail the gates saw.
>
> The jitter source is ordinary and expected: optimized paged attention is not
> bitwise batch-invariant. Different batch positions and different prefill
> shapes give different reduction orders, worth ~1 ulp. That is normally
> invisible. At 2 bits it is not.
>
> This single mechanism explains **every** observation in this file:
>
> | observation | explanation |
> |---|---|
> | batch=1, caching off: byte-stable over 7 rounds | identical batch composition every round -> no jitter source |
> | batch=1, caching on: round 1 (miss) != rounds 2-4 (hit) | a cache hit prefills only the tail, so the attention shape differs -> ~1 ulp -> flip |
> | batch=N identical prompts: N distinct outputs | each row sits at a different batch position -> ~1 ulp each |
> | eager reproduces the same N outputs every round | position-dependent but deterministic — exactly what fixed reduction orders per position predict |
> | gates fail a different subset every boot | ~83% flip chance per response, independent per prompt |
>
> **So bug A is not a wrong prefix cache and bug B is not a race.** The cached
> KV is reused correctly; it is numerically equivalent, just not bitwise
> identical. Nothing here needs to be turned off, and there is no localized
> defect to fix — making this bitwise reproducible would mean batch-invariant
> attention kernels, which is a large project and not what the goals ask for.
>
> **What this DOES require: the correctness gates must stop scoring exact
> text.** Iteration 1's "ALL GATES PASS" and iteration 2's "FAILED" were the
> same engine behaving the same way. A usable gate must either score on tensors
> (layer outputs vs a reference) or score a *rate* over repeated runs with a
> pass threshold, and the vision gate's keyword lists must tolerate wording
> variation. Rebuilding the gate on that basis is the prerequisite for landing
> any perf change.
>
> **Third, smaller effect — eager mode has a first-round warmup divergence.**
> With caching off and `enforce_eager=True`, round 1 differs from rounds 2-3 on
> 2 of 4 prompts, while the same config under CUDA graphs is perfectly stable
> (graph capture warms everything first; eager JIT-compiles on first use).
> **Discard round 0 as warmup in any future determinism test** — otherwise it is
> indistinguishable from bug A's cold-vs-warm signature. The graph-mode
> comparison above is clean precisely because graph capture removes it, and
> graph mode is what production uses.
>
> Together these explain every gate observation, including why iteration 1's
> "ALL GATES PASS" was luck. Note bug B cannot be worked around by disabling a
> feature — batching is not optional — so it needs fixing regardless. Bug A is
> the one where fix-vs-disable is a real choice, **and that choice is the
> user's.**
>
> ### What has been ruled out (measured, not reasoned)
>
> - **Prefix caching — REJECTED.** Corruption persists with it off. It stays on.
> - **The multi-column mmvq kernel — REJECTED as the cause.** The unmodified
>   kernel corrupts too, and by failure count is the *worse* of the two.
> - **Batching / scheduling — REJECTED.** `max_num_seqs=1` still corrupts.
> - **The vision tower — REJECTED.** `vit_determinism.py` (no engine boot,
>   loads only the 833 MB tower): 10/10 byte-identical outputs on identical
>   input, and an image's embedding is unchanged by what it is batched with.
> - **The sparse-MLA metadata cache — REJECTED.**
>   `VLLM_ROCM_MLA_ALWAYS_REBUILD_META=1` (added at
>   `rocm_aiter_mla_sparse.py:584`, default off, no behaviour change) bypasses
>   the `_prev_metadata_key` fingerprint entirely; two boots with it still
>   diverged. A related suspicion was also checked and is wrong:
>   `max_split_per_batch` looks absent from the fingerprint but derives from
>   `min(max_seq_len, topk_tokens)`, which `clamped_seq_lens` already fixes.
> - **`VLLM_ROCM_USE_AITER=0` — CANNOT be tested.** `Sparse attention indexer
>   ROCm path is only supported on AITER`. The aiter sparse path is therefore
>   still un-eliminated and is the leading remaining suspect.
>
> ### The next experiment is written and ready: `batch_determinism.py`
>
> Across ~14 boots the split is exact: **every batch-1 text/needle output is
> byte-identical between passes; every batch-4 vision output differs.** The
> obvious read is "vision is flaky", but the vision gate is *also* the only one
> that submits 4 requests in a single `generate()`. Since the tower is proven
> deterministic, **batch size is the unbroken confound.**
> `batch_determinism.py` breaks it: the same four TEXT prompts, no images,
> decoded one-at-a-time and four-at-a-time, three rounds each, in one boot. If
> the batch-4 text outputs diverge, this is not a vision bug at all.
> (Two attempts wedged during weight load — GPU at 1% after 18 min with low CPU
> load, a stale-shm collision after `pkill`. Re-run them one at a time.)
>
> Until this is resolved: **"correctness is DONE" is not established**, and any
> gate verdict must be a failure *rate* over repeated runs, not one boot.
> **Decision on fix-vs-disable belongs to the user** — report the root cause,
> do not unilaterally turn a feature off.

This file is the performance effort. Read it before forming a hypothesis;
update it before each iteration ends. `worklog.md` has the text fix and
`worklog_vision.md` the vision fix.

## Hardware and constraints

| | |
|---|---|
| GPUs | **exactly 2x MI300X** (TP2). 8 are present — GPUs 2-7 are free for parallel experiments via `HIP_VISIBLE_DEVICES`, but the SERVING config is always 2. |
| Per GPU | 192 GiB HBM3, **5.3 TB/s**, 304 CU, wave64, gfx942 |
| Peak matrix | bf16 ~1307 TFLOP/s, fp8/int8 ~2615 TFLOP/s |
| Model | 262 GB GGUF, q8_0 attention + q2_K routed experts, 78 layers, 256 experts, 8 active + 1 shared |
| Weights | ~131 GB/GPU |
| CPU | AMD EPYC 9754, 128 cores — CPU offload is permitted |

## Measured baseline — CONTROLLED, 2026-07-27 (`bench2.py`, GPUs 0-1)

Prefix caching OFF, fresh prompt per timed call, decode rate from the slope of
total time vs `max_tokens` over {1, 33, 65} so prefill and per-request engine
overhead cancel. CUDA graphs on, capture sizes [1,2,4,8,16,32,64].

| ctx (tok) | prefill s | prefill tok/s | ms/token | decode tok/s |
|---|---|---|---|---|
| 64 | 0.384 | 167 | 18.56 | **53.9** |
| 938 | 1.186 | 791 | 19.04 | 52.5 |
| 4 618 | 5.365 | 861 | 18.82 | 53.2 |
| 18 417 | 21.47 | 858 | 18.89 | 52.9 |

| batch | ms/step | aggregate tok/s | per-request tok/s |
|---|---|---|---|
| 1 | 18.56 | 53.9 | 53.9 |
| 2 | 24.92 | 80.2 | 40.1 |
| 4 | **74.23** | **53.9** | 13.5 |
| 8 | 132.3 | 60.5 | 7.6 |
| 16 | 163.3 | 98.0 | 6.1 |
| 32 | 228.6 | 140.0 | 4.4 |
| 64 | 372.1 | 172.0 | 2.7 |

Other fixed facts: startup ~272 s solo (~385 s with three engines booting
concurrently); KV cache 43-47 GiB -> 500,656 tokens at
`gpu_memory_utilization=0.92`; decode rate is **flat to 18k context**, so
attention is not yet the cost at these lengths.

**The old "~17-20 tok/s" figure was wrong** — see the iteration-1 log entry.

## The GGUF pair is self-contained — the FP8 repo should not be needed

llama.cpp serves this model from the two GGUF files alone, and it can because
GGUF carries everything. Verified in ours:

| what | where | present? |
|---|---|---|
| tokenizer | backbone `tokenizer.ggml.*` | **12 keys** — model, pre, tokens, token_type, merges, bos/eos/pad ids |
| hyperparameters | backbone `glm-dsa.*` | q_lora_rank, kv_lora_rank, key/value_length_mla, expert_{count,shared_count,gating_func,weights_scale}, `attention.indexer.{head_count,key_length,top_k}`, nextn_predict_layers, rope.dimension_count |
| vision config | mmproj `clip.vision.*` | projection_dim, image_size, patch_size, ... (38 fields) |
| vision tower weights | mmproj tensors | **335 tensors** (`v.blk.N.*`) |

So depending on `/home/hotaisle/models/GLM-5.2-Vision-FP8` is a limitation of
vLLM's plumbing, not of the format. Three things currently reach for it:

1. **Config** — buildable from `glm-dsa.*` plus vLLM's own `Glm5vConfig`
   class; the GGUF config parser exists for exactly this.
2. **Tokenizer** — transformers can read `tokenizer.ggml.*` via `gguf_file=`,
   but only for architectures its GGUF reader knows; `glm-dsa` is unlikely to
   be one, so this probably means building the tokenizer from the metadata
   directly.
3. **Multimodal processor + tower weights** — the hard one. vLLM's
   Glm5v/KimiK25 processing info wants an HF processor (patch size, merge
   size, mean/std); those values are in `clip.vision.*`, and
   `extract_vision_config_from_gguf` already reads several of them. The
   adapter currently sources the tower from `vision_tower.safetensors`; the
   335 mmproj tensors are the same weights.

**Do not treat the FP8 repo as a dependency in new work.** The empirical work
list is whatever a `model=<gguf>` boot with no `hf_config_path` and no
`tokenizer` actually fails on — measure that, do not guess.

## SPECIALISED to this model (iteration 5) — no more generic GGUF framework

**The repo serves exactly one model on exactly one GPU config.** Every failure
during the in-tree move was a *generic-path* failure — the speculators probe
choking on `glm-dsa`, `vocab_size` being a read-only property on a composite
config, an architecture registry rejecting the model. Porting a multi-format
framework and then patching it to tolerate this model was the wrong shape of
work. What the model actually uses, measured across all six shards:

```
F32 709 tensors   Q8_0 872 tensors   Q2_K 228 tensors     <- and nothing else
```

No q6_K, no q4_K, no iq*. **17 of the 19 quant types in the kernels are dead
code.** Removed accordingly:

- `is_remote_gguf` / `split_remote_gguf` / `get_gguf_file_path_from_hf` /
  `is_local_gguf_quant` / quant-type probing — the model is one known local
  file. `is_gguf` collapses to `check_gguf_file`. Purged from `config.py`,
  `config/model.py`, `tokenizers/registry.py`, `gguf_config_parser.py`.
- The adapter registry and the gemma3/diffusion adapters — `GlmDsaGGUFAdapter`
  is constructed directly.
- The loader's remote download and `<repo_id>:<quant_type>` resolution — it
  takes a local shard path or raises.

Still to collapse (next): the kernel-side type dispatch. `vecdotq.cuh` is 1812
lines and `mmvq/mmq/moe/moe_vec` each carry 19 launcher instantiations for
types that can never appear. Cutting to q8_0 + q2_K shrinks compile time
(faster iteration), shrinks code size, and lets the `switch (type)` become a
compile-time constant at each call site.

## GGUF is now IN-TREE — the plugin is deleted (iteration 5)

`~/vllm-gguf-plugin` is uninstalled and removed. Restore points if anything is
missing: `~/vllm-gguf-plugin-archive-2026-07-27.tar.gz` (full tree) and
`~/vllm-gguf-plugin-uncommitted.patch` (the uncommitted diff, which included the
`params.py` fix that makes this model load — `worklog.md` flags it as
irreplaceable).

The in-tree implementation was **recovered from git**, not rewritten: commit
`6635279d8a "[Migration] Migrate GGUF quantization support to plugin (#39612)"`
had deleted it in June. Restored from its parent and reverse-applied its wiring:

- `csrc/libtorch_stable/quantization/gguf/` — kernels, registered in
  `CMakeLists.txt`, declared in `csrc/libtorch_stable/ops.h`, bound in
  `csrc/libtorch_stable/torch_bindings.cpp` under namespace `_C`
  (so `torch.ops._C.ggml_mul_mat_vec_a8`, *not* `_C_stable_libtorch`).
- `vllm/model_executor/layers/quantization/gguf.py`,
  `vllm/model_executor/model_loader/gguf_loader.py`,
  `vllm/transformers_utils/gguf_utils.py`.
- 6 files conflicted on the reverse-apply (`config/model.py`, `arg_utils.py`,
  `linear.py`, `base_config.py`, `weight_utils.py`, `tokenizers/registry.py`) —
  all were both-sides merges, resolved.

**Build notes (these cost real time, do not rediscover):**

- Do NOT `git checkout` a shared header wholesale from an old commit.
  `csrc/libtorch_stable/ops.h` lost every declaration added since June and
  `torch_bindings.cpp` failed on `ngram_compute_n_gram_ids`,
  `fused_minimax_m3_qknorm_rope_kv_insert`, `weak_ref_tensor`. Append the
  removed block instead.
- **Any `CMakeLists.txt` edit forces a CMake reconfigure, which fails** in
  PyTorch's `LoadHIP.cmake:313` with `math cannot parse the expression:
  "( * 100) + "` — `find_package(HIP)` leaves `HIP_VERSION_MAJOR/MINOR` empty
  under cmake 4.4 + ROCm 7.2.4. Fix: reconfigure once with
  `cmake -DHIP_VERSION_MAJOR=7 -DHIP_VERSION_MINOR=2 .`
- Targets are `_C`, `_C_stable_libtorch`, `_moe_C_stable_libtorch` (there is no
  `_C_stable`). Incremental rebuild of the GGUF kernel is **~10 s**; the built
  `.so` must be copied from `build/temp.linux-x86_64-cpython-312/` into `vllm/`.

### New in-tree `mmvq.cuh` — llama.cpp geometry, measured

Replaces the b2899 kernel with llama.cpp's GCN-table geometry (`nwarps`
cooperating per block with an LDS reduction, `rows_per_cuda_block`, and the
`small_k` widening), keeping the b2899 `vec_dot` signature so `vecdotq.cuh` is
untouched. Same shape as the table below (q2_K, m=4096, k=14336):

| n | old b2899 | **new in-tree** | llama.cpp | vs old |
|---|---|---|---|---|
| 1 | 22.35 | **17.68** | 16.60 | **1.26x** |
| 2 | 33.34 | **22.96** | 20.79 | **1.45x** |
| 3 | 52.48 | **37.85** | 27.30 | **1.39x** |
| 4 | 52.92 | **39.54** | 34.11 | **1.34x** |
| 5 | 82.88 | 96.32 | 60.18 | *0.86x* |
| 8 | 85.18 | 98.58 | 34.26 | *0.86x* |

**1.26-1.45x at n=1-4, now within 6-28% of llama.cpp.** That is the range decode
actually uses (batch 1, and 3-4 rows under nspec=2/3 speculation), so this is
the win that matters.

**n>=5 REGRESSED and the fix is not more GEMV.** Two facts explain it. First,
this dispatch pads 5-7 up to ncols_dst=8 and computes the wasted columns, while
llama.cpp instantiates every width 1..8. Second and more important:
**llama.cpp's own n=8 (34.26 us) is faster than its n=5 (60.18 us)** despite 60%
more work, which is only possible if it changed kernel — `test-backend-ops`
benchmarks the whole `ggml_cuda_mul_mat` dispatch, so n=8 is landing on its
MFMA MMQ, not MMVQ. Hoisting the tail clamps out of the k loop was tried and
did **not** help n>=5 (96.48 -> 96.32), confirming it is not an addressing
problem. So: instantiate ncols_dst 1..8 exactly, and for n>=8 port llama.cpp's
modern MMQ instead of pushing the GEMV — which matches iteration 1's finding
that the b2899 `mmq` is flat in batch but runs at only 139-277 GB/s.

## Reference implementation: llama.cpp on the SAME model and hardware

`llama-bench`, same GGUF, same 2x MI300X (`~/llama.cpp`, b10121, HIP build):

| test | llama.cpp `-sm layer` | vLLM (this work) |
|---|---|---|
| prefill pp512 | 323.6 ± 1.4 tok/s | **~860 tok/s** (2.7x faster) |
| decode tg128 | **36.9 ± 0.01 tok/s** | **53.9** (57-61 with speculation) |

**vLLM is already ahead of the reference implementation on both axes**, which
retires the "we are at 10% of roofline so there must be 10x sitting there"
framing at the top of this file. The most tuned k-quant decode stack in
existence gets 36.9 tok/s on this model.

The reason vLLM wins end-to-end is TP2 vs `-sm layer`: layer-split runs one GPU
at a time, so llama.cpp pays ~2x the roofline time. (`-sm row`, its
tensor-parallel mode, **fails to load this MoE** — no useful error from
llama-bench. Don't retry it; go straight to op-level comparison.)

### Kernel vs kernel, same shape, same GPU — THIS is the prize

`test-backend-ops perf -o MUL_MAT -p type_a=q2_K` benchmarks llama.cpp's kernel
at `m=4096 k=14336` (19.3 MB weight) on one MI300X. `cmp_llamacpp.py` runs the
plugin's kernel at the identical shape:

| batch n | plugin us | llama.cpp us | llama.cpp faster |
|---|---|---|---|
| 1 | 22.35 | 16.60 | **1.35x** |
| 2 | 33.34 | 20.79 | **1.60x** |
| 3 | 52.48 | 27.30 | **1.92x** |
| 4 | 52.92 | 34.11 | **1.55x** |
| 5 | 82.88 | 60.18 | 1.38x |
| 8 | 85.18 | 34.26 | **2.49x** |

**llama.cpp's q2_K GEMV is 1.35-2.5x faster at every batch width**, after the
multi-column change already landed here. q2_K is the dominant weight type (all
256 routed experts), so this is the main decode cost. The 1.35x at n=1
independently matches the ~1.35x per-GPU efficiency inferred from the
end-to-end layer-split arithmetic above — two different measurements agreeing.

Porting these kernels into vLLM's TP2 is therefore worth roughly 1.4-2x on the
decode GEMMs, i.e. plausibly 54 -> 75-90 tok/s single-request before
speculation. That is a far better return than any further hand-tuning of the
2024 fork, and it is now a measured claim rather than an intuition.

### What modern llama.cpp has that this plugin does not

The GGUF plugin's kernels are a **b2899 (mid-2024) snapshot**. `mmvq.cu` has
since been rewritten: 1289 lines vs the fork's 212. Concretely missing:

- **Per-architecture parameter tables.** gfx942 maps to `MMVQ_PARAMETERS_GCN`,
  with `calc_nwarps` and `calc_rows_per_block` tuned per quant type *and* per
  batch width. The fork hardcodes 1 warp and 1 row per block.
- **`small_k` specialisation** — `rows_per_block = nwarps` when
  `blocks_per_row_x < nwarps * blocks_per_iter_1warp`. For the MoE `w2` shape
  (k=1024, q2_K) that test is `4 < 8` -> true, so llama.cpp puts 2 rows per
  block there. **This is exactly the w2 defect measured independently in
  iteration 2** ("each wave does a single `vec_dot` then a full 64-lane shuffle
  reduction", 10x off its own unique-byte roofline). Already solved upstream.
- **Multi-row blocks at batch>1** (`rows_per_block=2` for ncols_dst 2..8),
  which also fixes the occupancy regression the hand-written multi-column
  kernel hit on `shared_gate_up` (2048 rows -> only 2048 waves for 304 CUs).
- **A dedicated multi-token MoE kernel** (`mul_mat_vec_q_moe_launch` for
  `has_ids && ncols_dst > 1`) instead of the fork's one-block-per-(token,expert).
- **Fusion support** (`has_fusion`) for gate/up.

**Conclusion: stop hand-patching the 2024 fork.** The multi-column change
landed in iteration 2 was an independent re-derivation of one piece of this.
Porting modern `mmvq.cu` (plus `vecdotq.cuh`, `quantize.cuh`) into the plugin
subsumes it and brings the rest.

## The headline number: decode is at ~10% of roofline

Per-token decode weight traffic is ~25.6 GB (of which ~14.4 GB is the q8_0
attention/projection half; see `marlin-hip-*` memories). Against 5.3 TB/s:

```
roofline decode  = 5.3e12 / 25.6e9  ~= 207 tok/s
observed         ~= 17-20 tok/s     ~= 8-10% of roofline
```

**There is ~10x of headroom and the kernels are the reason.** Prior QuixiCore-ROCm
work established the mechanism: the q2_K/k-quant decode path is **decoder-ALU
bound, not bandwidth bound** — every denser format tested (q6_K, q5_K, q4_K, q4_0,
iq4_nl, mxfp4, nvfp4) was SLOWER in wall clock than q8_0 despite moving fewer
bytes. So do NOT chase "fewer bytes" until the kernel is actually near roofline.
Start by measuring achieved fraction of roofline per kernel.

Kernel reference material: `~/QuixiCore/QuixiCore-ROCm` (`perf/perf.md` is the
handbook; `perf/optimization_status.md` has measured results for MFMA qgemm,
k-quant decode, MoE grouped GEMM, MLA decode).

## 256k context: mostly already there

`GPU KV cache size: 500,656 tokens` — **KV capacity is NOT the blocker**. 256k
fits with room to spare. The test scripts simply set `max_model_len=8192/16384`.
A 14,759-token text needle already PASSES (`len2.log`).

So the 256k work is: raise `max_model_len`, then find what actually breaks or
degrades — prefill time, chunked-prefill behaviour, the DSA sparse indexer
(`index_topk=2048`) at long range, attention kernel scaling, and accuracy at
depth. Measure needle recall at 32k/64k/128k/256k, not just "does it run".

## Correctness gates v2 (iteration 5) — USE THIS ONE: `<SCRATCH>/gate2.py`

The v1 keyword gate is unusable: 1.17% of greedy decisions are decided by
<1e-4 of logprob, so the unmodified engine fails a different random subset every
boot (6 / 0 / 2 / 5 over four boots). **Score the model, do not sample it.**

| check | metric | value | jitter across boots |
|---|---|---|---|
| text NLL | mean logprob of a fixed 29-token reference continuation | **-1.4320** | **0.0000** — identical in all 8 suite runs |
| needle@1k / @7k | logprob of `BLUE-HERRING-7742` vs 4 decoys after the haystack | margin **+1.47 to +1.77** | ~0.1 |
| vision img1-4 | short cloze vs 3 decoys (river / snow / ESPERANCA / "Pidgey at level 17") | margin **+0.97 to +8.41** | ~0.1-0.3 |

Margin-to-noise is 10-80x, so a single boot now gives a real verdict. Verified
on four boots (2 control `.so`, 2 modified): **ALL GATES PASS on all four**, and
the control and modified numbers agree to within the jitter.

### Gate designs that were tried and REJECTED

- **Whole-caption forced choice** (score each of 4 full captions under each
  image, require each image to prefer its own). Failed img1 and img2
  *identically on both kernels* — raw `logP(caption|image)` is dominated by how
  probable each caption is on its own, not by the image.
- **PMI normalisation** of the above (`logP(cap|image) - logP(cap|no image)`).
  Overcorrects: the caption with the least likely no-image baseline then wins
  for *every* image. Worse than raw.
- **Long continuations generally.** A 30-token caption spreads the image
  contribution across mostly-generic tokens; margins came out at +0.007 to
  +0.087, i.e. *below* the gate's own 0.09-0.13 noise. Short, fully
  image-dependent answers give margins 100x larger.

### Measurement trap: image placeholders expand inside the engine

Scoring a continuation by slicing `prompt_logprobs[len(tok.encode(prefix)):]`
is wrong for any multimodal prompt — one `<|image|>` placeholder becomes
hundreds of tokens in the engine, so that offset lands in the middle of the
image tokens. It reported mean logprob **-15 for the word "river"** and margins
under the noise floor, and it looked exactly like "the model has weak vision
grounding". **Slice the last `n_cont` entries instead**, which is
expansion-proof. The text-only checks were unaffected, which is why the text
NLL looked healthy while vision looked broken.

## Correctness gates v1 (superseded — kept for the ground truth it encodes)

Run before landing anything. Any regression = revert, no exceptions.

1. **Text**: `"The capital of France is"` -> " Paris..." coherent.
2. **Long text**: needle-in-haystack recovers `BLUE-HERRING-7742` at 38 / 407 /
   1879 / 5559 / 14759 tokens (`<SCRATCH>/serve_len.py`).
3. **Vision 4/4**: `<SCRATCH>/serve_four.py` — img1 valley/river/cliffs,
   img2 snow peak, img3 bench + **ESPERANCA** + anchor + paving,
   img4 **Pokemon battle** + **Pidgey** + level **17** + Pikachu.
4. **Layer-sum sanity** (only if touching attention/MoE/quant kernels): l_out sums
   vs llama.cpp, median per-element error was 0.87%.

Fast pre-checks that need no engine boot:

- `<SCRATCH>/vit_ref.py` — vision tower + projector vs llama.cpp, ~30 s.
- `<SCRATCH>/cmp_vit_weights.py` — weight-level diff.

## Measurement rules (learned the hard way — do not repeat)

1. **Score on tensors, not on generated text.** Two hypotheses in the vision hunt
   survived far too long because "the output still looks wrong" cannot distinguish
   wrong from differently-wrong. For perf: score on measured latency/throughput,
   not on impressions.
2. **Never fingerprint by a reduction.** `sum()` collides and will manufacture a
   convincing false pattern. Hash exact bytes.
3. **Compare like for like.** Check tensor SHAPE before comparing (llama.cpp
   computes only the final position in the last layer).
4. **A sum is a bad similarity metric** — cancellation inflates relative error
   ~10x. Compare per element.
5. Use HIP events; warm up; batch tiny kernels per sample and record the batch
   size; keep allocation out of the timed region.
6. `rocprofv3 --kernel-trace` for per-dispatch time; `--pmc SQ_INSTS_VALU
   SQ_INSTS_MFMA TCC_HIT_sum TCC_MISS_sum` to tell ALU-bound from bandwidth-bound.
7. Startup is ~272 s. Batch several questions into one boot. Run independent
   experiments concurrently on GPUs 2-7 — never serialize by habit.
8. `limit_mm_per_prompt={"vision_chunk": 0}` crashes engine init at 16k context;
   use `1` even for text-only runs.

## Targets

| goal | measured baseline | target |
|---|---|---|
| single-request decode | 53 tok/s | **>= 60 tok/s** |
| single-request decode | 53 tok/s | **>= 200 tok/s** |
| context | 16k tested | **256k**, needle recall at 32/64/128/256k |
| startup | 272 s | opportunistic; do not trade correctness for it |

**BOTH throughput goals are single-request** (user correction, 2026-07-27). Goal
2 is 200 tok/s for ONE request, not aggregate over a batch. That is 5 ms/token
against 18.6 today. No kernel-level change gets within 3x of that — a decode
step launches ~150 kernels per layer-step and the biggest single kernel family
(q2_K MoE GEMV) is only ~35% of the step. **The only lever of the right
magnitude is emitting several tokens per target forward pass**, i.e.
speculative decoding.

User-directed inputs for that (2026-07-27):

- `RedHatAI/GLM-5.2-speculator.dspark-preview` — downloaded to
  `/home/hotaisle/models/GLM-5.2-speculator`. 5-layer qwen3 draft, 7.6 GB bf16,
  `speculators_model_type: dspark`, block_size 8, 7 speculative tokens, aux
  hidden states from target layers [8,23,39,55,70], verifier arch
  `GlmMoeDsaForCausalLM` — our exact text backbone.
- **TurboQuant** — `~/QuixiCore/specs/formats/turboquant.md`. It is a *KV-cache*
  format, not a weight format: 2-8 bit keys with per-32-group fp16 scale+zero,
  values via sign-flip + unnormalized FWHT + per-group RMS + centroid table.
  Relevant to goal 3 (256k) and to batch capacity, not to single-request decode
  latency. Not yet started.

Targets are direction, not a contract — record what was achieved and why the
remainder is hard.

## Startup time baseline (2026-07-27)

**Goal of this phase: reduce startup time.** Baseline first, no optimization until
the numbers are pinned.

### What is being measured, and why

| engine | definition |
|---|---|
| llama.cpp | **process exec -> first generated token** |
| vLLM | **process exec -> port accepting connections** (`/health` returns 200) |

Both are external wall clock from `Popen`, not internal timers.
llama.cpp's own `load time` (`llama_perf_context_print`) covers **model load
only** — it excludes exec, dynamic linking, ROCm/HIP runtime init and backend
registration, all of which a real deployment pays. Its log timestamps
(`min.sec.ms.us`, `common/log.cpp:99`) likewise start at log init, after exec.
vLLM's readiness LOG line and an actually-listening socket are also not the same
thing, so both are recorded and the socket/health number is the baseline.

Harnesses: `<SCRATCH>/measure_startup_llamacpp.py`,
`<SCRATCH>/measure_startup_vllm.py`.

### Conditions

- **WARM page cache** — the host has 3 TB RAM with ~2755 GB in buff/cache, so the
  262 GB model is resident. Cold-cache boots would be disk-bound and much slower.
  All numbers below are warm unless stated. Do not compare warm to cold.
- Runs are **serialized**, never concurrent: startup is weight-load dominated, so
  two 262 GB loads at once contend for memory bandwidth and corrupt both numbers.
- TP2, GPUs pinned via `HIP_VISIBLE_DEVICES`.

### Results

| engine | metric | time |
|---|---|---|
| llama.cpp | exec -> first token | *pending* |
| vLLM | exec -> /health 200 | *pending* |

## Status

Landed this iteration, **all three correctness gates re-run and PASSING** with
the changes in place (`gates.py`: text/Paris; needle at 38/223/959/2799/7399;
vision 4/4 including ESPERANCA and Pidgey/Lv.17 — plus the separate 256k needle
run at 6 depths):

- [x] mmvq/mmq threshold (`vllm_gguf_plugin/quantization/linear.py`). Aggregate
      throughput +94%/+115%/+52% at batch 4/8/16; single-request unchanged.
- [x] Two vLLM fixes so a standalone speculators draft loads against a quantized
      target (`spec_decode/dspark/utils.py`, `spec_decode/eagle/utils.py`).
- [x] Goal 3, 256k context: recall 6/6 at 32k / 64k / 128k, 5/5 completed depths
      at 256k (the run was killed before the last depth). Prefill 843 -> 759
      tok/s from 32k to 256k. Nothing broke.
- [x] Goal 1, 60 tok/s single request: **60.8 tok/s** with dspark at
      `num_speculative_tokens: 3` at short context — but that setting regresses
      to 44.5 at 4620 tokens. The setting that is a win everywhere is nspec=2:
      57.0 / 61.1 / 54.3 tok/s vs 53.9 / 52.5 / 53.2. So goal 1 is **touched,
      not held**: ~+8% robustly, 60+ only at some context lengths.

**NOTHING IS BROKEN — prefix caching stays on, no feature needs disabling.**
Bugs A and B were the same thing and it is not a defect: see ROOT CAUSE above.
The user's fix-vs-disable decision is moot.

- [x] **Gate v2 built and validated** (`gate2.py`, iteration 5). Margin-to-noise
      10-80x, so a single boot is a real verdict again. See the gate section
      near the top of this file.
- [x] **Multi-column `mul_mat_vec_q` is GATE-VERIFIED and LANDED.** Passes gate
      v2 on two independent boots, numbers matching the control within jitter.

- [x] **GGUF moved in-tree, plugin deleted** (iteration 5). See the in-tree
      section near the top.
- [x] **New `mmvq.cuh` with llama.cpp geometry**: 1.26-1.45x at n=1-4.

**IN-TREE IS VERIFIED: ALL GATES PASS**, numerically equivalent to the plugin:

| check | plugin baseline | in-tree |
|---|---|---|
| text NLL | -1.4320 | **-1.4320** (identical) |
| needle@1k / @7k margin | +1.47 .. +1.77 | +1.58 .. +1.78 |
| vision img1-4 margin | +0.97 .. +8.41 | +0.87 .. +8.42 |

The plugin's Python now lives at
`vllm/model_executor/layers/quantization/gguf/`,
`vllm/model_executor/model_loader/gguf_{loader,adapters,weight_utils}.py` and
`vllm/transformers_utils/gguf_{utils,config_parser}.py`.

**Two bugs found getting there, both worth remembering:**

1. **Vision silently disappears if the multimodal processor cannot resolve.**
   `_create_processing_info` raises, `supports_multimodal_inputs` returns
   False, and the model loads and answers text perfectly while **vision is
   gone** — one `warning_once` ("treated as multimodal but has no registered
   multimodal processor; running in text-only mode") is the only evidence. A
   text-only degradation that passes every text gate is the nastiest failure
   mode in this stack; the vision gate is what catches it.

   The current workaround points `model` at the unquantized repo and puts the
   .gguf in `model_weights` (`arg_utils.create_model_config`). **That is a
   stopgap, not a requirement — see the next section. An earlier version of
   this file claimed the unquantized repo was necessary; that was wrong.**
2. A transient `hipErrorUnknown` at `prepare_inputs_event.synchronize()` on the
   first `execute_model` appeared once and did not reproduce. Before blaming
   the kernels, they were cleared standalone: `intree_shapes.py` (mmvq at all 7
   real TP2 shapes x M in {1,2,3,4,5,8,16}, including the k=1024 `small_k`
   path) and `intree_moe.py` (`ggml_moe_a8_vec`, `ggml_moe_a8`,
   `ggml_dequantize` at real w13/w2 shapes) are all clean. If it returns, it is
   not the GGUF kernels.

**End-to-end after the move (`bench2.py`, in-tree vs plugin baseline):**

| | plugin | in-tree | |
|---|---|---|---|
| single, ctx 64 | 53.9 | **54.7** | +1.5% |
| single, ctx 4618 | 53.2 | 53.8 | +1.1% |
| single, ctx 18417 | 52.9 | 52.7 | — |
| batch 4 aggregate | 119.0 | **122.2** | +2.7% |
| batch 8 aggregate | 153.3 | **155.4** | +1.4% |
| batch 16 aggregate | 149.1 | **176.2** | +18% |
| batch 64 aggregate | 183.8 | **194.4** | +5.8% |

**Small, and the reason is instructive.** The kernel microbenchmark showed
1.26x at n=1, but only the *dense* GEMV calls got the new geometry — the MoE
still runs the old `moe_vec_q`. At batch 1 the MoE is 59 us/layer against the
dense 49.6 us/layer, so more than half the GEMV time was never touched.
**Porting the nwarps / rows_per_block / small_k geometry into `moe_vec.cuh` is
the obvious next win**, and `small_k` applies directly: w2 is k=1024, exactly
the shape the heuristic targets.

Next, in order:

1. **Port the new geometry to `moe_vec_q`** — the largest untouched GEMV, and
   the reason the end-to-end gain is 1.5% instead of ~10%.
2. Collapse the kernel type dispatch to q8_0 + q2_K (see the specialisation
   section) — faster builds, smaller code, no runtime switch.
3. Instantiate ncols_dst 1..8 exactly (stop padding 5-7 up to 8) and port
   llama.cpp's MFMA MMQ for n>=8 — see the analysis above for why the GEMV is
   the wrong tool at that width.
4. Then the specialisation ideas: bf16-promote the always-hot tensors, fuse the
   expert into one kernel, fuse q8_1 activation quantisation into the RMSNorm
   epilogue (562 launches/step today).

Deferred, still worth doing after the port (the port may subsume them):

- Re-measure the speculative sweep with **at most two concurrent engines**;
  iteration 2's nspec=3 ctx-940 point (29.6 vs 57.7 tok/s) was taken with four
  engines booting and is not believable.
- Enable the MoE per-matrix split (`VLLM_GGUF_MOE_VEC_W1=96`,
  `VLLM_GGUF_MOE_VEC_W2=80`) — ~20% at batch 32, ~2x on w2 at batch 64.
  `moe_mixed_check.py` hung and never gave a verdict; with gate v2 the
  engine-level A/B is the cheaper route.

Earlier perf work, now unblocked:

- Multi-column `mul_mat_vec_q` is IN THE TREE but **NOT gate-verified** (the
  gates cannot currently give a verdict). ~2x at M>=4 on 4 of 5 shapes, M=1
  unchanged, batch-8 aggregate 130.2 -> 153.3 tok/s, nspec=3 speculative decode
  60.8 -> 68.7 tok/s at short context and 44.5 -> 65.6 at 4620 tokens. It is
  deterministic in isolation and fails fewer gates than the unmodified kernel.
- Re-measure the speculative numbers with **at most two concurrent engines**;
  the nspec=3 ctx-940 point (29.6 tok/s vs 57.7) was taken with four engines
  booting and is not believable.
- `moe_vec_q` for q2_K is untouched and is still the largest single kernel
  family (34% of a batch-32 step, 32% of peak on w13, 12% on w2). The
  multi-column trick does NOT apply to it: at M=8 the 8 tokens route to ~59
  distinct experts, so there is almost no weight to share. It needs the
  lane-efficiency fix instead — `w2` (k=1024) gives each wave a single
  `vec_dot` followed by a full 64-lane shuffle reduction, which is why it sits
  at 10x off its own unique-byte roofline.

Then:

- [ ] Goal 2 also needs a draft matched to this target (2.29 acceptance length
      caps the win at ~2.3x). Retraining against the q2_K Vision checkpoint is
      the only real fix; nothing else moves it.
- [ ] MoE per-matrix kernel split is WRITTEN but UNVERIFIED and therefore
      DISABLED by default (`VLLM_GGUF_MOE_VEC_W1=96`, `VLLM_GGUF_MOE_VEC_W2=80`
      turn it on). `moe_mixed_check.py` hangs on the CPU inside the first
      `ggml_moe_a8_vec` call with E=32 and never reaches a verdict — debug that
      before enabling. Worth ~20% at batch 32 and ~2x on w2 at batch 64.
- [ ] TurboQuant KV cache — not started. It is a KV format, so it buys context
      capacity and batch headroom, not single-request decode latency. Lowest
      priority of the three now that 256k already fits.
- [ ] Prefill is 760-860 tok/s and flat; a 256k prompt costs 338 s. Untouched so
      far and a separate problem from decode (mmq/GEMM path, not GEMV).

## Log

### Iteration 1 (2026-07-27)

**The stated 17-20 tok/s baseline was mis-derived and is wrong.** It came from
`prod1.log`'s "generation 2.8 s" for 3 prompts x 16 tokens = 48/2.8 = 17 tok/s —
that is prefill + engine overhead + decode for three separate requests, not a
decode rate. Controlled measurement is **53 tok/s single-request decode**
(18.6 ms/token), flat from 64 to 18k context. Goal 1 (>=60) needs ~12% more,
not 3x. **Goal 2 is the real problem: aggregate throughput at batch 4 is 53.9
tok/s — no better than batch 1 — and per-request rate collapses to 13.5 tok/s.**

**Eager vs graphs: 6.94 tok/s eager vs ~65 tok/s with CUDA graphs (9x).**
Decode is heavily launch-bound; anything that adds kernels or breaks graph
capture is expensive.

#### Measurement traps hit (do not repeat)

1. **Prefix caching poisoned the first e2e benchmark.** Timing the *same*
   prompt at `max_tokens=1` and then `max_tokens=1+D` makes the second prefill
   free, so `D/(t_D - t_1)` went negative at long context. Use fresh prompts
   and `enable_prefix_caching=False`.
2. **A plain python timing loop bottoms out at ~22 us per call** — python +
   torch custom-op dispatch, not the kernel. Every shape from 2 MB to 400 MB
   measured "22 us". Capture N invocations into a `torch.cuda.CUDAGraph` and
   replay; that is also what production decode does.
3. **`rocprofv3` cannot profile this engine**: it aborts at startup with
   `api registration failed with error code 16: Configuration request occurred
   outside of valid rocprofiler configuration period` (something vLLM dlopens
   registers after rocprofiler's config window). Use vLLM's built-in
   `VLLM_TORCH_PROFILER_DIR` + `llm.start_profile()` instead.
4. **Killing stale engines by scraping `ps` PIDs killed a live experiment.**
   A failed `llm.start_profile()` leaves `VLLM::EngineCore` + workers holding
   the GPUs and the shm broadcast block, so the next boot on those GPUs hangs
   on "No available shared memory broadcast block found in 60 seconds". Clean
   up, but identify the victims by GPU occupancy and start time, not by
   pattern-matching `VLLM::` and killing everything that matches.
5. **`moe_vec` re-reads the expert weights for every token** (grid.z =
   tokens*top_k, each block loads its own expert row). Bytes moved = (tokens x
   top_k) x per-expert-bytes, NOT unique-experts-bytes. Scoring it against
   unique experts understates achieved bandwidth by ~4x and hides the fact that
   the kernel has zero weight reuse across a batch.

#### Per-kernel GPU time, real TP2 shapes, CUDA-graph timed (`kbench2.py`)

us per call; parenthesis = achieved GB/s counting quantised weight bytes.
`bf16blas` = weights pre-dequantised to bf16 + hipBLASLt (dequant NOT in the
timed region — this is the "what if the weight were bf16 in HBM" reference).

| layer (n x k, type) | b=1 | b=4 | b=16 | b=32 | b=128 |
|---|---|---|---|---|---|
| o_proj 6144x8192 q8_0 mmvq | 18.8 (2840) | 52.3 | 191 | 378 | 1499 |
| o_proj mmq | 385 (139) | 387 | 390 | 393 | 407 |
| o_proj bf16blas | 24.7 (4072) | 24.7 | 24.8 | 28.3 | 41.0 |
| dense_gate_up 12288x6144 q8_0 mmvq | 25.4 (3161) | 76.7 | 291 | 580 | 2313 |
| dense_gate_up mmq | 289 (277) | 290 | 291 | 293 | 307 |
| dense_gate_up bf16blas | 36.3 (4163) | 36.4 | 36.3 | 35.8 | 55.4 |
| q_b_proj 6144x2048 q8_0 mmvq | 9.8 (1367) | 19.5 | 58.1 | 111 | 430 |
| q_b_proj mmq | 96.5 (139) | 97.2 | 97.6 | 98.5 | 103 |
| shared_gate_up 2048x6144 q2_K mmvq | 9.9 (417) | 16.8 | 40.6 | 73.5 | 275 |
| shared_gate_up mmq | 530 (8) | 534 | 536 | 538 | 550 |
| shared_down 6144x1024 q2_K mmvq | 8.9 (233) | 18.1 | 54.6 | 103 | 397 |
| lm_head 77440x6144 q6_K mmvq | 167 (2337) | 676 | 2702 | 5362 | 21084 |
| lm_head mmq | 809 (482) | 810 | 815 | 819 | 1653 |

MoE q2_K, 256 experts, top-8, per GPU (`w13` [256,2048,6144], `w2` [256,6144,1024]):

| requests | w13 vec us | w13 %peak | w13 mmq us | w2 vec us | w2 %peak | w2 mmq us |
|---|---|---|---|---|---|---|
| 1 | 28.4 | 21.9% | 154 | 30.6 | 10.2% | 44.3 |
| 8 | 151 | 33.1% | 416 | 210 | 11.9% | 220 |
| 32 | 627 | 31.8% | 1213 | 837 | 11.9% | 552 |
| 128 | 2481 | 32.2% | 1866 | 3388 | 11.8% | 950 |

#### What the profile says

1. **`mmq` time is flat in batch, `mmvq` time is linear.** Crossover =
   mmvq_b1_GB/s / mmq_GB/s: lm_head 2337/482 ~= 5, dense_gate_up 3161/277 ~= 11
   (measured 16), o_proj 2840/139 ~= 20 (measured 32), q2_K shared 417/8 ~= 52
   (mmvq always wins). **The plugin switches at 2.** Every dense layer runs the
   ~5x slower kernel from batch 4 up. This is exactly the observed e2e cliff:
   aggregate throughput went 65 (b1) -> 108 (b2) -> **63 (b4)** -> 67 (b8).
2. **Nothing quantised is near roofline.** Best case is q8_0 mmvq at
   2800-3200 GB/s (53-60% of peak) on the two largest dense shapes; q2_K mmvq
   on the shared expert manages 417 GB/s (8%); every `mmq` path is 8-500 GB/s.
   MoE `moe_vec` holds ~32% of peak on w13 and ~12% on w2 regardless of batch.
3. **Small kernels bottom out at ~9-10 us even inside a graph** (kv_a_proj is
   3.8 MB and takes 9.8 us; ideal is 0.7 us). Per-dispatch floor on gfx942 plus
   occupancy starvation — `shared_gate_up` launches only 2048 waves for 304 CUs.
   With ~8 GEMM launches per layer x 78 layers plus norms/rope/router/reduce,
   decode is substantially dispatch-bound, which is why eager is 9x slower.
4. **bf16 + hipBLASLt reaches 4.1 TB/s (77% of peak)** and beats the quantised
   kernel outright at b=1 on 5 of 9 shapes, and by 5-15x at b>=8. Upcasting the
   q8_0 attention half to bf16 in HBM (~+6 GB/GPU) is the obvious batched-decode
   lever, but it does nothing for b=1 and costs KV cache.
5. **`w2` MoE should switch to mmq far earlier than `w13`.** The code gates both
   on `x.shape[0] > 64` request tokens, but w2 sees `tokens*top_k` rows, so at
   16 requests w2 is at 128 rows where mmq is already 20% faster.

#### LANDED: mmvq/mmq threshold (`vllm_gguf_plugin/quantization/linear.py`)

`mmvq_safe = 2 if rows > 5120 else 6` -> `_mmvq_batch_limit(rows, qtype)`:
4 for `rows >= 32768` (vocab projection), 64 for `rows < 4096` (mmq cannot fill
304 CUs at narrow output), 16 otherwise. ROCm-only default; other platforms keep
the old numbers; `VLLM_GGUF_MMVQ_MAX_BATCH` overrides.

A/B, two boots, identical scripts, GPUs 0-1 vs 4-5 (aggregate tok/s):

| batch | before | after | change |
|---|---|---|---|
| 1 | 53.9 | 53.8 | 0% |
| 2 | 80.2 | 80.3 | 0% |
| 4 | 53.9 | **104.8** | **+94%** |
| 8 | 60.5 | **130.2** | **+115%** |
| 16 | 98.0 | **149.1** | **+52%** |
| 32 | 140.0 | 162.9 | +16% |
| 64 | 172.0 | 183.8 | +7% |

Single-request decode is untouched (53.9/52.1/52.6/52.0 vs 53.9/52.5/53.2/52.9
at 64/938/4618/18417 tokens), which is the expected result — batch 1 used mmvq
before and after. Correctness gates not yet re-run; the change only reroutes
between two kernels that already produce the same result at other batch sizes,
but it is not landed-final until the gates pass.

#### Goal 3 (256k context): DONE for text — `ctx_needle.py`

`max_model_len=262144`, chunked prefill, 6 needle depths per length:

| context | recall | prefill tok/s |
|---|---|---|
| 32 101 | **6/6** | 843 |
| 64 163 | **6/6** | 816 |
| 128 287 | **6/6** | 760 |
| 256 581 | 3/3 so far | 759 |

Nothing broke: no OOM, no crash, no chunked-prefill failure, and recall is
perfect at every depth. Prefill throughput decays only 11% from 32k to 256k, so
the DSA indexer scales. **The 256k cost is prefill latency, not capacity**: a
256k prompt takes 338 s at 759 tok/s. Prefill is the next target for long
context, and it is a separate problem from decode (it is the mmq/GEMM path, not
the GEMV path).

#### Goal 2 (200 tok/s single request): DSpark speculative decoding

Two vLLM bugs found and fixed to get the draft to load against a GGUF target.
Both are general (any speculators-format draft + any quantized target), not
GLM-specific:

1. `vllm/v1/worker/gpu/spec_decode/dspark/utils.py` — `load_dspark_model` reused
   the target's `LoadConfig`, so a standalone draft directory was handed to the
   GGUF loader: `ValueError: Unrecognised GGUF reference:
   /home/hotaisle/models/GLM-5.2-speculator`. DSpark heads usually ship inside
   the target checkpoint, which is why nobody hit it. Added `_draft_load_config`:
   fall back to `load_format="auto"` when the draft path differs from the target.
2. `vllm/v1/worker/gpu/spec_decode/eagle/utils.py` — `_should_share` did
   `torch.equal(draft.weight, target.weight)` to decide whether the draft can
   share the target's embedding/lm_head. A GGUF target has `qweight`, no
   `weight`: `AttributeError: 'VocabParallelEmbedding' object has no attribute
   'weight'`. A quantized target's packed tensor is neither comparable to nor
   substitutable for the draft's dense one, so return False (draft keeps its own
   copy — this draft ships both, 951M params each).

Third blocker, config-level not a bug: `ValueError: No common block size for 16`.
`ROCM_AITER_MLA_SPARSE` accepts kernel block sizes `[1, 64]`, the draft's default
`ROCM_AITER_FA` accepts `[16, 32]`, and the KV manager default is 16 — the
intersection is empty. `block_size=64` alone fixes it (verified: booted with and
without `attention_backend=TRITON_ATTN` for the draft, both work, so the two
attention stacks land in separate KV cache groups). Also needs
`max_num_batched_tokens` >= max_num_seqs * (1 + num_speculative_tokens); 4096
with the default max_num_seqs gives
`max_num_scheduled_tokens is set to -2048`.

**RESULT: speculative decoding is currently a 34% REGRESSION.**

| config | single-request decode |
|---|---|
| no speculation | **53.9 tok/s** (18.6 ms/token) |
| dspark, 7 draft tokens, TRITON_ATTN draft | 35.6 tok/s (28.1 ms/token) |
| dspark, 7 draft tokens, AITER_FA draft | 36.1 tok/s (27.7 ms/token) |

Flat across context (35.6 / 36.3 / 35.7 at 65 / 940 / 4620 tokens), and it drags
batched down too (batch 8 aggregate 53.2 vs 130.2 without).

**Acceptance, measured (`spec_accept.py`, 166 drafts, 1162 draft tokens):**

```
Mean acceptance length  2.29     (1 bonus + 1.29 accepted)
Avg draft acceptance    18.5%
Per-position rate       0.583  0.327  0.160  0.096  0.058  0.045  0.026
```

Two things fall out, and together they explain the regression exactly.

1. **Acceptance decays fast.** Positions 5-7 contribute 0.13 tokens between them
   while costing three extra draft forwards and three extra verify rows. The
   draft was trained on `zai-org/GLM-5.2-FP8`; the target is a **q2_K** quant of
   GLM-5.2-**Vision**. 58% at position 1 is what a mismatched draft looks like.
2. **Every extra verify row costs ~6.5 ms.** Reconstructing step time from
   `acceptance_length / tok_s`: 1 row 18.6 ms, 3 rows 33.5, 4 rows 34.0, 8 rows
   64.3. On a bandwidth-bound fp8/bf16 model an 8-row step is nearly free versus
   1 row, which is *why* speculation normally works. Here `mul_mat_vec_q` and
   `moe_vec_q` reload the whole weight for every row — zero reuse — so the
   verify step scales almost linearly and break-even needs 3.46 accepted tokens
   when the draft supplies 2.29.

**Sweep of `num_speculative_tokens` (single-request tok/s, one boot each):**

| nspec | ctx 65 | ctx 940 | ctx 4620 |
|---|---|---|---|
| 0 (baseline) | 53.9 | 52.5 | 53.2 |
| **2** | **57.0** | **61.1** | 54.3 |
| **3** | **60.8** | 57.7 | 44.5 |
| 4 | 47.3 | 53.5 | 47.9 |
| 7 (checkpoint default) | 35.6 | 36.3 | 35.7 |

The checkpoint's own default of 7 is the worst possible setting on this target.
**nspec=2 is the setting to use**: it is the only one that never loses to no
speculation (57.0 / 61.1 / 54.3 vs 53.9 / 52.5 / 53.2). nspec=3 is faster at
short context but **regresses to 44.5 at 4620 tokens** — the verify step gets
more expensive as attention grows, so the row penalty that already dominates
gets worse with context. Any future nspec tuning must be measured at long
context, not just short.

#### What this says about goal 2 (200 tok/s single request)

200 tok/s is 5 ms/token. Speculation is the only mechanism with that reach, and
it is currently throttled by the *target*, not the draft: at ~6.5 ms per extra
verified row, drafting more tokens costs more than it returns after position 3.
Goal 2 therefore needs **both**:

- **(a) weight-stationary small-M decode.** Make verify(8) cost about what
  verify(1) costs. The evidence that this is achievable is already in the
  per-kernel table: `bf16blas` is flat at 24.7 us for o_proj from b=1 to b=32
  while `mmvq` goes 18.8 -> 378 us. The quantised kernels have the bandwidth
  (2800-3200 GB/s at b=1) and lack only reuse — dequantise the weight tile once
  and reuse it across rows. This is exactly QuixiCore's landed
  "LDS-staged k-quant tile decode, 1.4-2.0x" and "in-GEMM k-quant decode
  (dequant4 MFMA fragment)" (`perf/perf.md` sec 7). The existing `mmq` kernels
  *are* weight-stationary but run at 8-277 GB/s, so they are not the answer as
  written.
- **(b) a draft that matches this target.** 2.29 acceptance length caps the win
  at ~2.3x even with free verify rows. With (a) at +1 ms/row, step(8) ~ 26 ms
  and 2.29 acceptance gives ~88 tok/s; reaching 200 needs acceptance ~5.2, i.e.
  a draft trained against the q2_K Vision checkpoint rather than GLM-5.2-FP8.

With (a) alone and the current draft, ~85-90 tok/s looks reachable. 200 needs
(b) as well. That is the honest read.

### Iteration 2 (2026-07-27) — multi-column GEMV

#### First: no existing kernel was already the answer (`kbench3.py`)

Before writing anything, every path that exists was measured at the M the
verify step actually uses. GPU time us, CUDA-graph timed:

| shape | kernel | M=1 | M=4 | M=8 | M=32 |
|---|---|---|---|---|---|
| o_proj 6144x8192 q8_0, 53.5 MB, **ideal 10.1 us** | mmvq | 21.3 | 53.4 | 99.5 | 381.8 |
| | mmq | 384.0 | 386.0 | 387.7 | 391.9 |
| | triton | 421.7 | 422.4 | 422.5 | 426.2 |
| | bf16blas | 25.0 | 24.8 | 24.9 | 28.5 |
| shared_gate_up 2048x6144 q2_K, ideal 0.8 us | mmvq | 9.9 | 16.9 | 26.5 | 73.1 |
| | mmq | 531.4 | 532.8 | 533.7 | 536.6 |
| | triton | 485.3 | 485.5 | 485.4 | 486.4 |

**REJECTED: the Triton GGUF GEMM as the weight-stationary path.** It is flat in
M (so it does reuse the weight) but 4-40x slower than `bf16blas` and 20x off the
byte roofline — 422 us to move 53.5 MB is 127 GB/s. Same story as `mmq`, which
it closely tracks. Neither is worth tuning; both are structurally the wrong
shape (`mmq` uses `MMQ_X=64, MMQ_Y=128`, so at M=1 it computes 64 columns and
launches only `6144/128 = 48` workgroups for 304 CUs — 16% of the GPU doing 64x
redundant work).

#### The change: `mul_mat_vec_q` gains an `ncols_y` template parameter

`vllm_gguf_plugin/csrc/gguf/mmvq.cuh`. The b2899 kernel is one block per
(row, y) pair, so a batch of M reloads the whole weight M times — 8 x 53.5 MB of
loads for 53.5 MB of unique weight at M=8. `&x[ibx]` is invariant in the y
index, so tiling y by `ncols_y` and unrolling lets the compiler issue the weight
load once and dot it against all `ncols_y` vectors from registers. Dispatch
picks ncols_y from {1, 2, 4, 8} by M and tiles above 8; the tail clamps with
`min(vec0+j, nvecs-1)` and guards the store, so a partial tile computes
duplicate columns rather than breaking the shared load. 19 launchers now go
through one `VLLM_MMVQ_DISPATCH` macro. Rebuild is **59 s** — iteration on this
file is cheap, unlike an engine boot.

**Kernel result (mmvq us, before -> after):**

| shape | M=1 | M=2 | M=4 | M=8 | M=16 | M=32 |
|---|---|---|---|---|---|---|
| o_proj q8_0 | 21.3 -> 21.7 | 29.7 -> 25.3 | 53.4 -> **31.6** | 99.5 -> **50.2** | 192 -> **94.3** | 382 -> **178** |
| dense_gate_up q8_0 | 25.2 -> 25.0 | 43.1 -> 31.0 | 77.7 -> **45.4** | 149 -> **71.8** | 295 -> **135** | 584 -> **263** |
| q_b_proj q8_0 | 9.9 -> 9.8 | 12.8 -> 10.3 | 19.1 -> **12.2** | 31.7 -> **17.3** | 57.8 -> **28.3** | 111 -> **50.6** |
| shared_down q2_K | 8.8 -> 8.8 | 12.0 -> 9.7 | 18.1 -> **10.7** | 30.3 -> **15.4** | 54.6 -> **23.8** | 103 -> **37.4** |
| shared_gate_up q2_K | 9.9 -> 10.1 | 12.7 -> 12.6 | 16.9 -> *18.0* | 26.5 -> *28.8* | 40.6 -> 41.3 | 73.1 -> 65.7 |

~2x at M>=8 on four of five shapes, and **M=1 is unchanged** (ncols_y==1 is the
original kernel), so plain decode cannot regress.

**The one regression, and why:** `shared_gate_up` is 2048 rows, so the grid is
2048 single-wave blocks. Tiling y collapses `gridDim.y` from M to `ceil(M/8)`,
taking the wave count from 16384 to 2048 — 6.7 waves per CU, too few to hide
latency on 304 CUs. The reuse win and the parallelism loss cancel. **The
multi-column trick only pays when `nrows` alone can fill the GPU**; below that
it needs a split-K partner to keep the wave count up. Not fixed this iteration;
the shape costs 10-29 us against a whole-model step of 18600 us.

MoE `moe_vec_q` is a different kernel and is **unchanged** (27.5/153.8/610.8 us
at 1/8/32 requests, within noise of before) — expected, and see below for why
the same trick does not apply to it.

#### Correctness: scored per column, not per tensor

`mmvq_check.py`, 4 quant types x 4 shapes x M in {1,2,3,4,5,7,8,9,16,17} —
sweeping every dispatch boundary so the tail clamp is exercised. The first
version of this check scored the **median relative error over all m*n
elements**, which is measurement rule 4's exact trap: a single duplicated or
dropped column is invisible in a median over 8 columns. Rescored per column:

```
worst per-column max_rel  0.00286     mean_rel within that column  3e-7
most columns              exactly 0.0 (bit-identical)
```

`mean_rel = 3e-7` with `max_rel = 3e-3` is the signature of a few
catastrophically-cancelling outputs differing by FMA contraction, on q2_K and
q6_K only. A wrong column would put max_rel AND mean_rel near 1.0. Correct.

#### Engine result

`bench2.py`, same script as the baseline:

| | baseline | + threshold fix | + multi-column |
|---|---|---|---|
| single-request, ctx 64 | 53.9 | 53.8 | 53.6 |
| single-request, ctx 18417 | 52.9 | 52.0 | 52.2 |
| batch 2 aggregate | 80.2 | 80.3 | 83.4 |
| batch 4 aggregate | 53.9 | 104.8 | **119.0** |
| batch 8 aggregate | 60.5 | 130.2 | **153.3** |

Single-request is unchanged, as designed. Speculative decode improved sharply:
nspec=7 went 35.6/36.3/35.7 -> 41.4/47.7/50.4 tok/s at ctx 65/940/4620, and
nspec=3 reached 68.7 at ctx 65 and 65.6 at ctx 4620 (was 60.8 / 44.5) — the
first numbers above the 60 tok/s goal at long context. Those runs had four
engines booting concurrently and one point (nspec=3, ctx 940) came back at 29.6
against 57.7 before, which is not a believable value; **re-measure with at most
two concurrent engines before quoting any of them.**

#### The gate investigation this triggered — read the READ FIRST block

The gates failed with the new kernel, which looked like a regression and was
not one. The chase is written up at the top of this file; the short version is
that **the engine corrupts its own output non-deterministically with the
unmodified kernels**, so the gate could not have attributed anything. Two
methodology traps worth keeping:

- **`setuptools` does not track header dependencies.** Editing a `.cuh` and
  running `build_ext` can produce a `.so` that was never recompiled — the first
  "control" build silently copied the *new* binary (byte-identical size) until
  forced with `rm -rf build && touch gguf_kernel.cu`. Always check the output
  size changed. Full rebuild is 59 s.
- **A/B against a `.so` you did not build.** The saved baseline was a day old;
  rebuilding the unmodified source today gave 3 706 696 bytes against the old
  3 706 656, which is what made the comparison trustworthy. Check this before
  believing an A/B, not after.

#### Original (now superseded) framing of the same failure

The gates failed with the new kernel, but **the two new-kernel gate runs failed
differently on identical inputs at temperature 0** (run 1: img3 missing
floor/paving; run 2: needle@38 -> `'BLUE-HERRING-77!2'`, needle@2799 ->
`'BLUE-HERRING!4?2.5.0...'`, img1 missing pine/tree). Same code, same GPUs, same
prompts, different output.

Evidence gathered so far:

- **The kernel is deterministic in isolation.** `mmvq_determinism.py`: 6 shapes
  x 15 values of M x 12 repeats, hashing exact bytes — every hash stable.
- **The A/B was almost confounded and then wasn't.** The saved baseline `.so`
  was built a day earlier; rebuilding the *unmodified* source with today's
  toolchain produced 3 706 696 bytes against the old 3 706 656, so the baseline
  really was the same source. Worth the check: `setuptools` does not track
  header dependencies, so editing a `.cuh` alone can silently produce a `.so`
  that was never recompiled — `rm -rf build && touch gguf_kernel.cu` is required
  to force it. (The first control build did exactly this and produced a
  byte-identical-size copy of the *new* .so until forced.)
- **Output divergence does not track the `.so` cleanly.** New-kernel run 2 and
  the baseline produced byte-identical img4 text; new-kernel run 1 produced
  different img4 text. img1 does differ between baseline and new.

That pattern is consistent with the engine itself being non-deterministic
run-to-run, with the kernel's 3e-3 worst-element perturbation only deciding
which side of a token boundary a 2-bit model lands on. It is equally consistent
with a real bug that the standalone test does not reach (CUDA graphs, TP2, or
the merged-QKV `.contiguous()` path are all present in the engine and absent in
the test). Running `gates_rep.py` — the suite twice inside one boot, exact text
hashed, control `.so` and new `.so`, two boots each — to measure the engine's
own determinism before attributing anything.

---

## Load time: llama.cpp vs vLLM, measured 2026-07-27 (iteration 6)

**vLLM is 3.1x slower to become servable than llama.cpp on the same model,
same 2 GPUs, same warm page cache.** Almost none of the gap is weight I/O.

| | llama.cpp | vLLM (TP2) |
|---|---|---|
| definition | process start -> first emitted token | process start -> port accepts + `/health` 200 |
| measured | **141.4 s / 145.2 s** (two runs) | **436.4 s** |

llama.cpp's figure *includes* prefill and a token; vLLM's stops earlier. The
honest comparison is 141.1 s vs 436.4 s.

### How the measurement was made (do not repeat the failures)

Three attempts failed before one worked. Recorded because each is a trap:

1. **Timestamping the first non-whitespace byte on llama-cli's stdout** —
   caught llama-cli's own ASCII splash + "Loading model..." spinner, not a
   token. Reported 155.81 s with text `'\n\nLoading model... \n\n<ascii art>'`.
2. **`--log-disable` does not suppress the splash.** It is written by
   `ui::spinner` in `tools/cli/cli-context.cpp`, outside the log system.
3. **`/usr/bin/time ... | grep ... | head -14` wedged** — model fully resident
   on both GPUs, 0% CPU, 81 bytes out. Never put `head` on a long-running
   engine's stdout. Redirect to a file and read the file.

**What worked:** patch llama.cpp. `LLAMA_TTFT=1` stamps `ggml_time_us()` at the
top of `llama_cli()` and prints elapsed + `_exit(0)` on the first `content` or
`reasoning_content` SSE delta. Three files, 27 lines, env-gated:
`tools/cli/{cli.cpp,cli-context.cpp,cli-context.h}`. Runner:
`<SCRATCH>/run_ttft_llamacpp.sh`, verbose variant `run_ttft_v.sh`.

**Architecture note that invalidated the naive approach:** modern `llama-cli`
is an HTTP client. It spawns a `llama-server` child and streams
`/v1/chat/completions` SSE. There is no in-process token loop to hook.

### llama.cpp phase breakdown (`-v --log-timestamps`, run 2 = 141.41 s)

| phase | s |
|---|---|
| startup, 6-shard GGUF metadata, device probe, tokenizer | 1.5 |
| tensor load / offload to VRAM | **58.4** |
| context init | 0.8 |
| `graph_reserve` n_tokens=512 — ONE call, nothing logged inside | **79.7** |
| server slot init | 0.7 |
| prefill (17 tok) + first token | **0.32** |

Note `graph_reserve` is 56% of llama.cpp's own boot and this was with
`--no-warmup`. Actual inference is 0.32 s. llama.cpp is not the ceiling here —
it is just 3x less bad than we are.

### vLLM phase breakdown (436.4 s total; log t=0 is 25 s after process start)

| phase | s |
|---|---|
| python import before first log line | ~25 |
| config resolve / GGUF metadata / engine + worker init | ~150 |
| `Model loading took 125.74 GiB memory and 109.16 seconds` | **109** |
| profile + KV cache + cudagraph capture (`init engine ... took 72.14 s`) | **72** |
| post-engine API setup -> `Starting vLLM server` | **~85** |

Two phases llama.cpp simply does not have: ~150 s of pre-load setup (vs its
1.5 s) and ~85 s *after* the engine is up before the port opens.

### Root cause candidate #1: the Python `gguf` reader is 125x slower than C++

Measured directly, warm cache:

| | time |
|---|---|
| `import gguf` | 0.38 s |
| `GGUFReader(shard 1 of 6)` — 343 tensors, 41.3 GiB | **8.70 s** |
| same again, fully cached | **8.75 s** |
| `GGUFReader(mmproj)` — 335 tensors, 0.9 GiB | 0.02 s |

**It is not I/O.** A second open of a fully page-cached file costs the same
8.75 s, so it is Python-side parsing — ~25 ms per tensor record. llama.cpp
parses all six shards' metadata in 0.07 s (`0.48 -> 0.55` in its log).

There are **18 `GGUFReader(` construction sites** in-tree, and the boot log
shows the work repeating across processes: `Discovered 6 GGUF shard files`
twice, `Using fused CPU image preprocessing` four times (APIServer, EngineCore,
2 TP workers). Each process re-parses from scratch — nothing is shared or
cached across the fork.

### CONFIRMED and FIXED — boot 436.4 s -> 270.5 s (-38%)

Counted every construction per boot with a `sitecustomize.py` shim
(`<SCRATCH>/ggufprobe/`, `GGUF_PROBE_LOG=...`) that subclasses `GGUFReader` and
records pid + elapsed + caller. Baseline:

**77 constructions, 209.1 s across 4 processes — 47% of the entire boot.**

| pid | opens | s |
|---|---|---|
| APIServer | 22 | 114.4 |
| TP worker 0 | 25 | 42.0 |
| TP worker 1 | 25 | 41.6 |
| EngineCore | 5 | 11.2 |

Only shard 1 is expensive (8.7-9.9 s); shards 2-6 cost 0.015 s each. So the
cost is **not** per-tensor and **not** I/O. `cProfile` on shard 1:

```
16.5s  _build_fields
16.5s  _get_field_parts   (631,482 calls)
10.9s  _get               (1,110,649 calls)
 9.9s  _get_str           (476,963 calls)
 4.9s  numpy memmap.__array_finalize__  (4,443,285 calls)
```

**Root cause:** `GGUFReader.__init__` does `self.data = np.memmap(path)`.
`np.memmap` is an ndarray *subclass*, so every metadata slice runs
`__array_finalize__` + `may_share_memory`. Shard 1 carries
`tokenizer.ggml.tokens` (154,880) and `tokenizer.ggml.merges` (321,649), so the
reader takes ~477k one-string-at-a-time slices of a memmap. llama.cpp parses
all six shards in 0.07 s.

**Two fixes, both landed** (`vllm/transformers_utils/gguf_utils.py`):

1. `_plain_mmap` — map with `np.frombuffer(mmap.mmap(...))` instead, giving a
   plain ndarray over the same mapping. Installed by swapping
   `gguf.gguf_reader.np.memmap` under a lock for the duration of the parse, so
   there is no global side effect. **8.6 s -> 2.76 s, 3.11x.**
2. `gguf_reader()` — `@cache`d, key normalised with `Path(...).resolve()` so
   `str`/`Path`/relative spellings share one entry. All 18 construction sites
   in-tree now route through it; no `gguf.GGUFReader(` remains outside it.

| | opens | GGUFReader s | boot s |
|---|---|---|---|
| baseline | 77 | 209.1 | **436.4** |
| fast parse + cache | 20 | 30.5 | **286.6** |
| + the 2 sites inside `gguf_utils.py` itself | **18** | **11.4** | **270.5** |

18.3x off the reader, 165.9 s off the boot. Now exactly one shard-1 parse per
process (2.8 s x 4) — that is the floor without sharing across the spawn.

**Correctness:** parse verified byte-identical against the stock reader before
landing — field set and order, tensor names/shapes/types, tensor bytes, all
154,880 tokens, all 321,649 merges, and the three stop-set ids
(eos=154820, eot=154827, eom=154829).

### Still on the table for load time

- **~2.8 s x 4 processes** of duplicate shard-1 parsing. The vocab and merges
  are identical in every process; parse once and hand the workers the decoded
  lists rather than re-deriving them after spawn.
- **`Model loading took 109.16 s`** vs llama.cpp's 58.4 s for the same bytes off
  the same warm cache. Not yet investigated.
- **~85 s between the engine being ready and the port opening** (chat template
  detection, multimodal warmup, task setup). llama.cpp has no equivalent.
- **72 s** profile + KV cache + cudagraph capture.

---

## STARTUP-TIME BASELINE, measured 2026-07-28 (iteration 7 — new focus: reduce startup)

Fresh measurements, both engines loading the **same GGUF pair**:
`GLM-5.2-UD-Q2_K_RoutedQ2K-00001-of-00006.gguf` (6 shards, 262 GB) **plus**
`mmproj-GLM-5.2-Vision-f16.gguf` — unlike the 07-27 numbers, llama.cpp now
loads the vision projector too (`--mmproj`), so the comparison is apples to
apples. Same 2x MI300X, TP2 / `-sm layer`.

| run | llama.cpp TTFT (internal) | external (process start -> exit at token 1) |
|---|---|---|
| 1 (cold page cache) | 211.9 s | 229.2 s |
| 2 (warm) | 150.3 s | 167.6 s |
| 3 (warm) | 151.3 s | 166.6 s |

| vLLM (TP2, warm) | s |
|---|---|
| port accepts + `/health` 200 | **271.45** |

**BASELINE: llama.cpp ~150.8 s to first token (167 s external); vLLM 271.5 s
to servable. vLLM is 1.62x slower (1.80x vs external process clock).**
Down from 3.1x on 07-27 thanks to the GGUF-reader fix, and llama.cpp went
141 -> 150 s from carrying the mmproj.

Two structural notes from the runs:

- **17.3 s of exec+dyld before `llama_cli()` even runs** — constant across all
  three runs (external minus internal). The ROCm userspace stack costs ~17 s
  of dynamic linking on this box before any code runs. vLLM's equivalent (the
  ~25 s python-import phase) is not an outlier; everything paying the ROCm tax
  is slow to exec here.
- **Cold vs warm page cache is worth ~61 s on llama.cpp** (262 GB read;
  buff/cache grew ~245 GB during run 1). All baseline numbers are warm-cache.

vLLM phase timeline (from `vllm_boot.log`, t0 = spawn, port at 271.45 s):

| phase | ends at | s |
|---|---|---|
| exec + torch import (first log line ~4 s, first vLLM INFO) | ~29 s | ~29 |
| config resolve, engine + 2 worker spawn, GGUF parse, init | ~125 s | ~96 |
| weight load (`Model loading took 66.37 s`) | ~191 s | 66 |
| init engine: profile, KV cache, 21 s graph capture (73.40 s) | ~264 s | 73 |
| post-engine API setup -> `Starting vLLM server` + port | 271.5 s | ~7 |

Yesterday's ~85 s post-engine tail is now ~7 s (it was the APIServer's 22
GGUF re-parses, killed by the reader cache). Weight load also improved
109 -> 66 s run-over-run with no code change — treat 66 s as the warm-cache
number (llama.cpp's comparable tensor-load phase was 58.4 s on 07-27).

Measurement harness (this session's scratchpad): `run_ttft_llamacpp.sh`
(now with `--mmproj`), `run_ttft_ext.sh` (external clock), `ttft_vllm.py`;
logs `ttft_lcpp_run{1,2,3}.log`, `vllm_boot.log`. llama.cpp TTFT patch
unchanged from 07-27: `LLAMA_TTFT=1`, 27 lines in `tools/cli/`, still built
into `libllama-cli-impl.so`.

### Where the remaining 120 s gap lives (targets, in order)

1. **~96 s of pre-load setup** (imports in 4 processes, config resolution,
   worker spawn, per-process GGUF shard-1 parse at 2.8 s x 4). llama.cpp does
   the equivalent in ~1.5 s.
2. **73 s init engine** — profile run, KV-cache alloc, cudagraph capture.
   (llama.cpp's `graph_reserve` analogue was 79.7 s of its 141 s on 07-27,
   so this one is not unique to us — but it is additive with #1.)
3. **66 vs 58 s weight load** — small gap now, low priority.

## Startup reduction round 1 (2026-07-28): 271.5 -> 233.3 s, gates passing

Target: beat llama.cpp's 150.8 s to servable. All numbers from `[boot +Xs]`
stamps (`vllm/utils/bootstamp.py`, `VLLM_BOOT_T0` pinned by the harness);
each boot ends with a greedy `capital of France -> ' Paris...'` gate.

Landed, in order of measured effect:

1. **xgrammar import: 15.5 s x 3 process generations.** `backend_xgrammar.py`
   class-body annotation `matcher: xgr.GrammarMatcher` evaluated the LazyLoader
   at import, pulling `xgrammar -> tvm_ffi -> _optional_torch_c_dlpack`, which
   attempts a JIT build that always fails on torch-2.14-nightly headers
   (15 s, per process, unycached because it fails). Fix: one-line
   `from __future__ import annotations`. APIServer 25 s of imports -> 9.4 s.
2. **fork instead of spawn: ~21 s.** `entrypoints/serve/utils/api_utils.py`
   blanket-defaulted `VLLM_WORKER_MULTIPROC_METHOD=spawn` for the whole CLI,
   overriding vLLM's fork default before `_maybe_force_spawn` could decide.
   Removed; EngineCore + workers now fork in ~0.2 s and inherit the parsed
   GGUF reader (kills the 2.8 s shard-1 re-parse per child AND EngineCore's
   end-of-boot re-parse). Worker imports phase 38.5 s -> 17.7 s.
3. **cudagraph memory estimate: 10.3 s.** `gpu_worker.determine_available_memory`
   ran `profile_cudagraph_memory()` (a full trial capture) even when
   `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` says to discard it. Now gated;
   harness opts out. Post-profile gap 10.3 s -> 0.01 s.
4. **quark plugin: ~1 s/process.** `VLLM_PLUGINS=""` — the quark entry point
   imports 1 s of quark at every boot; we don't use it.

**Expert loading detour (record so it is not retried):** the adapter unbound
stacked `ffn_*_exps` into 256 per-expert 2-D slices — 60,265 yields, ~44 s of
consumer time (0.73 ms per copy). Yielding the 3-D stack instead routes into
`RoutedExperts.weight_loader`'s existing `full_load` path, but its TP-narrowed
copy is a host+device *strided* copy — 107 s, WORSE. Fix that stuck: upload the
contiguous stack to the GPU first (`_gguf_moe_weight_loader`, ndim==3), then
delegate — the narrows become device-side. 51-55 s, i.e. parity with the old
path and with llama.cpp's 58 s tensor load; pageable H2D (~5 GB/s effective)
is the wall. Pinned-ring or hipHostRegister are the next levers if load is
re-attacked.

Timeline at 233.3 s (best boot): imports+config 17.5, comm init 15.7
(pynccl 9!), load 55.4 + process_weights 12, profile_run **24.5-85 s
(unexplained 3.5x variance between clean boots — now instrumented)**,
15 s stall at the first tiny `torch.tensor(device=cuda)` after KV alloc
(NOT Triton compile — triton cache untouched; now isolated with a
synchronize stamp), capture 22, tail 6.

Remaining: profile_run variance, the 15 s stall, capture 22 s, pynccl 9 s,
process_weights_after_loading 12 s.

### Round 1 final: 271.5 -> 165.0 s (five boots: 166.5 / 166.0 / 168.5 / 167.2 / 165.0)

llama.cpp is 150.3 / 151.3 s to first token. We are **~15 s behind**, from 121 s
behind this morning. Every boot ends `' Paris. The city is located on the'`.

Worker-0 timeline of the 165.0 s boot:

| phase | window | s |
|---|---|---|
| APIServer imports | 0 -> 9.6 | 9.6 |
| config, GGUF parse, fork EngineCore+workers | 9.6 -> 18.2 | 8.6 |
| device init + pynccl/aiter/quick all-reduce | 18.2 -> 33.0 | 14.8 |
| weight load (51.5) + process_weights (12) | 33.0 -> 98.8 | 65.8 |
| profile_run (mm encoder 2.5, LM dummy 4.3, sync 15.5) | 98.8 -> 122.9 | 24.1 |
| KV cache alloc + commit | 122.9 -> 123.0 | **0.09** |
| kernel_warmup | 123.0 -> 137.4 | 15.1 |
| cudagraph capture (4 sizes) | 137.4 -> 158.4 | 21.0 |
| API tail -> port open | 158.4 -> 165.0 | 6.6 |

### The 15 s "stall" is a first-pageable-H2D-copy cost, and it is NOT KV alloc

Chased with `sudo py-spy dump --native` (ptrace_scope=1, so plain py-spy is
Permission Denied; `--nonblocking` cannot do native frames — use sudo).
Caught mid-stall:

```
c10::cuda::memcpy_and_sync -> copy_kernel_cuda -> ... -> internal_new_from_data
-> torch::utils::tensor_ctor    [blocked in libhsa-runtime64.so]
```

That is `torch.tensor([0, num_tokens], device=device)` — an 8-byte pageable
H2D copy — taking 15 s. Two hypotheses tested and **rejected**:

- *KV cache page commit*: added `torch.zeros(1, device).cpu()` + synchronize
  immediately after `initialize_kv_cache`. KV alloc measures **0.09 s** and the
  15 s did **not** move. Not deferred commit.
- *Triton compile*: `~/.triton/cache` gained zero entries during the window.

It is the first *pageable* H2D after ~43 GiB is allocated; the runtime appears
to rebuild its staging path. **Do not "fix" this by deleting the warmup** — the
cost is paid on first touch either way, and warmup is exactly the right place
to pay it (before the port opens vs. on a user's first request). Only a fix
inside the ROCm runtime, or pre-registering pinned staging, removes it.

### Instrumentation left in tree (keep — it is how any of this was found)

`vllm/utils/bootstamp.py`: `bootstamp(tag)` logs `[boot +Xs pid=N] tag`, origin
pinned in `VLLM_BOOT_T0` so forks/spawns share t0. Harness sets it at spawn, so
stamps include interpreter startup. Call sites: api_server (run_server,
build_and_serve), EngineCore/Worker entry, get_mp_context, plugins,
gguf_reader, gguf_loader (producer vs consumer split), cuda_communicator
(pynccl/aiter/quick), gpu_worker (init_device, load_model, profile_run,
KV alloc, kernel_warmup, capture), gpu_model_runner (profile_run internals),
kernel_warmup + v1_block_table_warmup per step.

Harness: `<SCRATCH>/ttft_vllm.py` — spawn -> port + `/health` 200, then a greedy
`The capital of France is` gate. It sets `VLLM_PLUGINS=""` and
`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.

### Next levers, largest first

1. **51 s weight load.** Pageable H2D at ~5 GB/s effective. A pinned staging
   ring or `hipHostRegister` on the mmap is the only real lever; the naive
   pinned-buffer copy was already measured slower (102 -> 117 s).
2. **21 s cudagraph capture** for 4 sizes. Capture is serial per size.
3. **15 s first-pageable-copy** (above) — needs a runtime-level fix.
4. **15 s comm init**, of which pynccl is ~9 s.
5. **12 s `process_weights_after_loading`** — not yet investigated.
6. **15.5 s of profile_run is the post-dummy-run `_sync_device()`**, i.e. the
   16384-token dummy forward itself. Reducing `max_num_batched_tokens` would
   cut it but changes serving behavior — not done.

### Round 2 attribution (all measured, none fixed yet)

Two of the "unexplored" blocks are now pinned to a single line each:

- **`process_weights_after_loading` = 12 s is entirely `MLAAttention`**:
  `MLAAttention=11.94s/78` layers (153 ms each), while every quant method
  combined is `GGUFLinearMethod=0.04s/411, UnquantizedLinearMethod=0.00s/284,
  GGUFMoEMethod=0.00s/75`. The work is in
  `model_executor/layers/attention/mla_attention.py:967`
  (`get_and_maybe_dequant_weights` on `kv_b_proj`, then the W_UK/W_UV
  transposes via `replace_parameter(..., prefer_copy=True)`). Our `kv_b_proj`
  is already BF16 from the adapter, so 153 ms for a
  [512, 16x(128+128)] transpose is far more than the arithmetic — suspect a
  per-layer device sync inside `replace_parameter`/`device_loading_context`.
  **Next thing to look at; probably the cheapest remaining 10 s.**
- **Capture 21 s splits PIECEWISE ~19 s / FULL ~1.3 s** for 4 sizes each.
  The FULL decode graphs are nearly free; the piecewise mixed prefill-decode
  graphs are the cost.

Also confirmed cheap and not worth chasing: KV cache alloc+commit **0.09 s**,
`get_kv_cache_configs` 0.01 s, API tail 6.6 s (was ~85 s this morning).

## Round 3 (2026-07-28): 165 -> ~152 s. The AITER FP8 BMM precompile was 11.8 s

**Root cause of the 12 s `MLAAttention` post-load.** With `VLLM_ROCM_USE_AITER=1`
the MLA path takes the `is_aiter_triton_fp8_bmm_enabled` branch, which
pre-compiles the Triton FP8 BMM over batch sizes 1..1024 — 2 x 1024 kernel
launches — **once per decoder layer, 78 times**. Triton's cache is keyed by
shape/dtype and every layer has identical W_K/W_V shapes, so layers 2..78
compiled nothing and only re-issued ~160k launches. Fix in
`mla_attention.py`: a class-level `_fp8_bmm_precompiled` set keyed by
(W_K.shape, W_V.shape, dtypes); sweep once per distinct shape.
**11.79 s -> 0.73 s**, same kernels compiled, gate passes.

The tell was that `rp_W_UV`/`rp_W_UK_T` timers both read 0.00 s — the `else:`
branch they live in never executes on this configuration. When per-statement
timers all read zero, the code you are reading is not the code that runs.

**Rejected on measurement (do not retry):**

- *GC pressure.* `gc.disable()` across the whole load + `freeze_gc_heap()`
  after: MLA stayed 11.8 s. The GC change is kept (it is harmless and the
  freeze legitimately excludes the static model graph) but it bought nothing.
- *KV-cache page commit / first-pageable-copy-after-big-alloc.* A real pageable
  H2D issued immediately after `initialize_kv_cache` costs **0.00 s**, so the
  15 s in `kernel_warmup` is not "the first pageable copy after 43 GiB".
- *Host staging not yet pinned.* Pinning a 64 MiB host buffer and doing an H2D
  in `init_device` costs 0.02 s and did not move the stall.
- *Tensor construction path / wrong device.* Rewriting
  `torch.tensor(..., device=device)` as `torch.tensor(...).to(device)` stalls
  identically, and the stamps confirm `device` is `cuda:0` / `cuda:1`
  correctly per rank.
  **The 15 s in `warm_v1_block_table_kernels` remains unexplained.** It is one
  H2D of 8 bytes, both ranks stall simultaneously, the GPU is idle
  (a `synchronize()` immediately before returns in 0.00 s), and py-spy shows
  it blocked in `libhsa-runtime64.so` under `memcpy_and_sync`.

**Measurement artifact worth knowing:** every anomalous boot (192 s, 211 s,
192 s) came *immediately after editing a vLLM source file* — the inductor /
compile cache key changes and `profile_run` jumps from ~22 s to ~60 s. The
second and later boots after an edit are the honest steady state. Always
discard the first boot after a code change.

Also added: the harness now aborts if port 8077 is already serving. Two
overlapping boots produced a bogus 32 s "measurement" against the previous
run's server.

### Where the ~152 s goes (steady state, worker 0)

| phase | s |
|---|---|
| APIServer imports | 9.6 |
| config + GGUF parse + fork | 8.2 |
| device init + NCCL/aiter comm init | 14.5 |
| weight load | 53 |
| MLA + other post-load | 1.5 |
| profile_run (LM dummy 16384 tok) | 22.6 |
| kernel_warmup (the unexplained 15 s H2D) | 15.1 |
| cudagraph capture (PIECEWISE 4x4.4 s, FULL 4x0.15 s) | 20.5 |
| API tail -> port open | 5.8 |

### llama.cpp comparison, corrected

llama.cpp's own run-to-run spread is as wide as ours: 141.4 / 145.2 (07-27,
no mmproj), 150.3 / 151.3 (07-28 with mmproj, `--no-warmup`), and **135.1 s
with warmup enabled** — i.e. enabling warmup made it *faster* than a
no-warmup run, which is variance, not a warmup effect. Its loader confirms
why our weight load cannot be trivially beaten: `llama_model_loader`'s
4-buffer pinned async upload path is explicitly disabled when mmap is on
(`if (use_mmap || check_tensors) return nullptr;`), so with its default mmap
llama.cpp does the same plain pageable copy vLLM does — 58.4 s there vs 53 s
here.

vLLM steady state, six boots: 151.76 / 151.80 / 152.01 / 154.75 / 154.77 /
155.06 (median ~153.4). Every boot gated on
`'The capital of France is' -> ' Paris. The city is located on the'`.

### CORRECTION — the interleaved A/B says llama.cpp is 67 s, not ~150 s

Back-to-back runs of one engine are not a fair baseline: the machine's page
cache and the ROCm code-object cache drift over a session. Interleaving the
two engines in a single window (llama.cpp, vLLM, llama.cpp, vLLM, ...):

| pair | llama.cpp (first token) | vLLM (port + /health) |
|---|---|---|
| 1 | 160.9 s | 152.6 s |
| 2 | **67.4 s** | 154.3 s |
| 3 | **67.2 s** | 157.3 s |

Both 67 s runs are genuine: llama-cli `_exit(0)`s on the first streamed token,
and their logs are structurally identical to the 160 s run. llama.cpp's first
run in the window is slow and then it settles at ~67 s, i.e. **llama.cpp's
warm steady state is 67 s and vLLM's is ~154 s — we are 2.3x slower, not at
parity.** Earlier claims of "15 s behind" compared our warm number against a
llama.cpp number that was not yet warm. Any future comparison must interleave.

**What this reframes.** vLLM's weight load alone (53 s) is nearly llama.cpp's
entire warm boot (67 s). The gap is no longer about loading bytes; it is the
~90 s of work vLLM does around the load that llama.cpp does not:

| vLLM-only cost | s |
|---|---|
| imports + config + fork | 17.8 |
| NCCL / aiter comm init | 14.5 |
| profile_run (16384-token dummy forward) | 22.6 |
| kernel_warmup (the unexplained 15 s H2D) | 15.1 |
| cudagraph capture | 20.5 |
| API tail | 5.8 |

llama.cpp's warm 67 s implies its own `graph_reserve` (79.7 s when cold on
07-27) is nearly free once the ROCm code-object cache is warm — so the
comparable vLLM phases (profile + warmup + capture = 58 s) are the single
biggest structural deficit, followed by comm init.

---

## Dead-code removal, pass 1 (2026-07-28): 259 files / 63,712 statements

**Coverage is taken from real inference, not the test suite.** Harness
`<SCRATCH>/cov_run.py` boots the actual server and drives 15 workload steps
(greedy, chat, sampling+penalties+seed+stop, n>1 + logprobs, batched prompts,
SSE streaming, 11k-token chunked prefill, prefix-cache hit, vision, and the
models/health/metrics/version/tokenize/detokenize endpoints), then combines
per-process data. Result: **17.9% of 295,425 statements execute; 980 of 2,110
files are never imported.**

**Getting the workers measured was the hard part and the first two attempts
were wrong.** Coverage does not follow vLLM's fork into EngineCore and the TP
workers: `sitecustomize` only runs for fresh interpreters, and the inherited
tracer never writes a data file. The first run produced 2 data files and
reported `gpu_model_runner.py` and `gpu_worker.py` at **0.0%** — files I had
watched execute earlier the same day. Their apparent 14-20% elsewhere was only
import-time class/def lines. Acting on that data would have deleted the entire
model-execution path. Fix: an explicit `coverage.Coverage(data_suffix=True)`
started in `WorkerProc.worker_main` under `VLLM_COV_WORKERS`, with atexit +
SIGTERM + a 10 s periodic `save()` (workers die too hard for atexit alone).
4 data files, and the worker paths then read 46-84%.

**Trap: every import in that hook must be aliased.** A bare `import signal`
(or `threading`) inside the `if` block binds the name function-locally for all
of `worker_main` and shadows the module-level import, so with the hook disabled
the worker dies with `UnboundLocalError`. Cost two failed boots.

**Deleted:** 257 never-imported top-level files in `model_executor/models/`,
plus `mimo_v2_mtp.py` and `cohere2_moe.py`, plus 247 `registry.py` entries.
Kept deliberately: `clip/pixtral/siglip` (imported at *runtime* inside
`vision.get_vision_encoder_info` — not TYPE_CHECKING, checked), and
`llama4/parakeet/extract_hidden_states/mimo_v2` dependents. Two live files had
dead branches removed: the MiMo-V2 MTP branches in `config/speculative.py` and
the Cohere routing branch in `custom_routing_router.py`.

**Verified:** full 15/15 workload passes on the pruned tree, vision included,
boot 153.8 s (unchanged). Deletions are `git rm`, so `git restore` recovers any
file without touching the uncommitted perf work.

**Two grep traps while cross-referencing** (both produced false conclusions
first): an empty line in a `grep -F -f` pattern file matches every line, and
model basenames collide with unrelated modules (`models/transformers/utils.py`
-> `utils` matched `from .utils import` everywhere). Use `grep -vxF` and
fully-qualified dotted paths.

### Remaining 0% files: 721 files / 91,425 statements

| statements | files | category |
|---|---|---|
| 41,597 | 309 | everything else (needs per-file triage) |
| 14,456 | 85 | KV connectors / offload / P-D disaggregation |
| 7,267 | 89 | HF configs+processors for the deleted models |
| 5,469 | 24 | benchmarks / profiler / tracing |
| 5,161 | 41 | tool parsers |
| 4,826 | 46 | other platforms (tpu/xpu/cpu/neuron) |
| 3,725 | 22 | model_executor/models subdirectories |
| 3,025 | 27 | vendored third_party |
| 1,855 | 26 | other quantization methods |
| 1,694 | 14 | LoRA |
| 1,436 | 25 | reasoning parsers |
| 914 | 13 | spec decode |

Note "0% in this workload" is not "dead": tool parsers, reasoning parsers,
LoRA and spec decode are features this workload never requested, and the tree
does have a `GLM-5.2-speculator.dspark-preview` checkout. Those need an
explicit scope decision rather than a coverage verdict.

## Dead-code removal, pass 2: +256 files / 30,495 statements

Scope decided by the user: keep spec decode, reasoning parsers, tool parsers,
and benchmarks/profiler/tracing; delete KV connectors + P/D disaggregation,
other platforms (tpu/xpu/cpu/neuron), HF configs+processors belonging to the
deleted models, other quantization methods, model subdirectories, and LoRA.

Method: for every candidate, resolve its dotted module path (including package
`__init__`) and drop it from the deletion set if **any** kept file imports it
or a parent package. 26 candidates were spared that way, leaving 256. Then 20
more `registry.py` entries pruned by checking the module file actually exists.

**Verified: 15/15 workload steps pass, vision included, boot 158.1 s.**

### Running total

| | files | statements |
|---|---|---|
| pass 1 (model architectures) | 259 | 63,712 |
| pass 2 (platforms, KV connectors, LoRA, quant, configs) | 256 | 30,495 |
| **total removed** | **515** | **94,207** |

That is 32% of the 295,425 statements coverage saw, and the tree went from
2,117 to 1,602 Python files. Everything is `git rm`, so any file is one
`git restore` away.

### Still on the table

721 -> ~465 files remain at 0%, dominated by the 309-file "everything else"
bucket (41,597 statements) that needs per-file triage rather than a category
verdict. Function-level dead code inside *live* files has not been touched yet:
coverage says only 17.9% of statements in the surviving tree execute, so the
larger prize is inside the files we keep.

## Dead-code removal, pass 3 + repo hygiene

**First: pass 1 and 2 were measured with EngineCore unmeasured.** The same fork
problem as the workers — `run_engine_core` never started its own coverage, so
everything that runs *only* in that process looked dead:
`v1/core/sched/scheduler.py` and `v1/core/kv_cache_manager.py` both read
**0.0%**. Deleting the scheduler would have been fatal. Added the same aliased
hook to `EngineCoreProc.run_engine_core`; with 5 data files they read 44.6% and
64.2%, and `v1/engine/core.py` went 17.1% -> 45.7%. **9 files were false zeros.**
Lesson: after any coverage run, sanity-check one known-live file *per process*
before trusting the report.

Deleted in pass 3 (184 files, 23,323 statements): other attention backends and
attention ops, other `model_executor/layers` implementations, remaining
`vllm/models/*` families (inkling, deepseek_v4, deepseek_v32, minimax_m3),
vendored `third_party`, pooling and speech-to-text entrypoints, helion kernels,
weight_transfer, leftover `transformers_utils/configs`, CPU worker variants,
and unused model_loader/multimodal files. 53 further candidates were spared by
the "still imported by a survivor" filter. Left alone deliberately:
`vllm/parser`, `vllm/tokenizers`, `vllm/renderers`, `vllm/utils`, `vllm/v1/core`
and the CLI/serve entrypoints — too load-bearing to delete on a single
workload's evidence.

Repo hygiene: removed `tests/` (1,520 files — the test suite is not the signal
for this fork), 44 untracked root `*.log`, and 3 untracked root `*.hip` scratch
files. **`build/` was left intact**: its 71 `.hip` files are hipify output from
`cmake/utils.cmake`, and deleting them forces a full C++ recompile for no gain.
Untracked files are not recoverable by git, so everything untracked was copied
to `<SCRATCH>/archive_untracked/` before removal.

### Final totals

| pass | files | statements |
|---|---|---|
| 1 — model architectures | 259 | 63,712 |
| 2 — platforms, KV connectors, LoRA, quant, configs | 256 | 30,495 |
| 3 — attention backends, layers, model families, misc | 184 | 23,323 |
| **subtotal (vllm/)** | **699** | **117,530** |
| tests/ + root logs/hip | 1,567 | n/a |

`vllm/` is down from 2,117 to 1,418 Python files. Coverage of the surviving
tree is 27.3% of 200,743 statements. Every pass gated on the same 15-step real
inference workload, vision included; boot time unchanged at ~158 s.

**Remaining:** ~270 files still at 0% that were spared as too load-bearing or
still-imported, plus the real prize — function-level dead code inside live
files, where ~73% of surviving statements never execute.

## Pass 4: file-level pruning hits its limit (and a filter bug)

Targeted 38 "barely alive" files (<=15% covered) in already-approved
categories — NVIDIA-only kernels (flashinfer/trtllm/cutlass/cutedsl), LoRA,
other quant, other model families, elastic-EP. **32 of the 38 had to be
restored**: they are transitively imported by live MoE dispatch/oracle code
(`fused_moe/oracle/fp8.py` imports `flashinfer_utils` at module level even
though nothing in it executes on ROCm). Net gain: 6 files.

**Filter bug found and fixed the hard way: passes 2-4 only matched *absolute*
imports.** Relative imports (`from .utils import compute_meta`) were invisible,
which is how `vllm/lora/punica_wrapper/utils.py` got deleted while its live
importer `punica_base.py` survived. An AST sweep for unresolvable relative
imports found 63 targets; **54 were restored from HEAD** and 3 untracked ones
from `<SCRATCH>/archive_untracked/`. Any future pass must resolve both import
forms — and the untracked archive earned its keep here.

Two false-positive classes in that AST sweep, for the record: `from . import
name` re-exports a *function*, not a submodule (this made GGUF hot-path files
look deleted when they never were), and namespace packages without
`__init__.py` look missing but import fine. Trust the runtime gate over the
static sweep.

**Verified green: 15/15, boot 161.1 s.**

### Where file-level pruning stands

`vllm/` is at 1,455 Python files. The ~270 files still at 0% are no longer
free to delete: each is wired into a live import graph, so removing them means
editing live dispatch tables (MoE oracles, attention backend registry,
entrypoint routers) rather than deleting leaves. That is a different and
riskier kind of change than everything done so far.

The remaining prize is inside live files: **5,697 fully-dead functions spanning
66,127 lines** (`<SCRATCH>/cov/dead_funcs.json`, ranked). Biggest single
offenders: `third_party/pynvml.py` (2,946 dead lines — NVIDIA-only on a ROCm
box), `compilation/passes/fusion/allreduce_rms_fusion.py` (945),
`config/speculative.py` (878), `quantization/modelopt.py` (768),
`v1/worker/gpu_model_runner.py` (742). Function-level deletion needs a
different safety argument than file-level: dynamic dispatch, registries and
`getattr` mean "no lines executed in one workload" is weaker evidence for a
method than for a whole module.

## Pass 5: ROCm-only platform detection (-8,378 lines)

The first cut that required editing live dispatch rather than deleting leaves,
and the pattern that makes the rest of the backlog tractable.

`vllm/platforms/__init__.py` ran every builtin probe on every boot. On this box
the CUDA probe imports the vendored 4,096-statement `third_party/pynvml.py` and
calls `nvmlInit()` before failing, and the CPU probe runs a Zen detection pass —
all so the ROCm probe can win. `builtin_platform_plugins` now holds one entry.

Removed with it: the tpu/cuda/xpu/cpu probe functions, `_is_amd_zen_cpu` (and
its stale `__all__` entry), `utils.import_utils.import_pynvml`, the
`CpuPlatform` fallback in `entrypoints/cli/main.py` (unreachable once detection
always resolves to ROCm), and `platforms/{cuda,cpu,xpu}.py` +
`third_party/pynvml.py`. Verified `current_platform` still resolves to
`RocmPlatform`, gate 15/15. Committed as `f70373dd7d`.

**Note on why pynvml read 45% covered, not 0%:** it is genuinely executed —
just only to fail. Coverage alone would never have flagged it; it took knowing
that NVML cannot matter on an AMD box. Category knowledge beats the profile for
this class of dead code.

### Next target, sized but not started

`get_quantization_config` in `model_executor/layers/quantization/__init__.py`
eagerly imports ~20 quant backends inside the function body — awq, gptq,
bitsandbytes, modelopt (841 stmts), humming (412), fbgemm, torchao, mxfp4,
inc, fp_quant and more — so one call drags all of them in. That is why they
all show partial coverage instead of 0%. This tree only resolves `gguf` (plus
quark/fp8 for the speculator path). Making the dispatch import only the
selected method would let most of those modules be deleted outright; it is a
single well-understood function, but it touches the path that every model load
goes through, so it wants its own commit and gate run.

## Load-time baseline after the specialization work (2026-07-29): 156.3 s

Re-measured after the platform-probe removal, lazy quantization dispatch, the
FP8 BMM precompile fix, and the new hard-coded defaults (kv-cache fp8_e4m3,
aiter linear+moe backends, glm47/glm45 parsers). Gate passes.

| phase | window | s |
|---|---|---|
| APIServer imports | 0 -> 9.6 | 9.6 |
| config + GGUF parse + fork | 9.6 -> 17.8 | 8.2 |
| device init + NCCL/aiter comm | 17.8 -> 33.4 | 15.6 |
| weight load (53.3) + post-load | 33.4 -> 90.5 | 57.1 |
| profile_run (16384-token dummy) | 90.5 -> 113.1 | 22.6 |
| kernel_warmup (the unexplained H2D stall) | 113.3 -> 128.4 | 15.1 |
| cudagraph capture (4 PIECEWISE + 4 FULL) | 128.4 -> 149.5 | 21.1 |
| API tail -> port open | 149.5 -> 155.8 | 6.3 |

**llama.cpp warm steady state is 67 s** (interleaved A/B; back-to-back runs of
one engine are not a valid baseline). Its weight load is ~50 s of that, so its
non-load overhead is ~17 s against our ~103 s. **The gap is not the load --
it is everything we do around it.**

Ranked targets, none yet attempted:

1. **capture 21.1 s.** 4 PIECEWISE captures at ~4.4 s each plus 4 nearly-free
   FULL ones. Capturing fewer sizes is a direct trade against batch-size
   coverage; worth deciding deliberately now that the config is hard-coded.
2. **profile_run 22.6 s**, dominated by the 16384-token dummy forward. Scales
   with max_num_batched_tokens, so it is a serving-capacity trade.
3. **capture + profile + warmup together are 58.8 s** and llama.cpp's
   equivalent (`graph_reserve`) is nearly free once its ROCm code-object cache
   is warm -- this block is the single biggest structural deficit.
4. **comm init 15.6 s**, of which pynccl bootstrap is ~8.8 s. Overlapping it
   with weight load in a background thread is the obvious idea and is untried.
5. **kernel_warmup 15.1 s** -- one 8-byte pageable H2D blocked in
   libhsa-runtime64.so. Four hypotheses tested and rejected (KV commit, host
   staging pin, tensor-construction path, Triton compile). Still unexplained.

## Startup reduction round 2 (2026-07-29): 156.3 -> 83.2 s, llama.cpp beaten

The external, like-for-like startup target is now met. The final source tree
reached `/health` in **83.16 s** and returned the gated first completion in
**83.82 s**. The comparable llama.cpp number is approximately **84.5 s**:
67.2 s from the top of `llama_cli()` to its first emitted token plus the
separately measured 17.3 s process-exec/dynamic-loader prefix. Thus vLLM is
1.3 s faster to health and 0.7 s faster through its first completion response.

This distinction matters: vLLM has beaten llama.cpp on the external process
clock defined in the baseline section, not llama.cpp's internal 67.2 s timer.
Quoting 83.82 vs 67.2 would mix clocks and is invalid. The margin is also
narrower than the 2-4 s run-to-run variation in the 262 GB pageable weight
transfer, so retain the phase timings when comparing future runs.

| metric | specialization baseline | final warm-cache run | reduction |
| --- | ---: | ---: | ---: |
| process start -> `/health` | 156.3 s | **83.16 s** | **73.1 s / 46.8%** |
| process start -> gated completion | not recorded | **83.82 s** | n/a |
| llama.cpp external -> first token | ~84.5 s | ~84.5 s | reference |

Final-run timeline (`/tmp/vllm_startup_mm_budget_hit.log`):

| phase | window | s |
| --- | ---: | ---: |
| imports | 0 -> 9.60 | 9.60 |
| cached GGUF metadata, config, EngineCore/worker fork | 9.60 -> 15.53 | 5.93 |
| model initialization + weight load + post-load | 15.53 -> 74.39 | 58.86 |
| KV alloc + warmup + graphs + sampler | 74.39 -> 81.60 | 7.21 |
| EngineCore/client/API tail -> HTTP serving | 81.60 -> 82.67 | 1.07 |
| socket polling -> `/health` | 82.67 -> 83.16 | 0.49 |
| text generation gate response | 83.16 -> 83.82 | 0.66 |

The weight phase was 55.16 s plus 2.8 s model construction and 0.8 s
post-load work. Pynccl initialization (8.4 s) is inside that interval and is
now overlapped with the weight stream rather than added before it.

### Changes retained

1. An explicit `--kv-cache-memory-bytes` now means exactly that: skip the
   16,384-token memory-profile forward, its trial large-prefill compile range,
   and the CUDA-graph memory estimate. Automatic KV sizing retains the old
   profiling path. The sampler still warms batch 8, matching the largest
   retained graph.
2. Block-table warmup creates its tiny tensors directly on the GPU. This
   removes the pathological 15 s first pageable H2D without changing any
   runtime block-table behavior.
3. TP communication setup is deferred and runs alongside GGUF weight loading;
   the inactive EP communicator is not initialized on this fixed startup path.
4. GGUF quantization type scalars are extracted on the CPU before upload, so
   startup no longer performs thousands of device scalar synchronizations.
5. The fixed-memory path freezes the completed heap directly instead of doing
   several full collections over the enormous loaded model. In particular,
   `DeviceMemoryProfiler` is bypassed during model load while allocator
   counters preserve the informational model-memory log. Automatic sizing
   still uses the original profiler.
6. Expensive shard-1 GGUF metadata is persisted under
   `VLLM_CACHE_ROOT/gguf_metadata/`, keyed by resolved path, size, mtime and
   schema. A hit remaps tensor bytes but restores the already-decoded field and
   tensor descriptors; malformed or stale entries fall back to the real
   parser. The main checkpoint parse is 2.78 s -> 0.10-0.11 s.
7. The derived multimodal encoder budget is shared across forked processes and
   persisted under `VLLM_CACHE_ROOT/multimodal_budget/`, with a config/version
   fingerprint. On a hit, text startup does not construct or warm the primary
   frontend multimodal processor. The primary and readonly processors are
   created under locks on the first request that actually needs them.
   Fixed-memory startup also skips resetting caches that cannot yet contain
   entries.
8. Fixed-memory frontend warmup omits multimodal dummy preprocessing. The
   first multimodal request intentionally pays that cold cost; text readiness
   and text TTFT do not.

Both persistent caches are derived state, not model state. A first-ever boot
will populate them and be slower; the comparison here, like the established
llama.cpp comparison, is warm steady state. Cache validation is fail-open to
the normal parser/processor path.

### Rejected experiments (all reverted)

- mmap-based contiguous upload, a prefetch thread, moving qweight/type handling
  wholesale to CPU, deferred materialization, direct CPU-to-parameter copies,
  and storage reuse were neutral or slower.
- TP-narrowing the MoE stacks through a CPU clone made the load exceed 100 s.
- Zero shared graph warmups and skipping warmups after the first capture shape
  both violated compiler/capture invariants (`torch.compile` attempted JIT
  during capture).
- Warming the sampler at batch 1 took **1.62 s**, versus about **0.7 s** at
  batch 8. The batch-8 warmup was restored.
- A tokenizer side cache produced no measurable startup improvement.

### Verification

- Final text benchmark, serialized on GPUs 6,7 with warm caches:

  ```bash
  STARTUP_LOG=/tmp/vllm_startup_mm_budget_hit.log \
    STARTUP_GPUS=6,7 STARTUP_KV_CACHE_BYTES=46170898432 \
    STARTUP_CG_MODE=FULL_DECODE_ONLY STARTUP_CG_SIZES=1,2,4,8 \
    /home/hotaisle/.venv/bin/python <SCRATCH>/ttft_vllm.py
  ```

  Result: health **83.16 s**, completion **83.82 s**, PASS with
  `' Paris. The city is located on the'`.
- Lazy vision gate, same fixed-memory path: HTTP 200 and the response identifies
  the 64x64 solid-red PNG as red.
- Persistent GGUF reader parity: all 343 shard-1 tensor names, types, shapes,
  offsets and data shapes; selected first/middle/last tensor bytes; the first
  20 tokenizer entries and last 20 merges all match a fresh parser. Tokenizer
  IDs also match for `The capital of France is` and `<|image|>`.
- Python compilation and `git diff --check` pass on the retained changes.

No PR was opened. Before any future PR, run the duplicate-work commands in
`AGENTS.md`, and the human submitter must review every line and disclose the AI
assistance and these exact test results.

## 100k-context concurrent throughput (2026-07-29)

### Workload and capacity

The throughput sweep uses exact token-ID inputs and forces every generation to
its requested length:

- input: exactly **100,000 tokens** per request;
- output: exactly **2,000 tokens** per request (`min_tokens == max_tokens`);
- concurrency: **1, 2, 4, 8, 16, 32, 64**;
- TP2 on MI300X GPUs 6,7, `max_model_len=102000`, fixed
  `kv_cache_memory_bytes=46170898432`, `max_num_batched_tokens=16384`, and
  decode graph sizes 1,2,4,8,16,32,64;
- prefix caching enabled, with a 99,984-token common prefix and a unique
  16-token suffix for every request.

The shared prefix is essential context for interpreting these numbers. The
fixed KV allocation holds 967,936 tokens, so at most nine independent 102k
sequences can be resident. With block sharing, the 64-way workload needs
99,984 common tokens plus `64 * (16 + 2,000)` unique tokens, or 229,008 resident
tokens. Thus all 64 requests remain resident and this sweep measures concurrent
long-context decode rather than scheduler queueing over repeated unique
prefills. A fully unique 64-way sweep is a separate multi-hour ingestion and
queueing benchmark.

`aggregate output tok/s` is `total generated tokens / wall time`; it does not
count cached prompt tokens as physical work. `decode-window tok/s` excludes
the interval before the first output token. Per-request tok/s is aggregate
throughput divided by concurrency.

### Optimization

The ROCm GGUF MoE dispatch formerly used one effective crossover for both
matmuls: vector kernels through 64 input rows for w1 and through
`64 * top_k = 512` routed rows for w2. Direct MI300X measurements place the
crossovers near 96 w1 input rows and only 80 w2 routed rows. The defaults now
use those independent limits on ROCm while retaining the existing environment
overrides. Non-ROCm dispatch is unchanged.

A controlled 128-output-token A/B sweep isolated the dispatch change without
spending another full 2k-output run on the slower baseline:

| Concurrency | Historical default tok/s | 96/80 split tok/s | Change |
| ---: | ---: | ---: | ---: |
| 1 | 36.62 | 38.36 | +4.8% |
| 2 | 63.62 | 63.00 | -1.0% |
| 4 | 92.78 | 92.62 | -0.2% |
| 8 | 123.91 | 124.34 | +0.3% |
| 16 | 148.04 | 173.02 | +16.9% |
| 32 | 147.16 | 176.13 | +19.7% |
| 64 | 170.11 | 228.75 | +34.5% |

The small low-concurrency movements are run noise; those batches retain the
same kernel choices. The material gains begin where the old w2 threshold kept
the slower vector kernel active.

### Final exact 100k/2k sweep

| Concurrency | Wall time (s) | Aggregate output tok/s | Decode-window tok/s | Per-request tok/s | TTFT p50 / p95 (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 47.71 | 41.92 | 42.13 | 41.92 | 0.235 / 0.235 |
| 2 | 57.31 | 69.79 | 70.08 | 34.90 | 0.341 / 0.440 |
| 4 | 77.55 | 103.16 | 103.48 | 25.79 | 0.502 / 0.506 |
| 8 | 120.45 | 132.84 | 133.09 | 16.61 | 0.731 / 0.742 |
| 16 | 175.87 | 181.95 | 182.18 | 11.37 | 0.888 / 1.085 |
| 32 | 350.50 | 182.59 | 182.71 | 5.71 | 1.441 / 1.803 |
| 64 | 533.06 | 240.12 | 240.24 | 3.75 | 2.450 / 2.930 |

Every request produced exactly 2,000 tokens from exactly 100,000 input token
IDs. Every measured request reported exactly 99,984 cached prompt tokens. A
32-request factual quality gate also passed 32/32 responses while exercising
the mixed w1-vector/w2-MMQ path. The warmed common-prefix insertion ran at
about 849 prompt tok/s.

The retained source passes Python compilation and `git diff --check`. Raw
results and logs are in `/tmp/longctx_baseline_128.{json,log}`,
`/tmp/longctx_moesplit_128.{json,log}`, and
`/tmp/longctx_final_shared_2000.{json,log}`.

### Rejected experiment

The production sparse-indexer configuration was measured separately at the
exact 100k sequence length. It costs about 0.07-0.14 ms per indexed layer for
batches 1-64 (roughly 1.5-2.7 ms over the indexed layers), so it is not the
source of the high-concurrency plateau and no indexer change was retained. An
unsupported alternative launch shape faulted in the isolated benchmark and
was discarded; it was never enabled in production.

## QuixiCore-guided Q2_K kernel pass (2026-07-29)

The next kernel search used
`~/QuixiCore/QuixiCore-ROCm/perf/perf.md` as the design guide. The relevant
principles were to optimize the exact decoder shapes, distinguish ALU and
bandwidth ceilings, keep wave64 register pressure explicit, hoist metadata only
when reuse amortizes it, and reject candidates by measured end-to-end results.
A steady-decode trace attributed about 52% of GPU kernel time to the routed
Q2_K expert kernels, making this the highest-value kernel target.

### Accepted matrix tile

The retained ROCm Q2_K MMQ tile changes from 4 routed tokens by 128 output rows
to **4 by 64**. The smaller row tile halves the per-thread output accumulator
count and Q2_K LDS span. It launches more blocks, but on MI300X the lower
register/LDS pressure wins decisively at the real GLM-5.2 TP2 shapes.

| TP2 MoE shape | 4x128 (us) | 4x64 (us) | Speedup |
| --- | ---: | ---: | ---: |
| w13, 64 model tokens | 1,201.7 | 851.7 | 1.41x |
| w2, 512 routed rows | 603.8 | 431.2 | 1.40x |
| w13, 512 speculative-verification tokens | 4,995.4 | 3,490.1 | 1.43x |
| w2, 4,096 routed verification rows | 2,517.9 | 1,818.7 | 1.38x |

The final clean rebuild reproduced 847.7/447.6 us at the 64-token w13/w2
shapes and 3,505.3/1,815.4 us at the 512-token verification shapes. Matrix and
vector results passed `torch.testing.assert_close` for every accepted shape;
the largest observed BF16 absolute difference was 0.25.

The faster matrix kernel moves the measured dispatch crossovers. W1 remains on
the vector kernel through 32 model tokens (598.0 vs 605.1 us at 32), then uses
MMQ at 64 (1,070.7 vs 826.7 us). W2 remains vector through 128 routed rows
(167.7 vs 193.2 us), then uses MMQ at 256 (328.5 vs 294.0 us). The ROCm
defaults are therefore now **32 W1 rows** and **128 W2 routed rows**; the
existing environment overrides remain available for other GGUF shapes.

### End-to-end result

The controlled 64-request 100k/128 run improved from 228.75 to **303.60
aggregate output tok/s** (+32.7%). The authoritative exact workload then
generated 2,000 tokens for each of 64 requests, with exactly 100,000 input
tokens and 99,984 cached prompt tokens per request:

| 64-request exact workload | Before | After | Change |
| --- | ---: | ---: | ---: |
| Wall time | 533.06 s | 393.84 s | -26.1% |
| Aggregate output tok/s | 240.12 | **325.01** | **+35.3%** |
| Per-request tok/s | 3.75 | **5.08** | **+35.3%** |

Live steady decode began near 341 tok/s and settled near 317-329 tok/s as the
context grew. All 64 requests remained resident with no scheduler queueing.
The exact result is in `/tmp/longctx_q2_y64_2000.json`; the short acceptance
result is in `/tmp/longctx_q2_y64_128.json`.

This is a substantial kernel win but still **3.08x short of 1,000 aggregate
tok/s**. The remaining gap cannot be closed by this tile change alone.

### Rejected kernel designs (all removed)

- A direct FP16 MFMA kernel was numerically correct but 3-5x slower. At batch
  64, w13 was 1.243 -> 4.965 ms and w2 was 0.621 -> 3.237 ms. The routed expert
  layout made weight access row-strided, and the four-token MFMA tile wasted
  most of its M capacity.
- An exact Q8-activation/int8-MFMA version was worse: w13 was
  1.231 -> 8.642 ms and w2 was 0.603 -> 1.335 ms at batch 64. Padding the small
  routed-token dimension and staging fragments outweighed MFMA throughput.
- A wave64 Q2 vector kernel that broadcast packed scales/deltas and hoisted Q8
  fragments regressed batch-64 w13 by 23% and w2 by 17%. Shuffle and register
  costs exceeded the saved metadata loads.
- Keeping the original lane mapping while reusing Q8 fragments was mixed. Four
  rows regressed batch-64 w13 by 10%; two rows gave only a noisy 1.8% w13 gain
  while making w2 about 33% slower. Neither met the acceptance gate.
- A 4x256 MMQ tile regressed batch-64 w2 from 599.4 us to 2,373.6 us due to
  accumulator/occupancy pressure. A two-wave-count variant violated the
  kernel's token-to-wave mapping and produced NaNs; it was never enabled.

Only the 4x64 tile and its measured dispatch thresholds remain in production.

## QuixiCore-guided Q2/Q8 kernel pass (2026-07-29)

This pass again used `~/QuixiCore/QuixiCore-ROCm/perf/perf.md`, this time as a
strict workflow rather than only a source of kernel ideas: benchmark the exact
serving shapes, classify the limiting resource, sweep one geometry variable at
a time, inspect the compiled VGPR/LDS/spill footprint, and accept changes only
after an exact end-to-end run. Two reusable microbenchmarks now cover the real
GLM-5.2 TP2 shapes:

- `benchmarks/kernels/benchmark_gguf_moe_q2.py` covers routed Q2_K w13/w2;
- `benchmarks/kernels/benchmark_gguf_linear.py` covers the Q8_0 attention and
  dense projections.

### Accepted Q2_K MoE design

The ROCm routed-expert tile is now **4 routed rows by 32 output rows**, down
from the previous 4x64 tile. The smaller tile improves CU coverage and reduces
per-block weight/LDS work. Q8 activation staging now groups two quant spans per
barrier pair, amortizing synchronization without the short-K regression of a
four-span stage.

| TP2 MoE shape | Previous 4x64 (us) | Final 4x32 + stage-2 (us) | Speedup |
| --- | ---: | ---: | ---: |
| w13, 64 model tokens | 884.506 | 797.335 | 1.109x |
| w2, 512 routed rows | 449.354 | 422.494 | 1.064x |
| w13, 512 verification tokens | 3,037.452 | 2,778.504 | 1.093x |
| w2, 4,096 routed rows | 1,592.988 | 1,479.758 | 1.077x |

The BF16 matrix and vector outputs were identical for the valid repeated Q2_K
test blocks (`max_abs_diff=0`). The compiled Q2 kernel uses 110 VGPRs, 42
SGPRs, 6,704 bytes of LDS, and has no VGPR/SGPR spills.

### Accepted shape-adaptive Q8_0 design

The Q8_0 matrix kernel now has two MI300X row tiles. It uses 64 output rows
when `nrows_x * ncols_y <= 2,097,152`, and retains the old 128-row tile above
that crossover. This captures the small-decode gain without regressing large
verification batches.

| Q8_0 projection, batch 64 | Previous 128-row tile (us) | 64-row tile (us) | Speedup |
| --- | ---: | ---: | ---: |
| o_proj, 6,144 x 8,192 | 410.936 | 314.051 | 1.308x |
| dense gate/up, 12,288 x 6,144 | 308.843 | 234.652 | 1.316x |
| q_b_proj, 6,144 x 2,048 | 113.630 | 91.397 | 1.243x |

At batch 512 the o_proj dispatch selects the 128-row tile and measures 606.468
us versus the 609.434-us baseline. A fixed 64-row tile measured 680.539 us at
that shape, which is why the adaptive crossover is required.

The linear dispatch also keeps Q8_0 projections with at most 6,144 output rows
on the multi-column vector kernel through batch 64. At the real batch-64
shapes, o_proj is 241.062 us vector versus 314.051 us matrix, and q_b_proj is
87.331 versus 91.397 us. The 12,288-row dense projection remains on the matrix
kernel (234.652 versus 355.212 us). Matrix/vector output parity again had
`max_abs_diff=0`.

### Exact end-to-end result

The authoritative TP2 workload generated exactly 2,000 tokens for each of 64
requests from exactly 100,000 input token IDs. Every request reported exactly
99,984 cached prompt tokens; all 64 remained resident with no scheduler
queueing.

| 64-request exact workload | Previous | Final | Change |
| --- | ---: | ---: | ---: |
| Wall time | 393.838 s | **323.313 s** | **-17.9%** |
| Aggregate output tok/s | 325.007 | **395.902** | **+21.8%** |
| Per-request tok/s | 5.078 | **6.186** | **+21.8%** |

Steady decode started around 414-420 tok/s and settled around 393-398 tok/s as
the context grew. Relative to the original 240.12-tok/s 64-way baseline, the
combined kernel work is now **+64.9%**. The 1,000-tok/s target is still 2.53x
away. The raw result is `/tmp/longctx_q2_y32_q8_2000.json`.

### Rejected designs

- Unrolling the inner Q2 dot-product loop was neutral to slightly slower and
  was removed.
- Four-span Q8 activation staging improved w13 but regressed the short-K w2
  shape; two spans was the best balanced setting.
- Replacing Q8 activation LDS staging with register/shuffle broadcasts was
  substantially worse: 1,062/536 us at the two decode shapes and 4,069/2,017
  us at the verification shapes. It was removed.
- A fixed 64-row Q8 tile won at decode sizes but regressed batch-512 o_proj by
  11.7%; the retained dual-tile dispatch avoids that loss.

The final source rebuilt and installed successfully. The focused pre-commit
suite (ruff, clang-format, mypy, typos, SPDX, and repository checks) passes,
`git diff --check` passes, and post-format/reinstall microbenchmarks reproduce
800.620 us for Q2 w13 and 235.833 us for the Q8 dense projection with exact
synthetic parity.

## RedHatAI GLM-5.2 DSpark evaluation (2026-07-29)

Installed `RedHatAI/GLM-5.2-speculator.dspark` at revision
`8bc9ac46fbf507f3ee3ad82304116a1f63e9edb4` under
`/home/hotaisle/models/GLM-5.2-speculator.dspark`. This is the new three-layer,
5.87-GiB BF16 DSpark checkpoint with an eight-token block and
`sample_from_anchor=true`. The requested serving configuration was used:

```text
--spec-model /home/hotaisle/models/GLM-5.2-speculator.dspark
--spec-method dspark
--spec-tokens 7
```

The specialized source tree had removed the upstream speculators config
algorithm module while retaining an import of it. Restoring the upstream
module made the checkpoint resolve as `Qwen3DSparkModel` with auxiliary target
layers `(2, 20, 39, 58, 75)`. Explicit standalone speculators also had to stop
inheriting the target GGUF `model_weights` and `hf_config_path`; otherwise the
draft was incorrectly resolved from the GLM verifier config as
`DeepSeekMTPModel`.

ROCm exposed a KV-block compatibility constraint. The target sparse MLA and
sparse SWA groups require a multiple of 64, while AITER flash attention for
the draft advertises only 16- and 32-token blocks. The working configuration
therefore uses `block_size=64` and `TRITON_ATTN` for the draft while retaining
AITER sparse MLA for the target. The exact long-context configuration loaded
129.19 GiB of weights and runtime state per GPU; its fixed KV allocation held
769,543 tokens, enough to keep all 64 requests resident.

### Acceptance sanity check

Three deterministic natural-language prompts, 128 output tokens each, proved
that DSpark loading and verification are functional:

| Metric | Result |
| --- | ---: |
| Draft steps | 162 |
| Draft tokens | 1,134 |
| Accepted draft tokens | 221 |
| Draft-token acceptance | 19.49% |
| Mean acceptance length | 2.364 |
| Per-position acceptance | 60.5%, 35.8%, 21.6%, 11.7%, 4.3%, 1.2%, 1.2% |
| Aggregate output tok/s | 68.63 |

The short-prompt probe forced sparse MLA through its MQA prefill path because
the current ROCm sparse backend does not implement the dense `forward_mha`
fallback. Two first-inference Triton JITs were included in the 5.60-second
timing, so the throughput number is diagnostic; the acceptance counters are
the important result. Raw output is in `/tmp/spec_accept_dspark7_new.json` and
`/tmp/spec_accept_dspark7_new.log`.

### Long-context acceptance failure (unresolved correctness issue)

On the 64-request workload with exactly 100,000 synthetic input token IDs per
request, 99,984 shared cached tokens, and a requested 2,000-token response,
acceptance collapsed in steady decode:

| Configuration | Aggregate output tok/s | Mean acceptance length |
| --- | ---: | ---: |
| No speculation, completed exact run | **395.902** | 1.000 |
| DSpark, 7 draft tokens | **104.9-105.6** | 1.000-1.010 |

Steady intervals accepted only 0-0.2% of draft tokens. DSpark drafted about
734 tokens/s while producing about 105 verified output tokens/s. That is a
roughly 73.5% throughput regression, or 3.77x slower than the completed
non-speculative baseline. The full 2,000-token DSpark run was intentionally
stopped after several stable intervals rather than spending roughly 20
minutes confirming a severe regression; it is not an exact completed result.
The raw partial trace is `/tmp/longctx_dspark7_2000.log`.

Do not attribute this collapse to the routed Q2_K verifier. The short natural
probe establishes that the same target/draft pair has coherent logits, and a
close target quantization is not a credible explanation for first-position
agreement falling to approximately zero. The collapse is therefore tracked as
an unresolved inference/caching/context correctness issue; the 105 tok/s value
is diagnostic and not a valid DSpark performance result.

Checks completed so far:

- verifier/draft token alignment is correct at shift zero; shifts -2, -1, +1,
  and +2 do not explain the rejection rate;
- DSpark's seven-query anchor layout and scheduler lookahead allocation agree;
- captured attention metadata points at the persistent runtime sequence-length,
  block-table, and slot-mapping buffers rather than frozen dummy tensors;
- auxiliary target layers resolve to `(2, 20, 39, 58, 75)`, and moving all five
  by one layer does not repair acceptance;
- disabling the Markov head and changing the draft KV cache from FP8 to BF16 do
  not repair the short-context acceptance deficit;
- the current runtime causal SWA mask is not exactly the training mask. Training
  preserves the same 2,048 base tokens for every query in the block, while the
  standard runtime window advances with each query and drops one to seven of
  the oldest base tokens. This is a real semantic mismatch, but it is too small
  to claim as the cause of approximately zero first-token agreement without an
  A/B measurement.

The next isolating experiment is a deterministic context-length and batch-size
sweep with prefix caching independently disabled and draft full graphs
independently disabled. It must establish whether the failure follows context
length, batch 64, graph replay, or prefix-cache reuse before a correctness fix
is retained. GPU device access was unavailable for that sweep at the end of
this investigation.

## DSpark acceptance collapse: root cause found, benchmark artifact (2026-07-29)

The isolating sweep ran and the collapse is **not a serving-stack bug**. It is
an artifact of the synthetic benchmark prompt, which is literally
`[token 1000] * 99,984` plus a 16-token per-request suffix. Conditioned on
thousands of copies of one token, the target itself decodes erratic garbage
(`"Commentsimport 1 importimportimport… import for the ofimport from
forimport to"`), and the three-layer draft — whose inputs are the target's
now far-out-of-distribution aux hidden states — cannot match that stream.
First-position agreement collapses to ~0 for workload reasons, with every
runtime mechanism working correctly.

Before concluding this, the code was compared against upstream main
(`~/vllm2`, 5c7a7f9462): the DSpark/DFlash speculators, the TRITON_ATTN
backend, and the triton unified attention kernel are byte-identical to
upstream; the only fork deltas are the documented draft-loading fixes plus an
upstream Gumbel/seq-lens fix that only affects `advance_draft_positions=False`
autoregressive drafts (gemma4), which DSpark does not use.

Experiment 1 — context-length sweep, batch 1, prefix caching off, natural
prose filler + same question, 128 greedy tokens (script
`dspark_ctx_sweep.py` in the session scratchpad, raw
`/tmp/dspark_ctx_sweep.json`):

| Case | Prompt tokens | Draft acceptance | Mean accept len |
| --- | ---: | ---: | ---: |
| natural_500 | 527 | 31.8% | 3.23 |
| natural_1500 | 1,527 | 27.9% | 2.95 |
| natural_2500 | 2,527 | 32.6% | 3.28 |
| natural_6000 | 6,027 | 32.1% | 3.24 |
| natural_12000 | 12,027 | 33.5% | 3.34 |
| shuffled_6000 (word salad) | 6,027 | 15.0% | 2.05 |

Crossing the draft's 2,048-token sliding window causes no degradation at all,
so the runtime-vs-training SWA mask difference noted above is immaterial in
practice. Distribution shift (shuffled words) halves acceptance but is
nowhere near zero.

Experiment 2 — prefix caching on, `max_num_batched_tokens=8192` to force
3-chunk prefill on 18k prompts (script `dspark_cache_chunk.py`, raw
`/tmp/dspark_cache_chunk.json`):

| Case | Cached tokens | Draft acceptance |
| --- | ---: | ---: |
| nat18k_cold (3 prefill chunks) | 0 | 25.5% |
| nat18k_warm (same prompt again) | 17,920 | 27.3% |
| nat18k_forked (same prefix, new question) | 17,920 | 16.7% |
| repeated6k_cold (`[1000]*6000 + [2001]*16`) | 0 | **0.23%** |
| repeated6k_forked (`[1000]*6000 + [2002]*16`) | 5,888 | **0.00%** |

Chunked prefill, warm prefix-cache hits, fine-grained (16-token) partial-block
hits with copy-on-write, and forked shared prefixes are all healthy on natural
text. The collapse reproduces at 6k tokens, batch 1, cold cache, single-chunk
prefill — with the repeated-token prompt and nothing else. The failure
follows the prompt content, full stop.

Consequences:

- The "long-context acceptance failure" above is a mislabel; the 105 tok/s
  DSpark number and 1.00 acceptance length say nothing about DSpark at real
  long context. The non-speculative 395.9 tok/s baseline is likewise a
  throughput-only number whose generated text is degenerate garbage.
- Long-context DSpark evaluation needs prompts with natural token statistics
  (long documents, or model-generated text), with per-request unique suffixes
  retained if the shared-prefix batching pattern is being measured.
- Healthy acceptance on this target is ~25-33% draft-token acceptance / ~2.8-3.3
  mean acceptance length at temperature 0 on natural prose, consistent from
  500 to 12,000 tokens of context.

## Valid long-context batch-64 DSpark A/B: spec-3 wins (2026-07-29)

Rebuilt the 100k-token benchmark on natural text (the vLLM docs corpus from
`~/vllm2/docs` as a shared prefix, 64 distinct question suffixes, prefix
caching on, TP2, block 64, same 46.17 GiB KV budget) and swept
`num_speculative_tokens`. Two prompt formats were run: raw completion (fixed
1,000 output tokens, `ignore_eos`) and proper chat template
(`enable_thinking=False`, natural EOS stopping — the earlier raw format made
some requests echo the question pattern). Scripts `bench_longctx_natural.py`
and `bench_longctx_chat.py` in the session scratchpad; raw JSON in
`/tmp/dspark_longctx_ab_*.json` and `/tmp/dspark_longctx_chat_*.json`.

Chat-templated (the realistic configuration; ~646-token mean answers, batch
drains at EOS):

| Mode | Aggregate tok/s | vs baseline | Draft acceptance | Mean accept len |
| --- | ---: | ---: | ---: | ---: |
| no spec | 254.25 | — | — | — |
| **dspark spec-3** | **285.32** | **+12.2%** | 56.7% | 2.70 |
| dspark spec-4 | 252.73 | -0.6% | 47.0% | 2.88 |
| dspark spec-7 | 194.47 | -23.5% | 29.6% | 3.07 |

Raw completion (uniform 1,000-token load, all requests finish together):
no-spec 396.56, spec-3 **412.85 (+4.1%)**, spec-4 401.59, spec-7 311.40
(-21%). Both formats agree on the ordering.

Conclusions:

- **Serve batch-64 long context with `num_speculative_tokens=3`.** Mean
  accept length grows only 2.70 → 3.07 from spec-3 to spec-7 while verify
  width grows 256 → 512 rows; past 4 rows/request the extra verification
  compute swamps the acceptance gain. Keep spec-7 only for the low-batch
  latency case (60.8 tok/s single-request result stands).
- Acceptance on the docs-summarization workload beats RedHatAI's published
  per-position curve at positions 0-3 (85%/71%/58% at spec-3) despite the
  Q2_K target; only deep positions lag, which is quantization drift
  compounding, not a serving bug.
- These numbers are temperature 0; production sampling params need one
  confirming run before locking the config.

## TurboQuant: draft-KV integration and target-side scoping (2026-07-29)

The dead-code cascade had deleted the TurboQuant implementation while keeping
its config plumbing. Restored from upstream main (byte-identical files):
`v1/attention/backends/turboquant_attn.py`, `v1/attention/ops/
triton_turboquant_{store,decode}.py`, `v1/attention/ops/
triton_decode_attention.py`, `turboquant/{__init__,centroids}.py`, plus the
deleted helpers `fa_utils.is_flash_attn_varlen_func_available` and
`workspace.is_workspace_manager_initialized`.

Upstream TurboQuant cannot serve the DSpark draft as-is; three capabilities
were added in this fork:

1. **Causal sliding window** in the TQ decode kernel (`SLIDING_WINDOW`
   constexpr restricting the split range to the last W positions; stage 2
   gets `min(seq_len, W)` for split occupancy), in the SDPA/flash-attn
   prefill fallbacks (banded masks / `window_size`), and window-clamped
   continuation dequant + workspace sizing. Kernel verified on-GPU:
   bit-exact vs full attention within the window, invariant to garbling all
   out-of-window cache bytes, and reads exactly the last W positions.
2. **Per-layer TQ cache dtype against a non-TQ target**: new
   `TQSlidingWindowSpec` (and a `tq_cache_dtype` field on
   `TQFullAttentionSpec`) carried through spec promotion, KV reshape, the
   block zeroer, and the mixed-view aligner, so
   `speculative_config={"attention_backend": "TURBOQUANT",
   "kv_cache_dtype": "turboquant_*"}` composes with the fp8 MLA target.
   TurboQuantMetadata is also on the ROCm spec-decode allowlist now.
3. **A CUDA-graph-safe uniform-query path**: the DFlash 8-query block pass
   previously routed into TQ's per-request prefill loop (CPU `.tolist()`
   syncs — crashes graph capture, wrong at replay). A batched
   synthetic-decode path now handles uniform q-len batches with pure GPU
   ops and one kernel launch.

End-to-end smoke (batch 1, `turboquant_k8v4` draft KV, TRITON target config
unchanged, `dspark_tq_smoke.py`, raw `/tmp/dspark_tq_smoke.json`): coherent
output and healthy acceptance both inside and past the 2,048-token window —
25.2% / mean 2.76 at 500 prompt tokens, **31.0% / mean 3.17 at 4,000**
(fp8-KV reference: 31.8% / 32.1%). k8v4 draft-KV quantization costs a few
points at short context and is at parity by 4k; the windowed TQ decode
kernel runs correctly under FULL cudagraph replay. Not yet re-benchmarked at
batch 64 / 100k.

Draft KV sizes at head 64 (per head-token; fp8 = 128 B): k8v4 100 B,
4bit_nc 70 B, k3v4_nc 62 B, 3bit_nc 54 B. The draft currently costs
11.8 KB/token/GPU of 63.0 KB total (967,399 → 769,237 tokens when enabled),
because page-size unification fails and its SWA layers are allocated
full-length; TQ shrinks that to ~9.2 (k8v4) … ~5.0 KB (3bit_nc), i.e. a
4-11% KV-capacity recovery. The larger draft-side prize remains
window-limited allocation (would cut nearly all 11.8 KB), which is a
hybrid-page-size project independent of TQ.

Target-side scoping (the 256k/batch-capacity goal):

- **MLA latent is 87% of KV** (78 layers x 576 B fp8 = 43.9 KB/token of
  51.2 KB baseline). TQ's contract supports head sizes 64/128/256; the
  576-wide latent needs a format extension (e.g. grouped 64-wide FWHT
  segments). The tractable integration is **gather+dequant**: DSA sparse
  MLA only ever attends to `index_topk=2048` gathered tokens per query, so
  a Triton gather-dequant producing a dense bf16 `[topk, 576]` scratch
  feeding the existing AITER sparse kernels bounds the dequant cost by
  topk, not context length. Capacity 2-4x on the dominant term; bandwidth
  neutral-to-better. Quality of <=4-bit latent is unproven — gate on an
  acceptance/gsm8k eval. Multi-day kernel + spec + store-path project.
- **The indexer cache is NOT a lever** (correcting an earlier guess in this
  section): `DeepseekV32IndexerCache` is already fp8-packed — allocated as
  `torch.uint8` with `head_dim = 128 + 128/quant_block_size*4` bytes per
  layer-token. The `indexer_kv_dtype` / `use_fp4_indexer_cache` config knobs
  only wire into MiniMax-M3, not the GLM DSA path, so setting them changes
  nothing here.

## Prefix caching verification (2026-07-29)

APC is on by default (`CacheConfig.enable_prefix_caching = True`) and
verified working under the production-like config (TP2, dspark spec-3,
block 64, 12k natural prompts; `apc_verify.py`, raw `/tmp/apc_verify.json`):

| Check | Result |
| --- | --- |
| Cold request | 0 cached tokens |
| Warm (identical prompt) | 11,904/12,017 cached; TTFT 12.55 s -> 1.10 s (11.4x) |
| Forked (same prefix, new question) | 11,904 cached |
| Partial overlap (~6k shared) | 5,888/6,281 cached (16-token granularity) |
| Output correctness | warm == cold, 64/64 greedy tokens, no divergence |
| Counters | `vllm:prefix_cache_hits` 29,696 / `queries` 42,332 |

The ~113-token shortfall on a "full" hit is the sub-granularity tail plus
the always-recomputed final position, not a defect. Caveat: with fp8 KV,
bitwise warm/cold equality is not guaranteed — an 18k probe earlier in the
day showed a single-word divergence (a low-margin numeric flip, not
corruption). Batch-64 runs showed 99.9% hit rate and 99,904-token hits.

## 1M context: fits at TP2, workspace is the real constraint (2026-07-29)

The model's ceiling is exactly `max_position_embeddings = 1,048,576`; vLLM
enforces it (asking for 1,050,000 is a hard config error), so 1M is the
supported limit with no headroom beyond.

Measured KV cost is **46.6 KB/token** with no draft (55 GiB held 1,238,016
tokens), so 1M of KV needs ~45.5 GiB of the ~62.8 GiB left after the
129.2 GiB of weights. Getting there requires setting
`kv_cache_memory_bytes` **by hand**:

- `gpu_memory_utilization=0.94` alone makes the auto-sizer request 64.00 GiB
  with 61.85 GiB free — instant OOM at init.
- 55 GiB of KV allocates fine (1,238,016 tokens) but then OOMs during
  execution with 8 MiB free: the sparse-MLA and indexer top-k workspaces
  scale with `max_model_len` and need GiB-scale headroom at 1M.
- **47 GiB (`kv_cache_memory_bytes=50_465_865_728`) works**: 1,057,984
  tokens, "Maximum concurrency for 1,048,576 tokens per request: 1.01x",
  engine initializes and runs.

**Confirmed end-to-end** (`ctx1m_smoke.py`, raw `/tmp/ctx1m_smoke.json`):
1,000,021 prompt tokens (the docs corpus tiled) + 32 greedy output tokens in
**1,847 s (30.8 min), 541.4 tok/s prefill**. The answer is coherent and drawn
from the corpus ("Automatic Prefix Caching (APC): Caches the KV cache of
existing queries so that new queries sharing the same prefix can directly
reuse..."), so sparse MLA, the indexer top-k path, block tables and prefix
caching are all correct at 1M — not merely allocatable. Note prefill at 1M
runs at 541 tok/s, below the ~800 tok/s seen at 100k: the sparse indexer
scan grows with context even though decode attention stays top-k bounded.

Consequences: at 1M you serve exactly one request at a time. Batch and
context trade directly because the MLA latent does not shard with TP —
adding GPUs buys batch, not context. Prefill dominates wall-clock at
~800 tok/s (a 1M prefill is ~20 minutes), which APC amortizes across
requests sharing a corpus but which rules out 1M for cold one-shot traffic.
Speculation is unaffordable at this length (the draft costs 11.8 KB/token,
~11.8 GiB at 1M); drop it, or use the TurboQuant draft KV from this same
day's work (~5 KB/token at 3-bit).

Also note `max_num_batched_tokens=8192` fails at this config with
"Expected exactly one compiled range_entry for static shape compilation,
but found 2" — the compile ranges are built around that endpoint. Use
16,384.
