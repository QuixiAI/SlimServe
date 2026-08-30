<!--
Two active campaign handoffs live in this file. They cover different
platforms and different hardware, and each is current for its own campaign;
neither supersedes the other. They met on this file in the 2026-08-28 merge
of origin/main and were joined rather than reconciled.

  1. NVFP4-on-Metal campaign (M1 Ultra / M5 Max)  -- section below
  2. MI300X GGUF profile record                   -- second section
-->

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
