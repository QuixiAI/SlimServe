# DeepSeek V4 0731 A100 Handoff

Updated: 2026-08-10 22:20 UTC

## Degeneration incident: ROOT-CAUSED AND FIXED (2026-08-13)

The entire 2026-08 incident family - the rare NaN seed, the BOS-loop
storms, and the deterministic no-spec degeneration - was ONE defect:
IEEE 0*NaN in the DSV4 sparse decode's split-K reduce
(csrc/quixicore/serving/paged_attn_v2_kernels.cuh). The persistent
writer publishes finite ml/es for balanced-away empty partitions but
never stores their tmp value vectors (torch::empty); the reducer
multiplied those unwritten vectors by a mathematically-zero weight,
and 0*NaN = NaN carried recycled allocator bytes into live attention
outputs. Boot/phase lottery = recycled pool content; geometry gating
= partition counts; the FULL-graph and aux-stream correlations were
allocation-pattern epiphenomena. Fix: skip !(weight > 0) partitions
(mathematically identical, garbage-proof), hardened in all four
reducers; ml/es workspace init kept as defense in depth (ceba09670).

Validation: the 13/13-storming no-spec repro ran clean (3 full
trigger runs, 0 NaN); production (spec, both daemons) redeployed on
the fixed kernel - dual c16 acceptance clean, 544-570 tok/s, zero NaN
events. --no-spec is UNBROKEN. Aux streams remain off on A100 purely
for performance (458 vs 427 c11 - the overlap was a net loss); it is
no longer a correctness mitigation. NAN_WATCH stays on both daemons
and the canary stays armed as regression tripwires. The full
investigation record - including every dead theory and both
instrument-bug retractions - is in perf/optimization_status.md;
diagnostics live in perf/diagnostics/dsv4-nan/ plus env-gated probes
(VLLM_DSV4_MLA_DEBUG_PARTIALS, VLLM_DSV4_MLA_TMP_SENTINEL).

## Read First

Work in `/home/ubuntu/SlimServe`. Read these before changing code or interpreting
performance:

- `AGENTS.md`
- `perf/perf.md`
- `perf/baseline_status.md`
- `perf/optimization_status.md` (see the "A100 TP2 Lifecycle Crash Root Cause"
  entry for the full debugging record of the crash below)
- `slimserve/profiles.json`, especially the dsv4-{hybrid,mxfp4}-{2,4,8} family
- `perf/dsv4_a100_kernel_history.md` for the longer TP2/TP4 kernel and
  ownership history (moved from the old root `handoff.md`; its serving
  tok/s numbers predate the sampler fix and are obsolete)

The worktree is intentionally very dirty and contains user and prior-agent work.
Do not reset, clean, checkout, or overwrite it. SlimServe downloads missing model
artifacts itself. Use the real profile, not an invented vLLM command.

## Objective

Finish the native DeepSeek V4 Flash 0731 Ampere A100 path, including DSpark,
TurboQuant, APC, long-context capacity, and TP2/TP4 performance. The routed MoE
production path must remain fused native IQ2_XXS gate/up + SwiGLU + Q2_K down +
weighted reduce. No dequant production fallback.

## Status: TP2 lifecycle crash SOLVED (2026-08-09 evening)

The "illegal CUDA memory access after ~7 decode tokens" that blocked TP2 is
fixed. The earlier aux-stream ownership theory was WRONG; serializing streams
only appeared to help because it changed allocator layout. Real chain:

1. A NaN target-logits row reached the rejection sampler.
2. `argmax_combine` in `csrc/quixicore/serving/v2_sample_kernels.cuh` used a
   negated comparison that let NaN replace the running best; the masked -inf
   tail lanes of the last vocab block then won with the lowest masked index,
   emitting token id 129280 == vocab_size as a "sampled" token.
3. 129280 entered `last_sampled_tokens`, was combined into `input_ids[0]`
   every step (self-sustaining poisoning; drafts degenerated, all rejected).
4. `dsv4_router::bf16_hash_router` dereferenced `tid2eid[129280*6]` (table,
   embedding, and lm head all have exactly 129280 rows) → garbage expert →
   wild weight-row read → MMU FAULT_PDE. Fault-vs-silent depended on what the
   allocator placed after the 2 MiB gate weight, hence the illusion that
   stream serialization "fixed" it. Serialized runs were silently
   quality-poisoned instead (all drafts rejected ≈ no spec gain).

Fixes, all retained and offline-verified
(`repro_rejection_oov.py` in the session scratchpad; NaN/-inf/garbage rows now
sample token 0, never an out-of-vocab id):

- `csrc/quixicore/serving/v2_sample_kernels.cuh`: positive-form
  `argmax_combine`, NaN-sanitized loads, in-vocab-only candidates, and
  sentinel-free reduction inits (`best_* = 0x7fffffff` could leak as a token
  id / wild index when every candidate was skipped). Ported to
  `/home/ubuntu/QuixiCore/QuixiCore-CUDA/kernels/serving/v2_sample_kernels.cuh`
  (the ROCm build includes the same header, nothing separate to port).
- Same NaN sanitize in the Triton fallbacks
  (`vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py`).
- `csrc/quixicore/serving/dsv4_router_ampere.cuh`: `bf16_hash_router` bounds-
  guards token ids (OOB → treated as padding) and records the first offender
  in a device slot readable via `quixicore_ops.dsv4_hash_router_debug()`
  (readout gated by `VLLM_DSV4_HASH_ROUTER_DEBUG=1` in the V2 runner).
- Rebuild: `cmake --build build/temp.linux-x86_64-cpython-312 --target
  _quixicore_C -j$(nproc)` then copy the `.so` into `vllm/`.

## Requalified numbers (clean server v33, full overlap, no debug env)

Canonical exact harness, TP2, `dsv4-2`, 19 GiB KV, PIECEWISE graphs, DSpark on:

| Run | tok/s | exact |
| --- | ---: | --- |
| 1K/2K run 1 | 168.0 | yes |
| 1K/2K run 2 | 168.7 | yes |
| 12K cold | 89.0 | yes |
| 12K hot | 93.3 | yes |
| 128K cold | 37.1 | yes |
| 128K hot | 38.2 | yes |
| post-128K 1K/2K continuation | 111.8 | yes |

Whole sequence ran against one server without restart (full lifecycle incl.
APC reuse). The post-128K continuation at 111.8 vs fresh-server 168 is noted
in `perf/baseline_status.md` as unexplained (suspect KV pool occupancy).

Previous "best" numbers (82.4 no-aux, 106.7 inner-serial) were poisoned runs
and are obsolete. Mean spec acceptance length is now ~3.5-3.7 (was ~1 while
poisoned). Raw JSONs: `perf/results/2026-08-09/dsv4-a100-tp2-kv-capacity/
control/clean-v33-*.json` (plus `sentinelfix-v32-*`, probe-era logs, and GPU
coredumps under `../coredumps/`).

The attention inner-overlap diagnostic switches
(`VLLM_DSV4_OVERLAP_INDEXER`, `VLLM_DSV4_OVERLAP_MLA_COMPRESSOR`,
`VLLM_DSV4_OVERLAP_INDEXER_INNER`, plus the pre-existing
`VLLM_DSV4_INNER_ATTENTION_OVERLAP` and `VLLM_DSV4_AUX_STREAMS`) remain,
default-on/no behavior change. The `VLLM_DSV4_MLA_DEBUG_SYNC` /
`VLLM_DSV4_ATTENTION_DEBUG_SYNC` host-sync diagnostics were removed.

## Quant strategy and profiles (2026-08-09 late)

Per user direction: A100 serves the **Q4K-tail hybrid** (Q4_K experts on
layers 37-42, IQ2_XXS/Q2_K elsewhere, 90.9 GiB); IQ2_XXS is the MacBook
quant. Profile changes (see `perf/optimization_status.md` for measurements):

- `dsv4-2` / `dsv4-4`: default quant now `Q4K-tail`. `registry.py` supports
  per-quant `quant_overrides`; the qualified 19 GiB TP2 KV budget stays with
  IQ2_XXS, hybrid runs a provisional 14 GiB (needs the 128K lifecycle
  qualification pass).
- New `dsv4-4-mxfp4` (MXFP4 on 4 GPUs) and `dsv4-8` (MXFP4-only TP8;
  TP4xDP2 is the alternative to benchmark).
- Hybrid baselines (exact): TP2 168.5 c1 / 185.5 c8; TP4 174.2 c1 /
  150.7 @12K / 416.4 c8. Hybrid TP4/TP2 scaling: 1.03x batch-1, 1.69x @12K,
  2.24x @c8.
- A100 MXFP4 experts had no fused path (32 tok/s TP4 c1). New fused decode
  kernels in `csrc/quixicore/quant/dsv4_mxfp4_moe_ampere.cuh` + op
  `ggml_dsv4_moe_a8_mxfp4` (correctness-tested in
  `tests/kernels/test_dsv4_mxfp4_moe.py`) lift `dsv4-4-mxfp4` to
  111.3 tok/s c1 (3.4x) and 75.0 @12K (2.6x); c8 stays at 27.1 because
  verify batches wider than 8 tokens still take the generic MMQ route -
  widening the fused path / MXFP4 MMQ tiles is the next kernel task, then
  `dsv4-8` TP8 vs TP4xDP2.

## 2026-08-10 completion pass

Everything from the quant-strategy discussion is implemented, tested, and
profiled (`perf/optimization_status.md` 2026-08-10 entry, baselines promoted
in `perf/baseline_status.md`):

- Hybrid TP2 KV budget qualified at **13 GiB** through the full 128K
  lifecycle (14 GiB crashed at 128K cold prefill; profile + tests updated).
- MXFP4 fused route widened to verify batches
  (`VLLM_GGUF_DSV4_MXFP4_ROWS`, default 64): c8 27.1 -> 98.6 agg, c1/12K
  unchanged. Remaining MXFP4 gap vs hybrid is prefill (generic MMQ tiles).
- `dsv4-8` finalized as **TP8**: 167.6 c1 / 117.4 @12K / 148.2 c8 agg =
  1.50-1.58x over `dsv4-4-mxfp4` (meets the >=1.5x gate). TP4xDP2 fails to
  initialize: DSV4's router `is_padding` mask is not DP-padding aware
  (`csrc/libtorch_stable/moe/topk_softplus_sqrt_kernels.cu:782`); DP
  enablement is an open item and its c8 comparison is unmeasured until then.

## 2026-08-10 DP enablement + throughput matrix

- DSV4 DP-padding FIXED: `_get_padding_mask` in both fused-topk router
  modules now hands out the mask only at exact width match (DP's naive
  dispatch all-gathers hidden+logits across ranks; the local mask cannot
  describe the gathered batch). TP4xDP2 boots and serves exact.
- Full {mxfp4, hybrid} x {TP4, TP2xDP2, TP8, TP4xDP2, TP2xDP4} matrix
  measured (table in `perf/optimization_status.md`). Winners encoded in
  profiles via `quant_overrides`:
  - 8-GPU total throughput: **hybrid TP4xDP2, 567.9 tok/s c8 agg**
    (`dsv4-8 --quant Q4K-tail`).
  - Single stream: **hybrid TP8, 329.5 tok/s c1** — despite losing the
    fused IQ2 path at per-rank intermediate=256; extending the fused
    kernels to 256 would lift TP8 further (open kernel item).
  - MXFP4 stays TP8 on `dsv4-8`; `dsv4-4` stays TP4.
  - mxfp4 TP2-shards don't fit 80 GB; hybrid TP2xDP4 fails engine init
    (illegal access in the DP4 dummy run, distinct from the fixed bug) and
    is marked illegal pending investigation.
- Follow-ups queued: dsv4-q4ktail-2 with FULL_DECODE_ONLY graphs (+12% c1
  suspected), TP2xDP4 init crash, MXFP4 prefill tiles.

## 2026-08-10 late: fused-256, concurrency curves, hot methodology

- Fused IQ2 at the TP8 shard (intermediate=256) is DONE: the silent
  512-instantiation fallback in `launch_q2_k_down_sum_repacked_topk` was
  the crash; explicit 256 branch + idle-lane guard added, CA-owned
  pending-down pinned to 512/1024. Harness accepts c{1,2,4,8}.
- MEASUREMENT RULE (learned the hard way): report APC-hot steady state
  (second of two identical runs); cold-vs-hot differs by >2x and KV-pool
  size changes (e.g. capture 64) masquerade as kernel regressions by
  evicting APC.
- Hot steady state (hybrid): TP4 519 c4 / 417 c8; TP8 680 c4 / 205 c8;
  TP4xDP2 282 c4 / **926 c8** (best on the box).
- CLIFF FIXED (user-prompted): the per-engine c8 collapse was 48-token
  verify batches running eager past capture 32. `dsv4-8` a100 now ships
  `max_cudagraph_capture_size: 64` (list gains 40/48/56/64): hybrid TP8
  hot c8 205 -> 465 (2.3x), mxfp4 TP8 148 -> 275 (1.9x), TP4xDP2 holds
  ~858-926, c4 unchanged. Residual TP8-vs-DP2 gap = the >=256 routed-row
  wide-layout switch (288 rows at c8) + DP2's doubled aggregate KV; a c6
  probe isolates the former if TP8 is to challenge 926.

## Open items

1. **NaN origin (open bug):** one v31-era run showed an entirely-NaN logits
   tensor at the first post-prefill verify. Current evidence says NaNs were
   downstream of the token-129280 poisoning (OOB embedding-row read produces
   garbage activations), and clean v32/v33 runs show zero NaN/OOV incidents —
   but the v31 warmup all-NaN is not fully explained. If quality issues or
   token-0 samples appear, re-run with `VLLM_DSV4_HASH_ROUTER_DEBUG=1` and a
   logits NaN probe in `RejectionSampler.__call__`.
2. **CPU APC offload:** unchanged from before — do not enable
   `kv_offloading_size` yet (implementation was removed in `81e7c5927`);
   restore only the needed pieces after the GPU baseline is promoted.
3. **MXFP4 prefill tiles: DONE 2026-08-10.** `moe_mxfp4_mmq_v2`
   (csrc/quixicore/quant/dsv4_mxfp4_mmq_ampere.cuh, int8 mma.sync 128x64
   tiles via the mmq_v2 machinery, env VLLM_GGUF_MXFP4_MMQ_V2 default on,
   alignment block for type 39 widened to 64): kernel 8.8-57x vs dp4a;
   e2e mxfp4-4 12K 76.6->110.6/115.8, c8 ~110->208/200; mxfp4-8 12K
   117->161/163, c8 275->302/328, all exact. Also fixed a latent moe_q
   bug (activation-scale gather only correct when mmq_x == nwarps).
   Remaining levers recorded in the notebook: cp.async + SoA repack,
   decode-gate crossover sweep. (Fused SwiGLU epilogue and permuted
   segments: DONE, see item 7.) QuixiCore-CUDA port DONE: commit
   e866bf16, kernels/quant/mxfp4_moe_ampere.cuh + standalone harness
   (fused decode + mma tile + segmented pipeline, all PASS on A100).
4. **TP2 post-128K dip** (168 -> 94.5 after a 1M-scale context): TP2-only —
   TP4/TP8 hold their fresh-server band post-128K, supporting KV-pool
   occupancy as the cause. Diagnose via KV-pool state, not kernels.
5. **DP prefix-affinity routing:** DP round-robin defeats APC for repeated
   long prefixes at low concurrency (hybrid-8 128K cold 3.0 tok/s c1).
   Only matters if c1 long-context on the throughput tier ever matters.
6. **dsv4-q4ktail-2 FULL_DECODE_ONLY graphs:** suspected +12% c1; unmeasured.
7. **QuixiCore-XPU code-review ideas: IMPLEMENTED 2026-08-10.**
   - Segmented MoE + fused SwiGLU+Q8_1 epilogue: DONE
     (dsv4_mxfp4_seg_ampere.cuh, op ggml_dsv4_moe_a8_mxfp4_seg, env
     VLLM_GGUF_DSV4_MXFP4_SEG default on, J16 threshold env
     VLLM_GGUF_DSV4_SEG_J16_ROWS=1536). Cold-c8: mxfp4-4 +33%, mxfp4-8
     +18%; decode stages par; capture-safe static grids, deterministic
     reduce. Notebook has the full entry.
   - NaN-guard audit: DONE -- six paged_attn_v2 reducer guard sites
     upgraded to `!(mp > NEG_INF)` (NaN partial degrades to empty).
   - Hybrid IQ2/Q2_K seg tiles: DONE 2026-08-10
     (dsv4_hybrid_seg_ampere.cuh, op ggml_dsv4_moe_a8_iq2_seg, measured
     crossover gate VLLM_GGUF_DSV4_IQ2_SEG_TOKENS=768 -- fused 8-wide
     pipeline keeps <768 tokens, tiles win 1.3-2x at prefill widths).
     dsv4-q4ktail-4 also ships capture 64 (hot c8 into the 500-750 band;
     mechanism = the TP8 eager-verify fix). Still open (smaller):
     test-discipline ports (oracle-from-stored-codes, memcmp cache
     contracts, bit-equal RoPE tails, worst_excess<=0); gate-768 e2e pair
     on the next qualification pass.
   - MXFP4 SoA repack + cp.async tile staging: the tile/seg loaders read
     raw unaligned 17-byte AoS; the byte-neutral repack is wired
     (ggml_dsv4_repack_mxfp4, REPACKED templates exist in every wide
     consumer now) but load-time enablement + flag threading through
     ggml_moe_a8 case 39 / the seg op is not. Expected 10-30% on the
     tile kernels; microbench before e2e.
   - Q4_K kernel family for A100 (the (12,12) pair): one effort, two
     beneficiaries -- accelerates q4ktail's 6 tail layers AND unlocks a
     dsv4-q4k-8 a100 quality tier (expected ~10-15% under MXFP4 speed at
     better quality). Quality can be evaluated today unoptimized via
     `dsv4-mxfp4-4 --quant Q4_K`. Verify-width tile routing for MXFP4
     was measured and REJECTED (fused GEMV wins below ~72 tokens).
   - Cross-platform contract watch (XPU-side bugs, do not port): XPU
     mqa_logits folds kv_scale inside the relu (our indexer_paged_logits
     placement is authoritative); XPU turboquant v2 rotated-key centroids
     look sigma-mismatched (our k8v4 unaffected); XPU all_reduce
     >=-acceptance rendezvous breaks at uint32 generation wrap (our !=
     design is immune).

## Canonical reproducer / harness

```bash
PYTHONPATH=. .venv/bin/python -m slimserve.cli dsv4-q4ktail-2 \
  --serve --host 127.0.0.1 --port 8012 -y

PYTHONPATH=. .venv/bin/python benchmarks/benchmark_dsv4_exact.py \
  --model /home/ubuntu/models/antirez-deepseek-v4-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  --served-model-name DeepSeek-V4-Flash \
  --source /home/ubuntu/ds4/tests/long_context_story_prompt.txt \
  --url http://127.0.0.1:8012/v1/completions \
  --concurrency 1 --input-tokens 1000 --output-tokens 2000 \
  --warmup-output-tokens 8 --timeout 900
```

Add `--repeat-source` for input lengths beyond ~30K tokens.

## Current Profile And Capacity

Unchanged: `dsv4-2` resolves to the IQ2_XXS artifact on 2x A100 with
`kv_cache_memory_bytes=20401094656` (19 GiB/worker), `max_model_len=1048576`,
APC on, native `fp8_ds_mla` target KV, native TurboQuant `turboquant_k8v4`
draft KV, DSpark k=5, PIECEWISE graphs (capture 32), async mHC. The planner
reports 3,646,636 logical KV tokens (~3.48 full 1M contexts).

No SlimServe server was left running; GPUs are idle.

## 2026-08-10 final: profile structure settled

DSV4 A100 profiles are now `dsv4-q4ktail-2` (TP2, 13 GiB KV, 128K-qualified),
`dsv4-q4ktail-4` (TP4), `dsv4-mxfp4-4` (TP4), `dsv4-mxfp4-8` (TP8,
capture 64). mxfp4-8 layout settled by hot pairs: TP8 (167.6 c1 / 275 c8)
beats TP4xDP2 (111 c1 / 118-271 unstable). The hybrid TP4xDP2 box record
(~858-926 tok/s hot c8) is intentionally unserved (quality-quant policy);
resurrect via mxfp4-8 --quant Q4K-tail + tp4/dp2 if ever wanted. The wide-
layout threshold was exonerated by a c6 probe; residual TP8-vs-DP2 hybrid
gap is per-engine width economics + measurement variance (acceptance
lengths swing 2.6-6.0 by text window). Old root `handoff.md` lives at
`perf/dsv4_a100_kernel_history.md` (obsolete serving numbers, valid kernel
history).

## 2026-08-10 final profile set (user-confirmed)

A100 dsv4 profiles: `dsv4-q4ktail-2` (TP2, 13 GiB KV, 128K-qualified),
`dsv4-q4ktail-4` (TP4), `dsv4-q4ktail-8` (TP4 x DP2, capture 64 -- throughput
tier, accepted at 921.8 tok/s hot c8 on the named profile),
`dsv4-mxfp4-4` (TP4), `dsv4-mxfp4-8` (TP8, capture 64 -- quality/latency
tier). All five run DSpark k=5 + TurboQuant draft KV + 1M max_model_len
(test-enforced).

## 2026-08-10: 128K lifecycle qualification — ALL FIVE PROFILES PASS

Full single-lifecycle sequence (1K/2K x2, 12K cold/hot, 128K cold/hot,
post-128K continuation; exact-token, c1, every stage `exact: true`, zero
preemptions):

| Stage (tok/s) | hybrid-4 | hybrid-8 | mxfp4-4 | mxfp4-8 |
| --- | ---: | ---: | ---: | ---: |
| 1K/2K r1 / r2 | 128.1 / 161.8 | 39.9 / 40.0 | 110.8 / 110.8 | 164.9 / 164.5 |
| 12K cold / hot | 152.6 / 155.2 | 26.9 / 37.3 | 76.6 / 75.5 | 117.1 / 116.5 |
| 128K cold / hot | 94.8 / 98.3 | 3.0 / 26.4 | 55.7 / 57.2 | 80.4 / 81.5 |
| post-128K | 171.4 | 40.0 | 110.5 | 165.0 |

hybrid-8's c1 numbers are the DP2 characteristic (~40 ceiling: one active
replica + per-step DP coordination; 128K cold 3.0 = round-robin sending the
timed request to the un-warmed replica → full prefill in the timed window).
Its service point is c8 (921.8 hot). Post-128K dip is TP2-only. Details:
`perf/baseline_status.md` (qualified table) and `perf/optimization_status.md`
(2026-08-10 lifecycle entry). Raw:
`perf/results/2026-08-10/dsv4-lifecycle-qual/`. GPUs left idle, no servers
running.

## 2026-08-10: unified profile naming

Profile ids now follow `<model>-<quant>-<gpus>` on every platform; a profile
lists exactly the platforms it is validated on and refuses elsewhere. Quant
tags: xxs=IQ2_XXS(-Q2_K), q4ktail=Q4K-tail, mxfp4=MXFP4, q4k=Q4_K, q2k=Q2_K.
Renames: dsv4-1 -> dsv4-xxs-1 (mi300x+metal; absorbed dsv4-mac, whose stale
pre-split Metal config was dropped in favor of the measured M5 Max one),
dsv4-2 -> the mi300x side of dsv4-q4ktail-2, dsv4-4 -> the mi300x side of
dsv4-mxfp4-4, dsv4-8 -> dsv4-q4k-8, glm52-2/4/8 -> glm52-q2k-*,
glm52-mac -> glm52-xxs-1, k3-6/8 -> k3-xxs-*. Merged-config parity with the
old profiles was verified per (profile, platform) before the switch.
