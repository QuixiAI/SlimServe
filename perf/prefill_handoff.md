# CLEANUP PHASE PLAN (2026-08-14, session 5) — EXECUTED in session 6, see STATUS UPDATE 8 — read this, then STATUS UPDATE 8

Both campaigns are CLOSED at measured ceilings: decode step 118.4-118.9
ms (plateau defended in the capstone entry), prefill 2048 at 5.522 s =
370.9 tok/s = 134% of ds4. The remaining job is to turn +13,777/-652
lines across 50 modified + 25 untracked files into a clean,
domain-driven, commit-ready tree. Cleanup is a ZERO-NUMERIC-CHANGE
class: nothing here may alter serving output — the full gate suite
(section 5) must pass BIT-EXACT at the end.

## 1. Ground rules

- Explain the per-file plan to the user before editing (standing rule).
- Commit ONLY when the user asks. Eric Hartford sole author. No
  co-author or assistance trailers; no automated-assistance mentions
  in commit messages.
- Learnings live in `perf/` (they are already recorded:
  optimization_status.md, baseline_status.md, the campaign docs, this
  file). Code comments state constraints the code cannot show —
  nothing narrative, nothing historical.
- When deleting a documented negative, keep at most a one-line pointer
  into the notebook where a future editor would otherwise re-try it
  (e.g. "no BM=128 twin: see optimization_status 2026-08-14 v12").

## 2. Inventory (what the uncommitted lines are)

- Kernels, modified: qgemm, qgemv, mla, paged_attn_v2, decode_linear,
  indexer, tile/dequant, tk_launch.h, qc_metal_serving.mm.
- Kernels, untracked NEW dirs: moe/moe_mm_id (production prefill MoE),
  serving/{dsv4_mhc, dsv4_router, moe_finalize, rms_norm, swiglu}
  (production), serving/prefetch (wave-14 RETIRED — delete),
  serving/probe (qc_probe instrumentation — delete unless a documented
  diagnostic case is made).
- Python serving path, modified: deepseek_v4/* (attention, compressor,
  metal.py, sparse_mla, common/ops/*), fused_moe, gguf quant path,
  platforms/metal*, v1/worker/gpu/* (model_runner, sampler, spec
  decode, buffers), metal_worker.
- Python, untracked NEW: metal_indexer.py + quixicore_metal_ops.py
  (production); metal_tape.py (PRODUCTION decode step tape — verify,
  do not mistake for a diagnostic); metal_phaseprof.py,
  metal_syncprof.py, metal_pysampler.py (opt-in profilers — decision
  item, section 3).
- Tests, untracked: 7 kernel oracles (KEEP — they are the gate suite).
- perf docs + campaign docs + this handoff (KEEP).
- benchmarks/benchmark_dsv4_exact.py (review the diff; note the
  client-side tokenizer-deadlock caveat is documented in the notebook).
- slimserve/profiles.json (max_num_batched_tokens 2176 etc. —
  production config, KEEP) and vllm/quixicore_metal.metallib (rebuilt
  artifact; final rebuild happens at the end of cleanup).

## 3. De-slopify passes (in this order)

1. **Diff review ledger.** Walk `git diff` + untracked files
   file-by-file; write a keep/fix/delete ledger before touching
   anything.
2. **Dead code and retired experiments.** Delete: serving/prefetch,
   serving/probe, the parked q2_K w64 twins and their
   VLLM_QC_MOE_MM_W64_Q2K diag env, any wave-8-era leftovers, the
   expert-grouped w13 verify kernel if still present (wave-10-decode,
   RETIRED), fused pair+SwiGLU opt-in if the user agrees (measured
   negative twice; the notebook already records it). CLAUDE.md allows
   intentionally retained documented diagnostics — each survivor needs
   an explicit reason in the ledger.
3. **Env-switch audit.** 36 VLLM_QC_* switches exist. Classify:
   (a) production config — keep, document in one place (e.g.
   LONGCTX_SYNC, DISABLE_NATIVE, MHC_METAL_MAX_TOKENS,
   ROUTER_SOFTPLUS_MODE if load-bearing);
   (b) reversion sentinels for ULP-class waves whose sha changed —
   keep the minimal set (MLA_PREFILL_FA_MMA at minimum) and document;
   (c) gate-era kill switches for BIT-EXACT-proven waves (MMQ_WIDE,
   MOE_MM_W64, MHC_DOTS_TG, MLA_PREFILL_FA, INDEXER_TOPK_PREFILL,
   MOE_SOA, MOE_SUM6, COMPRESS_FRONT, ...) — delete switch + dead
   branch; bit-exact gating means the old path proves nothing ongoing;
   (d) census/geometry/experiment switches (OP_CENSUS, IDS_CENSUS,
   *_GEO, MOE_MR, MEMO_*, OUT_RING, PREFETCH, MOE_GROUP, ...) —
   delete with their code unless production-load-bearing.
   Present the classification to the user before deleting (b) or
   anything ambiguous.
4. **Profiler modules.** phaseprof/syncprof/pysampler are cleanly
   opt-in and earned their keep during attribution. Recommendation:
   keep syncprof + phaseprof as documented diagnostics (one paragraph
   in perf/perf.md), delete pysampler if redundant — user call.
5. **Comment hygiene.** Remove narrative/history comments (rejected-
   experiment stories, "UPDATE:"-style notes, wave numbers) — the
   notebook holds them. Keep constraint comments (alignment
   requirements, uniformity assumptions like the q2_K il<2 hoist,
   layout contracts, the PSO thread-cap warning if BM128 ever
   returns). Remove the first-dispatch stderr breadcrumbs or gate them
   behind one debug env.
6. **Structure/naming.** Domain-driven layout: kernel twins deduped
   where provably identical (template or shared impl), tk_launch
   grouped by domain, qc_metal_serving.mm wrappers consistent
   (ggml_mul_mat_a8's wide/narrow split should read as one clear
   policy function), Python ops modules named for what they do, no
   dead imports. No behavior changes — textual/structural only.
7. **Wedge root-cause fix (optional but recommended).** The
   first-multi-chunk-request wedge (STATUS UPDATE 7) has a candidate
   fix in vllm/v1/worker/gpu/async_utils.py / metal_compat (cross-
   stream copy_event never signals when the machinery is hit cold).
   If fixed: prove it by a cold boot running primer -> 2500 DIRECTLY
   (the current deterministic wedge repro), then the full gate suite.
   If not fixed: it stays an ops protocol (ramp boots), documented in
   STATUS UPDATE 7 and the notebook — do not half-fix it.

## 4. Commit structuring (proposal — commit only when user asks)

Domain-sliced commits, each buildable: (1) metal kernels + tk_launch +
qc_metal_serving.mm + metallib; (2) python serving path (deepseek_v4,
fused_moe, worker, platforms); (3) profiles.json + benchmark harness;
(4) tests; (5) perf docs. Adjust granularity with the user.

## 5. Final gate suite (must pass before any commit)

- Rebuild: metallib (`xcrun metal -std=metal3.1 -O2 -I include/metal
  $(find kernels -name '*.metal') -o <repo>/vllm/quixicore_metal.metallib`
  from csrc/quixicore/metal) + extension (clang++ one-liner, STATUS
  UPDATE 6 §7-8). Python is `.venv/bin/python` from repo root.
- Oracles: tests/kernels/test_metal_moe_mm.py (40),
  test_metal_compress_front_c128.py (6), test_metal_prefill_fa.py,
  test_metal_indexer_topk_prefill.py (+ moe_group/moe_soa/moe_sum6
  while their code remains).
- Serving (BOOT RAMP PROTOCOL mandatory — see STATUS UPDATE 7):
  primer, then 8-tok before any multi-chunk request. Gates, all
  BIT-EXACT: 8-tok db2846cf721b 7/10/2; off1-2000 3fc700d9818b
  1496/2535/507, step ~118.4-118.9 ms; 2500 anchor dd5c1c87fe60.
  Walls within noise of UPDATE 26: 512 ~2.0-2.2 | 1000 ~3.1 | 2048
  <= ~5.55 (>=369 tok/s) | 3000 ~9.6-9.75. Use the server-side gate
  tooling (gate.py/probe.py — reconstruction recipe in the v12
  RE-GATE notebook entry; the harness client can deadlock).

--- current status below ---

# STATUS UPDATE 8 (2026-08-14, session 6) — CLEANUP WAVE 1 COMPLETE AND RE-GATED BIT-EXACT; ready to commit on request — read this first

The cleanup plan above is EXECUTED (wave 1). The working tree is the
ship tree, fully gated:

- All three serving shas BIT-EXACT on a fresh ramped boot (boot_final,
  after the review pass below): 8-tok db2846cf721b 7/10/2; off1-2000
  3fc700d9818b 1496/2535/507 (65.5 s); 2500 anchor dd5c1c87fe60
  (3.532 s, no wedge). Walls: 512 2.218 | 1000 3.102 | 2048
  5.512/371.6 (matches UPDATE 26 within noise) | 3000 9.649. All six
  kernel oracles pass; metallib 12.64 -> 12.18 MB.
- SECOND-PASS REVIEW (three independent reviewers over the final diff)
  fixed: dead qc_moe_mm_id_iq2_xxs kernel deleted (165 lines), mhc
  ARGS_PRE macro + always-true FUSED template collapsed, unused `lane`
  in mla_prefill_fa_mma, tape structural checks that self-disabled the
  tape on the SoA layout, plus ~a dozen comment/naming/pybind nits
  ("CLEANUP REVIEW PASS" notebook entry). TAPE KNOWN ISSUE surfaced:
  mode-2 verify mismatches on current routes (tape body unvalidated
  since 2026-08-11; documented in metal_tape.py; mode 1 untrusted).
- What changed: dead experiments deleted end-to-end (prefetch/probe
  kernels, pysampler, fused SwiGLU tile, w64 q2_K twins, _grp twin,
  decode_linear bricks, iq2-SoA family, ~28 geometry PSOs, censuses,
  tape bisect trio); env switches 37 -> 12; router ABI pre-softplus-
  only; MmqPlan struct + launcher grouping + segment-id dedupe;
  comment hygiene everywhere; tests hardened (SPDX, optional-pytest,
  real gates). One latent bug fixed (const char* vs "q2_K" pointer
  compare in the .mm tape).
- ONE RETENTION (new finding, documented in the code): removing the
  phaseprof brackets + marshalling-memo conditionals from
  vllm/models/deepseek_v4/{compressor,metal}.py makes even RAMPED
  multi-chunk requests park at completion (MPSEvent::synchronize) —
  boot-level bisect in the notebook "CLEANUP PHASE" entry. Those two
  files keep their pre-cleanup structure (comment-only edits,
  AST-identical, verified); metal_compat.py documents the race. The
  async-output event path root-cause fix is a follow-up campaign.
- Machine-boot variance note: walls on a given boot can sit ~3-5% off
  the band for ALL builds (A/B-proven earlier today); judge against
  the band across boots. Baseline UPDATE 26 unchanged; see
  baseline_status UPDATE 27.
- Wave 2 (separate PR recommended): .mm helper dedupes + kernel
  template dedups (list in the notebook entry). NOT started.
- Artifacts: perf/results/2026-08-14/cleanup_gate/. Gate tooling +
  tree snapshots (pre-cleanup and ship patches) in the session
  scratchpad (ledger.md there has the full disposition table).
- NOT COMMITTED — commit only on explicit request (Eric Hartford sole
  author, no co-author/assistance trailers).

# STATUS UPDATE 7 (2026-08-14, session 5) — v12 FULLY RE-GATED: 370.9 tok/s (134% of ds4); wedge root cause REVISED — superseded by UPDATE 8 (cleanup done)

**Campaign standing: 2048-token prefill 5.522 s = 370.9 tok/s = 134%
of ds4's 277.** All v12 gates passed BIT-EXACT post-restart on
boot_v12b (the restart did not roll the trajectory): 8-tok
db2846cf721b 7/10/2; off1-2000 3fc700d9818b 1496/2535/507, step
~118.9 ms; 2500 anchor dd5c1c87fe60, 3.541 s; breadcrumb liveness.
Walls: 512 2.110-2.175/235-243 | 1000 3.094/323.2 | **2048
5.522/370.9** | 3000 9.639-9.736/308-311. baseline_status UPDATE 26
is the standing baseline. NOTHING IS BLOCKED.

**WEDGE ROOT CAUSE REVISED (supersedes UPDATE 6):** the two-chunk
wedge was NEVER xctrace poisoning and NEVER the build. It is a
deterministic FIRST-MULTI-CHUNK-REQUEST ordering failure (2/2 wedge
repro post-restart on primer->2500-direct boots; 1/1 clean on the
ramped boot; identical MPSEvent::synchronize park, GPU idle, no CB
error, no kernel-log GPU fault). Park site: async-output copy_event
(AsyncOutput, vllm/v1/worker/gpu/async_utils.py — cross-stream
wait_stream/record; async scheduling is on by default). Full evidence:
optimization_status "v12 RE-GATE" entry.

**BOOT PROTOCOL (believed fixed; ramp stays mandatory until the repro
re-runs):** a root-cause fix landed — on MPS the async-output copy and
its completion event now stay on the producing stream
(vllm/v1/worker/gpu/async_utils.py; regression test
tests/kernels/test_metal_async_output.py with a poisoned stub
stream) — but the proof this document mandates in item 7 above (cold
boot running primer -> 2500 DIRECTLY, then the full gate suite) has
NOT been re-run, so per that item the ramp remains the ops protocol:
after startup: (1) tiny primer (4-5 tok in, 8 out); (2) a ~1000-token
single-chunk request WITH decode steps (the 8-tok gate qualifies)
BEFORE any multi-chunk (>2176-token) prefill. Skipping (2) wedged the
boot deterministically on its first multi-chunk request. Note also:
DraftTokensHandler (vllm/v1/worker/gpu/spec_decode/utils.py) and the
structured-outputs copies keep the same cross-stream shape the fix
removed from async-output — open follow-up.

Gate tooling (the harness client can still deadlock in HF tokenizers):
server-side reimplementation lives in the session scratchpad as
gate.py (sha gates, exact harness prompt construction via /tokenize +
/detokenize) and probe.py (streaming-TTFT walls, phase-distinct
cursors mod L=1085). /tmp is wiped by machine restarts — recreate from
the descriptions in the v12 RE-GATE notebook entry if lost.

Remaining extraction levers (unchanged, all optional): long-context
(>2048) MMA-FA for the topk branch (3000 marginal 4.4 ms/tok vs 2.3
dense — biggest), FA multi-request, small-n host floor (~0.8 s).
NEXT: user-confirmed CLEANUP PHASE in a new session (write its plan to
handoff.md first; review ~15k+ uncommitted lines across ~50 files,
quarantined negatives, diagnostics; wedge root-cause fix candidate).
Commit ONLY when the user asks. Eric Hartford sole author; no
assistance trailers or automated-assistance mentions.

--- previous status below ---

# STATUS UPDATE 6 (2026-08-14, session 4) — 369.5 tok/s (133% of ds4); BOX WEDGED, MACHINE RESTART REQUIRED — superseded by UPDATE 7 (wedge story revised)

**BLOCKER: the box needs a machine restart before any serving work.**
Post-xctrace CB/event poisoning (the documented Instruments-attach
failure mode): every boot since ~11:52 wedges — engine main thread
parks forever in torch Event.synchronize (async-output copy_event,
stock PyTorch, NOT a qc kernel). Reproduced across 5 fresh boots;
single-chunk 2048 prefill runs at full speed on the same boots where
two-chunk 2500 wedges; wedges persist with VLLM_QC_MMQ_WIDE=0 and with
VLLM_QC_MLA_PREFILL_FA=0 + VLLM_QC_INDEXER_TOPK_PREFILL=0 — the build
is exonerated (the identical build ran the 2500 anchor cleanly at
11:36 pre-xctrace). Full evidence: optimization_status v12 entry.

Session-4 waves (all uncommitted, on top of UPDATE 5):
- **v8a ablations**: w13 iq2_xxs is ~10-15% above its structural floor
  (dequant 2.7 ms, A-stores 3.8, MMA+loads floor 27.9 of 34.4) — CLOSED.
  Two rejected restructures documented (transposed loads +5.6 ms; load
  guard +4.7 ms). w2 q2_K: dequant 5.8 (SoA), floor 19.5, cull rejected.
- **v9 (RETAINED, bit-exact class)**: q2_K dequant load-shaping — per-
  block scalar hoist behind uniform il<2 + uchar4 qs loads. w2 SoA
  25.33 -> 24.16 ms. Oracle 40/40.
- **v10 (RETAINED, bit-exact)**: qgemm_wide_t_q8_0 — 64x64 tile +
  transposed [M,N] store for large-M q8_0 + host de-fluff (the aten
  transpose fluff was ~21.9 ms/call at wq_b!). wq_b op 31.0 -> 9.37 ms.
  VLLM_QC_MMQ_WIDE=0 reverts. GATED on v8 boot: all three shas
  identical + breadcrumb liveness + walls: **512 2.001/255.9 | 1000
  3.189/313.6 | 2048 5.543/369.5 | 3000 9.692/309.5** (fresh disjoint
  id-window TTFT probes; APC poisons repeat windows).
- **v11 attribution**: 2048 prefill is GPU-BOUND (~130 ms/layer; CBs
  chain gapless). xctrace RETENTION GOTCHA: attach traces keep only the
  last ~2 s — check first-interval timestamp before concluding idleness.
  The +0.8 s small-n "fixed" cost = per-layer host floor (only shows
  when GPU/layer < host/layer; hidden at 2048).
- **v12 (code RETAINED, serving re-gate PENDING post-restart)**:
  wide-gate tier rows>=512 && N>=8192 (wq_b M=1000: 6.05 -> 4.49 ms;
  gives 1000-token prompts + 3000 chunk-2 the v10 win). BM=128 twin
  REJECTED (wedge-risk suspect at the time; -0.5..-0.9 ms/layer if ever
  revisited WITH a maxTotalThreadsPerThreadgroup>=128 assert).
  Partial v12 gates already passed pre-wedge: 8-tok + off1-2000 shas
  identical, step 118.4 ms, new-tier breadcrumb fired.

**After machine restart**: sysctl iogpu.wired_limit_mb=122880; boot
dsv4-xxs-1; primer; then (1) long-ctx anchor — server-side repro (the
harness client can deadlock in HF tokenizers pre-request; use
/tokenize + /detokenize + text completions, ignore_eos, warmup 1 then
64 — expect dd5c1c87fe60), (2) walls 512/1000/2048/3000 via fresh
disjoint id windows (expect >= 369.5 at 2048; 1000/3000 should beat
v10 via the tier), (3) update baseline_status UPDATE 25.

Remaining measured pools at 2048 (all near floors): FFN mm ~60 ms/layer
(w13/w2 CLOSED), qc_mmq ~28 (within 8-27% of MPS fp16 ceiling), FA ~16,
mhc/misc ~15. Bigger levers if resumed: long-context (>2048) MMA-FA for
the topk branch (3000 marginal cost is 4.4 ms/tok vs 2.3 dense), FA
multi-request, small-n host floor (~0.8 s/request below ~1500 tokens).

--- previous status below ---

# STATUS UPDATE 5 (2026-08-14, session 3 final) — 344.6 tok/s, 124% of ds4

Final session state (all gated, all uncommitted):
- **2048-token prefill 5.943 s = 344.6 tok/s vs ds4's 277 = 124%.**
  Campaign 25.911 -> 5.943 s (4.36x). 1000: 317.4 | 3000: 303.0 |
  512: 235.5. Decode step 119-120 ms (start: 122.6-124.2).
- Waves since UPDATE 4: v6 single-chunk scheduling (profiles.json metal
  max_num_batched_tokens 2048 -> 2176; killed a 464 ms scheduler-boundary
  + 128-token tail chunk), v7 native prefill indexer top-k (robustness;
  eager python chain at ctx > 2048 replaced; walls-neutral).
- Serving baselines (all defaults): off1-2000 sha 3fc700d9818b
  1496/2535/507; 8-tok db2846cf721b 2/10/7; long-ctx anchor 2500-in/64-out
  offset-0 dd5c1c87fe60. Sentinels: VLLM_QC_MLA_PREFILL_FA_MMA=0 restores
  ec0cc6c5908e 1520/2410/482 bit-exactly; VLLM_QC_MOE_MM_W64=0,
  VLLM_QC_MHC_DOTS_TG=0, VLLM_QC_MLA_PREFILL_FA=0,
  VLLM_QC_INDEXER_TOPK_PREFILL=0 each revert their wave.
- Oracles on the final build: tests/kernels/test_metal_moe_mm.py (40),
  test_metal_compress_front_c128.py (6), test_metal_prefill_fa.py,
  test_metal_indexer_topk_prefill.py — all pass.
- Quarantined negatives (documented in the notebook, do not retry blind):
  fused pair+swiglu (41.1 ms vs 34.7 unfused even after the occupancy
  rebuild; opt-in), q2_K w64/aliasing (regress SoA), tile-level-softmax
  and dual-candidate attention walks (barrier-bound / neutral).
- Remaining pools (measured): FFN mm ~2.6 s (w13 34.4 ms at 53% peak,
  w2 25.3 ms — resists occupancy fixes), qc_mmq Q8 ~1.2 s (wq_b ~14,
  o_proj ~9.5 at ~66%), FA residue/misc ~0.7 s, 512-width fixed
  overhead. Next big lever if resumed: MMA-FA for the long-context topk
  branch + FA multi-request extension; or the w13 ceiling push.
- CLEANUP PHASE still required before any commit (user-confirmed review
  of all uncommitted lines — now ~15k); commit only when the user asks;
  Eric Hartford sole author, no assistance trailers.

# STATUS UPDATE 4 (2026-08-13, session 3) — superseded by UPDATE 5

**2048-token prefill: 6.584 s = 311.1 tok/s vs ds4's 277 = 112%.** All
five session-3 waves landed and gated (v3a/v3b/v4a bit-exact; v5 MMA FA
ULP-gated with bit-exact reversion sentinel). Campaign total 25.911 ->
6.584 s (3.94x). Decode untouched (paired step 120.2 ms; 8-tok sha
db2846cf721b 2/10/7 STILL unchanged). New off1-2000 baseline:
3fc700d9818b 1496/2535/507 (FA on); VLLM_QC_MLA_PREFILL_FA_MMA=0
restores ec0cc6c5908e 1520/2410/482 bit-exactly. Baselines:
perf/baseline_status.md UPDATE 23; notebook entries dated 2026-08-13.

Remaining extraction pools at 6.58 s (largest first): FFN mm ~2.8 s
(w13 34.4 ms + w2 25.3 ms + glue per layer; w2 q2_K resists occupancy
fixes — see notebook), qc_mmq Q8 GEMMs ~1.2 s (wq_b ~14 ms, o_proj
~9.5 ms at ~65% peak), attention residue ~0.7 s, mhc ~0.35 s, misc
eager ~0.5 s. Long-context (>2048) still runs v4a + eager python topk —
needed before any long-context claim. FA covers single-request prefill
steps only (multi-request extension via query_start_loc runs).

# STATUS UPDATE 3 (2026-08-13, session 3) — superseded by UPDATE 4

Score: 2048-token prefill 8.096 -> 7.679 s (253 -> 266.7 tok/s, 96.3% of
ds4's 277) via three more gated waves; decode untouched (8-tok
db2846cf721b 2/10/7, off1-2000 ec0cc6c5908e 1520/2410/482 both
BIT-IDENTICAL throughout). Walls: walls_v3a.log / walls_v4a.log.

New waves (all uncommitted, notebook entries dated 2026-08-13):
- v3a iq2_xxs dual-half 64-slot tile + dead-block cull + shmem aliasing
  (w13 45.3->34.4 ms; the fix that mattered was 18.6 KB -> 10.4 KB
  threadgroup memory => 3 TGs/core, barrier overlap; q2_K w64 REGRESSED,
  gated to iq2 only; VLLM_QC_MOE_MM_W64=0 reverts).
- v3b mhc_pre dots threadgroup staging (1.85->1.15 ms, bit-exact,
  VLLM_QC_MHC_DOTS_TG=0 reverts).
- v4a staged sparse-attention prefill twin (51.8/20.2 vs 53.7/20.6
  ms/layer, bit-exact, VLLM_QC_MLA_PREFILL_FA=0 reverts). Two REJECTED
  restructures documented in the notebook: tile-level softmax (4x SLOWER
  — barrier-bound at 2 TGs/core, register spills with runtime bounds)
  and barrier-free dual-candidate ILP walk (neutral). Conclusion: the
  scalar walk is ALU-issue-bound at ~52 ms; only MMA density moves it.

## IN FLIGHT: v5 dense-causal MMA FA (the ds4-decider, ~1.0 s expected)

Design (verified against build_comp_tables/build_swa_tables in
vllm/models/deepseek_v4/metal.py):
- Short-context dense-causal case: compressed_slots rows are a shared
  per-request prefix (row t masks -1 past len(t) = (pos+1)//cr); SWA is a
  band over the raw-position axis (lo=pos+1-len, hi=pos+1). Long-context
  topk lists stay on the v4a kernel.
- Pass 1 dequant scratch: decode candidate slots (584-B fp8 slots: 448
  fp8 dims scaled by exp2(e-127) from 8 exponent bytes at +576, dims
  448-511 bf16 rope at +448) -> contiguous half [cands x 512]; same for
  the SWA band positions from the raw cache. K == V == this 512 latent.
- Pass 2 FA kernel per (8-token q-tile, head): 128 threads / 4 sgs.
  Q staged half [8x512] (8 KB). S tile [8x32] via MMA with k-dim split
  across 4 sgs (partials to TG, 4 KB fp32), reduce+scale+causal
  mask+online stats on one sg, P half [8x32] in TG; PV via MMA with each
  sg owning a 128-dim V slice, O frags in registers (16 frags = 32
  regs/lane), alpha rescale via diagonal-matrix MMA (ggml trick, see
  kernel_flash_attn_ext in ~/Code/hy3/llama.cpp ggml-metal.metal:5925 —
  O in TG mem + diag rescale; ds4 ships this FA at dk576/dv512).
  simdgroup_load K^T/V directly from device scratch (no K/V staging!),
  ~3 barriers/block, ~13 KB TG => 2 TGs/core.
- Python: slice per-request prefill token ranges (md.prefill.chunks) to
  the FA path; decode/mixed tokens keep the old op. ULP gate class
  (P rounds to half — standard FA practice, ds4 does f16 FA): needs
  determinism x2, norm-relative oracle vs decode kernel, coherent text,
  paired walls, sha WILL change (new baseline to record).

Remaining pools after v5: w2 q2_K (25.3 ms x43 ~ 1.0 s; occupancy fixes
REGRESS it — cache-sensitive scattered dequant, see notebook), qc_mmq
Q8 GEMMs (o_proj 9.5 ms, wq_b ~14 ms, ~65% peak), eager glue ~0.5 s,
indexer eager top-k at ctx>2048 (metal_indexer.py — needed for any
long-context claim; decode-topk kernel is the template).

Box: server RUNNING on :8000 with v3b+v4a build (boot_v4a.log), primed.
Builds: metallib + extension current (clang++ one-liner in section 4 of
this doc; metallib via xcrun metal in csrc/quixicore/metal). Ops rules
unchanged. Commit only when user asks; cleanup phase first; Eric
Hartford sole author.

# PREFILL CAMPAIGN HANDOFF — dsv4-xxs-1 on M1 Ultra (written 2026-08-13)

> STATUS UPDATE 2 (2026-08-13, end of session): FIVE WAVES LANDED AND GATED.
> Prefill 77.6-81.1 -> 253-264 tok/s (3.2x); 2048-token TTFT 25.91 -> 8.10 s.
> ds4 target 277 tok/s: at 91%. Landed (all oracle+serving gated, notebook
> entries in optimization_status.md, artifacts perf/results/2026-08-13/
> prefill_mm_v1/): (1) w13 iq2_xxs tile GEMM qc_moe_mm_id_iq2_xxs; (2) w2
> q2_K tile qc_moe_mm_id_q2_K[_soa]; (3) cr=128 compressor front
> dsv4_compress_front_c128 (BITWISE vs eager; also removed decode's ~330 ms
> 128-boundary stalls -> step 119.2-119.5 ms); (4) native mhc at prefill
> widths (VLLM_QC_MHC_METAL_MAX_TOKENS default 2048); (5) map0 work queue
> (exact tile dispatch, bit-identical serving). Fused pair+SwiGLU
> (qc_moe_mm_id_iq2_xxs_swiglu) is BUILT and oracle-BITWISE vs
> tile+qc_swiglu but measured neutral-to-negative -> default OFF
> (VLLM_QC_MOE_PREFILL_MM_FUSED_ACT=1 opts in; re-A/B on top of the queue
> is untested). Current serving shas: 8-tok db2846cf721b (2/10/7,
> UNCHANGED from decode baseline); off1-2000 ec0cc6c5908e counters
> 1520/2410/482 (rolled at wave 4: mhc numerics; sentinel chain:
> MHC_MAX_TOKENS=32 -> ce6c5a586087 1584/2095/419; +PREFILL_MM=0 ->
> 4d18b4fac460 1409/2955/591). Oracles: tests/kernels/test_metal_moe_mm.py
> (40 checks), test_metal_compress_front_c128.py (6). Next targets from the
> xctrace method (metal-gpu-intervals export + label join — phaseprof
> phase totals MISATTRIBUTE via sync-drain, do not trust them): FFN block
> ~3.9 s (re-A/B fused pair; shexp qc_mmq transpose round trip at :1349),
> attention chain ~2.6 s (wq_b qc_mmq, indexer q/topk at prefill width,
> ds4 fused rope+insert refs), misc ~1 s. NOT committed: everything.
>
> STATUS UPDATE (2026-08-13, later session): WAVE-1 v1 IS LANDED AND GATED.
> qc_moe_mm_map0_{2,4,6,8} + qc_moe_mm_id_iq2_xxs implemented per §2, all
> §2.5 gates green (oracle 20/20; 8-tok sha db2846cf721b unchanged WITH the
> mm path live at prefill; off1-2000 determinism x2 sha 0adffb58c16a, step
> 123.3-124.1 ms; MM=0 sentinel reproduces 4d18b4fac460 1409/2955/591).
> Walls: 512 6.51->5.16 s, 1000 12.34->9.55, 2048 25.91->20.11, 3000
> 38.67->29.52 (~99-105 tok/s, was 77.6-81.1). See optimization_status.md
> "PREFILL WAVE 1 v1" and baseline_status.md UPDATE 22.
> FACT CORRECTIONS: (1) E=256 routed experts, NOT 64 (w13 [256,4096,1056];
> map0 sids[] sized 256; over-dispatch at 2048 tokens is ~32x empty TGs, not
> 9x — raises the priority of the work-queue item). (2) The 8-tok gate does
> NOT stay vec-path end-to-end — its 1000-token prefill takes the tile; the
> sha survived anyway (greedy top-1 robust to ULP prefill change).
> ds4 research findings (full report in the session log; ds4 refs verified):
> ds4's engine for 277 tok/s = same 64x32 mul_mm_id shape PLUS (a) map0 that
> also emits a compact (expert, r1) work queue -> exact dispatch
> (~/ds4/metal/moe.metal:7669-7689, host ds4_metal.m:31114-31142); (b) fused
> gate+up+SwiGLU*route_weight tile kernel, shared B tile, dual accumulators,
> f16 mid (moe.metal:8246-8451); (c) f16-RHS q2_K down tile GEMM
> (moe.metal:8710) + sum6 reduce; (d) attention: chunk-wide compressor GEMMs,
> fused norm+rope, ONE mixed FA pass w/ mask cached across layers/chunks +
> block-cull map (ds4_metal.m:26297, flash_attn.metal:139/:208); (e) HC fused
> to ~3 kernels/stage (dsv4_hc.metal:462/:652/:1220). Next: v2a tile w2 q2_K
> (SoA planes, b_per_slot=id mode), v2b pair+SwiGLU fusion, v2c work queue.

Start here in a fresh session. This file carries the campaign state, the
measured wave-1 baseline, a ready-to-implement kernel design, and the
standing ops rules. Companion docs: `perf/baseline_status.md` (UPDATE
21/21a/21b = decode closure), `perf/optimization_status.md` (all wave
entries incl. "PREFILL WAVE 1 BASELINE"), `perf/metal_m1ultra_
retrospective.md` (§4 do-not-redo, §7-8 cross-platform/external lessons).

## 0. Mission state

- DECODE IS CLOSED at the measured plateau 122.6-124.2 ms/step (~32
  tok/s, 7-offset mean). GPU saturated, streams at format ceilings,
  every lever measured. Do not reopen without new evidence; the do-not-
  redo list is retrospective §4.
- ACTIVE CAMPAIGN: PREFILL (task #37). Baseline this build: 77.6-81.1
  tok/s, 12.3-12.9 ms/token, FLAT 512->3000 tokens (zero chunk
  amortization). ds4 does 277 tok/s on this box; flat-tile physics
  supports ~500 tok/s class.
- AFTER the campaigns: a user-confirmed CLEANUP PHASE — full review of
  the ~11k uncommitted lines (quarantined negatives, diagnostics,
  dead branches, doc coherence) before any commit. Commit ONLY when the
  user asks. Eric Hartford is sole author: no co-author/assistance
  trailers, no automated-assistance mentions in commit messages.

## 1. Wave-1 baseline (measured 2026-08-13, artifacts in perf/results/2026-08-13/prefill_baseline/)

- Walls (clean boot, streaming TTFT, disjoint offsets to defeat APC):
  512 -> 6.506 s | 1000 -> 12.335 | 2048 -> 25.911 | 3000 -> 38.670.
- xctrace ground truth (metal-gpu-state-intervals): GPU Active 12507.6
  ms vs Idle 10.2 ms during prefill = 99.9% BUSY. Prefill is kernel-
  execution-bound, NOT host/encode-bound. ~4.6 CBs in flight.
- Phaseprof split of one 2048-token request (28.39 s bracketed, +10%
  sync inflation vs 25.9 clean; layer rows partition target_forward):
  * layer_ffn 12.26 s (43.2%) — trace encoder labels confirm chunk-width
    MoE still runs qc_moe_vec_mr_swiglu + qc_moe_vec_mr_sum per slot.
  * layer_attn 12.21 s (43.0%) — insert/compress complex ~10 s
    (attn_wqb_insert_c bracket WRAPS wq_b+kv_insert AND the parallel
    compressor; comp_full_compress nests inside). attn_mqa itself only
    1.85 s; oproj 0.96; indexer 0.60. The 128K split does NOT transfer.
  * layer_mhc 3.44 s (12.1%), layer_norm 0.10 s.
- FFN physics: decode ceilings (swiglu 0.458 + down 0.218 ms @36 slots)
  x341 slot scaling = 10.0 s/43 layers, matching the bracket => routed
  vec work ~10 s (w13 ~6.7 s, w2 ~3.2 s), rest shexp qc_mmq + router.
- Serving shapes (verified vs stream census byte math): E=64, topk=6,
  hidden=4096, inter=2048. w13 [64, 4096, 4096] iq2_xxs AoS (66 B/256w,
  4.33 MB/expert). w2 [64, 4096, 2048] q2_K SoA-repacked (84 B/256w,
  2.75 MB/expert). Flat-tile bound: ~26.6 TFLOP/chunk / ~7-8 TFLOPS
  effective simdgroup fp16 => ~2.5-3.5 s for BOTH mats vs ~10 s today.
- ROADMAP: wave 1 = flat MoE tile (43%); wave 2 = insert/compress chain
  (35%); wave 3 = mhc (12%). Wave-1 v1 (w13 only) projects prefill
  ~78 -> ~92+; +v2 (w2) -> ~105-125; waves 2-3 -> 200+.

## 2. Wave-1 v1 design (fully pinned; implement this first)

Scope v1: tile ONLY the w13 iq2_xxs GEMM (AoS, no repack — simplest,
biggest term). Output [slots, 4096] feeds the EXISTING act(out) and the
EXISTING sum6 q2_K down path unchanged. v2 tiles w2 afterwards.

### 2.1 New kernel file `csrc/quixicore/metal/kernels/moe/moe_mm_id/moe_mm_id.metal`

`#include <metal_stdlib>` + `#include "tk.metal"`, `namespace mittens`
(brings iq2xxs_grid/ksigns_iq2xs/kmask_iq2xs from dequant_tables.metal).
Port of llama.cpp kernel_mul_mm_id (+map0), NOT the tk rt/st path (tk's
`load(rt,st)` has simdgroup_load commented out; llama.cpp's hand-swizzle
+ simdgroup_load is the proven shape). llama.cpp source of truth:
`~/Code/hy3/llama.cpp/ggml/src/ggml-metal/ggml-metal.metal` — map0 @9823,
mul_mm_id @9889, dequantize_iq2_xxs @755, dequantize_q2_K @627.
(NOTE: ~/llama.cpp does NOT exist; the tree is under ~/Code/hy3/.)

Kernel A — `qc_moe_mm_map0<NE20>` (instantiate NE20 in {2,4,6,8}):
one threadgroup, one thread per expert (dispatch grid (1,1,1), threads
(E,1,1), E<=64 host-checked). Buffers: topk_ids i32 (tokens,topk) @0,
tpe u32 (E) @1, ids i32 (E*tokens) @2, tokens @3. Static threadgroup
`ushort sids[64*NE20]`. Logic = llama.cpp verbatim: cooperative stage of
ntg tokens' ids as ushort, barrier, every thread scans the chunk with
branchless `sel += (sids[j]==ide)*(j+1)`, writes `ids_e[n_all] =
(i21+t)*NE20 + sel - 1` unconditionally, `n_all += sel>0`, barrier;
final `tpe[ide] = n_all`. Entries past tpe[e] are stale garbage — never
read (phase-1 early-exit). Duplicate expert ids in one token's top-k
would corrupt `sel` — DSV4 router emits unique ids; negative ids would
be silently dropped (row never written) — the python gate requires
expert_map is None so ids are raw router output 0..63.

Kernel B — `qc_moe_mm_id_iq2_xxs`:
- Buffers: D half (tokens*topk, N) @0 | Wq uchar AoS @1 | X half
  (tokens,K) @2 | tpe u32 @3 | ids i32 @4 | N @5 K @6 tokens @7 topk @8.
- Dispatch: grid ((tokens+31)/32, N/64, E), threads (128,1,1) = 4 sgs.
- Constants: NR0=64 (weight rows), NR1=32 (tokens), NK=32, NL0=2, NL1=4,
  nl=16. K%32==0 and N%64==0 asserted at host (4096/4096 ✓) so all
  bounds tails compile away; keep the nr0/nr1 clamps anyway.
- Static threadgroup: `half sa[64*32]` (A tile, 8x8-swizzled [k][row]),
  `metal::half2x4 sbv[128]` with `threadgroup half *sb = (threadgroup
  half*)sbv` (B tile [t][k]; declare as half2x4 so the vectorized
  8-half store is 8-byte aligned — a bare `half sb[]` gives only 2-byte
  alignment and the half2x4 cast would be UB), `float sc[32*64]` output
  staging (SEPARATE buffer — no aliasing games), `ulong svalues[256]` +
  `uchar ssigns[128]` staged from the constant tables by all 128
  threads, then ONE threadgroup_barrier BEFORE the main loop (llama.cpp
  GEMM path hits constant memory — staging is the known win from our
  own qgemv; the barrier is mandatory, the first dequant call reads the
  tables before the in-loop barrier).
- Early exit: `im=tgid.z; r0=tgid.y*64; r1=tgid.x*32; neh1=tpe[im];
  if (r1 >= neh1) return;` Over-dispatch is ~9x trivially-exiting TGs at
  2048 tokens — accepted for v1 (v2 option: indirect dispatch via a
  prefix-sum tile queue).
- Thread map: lr0 = min(tiitg/2, nr0-1) (A row), il0 = tiitg%2 (K half,
  16-value dequant granule), lr1 = min(tiitg/4, nr1-1) (B row),
  iy = 8*(tiitg%4).
- A pointer: `xq = (device const block_iq2_xxs_t*)(Wq + (long)im*N*bpr*66
  + (long)(r0+lr0)*bpr*66)` with `struct block_iq2_xxs_t { half d;
  ushort qs[32]; }` (sizeof 66, align 2 — no padding).
- B indirection: `id = ids[im*tokens + r1 + lr1]; tok = id/topk;
  y = X + (long)tok*K + iy;` X rows are 64-B aligned, iy is 16-B
  multiples => the half2x4 vector load/store fast path is always legal.
- K loop (copy llama.cpp exactly):
  `temp_a = dequant16(xq, il, svalues, ssigns)` (registers) -> barrier
  (protects previous MMA reads) -> swizzle-store 16 values to sa
  (`sx=2*il0+i/8; sy=(tiitg/2)/8; lx=(tiitg/2)%8; ly=i%8; ib=8*sx+sy;
  *(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];` — keep the
  `*(ptr+expr)=` form, llama.cpp measured `ptr[expr]=` slower) ->
  B store `*(threadgroup half2x4*)(sb + 64*(4*(tiitg%4)+((tiitg/4)/8))
  + 8*((tiitg/4)%8)) = *(device const half2x4*)y` -> advance
  `il = (il+2<nl) ? il+2 : il%2; xq = (il<2) ? xq+1 : xq; y += NK;`
  -> barrier -> MMA: `lsma = sa + 4*64*(sgitg%2); lsmb = sb +
  2*64*(sgitg/2);` 4 ik-steps of {simdgroup_barrier(mem_none); load
  ma[0..3] from lsma+64*i stride 8; mb[0..1] from lsmb+64*i stride 8;
  8x simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
  lsma += 512; lsmb += 256;}. Accumulators simdgroup_float8x8 mc[8]
  zeroed with make_filled_simdgroup_matrix<float,8,8>(0.f).
- Epilogue: `temp_str = sc + 32*(sgitg&1) + (16*(sgitg>>1))*64;`
  simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*64*(i/4), 64) -> barrier
  -> scatter `for (j = sgitg; j < nr1; j += 4) { id = ids[im*tokens+r1+j];
  Dr = D + (long)id*N + r0;` (output row IS id: id = token*topk + slot =
  the vec path's flat slot order, so act() and sum6 consume it
  unchanged) `for (i = lane; i < nr0; i += 32) Dr[i] = half(C[i]); }`
  where C = sc + j*64. Accumulate fp32, store half.
- dequant16 hook (llama.cpp dequantize_iq2_xxs with threadgroup tables,
  float math then half store): d=float(xb->d); ib32=il/2; il%=2;
  q2=xb->qs+4*ib32; aux_g=q2[0]|q2[1]<<16; aux_s=q2[2]|q2[3]<<16;
  dl=d*(0.5f+float(aux_s>>28))*0.25f; two 8-value groups via
  svalues[aux8[2*il+{0,1}]] and ssigns[(aux_s>>(14*il{,+7}))&127],
  sign via kmask_iq2xs[i]. Exactly 16 values per call, il in [0,16).
- Threadgroup total: 4096+2048+8192+2048+128 = 16512 B < 32 KB.

### 2.2 Launcher (`csrc/quixicore/metal/kernels/common/tk_launch.h`)

`launch_moe_mm_map0(e, topk_ids, tpe, ids, tokens, topk, E)`:
pipeline "qc_moe_mm_map0_<topk>"; buffers as §2.1; dispatch(1,1,1,E,1,1).
`launch_moe_mm_id(e, d, wq, x, tpe, ids, N, K, tokens, topk)`:
pipeline "qc_moe_mm_id_iq2_xxs"; dispatch((tokens+31)/32, N/64, E-from-
caller, 128,1,1). Follow launch_qgemv_moe_mr_q2k_sum (tk_launch.h:6179)
style exactly.

### 2.3 Host op (`csrc/quixicore/tm_metal/qc_metal_serving.mm`)

`at::Tensor ggml_moe_mm_id(x, w, topk_ids_in, top_k, quant_type, row,
tokens)`: TORCH_CHECK fmt=="iq2_xxs" (ggml_type_to_format, 16), x fp16
[tokens,K] contiguous, K%256==0, N=row, N%64==0, w [E,N,K/256*66],
E<=64, top_k in {2,4,6,8}; topk_ids -> kInt contiguous; scratch
`tpe = at::empty({E}, kInt)`, `ids = at::empty({E*(long)tokens}, kInt)`
(int32 buffer is bit-compatible with the kernel's u32 counts); output
`at::empty({tokens*top_k, N}, x.options())` (do NOT ring_out — 100 MB
at 2048 tokens exceeds the 8 MiB ring bypass anyway). TWO encode()
calls — `encode("qc_moe_mm_map0", ...)` then `encode("qc_moe_mm_id",
...)`; separate encoders on torch's MPS command buffer serialize the
same way every existing dependent chain in this file does. pybind def
next to ggml_moe_a8_vec_sum.

### 2.4 Python plumbing

- `vllm/quixicore/ops.py`: wrapper `ggml_moe_mm_id(...)` (copy the
  ggml_moe_a8_vec_sum wrapper shape).
- `vllm/model_executor/layers/quantization/gguf/ops.py`: Metal-only
  wrapper (assert _is_metal(), copy ggml_moe_a8_vec_swiglu at :221).
- `vllm/model_executor/layers/quantization/gguf/fused_moe.py`, in the
  Metal w1 dispatch block (~line 668): add
  `use_mm_w1 = current_platform.is_metal() and qweight_type == 16 and
  expert_map is None and x.dtype == torch.float16 and not w1_repacked
  and num_tokens >= _qc_mm_min_tokens() and
  os.environ.get("VLLM_QC_MOE_PREFILL_MM", "1") != "0"`
  with `_qc_mm_min_tokens()` reading VLLM_QC_MOE_MM_MIN_TOKENS default
  32 (llama.cpp's exact GEMV/GEMM threshold: n_tokens>=32 && K>=64).
  Branch: `out = ops.ggml_moe_mm_id(...); ` then the EXISTING
  `out = act(out)` (act halves [...,2d]->[...,d], first-half gate —
  matches w13 row order gate [0,inter) | up [inter,2*inter)). Adjust the
  trailing `if not (w1_vec and use_fused_act): out = act(out)` to also
  exclude the mm branch (apply act exactly once). Down path (sum6)
  consumes [slots, inter] unchanged. Add a
  `logger.info_once("quixicore(metal): tiled MoE prefill GEMM (w13)
  active")` LIVENESS breadcrumb.
- Decode verify batch is 6 tokens (< 32) => the tile NEVER engages at
  decode; the 8-tok gate's whole request stays on the vec path.

### 2.5 Gates for v1

1. Oracle `tests/kernels/test_metal_moe_mm.py` (copy the
   test_metal_moe_sum6.py harness): random iq2_xxs stacks with FINITE
   scales, tokens {33, 64, 100}, topk 6, E 8/64, incl. ragged per-
   expert counts; compare ggml_moe_mm_id vs ggml_moe_a8_vec (plain, no
   swiglu) — TOLERANCE gate (MMA fp32 8x8-fragment accumulation !=
   GEMV dot-walk; expect ~1e-2 rel on fp16 out), plus determinism x2
   bitwise, plus tokens=6 python-gate check that the vec path is taken.
2. Serving: 8-tok sha db2846cf721b must stay BIT-IDENTICAL (vec path
   end to end). off1-2000 sha WILL roll (prefill numerics change):
   judge determinism x2 (same sha both runs), coherent text (dump
   completions), decode step ms unchanged (~123: compute (wall-5.5)/
   drafts), prefill wall improved. With VLLM_QC_MOE_PREFILL_MM=0 the
   off1 sha must be EXACTLY 4d18b4fac460 (counters 1409/2955/591) —
   that reversion run is the regression sentinel.
3. Prefill walls re-run (the §1 probe, same offsets): expect 2048-token
   TTFT 25.9 -> ~21-22 s for v1 (w13 only).

### 2.6 v2+ (after v1 gates)

- v2: tile w2 q2_K. The resident layout is SoA-planed (per expert:
  qs plane @0 (64 B/blk), scales plane @N*nb*64 (16 B/blk), d/dmin
  @N*nb*80 (4 B/blk)); either a dequant16_q2_K_soa hook (three plane
  pointers) or accept AoS via VLLM_QC_MOE_SOA=0 for an A/B first.
  llama.cpp dequantize_q2_K @627 is the AoS reference. Down input is
  [slots, inter=2048], K%32 ✓. Weighted sum: keep the existing
  moe_weighted_sum finalize in v2 (sum6 fold is vec-only), or extend.
- v2.5: fuse SwiGLU into the w13 tile epilogue (silu(gate)*up needs the
  gate row-tile and up row-tile in one TG — llama.cpp doesn't do this;
  our moe_grouped_gemm_swiglu (moe.metal:529) shows the two-accumulator
  pattern); kills the [slots,4096] round trip (~8.6 GB/chunk).
- Indirect dispatch (prefix-sum tile queue) to kill the ~9x over-launch.
- Wave 2: insert/compress chain (~10 s/chunk): profile inside
  attn_wqb_insert_c first (wq_b GEMM vs _fused_qnorm_rope_kv_insert vs
  compressor full-compress at prefill widths; comp_full_compress ~190
  ms/call at 2048 tokens is suspicious — check for per-call MPSGraph
  pathology, cf. the >64K encode-queue bug). ds4's fused insert-chain
  kernels are the reference. Also retire VLLM_QC_LONGCTX_SYNC via the
  real MPSGraph encode-queue fix.
- Wave 3: layer_mhc 3.44 s (419 calls x 8.2 ms at prefill widths).
- Sharpen shexp: qc_mmq host wrapper does a transpose+pad+transpose
  round trip per call (qc_metal_serving.mm:1349) — measurable at
  prefill; a native [tokens,K] path would kill it.

## 3. Key reference facts (so you don't re-explore)

- llama.cpp mul_mm_id anatomy (verified identical in both local trees;
  use ~/Code/hy3/llama.cpp): two-phase map0 (1 TG, thread-per-expert,
  ids i32 [E][tokens] dense-prefix + tpe u32 counts appended to dst
  alloc) -> mul_mm_id 64x32 tiles, 128 threads/4 sgs, early-exit
  `if (r1 >= tpe[im]) return`, A-rows clamped not masked, dequant hooks
  emit exactly 16 values (il in [0,16), nl=16 for q2_K/iq2_xxs), the
  GEMM path does NOT stage the iq2 tables (GEMV does — we stage),
  host threshold ne21>=32 && ne00>=64, FC bounds-check elision when
  K%32==0. Gotchas: short token indices (cap 32767), ne20 template set
  {1,2,4,5,6,8,10,16,22}, tensor-ops path is Metal4-only (dead on M1).
- Our infra: TorchEncoder (pipeline/in/out/bytes/dispatch, buffers
  from index 0, setBytes constants), encode(label, fn) per-encoder on
  torch's MPS CB, pipeline() PSO cache from g_library, ring_out bypasses
  >8 MiB. Metallib build GLOBs kernels/*.metal — a new file under
  csrc/quixicore/metal/kernels/ is picked up automatically (cmake) or
  included by the find-based xcrun one-liner below. tk substrate
  TILE_DIM=8; tk `load(rt,st)` does NOT use simdgroup_load (hence the
  raw-intrinsics port; precedent: attn_fwd_sg.metal uses raw
  simdgroup_* directly).
- In-repo negatives that CONSTRAIN this design (do not redo):
  4-warp/BM=128 staged tiles -20..26% (gemm_staged.metal:71); expert-
  grouped GEMV 123->230 ms (SLC already dedups; unbounded slot scans
  wedged prefill — width-gate everything); iq2_xxs SoA/pairing repack
  slower on Apple (w13 is AoS by measurement, fused_moe.py:965 comment);
  host fire-and-forget prefetch unaimable.

## 4. Ops rules (verbatim-critical)

1. `sysctl iogpu.wired_limit_mb` must be 122880 before any boot (resets
   on MACHINE restart only; user shell: `env -u TERMINFO sudo sysctl
   iogpu.wired_limit_mb=122880`).
2. Tiny primer curl after EVERY boot before anything big (big-first =>
   CB timeout => poisoned engine => reboot).
3. `kIOGPUCommandBufferCallbackError*` => pkill, wait `memory_pressure
   -Q` >70% free, ~6 s, reboot. Machine restarts roll the trajectory
   lottery; server reboots do not. Exclude fa_utils/FA2 lines from boot
   error greps.
4. `nohup` is broken in this harness — boot servers via the Bash tool's
   `run_in_background: true`; health-wait loops foreground (python
   poll loop; bare `sleep` is blocked). Hold the turn for <10-min waits.
5. Commit ONLY when the user asks. Explain diagnosis + proposed change
   in-message before code edits.
6. Boot: `PYTHONUNBUFFERED=1 .venv/bin/slimserve dsv4-xxs-1 --serve -y
   > <log> 2>&1` -> grep "Application startup complete" (~3 min) ->
   primer: `curl -s http://127.0.0.1:8000/v1/completions -H
   'Content-Type: application/json' -d '{"model":"DeepSeek-V4-Flash",
   "prompt":"Hello, my name is","max_tokens":8,"temperature":0}'`.
7. Gate harness: `.venv/bin/python benchmarks/benchmark_dsv4_exact.py
   --model /Users/seangherardi/models/antirez-deepseek-v4-gguf/DeepSeek-
   V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
   --served-model-name DeepSeek-V4-Flash --source perf/perf.md --url
   http://127.0.0.1:8000/v1/completions --concurrency 1 --input-tokens
   1000 --output-tokens {8|2000} --prompt-offset N
   --warmup-output-tokens 1` (step ms = (wall-5.5)/drafts;
   --dump-completions for text). Prefill walls: scratchpad
   prefill_probe.py from this session, or re-create (streaming TTFT,
   usage.prompt_tokens, disjoint offsets).
8. Metallib: `xcrun metal -std=metal3.1 -O2 -I csrc/quixicore/metal/
   include/metal -I csrc/quixicore/metal/kernels/common $(find
   csrc/quixicore/metal/kernels -name "*.metal") -o
   vllm/quixicore_metal.metallib` (verify new PSOs via unanchored
   `strings` grep). Ext build: TORCH_DIR=$PWD/.venv/lib/python3.12/
   site-packages/torch; PYINC=$(.venv/bin/python -c "import sysconfig;
   print(sysconfig.get_paths()['include'])"); clang++ -x objective-c++
   -std=c++17 -fobjc-arc -O3 -shared -DTORCH_EXTENSION_NAME=_quixicore_C
   -DTORCH_API_INCLUDE_EXTENSION_H -I$TORCH_DIR/include
   -I$TORCH_DIR/include/torch/csrc/api/include -I$PYINC
   -Icsrc/quixicore/metal/kernels/common
   csrc/quixicore/tm_metal/qc_metal_serving.mm -L$TORCH_DIR/lib -ltorch
   -ltorch_cpu -lc10 -ltorch_python -framework Metal -framework
   Foundation -framework QuartzCore -undefined dynamic_lookup
   -Wl,-rpath,$TORCH_DIR/lib -o vllm/_quixicore_C.cpython-312-darwin.so
9. Gate discipline: bit-exact => identical shas/counters PLUS a
   liveness proof (info_once breadcrumb / PSO-name check). ULP =>
   determinism x2 + coherent text + PAIRED step ms + means over >=5
   offsets. ±1.5 ms boot floor. Never trust interval logs.
10. Kill switches (default on): VLLM_QC_MLA_SPLITK, VLLM_QC_MOE_SUM6,
    VLLM_QC_MOE_SOA, VLLM_QC_MOE_MR, VLLM_QC_MOE_FUSED_ACT,
    VLLM_QC_MEMO_{COSSIN,POS,WOA}, VLLM_QC_COMPRESS_FRONT,
    VLLM_QC_OUT_RING, VLLM_QC_LONGCTX_SYNC, VLLM_METAL_MHC. Quarantined
    negatives (NEVER enable): VLLM_QC_MOE_GROUP, VLLM_QC_PREFETCH.
    Instruments: VLLM_QC_PHASE_PROF=1 (dump /tmp/phaseprof_<pid>.txt,
    snapshot-diff around a request isolates one request's split),
    VLLM_SYNCPROF=1, VLLM_QC_OP_CENSUS=<n>, VLLM_QC_MOE_IDS_CENSUS=<n>.
    New this campaign: VLLM_QC_MOE_PREFILL_MM (planned default 1 after
    gates), VLLM_QC_MOE_MM_MIN_TOKENS (default 32).

## 5. Box state at handoff (2026-08-13 ~19:45)

- A VLLM_QC_PHASE_PROF=1 server is RUNNING on :8000 (boot log
  perf/results/2026-08-13/prefill_baseline/boot_phaseprof.log). It is
  an instrumented dev boot (+10% step inflation) — kill and reboot
  clean before any wall/gate measurement. Wired limit verified 122880.
- Nothing committed this campaign (~11k lines pending across decode +
  this file). Task list: #37 in_progress (this campaign).
- Decode gates for regression sentinels: 8-tok sha db2846cf721b
  (2/10/7), off1-2000 matrix sha 4d18b4fac460 counters 1409/2955/591 at
  step 122.6-124.2 ms.
