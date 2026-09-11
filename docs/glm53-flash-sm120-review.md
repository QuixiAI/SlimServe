# GLM-5.3-Flash SM120 review and integration checklist

This is a review map for `glm53-flash-sm120`, not a replacement roadmap.
The item IDs and detailed history remain in
[the plan](glm53-flash-sm120-plan.md). Updated 2026-09-11.

## Scope and current evidence

Target: four RTX PRO 6000 Blackwell GPUs, TP4, profile `glm53-nvfp4-4`,
recipe `glm53-redhatai-nvfp4-fp8-kda-tp4-v1`. Preserve NVFP4 routed experts,
the pinned FP8 projection swap-set and FP32 repairs, BF16 KV/head, and
non-speculative execution. No claim against all B12X configurations or of
reaching a physical performance ceiling.

The fixed cold-prefix reference is 155.90 / 575.56 / 779.06 end-to-end output
tok/s at concurrency 1 / 8 / 16. The matched-workload B12X R28.1 reference is
132.59 / 466.69 / 589.28, with different W4A4/FP8-KV precision and scheduling.
These are complete-request rates, not pure steady-state decode. See
`perf/baseline_status.md` for spreads, exact workloads and limitations.

Subsequent retained work includes lossless paired mHC staging, wide Marlin
prefill and TP-row-sharded indexer prefill. The indexer serving pair reduces
cold 128K engine TTFT from 10.658 to 10.009 seconds; decode is neutral.
The first BF16 swapAB serving pair was a no-op: its dispatch required H32,
but the actual checkpoint has 64 global heads / TP4 = H16. Its H32 component
gain is not evidence for this profile. The corrected H16 pair qualifies a
small retained win: 32K/128K engine TTFT 2526.339/10001.190 ->
2519.428/9949.508 ms (-0.274/-0.517%). All three candidate timings beat all
controls at each length. Decode is neutral within the observed spread, not
improved. GPU/parity/sanitizer and serving checks pass; detailed ranges and
limitations are recorded in the baseline snapshot.

## Review order

1. Recipe/platform ownership: `slimserve/profiles.json`, preparation and digest
   verification, platform detection, CLI environment precedence and dry-run.
2. Retained decode path: BF16/FP8 projections, custom PCIe all-reduce,
   route/alignment synchronization, mHC, small-k sampler, sparse MLA reducer.
3. Retained prefill path: SM120 wide Marlin, pooled-indexer geometry, TP4 row
   sharding, sparse prefill dispatch and candidate synchronization.
4. Bounds and platform fallback: packed-page strides, request-local tables,
   empty/tail rows, graph replay, device-local launch setup, unchanged
   non-SM120 dispatch and explicit opt-outs.
5. Diagnostic isolation: failed candidates must remain benchmark-only or
   explicitly opt-in; no disabled heavyweight serving cache or default
   numerical policy should be introduced by cleanup.

The accumulated upstream diff is large (365 files after the main merge),
including historical notebooks and diagnostic tests. Do not equate notebook
volume with retained kernel code. Preserve the evidence and user edits.
Existing draft [PR #24](https://github.com/QuixiAI/SlimServe/pull/24) uses the
upstream `glm53-flash-sm120` branch. Main f6c6ed429 is merged as725b4ba01;
all17 conflicts are resolved, and GitHub reports it mergeable.

One fixed merged-tree boot passed all timed batches, text/image canaries and
six retrieval contrasts. Exact1000/300 cold-prefix E2E medians at c1/c8/c16:
156.841/577.473/781.668 tok/s; cold32K/128K engine TTFT2509.907/9920.798ms.
Three repetitions, no restarts or timing exclusions; native binaries/recipe
unchanged. See the baseline snapshot for spreads and raw source receipts.
Post-run cleanup scopes raw-indexer matmul defaults to SM120 with ten focused
dispatch tests; SM120 behavior is unchanged from the measured tree.

## Qualification record map

These existing notebook records supply the baseline, hypothesis, checks, timing,
decision and raw receipts requested in the serving review. They are historical
evidence, not a claim that old best-start rates are current baselines or that
every optional diagnostic is enabled. Raw paths below are relative to
`perf/results/`; the linked records preserve failed runs and caveats.

| Implementation | Existing qualification | Scope and limitation |
|---|---|---|
| F32 repairs / offline partial reads | [2026-09-04: F32 sidecar for the RedHatAI NVFP4 conversion (router bias, KDA decay, mHC vectors) - RETAINED](../perf/optimization_status.md#2026-09-04-f32-sidecar-for-the-redhatai-nvfp4-conversion-router-bias-kda-decay-mhc-vectors---retained) | Native-value repair versus downcast values; throughput-neutral in quiet paired readings. Range reads are offline I/O hygiene, not a separately measured serving optimization. Raw `2026-09-04/glm53-nvfp4-4-rtx6000-f32ab*`. |
| MoE output alias and combined sum | [2026-09-07: Phase 1 item 2 remainder - MoE finalize copy removed, shared-expert add fused into the Marlin sum](../perf/optimization_status.md#2026-09-07-phase-1-item-2-remainder---moe-finalize-copy-removed-shared-expert-add-fused-into-the-marlin-sum) | Baseline, separate alias/combine factors, correctness and exact-token comparisons are recorded. Raw `2026-09-07/item2-{A,B}-pass{1,2}`. Existing ROCm/AITER predicate is unchanged; no new ROCm qualification claimed. |
| Sparse MLA partition/reducer | [2026-09-07 Sparse MLA decode: partition and reduce (DSA layers; c1 lever from the kernel sequence)](../perf/optimization_status.md#2026-09-07-sparse-mla-decode-partition-and-reduce-dsa-layers-c1-lever-from-the-kernel-sequence) | Actual H16/TP4 component comparison and serving attribution. Historical best-boot rates are superseded by the fixed cold-prefix baseline, not repeated as a current speedup. Raw `2026-09-08/dsa-A-pass{1,2}`. |
| H16 sparse prefill dispatch | [H16 serving pair complete — retain small prefill win](../perf/optimization_status.md#h16-serving-pair-complete--retain-small-prefill-win) | Corrected active-H16 flag0/flag1 comparison; cold32K/128K TTFT -0.274/-0.517%, decode neutral. Initial H32 pair was a no-op and remains preserved. Raw `2026-09-11/sparse-swapab-h16-serving-{control,candidate}`. |
| State-copy warmup | [2026-09-08: Startup-copy warmup passes the real three-start workload](../perf/optimization_status.md#2026-09-08-startup-copy-warmup-passes-the-real-three-start-workload) | Preceding baseline identifies the first-use157ms JIT stall; tiny private scratch compiles the production signature. Exact requests/canaries pass; steady-state throughput neutral. Raw `2026-09-08/repro-baseline` and `warmup-boundary`. |
| Corrected small-k sampler | [2026-09-08: Corrected sampler passes the fixed three-start serving series](../perf/optimization_status.md#2026-09-08-corrected-sampler-passes-the-fixed-three-start-serving-series) | Prior entry documents corrected ties/noise and kernel/sanitizer checks. All225 measured requests pass;156.084/575.670/777.037 E2E tok/s, neutral versus the prior sampler. Raw `2026-09-08/sampler-serving`. |
| Optional fused indexer ordering | [2026-09-09 - Fused selected-pool ordering preserves full-model results](../perf/optimization_status.md#2026-09-09---fused-selected-pool-ordering-preserves-full-model-results) | Preceding bitonic comparison replaces selector+sort with ordered selection: actual7616x1904 warm66.191->57.661us; full-model score/tensor parity passes. Instrumented107.845/457.533/645.385 tok/s is NOT a production baseline. Raw `2026-09-09/index-bitonic-timing` and `index-fused-quality-diagnostic`. |
| Optional deterministic reductions | [2026-09-10 - Full no-combo workload is repeatable within start but fails quality windows](../perf/optimization_status.md#2026-09-10---full-no-combo-workload-is-repeatable-within-start-but-fails-quality-windows) | Not promoted: native-order diagnostic repeats scores but fails12/32 historical windows;155.740/574.245/777.768 E2E tok/s is not a win. Raw `2026-09-10/deterministic-no-combo-serving/fresh-a`. |
| Optional GLM prompt-score chunking | [2026-09-10 - Exact model parity with recovered allocation failures eliminated](../perf/optimization_status.md#2026-09-10---exact-model-parity-with-recovered-allocation-failures-eliminated) | GLM-specific control/chunked/return, not Qwen evidence: exact4096+168 scores, allocator warnings8->0->8; candidate156.983/579.758/778.748 tok/s, timing neutral. Raw `2026-09-10/prompt-score-serving-v1`. Remains opt-in. |
| Prompt-score default rollout limit | [2026-09-10 - Fresh production rollout stops at unchanged control quality](../perf/optimization_status.md#2026-09-10---fresh-production-rollout-stops-at-unchanged-control-quality) | Native-order0 control failed historical windows with chunking OFF; candidates/return never ran. Raw `2026-09-10/prompt-score-rollout-v1`. Not a chunking failure, no default promotion, and no instruction to restart a scoring campaign. |

No separate full-load-versus-range-read timing exists for the offline F32
builder; do not invent one. Its tensor/value identity and model-repair evidence
are the relevant contract. Likewise, the unchanged ROCm alias branch is not
requalified by the SM120 comparison. New performance claims still need a paired
measurement on the selected profile.

## Remaining deep-kernel decisions

### 2.3: broader next-layer prefetch — unimplemented, deferred

The implemented 8 MiB KDA output-weight prefetch is rejected: complete-window
c1 saves only about 0.2 us per KDA layer, while c8/c16 regress about 4.4%.
That experiment does not test next-layer weights during a TP reduction.

The broader proposal would overlap a known future backbone read with the
remaining reduction wait. The retained custom reduction is approximately
5.1 us per call, rather than the original approximately 11 us NCCL call.
At the theoretical 1.79 TB/s rate, 5.1 us corresponds to at most 9.1 MB of
GDDR traffic; this is an optimistic overlap budget, not measured prefetch
bandwidth, nor a strict ceiling if prefetch starts earlier. Starting earlier
also overlaps current expert/projection traffic. The source precedent
local-inference-lab/vllm #576 records losses from expert-stream contention and
per-layer joins; its BF16/speculative gains do not transfer directly here.

Defer the broader design for this retention pass. Reopen with evidence of a
specific independent next-weight window and cache-miss benefit that exceeds
issue/synchronization cost on this FP8/no-spec route. Do not label it tested,
impossible, or exhausted, and do not substitute a warmed-GEMM timing for the
whole dependency window.

### 4.4: persistent per-layer decode — unimplemented, deferred

The recorded c1 graph span is about 5.740 ms, with 0.427 ms uncovered by GPU
kernels. Removing every such gap would remove about 7.4% of that graph span;
it is not an achievable speedup prediction or a bound on all fusion benefits.
Persistent execution cannot simply erase TP dependencies or reuse existing
host-launched kernels inside a device loop. It needs new cross-CTA scheduling
and compatible operator ownership, while preserving the measured arithmetic.

The enabling experiments do not currently justify that replacement:

- Marlin already uses DP plus two-tile stream-K; adding stream-K is not new.
- Cross-item expert pipelining remains slower than retained Marlin.
- Paired Marlin activation fusion offers only about 54 us per c1 step and is
  batched-neutral; amplified-input comparisons also remain outside tolerance.
- Whole-head KDA conv/state/norm fusion regresses c1/c16 despite passing its
  corrected 306 state/output checks.
- The swapAB proposal addresses prefill, not persistent decode; its original
  H32 measurements did not match this TP4 checkpoint.

Defer the monolithic design, without calling it implemented or exhausted.
Reopen when a concrete region has a measured traffic/dependency saving and a
compatible resident schedule; launch-count reduction alone is insufficient.

## Deferred integration work — not SlimServe PR gates

The operator explicitly excludes Foundry integration from this PR. Disabled
host/NVMe tier qualification and the external QuixiCore port-back are follow-ups.
Preserve the integration notes below for that later work; do not use them to
block publishing or reviewing the retained SlimServe implementation.

### 5.1 Foundry

The current Foundry backend is native-FP8 SGLang, served name
`glm-5.3-flash`, dialect `sglang`, health `/health_generate`, context 300K,
minimum thinking budget 163840 and no request timeout. SlimServe uses the
registry-owned `GLM-5.3-Flash` name and vLLM/OpenAI-compatible request handling.
An integration must update the launcher, health route, dialect and model-name
contract together; swapping only the executable is insufficient.

Foundry already sets `_LLM_PARALLEL_WORKERS = 8` (twice), while its current
launcher defaults to four running requests. Do not blindly raise the worker
count. Its historical eight-director fixture caps outputs at 4096 tokens and
does not qualify the current long-thinking application policy. The original
cutover gate is a matched, usable eight-wide director result against B12X,
not the generic exact-token result above.

The Foundry worktree has overlapping uncommitted registry, deployment and
orchestrator edits. None were changed or reverted in this pass; no Foundry
service was restarted. Integration and application qualification remain open.

### 5.2 Host/NVMe KV tier

RTX6000 tiers remain disabled. The proposed opt-in test budget is 8 GiB host
plus 16 GiB NVMe per rank (32/64 GiB total), disk directory
`/raid/slimserve-kv-tier`, subject to the outstanding operator choice. Do not
inherit A100's 64 GiB/rank host allocation on this 188 GiB host. Qualification
must use the registered profile, forced eviction/restore and
`VLLM_KV_TIER_VERIFY=1`; speed without zero-mismatch restore is not acceptance.

### 5.3 Minimal QuixiCore port-back

Canonical repository: `https://github.com/QuixiAI/QuixiCore-CUDA.git`.
Read-only comparison against clean `main` at `5e89180` found:

| Retained family | Minimal port |
|---|---|
| Small-M BF16/FP8 projections | `bf16_decode_gemm.cuh`, `fp8_decode_gemm.cuh`, retained bindings |
| Router and alignment | BF16-input/FP32-output router GEMV; atomic GLM route/alignment with both warp barriers |
| Expert combination | `moe_sum_add` only |
| mHC | Retained cooperative/split/post/head paths and paired-BF16 prefill, not tensor-core/last-block experiments |
| Small-k sampler | Corrected cutoff/tie/nucleus/noise semantics |
| Sparse decode | Channel-owned NoPE reducer and launch selection |
| Sparse prefill | Retained H16 BF16 swapAB, including all-reader barriers, checked binding and focused tests |

`mla_kernels.cuh` is already byte-identical, including packed-page stride.
Wide Marlin and TP indexer sharding live in vLLM; do not import vLLM wholesale.
Sparse H16 swapAB is now retained and belongs in the port; its prior H32
prototype does not. Extract bindings; do not copy the entire
`tm_cuda_serving.cu` translation unit.

Port prerequisites: user-selected branch (QuixiCore forbids choosing one),
scoped build plumbing (`setup.py` omits nine initializer dependencies and
`build_ext.sh` hard-codes SM86/a missing interpreter), accurate SM120 metadata,
and focused correctness/performance through the rebuilt public route. The
existing SlimServe measurements are provenance, not standalone port validation.

## Publication

GitHub authentication was restored as `auroter` on 2026-09-11; the earlier
HTTP401 failure remains in the historical notebook. Both API access and
upstream push permission are confirmed. Update existing PR #24; do not create
a duplicate or force-push. Use Auroter <auroter@users.noreply.github.com> for
author and committer.

The semantic merge preserves SM120's padded FP8 KDA projections, raw indexer
and measured V1 runner, alongside A100's gate-pair path, compact cache/decode
sharding and V2/DFlash2. Both platform-specific MTP adapters remain intact.
Canonical glm53f profile names retain historical aliases. Sliced prefill query
maps now gather actual request IDs rather than renumbering from zero.

The user permits splitting only if necessary. CodeRabbit explicitly confirmed
that it cannot batch selected files within one PR: incremental review only
covers new changes, while path filters permanently exclude review scope. The
365-file diff exceeds the current100-file limit and absolute300-file maximum.
Therefore publish a dependency-ordered stack below100 files per PR; do not
exclude tests/diagnostics. Keep PR24 as the tip and preserve campaign commits
with a non-rewriting history join. Combined-tree qualification above is not
a claim that every intermediate review layer is independently serving-qualified.

The stack is published; review/merge dependency order is:

| Part | PR | Scope | Changed files |
|---|---|---|---:|
| 1 | [#26](https://github.com/QuixiAI/SlimServe/pull/26) | Native kernels, bindings, focused probes/tests | 94 |
| 2 | [#27](https://github.com/QuixiAI/SlimServe/pull/27) | Serving recipe, platform integration and regressions | 83 |
| 3 | [#28](https://github.com/QuixiAI/SlimServe/pull/28) | Numerical diagnostics and prefill qualification | 83 |
| 4 | [#25](https://github.com/QuixiAI/SlimServe/pull/25) | Indexer/routing diagnostics and qualification | 57 |
| 5 | [#24](https://github.com/QuixiAI/SlimServe/pull/24) | Campaign harnesses, evidence and roadmap | 55 |

History join e5d0ec096 is byte-identical to qualified-content c41148025 and
retains every original campaign commit. No force-push or merge into main.
Layer1 targets main; each later PR targets the preceding branch. Keep this
dependency order when landing; retarget the next PR after its base lands.
All review/cleanup gates are complete; the stack is ready for review/landing in
dependency order, not authorized for automatic merge. CodeRabbit completed
part5 at18:58 UTC on2026-09-11:
ten actionable comments and two small test cleanups are addressed. Fixes cover
failed/timeout process exit status, interrupted workload receipts, unavailable
decode rates for one-event streams, captured graph-output checks, Python3.10
digest compatibility at the reported sites, isolated test environments and
matching Auroter authorship instructions. The shared fixture helper is committed
in part3 and propagated forward; no history rewrite or serving-code change.
Focused CPU suite:128 passed in30.41s, raw XML
`perf/results/2026-09-11/pr-review/part5.xml`. This is harness fault-injection
coverage, not another serving benchmark or native-kernel qualification.

Part1 review completed at20:00 UTC:13 inline findings and14 nits. Published
03875b05a addresses valid contracts, optional-symbol/platform fallbacks, private
diagnostic permissions and probe hygiene. The optional mode1 mHC completion
counter is invocation-local, including independently captured graphs; the
profile's cooperative default is unchanged. The bitmap-overflow report was
already bounded by both public callers; a rejection regression now locks it.
The proposed sampler sentinel/barrier is declined because the histogram
invariant guarantees exactly one writer; no supported failure was identified.
All inline findings have evidence replies and the nits have a disposition.

Checks:99 focused CPU tests and172 GPU tests pass after rebuilding QuixiCore
with80GiB/no swap/-j2. Stable-libtorch stays unchanged. Local serving review
also found an unsupported-dtype combine handoff; d19a1744a preserves the
unfused fallback for FP16/misaligned inputs, with10 CPU regressions passing.
Fixes live in their owning branches and are merged forward without rewriting.

The first review-tree serving boot (652bb8014) failed before timing: Dynamo
traced the new capability check into NVML ctypes. Fix03382cb49 evaluates the
process-fixed gates as compile-time constants. The initial regression wrongly
treated is_compiling as proof of symbolic execution; e35bd7976 replaces it
with an actual ctypes call. All18 dispatch tests now pass, including four
cold-cache full-graph checks. A single corrected-tree boot on eb8c345a9 passed
all three prescribed repeats, text/image and six retrieval contrasts. E2E
c1/c8/c16 medians156.608/578.251/783.254 tok/s; cold32K/128K engine TTFT
2508.750/9906.629ms. Preserves performance, not a paired speedup claim.
Failures and successful checks remain under `perf/results/2026-09-11/pr-review/`.

Main advanced nine commits to0313f5228 during that run. Binding and indexer
conflicts are resolved in72e3ed10b/8720c75e7 and merged forward as24bd601c5.
Both new A100 FP8 decode and SM120 BF16 prefill bindings are retained, as are
grouped speculative scoring and request-grouped raw prefill. The RTX6000
profile record is identical as parsed JSON; upstream thinking-budget changes
are confined to A100.120 focused CPU checks pass, eight GPU-only cases skipped.
The native rebuild passes (QC61616000, stable-libtorch unchanged), followed by
66 GPU tests for merged indexer/grouped-scoring/BF16-FP8 sparse-MLA behavior.
GPUs released. That serving result predates this merge; the final combined
qualification below includes it and the serving-review fixes.

CodeRabbit resolved all13 part1 threads. Part2 review completed21:05 UTC with
17 findings/six nits. Fixesceafefbed/311f5accf/dbb4ab817 plus part3aa2efae02
cover invalid TP counts, truncated sidecars, packaged case dispatch, native
symbol fallback, folded MTP projections, aborted journal frames, sampler
fallback metadata and stronger tests.137 CPU tests/11 GPU routing tests pass;
final scheduler/import cleanup passes17 focused CPU tests. No native rebuild.
The51-file source catalog is complete on the combined tip; the missing-file
report inspected a partial review layer. The registered RTX6000 V1 override
also makes the requested V2 port inapplicable. Keep the distinct MTP adapters;
the SM120 block now has an unambiguous name. Decline unmeasured persistent
pooled-key caches. Qualification links above preserve existing evidence.
All17 individual review threads are resolved and all six nits dispositioned.
Part3/#28 received a full83-file static review at21:44:39 UTC with no actionable
defects ([review comment](https://github.com/QuixiAI/SlimServe/pull/28#issuecomment-5640985715)).
This is an explicit CodeRabbit chat review, not a formal GitHub approval. It
did not consume the formal review slot: part4/#25 was triggered21:48:31 UTC
and completed22:08:35 UTC with nine findings/four nits. All nine threads are
resolved; the [nit disposition](https://github.com/QuixiAI/SlimServe/pull/25#issuecomment-5641266239)
records every decision. No paid overages or repeated quota retries.

Part4 fixes172c68bf7 (forward mergef66cfdc70) reject unknown routing-record
kinds; keep pool-set and positional-tail checks separate after changed input;
record gathered per-rank checks before failing; anchor source/Git provenance
to the repo root; detect added SASS copies; validate serialized archive paths;
and isolate environment/device/probe assumptions in tests. Shared replay SHA
uses Python3.10-compatible chunked reads; tie replay derives capture geometry.
The old eight-arm AOT diagnostic occupies2.752GiB allocated. Independent caches
and failed-preparation evidence remain intentionally preserved; the protocol
documents cost and fresh-path recovery rather than sharing writable artifacts.

Focused combined-tree checks:107 CPU passes, one inherited-NATIVE_ORDER test
pass, one compiled-indexer GPU archive-path pass,23 expected unavailable-device/
probe skips. Raw `pr-review/part4-{cpu,env,gpu}.xml`; Ruff/diff checks pass.
No changes under csrc/, slimserve/ or vllm/ since qualified f9963f878. Native
binaries/recipe unchanged, GPUs released; no model boot or performance rerun.
Final audit:49/49 inline threads resolved (13/17/0/9/10 by dependency order),
all five PRs mergeable and CI guards passing. No merge into main.

Local ownership review found that Marlin's per-device lock cache could alias
overlapping layers or microbatches. Fix166d19ced uses layer/device/microbatch
ownership for normal, LoRA and batched expert paths; standalone calls without
explicit scratch allocate per invocation. Steady-state serving retains reuse
without a per-call fill. The target needs about126KiB total scratch per GPU
across42 layers.18 focused ownership/fallback tests pass on the combined tree,
raw `part2-workspace-fixed.xml`; the first fixture's wrong positional argument
is preserved in `part2-workspace.xml`. No DBO serving or performance claim.

Final combined qualification on clean f9963f878 completed successfully with
main0313f5228, the owned scratch fix and all part2 fixes present. One boot,
three exact1000/300 cold-prefix repetitions: c1/c8/c16 E2E medians
156.838/577.605/782.960 tok/s; cold32K/128K engine TTFT2511.369/9921.773ms.
All timed batches, text/image and six retrieval contrasts pass;4096 scored
tokens, mean logprob -2.731236241. Known recovered scoring allocation warnings
remain; no diagnostic default changed. Sources/binaries stayed fixed, exit0,
GPU release independently verified. Full ranges/identity/limitations are in
the baseline snapshot; raw `pr-review/serving-final-review/`. This preserves
performance, not a paired speedup or resolution of score-repeatability limits.
Later diagnostic-only review fixes need focused tests, not another model boot;
do not extend this qualification to subsequent serving-path changes.

Main's existing
glm53f-q2-1/metal record references a missing glm53f-gguf source and lacks its
FP8-KV note; eight broad registry failures reproduce on main and are recorded
separately, not hidden by invented model metadata. No Foundry service or external
port is required here. Use REST for PR edits: the installed gh pr edit command
fails on deprecated GraphQL projectCards, unrelated to authentication.
