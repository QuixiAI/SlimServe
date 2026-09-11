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

The accumulated upstream diff is large (363 files and about 85K added lines
at `b4cb567e6`), including historical notebooks and diagnostic tests. Do not
equate that with 85K lines of retained kernels. Review runtime dependencies
before moving or deleting diagnostics; preserve the evidence and user edits.
Existing draft [PR #24](https://github.com/QuixiAI/SlimServe/pull/24) uses the
upstream `glm53-flash-sm120` branch. Its published head `7619685d8` is an
ancestor of the retained checkpoint, so publication needs only a fast-forward.
CodeRabbit skipped the draft; no substantive review has occurred yet.

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

The current upstream main `f6c6ed429` has 17 conflicting files against the
retained branch, spanning profiles, shared GLM/KDA paths, CLI, runner and
notebooks. A read-only merge-tree inspection identified these without changing
the worktree. Preserve the independently developed main-side improvements and
the qualified SM120 recipe when reconciling; the existing serving numbers apply
to the recorded source, not automatically to a future merged tree.

Remaining: publish the retained checkpoint, request substantive review, address
feedback and semantic merge conflicts, run focused checks for resulting code
changes, and clean up. No Foundry service or external port is required here.
