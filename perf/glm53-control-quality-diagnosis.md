# GLM53 production-control quality diagnosis — 2026-09-10

Status: CPU evidence complete; future acceptance-methodology revision proposed,
not applied. No serving, quant, compiler, native library, or profile-default change.
The failed `prompt-score-rollout-v1` remains terminal. Do not resume it.

## What the historical controls actually establish

Take all three chronological BF16fn1/native-order0 starts from the retained
2026-09-08 paired-storage series: `mhc-paired-serving/boot-1`, then
`mhc-paired-serving-remainder/boot-1` and `boot-2`. Hold each out once against
the other two using the existing comparator, unchanged 0.01 nat/token aggregate
and per-window limits, original responses and exact prompt IDs. No observation
is replicated or excluded and no reference is selected by score.

| Held-out historical start | Mean logprob | Failed windows / 32 | Largest deficit below allowed floor |
| --- | ---: | ---: | ---: |
| First BF16 start | -2.7313757382 | 6 | 0.086295 |
| Remaining BF16 start 1 | -2.7318342249 | 6 | 0.055896 |
| Remaining BF16 start 2 | -2.7353993114 | 10 | 0.105111 |

All aggregate and needle gates pass; every held-out start fails the window gate.
The historical environment, engine, recipe, quant, packages, native hashes and
CPU affinity match. Between their recorded commits, only the benchmark's teardown/
GPU-release bookkeeping, its tests, and documentation changed; serving code did
not. The original outer-chain interruption is retained, not reclassified: all
three quality-bearing boots completed with serving exit0.

The new unchunked production control fails five windows, with largest deficit
0.090025. Its two native-library hashes differ from the historical runs, and it
compiled fresh. Therefore this is not an identical-binary/cache A/A comparison.
The historical result alone is enough to show that the gate can reject the
unchanged reference policy. It does NOT establish that the new control is safe,
estimate a false-rejection rate from three dependent folds, or attribute the
new discrepancy to the compiler/native changes.

## Reuse the existing causal evidence

The 2026-09-09 model/first-MoE journals already isolate within-run variation in
the production expert alignment; canonical ordering removes the short-context
variation. Stable pool ordering resolves the separately traced long-context
variation. The later native-order path repeats exactly with a fixed warm AOT
cache, not across independently fresh compilations.

Do not repeat those investigations or treat an arbitrary historical score vector
as an independent mathematical oracle. Geometry and attention interventions have
already demonstrated that changing FP32 reduction order can shift the complete
score vector; their failed quality gates stay failed. The qualified isolated KDA
choice comparisons were bit-exact on their recorded matrices, not a proof of
whole-model compiler invariance.

## The memory-fix workload coverage gap

Each text quality request has 512 prefix + 128 continuation tokens. The runner
projects at most 639 prompt-score rows, while chunking is used only ABOVE 1024
rows. Consequently NONE of the 32 text continuation windows exercises the new
chunked scorer, even with its flag enabled. Their exact parity in the earlier
fixed-cache diagnostic is useful model-repeatability evidence, not coverage of
the memory fix. Long needle prompts do exercise large scoring requests; the
completed 8 -> 0 -> 8 allocation-warning result remains useful memory evidence.

New CPU tests observe actual calls to the helper in the extracted real runner
method at 639, 1024, and 1025 rows, with the option both off and on. Equality of
outputs alone cannot establish branch coverage.

## Proposed next qualification (requires methodology decision)

Separate three questions instead of interpreting one cross-start score gate as
an answer to all three:

1. **Scorer equivalence:** in a diagnostic start, compute both scorers from the
   SAME complete live logits and target IDs, retaining exact token IDs, FP32 score
   bits, and ranks. Record nonzero coverage above1024 rows and the actual request/
   chunk geometry. Keep projection and TP gather once and unchanged. A shadow run
   executes the unbounded reference too, so it is not a memory-peak or TPS result.
2. **Repeatability:** measure within-start and independent-fresh-cache variation
   separately, with prescribed repetitions and all outcomes retained. Fixed-cache
   repeatability does not qualify fresh-cache invariance. Do not change native
   ordering or normalization merely to get a favorable historical score.
3. **Model quality:** prescribe a broader held-out, paired evaluation and its
   statistical decision rule BEFORE measuring candidate results. Report aggregate,
   per-window/task effects, uncertainty and retrieval/generation failures. A
   repeatable implementation can still be wrong; a small aggregate improvement
   cannot waive a substantive regression. No new quality rule is adopted here.

Do not promote the memory flag, stable ordering, deterministic compiler policy,
or any speed baseline from this CPU diagnosis. No next GPU command is prescribed.
The operator was asked whether to revise future qualification; existing protocols,
tolerances, frozen failed reports and selected quant remain unchanged.

## Reproducible evidence

`benchmarks.analyze_glm53_quality_pair.held_out_controls` is a read-only diagnostic;
the existing production comparator and rollout behavior are unchanged. Tests
cover unchanged limits, every actual observation held out exactly once, aggregate
pass/window failure, mismatched input rejection, and no input mutation.

Raw under `perf/results/2026-09-10/runtime-control/`:

- `control-variation-v1.py`: exact analysis command body, pinned historical
  response/summary hashes, policy checks, source-diff census and fresh comparison.
- `control-variation-v1.json`, SHA
  `f8cfbb2cbcdda3be5fd1aeeb89cc2031b5eb74d7bedda9fbcfbb0491a540eb8d`.
- `control-variation-cpu-v1.xml`: 36 tests pass/12.74s; expanded v2: 271 tests
  pass/36.16s, including actual runner branch-coverage and bounded-scoring tests.
  Ruff and diff checks pass. CPU8GiB/swap0,
  GPUs hidden. No serving/GPU run or cache mutation was used for this diagnosis.
